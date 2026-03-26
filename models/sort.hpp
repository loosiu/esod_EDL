#pragma once
/*
 * C++ port of:
 * "SORT: A Simple, Online and Realtime Tracker"
 * Original Python implementation by Alex Bewley (GPLv3).
 *
 * This is a fairly direct translation of the Python version you pasted:
 *   - KalmanBoxTracker(dim_x=7, dim_z=4)
 *   - IoU-based association
 *   - Hungarian assignment (linear_assignment)
 *   - Sort::update(dets) API
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

// =================== 기본 구조체 ===================

struct SortBBox {
    double x1, y1, x2, y2, score;
};

struct SortTrack {
    double x1, y1, x2, y2;
    int id;   // tracker ID (Python에서 +1 되어 나가는 값)
};

// =================== 유틸 함수들 ===================

inline double iou_single(const SortBBox& a, const SortBBox& b) {
    double xx1 = std::max(a.x1, b.x1);
    double yy1 = std::max(a.y1, b.y1);
    double xx2 = std::min(a.x2, b.x2);
    double yy2 = std::min(a.y2, b.y2);

    double w = std::max(0.0, xx2 - xx1);
    double h = std::max(0.0, yy2 - yy1);
    double inter = w * h;
    double areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    double areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    double denom = areaA + areaB - inter;
    if (denom <= 0.0) return 0.0;
    return inter / denom;
}

// [x1,y1,x2,y2] -> [x,y,s,r]
inline void convert_bbox_to_z(const SortBBox& bb, double z[4]) {
    double w = bb.x2 - bb.x1;
    double h = bb.y2 - bb.y1;
    double x = bb.x1 + w * 0.5;
    double y = bb.y1 + h * 0.5;
    double s = w * h;                       // scale = area
    double r = w / std::max(h, 1e-9);       // aspect ratio
    z[0] = x; z[1] = y; z[2] = s; z[3] = r;
}

// [x,y,s,r] -> [x1,y1,x2,y2]
inline SortBBox convert_x_to_bbox(const double x[7], double score = -1.0) {
    double s = x[2];
    double r = x[3];
    double w = std::sqrt(std::max(s * r, 1e-9));
    double h = s / std::max(w, 1e-9);
    double xc = x[0];
    double yc = x[1];

    SortBBox bb;
    bb.x1 = xc - w * 0.5;
    bb.y1 = yc - h * 0.5;
    bb.x2 = xc + w * 0.5;
    bb.y2 = yc + h * 0.5;
    bb.score = score;
    return bb;
}

// =================== 작은 행렬 유틸(7x7, 7x4, 4x7, 4x4) ===================

inline void mat7x7_set_identity(double A[7][7]) {
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            A[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

inline void mat4x4_set_identity(double A[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// C = A * B, (7x7 * 7x7)
inline void mat7x7_mul(const double A[7][7], const double B[7][7], double C[7][7]) {
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 7; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// C = A * B, (4x7 * 7x7 = 4x7)
inline void mat4x7_mul(const double A[4][7], const double B[7][7], double C[4][7]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 7; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 7; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// C = A * B, (7x7 * 7x4 = 7x4)
inline void mat7x4_mul(const double A[7][7], const double B[7][4], double C[7][4]) {
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 7; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// C = A * B, (7x4 * 4x4 = 7x4)
inline void mat7x4_mul_4x4(const double A[7][4], const double B[4][4], double C[7][4]) {
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// C = A * B, (4x7 * 7x4 = 4x4)
inline void mat4x4_mul_4x7_7x4(const double A[4][7], const double B[7][4], double C[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 7; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// y = A * x, A:4x7, x:7
inline void mat4x7_vec7_mul(const double A[4][7], const double x[7], double y[4]) {
    for (int i = 0; i < 4; ++i) {
        double sum = 0.0;
        for (int j = 0; j < 7; ++j) sum += A[i][j] * x[j];
        y[i] = sum;
    }
}

// y = A * x, A:7x7, x:7
inline void mat7x7_vec7_mul(const double A[7][7], const double x[7], double y[7]) {
    for (int i = 0; i < 7; ++i) {
        double sum = 0.0;
        for (int j = 0; j < 7; ++j) sum += A[i][j] * x[j];
        y[i] = sum;
    }
}

// A += B (same shape 7x7)
inline void mat7x7_add_inplace(double A[7][7], const double B[7][7]) {
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A[i][j] += B[i][j];
}

// A += B (4x4)
inline void mat4x4_add_inplace(double A[4][4], const double B[4][4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            A[i][j] += B[i][j];
}

// C = A^T (7x7)
inline void mat7x7_transpose(const double A[7][7], double C[7][7]) {
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            C[j][i] = A[i][j];
}

// C = A^T (4x7 -> 7x4)
inline void mat4x7_transpose(const double A[4][7], double C[7][4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 7; ++j)
            C[j][i] = A[i][j];
}

// 4x4 역행렬 (간단 Gaussian elimination)
inline bool mat4x4_inverse(const double A[4][4], double invA[4][4]) {
    double aug[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) aug[i][j] = A[i][j];
        for (int j = 4; j < 8; ++j) aug[i][j] = (i == (j-4)) ? 1.0 : 0.0;
    }
    for (int i = 0; i < 4; ++i) {
        // pivot
        int pivot = i;
        double mx = std::fabs(aug[i][i]);
        for (int r = i+1; r < 4; ++r) {
            double v = std::fabs(aug[r][i]);
            if (v > mx) { mx = v; pivot = r; }
        }
        if (mx < 1e-12) return false;
        if (pivot != i) {
            for (int c = 0; c < 8; ++c)
                std::swap(aug[i][c], aug[pivot][c]);
        }
        double diag = aug[i][i];
        for (int c = 0; c < 8; ++c) aug[i][c] /= diag;
        for (int r = 0; r < 4; ++r) {
            if (r == i) continue;
            double factor = aug[r][i];
            for (int c = 0; c < 8; ++c)
                aug[r][c] -= factor * aug[i][c];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            invA[i][j] = aug[i][j+4];
    return true;
}

// =================== KalmanBoxTracker ===================

class KalmanBoxTracker {
public:
    static int count;

    explicit KalmanBoxTracker(const SortBBox& bbox)
        : time_since_update(0),
          id(count++),
          hits(0),
          hit_streak(0),
          age(0)
    {
        // F
        // [[1,0,0,0,1,0,0],
        //  [0,1,0,0,0,1,0],
        //  [0,0,1,0,0,0,1],
        //  [0,0,0,1,0,0,0],
        //  [0,0,0,0,1,0,0],
        //  [0,0,0,0,0,1,0],
        //  [0,0,0,0,0,0,1]]
        mat7x7_set_identity(F);
        F[0][4] = 1.0;
        F[1][5] = 1.0;
        F[2][6] = 1.0;

        // H
        // [[1,0,0,0,0,0,0],
        //  [0,1,0,0,0,0,0],
        //  [0,0,1,0,0,0,0],
        //  [0,0,0,1,0,0,0]]
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 7; ++j)
                H[i][j] = 0.0;
        H[0][0] = H[1][1] = H[2][2] = H[3][3] = 1.0;

        // R = I, R[2:,2:] *= 10
        mat4x4_set_identity(R);
        R[2][2] *= 10.0;
        R[3][3] *= 10.0;

        // P = I, P[4:,4:] *= 1000, P *= 10
        mat7x7_set_identity(P);
        for (int i = 4; i < 7; ++i)
            P[i][i] *= 1000.0;
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                P[i][j] *= 10.0;

        // Q = I, Q[-1,-1] *= 0.01, Q[4:,4:] *= 0.01
        mat7x7_set_identity(Q);
        Q[6][6] *= 0.01;
        for (int i = 4; i < 7; ++i)
            Q[i][i] *= 0.01;

        // x 초기화
        for (int i = 0; i < 7; ++i) x[i] = 0.0;
        double z[4];
        convert_bbox_to_z(bbox, z);
        for (int i = 0; i < 4; ++i) x[i] = z[i];
    }

    void update(const SortBBox& bbox) {
        // time_since_update = 0, history clear, hits++, hit_streak++
        time_since_update = 0;
        hits += 1;
        hit_streak += 1;

        double z[4];
        convert_bbox_to_z(bbox, z);

        // y = z - Hx
        double Hx[4];
        mat4x7_vec7_mul(H, x, Hx);
        double y[4];
        for (int i = 0; i < 4; ++i) y[i] = z[i] - Hx[i];

        // S = HPH^T + R
        double HP[4][7];
        mat4x7_mul(P, H, HP);  // 7x7 * 7x4? -> 바꿔야: 먼저 P * H^T? 편의상: (H * P)
        // (정확히는 HP = H * P)
        // 위에서 mat4x7_mul 을 4x7 * 7x7 로 정의했으므로:
        // mat4x7_mul(H, P, HP); 로 수정
        mat4x7_mul(H, P, HP);

        double HPT[4][4];
        // HPH^T = HP * H^T (4x7 * 7x4)
        double HT[7][4];
        mat4x7_transpose(H, HT);
        mat4x4_mul_4x7_7x4(HP, HT, HPT);

        double S[4][4];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                S[i][j] = HPT[i][j];
        mat4x4_add_inplace(S, R);

        // K = P H^T S^-1
        double invS[4][4];
        if (!mat4x4_inverse(S, invS)) {
            // 역행렬 실패 -> 업데이트 스킵
            return;
        }

        double PHT[7][4]; // P * H^T
        double HT2[7][4];
        mat4x7_transpose(H, HT2);
        mat7x4_mul(P, HT2, PHT); // 7x7 * 7x4 = 7x4

        double K[7][4];
        mat7x4_mul_4x4(PHT, invS, K);  // 7x4 * 4x4 = 7x4

        // x = x + K y
        double Ky[7];
        for (int i = 0; i < 7; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 4; ++j) sum += K[i][j] * y[j];
            Ky[i] = sum;
        }
        for (int i = 0; i < 7; ++i) x[i] += Ky[i];

        // P = (I - K H) P
        double KH[7][7];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k)
                    sum += K[i][k] * H[k][j];
                KH[i][j] = sum;
            }

        double I[7][7];
        mat7x7_set_identity(I);
        double IKH[7][7];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                IKH[i][j] = I[i][j] - KH[i][j];

        double newP[7][7];
        mat7x7_mul(IKH, P, newP);
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                P[i][j] = newP[i][j];
    }

    SortBBox predict() {
        // if ((x[6] + x[2]) <= 0): x[6] = 0
        if ((x[6] + x[2]) <= 0.0) x[6] = 0.0;

        // x = F x
        double newx[7];
        mat7x7_vec7_mul(F, x, newx);
        for (int i = 0; i < 7; ++i) x[i] = newx[i];

        // P = F P F^T + Q
        double FP[7][7];
        mat7x7_mul(F, P, FP);
        double FT[7][7];
        mat7x7_transpose(F, FT);
        double FPFt[7][7];
        mat7x7_mul(FP, FT, FPFt);
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                P[i][j] = FPFt[i][j] + Q[i][j];

        age += 1;
        if (time_since_update > 0) {
            hit_streak = 0;
        }
        time_since_update += 1;

        return convert_x_to_bbox(x);
    }

    SortBBox get_state() const {
        return convert_x_to_bbox(x);
    }

    int time_since_update;
    int id;
    int hits;
    int hit_streak;
    int age;

private:
    double x[7];     // state
    double F[7][7];
    double H[4][7];
    double Q[7][7];
    double R[4][4];
    double P[7][7];
};

int KalmanBoxTracker::count = 0;

// =================== Hungarian (linear_assignment) ===================

// 헝가리안 알고리즘: 최소 비용 할당
// cost: NxM, nRows=N, nCols=M
// 결과: vector<pair<row,col>>
inline std::vector<std::pair<int,int>> hungarian_assignment(
    const std::vector<std::vector<double>>& cost)
{
    int nRows = (int)cost.size();
    int nCols = nRows ? (int)cost[0].size() : 0;
    int n = std::max(nRows, nCols);
    const double BIG = 1e10;

    // 정사각형으로 패딩
    std::vector<std::vector<double>> c(n, std::vector<double>(n, BIG));
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
            c[i][j] = cost[i][j];

    // Hungarian
    std::vector<double> u(n+1), v(n+1);
    std::vector<int> p(n+1), way(n+1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n+1, BIG);
        std::vector<char> used(n+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = BIG;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = c[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // 결과 추출
    std::vector<std::pair<int,int>> matches;
    matches.reserve(n);
    for (int j = 1; j <= n; ++j) {
        if (p[j] == 0) continue;
        int i = p[j] - 1;
        int jj = j - 1;
        if (i < nRows && jj < nCols) {
            // 유효 범위
            matches.emplace_back(i, jj);
        }
    }
    return matches;
}

// Python associate_detections_to_trackers와 동일한 인터페이스
inline void associate_detections_to_trackers(
    const std::vector<SortBBox>& dets,
    const std::vector<SortBBox>& trks,
    double iou_threshold,
    std::vector<std::pair<int,int>>& matches,
    std::vector<int>& unmatched_dets,
    std::vector<int>& unmatched_trks
) {
    matches.clear();
    unmatched_dets.clear();
    unmatched_trks.clear();

    int N = (int)dets.size();
    int M = (int)trks.size();

    if (M == 0) {
        // (len(trackers)==0) -> 모든 detection unmatched
        for (int i = 0; i < N; ++i) unmatched_dets.push_back(i);
        return;
    }

    // IoU matrix
    std::vector<std::vector<double>> iou_matrix(N, std::vector<double>(M, 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            iou_matrix[i][j] = iou_single(dets[i], trks[j]);

    if (N > 0 && M > 0) {
        // a = (iou > iou_threshold)
        std::vector<std::vector<int>> a(N, std::vector<int>(M, 0));
        std::vector<int> rowSum(N, 0), colSum(M, 0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (iou_matrix[i][j] > iou_threshold) {
                    a[i][j] = 1;
                    rowSum[i] += 1;
                    colSum[j] += 1;
                }
            }
        }

        int maxRow = 0, maxCol = 0;
        for (int i = 0; i < N; ++i) maxRow = std::max(maxRow, rowSum[i]);
        for (int j = 0; j < M; ++j) maxCol = std::max(maxCol, colSum[j]);

        std::vector<std::pair<int,int>> m_inds;
        if (maxRow == 1 && maxCol == 1) {
            // 단순히 a==1인 위치를 매칭으로 사용
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < M; ++j)
                    if (a[i][j] == 1)
                        m_inds.emplace_back(i, j);
        } else {
            // linear_assignment(-iou_matrix)
            std::vector<std::vector<double>> cost(N, std::vector<double>(M, 0.0));
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < M; ++j)
                    cost[i][j] = -iou_matrix[i][j];
            m_inds = hungarian_assignment(cost);
        }

        // unmatched 계산
        std::vector<bool> matchedDet(N, false);
        std::vector<bool> matchedTrk(M, false);
        for (auto& m : m_inds) {
            int d = m.first;
            int t = m.second;
            // filter out low IOU
            if (iou_matrix[d][t] < iou_threshold) continue;
            matches.emplace_back(d, t);
            matchedDet[d] = true;
            matchedTrk[t] = true;
        }

        for (int i = 0; i < N; ++i)
            if (!matchedDet[i]) unmatched_dets.push_back(i);
        for (int j = 0; j < M; ++j)
            if (!matchedTrk[j]) unmatched_trks.push_back(j);
    } else {
        // no valid iou_matrix
        // 모든 detection unmatched
        for (int i = 0; i < N; ++i) unmatched_dets.push_back(i);
        // trackers는 모두 unmatched_trks로 둘 수도 있지만, Python 코드는
        // matched_indices.shape=(0,2)인 경우 → 아래에서 time_since_update로 죽음
        for (int j = 0; j < M; ++j) unmatched_trks.push_back(j);
    }
}

// =================== SORT 클래스 ===================

class Sort {
public:
    Sort(int max_age = 5, int min_hits = 3, double iou_threshold = 0.3)
        : max_age_(max_age),
          min_hits_(min_hits),
          iou_threshold_(iou_threshold),
          frame_count_(0)
    {}

    // dets: [ [x1,y1,x2,y2,score], ... ]
    // return: [ {x1,y1,x2,y2,id}, ... ]  (Python에서는 (N,5) ndarray)
    std::vector<SortTrack> update(const std::vector<SortBBox>& dets) {
        frame_count_ += 1;

        // 1) 예측 (trks = zeros(len(trackers),5))
        std::vector<SortBBox> trks;
        trks.reserve(trackers_.size());
        std::vector<int> to_del;

        for (size_t t = 0; t < trackers_.size(); ++t) {
            SortBBox pos = trackers_[t].predict();
            // pos: [x1,y1,x2,y2]
            if (std::isnan(pos.x1) || std::isnan(pos.y1) ||
                std::isnan(pos.x2) || std::isnan(pos.y2)) {
                to_del.push_back((int)t);
            } else {
                SortBBox trk = pos;
                trk.score = 0.0;
                trks.push_back(trk);
            }
        }

        // NaN 제거
        for (int i = (int)to_del.size() - 1; i >= 0; --i) {
            trackers_.erase(trackers_.begin() + to_del[i]);
        }

        // 2) 매칭
        std::vector<std::pair<int,int>> matches;
        std::vector<int> unmatched_dets, unmatched_trks;
        associate_detections_to_trackers(
            dets, trks, iou_threshold_,
            matches, unmatched_dets, unmatched_trks
        );

        // 3) matched trackers 업데이트
        for (auto& m : matches) {
            int det_idx = m.first;
            int trk_idx = m.second;
            trackers_[trk_idx].update(dets[det_idx]);
        }

        // 4) unmatched_dets -> 새 tracker 생성
        for (int idx : unmatched_dets) {
            trackers_.emplace_back(dets[idx]);
        }

        // 5) 결과 생성 + 죽은 트랙 제거
        std::vector<SortTrack> ret;
        // Python: for trk in reversed(self.trackers):
        for (int i = (int)trackers_.size() - 1; i >= 0; --i) {
            auto& trk = trackers_[i];
            SortBBox d = trk.get_state();

            // if (trk.time_since_update < 1) and (trk.hit_streak >= min_hits or frame_count <= min_hits)
            if (trk.time_since_update < 1 &&
               (trk.hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
                SortTrack out;
                out.x1 = d.x1;
                out.y1 = d.y1;
                out.x2 = d.x2;
                out.y2 = d.y2;
                out.id = trk.id + 1;  // +1 as MOT benchmark requires positive
                ret.push_back(out);
            }

            // if (trk.time_since_update > max_age): remove
            if (trk.time_since_update > max_age_) {
                trackers_.erase(trackers_.begin() + i);
            }
        }

        return ret;
    }

private:
    int max_age_;
    int min_hits_;
    double iou_threshold_;
    int frame_count_;
    std::vector<KalmanBoxTracker> trackers_;
};
