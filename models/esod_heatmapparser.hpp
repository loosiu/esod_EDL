// esod_heatmapparser.hpp
#pragma once
#include "esod_core.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

// 추가: cluster 박스 표현용
struct Rect {
    int x1;
    int y1;
    int x2;
    int y2;
};

class HeatMapParserCpp {
public:
    HeatMapParserCpp(int c, int ratio = 8, float threshold = 0.5f,
                     bool mask_only = false, bool cluster_only = false)
        : c_(c),
          ratio_(ratio),
          thresh_(threshold),
          mask_only_(mask_only),
          cluster_only_(cluster_only) {}

    /**
     * feat:  [B, C, H, W]
     * mask:  [B, 1, H, W]  (heatmaps[0])
     *
     * out_patches: [N_patch, C, ph, pw]
     * out_offsets: [N_patch, 5]  (b, x1, y1, x2, y2)
     *
     * Python HeatMapParser.forward 의 inference 분기:
     *   mask_pred = heatmaps[0]
     *   if max>1 or min<0: sigmoid
     *   mask_pred = mask_pred[:,0,:,:]
     *   clusters = ada_slicer_fast(mask_pred, ratio, threshold)
     *   → patches, offsets
     */
    void forward(const Tensor& feat, const Tensor& mask,
                 Tensor& out_patches, Tensor& out_offsets) const
    {
        // feat: [B,C,H,W]
        if (feat.shape.size() != 4)
            throw std::runtime_error("HeatMapParserCpp::forward - feat must be 4D [B,C,H,W]");
        if (mask.shape.size() != 4)
            throw std::runtime_error("HeatMapParserCpp::forward - mask must be 4D [B,1,H,W]");

        int B = feat.shape[0];
        int C = feat.shape[1];
        int H = feat.shape[2];
        int W = feat.shape[3];

        if (C != c_) {
            throw std::runtime_error("HeatMapParserCpp::forward - channel mismatch");
        }
        if (mask.shape[0] != B || mask.shape[2] != H || mask.shape[3] != W) {
            throw std::runtime_error("HeatMapParserCpp::forward - mask shape mismatch with feat");
        }

        // 1) mask 를 [B, H, W] 로 변환 + 필요하면 sigmoid 적용
        Tensor mask_pred({B, H, W});
        float vmin = 1e9f, vmax = -1e9f;
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float v = mask.at4(b, 0, y, x);
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
            }
        }
        bool need_sigmoid = (vmax > 1.0f || vmin < 0.0f);
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float v = mask.at4(b, 0, y, x);
                    if (need_sigmoid) v = sigmoid(v);
                    mask_pred.at3(b, y, x) = v;
                }
            }
        }

        // mask_only 모드는 Python에서 (x, threshold)만 리턴하는 용도.
        // 추론 파이프라인에서는 안 쓰니까 여기서는 구현만 간단히 흉내
        if (mask_only_) {
            // out_patches에는 feat 그대로, out_offsets에는 threshold만 저장하는 식으로 써도 되고
            // 실 사용 안 한다면 그냥 예외 던져도 됨.
            throw std::runtime_error("mask_only_ 모드는 C++에서는 사용하지 않도록 하세요.");
        }

        // 2) cluster 크기 계산 (Python ada_slicer_fast와 동일)
        int cluster_w = make_divisible_int(static_cast<float>(W) / ratio_, 4);
        int cluster_h = make_divisible_int(static_cast<float>(H) / ratio_, 4);

        // 3) adaptive slicing: ada_slicer_fast (Python과 동일한 로직)
        std::vector<std::vector<Rect>> clusters;  // [B][num_clusters_bi]
        ada_slicer_fast(mask_pred, cluster_w, cluster_h, clusters);

        if (cluster_only_) {
            // Python: get_offsets_by_clusters(total_clusters)만 리턴하는 모드
            // 여기서는 offsets만 채워주고 patches는 비워둔다.
            int total = 0;
            for (int b = 0; b < B; ++b) total += static_cast<int>(clusters[b].size());
            out_patches = Tensor({0, 0, 0, 0});  // 사용 안 함
            out_offsets = Tensor({total, 5});
            int idx = 0;
            for (int b = 0; b < B; ++b) {
                for (const auto& r : clusters[b]) {
                    out_offsets.at2(idx, 0) = static_cast<float>(b);
                    out_offsets.at2(idx, 1) = static_cast<float>(r.x1);
                    out_offsets.at2(idx, 2) = static_cast<float>(r.y1);
                    out_offsets.at2(idx, 3) = static_cast<float>(r.x2);
                    out_offsets.at2(idx, 4) = static_cast<float>(r.y2);
                    ++idx;
                }
            }
            return;
        }

        // 4) clusters 기반으로 patch 추출 (Python forward 마지막 부분)
        int N_patch = 0;
        for (int b = 0; b < B; ++b) {
            N_patch += static_cast<int>(clusters[b].size());
        }

        if (N_patch == 0) {
            // Python: return zeros((0,c,ny,nx)), zeros((0,5))
            out_patches = Tensor({0, C, H, W});
            out_offsets = Tensor({0, 5});
            return;
        }

        out_patches = Tensor({N_patch, C, cluster_h, cluster_w});
        out_offsets = Tensor({N_patch, 5});

        int patch_idx = 0;
        for (int b = 0; b < B; ++b) {
            for (const auto& r : clusters[b]) {
                int x1 = r.x1;
                int y1 = r.y1;
                int x2 = r.x2;
                int y2 = r.y2;
                int box_w = x2 - x1;
                int box_h = y2 - y1;

                // feat[b, :, y1:y2, x1:x2] → out_patches[patch_idx, :, :, :]
                for (int c = 0; c < C; ++c) {
                    for (int yy = 0; yy < cluster_h; ++yy) {
                        int gy = y1 + yy;
                        for (int xx = 0; xx < cluster_w; ++xx) {
                            int gx = x1 + xx;
                            float v = 0.0f;
                            if (yy < box_h && xx < box_w &&
                                gy >= 0 && gy < H &&
                                gx >= 0 && gx < W)
                            {
                                v = feat.at4(b, c, gy, gx);
                            }
                            Tensor& P = out_patches;
                            int C_ = P.shape[1];
                            int H_ = P.shape[2];
                            int W_ = P.shape[3];
                            int idx_patch = ((patch_idx * C_ + c) * H_ + yy) * W_ + xx;
                            P.data[idx_patch] = v;
                        }
                    }
                }

                // offsets: [b, x1, y1, x2, y2]
                out_offsets.at2(patch_idx, 0) = static_cast<float>(b);
                out_offsets.at2(patch_idx, 1) = static_cast<float>(x1);
                out_offsets.at2(patch_idx, 2) = static_cast<float>(y1);
                out_offsets.at2(patch_idx, 3) = static_cast<float>(x2);
                out_offsets.at2(patch_idx, 4) = static_cast<float>(y2);

                ++patch_idx;
            }
        }
    }

private:
    int  c_;
    int  ratio_;
    float thresh_;
    bool mask_only_;
    bool cluster_only_;

    // Python ada_slicer_fast와 같은 동작을 하는 C++ 버전
    void ada_slicer_fast(const Tensor& mask_pred,  // [B,H,W] float [0,1]
                         int cluster_w,
                         int cluster_h,
                         std::vector<std::vector<Rect>>& outs) const
    {
        int B = mask_pred.shape[0];
        int H = mask_pred.shape[1];
        int W = mask_pred.shape[2];

        int ratio_x = static_cast<int>(std::ceil(static_cast<float>(W) / cluster_w));
        int ratio_y = static_cast<int>(std::ceil(static_cast<float>(H) / cluster_h));

        outs.clear();
        outs.resize(B);

        // 1) activated & maxima → obj_centers
        std::vector<uint8_t> activated(B * H * W, 0);
        std::vector<uint8_t> maxima(B * H * W, 0);
        std::vector<uint8_t> centers(B * H * W, 0);

        auto idx3 = [H, W](int b, int y, int x) {
            return (b * H + y) * W + x;
        };

        // activated
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float v = mask_pred.at3(b, y, x);
                    activated[idx3(b, y, x)] = (v >= thresh_) ? 1 : 0;
                }
            }
        }

        // maxima: 3x3 local max with zero padding
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float center_v = mask_pred.at3(b, y, x);
                    float max_v = center_v;
                    for (int dy = -1; dy <= 1; ++dy) {
                        int yy = y + dy;
                        if (yy < 0 || yy >= H) continue;
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = x + dx;
                            if (xx < 0 || xx >= W) continue;
                            float v = mask_pred.at3(b, yy, xx);
                            if (v > max_v) max_v = v;
                        }
                    }
                    if (std::abs(center_v - max_v) < 1e-6f) {
                        maxima[idx3(b, y, x)] = 1;
                    }
                }
            }
        }

        // obj_centers = activated & maxima
        float center_sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int id = idx3(b, y, x);
                    centers[id] = (activated[id] && maxima[id]) ? 1 : 0;
                    center_sum += static_cast<float>(centers[id]);
                }
            }
        }

        // 아무 객체도 없으면 각 batch마다 빈 텐서
        if (center_sum == 0.0f) {
            for (int b = 0; b < B; ++b) {
                outs[b].clear();
            }
            return;
        }

        // 2) 각 cluster 영역(ratio_y x ratio_x) 중 obj_center 가 하나라도 있는 곳만 후보로
        std::vector<int> cb;   // candidate batch indices
        std::vector<int> x1v;  // candidate x1
        std::vector<int> y1v;  // candidate y1

        for (int b = 0; b < B; ++b) {
            for (int ry = 0; ry < ratio_y; ++ry) {
                for (int rx = 0; rx < ratio_x; ++rx) {
                    int x1 = rx * cluster_w;
                    int y1 = ry * cluster_h;
                    int x2 = std::min(x1 + cluster_w, W);
                    int y2 = std::min(y1 + cluster_h, H);

                    bool has_center = false;
                    for (int yy = y1; yy < y2 && !has_center; ++yy) {
                        for (int xx = x1; xx < x2; ++xx) {
                            if (centers[idx3(b, yy, xx)]) {
                                has_center = true;
                                break;
                            }
                        }
                    }
                    if (has_center) {
                        cb.push_back(b);
                        x1v.push_back(x1);
                        y1v.push_back(y1);
                    }
                }
            }
        }

        int Ncand = static_cast<int>(cb.size());
        if (Ncand == 0) {
            // center는 있는데, valid_regions 계산에서 다 날아간 케이스는 거의 없지만
            // 방어적으로 처리
            for (int b = 0; b < B; ++b) outs[b].clear();
            return;
        }

        // 3) 각 후보 cluster에 대해 activated patch를 보고 dx,dy refine
        for (int i = 0; i < Ncand; ++i) {
            int b = cb[i];
            int x1 = x1v[i];
            int y1 = y1v[i];

            // act[h][w] = cluster 내 활성화 여부
            std::vector<uint8_t> act(cluster_w * cluster_h, 0);
            auto idx2 = [cluster_w](int y, int x) {
                return y * cluster_w + x;
            };

            for (int yy = 0; yy < cluster_h; ++yy) {
                int gy = y1 + yy;
                for (int xx = 0; xx < cluster_w; ++xx) {
                    int gx = x1 + xx;
                    uint8_t v = 0;
                    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
                        v = activated[idx3(b, gy, gx)];
                    }
                    act[idx2(yy, xx)] = v;
                }
            }

            // act_x: 각 column에 하나라도 활성화가 있는지
            std::vector<uint8_t> act_x(cluster_w, 0);
            std::vector<uint8_t> act_y(cluster_h, 0);
            for (int yy = 0; yy < cluster_h; ++yy) {
                for (int xx = 0; xx < cluster_w; ++xx) {
                    if (act[idx2(yy, xx)]) {
                        act_x[xx] = 1;
                        act_y[yy] = 1;
                    }
                }
            }

            // dx1 = 첫 활성화 column 위치, dx2 = 뒤에서부터 첫 활성화 column 위치의 -index
            int dx1 = 0;
            int dx2 = 0;
            {
                int argmin = 0;
                float best = 1e9f;
                for (int x = 0; x < cluster_w; ++x) {
                    float val = 1.0f - (act_x[x] ? 1.0f : 0.0f);
                    if (val < best) {
                        best = val;
                        argmin = x;
                    }
                }
                dx1 = argmin;
            }
            {
                int argmin = 0;
                float best = 1e9f;
                for (int xr = 0; xr < cluster_w; ++xr) {
                    int x = cluster_w - 1 - xr;  // reversed index
                    float val = 1.0f - (act_x[x] ? 1.0f : 0.0f);
                    if (val < best) {
                        best = val;
                        argmin = xr;
                    }
                }
                dx2 = -argmin;
            }

            int dy1 = 0;
            int dy2 = 0;
            {
                int argmin = 0;
                float best = 1e9f;
                for (int y = 0; y < cluster_h; ++y) {
                    float val = 1.0f - (act_y[y] ? 1.0f : 0.0f);
                    if (val < best) {
                        best = val;
                        argmin = y;
                    }
                }
                dy1 = argmin;
            }
            {
                int argmin = 0;
                float best = 1e9f;
                for (int yr = 0; yr < cluster_h; ++yr) {
                    int y = cluster_h - 1 - yr;
                    float val = 1.0f - (act_y[y] ? 1.0f : 0.0f);
                    if (val < best) {
                        best = val;
                        argmin = yr;
                    }
                }
                dy2 = -argmin;
            }

            int dx = (std::abs(dx1) > std::abs(dx2)) ? dx1 : dx2;
            int dy = (std::abs(dy1) > std::abs(dy2)) ? dy1 : dy2;

            int ref_x1 = x1 + dx;
            int ref_y1 = y1 + dy;

            // clamp(0, width - cluster_w / height - cluster_h)
            ref_x1 = std::max(0, std::min(ref_x1, W - cluster_w));
            ref_y1 = std::max(0, std::min(ref_y1, H - cluster_h));

            Rect r;
            r.x1 = ref_x1;
            r.y1 = ref_y1;
            r.x2 = ref_x1 + cluster_w;
            r.y2 = ref_y1 + cluster_h;

            outs[b].push_back(r);
        }
    }
};
