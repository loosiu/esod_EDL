// esod_head_decoder.hpp
#pragma once
#include "esod_core.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>

class EsodHeadDecoder {
public:
    // conf_low: 이보다 낮으면 버림
    // conf_high: 이 이상이면 known(class 부여), 그 사이는 unknown(cls=-1)
    EsodHeadDecoder(float conf_low = 0.30f,
                    float conf_high = 0.40f,
                    float nms_thresh = 0.45f,
                    int num_classes = 10,
                    float unknown_iou_thresh = 0.30f,     // unknown이 kept known과 겹치면 제거
                    bool  nms_unknown = true,             // unknown끼리도 NMS 할지
                    float nms_unknown_thresh = -1.0f)     // unknown NMS IoU (음수면 nms_thresh 사용)
        : conf_low_(conf_low),
          conf_high_(conf_high),
          nms_thresh_(nms_thresh),
          num_classes_(num_classes),
          na_(3),
          unknown_iou_thresh_(unknown_iou_thresh),
          nms_unknown_(nms_unknown),
          nms_unknown_thresh_(nms_unknown_thresh)
    {
        // strides (Detect.stride = [8, 16, 32])
        strides_ = {8.0f, 16.0f, 32.0f};

        // anchors (픽셀 단위)
        anchors_[0] = { {6.0f, 8.0f}, {10.0f, 22.0f}, {19.0f, 15.0f} };
        anchors_[1] = { {19.0f, 32.0f}, {39.0f, 22.0f}, {32.0f, 47.0f} };
        anchors_[2] = { {74.0f, 38.0f}, {65.0f, 74.0f}, {136.0f, 112.0f} };

        // grid 크기 (patch 24×24 기준)
        ny_ = {24, 12, 6};
        nx_ = {24, 12, 6};

        if (nms_unknown_thresh_ < 0.0f) nms_unknown_thresh_ = nms_thresh_;
    }

    void decode(const Tensor& head_raw,
                const Tensor& offsets,
                std::vector<Detection>& out_dets) const
    {
        if (head_raw.shape.size() != 3)
            throw std::runtime_error("EsodHeadDecoder::decode - head_raw must be 3D [N_patch, M, no]");
        if (offsets.shape.size() != 2 || offsets.shape[1] != 5)
            throw std::runtime_error("EsodHeadDecoder::decode - offsets must be [N_patch, 5]");

        const int N_patch = head_raw.shape[0];
        const int M       = head_raw.shape[1];
        const int no      = head_raw.shape[2];

        if (no != 5 + num_classes_)
            throw std::runtime_error("EsodHeadDecoder::decode - no must be 5+num_classes");

        int num_l0 = ny_[0] * nx_[0] * na_; // 1728
        int num_l1 = ny_[1] * nx_[1] * na_; // 432
        int num_l2 = ny_[2] * nx_[2] * na_; // 108
        if (num_l0 + num_l1 + num_l2 != M)
            throw std::runtime_error("EsodHeadDecoder::decode - M != sum(level sizes)");

        std::vector<Detection> all_dets;
        all_dets.reserve(N_patch * 256);

        for (int p = 0; p < N_patch; ++p) {
            float b_idx_f  = offsets.at(p, 0);
            int   img_id   = static_cast<int>(b_idx_f + 0.5f);
            float x1_patch = offsets.at(p, 1);
            float y1_patch = offsets.at(p, 2);

            const float* row = &head_raw.data[(p * M) * no];

            int idx_offset = 0;
            for (int l = 0; l < 3; ++l) {
                int ny = ny_[l];
                int nx = nx_[l];
                float stride = strides_[l];
                int num_this_level = ny * nx * na_;

                float patch_off_x = x1_patch / stride;
                float patch_off_y = y1_patch / stride;

                for (int idx = 0; idx < num_this_level; ++idx) {
                    int j = idx_offset + idx;

                    int anchor_id = idx / (ny * nx);
                    int rem       = idx % (ny * nx);
                    int gy        = rem / nx;
                    int gx        = rem % nx;

                    const float* p_raw = row + j * no;

                    float tx  = sigmoid(p_raw[0]);
                    float ty  = sigmoid(p_raw[1]);
                    float tw  = sigmoid(p_raw[2]);
                    float th  = sigmoid(p_raw[3]);
                    float obj = sigmoid(p_raw[4]);

                    int best_cls = -1;
                    float best_cls_score = 0.0f;
                    for (int c = 0; c < num_classes_; ++c) {
                        float cls_score = sigmoid(p_raw[5 + c]);
                        if (cls_score > best_cls_score) {
                            best_cls_score = cls_score;
                            best_cls = c;
                        }
                    }

                    float conf = obj * best_cls_score;

                    // ✅ 1) conf_low 미만은 버림
                    if (conf < conf_low_) continue;

                    // ✅ 2) conf_low~conf_high는 unknown
                    int out_cls = (conf >= conf_high_) ? best_cls : -1;

                    float gx_off = gx + patch_off_x;
                    float gy_off = gy + patch_off_y;

                    float bx = (tx * 2.0f - 0.5f + gx_off) * stride;
                    float by = (ty * 2.0f - 0.5f + gy_off) * stride;

                    float aw = anchors_[l][anchor_id].first;
                    float ah = anchors_[l][anchor_id].second;

                    float bw = (tw * 2.0f); bw *= bw; bw *= aw;
                    float bh = (th * 2.0f); bh *= bh; bh *= ah;

                    Detection det;
                    det.x = bx;
                    det.y = by;
                    det.w = bw;
                    det.h = bh;
                    det.score = conf;
                    det.cls = out_cls;        // known or unknown(-1)
                    det.image_id = img_id;

                    all_dets.push_back(det);
                }

                idx_offset += num_this_level;
            }
        }

        out_dets.clear();
        if (all_dets.empty()) return;

        int max_img_id = 0;
        for (const auto& d : all_dets)
            max_img_id = std::max(max_img_id, d.image_id);

        for (int img = 0; img <= max_img_id; ++img) {
            std::vector<Detection> known, unknown;
            for (const auto& d : all_dets) {
                if (d.image_id != img) continue;
                if (d.cls == -1) unknown.push_back(d);
                else known.push_back(d);
            }
            if (known.empty() && unknown.empty()) continue;

            // ===========================
            // ✅ A) known만 class-wise NMS
            // ===========================
            std::sort(known.begin(), known.end(),
                      [](const Detection& a, const Detection& b) { return a.score > b.score; });

            std::vector<Detection> keep_known;
            std::vector<bool> removed_k(known.size(), false);

            for (size_t i = 0; i < known.size(); ++i) {
                if (removed_k[i]) continue;
                keep_known.push_back(known[i]);

                float x1i, y1i, x2i, y2i;
                xywh_to_xyxy(known[i], x1i, y1i, x2i, y2i);

                for (size_t j = i + 1; j < known.size(); ++j) {
                    if (removed_k[j]) continue;
                    if (known[j].cls != known[i].cls) continue;

                    float x1j, y1j, x2j, y2j;
                    xywh_to_xyxy(known[j], x1j, y1j, x2j, y2j);

                    float iou = iou_xyxy(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j);
                    if (iou > nms_thresh_) removed_k[j] = true;
                }
            }

            // ==========================================
            // ✅ B) unknown은 kept known과 겹치면 제거
            // ==========================================
            std::vector<Detection> unknown_filtered;
            unknown_filtered.reserve(unknown.size());

            for (const auto& u : unknown) {
                float ux1, uy1, ux2, uy2;
                xywh_to_xyxy(u, ux1, uy1, ux2, uy2);

                bool overlapped = false;
                for (const auto& k : keep_known) {
                    float kx1, ky1, kx2, ky2;
                    xywh_to_xyxy(k, kx1, ky1, kx2, ky2);

                    float iou = iou_xyxy(ux1, uy1, ux2, uy2, kx1, ky1, kx2, ky2);
                    if (iou >= unknown_iou_thresh_) {
                        overlapped = true;
                        break;
                    }
                }
                if (!overlapped) unknown_filtered.push_back(u);
            }

            // ===========================
            // ✅ C) unknown끼리 NMS (옵션)
            // ===========================
            std::vector<Detection> keep_unknown;

            if (nms_unknown_ && !unknown_filtered.empty()) {
                std::sort(unknown_filtered.begin(), unknown_filtered.end(),
                          [](const Detection& a, const Detection& b) { return a.score > b.score; });

                std::vector<bool> removed_u(unknown_filtered.size(), false);

                for (size_t i = 0; i < unknown_filtered.size(); ++i) {
                    if (removed_u[i]) continue;
                    keep_unknown.push_back(unknown_filtered[i]);

                    float x1i, y1i, x2i, y2i;
                    xywh_to_xyxy(unknown_filtered[i], x1i, y1i, x2i, y2i);

                    for (size_t j = i + 1; j < unknown_filtered.size(); ++j) {
                        if (removed_u[j]) continue;

                        float x1j, y1j, x2j, y2j;
                        xywh_to_xyxy(unknown_filtered[j], x1j, y1j, x2j, y2j);

                        float iou = iou_xyxy(x1i, y1i, x2i, y2i, x1j, y1j, x2j, y2j);
                        if (iou > nms_unknown_thresh_) removed_u[j] = true;
                    }
                }
            } else {
                keep_unknown = std::move(unknown_filtered);
            }

            // 최종 합치기 (known 먼저, unknown 뒤)
            out_dets.insert(out_dets.end(), keep_known.begin(), keep_known.end());
            out_dets.insert(out_dets.end(), keep_unknown.begin(), keep_unknown.end());
        }
    }

private:
    float conf_low_;
    float conf_high_;
    float nms_thresh_;
    int   num_classes_;

    int na_;
    std::vector<float> strides_;
    std::vector<std::pair<float,float>> anchors_[3];
    std::vector<int> ny_, nx_;

    float unknown_iou_thresh_;
    bool  nms_unknown_;
    float nms_unknown_thresh_;
};
