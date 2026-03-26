// esod_core.hpp
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;

    Tensor(const std::vector<int>& s) : shape(s) {
        int n = 1;
        for (int v : shape) n *= v;
        data.assign(n, 0.0f);
    }

    // ---- 4D index: [N,C,H,W] ----
    float& at4(int n, int c, int h, int w) {
        if (shape.size() != 4)
            throw std::runtime_error("Tensor::at4 - shape must be 4D");
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];
        int idx = ((n * C + c) * H + h) * W + w;
        return data[idx];
    }

    const float& at4(int n, int c, int h, int w) const {
        if (shape.size() != 4)
            throw std::runtime_error("Tensor::at4 - shape must be 4D");
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];
        int idx = ((n * C + c) * H + h) * W + w;
        return data[idx];
    }

    // ---- 3D index: [N,H,W] ----
    float& at3(int n, int h, int w) {
        if (shape.size() != 3)
            throw std::runtime_error("Tensor::at3 - shape must be 3D");
        int H = shape[1];
        int W = shape[2];
        int idx = (n * H + h) * W + w;
        return data[idx];
    }

    const float& at3(int n, int h, int w) const {
        if (shape.size() != 3)
            throw std::runtime_error("Tensor::at3 - shape must be 3D");
        int H = shape[1];
        int W = shape[2];
        int idx = (n * H + h) * W + w;
        return data[idx];
    }

    // ---- 2D index: [N,K] ----
    float& at2(int n, int k) {
        if (shape.size() != 2)
            throw std::runtime_error("Tensor::at2 - shape must be 2D");
        int K = shape[1];
        int idx = n * K + k;
        return data[idx];
    }

    const float& at2(int n, int k) const {
        if (shape.size() != 2)
            throw std::runtime_error("Tensor::at2 - shape must be 2D");
        int K = shape[1];
        int idx = n * K + k;
        return data[idx];
    }

    // ---- 호환용 at(): 2D/3D를 위해 alias 제공 ----

    // offsets.at(p, 0) 같은 코드용 (2D)
    float& at(int n, int k) {
        return at2(n, k);
    }

    const float& at(int n, int k) const {
        return at2(n, k);
    }

    // head_raw.at(p, m, no) 같은 3D 접근이 들어갈 수도 있으니 미리 지원
    float& at(int a, int b, int c) {
        if (shape.size() != 3)
            throw std::runtime_error("Tensor::at(3) - shape must be 3D");
        int B = shape[1];
        int C = shape[2];
        int idx = (a * B + b) * C + c;
        return data[idx];
    }

    const float& at(int a, int b, int c) const {
        if (shape.size() != 3)
            throw std::runtime_error("Tensor::at(3) - shape must be 3D");
        int B = shape[1];
        int C = shape[2];
        int idx = (a * B + b) * C + c;
        return data[idx];
    }

    // 필요하면 numel() 같은 함수도 여기에...
};

// math helpers
inline int make_divisible_int(int x, int divisor) {
    return ((x + divisor - 1) / divisor) * divisor;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

struct Detection {
    float x, y, w, h;
    float score;
    int cls;
    int image_id;
    int   track_id;   // ✅ SORT 또는 트래커에서 쓸 ID

    Detection()
        : x(0), y(0), w(0), h(0),
          score(0),
          cls(-1),
          image_id(-1),
          track_id(-1)      // ✅ 기본은 미할당 상태
    {}
    
};

// center-x,center-y,width,height -> x1,y1,x2,y2
inline void xywh_to_xyxy(const Detection& d,
                         float& x1, float& y1,
                         float& x2, float& y2) {
    x1 = d.x - d.w * 0.5f;
    y1 = d.y - d.h * 0.5f;
    x2 = d.x + d.w * 0.5f;
    y2 = d.y + d.h * 0.5f;
}

inline float iou_xyxy(float x1, float y1, float x2, float y2,
                      float x1b, float y1b, float x2b, float y2b) {
    float xx1 = std::max(x1,  x1b);
    float yy1 = std::max(y1,  y1b);
    float xx2 = std::min(x2,  x2b);
    float yy2 = std::min(y2,  y2b);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area2 = std::max(0.0f, x2b - x1b) * std::max(0.0f, y2b - y1b);
    float uni   = area1 + area2 - inter + 1e-6f;

    return inter / uni;
}
