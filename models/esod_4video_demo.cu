// esod_4video_demo.cu

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <array>
#include <algorithm>  // std::min, std::max
#include <cmath>
#include <limits>

#include "esod_core.hpp"
#include "esod_head_decoder.hpp"

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "cuda_runtime.h"

using namespace nvinfer1;

// ======================= 공통 상수/구조체 =======================

// VisDrone 10-class → 3-class(person, bicycle, car)
static const char* kThreeClassNames[3] = {
    "person",   // 0
    "bicycle",  // 1
    "car"       // 2
};

static inline int map_visdrone10_to3(int cls10)
{
    static const int table[10] = {
        0, // 0 pedestrian      → person
        0, // 1 people          → person
        1, // 2 bicycle         → bicycle
        2, // 3 car             → car
        2, // 4 van             → car
        2, // 5 truck           → car
        1, // 6 tricycle        → bicycle
        1, // 7 awning-tricycle → bicycle
        2, // 8 bus             → car
        1  // 9 motor           → bicycle
    };

    if (cls10 < 0 || cls10 >= 10) return -1;
    return table[cls10];
}

// 전처리(letterbox) 정보
struct PreprocInfo {
    float gain = 1.0f;
    int   pad_x = 0;
    int   pad_y = 0;
};

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

// ================== TensorRT Logger ==================
class TrtLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << "\n";
        }
    }
} gLogger;

// ================== 공통: 엔진 파일 로딩 ==================
static std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << path << "\n";
        std::exit(1);
    }
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// ======================================================
// 1) engine1 (backbone) TensorRT 핸들 / 초기화
// ======================================================
static IRuntime*          gRuntime1  = nullptr;
static ICudaEngine*       gEngine1   = nullptr;
static IExecutionContext* gContext1  = nullptr;
static cudaStream_t       gStream1;

// engine1용 device 버퍼
static float* d_input1 = nullptr;
static float* d_feat1  = nullptr;
static float* d_mask1  = nullptr;
static size_t cap_input1 = 0;
static size_t cap_feat1  = 0;
static size_t cap_mask1  = 0;

// feat/mask feature 크기 (고정 가정: B,C,H,W에서 C,H,W)
static int gFeatC = 192;
static int gFeatH = 192;
static int gFeatW = 192;

static const char* kE1InputName = "images";      // INPUT
static const char* kE1FeatName  = "feat_stage1"; // OUTPUT
static const char* kE1MaskName  = "mask";

void init_engine1(const std::string& enginePath) {
    auto engineData = loadEngineFile(enginePath);

    gRuntime1 = createInferRuntime(gLogger);
    if (!gRuntime1) {
        std::cerr << "Failed to create TRT runtime for engine1\n";
        std::exit(1);
    }

    gEngine1 = gRuntime1->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!gEngine1) {
        std::cerr << "Failed to deserialize engine1\n";
        std::exit(1);
    }

    std::cout << "[engine1 IO tensors]\n";
    int nIO = gEngine1->getNbIOTensors();
    for (int i = 0; i < nIO; ++i) {
        const char* name = gEngine1->getIOTensorName(i);
        auto mode = gEngine1->getTensorIOMode(name);
        std::cout << "  " << i << ": " << name
                  << " (" << (mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << ")\n";
    }

    gContext1 = gEngine1->createExecutionContext();
    if (!gContext1) {
        std::cerr << "Failed to create context for engine1\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamCreate(&gStream1));
}

void ensure_engine1_buffers(size_t inputSize, size_t featSize, size_t maskSize) {
    if (cap_input1 < inputSize) {
        if (d_input1) cudaFree(d_input1);
        CHECK_CUDA(cudaMalloc(&d_input1, inputSize));
        cap_input1 = inputSize;
    }
    if (cap_feat1 < featSize) {
        if (d_feat1) cudaFree(d_feat1);
        CHECK_CUDA(cudaMalloc(&d_feat1, featSize));
        cap_feat1 = featSize;
    }
    if (cap_mask1 < maskSize) {
        if (d_mask1) cudaFree(d_mask1);
        CHECK_CUDA(cudaMalloc(&d_mask1, maskSize));
        cap_mask1 = maskSize;
    }
}

// ======================================================
// 2) engine2 (head) TensorRT 핸들 / 초기화
// ======================================================
static IRuntime*          gRuntime2  = nullptr;
static ICudaEngine*       gEngine2   = nullptr;
static IExecutionContext* gContext2  = nullptr;
static cudaStream_t       gStream2;

// engine2용 device 버퍼 (chunk 단위로 사용)
static float* d_in2  = nullptr;
static float* d_out2 = nullptr;
static size_t cap_in2  = 0;
static size_t cap_out2 = 0;

// patch 텐서 shape (N_patch, C_patch, ph, pw) 중 C, ph, pw
static int gPatchC = 192;
static int gPatchH = 24;
static int gPatchW = 24;

static const char* kE2InputName  = "patch";      // INPUT
static const char* kE2OutputName = "head_raw";

void init_engine2(const std::string& enginePath) {
    auto engineData = loadEngineFile(enginePath);

    gRuntime2 = createInferRuntime(gLogger);
    if (!gRuntime2) {
        std::cerr << "Failed to create TRT runtime for engine2\n";
        std::exit(1);
    }

    gEngine2 = gRuntime2->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!gEngine2) {
        std::cerr << "Failed to deserialize engine2\n";
        std::exit(1);
    }

    std::cout << "[engine2 IO tensors]\n";
    int nIO = gEngine2->getNbIOTensors();
    for (int i = 0; i < nIO; ++i) {
        const char* name = gEngine2->getIOTensorName(i);
        auto mode = gEngine2->getTensorIOMode(name);
        std::cout << "  " << i << ": " << name
                  << " (" << (mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << ")\n";
    }

    Dims patchDims = gEngine2->getTensorShape(kE2InputName);
    if (patchDims.nbDims == 4) {
        gPatchC = patchDims.d[1];
        gPatchH = patchDims.d[2];
        gPatchW = patchDims.d[3];
        std::cout << "engine2 patch shape: [N," << gPatchC
                  << "," << gPatchH << "," << gPatchW << "]\n";
    }

    gContext2 = gEngine2->createExecutionContext();
    if (!gContext2) {
        std::cerr << "Failed to create context for engine2\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamCreate(&gStream2));
}

void ensure_engine2_buffers(size_t inSize, size_t outSize) {
    if (cap_in2 < inSize) {
        if (d_in2) cudaFree(d_in2);
        CHECK_CUDA(cudaMalloc(&d_in2, inSize));
        cap_in2 = inSize;
    }
    if (cap_out2 < outSize) {
        if (d_out2) cudaFree(d_out2);
        CHECK_CUDA(cudaMalloc(&d_out2, outSize));
        cap_out2 = outSize;
    }
}

// ======================================================
// 3) engine1 실행 (GPU만 사용, feat/mask는 d_feat1/d_mask1에만 저장)
// ======================================================
void run_engine1(
    const Tensor& input,
    int B, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout
) {
    if (!gContext1) {
        std::cerr << "[run_engine1] engine1 not initialized!\n";
        std::exit(1);
    }

    Dims inDims;
    inDims.nbDims = 4;
    inDims.d[0] = B;
    inDims.d[1] = Cin;
    inDims.d[2] = Hin;
    inDims.d[3] = Win;

    if (!gContext1->setInputShape(kE1InputName, inDims)) {
        std::cerr << "Failed to set input shape for engine1\n";
        std::exit(1);
    }

    size_t inputSize = static_cast<size_t>(B) * Cin  * Hin  * Win  * sizeof(float);
    size_t featSize  = static_cast<size_t>(B) * Cout * Hout * Wout * sizeof(float);
    size_t maskSize  = static_cast<size_t>(B) * 1    * Hout * Wout * sizeof(float);

    ensure_engine1_buffers(inputSize, featSize, maskSize);

    CHECK_CUDA(cudaMemcpyAsync(
        d_input1,
        input.data.data(),
        inputSize,
        cudaMemcpyHostToDevice,
        gStream1));

    if (!gContext1->setInputTensorAddress(kE1InputName, d_input1) ||
        !gContext1->setOutputTensorAddress(kE1FeatName, d_feat1)  ||
        !gContext1->setOutputTensorAddress(kE1MaskName, d_mask1)) {
        std::cerr << "Failed to set tensor addresses for engine1\n";
        std::exit(1);
    }

    if (!gContext1->enqueueV3(gStream1)) {
        std::cerr << "Failed to enqueueV3 for engine1\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamSynchronize(gStream1));
}

// ======================================================
// 4) HeatMapParser CUDA 버전 (local maxima + per-image cap)
// ======================================================

static const int MAX_PATCHES_PER_IMAGE = 128;  // 필요시 조절

// 간단 sigmoid (mask가 0~1이면 그냥 identity로 써도 됨)
__device__ inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__
void heatmap_parser_kernel(
    const float* __restrict__ feat,   // [B,C,H,W]
    const float* __restrict__ mask,   // [B,1,H,W]
    int B, int C, int H, int W,
    float thresh,
    int stride,
    int patchC, int patchH, int patchW,
    float* __restrict__ patches,  // [Nmax, C, patchH, patchW]
    float* __restrict__ offsets,  // [Nmax, 5] (b,x1,y1,x2,y2)
    int*   __restrict__ counter_global,
    int*   __restrict__ counter_per_img,
    int    maxPatchesTotal
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // W
    int y = blockIdx.y * blockDim.y + threadIdx.y; // H
    int b = blockIdx.z;                            // B

    if (b >= B || y >= H || x >= W) return;

    int idx_mask = ((b * H + y) * W + x);
    float m = mask[idx_mask];

    // 필요하면 sigmoid
    // m = sigmoid_f(m);

    if (m <= thresh) return;

    // 3x3 local maximum 체크
    bool is_max = true;
    for (int dy = -1; dy <= 1 && is_max; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            if (dx == 0 && dy == 0) continue;
            int n_idx = ((b * H + yy) * W + xx);
            float nm = mask[n_idx];
            // nm = sigmoid_f(nm);
            if (nm > m) {
                is_max = false;
                break;
            }
        }
    }
    if (!is_max) return;

    // per-image patch 개수 제한
    int my_img_count = atomicAdd(&counter_per_img[b], 1);
    if (my_img_count >= MAX_PATCHES_PER_IMAGE) {
        return;
    }

    // global index는 연속되게
    int patch_idx = atomicAdd(counter_global, 1);
    if (patch_idx >= maxPatchesTotal) {
        return;
    }

    // center in network coordinate (YOLO 스타일)
    float cx = (x + 0.5f) * stride;
    float cy = (y + 0.5f) * stride;

    float box_w = patchW * stride;
    float box_h = patchH * stride;

    float x1 = cx - box_w * 0.5f;
    float y1 = cy - box_h * 0.5f;
    float x2 = cx + box_w * 0.5f;
    float y2 = cy + box_h * 0.5f;

    offsets[patch_idx * 5 + 0] = (float)b;
    offsets[patch_idx * 5 + 1] = x1;
    offsets[patch_idx * 5 + 2] = y1;
    offsets[patch_idx * 5 + 3] = x2;
    offsets[patch_idx * 5 + 4] = y2;

    int half_h = patchH / 2;
    int half_w = patchW / 2;

    int top  = y - half_h;
    int left = x - half_w;

    for (int c = 0; c < C; ++c) {
        for (int yy = 0; yy < patchH; ++yy) {
            int src_y = top + yy;
            if (src_y < 0) src_y = 0;
            else if (src_y >= H) src_y = H - 1;

            for (int xx = 0; xx < patchW; ++xx) {
                int src_x = left + xx;
                if (src_x < 0) src_x = 0;
                else if (src_x >= W) src_x = W - 1;

                int feat_idx =
                    ((b * C + c) * H + src_y) * W + src_x;
                int patch_idx_flat =
                    (((patch_idx * C + c) * patchH) + yy) * patchW + xx;

                patches[patch_idx_flat] = feat[feat_idx];
            }
        }
    }
}

// host wrapper
void heatmap_parser_cuda_forward(
    int B, int C, int H, int W,
    float thresh,
    int stride,
    Tensor& patches,  // out (host)
    Tensor& offsets   // out (host)
) {
    int maxPatches = B * MAX_PATCHES_PER_IMAGE;

    size_t patchesDevSize =
        (size_t)maxPatches * gPatchC * gPatchH * gPatchW * sizeof(float);
    size_t offsetsDevSize =
        (size_t)maxPatches * 5 * sizeof(float);

    float* d_patches = nullptr;
    float* d_offsets = nullptr;
    int*   d_counter_global = nullptr;
    int*   d_counter_per_img = nullptr;

    CHECK_CUDA(cudaMalloc(&d_patches, patchesDevSize));
    CHECK_CUDA(cudaMalloc(&d_offsets, offsetsDevSize));
    CHECK_CUDA(cudaMalloc(&d_counter_global, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counter_per_img, sizeof(int) * B));

    CHECK_CUDA(cudaMemset(d_counter_global, 0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_counter_per_img, 0, sizeof(int) * B));

    dim3 block(16, 16, 1);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              B);

    heatmap_parser_kernel<<<grid, block>>>(
        d_feat1, d_mask1,
        B, C, H, W,
        thresh,
        stride,
        gPatchC, gPatchH, gPatchW,
        d_patches,
        d_offsets,
        d_counter_global,
        d_counter_per_img,
        maxPatches
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    int h_count = 0;
    CHECK_CUDA(cudaMemcpy(&h_count, d_counter_global, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (h_count > maxPatches) h_count = maxPatches;

    patches.shape = { h_count, gPatchC, gPatchH, gPatchW };
    patches.data.resize((size_t)h_count * gPatchC * gPatchH * gPatchW);

    offsets.shape = { h_count, 5 };
    offsets.data.resize((size_t)h_count * 5);

    size_t patchesHostSize =
        (size_t)h_count * gPatchC * gPatchH * gPatchW * sizeof(float);
    size_t offsetsHostSize =
        (size_t)h_count * 5 * sizeof(float);

    CHECK_CUDA(cudaMemcpy(patches.data.data(), d_patches, patchesHostSize,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(offsets.data.data(), d_offsets, offsetsHostSize,
                          cudaMemcpyDeviceToHost));

    cudaFree(d_patches);
    cudaFree(d_offsets);
    cudaFree(d_counter_global);
    cudaFree(d_counter_per_img);
}

// ======================================================
// 5) engine2 실행 (TensorRT enqueueV3, N_patch > 128 이면 chunk)
// ======================================================
void run_engine2(
    const Tensor& patches,
    Tensor& head_raw
) {
    if (!gContext2) {
        std::cerr << "[run_engine2] engine2 not initialized!\n";
        std::exit(1);
    }

    int N_total = patches.shape[0];
    int C       = patches.shape[1];
    int ph      = patches.shape[2];
    int pw      = patches.shape[3];

    int M       = head_raw.shape[1];
    int no      = head_raw.shape[2];

    const int maxBatchTRT = 128;

    int offset = 0;
    while (offset < N_total) {
        int curN = std::min(maxBatchTRT, N_total - offset);

        Dims inDims;
        inDims.nbDims = 4;
        inDims.d[0] = curN;
        inDims.d[1] = C;
        inDims.d[2] = ph;
        inDims.d[3] = pw;

        if (!gContext2->setInputShape(kE2InputName, inDims)) {
            std::cerr << "Failed to set input shape for engine2\n";
            std::exit(1);
        }

        size_t inSize  = (size_t)curN * C * ph * pw * sizeof(float);
        size_t outSize = (size_t)curN * M * no * sizeof(float);

        ensure_engine2_buffers(inSize, outSize);

        const float* h_in_ptr =
            patches.data.data() + (size_t)offset * C * ph * pw;
        float* h_out_ptr =
            head_raw.data.data() + (size_t)offset * M * no;

        CHECK_CUDA(cudaMemcpyAsync(
            d_in2,
            h_in_ptr,
            inSize,
            cudaMemcpyHostToDevice,
            gStream2));

        if (!gContext2->setInputTensorAddress(kE2InputName, d_in2) ||
            !gContext2->setOutputTensorAddress(kE2OutputName, d_out2)) {
            std::cerr << "Failed to set tensor addresses for engine2\n";
            std::exit(1);
        }

        if (!gContext2->enqueueV3(gStream2)) {
            std::cerr << "Failed to enqueueV3 for engine2\n";
            std::exit(1);
        }

        CHECK_CUDA(cudaMemcpyAsync(
            h_out_ptr,
            d_out2,
            outSize,
            cudaMemcpyDeviceToHost,
            gStream2));

        CHECK_CUDA(cudaStreamSynchronize(gStream2));

        offset += curN;
    }
}

// ======================================================
// 6) ESOD 배치 추론 (HeatMapParser CUDA 버전 사용)
// ======================================================
void esod_infer_batch(
    const Tensor& img_batch,         // [B,3,H_in,W_in]
    int num_classes,
    std::vector<Detection>& detections  // out (image_id = batch index)
) {
    int B = img_batch.shape[0];

    int Cin = 3;
    int Hin = img_batch.shape[2];
    int Win = img_batch.shape[3];

    int C = gFeatC;
    int H = gFeatH;
    int W = gFeatW;

    // 1) backbone
    run_engine1(img_batch,
                B, Cin, Hin, Win,
                C, H, W);

    // 2) HeatMapParser (CUDA)
    Tensor patches;
    Tensor offsets;   // [N_patch, 5] (b, x1, y1, x2, y2)

    float heatmap_thresh = 0.5f;  // 필요시 0.6~0.7로 조절 가능
    int   stride         = 8;     // 1536 / 192

    heatmap_parser_cuda_forward(
        B, C, H, W,
        heatmap_thresh,
        stride,
        patches,
        offsets
    );

    int N_patch = patches.shape[0];
    if (N_patch == 0) {
        detections.clear();
        return;
    }

    // 3) head (TensorRT)
    int no = 5 + num_classes;
    int M  = 24*24*3 + 12*12*3 + 6*6*3;  // 2268

    Tensor head_raw({N_patch, M, no});
    run_engine2(patches, head_raw);

    // 4) HeadDecoder + NMS (CPU)
    // conf_low ~ conf_high 구간은 unknown(-1)로 라벨링
    float conf_low   = 0.30f;   // 이 아래는 버림
    float conf_high  = 0.40f;   // 이 이상은 known
    float nms_thresh = 0.50f;

    // unknown 정책 (esod_head_decoder.hpp에서 추가한 파라미터와 맞춰야 함)
    float unknown_iou = 0.30f;  // unknown이 known과 이 IoU 이상 겹치면 제거
    bool  nms_unknown = true;   // unknown끼리도 NMS 할지
    float nms_unknown_thresh = 0.35f;  // (추천) unknown끼리 더 강하게 제거

    EsodHeadDecoder decoder(conf_low, conf_high, nms_thresh,
                        num_classes, unknown_iou, nms_unknown, nms_unknown_thresh);

    std::vector<Detection> out_dets;
    decoder.decode(head_raw, offsets, out_dets);  // image_id = b index

    detections = out_dets;
}

// ======================================================
// 7) 전처리: letterbox
// ======================================================
void preprocess_frame_to_batch_letterbox(
    const cv::Mat& frame_bgr,
    int H_in, int W_in,
    Tensor& img_batch,   // [B,3,H_in,W_in]
    int b_idx,
    PreprocInfo& info
) {
    int orig_w = frame_bgr.cols;
    int orig_h = frame_bgr.rows;

    float r = std::min((float)W_in / (float)orig_w,
                       (float)H_in / (float)orig_h);
    int new_w = (int)std::round(orig_w * r);
    int new_h = (int)std::round(orig_h * r);

    cv::Mat resized;
    cv::resize(frame_bgr, resized, cv::Size(new_w, new_h));

    int pad_x = (W_in - new_w) / 2;
    int pad_y = (H_in - new_h) / 2;

    cv::Mat canvas(H_in, W_in, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    info.gain  = r;
    info.pad_x = pad_x;
    info.pad_y = pad_y;

    cv::Mat rgb, f32;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0f);

    int B = img_batch.shape[0];
    int C = 3;
    if (b_idx < 0 || b_idx >= B) {
        throw std::runtime_error("preprocess_frame_to_batch_letterbox: invalid b_idx");
    }

    int H = H_in;
    int W = W_in;
    size_t batch_offset = (size_t)b_idx * C * H * W;

    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 3; ++c) {
                size_t idx = batch_offset + c * H * W + y * W + x;
                img_batch.data[idx] = row[x][c];
            }
        }
    }
}

// ======================================================
// 8) (참고용) 후처리: letterbox 역변환 + 3클래스 draw (현재는 사용 X)
// ======================================================
void draw_detections_letterbox(
    cv::Mat& frame,
    const std::vector<Detection>& dets,
    const PreprocInfo& info
) {
    cv::Scalar colors[3] = {
        cv::Scalar(0, 255, 0),   // person
        cv::Scalar(255, 0, 0),   // bicycle
        cv::Scalar(0, 0, 255)    // car
    };

    for (const auto& d : dets) {
        int cls = d.cls;

        float x_net = d.x;
        float y_net = d.y;
        float w_net = d.w;
        float h_net = d.h;

        float x = (x_net - info.pad_x) / info.gain;
        float y = (y_net - info.pad_y) / info.gain;
        float w =  w_net / info.gain;
        float h =  h_net / info.gain;

        int x1 = (int)(x - w / 2.0f);
        int y1 = (int)(y - h / 2.0f);
        int x2 = (int)(x + w / 2.0f);
        int y2 = (int)(y + h / 2.0f);

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        cv::Scalar color = (0 <= cls && cls < 3)
            ? colors[cls]
            : cv::Scalar(0, 255, 255);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                      color, 2);

        std::string label;
        if (0 <= cls && cls < 3) {
            label = kThreeClassNames[cls];
        } else {
            label = "cls_" + std::to_string(cls);
        }

        char text[128];
        std::snprintf(text, sizeof(text), "%s %.2f", label.c_str(), d.score);

        int baseLine = 0;
        cv::getTextSize(text,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5, 1, &baseLine);

        int yy = std::max(0, y1 - 5);
        cv::putText(frame, text,
                    cv::Point(x1, yy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1);
    }
}

// ======================================================
// 8.5) SORT: Kalman + IoU + Hungarian (Python 코드 C++ 포팅)
// ======================================================

// ------------ IoU (xyxy) 이미 esod_core.hpp에 있음, 그대로 사용 ------------
// inline float iou_xyxy( ... )  // esod_core.hpp 정의 사용

struct SortBBox {
    float x1, y1, x2, y2;
    float score;
    int   cls;
};

struct TrackResult {
    float x1, y1, x2, y2;
    float score;
    int   cls;
    int   id;   // track id
};

// ------------------ 간단 KalmanFilter(7x4) 구현 ------------------

class KalmanBoxFilter {
public:
    KalmanBoxFilter() {
        // zero init
        for (int i=0;i<7;++i) {
            x_[i] = 0.0;
            for (int j=0;j<7;++j) {
                F_[i][j] = 0.0;
                P_[i][j] = 0.0;
                Q_[i][j] = 0.0;
            }
        }
        for (int i=0;i<4;++i) {
            for (int j=0;j<7;++j) H_[i][j] = 0.0;
            for (int j=0;j<4;++j) R_[i][j] = 0.0;
        }

        // F (7x7)
        // [1 0 0 0 1 0 0
        //  0 1 0 0 0 1 0
        //  0 0 1 0 0 0 1
        //  0 0 0 1 0 0 0
        //  0 0 0 0 1 0 0
        //  0 0 0 0 0 1 0
        //  0 0 0 0 0 0 1]
        for (int i=0;i<7;++i) F_[i][i] = 1.0;
        F_[0][4] = 1.0;
        F_[1][5] = 1.0;
        F_[2][6] = 1.0;

        // H (4x7)
        // [[1,0,0,0,0,0,0],
        //  [0,1,0,0,0,0,0],
        //  [0,0,1,0,0,0,0],
        //  [0,0,0,1,0,0,0]]
        for (int i=0;i<4;++i) {
            H_[i][i] = 1.0;
        }

        // R (4x4) ~ diag(1,1,10,10)
        for (int i=0;i<4;++i) R_[i][i] = 1.0;
        R_[2][2] = 10.0;
        R_[3][3] = 10.0;

        // P 초기화: 대략 diag(10,10,10,10,10000,10000,10000)
        for (int i=0;i<7;++i) {
            double val = (i>=4) ? 10000.0 : 10.0;
            P_[i][i] = val;
        }

        // Q 초기화: 대략 단위행렬 기반, 마지막 3원소는 작은 값
        for (int i=0;i<7;++i) {
            Q_[i][i] = 1.0;
        }
        // python: Q[-1,-1] *= 0.01; Q[4:,4:] *= 0.01
        Q_[6][6] *= 0.01;
        for (int i=4;i<7;++i) {
            for (int j=4;j<7;++j) {
                Q_[i][j] *= 0.01;
            }
        }
    }

    // bbox [x1,y1,x2,y2] → 상태 초기화 [x,y,s,r,0,0,0]
    void init(const float* bbox4) {
        double w = bbox4[2] - bbox4[0];
        double h = bbox4[3] - bbox4[1];
        double x = bbox4[0] + w * 0.5;
        double y = bbox4[1] + h * 0.5;
        double s = w * h;
        double r = (h > 1e-6) ? (w / h) : 1.0;

        x_[0] = x;
        x_[1] = y;
        x_[2] = s;
        x_[3] = r;
        x_[4] = 0.0;
        x_[5] = 0.0;
        x_[6] = 0.0;
    }

    // 예측: x, P
    void predict() {
        // python:
        // if (x[6] + x[2] <= 0): x[6] *= 0
        if (x_[6] + x_[2] <= 0.0) {
            x_[6] = 0.0;
        }

        // x = F * x
        double x_new[7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                x_new[i] += F_[i][j] * x_[j];
            }
        }
        for (int i=0;i<7;++i) x_[i] = x_new[i];

        // P = F P F^T + Q
        double FP[7][7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                for (int k=0;k<7;++k) {
                    FP[i][j] += F_[i][k] * P_[k][j];
                }
            }
        }
        double FPFt[7][7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                for (int k=0;k<7;++k) {
                    FPFt[i][j] += FP[i][k] * F_[j][k]; // FPF^T
                }
            }
        }
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                P_[i][j] = FPFt[i][j] + Q_[i][j];
            }
        }
    }

    // 관측 업데이트: bbox [x1,y1,x2,y2]
    void update(const float* bbox4) {
        // z = [x,y,s,r]
        double w = bbox4[2] - bbox4[0];
        double h = bbox4[3] - bbox4[1];
        double x = bbox4[0] + w * 0.5;
        double y = bbox4[1] + h * 0.5;
        double s = w * h;
        double r = (h > 1e-6) ? (w / h) : 1.0;

        double z[4] = { x, y, s, r };

        // y = z - H x
        double Hx[4] = {0};
        for (int i=0;i<4;++i) {
            for (int j=0;j<7;++j) {
                Hx[i] += H_[i][j] * x_[j];
            }
        }
        double y_vec[4];
        for (int i=0;i<4;++i) {
            y_vec[i] = z[i] - Hx[i];
        }

        // S = H P H^T + R (4x4)
        double HP[4][7] = {0};
        for (int i=0;i<4;++i) {
            for (int j=0;j<7;++j) {
                for (int k=0;k<7;++k) {
                    HP[i][j] += H_[i][k] * P_[k][j];
                }
            }
        }
        double S[4][4] = {0};
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                for (int k=0;k<7;++k) {
                    S[i][j] += HP[i][k] * H_[j][k]; // HP H^T
                }
                S[i][j] += R_[i][j];
            }
        }

        // S^-1 (4x4) : Gauss-Jordan
        double Sinv[4][4];
        invert4x4(S, Sinv);

        // K = P H^T S^-1 (7x4)
        double PHt[7][4] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<4;++j) {
                for (int k=0;k<7;++k) {
                    PHt[i][j] += P_[i][k] * H_[j][k];
                }
            }
        }
        double K[7][4] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<4;++j) {
                for (int k=0;k<4;++k) {
                    K[i][j] += PHt[i][k] * Sinv[k][j];
                }
            }
        }

        // x = x + K y
        double Ky[7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<4;++j) {
                Ky[i] += K[i][j] * y_vec[j];
            }
        }
        for (int i=0;i<7;++i) {
            x_[i] += Ky[i];
        }

        // P = (I - K H) P
        double KH[7][7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                for (int k=0;k<4;++k) {
                    KH[i][j] += K[i][k] * H_[k][j];
                }
            }
        }
        double I_KH[7][7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                double val = (i==j ? 1.0 : 0.0) - KH[i][j];
                I_KH[i][j] = val;
            }
        }
        double newP[7][7] = {0};
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                for (int k=0;k<7;++k) {
                    newP[i][j] += I_KH[i][k] * P_[k][j];
                }
            }
        }
        for (int i=0;i<7;++i) {
            for (int j=0;j<7;++j) {
                P_[i][j] = newP[i][j];
            }
        }
    }

    // 현재 상태 → bbox [x1,y1,x2,y2]
    void get_state(float* bbox4_out) const {
        double x = x_[0];
        double y = x_[1];
        double s = x_[2];
        double r = x_[3];
        double w = std::sqrt(std::max(0.0, s * r));
        double h = (r > 1e-6) ? (s / w) : 0.0;

        bbox4_out[0] = static_cast<float>(x - w * 0.5);
        bbox4_out[1] = static_cast<float>(y - h * 0.5);
        bbox4_out[2] = static_cast<float>(x + w * 0.5);
        bbox4_out[3] = static_cast<float>(y + h * 0.5);
    }

private:
    double x_[7];
    double P_[7][7];
    double F_[7][7];
    double Q_[7][7];
    double H_[4][7];
    double R_[4][4];

    static void invert4x4(const double A[4][4], double invA[4][4]) {
        // Gauss-Jordan
        double aug[4][8];
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                aug[i][j] = A[i][j];
            }
            for (int j=4;j<8;++j) {
                aug[i][j] = (j-4 == i) ? 1.0 : 0.0;
            }
        }

        for (int col=0; col<4; ++col) {
            // pivot
            int pivot = col;
            double max_val = std::fabs(aug[col][col]);
            for (int r=col+1; r<4; ++r) {
                double v = std::fabs(aug[r][col]);
                if (v > max_val) {
                    max_val = v;
                    pivot = r;
                }
            }
            if (max_val < 1e-12) {
                // singular; 그냥 identity에 가깝게 fallback
                for (int i=0;i<4;++i) {
                    for (int j=0;j<4;++j) {
                        invA[i][j] = (i==j ? 1.0 : 0.0);
                    }
                }
                return;
            }
            if (pivot != col) {
                for (int j=0;j<8;++j) {
                    std::swap(aug[col][j], aug[pivot][j]);
                }
            }

            // normalize row
            double diag = aug[col][col];
            for (int j=0;j<8;++j) aug[col][j] /= diag;

            // eliminate others
            for (int r=0;r<4;++r) {
                if (r == col) continue;
                double factor = aug[r][col];
                for (int j=0;j<8;++j) {
                    aug[r][j] -= factor * aug[col][j];
                }
            }
        }

        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                invA[i][j] = aug[i][j+4];
            }
        }
    }
};

// --------------- KalmanBoxTracker (Python 클래스 포트) ---------------

class KalmanBoxTracker {
public:
    static int global_count;
    static constexpr int kMaxID = 1000;

    KalmanBoxTracker(const float* bbox4) {
        kf_.init(bbox4);
        time_since_update_ = 0;

        id_ = global_count;                 // 0~999
        global_count = (global_count + 1) % kMaxID;

        hits_ = 0;
        hit_streak_ = 0;
        age_ = 0;
    }

    // z 업데이트
    void update(const float* bbox4) {
        time_since_update_ = 0;
        history_.clear();
        hits_ += 1;
        hit_streak_ += 1;
        kf_.update(bbox4);
    }

    // 예측: bbox 반환
    std::array<float,4> predict() {
        // if((kf.x[6]+kf.x[2])<=0) -> 내부에서 처리
        kf_.predict();
        age_ += 1;
        if (time_since_update_ > 0) {
            hit_streak_ = 0;
        }
        time_since_update_ += 1;

        float bbox4[4];
        kf_.get_state(bbox4);
        std::array<float,4> res = {bbox4[0], bbox4[1], bbox4[2], bbox4[3]};
        history_.push_back(res);
        return history_.back();
    }

    std::array<float,4> get_state() const {
        float bbox4[4];
        kf_.get_state(bbox4);
        return {bbox4[0], bbox4[1], bbox4[2], bbox4[3]};
    }

    int id() const { return id_; }
    int time_since_update() const { return time_since_update_; }
    int hit_streak() const { return hit_streak_; }

    void mark_deleted() {
        // 필요시 플래그 추가 가능
    }

private:
    KalmanBoxFilter kf_;
    int time_since_update_;
    int id_;
    int hits_;
    int hit_streak_;
    int age_;
    std::vector<std::array<float,4>> history_;
};

int KalmanBoxTracker::global_count = 0;

// ------------------- IoU 행렬 + Hungarian 매칭 -------------------

static void linear_assignment_hungarian(
    const std::vector<std::vector<double>>& cost_matrix,
    std::vector<std::pair<int,int>>& matches)
{
    int n = (int)cost_matrix.size();
    int m = (n > 0) ? (int)cost_matrix[0].size() : 0;
    if (n == 0 || m == 0) return;

    int N = std::max(n, m);
    // pad square
    std::vector<std::vector<double>> cost(N, std::vector<double>(N, 0.0));
    for (int i=0;i<N;++i){
        for (int j=0;j<N;++j){
            if (i<n && j<m) cost[i][j] = cost_matrix[i][j];
            else cost[i][j] = 0.0;
        }
    }

    // Hungarian (potential-based 구현)
    std::vector<double> u(N+1, 0.0), v(N+1, 0.0);
    std::vector<int> p(N+1, 0), way(N+1, 0);
    for (int i=1; i<=N; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(N+1, std::numeric_limits<double>::infinity());
        std::vector<char> used(N+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = std::numeric_limits<double>::infinity();
            for (int j=1; j<=N; ++j) {
                if (used[j]) continue;
                double cur = cost[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j=0; j<=N; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // p[j] = row matched with column j
    std::vector<int> assignment_row(n, -1);
    for (int j=1; j<=N; ++j) {
        int i = p[j];
        if (i >= 1 && i <= n && j >= 1 && j <= m) {
            assignment_row[i-1] = j-1;
        }
    }

    matches.clear();
    for (int i=0;i<n;++i) {
        if (assignment_row[i] >= 0) {
            matches.push_back({i, assignment_row[i]});
        }
    }
}

// dets: [N,4], trackers: [M,4]
static void associate_detections_to_trackers(
    const std::vector<std::array<float,4>>& detections,
    const std::vector<std::array<float,4>>& trackers,
    float iou_threshold,
    std::vector<std::pair<int,int>>& matches,
    std::vector<int>& unmatched_dets,
    std::vector<int>& unmatched_trks
) {
    int N = (int)detections.size();
    int M = (int)trackers.size();

    matches.clear();
    unmatched_dets.clear();
    unmatched_trks.clear();

    if (M == 0) {
        // 모든 detection unmatched
        unmatched_dets.resize(N);
        for (int i=0;i<N;++i) unmatched_dets[i] = i;
        return;
    }

    // IoU matrix [N,M]
    std::vector<std::vector<float>> iou_matrix(N, std::vector<float>(M, 0.0f));
    for (int i=0;i<N;++i) {
        float x1 = detections[i][0];
        float y1 = detections[i][1];
        float x2 = detections[i][2];
        float y2 = detections[i][3];
        for (int j=0;j<M;++j) {
            float x1b = trackers[j][0];
            float y1b = trackers[j][1];
            float x2b = trackers[j][2];
            float y2b = trackers[j][3];
            float iou = iou_xyxy(x1, y1, x2, y2, x1b, y1b, x2b, y2b);
            iou_matrix[i][j] = iou;
        }
    }

    // binary mask a: iou > threshold
    std::vector<std::vector<int>> a(N, std::vector<int>(M, 0));
    for (int i=0;i<N;++i) {
        for (int j=0;j<M;++j) {
            if (iou_matrix[i][j] > iou_threshold) {
                a[i][j] = 1;
            }
        }
    }

    auto max_in_row = [&](int i)->int {
        int s=0;
        for (int j=0;j<M;++j) s += a[i][j];
        return s;
    };
    auto max_in_col = [&](int j)->int {
        int s=0;
        for (int i=0;i<N;++i) s += a[i][j];
        return s;
    };

    int max_row = 0, max_col = 0;
    for (int i=0;i<N;++i) max_row = std::max(max_row, max_in_row(i));
    for (int j=0;j<M;++j) max_col = std::max(max_col, max_in_col(j));

    std::vector<std::pair<int,int>> raw_matches;

    if (N>0 && M>0 && max_row == 1 && max_col == 1) {
        // simple unique matching
        for (int i=0;i<N;++i) {
            for (int j=0;j<M;++j) {
                if (a[i][j]) {
                    raw_matches.push_back({i,j});
                }
            }
        }
    } else {
        // Hungarian on cost = 1 - IoU (minimize)
        std::vector<std::vector<double>> cost(N, std::vector<double>(M, 0.0));
        for (int i=0;i<N;++i) {
            for (int j=0;j<M;++j) {
                cost[i][j] = 1.0 - (double)iou_matrix[i][j];
            }
        }
        linear_assignment_hungarian(cost, raw_matches);
    }

    // unmatched 초기화
    std::vector<bool> det_used(N,false), trk_used(M,false);

    // iou < threshold 인 매칭은 버리기
    for (auto& m : raw_matches) {
        int di = m.first;
        int ti = m.second;
        if (di<0 || di>=N || ti<0 || ti>=M) continue;
        if (iou_matrix[di][ti] < iou_threshold) continue;
        matches.push_back({di,ti});
        det_used[di] = true;
        trk_used[ti] = true;
    }

    for (int i=0;i<N;++i) if (!det_used[i]) unmatched_dets.push_back(i);
    for (int j=0;j<M;++j) if (!trk_used[j]) unmatched_trks.push_back(j);
}

// --------------------- Sort (Python 클래스 포트) ---------------------

class Sort {
public:
    Sort(int max_age=10, int min_hits=3, float iou_threshold=0.3f)
        : max_age_(max_age),
          min_hits_(min_hits),
          iou_threshold_(iou_threshold),
          frame_count_(0) {}

    // dets: [x1,y1,x2,y2,score] + cls 는 따로
    std::vector<TrackResult> update(const std::vector<SortBBox>& dets) {
        frame_count_++;

        // 1) 기존 트랙으로부터 예측 위치 얻기
        int T = (int)trackers_.size();
        std::vector<std::array<float,4>> trks_pred;
        trks_pred.reserve(T);

        // 이거 추가
        std::vector<int> active_idx;
        active_idx.reserve(T);

        std::vector<int> to_del;
        to_del.reserve(T);

        for (int t=0; t<T; ++t) {
            std::array<float,4> pos = trackers_[t].predict();
            bool has_nan = false;
            for (int k=0;k<4;++k) {
                if (!std::isfinite(pos[k])) {
                    has_nan = true;
                    break;
                }
            }
            if (has_nan) {
                to_del.push_back(t);
            } else {
                trks_pred.push_back(pos);
                active_idx.push_back(t);   // <-- 이 트랙은 trackers_의 t번째
            }
        }
        // to_del 의 실제 인덱스를 맞추려면 조금 복잡하지만,
        // 여기서는 NaN 거의 없다고 보고, 단순 pop_back 형태로 처리해도 됨.
        // (완벽히 하려면 인덱스 매핑 필요)

        // 2D IoU 기반 할당
        std::vector<std::array<float,4>> dets_xyxy;
        dets_xyxy.reserve(dets.size());
        for (auto& d : dets) {
            dets_xyxy.push_back({d.x1, d.y1, d.x2, d.y2});
        }

        std::vector<std::pair<int,int>> matches;
        std::vector<int> unmatched_dets, unmatched_trks;
        associate_detections_to_trackers(
            dets_xyxy, trks_pred, iou_threshold_,
            matches, unmatched_dets, unmatched_trks
        );

        // 매칭된 트랙 업데이트
        for (auto& m : matches) {
            int di = m.first;   // detection index
            int ti = m.second;  // trks_pred index
            int tracker_idx = active_idx[ti];   // <-- 진짜 trackers_ 인덱스

            float box4[4] = {
                dets[di].x1, dets[di].y1, dets[di].x2, dets[di].y2
            };
            trackers_[tracker_idx].update(box4);
        }
        // 매칭 안된 detection -> 새 트랙
        for (int di : unmatched_dets) {
            float box4[4] = {
                dets[di].x1, dets[di].y1, dets[di].x2, dets[di].y2
            };
            trackers_.emplace_back(box4);
            int idx_new = (int)trackers_.size() - 1;
            // Python에선 id는 KalmanBoxTracker 내부에서 할당됨
            (void)idx_new;
        }

        // 리턴 + 오래된 트랙 제거
        std::vector<TrackResult> ret;
        int i = 0;
        for (int t = (int)trackers_.size() - 1; t >= 0; --t) {
            auto state = trackers_[t].get_state();
            int tsu = trackers_[t].time_since_update();
            int hs  = trackers_[t].hit_streak();
            int id  = trackers_[t].id();

            if ( (tsu < 1) &&
                 (hs >= min_hits_ || frame_count_ <= min_hits_) ) {
                // detection과 동일하게 score / cls를 넣어줘야 하는데,
                // Python 원본은 dets[i]에서 가져옴. 여기서는
                // 가장 최근 매칭된 detection의 score/cls를 그대로 쓴다고 가정.
                // (위 update에서 track에 score/cls를 직접 저장하려면
                //  SimpleTrack 구조를 따로 두면 됨. 하지만 간단하게
                //  여기서는 dets 중 iou max인 걸 찾아서 사용.)
                float best_iou = 0.0f;
                float best_score = 0.0f;
                int   best_cls = -1;
                for (auto& d : dets) {
                    float iou = iou_xyxy(
                        state[0], state[1], state[2], state[3],
                        d.x1, d.y1, d.x2, d.y2
                    );
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_score = d.score;
                        best_cls = d.cls;
                    }
                }

                TrackResult r;
                r.x1 = state[0];
                r.y1 = state[1];
                r.x2 = state[2];
                r.y2 = state[3];
                r.score = best_score;
                r.cls   = best_cls;
                // r.id    = id + 1; // Python에서 +1 해서 반환
                r.id = id;        // 0~999 그대로
                ret.push_back(r);
            }

            // 오래된 트랙 제거
            if (trackers_[t].time_since_update() > max_age_) {
                trackers_.erase(trackers_.begin() + t);
            }
            i++;
        }

        return ret;
    }

private:
    int max_age_;
    int min_hits_;
    float iou_threshold_;
    int frame_count_;   

    std::vector<KalmanBoxTracker> trackers_;
};

// 4개 영상 각각에 대한 Sort 인스턴스
static Sort gSortTrackers[4] = {
    Sort(), Sort(), Sort(), Sort()
};

// ======================================================
// 8.6) TRACK 결과 그리기 (원본 프레임 좌표계에서 ID 포함)
// ======================================================
void draw_tracks(
    cv::Mat& frame,
    const std::vector<TrackResult>& tracks
) {
    cv::Scalar colors[3] = {
        cv::Scalar(0, 255, 0),   // person
        cv::Scalar(255, 0, 0),   // bicycle
        cv::Scalar(0, 0, 255)    // car
    };

    for (const auto& t : tracks) {
        int cls = t.cls;

        int x1 = (int)std::round(t.x1);
        int y1 = (int)std::round(t.y1);
        int x2 = (int)std::round(t.x2);
        int y2 = (int)std::round(t.y2);

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        cv::Scalar color = (0 <= cls && cls < 3)
            ? colors[cls]
            : cv::Scalar(0, 255, 255);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                      color, 2);

        std::string cls_name;
        if (cls == -1) {
            cls_name = "unknown";
        } else if (0 <= cls && cls < 3) {
            cls_name = kThreeClassNames[cls];
        } else {
            cls_name = "cls_" + std::to_string(cls);
        }

        char text[128];
        std::snprintf(text, sizeof(text), "%s id=%d %.2f",
                      cls_name.c_str(), t.id, t.score);

        int baseLine = 0;
        cv::Size ts = cv::getTextSize(text,
                                      cv::FONT_HERSHEY_SIMPLEX,
                                      0.5, 1, &baseLine);
        int ty = std::max(0, y1 - 5);
        if (ty - ts.height < 0) ty = y1 + ts.height + 5;

        cv::putText(frame, text,
                    cv::Point(x1, ty),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1);
    }
}

// ======================================================
// 9) main: 비디오 4개 배치 추론 + SORT 트래킹
// ======================================================
int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "사용법: " << argv[0]
                  << " in0 in1 in2 in3 out0 out1 out2 out3\n";
        return 1;
    }

    std::array<std::string,4> in_paths  = {argv[1], argv[2], argv[3], argv[4]};
    std::array<std::string,4> out_paths = {argv[5], argv[6], argv[7], argv[8]};

    init_engine1("/data/siwoo/esod/esod/esod_stage1_fp16.engine");
    init_engine2("/data/siwoo/esod/esod/esod_head_fp16.engine");

    std::array<cv::VideoCapture,4> caps;
    std::array<cv::VideoWriter,4>  writers;
    std::array<bool,4>             alive = {true, true, true, true};

    int in_width  = 0;
    int in_height = 0;
    double fps = 0.0;

    for (int i = 0; i < 4; ++i) {
        caps[i].open(in_paths[i]);
        if (!caps[i].isOpened()) {
            std::cerr << "비디오 열기 실패: " << in_paths[i] << "\n";
            return 1;
        }

        int w = (int)caps[i].get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)caps[i].get(cv::CAP_PROP_FRAME_HEIGHT);
        double f = caps[i].get(cv::CAP_PROP_FPS);
        if (f <= 0) f = 25.0;

        if (i == 0) {
            in_width  = w;
            in_height = h;
            fps       = f;
        } else {
            if (w != in_width || h != in_height) {
                std::cerr << "모든 입력 영상의 해상도가 같아야 합니다. "
                          << "cam" << i << " 해상도: " << w << "x" << h << "\n";
                return 1;
            }
        }
    }

    std::cout << "input size: " << in_width << " x " << in_height
              << ", fps=" << fps << "\n";

    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    for (int i = 0; i < 4; ++i) {
        writers[i].open(out_paths[i], fourcc, fps, cv::Size(in_width, in_height));
        if (!writers[i].isOpened()) {
            std::cerr << "출력 비디오 열기 실패: " << out_paths[i] << "\n";
            return 1;
        }
    }

    const int H_in = 1536;
    const int W_in = 1536;
    int num_classes = 10;

    int global_frame_idx = 0;
    double total_infer_ms = 0.0;
    int total_batches = 0;
    int total_frames_processed = 0;

    std::array<PreprocInfo,4> preproc_infos;

    while (true) {
        std::vector<cv::Mat> frames(4);
        std::vector<int> batch2vid;
        batch2vid.reserve(4);

        for (int vid = 0; vid < 4; ++vid) {
            if (!alive[vid]) continue;

            cv::Mat f;
            if (!caps[vid].read(f)) {
                alive[vid] = false;
                continue;
            }
            frames[vid] = f;
            batch2vid.push_back(vid);
        }

        int B = (int)batch2vid.size();
        if (B == 0) break;
        total_frames_processed += B;

        Tensor img_batch({B, 3, H_in, W_in});
        for (int b = 0; b < B; ++b) {
            int vid = batch2vid[b];
            preprocess_frame_to_batch_letterbox(
                frames[vid], H_in, W_in, img_batch, b, preproc_infos[vid]);
        }

        std::vector<Detection> batch_dets;
        auto t0 = std::chrono::high_resolution_clock::now();
        esod_infer_batch(img_batch, num_classes, batch_dets);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_infer_ms += ms;
        total_batches  += 1;

        double ms_per_img = ms / B;
        std::cout << "batch " << total_batches
                  << " (B=" << B << ") infer time = "
                  << ms << " ms (" << ms_per_img << " ms / image), dets="
                  << batch_dets.size() << "\n";

        // 배치 → 각 비디오별 Detection 분리
        std::array<std::vector<Detection>,4> dets_per_video;
        for (const auto& d : batch_dets) {
            int b = d.image_id;
            if (b < 0 || b >= B) continue;
            int vid = batch2vid[b];

            int new_cls;
            if (d.cls == -1) {
                new_cls = -1;   // unknown 그대로 유지
            } else {
                new_cls = map_visdrone10_to3(d.cls);
                if (new_cls < 0) continue;  // 매핑 불가만 제거
            }

            Detection d2 = d;
            d2.image_id = vid;
            d2.cls = new_cls;

            dets_per_video[vid].push_back(d2);
        }

        // 각 비디오별 SORT 업데이트 + 그리기
        for (int vid = 0; vid < 4; ++vid) {
            if (!alive[vid]) continue;
            if (frames[vid].empty()) continue;

            const auto& info = preproc_infos[vid];

            // ESOD detection(네트워크 좌표) → 원본 좌표 SortBBox
            std::vector<SortBBox> dets_for_sort;
            dets_for_sort.reserve(dets_per_video[vid].size());
            for (const auto& d : dets_per_video[vid]) {
                float x_net = d.x;
                float y_net = d.y;
                float w_net = d.w;
                float h_net = d.h;

                float x = (x_net - info.pad_x) / info.gain;
                float y = (y_net - info.pad_y) / info.gain;
                float w =  w_net / info.gain;
                float h =  h_net / info.gain;

                float x1 = x - w * 0.5f;
                float y1 = y - h * 0.5f;
                float x2 = x + w * 0.5f;
                float y2 = y + h * 0.5f;

                SortBBox sb;
                sb.x1 = x1;
                sb.y1 = y1;
                sb.x2 = x2;
                sb.y2 = y2;
                sb.score = d.score;
                sb.cls   = d.cls;
                dets_for_sort.push_back(sb);
            }

            // SORT 업데이트
            std::vector<TrackResult> tracks = gSortTrackers[vid].update(dets_for_sort);

            // 트랙 결과 그리기
            draw_tracks(frames[vid], tracks);

            writers[vid].write(frames[vid]);
        }

        ++global_frame_idx;
    }

    std::cout << "processed batches: " << total_batches << "\n";
    std::cout << "processed frames (all videos total): "
              << total_frames_processed << "\n";

    if (total_batches > 0) {
        std::cout << "avg infer time per batch: "
                  << (total_infer_ms / total_batches) << " ms\n";
    }
    if (total_frames_processed > 0) {
        std::cout << "avg infer time per image: "
                  << (total_infer_ms / total_frames_processed) << " ms\n";
    }

    if (d_input1) cudaFree(d_input1);
    if (d_feat1)  cudaFree(d_feat1);
    if (d_mask1)  cudaFree(d_mask1);
    if (d_in2)    cudaFree(d_in2);
    if (d_out2)   cudaFree(d_out2);

    for (int i = 0; i < 4; ++i) {
        caps[i].release();
        writers[i].release();
    }

    return 0;
}
