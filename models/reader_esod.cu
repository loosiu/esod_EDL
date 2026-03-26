// reader_esod.cu
// 기존 reader.cpp(IPC/SHM/ICD) + ESOD 2-stage TRT 추론 통합본

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <csignal>
#include <condition_variable>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "icd.h"
#include "sort_safe.hpp"

// ===== ESOD headers (당신 프로젝트 경로에 맞게 include) =====
#include "esod_core.hpp"
#include "esod_head_decoder.hpp"

using namespace nvinfer1;

static std::atomic<int> g_active_infer{0};

#define READER_PRINT_DETS 1
#define READER_PRINT_TIME 1

// ================= TRT Logger =================
class Logger : public ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << "\n";
    }
} gLogger;

// ================= CUDA 체크 =================
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

// ================= 전처리(letterbox) =================
struct PreprocInfo {
    float gain = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

// ================= util =================
static inline float clampf(float v, float lo, float hi){
    return std::max(lo, std::min(v, hi));
}
static inline int clampi(int v, int lo, int hi){
    return std::max(lo, std::min(v, hi));
}

// ======================= ESOD 2-stage TensorRT 런타임 =======================

// ---------- engine file loader ----------
static std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[ESOD] Failed to open engine file: " << path << "\n";
        std::exit(1);
    }
    file.seekg(0, std::ifstream::end);
    size_t size = (size_t)file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), (std::streamsize)size);
    return buffer;
}

// ======================================================
// engine1 (backbone)
// ======================================================
static IRuntime*          gRuntime1  = nullptr;
static ICudaEngine*       gEngine1   = nullptr;
static IExecutionContext* gContext1  = nullptr;
static cudaStream_t       gStream1;

static float* d_input1 = nullptr;
static float* d_feat1  = nullptr;
static float* d_mask1  = nullptr;
static size_t cap_input1 = 0;
static size_t cap_feat1  = 0;
static size_t cap_mask1  = 0;

// engine1 output feat/mask shape
static int gFeatC = 192;
static int gFeatH = 192;
static int gFeatW = 192;

static const char* kE1InputName = "images";
static const char* kE1FeatName  = "feat_stage1";
static const char* kE1MaskName  = "mask";

static void ensure_engine1_buffers(size_t inputSize, size_t featSize, size_t maskSize) {
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

static void init_engine1(const std::string& enginePath) {
    auto engineData = loadEngineFile(enginePath);

    initLibNvInferPlugins(&gLogger,"");

    gRuntime1 = createInferRuntime(gLogger);
    if (!gRuntime1) {
        std::cerr << "[ESOD] Failed to create TRT runtime for engine1\n";
        std::exit(1);
    }

    gEngine1 = gRuntime1->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!gEngine1) {
        std::cerr << "[ESOD] Failed to deserialize engine1\n";
        std::exit(1);
    }

    gContext1 = gEngine1->createExecutionContext();
    if (!gContext1) {
        std::cerr << "[ESOD] Failed to create context for engine1\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamCreate(&gStream1));

    // 가능하면 output tensor shape를 엔진에서 읽어서 세팅(정적 shape라면)
    Dims featDims = gEngine1->getTensorShape(kE1FeatName);
    // 보통 [B,C,H,W]
    if (featDims.nbDims == 4) {
        gFeatC = featDims.d[1];
        gFeatH = featDims.d[2];
        gFeatW = featDims.d[3];
        std::cout << "[ESOD] engine1 feat shape: [B," << gFeatC << "," << gFeatH << "," << gFeatW << "]\n";
    } else {
        std::cout << "[ESOD] engine1 featDims.nbDims=" << featDims.nbDims
                  << " (런타임에서 setInputShape 후 context getTensorShape로 확인 권장)\n";
    }
}

// engine1 실행: input(host Tensor) -> feat/mask는 GPU(d_feat1/d_mask1)에 남겨둠
static void run_engine1(
    const Tensor& input, // [B,3,H,W], host float
    int B, int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout
) {
    Dims inDims;
    inDims.nbDims = 4;
    inDims.d[0] = B;
    inDims.d[1] = Cin;
    inDims.d[2] = Hin;
    inDims.d[3] = Win;

    if (!gContext1->setInputShape(kE1InputName, inDims)) {
        std::cerr << "[ESOD] Failed to set input shape for engine1\n";
        std::exit(1);
    }

    size_t inputSize = (size_t)B * Cin  * Hin  * Win  * sizeof(float);
    size_t featSize  = (size_t)B * Cout * Hout * Wout * sizeof(float);
    size_t maskSize  = (size_t)B * 1    * Hout * Wout * sizeof(float);

    ensure_engine1_buffers(inputSize, featSize, maskSize);

    CHECK_CUDA(cudaMemcpyAsync(
        d_input1, input.data.data(), inputSize,
        cudaMemcpyHostToDevice, gStream1));

    if (!gContext1->setInputTensorAddress(kE1InputName, d_input1) ||
        !gContext1->setOutputTensorAddress(kE1FeatName, d_feat1)  ||
        !gContext1->setOutputTensorAddress(kE1MaskName, d_mask1)) {
        std::cerr << "[ESOD] Failed to set tensor addresses for engine1\n";
        std::exit(1);
    }

    if (!gContext1->enqueueV3(gStream1)) {
        std::cerr << "[ESOD] Failed to enqueueV3 for engine1\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamSynchronize(gStream1));

    // 동적 엔진일 경우 context에서 실제 output shape 확인 가능
    Dims realFeat = gContext1->getTensorShape(kE1FeatName);
    if (realFeat.nbDims == 4) {
        gFeatC = realFeat.d[1];
        gFeatH = realFeat.d[2];
        gFeatW = realFeat.d[3];
    }
}

// ======================================================
// engine2 (head)
// ======================================================
static IRuntime*          gRuntime2  = nullptr;
static ICudaEngine*       gEngine2   = nullptr;
static IExecutionContext* gContext2  = nullptr;
static cudaStream_t       gStream2;

static float* d_in2  = nullptr;
static float* d_out2 = nullptr;
static size_t cap_in2  = 0;
static size_t cap_out2 = 0;

// patch shape: [N,C,ph,pw]
static int gPatchC = 192;
static int gPatchH = 24;
static int gPatchW = 24;

static const char* kE2InputName  = "patch";
static const char* kE2OutputName = "head_raw";

static void ensure_engine2_buffers(size_t inSize, size_t outSize) {
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

static void init_engine2(const std::string& enginePath) {
    auto engineData = loadEngineFile(enginePath);

    gRuntime2 = createInferRuntime(gLogger);
    if (!gRuntime2) {
        std::cerr << "[ESOD] Failed to create TRT runtime for engine2\n";
        std::exit(1);
    }

    gEngine2 = gRuntime2->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!gEngine2) {
        std::cerr << "[ESOD] Failed to deserialize engine2\n";
        std::exit(1);
    }

    gContext2 = gEngine2->createExecutionContext();
    if (!gContext2) {
        std::cerr << "[ESOD] Failed to create context for engine2\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamCreate(&gStream2));

    Dims patchDims = gEngine2->getTensorShape(kE2InputName);
    if (patchDims.nbDims == 4) {
        gPatchC = patchDims.d[1];
        gPatchH = patchDims.d[2];
        gPatchW = patchDims.d[3];
        std::cout << "[ESOD] engine2 patch shape: [N," << gPatchC << "," << gPatchH << "," << gPatchW << "]\n";
    }
}

// ======================================================
// HeatMapParser CUDA (feat/mask GPU -> patches/offsets host)
// ======================================================
static const int MAX_PATCHES_PER_IMAGE = 128;

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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= B || y >= H || x >= W) return;

    int idx_mask = ((b * H + y) * W + x);
    float m = mask[idx_mask];
    if (m <= thresh) return;

    // 3x3 local max
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
            if (nm > m) { is_max = false; break; }
        }
    }
    if (!is_max) return;

    int my_img_count = atomicAdd(&counter_per_img[b], 1);
    if (my_img_count >= MAX_PATCHES_PER_IMAGE) return;

    int patch_idx = atomicAdd(counter_global, 1);
    if (patch_idx >= maxPatchesTotal) return;

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

static void heatmap_parser_cuda_forward(
    int B, int C, int H, int W,
    float thresh,
    int stride,
    Tensor& patches,  // out host
    Tensor& offsets   // out host
) {
    int maxPatches = B * MAX_PATCHES_PER_IMAGE;

    size_t patchesDevSize = (size_t)maxPatches * gPatchC * gPatchH * gPatchW * sizeof(float);
    size_t offsetsDevSize = (size_t)maxPatches * 5 * sizeof(float);

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
    CHECK_CUDA(cudaMemcpy(&h_count, d_counter_global, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_count > maxPatches) h_count = maxPatches;

    patches.shape = { h_count, gPatchC, gPatchH, gPatchW };
    patches.data.resize((size_t)h_count * gPatchC * gPatchH * gPatchW);

    offsets.shape = { h_count, 5 };
    offsets.data.resize((size_t)h_count * 5);

    size_t patchesHostSize = (size_t)h_count * gPatchC * gPatchH * gPatchW * sizeof(float);
    size_t offsetsHostSize = (size_t)h_count * 5 * sizeof(float);

    CHECK_CUDA(cudaMemcpy(patches.data.data(), d_patches, patchesHostSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(offsets.data.data(), d_offsets, offsetsHostSize, cudaMemcpyDeviceToHost));

    cudaFree(d_patches);
    cudaFree(d_offsets);
    cudaFree(d_counter_global);
    cudaFree(d_counter_per_img);
}

// ======================================================
// engine2 실행: patches(host) -> head_raw(host)
// ======================================================
static void run_engine2(const Tensor& patches, Tensor& head_raw) {
    int N_total = patches.shape[0];
    int C       = patches.shape[1];
    int ph      = patches.shape[2];
    int pw      = patches.shape[3];

    int M  = head_raw.shape[1];
    int no = head_raw.shape[2];

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
            std::cerr << "[ESOD] Failed to set input shape for engine2\n";
            std::exit(1);
        }

        size_t inSize  = (size_t)curN * C * ph * pw * sizeof(float);
        size_t outSize = (size_t)curN * M * no * sizeof(float);

        ensure_engine2_buffers(inSize, outSize);

        const float* h_in_ptr = patches.data.data() + (size_t)offset * C * ph * pw;
        float* h_out_ptr      = head_raw.data.data() + (size_t)offset * M * no;

        CHECK_CUDA(cudaMemcpyAsync(d_in2, h_in_ptr, inSize, cudaMemcpyHostToDevice, gStream2));

        if (!gContext2->setInputTensorAddress(kE2InputName, d_in2) ||
            !gContext2->setOutputTensorAddress(kE2OutputName, d_out2)) {
            std::cerr << "[ESOD] Failed to set tensor addresses for engine2\n";
            std::exit(1);
        }

        if (!gContext2->enqueueV3(gStream2)) {
            std::cerr << "[ESOD] Failed to enqueueV3 for engine2\n";
            std::exit(1);
        }

        CHECK_CUDA(cudaMemcpyAsync(h_out_ptr, d_out2, outSize, cudaMemcpyDeviceToHost, gStream2));
        CHECK_CUDA(cudaStreamSynchronize(gStream2));

        offset += curN;
    }
}

// ======================================================
// ESOD single-image infer: cv::Mat(BGR 1080x1920) -> vector<xyxy dets on original coords>
// ======================================================

// reader.cpp 스타일 Detection (xyxy)
struct DetXYXY {
    int cls;
    float score;
    float x1,y1,x2,y2;
};

// letterbox preprocess: BGR -> RGB float(0~1), CHW, 1536x1536
static void preprocess_letterbox(
    const cv::Mat& frame_bgr,
    int H_in, int W_in,
    Tensor& img_batch, // shape [1,3,H_in,W_in]
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

    img_batch.shape = {1, 3, H_in, W_in};
    img_batch.data.resize((size_t)1 * 3 * H_in * W_in);

    int H = H_in, W = W_in;
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            // CHW
            img_batch.data[0 * (size_t)H * W + (size_t)y * W + x] = row[x][0];
            img_batch.data[1 * (size_t)H * W + (size_t)y * W + x] = row[x][1];
            img_batch.data[2 * (size_t)H * W + (size_t)y * W + x] = row[x][2];
        }
    }
}

// ESOD 출력 Detection(네트워크 좌표 x,y,w,h) -> 원본 좌표 xyxy 변환
static DetXYXY net_to_orig_xyxy(
    const Detection& d,           // esod_core.hpp 의 Detection 가정: (x,y,w,h,score,cls,image_id)
    const PreprocInfo& info,
    int orig_w, int orig_h
) {
    // 네트워크 좌표 (letterbox canvas 기준)
    float x_net = d.x;
    float y_net = d.y;
    float w_net = d.w;
    float h_net = d.h;

    // canvas -> 원본 역변환
    float x = (x_net - info.pad_x) / info.gain;
    float y = (y_net - info.pad_y) / info.gain;
    float w =  w_net / info.gain;
    float h =  h_net / info.gain;

    float x1 = x - w * 0.5f;
    float y1 = y - h * 0.5f;
    float x2 = x + w * 0.5f;
    float y2 = y + h * 0.5f;

    x1 = clampf(x1, 0.f, (float)orig_w);
    y1 = clampf(y1, 0.f, (float)orig_h);
    x2 = clampf(x2, 0.f, (float)orig_w);
    y2 = clampf(y2, 0.f, (float)orig_h);

    DetXYXY o;
    o.cls = d.cls;
    o.score = d.score;
    o.x1=x1; o.y1=y1; o.x2=x2; o.y2=y2;
    return o;
}

static std::vector<DetXYXY> esod_infer_one(
    const cv::Mat& frame_bgr,
    int num_classes,
    float conf_low,
    float conf_high,
    float nms_thresh,
    float heatmap_thresh,
    int stride
) {
#if READER_PRINT_TIME
    auto t_all0 = std::chrono::steady_clock::now();
#endif

    // 1) preprocess
    const int H_in = 1536, W_in = 1536;
    Tensor img_batch;
    PreprocInfo pinfo;
    preprocess_letterbox(frame_bgr, H_in, W_in, img_batch, pinfo);

    int B = 1;
    int Cin = 3;
    int Hin = H_in;
    int Win = W_in;

    // 2) backbone
    int C = gFeatC, H = gFeatH, W = gFeatW;
    run_engine1(img_batch, B, Cin, Hin, Win, C, H, W);

    // (정말로 H,W가 192가 아니라면 stride도 바뀌어야 함)
    // 기본: stride = 1536 / gFeatH (예: 1536/192=8)
    if (gFeatH > 0) stride = (int)std::lround((double)H_in / (double)gFeatH);

    // 3) heatmap parser (CUDA)
    Tensor patches, offsets; // offsets: [N,5] (b,x1,y1,x2,y2) in net coord
    heatmap_parser_cuda_forward(B, C, gFeatH, gFeatW, heatmap_thresh, stride, patches, offsets);

    int N_patch = patches.shape.empty() ? 0 : patches.shape[0];
    if (N_patch == 0) return {};

    // 4) head
    int no = 5 + num_classes;
    int M  = 24*24*3 + 12*12*3 + 6*6*3; // demo 고정값 (당신 head에 맞게 확인!)
    Tensor head_raw({N_patch, M, no});
    run_engine2(patches, head_raw);

    // 5) decode + NMS (CPU)
    // unknown 정책도 원하면 여기서 사용(아래는 demo 기본값)
    float unknown_iou = 0.30f;
    bool  nms_unknown = true;
    float nms_unknown_thresh = 0.35f;

    EsodHeadDecoder decoder(conf_low, conf_high, nms_thresh,
                            num_classes, unknown_iou, nms_unknown, nms_unknown_thresh);

    std::vector<Detection> out_dets; // esod_core.hpp Detection
    decoder.decode(head_raw, offsets, out_dets); // image_id=0

    // 6) net->orig xyxy
    std::vector<DetXYXY> dets;
    dets.reserve(out_dets.size());
    for (auto& d : out_dets) {
        if (d.image_id != 0) continue;
        dets.push_back(net_to_orig_xyxy(d, pinfo, frame_bgr.cols, frame_bgr.rows));
    }

#if READER_PRINT_TIME
    auto t_all1 = std::chrono::steady_clock::now();
    double all_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
    std::cerr << "[Reader][TIME] ESOD_total(ms)=" << all_ms
              << "  patches=" << N_patch
              << "  dets=" << dets.size()
              << "  stride=" << stride
              << " feat=" << gFeatC << "x" << gFeatH << "x" << gFeatW
              << "\n";
#endif

    return dets;
}

// ===================== ReaderApp (기존 구조 유지) =====================
class ReaderApp{
public:
    static inline std::atomic<bool> running{true};

    ReaderApp(std::string e1, std::string e2,
              float confLow, float confHigh, float nmsThr,
              float heatmapThr,
              int maxAge,int minHits,double iouThr)
        : engine1Path(std::move(e1)), engine2Path(std::move(e2)),
          conf_low(confLow), conf_high(confHigh), nms_thresh(nmsThr),
          heatmap_thresh(heatmapThr),
          sortMaxAge(maxAge), sortMinHits(minHits), sortIouThr(iouThr) {}

    bool init(){
        // SHM open
        shm_fd.assign(SHM_COUNT,-1);
        shm_ptr.assign(SHM_COUNT,nullptr);
        for(int i=0;i<SHM_COUNT;++i){
            shm_fd[i]=shm_open(SHM_NAMES[i],O_RDWR,0666);
            if(shm_fd[i]<0){ perror("shm_open"); return false; }
            void* p=mmap(nullptr,SHM_SIZE,PROT_READ,MAP_SHARED,shm_fd[i],0);
            if(p==MAP_FAILED){ perror("mmap"); return false; }
            shm_ptr[i]=(uint8_t*)p;
        }

        // init ESOD engines
        init_engine1(engine1Path);
        init_engine2(engine2Path);

        // per-cam SORT 초기화(기존 reader.cpp 스타일: class별 sorter)
        // num_classes는 ESOD head decoder에 넣는 값과 동일해야 함
        numCls = esod_num_classes;

        for(int cam=1; cam<=4; ++cam){
            sorters[cam].clear();
            sorters[cam].reserve(numCls);
            for(int c=0;c<numCls;++c) sorters[cam].emplace_back(sortMaxAge,sortMinHits,sortIouThr);
            hasNew[cam].store(false);
        }

        return connect9000();
    }

    bool start5555(){
        trigListen=socket(AF_INET,SOCK_STREAM,0);
        if(trigListen<0){ perror("socket"); return false; }
        int opt=1; setsockopt(trigListen,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));

        sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(W2RPORT); a.sin_addr.s_addr=INADDR_ANY;
        if(bind(trigListen,(sockaddr*)&a,sizeof(a))<0){ perror("bind"); return false; }
        if(listen(trigListen,1)<0){ perror("listen"); return false; }

        std::cout<<"[Reader] waiting writer trigger on "<<W2RPORT<<"...\n";
        trigSock=accept(trigListen,nullptr,nullptr);
        if(trigSock<0){ perror("accept"); return false; }
        std::cout<<"[Reader] writer trigger connected.\n";
        return true;
    }

    void run(){
        for(int cam=1;cam<=4;++cam) workers.emplace_back([&,cam](){ camWorker(cam); });
        trigLoop(); // main thread
        for(auto& t:workers) if(t.joinable()) t.join();
    }

    void cleanup(){
        running.store(false);
        if(trigSock>=0){ shutdown(trigSock,SHUT_RDWR); close(trigSock); }
        if(trigListen>=0){ shutdown(trigListen,SHUT_RDWR); close(trigListen); }
        if(resSock>=0){ shutdown(resSock,SHUT_RDWR); close(resSock); }

        for(int i=0;i<SHM_COUNT;++i){
            if(shm_ptr[i]) munmap(shm_ptr[i],SHM_SIZE);
            if(shm_fd[i]>=0) close(shm_fd[i]);
        }

        if (d_input1) cudaFree(d_input1);
        if (d_feat1)  cudaFree(d_feat1);
        if (d_mask1)  cudaFree(d_mask1);
        if (d_in2)    cudaFree(d_in2);
        if (d_out2)   cudaFree(d_out2);

        if (gStream1) cudaStreamDestroy(gStream1);
        if (gStream2) cudaStreamDestroy(gStream2);

        if (gContext1) { gContext1->destroy(); gContext1=nullptr; }
        if (gContext2) { gContext2->destroy(); gContext2=nullptr; }

        if (gEngine1) { gEngine1->destroy(); gEngine1=nullptr; }
        if (gEngine2) { gEngine2->destroy(); gEngine2=nullptr; }

        if (gRuntime1) { gRuntime1->destroy(); gRuntime1=nullptr; }
        if (gRuntime2) { gRuntime2->destroy(); gRuntime2=nullptr; }
    }

    static void onSignal(int){ running.store(false); }

private:
    // ESOD engine paths
    std::string engine1Path;
    std::string engine2Path;

    // ESOD params
    float conf_low = 0.30f;
    float conf_high = 0.40f;
    float nms_thresh = 0.50f;
    float heatmap_thresh = 0.50f;
    int   esod_num_classes = 10;
    int   stride = 8;

    // SORT params
    int sortMaxAge, sortMinHits;
    double sortIouThr;

    int numCls=0;

    std::vector<int> shm_fd;
    std::vector<uint8_t*> shm_ptr;

    int trigListen=-1, trigSock=-1;
    int resSock=-1;

    std::mutex mtx[7];
    std::condition_variable cv[7];
    std::atomic<bool> hasNew[7];

    std::vector<Sort> sorters[7];
    std::vector<std::thread> workers;

    bool connect9000(){
        resSock=socket(AF_INET,SOCK_STREAM,0);
        if(resSock<0){ perror("socket"); return false; }

        sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(R2WPORT);
        inet_pton(AF_INET,"127.0.0.1",&a.sin_addr);

        while(running.load()){
            if(connect(resSock,(sockaddr*)&a,sizeof(a))==0){
                std::cout<<"[Reader] connected writer result on "<<R2WPORT<<"\n";
                return true;
            }
            std::cerr<<"[Reader] connect 9000 retry...\n";
            sleep(1);
        }
        return false;
    }

    static bool sendAll(int fd,const uint8_t* p,size_t n){
        while(n>0){
            ssize_t s=send(fd,p,n,0);
            if(s<=0) return false;
            p+= (size_t)s; n-= (size_t)s;
        }
        return true;
    }

    void sendICD(uint8_t cmd, const std::vector<uint8_t>& payload){
        uint8_t len=(uint8_t)payload.size();
        size_t total=(size_t)(ICD_END + len);
        std::vector<uint8_t> pkt(total,0);
        pkt[ICD_START]=START_ICD;
        pkt[ICD_CMD]=cmd;
        pkt[ICD_LENGTH]=len;
        if(len>0) std::memcpy(&pkt[ICD_DATA], payload.data(), len);

        size_t chk_idx=(size_t)(ICD_CHECKSUM + len - 1);
        size_t end_idx=(size_t)(ICD_END + len - 1);
        pkt[chk_idx]=calcICDChecksum(pkt[ICD_CMD],pkt[ICD_LENGTH],&pkt[ICD_DATA]);
        pkt[end_idx]=END_ICD;

        (void)sendAll(resSock,pkt.data(),pkt.size());
    }

    void camWorker(int cam_id){
        while(running.load()){
            {
                std::unique_lock<std::mutex> lk(mtx[cam_id]);
                cv[cam_id].wait_for(lk,std::chrono::milliseconds(100),
                                    [&](){return !running.load() || hasNew[cam_id].load();});
                if(!running.load()) break;
                hasNew[cam_id].store(false);
            }

            // SHM -> frame copy
            cv::Mat shm(1080,1920,CV_8UC3, shm_ptr[cam_id]);
            cv::Mat frame=shm.clone();

            auto t0 = std::chrono::steady_clock::now();
            int cur = ++g_active_infer;
            std::cerr << "[Reader][BEGIN] cam=" << cam_id
                      << " active=" << cur
                      << " tid=" << std::this_thread::get_id() << "\n";

            // ===== ESOD inference =====
            auto dets = esod_infer_one(frame, esod_num_classes, conf_low, conf_high,
                                       nms_thresh, heatmap_thresh, stride);

#if READER_PRINT_DETS
            std::cerr << "\n[Reader][DETS] cam_id=" << cam_id
                      << " dets=" << dets.size() << "\n";
            size_t k = std::min<size_t>(dets.size(), 10);
            for (size_t i=0;i<k;++i){
                auto& d=dets[i];
                std::cerr<<"  ["<<i<<"] cls="<<d.cls<<" score="<<d.score
                         <<" box=("<<d.x1<<","<<d.y1<<","<<d.x2<<","<<d.y2<<")\n";
            }
#endif

            cur = --g_active_infer;
            auto t1 = std::chrono::steady_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cerr << "[Reader][END]   cam=" << cam_id
                      << " active=" << cur
                      << " wall(ms)=" << wall_ms << "\n";

            // ===== class-wise sort (기존 reader.cpp 방식 유지) =====
            std::vector<std::vector<DetXYXY>> detByCls(numCls);
            for(auto& d: dets){
                if(d.cls>=0 && d.cls<numCls) detByCls[d.cls].push_back(d);
                // unknown(cls=-1) 처리 원하면 여기 분기 추가
            }

            struct Out{int id,cls; float conf; float x1,y1,x2,y2;};
            std::vector<Out> outs; outs.reserve(64);

            for(int c=0;c<numCls;++c){
                std::vector<SortBBox> sd; sd.reserve(detByCls[c].size());
                for(auto& d: detByCls[c]){
                    sd.push_back(SortBBox{d.x1,d.y1,d.x2,d.y2,d.score});
                }
                auto tr = sorters[cam_id][c].update(sd);

                for(auto& t: tr){
                    // conf 매칭: 가장 겹치는 det score
                    float bestI=0.f, bestS=0.f;
                    DetXYXY td{c,0.f,(float)t.x1,(float)t.y1,(float)t.x2,(float)t.y2};
                    for(auto& d: detByCls[c]){
                        // IoU
                        float xx1=std::max(td.x1,d.x1), yy1=std::max(td.y1,d.y1);
                        float xx2=std::min(td.x2,d.x2), yy2=std::min(td.y2,d.y2);
                        float w=std::max(0.f,xx2-xx1), h=std::max(0.f,yy2-yy1);
                        float inter=w*h;
                        float areaA=std::max(0.f,td.x2-td.x1)*std::max(0.f,td.y2-td.y1);
                        float areaB=std::max(0.f,d.x2-d.x1)*std::max(0.f,d.y2-d.y1);
                        float uni=areaA+areaB-inter;
                        float iou=(uni>0)?(inter/uni):0.f;
                        if(iou>bestI){ bestI=iou; bestS=d.score; }
                    }
                    int uniqId = (c+1)*10000 + t.id;
                    outs.push_back({uniqId,c,bestS,(float)t.x1,(float)t.y1,(float)t.x2,(float)t.y2});
                }
            }

            std::sort(outs.begin(),outs.end(),[](auto& a,auto& b){return a.conf>b.conf;});
            if(outs.size()>20) outs.resize(20); // ICD 255 제한 고려

#if READER_PRINT_DETS
            if (!outs.empty()) {
                std::cerr << "\n[Reader][SEND] cam_id=" << cam_id
                          << " tracks=" << outs.size() << "\n";
                for (size_t i=0;i<outs.size();++i){
                    auto& o=outs[i];
                    std::cerr<<"  ["<<i<<"] box_id="<<o.id<<" cls="<<o.cls<<" conf="<<o.conf
                             <<" box=("<<o.x1<<","<<o.y1<<","<<o.x2<<","<<o.y2<<")\n";
                }
            } else {
                std::cerr << "\n[Reader][SEND] cam_id=" << cam_id << " tracks=0\n";
            }
#endif

            // ===== JSON payload build (<=255 bytes) =====
            const size_t MAX_BYTES = 255;
            auto clamp_i = [&](float v, int lo, int hi)->int{
                if (!std::isfinite(v)) v = (float)lo;
                int x = (int)std::lround(v);
                if (x < lo) x = lo;
                if (x > hi) x = hi;
                return x;
            };

            std::string js;
            js.reserve(256);
            js += "{\"cam_id\":";
            js += std::to_string(cam_id);
            js += ",\"detections\":[";

            int used = 0;
            for (size_t i=0;i<outs.size();++i) {
                const auto& o = outs[i];

                int x1 = clamp_i(o.x1, 0, 1920);
                int y1 = clamp_i(o.y1, 0, 1080);
                int x2 = clamp_i(o.x2, 0, 1920);
                int y2 = clamp_i(o.y2, 0, 1080);
                if (x2 <= x1 || y2 <= y1) continue;

                float conf = std::min(std::max(o.conf, 0.f), 1.f);

                char one[200];
                int n = std::snprintf(
                    one, sizeof(one),
                    "{\"box_id\":%d,\"class_id\":%d,\"confidence\":%.3f,\"box\":[%d,%d,%d,%d]}",
                    o.id, o.cls, (double)conf, x1, y1, x2, y2
                );
                if (n <= 0) continue;

                size_t need = (size_t)n + (used > 0 ? 1 : 0);
                if (js.size() + need + 2 > MAX_BYTES) break;

                if (used > 0) js.push_back(',');
                js.append(one, (size_t)n);
                used++;
            }

            js += "]}";

#if READER_PRINT_DETS
            std::cerr << "[Reader][SEND_JSON] cam_id=" << cam_id
                      << " det_sent=" << used
                      << " det_total=" << outs.size()
                      << " json_bytes=" << js.size() << "\n";
#endif

            std::vector<uint8_t> payload;
            payload.insert(payload.end(), js.begin(), js.end());
            sendICD(CAM_INFERENCE_RESULT, payload);
        }
    }

    void trigLoop(){
        std::vector<uint8_t> buf; buf.reserve(4096);
        uint8_t tmp[1024];

        while(running.load()){
            ssize_t n=recv(trigSock,tmp,sizeof(tmp),0);
            if(n<=0) break;
            buf.insert(buf.end(),tmp,tmp+n);

            while(true){
                auto it=std::find(buf.begin(),buf.end(),START_ICD);
                if(it==buf.end()){ buf.clear(); break; }
                size_t start=(size_t)std::distance(buf.begin(),it);
                if(buf.size()<start+3){ if(start>0) buf.erase(buf.begin(),buf.begin()+(long)start); break; }

                uint8_t cmd=buf[start+ICD_CMD];
                uint8_t len=buf[start+ICD_LENGTH];
                size_t total=(size_t)(ICD_END + len);
                if(buf.size()<start+total){ if(start>0) buf.erase(buf.begin(),buf.begin()+(long)start); break; }

                std::vector<uint8_t> pkt(buf.begin()+(long)start, buf.begin()+(long)(start+total));
                buf.erase(buf.begin(), buf.begin()+(long)(start+total));

                size_t end_idx=(size_t)(ICD_END + len - 1);
                if(pkt[end_idx]!=END_ICD) continue;

                size_t chk_idx=(size_t)(ICD_CHECKSUM + len - 1);
                uint8_t chk=calcICDChecksum(pkt[ICD_CMD],pkt[ICD_LENGTH],&pkt[ICD_DATA]);
                if(pkt[chk_idx]!=chk) continue;

                uint8_t cam_id=(cmd & ~(CMD_MASK));
                if(cam_id<1 || cam_id>4) continue;

                hasNew[cam_id].store(true);
                cv[cam_id].notify_one();
            }
        }
        running.store(false);
    }
};

// ===================== main =====================
int main(int argc,char** argv){
    if(argc < 7){
        std::cout <<
        "Usage:\n  " << argv[0] <<
        " <engine1_backbone.plan> <engine2_head.plan>"
        " <conf_low> <conf_high> <nms_thr> <heatmap_thr>"
        " [num_classes stride sort_max_age sort_min_hits sort_iou_thr]\n\n"
        "Example:\n  " << argv[0] <<
        " esod_stage1_fp16.engine esod_head_fp16.engine"
        " 0.30 0.40 0.50 0.50 10 8 10 3 0.30\n";
        return 0;
    }

    std::string e1 = argv[1];
    std::string e2 = argv[2];

    float conf_low   = std::stof(argv[3]);
    float conf_high  = std::stof(argv[4]);
    float nms_thr    = std::stof(argv[5]);
    float heat_thr   = std::stof(argv[6]);

    int num_classes = 10;
    int stride = 8;
    int maxAge=10, minHits=3; double iouThr=0.3;

    if(argc >= 9){
        num_classes = std::stoi(argv[7]);
        stride      = std::stoi(argv[8]);
    }
    if(argc >= 12){
        maxAge  = std::stoi(argv[9]);
        minHits = std::stoi(argv[10]);
        iouThr  = std::stod(argv[11]);
    }

    ReaderApp app(e1, e2, conf_low, conf_high, nms_thr, heat_thr, maxAge, minHits, iouThr);

    // num_classes/stride는 내부 멤버로 고정돼 있으니,
    // 필요하면 ReaderApp 생성자에 추가하거나 내부 변수로 노출해서 세팅하세요.
    // (여기서는 간단히 default 10/8 기반)

    signal(SIGINT, ReaderApp::onSignal);
    signal(SIGTERM, ReaderApp::onSignal);

    if(!app.init()) return 1;
    if(!app.start5555()) return 1;

    app.run();
    app.cleanup();
    return 0;
}
