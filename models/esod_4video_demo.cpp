// esod_4video_demo.cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <array>
#include <algorithm>  // std::min, std::max

#include "esod_core.hpp"
#include "esod_heatmapparser.hpp"
#include "esod_head_decoder.hpp"

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "cuda_runtime.h"

// VisDrone 10-class → 3-class(person, bicycle, car) 매핑
// 0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van,
// 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor

static const char* kThreeClassNames[3] = {
    "person",   // 0
    "bicycle",  // 1
    "car"       // 2
};

// 10-class → 3-class 매핑
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

// 전처리(letterbox) 정보: 나중에 박스 좌표를 원본으로 되돌릴 때 사용
struct PreprocInfo {
    float gain = 1.0f;  // resize scale
    int   pad_x = 0;    // left pad
    int   pad_y = 0;    // top pad
};

// ESOD 단계별 타이밍
struct EsodTiming {
    double t_engine1_ms = 0.0;
    double t_parser_ms  = 0.0;
    double t_engine2_ms = 0.0;
    double t_decode_ms  = 0.0;  // NMS 포함
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

using namespace nvinfer1;

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

// engine1용 device 버퍼 재사용
static float* d_input1 = nullptr;
static float* d_feat1  = nullptr;
static float* d_mask1  = nullptr;
static size_t cap_input1 = 0;
static size_t cap_feat1  = 0;
static size_t cap_mask1  = 0;

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

// engine1 버퍼 확보/재사용
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

// engine2용 device 버퍼 재사용
static float* d_in2  = nullptr;
static float* d_out2 = nullptr;
static size_t cap_in2  = 0;
static size_t cap_out2 = 0;

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

    gContext2 = gEngine2->createExecutionContext();
    if (!gContext2) {
        std::cerr << "Failed to create context for engine2\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaStreamCreate(&gStream2));
}

// engine2 버퍼 확보/재사용
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
// 3) engine1 실행 (TensorRT enqueueV3)
//    input:  [B,3,H_in,W_in]
//    output: feat [B,C,H,W], mask [B,1,H,W]
// ======================================================
void run_engine1(
    const Tensor& input,
    Tensor& feat,
    Tensor& mask
) {
    if (!gContext1) {
        std::cerr << "[run_engine1] engine1 not initialized!\n";
        std::exit(1);
    }

    int B    = input.shape[0];
    int Cin  = input.shape[1];
    int Hin  = input.shape[2];
    int Win  = input.shape[3];

    int Cout = feat.shape[1];
    int Hout = feat.shape[2];
    int Wout = feat.shape[3];

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

    size_t inputSize = static_cast<size_t>(B) * Cin * Hin * Win * sizeof(float);
    size_t featSize  = static_cast<size_t>(B) * Cout * Hout * Wout * sizeof(float);
    size_t maskSize  = static_cast<size_t>(B) * 1    * Hout * Wout * sizeof(float);

    // device 버퍼 재사용
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

    CHECK_CUDA(cudaMemcpyAsync(
        feat.data.data(),
        d_feat1,
        featSize,
        cudaMemcpyDeviceToHost,
        gStream1));

    CHECK_CUDA(cudaMemcpyAsync(
        mask.data.data(),
        d_mask1,
        maskSize,
        cudaMemcpyDeviceToHost,
        gStream1));

    CHECK_CUDA(cudaStreamSynchronize(gStream1));
}

// ======================================================
// 4) engine2 실행 (TensorRT enqueueV3)
//    input:  patches [N_patch,C,ph,pw]
//    output: head_raw [N_patch,M,no]
// ======================================================
void run_engine2(
    const Tensor& patches,
    Tensor& head_raw
) {
    if (!gContext2) {
        std::cerr << "[run_engine2] engine2 not initialized!\n";
        std::exit(1);
    }

    int N_patch = patches.shape[0];
    int C       = patches.shape[1];
    int ph      = patches.shape[2];
    int pw      = patches.shape[3];

    int M       = head_raw.shape[1];
    int no      = head_raw.shape[2];

    Dims inDims;
    inDims.nbDims = 4;
    inDims.d[0] = N_patch;
    inDims.d[1] = C;
    inDims.d[2] = ph;
    inDims.d[3] = pw;

    if (!gContext2->setInputShape(kE2InputName, inDims)) {
        std::cerr << "Failed to set input shape for engine2\n";
        std::exit(1);
    }

    size_t inSize  = static_cast<size_t>(N_patch) * C * ph * pw * sizeof(float);
    size_t outSize = static_cast<size_t>(N_patch) * M * no * sizeof(float);

    // device 버퍼 재사용
    ensure_engine2_buffers(inSize, outSize);

    CHECK_CUDA(cudaMemcpyAsync(
        d_in2,
        patches.data.data(),
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
        head_raw.data.data(),
        d_out2,
        outSize,
        cudaMemcpyDeviceToHost,
        gStream2));

    CHECK_CUDA(cudaStreamSynchronize(gStream2));
}

// ======================================================
// 5) ESOD 배치 추론 (B 프레임, 10-class 그대로 출력)
//    timing != nullptr 인 경우, 단계별 시간(ms) 채워줌
// ======================================================
void esod_infer_batch(
    const Tensor& img_batch,         // [B,3,H_in,W_in]
    int num_classes,
    std::vector<Detection>& detections,  // out (image_id = batch index)
    EsodTiming* timing                       // out (nullable)
) {
    int B = img_batch.shape[0];

    // engine2 config 기준
    int C = 192;   // channel
    int H = 192;   // 1536 / 8
    int W = 192;

    Tensor feat({B, C, H, W});
    Tensor mask({B, 1, H, W});

    HeatMapParserCpp parser(C, 8, 0.5f, false, false);

    // 타이머
    auto t0 = std::chrono::high_resolution_clock::now();
    // ---- engine1 ----
    run_engine1(img_batch, feat, mask);
    auto t1 = std::chrono::high_resolution_clock::now();

    // ---- HeatMapParser ----
    Tensor patches;
    Tensor offsets;   // [N_patch, 5] (b, x1, y1, x2, y2)
    parser.forward(feat, mask, patches, offsets);
    auto t2 = std::chrono::high_resolution_clock::now();

    int N_patch = patches.shape[0];
    if (N_patch == 0) {
        detections.clear();
        if (timing) {
            timing->t_engine1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timing->t_parser_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
            timing->t_engine2_ms = 0.0;
            timing->t_decode_ms  = 0.0;
        }
        return;
    }

    // ---- engine2 ----
    int no = 5 + num_classes;
    int M  = 24*24*3 + 12*12*3 + 6*6*3;  // 2268

    Tensor head_raw({N_patch, M, no});
    run_engine2(patches, head_raw);
    auto t3 = std::chrono::high_resolution_clock::now();

    // ---- HeadDecoder + NMS ----
    float conf_thresh = 0.25f;
    float nms_thresh  = 0.50f;
    EsodHeadDecoder decoder(conf_thresh, nms_thresh, num_classes);

    std::vector<Detection> out_dets;
    decoder.decode(head_raw, offsets, out_dets);  // image_id = b index
    auto t4 = std::chrono::high_resolution_clock::now();

    detections = out_dets; // 여기까지는 여전히 10-class

    if (timing) {
        timing->t_engine1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing->t_parser_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
        timing->t_engine2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        timing->t_decode_ms  = std::chrono::duration<double, std::milli>(t4 - t3).count();
    }
}

// ======================================================
// 6) 전처리: letterbox로 배치 텐서에 채우고 gain/pad 기록
// ======================================================
void preprocess_frame_to_batch_letterbox(
    const cv::Mat& frame_bgr,
    int H_in, int W_in,
    Tensor& img_batch,   // [B,3,H_in,W_in]
    int b_idx,
    PreprocInfo& info    // gain/pad 저장
) {
    int orig_w = frame_bgr.cols;
    int orig_h = frame_bgr.rows;

    // 1) 비율 유지하는 scale
    float r = std::min((float)W_in / (float)orig_w,
                       (float)H_in / (float)orig_h);
    int new_w = (int)std::round(orig_w * r);
    int new_h = (int)std::round(orig_h * r);

    // 2) resize
    cv::Mat resized;
    cv::resize(frame_bgr, resized, cv::Size(new_w, new_h));

    // 3) 패딩 계산 (center)
    int pad_x = (W_in - new_w) / 2;
    int pad_y = (H_in - new_h) / 2;

    // 4) 114로 채운 1536×1536 캔버스에 복사
    cv::Mat canvas(H_in, W_in, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // 5) info 저장
    info.gain  = r;
    info.pad_x = pad_x;
    info.pad_y = pad_y;

    // 6) canvas → RGB / float / [0,1] → 텐서 (CHW)
    cv::Mat rgb, f32;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

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
// 7) 후처리: letterbox 역변환으로 원본 좌표에 그리기 + 3클래스 색/라벨
// ======================================================
void draw_detections_letterbox(
    cv::Mat& frame,  // 원본 프레임
    const std::vector<Detection>& dets,
    const PreprocInfo& info
) {
    // BGR 색상:
    // 0: person  → 초록
    // 1: bicycle → 파랑
    // 2: car     → 빨강
    cv::Scalar colors[3] = {
        cv::Scalar(0, 255, 0),   // green
        cv::Scalar(255, 0, 0),   // blue
        cv::Scalar(0, 0, 255)    // red
    };

    for (const auto& d : dets) {
        int cls = d.cls; // 이미 0~2 로 매핑

        // 1) 네트워크 좌표(1536×1536 + pad) → 원본 좌표로 역변환
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
            : cv::Scalar(0, 255, 255); // fallback: yellow

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
        (void)baseLine;
        cv::Size textSize = cv::getTextSize(text,
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
// 8) main: 비디오 4개 배치 추론
//    사용법:
//    ./esod_4video_demo v0.avi v1.avi v2.avi v3.avi out0 out1 out2 out3
// ======================================================
int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "사용법: " << argv[0]
                  << " in0 in1 in2 in3 out0 out1 out2 out3\n";
        return 1;
    }

    std::array<std::string,4> in_paths  = {argv[1], argv[2], argv[3], argv[4]};
    std::array<std::string,4> out_paths = {argv[5], argv[6], argv[7], argv[8]};

    // 1) 엔진 초기화 (경로는 실제 파일명으로 수정)
    init_engine1("/data/siwoo/esod/esod/esod_stage1_fp16.engine");
    init_engine2("/data/siwoo/esod/esod/esod_head_fp16.engine");

    // 2) 비디오 4개 열기
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

        int w = static_cast<int>(caps[i].get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(caps[i].get(cv::CAP_PROP_FRAME_HEIGHT));
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

    // 3) writer 4개 (원본 해상도로 저장, 코덱은 MJPG 예시)
    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    for (int i = 0; i < 4; ++i) {
        writers[i].open(out_paths[i], fourcc, fps, cv::Size(in_width, in_height));
        if (!writers[i].isOpened()) {
            std::cerr << "출력 비디오 열기 실패: " << out_paths[i] << "\n";
            return 1;
        }
    }

    // 네트워크 입력 크기
    const int H_in = 1536;
    const int W_in = 1536;
    int num_classes = 10; // 엔진/헤드는 여전히 10-class

    int global_frame_idx = 0;
    double total_infer_ms = 0.0;
    int total_batches = 0;
    int total_frames_processed = 0;

    // 각 비디오별 전처리 정보
    std::array<PreprocInfo,4> preproc_infos;

    while (true) {
        // 1) 각 카메라에서 한 프레임씩 읽어서 배치 구성
        std::vector<cv::Mat> frames(4);
        std::vector<int> batch2vid;  // batch index -> video index
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

        int B = static_cast<int>(batch2vid.size());
        if (B == 0) break;  // 더 이상 읽을 프레임 없음
        total_frames_processed += B;

        // 2) 배치 텐서 만들기
        Tensor img_batch({B, 3, H_in, W_in});
        for (int b = 0; b < B; ++b) {
            int vid = batch2vid[b];
            preprocess_frame_to_batch_letterbox(
                frames[vid], H_in, W_in, img_batch, b, preproc_infos[vid]);
        }

        // 3) ESOD 배치 추론 + 시간 측정 (10-class 결과)
        std::vector<Detection> batch_dets;
        EsodTiming et;

        auto t0 = std::chrono::high_resolution_clock::now();
        esod_infer_batch(img_batch, num_classes, batch_dets, &et);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_infer_ms += ms;
        total_batches  += 1;

        double ms_per_img        = ms / B;
        double no_nms_ms         = et.t_engine1_ms + et.t_parser_ms + et.t_engine2_ms;
        double no_nms_per_img_ms = no_nms_ms / B;

        std::cout << "batch " << total_batches
                  << " (B=" << B << ") infer total = "
                  << ms << " ms (" << ms_per_img << " ms / image)"
                  << ", no-NMS = " << no_nms_ms << " ms ("
                  << no_nms_per_img_ms << " ms / image)"
                  << ", decode(NMS) = " << et.t_decode_ms << " ms"
                  << ", dets=" << batch_dets.size() << "\n";

        // 4) 배치 결과(10-class)를 비디오별로 분리 + 3-class로 리매핑
        std::array<std::vector<Detection>,4> dets_per_video;
        for (const auto& d : batch_dets) {
            int b = d.image_id;          // 0..B-1
            if (b < 0 || b >= B) continue;
            int vid = batch2vid[b];      // 실제 비디오 index

            int new_cls = map_visdrone10_to3(d.cls);
            if (new_cls < 0) continue;   // 필요 없는 클래스면 스킵

            Detection d2 = d;
            d2.image_id = vid;           // video index
            d2.cls = new_cls;            // 0:person, 1:bicycle, 2:car

            dets_per_video[vid].push_back(d2);
        }

        // 5) 각 비디오 프레임에 박스 그리고 저장 (3-class 기준으로 draw)
        for (int vid = 0; vid < 4; ++vid) {
            if (!alive[vid]) continue;
            if (frames[vid].empty()) continue;

            draw_detections_letterbox(frames[vid],
                                      dets_per_video[vid],
                                      preproc_infos[vid]);

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

    // ====== device 메모리 정리 ======
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
