// esod_video_demo.cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "esod_core.hpp"
#include "esod_heatmapparser.hpp"
#include "esod_head_decoder.hpp"

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "cuda_runtime.h"

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

// ======================================================
// 2) engine2 (head) TensorRT 핸들 / 초기화
// ======================================================
static IRuntime*          gRuntime2  = nullptr;
static ICudaEngine*       gEngine2   = nullptr;
static IExecutionContext* gContext2  = nullptr;
static cudaStream_t       gStream2;

static const char* kE2InputName  = "patch";      // INPUT
static const char* kE2OutputName = "head_raw";   // OUTPUT

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

// ======================================================
// 3) engine1 실행
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

    float* d_input = nullptr;
    float* d_feat  = nullptr;
    float* d_mask  = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA(cudaMalloc(&d_feat,  featSize));
    CHECK_CUDA(cudaMalloc(&d_mask,  maskSize));

    CHECK_CUDA(cudaMemcpyAsync(
        d_input,
        input.data.data(),
        inputSize,
        cudaMemcpyHostToDevice,
        gStream1));

    if (!gContext1->setInputTensorAddress(kE1InputName, d_input) ||
        !gContext1->setOutputTensorAddress(kE1FeatName, d_feat)  ||
        !gContext1->setOutputTensorAddress(kE1MaskName, d_mask)) {
        std::cerr << "Failed to set tensor addresses for engine1\n";
        std::exit(1);
    }

    if (!gContext1->enqueueV3(gStream1)) {
        std::cerr << "Failed to enqueueV3 for engine1\n";
        std::exit(1);
    }

    CHECK_CUDA(cudaMemcpyAsync(
        feat.data.data(),
        d_feat,
        featSize,
        cudaMemcpyDeviceToHost,
        gStream1));

    CHECK_CUDA(cudaMemcpyAsync(
        mask.data.data(),
        d_mask,
        maskSize,
        cudaMemcpyDeviceToHost,
        gStream1));

    CHECK_CUDA(cudaStreamSynchronize(gStream1));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_feat));
    CHECK_CUDA(cudaFree(d_mask));
}

// ======================================================
// 4) engine2 실행
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

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  inSize));
    CHECK_CUDA(cudaMalloc(&d_out, outSize));

    // host -> device
    CHECK_CUDA(cudaMemcpyAsync(
        d_in,
        patches.data.data(),
        inSize,
        cudaMemcpyHostToDevice,
        gStream2));

    if (!gContext2->setInputTensorAddress(kE2InputName, d_in) ||
        !gContext2->setOutputTensorAddress(kE2OutputName, d_out)) {
        std::cerr << "Failed to set tensor addresses for engine2\n";
        std::exit(1);
    }

    if (!gContext2->enqueueV3(gStream2)) {
        std::cerr << "Failed to enqueueV3 for engine2\n";
        std::exit(1);
    }

    // device -> host  ✅ 여기 수정
    CHECK_CUDA(cudaMemcpyAsync(
        head_raw.data.data(),
        d_out,
        outSize,
        cudaMemcpyDeviceToHost,
        gStream2));

    CHECK_CUDA(cudaStreamSynchronize(gStream2));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

// ======================================================
// 5) ESOD 추론 (한 프레임)
// ======================================================
void esod_infer(
    const Tensor& img_tensor,           // [1,3,H_in,W_in]
    int num_classes,
    std::vector<Detection>& detections  // out
) {
    int B = img_tensor.shape[0];
    if (B != 1) {
        throw std::runtime_error("esod_infer: only B=1 supported in this demo");
    }

    int C = 192;   // engine2 input: [-1,192,24,24]
    int H = 192;   // 1536 / 8
    int W = 192;

    Tensor feat({B, C, H, W});
    Tensor mask({B, 1, H, W});

    run_engine1(img_tensor, feat, mask);

    HeatMapParserCpp parser(C, 8, 0.5f, false, false);

    Tensor patches;
    Tensor offsets;
    parser.forward(feat, mask, patches, offsets);

    int N_patch = patches.shape[0];
    if (N_patch == 0) {
        detections.clear();
        return;
    }

    int no = 5 + num_classes;
    int M  = 24*24*3 + 12*12*3 + 6*6*3;

    Tensor head_raw({N_patch, M, no});
    run_engine2(patches, head_raw);

    float conf_thresh = 0.25f;
    float nms_thresh  = 0.45f;
    EsodHeadDecoder decoder(conf_thresh, nms_thresh, num_classes);

    std::vector<Detection> out_dets;
    decoder.decode(head_raw, offsets, out_dets);
    detections = out_dets;
}

// ======================================================
// 6) 전처리: OpenCV BGR frame -> Tensor [1,3,H_in,W_in]
// ======================================================
void preprocess_frame(
    const cv::Mat& frame_bgr,
    int H_in, int W_in,
    Tensor& img_tensor   // [1,3,H_in,W_in]
) {
    cv::Mat resized, rgb, f32;
    cv::resize(frame_bgr, resized, cv::Size(W_in, H_in));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    int B = 1, C = 3;

    if (img_tensor.shape.size() != 4 ||
        img_tensor.shape[0] != B ||
        img_tensor.shape[1] != C ||
        img_tensor.shape[2] != H_in ||
        img_tensor.shape[3] != W_in) {
        img_tensor = Tensor({B, C, H_in, W_in});
    }

    // HWC -> CHW
    for (int y = 0; y < H_in; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W_in; ++x) {
            for (int c = 0; c < 3; ++c) {
                int idx = c * H_in * W_in + y * W_in + x;
                img_tensor.data[idx] = row[x][c];
            }
        }
    }
}

// ======================================================
// 7) 후처리: bbox 그리기
// ======================================================
void draw_detections_rescaled(
    cv::Mat& frame,                  // 원본 프레임
    const std::vector<Detection>& dets,
    int net_h, int net_w             // H_in, W_in (1536,1536)
) {
    float scale_x = static_cast<float>(frame.cols) / static_cast<float>(net_w);
    float scale_y = static_cast<float>(frame.rows) / static_cast<float>(net_h);

    for (const auto& d : dets) {
        float cx = d.x * scale_x;
        float cy = d.y * scale_y;
        float w  = d.w * scale_x;
        float h  = d.h * scale_y;

        int x1 = static_cast<int>(cx - w / 2.0f);
        int y1 = static_cast<int>(cy - h / 2.0f);
        int x2 = static_cast<int>(cx + w / 2.0f);
        int y2 = static_cast<int>(cy + h / 2.0f);

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);
        char text[64];
        std::snprintf(text, sizeof(text), "cls:%d %.2f", d.cls, d.score);
        cv::putText(frame, text, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }
}

// ======================================================
// 8) main: 비디오 1개 추론
// ======================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용법: " << argv[0] << " input_video output_video\n";
        return 1;
    }

    std::string input_video  = argv[1];
    std::string output_video = argv[2];

    // 1) 엔진 초기화
    init_engine1("/data/siwoo/esod/esod/esod_stage1_fp16.engine");
    init_engine2("/data/siwoo/esod/esod/esod_head_fp16.engine");

    // 2) 비디오 열기
    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "비디오 열기 실패: " << input_video << "\n";
        return 1;
    }

    // ★ 여기서 총 프레임 수 출력
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "total frames in input: " << total_frames << std::endl;

    int   in_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int   in_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps      = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;

    const int H_in = 1536;
    const int W_in = 1536;

    cv::VideoWriter writer;

    // 1) AVI + XVID
    // int fourcc = cv::VideoWriter::fourcc('X','V','I','D');

    // 2) 또는 MJPG (용량 크지만 거의 항상 됨)
    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');

    writer.open(output_video, fourcc, fps, cv::Size(in_width, in_height));
    if (!writer.isOpened()) {
        std::cerr << "출력 비디오 열기 실패: " << output_video << "\n";
        return 1;
    }

    Tensor img_tensor({1, 3, H_in, W_in});
    int num_classes = 10;

    cv::Mat frame;
    int frame_idx = 0;

    while (true) {
        if (!cap.read(frame)) break;  // frame: 원본 해상도 (in_height × in_width)

        // 1) 전처리: 원본 -> 1536×1536 텐서
        preprocess_frame(frame, H_in, W_in, img_tensor);

        // 2) ESOD 추론 (결과는 1536×1536 좌표계)
        std::vector<Detection> dets;
        esod_infer(img_tensor, num_classes, dets);

        // 3) 박스를 "원본 frame" 위에 그리되, 좌표는 스케일해서 매핑
        draw_detections_rescaled(frame, dets, H_in, W_in);  

        // 4) 원본 해상도 그대로 저장
        writer.write(frame);

        std::cout << "frame " << frame_idx++ << " done, dets=" << dets.size() << "\n";
    }

    cap.release();
    writer.release();

    // ★ 실제 처리한 프레임 수 출력
    std::cout << "processed frames: " << frame_idx << std::endl;

    return 0;
}
