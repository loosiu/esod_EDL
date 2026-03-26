#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime.h"

using namespace nvinfer1;

// ======================= 공통 구조체/상수 =======================

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   cls;   // 0=person, 1=bicycle, 2=car(차량군)
    int   b;     // batch index (0..B-1)
};

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

// ================== 엔진 파일 로딩 ==================
static std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << path << "\n";
        std::exit(1);
    }
    file.seekg(0, std::ifstream::end);
    size_t size = (size_t)file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// ================== YOLO 엔진 핸들 ==================
static IRuntime*          gRuntime = nullptr;
static ICudaEngine*       gEngine  = nullptr;
static IExecutionContext* gContext = nullptr;
static cudaStream_t       gStream  = 0;

static float* d_input  = nullptr;
static float* d_output = nullptr;
static size_t cap_in   = 0;
static size_t cap_out  = 0;

// 엔진 IO 이름
static const char* kInputName  = "images";
static const char* kOutputName = "output0";

// 입력 크기
static const int YOLO_IN_H = 640;
static const int YOLO_IN_W = 640;

// 출력 shape: 런타임에 채움
static int gOutC     = -1;  // 예: 84 또는 85
static int gNumPreds = -1;  // 예: 8400

// 우리가 표시할 3클래스
static const char* kThreeNames[3] = {"person","bicycle","car"};

// ================== COCO -> 3클래스 매핑 ==================
// COCO: person=0, bicycle=1, car=2, bus=5, truck=7
static inline int map_coco_to3(int coco_id) {
    switch (coco_id) {
        case 0: return 0; // person
        case 1: return 1; // bicycle
        case 2: return 2; // car
        case 5: return 2; // bus -> car group
        case 7: return 2; // truck -> car group
        default: return -1;
    }
}

// ================== 엔진 초기화 ==================
void init_yolo_engine(const std::string& enginePath) {
    auto engineData = loadEngineFile(enginePath);

    gRuntime = createInferRuntime(gLogger);
    if (!gRuntime) { std::cerr << "Failed to create TRT runtime\n"; std::exit(1); }

    gEngine = gRuntime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!gEngine) { std::cerr << "Failed to deserialize engine\n"; std::exit(1); }

    std::cout << "[YOLO engine IO]\n";
    int nIO = gEngine->getNbIOTensors();
    for (int i = 0; i < nIO; ++i) {
        const char* name = gEngine->getIOTensorName(i);
        auto mode = gEngine->getTensorIOMode(name);
        std::cout << "  " << i << ": " << name
                  << " (" << (mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << ")\n";
    }

    gContext = gEngine->createExecutionContext();
    if (!gContext) { std::cerr << "Failed to create context\n"; std::exit(1); }

    CHECK_CUDA(cudaStreamCreate(&gStream));
}

void ensure_buffers(size_t inSize, size_t outSize) {
    if (cap_in < inSize) {
        if (d_input) cudaFree(d_input);
        CHECK_CUDA(cudaMalloc(&d_input, inSize));
        cap_in = inSize;
    }
    if (cap_out < outSize) {
        if (d_output) cudaFree(d_output);
        CHECK_CUDA(cudaMalloc(&d_output, outSize));
        cap_out = outSize;
    }
}

// ================== letterbox 전처리 (NCHW) ==================
void preprocess_letterbox_nchw(
    const cv::Mat& frame_bgr,
    int H_in, int W_in,
    float* dst,         // [3, H_in, W_in]
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
    rgb.convertTo(f32, CV_32FC3, 1.0f / 255.0f);

    for (int y = 0; y < H_in; ++y) {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W_in; ++x) {
            dst[0 * H_in * W_in + y * W_in + x] = row[x][0];
            dst[1 * H_in * W_in + y * W_in + x] = row[x][1];
            dst[2 * H_in * W_in + y * W_in + x] = row[x][2];
        }
    }
}

// ================== IoU / NMS ==================
static float iou_xyxy(float ax1, float ay1, float ax2, float ay2,
                      float bx1, float by1, float bx2, float by2)
{
    float xx1 = std::max(ax1, bx1);
    float yy1 = std::max(ay1, by1);
    float xx2 = std::min(ax2, bx2);
    float yy2 = std::min(ay2, by2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
    float areaB = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
    return inter / (areaA + areaB - inter + 1e-6f);
}

static void nms_classwise(std::vector<Detection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].cls != dets[j].cls) continue;

            float iou = iou_xyxy(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
                                 dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2);
            if (iou > iou_thresh) removed[j] = true;
        }
    }

    std::vector<Detection> out;
    out.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!removed[i]) out.push_back(dets[i]);
    }
    dets.swap(out);
}

// ================== 출력 텐서 shape 검사/세팅 ==================
static void update_output_shape_or_die(int B) {
    Dims4 inDims(B, 3, YOLO_IN_H, YOLO_IN_W);
    if (!gContext->setInputShape(kInputName, inDims)) {
        std::cerr << "Failed to set input shape for " << kInputName << "\n";
        std::exit(1);
    }

    Dims outDims = gContext->getTensorShape(kOutputName);
    if (outDims.nbDims != 3) {
        std::cerr << "Unexpected output dims nbDims=" << outDims.nbDims << "\n";
        std::exit(1);
    }

    gOutC     = outDims.d[1];
    gNumPreds = outDims.d[2];

    if (!(gOutC == 84 || gOutC == 85)) {
        std::cerr << "Warning: output channel dim is " << gOutC
                  << " (expected 84 or 85). You may need to adjust decode logic.\n";
    }
}

// ================== YOLO 배치 추론 (80클래스->3클래스 매핑) ==================
void yolo_infer_batch(
    const std::vector<cv::Mat>& frames,
    const std::vector<int>& batch2vid,
    std::array<PreprocInfo,4>& preproc_infos,
    std::vector<Detection>& out_dets,
    float conf_thresh,
    float nms_thresh,
    double& infer_ms,      // inference(H2D+enqueue+D2H) 시간
    double& post_ms        // 후처리(디코드+NMS) 시간
) {
    infer_ms = 0.0;
    post_ms  = 0.0;

    int B = (int)batch2vid.size();
    if (B == 0) { out_dets.clear(); return; }

    update_output_shape_or_die(B);

    const bool has_obj = (gOutC == 85);     // 5+80
    const int  cls_off = has_obj ? 5 : 4;   // cls start index
    const int  nc      = has_obj ? (gOutC - 5) : (gOutC - 4);

    if (nc <= 0) {
        std::cerr << "Invalid num classes derived from outputC=" << gOutC << "\n";
        std::exit(1);
    }

    // 1) host input
    std::vector<float> host_in((size_t)B * 3 * YOLO_IN_H * YOLO_IN_W);
    for (int b = 0; b < B; ++b) {
        int vid = batch2vid[b];
        float* dst = host_in.data() + (size_t)b * 3 * YOLO_IN_H * YOLO_IN_W;
        preprocess_letterbox_nchw(frames[vid], YOLO_IN_H, YOLO_IN_W, dst, preproc_infos[vid]);
    }

    size_t inSize  = (size_t)B * 3 * YOLO_IN_H * YOLO_IN_W * sizeof(float);
    size_t outSize = (size_t)B * gOutC * gNumPreds * sizeof(float);

    ensure_buffers(inSize, outSize);

    // ================== INFERENCE 타이밍 ==================
    auto t_infer0 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpyAsync(d_input, host_in.data(), inSize,
                               cudaMemcpyHostToDevice, gStream));

    if (!gContext->setInputTensorAddress(kInputName, d_input) ||
        !gContext->setOutputTensorAddress(kOutputName, d_output)) {
        std::cerr << "Failed to set tensor addresses\n";
        std::exit(1);
    }

    if (!gContext->enqueueV3(gStream)) {
        std::cerr << "enqueueV3 failed\n";
        std::exit(1);
    }

    std::vector<float> host_out((size_t)B * gOutC * gNumPreds);
    CHECK_CUDA(cudaMemcpyAsync(host_out.data(), d_output, outSize,
                               cudaMemcpyDeviceToHost, gStream));
    CHECK_CUDA(cudaStreamSynchronize(gStream));

    auto t_infer1 = std::chrono::high_resolution_clock::now();
    infer_ms = std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count();

    // ================== POSTPROCESS 타이밍 ==================
    auto t_post0 = std::chrono::high_resolution_clock::now();

    auto at = [&](int b, int c, int i) -> float {
        return host_out[(size_t)b * gOutC * gNumPreds + (size_t)c * gNumPreds + (size_t)i];
    };

    std::vector<std::vector<Detection>> dets_per_b(B);

    for (int b = 0; b < B; ++b) {
        int vid = batch2vid[b];
        const auto& info = preproc_infos[vid];

        // (디버그) 매핑 확인용 카운트
        // int cnt3[3] = {0,0,0};

        for (int i = 0; i < gNumPreds; ++i) {
            float cx = at(b, 0, i);
            float cy = at(b, 1, i);
            float w  = at(b, 2, i);
            float h  = at(b, 3, i);

            float obj = 1.0f;
            if (has_obj) {
                obj = at(b, 4, i);
                if (obj < 1e-6f) continue;
            }

            // 80클래스 전체에서 best class를 구한 뒤 -> 3클래스로 매핑
            int best_coco = -1;
            float best_prob = 0.0f;
            for (int c = 0; c < nc; ++c) {
                float p = at(b, cls_off + c, i);
                if (p > best_prob) {
                    best_prob = p;
                    best_coco = c;
                }
            }

            int cls3 = map_coco_to3(best_coco);
            if (cls3 < 0) continue;

            float score = obj * best_prob;
            if (score < conf_thresh) continue;

            float x1_l = cx - w * 0.5f;
            float y1_l = cy - h * 0.5f;
            float x2_l = cx + w * 0.5f;
            float y2_l = cy + h * 0.5f;

            float x1 = (x1_l - info.pad_x) / info.gain;
            float y1 = (y1_l - info.pad_y) / info.gain;
            float x2 = (x2_l - info.pad_x) / info.gain;
            float y2 = (y2_l - info.pad_y) / info.gain;

            x1 = std::max(0.0f, std::min(x1, (float)frames[vid].cols - 1));
            y1 = std::max(0.0f, std::min(y1, (float)frames[vid].rows - 1));
            x2 = std::max(0.0f, std::min(x2, (float)frames[vid].cols - 1));
            y2 = std::max(0.0f, std::min(y2, (float)frames[vid].rows - 1));

            if ((x2 - x1) < 2.0f || (y2 - y1) < 2.0f) continue;

            Detection d;
            d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
            d.score = score;
            d.cls = cls3;
            d.b = b;
            dets_per_b[b].push_back(d);

            // cnt3[cls3]++;
        }

        nms_classwise(dets_per_b[b], nms_thresh);

        // std::cout << "[vid " << vid << "] det3 counts: "
        //           << "person=" << cnt3[0] << ", "
        //           << "bicycle=" << cnt3[1] << ", "
        //           << "car=" << cnt3[2] << "\n";
    }

    out_dets.clear();
    for (int b = 0; b < B; ++b) {
        out_dets.insert(out_dets.end(), dets_per_b[b].begin(), dets_per_b[b].end());
    }

    auto t_post1 = std::chrono::high_resolution_clock::now();
    post_ms = std::chrono::duration<double, std::milli>(t_post1 - t_post0).count();
}

// ================== draw (색상 person/bicycle/car 고정) ==================
static void draw_dets(cv::Mat& frame, const std::vector<Detection>& dets) {
    cv::Scalar colors[3] = {
        cv::Scalar(0, 255, 0),   // person (green)
        cv::Scalar(255, 0, 0),   // bicycle (blue)  [BGR]
        cv::Scalar(0, 0, 255)    // car group (red) [BGR]
    };

    for (const auto& d : dets) {
        int cls = d.cls;
        cv::Scalar color = (0 <= cls && cls < 3) ? colors[cls] : cv::Scalar(0,255,255);

        int x1 = (int)std::round(d.x1);
        int y1 = (int)std::round(d.y1);
        int x2 = (int)std::round(d.x2);
        int y2 = (int)std::round(d.y2);

        cv::rectangle(frame, cv::Point(x1,y1), cv::Point(x2,y2), color, 2);

        char txt[128];
        const char* name = (0 <= cls && cls < 3) ? kThreeNames[cls] : "cls";
        std::snprintf(txt, sizeof(txt), "%s %.2f", name, d.score);

        int ty = std::max(0, y1 - 5);
        cv::putText(frame, txt, cv::Point(x1, ty),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

// ================== main ==================
int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "사용법: " << argv[0]
                  << " in0 in1 in2 in3 out0 out1 out2 out3\n";
        return 1;
    }

    std::array<std::string,4> in_paths  = {argv[1], argv[2], argv[3], argv[4]};
    std::array<std::string,4> out_paths = {argv[5], argv[6], argv[7], argv[8]};

    init_yolo_engine("/data/siwoo/yolo11n_fp16.engine"); // 경로 수정

    std::array<cv::VideoCapture,4> caps;
    std::array<cv::VideoWriter,4>  writers;
    std::array<bool,4>             alive = {true,true,true,true};

    int in_w=0, in_h=0;
    double fps=0.0;

    for (int i=0;i<4;++i) {
        caps[i].open(in_paths[i]);
        if (!caps[i].isOpened()) {
            std::cerr << "비디오 열기 실패: " << in_paths[i] << "\n";
            return 1;
        }
        int w = (int)caps[i].get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)caps[i].get(cv::CAP_PROP_FRAME_HEIGHT);
        double f = caps[i].get(cv::CAP_PROP_FPS);
        if (f <= 0) f = 25.0;

        if (i==0) { in_w=w; in_h=h; fps=f; }
        else if (w!=in_w || h!=in_h) {
            std::cerr << "모든 입력 영상 해상도 동일해야 함. cam" << i
                      << " : " << w << "x" << h << "\n";
            return 1;
        }
    }

    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    for (int i=0;i<4;++i) {
        writers[i].open(out_paths[i], fourcc, fps, cv::Size(in_w, in_h));
        if (!writers[i].isOpened()) {
            std::cerr << "출력 비디오 열기 실패: " << out_paths[i] << "\n";
            return 1;
        }
    }

    std::array<PreprocInfo,4> preproc_infos;

    int total_batches = 0;
    double total_infer_ms = 0.0;
    double total_post_ms  = 0.0;
    double total_total_ms = 0.0;
    int total_frames_processed = 0;

    const float conf_thresh = 0.25f;
    const float nms_thresh  = 0.50f;

    while (true) {
        std::vector<cv::Mat> frames(4);
        std::vector<int> batch2vid;
        batch2vid.reserve(4);

        for (int vid=0; vid<4; ++vid) {
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
        if (B==0) break;

        total_frames_processed += B;

        std::vector<Detection> batch_dets;

        double infer_ms = 0.0, post_ms = 0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        yolo_infer_batch(frames, batch2vid, preproc_infos, batch_dets,
                         conf_thresh, nms_thresh,
                         infer_ms, post_ms);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_batches++;
        total_infer_ms += infer_ms;
        total_post_ms  += post_ms;
        total_total_ms += total_ms;

        std::cout << "batch " << total_batches
                  << " (B=" << B << ") "
                  << "infer=" << infer_ms << " ms, "
                  << "post=" << post_ms << " ms, "
                  << "total=" << total_ms << " ms, "
                  << "dets=" << batch_dets.size()
                  << " (outputC=" << gOutC << ", preds=" << gNumPreds << ")\n";

        std::array<std::vector<Detection>,4> dets_per_vid;
        for (auto& d : batch_dets) {
            int vid = batch2vid[d.b];
            dets_per_vid[vid].push_back(d);
        }

        for (int vid=0; vid<4; ++vid) {
            if (!alive[vid]) continue;
            if (frames[vid].empty()) continue;

            draw_dets(frames[vid], dets_per_vid[vid]);
            writers[vid].write(frames[vid]);
        }
    }

    std::cout << "processed batches: " << total_batches << "\n";
    std::cout << "processed frames total: " << total_frames_processed << "\n";
    if (total_batches > 0) {
        std::cout << "avg per batch: "
                  << "infer=" << (total_infer_ms / total_batches) << " ms, "
                  << "post="  << (total_post_ms  / total_batches) << " ms, "
                  << "total=" << (total_total_ms / total_batches) << " ms\n";
    }
    if (total_frames_processed > 0) {
        std::cout << "avg per image: "
                  << "infer=" << (total_infer_ms / total_frames_processed) << " ms, "
                  << "post="  << (total_post_ms  / total_frames_processed) << " ms, "
                  << "total=" << (total_total_ms / total_frames_processed) << " ms\n";
    }

    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    CHECK_CUDA(cudaStreamDestroy(gStream));

    for (int i=0;i<4;++i) { caps[i].release(); writers[i].release(); }

    return 0;
}
