# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import time
from pathlib import Path
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# NVML (GPU 사용률/메모리 측정용)
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, norm_imgs
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
    save_one_box,
    target2mask,
)
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
    )
    batch_size = opt.batch_size
    save_img = not opt.nosave and (not source.endswith(".txt") or True)
    webcam = source.isnumeric() or (
        source.endswith(".txt") and False
    ) or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    # Directories
    save_dir = increment_path(
        Path(opt.project) / opt.name, exist_ok=opt.exist_ok
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != "cpu"  # half precision only supported on CUDA

    # ---------------- NVML 초기화 (GPU 사용률/메모리 측정용) ----------------
    max_gpu_util = 0          # %
    max_gpu_mem = 0           # bytes
    nvml_handle = None
    if opt.device != "cpu":
        try:
            nvmlInit()
            # opt.device 예: "0", "1", "0,1"
            dev_str = str(opt.device)
            first_gpu_str = dev_str.split(",")[0].strip()
            gpu_index = int(first_gpu_str)  # 물리 GPU 인덱스 그대로 사용
            nvml_handle = nvmlDeviceGetHandleByIndex(gpu_index)
            print(f"[NVML] monitoring physical GPU index {gpu_index}")
        except Exception as e:
            print(f"[NVML] 초기화 실패, GPU 사용률/메모리 측정 건너뜀: {e}")
            nvml_handle = None
    # ----------------------------------------------------------------

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size (정사각 한 변)
    names = model.module.names if hasattr(model, "module") else model.names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        batch_size = 1  # 스트림 모드는 배치 1로 강제
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 이미지 폴더가 아니면(비디오 파일 등) 배치 1로
    if not webcam and getattr(dataset, "mode", "") != "image":
        batch_size = 1

    print(f"[INFO] Using batch size = {batch_size}")

    # Run inference
    if device.type != "cpu":
        # warmup
        model(
            torch.zeros(1, 3, imgsz, imgsz)
            .to(device)
            .type_as(next(model.parameters()))
        )

    t0 = time.time()

    # --------- 배치 처리용 헬퍼 함수 ---------
    def process_batch(batch_paths, batch_imgs_np, batch_im0s, batch_vid_caps):
        nonlocal max_gpu_util, max_gpu_mem, nvml_handle, model, vid_path, vid_writer

        if len(batch_imgs_np) == 0:
            return

        # numpy -> torch (B,3,H,W)
        imgs_np = np.stack(batch_imgs_np, axis=0)
        imgs = torch.from_numpy(imgs_np).to(device)
        imgs = imgs.half() if half else imgs.float()
        if imgs.ndimension() == 3:
            imgs = imgs.unsqueeze(0)
        imgs = norm_imgs(imgs, model)

        # Inference
        t1 = time_synchronized()
        (pred, p_det), masks = model(imgs, augment=opt.augment)

        # GPU 사용률/메모리 업데이트
        if nvml_handle is not None:
            try:
                util = nvmlDeviceGetUtilizationRates(nvml_handle)
                mem_info = nvmlDeviceGetMemoryInfo(nvml_handle)
                if util.gpu > max_gpu_util:
                    max_gpu_util = util.gpu
                if mem_info.used > max_gpu_mem:
                    max_gpu_mem = mem_info.used
            except Exception as e:
                print(f"[NVML] 사용률/메모리 조회 실패: {e}")
                nvml_handle = None

        if pred is None:
            print("[WARN] prediction is None for this batch, skipping batch.")
            return

        # NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            opt.classes,
            opt.agnostic_nms,
            max_det=opt.max_det,
        )
        t2 = time_synchronized()

        # 배치 안 각 이미지별로 후처리
        for b_i, det in enumerate(pred):
            path = batch_paths[b_i]
            im0s = batch_im0s[b_i]
            vid_cap = batch_vid_caps[b_i]

            if webcam:
                # 실제로는 batch_size=1이라 여기 안 들어옴
                p, s, im0, frame = (
                    path[b_i],
                    f"{b_i}: ",
                    im0s[b_i].copy(),
                    dataset.count,
                )
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

            image_name = osp.basename(p).split(".")[0]
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = (
                str(save_dir / "labels" / p.stem)
                + ("" if dataset.mode == "image" else f"_{frame}")
            )  # img.txt

            s += "%gx%g " % imgs.shape[2:]  # H, W
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (
                            xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
                        ).view(-1).tolist()  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        )  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if opt.hide_labels
                            else (
                                names[c]
                                if opt.hide_conf
                                else f"{names[c]} {conf:.2f}"
                            )
                        )
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors(c, True),
                            line_thickness=opt.line_thickness,
                        )
                        if opt.save_crop:
                            save_one_box(
                                xyxy,
                                imc,
                                file=save_dir
                                / "crops"
                                / names[c]
                                / f"{p.stem}.jpg",
                                BGR=True,
                            )

            print(f"{s}Done. ({t2 - t1:.3f}s)")

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # video/stream (여기선 batch=1일 때만)
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h),
                        )
                    vid_writer.write(im0)

    # ---------------- 메인 루프: 배치 모아서 처리 ----------------
    batch_imgs = []
    batch_paths = []
    batch_im0s = []
    batch_vid_caps = []

    for path, img, im0s, vid_cap in dataset:
        if batch_size == 1:
            # 그냥 바로 배치 1개로 처리
            batch_imgs = [img]
            batch_paths = [path]
            batch_im0s = [im0s]
            batch_vid_caps = [vid_cap]
            process_batch(batch_paths, batch_imgs, batch_im0s, batch_vid_caps)
            batch_imgs, batch_paths, batch_im0s, batch_vid_caps = [], [], [], []
        else:
            # 배치에 쌓았다가 한 번에 처리
            batch_imgs.append(img)
            batch_paths.append(path)
            batch_im0s.append(im0s)
            batch_vid_caps.append(vid_cap)

            if len(batch_imgs) == batch_size:
                process_batch(batch_paths, batch_imgs, batch_im0s, batch_vid_caps)
                batch_imgs, batch_paths, batch_im0s, batch_vid_caps = [], [], [], []

    # 마지막 남은 배치 처리
    if batch_size > 1 and len(batch_imgs) > 0:
        process_batch(batch_paths, batch_imgs, batch_im0s, batch_vid_caps)

    # 결과 출력
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        print(f"Results saved to {save_dir}{s}")

    # ----------------- 여기서 최대 GPU 사용률/메모리 출력 -------------------
    if device.type != "cpu" and nvml_handle is not None:
        print(f"Max GPU utilization during inference: {max_gpu_util}%")
        print(
            f"Max GPU memory used during inference: {max_gpu_mem / (1024 ** 2):.1f} MiB"
        )
        try:
            nvmlShutdown()
        except Exception:
            pass
    # -----------------------------------------------------------------------

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="source",  # file/folder, 0 for webcam
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="maximum number of detections per image",
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--view-img", action="store_true", help="display results"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="save results to *.txt"
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="save confidences in --save-txt labels",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="save cropped prediction boxes",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--augment", action="store_true", help="augmented inference"
    )
    parser.add_argument(
        "--update", action="store_true", help="update all models"
    )
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument(
        "--name", default="exp", help="save results to project/name"
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness",
        default=3,
        type=int,
        help="bounding box thickness (pixels)",
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--view-cluster",
        action="store_true",
        help="(disabled) visualize clusters (ignored in this version)",
    )
    parser.add_argument(
        "--view-center",
        action="store_true",
        help="(disabled) visualize heatmap centers (ignored in this version)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="inference batch size (image folder only)",
    )

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=("tensorboard", "pycocotools", "thop"))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
