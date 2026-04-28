# ESOD-EDL: Evidential Objectness for Efficient Small Object Detection

This repository is based on the official implementation of 
[ESOD: Efficient Small Object Detection on High-Resolution Images](https://arxiv.org/abs/2407.16424),
with additional modifications for Evidential Deep Learning (EDL).

This project modifies the objectness prediction branch of ESOD by introducing
foreground/background evidence estimation and an EDL-based objectness loss.

## Key Modifications

- Extended the objectness prediction branch to estimate foreground/background evidence
- Replaced the standard objectness BCE loss with an EDL-based loss
- Used evidential objectness to improve heatmap reliability for patch selection

<!-- 
## Installation

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):

```bash
# (optional) conda install cuda-toolkit=11.8 -c pytorch

pip install -r requirements.txt 
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0
```

## Data Preparation

We currently support [**VisDrone**](https://github.com/VisDrone/VisDrone-Dataset), [**UAVDT**](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5), and [**TinyPerson**](https://github.com/ucas-vg/TinyBenchmark) datasets. Follow the instructions below to prepare datasets.

<details>
<summary>Data Prepare (click to expand)</summary>

* **Darknet Format**: The Darknet framework **locates labels automatically for each image** by replacing the last instance of `/images/` in each image path with `/labels/`. For example:

```  
dataset/images/im0.jpg
dataset/labels/im0.txt
```

The images and labels from VisDrone, UAVDT, TinyPerson are all organized in this format.

* **Ground-Truth Heatmap**: We recommend to leverage the [segment-anything](https://github.com/facebookresearch/segment-anything) model (SAM) to introduce precise shape prior to the GT heatmaps for training. You need to install SAM first:

```bash
cd third_party/segment-anything
pip install -e .
```

* **Dataset - VisDrone**: Download the [data](https://github.com/VisDrone/VisDrone-Dataset), and ensure the subsets under the `/path/to/visdrone` directory are as follows:

```
VisDrone2019-DET-train  VisDrone2019-DET-val  VisDrone2019-DET-test-dev  VisDrone2019-DET-test-challenge
```

Then make a soft-link to your directory, and run the `scripts/data_prepare.py` script to reorganize the images and labels:

```bash
ln -sf /path/to/visdrone VisDrone
python scripts/data_prepare.py --dataset VisDrone
```

* **Dataset - UAVDT**: Download the [data](https://drive.google.com/file/d/1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc/view?usp=sharing) and perform similar preprocessing:

```bash
ln -sf /path/to/uavdt UAVDT
python scripts/data_prepare.py --dataset UAVDT
```


* **Dataset - TinyPerson**: Download the [data](https://drive.google.com/open?id=1KrH9uEC9q4RdKJz-k34Q6v5hRewU5HOw) and perform similar preprocessing:

```bash
ln -sf /path/to/tinyperson TinyPerson
python scripts/data_prepare.py --dataset TinyPerson
```

</details>

## Training

Run commands below to reproduce results on the datasets, *e.g.*, [VisDrone](https://github.com/VisDrone/VisDrone-Dataset). Download the pretrained weights (e.g., [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt)) and put them to the `weights/pretrained/` directory first.

* **Training on Single GPU**

Here is the default setting for training YOLOv5m on VisDrone using the ESOD-EDL framework:

```bash
DATASET=visdrone MODEL=yolov5m GPUS=0 BATCH_SIZE=8 IMAGE_SIZE=1536 EPOCHS=50 bash ./scripts/train.sh
```

* **DDP Training**

When multiple GPUs are available, the [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) mode can be applied to speed up the training.
Simply set `GPUS` according to your devices, e.g. `GPUS=0,1,2,3`

```bash
DATASET=visdrone MODEL=yolov5m GPUS=0,1,2,3 BATCH_SIZE=32 IMAGE_SIZE=1536 EPOCHS=50 bash ./scripts/train.sh
```

* **Model Card**

We support the YOLOv5 series ([YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt), [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt), [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt), [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt)). After downloading them to `weights/`, simply change the `MODEL` (as well as `IMAGE_SIZE`) to apply different model:

```bash
DATASET=visdrone MODEL=yolov5s GPUS=0 BATCH_SIZE=8 IMAGE_SIZE=1536 EPOCHS=50 bash ./scripts/train.sh
                       yolov5m                   8            1536
                       yolov5l                   4            1920
                       yolov5x                   2            1920
```

Besides, we also support the [RetinaNet](https://pytorch.org/vision/main/models/retinanet.html), [RTMDet](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet), [YOLOv8](https://github.com/ultralytics/ultralytics), and [GPViT](https://github.com/ChenhongyiYang/GPViT) models. You can download the pre-trained weights, convert to ESOD initialization, and train the models to specific datasets:

```bash
python scripts/model_convert.py --model retinanet
DATASET=visdrone MODEL=retinanet GPUS=0 BATCH_SIZE=8 IMAGE_SIZE=1536 EPOCHS=50 bash ./scripts/train.sh
```

Feel free to set `MODEL` as `retinanet`, `rtmdet`, or `gpvit` (`yolov8m` does not require model convert). The detailed instructions will come soon.

* **Dataset Supported**

Beside **VisDrone**, we also support model training on **UAVDT** and **TinyPerson** datasets:

```bash
DATASET=uavdt MODEL=yolov5m GPUS=0 BATCH_SIZE=8 IMAGE_SIZE=1280 EPOCHS=50 bash ./scripts/train.sh
        tinyperson                                         2048
```


## Testing

### Vanilla Evaluation

Run commands below to compute evaluation results (AP, AP<sub>50</sub>, empty rate, missing rate) with intergrated `utils/metrics.py`.

```bash
python test.py --data data/visdrone.yaml --weights weights/yolov5m.pt --batch-size 8 --img-size 1536 --device 0
```

For computational analysis (including GFLOPs and FPS), use the following command:

```bash
python test.py --data data/visdrone.yaml --weights weights/yolov5m.pt --batch-size 1 --img-size 1536 --device 0 --task measure
```


### Official Evaluation

The organizations of data for official evaluation tools are different from Darknet. So an intermediate data conversion is required. Run command below to get the results in Darknet format.

```bash
python test.py --data data/visdrone.yaml --weights weights/yolov5m.pt --batch-size 8 --img-size 1536 --device 0 --task test --save-txt --save-conf
```

Then run the specified script `data_convert.py` for corresponding data formats and perform official evaluations.

* **VisDrone** test-dev set:

```bash
cp -r ./evaluate/VisDrone2018-DET-toolkit ./VisDrone/
python data_convert.py --dataset VisDrone --pred runs/test/exp/labels
cd ./VisDrone/VisDrone2018-DET-toolkit
matlab -nodesktop -nosplash -r evalDET
```

* **UAVDT** : coming soon.

* **TinyPerson**: coming soon.

## Inference

The script `detect.py` runs inference and saves the results to `runs/detect`.

```bash
python detect.py --weights weights/yolov5m.pt --source data/images/visdrone.txt --img-size 1536 --device 0 --view-cluster --line-thickness 1
```

`--view-cluster` will draw the generated patches in green boxes and save the heat maps from both prediction and ground truth.

## Pretrained Weights

- [x] Please find the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1jrrtG34q6gqdGMx7SflNK1uM6iiVKpin?usp=drive_link)


## Acknowledgment

- This work is based on the official ESOD implementation.
- We thank the original authors for their contributions.

## Citation

If you find this work useful in your research, please kindly cite the paper:

```
@article{liu2025esod,
    title={ESOD: Efficient Small Object Detection on High-Resolution Images},
    author={Liu, Kai and Fu, Zhihang and Jin, Sheng and Chen, Ze and Zhou, Fan and Jiang, Rongxin and Chen, Yaowu and Ye, Jieping},
    journal={IEEE Transactions on Image Processing},
    volume={34},
    pages={183--195},
    year={2025}
}
```
-->

