## 📋 Table of Contents
- [Model Zoo](#-model-zoo)
- [Dataset Preparation](#-dataset-preparation)
- [Model Configuration](#-model-configuration)
- [Usage](#-usage)
- [Tools](#-tools)

---

## 🏆 Model Zoo

### COCO2017 Validation Results (Keypoints)


| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECPose-S** | 640 | 68.9 |  10 | 30 | 5.54 | [config](configs/ecpose/ecpose_s_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth) |
| **ECPose-M** | 640 | 72.4 |  20 | 63 | 9.25 | [config](configs/ecpose/ecpose_m_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_m.pth) |
| **ECPose-L** | 640 | 73.5 |  34 | 112 | 11.83 | [config](configs/ecpose/ecpose_l_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth) |
| **ECPose-X** | 640 | 74.8 |  51 | 172 | 14.31 | [config](configs/ecpose/ecpose_x_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_x.pth) |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 📁 Dataset Preparation

### COCO2017 Keypoints

1. Download COCO2017 and keypoint annotations from the official COCO website.
2. Organize data as:

```text
/path/to/COCO2017/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

3. Update paths in [`configs/dataset/coco_pose.yml`](./configs/dataset/coco_pose.yml):

```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/COCO2017/train2017
    ann_file: /path/to/COCO2017/annotations/person_keypoints_train2017.json

val_dataloader:
  dataset:
    img_folder: /path/to/COCO2017/val2017
    ann_file: /path/to/COCO2017/annotations/person_keypoints_val2017.json
```

### Custom Dataset (COCO Keypoints Format)

Use the same format as COCO keypoints and adapt `configs/dataset/coco_pose.yml`:

- set `img_folder` / `ann_file` to your dataset paths
- keep `task: pose`
- adjust `num_classes` and remapping behavior if needed

---

## 🔌 Model Configuration

Model configs are in [`configs/ecpose`](./configs/ecpose/):

- [`ecpose_s_coco.yml`](./configs/ecpose/ecpose_s_coco.yml)
- [`ecpose_m_coco.yml`](./configs/ecpose/ecpose_m_coco.yml)
- [`ecpose_l_coco.yml`](./configs/ecpose/ecpose_l_coco.yml)
- [`ecpose_x_coco.yml`](./configs/ecpose/ecpose_x_coco.yml)

Base model definition is in [`ecpose.yml`](./configs/ecpose/ecpose.yml) with:

- `task: pose`
- `model: ECPose`
- postprocess outputs `(scores, labels, keypoints)`

---

## 🎮 Usage

### Training

```bash
# Generic (single node, 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --use-amp --seed=0
```

Replace `{SIZE}` with `s`, `m`, `l`, or `x`.

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --test-only -r /path/to/model.pth
```

### Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --use-amp --seed=0 -t /path/to/model.pth
```

---

## 🔧 Tools

### PyTorch Inference (image/video)

```bash
python tools/inference/torch_inf.py \
  -c configs/ecpose/ecpose_{SIZE}_coco.yml \
  -r ecpose_{SIZE}.pth \
  -i /path/to/image_or_video
```

Optional flags: `-d cuda:0`, `-t 0.4`, `--fp32`, `--no-skeleton`.

### ONNX Export

```bash
python tools/deployment/export_onnx.py \
  -c configs/ecpose/ecpose_{SIZE}_coco.yml \
  -r ecpose_{SIZE}.pth \
  --check --simplify
```

### ONNX Inference (image/video)

```bash
python tools/inference/onnx_inf.py \
  --onnx ecpose_{SIZE}.onnx \
  --input /path/to/image_or_video \
  --device cuda
```
