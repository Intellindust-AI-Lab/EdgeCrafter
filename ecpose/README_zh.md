<h3 align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</h3>

## 📋 目录
- [模型库](#-模型库)
- [数据集准备](#-数据集准备)
- [模型配置](#-模型配置)
- [快速上手](#-快速上手)
- [实用工具](#-实用工具)

---

## 🏆 模型库

### COCO2017 Validation Results (Keypoints)


| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECPose-S** | 640 | 68.9 |  10 | 30 | 5.54 | [config](configs/ecpose/ecpose_s_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth) |
| **ECPose-M** | 640 | 72.4 |  20 | 63 | 9.25 | [config](configs/ecpose/ecpose_m_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_m.pth) |
| **ECPose-L** | 640 | 73.5 |  34 | 112 | 11.83 | [config](configs/ecpose/ecpose_l_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth) |
| **ECPose-X** | 640 | 74.8 |  51 | 172 | 14.31 | [config](configs/ecpose/ecpose_x_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_x.pth) |

---

## 📦 安装

```bash
pip install -r requirements.txt
```

### ⚡ 快速上手 (推理测试)
使用预训练模型对示例图像进行推理是上手 ECPose 最快捷的方式。
```bash
# 1. 下载预训练权重（以 ECPose-L 为例）
wget https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth

# 2. 执行 PyTorch 推理
# 请将 `path/to/your/image.jpg` 替换为实际的图像路径
python tools/inference/torch_inf.py -c configs/ecpose/ecpose_l_coco.yml -r ecpose_l.pth -i path/to/your/image.jpg
```

## 📁 数据集准备

### COCO2017 关键点数据集

1. 从 COCO 官网下载 COCO2017 数据集及关键点标注。
2. 参照以下结构组织数据：

```text
/path/to/COCO2017/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

3. 在 [`configs/dataset/coco_pose.yml`](./configs/dataset/coco_pose.yml) 中配置相应路径：

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

### 自定义数据集 (符合 COCO 关键点格式)

如果使用自定义数据集，请确保其格式与 COCO 关键点格式一致，并调整 `configs/dataset/coco_pose.yml`：

- 将 `img_folder` 和 `ann_file` 设置为实际的数据路径。
- 确保 `task: pose` 保持不变。
- 根据实际需求调整 `num_classes` 以及类别映射逻辑。

---

## 🔌 模型配置

模型配置文件位于 [`configs/ecpose`](./configs/ecpose/)。以 [ecpose_s_coco.yml](./configs/ecpose/ecpose_s_coco.yml) 为例：

```yaml
__include__: [
  '../dataset/coco_pose.yml',
  'ecpose.yml'
]

output_dir: outputs/ecpose_s

ViTAdapter:
  name: ecpose_vitt
  embed_dim: 192
  weights_path: ecvits/ecpose_vitt.pth    # 预训练骨干网络；首次运行时将自动下载
  interaction_indexes: [10, 11]
  num_heads: 3

eval_spatial_size: [640, 640]   # 输入分辨率

epoches: 92  # 总训练轮数
grad_accum_steps: 1

## 优化器配置
optimizer:
  type: AdamW
  params: 
    -
      params: '^(?=.*.backbone)(?!.*(?:norm|bn|bias)).*$'  # 骨干网络参数（不含归一化层和偏置）
      lr: 0.000025
    -
      params: '^(?=.*.backbone)(?=.*(?:norm|bn|bias)).*$'  # 骨干网络中的归一化层 (norm/bn) 和偏置参数
      lr: 0.000025
      weight_decay: 0.
    - 
      params: '^(?!.*\.backbone)(?=.*(?:norm|bn|bias)).*$'  # 非骨干网络部分的归一化层和偏置参数
      weight_decay: 0.

  lr: 0.0005  # 非骨干网络部分的基础学习率
  betas: [0.9, 0.999]
  weight_decay: 0.0001

train_dataloader: 
  dataset: 
    transforms:
      ops:
        - {type: PoseMosaic, output_size: 320}
        - {type: MixUpCopyPaste, mixup_prob: 0.5}
        - {type: RandomZoomOut, p: 0.5}
        - {type: RandomHorizontalFlip}
        - {type: ColorJitter}
        - {type: Resize, size: [640, 640]}
        - {type: ToTensor}
        - {type: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]} 
      mosaic_prob: 0.5  # 使用 Mosaic 增强的概率
      policy:
        epoch: [4, 45, 90]   # Mosaic 在第 4 轮开启，第 45 轮停止；所有数据增强在第 90 轮关闭。
       
  collate_fn:
    stop_epoch: 90  # 所有数据增强在第 90 轮关闭。
```

---

## 🎮 快速上手

### 训练

```bash
# 通用训练指令 (单节点，4 张 GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --use-amp --seed=0
```

请将 `{SIZE}` 替换为 `s`、`m`、`l` 或 `x`。

### 评估

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --test-only -r /path/to/model.pth
```

### 微调

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecpose/ecpose_{SIZE}_coco.yml --use-amp --seed=0 -t /path/to/model.pth
```

---

## 🔧 实用工具

### PyTorch 推理 (图像/视频)

```bash
python tools/inference/torch_inf.py \
  -c configs/ecpose/ecpose_{SIZE}_coco.yml \
  -r ecpose_{SIZE}.pth \
  -i /path/to/image_or_video
```

可选参数：`-d cuda:0` (指定设备)、`-t 0.4` (置信度阈值)、`--no-skeleton` (不绘制骨架)。

### 导出 ONNX 模型

```bash
python tools/deployment/export_onnx.py \
  -c configs/ecpose/ecpose_{SIZE}_coco.yml \
  -r ecpose_{SIZE}.pth \
  --check --simplify
```

### ONNX 推理 (图像/视频)

```bash
python tools/inference/onnx_inf.py \
  --onnx ecpose_{SIZE}.onnx \
  --input /path/to/image_or_video \
  --device cuda
```
