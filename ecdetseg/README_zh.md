<h3 align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</h3>

## 📋 目录
- [模型库](#-模型库)
- [安装指南](#-安装指南)
- [数据集准备](#-数据集准备)
- [模型配置](#-模型配置)
- [基本用法](#-基本用法)
- [实用工具](#-实用工具)

---

## 🏆 模型库

### COCO2017 Validation Results

> **Note**: Latency is measured on an NVIDIA T4 GPU with batch size 1 under FP16 precision using TensorRT (v10.6).

### Object Detection

| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECDet-S** | 640 | 51.7 | 10 | 26 | 5.41 | [config](./configs/ecdet/ecdet_s.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth) |
| **ECDet-M** | 640 | 54.3 | 18 | 53 | 7.98 | [config](./configs/ecdet/ecdet_m.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_m.pth) |
| **ECDet-L** | 640 | 57.0 | 31 | 101 | 10.49 | [config](./configs/ecdet/ecdet_l.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth) |
| **ECDet-X** | 640 | 57.9 | 49 | 151 | 12.70 | [config](./configs/ecdet/ecdet_x.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_x.pth) |

### Instance Segmentation

| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECSeg-S** | 640 | 43.0 | 10 | 33 | 6.96 | [config](./configs/ecseg/ecseg_s.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_s.pth) |
| **ECSeg-M** | 640 | 45.2 | 20 | 64 | 9.85 | [config](./configs/ecseg/ecseg_m.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_m.pth) |
| **ECSeg-L** | 640 | 47.1 | 34 | 111 | 12.56 | [config](./configs/ecseg/ecseg_l.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth) |
| **ECSeg-X** | 640 | 48.4 | 50 | 168 | 14.96 | [config](./configs/ecseg/ecseg_x.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_x.pth) |

---
## 📦 安装指南

```bash
# 安装依赖
pip install -r requirements.txt
```

### ⚡ 快速开始（推理示例）
测试 EdgeCrafter 最直接的方法是使用预训练模型对示例图像进行推理。
```bash
# 1. 下载预训练模型（以 ECDet-L 为例）
wget https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth
# 2. 运行 PyTorch 推理
# 请将 `path/to/your/image.jpg` 替换为实际图像路径
python tools/inference/torch_inf.py -c configs/ecdet/ecdet_l.yml -r ecdet_l.pth -i path/to/your/image.jpg
```

---

## 📁 数据集准备

### 自定义数据集

<details>
  <summary><strong>目标检测</strong></summary>
  
  若要使用 COCO 格式的自定义数据集进行训练，请参考并修改 <a href="./configs/dataset/custom.yml">custom.yml</a> 配置文件：

<pre><code class="language-yaml">task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']
  verbose: False  # 设置为 True 以显示每个类别的 AP 结果
blocks
num_classes: 80  # 数据集类别总数
remap_mscoco_category: False  # 设置为 False 以禁用类别 ID 的自动映射

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/your/dataset/train
    ann_file: /path/to/your/dataset/train/annotations.json
    ...
val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/your/dataset/val
    ann_file: /path/to/your/dataset/val/annotations.json
    ...
</code></pre>

  <strong>可选操作</strong>：若需在评估时查看具体每个类别的 AP，请在配置文件中设置 <code>verbose: True</code>：

<pre><code class="language-yaml">evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']
  verbose: True  # 输出详细的分类别 AP
</code></pre>

</details>

<details close>
  <summary><strong>实例分割</strong></summary>
若要使用 COCO 格式的自定义分割数据集，其配置流程与检测任务一致，只需同步更新 <strong>img_folder</strong> 和 <strong>ann_file</strong> 对应的路径即可。
</details>

### COCO2017 数据集

按照以下步骤复现我们在 COCO2017 上的实验结果：

> **注意**：由于 PyTorch 中 `grid_sample` 算子在反向传播时的非确定性（non-deterministic），实验结果可能会有约 0.2 AP 的微小波动。详见 [PyTorch 社区讨论](https://discuss.pytorch.org/t/f-grid-sample-non-deterministic-backward-results/27566)。

1. **获取数据集**：从 [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) 或 [COCO 官网](https://cocodataset.org/#download)下载 COCO2017。

2. **目录结构**：请按以下格式组织数据集文件：
   ```
   /path/to/COCO2017/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/
   └── val2017/
   ```

3. **配置路径**：在 [coco.yml](./configs/dataset/coco.yml) 中更新相关路径：
   ```yaml
   train_dataloader:
     dataset:
       img_folder: /path/to/COCO2017/train2017/
       ann_file: /path/to/COCO2017/annotations/instances_train2017.json
   
   val_dataloader:
     dataset:
       img_folder: /path/to/COCO2017/val2017/
       ann_file: /path/to/COCO2017/annotations/instances_val2017.json
   ```

---

## 🔌 模型配置

### 目标检测

配置文件位于 [configs/ecdet](./configs/ecdet/) 目录下。请根据具体的计算资源和精度需求选择合适的配置。

对于自定义数据集，通常需要调整配置文件（如 **[ecdet_s.yml](./configs/ecdet/ecdet_s.yml)**）中的以下关键参数：

```yaml
__include__: [
  '../dataset/coco.yml',  # 基础数据集配置。若使用自定义数据请替换为 custom.yml
  'ecdet.yml',                      
]

ViTAdapter:
  name: ecvitt
  embed_dim: 192
  num_heads: 3
  interaction_indexes: [10, 11]     # 用于特征交互与融合的 Transformer 块索引
  weights_path: ecvits/ecvitt.pth   # 预训练骨干网络路径。首次运行会自动下载。
  skip_load_backbone: False         # 若设为 True，则骨干网络将随机初始化（不加载预训练权重）

optimizer:
  type: AdamW
  params: 
    - # 骨干网络参数（不含归一化层和偏置）
      params: '^(?=.*backbone)(?!.*(?:norm|bn|bias)).*$'
      lr: 0.000025

    - # 骨干网络中的归一化层 (norm/bn) 和偏置
      params: '^(?=.*backbone)(?=.*(?:norm|bn|bias)).*$'
      lr: 0.000025
      weight_decay: 0.0

    - # 非骨干网络部分的归一化层和偏置
      params: '^(?!.*\.backbone)(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0

  lr: 0.0005                # 非骨干网络参数的基础学习率
  betas: [0.9, 0.999]      # AdamW 动量系数
  weight_decay: 0.0001     # 权重衰减系数（不作用于归一化层和偏置）

epochs: 74                 # 总训练轮数。COCO 6x 策略（1x = 12 轮）外加 2 轮关闭增强的微调
warmup_iter: 2000         # 学习率预热步数
lr_gamma: 0.5             # 学习率衰减率

eval_spatial_size: [640, 640]  # 训练及评估的输入分辨率 (H, W)。高分辨率训练建议设为 [1280, 1280]

train_dataloader:
  total_batch_size: 32      # 全局 Batch Size（需根据显存容量调整）
  dataset:
    transforms:
      mosaic_epoch: 36        # Mosaic 增强截止轮次。通常建议设为 stop_epoch 的一半。
      mosaic_prob: 0.75       # Mosaic 增强触发概率
      stop_epoch: 72          # 增强停止轮次（最后 2 轮通常关闭数据增强以利于收敛）

  collate_fn:
    mixup_prob: 0.75          # MixUp 增强概率
    mixup_epoch: 36           # MixUp 增强截止轮次
```

### 实例分割

配置文件位于 [configs/ecseg](./configs/ecseg/) 目录下。其配置结构与检测任务完全一致，通过继承检测配置并补充分割相关的特定设置实现。

对于自定义数据集，建议在配置文件（如 **[ecseg_s.yml](./configs/ecseg/ecseg_s.yml)**）中重点关注以下差异项：

```yaml
__include__: [
  '../dataset/coco.yml',  # 基础数据集配置
  '../ecdet/ecdet_s.yml',           # 继承检测模型配置
  'ecseg.yml',                      # 叠加分割任务专属配置
]

train_dataloader:  # 仅需列出需要覆盖或新增的参数
...
```

---

## 🎮 基本用法

### 训练

使用 4 张 GPU 从头开始训练：

```bash
# 目标检测
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecdet/ecdet_{SIZE}.yml --use-amp --seed=0

# 实例分割
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecseg/ecseg_{SIZE}.yml --use-amp --seed=0
```

请将 `{SIZE}` 替换为实际的模型规格：`s`、`m`、`l` 或 `x`。

### 评估

对已有的模型权重进行评估：

```bash
# 目标检测
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecdet/ecdet_{SIZE}.yml --test-only -r /path/to/model.pth

# 实例分割
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecseg/ecseg_{SIZE}.yml --test-only -r /path/to/model.pth
```

### 微调

加载预训练权重进行微调：

```bash
# 目标检测
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecdet/ecdet_{SIZE}.yml --use-amp --seed=0 -t /path/to/model.pth

# 实例分割
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecseg/ecseg_{SIZE}.yml --use-amp --seed=0 -t /path/to/model.pth
```

---

## 🔧 实用工具

更多功能脚本详见 [tools](./tools/) 目录：

- **可视化工具**

  PyTorch 推理：
  ```bash
  # 检测
  python tools/inference/torch_inf.py -c configs/ecdet/ecdet_{SIZE}.yml -r ecdet_{SIZE}.pth -i example.jpg

  # 分割
  python tools/inference/torch_inf.py -c configs/ecseg/ecseg_{SIZE}.yml -r ecseg_{SIZE}.pth -i example.jpg
  ```

  ONNX 推理：
  ```bash
  # 检测
  python tools/inference/onnx_inf.py -o ecdet_{SIZE}.onnx -i example.jpg

  # 分割
  python tools/inference/onnx_inf.py -o ecseg_{SIZE}.onnx -i example.jpg
  ```

- **导出工具**

  导出为 ONNX 格式：
  ```bash
  # 检测
  python tools/deployment/export_onnx.py -c configs/ecdet/ecdet_{SIZE}.yml -r ecdet_{SIZE}.pth

  # 分割
  python tools/deployment/export_onnx.py -c configs/ecseg/ecseg_{SIZE}.yml -r ecseg_{SIZE}.pth
  ```
