
<h3 align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</h3>

## 📋 Table of Contents
- [Model Zoo](#-model-zoo)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Model Configuration](#-model-configuration)
- [Usage](#-usage)
- [Tools](#-tools)

---

## 🏆 Model Zoo

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

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### ⚡ Quick Start (Inference)
The easiest way to test EdgeCrafter is to run inference on a sample image using a pre-trained model.
```bash
# 1. Download a pre-trained model (e.g., ECDet-L)
wget https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth
# 2. Run PyTorch inference
# Make sure to replace `path/to/your/image.jpg` with an actual image path
python tools/inference/torch_inf.py -c configs/ecdet/ecdet_l.yml -r ecdet_l.pth -i path/to/your/image.jpg
```


---

## 📁 Dataset Preparation

### Custom Dataset

<details>
  <summary><strong>Object Detection</strong></summary>
  
  To train on your custom detection dataset in the COCO format, modify the <a href="./configs/dataset/custom.yml">custom.yml</a> configuration file:

<pre><code class="language-yaml">task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']
  verbose: False  # Set to True to output per-category AP

num_classes: 80  # Number of classes in your dataset
remap_mscoco_category: False  # Set to False to prevent automatic remapping of category IDs

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

  <strong>Optional</strong>: To output per-category AP during evaluation, set <code>verbose: True</code> in your dataset configuration:

<pre><code class="language-yaml">evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']
  verbose: True  # Output per-category AP
</code></pre>

</details>

<details close>
  <summary><strong>Instance Segmentation</strong></summary>
To train on a custom segmentation dataset in COCO format, simply follow the detection and update the <strong>img_folder</strong> and <strong>ann_file</strong> paths.
</details>

### COCO2017 Dataset

To reproduce our results on COCO2017, follow these steps:

> **Note**: Due to the non-deterministic nature of grid_sample during backward operations, results may vary slightly (approx. 0.2 AP). For further details, refer to this [PyTorch discussion](https://discuss.pytorch.org/t/f-grid-sample-non-deterministic-backward-results/27566).

1. **Download COCO2017** from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) or the [official COCO website](https://cocodataset.org/#download).

2. **Organize the dataset** as follows:
   ```
   /path/to/COCO2017/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/
   └── val2017/
   ```

3. **Update paths** in [coco.yml](./configs/dataset/coco.yml):
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

## 🔌 Model Configuration

### Object Detection

Model configuration files are located in [configs/ecdet](./configs/ecdet/). Choose the appropriate configuration based on your computational budget and accuracy requirements.

For custom datasets, you may need to adjust specific parameters in your configuration file (e.g. **[ecdet_s.yml](./configs/ecdet/ecdet_s.yml)**):

```yaml
__include__: [
  '../dataset/coco.yml',  # Base dataset configuration. Replace with custom.yml when using a custom dataset
  'ecdet.yml',                      
]

ViTAdapter:
  name: ecvitt
  embed_dim: 192
  num_heads: 3
  interaction_indexes: [10, 11]     # Indices of transformer blocks used for feature interaction/fusion
  weights_path: ecvits/ecvitt.pth   # Pretrained backbone. Automatically downloaded on first use.
  skip_load_backbone: False         # If True, the backbone will be initialized from scratch (no pretrained weights)

optimizer:
  type: AdamW
  params: 
    - # Backbone parameters excluding normalization layers and bias
      params: '^(?=.*backbone)(?!.*(?:norm|bn|bias)).*$'
      lr: 0.000025

    - # Backbone normalization layers (norm/bn) and bias parameters
      params: '^(?=.*backbone)(?=.*(?:norm|bn|bias)).*$'
      lr: 0.000025
      weight_decay: 0.0

    - # Non-backbone normalization layers and bias parameters
      params: '^(?!.*\.backbone)(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0

  lr: 0.0005                # Base learning rate for non-backbone parameters
  betas: [0.9, 0.999]      # AdamW beta coefficients
  weight_decay: 0.0001     # Weight decay applied to parameters except norm/bias

epochs: 74                 # Total training epochs. COCO 6× schedule (12 epochs = 1×) plus 2 extra epochs without augmentation
warmup_iter: 2000         # Number of iterations used for learning rate warmup
lr_gamma: 0.5             # Learning rate decay factor during LR scheduling

eval_spatial_size: [640, 640]  # Input resolution for training/evaluation (height, width). Use [1280,1280] for high-resolution training

train_dataloader:
  total_batch_size: 32      # Global batch size across all GPUs (adjust depending on GPU memory)
  dataset:
    transforms:
      mosaic_epoch: 36        # Apply Mosaic augmentation until this epoch. Recommended to set this to half of stop_epoch
      mosaic_prob: 0.75       # Probability of applying Mosaic augmentation
      stop_epoch: 72          # Disable all augmentations after this epoch (last 2 epochs without augmentation)

  collate_fn:
    mixup_prob: 0.75          # Probability of applying MixUp augmentation
    mixup_epoch: 36           # Apply MixUp augmentation until this epoch. Recommended to set this to half of stop_epoch
```

### Instance Segmentation

Model configuration files are located in [configs/ecseg](./configs/ecseg/). The configuration structure is identical to detection, but inherits from the detection config and adds segmentation-specific settings. 

For custom datasets, you may need to adjust the following settings in your config file(e.g. **[ecseg_s.yml](./configs/ecseg/ecseg_s.yml)**):

```yaml
__include__: [
  '../dataset/coco.yml',  # Base dataset configuration. Replace with custom.yml when using a custom dataset
  '../ecdet/ecdet_s.yml',           # Inherit detection model configuration
  'ecseg.yml',                      # Segmentation-specific configuration
]

train_dataloader:  # Add only the parameters you wish to override
...
```


---

## 🎮 Usage

### Training

Train from scratch using 4 GPUs:

```bash
# Detection
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecdet/ecdet_{SIZE}.yml --use-amp --seed=0

# Segmentation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecseg/ecseg_{SIZE}.yml --use-amp --seed=0
```

Replace `{SIZE}` with `s`, `m`, `l`, or `x` based on your chosen model size.

### Evaluation

Evaluate a trained model:

```bash
# Detection
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecdet/ecdet_{SIZE}.yml --test-only -r /path/to/model.pth

# Segmentation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecseg/ecseg_{SIZE}.yml --test-only -r /path/to/model.pth
```

### Fine-tuning

Fine-tune from a pre-trained checkpoint:

```bash
# Detection
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecdet/ecdet_{SIZE}.yml --use-amp --seed=0 -t /path/to/model.pth

# Segmentation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -c configs/ecseg/ecseg_{SIZE}.yml --use-amp --seed=0 -t /path/to/model.pth
```

---

## 🔧 Tools

Additional utilities and tools can be found in the [tools](./tools/) directory:

- **Visualization Tools**

  PyTorch inference:
  ```bash
  # Detection
  python tools/inference/torch_inf.py -c configs/ecdet/ecdet_{SIZE}.yml -r ecdet_{SIZE}.pth -i example.jpg

  # Segmentation
  python tools/inference/torch_inf.py -c configs/ecseg/ecseg_{SIZE}.yml -r ecseg_{SIZE}.pth -i example.jpg
  ```

  ONNX inference:
  ```bash
  # Detection
  python tools/inference/onnx_inf.py -o ecdet_{SIZE}.onnx -i example.jpg

  # Segmentation
  python tools/inference/onnx_inf.py -o ecseg_{SIZE}.onnx -i example.jpg
  ```

- **Export Tools**

  Export to ONNX format:
  ```bash
  # Detection
  python tools/deployment/export_onnx.py -c configs/ecdet/ecdet_{SIZE}.yml -r ecdet_{SIZE}.pth

  # Segmentation
  python tools/deployment/export_onnx.py -c configs/ecseg/ecseg_{SIZE}.yml -r ecseg_{SIZE}.pth
  ```

