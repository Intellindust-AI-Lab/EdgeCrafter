<h1 align="center">EdgeCrafter: Compact ViTs for Edge Dense Prediction via
Task-Specialized Distillation</h1>

<h3 align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</h3>

<p align="center">
  <a href="https://intellindust-ai-lab.github.io/projects/EdgeCrafter/"><img src="https://img.shields.io/badge/Webpage-EdgeCrafter-blue.svg" alt="Webpage"></a>
  <a href="https://arxiv.org/abs/2603.18739"><img src="https://img.shields.io/badge/arXiv-EdgeCrafter-orange.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>

</p>

<p align="center">
        <a href="https://capsule2077.github.io/">Longfei Liu <sup>*</sup><sup>‡</sup></a>&nbsp;
        <a >Yongjie Hou <sup>*</sup></a>&nbsp;
        <a >Yang Li <sup>*</sup></a>&nbsp;
        <a href='https://qiruiwang0728.github.io/homepage/'>Qirui Wang <sup>*</sup></a>&nbsp;
        <a >Youyang Sha</a>&nbsp; </br>
        <a >Yongjun Yu</a>&nbsp;
        <a >Yinzhi Wang</a>&nbsp;
        <a >Peizhe Ru</a>&nbsp;
        <a href="https://xuanlong-yu.github.io/">Xuanlong Yu<sup>†</sup></a>&nbsp
        <a href="https://xishen0220.github.io/">Xi Shen <sup>†</sup></a> <br><br>
      <a> * Equal Contribution &nbsp;&nbsp; ‡ Project Leader &nbsp;&nbsp; † Corresponding Author</a> <br>
</p>

<p align="center">
    <sup></sup> <a href="https://intellindust-ai-lab.github.io">Intellindust AI Lab</a> <br> 
</p>

<p align="center" style="margin:0; padding:0;">
  <img src=".github/teaser.png">
</p>

---

## 🚀 更新日志

- **[2026-03-21]** <a href="https://huggingface.co/Intellindust">模型已发布至 🤗 Hugging Face</a>。
- **[2026-03-19]** EdgeCrafter 初始版本正式发布。

---

## 🤗 Hugging Face

模型已在 <a href="https://huggingface.co/Intellindust">🤗 Hugging Face</a> 开放下载！也可以通过 [hf_models.ipynb](./hf_models.ipynb) 快速调用模型。欢迎尝试！

---

## 📍 结果复现

- **目标检测与实例分割：** [复现指南](./ecdetseg)
- **姿态估计：** [复现指南](./ecpose)

---

## 🏆 模型库

### COCO2017 Validation Results

> **Note**: Latency is measured on an NVIDIA T4 GPU with batch size 1 under FP16 precision using TensorRT (v10.6).

### Object Detection

| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECDet-S** | 640 | 51.7 | 10 | 26 | 5.41 | [config](ecdetseg/configs/ecdet/ecdet_s.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth) |
| **ECDet-M** | 640 | 54.3 | 18 | 53 | 7.98 | [config](ecdetseg/configs/ecdet/ecdet_m.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_m.pth) |
| **ECDet-L** | 640 | 57.0 | 31 | 101 | 10.49 | [config](ecdetseg/configs/ecdet/ecdet_l.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth) |
| **ECDet-X** | 640 | 57.9 | 49 | 151 | 12.70 | [config](ecdetseg/configs/ecdet/ecdet_x.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_x.pth) |

### Instance Segmentation

| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECSeg-S** | 640 | 43.0 | 10 | 33 | 6.96 | [config](ecdetseg/configs/ecseg/ecseg_s.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_s.pth) |
| **ECSeg-M** | 640 | 45.2 | 20 | 64 | 9.85 | [config](ecdetseg/configs/ecseg/ecseg_m.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_m.pth) |
| **ECSeg-L** | 640 | 47.1 | 34 | 111 | 12.56 | [config](ecdetseg/configs/ecseg/ecseg_l.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth) |
| **ECSeg-X** | 640 | 48.4 | 50 | 168 | 14.96 | [config](ecdetseg/configs/ecseg/ecseg_x.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_x.pth) |

### Pose Estimation

| Model | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----:|:--:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECPose-S** | 640 | 68.9 |  10 | 30 | 5.54 | [config](ecpose/configs/ecpose/ecpose_s_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth) |
| **ECPose-M** | 640 | 72.4 |  20 | 63 | 9.25 | [config](ecpose/configs/ecpose/ecpose_m_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_m.pth) |
| **ECPose-L** | 640 | 73.5 |  34 | 112 | 11.83 | [config](ecpose/configs/ecpose/ecpose_l_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth) |
| **ECPose-X** | 640 | 74.8 |  51 | 172 | 14.31 | [config](ecpose/configs/ecpose/ecpose_x_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_x.pth) |
---

## 📦 安装

```bash
# 创建并激活 conda 环境
conda create -n ec python=3.11 -y
conda activate ec

# 安装依赖
pip install -r requirements.txt
```

### ⚡ 快速上手（模型推理）
可以通过预训练模型对示例图像进行推理，以快速测试 EdgeCrafter 的性能。
```bash
# 1. 进入对应目录并下载预训练权重（以 ECDet-L 为例）
cd ecdetseg
wget https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth

# 2. 运行 PyTorch 推理
# 请将 `path/to/your/image.jpg` 替换为实际图像的路径
python tools/inference/torch_inf.py -c configs/ecdet/ecdet_l.yml -r ecdet_l.pth -i path/to/your/image.jpg
```

## 📄 开源协议

本项目遵循 [Apache 2.0 许可证](./LICENSE) 开源。

---

## 🙏 致谢

感谢以下开源项目为本工作提供的支持与启发：[RT-DETR](https://github.com/lyuwenyu/RT-DETR)、[D-FINE](https://github.com/Peterande/D-FINE)、[DEIM](https://github.com/Intellindust-AI-Lab/DEIM)、[lightly-train](https://github.com/lightly-ai/lightly-train)、[DETRPose](https://github.com/SebastianJanampa/DETRPose)、[RF-DETR](https://github.com/roboflow/rf-detr)、[DINOv3](https://github.com/facebookresearch/dinov3)

--- 

## 📚 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{liu2026edgecrafter,
  title={EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation},
  author={Liu, Longfei and Hou, Yongjie and Li, Yang and Wang, Qirui and Sha, Youyang and Yu, Yongjun and Wang, Yinzhi and Ru, Peizhe and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2026}
}
```
