# ViT Inference Acceleration for Breast Cancer Histopathology

## 📖 Project Overview
This repository provides a comprehensive inference acceleration framework for Vision Transformers (ViT) specialized in breast cancer histopathology four-class classification. The project addresses the computational bottlenecks associated with large-scale vision foundation models in clinical diagnostic settings. 

By leveraging the **Huawei Cloud Ascend NPU** platform, this implementation achieves significant throughput improvements through a multi-stage optimization pipeline, including structured pruning, decoupled knowledge distillation, and SVD-based low-rank decomposition.

## 🛠️ Technical Innovations

* **Feature Enhancement**: Incorporates a Gated Feature Enhancer and Prior-based Attention Refiner to bridge the gap between general-purpose foundation features and specific pathological diagnostic requirements.
* **Multi-Stage Compression**: A cascaded optimization pipeline consisting of:
    * **Transformer Block Truncation**: Reducing depth for faster sequential processing.
    * **Global L1 Unstructured Pruning**: Removing redundant weights while maintaining structural integrity.
    * **Decoupled Knowledge Distillation (DKD)**: Transferring complex diagnostic knowledge from a heavy teacher model to a lightweight student.
* **NPU Optimization**: Seamless integration with the **Ascend Model Compression Toolkit (AMCT)** for INT8 quantization and the **Ascend Tensor Compiler (ATC)** for generating high-performance offline models (.om).

## 📁 Repository Structure

```text
Huawei-ICT-ViT-Acceleration/
├── src/                           # Core logic modules
│   ├── data/                      # Preprocessing, balancing, and loaders
│   ├── models/                    # UNI backbone, enhancers, and MIL aggregators
│   ├── compression/               # Pruning, DKD distillation, and SVD logic
│   └── engine/                    # Training, evaluation, and NPU quantization
├── notebooks/                     # Documented experiment logs and visualizations
├── tools/                         # Cloud-sync, hardware monitoring, and export utils
├── configs/                       # Global YAML configurations
└── docs/                          # Design reports and support materials
```

## 💻 Environment & Huawei Cloud Images

Development and optimization were conducted on **Huawei Cloud ModelArts**. Specific images were utilized to ensure hardware compatibility:

* **Training & Distillation**: `PyTorch-1.10.2` (Cuda 10.2, Python 3.7). Used for initial fine-tuning and knowledge distillation on Tesla V100 instances.
* **NPU Optimization**: `PyTorch-2.1.0` (Ascend CANN 7.0, Python 3.9). Used for AMCT quantization, NPU resource monitoring, and ATC model compilation.

## 📊 Data & Weight Sources

* **Dataset**: Sourced from the **ICIAR 2018 Grand Challenge (BACH)** for breast cancer histopathology.
* **Foundation Model**: The backbone utilizes the **UNI Model**, a vision foundation model for pathology developed by Mahmood Lab, Harvard Medical School.

---

# 乳腺癌病理图像 ViT 推理加速方案

## 📖 项目概览
本项目提供了一个针对乳腺癌病理图像四分类任务的 Vision Transformer (ViT) 推理加速框架。该项目旨在解决大规模视觉基座模型在临床诊断场景中的计算瓶颈问题。

通过利用**华为云昇腾 (Ascend) NPU** 平台，本方案通过多级优化流水线（包括结构化剪枝、解耦知识蒸馏和基于 SVD 的低秩分解）实现了显著的推理吞吐量提升。

## 🛠️ 技术创新

* **特征增强**：引入门控特征增强器（Gated Feature Enhancer）和基于先验的注意力精炼器（Prior-based Attention Refiner），弥合通用基座特征与特定病理诊断需求之间的差距。
* **多级模型压缩**：采用级联优化流水线，包括：
    * **Transformer 块截断**：通过减小深度加速顺序处理。
    * **全局 L1 非结构化剪枝**：在保持结构完整性的同时去除冗余权重。
    * **解耦知识蒸馏 (DKD)**：将复杂的诊断知识从重型教师模型迁移至轻量级学生模型。
* **昇腾 NPU 优化**：无缝集成**昇腾模型压缩工具包 (AMCT)** 进行 INT8 量化，并使用**昇腾张量编译器 (ATC)** 生成高性能离线模型 (.om)。

## 📁 仓库结构

```text
Huawei-ICT-ViT-Acceleration/
├── src/                           # 核心逻辑模块
│   ├── data/                      # 预处理、数据均衡与加载器
│   ├── models/                    # UNI 骨干网络、增强器与 MIL 聚合器
│   ├── compression/               # 剪枝、DKD 蒸馏与 SVD 逻辑
│   └── engine/                    # 训练、评估与 NPU 量化
├── notebooks/                     # 实验日志记录与可视化
├── tools/                         # 云同步、硬件监控与导出工具
├── configs/                       # 全局 YAML 配置
└── docs/                          # 设计报告与支持材料
```

## 💻 环境与华为云镜像

开发与优化工作在**华为云 ModelArts** 上完成。为确保硬件兼容性，不同阶段使用了特定的镜像：

* **训练与蒸馏阶段**：`PyTorch-1.10.2` (Cuda 10.2, Python 3.7)。用于在 Tesla V100 实例上进行初始微调和知识蒸馏。
* **NPU 优化阶段**：`PyTorch-2.1.0` (Ascend CANN 7.0, Python 3.9)。用于 AMCT 量化、NPU 资源监控及 ATC 模型编译。

## 📊 数据与权重来源

* **数据集**：采用 **ICIAR 2018 Grand Challenge (BACH)** 乳腺癌病理图像数据集。
* **基座模型**：骨干网络采用 **UNI 模型**，这是由哈佛医学院 Mahmood 实验室开发的病理视觉基座模型。
