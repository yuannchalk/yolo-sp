# YOLO + SuperPoint 知识蒸馏项目

基于 YOLO 目标检测模型的特征图，蒸馏 SuperPoint 特征点检测能力。

## 项目概述

本项目实现了一个知识蒸馏框架，将预训练的 SuperPoint 教师模型的知识迁移到基于 YOLO backone 的学生模型中。通过这种方式，可以在保持 YOLO 目标检测能力的同时，获得特征点检测功能。

### 核心思想

```
原始方案：图像 → SuperPoint → 特征点
    问题：计算量大，无法利用目标检测的先验信息

蒸馏方案：图像 → YOLO → 特征图 → Adapter → SP Head → 特征点
                    ↓
              SuperPoint (教师) → 特征点 (提供监督信号)
```

### 技术亮点

- **通道适配器**：将 YOLO 的 256 通道特征图适配到 SuperPoint 的 128 维描述子空间
- **输出级蒸馏**：直接对齐学生与教师的热图和描述子输出
- **轻量级**：仅训练 Adapter，YOLO 和 SuperPoint 保持冻结

## 项目结构

```
YOLO_SP_Project/
├── models/                      # 核心模型定义
│   ├── yolo_sp.py              # YOLO+SP 融合蒸馏模型
│   ├── adapter.py              # 通道适配器 (StrongAdapter)
│   └── losses.py               # 蒸馏损失函数
├── tools/                       # 训练和推理脚本
│   ├── train.py                # 蒸馏训练脚本
│   ├── infer_final_robust.py   # 融合模型推理
│   └── test_pure_yolo.py       # 纯 YOLO 检测对比
├── superpoint_lib/             # SuperPoint 教师模型
│   └── superpoint_offical.py   # SuperPoint PyTorch 实现
├── superpoints/                # 辅助工具
│   ├── generate_pseudo_labels.py  # 伪标签生成
│   └── superpoint_pytorch.py      # SuperPoint 参考实现
├── data/                       # 数据集加载
│   └── dataset.py             # 蒸馏数据集实现
├── utils/                      # 公共工具
│   └── common.py              # 公共函数 (NMS, letterbox 等)
├── configs/                    # 配置文件
│   └── config.yaml            # 训练配置
├── spilt_dataset.py           # 数据集划分脚本
├── creat_config.py            # 配置文件生成脚本
├── degub_hook.py              # Hook 调试脚本
└── .gitignore                 # Git 忽略配置
```

## 技术细节

### 模型架构

#### 1. YOLOSP_Distiller (models/yolo_sp.py)

主模型类，整合 YOLO、SuperPoint 和 Adapter：

```python
class YOLOSP_Distiller(nn.Module):
    def __init__(self, yolo_weights, sp_weights, device):
        # YOLO Backbone (冻结)
        self.yolo = YOLO(yolo_weights).model
        # SuperPoint 教师 (冻结)
        self.sp_teacher = SuperPoint()
        # 通道适配器 (可训练)
        self.adapter = StrongAdapter(in_channels=256, out_channels=128)
        # SP 检测头和描述子头 (复用教师)
        self.sp_detector = self.sp_teacher.detector
        self.sp_descriptor = self.sp_teacher.descriptor
```

#### 2. StrongAdapter (models/adapter.py)

通道适配器，将 YOLO 特征 (256ch) 转换为 SuperPoint 格式 (128ch)：

```
输入 [B, 256, H, W]
    ↓ Conv1x1 + BN + ReLU
[B, 128, H, W]
    ↓ Channel Attention (通道注意力)
[B, 128, H, W]
    ↓ Spatial Attention (空间注意力)
[B, 128, H, W]
    ↓ Residual Block × 2 (残差增强)
输出 [B, 128, H, W]
```

#### 3. 损失函数 (models/losses.py)

**OutputDistillationLoss** - 输出级蒸馏损失：

```python
# 1. 热图对齐损失 (KL 散度)
loss_heatmap = KL_divergence(
    softmax(student_scores / T),
    softmax(teacher_scores / T)
) * T²

# 2. 描述子对齐损失 (余弦相似度)
loss_descriptor = 1 - mean(cosine_similarity(
    student_descriptors_at_keypoints,
    teacher_descriptors_at_keypoints
))

# 总损失
total_loss = heatmap_weight * loss_heatmap + desc_weight * loss_descriptor
```

### 关键函数

#### batched_nms (utils/common.py)

批处理非极大值抑制，用于抑制密集热图中的冗余检测点：

```
原理：使用 max_pool 检测局部最大值，迭代 2 次处理邻近响应
输入：scores [B, H, W], nms_radius
输出：scores [B, H, W] (仅保留局部最大值)
```

#### letterbox (utils/common.py)

保持宽高比的图像预处理：

```
输入：原始图像 [H, W, 3], target_size=640
处理：
  1. 计算缩放比例 = min(640/H, 640/W)
  2. 按比例 resize
  3. 边缘填充灰色 (114, 114, 114)
输出：padded [640, 640, 3], scale, pad_left, pad_top
```

## 快速开始

### 环境要求

```
Python >= 3.8
PyTorch >= 1.10
OpenCV >= 4.5
NumPy
PyYAML
tqdm
ultralytics (YOLOv8)
```

### 安装依赖

```bash
pip install torch torchvision opencv-python numpy pyyaml tqdm ultralytics
```

### 准备数据

1. 将图像放入 `datasets/fire/images/` 目录
2. 使用 `spilt_dataset.py` 划分训练/验证集：

```python
# 修改 spilt_dataset.py 中的路径配置
SOURCE_IMG_DIR = "your/images/path"
SOURCE_LABEL_DIR = "your/labels/path"
OUTPUT_DIR = "datasets/fire"

# 运行
python spilt_dataset.py
```

### 训练模型

1. 修改 `configs/config.yaml`：

```yaml
weights:
  yolo: weights/yolov8n-fire.pt
  superpoint: weights/superpoint_v6_from_tf.pth

data:
  train_img_dir: datasets/fire/images/train
  val_img_dir: datasets/fire/images/val

train:
  distill_epochs: 80
  batch_size: 8
  distill_lr: 0.001
  temperature: 3.0
```

2. 开始训练：

```bash
python tools/train.py
```

训练完成后，模型保存在 `tools/checkpoints/best_adapter.pth`

### 模型推理

```bash
# 融合模型推理
python tools/infer_final_robust.py

# 或使用纯 YOLO 对比
python tools/test_pure_yolo.py
```

## 配置文件说明 (configs/config.yaml)

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `weights.yolo` | YOLO 权重路径 | yolov8n-fire.pt |
| `weights.superpoint` | SuperPoint 预训练权重 | superpoint_v6_from_tf.pth |
| `model.input_size` | 输入图像尺寸 | 640 |
| `train.distill_epochs` | 蒸馏训练轮数 | 80 |
| `train.batch_size` | 批大小 | 8 |
| `train.distill_lr` | 蒸馏学习率 | 0.001 |
| `train.temperature` | KL 散度温度参数 | 3.0 |

## 推理输出说明

运行推理脚本后：

```
tools/checkpoints/
├── best_adapter.pth           # 训练好的 Adapter 权重
├── final_result_robust.jpg   # 融合模型结果 (红色框=火焰, 绿色点=特征点)
└── final_result_all_keypoints.jpg  # 所有检测到的特征点
```

## 贡献指南

### 代码规范

- 使用 Python 3.8+ 语法
- 公共函数添加 docstring
- 异常使用具体类型捕获

### 添加新模块

1. 在对应目录创建模块文件
2. 如有公共函数，添加到 `utils/common.py`
3. 更新本 README

### 测试

```bash
# 调试 Hook
python degub_hook.py

# 测试数据集加载
python -c "from data.dataset import DistillDataset; ds = DistillDataset('path')"
```

## 相关论文

- **SuperPoint**: Self-Supervised Interest Point Detection and Description
- **YOLOv8**: Ultralytics YOLOv8
- **Knowledge Distillation**: Model Compression and Speedup

## 许可证

MIT License

## 致谢

- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) - Magic Leap
- [ultralytics](https://github.com/ultralytics/ultralytics) - Ultralytics YOLOv8
