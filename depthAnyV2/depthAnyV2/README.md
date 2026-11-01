# Depth Anything V2 本地部署说明书（度量深度版）

## 适用场景

固定摄像头车辆监控（单目度量深度估计）。**输出真实米数**，可直接得到"车离镜头多少米"。无需多视图，逐帧处理。

## 与旧版（相对深度版）的区别

| | 旧版（相对深度） | 本版（度量深度） |
|---|---|---|
| 教师模型 | Depth-Anything-V2-Large-hf | **Depth-Anything-V2-Metric-Outdoor-Large-hf** |
| 学生模型 | Depth-Anything-V2-Base-hf | **Depth-Anything-V2-Metric-Outdoor-Base-hf** |
| 输出 | 相对深度（无量纲，只比远近） | **真实米数（0~80m）** |
| 训练数据存储 | 单个 .npz（爆内存风险） | **逐张 .npy（支持几万张）** |
| 适用 | 仅看深浅 | 算绝对距离 |

## 一、训练流程

### 前置准备

```bash
conda activate yolo_depth
set HF_ENDPOINT=https://hf-mirror.com
set CUDA_VISIBLE_DEVICES=0
```

依赖（环境 `yolo_depth` 已全部安装，无需额外操作）：
- torch 2.5.1+cu121、torchvision 0.20.1、transformers 5.15.0
- opencv-python、numpy、pillow、safetensors

### Step 1: 放置原始图片

将视频抽帧得到的 JPG 图片放入：

```
datasets/my_dataset/images/
  ├── frame_001.jpg
  ├── frame_002.jpg
  └── ...
```

### Step 2: 数据准备（生成米数伪标签）

```bash
python prepare_data.py --image-dir datasets/my_dataset/images --output-dir datasets/my_dataset/processed
```

用 DA2-Metric-Outdoor-Large 教师模型生成真实米数伪标签（0~80m）。逐张保存，支持几万张不爆内存。

输出文件（保存在 `datasets/my_dataset/processed/`）：

```
processed/
  ├── image_files.npy          (图片路径列表)
  └── depths/
      ├── 00000.npy            (第0张深度图, float32, 单位米)
      ├── 00001.npy
      └── ...
```

### Step 3: 模型训练

```bash
# 阶段 1: Head 微调（冻结 backbone，仅训练 DPT head）
python train.py --data-dir datasets/my_dataset/processed --output-dir checkpoints/my_model --stage 1 --epochs 5

# 阶段 2: 全量微调（解冻 backbone 联合训练）
python train.py --data-dir datasets/my_dataset/processed --output-dir checkpoints/my_model --stage 2 --epochs 10 --resume checkpoints/my_model/best_model.pt
```

训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | 必填 | prepare_data.py 的输出目录 |
| `--output-dir` | 必填 | checkpoint 保存目录 |
| `--model-name` | `Depth-Anything-V2-Metric-Outdoor-Base-hf` | 预训练学生模型 |
| `--stage` | 1 | 1=Head微调, 2=全量微调 |
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 2 | 批大小（8GB显存建议1-2） |
| `--process-res` | 518 | 处理分辨率（须为14的倍数） |
| `--grad-accumulate` | 4 | 梯度累积步数 |
| `--resume` | 无 | 从 checkpoint 恢复训练 |

### Step 4: 模型评估

```bash
python evaluate.py --checkpoint checkpoints/my_model/best_model.pt --image-dir datasets/my_dataset/images --output-dir eval_results --compare
```

输出米数统计（最小/最大/均值/中位数）+ 以下定量指标：

| 指标 | 含义 | 方向 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|------|------|-----|
| **AbsRel** | 绝对相对误差：\|预测-标签\|/标签 的均值 | 越低越好 | < 0.10 | 0.10~0.20 | 0.20~0.30 | > 0.30 |
| **RMSE** | 均方根误差（单位：米） | 越低越好 | < 1.0m | 1.0~3.0m | 3.0~5.0m | > 5.0m |
| **δ<1.25** | 预测值与标签比值 < 1.25 的像素占比 | 越高越好 | > 90% | 70%~90% | 50%~70% | < 50% |

**指标怎么读**：
- 微调后 AbsRel 和 RMSE 应**下降**，δ<1.25 应**上升** → 说明微调有效
- 如果指标没变好甚至变差 → 说明训练参数有问题或过拟合，需调整

> **注意**：本项目用自蒸馏方案，"标签"是教师模型（DA2-Large）生成的伪标签，非真实深度。因此上述指标衡量的是"学生模型与教师模型的接近程度"，不是绝对精度。如需评估绝对精度，需配真实深度相机/LiDAR 数据。

---

## 二、验证/推理流程

### 放置待验证文件

```
detect/input/
  ├── test.jpg
  └── test.mp4
```

### 运行推理

```bash
# 单图推理（输出对比图 + 深度图 + 米数 .npy）
python detect/detect.py --source detect/input/test.jpg --model-depth checkpoints/my_model/best_model.pt --output detect/output

# 视频推理（逐帧生成深度，输出左右对比视频）
python detect/detect.py --source detect/input/test.mp4 --model-depth checkpoints/my_model/best_model.pt --output detect/output

# 目录批量推理
python detect/detect.py --source detect/input/ --model-depth checkpoints/my_model/best_model.pt --output detect/output

# 使用预训练模型（未训练前可直接用，已默认 Metric-Outdoor-Base）
python detect/detect.py --source detect/input/ --output detect/output
```

单图输出文件：

| 文件 | 内容 |
|------|------|
| `xxx_comparison.png` | 原图 + 深度图左右对比（标注米数范围） |
| `xxx_depth.png` | 彩色深度图 |
| `xxx_depth.npy` | 深度矩阵（float32，单位米，可直接 `np.load` 读取） |

---

## 三、完整流程速查

```
1. 放图片 → datasets/my_dataset/images/
2. python prepare_data.py --image-dir datasets/my_dataset/images --output-dir datasets/my_dataset/processed
3. python train.py --data-dir datasets/my_dataset/processed --output-dir checkpoints/my_model --stage 1 --epochs 5
4. python train.py --data-dir datasets/my_dataset/processed --output-dir checkpoints/my_model --stage 2 --epochs 10 --resume checkpoints/my_model/best_model.pt
5. python evaluate.py --checkpoint checkpoints/my_model/best_model.pt --image-dir datasets/my_dataset/images --output-dir eval_results --compare
6. 放验证文件 → detect/input/
7. python detect/detect.py --source detect/input/ --model-depth checkpoints/my_model/best_model.pt --output detect/output
```

---

## 四、可用模型

### 度量深度模型（输出真实米数，本版使用）

| 模型 | HuggingFace ID | 训练数据 | max_depth | 显存 |
|------|----------------|---------|-----------|------|
| Metric-Outdoor-Small | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | Virtual KITTI 2（室外车载） | 80m | 4GB+ |
| Metric-Outdoor-Base | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf` | Virtual KITTI 2 | 80m | 6GB+ |
| Metric-Outdoor-Large | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | Virtual KITTI 2 | 80m | 10GB+ |
| Metric-Indoor-Small | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Hypersim（室内） | 20m | 4GB+ |
| Metric-Indoor-Base | `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf` | Hypersim | 20m | 6GB+ |
| Metric-Indoor-Large | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | Hypersim | 20m | 10GB+ |

> **车辆监控选 Outdoor 系列**（Virtual KITTI 2 是自动驾驶场景训练，匹配度高）。
> 教师=Large（精度最高），学生=Base（平衡）。

### 相对深度模型（旧版，仅比远近，无米数）

| 模型 | HuggingFace ID |
|------|----------------|
| DA2-Small | `depth-anything/Depth-Anything-V2-Small-hf` |
| DA2-Base | `depth-anything/Depth-Anything-V2-Base-hf` |
| DA2-Large | `depth-anything/Depth-Anything-V2-Large-hf` |

---

## 五、文件说明

| 文件 | 作用 |
|------|------|
| `prepare_data.py` | 数据准备：DA2-Metric-Large 教师生成米数伪标签（逐张保存） |
| `train.py` | 训练脚本：SILoss + 梯度匹配损失，分阶段微调，lazy loading |
| `evaluate.py` | 评估脚本：与预训练模型对比，米数统计 + 定量指标 |
| `detect/detect.py` | 推理脚本：支持图片/视频/目录，输出米数 + 左右对比图 |
| `detect/input/` | 放入验证文件 |
| `detect/output/` | 推理输出目录 |
| `datasets/my_dataset/images/` | 放入训练图片 |
| `datasets/my_dataset/processed/` | 数据准备输出（image_files.npy + depths/） |
| `checkpoints/my_model/` | 模型输出目录（best_model.pt / latest_model.pt） |
| `eval_results/` | 评估输出目录 |

---

## 六、关键改动（相对旧版）

1. **教师模型** → Metric-Outdoor-Large（输出米数而非相对深度）
2. **学生模型** → Metric-Outdoor-Base（输出米数）
3. **数据存储** → 逐张 `.npy`（修复 ragged array，支持几万张）
4. **训练加载** → lazy loading（按 idx 读盘，不爆内存）
5. **amp API** → 新版 `torch.amp.autocast/GradScaler`（修复弃用警告）
6. **损失函数** → 适配米数范围（有效区间 0.1~100m）
7. **Stage 2 LR** → backbone 5e-6（原 1e-5，避免破坏预训练特征）
8. **detect 单图** → 增加左右对比图 + 米数标注
9. **evaluate** → 修复 loss 格式化 bug，增加米数统计
