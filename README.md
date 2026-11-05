# 深目 DepthVision — 深度视觉检测系统

基于 **YOLO 目标检测** 与 **Depth Anything V2 单目度量深度估计** 的车辆视觉检测系统。
从固定摄像头画面中同时回答两个问题：**画面里有没有车（YOLO）**、**车离镜头多少米（度量深度）**，并通过多规则融合输出可靠的"有车/无车"判定，支持夜间、眩光、远距离等复杂场景。

## 核心特性

- **单目度量深度**：无需 LiDAR / 双目相机，单个摄像头直接输出 0~80m 真实米数深度图
- **自蒸馏微调**：教师模型（DA2-Metric-Outdoor-Large）生成伪标签 → 学生模型（Metric-Outdoor-Base）微调，支持 Stage1 Head 微调 / Stage2 全量微调 / LoRA / 时序一致性损失
- **多规则融合判定**：深度监测点变化（主规则 70%）+ YOLO 视觉检测（30%）+ MOG2 运动检测，加权融合输出有车置信度
- **夜间鲁棒性**：CLAHE 去眩光、亮度自适应阈值、过曝点饱和屏蔽、夜间时序平滑与冷却期防误报
- **全工程流水线**：视频抽帧 → 数据清洗 → 数据集划分 → 训练 → 评估（AbsRel/RMSE/δ<1.25）→ 推理部署
- **车辆侧面图像拼接**：YOLO 定位 + SIFT/ORB 特征匹配 + 多帧融合，从视频重建完整车辆侧面图
- **配置驱动**：所有参数集中在 `config.yaml`，改配置即生效，支持命令行覆盖
- **工程化保障**：OOM 自愈训练管理器、断点续训、增量数据准备（逐张 .npy 存储不爆内存）、Windows 兼容适配

## 项目结构

```
Visual_inspection/
├── depthAnyV2/                  # 子项目一：单目度量深度估计 + 车辆融合检测
│   └── depthAnyV2/
│       ├── config.yaml              # 统一配置（8 个段落控制全部脚本）
│       ├── prepare_data.py          # 数据准备：教师模型生成米数伪标签
│       ├── train.py                 # 深度模型自蒸馏微调
│       ├── evaluate.py              # 评估：AbsRel / RMSE / δ<1.25
│       ├── train_judge.py           # 判定 MLP 训练（知识蒸馏）
│       ├── prepare_judge_slices.py
│       ├── run_prepare.py           # 包装脚本（自动设镜像）
│       ├── run_train.py
│       └── detect/
│           ├── detect.py                # 通用深度推理（图片/视频/目录）
│           ├── detect_car.py            # 车辆检测：深度+YOLO+MOG2 融合判定
│           ├── detect_sel_videos.py     # 批量视频挑选（人工复核判错案例）
│           ├── judge_integration.py     # MLP 判定模型集成模块
│           └── input/                   # 待检测素材目录
│
└── yolov26/                     # 子项目二：YOLO 车辆检测全流水线
    └── yolov26/
        ├── train.py                 # YOLO 训练（AdamW + 余弦退火 + Mosaic）
        ├── detect.py                # 推理（图片/视频/摄像头 + BoT-SORT 跟踪）
        ├── split_dataset.py         # 数据集划分
        ├── memory_manager.py        # OOM 自愈训练管理器
        ├── clean/
        │   └── clean_dataset.py         # 数据清洗（带清洗报告）
        ├── Video label/
        │   └── video_slicer.py          # 视频抽帧打标
        ├── vehicle_stitcher/        # 车辆侧面图像拼接（特征匹配+多帧融合）
        └── ultralytics-main/        # Ultralytics 框架源码
```

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.5.1+cu121、torchvision 0.20.1
- transformers、opencv-python、numpy、pillow、safetensors、pyyaml、ultralytics
- 建议 GPU：8GB+ 显存

```bash
conda create -n yolo_depth python=3.10
conda activate yolo_depth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers opencv-python numpy pillow safetensors pyyaml ultralytics
```

### 深度估计流水线（depthAnyV2）

```bash
cd depthAnyV2/depthAnyV2

# 国内镜像，避免模型下载失败
set HF_ENDPOINT=https://hf-mirror.com

# 1. 放图片到 datasets/my_dataset/images/
# 2. 数据准备：教师模型生成米数伪标签（逐张 .npy，增量跳过）
python prepare_data.py

# 3. 训练（Stage/epochs/lr 等参数都在 config.yaml 里）
python train.py

# 4. 评估（与预训练模型对比）
python evaluate.py

# 5. 通用深度推理：放图片/视频到 detect/input/
python detect/detect.py

# 6. 车辆检测（有车/无车判定）
python detect/detect_car.py
```

### YOLO 检测流水线（yolov26）

```bash
cd yolov26/yolov26

# 视频抽帧 → 标注 → 清洗 → 划分
python "Video label/video_slicer.py"
python clean/clean_dataset.py
python split_dataset.py

# 训练与推理
python train.py
python detect.py
```

## 判定逻辑（detect_car.py）

多规则加权融合，规则权重总和 = 1，阈值 0.47 判有车：

| 规则 | 内容 | 权重 |
|------|------|------|
| R1 | 深度骤变检测 | 0.10 |
| R2 | 绝对深度分布（主规则） | 0.25 |
| R3 | 基线对比下降（主规则） | 0.25 |
| R4 | 车辆区域统计特征 | 0.10 |
| R5 | YOLO 视觉检测 | 0.30 |

- **检测线监测点**：默认 15 点双列分布（x=40 共 7 点 + x=55 共 8 点），≥2 点深度骤减（绝对 ≥0.4m 且相对 ≥8%）连续 3 帧确认有车
- **冷却期**：车辆离开后一段时间内不确认新车，防止两车间隔噪声误报
- **夜间自适应**：平均灰度 < 80 判夜间，自动放宽确认帧、延长冷却、EMA 平滑
- **过曝屏蔽**：车灯直射导致像素饱和（>250）的监测点直接丢弃，不参与判定

## 评估指标

| 指标 | 含义 | 优秀 | 良好 | 一般 |
|------|------|------|------|------|
| AbsRel | 绝对相对误差（越低越好） | < 0.10 | 0.10~0.20 | 0.20~0.30 |
| RMSE | 均方根误差，单位米（越低越好） | < 1.0m | 1.0~3.0m | 3.0~5.0m |
| δ<1.25 | 预测/标签比值 < 1.25 的像素占比（越高越好） | > 90% | 70%~90% | 50%~70% |

> **注意**：本项目采用自蒸馏方案，评估"标签"为教师模型生成的伪标签而非 LiDAR 真值，上述指标衡量的是学生模型与教师模型的接近程度；绝对精度需真实深度数据验证。

## 技术栈

- **深度估计**：Depth Anything V2（Metric-Outdoor 系列，Virtual KITTI 2 室外车载数据训练）
- **目标检测**：Ultralytics YOLO + BoT-SORT 跟踪
- **图像拼接**：SIFT/ORB 特征匹配 + Lowe's ratio test + 多帧线性插值融合
- **运动检测**：OpenCV MOG2 背景减除（亮度动态权重）
- **训练技术**：自蒸馏、知识蒸馏（规则 → MLP）、LoRA、时序一致性损失、夜间模拟增强
- **工程能力**：统一 YAML 配置、断点续训、OOM 自愈、增量处理、Git LFS 管理模型权重

## 常见问题

1. **怎么改参数？** 所有脚本默认读 `config.yaml`，改对应段落即可；命令行参数（如 `--epochs 50`）会临时覆盖。
2. **训练中断怎么续？** 把 `train.resume` 改为 `checkpoints/my_model_stage1/latest_model.pt` 再运行。
3. **显存不足？** 降低 `train.batch_size`（8GB 显存建议 1~2），或降低 `process_res`（须为 14 的倍数）。
4. **HF 模型下载失败？** 确保设置 `HF_ENDPOINT=https://hf-mirror.com`（包装脚本已内置）。
5. **夜间误报/闪烁？** 依次调整 `night` 段参数、`saturation_threshold` 饱和屏蔽、`clahe` 去眩光。

## 说明

- 数据集与训练视频未包含在本仓库中（体积过大），仓库内为代码、配置与模型权重（Git LFS）
- 详细使用说明见 [depthAnyV2/README.md](depthAnyV2/depthAnyV2/README.md)
