# YOLOv26 道路车辆检测系统

## 项目简介

基于 Ultralytics YOLOv26 框架的道路车辆目标检测系统，覆盖**视频抽帧 → 数据清洗 → 数据集划分 → 模型训练 → 验证评估 → 推理部署**完整工程流水线。在约 1.5 万张自采集道路监控图片（4256 张有效标注，car 单类别）上训练车辆检测模型，为固定机位监控场景的车辆存在性判定与车辆侧面图像重建提供检测能力。

## 核心技术亮点

| # | 技术点 | 说明 |
|---|--------|------|
| 1 | **全流程数据工程** | 视频批量抽帧打标（video_slicer）→ 无效样本自动清洗（clean_dataset，生成清洗报告）→ 训练/验证集划分（split_dataset），构建可复现的数据流水线 |
| 2 | **训练策略精调** | AdamW + 余弦退火学习率（lr0=9e-4）+ 3 epoch warmup + label smoothing + 早停（patience=10），单类别任务关闭多类别干扰 |
| 3 | **数据增强策略** | Mosaic 0.8 + 训练末期 10 epoch 关闭 Mosaic（close_mosaic）+ HSV 色彩扰动 + 随机旋转/平移/缩放/剪切/透视，提升复杂路况泛化能力 |
| 4 | **混合精度训练** | AMP 自动混合精度 + 确定性训练（deterministic + 固定 seed），兼顾训练速度与结果可复现 |
| 5 | **OOM 自愈训练管理器** | memory_manager 实时监测 CUDA OOM，按优先级渐进式调低 batch/参数，自动保存并恢复训练状态；内置文件版本控制（MD5 校验 + 备份历史） |
| 6 | **工程细节修复** | 路径含空格导致 TensorBoard event 写入失败的符号链接方案 + event 文件完整性验证 |
| 7 | **车辆侧面图像拼接** | vehicle_stitcher：YOLO 检测定位车辆区域 + SIFT/ORB 特征点匹配（Lowe's ratio test）计算帧间位移，多帧线性插值融合 + 防漂移机制，从视频重建完整车辆侧面图 |
| 8 | **多源推理支持** | 图片目录 / 单图 / 视频 / 摄像头实时检测，BoT-SORT 目标跟踪，结果可视化输出 |

## 系统架构

```
道路监控视频
   │
   ├─► 视频抽帧打标 ──► 原始图片 + YOLO 标签
   │        │
   │        ▼
   ├─► 数据清洗（clean_dataset）──► 无效样本剔除 + 清洗报告
   │        │
   │        ▼
   ├─► 数据集划分（split_dataset）──► train / val / test
   │        │
   │        ▼
   ├─► YOLOv26 训练（train.py + args.yaml）
   │     ├─ AMP 混合精度 / 余弦退火 / Mosaic 增强 / 早停
   │     ├─ memory_manager：OOM 监测 + 参数降级 + 状态恢复
   │     └─ TensorBoard 训练曲线
   │        │
   │        ▼
   ├─► 验证评估（mAP@0.5 / mAP@0.5:0.95 / 混淆矩阵）
   │        │
   │        ▼
   └─► 推理部署（detect.py）
         ├─ 图片 / 视频 / 摄像头实时检测（BoT-SORT 跟踪）
         └─ vehicle_stitcher：车辆侧面图像拼接重建
```

## 目录结构

```
yolov26/
├── dataset/                  # 数据集目录（images_raw/labels_raw → 划分后 images/labels）
├── output/                   # 训练结果输出（weights/best.pt、results.png、混淆矩阵）
├── runs/                     # 检测结果输出
├── Video label/              # 视频抽帧打标（video_slicer.py）
├── clean/                    # 数据集清洗（clean_dataset.py + 清洗报告）
├── vehicle_stitcher/         # 车辆侧面图像拼接系统
├── ultralytics-main/         # Ultralytics YOLO 框架源码
├── args.yaml                 # 训练超参数配置（含逐项中文注释）
├── data.yaml                 # 数据集配置（路径 + 类别）
├── train.py                  # 训练脚本（环境检查 / TensorBoard 验证 / 自动验证）
├── detect.py                 # 推理脚本（图片/视频/摄像头/验证模式）
├── split_dataset.py          # 数据集划分脚本
├── clean_dataset.py          # 数据清洗入口
├── memory_manager.py         # CUDA OOM 自愈训练管理器
├── init_directories.py       # 目录初始化脚本
├── yolo26n/s/m.pt            # 预训练权重（n/s/m 三档）
└── requirements.txt          # 依赖清单
```

## 快速开始

详细操作步骤见 [readme_windows.txt](readme_windows.txt)（Windows）与 [readme_linux.txt](readme_linux.txt)（Linux 服务器）。

```bash
# 1. 环境准备
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 数据准备：图片放 dataset/images_raw/，标签放 dataset/labels_raw/
python split_dataset.py        # 划分训练/验证集
python clean/clean_dataset.py  # 数据清洗

# 3. 训练（args.yaml 可调 epochs/batch/model 等参数）
python train.py

# 显存不足时使用 OOM 自愈管理器
python memory_manager.py

# 4. 验证与推理
python detect.py --mode validate --model output/exp-det/weights/best.pt
python detect.py --source dataset/images/val/        # 图片目录
python detect.py --source test.mp4                   # 视频
python detect.py --source 0                          # 摄像头实时

# 5. 车辆侧面图像拼接
python vehicle_stitcher/vehicle_stitcher.py --video xxx.mp4 --model best.pt
```

## 关键训练配置（args.yaml）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| epochs / patience | 100 / 10 | 训练轮数与早停耐心值 |
| batch / imgsz | 8 / 640 | 批大小与输入分辨率 |
| optimizer / lr0 | AdamW / 9e-4 | 优化器与初始学习率 |
| cos_lr / warmup_epochs | true / 3.0 | 余弦退火 + 热身 |
| mosaic / close_mosaic | 0.8 / 10 | Mosaic 增强 + 末期关闭 |
| label_smoothing | 0.03 | 标签平滑防过拟合 |
| amp / deterministic | true / true | 混合精度 + 可复现 |
| box / cls / dfl | 7.5 / 1.2 / 1.5 | 损失权重 |
| tracker | botsort.yaml | 目标跟踪器 |

## 输出结果

训练完成后保存在 `output/exp-det/`：
- `weights/best.pt` / `last.pt`：最佳与最终权重
- `results.png`：训练曲线（loss / mAP）
- `confusion_matrix.png`：混淆矩阵
- TensorBoard：`tensorboard --logdir=output/exp-det --port=6006`
