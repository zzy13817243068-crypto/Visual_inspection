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

## 快速开始（完整流程速查）

所有脚本的默认参数都来自 `config.yaml`，**改配置即可生效，不需要传命令行参数**；命令行参数（如 `--epochs 50`）会覆盖配置文件。

```bash
conda activate yolo_depth
set HF_ENDPOINT=https://hf-mirror.com

# 1. 放图片到 datasets/my_dataset/images/（如需改路径，改 config.yaml 的 paths 段）
# 2. 数据准备：教师模型生成米数伪标签
python prepare_data.py          # 或双击/运行 run_prepare.py

# 3. 训练（config 中 train.stage / train.resume 决定阶段与续训）
python train.py                 # 或 run_train.py

# 4. 评估
python evaluate.py

# 5. 验证：放图片/视频到 detect/input/
# 6. 通用深度推理（单图/视频/目录）
python detect/detect.py

# 7. 车辆检测（有车/无车判定，深度+YOLO+MOG2 融合）
python detect/detect_car.py
```

---

## 一、环境准备

依赖（环境 `yolo_depth` 已全部安装，无需额外操作）：
- torch 2.5.1+cu121、torchvision 0.20.1、transformers 5.15.0
- opencv-python、numpy、pillow、safetensors、pyyaml

```bash
conda activate yolo_depth
set HF_ENDPOINT=https://hf-mirror.com      # 国内镜像，避免模型下载失败
set CUDA_VISIBLE_DEVICES=0                 # 单卡时指定
```

## 二、统一配置 config.yaml

项目所有脚本共用一份配置，修改后立即生效。共 8 个段落：

| 段落 | 作用 | 对应脚本 |
|------|------|----------|
| `paths` | 输入/输出目录 | 全部脚本 |
| `model` | 教师/学生/基准模型、夜间增强开关 | prepare_data / train / evaluate |
| `train` | 训练阶段、轮数、学习率、早停、LoRA 等 | train.py |
| `eval` | 评估 checkpoint、是否对比预训练 | evaluate.py |
| `detect` | 深度推理：模型、预览模式、像素点监测 | detect/detect.py |
| `car_detect` | 车辆检测：YOLO、判定阈值、夜间自适应、MOG2、MLP 等 | detect/detect_car.py |
| `classifier` | 有车/无车图像分类器训练 | detect/training/train_classifier.py |
| `judge` | 判定 MLP 训练超参 | train_judge.py |

配置优先级：**命令行参数 > config.yaml > 内置默认值**。

---

## 三、数据准备 prepare_data.py

用 DA2-Metric-Outdoor-Large 教师模型生成真实米数伪标签（0~80m），**逐张保存不爆内存**，支持几万张；已生成深度标签的图片自动跳过（增量处理）。

```bash
python prepare_data.py
```

输出结构（`datasets/my_dataset/processed/`）：

```
processed/
  ├── image_files.npy          (图片路径列表)
  └── depths/
      └── frame_001_depth.npy  (逐张米数深度图, float32, 单位米)
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--image-dir` / `--output-dir` | 默认读 `paths.image_dir` / `paths.processed_dir` |
| `--teacher-model` | 默认读 `model.teacher` |
| `--no-tta` | 禁用 TTA 多尺度增强（默认开启） |
| `--night-enhance` | 夜间去眩光：低亮度图先 CLAHE 压高光再推理（默认读 `model.night_enhance`） |
| `--force` | 强制重新生成所有伪标签 |
| `--force-night` | 只强制重跑夜间旧图（白天旧图仍跳过），配合 `--night-enhance` 使用 |

## 四、模型训练 train.py

自蒸馏微调：学生模型（Metric-Outdoor-Base）拟合教师模型生成的米数伪标签。支持分阶段微调、断点续训、LoRA、时序一致性损失、夜间模拟增强。

```bash
# 一切按 config.yaml 的 train 段运行（stage/resume/epochs 等都在配置里）
python train.py

# 或命令行覆盖配置
python train.py --stage 1 --epochs 100
python train.py --stage 2 --resume checkpoints/my_model/best_model.pt
```

### 分阶段训练

| 阶段 | 内容 | 说明 |
|------|------|------|
| Stage 1 | Head 微调 | 冻结 backbone，只训 DPT head，快速适应数据 |
| Stage 2 | 全量微调 | 解冻 backbone 联合训练；backbone LR = `base_lr`（默认 5e-6，避免破坏预训练特征），head 自动 ×10 |

Stage 2 通过 `resume: checkpoints/my_model/best_model.pt` 承接 Stage 1：**只继承模型权重**，优化器与训练计数清零、从头开始计数 epoch。

### 断点续训

- 中断后续跑：把 `train.resume` 改为 `checkpoints/my_model/latest_model.pt`，从断点完整恢复（含优化器状态）。
- `train.resume: null` = 从头开始。

### LoRA 微调（可选）

`train.lora: true` 时冻结 ViT，只在 DPT head 的 1x1 卷积上加低秩适配（秩 `lora_rank`，默认 8），参数量小、训练快、防过拟合；保存 checkpoint 时自动合并 LoRA 权重为标准键名，加载无额外要求。

### 时序一致性损失

约束相邻帧深度平滑：`temporal_weight`（0=禁用，0.1~0.3 推荐）控制权重；`temporal_every` 为子采样间隔（每 N 个 batch 做一次额外前向，权重 ×N 自动补偿，2 约省 40% 训练时间且期望梯度不变）。

### 夜间模拟增强

`night_aug_prob`（触发概率）+ `bloom_kernel`（眩光光晕模糊核）：随机给低照度样本加 bloom，提升对夜间车灯场景的鲁棒性。验证只算纯 SILog loss，不算时序 loss。

### 主要参数（train 段，CLI 同名 `--xxx`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `stage` | 2 | 1=Head 微调, 2=全量微调 |
| `epochs` | 25 | 训练轮数 |
| `batch_size` | 4 | 批大小（8GB 显存建议 1-2，16GB 可 4） |
| `process_res` | 448 | 处理分辨率（须为 14 的倍数；518 为标准值，448 提速约 25%） |
| `base_lr` | 5e-6 | backbone 学习率（head ×10） |
| `weight_decay` | 0.01 | AdamW 权重衰减 |
| `loss_lambd` | 0.5 | SILog 方差惩罚系数 λ |
| `temporal_weight` / `temporal_every` | 0.2 / 2 | 时序一致性损失权重 / 子采样间隔 |
| `night_aug_prob` / `bloom_kernel` | 0.3 / 11 | 夜间模拟增强概率 / 光晕核大小 |
| `val_every` | 2 | 每 N 轮验证一次（N>1 省时，早停按验证次数计） |
| `patience` / `min_delta` | 7 / 1e-4 | 早停耐心 / 最小改善幅度 |
| `seed` / `num_workers` / `prefetch_factor` | 42 / 6 / 4 | 随机种子 / DataLoader 进程数 / 预取数 |
| `lora` / `lora_rank` | false / 8 | LoRA 开关 / 秩 |
| `resume` | best_model.pt | 续训 checkpoint（null=从头开始） |

输出：`checkpoints/my_model/best_model.pt` 与 `latest_model.pt`（含 model/optimizer 状态与训练标量）。

## 五、模型评估 evaluate.py

```bash
python evaluate.py                 # 默认读 eval 段：checkpoint + compare=true
python evaluate.py --checkpoint checkpoints/my_model/best_model.pt --compare
python evaluate.py --limit 0       # 评估全部图片（默认抽样 100 张避免太慢）
```

输出米数统计（最小/最大/均值/中位数）+ 以下定量指标：

| 指标 | 含义 | 方向 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|------|------|-----|
| **AbsRel** | 绝对相对误差：\|预测-标签\|/标签 的均值 | 越低越好 | < 0.10 | 0.10~0.20 | 0.20~0.30 | > 0.30 |
| **RMSE** | 均方根误差（单位：米） | 越低越好 | < 1.0m | 1.0~3.0m | 3.0~5.0m | > 5.0m |
| **δ<1.25** | 预测值与标签比值 < 1.25 的像素占比 | 越高越好 | > 90% | 70%~90% | 50%~70% | < 50% |

**指标怎么读**：微调后 AbsRel、RMSE 应**下降**，δ<1.25 应**上升**，说明微调有效；否则需调整训练参数。

> **注意**：本项目用自蒸馏方案，"标签"是教师模型（DA2-Large）生成的伪标签，非真实深度。上述指标衡量的是"学生模型与教师模型的接近程度"，并非绝对精度；要评估绝对精度需真实深度相机/LiDAR 数据。

## 六、深度推理 detect.py

支持图片 / 视频 / 目录输入，输出原图+深度对比图、彩色深度图、米数矩阵；视频输出左右对比视频，并可叠加**像素点深度监测**与**检测线车辆状态标注**。

```bash
python detect/detect.py                                           # 默认读 paths.detect_input（detect/input）
python detect/detect.py --source detect/input/test.jpg
python detect/detect.py --source detect/input/test.mp4 --model-depth checkpoints/my_model/best_model.pt
python detect/detect.py --source detect/input/ --output detect/output
```

> checkpoint 不存在时自动回退到预训练 Metric-Outdoor-Base 模型，方便未训练前直接试用。

单图输出：

| 文件 | 内容 |
|------|------|
| `xxx_comparison.png` | 原图 + 深度图左右对比（标注米数范围） |
| `xxx_depth.png` | 彩色深度图 |
| `xxx_depth.npy` | 深度矩阵（float32，单位米，可直接 `np.load` 读取） |

### 预览模式

`detect.preview_enabled: true` 时，随机选 `preview_count` 个文件用 OpenCV imshow 窗口预览**不落盘**（省内存）；设为 false 才保存结果文件。

### 像素点深度监测

在画面指定像素点上实时显示米数深度：

| 配置 | 说明 |
|------|------|
| `pixel_depth_enabled` | 总开关 |
| `pixel_depth_points` | 自定义点，格式 `"x1,y1;x2,y2;..."`，留空自动分布 |
| `pixel_depth_count` | 自动监测点数量 |

自动分布支持三种布局：15 = 非对称双列（x=40 共 7 点 + x=55 共 8 点，默认）、14 = 对称双列（7+7 聚合模式）、7 = x=40 单列。

其他常用：`save_depth`（视频每帧深度存 .npz，1=存但慢）、`frame_step`（隔帧采样）、`max_frames`（最大帧数）。

## 七、车辆检测 detect_car.py

基于**检测线监测点深度变化 + YOLO 视觉 + MOG2 运动**的多规则融合"有车/无车"判定，支持夜间场景。

```bash
python detect/detect_car.py                        # 默认读 car_detect.input_dir（detect/input，递归子目录）
python detect/detect_car.py --source detect/input/xxx.mp4
python detect/detect_car.py --preview-count 0      # 处理全部视频（默认随机抽 5 个）
```

输出：`detect/car_output/run_时间戳/`，含逐帧标注视频（左原图 + 右深度图 + 监测点米数 + 状态条 + MOG2 运动带）、无车片段截取帧、测试报告；控制台每视频仅 3 行总结，详细日志在 `run_log.txt`。

### 判定逻辑

1. **深度监测点**（主规则）：默认 15 点双列分布在检测线（x=40 共 7 点、x=55 共 8 点）。
   - 有车：≥`min_car_points`（2）个可靠点相对基线骤减（绝对 ≥`distance_drop_threshold`(0.4m) 且相对 ≥`relative_drop_threshold`(8%)），连续 `car_confirm_frames`（3）帧确认。
   - 无车：≥`min_stable_points`（3）个路面可靠点偏差 <`normal_fluctuation`(1.5m)，连续 `stable_confirm_frames`（5）帧稳定。
   - 只统计基线 ≤`max_detect_depth`(30m) 的点，过滤天空/远景。
2. **YOLO 视觉（R5）**：检测到车辆框且已过检测线（过线容差 `cross_tolerance_px`）时提高置信权重；本帧未检测到框时，深度判定 + MOG2 为主。YOLO 无跨帧延续，只影响当前帧。
3. **MOG2 运动检测**：检测线窄带（`strip_width`）内前景像素辅助确认；深度模型返回 unknown 时需 MOG2 确认 + 深度变化验证（绝对 ≥0.1m 且相对 ≥1%）才判有车；按环境亮度动态调整灵敏度（夜间更敏感）。
4. **权重融合**：规则 R1~R5 权重总和 =1，`car_confidence_threshold`（0.47）为有车阈值；R2 绝对深度 0.25 + R3 基线对比 0.25 为主规则，R5 YOLO 0.30。
5. **冷却期**：车辆离开后 40 帧（约 1.3s@30fps）内不允许新确认，防止两车间隔的深度噪声误报。
6. **MLP 判定模型（实验性，见下）**。

### 夜间自适应与去眩光

| 配置 | 作用 |
|------|------|
| `night.enabled` / `bright_threshold` | 夜间自适应开关 / 亮度阈值（平均灰度 <80 判夜间） |
| `night.smooth_alpha` | 夜间道路深度 EMA 平滑系数（越小越平滑） |
| `night.confirm_extra` / `cooldown_extra` | 夜间额外确认帧数 / 额外冷却帧数（放宽门槛防闪断） |
| `saturation_threshold` | 饱和屏蔽：监测点过曝（车灯直射）时丢弃该点，0=关闭 |
| `clahe.enabled` 等 | CLAHE 去眩光预处理（Lab 空间 L 通道对比度均衡，当前默认关闭） |
| `baseline_dir` | 预设背景基线目录（视频开头就有车污染基线时用），留空则用前 `baseline_warmup` 帧预热 |

### MLP 判定模型（实验性）

`car_detect.judge_mlp` 将 train_judge.py 训练的 MLP 接入判定流程，仅在高置信度时辅助纠正，不替代 v8 规则；开关关闭/加载失败/推理异常时自动回退。

> **当前状态**：`judge_mlp.enabled: false`。实测现有 judge_mlp.pt 会把有车帧全部纠正为无车，模型重训到位前保持关闭；配置透传能力已保留，训好后再开。

## 八、视频挑选 detect_sel_videos.py

批量抽取视频生成完整标注视频（与 detect_car.py 同一判定逻辑，原图叠加 YOLO 框），用于**人工挑选 v8 判错案例**做针对性优化。

```bash
python detect/detect_sel_videos.py                 # 默认从 ≥2s 的视频随机抽 50 个
python detect/detect_sel_videos.py --count 100 --min-duration 3
```

输出：`detect/sel_output/run_时间戳/{视频名}_depth_video.mp4` + `run_log.txt`。挑选流程：播放器查看 → 把判错（误报/漏报/闪烁）的视频拷到单独文件夹 → 针对性优化规则。

## 九、判定模型训练 train_judge.py

用样本视频训练轻量 MLP（每帧约 25 维特征：深度统计/YOLO/MOG2/亮度/时序窗口/v8 内部信号 → 3 类 **car / no_car / unknown**），本阶段标签来自 v8 规则输出（**知识蒸馏**），建立"特征提取 → 训练 → 验证 → 保存"流水线；后续可替换人工真值帧级标签以真正超越 v8。

```bash
python train_judge.py                       # 增量提取：旧视频复用特征缓存，只处理新增视频
python train_judge.py --use-cache           # 纯调参：跳过特征提取，直接用已有缓存
python train_judge.py --force-extract       # 强制全量重新提取
```

- 样本目录自动推断标签：子目录名含"误识"→no_car，含"闪烁"→car，其余用 v8 伪标签；可用 `--label-overrides` 手动指定。
- 按**视频分组** 8:2 划分训练/验证（`val_video_frac`），避免帧级泄漏；类权重加权 CrossEntropy 处理类别不平衡；早停防过拟合。
- 超参（epochs/batch/window/lr/weight_decay/dropout/patience/seed/val_video_frac）默认读 `judge` 段，CLI 可覆盖。

输出：`checkpoints/judge_model/judge_mlp.pt` + `judge_training_output/`（`feature_cache.npz` 特征缓存、`metrics.json`、`training_curve.png`、`classification_report.txt`、`confusion_matrix.png`）。

## 十、分类器训练 detect/training/train_classifier.py

有车/无车**图像二分类器**训练（与深度判定互补的视觉分类器）。

数据集目录结构（`classifier.data_dir`，默认 `detect/training/dataset/`）：

```
dataset/
  ├── car/           # 有车类别图片
  └── no_car/        # 无车类别图片
```

```bash
python detect/training/train_classifier.py
python detect/training/train_classifier.py --epochs 50 --img-size 224
python detect/training/train_classifier.py --resume checkpoints/classifier/best_model.pt
```

- 模型可选：`efficientnet_b0`（默认）/ `resnet18` / `resnet50` / `mobilenet_v3_small` / `vit_b_16`。
- 支持预训练权重、冻结骨干、数据增强（翻转/旋转/亮度/对比度）、早停、cosine/step/reduce 调度器；`--no-augmentation` / `--no-early-stopping` 可关闭。
- 输出：`checkpoints/classifier/` 下 `best_model.pt`、`latest_model.pt`、`deployed_model.pt`、`classifier_config.json`、`training_curves.png`、`evaluation_report.txt`。

## 十一、可用模型

### 度量深度模型（输出真实米数，本版使用）

| 模型 | HuggingFace ID | 训练数据 | max_depth | 显存 |
|------|----------------|---------|-----------|------|
| Metric-Outdoor-Small | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | Virtual KITTI 2（室外车载） | 80m | 4GB+ |
| Metric-Outdoor-Base | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf` | Virtual KITTI 2 | 80m | 6GB+ |
| Metric-Outdoor-Large | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | Virtual KITTI 2 | 80m | 10GB+ |
| Metric-Indoor-Small | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Hypersim（室内） | 20m | 4GB+ |
| Metric-Indoor-Base | `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf` | Hypersim | 20m | 6GB+ |
| Metric-Indoor-Large | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | Hypersim | 20m | 10GB+ |

> **车辆监控选 Outdoor 系列**（Virtual KITTI 2 是自动驾驶场景训练，匹配度高）。教师=Large（精度最高），学生=Base（平衡）。

### 相对深度模型（旧版，仅比远近，无米数）

| 模型 | HuggingFace ID |
|------|----------------|
| DA2-Small | `depth-anything/Depth-Anything-V2-Small-hf` |
| DA2-Base | `depth-anything/Depth-Anything-V2-Base-hf` |
| DA2-Large | `depth-anything/Depth-Anything-V2-Large-hf` |

## 十二、文件说明

| 文件 | 作用 |
|------|------|
| `config.yaml` | **统一配置**：8 个段落控制全部脚本，改值即生效 |
| `prepare_data.py` | 数据准备：DA2-Metric-Large 教师生成米数伪标签（逐张保存，增量跳过） |
| `train.py` | 深度训练：自蒸馏微调，Stage 1/2、LoRA、时序损失、夜间增强、早停 |
| `evaluate.py` | 评估：与预训练模型对比，米数统计 + AbsRel/RMSE/δ 指标 |
| `train_judge.py` | 判定 MLP 训练：v8 伪标签知识蒸馏（car/no_car/unknown 三分类） |
| `run_train.py` / `run_prepare.py` | 简易包装脚本（自动设镜像、异常兜底），直接运行 train.py / prepare_data.py |
| `detect/detect.py` | 通用深度推理：图片/视频/目录，预览模式 + 像素点监测 + 车辆状态标注 |
| `detect/detect_car.py` | 车辆检测：深度+YOLO+MOG2 融合判定，夜间自适应，冷却期防误报 |
| `detect/detect_sel_videos.py` | 批量视频挑选：出标注视频供人工挑 v8 判错案例 |
| `detect/judge_integration.py` | MLP 判定模型集成模块（无独立入口，开关控制） |
| `detect/training/train_classifier.py` | 有车/无车图像分类器训练与部署导出 |

| 目录 | 内容 |
|------|------|
| `datasets/my_dataset/images/` | 放入训练图片 |
| `datasets/my_dataset/processed/` | 数据准备输出（image_files.npy + depths/） |
| `checkpoints/my_model/` | 深度模型输出（best_model.pt / latest_model.pt） |
| `checkpoints/yolo/` | YOLO 权重目录（`car_detect.yolo.model_path` 指向此处） |
| `checkpoints/judge_model/` | 判定 MLP 权重（train_judge.py 输出） |
| `checkpoints/classifier/` | 分类器输出（train_classifier.py 输出） |
| `eval_results/` | 评估输出目录 |
| `detect/input/` | 放入待验证/待检测图片或视频 |
| `detect/output/` | 通用推理输出目录 |
| `detect/car_output/` | 车辆检测输出（按运行时间戳分子目录） |
| `detect/sel_output/` | 视频挑选输出 |
| `judge_training_output/` | 判定 MLP 训练输出（特征缓存/曲线/报告） |
| `*_backup_*/`、`REFACTOR_CHANGELOG.md`、`TRAIN_REFACTOR_CHANGELOG.md` | 历史版本备份与路线优化记录（不影响日常使用） |

## 十三、常见问题

1. **怎么改参数？** 所有脚本默认读 `config.yaml`，改对应段落的值即可；临时覆盖用命令行参数（如 `--epochs 50`）。
2. **Windows 上训练大数据集内存/显存异常？** 代码已内置保护：Windows（WDDM 驱动）下 `pin_memory` 自动禁用，数据集超过 4GB 时必须保持该行为；如遇显存不足，降低 `train.batch_size`（8GB 显存建议 1-2）。
3. **训练中断怎么续？** 把 `train.resume` 改为 `checkpoints/my_model/latest_model.pt` 再运行 `python train.py`（完整恢复含优化器）；Stage 2 用 `best_model.pt` 只继承权重。
4. **detect 提示 checkpoint 不存在？** 会自动回退到预训练 Model，属正常行为；要用微调模型请确认 `detect.model_depth` 路径正确。
5. **为什么 `judge_mlp.enabled` 是 false？** 当前 judge_mlp.pt 会把有车帧误判为无车，重训到位前保持关闭；训练好后再开启。
6. **夜间检测闪烁/误报？** 依次可调：`night` 段（平滑系数/确认帧/冷却帧）、`saturation_threshold` 饱和屏蔽、`clahe.enabled` 去眩光；训练侧可提高 `night_aug_prob` 增强夜间样本。
7. **HF 模型下载慢/失败？** 确保设置了 `HF_ENDPOINT=https://hf-mirror.com`（scripts 内已 setdefault，包装脚本 run_*.py 已内置）。
8. **时序损失拖慢训练？** `temporal_every: 2` 已默认开启子采样（省约 40% 时间、期望梯度不变）；`val_every: 2` 每 2 轮验证一次，也可继续调大省时。