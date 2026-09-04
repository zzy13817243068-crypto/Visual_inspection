# ============================================================
# 车辆侧面图像拼接系统
# Vehicle Side Image Stitching System
# ============================================================

## 项目概述

基于 OpenCV 和 YOLO 深度学习模型，从固定机位拍摄的视频中自动拼接车辆侧面完整图像。系统通过逐帧检测车辆、提取特征点、计算位移、平滑融合等步骤，生成一张比例正常、无拉伸变形的完整车辆侧面图。

## 目录结构

```
vehicle_stitcher/
├── vehicle_stitcher.py    # 主程序（包含所有功能模块）
├── requirements.txt       # 依赖清单
├── models/                # 存放车辆检测模型
├── output/                # 存放拼接结果
└── README.txt             # 本说明文档
```

## 环境要求

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- Ultralytics YOLO 8.4+

## 安装步骤

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python vehicle_stitcher.py \
    --video /path/to/video.mp4 \
    --model /path/to/best.pt
```

### 完整参数

```bash
python vehicle_stitcher.py \
    --video /path/to/video.mp4 \
    --model /path/to/best.pt \
    --output ./result/ \
    --confidence 0.5 \
    --feature orb \
    --min-displacement 5.0 \
    --format png \
    --save-intermediate
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | (必填) | 输入视频路径 |
| `--model` | (必填) | 车辆检测模型路径 |
| `--output` | `./output` | 输出目录 |
| `--confidence` | `0.5` | 检测置信度阈值 (0-1) |
| `--iou` | `0.45` | NMS IoU阈值 (0-1) |
| `--feature` | `orb` | 特征点方法: orb/sift/surf |
| `--max-features` | `2000` | 最大特征点数量 |
| `--min-match` | `15` | 最少有效匹配点数 |
| `--min-displacement` | `5.0` | 最小位移阈值(像素) |
| `--frame-stride` | `1` | 帧间隔（跳帧处理） |
| `--format` | `png` | 输出格式: png/jpg |
| `--scale` | `1.0` | 输出缩放因子 |
| `--save-intermediate` | False | 保存中间处理帧 |

## 核心算法流程

```
1. 视频读取 ──→ 2. 车辆检测 ──→ 3. 特征点提取
                                            ↓
6. 结果保存 ←── 5. 融合拼接 ←── 4. 位移计算
```

### 详细步骤

1. **视频读取**: 逐帧读取视频，支持 mp4/avi/mov 等格式
2. **车辆检测**: 使用 YOLO 模型检测每帧中的车辆位置
3. **特征提取**: 在车辆区域提取 ORB/SIFT/SURF 特征点
4. **位移计算**: 匹配相邻帧特征点，计算车辆位移向量
5. **融合拼接**: 按位移量将多帧车辆区域平滑融合
6. **结果输出**: 裁剪有效区域，保存完整车辆侧面图

## 参数调优建议

### 检测置信度 (confidence)
- **0.3-0.4**: 召回率高，可能有误检
- **0.5-0.6**: 平衡（推荐）
- **0.7-0.9**: 精确率高，可能漏检

### 最小位移阈值 (min-displacement)
- **1-3px**: 适用于低速场景
- **5-10px**: 适用于常规速度（推荐）
- **10-30px**: 适用于高速场景

### 特征点方法 (feature)
- **ORB**: 速度快，适合实时处理
- **SIFT**: 精度高，适合复杂场景
- **SURF**: 速度与精度平衡

### 帧间隔 (frame-stride)
- **1**: 每帧处理，精度高
- **2-3**: 跳帧处理，速度快
- **4+**: 快速处理，可能漏帧

## 输出结果

```
output/
├── vehicle_side_20260807_120000.png   # 拼接结果
├── process_log.json                    # 处理日志
└── intermediate/                       # 中间帧（可选）
    ├── intermediate_frame_0100.png
    └── ...
```

## 常见问题

### Q1: 检测不到车辆怎么办？
降低 `--confidence` 阈值到 0.3-0.4

### Q2: 拼接结果有重复或断裂？
- 检查 `--min-displacement` 阈值
- 尝试更换特征点方法 `--feature sift`
- 增加 `--max-features` 到 3000

### Q3: 内存不足？
- 增大 `--frame-stride` 跳帧处理
- 设置 `--scale 0.5` 降低输出分辨率

### Q4: 车辆检测不准确？
- 使用更专业的车辆检测模型
- 提高 `--confidence` 阈值
- 调整 `--iou` 阈值

### Q5: 拼接结果有黑边？
脚本自动裁剪有效区域，如仍有黑边可手动调整 `padding` 参数

## 技术细节

### 特征匹配算法
- 使用 Lowe's Ratio Test 过滤低质量匹配
- 计算位移中位数并过滤异常值
- 评估匹配置信度

### 融合算法
- 重叠区域线性插值融合
- 非重叠区域直接复制
- 画布动态扩展

### 防漂移机制
- 实时验证特征匹配质量
- 位移异常检测与跳过
- 多条件复合判断

## 许可协议

本项目仅供学习和研究使用。