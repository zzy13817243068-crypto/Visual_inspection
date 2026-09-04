"""
模型验证与推理脚本（Depth Anything V2 - 度量深度，输出真实米数）
支持图片、视频、目录输入

用法：
  python detect.py --source input/ --model-depth chqeckpoints/my_model/best_model.pt

输出：
  - 单图：原图+深度对比图、深度彩色图、深度 .npy（米数）
  - 视频：左右对比视频、深度 .npz
"""

import os
import sys
import time
import random
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 默认模型：室外度量深度 Base（输出真实米数）
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"


def load_model(model_path, device="cuda"):
    """加载模型（支持预训练模型或微调 checkpoint）"""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # checkpoint 文件不存在时自动回退到预训练模型
    if model_path and model_path.endswith(".pt") and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        base_model = ckpt.get("model_name", DEFAULT_MODEL)
        print(f"[Model] 加载基础模型: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)
        model = AutoModelForDepthEstimation.from_pretrained(base_model).to(device)
        print(f"[Model] 加载微调权重: {model_path}")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        if model_path and model_path.endswith(".pt") and not os.path.exists(model_path):
            print(f"[Model] checkpoint 不存在: {model_path}，回退到预训练模型")
        model_name = DEFAULT_MODEL
        print(f"[Model] 加载预训练模型: {model_name}")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    model.eval()
    print(f"[Model] 已加载，使用设备: {device}")
    return model, processor


def depth_to_colormap(depth):
    """米数深度图转彩色可视化，返回 (vis, d_min, d_max)"""
    valid = depth > 0
    if valid.sum() > 0:
        d_min, d_max = float(depth[valid].min()), float(depth[valid].max())
        vis = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    else:
        d_min, d_max = 0.0, 0.0
        vis = np.zeros_like(depth, dtype=np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis[~valid] = 0
    return vis, d_min, d_max


def annotate_depth(vis, d_min, d_max, img_h, pixel_points=None, pixel_depths=None, frame_time=None):
    """在深度图标注米数范围 + 帧检测时间 + 多点像素深度监测"""
    # 左上角：深度范围 + 检测时间
    lines = [f"{d_min:.1f}m ~ {d_max:.1f}m"]
    if frame_time is not None:
        lines.append(f"time: {frame_time:.0f}ms")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    # 计算背景框大小
    max_w = 0
    total_h = 0
    text_sizes = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_sizes.append((tw, th))
        max_w = max(max_w, tw)
        total_h += th + 4
    # 绘制黑色背景框
    cv2.rectangle(vis, (8, 8), (8 + max_w + 12, 8 + total_h + 4), (0, 0, 0), -1)
    # 逐行绘制文字
    y = 12
    for line, (tw, th) in zip(lines, text_sizes):
        cv2.putText(vis, line, (12, y + th), font, font_scale, (255, 255, 255), thickness)
        y += th + 4

    # 多点像素深度监测
    if pixel_points and pixel_depths and len(pixel_points) == len(pixel_depths):
        h, w = vis.shape[:2]
        for i, ((px, py), pdepth) in enumerate(zip(pixel_points, pixel_depths)):
            if not (0 <= px < w and 0 <= py < h):
                continue
            # 缩小十字标记：半径6，线条宽1
            cr = 6
            cv2.line(vis, (px - cr, py), (px + cr, py), (0, 0, 255), 1)
            cv2.line(vis, (px, py - cr), (px, py + cr), (0, 0, 255), 1)
            cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)

            # 绘制该点深度值标签
            p_txt = f"P{i+1}({px},{py}): {pdepth:.2f}m"
            (ptw, pth), _ = cv2.getTextSize(p_txt, font, 0.5, 1)
            lbl_x = px + cr + 3
            if lbl_x + ptw + 6 > w:
                lbl_x = px - cr - ptw - 10
            lbl_y = max(py - cr - 4, pth + 2)
            cv2.rectangle(vis, (lbl_x - 1, lbl_y - pth - 1),
                          (lbl_x + ptw + 4, lbl_y + 2), (0, 0, 0), -1)
            cv2.putText(vis, p_txt, (lbl_x + 1, lbl_y),
                        font, 0.5, (0, 255, 255), 1)

    cv2.putText(vis, "near", (10, img_h - 15), font, 0.6, (0, 255, 0), 1)
    cv2.putText(vis, "far", (vis.shape[1] - 50, img_h - 15), font, 0.6, (0, 0, 255), 1)
    return vis


# ============================================================
# 车辆检测器：核心判定逻辑（与 detect_car.py 同步）
# ============================================================
# 全局 YOLO 模型缓存：多视频/多 detector 实例共享同一权重，避免每个视频都懒加载打印打断 tqdm
_YOLO_CACHE: dict = {}  # key=(model_path_norm, device_norm) -> yolo_model_obj

class VehicleDetector:
    """
    基于检测线多点深度变化的车辆存在判定（v8: 非对称双列 + 全点检测）。

    核心改进：
    - 支持非对称双列检测线（x=40共7个点，x=55共8个点，共15个点）
    - 奇数点数（如15）自动走全点模式，所有点独立参与检测
    - 偶数且≥10的点数走双列对称聚合模式
    - 预热阶段前3帧直接加入buffer，第4帧开始用buffer中位数检测异常
    - 异常帧（有车）不加入buffer，避免基线被污染
    - 使用中位数而非均值建立基线，抗深度模型不稳定
    """

    def __init__(self, n_points=5,
                 distance_drop_threshold=0.8,
                 normal_fluctuation=1.5,
                 fps=15.0,
                 baseline_warmup=5,
                 baseline_alpha=0.05,
                 min_car_points=1,
                 min_stable_points=3,
                 car_confirm_frames=3,
                 stable_confirm_frames=5,
                 max_detect_depth=30.0,
                 relative_drop_threshold=0.15,
                 max_vehicle_depth=20.0,
                 # ===== YOLO 视觉检测参数 =====
                 yolo_enabled=False,
                 yolo_model_path="yolov8n.pt",
                 yolo_device="",
                 yolo_confidence=0.40,
                 yolo_iou=0.45,
                 yolo_vehicle_classes=(1, 2, 3, 5, 7),
                 yolo_enter_bottom_ratio=0.30,
                 yolo_min_box_ratio=0.02,
                 # ===== 多规则融合权重 (R1+R2+R3+R4+R5 = 1.00) =====
                 rule_weights=None,
                 car_confidence_threshold=0.470):
        self.n_points = n_points
        self.distance_drop_threshold = distance_drop_threshold
        self.normal_fluctuation = normal_fluctuation
        self.fps = fps
        self.baseline_warmup = baseline_warmup
        self.baseline_alpha = baseline_alpha
        self.min_car_points = min_car_points
        self.min_stable_points = min_stable_points
        self.car_confirm_frames = car_confirm_frames
        self.stable_confirm_frames = stable_confirm_frames
        self.max_detect_depth = max_detect_depth
        self.relative_drop_threshold = relative_drop_threshold
        self.max_vehicle_depth = max_vehicle_depth

        # 双列对称聚合模式：偶数且≥10
        # 非对称双列（如15点）走全点模式，不聚合
        self.is_dual_col = (n_points % 2 == 0 and n_points >= 10)
        if self.is_dual_col:
            self.col_size = n_points // 2
            self.n_effective = self.col_size
            # 对称双列：跳过P1天空点
            self.road_zone_start = 1
        else:
            self.col_size = n_points
            self.n_effective = n_points
            # 全点模式：跳过前2个天空/远景点
            self.road_zone_start = min(2, max(1, n_points // 7))

        self.n_road_points = self.n_effective - self.road_zone_start

        # 预热阶段异常检测所需的最小buffer帧数
        # 至少2帧后开始检测，但需要至少min_car_points个点同时触发
        self.warmup_detect_min = 2

        # 滑动窗口参数
        self.window_size = 10

        # ===== 多规则权重：深度变化(70%) + 视觉(30%) =====
        default_weights = {
            'R1': 0.10,   # Way4 深度骤变
            'R2': 0.25,   # Way5 绝对深度分布 (主规则)
            'R3': 0.25,   # Way1 基线对比下降 (主规则)
            'R4': 0.10,   # 车辆区域统计特征
            'R5': 0.30,   # YOLO 视觉检测 (视觉高权重)
        }
        self.rule_weights = rule_weights if rule_weights else default_weights

        # 加权置信度阈值（≥此值判定有车）
        self.car_confidence_threshold = car_confidence_threshold

        # ===== YOLO 视觉检测 (延迟加载，禁用或模型不存在时静默降级) =====
        self.yolo_enabled = yolo_enabled
        self.yolo_model_path = yolo_model_path
        self.yolo_device = yolo_device
        self.yolo_confidence = yolo_confidence
        self.yolo_iou = yolo_iou
        self.yolo_vehicle_classes = tuple(int(c) for c in yolo_vehicle_classes)
        self.yolo_enter_bottom_ratio = yolo_enter_bottom_ratio
        self.yolo_min_box_ratio = yolo_min_box_ratio
        self._yolo_model = None
        self._yolo_load_error = None  # 记录加载失败原因（静默降级不崩溃）

        # 状态变量
        self.baseline = None       # 正式基线数组
        self.warmup_buffer = []    # 预热缓冲区（只存无车帧）
        self.history_window = []   # 滑动窗口
        self.current_state = "unknown"
        self.pending_state = None
        self.confirm_count = 0
        self.last_depths = None
        self.last_confidence = 0.0  # 最近帧的总置信度
        self.last_vision_score = 0.0  # 最近帧的 YOLO 视觉评分
        self._last_scores = {}     # 最近帧各规则分数
        self._last_yolo_boxes = []  # 最近帧的 YOLO 车辆检测框 [(x1,y1,x2,y2,conf,cls),...]

    # ============================================================
    # YOLO 视觉检测（延迟加载 + 静默降级 + 进入检测区域判定）
    # ============================================================
    def _ensure_yolo_loaded(self):
        """
        延迟加载 YOLO 模型（首次调用时执行，失败则记录错误并永久关闭，不影响深度检测）。
        全局缓存：同一 (model_path, device) 只真正加载一次，后续实例共享对象，避免重复加载/打印。
        返回 True=已就绪, False=不可用（禁用/加载失败）。
        """
        global _YOLO_CACHE
        if not self.yolo_enabled:
            return False
        if self._yolo_model is not None:
            return True
        if self._yolo_load_error is not None:
            return False  # 已失败过，不再重试

        try:
            from ultralytics import YOLO
            import torch

            # 确定设备：优先使用 yolo_device 参数，否则 cuda/cpu 自动判断
            device = self.yolo_device
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # 兼容路径格式（Windows/Posix），并支持 ultralytics 官方模型名（如 yolov8n.pt）
            model_path = self.yolo_model_path
            if (isinstance(model_path, str) and model_path.endswith(".pt")
                    and not os.path.isabs(model_path) and not os.path.exists(model_path)):
                # 相对路径尝试拼接项目根目录
                candidate = os.path.join(os.getcwd(), model_path)
                if os.path.exists(candidate):
                    model_path = candidate
                # 否则交给 ultralytics 自动下载官方权重

            # ---- 全局缓存命中 ----
            cache_key = (os.path.normcase(os.path.abspath(model_path))
                         if isinstance(model_path, str) and os.path.isabs(model_path) else str(model_path),
                         str(device).lower())
            if cache_key in _YOLO_CACHE:
                self._yolo_model = _YOLO_CACHE[cache_key]
                return True  # 命中时静默，不再打印

            # ---- 首次加载 ----
            yolo = YOLO(model_path)
            # 预热一次空推断（加速首帧），但不读结果
            try:
                yolo.predict(
                    np.zeros((64, 64, 3), dtype=np.uint8),
                    verbose=False, conf=0.01, imgsz=64
                )
            except Exception:
                pass
            print(f"[YOLO] 已加载模型: {model_path} (设备={device})")
            _YOLO_CACHE[cache_key] = yolo
            self._yolo_model = yolo
            return True
        except Exception as e:
            self._yolo_load_error = str(e)
            self.yolo_enabled = False  # 永久关闭，避免每帧重试
            print(f"[WARN] YOLO 加载失败，已禁用视觉检测: {e}")
            return False

    def yolo_detect_frame(self, frame_bgr):
        """
        在单帧图像上运行 YOLO 车辆检测，并判定车辆是否进入检测区域。

        视觉协同原则（非独立触发）：
        1. 只识别配置列表中的 vehicle_classes（car/truck/bus/motorcycle/bicycle）
        2. 只统计画面下方 enter_bottom_ratio 区域内的框（判定车已到检测线附近）
        3. 忽略极小框 min_box_ratio（过滤远处背景车辆误检）

        Args:
            frame_bgr: BGR 格式原始画面（cv2.imread/cap.read 返回格式）

        Returns:
            vision_score: float [0..1]  — YOLO 视觉检测+进入区域综合评分
            boxes: list[(x1,y1,x2,y2,conf,cls)]  — 命中的车辆检测框
        """
        self._last_yolo_boxes = []
        if frame_bgr is None:
            self.last_vision_score = 0.0
            return 0.0, []
        if not self._ensure_yolo_loaded():
            self.last_vision_score = 0.0
            return 0.0, []

        h, w = frame_bgr.shape[:2]
        img_area = float(w * h)
        # 检测区域底部线：画面最下方 y >= h * (1 - ratio) 才算"进入检测区"
        enter_y = h * (1.0 - self.yolo_enter_bottom_ratio)
        min_box_area = img_area * self.yolo_min_box_ratio * self.yolo_min_box_ratio

        try:
            # YOLO 预测 (BGR 直接喂入，ultralytics 内部处理)
            with torch.no_grad():
                results = self._yolo_model.predict(
                    frame_bgr,
                    verbose=False,
                    conf=float(self.yolo_confidence),
                    iou=float(self.yolo_iou),
                    classes=list(self.yolo_vehicle_classes),
                    device=self.yolo_device if self.yolo_device else None,
                )
        except Exception as e:
            print(f"[WARN] YOLO 预测失败: {e}")
            self.last_vision_score = 0.0
            return 0.0, []

        hits = []  # 命中: (框, conf, cls)
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else np.array(r.boxes.xyxy)
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
                clss = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, 'cpu') else np.array(r.boxes.cls)
                for k in range(len(xyxy)):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[k]]
                    c = float(confs[k])
                    cls_id = int(clss[k])
                    # 过滤：非车辆类、极小框（远处误检）、未进入检测区
                    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    if bw * bh < min_box_area:
                        continue
                    if y2 < enter_y:
                        # 框底部在检测线上方（车未到线）
                        continue
                    hits.append((int(x1), int(y1), int(x2), int(y2), c, cls_id))

        # 计算视觉评分 (0..1)：取命中框中最高置信度，再乘进入深度惩罚
        if not hits:
            self.last_vision_score = 0.0
            return 0.0, []

        # 最大框置信度
        max_conf = max(h[4] for h in hits)
        # 进入惩罚：框底部越靠近下方(enter_y)分越高，越过底部(接近摄像头)分最高
        depth_penalties = []
        for h_box in hits:
            y2_box = h_box[3]
            # enter_y 以下归一化到 [0,1]
            span = float(h - enter_y) if h > enter_y else 1.0
            depth_p = min(1.0, max(0.0, (y2_box - enter_y) / span))
            depth_penalties.append(depth_p)
        max_depth_p = max(depth_penalties)

        # 多车加分：≥2辆车进入检测区 → 更强信号
        multi_bonus = min(0.15, 0.05 * max(0, len(hits) - 1))
        vision_score = float(max_conf * 0.65 + max_depth_p * 0.35 + multi_bonus)
        vision_score = max(0.0, min(1.0, vision_score))

        self._last_yolo_boxes = hits
        self.last_vision_score = vision_score
        return vision_score, hits

    def _column_aggregate(self, depths):
        """
        双列聚合：将两列同高度点取min，聚合为等效单数组。

        双列模式下，depths格式为[col0_p0, col0_p1, ..., col0_p6, col1_p0, col1_p1, ..., col1_p6]
        聚合后返回[min(p0,p0'), min(p1,p1'), ..., min(p6,p6')]
        """
        if not self.is_dual_col:
            return depths
        d = np.array(depths, dtype=np.float32)
        col0 = d[:self.col_size]
        col1 = d[self.col_size:2 * self.col_size]
        # 取两列中更浅的深度（车辆覆盖时深度更近）
        aggregated = np.minimum(col0, col1)
        return aggregated

    def _get_road_depths(self, depths):
        """提取检测线上P2-P7的深度值（双列时已聚合）"""
        if self.is_dual_col:
            depths = self._column_aggregate(depths)
        return depths[self.road_zone_start:]

    def _get_valid_road_mask(self, depths):
        """获取检测线点的有效掩码"""
        road = self._get_road_depths(depths)
        return (~np.isnan(road)) & (road > 0) & (road <= self.max_detect_depth)

    def _get_ref_baseline(self):
        """获取参考基线：正式基线优先，无基线时用预热buffer中位数"""
        if self.baseline is not None:
            return self.baseline
        if len(self.warmup_buffer) >= self.warmup_detect_min:
            buf = np.array(self.warmup_buffer)
            return np.median(buf, axis=0)
        return None

    @staticmethod
    def detect_vehicle_present(depths, valid_mask, road_zone_start,
                                min_points=3, bounce_min=0.8, max_depth=15.0):
        """保留兼容接口：反弹检测（已过滤异常深度点）

        Args:
            depths: 深度数组
            valid_mask: 有效点掩码
            road_zone_start: 检测区域起始索引
            min_points: 最少有效点数
            bounce_min: 反弹最小深度差
            max_depth: 参与反弹检测的最大深度（超过此值的点视为天空/远景）
        """
        n = len(depths)
        road_indices = np.arange(road_zone_start, n)
        road_valid = valid_mask[road_indices] & (depths[road_indices] > 0) & (depths[road_indices] <= max_depth)
        valid_road = road_indices[road_valid]
        if len(valid_road) < min_points:
            return False, None
        bounces = 0
        for i in range(len(valid_road) - 1):
            if valid_road[i + 1] == valid_road[i] + 1:
                d_upper = depths[valid_road[i]]
                d_lower = depths[valid_road[i + 1]]
                if d_upper > 0 and d_lower > d_upper + bounce_min:
                    bounces += 1
        if bounces >= 1:
            return True, valid_road
        return False, None

    def _analyze_depth_pattern(self, road_depths, valid_mask):
        """分析深度分布模式，判断是否具有车辆特征的双峰分布

        车辆特征：深度呈双峰分布（近=路面+车辆覆盖，远=车辆顶面/远端），
        且近/远簇之间有明确间隙（车辆高度导致）。
        场景特征：深度呈连续渐变，无明确间隙分隔。

        Returns:
            depth_span_ok: 是否具有车辆特征的深度跨度模式
            span_valid_depths: 有效深度值列表
        """
        span_valid_depths = []
        for i in range(len(road_depths)):
            if not valid_mask[i]:
                continue
            d = road_depths[i]
            if d <= 0 or d > self.max_detect_depth:
                continue
            span_valid_depths.append(d)

        if len(span_valid_depths) < 4:
            return False, span_valid_depths

        max_depth = max(span_valid_depths)

        # 核心约束：车辆在检测线上会产生足够的浅层点（≤max_vehicle_depth）
        # 部分覆盖场景：浅层簇=车辆覆盖（5-10m），深层簇=未覆盖的背景（20-50m）
        # 桥梁场景：所有点都在深层（17-28m），无有效浅层簇
        shallow_depths = [d for d in span_valid_depths if d <= self.max_vehicle_depth]
        if len(shallow_depths) < self.min_car_points:
            return False, span_valid_depths

        sorted_depths = sorted(span_valid_depths)

        # 路径A：传统阈值（近点5-12m + 远点≥15m + 总跨度≥8m + 间隙检查）
        # 增加间隙检查：近簇与远簇之间必须有明确分离（非平滑渐变）
        span_car_depths = [d for d in span_valid_depths if 5.0 <= d <= 12.0]
        span_far_depths = [d for d in span_valid_depths if d >= 15.0]
        if len(span_car_depths) >= self.min_car_points and len(span_far_depths) >= 1:
            depth_span = max_depth - min(span_valid_depths)
            if depth_span >= 8.0:
                # 间隙检查：近簇最大值与远簇最小值之间必须有≥2m间隙
                # 且间隙区内不能有太多过渡点（≤1个），否则是渐变而非双峰
                near_max = max(span_car_depths)
                far_min = min(span_far_depths)
                if far_min - near_max >= 2.0:
                    gap_points = [d for d in span_valid_depths
                                  if near_max < d < far_min]
                    if len(gap_points) <= 1:
                        return True, span_valid_depths

        # 路径B：动态间隙检测 + 聚类紧密度约束
        # 适用于车辆部分覆盖、场景深度有限的情况
        max_gap = 0.0
        max_gap_idx = 0
        for j in range(len(sorted_depths) - 1):
            gap = sorted_depths[j + 1] - sorted_depths[j]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = j

        if max_gap > 3.0:
            near_count = max_gap_idx + 1
            far_count = len(sorted_depths) - near_count
            if near_count >= self.min_car_points and far_count >= 1:
                # 聚类紧密度检查：
                # 车辆的近/远簇各自紧凑（簇内range≤5m），
                # 而场景渐变则各点分散、簇内range很大
                near_cluster = sorted_depths[:max_gap_idx + 1]
                far_cluster = sorted_depths[max_gap_idx + 1:]
                near_range = max(near_cluster) - min(near_cluster)
                far_range = max(far_cluster) - min(far_cluster)

                # 安全检查：近簇最大深度应≤15m（车辆合理深度）
                # 桥梁场景的"浅"点在17-20m范围，远超车辆深度
                near_max_depth = max(near_cluster)
                if near_max_depth <= 15.0:
                    # 近簇：路面+车辆覆盖，深度变化≤5m（含路面坡度）
                    # 远簇：车辆顶面/远端，深度变化≤6m
                    # 部分覆盖场景：远簇为背景（>max_vehicle_depth），无需紧密度约束
                    has_background = max(far_cluster) > self.max_vehicle_depth
                    if near_range <= 5.0:
                        if has_background:
                            # 部分覆盖：远簇是背景，仅需近簇紧凑+间隙足够
                            return True, span_valid_depths
                        elif far_range <= 6.0:
                            # 正常车辆：近远簇都是车辆特征，都需紧凑
                            return True, span_valid_depths

        return False, span_valid_depths

    def _detect_car_by_depth(self, road_depths, valid_mask, ref_baseline=None):
        """
        基于加权置信度的车辆检测（消除一票否决，综合多规则评分）。

        置信度规则：
        - R1 [15%]: Way4 深度骤变 — depth_span_ok + 基线对比（双峰+下降）
        - R2 [30%]: Way5 绝对深度 — 深背景/双峰/全覆盖 + 紧凑簇特征
        - R3 [30%]: Way1 基线对比 — 独立基线下降（增强分布特征/比例门控）
        - R4 [25%]: 车辆区域统计 — car_depths(5-15m)紧凑度 + shallow点 + 深背景上下文

        每条规则输出 [0..1] 置信度分数，加权求和后 ≥0.5 判定有车。
        阈值可通过 self.car_confidence_threshold 调整。
        """
        if ref_baseline is None:
            ref_baseline = self._get_ref_baseline()

        depth_span_ok, span_valid_depths = self._analyze_depth_pattern(
            road_depths, valid_mask
        )

        all_valid_depths = []
        for i in range(len(road_depths)):
            if valid_mask[i]:
                all_valid_depths.append(road_depths[i])

        n_valid = len(all_valid_depths)
        if n_valid < 4:
            self._last_scores = {}
            self.last_confidence = 0.0
            return False

        max_depth = max(all_valid_depths)
        min_depth = min(all_valid_depths)
        depth_range = max_depth - min_depth
        median_depth = float(np.median(all_valid_depths))

        absolute_car_points = 0
        car_depths = []
        has_deep_background = False
        shallow_depths = []
        for i in range(len(road_depths)):
            if not valid_mask[i]:
                continue
            d = road_depths[i]
            if 5.0 <= d <= 15.0:
                absolute_car_points += 1
                car_depths.append(d)
            if d > 20.0:
                has_deep_background = True
            if d < 5.0:
                shallow_depths.append(d)

        shallow_count = len(shallow_depths)
        deep_bg_ratio = sum(1 for d in all_valid_depths if d > 20.0) / n_valid

        # ==================== 提前计算基线对比（R1/R3共用） ====================
        car_points_w1 = 0
        drops_w1 = []
        if ref_baseline is not None:
            for i in range(len(road_depths)):
                if not valid_mask[i] or i >= len(ref_baseline):
                    continue
                d = road_depths[i]
                base = ref_baseline[i]
                if base <= 0:
                    continue
                abs_drop = base - d
                rel_drop = abs_drop / max(base, 0.01)
                if (abs_drop > self.distance_drop_threshold and
                        rel_drop > self.relative_drop_threshold):
                    car_points_w1 += 1
                    drops_w1.append(abs_drop)

        has_some_drop = car_points_w1 >= 2
        ratio_w1 = car_points_w1 / n_valid if n_valid > 0 else 0
        mean_drop_w1 = float(np.mean(drops_w1)) if drops_w1 else 0

        # all_in_vehicle_range: 几乎所有点都在车辆深度范围(5-15m) → 全覆盖
        # 放宽 max_depth 到 20m（车顶/货车顶部可达20m，但不会更高）
        # 邻近车辆不可能满足此条件（大部分点仍是正常路面深度）
        all_in_vehicle_range = (n_valid > 0 and
                                absolute_car_points + shallow_count >= n_valid - 1 and
                                absolute_car_points >= self.min_car_points and
                                max_depth <= 20.0 and
                                (has_some_drop or depth_range > 1.0))

        # ==================== R1: Way4 深度骤变 ====================
        r1_score = 0.0
        if depth_span_ok and ref_baseline is not None and car_points_w1 > 0:
            if drops_w1:
                ratio_score = min(1.0, car_points_w1 / max(self.min_car_points + 2, n_valid * 0.5))
                mag_score = min(1.0, mean_drop_w1 / 5.0)
                r1_score = (ratio_score + mag_score) / 2.0
        elif (not depth_span_ok and ref_baseline is not None and
              mean_drop_w1 >= 5.0 and car_points_w1 >= self.min_car_points * 2):
            ratio_score = min(1.0, car_points_w1 / max(self.min_car_points + 2, n_valid * 0.5))
            mag_score = min(1.0, mean_drop_w1 / 8.0)
            r1_score = (ratio_score + mag_score) / 2.0 * 0.7

        # ==================== R2: Way5 绝对深度 ====================
        r2_score = 0.0
        if absolute_car_points >= self.min_car_points and car_depths:
            # 路径1: 有深背景（部分覆盖货车经典场景）
            # 路径2: 双峰分布（有近远两簇）
            # 路径3: all_in_vehicle_range 全覆盖
            # 路径4: 渐进接近（整体浅+有基线下降+无深背景）
            progressive_ok = (not has_deep_background and
                              not depth_span_ok and
                              max_depth <= 15.0 and
                              depth_range <= 10.0 and
                              has_some_drop)
            path_ok = (has_deep_background or depth_span_ok or
                       all_in_vehicle_range or progressive_ok)
            if path_ok and len(car_depths) >= 2:
                sorted_car = sorted(car_depths)
                internal_gaps = [sorted_car[j+1] - sorted_car[j]
                                 for j in range(len(sorted_car) - 1)]
                median_gap = float(np.median(internal_gaps))
                car_range = max(car_depths) - min(car_depths)

                # 渐进接近场景 depth_range 本身较大 + gap 可能因远离车辆拉大
                if progressive_ok:
                    gap_score = max(0.0, min(1.0, (2.0 - median_gap) / 2.0))
                    range_score = max(0.0, min(1.0, (14.0 - car_range) / 8.0))
                else:
                    gap_score = max(0.0, min(1.0, (1.5 - median_gap) / 1.5))
                    range_score = max(0.0, min(1.0, (9.0 - car_range) / 4.0))
                r2_score = (gap_score + range_score) / 2.0

                if has_deep_background:
                    car_ratio = absolute_car_points / n_valid
                    context_bonus = max(0.0, min(0.4, car_ratio * 0.6))
                    r2_score = min(1.0, r2_score + context_bonus)

        # ==================== R3: Way1 基线对比 ====================
        r3_score = 0.0
        if ref_baseline is not None and car_points_w1 >= self.min_car_points:
            # Way1 增强检查（多条件互补，防邻近车辆误检）
            # 条件A: 分布特征门控（防邻近车辆误检）
            cond_dist = (depth_span_ok or max_depth <= 10.0 or depth_range <= 7.0)
            # 条件B: 极高下降比例（90%以上点显著下降 → 整体变化剧烈）
            cond_ratio = ratio_w1 >= 0.90
            # 条件C: 大平均下降幅度（>5m平均下降 → 变化非常剧烈）
            cond_mag = mean_drop_w1 >= 5.0

            dist_ok = cond_dist or cond_ratio or cond_mag
            if dist_ok:
                ratio_score_r3 = min(1.0, ratio_w1 * 1.5)
                mag_score_r3 = min(1.0, mean_drop_w1 / 5.0)
                r3_score = max(ratio_score_r3, mag_score_r3)

        # ==================== R4: 车辆区域统计特征 ====================
        r4_score = 0.0

        # 4a: car_depths(5-15m) 紧凑度评分
        if car_depths and len(car_depths) >= 2:
            car_max = max(car_depths)
            car_min = min(car_depths)
            car_range_v = car_max - car_min
            car_median = float(np.median(car_depths))

            r4a_max = max(0.0, min(1.0, (15.0 - car_max) / 7.0))
            r4a_range = max(0.0, min(1.0, (8.0 - car_range_v) / 4.0))
            r4a_med = max(0.0, min(1.0, (8.0 - car_median) / 4.0))
            car_compact = r4a_max * 0.3 + r4a_range * 0.35 + r4a_med * 0.35
        else:
            car_compact = 0.0

        # 4b: shallow点(<5m)评分（近地面/车辆底部特征）
        if n_valid > 0:
            shallow_ratio = shallow_count / n_valid
            r4b_shallow = max(0.0, min(1.0, (shallow_ratio - 0.15) / 0.35))
        else:
            r4b_shallow = 0.0

        # 4c: 深背景+车辆上下文评分（有深背景时car点比例越高越好）
        if has_deep_background and car_depths:
            car_ratio_bg = absolute_car_points / n_valid
            r4c_context = max(0.0, min(1.0, (car_ratio_bg - 0.4) / 0.4))
        else:
            r4c_context = 0.0

        # 4d: 全覆盖模式评分（所有点都是车辆/shallow）
        if all_in_vehicle_range:
            all_v_ratio = (absolute_car_points + shallow_count) / n_valid
            r4d_all = max(0.0, min(1.0, (all_v_ratio - 0.6) / 0.3))
        else:
            r4d_all = 0.0

        # R4 = 取各子项最大值（多路径互补）
        r4_score = max(car_compact, r4b_shallow, r4c_context, r4d_all)

        # ==================== 加权求和（深度变化+视觉协同，自适应模式） ====================
        # 两种模式：
        # A. 纯深度模式（YOLO禁用 或 R5=0 无视觉信号）→ 硬编码老权重 R1=0.15/R2=0.30/R3=0.30/R4=0.25
        #    → 保持老 25 用例判定边界（100% 兼容）
        # B. 视觉协同模式（YOLO启用 且 R5>0）→ 用配置权重 R1~R5，默认深度变化70%+视觉30%
        weights = self.rule_weights
        r5_score = float(getattr(self, 'last_vision_score', 0.0))
        yolo_active = bool(getattr(self, 'yolo_enabled', False))

        if (not yolo_active) or r5_score <= 1e-9:
            # ----- 模式 A：纯深度，100% 兼容老判定 -----
            w1, w2, w3, w4, w5_used = 0.15, 0.30, 0.30, 0.25, 0.0
        else:
            # ----- 模式 B：深度 + 视觉协同 -----
            w1 = float(weights.get('R1', 0.10))
            w2 = float(weights.get('R2', 0.25))
            w3 = float(weights.get('R3', 0.25))
            w4 = float(weights.get('R4', 0.10))
            w5_used = float(weights.get('R5', 0.30))

        confidence = (r1_score * w1 +
                      r2_score * w2 +
                      r3_score * w3 +
                      r4_score * w4 +
                      r5_score * w5_used)

        # ==================== 已确认 CAR 状态下的 YOLO 视觉高置信度增强（防闪烁） ====================
        # 用户反馈：车经过检测线的过程中（已进入 CAR 状态），车速不均匀停一下时会闪烁几帧 no_car
        # 根因：车停住后，深度骤变(R1/R3)归零，MOG2运动检测认为"没运动"→ 在融合阶段把中等信号打为 no_car
        # 修复：若 current_state 已经是 car 且 YOLO 视觉仍确认车在检测区(R5≥0.40且至少1框)，
        #       直接对 confidence 做 boost：min(0.98, confidence * 1.30 + 0.10)
        #       → 让 0.38~0.46 的边缘信号跨过 0.470 阈值，并冲到 0.60+ 绕过 MOG2 中等信号门槛
        car_vision_boost_applied = False
        r5_boxes_n = len(getattr(self, '_last_yolo_boxes', []))
        if (getattr(self, 'current_state', None) == 'car'
                and r5_score >= 0.40
                and r5_boxes_n >= 1):
            boosted = min(0.98, confidence * 1.30 + 0.10)
            if boosted > confidence:
                confidence = boosted
                car_vision_boost_applied = True

        # ==================== 距离门控（Vehicle Proximity Gate） ====================
        # 有远处深背景(max_depth>25m 桥梁/天空) + 有基线下降(车辆接近导致)
        # 但最近点仍在检测线外(>5.5m) 且无点到达检测线(shallow_count==0)
        # → 强制压低置信度，防止远处车辆+深背景被误判为"部分覆盖货车"
        # 例外1: baseline==current 时（基线污染场景），无基线下降则不触发
        # 例外2: 有密集近点簇(≥4个点在near_point±2m范围) → 车辆真正到达检测线
        # 例外3: YOLO 视觉强信号 (r5≥0.70 且 box≥1) → 车在视觉上已到检测线，不拦
        # 例外4: 已CAR状态+视觉仍然见车(boost) → 不允许门控把已确认的车打回低置信度
        near_point = min_depth
        has_baseline_drop = car_points_w1 >= self.min_car_points if ref_baseline is not None else False
        near_cluster = int(np.sum((road_depths >= near_point) & (road_depths <= near_point + 2.0)))
        vision_strong = (r5_score >= 0.70 and r5_boxes_n >= 1)
        proximity_gate = (max_depth > 25.0 and
                          near_point > 5.5 and
                          shallow_count == 0 and
                          has_baseline_drop and
                          near_cluster < 4 and
                          not vision_strong and
                          not car_vision_boost_applied)  # 例外4
        if proximity_gate:
            confidence = min(confidence, 0.35)

        self._last_scores = {
            'R1_Way4_depth_jump': r1_score,
            'R2_Way5_absolute_depth': r2_score,
            'R3_Way1_baseline_drop': r3_score,
            'R4_vehicle_region_stats': r4_score,
            'R5_YOLO_vision': r5_score,
            '  R4a_car_compact': car_compact,
            '  R4b_shallow': r4b_shallow,
            '  R4c_deep_context': r4c_context,
            '  R4d_all_vehicle': r4d_all,
            'depth_span_ok': 1.0 if depth_span_ok else 0.0,
            'has_deep_bg': 1.0 if has_deep_background else 0.0,
            'all_in_vehicle_range': 1.0 if all_in_vehicle_range else 0.0,
            'car_points': absolute_car_points,
            'shallow_count': shallow_count,
            'max_depth': float(max_depth),
            'depth_range': float(depth_range),
            'median_depth': float(median_depth),
            'total_confidence': confidence,
            'proximity_gate': 1.0 if proximity_gate else 0.0,
            'near_point_m': float(near_point),
            'vision_strong_bypass': 1.0 if vision_strong else 0.0,
            'car_vision_boost': 1.0 if car_vision_boost_applied else 0.0,
        }
        self.last_confidence = confidence

        return confidence >= self.car_confidence_threshold

    def update(self, depths, frame_bgr=None):
        """
        更新车辆状态（支持 YOLO 视觉协同）。

        协同原则：视觉不是独立触发，而是作为 R5 规则加权进入综合判定。
        - 只传 depths: 走纯深度检测（向后兼容，YOLO 不参与，R5=0）
        - 传 depths + frame_bgr: 先跑 YOLO 得到 R5，再深度得到 R1-R4，加权综合
        """
        # Step A: 若传了画面，先跑 YOLO 视觉检测（R5 评分写入 self.last_vision_score）
        if self.yolo_enabled and frame_bgr is not None:
            self.yolo_detect_frame(frame_bgr)
        else:
            # 无画面或禁用 YOLO 时，视觉分清零，不干扰纯深度路径
            self.last_vision_score = 0.0
            self._last_yolo_boxes = []

        depths = np.array(depths, dtype=np.float32)
        valid_mask = ~np.isnan(depths) & (depths > 0)
        depths_clean = np.where(valid_mask, depths, 0.0)

        road_depths = self._get_road_depths(depths_clean)
        road_valid = self._get_valid_road_mask(depths_clean)

        # 更新滑动窗口
        self.history_window.append(road_depths.copy())
        if len(self.history_window) > self.window_size:
            self.history_window.pop(0)

        # 获取参考基线（正式基线或预热buffer中位数）
        ref_baseline = self._get_ref_baseline()

        # 检测车辆：始终调用（Way 5不依赖基线，可在预热期即生效）
        # ref_baseline为None时，Way 1/2/4被自动跳过，Way 5独立工作
        car_detected = self._detect_car_by_depth(
            road_depths, road_valid, ref_baseline
        )

        if car_detected:
            # 检测到车辆
            if self.current_state != "car":
                # 强信号快速通道：conf≥0.7 直接确认，跳过confirm_count
                if self.last_confidence >= 0.70:
                    self.confirm_count = self.car_confirm_frames
                else:
                    self.confirm_count += 1
                self.pending_state = "car"
            else:
                self.confirm_count = self.car_confirm_frames
            if self.confirm_count >= self.car_confirm_frames:
                self.current_state = "car"
                # 预热阶段：异常帧不加入buffer（保护基线）
                # 正式阶段：清空warmup_buffer
                if self.baseline is not None:
                    self.warmup_buffer = []

        elif self.current_state == "car":
            # 检查车辆是否离开
            # ============== YOLO 视觉 HOLD：车停住时禁止状态机回退 ==============
            # 防闪烁：已确认car后，若YOLO视觉仍确认车在检测区（R5≥0.5 且至少1框），
            # 跳过"稳定点计数"离开判定；R5≥0.30 弱信号时回退帧门槛 5→12（更难回退）
            # 与 _detect_car_by_depth 里的 boost 对齐：R5≥0.40 时已有 car 确认增强，
            # 因此 HOLD 强门槛从 0.50 降到 0.40，保证"boost时绝不回退"
            r5_hold = float(getattr(self, 'last_vision_score', 0.0))
            r5_boxes_n_hold = len(getattr(self, '_last_yolo_boxes', []))
            last_conf = float(getattr(self, 'last_confidence', 0.0))
            # HOLD 强条件：视觉仍确认有车（R5≥0.40 + 有框），与检测侧boost门槛对齐
            # 额外：即使视觉分不高，只要综合 conf≥0.40（已在车检阈值附近）也不回退
            vision_hold_strong = (
                (r5_hold >= 0.40 and r5_boxes_n_hold >= 1)
                or last_conf >= 0.50  # 综合仍有车信号 → 强锁定
            )
            vision_hold_soft = r5_hold >= 0.30 and r5_boxes_n_hold >= 1
            effective_stable_frames = (
                12 if vision_hold_soft else self.stable_confirm_frames
            )
            # 稳定点门槛从 3 提高到 7（15个点里≥7稳定才真正"回到路面"）
            # 避免 3 个噪声点就触发回退
            min_stable_pts_for_leave = max(self.min_stable_points + 4, 7)
            return_to_no_car = False
            if not vision_hold_strong and ref_baseline is not None:
                stable_count = 0
                for i in range(len(road_depths)):
                    if (road_valid[i] and i < len(ref_baseline)
                            and ref_baseline[i] > 0):
                        deviation = abs(road_depths[i] - ref_baseline[i])
                        if deviation < self.normal_fluctuation:
                            stable_count += 1
                if stable_count >= min_stable_pts_for_leave:
                    return_to_no_car = True

            if return_to_no_car:
                self.pending_state = "no_car"
                self.confirm_count += 1
                if self.confirm_count >= effective_stable_frames:
                    self.current_state = "no_car"
                    if self.baseline is None:
                        # 无基线，用预热buffer建立基线
                        if len(self.warmup_buffer) >= self.warmup_detect_min:
                            buf = np.array(self.warmup_buffer)
                            self.baseline = np.median(buf, axis=0)
                    else:
                        # 有基线，缓慢更新
                        for i in range(len(self.baseline)):
                            if road_valid[i] and self.baseline[i] > 0:
                                self.baseline[i] = (
                                    self.baseline[i] * (1 - self.baseline_alpha) +
                                    road_depths[i] * self.baseline_alpha
                                )
                    self.confirm_count = 0
            else:
                self.pending_state = "car"
                self.confirm_count = self.car_confirm_frames

        else:
            # 无车或未知状态
            self.confirm_count = 0
            if self.baseline is None:
                # 预热阶段：加入buffer
                self.warmup_buffer.append(road_depths.copy())
                if len(self.warmup_buffer) >= self.baseline_warmup:
                    buf = np.array(self.warmup_buffer)
                    self.baseline = np.median(buf, axis=0)
                    self.current_state = "no_car"
                    self.pending_state = "no_car"
                    self.confirm_count = 0
                else:
                    self.current_state = "unknown"
            else:
                # 正式阶段，no_car状态，缓慢更新基线
                for i in range(len(self.baseline)):
                    if road_valid[i] and self.baseline[i] > 0:
                        deviation = abs(road_depths[i] - self.baseline[i])
                        if deviation < self.normal_fluctuation * 2:
                            self.baseline[i] = (
                                self.baseline[i] * (1 - self.baseline_alpha) +
                                road_depths[i] * self.baseline_alpha
                            )
                self.current_state = "no_car"

        self.last_depths = depths_clean.copy()
        return self.current_state

    def has_depth_change(self, depths, abs_threshold=0.2, rel_threshold=0.03, min_points=2):
        """
        检查检测线路面点深度是否相对基线有实质性下降（辅助 MOG2 验证）。

        支持预热阶段（使用buffer中位数作为临时基线）。
        要求至少min_points个点同时满足条件才返回True，避免单点噪声误判。
        """
        depths = np.array(depths, dtype=np.float32)
        road_depths = self._get_road_depths(depths)
        road_valid = self._get_valid_road_mask(depths)

        # 获取基线：优先使用正式基线，否则用buffer中位数
        baseline = self._get_ref_baseline()
        if baseline is None:
            return False

        match_points = 0
        for i in range(len(road_depths)):
            if not road_valid[i] or i >= len(baseline):
                continue
            baseline_val = float(baseline[i])
            if baseline_val <= 0:
                continue
            d = float(road_depths[i])
            abs_drop = baseline_val - d
            rel_drop = abs_drop / max(baseline_val, 0.01)
            if abs_drop > abs_threshold and rel_drop > rel_threshold:
                match_points += 1

        return match_points >= min_points


# ============================================================
# 亮度监听器：实时监测环境亮度，动态调整 MOG2 判定权重
# ============================================================
class BrightnessMonitor:
    """
    实时监测环境亮度，动态调整 MOG2 运动检测的判定权重。

    原理：
    - 亮度高（白天）：深度模型可靠，MOG2 权重降低（减少误触发）
    - 亮度中等（黄昏）：MOG2 权重逐步提升
    - 亮度低（夜间）：深度模型易失效，MOG2 权重最大（辅助判定）

    权重范围：[min_weight, max_weight]，通过 EMA 平滑过渡，避免突变。
    """

    def __init__(self,
                 bright_high=180.0,
                 bright_mid=80.0,
                 min_weight=0.4,
                 max_weight=1.0,
                 weight_alpha=0.05,
                 sample_width=160,
                 sample_height=90,
                 log_interval=50):
        """
        Args:
            bright_high: 高亮度阈值（0-255），高于此值为白天，MOG2 权重最低
            bright_mid: 中亮度阈值（0-255），低于此值为夜间，MOG2 权重最高
            min_weight: 白天 MOG2 最小权重（0~1）
            max_weight: 夜间 MOG2 最大权重（0~1）
            weight_alpha: 权重 EMA 平滑系数（0~1，越小过渡越慢）
            sample_width: 亮度采样宽度（降采样以降低开销）
            sample_height: 亮度采样高度
            log_interval: 每 N 帧记录一次日志
        """
        self.bright_high = bright_high
        self.bright_mid = bright_mid
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_alpha = weight_alpha
        self.sample_width = sample_width
        self.sample_height = sample_height
        self.log_interval = log_interval

        # 运行时状态
        self.smoothed_brightness = None  # EMA 平滑亮度
        self.current_weight = min_weight  # 当前 MOG2 权重
        self.frame_count = 0

        # 日志记录
        self.brightness_log = []  # [(frame, brightness, weight), ...]
        self.performance_log = []  # [(frame, detect_time_ms, mog2_time_ms), ...]

    def compute_brightness(self, frame):
        """
        计算帧的平均亮度（降采样加速）。

        Args:
            frame: BGR 图像

        Returns:
            float: 平均亮度值 (0-255)
        """
        h, w = frame.shape[:2]
        # 降采样到小尺寸以减少计算量
        scale_x = min(1.0, self.sample_width / w)
        scale_y = min(1.0, self.sample_height / h)
        if scale_x < 1.0 or scale_y < 1.0:
            small = cv2.resize(frame, None, fx=scale_x, fy=scale_y,
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def brightness_to_weight(self, brightness):
        """
        将亮度映射到 MOG2 判定权重（线性分段）。

        - brightness >= bright_high → min_weight（白天，MOG2 低权重）
        - bright_mid <= brightness < bright_high → 线性插值
        - brightness < bright_mid → max_weight（夜间，MOG2 高权重）

        Args:
            brightness: 当前亮度值 (0-255)

        Returns:
            float: MOG2 权重 [min_weight, max_weight]
        """
        if brightness >= self.bright_high:
            return self.min_weight
        elif brightness <= self.bright_mid:
            return self.max_weight
        else:
            ratio = (brightness - self.bright_mid) / (self.bright_high - self.bright_mid)
            return self.max_weight - ratio * (self.max_weight - self.min_weight)

    def update(self, frame):
        """
        更新亮度监测和 MOG2 权重（每帧调用）。

        Args:
            frame: BGR 图像

        Returns:
            tuple: (current_brightness, current_weight, brightness_level)
        """
        self.frame_count += 1
        raw_brightness = self.compute_brightness(frame)

        # EMA 平滑亮度
        if self.smoothed_brightness is None:
            self.smoothed_brightness = raw_brightness
        else:
            self.smoothed_brightness = (
                self.smoothed_brightness * (1 - self.weight_alpha) +
                raw_brightness * self.weight_alpha
            )

        # 计算目标权重
        target_weight = self.brightness_to_weight(self.smoothed_brightness)

        # 权重 EMA 平滑过渡（防止突变）
        self.current_weight = (
            self.current_weight * (1 - self.weight_alpha) +
            target_weight * self.weight_alpha
        )

        # 亮度等级（用于日志和可视化）
        if self.smoothed_brightness >= self.bright_high:
            level = "daytime"
        elif self.smoothed_brightness >= self.bright_mid:
            level = "twilight"
        else:
            level = "night"

        # 定期记录日志
        if self.frame_count % self.log_interval == 0:
            self.brightness_log.append({
                "frame": self.frame_count,
                "brightness": round(self.smoothed_brightness, 1),
                "raw_brightness": round(raw_brightness, 1),
                "weight": round(self.current_weight, 3),
                "level": level,
            })

        return self.smoothed_brightness, self.current_weight, level

    def get_weight(self):
        """获取当前 MOG2 权重"""
        return self.current_weight

    def get_brightness_level(self):
        """获取当前亮度等级"""
        if self.smoothed_brightness is None:
            return "unknown"
        if self.smoothed_brightness >= self.bright_high:
            return "daytime"
        elif self.smoothed_brightness >= self.bright_mid:
            return "twilight"
        else:
            return "night"

    def get_logs(self):
        """获取亮度日志"""
        return self.brightness_log

    def reset(self):
        """重置状态（用于新视频）"""
        self.smoothed_brightness = None
        self.current_weight = self.min_weight
        self.frame_count = 0
        self.brightness_log = []


# ============================================================
# MOG2 运动检测器：检测运动目标是否经过检测线
# ============================================================
class MotionLineDetector:
    """
    基于 MOG2 背景减除的运动检测器（带确认帧机制 + 动态亮度权重）。

    工作原理：
    1. 用 MOG2 建立背景模型，提取前景掩码（运动物体）
    2. 在检测线（像素监测点所在的垂直线）附近设置一条窄带
    3. 统计窄带内的前景像素数量
    4. 必须连续 N 帧检测到运动才确认 → 防止单帧噪声误判
    5. 根据环境亮度动态调整判定灵敏度：
       - 白天（亮度高）：提高 min_motion_pixels 和 confirm_frames，降低误触发
       - 夜间（亮度低）：降低 min_motion_pixels 和 confirm_frames，更敏感

    注意：MOG2 单独检测到运动不直接判定有车，必须与深度检测配合：
    - 深度检测到 car → 直接判定有车（可靠）
    - MOG2 确认运动 + 深度有变化 → 判定有车
    - MOG2 确认运动 + 深度稳定 → 可能是反光/噪声，不判定
    """

    def __init__(self, detect_x=40, strip_width=30,
                 min_motion_pixels=50, history=300, var_threshold=25,
                 detect_shadow=False, warmup_frames=30,
                 confirm_frames=3, brightness_monitor=None):
        """
        Args:
            detect_x: 检测线 x 坐标（像素监测点所在垂直线）
            strip_width: 检测带宽度（以 detect_x 为中心的左右各 strip_width/2 像素）
            min_motion_pixels: 基础最小运动像素数（会被亮度权重动态调整）
            history: MOG2 背景模型的历史帧数
            var_threshold: MOG2 像素与背景模型的方差阈值
            detect_shadow: 是否检测阴影（True=检测，但可能误判）
            warmup_frames: MOG2 预热帧数，前 N 帧不检测（背景模型未稳定）
            confirm_frames: 基础确认帧数（会被亮度权重动态调整）
            brightness_monitor: BrightnessMonitor 实例，提供动态权重
        """
        self.detect_x = detect_x
        self.strip_width = max(4, strip_width)
        self.base_min_motion_pixels = min_motion_pixels
        self.base_confirm_frames = confirm_frames
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.current_state = "unknown"  # "motion" / "no_motion" / "unknown"
        self.motion_pixel_count = 0
        self.motion_confirm_count = 0  # 连续运动帧数
        self.confirmed = False  # 是否已确认运动
        self.brightness_monitor = brightness_monitor
        self.current_weight = 0.5  # 当前动态权重

        # 存储有效阈值（由权重调整）
        self._eff_min_motion = min_motion_pixels
        self._eff_confirm_frames = confirm_frames

        # MOG2 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadow
        )

    def _apply_weight(self, weight):
        """
        根据权重调整有效检测阈值。

        权重越高（夜间）：
        - 降低 min_motion_pixels（更敏感，能检测低光下的微弱运动）
        - 降低 confirm_frames（更快确认，减少延迟）

        权重越低（白天）：
        - 提高 min_motion_pixels（更严格，过滤误触发）
        - 提高 confirm_frames（更慢确认，确保可靠）

        Args:
            weight: 当前权重 [0.4, 1.0]
        """
        self.current_weight = weight
        # 高权重 → 低阈值（夜间敏感），低权重 → 高阈值（白天严格）
        inv_w = 1.0 - weight  # 反转：weight=1.0 → 0.0（最敏感）
        # min_motion_pixels: 权重1.0(夜间)时降到50%，权重0.4(白天)时升到150%
        self._eff_min_motion = max(
            10,
            int(self.base_min_motion_pixels * (0.5 + inv_w * 1.0))
        )
        # confirm_frames: 权重1.0(夜间)时降到50%，权重0.4(白天)时升到150%
        self._eff_confirm_frames = max(
            1,
            int(self.base_confirm_frames * (0.5 + inv_w * 1.0))
        )

    def update(self, frame):
        """
        输入一帧图像，返回是否有运动目标经过检测线。

        Args:
            frame: BGR 图像 (h, w, 3)

        Returns:
            state: "motion" / "no_motion" / "unknown"
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        # 更新亮度监测（如有）
        if self.brightness_monitor:
            _, weight, _ = self.brightness_monitor.update(frame)
            self._apply_weight(weight)

        # 确保检测线在画面内
        cx = max(0, min(self.detect_x, w - 1))

        # MOG2 前景提取
        fg_mask = self.bg_subtractor.apply(frame)

        # 预热期内背景模型不稳定，不检测
        if self.frame_count <= self.warmup_frames:
            self.current_state = "unknown"
            self.motion_pixel_count = 0
            self.motion_confirm_count = 0
            self.confirmed = False
            return self.current_state

        # 阴影处理：MOG2 中阴影值为127，运动前景为255
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 形态学操作：去噪 + 连通
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 提取检测带：以 cx 为中心的垂直窄带
        half_w = self.strip_width // 2
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w + 1)
        strip_mask = fg_mask[:, x1:x2]

        # 统计检测带内的前景像素数
        self.motion_pixel_count = int(np.sum(strip_mask > 0))

        # 确认帧机制（使用权重调整后的有效阈值）
        if self.motion_pixel_count >= self._eff_min_motion:
            self.motion_confirm_count += 1
        else:
            self.motion_confirm_count = max(0, self.motion_confirm_count - 1)

        if self.motion_confirm_count >= self._eff_confirm_frames:
            self.current_state = "motion"
            self.confirmed = True
        elif self.motion_pixel_count < self._eff_min_motion:
            self.current_state = "no_motion"
            self.confirmed = False
        else:
            self.current_state = "no_motion"

        return self.current_state

    def is_confirmed(self):
        """返回运动是否已确认（连续 N 帧检测到）"""
        return self.confirmed

    def get_current_weight(self):
        """返回当前 MOG2 权重（由亮度监测动态调整）"""
        return self.current_weight

    def get_effective_thresholds(self):
        """返回当前生效的检测阈值"""
        return {
            "min_motion_pixels": self._eff_min_motion,
            "confirm_frames": self._eff_confirm_frames,
            "base_min_motion": self.base_min_motion_pixels,
            "base_confirm": self.base_confirm_frames,
        }

    def get_strip_region(self, w):
        """返回检测带的 x 坐标范围 (x1, x2)，用于可视化"""
        cx = max(0, min(self.detect_x, w - 1))
        half_w = self.strip_width // 2
        return max(0, cx - half_w), min(w, cx + half_w + 1)


def _draw_car_state(vis, state, w, motion_state=None, motion_count=0,
                    brightness_level=None, mog2_weight=None):
    """在深度可视化图右上角绘制车辆状态 + MOG2 运动状态 + 亮度权重"""
    state_color = {"car": (0, 0, 255), "no_car": (0, 255, 0), "unknown": (0, 165, 255)}
    state_text = {"car": "CAR DETECTED", "no_car": "NO CAR", "unknown": "UNKNOWN"}
    color = state_color.get(state, (255, 255, 255))
    text = state_text.get(state, state.upper())

    # 计算总高度
    total_h = 48  # 状态框
    if motion_state is not None:
        total_h += 30  # 运动框
    if brightness_level is not None:
        total_h += 24  # 亮度框

    cv2.rectangle(vis, (w - 220, 8), (w - 8, 8 + total_h), (0, 0, 0), -1)
    cv2.putText(vis, text, (w - 210, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    y_offset = 48
    if motion_state is not None:
        motion_color = {"motion": (0, 0, 255), "no_motion": (0, 255, 0), "unknown": (0, 165, 255)}
        motion_text = {"motion": f"MOTION {motion_count}px", "no_motion": "NO MOTION", "unknown": "MOTION ?"}
        m_color = motion_color.get(motion_state, (255, 255, 255))
        m_text = motion_text.get(motion_state, motion_state.upper())
        cv2.putText(vis, m_text, (w - 210, 8 + y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_color, 2)
        y_offset += 30

    if brightness_level is not None:
        bright_colors = {"daytime": (255, 255, 0), "twilight": (0, 165, 255), "night": (128, 128, 255)}
        bright_labels = {"daytime": "DAY", "twilight": "DUSK", "night": "NIGHT"}
        b_color = bright_colors.get(brightness_level, (255, 255, 255))
        b_label = bright_labels.get(brightness_level, brightness_level)
        w_str = f"{mog2_weight:.1f}" if mog2_weight is not None else "?"
        cv2.putText(vis, f"Bright:{b_label} W:{w_str}", (w - 210, 8 + y_offset + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, b_color, 2)
    return vis


def _draw_motion_strip(vis, detector, w, h):
    """在原始帧上绘制 MOG2 检测带"""
    if detector is None:
        return vis
    x1, x2 = detector.get_strip_region(w)
    # 半透明绿色带
    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 255, 0), -1)
    vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
    # 边界线
    cv2.line(vis, (x1, 0), (x1, h), (0, 255, 0), 1)
    cv2.line(vis, (x2, 0), (x2, h), (0, 255, 0), 1)
    return vis


def process_image(model, processor, img_path, output_dir, device, pixel_points=None):
    """处理单张图片：输出对比图 + 深度图 + 米数 .npy"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(images=[img_rgb], return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth
    frame_time = (time.time() - t0) * 1000  # 毫秒

    # 统一成 4D 再插值
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    h, w = img.shape[:2]
    depth = torch.nn.functional.interpolate(
        depth, size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()

    # 计算所有像素点深度值
    pixel_depths = []
    if pixel_points:
        for px, py in pixel_points:
            if 0 <= px < w and 0 <= py < h:
                pixel_depths.append(float(depth[py, px]))
            else:
                pixel_depths.append(None)

    depth_vis, d_min, d_max = depth_to_colormap(depth)
    depth_vis = annotate_depth(depth_vis, d_min, d_max, h, pixel_points, pixel_depths if pixel_depths else None, frame_time)

    # 左右对比图（原图 | 深度图）
    comparison = np.hstack([img, depth_vis])

    name = Path(img_path).stem
    cv2.imwrite(os.path.join(output_dir, f"{name}_comparison.png"), comparison)
    cv2.imwrite(os.path.join(output_dir, f"{name}_depth.png"), depth_vis)
    np.save(os.path.join(output_dir, f"{name}_depth.npy"), depth)

    print(f"  [OK] {img_path}")
    print(f"       深度范围: {d_min:.2f}m ~ {d_max:.2f}m")
    if pixel_depths:
        for i, ((px, py), pd) in enumerate(zip(pixel_points, pixel_depths)):
            if pd is not None:
                print(f"       P{i+1} ({px},{py}): {pd:.2f}m")
    print(f"       输出: {name}_comparison.png, {name}_depth.png, {name}_depth.npy")
    return depth


def process_video(model, processor, video_path, output_dir, device,
                  frame_step=1, max_frames=None, save_depth=True, pixel_points=None,
                  car_detector_config=None):
    """处理视频：输出左右对比视频 + 米数 .npz + 车辆检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name = Path(video_path).stem
    # MJPG + AVI 是 Windows 上兼容性最好的编码（所有播放器都支持）
    video_out_path = os.path.join(output_dir, f"{name}_depth_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w * 2, h))
    if not writer.isOpened():
        video_out_path = os.path.join(output_dir, f"{name}_depth_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w * 2, h))

    frame_idx = 0
    processed = 0
    depths_list = []

    # 初始化深度车辆检测器
    detector = None
    if car_detector_config and pixel_points:
        effective_fps = fps / frame_step if fps > 0 and frame_step > 0 else fps
        detector = VehicleDetector(
            n_points=len(pixel_points),
            distance_drop_threshold=car_detector_config["distance_drop_threshold"],
            normal_fluctuation=car_detector_config["normal_fluctuation"],
            fps=effective_fps,
            baseline_warmup=car_detector_config["baseline_warmup"],
            baseline_alpha=car_detector_config["baseline_alpha"],
            min_car_points=car_detector_config["min_car_points"],
            min_stable_points=car_detector_config["min_stable_points"],
            car_confirm_frames=car_detector_config["car_confirm_frames"],
            stable_confirm_frames=car_detector_config["stable_confirm_frames"],
            max_detect_depth=car_detector_config.get("max_detect_depth", 30.0),
            relative_drop_threshold=car_detector_config.get("relative_drop_threshold", 0.3),
        )
        print(f"  [Car] 深度车辆检测已启用")

    # 初始化 MOG2 运动检测器
    motion_detector = None
    mog2_config = car_detector_config.get("mog2", {}) if car_detector_config else {}
    mog2_enabled = mog2_config.get("enabled", True) if mog2_config else True

    # 初始化亮度监听器（依赖 mog2_config，需在之后）
    brightness_monitor = None
    if mog2_enabled and pixel_points:
        mog2_bright_cfg = mog2_config.get("brightness", {})
        brightness_monitor = BrightnessMonitor(
            bright_high=mog2_bright_cfg.get("bright_high", 180.0),
            bright_mid=mog2_bright_cfg.get("bright_mid", 80.0),
            min_weight=mog2_bright_cfg.get("min_weight", 0.4),
            max_weight=mog2_bright_cfg.get("max_weight", 1.0),
            weight_alpha=mog2_bright_cfg.get("weight_alpha", 0.05),
            log_interval=mog2_bright_cfg.get("log_interval", 50),
        )

    if mog2_enabled and pixel_points:
        detect_x = pixel_points[0][0] if pixel_points else 40
        motion_detector = MotionLineDetector(
            detect_x=detect_x,
            strip_width=mog2_config.get("strip_width", 30),
            min_motion_pixels=mog2_config.get("min_motion_pixels", 50),
            history=mog2_config.get("history", 300),
            var_threshold=mog2_config.get("var_threshold", 25),
            detect_shadow=mog2_config.get("detect_shadow", False),
            warmup_frames=mog2_config.get("warmup_frames", 30),
            confirm_frames=mog2_config.get("confirm_frames", 3),
            brightness_monitor=brightness_monitor,
        )
        print(f"  [MOG2] 运动检测已启用 | 检测线 x={detect_x}, 带宽={mog2_config.get('strip_width', 30)}px, 确认={mog2_config.get('confirm_frames', 3)}帧")
        if brightness_monitor:
            print(f"  [亮度] 动态权重已启用 | 白天≥{brightness_monitor.bright_high}→权重={brightness_monitor.min_weight:.1f} | 夜间<{brightness_monitor.bright_mid}→权重={brightness_monitor.max_weight:.1f}")

    print(f"[Video] 处理视频: {Path(video_path).name}")
    print(f"  分辨率: {w}x{h}, FPS: {fps:.1f}, 总帧数: {total_frames}")

    # 计算实际要处理的帧数（用于进度条总长度）
    n_to_process = (total_frames + frame_step - 1) // frame_step
    if max_frames:
        n_to_process = min(n_to_process, max_frames)

    # ===== 显示平滑：监测点深度 + 画面深度范围 EMA（去帧间抖动，不影响检测算法）=====
    _ema_depths = None          # 监测点 EMA 缓冲
    _ema_d_min = None           # 画面最小深度 EMA
    _ema_d_max = None           # 画面最大深度 EMA
    _ema_alpha = 0.35           # EMA 系数（越小越平滑，响应越慢）

    pbar = tqdm(total=n_to_process, desc=f"视频 {name[:15]}", ncols=120, leave=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=[img_rgb], return_tensors="pt").to(device)

            t0 = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                depth = outputs.predicted_depth
            frame_time = (time.time() - t0) * 1000  # 毫秒

            # 统一成 4D 再插值
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            depth = torch.nn.functional.interpolate(
                depth, size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

            depth_vis, d_min_raw, d_max_raw = depth_to_colormap(depth)

            # 计算所有像素点深度值（原始值，用于检测算法）
            pixel_depths = []
            if pixel_points:
                for px, py in pixel_points:
                    if 0 <= px < w and 0 <= py < h:
                        pixel_depths.append(float(depth[py, px]))
                    else:
                        pixel_depths.append(None)

            # ===== 仅用于显示的 EMA 平滑（不影响检测逻辑）=====
            # _ema_* 变量在外层 while True 循环前定义，与本分支同属函数作用域，无需 nonlocal
            # 监测点深度平滑
            if pixel_depths:
                if _ema_depths is None or len(_ema_depths) != len(pixel_depths):
                    _ema_depths = list(pixel_depths)
                else:
                    for i, d in enumerate(pixel_depths):
                        if d is None:
                            continue
                        if _ema_depths[i] is None:
                            _ema_depths[i] = d
                        else:
                            _ema_depths[i] = _ema_alpha * d + (1 - _ema_alpha) * _ema_depths[i]
            # 画面深度范围平滑
            if _ema_d_min is None:
                _ema_d_min, _ema_d_max = d_min_raw, d_max_raw
            else:
                _ema_d_min = _ema_alpha * d_min_raw + (1 - _ema_alpha) * _ema_d_min
                _ema_d_max = _ema_alpha * d_max_raw + (1 - _ema_alpha) * _ema_d_max
            # 显示用值（检测仍用 pixel_depths / d_min_raw）
            display_depths = list(_ema_depths) if _ema_depths else pixel_depths
            display_d_min, display_d_max = _ema_d_min, _ema_d_max

            # 车辆检测（深度 + MOG2 融合）
            car_state = None
            motion_state = None
            motion_count = 0

            # 深度检测
            if detector and pixel_depths:
                car_state = detector.update(pixel_depths)

            # MOG2 运动检测
            if motion_detector:
                motion_state = motion_detector.update(frame)
                motion_count = motion_detector.motion_pixel_count
                # 在原始帧上绘制检测带
                frame = _draw_motion_strip(frame, motion_detector, w, h)

            # 融合判定（v4: 亮度权重 + 深度变化验证）
            # 深度检测可靠：car → 直接判定有车
            # MOG2 辅助：根据当前亮度权重调整判定策略
            #   - 夜间（weight≥0.7）：MOG2确认+深度基本变化 → 有车
            #   - 黄昏（0.5≤weight<0.7）：MOG2确认+深度明显变化 → 有车
            #   - 白天（weight<0.5）：MOG2确认+深度强烈变化 → 有车
            # 深度为 no_car 时，MOG2 不覆盖（可能是反光/远处运动）
            # 融合判定（v5: 深度+MOG2双重验证）
            mog2_confirmed = motion_detector.is_confirmed() if motion_detector else False
            mog2_weight = motion_detector.get_current_weight() if motion_detector else 0.5
            depth_conf = detector.last_confidence if detector else 0.0

            # 基础深度变化检查（阈值随权重调整）
            if detector and mog2_confirmed and pixel_depths:
                depth_abs_threshold = 0.5 + (1.0 - mog2_weight) * 1.5
                depth_rel_threshold = 0.05 + (1.0 - mog2_weight) * 0.20
                mog2_depth_ok = detector.has_depth_change(
                    pixel_depths,
                    abs_threshold=depth_abs_threshold,
                    rel_threshold=depth_rel_threshold
                )
            else:
                mog2_depth_ok = False

            if car_state == "car":
                if mog2_confirmed:
                    final_state = "car"
                elif depth_conf >= 0.60:
                    final_state = "car"
                else:
                    final_state = "no_car"
            elif car_state == "no_car":
                final_state = "no_car"
            elif car_state == "unknown" and mog2_confirmed and mog2_depth_ok:
                final_state = "car"
            else:
                final_state = car_state or "unknown"

            depth_vis = _draw_car_state(
                depth_vis, final_state, w,
                motion_state, motion_count,
                brightness_level=brightness_monitor.get_brightness_level() if brightness_monitor else None,
                mog2_weight=motion_detector.get_current_weight() if motion_detector else None,
            )

            depth_vis = annotate_depth(depth_vis, display_d_min, display_d_max, h, pixel_points, display_depths if display_depths else None, frame_time)

            combined = np.hstack([frame, depth_vis])
            writer.write(combined)
            depths_list.append(depth)
            processed += 1

            # 进度条实时显示深度范围 + 监测点深度 + 车辆状态（使用显示用平滑值，避免抖动刷屏）
            postfix = f"depth={display_d_min:.1f}~{display_d_max:.1f}m"
            if display_depths:
                valid_depths = [d for d in display_depths if d is not None]
                if valid_depths:
                    postfix += " pts=[" + ",".join(f"{d:.1f}" for d in valid_depths) + "]m"
            if final_state:
                postfix += f" state={final_state}"
            if motion_state:
                postfix += f" mot={motion_state}"
            pbar.set_postfix_str(postfix)
            pbar.update(1)

            if max_frames and processed >= max_frames:
                break

        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()

    # 等待文件完全写入磁盘（避免 ffmpeg 读取时文件被占用）
    time.sleep(0.5)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 自动尝试用 ffmpeg 转码为 MP4
    mp4_path = os.path.abspath(os.path.join(output_dir, f"{name}_depth_video.mp4"))
    video_out_path_abs = os.path.abspath(video_out_path)
    ffmpeg_exe = _get_ffmpeg_path()
    if ffmpeg_exe:
        print(f"  转码 MP4 中...", end="", flush=True)
        try:
            import subprocess
            result = subprocess.run(
                [ffmpeg_exe, "-y", "-i", video_out_path_abs,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 mp4_path],
                capture_output=True, timeout=300
            )
            if result.returncode == 0 and os.path.exists(mp4_path):
                os.remove(video_out_path_abs)
                video_out_path = mp4_path
                print(f" 已转为 MP4")
            else:
                err_msg = result.stderr.decode(errors='ignore')[-200:] if result.stderr else "未知错误"
                print(f" 转码失败，保留 AVI 格式 (原因: {err_msg.strip()[:80]})")
        except Exception as e:
            print(f" 转码异常: {e}，保留 AVI 格式")
    else:
        print(f"  提示: 未找到 ffmpeg，当前为 AVI 格式（可用 VLC/完美解码 播放）")

    if save_depth and depths_list:
        print(f"  保存深度数据: {len(depths_list)} 帧...", end="", flush=True)
        np.savez_compressed(
            os.path.join(output_dir, f"{name}_depths.npz"),
            depth=np.array(depths_list)
        )
        print(f" 已保存 ({len(depths_list)} 帧)")
    elif not save_depth:
        # 释放内存
        depths_list.clear()

    # 保存亮度/权重日志
    if brightness_monitor and brightness_monitor.get_logs():
        log_path = os.path.join(output_dir, f"{name}_brightness_log.json")
        import json
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "video": str(video_path),
                "brightness_thresholds": {
                    "bright_high": brightness_monitor.bright_high,
                    "bright_mid": brightness_monitor.bright_mid,
                },
                "weight_range": {
                    "min": brightness_monitor.min_weight,
                    "max": brightness_monitor.max_weight,
                },
                "logs": brightness_monitor.get_logs(),
            }, f, ensure_ascii=False, indent=2)
        print(f"  [亮度] 日志已保存: {log_path} ({len(brightness_monitor.get_logs())} 条记录)")

    print(f"  [OK] 完成: {Path(video_out_path).name} | 共 {processed} 帧")


def _get_ffmpeg_path():
    """获取 ffmpeg 路径：优先系统 ffmpeg，其次 imageio-ffmpeg"""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_files(source):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')

    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in image_exts:
            return [("image", source)]
        elif ext in video_exts:
            return [("video", source)]
        else:
            print(f"[ERROR] 不支持的文件格式: {ext}")
            return []
    elif os.path.isdir(source):
        files = []
        # 递归扫描目录及子目录
        for dirpath, dirnames, filenames in os.walk(source):
            dirnames.sort()  # 子目录按名称排序
            for f in sorted(filenames):
                fpath = os.path.join(dirpath, f)
                ext = os.path.splitext(f)[1].lower()
                if ext in image_exts:
                    files.append(("image", fpath))
                elif ext in video_exts:
                    files.append(("video", fpath))
        return files
    else:
        print(f"[ERROR] 路径不存在: {source}")
        return []


def _load_config():
    """加载项目根目录的 config.yaml（detect.py 在子目录，向上找一级）"""
    import yaml
    root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(root) == "detect":
        root = os.path.dirname(root)
    cfg_path = os.path.join(root, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        # 将所有相对路径转为绝对路径（相对于 config.yaml 所在目录）
        cfg["_root"] = root
        return cfg
    return {"_root": root}


def _abs_path(root, path):
    """将相对路径转为绝对路径；已是绝对路径则原样返回"""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _parse_pixel_points(points_str, count, img_w, img_h):
    """解析像素点配置，返回 [(x,y), ...] 列表
    - points_str: 分号分隔的多坐标字符串，如 "320,240;640,480"
    - count: 自动生成的点数量（仅当 points_str 为空时生效）
    - img_w, img_h: 画面尺寸
    规则：
    1. points_str 非空 → 解析自定义坐标列表
    2. points_str 为空且 count=15 → 双列非对称布局（左列x=40共7个点，右列x=55共8个点）
    3. points_str 为空且 count 为偶数 → 双列对称布局（左列x=40，右列x=55）
    4. points_str 为空且 count 为奇数（≠15） → 单列布局（x=40）
    """
    points_str = str(points_str or "").strip()
    if points_str:
        # 解析自定义坐标
        points = []
        for pair in points_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            try:
                parts = pair.split(",")
                if len(parts) == 2:
                    x, y = int(parts[0].strip()), int(parts[1].strip())
                    x = max(0, min(x, img_w - 1))
                    y = max(0, min(y, img_h - 1))
                    points.append((x, y))
            except (ValueError, TypeError):
                continue
        if points:
            return points

    # 默认：双列非对称交错布局（count=15）
    n = max(1, int(count))
    if n == 15:
        # 左列x=40共7个点，右列x=55共8个点（交错分布）
        # 左列：均匀分布在整个画面高度
        left_x = max(0, min(40, img_w - 1))
        right_x = max(0, min(55, img_w - 1))
        left_ys = []
        for i in range(7):
            y = int((i + 0.5) * img_h / 7)
            y = max(0, min(y, img_h - 1))
            left_ys.append(y)

        # 右列：交错分布在左列相邻两点之间
        # 共8个点：首点在第一个左列点上方，6个在间隙中间，末点在最后一个左列点下方
        right_ys = []
        # 首点：第一个左列点上方的间隙
        first_gap = left_ys[0] - 0  # 从0到第一个左列点的距离
        right_ys.append(max(0, min(int(left_ys[0] - first_gap / 2), img_h - 1)))
        # 中间6个点：左列相邻两点的中间
        for j in range(6):
            mid_y = (left_ys[j] + left_ys[j + 1]) // 2
            right_ys.append(max(0, min(mid_y, img_h - 1)))
        # 末点：最后一个左列点下方的间隙
        last_gap = img_h - left_ys[6]
        right_ys.append(max(0, min(int(left_ys[6] + last_gap / 2), img_h - 1)))

        points = []
        for y in left_ys:
            points.append((left_x, y))
        for y in right_ys:
            points.append((right_x, y))
        return points

    # 双列对称布局（count为偶数且≥10）
    if n % 2 == 0 and n >= 10:
        per_col = n // 2
        col_xs = [max(0, min(40, img_w - 1)), max(0, min(55, img_w - 1))]
        points = []
        for col_x in col_xs:
            for i in range(per_col):
                y = int((i + 0.5) * img_h / per_col)
                y = max(0, min(y, img_h - 1))
                points.append((col_x, y))
        return points

    # 单列布局
    cx = max(0, min(40, img_w - 1))
    points = []
    for i in range(n):
        if n == 1:
            y = img_h // 2
        else:
            y = int((i + 0.5) * img_h / n)
        y = max(0, min(y, img_h - 1))
        points.append((cx, y))
    return points


def preview_image(model, processor, img_path, device, pixel_points=None):
    """预览单张图片：imshow 显示，不保存文件"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(images=[img_rgb], return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth
    frame_time = (time.time() - t0) * 1000

    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    h, w = img.shape[:2]
    depth = torch.nn.functional.interpolate(
        depth, size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()

    pixel_depths = []
    if pixel_points:
        for px, py in pixel_points:
            if 0 <= px < w and 0 <= py < h:
                pixel_depths.append(float(depth[py, px]))
            else:
                pixel_depths.append(None)

    depth_vis, d_min, d_max = depth_to_colormap(depth)
    depth_vis = annotate_depth(depth_vis, d_min, d_max, h, pixel_points, pixel_depths if pixel_depths else None, frame_time)
    combined = np.hstack([img, depth_vis])

    print(f"  [Preview] {Path(img_path).name} | {d_min:.2f}~{d_max:.2f}m | {frame_time:.0f}ms")
    print(f"  [提示] 按任意键关闭窗口")

    cv2.imshow(f"Preview - {Path(img_path).name}", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preview_video(model, processor, video_path, device,
                  frame_step=1, max_frames=None, pixel_points=None,
                  car_detector_config=None):
    """预览视频：逐帧 imshow 显示，不保存文件。按 q 退出，空格暂停"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name = Path(video_path).stem
    print(f"[Video] 预览: {Path(video_path).name}")
    print(f"  分辨率: {w}x{h}, FPS: {fps:.1f}, 总帧数: {total_frames}")
    print(f"  [提示] 按 q 退出, 空格暂停/继续")

    # 初始化深度车辆检测器
    detector = None
    if car_detector_config and pixel_points:
        effective_fps = fps / frame_step if fps > 0 and frame_step > 0 else fps
        detector = VehicleDetector(
            n_points=len(pixel_points),
            distance_drop_threshold=car_detector_config["distance_drop_threshold"],
            normal_fluctuation=car_detector_config["normal_fluctuation"],
            fps=effective_fps,
            baseline_warmup=car_detector_config["baseline_warmup"],
            baseline_alpha=car_detector_config["baseline_alpha"],
            min_car_points=car_detector_config["min_car_points"],
            min_stable_points=car_detector_config["min_stable_points"],
            car_confirm_frames=car_detector_config["car_confirm_frames"],
            stable_confirm_frames=car_detector_config["stable_confirm_frames"],
            max_detect_depth=car_detector_config.get("max_detect_depth", 30.0),
            relative_drop_threshold=car_detector_config.get("relative_drop_threshold", 0.3),
        )
        print(f"  [Car] 深度车辆检测已启用")

    # 初始化 MOG2 运动检测器
    motion_detector = None
    mog2_config = car_detector_config.get("mog2", {}) if car_detector_config else {}
    mog2_enabled = mog2_config.get("enabled", True) if mog2_config else True

    # 初始化亮度监听器（依赖 mog2_config，需在之后）
    brightness_monitor = None
    if mog2_enabled and pixel_points:
        mog2_bright_cfg = mog2_config.get("brightness", {})
        brightness_monitor = BrightnessMonitor(
            bright_high=mog2_bright_cfg.get("bright_high", 180.0),
            bright_mid=mog2_bright_cfg.get("bright_mid", 80.0),
            min_weight=mog2_bright_cfg.get("min_weight", 0.4),
            max_weight=mog2_bright_cfg.get("max_weight", 1.0),
            weight_alpha=mog2_bright_cfg.get("weight_alpha", 0.05),
            log_interval=mog2_bright_cfg.get("log_interval", 50),
        )

    if mog2_enabled and pixel_points:
        detect_x = pixel_points[0][0] if pixel_points else 40
        motion_detector = MotionLineDetector(
            detect_x=detect_x,
            strip_width=mog2_config.get("strip_width", 30),
            min_motion_pixels=mog2_config.get("min_motion_pixels", 50),
            history=mog2_config.get("history", 300),
            var_threshold=mog2_config.get("var_threshold", 25),
            detect_shadow=mog2_config.get("detect_shadow", False),
            warmup_frames=mog2_config.get("warmup_frames", 30),
            confirm_frames=mog2_config.get("confirm_frames", 3),
            brightness_monitor=brightness_monitor,
        )
        print(f"  [MOG2] 运动检测已启用 | 检测线 x={detect_x}, 带宽={mog2_config.get('strip_width', 30)}px, 确认={mog2_config.get('confirm_frames', 3)}帧")
        if brightness_monitor:
            print(f"  [亮度] 动态权重已启用 | 白天≥{brightness_monitor.bright_high}→权重={brightness_monitor.min_weight:.1f} | 夜间<{brightness_monitor.bright_mid}→权重={brightness_monitor.max_weight:.1f}")

    # 缩放比例（如果画面太大，缩放到合适尺寸显示）
    max_display_w = 1920
    scale = 1.0
    if w * 2 > max_display_w:
        scale = max_display_w / (w * 2)

    frame_idx = 0
    processed = 0
    paused = False

    n_to_process = (total_frames + frame_step - 1) // frame_step
    if max_frames:
        n_to_process = min(n_to_process, max_frames)

    pbar = tqdm(total=n_to_process, desc=f"预览 {name[:15]}", ncols=120, leave=True)
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = processor(images=[img_rgb], return_tensors="pt").to(device)

                t0 = time.time()
                with torch.no_grad():
                    outputs = model(**inputs)
                    depth = outputs.predicted_depth
                frame_time = (time.time() - t0) * 1000

                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)
                depth = torch.nn.functional.interpolate(
                    depth, size=(h, w), mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()

                depth_vis, d_min, d_max = depth_to_colormap(depth)

                pixel_depths = []
                if pixel_points:
                    for px, py in pixel_points:
                        if 0 <= px < w and 0 <= py < h:
                            pixel_depths.append(float(depth[py, px]))
                        else:
                            pixel_depths.append(None)

                # 车辆检测（深度 + MOG2 融合）
                car_state = None
                motion_state = None
                motion_count = 0

                if detector and pixel_depths:
                    car_state = detector.update(pixel_depths)

                if motion_detector:
                    motion_state = motion_detector.update(frame)
                    motion_count = motion_detector.motion_pixel_count
                    frame = _draw_motion_strip(frame, motion_detector, w, h)

                # 融合判定（v5: 深度+MOG2双重验证）
                mog2_confirmed = motion_detector.is_confirmed() if motion_detector else False
                mog2_weight = motion_detector.get_current_weight() if motion_detector else 0.5
                depth_conf = detector.last_confidence if detector else 0.0

                if detector and mog2_confirmed and pixel_depths:
                    depth_abs_threshold = 0.5 + (1.0 - mog2_weight) * 1.5
                    depth_rel_threshold = 0.05 + (1.0 - mog2_weight) * 0.20
                    mog2_depth_ok = detector.has_depth_change(
                        pixel_depths,
                        abs_threshold=depth_abs_threshold,
                        rel_threshold=depth_rel_threshold
                    )
                else:
                    mog2_depth_ok = False

                if car_state == "car":
                    if mog2_confirmed:
                        final_state = "car"
                    elif depth_conf >= 0.60:
                        final_state = "car"
                    else:
                        final_state = "no_car"
                elif car_state == "no_car":
                    final_state = "no_car"
                elif car_state == "unknown" and mog2_confirmed and mog2_depth_ok:
                    final_state = "car"
                else:
                    final_state = car_state or "unknown"

                depth_vis = _draw_car_state(
                depth_vis, final_state, w,
                motion_state, motion_count,
                brightness_level=brightness_monitor.get_brightness_level() if brightness_monitor else None,
                mog2_weight=motion_detector.get_current_weight() if motion_detector else None,
            )

                depth_vis = annotate_depth(depth_vis, d_min, d_max, h, pixel_points, pixel_depths if pixel_depths else None, frame_time)
                combined = np.hstack([frame, depth_vis])

                # 缩放显示
                if scale < 1.0:
                    disp = cv2.resize(combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale)))
                else:
                    disp = combined

                cv2.imshow(f"Preview - {Path(video_path).name} (q=退出, 空格=暂停)", disp)
                processed += 1

                postfix = f"depth={d_min:.1f}~{d_max:.1f}m time={frame_time:.0f}ms"
                if pixel_depths:
                    valid_depths = [d for d in pixel_depths if d is not None]
                    if valid_depths:
                        postfix += " pts=[" + ",".join(f"{d:.1f}" for d in valid_depths) + "]m"
                if final_state:
                    postfix += f" state={final_state}"
                if motion_state:
                    postfix += f" mot={motion_state}"
                pbar.set_postfix_str(postfix)
                pbar.update(1)

                if max_frames and processed >= max_frames:
                    break

            frame_idx += 1

        # 检测按键
        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key == ord('q'):
            print("\n  [用户退出预览]")
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("\n  [已暂停，按空格继续]")
            else:
                print("  [继续播放]")

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"  [OK] 预览完成: 共 {processed} 帧")


def main():
    cfg = _load_config()
    root = cfg.get("_root", os.getcwd())
    paths = cfg.get("paths", {})
    detect_cfg = cfg.get("detect", {})
    car_cfg = cfg.get("car_detect", {})

    parser = argparse.ArgumentParser(description="DA2-Metric 模型验证与推理 (输出米数)")
    parser.add_argument("--source", type=str,
                        default=_abs_path(root, paths.get("detect_input", "detect/input")),
                        help="输入源: 图片/视频文件或目录路径")
    parser.add_argument("--model-depth", type=str,
                        default=_abs_path(root, detect_cfg.get("model_depth", DEFAULT_MODEL)),
                        help="模型路径 (预训练模型或微调 checkpoint)")
    parser.add_argument("--output", type=str,
                        default=_abs_path(root, paths.get("detect_output", "detect/output")),
                        help="输出目录")
    parser.add_argument("--device", type=str,
                        default=detect_cfg.get("device", "cuda"))
    parser.add_argument("--frame-step", type=int,
                        default=detect_cfg.get("frame_step", 1),
                        help="视频帧采样间隔 (1=每帧都处理)")
    parser.add_argument("--max-frames", type=int,
                        default=detect_cfg.get("max_frames"),
                        help="视频最大处理帧数")
    parser.add_argument("--save-depth", type=int, choices=[0, 1],
                        default=int(detect_cfg.get("save_depth", 0)),
                        help="是否保存每帧深度 .npz (1=保存, 0=不保存)")
    parser.add_argument("--preview-enabled", type=int, choices=[0, 1],
                        default=int(detect_cfg.get("preview_enabled", 1)),
                        help="是否启用预览模式 (1=imshow显示不保存, 0=保存到磁盘)")
    parser.add_argument("--preview-count", type=int,
                        default=int(detect_cfg.get("preview_count", 3)),
                        help="预览模式下随机选取的文件数量 (0=全部)")
    parser.add_argument("--pixel-depth-enabled", type=int, choices=[0, 1],
                        default=int(detect_cfg.get("pixel_depth_enabled", 1)),
                        help="是否启用像素点深度监测 (1=启用, 0=关闭)")
    parser.add_argument("--pixel-depth-points", type=str,
                        default=str(detect_cfg.get("pixel_depth_points", "") or ""),
                        help="监测像素点列表 (格式: x1,y1;x2,y2;... 或留空使用自动分布)")
    parser.add_argument("--pixel-depth-count", type=int,
                        default=int(detect_cfg.get("pixel_depth_count", 5)),
                        help="自动分布的监测点数量 (仅当未指定自定义坐标时生效)")
    # 车辆检测参数
    parser.add_argument("--car-detect-enabled", type=int, choices=[0, 1],
                        default=int(car_cfg.get("car_detect_enabled", 1)),
                        help="是否启用车辆检测 (1=启用, 0=关闭)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.preview_enabled:
        os.makedirs(args.output, exist_ok=True)

    model, processor = load_model(args.model_depth, device)

    files = get_files(args.source)
    if not files:
        print("[ERROR] 没有找到可处理的文件")
        sys.exit(1)

    # 预览模式：随机选取文件
    preview_enabled = bool(args.preview_enabled)
    preview_count = args.preview_count
    if preview_enabled and preview_count > 0 and len(files) > preview_count:
        random.shuffle(files)
        selected_files = files[:preview_count]
        print(f"[Preview] 预览模式: 从 {len(files)} 个文件中随机选取 {preview_count} 个")
        print(f"[Preview] 选取文件: {[Path(f[1]).name for f in selected_files]}")
        files = selected_files
    elif preview_enabled:
        print(f"[Preview] 预览模式: 处理全部 {len(files)} 个文件")

    # 读取像素点深度监测配置（仅记录开关，坐标在循环内按每个文件实际尺寸生成）
    pixel_depth_enabled = bool(args.pixel_depth_enabled)
    pixel_depth_points_str = args.pixel_depth_points if args.pixel_depth_points else ""
    pixel_depth_count = args.pixel_depth_count
    if pixel_depth_enabled:
        reason = "自定义坐标" if pixel_depth_points_str else f"中分线自动分布(每文件按实际尺寸生成{pixel_depth_count}点)"
        print(f"[Pixel] 像素点深度监测已启用: {reason}")
    else:
        print("[Pixel] 像素点深度监测已关闭")

    # 车辆检测配置（深度 + MOG2）
    car_detect_enabled = bool(args.car_detect_enabled) and pixel_depth_enabled
    car_detector_config = None
    if car_detect_enabled:
        mog2_cfg = car_cfg.get("mog2", {})
        car_detector_config = {
            "distance_drop_threshold": float(car_cfg.get("distance_drop_threshold", 2.0)),
            "normal_fluctuation": float(car_cfg.get("normal_fluctuation", 1.5)),
            "min_car_points": int(car_cfg.get("min_car_points", 1)),
            "min_stable_points": int(car_cfg.get("min_stable_points", 3)),
            "car_confirm_frames": int(car_cfg.get("car_confirm_frames", 1)),
            "stable_confirm_frames": int(car_cfg.get("stable_confirm_frames", 3)),
            "baseline_warmup": int(car_cfg.get("baseline_warmup", 5)),
            "baseline_alpha": float(car_cfg.get("baseline_alpha", 0.05)),
            "max_detect_depth": float(car_cfg.get("max_detect_depth", 30.0)),
            "relative_drop_threshold": float(car_cfg.get("relative_drop_threshold", 0.3)),
            "mog2": {
                "enabled": bool(mog2_cfg.get("enabled", True)),
                "strip_width": int(mog2_cfg.get("strip_width", 30)),
                "min_motion_pixels": int(mog2_cfg.get("min_motion_pixels", 50)),
                "history": int(mog2_cfg.get("history", 300)),
                "var_threshold": int(mog2_cfg.get("var_threshold", 25)),
                "detect_shadow": bool(mog2_cfg.get("detect_shadow", False)),
                "warmup_frames": int(mog2_cfg.get("warmup_frames", 30)),
                "confirm_frames": int(mog2_cfg.get("confirm_frames", 3)),
                "brightness": dict(mog2_cfg.get("brightness", {})),
            },
        }
        mog2_status = "深度+MOG2" if mog2_cfg.get("enabled", True) else "仅深度"
        print(f"[Car] 车辆检测已启用 ({mog2_status}) | 骤减阈值={car_detector_config['distance_drop_threshold']}m | 波动阈值={car_detector_config['normal_fluctuation']}m")
    else:
        print("[Car] 车辆检测未启用")

    print(f"\n[Detect] 待处理文件: {len(files)} 个 | 模式: {'预览(imshow)' if preview_enabled else '保存到磁盘'}\n")

    def _gen_points_for_file(ftype, fpath):
        """根据该文件实际尺寸生成监测点（保证不同分辨率视频点位正确）"""
        if not pixel_depth_enabled:
            return None
        if ftype == "video":
            cap_test = cv2.VideoCapture(fpath)
            w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap_test.isOpened() else 0
            h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap_test.isOpened() else 0
            cap_test.release()
        else:
            img = cv2.imread(fpath)
            if img is None:
                return None
            h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return None
        pts = _parse_pixel_points(pixel_depth_points_str, pixel_depth_count, w, h)
        if pts:
            coords_str = ", ".join(f"({x},{y})" for x, y in pts)
            print(f"[Pixel] {Path(fpath).name} ({w}x{h}): {coords_str}")
        return pts

    for file_type, file_path in files:
        print(f"\n{'=' * 60}")
        # 每个文件单独生成监测点（按该文件实际分辨率）
        file_pixel_points = _gen_points_for_file(file_type, file_path)
        if preview_enabled:
            # 预览模式：imshow 显示，不保存
            if file_type == "image":
                preview_image(model, processor, file_path, device, pixel_points=file_pixel_points)
            elif file_type == "video":
                preview_video(model, processor, file_path, device,
                              args.frame_step, args.max_frames, pixel_points=file_pixel_points,
                              car_detector_config=car_detector_config)
        else:
            # 保存模式：保存到磁盘
            if file_type == "image":
                process_image(model, processor, file_path, args.output, device,
                             pixel_points=file_pixel_points)
            elif file_type == "video":
                process_video(model, processor, file_path, args.output, device,
                              args.frame_step, args.max_frames, bool(args.save_depth),
                              pixel_points=file_pixel_points,
                              car_detector_config=car_detector_config)

    print(f"\n{'=' * 60}")
    if preview_enabled:
        print(f"预览完成！")
    else:
        print(f"所有文件处理完成！")
        print(f"输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
