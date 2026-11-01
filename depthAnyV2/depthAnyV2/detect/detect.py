"""
DA2 度量深度的验证/推理脚本，出来的就是真实米数。
图片、视频、整个目录都能丢进来。

跑法：
  python detect.py --source input/ --model-depth chqeckpoints/my_model/best_model.pt

出来后：
  - 单图：原图+深度的对比图、深度彩色图，还有米数的 .npy
  - 视频：左右对比视频 + 深度 .npz
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


# 车辆检测器：核心判定逻辑，跟 detect_car.py 保持一致。
# 全局 YOLO 模型缓存：多个视频/多个 detector 共用同一份权重，
# 不然每个视频都要懒加载一次、打印还会打断 tqdm。
# 背景图片/视频扩展名
_BG_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_BG_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".asf")


def load_baseline_depths(baseline_dir, model, processor, device, pixel_points, max_frames=30):
    """
    把背景目录里的图片和视频都翻一遍，抽监测点的深度，用来预设基线。
    baseline_dir 是背景目录；model/processor 是 DA2 深度模型；device 是算的设备；
    pixel_points 是监测点 [(x,y), ...]；max_frames 是每个背景视频最多抽几帧（图片不限）。
    返回每帧背景的 pixel_depths 列表。
    """
    if not baseline_dir or not os.path.isdir(baseline_dir):
        return []
    image_files = []
    video_files = []
    for dp, _, files in os.walk(baseline_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(dp, f)
            if ext in _BG_IMAGE_EXTS:
                image_files.append(full)
            elif ext in _BG_VIDEO_EXTS:
                video_files.append(full)
    if not image_files and not video_files:
        print(f"[Baseline] 背景目录无图片/视频: {baseline_dir}")
        return []

    depths_list = []

    # 图片：每张跑一次深度
    for ip in image_files:
        img = cv2.imread(ip)
        if img is None:
            continue
        depths = _infer_pixel_depths(model, processor, img, device, pixel_points)
        if depths is not None:
            depths_list.append(depths)

    # 视频：等间隔抽帧
    for vp in video_files:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        step = max(1, total // max_frames)
        frame_idx = 0
        got = 0
        while got < max_frames and frame_idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            depths = _infer_pixel_depths(model, processor, frame, device, pixel_points)
            if depths is not None:
                depths_list.append(depths)
                got += 1
            frame_idx += step
        cap.release()

    print(f"[Baseline] 从 {len(image_files)} 张图片 + {len(video_files)} 个视频 "
          f"提取 {len(depths_list)} 帧背景深度")
    return depths_list


def apply_clahe(frame_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """夜里去眩光、提亮暗部：到 Lab 空间只对 L 通道做 CLAHE。
    返回增强后的 BGR 帧。深度模型和 YOLO 得吃同一张增强帧，判定口径才对得上。
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _frame_mean_brightness(frame_bgr, target_w=160.0, target_h=90.0, independent_scale=False):
    """先把帧降采样再算平均灰度(0~255)，这样快。

    - independent_scale=False：x/y 用同一个缩放系数（VehicleDetector 夜间判定的老做法）
    - independent_scale=True：x/y 各自缩放（BrightnessMonitor 的老做法，用 sample_width/height）
    这两处调用都跟重构前的数值结果一模一样，没差。
    """
    h, w = frame_bgr.shape[:2]
    if independent_scale:
        sx = min(1.0, target_w / w)
        sy = min(1.0, target_h / h)
        need_resize = (sx < 1.0 or sy < 1.0)
        fx, fy = sx, sy
    else:
        scale = min(1.0, target_w / w, target_h / h)
        need_resize = scale < 1.0
        fx = fy = scale
    if need_resize:
        small = cv2.resize(frame_bgr, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _infer_depth_map(model, processor, frame_bgr, device):
    """单帧跑一次深度推理，返回 (depth_np, frame_time_ms)。
    跟以前各流水线里内联的那段推断块语义一样（连计时都算），出异常照样往上抛。"""
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
    return depth, frame_time


def _extract_pixel_depths(depth, pixel_points, frame_bgr=None, sat_threshold=0):
    """从深度图里把监测点的深度抠出来。点越界就返回 nan。
    - frame_bgr / sat_threshold 是车灯过曝点的屏蔽（P0-2）：某个像素最大通道 ≥ sat_threshold 就当过曝，
      这块深度不可信 → 返回 nan。sat_threshold<=0 就不屏蔽（跟原来一样）。
    """
    h, w = depth.shape[:2]
    out = []
    for px, py in pixel_points:
        if 0 <= px < w and 0 <= py < h:
            if sat_threshold > 0 and frame_bgr is not None:
                b = int(frame_bgr[py, px, 0])
                g = int(frame_bgr[py, px, 1])
                r = int(frame_bgr[py, px, 2])
                if max(b, g, r) >= sat_threshold:
                    out.append(float("nan"))
                    continue
            out.append(float(depth[py, px]))
        else:
            out.append(float("nan"))
    return out


def _infer_pixel_depths(model, processor, frame_bgr, device, pixel_points):
    """对单帧图像跑深度推理，返回 pixel_points 对应的深度列表；失败返回 None"""
    try:
        depth, _ = _infer_depth_map(model, processor, frame_bgr, device)
        return _extract_pixel_depths(depth, pixel_points)
    except Exception:
        return None


_YOLO_CACHE: dict = {}  # key=(model_path_norm, device_norm) -> yolo_model_obj

class VehicleDetector:
    """
    靠检测线上多点深度的变化来判有没有车（v8：非对称双列 + 全点检测）。

    几点关键的：
    - 支持非对称双列检测线（x=40 摆 7 个点，x=55 摆 8 个点，一共 15 个）
    - 点数是奇数（比如 15）就走全点模式，每个点各算各的
    - 点数是偶数且 ≥10 就走双列对称聚合
    - 预热时前 3 帧先塞进 buffer，第 4 帧起拿 buffer 中位数看异常
    - 异常帧（有车）不进 buffer，免得把基线搞脏
    - 基线用中位数不用均值，扛得住深度模型抖动
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
                 # YOLO 视觉检测参数
                 yolo_enabled=False,
                 yolo_model_path="yolov8n.pt",
                 yolo_device="",
                 yolo_confidence=0.40,
                 yolo_iou=0.45,
                 yolo_vehicle_classes=(1, 2, 3, 5, 7),
                 yolo_enter_bottom_ratio=0.30,
                 yolo_min_box_ratio=0.02,
                 yolo_detect_x=40,
                 yolo_cross_tolerance=15,
                 # 多规则融合权重，R1+R2+R3+R4+R5 = 1.00
                 rule_weights=None,
                 car_confidence_threshold=0.470,
                 # 夜间自适应，专门对付车头光源干扰
                 night_bright_threshold=80.0,
                 night_smooth_alpha=0.35,
                 night_confirm_extra=2,
                 night_cooldown_extra=20,
                 # 一车一信号相关
                 min_car_hold_frames=10,
                 min_signal_interval_frames=15,
                 spike_abs_threshold=8.0,
                 car_exit_cooldown_frames=15,
                 baseline_rescue_frames=30,
                 baseline_up_snap_frames=10,
                 baseline_down_snap_frames=10):
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

        # 多规则权重：深度变化 70% + 视觉 30%
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

        # YOLO 视觉检测：延迟加载，禁用或者模型不在就静默降级，不崩
        self.yolo_enabled = yolo_enabled
        self.yolo_model_path = yolo_model_path
        self.yolo_device = yolo_device
        self.yolo_confidence = yolo_confidence
        self.yolo_iou = yolo_iou
        self.yolo_vehicle_classes = tuple(int(c) for c in yolo_vehicle_classes)
        self.yolo_enter_bottom_ratio = yolo_enter_bottom_ratio
        self.yolo_min_box_ratio = yolo_min_box_ratio
        # 检测线 x 坐标与过线容差(px)：YOLO 框左沿必须到达 detect_x + 容差 才计入视觉分
        # 防止车还没到检测线（只是从右侧靠近）就触发 R5 误报
        self.yolo_detect_x = int(yolo_detect_x)
        self.yolo_cross_tolerance = float(yolo_cross_tolerance)
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
        # 车辆离开冷却期：car→no_car 转换后，提高新 car 触发门槛
        # 防止两车间隔阶段噪声波动误触发 car
        self._car_exit_cooldown = 0  # 剩余冷却帧数
        # 冷却期总帧数（默认40≈1.33s@30fps；微波替代场景建议15≈0.5s，跟车间隔短才分得开）
        self._cooldown_frames = int(car_exit_cooldown_frames)

        # 夜间自适应状态
        self.night_bright_threshold = float(night_bright_threshold)
        self.night_smooth_alpha = float(night_smooth_alpha)
        self.night_confirm_extra = int(night_confirm_extra)
        self.night_cooldown_extra = int(night_cooldown_extra)
        self._ema_road_depths = None   # 夜间深度 EMA 平滑缓存
        self._is_night = False          # 当前帧是否夜间

        # 一车一信号三件套：
        # 状态锁——确认 car 后锁 min_car_hold_frames 帧，锁内不做回退判定，防"刚确认就闪退"
        self.min_car_hold_frames = int(min_car_hold_frames)
        self._car_hold_remaining = 0
        # 信号层——只在非car→car上升沿发一次信号，彻底回no_car且冷却结束才解锁下一次
        self.min_signal_interval_frames = int(min_signal_interval_frames)
        self.signal_count = 0
        self._signal_armed = True
        self._last_signal_update = -10 ** 9
        self._update_n = 0
        # 孤立尖峰过滤——单点比上一帧跳变超过该米数且其他点没联动，就当噪声丢掉；0=关闭
        self.spike_abs_threshold = float(spike_abs_threshold)
        self._prev_valid_depths = None
        # 基线救援——视频开头就有车占线、预热一直收不到无车帧导致基线建不起来时，
        # 用"逐点历史最大深度"兜底当基线（路面=该点能看到的最远距离，遮挡只会更近）；0=关闭
        self.baseline_rescue_frames = int(baseline_rescue_frames)
        self._road_max = None
        # 基线上浮校正——深度只会被遮挡压小，不可能比真实路面更远；
        # 某点连续 N 帧高于基线就说明基线是错的（被占线车污染），抬回观测值；0=关闭
        self.baseline_up_snap_frames = int(baseline_up_snap_frames)
        self._above_baseline_cnt = None
        # 基线下沉校正——噪声/漂移会让基线比真实路面偏高零点几米（8%相对下降在10m处才0.8m，
        # 与噪声同量级），这种"亚米级假下降"会在空路面持续撑着 R1/R3 分数、把状态机锁死在 car
        # （两车共用一个信号的根因之一）。真车遮挡是米级下降（>band）；
        # obs 持续低于基线 (0.3, band] 米 → 基线噪声偏高，压回观测值；0=关闭
        self.baseline_down_snap_frames = int(baseline_down_snap_frames)
        self._below_baseline_cnt = None

    # YOLO 视觉检测：延迟加载 + 静默降级 + 判定车有没有进检测区域
    def _ensure_yolo_loaded(self):
        """
        YOLO 模型延迟加载，第一次用到才加载；失败就记下错误并永久关掉，不影响深度检测。
        全局缓存：同一个 (model_path, device) 只真加载一次，后面实例共用，不重复加载/打印。
        返回 True=能用，False=用不了（没开或加载失败）。
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

            # 定设备：优先用 yolo_device 参数，没填就 cuda/cpu 自己判断
            device = self.yolo_device
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # 兼容路径格式（Windows/Posix），并支持 ultralytics 官方模型名（如 yolov8n.pt）
            model_path = self.yolo_model_path
            if (isinstance(model_path, str) and model_path.endswith(".pt")
                    and not os.path.isabs(model_path) and not os.path.exists(model_path)):
                # 相对路径就试着拼项目根目录
                candidate = os.path.join(os.getcwd(), model_path)
                if os.path.exists(candidate):
                    model_path = candidate
                # 找不到就交给 ultralytics 自动下官方权重

            # 全局缓存命中
            cache_key = (os.path.normcase(os.path.abspath(model_path))
                         if isinstance(model_path, str) and os.path.isabs(model_path) else str(model_path),
                         str(device).lower())
            if cache_key in _YOLO_CACHE:
                self._yolo_model = _YOLO_CACHE[cache_key]
                return True  # 命中时静默，不再打印

            # 第一次加载
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
        单帧跑一遍 YOLO 车辆检测，看车有没有进检测区域。

        视觉只是协同，不单独触发：
        1. 只认配置里的 vehicle_classes（car/truck/bus/motorcycle/bicycle）
        2. 只数画面下方 enter_bottom_ratio 区域里的框（说明车已经到检测线附近）
        3. 太小的框 min_box_ratio 直接忽略（那是远处背景车的误检）

        frame_bgr 是 BGR 原始画面（cv2.imread/cap.read 出来的格式）。
        返回 vision_score [0..1] 是视觉+进区的综合分，
        boxes 是命中的框 [(x1,y1,x2,y2,conf,cls)]。
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
            # YOLO 预测，BGR 直接喂，ultralytics 内部自己处理
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
                    # 过线过滤：车辆从画面右侧驶来，框左沿 x1 必须到达检测线
                    # （x1 <= detect_x + 容差），车只是在右侧靠近/行驶时不计入视觉分，
                    # 避免"车还没到检测线就报 car"。容差用于吸收检测框抖动。
                    if x1 > self.yolo_detect_x + self.yolo_cross_tolerance:
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
        双列聚合：同高度两列取 min，合成一个等效的单数组。

        双列模式下 depths 是 [col0_p0...col0_p6, col1_p0...col1_p6]，
        聚合后变成 [min(p0,p0'), min(p1,p1'), ..., min(p6,p6')]。
        """
        if not self.is_dual_col:
            return depths
        d = np.array(depths, dtype=np.float32)
        col0 = d[:self.col_size]
        col1 = d[self.col_size:2 * self.col_size]
        # 取两列中更浅的深度（车辆覆盖时深度更近）
        aggregated = np.minimum(col0, col1)
        return aggregated

    @staticmethod
    def _compute_frame_brightness(frame_bgr):
        """降采样计算帧平均灰度(0~255)，用于夜间判定。"""
        return _frame_mean_brightness(frame_bgr, 160.0, 90.0, independent_scale=False)

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

    def preset_baseline(self, depths_list):
        """
        拿外部背景图/视频的深度点当基线，这样能跳过视频预热那一步。

        depths_list 每项是一帧背景的 pixel_depths（长度 = n_points，里面有 NaN 就是无效点）。
        返回 bool，看基线设成没成。
        """
        if not depths_list:
            return False
        road_frames = []
        for depths in depths_list:
            road = self._get_road_depths(depths)
            road_arr = np.asarray(road, dtype=np.float32)
            # 用中位数过滤 NaN：每点取所有帧的中位数，NaN 不参与
            road_frames.append(road_arr)
        if not road_frames:
            return False
        buf = np.stack(road_frames, axis=0)  # (N, n_road_points)
        # 沿帧轴取中位数，忽略 NaN
        with np.errstate(all="ignore"):
            baseline = np.nanmedian(buf, axis=0).astype(np.float32)
        # 剩余 NaN 用 0 填充（无效点）
        baseline = np.where(np.isnan(baseline), 0.0, baseline)
        if baseline.size == 0 or np.all(baseline == 0):
            print(f"[Baseline] 背景深度全部无效，预设失败")
            return False
        self.baseline = baseline
        self.current_state = "no_car"
        self.pending_state = "no_car"
        self.confirm_count = 0
        self._car_exit_cooldown = 0
        self.warmup_buffer = []
        # 一车一信号相关状态也一起复位
        self._car_hold_remaining = 0
        self.signal_count = 0
        self._signal_armed = True
        self._last_signal_update = -10 ** 9
        self._prev_valid_depths = None
        self._road_max = None
        self._above_baseline_cnt = None
        self._below_baseline_cnt = None
        print(f"[Baseline] 已用 {len(depths_list)} 帧背景预设基线 "
              f"({self.n_road_points} 点, 中位深度 {np.median(baseline[baseline>0]):.1f}m)")
        return True

    def _analyze_depth_pattern(self, road_depths, valid_mask):
        """看深度分布像不像有车，重点是有没有双峰。

        有车：深度是双峰（近的一簇=路面+车压住，远的一簇=车顶/远端），
        而且近远两簇中间有明显空隙（车的高度撑出来的）。
        没车（普通场景）：深度一路渐变，没有明确的分隔。
        返回 depth_span_ok（是不是车辆那种跨度模式）和 span_valid_depths（有效深度值）。
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
        加权置信度判车，不再一票否决，靠多条规则一起打分。

        四条规则：
        - R1 [15%]：Way4 深度骤变 —— depth_span_ok 再加基线对比（双峰+下降）
        - R2 [30%]：Way5 绝对深度 —— 深背景/双峰/全覆盖 + 簇要紧凑
        - R3 [30%]：Way1 基线对比 —— 独立的基线下降（分布特征/比例门控兜着）
        - R4 [25%]：车辆区域统计 —— car_depths(5-15m) 紧凑度 + shallow 点 + 深背景上下文

        每条规则出 [0..1] 的分，加权求和 ≥0.5 就算有车。
        阈值在 self.car_confidence_threshold 上调。
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

        # 先把基线对比算出来，R1/R3 共用
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

        # R1: Way4 深度骤变
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

        # R2: Way5 绝对深度
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

        # R3: Way1 基线对比
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

        # R4: 车辆区域统计特征
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

        # 加权求和：深度变化 + 视觉协同，自适应切换
        # 分两种模式：
        # A. 纯深度（YOLO 禁用或 R5=0 没视觉信号）→ 用写死的老权重 R1=0.15/R2=0.30/R3=0.30/R4=0.25，
        #    这样老的 25 个用例判定边界一点不变（100% 兼容）
        # B. 视觉协同（YOLO 启用且 R5>0）→ 用配置里的 R1~R5，默认深度变化 70% + 视觉 30%
        weights = self.rule_weights
        r5_score = float(getattr(self, 'last_vision_score', 0.0))
        yolo_active = bool(getattr(self, 'yolo_enabled', False))

        if (not yolo_active) or r5_score <= 1e-9:
            # 模式 A：纯深度，跟老判定 100% 一样
            w1, w2, w3, w4, w5_used = 0.15, 0.30, 0.30, 0.25, 0.0
        else:
            # 模式 B：深度 + 视觉协同
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

        # 已经是 CAR 状态时，YOLO 视觉也还看到车，就把置信度抬一把（防闪烁）
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

        # 距离门控（Vehicle Proximity Gate）
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
        更新车辆状态（带 YOLO 视觉协同）。

        视觉不单独触发，而是当成 R5 规则加权进综合判定：
        - 只传 depths：走纯深度检测（向后兼容，YOLO 不参与，R5=0）
        - 传 depths + frame_bgr：先跑 YOLO 拿到 R5，再跑深度拿 R1-R4，最后加权综合
        """
        self._update_n += 1
        # Step A: 若传了画面，先跑 YOLO 视觉检测（R5 评分写入 self.last_vision_score）
        if self.yolo_enabled and frame_bgr is not None:
            self.yolo_detect_frame(frame_bgr)
        else:
            # 无画面或禁用 YOLO 时，视觉分清零，不干扰纯深度路径
            self.last_vision_score = 0.0
            self._last_yolo_boxes = []

        # P0 夜间自适应：判断亮度和有效门槛
        self._is_night = False
        if frame_bgr is not None and (
                self.night_smooth_alpha > 0 or self.night_confirm_extra > 0
                or self.night_cooldown_extra > 0):
            self._is_night = (
                self._compute_frame_brightness(frame_bgr)
                < self.night_bright_threshold
            )
        is_night = self._is_night
        self._eff_confirm = (
            self.car_confirm_frames + (self.night_confirm_extra if is_night else 0)
        )
        self._eff_cooldown = (
            self._cooldown_frames + (self.night_cooldown_extra if is_night else 0)
        )

        depths = np.array(depths, dtype=np.float32)
        valid_mask = ~np.isnan(depths) & (depths > 0)
        depths_clean = np.where(valid_mask, depths, 0.0)

        # 孤立尖峰过滤：单点比上一帧跳变超过 spike_abs_threshold 且全场跳变点占比 <30%，
        # 就当是深度模型抽风的噪声点，这帧作废；真车进场是整列点联动，不会误杀
        if self.spike_abs_threshold > 0:
            prev = self._prev_valid_depths
            if prev is not None and prev.shape == depths_clean.shape:
                both = valid_mask & (prev > 0)
                n_both = int(both.sum())
                if n_both > 0:
                    jumped = both & (np.abs(depths_clean - prev) > self.spike_abs_threshold)
                    n_jumped = int(jumped.sum())
                    if 0 < n_jumped < max(1, int(n_both * 0.3)):
                        valid_mask = valid_mask & ~jumped
                        depths_clean = np.where(valid_mask, depths_clean, 0.0)
            self._prev_valid_depths = np.where(
                valid_mask, depths_clean,
                self._prev_valid_depths if self._prev_valid_depths is not None else 0.0)
        else:
            self._prev_valid_depths = None

        road_depths = self._get_road_depths(depths_clean)
        road_valid = self._get_valid_road_mask(depths_clean)

        # P0-1 夜间时序平滑：把车灯扫过引起的深度尖峰压下去
        if is_night and self.night_smooth_alpha > 0:
            a = self.night_smooth_alpha
            rd = road_depths.astype(np.float32)
            if self._ema_road_depths is None or self._ema_road_depths.shape != rd.shape:
                self._ema_road_depths = rd.copy()
            else:
                blended = self._ema_road_depths * (1.0 - a) + rd * a
                self._ema_road_depths = np.where(road_valid, blended, self._ema_road_depths)
            road_depths = np.where(road_valid, self._ema_road_depths, 0.0).astype(np.float32)
        else:
            self._ema_road_depths = None

        # 基线救援：逐点追踪历史最大深度（在尖峰过滤/夜间平滑之后，污染已被压过）。
        # 基线迟迟建不起来（视频开头车就占线，预热buffer收不到无车帧）时兜底，
        # 否则"回no_car要对比基线"永远不满足，状态机会永久卡死在car（两车共用一个信号）
        if self.baseline_rescue_frames > 0:
            rd_valid_vals = np.where(road_valid, road_depths, -np.inf)
            if self._road_max is None or self._road_max.shape != rd_valid_vals.shape:
                self._road_max = rd_valid_vals
            else:
                self._road_max = np.maximum(self._road_max, rd_valid_vals)
            if (self.baseline is None
                    and self._update_n >= self.baseline_rescue_frames):
                rescued = np.where(
                    np.isfinite(self._road_max) & (self._road_max > 0),
                    self._road_max, 0.0).astype(np.float32)
                if int((rescued > 0).sum()) >= self.min_stable_points:
                    self.baseline = rescued

        # 基线上浮校正：深度只会被遮挡压小，不可能比真实路面更远。
        # 某点连续高于基线 N 帧 → 基线错了（被占线车污染/救援兜底时该点还没见过路面），抬回观测值
        if self.baseline is not None and self.baseline_up_snap_frames > 0:
            if self._above_baseline_cnt is None or len(self._above_baseline_cnt) != len(road_depths):
                self._above_baseline_cnt = np.zeros(len(road_depths), dtype=np.int32)
            up_margin = max(self.normal_fluctuation * 2, 2.0)
            for i in range(min(len(road_depths), len(self.baseline))):
                if (road_valid[i] and self.baseline[i] > 0
                        and road_depths[i] - self.baseline[i] > up_margin):
                    self._above_baseline_cnt[i] += 1
                    if self._above_baseline_cnt[i] >= self.baseline_up_snap_frames:
                        self.baseline[i] = float(road_depths[i])
                        self._above_baseline_cnt[i] = 0
                else:
                    self._above_baseline_cnt[i] = 0

        # 基线下沉校正：噪声/漂移导致基线比真实路面偏高零点几米时，
        # 观测会持续略低于基线 → 被 R1/R3 当真车下降打分，空路面锁死 car。
        # 真车遮挡是米级下降（>band），亚米级持续偏低 → 基线噪声偏高，压回观测值
        if self.baseline is not None and self.baseline_down_snap_frames > 0:
            if self._below_baseline_cnt is None or len(self._below_baseline_cnt) != len(road_depths):
                self._below_baseline_cnt = np.zeros(len(road_depths), dtype=np.int32)
            down_band = max(self.normal_fluctuation, 1.5)
            for i in range(min(len(road_depths), len(self.baseline))):
                if road_valid[i] and self.baseline[i] > 0:
                    offset = self.baseline[i] - road_depths[i]
                    if 0.3 < offset <= down_band:
                        self._below_baseline_cnt[i] += 1
                        if self._below_baseline_cnt[i] >= self.baseline_down_snap_frames:
                            self.baseline[i] = float(road_depths[i])
                            self._below_baseline_cnt[i] = 0
                    else:
                        self._below_baseline_cnt[i] = 0
                else:
                    self._below_baseline_cnt[i] = 0

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
                # 冷却期内完全阻止新 car 触发：防止两车间隔阶段噪声误触发
                # 冷却帧数走 car_exit_cooldown_frames（默认15≈0.5s@30fps），真车最多延迟这点时间确认
                if self._car_exit_cooldown > 0:
                    # 冷却期内：不累积 confirm_count，保持当前状态
                    self.pending_state = self.current_state
                    # 冷却期内也要递减计数器（否则冷却期会因噪声帧无限延长）
                    self._car_exit_cooldown -= 1
                else:
                    # 非冷却期：正常逻辑
                    # 强信号快速通道：conf≥0.7 直接确认，跳过confirm_count
                    if self.last_confidence >= 0.70:
                        self.confirm_count = self._eff_confirm
                    else:
                        self.confirm_count += 1
                    self.pending_state = "car"
            else:
                self.confirm_count = self._eff_confirm
            if self.confirm_count >= self._eff_confirm:
                if self.current_state != "car":
                    # 刚进 car：上状态锁，锁内不允许回退，防"确认后立刻闪退再闪进"
                    self._car_hold_remaining = self.min_car_hold_frames
                self.current_state = "car"
                # 车辆确认后清零冷却计数器
                self._car_exit_cooldown = 0
                # 预热阶段：异常帧不加入buffer（保护基线）
                # 正式阶段：清空warmup_buffer
                if self.baseline is not None:
                    self.warmup_buffer = []

        elif self.current_state == "car":
            # 状态锁：刚确认 car 的 min_car_hold_frames 帧内不做任何回退判定
            if self._car_hold_remaining > 0:
                self._car_hold_remaining -= 1
                self.pending_state = "car"
                self.confirm_count = self._eff_confirm
                self.last_depths = depths_clean.copy()
                # 信号层：car 状态下不发新信号（armed=False 已在上升沿置过）
                return self.current_state
            # 检查车辆是否离开
            # YOLO 视觉 HOLD：车停住时不让状态机回退
            # 防闪烁：已经确认 car 后，YOLO 视觉还看到车在检测区（R5≥0.5 且有框），
            # 就跳过"稳定点计数"那套离开判定；R5≥0.30 是弱信号，把回退帧门槛 5→12（更难回退）
            # 跟 _detect_car_by_depth 里的 boost 对齐：R5≥0.40 时本来就有 car 增强，
            # 所以 HOLD 强门槛从 0.50 降到 0.40，保证"boost 的时候绝不回退"
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
                    # 启动冷却期：防止两车间隔阶段噪声误触发新 car
                    self._car_exit_cooldown = self._eff_cooldown
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
                self.confirm_count = self._eff_confirm

        else:
            # 无车或未知状态
            self.confirm_count = 0
            # 递减冷却计数器
            if self._car_exit_cooldown > 0:
                self._car_exit_cooldown -= 1
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

        # 信号层：非car→car 上升沿发一次信号并锁存；
        # 只有状态彻底回 no_car 且冷却结束才解锁，保证一辆车经过只出一个有车信号
        if self.current_state == "car" and self._signal_armed:
            if self._update_n - self._last_signal_update >= self.min_signal_interval_frames:
                self.signal_count += 1
                self._last_signal_update = self._update_n
                self._signal_armed = False
        elif self.current_state == "no_car" and self._car_exit_cooldown <= 0:
            self._signal_armed = True

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


# 亮度监听器：实时看环境亮度，动态调 MOG2 的判定权重
class BrightnessMonitor:
    """
    实时盯着环境亮度，动态调 MOG2 运动检测的判定权重。

    道理很简单：
    - 亮（白天）：深度模型靠谱，MOG2 权重压低点（少误触发）
    - 中等（黄昏）：MOG2 权重一点点往上抬
    - 暗（夜里）：深度模型容易失效，MOG2 权重顶到最大来帮着判

    权重在 [min_weight, max_weight] 之间，用 EMA 平滑过去，免得突然跳变。
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
        参数：
            bright_high：高亮度阈值（0-255），比它亮就是白天，MOG2 权重最低
            bright_mid：中亮度阈值（0-255），比它暗就是夜里，MOG2 权重最高
            min_weight：白天 MOG2 的最小权重（0~1）
            max_weight：夜里 MOG2 的最大权重（0~1）
            weight_alpha：权重 EMA 平滑系数（0~1，越小过渡越慢）
            sample_width：亮度采样宽度（降采样省开销）
            sample_height：亮度采样高度
            log_interval：每 N 帧记一次日志
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
        """算一下这帧的平均亮度（降采样更快）。frame 是 BGR 图，返回 0-255 的亮度值。"""
        return _frame_mean_brightness(frame, self.sample_width, self.sample_height,
                                      independent_scale=True)

    def brightness_to_weight(self, brightness):
        """
        把亮度映射成 MOG2 权重（分段线性）。

        - 亮度 ≥ bright_high → min_weight（白天，MOG2 低权重）
        - bright_mid ≤ 亮度 < bright_high → 线性插值
        - 亮度 < bright_mid → max_weight（夜里，MOG2 高权重）

        传入当前亮度（0-255），返回 [min_weight, max_weight] 里的权重。
        """
        if brightness >= self.bright_high:
            return self.min_weight
        elif brightness <= self.bright_mid:
            return self.max_weight
        else:
            ratio = (brightness - self.bright_mid) / (self.bright_high - self.bright_mid)
            return self.max_weight - ratio * (self.max_weight - self.min_weight)

    def update(self, frame):
        """每帧调一次，更新亮度监测和 MOG2 权重。
        frame 是 BGR 图，返回 (当前亮度, 当前权重, 亮度等级)。"""
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


# MOG2 运动检测器：看有没有运动目标从检测线经过
class MotionLineDetector:
    """
    靠 MOG2 背景减除做运动检测（带确认帧 + 动态亮度权重）。

    大概这么走：
    1. MOG2 先建背景模型，把前景（动的东西）抠出来
    2. 在检测线（监测点所在的垂直线）附近划一条窄带
    3. 数窄带里的前景像素有多少
    4. 得连着 N 帧都有运动才算数 → 防单帧噪声乱判
    5. 按环境亮度动态调灵敏度：
       - 白天（亮）：抬高 min_motion_pixels 和 confirm_frames，少误触发
       - 夜里（暗）：压低 min_motion_pixels 和 confirm_frames，更灵敏

    注意：MOG2 光看到运动不能直接说有车，得跟深度配合：
    - 深度判到 car → 直接算有车（最靠谱）
    - MOG2 确认有运动 + 深度有变化 → 算有车
    - MOG2 确认有运动 + 深度没动 → 多半是反光/噪声，不算
    """

    def __init__(self, detect_x=40, strip_width=30,
                 min_motion_pixels=50, history=300, var_threshold=25,
                 detect_shadow=False, warmup_frames=30,
                 confirm_frames=3, brightness_monitor=None):
        """
        参数：
            detect_x：检测线 x 坐标（监测点所在的那条竖线）
            strip_width：检测带宽度（以 detect_x 为中心，左右各 strip_width/2 像素）
            min_motion_pixels：基础最小运动像素数（会被亮度权重动态改）
            history：MOG2 背景模型的历史帧数
            var_threshold：MOG2 像素跟背景模型的方差阈值
            detect_shadow：要不要检测阴影（True=检测，但可能误判）
            warmup_frames：MOG2 预热帧数，前 N 帧先不检测（背景模型还没稳）
            confirm_frames：基础确认帧数（会被亮度权重动态改）
            brightness_monitor：BrightnessMonitor 实例，给动态权重
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
        按权重调一下真正生效的检测阈值。

        权重高（夜里）：
        - min_motion_pixels 调低（更敏感，暗光下微弱运动也抓得到）
        - confirm_frames 调低（更快确认，少点延迟）

        权重低（白天）：
        - min_motion_pixels 调高（更严，滤掉误触发）
        - confirm_frames 调高（确认慢一点，稳当）

        weight 是当前权重，范围 [0.4, 1.0]。
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
        """喂一帧进来，看有没有运动目标从检测线过。
        frame 是 BGR 图 (h, w, 3)，返回 "motion" / "no_motion" / "unknown"。"""
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
    depth, frame_time = _infer_depth_map(model, processor, img, device)
    h, w = img.shape[:2]

    # 计算所有像素点深度值
    pixel_depths = _extract_pixel_depths(depth, pixel_points or [])

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

    # 初始化检测流水线（深度车辆检测器 + 亮度监听 + MOG2 运动检测）
    effective_fps = fps / frame_step if fps > 0 and frame_step > 0 else fps
    detector, brightness_monitor, motion_detector = build_detector_pipeline(
        car_detector_config, pixel_points, effective_fps, verbose="basic")

    print(f"[Video] 处理视频: {Path(video_path).name}")
    print(f"  分辨率: {w}x{h}, FPS: {fps:.1f}, 总帧数: {total_frames}")

    # 计算实际要处理的帧数（用于进度条总长度）
    n_to_process = (total_frames + frame_step - 1) // frame_step
    if max_frames:
        n_to_process = min(n_to_process, max_frames)

    # 显示平滑：监测点深度 + 画面深度范围做 EMA，去帧间抖动，不碰检测算法
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
            depth, frame_time = _infer_depth_map(model, processor, frame, device)

            depth_vis, d_min_raw, d_max_raw = depth_to_colormap(depth)

            # 计算所有像素点深度值（原始值，用于检测算法）
            pixel_depths = _extract_pixel_depths(depth, pixel_points or [])

            # 纯为显示做的 EMA 平滑，不影响检测逻辑
            # _ema_* 变量在外层 while True 循环前定义，与本分支同属函数作用域，无需 nonlocal
            # 监测点深度平滑
            if pixel_depths:
                if _ema_depths is None or len(_ema_depths) != len(pixel_depths):
                    _ema_depths = list(pixel_depths)
                else:
                    for i, d in enumerate(pixel_depths):
                        if d is None or (isinstance(d, float) and d != d):
                            continue
                        if _ema_depths[i] is None or (isinstance(_ema_depths[i], float) and _ema_depths[i] != _ema_depths[i]):
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

            # 融合判定（主流水线简化版：深度 + MOG2 双重验证，没带 YOLO 视觉协同）
            # 注意：这跟 detect_car.py 里的 fuse_v8_car_state()（YOLO + 多规则加权）是两码事，
            # 这里只给 detect.py 状态条显示用，别混。
            #   - 深度靠谱：car → 直接算有车
            #   - MOG2 辅助：按当前亮度权重换策略
            #     - 夜里（weight≥0.7）：MOG2确认 + 深度基本变化 → 有车
            #     - 黄昏（0.5≤weight<0.7）：MOG2确认 + 深度明显变化 → 有车
            #     - 白天（weight<0.5）：MOG2确认 + 深度剧烈变化 → 有车
            #   - 深度是 no_car 时 MOG2 不覆盖（可能是反光/远处在动）
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
                valid_depths = [d for d in display_depths
                                if d is not None and not (isinstance(d, float) and d != d)]
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


def build_detector_config(car_cfg, root, scalar_values, yolo_model_default="yolov8n.pt"):
    """
    从 car_detect 配置段拼出统一的 detector_config 字典（detect_car / detect_sel_videos 一起用）。
    - scalar_values：各脚本自己解析好的标量参数 dict，直接并进来。
      detect_car 走 CLI(args.*)、detect_sel_videos 走 car_cfg.get(...)，两边默认值不一样，
      所以让调用方传进来，各自的老行为都不动。
    - yolo / mog2 / weights 子块从 car_cfg 读。
    - P0-1 配置透传：judge_mlp / night / saturation_threshold / clahe 四个子块直接并进来，
      检测器初始化时按 config.yaml 生效，不再掉回代码里写死的默认。
    """
    car_cfg = car_cfg or {}
    yolo_cfg = car_cfg.get("yolo", {}) or {}
    mog2_cfg = car_cfg.get("mog2", {}) or {}
    weights_cfg = car_cfg.get("weights", {}) or {}

    config = dict(scalar_values or {})
    config["yolo"] = {
        "enabled": bool(yolo_cfg.get("enabled", True)),
        # 转绝对路径：避免当前工作目录不对时找不着
        "model_path": _abs_path(root, str(yolo_cfg.get("model_path", yolo_model_default))),
        "device": str(yolo_cfg.get("device", "") or ""),
        "confidence": float(yolo_cfg.get("confidence", 0.40)),
        "iou": float(yolo_cfg.get("iou", 0.45)),
        "vehicle_classes": list(yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
        "enter_bottom_ratio": float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
        "min_box_ratio": float(yolo_cfg.get("min_box_ratio", 0.02)),
        "cross_tolerance_px": float(yolo_cfg.get("cross_tolerance_px", 15)),
    }
    config["weights"] = dict(weights_cfg) if weights_cfg else None
    config["mog2"] = {
        "enabled": bool(mog2_cfg.get("enabled", True)),
        "strip_width": int(mog2_cfg.get("strip_width", 30)),
        "min_motion_pixels": int(mog2_cfg.get("min_motion_pixels", 50)),
        "history": int(mog2_cfg.get("history", 300)),
        "var_threshold": int(mog2_cfg.get("var_threshold", 25)),
        "detect_shadow": bool(mog2_cfg.get("detect_shadow", False)),
        "warmup_frames": int(mog2_cfg.get("warmup_frames", 30)),
        "confirm_frames": int(mog2_cfg.get("confirm_frames", 3)),
        "brightness": dict(mog2_cfg.get("brightness", {}) or {}),
    }
    # P0-1 新透传这四个配置子块（之前两个脚本都没从 config.yaml 透传）
    config["judge_mlp"] = dict(car_cfg.get("judge_mlp", {}) or {})
    config["night"] = dict(car_cfg.get("night", {}) or {})
    config["saturation_threshold"] = int(car_cfg.get("saturation_threshold", 235))
    config["clahe"] = dict(car_cfg.get("clahe", {}) or {})
    # 一车一信号 + 夜间专属屏蔽透传
    config["sat_night_only"] = bool(car_cfg.get("sat_night_only", True))
    config["spike_abs_threshold"] = float(car_cfg.get("spike_abs_threshold", 8.0))
    config["min_car_hold_frames"] = int(car_cfg.get("min_car_hold_frames", 10))
    config["min_signal_interval_frames"] = int(car_cfg.get("min_signal_interval_frames", 15))
    config["car_exit_cooldown_frames"] = int(car_cfg.get("car_exit_cooldown_frames", 15))
    config["baseline_rescue_frames"] = int(car_cfg.get("baseline_rescue_frames", 30))
    config["baseline_up_snap_frames"] = int(car_cfg.get("baseline_up_snap_frames", 10))
    config["baseline_down_snap_frames"] = int(car_cfg.get("baseline_down_snap_frames", 10))
    return config


# 检测流水线工厂：4 处初始化块统一走这里，省得重复写
def build_detector_pipeline(detector_config, pixel_points, effective_fps,
                            with_vision=False, verbose="none", yolo_model_path_default="yolov8n.pt"):
    """
    搭检测流水线：VehicleDetector + BrightnessMonitor + MotionLineDetector。

    原来 detect.py(process_video/preview_video)、detect_car.py、detect_sel_videos.py
    里散着 4 份一模一样的初始化代码，这里合成一份：
    - with_vision=False（detect.py 主流水线，原来不带视觉/夜间参数）：跟重构前一模一样
    - with_vision=True（car/sel）：YOLO 视觉 + 夜间自适应参数跟着 detector_config 走
    - verbose 管打印，保持各脚本原来的输出风格：
      "none"  = 不打印（detect_sel_videos.py 批量跑时静默）
      "basic" = [Car]/[MOG2]/[亮度]（detect.py 主流水线）
      "full"  = [YOLO]/[权重]/[阈值]/[MOG2]/[亮度]（detect_car.py）
    返回 (detector, brightness_monitor, motion_detector)，没启用的那项是 None。
    """
    cfg = detector_config or {}
    yolo_cfg = cfg.get("yolo", {}) or {}
    weights_cfg = cfg.get("weights", {}) or {}
    mog2_config = cfg.get("mog2", {}) or {}
    mog2_enabled = bool(mog2_config.get("enabled", True))

    # 深度车辆检测器
    detector = None
    if detector_config and pixel_points:
        rule_weights = None
        if with_vision and weights_cfg:
            # 从配置映射到 detector 内部 key (R1..R5)
            rule_weights = {
                'R1': float(weights_cfg.get("R1_way4_depth_jump", 0.10)),
                'R2': float(weights_cfg.get("R2_way5_abs_depth", 0.25)),
                'R3': float(weights_cfg.get("R3_way1_baseline", 0.25)),
                'R4': float(weights_cfg.get("R4_vehicle_region", 0.10)),
                'R5': float(weights_cfg.get("R5_yolo_vision", 0.30)),
            }
        kwargs = dict(
            n_points=len(pixel_points),
            distance_drop_threshold=cfg["distance_drop_threshold"],
            normal_fluctuation=cfg["normal_fluctuation"],
            fps=effective_fps,
            baseline_warmup=cfg["baseline_warmup"],
            baseline_alpha=cfg["baseline_alpha"],
            min_car_points=cfg["min_car_points"],
            min_stable_points=cfg["min_stable_points"],
            car_confirm_frames=cfg["car_confirm_frames"],
            stable_confirm_frames=cfg["stable_confirm_frames"],
            max_detect_depth=cfg.get("max_detect_depth", 30.0),
            relative_drop_threshold=cfg.get("relative_drop_threshold", 0.3),
            rule_weights=rule_weights,
            car_confidence_threshold=float(cfg.get("car_confidence_threshold", 0.470)),
            # 一车一信号三件套（config 透传，键不存在就用默认）
            min_car_hold_frames=int(cfg.get("min_car_hold_frames", 10)),
            min_signal_interval_frames=int(cfg.get("min_signal_interval_frames", 15)),
            spike_abs_threshold=float(cfg.get("spike_abs_threshold", 8.0)),
            car_exit_cooldown_frames=int(cfg.get("car_exit_cooldown_frames", 15)),
            baseline_rescue_frames=int(cfg.get("baseline_rescue_frames", 30)),
            baseline_up_snap_frames=int(cfg.get("baseline_up_snap_frames", 10)),
            baseline_down_snap_frames=int(cfg.get("baseline_down_snap_frames", 10)),
        )
        if with_vision:
            kwargs.update(
                # YOLO 视觉检测参数
                yolo_enabled=bool(yolo_cfg.get("enabled", False)),
                yolo_model_path=str(yolo_cfg.get("model_path", yolo_model_path_default)),
                yolo_device=str(yolo_cfg.get("device", "") or ""),
                yolo_confidence=float(yolo_cfg.get("confidence", 0.40)),
                yolo_iou=float(yolo_cfg.get("iou", 0.45)),
                yolo_vehicle_classes=tuple(int(c) for c in yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
                yolo_enter_bottom_ratio=float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
                yolo_min_box_ratio=float(yolo_cfg.get("min_box_ratio", 0.02)),
                yolo_detect_x=(pixel_points[0][0] if pixel_points else 40),
                yolo_cross_tolerance=float(yolo_cfg.get("cross_tolerance_px", 15)),
                # 夜间自适应参数
                night_bright_threshold=float(cfg.get("night", {}).get("bright_threshold", 80.0)),
                night_smooth_alpha=float(cfg.get("night", {}).get("smooth_alpha", 0.35)) if cfg.get("night", {}).get("enabled", True) else 0.0,
                night_confirm_extra=int(cfg.get("night", {}).get("confirm_extra", 2)) if cfg.get("night", {}).get("enabled", True) else 0,
                night_cooldown_extra=int(cfg.get("night", {}).get("cooldown_extra", 20)) if cfg.get("night", {}).get("enabled", True) else 0,
            )
        detector = VehicleDetector(**kwargs)
        if verbose == "basic":
            print(f"  [Car] 深度车辆检测已启用")
        if kwargs.get("yolo_enabled", False):
            # 进入 tqdm 进度条之前先确保 YOLO 已加载（避免在进度条首帧 print 打断它）
            detector._ensure_yolo_loaded()
            if verbose == "full":
                print(f"  [YOLO] 视觉检测已启用 | 模型={yolo_cfg.get('model_path')} | 进入检测区阈值=下{float(yolo_cfg.get('enter_bottom_ratio', 0.30)) * 100:.0f}% | 置信度≥{float(yolo_cfg.get('confidence', 0.40)):.2f}")
                if rule_weights:
                    print(f"  [权重] R1={rule_weights['R1']:.2f}(骤变) R2={rule_weights['R2']:.2f}(绝对深度) R3={rule_weights['R3']:.2f}(基线) R4={rule_weights['R4']:.2f}(统计) R5={rule_weights['R5']:.2f}(视觉) = {sum(rule_weights.values()):.2f}")
                    print(f"  [阈值] car_confidence_threshold = {cfg.get('car_confidence_threshold', 0.470):.3f}")

    # MOG2 运动检测 + 亮度监听
    brightness_monitor = None
    motion_detector = None
    if mog2_enabled and pixel_points:
        mog2_bright_cfg = mog2_config.get("brightness", {}) or {}
        brightness_monitor = BrightnessMonitor(
            bright_high=mog2_bright_cfg.get("bright_high", 180.0),
            bright_mid=mog2_bright_cfg.get("bright_mid", 80.0),
            min_weight=mog2_bright_cfg.get("min_weight", 0.4),
            max_weight=mog2_bright_cfg.get("max_weight", 1.0),
            weight_alpha=mog2_bright_cfg.get("weight_alpha", 0.05),
            log_interval=mog2_bright_cfg.get("log_interval", 50),
        )
        detect_x = pixel_points[0][0] if pixel_points else 40
        motion_detector = MotionLineDetector(
            detect_x=detect_x,
            strip_width=int(mog2_config.get("strip_width", 30)),
            min_motion_pixels=int(mog2_config.get("min_motion_pixels", 50)),
            history=int(mog2_config.get("history", 300)),
            var_threshold=int(mog2_config.get("var_threshold", 25)),
            detect_shadow=bool(mog2_config.get("detect_shadow", False)),
            warmup_frames=int(mog2_config.get("warmup_frames", 30)),
            confirm_frames=int(mog2_config.get("confirm_frames", 3)),
            brightness_monitor=brightness_monitor,
        )
        if verbose in ("basic", "full"):
            print(f"  [MOG2] 运动检测已启用 | 检测线 x={detect_x}, 带宽={mog2_config.get('strip_width', 30)}px, 确认={mog2_config.get('confirm_frames', 3)}帧")
            if brightness_monitor:
                print(f"  [亮度] 动态权重已启用 | 白天≥{brightness_monitor.bright_high}→权重={brightness_monitor.min_weight:.1f} | 夜间<{brightness_monitor.bright_mid}→权重={brightness_monitor.max_weight:.1f}")

    return detector, brightness_monitor, motion_detector


def _parse_pixel_points(points_str, count, img_w, img_h):
    """解析像素点配置，返回 [(x,y), ...] 列表。
    points_str 是分号隔开的坐标串，比如 "320,240;640,480"；
    count 是自动生成的点数（只在 points_str 为空时才用）；
    img_w/img_h 是画面尺寸。
    规则：
    1. points_str 有内容 → 直接解析自定义坐标
    2. points_str 为空且 count=15 → 双列非对称布局（左列 x=40 共 7 个点，右列 x=55 共 8 个点）
    3. points_str 为空且 count 为偶数 → 双列对称布局（左列 x=40，右列 x=55）
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
    depth, frame_time = _infer_depth_map(model, processor, img, device)
    h, w = img.shape[:2]

    pixel_depths = _extract_pixel_depths(depth, pixel_points or [])

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

    # 初始化检测流水线（深度车辆检测器 + 亮度监听 + MOG2 运动检测）
    effective_fps = fps / frame_step if fps > 0 and frame_step > 0 else fps
    detector, brightness_monitor, motion_detector = build_detector_pipeline(
        car_detector_config, pixel_points, effective_fps, verbose="basic")

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
                depth, frame_time = _infer_depth_map(model, processor, frame, device)

                depth_vis, d_min, d_max = depth_to_colormap(depth)

                pixel_depths = _extract_pixel_depths(depth, pixel_points or [])

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
