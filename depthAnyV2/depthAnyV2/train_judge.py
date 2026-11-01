"""
判定规则可学习模型训练脚本（实验性）
================================================================
目标：用目录中的样本视频，训练一个轻量 MLP 替代/补充当前 v8 规则的"是否来车"
判定。本阶段为实验基础：先用 v8 规则的输出作为伪标签（知识蒸馏），建立完整的
"特征提取 → 训练 → 验证 → 保存"流水线；后续可替换为人工真值标签以真正超越 v8。

设计要点：
- 不修改现有 detect 代码，只 import 复用其深度推断/检测器
- 特征：每帧约 30 维（深度统计/YOLO/MOG2/亮度/时序窗口/v8内部信号）
- 模型：MLP（30 → 128 → 64 → 32 → 3），Dropout 防过拟合
- 标签：car / no_car / unknown（用 v8 的 fuse_v8 输出作伪标签）
- 划分：按视频分组 8:2，避免帧级泄漏
- 类别不平衡：用类权重加权 CrossEntropy
- 早停：验证集 loss 连续 patience 轮不降则停

输出（默认）：
  checkpoints/judge_model/judge_mlp.pt          模型权重
  judge_training_output/feature_cache.npz        特征缓存（避免重复深度推断）
  judge_training_output/metrics.json             训练指标
  judge_training_output/training_curve.png        训练/验证曲线
  judge_training_output/classification_report.txt 分类报告
  judge_training_output/confusion_matrix.png      混淆矩阵

使用方式：
  # 默认：从 detect/1 找标准视频，在 detect/input 找原始视频，训练 100 轮
  python train_judge.py

  # 自定义目录、轮数、窗口
  python train_judge.py --std-dir detect/1 --input-root detect/input \
      --epochs 100 --window 5 --batch 256

  # 跳过特征提取（用已有缓存，调参时加快迭代）
  python train_judge.py --use-cache

输出解读：
  - training_curve.png：train_loss 与 val_loss 曲线，val 上升=过拟合
  - classification_report.txt：每类的 precision/recall/f1，关注 car 的 recall（漏报）
  - confusion_matrix.png：行=真实，列=预测，看 car↔no_car 误判格
  - metrics.json：最终准确率、macro-F1、最佳轮次
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 允许导入 detect 包下的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect"))
from detect import (
    load_model, _load_config, _abs_path, _parse_pixel_points, DEFAULT_MODEL,
    VehicleDetector, MotionLineDetector, BrightnessMonitor,
)
from detect_car import fuse_v8_car_state

# 尝试导入 matplotlib（可选，缺失则跳过画图）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ============================================================
# 配置常量
# ============================================================
STATE_LIST = ["car", "no_car", "unknown"]          # 类别顺序（与 one-hot 索引对应）
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LIST)}
WINDOW_DEFAULT = 5                                 # 时序窗口：拼接前 W 帧特征
FEATURE_DIM = 30                                   # 单帧特征维度（见 build_feature）
LABELS = {"car": 0, "no_car": 1, "unknown": 2}


# ============================================================
# 1. 视频配对：从标准视频反推原始视频
# ============================================================
def find_standard_pairs(std_dir, input_root, label_override=None):
    """
    扫描 std_dir 下 *_depth_video.mp4，去掉后缀得到原始视频名，
    再到 input_root 递归查找同名（任意扩展名）的原始视频。
    label_override: 若非 None（"car"/"no_car"/"unknown"），则该目录所有帧的标签
    被强制为该类别（用于"无车误识"等错误样本：v8 判错，用真值覆盖）。
    返回 [(video_name, original_path, label_override), ...]
    """
    if not os.path.isdir(std_dir):
        raise FileNotFoundError(f"标准视频目录不存在: {std_dir}")
    exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    # 建立 input 根目录下所有视频的 stem → 路径 索引
    inp_index = {}
    for dp, _, ns in os.walk(input_root):
        for n in ns:
            if n.lower().endswith(exts):
                inp_index[os.path.splitext(n)[0]] = os.path.join(dp, n)
    pairs = []
    for f in sorted(os.listdir(std_dir)):
        if not f.endswith("_depth_video.mp4"):
            continue
        stem = os.path.splitext(f)[0].replace("_depth_video", "")
        if stem in inp_index:
            pairs.append((stem, inp_index[stem], label_override))
    return pairs


def find_pairs_from_sample_root(sample_root, input_root, label_overrides=None):
    """
    递归扫描 sample_root 下所有子目录的 *_depth_video.mp4，按子目录名推断标签策略：
      - label_overrides: dict {子目录名: "v8"|"car"|"no_car"|"unknown"}
        "v8" 表示用 v8 输出作伪标签（正确样本），其他值强制覆盖标签（错误样本纠正）。
      - 默认推断：含"误识"→no_car，含"闪烁"→car，含"标准"→v8，其余→v8
    返回 [(video_name, original_path, label_override|None), ...]
    """
    label_overrides = label_overrides or {}
    pairs = []
    if not os.path.isdir(sample_root):
        raise FileNotFoundError(f"样本根目录不存在: {sample_root}")
    for sub_dp, _, files in os.walk(sample_root):
        sub_name = os.path.basename(sub_dp) or os.path.basename(sample_root)
        # 推断该子目录的标签策略
        strategy = label_overrides.get(sub_name)
        if strategy is None:
            if "误识" in sub_name:
                strategy = "no_car"      # 无车误识：v8 误报 car，真值是 no_car
            elif "闪烁" in sub_name:
                strategy = "car"         # 闪烁：本应连续 car，真值是 car
            else:
                strategy = "v8"          # 标准/其他：用 v8 输出作伪标签
        override = None if strategy == "v8" else strategy
        for f in sorted(files):
            if not f.endswith("_depth_video.mp4"):
                continue
            stem = os.path.splitext(f)[0].replace("_depth_video", "")
            # 在 input_root 查找同名原始视频
            exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
            found_path = None
            for dp2, _, ns in os.walk(input_root):
                for n in ns:
                    if n.lower().endswith(exts) and os.path.splitext(n)[0] == stem:
                        found_path = os.path.join(dp2, n)
                        break
                if found_path:
                    break
            if found_path:
                pairs.append((stem, found_path, override))
    return pairs


# ============================================================
# 2. 检测器配置构建（与 detect_car.py 完全一致，保证伪标签可复现）
# ============================================================
def build_detector_config(cfg):
    root = os.path.dirname(os.path.abspath(__file__))
    car_cfg = cfg.get("car_detect", {}) or {}
    mog2_cfg = car_cfg.get("mog2", {}) or {}
    yolo_cfg = car_cfg.get("yolo", {}) or {}
    weights_cfg = car_cfg.get("weights", {}) or {}
    rule_weights = None
    if weights_cfg:
        rule_weights = {
            'R1': float(weights_cfg.get("R1_way4_depth_jump", 0.10)),
            'R2': float(weights_cfg.get("R2_way5_abs_depth", 0.25)),
            'R3': float(weights_cfg.get("R3_way1_baseline", 0.25)),
            'R4': float(weights_cfg.get("R4_vehicle_region", 0.10)),
            'R5': float(weights_cfg.get("R5_yolo_vision", 0.30)),
        }
    return {
        "distance_drop_threshold": float(car_cfg.get("distance_drop_threshold", 0.4)),
        "normal_fluctuation": float(car_cfg.get("normal_fluctuation", 1.5)),
        "min_car_points": int(car_cfg.get("min_car_points", 2)),
        "min_stable_points": int(car_cfg.get("min_stable_points", 3)),
        "car_confirm_frames": int(car_cfg.get("car_confirm_frames", 3)),
        "stable_confirm_frames": int(car_cfg.get("stable_confirm_frames", 5)),
        "baseline_warmup": int(car_cfg.get("baseline_warmup", 5)),
        "baseline_alpha": float(car_cfg.get("baseline_alpha", 0.05)),
        "max_detect_depth": float(car_cfg.get("max_detect_depth", 30.0)),
        "relative_drop_threshold": float(car_cfg.get("relative_drop_threshold", 0.08)),
        "car_confidence_threshold": float(car_cfg.get("car_confidence_threshold", 0.470)),
        "yolo": {
            "enabled": bool(yolo_cfg.get("enabled", True)),
            "model_path": _abs_path(root, str(yolo_cfg.get("model_path", "checkpoints/yolo/best.pt"))),
            "device": str(yolo_cfg.get("device", "") or ""),
            "confidence": float(yolo_cfg.get("confidence", 0.40)),
            "iou": float(yolo_cfg.get("iou", 0.45)),
            "vehicle_classes": list(yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
            "enter_bottom_ratio": float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
            "min_box_ratio": float(yolo_cfg.get("min_box_ratio", 0.02)),
        },
        "rule_weights": rule_weights,
        "mog2": mog2_cfg,
    }


def make_detectors(dc, fps, pixel_points):
    """根据 detector_config 创建 VehicleDetector / MotionLineDetector / BrightnessMonitor"""
    yolo_cfg = dc["yolo"]
    mog2_cfg = dc["mog2"]
    detector = VehicleDetector(
        n_points=len(pixel_points) if pixel_points else 5,
        distance_drop_threshold=dc["distance_drop_threshold"],
        normal_fluctuation=dc["normal_fluctuation"],
        fps=fps or 25.0,
        baseline_warmup=dc["baseline_warmup"],
        baseline_alpha=dc["baseline_alpha"],
        min_car_points=dc["min_car_points"],
        min_stable_points=dc["min_stable_points"],
        car_confirm_frames=dc["car_confirm_frames"],
        stable_confirm_frames=dc["stable_confirm_frames"],
        max_detect_depth=dc["max_detect_depth"],
        relative_drop_threshold=dc["relative_drop_threshold"],
        yolo_enabled=bool(yolo_cfg.get("enabled", False)),
        yolo_model_path=str(yolo_cfg.get("model_path", "")),
        yolo_device=str(yolo_cfg.get("device", "") or ""),
        yolo_confidence=float(yolo_cfg.get("confidence", 0.40)),
        yolo_iou=float(yolo_cfg.get("iou", 0.45)),
        yolo_vehicle_classes=tuple(int(c) for c in yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
        yolo_enter_bottom_ratio=float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
        yolo_min_box_ratio=float(yolo_cfg.get("min_box_ratio", 0.02)),
        rule_weights=dc["rule_weights"],
        car_confidence_threshold=dc["car_confidence_threshold"],
    )
    if yolo_cfg.get("enabled", False):
        detector._ensure_yolo_loaded()

    bright_cfg = mog2_cfg.get("brightness", {}) or {}
    brightness_monitor = BrightnessMonitor(
        bright_high=float(bright_cfg.get("bright_high", 180.0)),
        bright_mid=float(bright_cfg.get("bright_mid", 80.0)),
        min_weight=float(bright_cfg.get("min_weight", 0.4)),
        max_weight=float(bright_cfg.get("max_weight", 1.0)),
        weight_alpha=float(bright_cfg.get("weight_alpha", 0.05)),
        log_interval=9999,
    )
    detect_x = pixel_points[0][0] if pixel_points else 40
    motion_detector = MotionLineDetector(
        detect_x=detect_x,
        strip_width=int(mog2_cfg.get("strip_width", 30)),
        min_motion_pixels=int(mog2_cfg.get("min_motion_pixels", 50)),
        history=int(mog2_cfg.get("history", 300)),
        var_threshold=int(mog2_cfg.get("var_threshold", 25)),
        detect_shadow=bool(mog2_cfg.get("detect_shadow", False)),
        warmup_frames=int(mog2_cfg.get("warmup_frames", 30)),
        confirm_frames=int(mog2_cfg.get("confirm_frames", 3)),
        brightness_monitor=brightness_monitor,
    )
    return detector, motion_detector, brightness_monitor


# ============================================================
# 3. 单帧特征向量构建（约 30 维）
# ============================================================
def build_feature(pixel_depths, detector, motion_detector, brightness_monitor,
                  prev_d_min, prev_depth_conf, prev_state_idx):
    """
    构建单帧特征向量（numpy float32, 长度 FEATURE_DIM）。
    特征组：
      A. 深度统计(7): d_min/d_max/d_mean/d_std + baseline_min/baseline_mean + drop_abs
      B. YOLO(2): max_conf + n_boxes
      C. MOG2/亮度(4): motion_pixels + confirmed + brightness_val + mog2_weight
      D. v8 内部信号(2): depth_conf + R5_vision
      E. 时序(4): prev_d_min + prev_depth_conf + drop_rel + depth_diff_rate
      F. 前状态 one-hot(3): prev_state
    缺失值用 0 填充，保证维度恒定。
    """
    valid = [d for d in (pixel_depths or []) if d is not None and not (isinstance(d, float) and d != d)]
    if valid:
        d_min = float(min(valid)); d_max = float(max(valid))
        d_mean = float(np.mean(valid)); d_std = float(np.std(valid))
    else:
        d_min = d_max = d_mean = d_std = 0.0
    baseline = getattr(detector, "baseline", None) if detector else None
    if baseline is not None and len(baseline):
        bl_arr = np.asarray(baseline, dtype=np.float32)
        bl_min = float(np.nanmin(bl_arr)); bl_mean = float(np.nanmean(bl_arr))
    else:
        bl_min = bl_mean = 0.0
    drop_abs = (bl_min - d_min) if bl_min > 0 else 0.0
    drop_rel = (drop_abs / bl_min) if bl_min > 0 else 0.0

    yolo_boxes = list(getattr(detector, "_last_yolo_boxes", []) or []) if detector else []
    yolo_max_conf = max((float(b[4]) for b in yolo_boxes), default=0.0)
    yolo_n = len(yolo_boxes)

    motion_px = float(getattr(motion_detector, "motion_pixel_count", 0) if motion_detector else 0)
    mog2_confirmed = 1.0 if (motion_detector and motion_detector.is_confirmed()) else 0.0
    bright_val = 0.0
    if brightness_monitor:
        lvl = brightness_monitor.get_brightness_level()
        bright_val = float(lvl) if lvl is not None else 0.0
    mog2_w = float(motion_detector.get_current_weight() if motion_detector and hasattr(motion_detector, "get_current_weight") else 1.0)

    depth_conf = float(getattr(detector, "last_confidence", 0.0) or 0.0) if detector else 0.0
    r5 = float(getattr(detector, "last_vision_score", 0.0) or 0.0) if detector else 0.0

    p_dmin = float(prev_d_min) if prev_d_min is not None else d_min
    p_conf = float(prev_depth_conf) if prev_depth_conf is not None else depth_conf
    depth_diff_rate = (d_min - p_dmin) / p_dmin if p_dmin > 0 else 0.0

    prev_oh = [0.0, 0.0, 0.0]
    if 0 <= prev_state_idx < 3:
        prev_oh[prev_state_idx] = 1.0

    feat = [
        d_min, d_max, d_mean, d_std, bl_min, bl_mean, drop_abs,   # A 7
        yolo_max_conf, yolo_n,                                       # B 2
        motion_px, mog2_confirmed, bright_val, mog2_w,               # C 4
        depth_conf, r5,                                              # D 2
        p_dmin, p_conf, drop_rel, depth_diff_rate,                  # E 4
        prev_oh[0], prev_oh[1], prev_oh[2],                         # F 3
    ]
    # 确保维度恒定（防止中间算错）
    if len(feat) < FEATURE_DIM:
        feat += [0.0] * (FEATURE_DIM - len(feat))
    elif len(feat) > FEATURE_DIM:
        feat = feat[:FEATURE_DIM]
    return np.asarray(feat, dtype=np.float32)


# ============================================================
# 4. 特征提取（跑深度推断 + v8，逐帧记录特征与伪标签）
# ============================================================
def extract_features_for_video(model, processor, video_path, device, dc,
                                pixel_points, pbar_desc="", label_override=None):
    """
    对单个视频逐帧提取 (feature, label_idx, frame_idx) 列表。
    label_override: 若非 None（"car"/"no_car"/"unknown"），强制标签为该类别，
    用于"无车误识/闪烁"等错误样本：v8 判错时用真值覆盖，让模型学到 v8 的错误边界。
    默认 None → 用 v8 fuse 的最终 state 作伪标签（知识蒸馏）。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector, motion_detector, brightness_monitor = make_detectors(dc, fps, pixel_points)

    last_final_state = "unknown"
    prev_d_min = None
    prev_depth_conf = None
    prev_state_idx = STATE_TO_IDX["unknown"]

    records = []  # [(feat, label_idx, frame_idx), ...]
    frame_idx = 0
    pbar = tqdm(total=total, desc=pbar_desc[:18], ncols=110, leave=True, mininterval=0.5)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[img_rgb], return_tensors="pt").to(device)
        with torch.no_grad():
            depth = model(**inputs).predicted_depth
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h, w), mode="bicubic", align_corners=False).squeeze().cpu().numpy()

        pixel_depths = []
        for px, py in pixel_points:
            if 0 <= px < w and 0 <= py < h:
                pixel_depths.append(float(depth[py, px]))
            else:
                pixel_depths.append(None)

        car_state = None
        if pixel_depths:
            car_state = detector.update(pixel_depths, frame)
        if motion_detector:
            motion_detector.update(frame)

        state, last_final_state = fuse_v8_car_state(
            car_state=car_state, detector=detector, motion_detector=motion_detector,
            pixel_depths=pixel_depths,
            yolo_boxes=list(getattr(detector, "_last_yolo_boxes", []) or []),
            last_final_state=last_final_state,
        )

        feat = build_feature(
            pixel_depths, detector, motion_detector, brightness_monitor,
            prev_d_min, prev_depth_conf, prev_state_idx,
        )
        # 标签：若指定 override（错误样本），强制用真值；否则用 v8 输出（伪标签）
        if label_override is not None:
            lbl = STATE_TO_IDX.get(label_override, 2)
        else:
            lbl = STATE_TO_IDX.get(state, 2)
        records.append((feat, lbl, frame_idx))

        # 更新时序缓存
        valid = [d for d in pixel_depths if d is not None]
        if valid:
            prev_d_min = float(min(valid))
        prev_depth_conf = float(getattr(detector, "last_confidence", 0.0) or 0.0)
        prev_state_idx = STATE_TO_IDX.get(state, 2)

        pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()
    return records, fps


def extract_all(pairs, model, processor, device, dc, pixel_points_fn, cache_path,
                force=False):
    """
    提取所有视频的特征，缓存到 cache_path (.npz)。
    pairs: [(name, path, label_override|None), ...]
    返回:
      features: (N, FEATURE_DIM) float32
      labels:   (N,) int64
      video_ids:(N,) int  —— 每帧所属视频索引，用于按视频分组
      video_names: list[str]
    """
    if not force and os.path.exists(cache_path):
        print(f"[Cache] 读取已有特征缓存: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], data["video_ids"], list(data["video_names"])

    print(f"[Extract] 提取特征 for {len(pairs)} 个视频...")
    all_feats, all_labels, all_vids = [], [], []
    video_names = []
    for vi, item in enumerate(pairs, 1):
        name, vp = item[0], item[1]
        override = item[2] if len(item) > 2 else None
        tag = f"[{override or 'v8'}]" if override else ""
        print(f"\n[Extract] {vi}/{len(pairs)}: {name} {tag}")
        try:
            cap_t = cv2.VideoCapture(str(vp))
            w = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_t.release()
            pts = pixel_points_fn(w, h)
            if not pts:
                print(f"  [WARN] 监测点解析失败，跳过")
                continue
            recs, fps = extract_features_for_video(
                model, processor, vp, device, dc, pts, pbar_desc=name,
                label_override=override)
            for feat, label, fi in recs:
                all_feats.append(feat)
                all_labels.append(label)
                all_vids.append(vi - 1)
            video_names.append(name)
            sc = Counter(STATE_LIST[l] for _, l, _ in recs)
            print(f"  [Done] {len(recs)}帧 | car={sc.get('car',0)} "
                  f"no_car={sc.get('no_car',0)} unk={sc.get('unknown',0)}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            continue

    if not all_feats:
        raise RuntimeError("未提取到任何特征，检查视频/模型/监测点配置")

    features = np.stack(all_feats).astype(np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    video_ids = np.asarray(all_vids, dtype=np.int64)
    # 对齐 video_names 长度到 video_ids 范围
    max_vid = int(video_ids.max()) if len(video_ids) else 0
    while len(video_names) <= max_vid:
        video_names.append(f"unknown_{len(video_names)}")
    video_names_arr = np.asarray(video_names, dtype=object)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels,
                        video_ids=video_ids, video_names=video_names_arr)
    print(f"\n[Cache] 保存特征缓存: {cache_path} | {len(features)}帧 × {features.shape[1]}维")
    return features, labels, video_ids, video_names


# ============================================================
# 5. 时序窗口数据集（拼接前 W 帧特征）
# ============================================================
class JudgeDataset(Dataset):
    """
    滑动窗口数据集：对每个视频内部，把当前帧与前 W-1 帧的特征拼接成一个
    长样本（W*FEATURE_DIM 维），避免跨视频拼接（跨视频时序无意义）。
    标签取当前帧的 label。
    """
    def __init__(self, features, labels, video_ids, window=1):
        self.window = window
        self.samples = []  # [(window_feat (W*D,), label)]
        # 按视频分组帧索引
        vid_to_frames = defaultdict(list)
        for i, vid in enumerate(video_ids):
            vid_to_frames[int(vid)].append(i)
        for vid, frames in vid_to_frames.items():
            frames_sorted = sorted(frames)
            for pos, fi in enumerate(frames_sorted):
                lo = max(0, pos - window + 1)
                window_frames = frames_sorted[lo:pos + 1]
                # 左填充：不足 W 帧时前面补 0
                feats = [features[j] for j in window_frames]
                while len(feats) < window:
                    feats.insert(0, np.zeros_like(features[0]))
                window_feat = np.concatenate(feats, axis=0).astype(np.float32)
                self.samples.append((window_feat, int(labels[fi])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


# ============================================================
# 6. 模型：MLP
# ============================================================
class JudgeMLP(nn.Module):
    """
    输入: (batch, window * FEATURE_DIM)
    输出: (batch, 3) logits —— car / no_car / unknown
    """
    def __init__(self, in_dim, hidden=(128, 64, 32), n_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 7. 训练 + 验证
# ============================================================
def train_model(train_loader, val_loader, in_dim, device, epochs=100,
               lr=1e-3, weight_decay=1e-4, patience=15, class_weights=None,
               save_path=None, verbose=True):
    """
    训练 MLP，带早停。返回 (best_model, history_dict)
    history: {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    """
    model = JudgeMLP(in_dim=in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 类别权重处理不平衡（car 帧多，no_car 帧少时加权）
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        cw_tensor = None
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot_loss = tot_correct = tot_n = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(labels)
            tot_correct += (logits.argmax(1) == labels).sum().item()
            tot_n += len(labels)
        tr_loss = tot_loss / max(1, tot_n)
        tr_acc = tot_correct / max(1, tot_n)

        # ---- val ----
        model.eval()
        v_loss = v_correct = v_n = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                loss = criterion(logits, labels)
                v_loss += loss.item() * len(labels)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_n += len(labels)
        vl_loss = v_loss / max(1, v_n)
        vl_acc = v_correct / max(1, v_n)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        improved = vl_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or improved or no_improve == 0):
            mark = "*" if improved else " "
            print(f"  Epoch {epoch:3d}/{epochs} {mark}| tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
                  f"| val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}")

        if no_improve >= patience:
            print(f"  [EarlyStop] {patience} 轮无改善，停止于 epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "n_classes": 3,
                    "feature_dim": FEATURE_DIM,
                    "state_list": STATE_LIST}, save_path)
        print(f"[Save] 模型保存: {save_path} (best epoch={best_epoch})")
    return model, history, best_epoch


# ============================================================
# 8. 评估：分类报告 + 混淆矩阵
# ============================================================
def evaluate_model(model, loader, device):
    """返回 (y_true, y_pred)"""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            logits = model(feats).cpu()
            ys.extend(labels.tolist())
            ps.extend(logits.argmax(1).tolist())
    return np.array(ys), np.array(ps)


def classification_report_txt(y_true, y_pred, target_names):
    """生成简易分类报告文本"""
    lines = []
    lines.append("分类报告 (Classification Report)")
    lines.append("=" * 60)
    lines.append(f"{'类别':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}")
    lines.append("-" * 60)
    for i, name in enumerate(target_names):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        sup = int((y_true == i).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(f"{name:<10}{prec:<12.4f}{rec:<12.4f}{f1:<12.4f}{sup:<10}")
    acc = (y_true == y_pred).mean()
    lines.append("-" * 60)
    lines.append(f"总体准确率 Accuracy: {acc:.4f}")
    # macro
    f1s = []
    for i in range(len(target_names)):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    lines.append(f"Macro-F1: {np.mean(f1s):.4f}")
    return "\n".join(lines)


def plot_training_curve(history, save_path):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)


def plot_confusion(y_true, y_pred, target_names, save_path):
    if not HAS_MPL:
        return
    n = len(target_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(target_names); ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)


# ============================================================
# 9. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="判定规则可学习模型训练（实验性）")
    parser.add_argument("--std-dir", type=str, default=None,
                        help="标准视频目录（默认 detect/样本/标准）")
    parser.add_argument("--sample-root", type=str, default=None,
                        help="样本根目录：递归扫描所有子目录（默认 detect/样本）。"
                        "会按子目录名自动推断标签策略：含'误识'→no_car，含'闪烁'→car，其余→v8伪标签")
    parser.add_argument("--label-overrides", type=str, default="",
                        help="手动指定子目录标签策略，逗号分隔，如 '无车误识=no_car,闪烁=car,标准=v8'")
    parser.add_argument("--input-root", type=str, default=None,
                        help="原始视频根目录（默认 detect/input）")
    parser.add_argument("--output", type=str, default="judge_training_output",
                        help="训练输出目录")
    parser.add_argument("--model-out", type=str, default=None,
                        help="模型保存路径（默认 checkpoints/judge_model/judge_mlp.pt）")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--batch", type=int, default=256, help="batch size")
    parser.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                        help="时序窗口帧数（1=单帧，5=前5帧拼接）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use-cache", action="store_true", help="用已有特征缓存（跳过提取）")
    parser.add_argument("--force-extract", action="store_true", help="强制重新提取特征")
    parser.add_argument("--val-video-frac", type=float, default=0.2,
                        help="验证视频占比（按视频分组，默认20%）")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---------- 路径 ----------
    root = os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config()
    detect_cfg = cfg.get("detect", {}) or {}
    std_dir = args.std_dir or os.path.join(root, "detect", "样本", "标准")
    sample_root = args.sample_root or os.path.join(root, "detect", "样本")
    input_root = args.input_root or _abs_path(root, detect_cfg.get("input_dir", "detect/input"))
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "feature_cache.npz")
    model_out = args.model_out or os.path.join(root, "checkpoints", "judge_model", "judge_mlp.pt")

    # 解析 label_overrides 字符串
    label_overrides = {}
    if args.label_overrides:
        for part in args.label_overrides.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                label_overrides[k.strip()] = v.strip()

    print("=" * 70)
    print("判定规则可学习模型训练（实验性）")
    print("=" * 70)
    print(f"输出目录: {out_dir}")
    print(f"模型保存: {model_out}")
    print(f"参数: epochs={args.epochs} batch={args.batch} window={args.window} "
          f"lr={args.lr} dropout={args.dropout} patience={args.patience}")

    # ---------- 配对视频 ----------
    # 优先用 --sample-root（递归多目录，按子目录名推断标签策略）；
    # 若未指定 sample-root 或其不存在，回退到 --std-dir 单目录
    pairs = []
    if os.path.isdir(sample_root):
        pairs = find_pairs_from_sample_root(sample_root, input_root, label_overrides)
        mode = f"递归样本根目录 {sample_root}（按子目录名推断标签策略）"
    else:
        pairs = find_standard_pairs(std_dir, input_root)
        mode = f"单目录 {std_dir}（全部用 v8 伪标签）"
    print(f"[Pair] {mode}")
    print(f"[Pair] 匹配到 {len(pairs)} 个样本视频 ↔ 原始视频")
    # 统计各标签策略数量
    strat_cnt = Counter((p[2] or "v8") for p in pairs)
    print(f"[Pair] 标签策略分布: " + " ".join(f"{k}={v}" for k, v in sorted(strat_cnt.items())))
    if not pairs:
        print("[ERROR] 未匹配到视频。检查 sample-root/std-dir 和 input-root 是否正确，"
              "以及标准视频命名是否为 xxx_depth_video.mp4")
        sys.exit(1)

    # ---------- 监测点解析函数 ----------
    ppoints_str = str(detect_cfg.get("pixel_depth_points", "") or "").strip()
    pcount = int(detect_cfg.get("pixel_depth_count", 5))
    print(f"[Pixel] pixel_depth_points='{ppoints_str}' pixel_depth_count={pcount}")

    def pixel_points_fn(w, h):
        return _parse_pixel_points(ppoints_str, pcount, w, h)

    # ---------- 特征提取 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    dc = build_detector_config(cfg)
    if not args.use_cache:
        print("[Model] 加载深度模型...")
        model, processor = load_model(cfg.get("model", {}).get("student", DEFAULT_MODEL), device)
    else:
        model = processor = None
        print("[Cache] 跳过深度模型加载（使用缓存）")

    features, labels, video_ids, video_names = extract_all(
        pairs, model, processor, device, dc, pixel_points_fn,
        cache_path, force=args.force_extract,
    )
    print(f"\n[Data] 总帧数: {len(features)} | 特征维度: {features.shape[1]}")
    label_dist = Counter(labels.tolist())
    print(f"[Data] 标签分布: " + " ".join(f"{STATE_LIST[k]}={v}" for k, v in sorted(label_dist.items())))

    # ---------- 按视频分组 train/val ----------
    unique_vids = sorted(set(video_ids.tolist()))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_vids)
    n_val = max(1, int(len(unique_vids) * args.val_video_frac))
    val_vids = set(unique_vids[:n_val])
    train_vids = set(unique_vids[n_val:])
    train_mask = np.array([int(v) in train_vids for v in video_ids])
    val_mask = ~train_mask
    print(f"[Split] 视频: train={len(train_vids)} val={len(val_vids)} | "
          f"帧: train={int(train_mask.sum())} val={int(val_mask.sum())}")

    train_ds = JudgeDataset(features[train_mask], labels[train_mask],
                            video_ids[train_mask], window=args.window)
    val_ds = JudgeDataset(features[val_mask], labels[val_mask],
                          video_ids[val_mask], window=args.window)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    in_dim = args.window * FEATURE_DIM
    print(f"[Model] MLP in_dim={in_dim} | 训练样本={len(train_ds)} 验证样本={len(val_ds)}")

    # ---------- 类权重 ----------
    train_labels = labels[train_mask]
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    class_weights = counts.sum() / (3.0 * counts)
    print(f"[Weight] 类权重: " + " ".join(f"{STATE_LIST[k]}={class_weights[k]:.3f}" for k in range(3)))

    # ---------- 训练 ----------
    print("\n[Train] 开始训练...")
    t0 = time.time()
    best_model, history, best_epoch = train_model(
        train_loader, val_loader, in_dim, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        class_weights=class_weights, save_path=model_out,
    )
    train_sec = time.time() - t0
    print(f"[Train] 完成 | 耗时 {train_sec:.1f}s | best epoch={best_epoch}")

    # ---------- 评估 ----------
    y_true, y_pred = evaluate_model(best_model, val_loader, device)
    report = classification_report_txt(y_true, y_pred, STATE_LIST)
    acc = float((y_true == y_pred).mean())

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    # ---------- 保存报告/图 ----------
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\n\nbest_epoch={best_epoch}\ntrain_time_sec={train_sec:.1f}\n")
        f.write(f"window={args.window} lr={args.lr} dropout={args.dropout}\n")

    if HAS_MPL:
        plot_training_curve(history, os.path.join(out_dir, "training_curve.png"))
        plot_confusion(y_true, y_pred, STATE_LIST, os.path.join(out_dir, "confusion_matrix.png"))
        print(f"[Plot] 训练曲线 + 混淆矩阵已保存到 {out_dir}")
    else:
        print("[Plot] matplotlib 不可用，跳过画图")

    # ---------- metrics.json ----------
    metrics = {
        "run_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_videos": len(pairs),
        "n_frames": int(len(features)),
        "n_train_frames": int(train_mask.sum()),
        "n_val_frames": int(val_mask.sum()),
        "label_distribution": {STATE_LIST[k]: int(label_dist.get(k, 0)) for k in range(3)},
        "window": args.window,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_acc": float(max(history["val_acc"])) if history["val_acc"] else 0.0,
        "final_val_acc": float(history["val_acc"][-1]) if history["val_acc"] else 0.0,
        "accuracy": acc,
        "train_time_sec": train_sec,
        "model_path": model_out,
        "note": "伪标签=v8输出（知识蒸馏）。要超越v8需替换为人工真值标签。",
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] 全部输出保存在: {out_dir}")
    print(f"[Done] 模型: {model_out}")
    print("\n解读：")
    print("  - training_curve.png: val_loss 上升=过拟合，需增大 dropout 或减小 window")
    print("  - classification_report.txt: 关注 car 的 recall（漏报）和 no_car 的 precision（误报）")
    print("  - 本模型为 v8 伪标签蒸馏结果，准确率≈v8；要真正超越需独立人工标注帧级真值")


if __name__ == "__main__":
    main()
