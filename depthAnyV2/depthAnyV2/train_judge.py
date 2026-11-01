"""
判定规则可学习模型训练脚本，实验性质的。
拿目录里的样本视频训一个轻量 MLP，先看看能不能替代或补一下现在 v8 那套"是否来车"判定。
这版先打基础：用 v8 的输出当伪标签（其实就是知识蒸馏），把"提特征 → 训练 → 验证 → 保存"
这条流水线搭完整；以后换成人工真值，才有可能真超过 v8。

几点说明：
- detect 那边代码一行没动，只是 import 进来复用它的深度推断和检测器
- 特征每帧 25 维：深度统计 / YOLO / MOG2 / 亮度 / 时序窗口
- 模型是 MLP：25 → 128 → 64 → 32 → 3，加 Dropout 防过拟合
- 标签三类 car / no_car / unknown，用 v8 的 fuse_v8 输出当伪标签
- 划分按视频分组 8:2，免得同一视频的帧又进训练又进验证（帧级泄漏）
- car 帧多、no_car 帧少，用类权重给 CrossEntropy 加权
- 验证集 loss 连着 patience 轮不降就早停

默认输出这些文件：
  checkpoints/judge_model/judge_mlp.pt           模型权重
  judge_training_output/feature_cache.npz          特征缓存（免得重复跑深度推断）
  judge_training_output/metrics.json              训练指标
  judge_training_output/training_curve.png         训练/验证曲线
  judge_training_output/classification_report.txt  分类报告
  judge_training_output/confusion_matrix.png       混淆矩阵

怎么跑：
  默认就是增量提取——老视频走缓存，只提新增视频的特征。
  python train_judge.py
  第一次跑或者没缓存会自动全量提取。

  想改目录、轮数、窗口：
  python train_judge.py --std-dir detect/1 --input-root detect/input \
      --epochs 100 --window 5 --batch 256

  只调参，跳过提取和深度模型加载，直接用缓存：
  python train_judge.py --use-cache

  忽略缓存、所有视频重跑一遍深度推断：
  python train_judge.py --force-extract

结果怎么看：
  - training_curve.png：train_loss 和 val_loss 曲线，val 一直涨就是过拟合
  - classification_report.txt：各类 precision/recall/f1，重点盯 car 的 recall（漏报）
  - confusion_matrix.png：行是真实、列是预测，看 car 和 no_car 互相判错的格子
  - metrics.json：最终准确率、macro-F1、最优轮次
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
    VehicleDetector, MotionLineDetector, BrightnessMonitor, _extract_pixel_depths,
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


# 配置常量
STATE_LIST = ["car", "no_car", "unknown"]          # 类别顺序，要跟 one-hot 下标对上
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LIST)}
WINDOW_DEFAULT = 5                                 # 时序窗口：拼前 W 帧特征
FEATURE_DIM = 25                                   # 单帧特征维度（A7+B4+C4+D3+E4+F3，不含 v8 内部信号）
YOLO_CONF_THRESHOLD = 0.40                         # YOLO 框置信度阈值，跟 v8 的 R5 一致


# 车过没过线：拿 YOLO 框的 x 坐标来模拟"看画面"
def car_crossed_line(yolo_boxes, detect_x=40, direction="any"):
    """
    拿 YOLO 框坐标判车过没过检测线，跟人眼看差不多。
    yolo_boxes: [(x1,y1,x2,y2,conf,cls), ...]
    detect_x: 检测线 x 像素坐标（默认 40，左列监测点）
    direction: "any"=框盖住线就算 | "left"=框在检测线左侧 | "right"=框在右侧还没过
    返回: (crossed: bool, box_info: dict|None)
      - crossed=True 就是有框过线了
      - box_info 带框坐标和置信度，取样日志要用
    """
    if not yolo_boxes:
        return False, None
    for (x1, y1, x2, y2, conf, cls) in yolo_boxes:
        if float(conf) < YOLO_CONF_THRESHOLD:
            continue
        bx1, bx2 = float(x1), float(x2)
        cx = (bx1 + bx2) / 2  # 框中心 x
        if direction == "any":
            # 框压着线（左边界 ≤ 检测线 ≤ 右边界），或者框中心已经过线
            crossed = (bx1 <= detect_x <= bx2) or (cx <= detect_x)
        elif direction == "right":
            # 框整个在检测线右边，车还没过
            crossed = bx1 > detect_x
        elif direction == "left":
            # 框在检测线左边或者压线，车已经过了或在过
            crossed = bx2 >= detect_x
        else:
            crossed = (bx1 <= detect_x <= bx2) or (cx <= detect_x)
        if crossed:
            return True, {"x1": bx1, "y1": float(y1), "x2": bx2, "y2": float(y2),
                          "conf": float(conf), "cx": cx}
    return False, None


def _check_depth_drop(pixel_depths, detector):
    """看深度有没有骤降，YOLO 失灵时拿它当备用证据。
    判法：15 点深度的 min 跟 baseline 的 min 比，绝对差 ≥0.4m 且相对 ≥8%。
    返回 True 就是骤降了，可能有过车。
    """
    if not pixel_depths or not detector:
        return False
    valid = [d for d in pixel_depths if d is not None]
    if not valid:
        return False
    d_min = min(valid)
    bl = getattr(detector, "baseline", None)
    if bl is None or not len(bl):
        return False
    bl_arr = np.asarray(bl, dtype=np.float32)
    bl_min = float(np.nanmin(bl_arr))
    if bl_min <= 0:
        return False
    drop_abs = bl_min - d_min
    drop_rel = drop_abs / bl_min
    return drop_abs >= 0.4 and drop_rel >= 0.08


def should_sample_by_type(label_override, v8_state, yolo_boxes, pixel_points,
                          pixel_depths=None, detector=None):
    """
    按样本类型挑帧：只取那些"有问题"的片段，别把判对的帧也掺进来。
    返回: (should_sample: bool, corrected_label: str|None, reason: str)
      - should_sample=False → 这帧不要，不进训练集
      - corrected_label 不是 None → 用它盖掉 v8 的输出
      - reason 是给日志看的，方便回头查为什么取这帧
    pixel_depths/detector: 给标准样本判深度骤降用（不再看 v8 输出）
    """
    detect_x = pixel_points[0][0] if pixel_points else 40

    if label_override is None:
        # 标准样本：靠 YOLO+深度自己打标签（不看 v8）
        crossed, _ = car_crossed_line(yolo_boxes, detect_x, direction="any")
        # 判深度骤降，拿原始深度点和 baseline 比
        depth_drop = False
        valid = [d for d in (pixel_depths or []) if d is not None]
        d_min_local = min(valid) if valid else 0.0
        bl = getattr(detector, "baseline", None) if detector else None
        if bl is not None and len(bl) and valid:
            bl_min_local = float(np.nanmin(np.asarray(bl, dtype=np.float32)))
            drop_abs_local = bl_min_local - d_min_local if bl_min_local > 0 else 0.0
            drop_rel_local = (drop_abs_local / bl_min_local) if bl_min_local > 0 else 0.0
            depth_drop = (drop_abs_local >= 0.4 and drop_rel_local >= 0.08)
        if crossed or depth_drop:
            return True, "car", "标准-车过线/深度骤降(自动标car)"
        else:
            return True, "no_car", "标准-无车证据(自动标no_car)"

    if label_override == "no_car":
        # 无车误识：只取"车没过线、可 v8 报了 car"的帧
        # 判法：YOLO 没框，或者框在检测线右边（车还在右边没过线）
        if v8_state != "car":
            # v8 自己也判 no_car/unknown → 这帧 v8 可能判对了，照样取、标签保持 v8
            return True, None, "无车误识-v8正确帧(保持)"
        # v8 判了 car，那就查车到底过线没有（YOLO + 深度兜底）
        crossed, _ = car_crossed_line(yolo_boxes, detect_x, direction="any")
        depth_drop = _check_depth_drop(pixel_depths, detector)
        if not crossed and not depth_drop:
            # YOLO 没框/框在右边 且 深度也没骤降 → v8 误报，取这帧并改成 no_car
            return True, "no_car", "无车误识-车未过线v8误报car(纠正)"
        else:
            # 车真过线了或者深度骤降 → v8 判 car 可能没错，照样取、保持 car（别污染）
            return True, None, "无车误识-车真过线v8正确(保持)"

    if label_override == "car":
        # 闪烁：只取"车过线了、可 v8 判了 no_car"的帧
        # 判法：YOLO 有框且框越过检测线（车到线了），或者深度骤降兜底
        if v8_state != "no_car":
            # v8 自己也判 car/unknown → 这帧 v8 可能判对了，照样取
            return True, None, "闪烁-v8正确帧(保持)"
        # v8 判了 no_car，那就查车到底过线没有（YOLO + 深度兜底）
        crossed, _ = car_crossed_line(yolo_boxes, detect_x, direction="left")
        depth_drop = _check_depth_drop(pixel_depths, detector)
        if crossed or depth_drop:
            # 车过线了或者深度骤降 → v8 漏判，取这帧并改成 car
            return True, "car", "闪烁-车过线v8漏判no_car(纠正)"
        else:
            # 车没过线且深度没骤降 → v8 判 no_car 可能没错，照样取
            return True, None, "闪烁-车未过线v8正确(保持)"

    if label_override == "car_miss":
        # 有车不识：车来了可 v8 漏判（v8 说 no_car/unknown，真值是 car）
        # 判法：只要 YOLO 有框 或 深度骤降 → 认定车来了，改成 car
        # v8 判 car 时保持（它可能判对了）
        if v8_state == "car":
            return True, None, "有车不识-v8正确判car(保持)"
        crossed, _ = car_crossed_line(yolo_boxes, detect_x, direction="any")
        depth_drop = _check_depth_drop(pixel_depths, detector)
        if crossed or depth_drop:
            return True, "car", "有车不识-车来了v8漏判(纠正car)"
        # 没证据 → v8 判 no_car 可能没错，照样取
        return True, None, "有车不识-无车证据v8正确(保持)"

    if label_override == "yolo_blind":
        # yolo失灵：YOLO 漏检了，只能靠深度兜底
        # 这类样本 YOLO 不会有框，所以只用深度骤降当证据
        # v8 判 car → 保持（它可能靠深度判对了）
        # v8 判 no_car 但深度骤降 → 改成 car（YOLO 漏检，但深度说有车）
        # v8 判 no_car 且深度没骤降 → 保持 no_car
        if v8_state == "car":
            return True, None, "yolo失灵-v8靠深度判car(保持)"
        depth_drop = _check_depth_drop(pixel_depths, detector)
        if v8_state == "no_car" and depth_drop:
            return True, "car", "yolo失灵-深度骤降v8漏判(纠正car)"
        return True, None, "yolo失灵-无深度证据v8正确(保持)"

    if label_override == "yolo_recovered":
        # 克服yolo失灵：YOLO 恢复后判对了，直接用 v8 输出（蒸馏）
        return True, None, "克服yolo失灵-用v8输出(蒸馏)"

    # 其他策略就全取
    return True, None, "其他-全部取样"


# 视频配对：从标准视频反推原始视频
def find_standard_pairs(std_dir, input_root, label_override=None):
    """
    扫 std_dir 下的 *_depth_video.mp4，去掉后缀就是原始视频名，
    再去 input_root 里递归找同名的原始视频（扩展名不限）。
    label_override: 不是 None 的话（"car"/"no_car"/"unknown"），这个目录所有帧的标签
    都按这个类别算（给"无车误识"这类错样本用：v8 判错，拿真值盖掉）。
    返回 [(video_name, original_path, label_override), ...]
    """
    if not os.path.isdir(std_dir):
        raise FileNotFoundError(f"标准视频目录不存在: {std_dir}")
    exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    # 把 input 根目录下所有视频建成 stem → 路径 的索引
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
    递归扫 sample_root 下所有子目录里的 *_depth_video.mp4，按子目录名猜标签策略：
      - label_overrides: dict {子目录名: "v8"|"car"|"no_car"|"unknown"}
        "v8" 就是拿 v8 输出当伪标签（正确样本），别的值强制覆盖标签（错样本纠正）。
      - 没手动指定时按名字猜：含"误识"→no_car，含"闪烁"→car，含"标准"→v8，其余→v8
    返回 [(video_name, original_path, label_override|None), ...]
    """
    label_overrides = label_overrides or {}
    pairs = []
    if not os.path.isdir(sample_root):
        raise FileNotFoundError(f"样本根目录不存在: {sample_root}")
    # 一次性把 input_root 的视频建成索引（同名取第一个，跟以前逐个查是一个效果）
    exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    inp_index = {}
    for dp, _, ns in os.walk(input_root):
        for n in ns:
            if n.lower().endswith(exts):
                stem = os.path.splitext(n)[0]
                if stem not in inp_index:
                    inp_index[stem] = os.path.join(dp, n)
    for sub_dp, _, files in os.walk(sample_root):
        sub_name = os.path.basename(sub_dp) or os.path.basename(sample_root)
        # 猜这个子目录用哪种标签策略（六类样本各不一样）
        strategy = label_overrides.get(sub_name)
        if strategy is None:
            if "误识" in sub_name:
                strategy = "no_car"          # 无车误识：v8 误报 car，真值是 no_car
            elif "闪烁" in sub_name:
                strategy = "car"             # 闪烁：本该一直是 car，真值是 car
            elif "有车不识" in sub_name:
                strategy = "car_miss"        # 有车不识：车来了 v8 漏判，真值是 car
            elif "克服" in sub_name and "yolo" in sub_name:
                strategy = "yolo_recovered"  # 克服yolo失灵：YOLO恢复后判对，用 v8 输出
            elif "yolo" in sub_name and "失灵" in sub_name:
                strategy = "yolo_blind"       # yolo失灵：YOLO漏检，靠深度兜底
            else:
                strategy = "v8"               # 标准：靠 YOLO+深度自己打标（不看 v8）
        # override 传给 should_sample_by_type：None=标准样本自己打标，其他=策略名字符串
        override = None if strategy == "v8" else strategy
        for f in sorted(files):
            if not f.endswith("_depth_video.mp4"):
                continue
            stem = os.path.splitext(f)[0].replace("_depth_video", "")
            found_path = inp_index.get(stem)
            if found_path:
                pairs.append((stem, found_path, override))
    return pairs


# 检测器配置构建：跟 detect_car.py 保持一致，这样伪标签才复现得出来
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
        # 一车一信号 + 夜间专属屏蔽透传（与 detect.py build_detector_config 对齐）
        "saturation_threshold": int(car_cfg.get("saturation_threshold", 0)),
        "sat_night_only": bool(car_cfg.get("sat_night_only", True)),
        "spike_abs_threshold": float(car_cfg.get("spike_abs_threshold", 8.0)),
        "min_car_hold_frames": int(car_cfg.get("min_car_hold_frames", 10)),
        "min_signal_interval_frames": int(car_cfg.get("min_signal_interval_frames", 15)),
        "car_exit_cooldown_frames": int(car_cfg.get("car_exit_cooldown_frames", 15)),
        "baseline_rescue_frames": int(car_cfg.get("baseline_rescue_frames", 30)),
        "baseline_up_snap_frames": int(car_cfg.get("baseline_up_snap_frames", 10)),
        "baseline_down_snap_frames": int(car_cfg.get("baseline_down_snap_frames", 10)),
        "night": car_cfg.get("night", {}) or {},
        "yolo": {
            "enabled": bool(yolo_cfg.get("enabled", True)),
            "model_path": _abs_path(root, str(yolo_cfg.get("model_path", "checkpoints/yolo/best.pt"))),
            "device": str(yolo_cfg.get("device", "") or ""),
            "confidence": float(yolo_cfg.get("confidence", 0.40)),
            "iou": float(yolo_cfg.get("iou", 0.45)),
            "vehicle_classes": list(yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
            "enter_bottom_ratio": float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
            "min_box_ratio": float(yolo_cfg.get("min_box_ratio", 0.02)),
            "cross_tolerance_px": float(yolo_cfg.get("cross_tolerance_px", 15)),
        },
        "rule_weights": rule_weights,
        "mog2": mog2_cfg,
    }


def make_detectors(dc, fps, pixel_points):
    """按 detector_config 造 VehicleDetector / MotionLineDetector / BrightnessMonitor 这三个"""
    yolo_cfg = dc["yolo"]
    mog2_cfg = dc["mog2"]
    night_cfg = dc.get("night", {}) or {}
    night_enabled = bool(night_cfg.get("enabled", True))
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
        yolo_detect_x=(pixel_points[0][0] if pixel_points else 40),
        yolo_cross_tolerance=float(yolo_cfg.get("cross_tolerance_px", 15)),
        rule_weights=dc["rule_weights"],
        car_confidence_threshold=dc["car_confidence_threshold"],
        # 夜间自适应 + 一车一信号三件套：与 detect.py build_detector_pipeline 对齐
        night_bright_threshold=float(night_cfg.get("bright_threshold", 80.0)),
        night_smooth_alpha=float(night_cfg.get("smooth_alpha", 0.35)) if night_enabled else 0.0,
        night_confirm_extra=int(night_cfg.get("confirm_extra", 2)) if night_enabled else 0,
        night_cooldown_extra=int(night_cfg.get("cooldown_extra", 20)) if night_enabled else 0,
        min_car_hold_frames=int(dc.get("min_car_hold_frames", 10)),
        min_signal_interval_frames=int(dc.get("min_signal_interval_frames", 15)),
        spike_abs_threshold=float(dc.get("spike_abs_threshold", 8.0)),
        car_exit_cooldown_frames=int(dc.get("car_exit_cooldown_frames", 15)),
        baseline_rescue_frames=int(dc.get("baseline_rescue_frames", 30)),
        baseline_up_snap_frames=int(dc.get("baseline_up_snap_frames", 10)),
        baseline_down_snap_frames=int(dc.get("baseline_down_snap_frames", 10)),
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


# 单帧特征向量构建（FEATURE_DIM 维）
def build_feature(pixel_depths, detector, motion_detector, brightness_monitor,
                  prev_d_min, prev_depth_conf, prev_state_idx):
    """
    拼一帧的特征向量（numpy float32，长度 FEATURE_DIM）。
    分这几组：
      A. 深度统计(7): d_min/d_max/d_mean/d_std + baseline_min/baseline_mean + drop_abs
      B. YOLO(4): max_conf + n_boxes + 最左框x1 + 最右框x2
      C. MOG2/亮度(4): motion_pixels + confirmed + brightness_val + mog2_weight
      D. 过线/骤降(3): depth_drop_flag + yolo_crossed + yolo_nearest_cx
      E. 时序(4): prev_d_min + prev_depth_conf + drop_rel + depth_diff_rate
      F. 前状态 one-hot(3): prev_state
    缺的值填 0，保证维度恒定。
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
    # YOLO 框的 x 坐标，替掉 v8 那套内部信号，让模型自己判过没过线
    yolo_min_x1 = min((float(b[0]) for b in yolo_boxes), default=9999.0)  # 最左框左边界
    yolo_max_x2 = max((float(b[2]) for b in yolo_boxes), default=0.0)     # 最右框右边界
    yolo_nearest_cx = min((abs(((float(b[0]) + float(b[2])) / 2) - 40) for b in yolo_boxes), default=9999.0)  # 框中心到检测线距离

    motion_px = float(getattr(motion_detector, "motion_pixel_count", 0) if motion_detector else 0)
    mog2_confirmed = 1.0 if (motion_detector and motion_detector.is_confirmed()) else 0.0
    bright_val = 0.0
    if brightness_monitor:
        # 优先用 smoothed_brightness（0-255 的 float），取不到就拿 level 字符串映射
        sb = getattr(brightness_monitor, "smoothed_brightness", None)
        if sb is not None:
            bright_val = float(sb)
        else:
            lvl_map = {"daytime": 2.0, "twilight": 1.0, "night": 0.0}
            lvl = brightness_monitor.get_brightness_level()
            bright_val = lvl_map.get(lvl, 0.0) if lvl else 0.0
    mog2_w = float(motion_detector.get_current_weight() if motion_detector and hasattr(motion_detector, "get_current_weight") else 1.0)

    # 深度骤降标志，替掉 v8 的 depth_conf，用原始信号让模型自己学
    depth_drop_flag = 1.0 if (drop_abs >= 0.4 and drop_rel >= 0.08) else 0.0
    # YOLO 过线标志，替掉 v8 的 r5，用框坐标让模型自己学
    yolo_crossed = 1.0 if car_crossed_line(yolo_boxes, detect_x=40, direction="any")[0] else 0.0

    p_dmin = float(prev_d_min) if prev_d_min is not None else d_min
    p_conf = float(prev_depth_conf) if prev_depth_conf is not None else 0.0
    depth_diff_rate = (d_min - p_dmin) / p_dmin if p_dmin > 0 else 0.0

    prev_oh = [0.0, 0.0, 0.0]
    if 0 <= prev_state_idx < 3:
        prev_oh[prev_state_idx] = 1.0

    feat = [
        d_min, d_max, d_mean, d_std, bl_min, bl_mean, drop_abs,   # A 7
        yolo_max_conf, yolo_n, yolo_min_x1, yolo_max_x2,           # B 4 (原2+新增2坐标)
        motion_px, mog2_confirmed, bright_val, mog2_w,             # C 4
        depth_drop_flag, yolo_crossed, yolo_nearest_cx,           # D 3 (原2替换+新增1)
        p_dmin, p_conf, drop_rel, depth_diff_rate,                # E 4
        prev_oh[0], prev_oh[1], prev_oh[2],                       # F 3
    ]
    # 兜底，保证维度恒定，防止中间算错
    if len(feat) < FEATURE_DIM:
        feat += [0.0] * (FEATURE_DIM - len(feat))
    elif len(feat) > FEATURE_DIM:
        feat = feat[:FEATURE_DIM]
    return np.asarray(feat, dtype=np.float32)


# 特征提取：跑一遍深度推断 + v8，逐帧记下特征和伪标签
def extract_features_for_video(model, processor, video_path, device, dc,
                                pixel_points, pbar_desc="", label_override=None):
    """
    对单个视频逐帧提，返回 (feature, label_idx, frame_idx) 的列表。
    label_override: 不是 None 的话（"car"/"no_car"/"unknown"），标签强制按这个类别算，
    给"无车误识/闪烁"这类错样本用：v8 判错就拿真值盖掉，让模型学到 v8 出错的那条边界。
    默认 None → 用 v8 fuse 出来的最终 state 当伪标签（知识蒸馏）。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector, motion_detector, brightness_monitor = make_detectors(dc, fps, pixel_points)

    # 监测点过曝屏蔽：与 detect_car.py 同一套"夜间专属"逻辑，保证训练特征 = 线上推理特征
    sat_threshold = int(dc.get("saturation_threshold", 0))
    sat_night_only = bool(dc.get("sat_night_only", True))

    last_final_state = "unknown"
    prev_d_min = None
    prev_depth_conf = None
    prev_state_idx = STATE_TO_IDX["unknown"]

    records = []  # [(特征, 标签下标, 帧号), ...]
    sample_stats = {"sampled": 0, "skipped": 0, "corrected": 0}  # 取样统计
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

        # 屏蔽夜间专属：白天白车身像素顶到 255 也会被误杀成 nan，
        # 与 detect_car.py 完全一致（_is_night 取自上一帧，首帧按白天，线上同款行为）
        sat_now = sat_threshold if (not sat_night_only or getattr(detector, "_is_night", False)) else 0
        pixel_depths = _extract_pixel_depths(depth, pixel_points or [], frame, sat_now)

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
        # 标签：靠 YOLO 框坐标判"车过没过线"，只在有问题的帧上纠正
        yolo_boxes_frame = list(getattr(detector, "_last_yolo_boxes", []) or [])
        should_sample, corrected_label, reason = should_sample_by_type(
            label_override, state, yolo_boxes_frame, pixel_points,
            pixel_depths=pixel_depths, detector=detector
        )
        if not should_sample:
            # 这帧不要（跳过，不进训练集）
            sample_stats["skipped"] += 1
            pbar.update(1)
            frame_idx += 1
            continue

        # 标签默认用 v8 输出；should_sample 给了纠正标签就用它盖掉
        if corrected_label is not None:
            lbl = STATE_TO_IDX.get(corrected_label, 2)
            sample_stats["corrected"] += 1
        else:
            lbl = STATE_TO_IDX.get(state, 2)
        sample_stats["sampled"] += 1
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
    return records, fps, sample_stats


def _extract_videos(pairs, model, processor, device, dc, pixel_points_fn):
    """提一批视频的特征，返回 {name: (特征列表, 标签列表)} 的映射"""
    result = {}
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
            recs, fps, sstats = extract_features_for_video(
                model, processor, vp, device, dc, pts, pbar_desc=name,
                label_override=override)
            result[name] = ([f for f, l, fi in recs],
                            [l for f, l, fi in recs])
            sc = Counter(STATE_LIST[l] for _, l, _ in recs)
            print(f"  [Done] {len(recs)}帧(取样{sstats['sampled']}/跳过{sstats['skipped']}/纠正{sstats['corrected']}) "
                  f"| car={sc.get('car',0)} no_car={sc.get('no_car',0)} unk={sc.get('unknown',0)}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            continue
    return result


def extract_all(pairs, model, processor, device, dc, pixel_points_fn, cache_path,
                force=False):
    """
    把所有视频的特征都提出来，缓存到 cache_path（.npz）。
    支持增量：缓存里有的视频直接复用，只提新增的那些。
    pairs 里删掉的视频，缓存里也顺手清掉。

    pairs: [(name, path, label_override|None), ...]
    返回:
      features: (N, FEATURE_DIM) float32
      labels:   (N,) int64
      video_ids:(N,) int  —— 每帧属于哪个视频，按视频分组要用
      video_names: list[str]
    """
    current_names = [p[0] for p in pairs]
    current_name_set = set(current_names)

    # 强制提取，或者压根没缓存：全部从头来
    if force or not os.path.exists(cache_path):
        print(f"[Extract] 全量提取 {len(pairs)} 个视频...")
        new_data = _extract_videos(pairs, model, processor, device, dc, pixel_points_fn)
        if not new_data:
            raise RuntimeError("未提取到任何特征，检查视频/模型/监测点配置")
        return _build_and_save(new_data, pairs, cache_path)

    # 增量模式：读旧缓存，只提新视频
    data = np.load(cache_path, allow_pickle=True)
    old_features = data["features"]
    old_labels = data["labels"]
    old_video_ids = data["video_ids"]
    old_video_names = list(data["video_names"])
    old_name_set = set(old_video_names)
    data.close()

    new_pairs = [p for p in pairs if p[0] not in old_name_set]
    removed_names = [n for n in old_video_names if n not in current_name_set]
    kept_names = [n for n in old_video_names if n in current_name_set]

    if not new_pairs and not removed_names:
        print(f"[Cache] 缓存与当前数据集完全一致 ({len(kept_names)} 视频)，直接使用")
        return old_features, old_labels, old_video_ids, old_video_names

    print(f"[Cache] 增量模式: 保留 {len(kept_names)} | 新增 {len(new_pairs)} | 剔除 {len(removed_names)}")

    # 建个旧视频名 → (特征数组, 标签数组) 的映射
    old_by_name = {}
    for vid_idx, name in enumerate(old_video_names):
        mask = old_video_ids == vid_idx
        old_by_name[name] = (old_features[mask], old_labels[mask])

    # 提新视频的特征
    new_by_name = {}
    if new_pairs:
        if model is None or processor is None:
            raise RuntimeError("检测到新视频需要提取特征，但深度模型未加载。"
                               "请去掉 --use-cache 参数后重试。")
        print(f"[Extract] 提取新视频特征: {len(new_pairs)} 个...")
        new_by_name = _extract_videos(new_pairs, model, processor, device, dc, pixel_points_fn)

    # 合并：照当前 pairs 的顺序重建 video_ids
    all_feats, all_labels, all_vids = [], [], []
    video_names = []
    for vi, pair in enumerate(pairs):
        name = pair[0]
        if name in old_by_name:
            feats, labs = old_by_name[name]
        elif name in new_by_name:
            feats, labs = new_by_name[name]
        else:
            continue  # 提失败的，跳过

        for feat, label in zip(feats, labs):
            all_feats.append(feat)
            all_labels.append(label)
            all_vids.append(vi)
        video_names.append(name)

    if not all_feats:
        raise RuntimeError("未提取到任何特征，检查视频/模型/监测点配置")

    features = np.stack(all_feats).astype(np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    video_ids = np.asarray(all_vids, dtype=np.int64)
    video_names_arr = np.asarray(video_names, dtype=object)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels,
                        video_ids=video_ids, video_names=video_names_arr)
    print(f"\n[Cache] 更新特征缓存: {cache_path} | {len(features)}帧 × {features.shape[1]}维")
    return features, labels, video_ids, video_names


def _build_and_save(new_data, pairs, cache_path):
    """把提取结果拼成特征数组，顺手写进缓存"""
    all_feats, all_labels, all_vids = [], [], []
    video_names = []
    for vi, pair in enumerate(pairs):
        name = pair[0]
        if name not in new_data:
            continue
        feats, labs = new_data[name]
        for feat, label in zip(feats, labs):
            all_feats.append(feat)
            all_labels.append(label)
            all_vids.append(vi)
        video_names.append(name)

    features = np.stack(all_feats).astype(np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    video_ids = np.asarray(all_vids, dtype=np.int64)
    video_names_arr = np.asarray(video_names, dtype=object)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels,
                        video_ids=video_ids, video_names=video_names_arr)
    print(f"\n[Cache] 保存特征缓存: {cache_path} | {len(features)}帧 × {features.shape[1]}维")
    return features, labels, video_ids, video_names


# 时序窗口数据集：把前 W 帧特征拼起来
class JudgeDataset(Dataset):
    """
    滑动窗口数据集：在每个视频内部，把当前帧跟前面 W-1 帧的特征拼成一条
    长样本（W*FEATURE_DIM 维）。不跨视频拼，跨视频的时序没意义。
    标签就用当前帧的。
    """
    def __init__(self, features, labels, video_ids, window=1):
        self.window = window
        self.samples = []  # [(窗口特征 (W*D,), 标签)]
        # 按视频把帧索引分好组
        vid_to_frames = defaultdict(list)
        for i, vid in enumerate(video_ids):
            vid_to_frames[int(vid)].append(i)
        for vid, frames in vid_to_frames.items():
            frames_sorted = sorted(frames)
            for pos, fi in enumerate(frames_sorted):
                lo = max(0, pos - window + 1)
                window_frames = frames_sorted[lo:pos + 1]
                # 不够 W 帧就在前面补 0
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


# 模型：MLP
class JudgeMLP(nn.Module):
    """
    输入: (batch, window * FEATURE_DIM)
    输出: (batch, 3) 的 logits —— car / no_car / unknown
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


# 训练 + 验证
def train_model(train_loader, val_loader, in_dim, device, epochs=100,
               lr=1e-3, weight_decay=1e-4, patience=15, class_weights=None,
               save_path=None, verbose=True, dropout=0.3):
    """
    训 MLP，带早停。返回 (best_model, history_dict)。
    history: {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    """
    model = JudgeMLP(in_dim=in_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 类权重，用来对付不平衡（car 帧多、no_car 帧少就加权）
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
        # 训练
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

        # 验证
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


# 评估：分类报告 + 混淆矩阵
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


# 主流程
def main():
    # 读配置：judge 段给训练超参的默认值，命令行能覆盖
    root = os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config()
    judge_cfg = cfg.get("judge", {}) or {}
    detect_cfg = cfg.get("detect", {}) or {}

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
    parser.add_argument("--epochs", type=int, default=judge_cfg.get("epochs", 100), help="最大训练轮数")
    parser.add_argument("--batch", type=int, default=judge_cfg.get("batch", 256), help="batch size")
    parser.add_argument("--window", type=int, default=judge_cfg.get("window", WINDOW_DEFAULT),
                        help="时序窗口帧数（1=单帧，5=前5帧拼接）")
    parser.add_argument("--lr", type=float, default=judge_cfg.get("lr", 1e-3), help="学习率")
    parser.add_argument("--weight-decay", type=float, default=judge_cfg.get("weight_decay", 1e-4),
                        help="AdamW 权重衰减")
    parser.add_argument("--patience", type=int, default=judge_cfg.get("patience", 15), help="早停耐心")
    parser.add_argument("--dropout", type=float, default=judge_cfg.get("dropout", 0.3), help="Dropout")
    parser.add_argument("--seed", type=int, default=judge_cfg.get("seed", 42), help="随机种子")
    parser.add_argument("--use-cache", action="store_true", help="用已有特征缓存（跳过提取）")
    parser.add_argument("--force-extract", action="store_true", help="强制重新提取特征")
    parser.add_argument("--val-video-frac", type=float, default=judge_cfg.get("val_video_frac", 0.2),
                        help="验证视频占比（按视频分组，默认20%）")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 路径
    std_dir = args.std_dir or os.path.join(root, "detect", "样本", "标准")
    sample_root = args.sample_root or os.path.join(root, "detect", "样本")
    input_root = args.input_root or _abs_path(root, detect_cfg.get("input_dir", "detect/input"))
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "feature_cache.npz")
    model_out = args.model_out or os.path.join(root, "checkpoints", "judge_model", "judge_mlp.pt")

    # 解析 label_overrides 这个字符串
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

    # 配对视频
    # 优先用 --sample-root（递归多目录，按子目录名猜标签策略）；
    # 没指定 sample-root、或者它不存在，就退回 --std-dir 单目录
    pairs = []
    if os.path.isdir(sample_root):
        pairs = find_pairs_from_sample_root(sample_root, input_root, label_overrides)
        mode = f"递归样本根目录 {sample_root}（按子目录名推断标签策略）"
    else:
        pairs = find_standard_pairs(std_dir, input_root)
        mode = f"单目录 {std_dir}（全部用 v8 伪标签）"
    print(f"[Pair] {mode}")
    print(f"[Pair] 匹配到 {len(pairs)} 个样本视频 ↔ 原始视频")
    # 数一下各标签策略各有多少
    strat_cnt = Counter((p[2] or "v8") for p in pairs)
    print(f"[Pair] 标签策略分布: " + " ".join(f"{k}={v}" for k, v in sorted(strat_cnt.items())))
    if not pairs:
        print("[ERROR] 未匹配到视频。检查 sample-root/std-dir 和 input-root 是否正确，"
              "以及标准视频命名是否为 xxx_depth_video.mp4")
        sys.exit(1)

    # 监测点解析函数
    ppoints_str = str(detect_cfg.get("pixel_depth_points", "") or "").strip()
    pcount = int(detect_cfg.get("pixel_depth_count", 5))
    print(f"[Pixel] pixel_depth_points='{ppoints_str}' pixel_depth_count={pcount}")

    def pixel_points_fn(w, h):
        return _parse_pixel_points(ppoints_str, pcount, w, h)

    # 特征提取
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    dc = build_detector_config(cfg)

    # 判断要不要加载深度模型：
    # --use-cache      → 不加载（纯调参）
    # --force-extract  → 加载（全量重提）
    # 没缓存           → 加载（首次提取）
    # 有缓存 + 有新视频 → 加载（增量提取）
    # 有缓存 + 没变化   → 不加载（全走缓存）
    need_model = args.force_extract or not os.path.exists(cache_path)
    if not need_model and not args.use_cache:
        _data = np.load(cache_path, allow_pickle=True)
        _old_names = set(_data["video_names"])
        _data.close()
        _new_count = sum(1 for p in pairs if p[0] not in _old_names)
        need_model = _new_count > 0
        if not need_model:
            print(f"[Cache] 无新视频，跳过深度模型加载")

    if need_model:
        print("[Model] 加载深度模型...")
        model, processor = load_model(cfg.get("model", {}).get("student", DEFAULT_MODEL), device)
    else:
        model = processor = None

    features, labels, video_ids, video_names = extract_all(
        pairs, model, processor, device, dc, pixel_points_fn,
        cache_path, force=args.force_extract,
    )
    print(f"\n[Data] 总帧数: {len(features)} | 特征维度: {features.shape[1]}")
    label_dist = Counter(labels.tolist())
    print(f"[Data] 标签分布: " + " ".join(f"{STATE_LIST[k]}={v}" for k, v in sorted(label_dist.items())))

    # 按视频分组切 train/val
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

    # 类权重
    train_labels = labels[train_mask]
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    class_weights = counts.sum() / (3.0 * counts)
    print(f"[Weight] 类权重: " + " ".join(f"{STATE_LIST[k]}={class_weights[k]:.3f}" for k in range(3)))

    # 训练
    print("\n[Train] 开始训练...")
    t0 = time.time()
    best_model, history, best_epoch = train_model(
        train_loader, val_loader, in_dim, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, dropout=args.dropout,
        class_weights=class_weights, save_path=model_out,
    )
    train_sec = time.time() - t0
    print(f"[Train] 完成 | 耗时 {train_sec:.1f}s | best epoch={best_epoch}")

    # 评估
    y_true, y_pred = evaluate_model(best_model, val_loader, device)
    report = classification_report_txt(y_true, y_pred, STATE_LIST)
    acc = float((y_true == y_pred).mean())

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    # 保存报告和图
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

    # metrics.json
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
