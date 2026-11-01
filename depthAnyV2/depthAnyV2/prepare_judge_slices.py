"""
智能切片分类 + 特征提取脚本（图片/视频混合数据源）。
把 detect/样本/ 下各种子目录的切片统一处理，输出能直接喂给 train_judge.py 的特征缓存。

和 train_judge.py 的区别：
  - train_judge.py 只认 *_depth_video.mp4，还得去 detect/input 找原始视频配对
  - 这脚本同时支持图片（jpg/png 等）和视频（depth_video / 原始 mp4 都能直接处理）
  - 输出 feature_cache.npz 格式完全兼容 train_judge.py，后者 --use-cache 直接训
  - 支持增量缓存：只提新增的图片/视频，旧缓存复用

视频处理：逐帧跑深度 + YOLO + MOG2，按子目录名对应的策略筛选帧（复用 train_judge.py
的 should_sample_by_type），支持 --frame-step 采样间隔省算力。
图片处理：直接跑一遍深度 + YOLO + MOG2 提特征，标签由子目录名直接决定（不做帧级筛选）。

跑法：
  python prepare_judge_slices.py                                    # 全量（或增量）提特征
  python prepare_judge_slices.py --sample-root detect/样本 --frame-step 2
  python prepare_judge_slices.py --force --output-dir judge_training_output
  python train_judge.py --use-cache                                 # 用缓存直接训
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
import torch.nn.functional as F
from tqdm import tqdm

# ---- import detect 包 ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect"))
from detect import (
    load_model, _load_config, _abs_path, _parse_pixel_points, DEFAULT_MODEL,
    _extract_pixel_depths,
)

# ---- import train_judge.py 的共享函数（常量 + 检测器构建 + 特征构建 + 帧筛选策略）----
# train_judge.py 不会反向 import prepare_judge_slices，所以无循环依赖
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from train_judge import (
    STATE_LIST, STATE_TO_IDX, FEATURE_DIM,
    build_detector_config as _tj_build_detector_config,
    make_detectors as _tj_make_detectors,
    build_feature,
    should_sample_by_type as _tj_should_sample_by_type,
)

# ---- import detect_car ----
from detect_car import fuse_v8_car_state


# =============================================================================
# 文件类型常量
# =============================================================================

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}


# =============================================================================
# 目录名 → 标签策略映射
# =============================================================================

def _guess_strategy(sub_name):
    """根据子目录名猜 (视频 label_override, 图片标签策略)。

    视频 override=None → 用 v8 输出当伪标签；其他值=纠正策略名
    图片策略 "direct" → 直接用覆盖标签；"v8" → 跑一遍 v8 打标签
    """
    if "误识" in sub_name:
        return "no_car", "direct"
    if "闪烁" in sub_name:
        return "car", "direct"
    if "有车不识" in sub_name:
        return "car_miss", "direct"
    if "克服" in sub_name and "yolo" in sub_name:
        return "yolo_recovered", "v8"
    if "yolo" in sub_name and "失灵" in sub_name:
        return "yolo_blind", "direct"
    return None, "v8"  # 标准/其他：视频用 v8 蒸馏，图片也跑 v8


def _image_direct_label(sub_name):
    """图片 direct 策略下的直接标签值。"""
    if "误识" in sub_name:
        return "no_car"
    if "闪烁" in sub_name or "有车不识" in sub_name or ("yolo" in sub_name and "失灵" in sub_name):
        return "car"
    return "unknown"


# =============================================================================
# 薄包装：复用 train_judge.py 的函数，跟本脚本的 dc 结构对齐
# =============================================================================

def build_detector_config(cfg):
    return _tj_build_detector_config(cfg)


def make_detectors(dc, fps, pixel_points):
    return _tj_make_detectors(dc, fps, pixel_points)


# =============================================================================
# 样本目录扫描
# =============================================================================

def scan_samples(sample_root, input_root):
    """扫 sample_root 下所有子目录，返回 (video_items, image_items)。

    video_items: [(stem, video_path, label_override_or_None, sub_dir_name), ...]
    image_items: [(stem, image_path, image_label_strategy, direct_label, sub_dir_name), ...]
    """
    video_items = []
    image_items = []

    inp_index = {}
    if os.path.isdir(input_root):
        for dp, _, ns in os.walk(input_root):
            for n in ns:
                ext = os.path.splitext(n)[1].lower()
                if ext in VIDEO_EXTS:
                    stem = os.path.splitext(n)[0]
                    if stem not in inp_index:
                        inp_index[stem] = os.path.join(dp, n)

    for sub_dp, _, files in os.walk(sample_root):
        sub_name = os.path.basename(sub_dp) or os.path.basename(sample_root)
        label_override, image_strategy = _guess_strategy(sub_name)
        direct_label = _image_direct_label(sub_name)

        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            fpath = os.path.join(sub_dp, f)

            if f.endswith("_depth_video.mp4"):
                stem = os.path.splitext(f)[0].replace("_depth_video", "")
                orig_path = inp_index.get(stem)
                video_path = orig_path if orig_path else fpath
                video_items.append((stem, video_path, label_override, sub_name))

            elif ext in VIDEO_EXTS:
                stem = os.path.splitext(f)[0]
                video_items.append((stem, fpath, label_override, sub_name))

            elif ext in IMAGE_EXTS:
                stem = os.path.splitext(f)[0]
                image_items.append((stem, fpath, image_strategy, direct_label, sub_name))

    return video_items, image_items


# =============================================================================
# 视频特征提取（加了 frame_step 支持）
# =============================================================================

def extract_video(model, processor, video_path, device, dc, pixel_points,
                  frame_step, label_override, pbar_desc=""):
    """对单个视频逐帧提特征。

    返回 records: [(feat, label_idx, frame_idx, confidence), ...]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_step = max(1, int(frame_step))
    effective_total = (total + frame_step - 1) // frame_step

    detector, motion_detector, brightness_monitor = make_detectors(dc, fps, pixel_points)

    last_final_state = "unknown"
    prev_d_min = None
    prev_depth_conf = None
    prev_state_idx = STATE_TO_IDX["unknown"]

    clahe_enabled = bool(dc.get("clahe", False))
    clahe_clip = float(dc.get("clahe_clip", 2.0))
    clahe_tile = int(dc.get("clahe_tile", 8))

    # 监测点过曝屏蔽：与 detect_car.py 同一套"夜间专属"逻辑，保证训练特征 = 线上推理特征
    sat_threshold = int(dc.get("saturation_threshold", 0))
    sat_night_only = bool(dc.get("sat_night_only", True))

    records = []
    sample_stats = {"sampled": 0, "skipped": 0, "corrected": 0}
    frame_idx = 0
    pbar = tqdm(total=effective_total, desc=pbar_desc[:20], ncols=110,
                leave=True, mininterval=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        if clahe_enabled:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=clahe_clip,
                                            tileGridSize=(clahe_tile, clahe_tile)).apply(lab[:, :, 0])
            frame_infer = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            frame_infer = frame

        img_rgb = cv2.cvtColor(frame_infer, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[img_rgb], return_tensors="pt").to(device)
        with torch.no_grad():
            depth = model(**inputs).predicted_depth
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h, w), mode="bicubic",
                              align_corners=False).squeeze().cpu().numpy()

        # 屏蔽夜间专属：白天白车身像素顶到 255 也会被误杀成 nan，
        # 与 detect_car.py 完全一致（_is_night 取自上一帧，首帧按白天，线上同款行为）
        sat_now = sat_threshold if (not sat_night_only or getattr(detector, "_is_night", False)) else 0
        pixel_depths = _extract_pixel_depths(depth, pixel_points or [], frame, sat_now)

        car_state = None
        if pixel_depths:
            car_state = detector.update(pixel_depths, frame_infer)
        if motion_detector:
            motion_detector.update(frame_infer)

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

        yolo_boxes_frame = list(getattr(detector, "_last_yolo_boxes", []) or [])
        should_sample, corrected_label, reason = _tj_should_sample_by_type(
            label_override, state, yolo_boxes_frame, pixel_points,
            pixel_depths=pixel_depths, detector=detector,
        )

        if not should_sample:
            sample_stats["skipped"] += 1
            frame_idx += 1
            pbar.update(1)
            continue

        if corrected_label is not None:
            lbl = STATE_TO_IDX.get(corrected_label, 2)
            sample_stats["corrected"] += 1
        else:
            lbl = STATE_TO_IDX.get(state, 2)
        sample_stats["sampled"] += 1

        conf = float(getattr(detector, "last_confidence", 0.0) or 0.0)
        records.append((feat, lbl, frame_idx, conf))

        valid = [d for d in pixel_depths if d is not None]
        if valid:
            prev_d_min = float(min(valid))
        prev_depth_conf = conf
        prev_state_idx = STATE_TO_IDX.get(state, 2)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return records, fps, sample_stats


# =============================================================================
# 图片特征提取（单帧，时序特征退化）
# =============================================================================

def extract_image(model, processor, image_path, device, dc, pixel_points,
                  label_strategy, direct_label, pbar_desc=""):
    """对单张图片提特征。返回 records: [(feat, label_idx, 0, confidence), ...]"""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    h, w = frame.shape[:2]
    fps = 25.0
    detector, motion_detector, brightness_monitor = make_detectors(dc, fps, pixel_points)

    clahe_enabled = bool(dc.get("clahe", False))
    clahe_clip = float(dc.get("clahe_clip", 2.0))
    clahe_tile = int(dc.get("clahe_tile", 8))

    if clahe_enabled:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=clahe_clip,
                                        tileGridSize=(clahe_tile, clahe_tile)).apply(lab[:, :, 0])
        frame_infer = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        frame_infer = frame

    img_rgb = cv2.cvtColor(frame_infer, cv2.COLOR_BGR2RGB)
    inputs = processor(images=[img_rgb], return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    depth = F.interpolate(depth, size=(h, w), mode="bicubic",
                          align_corners=False).squeeze().cpu().numpy()

    # 监测点过曝屏蔽：单图无前后帧，用本图亮度自判夜间（判定条件与 VehicleDetector 一致）
    sat_threshold = int(dc.get("saturation_threshold", 0))
    sat_night_only = bool(dc.get("sat_night_only", True))
    night_active = (detector.night_smooth_alpha > 0 or detector.night_confirm_extra > 0
                    or detector.night_cooldown_extra > 0)
    is_night = (night_active
                and detector._compute_frame_brightness(frame_infer) < detector.night_bright_threshold)
    sat_now = sat_threshold if (not sat_night_only or is_night) else 0
    pixel_depths = _extract_pixel_depths(depth, pixel_points or [], frame, sat_now)

    car_state = None
    if pixel_depths:
        car_state = detector.update(pixel_depths, frame_infer)
    if motion_detector:
        motion_detector.update(frame_infer)

    state, _ = fuse_v8_car_state(
        car_state=car_state, detector=detector, motion_detector=motion_detector,
        pixel_depths=pixel_depths,
        yolo_boxes=list(getattr(detector, "_last_yolo_boxes", []) or []),
        last_final_state="unknown",
    )

    feat = build_feature(
        pixel_depths, detector, motion_detector, brightness_monitor,
        prev_d_min=None, prev_depth_conf=None, prev_state_idx=2,
    )

    if label_strategy == "direct":
        lbl = STATE_TO_IDX.get(direct_label, 2)
    else:
        lbl = STATE_TO_IDX.get(state, 2)

    conf = float(getattr(detector, "last_confidence", 0.0) or 0.0)
    return [(feat, lbl, 0, conf)]


# =============================================================================
# 统一提取 + 增量缓存 + manifest 生成
# =============================================================================

def extract_all(model, processor, device, dc, pixel_points_fn,
                video_items, image_items, frame_step,
                cache_path, manifest_path, force=False):
    """统一处理视频 + 图片，输出 npz 缓存 + json manifest。"""

    # ---------- 读旧缓存，做增量 ----------
    old_by_name = {}
    if not force and os.path.exists(cache_path):
        try:
            old_data = np.load(cache_path, allow_pickle=True)
            old_feats = old_data["features"]
            old_labs = old_data["labels"]
            old_vids = old_data["video_ids"]
            old_names = list(old_data["video_names"])
            old_data.close()
            for vid_idx, name in enumerate(old_names):
                mask = old_vids == vid_idx
                old_by_name[name] = (old_feats[mask], old_labs[mask])
            print(f"[Cache] 读旧缓存: {len(old_by_name)} 个样本")
        except Exception as e:
            print(f"[Cache] 旧缓存读失败，全量重提: {e}")
            force = True

    # ---------- 视频 ----------
    video_results = {}  # stem → (feats, labs, confs, frame_indices)
    for stem, vp, override, sub_name in video_items:
        if not force and stem in old_by_name:
            feats, labs = old_by_name[stem]
            video_results[stem] = (feats, labs, np.zeros(len(feats), dtype=np.float32),
                                   np.zeros(len(feats), dtype=np.int64))
            continue
        tag = f"[{override or 'v8'}]" if override else ""
        try:
            cap_t = cv2.VideoCapture(str(vp))
            w = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_t.release()
            pts = pixel_points_fn(w, h)
            if not pts:
                print(f"  [WARN] {stem} 监测点解析失败，跳过")
                continue
            recs, fps, sstats = extract_video(
                model, processor, vp, device, dc, pts, frame_step,
                override, pbar_desc=f"{stem[:16]}{tag}",
            )
            if not recs:
                print(f"  [WARN] {stem} 没提取到任何帧（全部被策略跳过？）")
                continue
            video_results[stem] = (
                np.stack([r[0] for r in recs]).astype(np.float32),
                np.asarray([r[1] for r in recs], dtype=np.int64),
                np.asarray([r[3] for r in recs], dtype=np.float32),
                np.asarray([r[2] for r in recs], dtype=np.int64),  # 真实帧号
            )
            sc = Counter(STATE_LIST[r[1]] for r in recs)
            print(f"  [Done] {stem} {tag}: {len(recs)}帧 "
                  f"(取样{sstats['sampled']}/跳过{sstats['skipped']}/纠正{sstats['corrected']}) "
                  f"| car={sc.get('car',0)} no_car={sc.get('no_car',0)} unk={sc.get('unknown',0)}")
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            traceback.print_exc()
            continue

    # ---------- 图片 ----------
    image_results = {}
    for stem, ipath, label_strat, direct_label, sub_name in image_items:
        if not force and stem in old_by_name:
            feats, labs = old_by_name[stem]
            image_results[stem] = (feats, labs, np.zeros(len(feats), dtype=np.float32),
                                   np.zeros(len(feats), dtype=np.int64))
            continue
        try:
            frame = cv2.imread(str(ipath))
            if frame is None:
                print(f"  [WARN] {stem} 无法读取，跳过")
                continue
            h, w = frame.shape[:2]
            pts = pixel_points_fn(w, h)
            if not pts:
                print(f"  [WARN] {stem} 监测点解析失败，跳过")
                continue
            recs = extract_image(
                model, processor, ipath, device, dc, pts,
                label_strat, direct_label, pbar_desc=stem[:16],
            )
            image_results[stem] = (
                np.stack([r[0] for r in recs]).astype(np.float32),
                np.asarray([r[1] for r in recs], dtype=np.int64),
                np.asarray([r[3] for r in recs], dtype=np.float32),
                np.asarray([r[2] for r in recs], dtype=np.int64),
            )
            lbl_state = STATE_LIST[recs[0][1]]
            conf = recs[0][3]
            print(f"  [Done] 图片 {stem} → {lbl_state} (conf={conf:.2f})")
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            traceback.print_exc()
            continue

    # ---------- 合并 ----------
    all_feats, all_labels, all_vids, all_names = [], [], [], []
    slices_manifest = {"total_slices": 0, "categories": defaultdict(lambda: {"videos": 0, "images": 0}),
                       "slices": []}

    sample_idx = 0
    for stem, vp, override, sub_name in video_items:
        if stem not in video_results:
            continue
        feats, labs, confs, frame_idxs = video_results[stem]
        video_ids = np.full(len(feats), sample_idx, dtype=np.int64)
        all_feats.append(feats); all_labels.append(labs); all_vids.append(video_ids)
        all_names.append(stem)
        sample_idx += 1

        slices_manifest["categories"][sub_name]["videos"] += 1
        slices_manifest["slices"].append({
            "source": vp, "source_type": "video", "category": sub_name,
            "strategy": override or "v8",
            "frames": [
                {"frame_idx": int(frame_idxs[i]), "label": STATE_LIST[labs[i]],
                 "confidence": round(float(confs[i]), 4)}
                for i in range(len(feats))
            ],
        })

    for stem, ipath, label_strat, direct_label, sub_name in image_items:
        if stem not in image_results:
            continue
        feats, labs, confs, frame_idxs = image_results[stem]
        video_ids = np.full(len(feats), sample_idx, dtype=np.int64)
        all_feats.append(feats); all_labels.append(labs); all_vids.append(video_ids)
        all_names.append(stem)
        sample_idx += 1

        slices_manifest["categories"][sub_name]["images"] += 1
        label_state = STATE_LIST[labs[0]]
        slices_manifest["slices"].append({
            "source": ipath, "source_type": "image", "category": sub_name,
            "strategy": label_strat, "direct_label": direct_label,
            "frames": [{"frame_idx": int(frame_idxs[0]), "label": label_state,
                        "confidence": round(float(confs[0]), 4)}],
        })

    if not all_feats:
        raise RuntimeError("未提取到任何特征，检查样本目录/模型/监测点配置")

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    video_ids = np.concatenate(all_vids, axis=0).astype(np.int64)
    video_names_arr = np.asarray(all_names, dtype=object)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels,
                        video_ids=video_ids, video_names=video_names_arr)
    print(f"\n[Cache] 特征缓存: {cache_path} | {len(features)}帧 × {features.shape[1]}维 "
          f"| {len(all_names)}个样本")

    slices_manifest["total_slices"] = sum(len(s["frames"]) for s in slices_manifest["slices"])
    slices_manifest["run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slices_manifest["feature_dim"] = FEATURE_DIM
    slices_manifest["video_frame_step"] = frame_step

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(slices_manifest, f, ensure_ascii=False, indent=2)
    print(f"[Manifest] 结构化清单: {manifest_path}")

    return features, labels, video_ids, all_names


# =============================================================================
# main
# =============================================================================

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config()
    detect_cfg = cfg.get("detect", {}) or {}
    car_cfg = cfg.get("car_detect", {}) or {}

    parser = argparse.ArgumentParser(description="智能切片分类 + 特征提取（图片/视频混合）")
    parser.add_argument("--sample-root", type=str, default=None,
                        help="样本根目录（默认 detect/样本）")
    parser.add_argument("--input-root", type=str, default=None,
                        help="原始视频根目录，depth_video 会去这里找配对（默认 detect/input）")
    parser.add_argument("--model-depth", type=str,
                        default=_abs_path(root, detect_cfg.get("model_depth", DEFAULT_MODEL)))
    parser.add_argument("--device", type=str, default=detect_cfg.get("device", "cuda"))
    parser.add_argument("--output-dir", type=str, default="judge_training_output",
                        help="输出目录（默认 judge_training_output）")
    parser.add_argument("--frame-step", type=int, default=int(car_cfg.get("frame_step", 1)),
                        help="视频帧采样间隔（1=每帧，2=隔帧，省算力）")
    parser.add_argument("--pixel-depth-points", type=str,
                        default=str(detect_cfg.get("pixel_depth_points", "") or ""))
    parser.add_argument("--pixel-depth-count", type=int,
                        default=int(detect_cfg.get("pixel_depth_count", 5)))
    parser.add_argument("--force", action="store_true",
                        help="强制重新提取（忽略旧缓存，全量重提）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sample_root = args.sample_root or os.path.join(root, "detect", "样本")
    input_root = args.input_root or _abs_path(root, detect_cfg.get("input_dir", "detect/input"))
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(root, args.output_dir)
    cache_path = os.path.join(out_dir, "feature_cache.npz")
    manifest_path = os.path.join(out_dir, "slices_manifest.json")
    dc = build_detector_config(cfg)

    print("=" * 70)
    print("智能切片分类 + 特征提取（图片/视频混合数据源）")
    print("=" * 70)
    print(f"样本目录:   {sample_root}")
    print(f"原始视频:   {input_root}")
    print(f"输出目录:   {out_dir}")
    print(f"帧采样间隔: {args.frame_step}")
    print(f"设备:       {device}")
    print(f"强制重提:   {args.force}")

    if not os.path.isdir(sample_root):
        print(f"[ERROR] 样本目录不存在: {sample_root}")
        sys.exit(1)

    print(f"\n[Scan] 扫描 {sample_root} ...")
    video_items, image_items = scan_samples(sample_root, input_root)

    print(f"  视频: {len(video_items)} 个样本")
    categories_v = Counter(it[3] for it in video_items)
    for cat, cnt in sorted(categories_v.items()):
        label_override = _guess_strategy(cat)[0]
        tag = f"[{label_override or 'v8'}]" if label_override else "[v8]"
        print(f"    {cat}: {cnt} {tag}")

    print(f"  图片: {len(image_items)} 张")
    categories_i = Counter(it[4] for it in image_items)
    for cat, cnt in sorted(categories_i.items()):
        _video_override, img_strat = _guess_strategy(cat)
        direct_label = _image_direct_label(cat)
        label_tag = direct_label if img_strat == "direct" else "v8"
        print(f"    {cat}: {cnt} → {label_tag}")

    if not video_items and not image_items:
        print("[ERROR] 没找到任何视频或图片样本")
        sys.exit(1)

    print(f"\n[Model] 加载深度模型 {args.model_depth} ...")
    model, processor = load_model(args.model_depth, device)
    print(f"[Model] 已加载, device={device}")

    pixel_points_fn = lambda w, h: _parse_pixel_points(
        args.pixel_depth_points, args.pixel_depth_count, w, h)

    print(f"\n[Extract] 开始提取特征 ...")
    t0 = time.time()
    extract_all(model, processor, device, dc, pixel_points_fn,
                video_items, image_items, args.frame_step,
                cache_path, manifest_path, force=args.force)
    t1 = time.time()

    print(f"\n[Done] 全部完成，耗时 {t1 - t0:.1f}s")
    print(f"  → 训练特征缓存: {cache_path}  (train_judge.py --use-cache 直接训)")
    print(f"  → 结构化清单:    {manifest_path}")


if __name__ == "__main__":
    main()
