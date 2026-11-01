"""
视频挑选用批量检测脚本（v8 融合判定，与 detect_car.py 同一逻辑）
================================================================
输出完整带标注视频，画面与 detect 结构一致（左原图+右深度图+监测点米数+
状态条+MOG2 运动带），但状态判定用当前 v8 规则（fuse_v8 + YOLO 视觉协同），
原图额外画 YOLO 框。用于人工挑选 v8 判错案例做针对性优化。

使用方式:
  python detect/detect_sel_videos.py                  # 默认: 从≥2s视频中抽50个
  python detect/detect_sel_videos.py --count 100     # 跑 100 个
  python detect/detect_sel_videos.py --min-duration 3 # 只跑≥3秒的视频
  python detect/detect_sel_videos.py --source ... --output ...

输出:
  detect/sel_output/run_YYYYMMDD_HHMMSS/
    ├─ {视频名}_depth_video.mp4   ← 左原图(YOLO框)+右深度图(15点米数+v8状态条)
    └─ run_log.txt

挑选流程:
  1. 用播放器打开 *_depth_video.mp4 看;
  2. 发现 v8 判错的(误报/漏报/闪烁) → 把 mp4 拷到另一个文件夹;
  3. 把那个文件夹路径发回给我, 我针对性优化 v8 规则。
"""

import os
import sys
import time
import json
import random
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 允许直接在本目录运行：import 兄弟模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import (
    load_model, _load_config, _abs_path, _parse_pixel_points,
    depth_to_colormap, annotate_depth, DEFAULT_MODEL,
    VehicleDetector, MotionLineDetector, BrightnessMonitor,
    _get_ffmpeg_path, _draw_motion_strip, _draw_car_state,
)
from detect_car import fuse_v8_car_state


def _collect_video_files(source: str) -> list:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if os.path.isfile(source):
        return [source]
    files = []
    for dp, _, names in os.walk(source):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in exts:
                files.append(os.path.join(dp, n))
    return files


def _filter_by_duration(files, min_duration, desc="时长过滤"):
    """过滤掉时长 < min_duration 的视频（带进度条）"""
    kept = []
    for f in tqdm(files, desc=desc, ncols=100, leave=True):
        try:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            if fps <= 0 or n <= 0:
                continue
            if n / fps >= min_duration:
                kept.append(f)
        except Exception:
            continue
    return kept


def process_one_video(
    model, processor, video_path, output_dir, device,
    detector_config, pixel_points, frame_step=1, run_timestamp="",
):
    """处理单个视频：v8 融合判定 + 左右对比视频输出"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = Path(video_path).stem

    avi_out = os.path.join(output_dir, f"{name}_depth_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(avi_out, fourcc, int(fps) or 25, (w * 2, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(avi_out, fourcc, int(fps) or 25, (w * 2, h))

    # ---------- 检测器（参数与 detect_car.py 完全一致）----------
    yolo_cfg = detector_config.get("yolo", {})
    weights_cfg = detector_config.get("weights", {})
    mog2_config = detector_config.get("mog2", {})

    rule_weights = None
    if weights_cfg:
        rule_weights = {
            'R1': float(weights_cfg.get("R1_way4_depth_jump", 0.10)),
            'R2': float(weights_cfg.get("R2_way5_abs_depth", 0.25)),
            'R3': float(weights_cfg.get("R3_way1_baseline", 0.25)),
            'R4': float(weights_cfg.get("R4_vehicle_region", 0.10)),
            'R5': float(weights_cfg.get("R5_yolo_vision", 0.30)),
        }
    effective_fps = fps / frame_step if fps > 0 and frame_step > 0 else fps
    detector = VehicleDetector(
        n_points=len(pixel_points) if pixel_points else 5,
        distance_drop_threshold=detector_config["distance_drop_threshold"],
        normal_fluctuation=detector_config["normal_fluctuation"],
        fps=effective_fps,
        baseline_warmup=detector_config["baseline_warmup"],
        baseline_alpha=detector_config["baseline_alpha"],
        min_car_points=detector_config["min_car_points"],
        min_stable_points=detector_config["min_stable_points"],
        car_confirm_frames=detector_config["car_confirm_frames"],
        stable_confirm_frames=detector_config["stable_confirm_frames"],
        max_detect_depth=detector_config.get("max_detect_depth", 30.0),
        relative_drop_threshold=detector_config.get("relative_drop_threshold", 0.3),
        yolo_enabled=bool(yolo_cfg.get("enabled", False)),
        yolo_model_path=str(yolo_cfg.get("model_path", "")),
        yolo_device=str(yolo_cfg.get("device", "") or ""),
        yolo_confidence=float(yolo_cfg.get("confidence", 0.40)),
        yolo_iou=float(yolo_cfg.get("iou", 0.45)),
        yolo_vehicle_classes=tuple(int(c) for c in yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
        yolo_enter_bottom_ratio=float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
        yolo_min_box_ratio=float(yolo_cfg.get("min_box_ratio", 0.02)),
        rule_weights=rule_weights,
        car_confidence_threshold=float(detector_config.get("car_confidence_threshold", 0.470)),
    )
    # 进入 tqdm 前先加载 YOLO，避免打断进度条
    if yolo_cfg.get("enabled", False):
        detector._ensure_yolo_loaded()

    mog2_enabled = mog2_config.get("enabled", True)
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
    motion_detector = None
    if mog2_enabled and pixel_points:
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

    print(f"  [Video] {Path(video_path).name} ({w}x{h} @ {fps:.0f}fps, {total_frames}帧)")

    # ---------- 显示 EMA（只影响画面显示，不影响检测算法）----------
    _ema_depths = None
    _ema_d_min = None
    _ema_d_max = None
    _ema_alpha = 0.35

    n_to_process = (total_frames + frame_step - 1) // frame_step
    processed = 0
    frame_idx = 0
    last_final_state = "unknown"

    pbar = tqdm(total=n_to_process, desc=f"  {name[:14]}", ncols=120, leave=True,
                mininterval=0.5, maxinterval=2.0, smoothing=0.05)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # ---------- 深度推断 ----------
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=[img_rgb], return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                depth = model(**inputs).predicted_depth
            infer_ms = (time.time() - t0) * 1000.0

            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            depth = torch.nn.functional.interpolate(
                depth, size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

            depth_vis, d_min_raw, d_max_raw = depth_to_colormap(depth)

            # ---------- 监测点深度 ----------
            pixel_depths = []
            if pixel_points:
                for px, py in pixel_points:
                    if 0 <= px < w and 0 <= py < h:
                        pixel_depths.append(float(depth[py, px]))
                    else:
                        pixel_depths.append(None)

            # ---------- 显示 EMA 平滑 ----------
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
            if _ema_d_min is None:
                _ema_d_min, _ema_d_max = d_min_raw, d_max_raw
            else:
                _ema_d_min = _ema_alpha * d_min_raw + (1 - _ema_alpha) * _ema_d_min
                _ema_d_max = _ema_alpha * d_max_raw + (1 - _ema_alpha) * _ema_d_max
            display_depths = list(_ema_depths) if _ema_depths else pixel_depths
            display_d_min, display_d_max = _ema_d_min, _ema_d_max

            # ---------- 深度检测 + MOG2 ----------
            car_state = None
            yolo_boxes = []
            if pixel_depths:
                car_state = detector.update(pixel_depths, frame)
                yolo_boxes = list(getattr(detector, "_last_yolo_boxes", []) or [])
            motion_state = None
            if motion_detector:
                motion_state = motion_detector.update(frame)
                frame = _draw_motion_strip(frame, motion_detector, w, h)

            # ---------- V8 融合判定（与 detect_car.py 同一函数）----------
            state, last_final_state = fuse_v8_car_state(
                car_state=car_state,
                detector=detector,
                motion_detector=motion_detector,
                pixel_depths=pixel_depths,
                yolo_boxes=yolo_boxes,
                last_final_state=last_final_state,
            )

            # ---------- 画 YOLO 框到原图（便于肉眼判断 YOLO 是否打对）----------
            if yolo_boxes:
                for (x1, y1, x2, y2, conf, cls) in yolo_boxes:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"yolo:{conf:.2f}",
                                (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # ---------- 画 v8 状态条到深度图 ----------
            blevel = brightness_monitor.get_brightness_level() if brightness_monitor else None
            mweight = motion_detector.get_current_weight() if motion_detector else None
            motion_count = motion_detector.motion_pixel_count if motion_detector else 0
            depth_vis = _draw_car_state(
                depth_vis, state, w, motion_state, motion_count,
                brightness_level=blevel, mog2_weight=mweight,
            )

            # ---------- 监测点深度米数 ----------
            depth_vis = annotate_depth(
                depth_vis, display_d_min, display_d_max, h,
                pixel_points,
                display_depths if display_depths else None,
                infer_ms,
            )

            # ---------- 画 CAR 置信度到左上角 ----------
            conf_val = float(getattr(detector, "last_confidence", 0.0) or 0.0)
            vision_val = float(getattr(detector, "last_vision_score", 0.0) or 0.0)
            text = f"conf={conf_val:.2f}  R5={vision_val:.2f}"
            cv2.rectangle(depth_vis, (0, 0), (220, 26), (0, 0, 0), -1)
            cv2.putText(depth_vis, text, (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            combined = np.hstack([frame, depth_vis])
            writer.write(combined)
            processed += 1

            valid_depths = [d for d in display_depths or [] if d is not None]
            pts_s = " pts=[" + ",".join(f"{d:.1f}" for d in valid_depths) + "]m" if valid_depths else ""
            pbar.set_postfix_str(
                f"{state} | {display_d_min:.1f}~{display_d_max:.1f}m{pts_s}"
            )
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    time.sleep(0.4)

    # ---------- 转 ffmpeg mp4 ----------
    final_path = avi_out
    mp4_path = os.path.join(output_dir, f"{name}_depth_video.mp4")
    ffmpeg_exe = _get_ffmpeg_path()
    if ffmpeg_exe:
        print(f"  转码 MP4...", end="", flush=True)
        try:
            import subprocess
            result = subprocess.run(
                [ffmpeg_exe, "-y", "-i", os.path.abspath(avi_out),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 os.path.abspath(mp4_path)],
                capture_output=True, timeout=300,
            )
            if result.returncode == 0 and os.path.exists(mp4_path):
                try:
                    os.remove(os.path.abspath(avi_out))
                except Exception:
                    pass
                final_path = mp4_path
                print(f" 完成 -> {Path(mp4_path).name}")
            else:
                err = (result.stderr.decode(errors='ignore')[-200:] or "").strip()
                print(f" 失败，保留 AVI: {err[:80]}")
        except Exception as e:
            print(f" 异常，保留 AVI: {e}")
    else:
        print("  ffmpeg 不可用，输出为 AVI")

    state_counts = getattr(detector, "state_counts", None)
    if not state_counts:
        sc_car = sc_nocar = sc_unk = "?"
    else:
        sc_car = state_counts.get("car", 0)
        sc_nocar = state_counts.get("no_car", 0)
        sc_unk = state_counts.get("unknown", 0)
    print(f"  done {Path(final_path).name} | car={sc_car} no_car={sc_nocar} unk={sc_unk} | {processed}帧")
    return {
        "video_name": name,
        "video_path": str(video_path),
        "output_video": final_path,
        "frames": processed,
    }


def main():
    parser = argparse.ArgumentParser(description="视频挑选用: 批量输出 v8 判定带标注视频")
    parser.add_argument("--source", type=str, default=None,
                        help="输入视频/目录（默认: config 中 detect.input_dir）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录（默认: detect/sel_output）")
    parser.add_argument("--count", type=int, default=50,
                        help="抽取多少个视频（默认 50，0=全部）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机抽取种子，固定种子可复现")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="帧采样间隔（1=每帧）")
    parser.add_argument("--min-duration", type=float, default=2.0,
                        help="只处理时长≥该秒数的视频（默认 2.0 秒，过滤掉短片段）")
    args = parser.parse_args()

    cfg = _load_config()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detect_cfg = cfg.get("detect", {}) or {}
    car_cfg = cfg.get("car_detect", {}) or {}

    source = args.source or _abs_path(root, detect_cfg.get("input_dir", "detect/input"))
    output_root = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sel_output")

    # 与 detect.py 完全一致：从 config 读取监测点配置
    pixel_depth_points_str = str(detect_cfg.get("pixel_depth_points", "") or "").strip()
    pixel_depth_count = int(detect_cfg.get("pixel_depth_count", 5))
    print(f"[Pixel] pixel_depth_points='{pixel_depth_points_str}' pixel_depth_count={pixel_depth_count}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 设备: {device}")

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_root, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"[Output] {os.path.abspath(run_dir)}")
    log(f"[Run] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[Config] 输入: {source}")
    log(f"[Config] 抽样: {args.count} 个 | 随机种子: {args.seed} | 帧步长: {args.frame_step} | 最小时长: {args.min_duration}s")

    # ---------- 加载模型 ----------
    print("[Model] 加载模型...")
    model, processor = load_model(
        cfg.get("model", {}).get("student", DEFAULT_MODEL),
        device,
    )
    print(f"[Model] 已加载: {cfg.get('model', {}).get('student', DEFAULT_MODEL)}")

    # ---------- 视频列表 ----------
    files = _collect_video_files(source)
    if not files:
        print(f"[ERROR] 未找到视频: {source}")
        sys.exit(1)

    total_found = len(files)

    # ---------- 时长过滤 ----------
    if args.min_duration > 0:
        log(f"[Filter] 过滤时长 < {args.min_duration}s 的短片段 (共 {total_found} 个)...")
        files = _filter_by_duration(files, args.min_duration)
        log(f"[Filter] 保留 {len(files)} 个 (≥{args.min_duration}s)，丢弃 {total_found - len(files)} 个")

    count = args.count
    if count > 0 and len(files) > count:
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(files)
        files = files[:count]
        log(f"[Detect] 从 {len(files)} 个合格视频中抽取 {count} 个")
    else:
        log(f"[Detect] 全部 {len(files)} 个合格视频")

    # ---------- 检测器配置（与 detect_car.py 完全一致）----------
    mog2_cfg = car_cfg.get("mog2", {}) or {}
    yolo_cfg = car_cfg.get("yolo", {}) or {}
    weights_cfg = car_cfg.get("weights", {}) or {}
    detector_config = {
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
        "weights": dict(weights_cfg),
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
    log(f"[Config] 阈值: drop={detector_config['distance_drop_threshold']}m "
        f"fluct={detector_config['normal_fluctuation']}m "
        f"car={detector_config['car_confirm_frames']}f "
        f"nocar={detector_config['stable_confirm_frames']}f")
    log(f"[Config] car_confidence_threshold = {detector_config['car_confidence_threshold']}")
    log(f"[Config] YOLO = {detector_config['yolo']['enabled']}")

    # ---------- 批处理每个视频 ----------
    results = []
    errors = []
    for vi, vp in enumerate(files, 1):
        sep = "\n" + "=" * 70
        print(sep)
        log(f"[Progress] {vi}/{len(files)}: {Path(vp).name}")

        try:
            cap_test = cv2.VideoCapture(str(vp))
            w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_test.release()
            pixel_points = _parse_pixel_points(pixel_depth_points_str, pixel_depth_count, w, h)
            if pixel_points:
                xs = sorted(set(x for x, _ in pixel_points))
                print(f"  [Pixel] 自动分布({len(pixel_points)}点) x∈{{{','.join(map(str,xs))}}}")
        except Exception as e:
            err = f"[ERROR] 视频 {Path(vp).name} 元数据读取失败: {e}"
            print(err)
            errors.append({"video": Path(vp).name, "error": str(e),
                           "traceback": traceback.format_exc()})
            continue

        try:
            r = process_one_video(
                model=model,
                processor=processor,
                video_path=vp,
                output_dir=run_dir,
                device=device,
                detector_config=detector_config,
                pixel_points=pixel_points,
                frame_step=args.frame_step,
                run_timestamp=run_timestamp,
            )
            if r:
                results.append(r)
        except Exception as e:
            err_msg = f"[ERROR] {Path(vp).name} 处理失败: {e}"
            print(err_msg)
            tb = traceback.format_exc()
            print(tb)
            errors.append({"video": Path(vp).name, "error": str(e), "traceback": tb})

    # ---------- 收尾 ----------
    sep = "\n" + "=" * 70
    print(sep)
    fail_s = f", {len(errors)} 个失败" if errors else ""
    log(f"[汇总] {len(results)}/{len(files)} 个视频成功{fail_s}")
    log(f"[汇总] 输出目录: {os.path.abspath(run_dir)}")
    log(f"[Run] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with open(os.path.join(run_dir, f"{run_timestamp}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_timestamp": run_timestamp,
            "total": len(files),
            "succeeded": len(results),
            "failed": len(errors),
            "config": detector_config,
            "results": results,
            "errors": errors,
        }, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, f"{run_timestamp}_run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        if errors:
            f.write("\n\n===== 错误明细 =====\n")
            for e in errors:
                f.write(f"\n视频: {e['video']}\n错误: {e['error']}\n{e.get('traceback','')}\n")


if __name__ == "__main__":
    main()
