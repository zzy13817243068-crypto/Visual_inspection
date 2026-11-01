"""
车辆检测测试脚本（精简版）
==========================
随机抽取150个视频，只保存检测为"no_car"的帧到 test 目录。

用法：
  python detect/test_no_car.py                    # 默认抽150个视频
  python detect/test_no_car.py --count 20         # 自定义数量
  python detect/test_no_car.py --source xxx.mp4   # 单个视频
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 允许从 detect 目录直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import (
    load_model, _load_config, _abs_path, _parse_pixel_points,
    depth_to_colormap, annotate_depth, DEFAULT_MODEL,
    VehicleDetector, MotionLineDetector, BrightnessMonitor
)


def process_video(model, processor, video_path, output_dir, device,
                  detector_config, pixel_points=None, frame_step=1):
    """处理单个视频，只保存 no_car 帧到输出目录"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return {"video": Path(video_path).name, "no_car_saved": 0, "error": "open_failed"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_step = max(1, int(frame_step))
    effective_fps = fps / frame_step

    video_name = Path(video_path).stem
    # 所有视频的帧平铺到一个目录，文件名里带 video_name 区分
    video_out_dir = output_dir
    os.makedirs(video_out_dir, exist_ok=True)

    # 初始化深度检测器
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
        relative_drop_threshold=detector_config.get("relative_drop_threshold", 0.15),
    )

    # 初始化 MOG2 运动检测器
    motion_detector = None
    mog2_config = detector_config.get("mog2", {})
    mog2_enabled = mog2_config.get("enabled", True)

    # 初始化亮度监听器
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

    state_counts = {"car": 0, "no_car": 0, "unknown": 0}
    no_car_saved = 0
    frame_idx = 0
    processed_total = (total_frames + frame_step - 1) // frame_step
    pbar = tqdm(total=processed_total, desc=f"测试 {video_name[:15]}", ncols=120, leave=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps if fps > 0 else 0
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[img_rgb], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = torch.nn.functional.interpolate(
            depth, size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

        # 提取监测点深度
        pixel_depths = []
        if pixel_points:
            for px, py in pixel_points:
                if 0 <= px < w and 0 <= py < h:
                    pixel_depths.append(float(depth[py, px]))
                else:
                    pixel_depths.append(float('nan'))

        # 更新检测器（深度 + MOG2 融合）
        car_state = None
        motion_state = None
        if detector and pixel_depths:
            car_state = detector.update(pixel_depths)
        if motion_detector:
            motion_state = motion_detector.update(frame)

        # 融合判定（v4: 亮度权重 + 深度变化验证）
        mog2_confirmed = motion_detector.is_confirmed() if motion_detector else False
        mog2_weight = motion_detector.get_current_weight() if motion_detector else 0.5

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
            state = "car"
        elif car_state == "no_car":
            state = "no_car"
        elif car_state == "unknown" and mog2_confirmed and mog2_depth_ok:
            state = "car"
        else:
            state = car_state or "unknown"

        state_counts[state] = state_counts.get(state, 0) + 1

        # 只保存 no_car 帧（原始画面，无 UI）
        if state == "no_car":
            img_ext = detector_config.get("image_format", "jpg")
            jpeg_q = int(detector_config.get("jpeg_quality", 95))
            img_path = os.path.join(
                video_out_dir,
                f"{video_name}_frame{frame_idx:06d}_t{timestamp:06.2f}s.{img_ext}"
            )
            if img_ext == "jpg":
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            else:
                cv2.imwrite(img_path, frame)
            no_car_saved += 1

        # 进度条
        valid_depths = [d for d in pixel_depths if not (d != d)]
        pts_str = "[" + ",".join(f"{d:.1f}" for d in valid_depths) + "]m" if valid_depths else "[]"
        pbar.set_postfix_str(f"state={state} pts={pts_str} saved={no_car_saved}")
        pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()

    # 保存亮度/权重日志（文件名带 video_name，平铺目录下能区分）
    if brightness_monitor and brightness_monitor.get_logs():
        bright_log_path = os.path.join(video_out_dir, f"{video_name}_brightness_log.json")
        with open(bright_log_path, "w", encoding="utf-8") as f:
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

    return {
        "video": Path(video_path).name,
        "total_frames": total_frames,
        "state_counts": state_counts,
        "no_car_saved": no_car_saved,
        "output_dir": video_out_dir,
    }


def is_night_video(video_path, threshold=80.0, sample_frame_idx=None):
    """判断视频是否为夜间（取中间帧计算平均亮度）。

    Args:
        video_path: 视频路径
        threshold: 亮度阈值，低于此值为夜间
        sample_frame_idx: 指定采样帧索引，None 则取中间帧
    Returns:
        (bool, float): (是否夜间, 平均亮度值)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, -1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if sample_frame_idx is None:
        sample_frame_idx = max(0, total // 2)  # 取中间帧，避开开头黑场
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False, -1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    return brightness < threshold, brightness


def main():
    cfg = _load_config()
    root = cfg.get("_root", os.getcwd())
    paths = cfg.get("paths", {})
    detect_cfg = cfg.get("detect", {})
    car_cfg = cfg.get("car_detect", {})

    parser = argparse.ArgumentParser(description="车辆检测测试 - 只保存no_car帧")
    parser.add_argument("--source", type=str,
                        default=_abs_path(root, car_cfg.get("input_dir", "detect/input")),
                        help="输入视频文件或目录")
    parser.add_argument("--output", type=str,
                        default=_abs_path(root, "detect/test"),
                        help="输出目录（默认 detect/test）")
    parser.add_argument("--model-depth", type=str,
                        default=_abs_path(root, detect_cfg.get("model_depth", DEFAULT_MODEL)))
    parser.add_argument("--device", type=str,
                        default=detect_cfg.get("device", "cuda"))
    parser.add_argument("--count", type=int, default=500,
                        help="随机抽取视频数量（0=全部）")
    parser.add_argument("--frame-step", type=int,
                        default=int(car_cfg.get("frame_step", 1)))
    parser.add_argument("--seed", type=int,
                        default=car_cfg.get("seed"))
    parser.add_argument("--pixel-depth-points", type=str,
                        default=str(detect_cfg.get("pixel_depth_points", "") or ""))
    parser.add_argument("--pixel-depth-count", type=int,
                        default=int(detect_cfg.get("pixel_depth_count", 5)))
    parser.add_argument("--night-only", action="store_true",
                        help="只处理夜间视频（平均亮度低于阈值）")
    parser.add_argument("--night-threshold", type=float, default=80.0,
                        help="夜间亮度阈值（0-255，低于此值为夜间，默认80）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 每次运行创建带时间戳的子目录
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[Test] 输出目录: {os.path.abspath(run_dir)}")
    print(f"[Test] 抽取数量: {args.count} | 帧间隔: {args.frame_step} | 种子: {args.seed}")

    # 加载模型
    print("[Model] 加载模型...")
    model, processor = load_model(args.model_depth, device)
    print(f"[Model] 已加载，设备: {device}")

    # 收集视频文件（支持子目录递归）
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if os.path.isfile(args.source):
        files = [args.source]
    elif os.path.isdir(args.source):
        files = []
        for dirpath, _, filenames in os.walk(args.source):
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() in video_exts:
                    files.append(os.path.join(dirpath, f))
    else:
        print(f"[ERROR] 路径不存在: {args.source}")
        sys.exit(1)

    if not files:
        print(f"[ERROR] 未找到视频文件: {args.source}")
        sys.exit(1)

    # 夜间筛选（--night-only 时只保留平均亮度低于阈值的视频）
    if args.night_only:
        total_before = len(files)
        print(f"[Night] 筛选夜间视频（阈值={args.night_threshold}）...")
        night_files = []
        for vp in tqdm(files, desc="夜间筛查", ncols=100):
            is_night, bright = is_night_video(vp, args.night_threshold)
            if is_night:
                night_files.append(vp)
        files = night_files
        print(f"[Night] 夜间视频: {len(files)}/{total_before} 个")
        if not files:
            print(f"[ERROR] 未找到夜间视频，请降低 --night-threshold")
            sys.exit(1)

    # 随机选取
    if args.seed is not None:
        random.seed(args.seed)
    if args.count > 0 and len(files) > args.count:
        random.shuffle(files)
        selected = files[:args.count]
        print(f"[Test] 找到 {len(files)} 个视频，随机选取 {args.count} 个")
    else:
        selected = files
        print(f"[Test] 处理全部 {len(files)} 个视频")

    # 检测器配置（深度 + MOG2）
    mog2_cfg = car_cfg.get("mog2", {})
    detector_config = {
        "distance_drop_threshold": float(car_cfg.get("distance_drop_threshold", 1.5)),
        "normal_fluctuation": float(car_cfg.get("normal_fluctuation", 1.5)),
        "min_car_points": int(car_cfg.get("min_car_points", 1)),
        "min_stable_points": int(car_cfg.get("min_stable_points", 3)),
        "car_confirm_frames": int(car_cfg.get("car_confirm_frames", 1)),
        "stable_confirm_frames": int(car_cfg.get("stable_confirm_frames", 5)),
        "baseline_warmup": int(car_cfg.get("baseline_warmup", 5)),
        "baseline_alpha": float(car_cfg.get("baseline_alpha", 0.05)),
        "max_detect_depth": float(car_cfg.get("max_detect_depth", 30.0)),
        "relative_drop_threshold": float(car_cfg.get("relative_drop_threshold", 0.15)),
        "image_format": car_cfg.get("image_format", "jpg"),
        "jpeg_quality": int(car_cfg.get("jpeg_quality", 95)),
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

    # 处理每个视频
    all_results = []
    for i, video_path in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {Path(video_path).name}")

        # 获取视频尺寸用于解析监测点
        cap_test = cv2.VideoCapture(video_path)
        test_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
        test_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_test.release()

        pixel_points = _parse_pixel_points(
            args.pixel_depth_points, args.pixel_depth_count, test_w, test_h
        )

        try:
            result = process_video(
                model, processor, video_path, run_dir, device,
                detector_config, pixel_points, args.frame_step
            )
            all_results.append(result)
            print(f"  -> no_car帧: {result['no_car_saved']} | 状态分布: {result['state_counts']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results.append({"video": Path(video_path).name, "error": str(e)})

    # 汇总
    total_no_car = sum(r.get("no_car_saved", 0) for r in all_results)
    success_count = sum(1 for r in all_results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"[完成] 处理视频: {success_count}/{len(selected)}")
    print(f"[完成] 保存no_car帧总数: {total_no_car}")
    print(f"[完成] 输出目录: {os.path.abspath(run_dir)}")

    # 保存汇总报告
    summary = {
        "run_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_videos": len(selected),
        "success_count": success_count,
        "total_no_car_frames": total_no_car,
        "results": all_results,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[完成] 汇总报告: {summary_path}")


if __name__ == "__main__":
    main()
