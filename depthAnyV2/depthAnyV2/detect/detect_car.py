"""
车辆检测测试脚本
=================
基于五个监测点距离变化的车辆存在判定逻辑。

判定原理：
- 有车经过：车辆遮挡背景导致监测点深度值骤减
- 无车状态：监测点深度值稳定在背景基线附近

工作流程：
1. 读取视频，用 DA2-Metric 模型提取每帧5个监测点的深度值
2. 维护背景基线（无车时的深度），检测深度骤减判定有车
3. 识别"无车状态"片段，截取帧保存到输出目录
4. 生成测试报告（视频时长、无车片段数、截取帧数等）

使用方式：
  python detect/detect_car.py
  python detect/detect_car.py --source detect/input/xxx.mp4
  python detect/detect_car.py --distance-drop 4.0 --stable-time 3.0
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm
import yaml

# 允许从 detect 目录直接运行，导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import (
    load_model, _load_config, _abs_path, _parse_pixel_points,
    depth_to_colormap, annotate_depth, DEFAULT_MODEL,
    VehicleDetector, MotionLineDetector, BrightnessMonitor
)


# ============================================================
# v8 融合判定（单帧，共享公共函数）
# detect_car.py 和 detect_no_car.py 都直接调用此处，避免两份逻辑错位
# 原则：YOLO-only-boost（仅当帧检测到时加权，否则深度主导）
# ============================================================
def fuse_v8_car_state(car_state, detector, motion_detector, pixel_depths, yolo_boxes, last_final_state):
    """
    单帧 v8 综合判定：
      - car_state: VehicleDetector.update() 内部加权 + 状态机 的输出（car/no_car/unknown）
      - 其他检测器/上下文：detector(取conf/R5)、motion_detector、pixel_depths(深度点)、
        yolo_boxes(YOLO当前帧框列表)、last_final_state(上一帧最终state，用于当前帧HOLD)
    返回: (state 最终单帧状态, 新的 last_final_state)
    """
    mog2_confirmed = motion_detector.is_confirmed() if motion_detector else False
    depth_conf = detector.last_confidence if detector else 0.0
    vision_score = detector.last_vision_score if detector else 0.0
    yolo_boxes_n = len(yolo_boxes)
    vision_strong = vision_score >= 0.70 and yolo_boxes_n >= 1
    vision_weak_see = vision_score >= 0.45 and yolo_boxes_n >= 1
    yolo_blind = (yolo_boxes_n == 0) or (vision_score < 0.30)

    # 当前帧 HOLD（仅当 YOLO 真检测到 → 防闪烁）
    yolo_current_frame_hold = (last_final_state == "car") and vision_weak_see

    mog2_depth_ok = False
    if detector and mog2_confirmed and pixel_depths:
        mog2_depth_ok = detector.has_depth_change(
            pixel_depths, abs_threshold=0.3, rel_threshold=0.05, min_points=2
        )

    if yolo_current_frame_hold:
        state = "car"
    elif car_state == "car":
        if vision_strong or depth_conf >= 0.70:
            state = "car"
        elif mog2_confirmed:
            state = "car"
        elif depth_conf >= 0.60:
            state = "car"
        elif vision_weak_see:
            state = "car"
        elif yolo_blind and depth_conf >= 0.35:
            # 漏检修复（深度主导）：YOLO漏检但深度信号未坍塌(≥0.35) → 放行 car
            # 防闪断：车停住+YOLO漏几帧时，depth_conf 通常 0.35~0.49，旧门槛0.50 会闪断
            state = "car"
        elif depth_conf > 0.20:
            # 已car状态深度兜底惯性：状态机仍判定car，深度未彻底坍塌(>0.20) → 保持 car
            # 不靠YOLO也不靠高confidence，纯状态惯性，避免车过检测线过程中闪几帧no_car
            state = "car"
        else:
            state = "no_car"
    elif car_state == "no_car":
        if last_final_state == "car":
            if vision_weak_see and depth_conf >= 0.35:
                state = "car"
            else:
                state = "no_car"
        else:
            if vision_score >= 0.75 and yolo_boxes_n >= 1:
                state = "car"
            else:
                state = "no_car"
    elif car_state == "unknown" and mog2_confirmed and mog2_depth_ok:
        state = "car"
    else:
        state = car_state or "unknown"

    last_final_state = state
    return state, last_final_state


# ============================================================
# 无车片段记录器
# ============================================================
class NoCarSegmentRecorder:
    """
    记录"无车状态"片段，截取帧保存到磁盘。

    片段定义：从状态变为 no_car 开始，到状态离开 no_car 结束。
    """

    def __init__(self, output_dir, video_name, image_format="jpg", jpeg_quality=95, run_timestamp=""):
        self.output_dir = output_dir
        self.video_name = video_name
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.run_timestamp = run_timestamp

        # 片段状态
        self.segments = []          # 已完成的无车片段列表
        self.current_segment = None  # 正在记录的片段
        self.frame_buffer = []      # 当前片段的帧缓存
        self.total_saved = 0        # 总保存帧数

    def start_segment(self, start_frame, start_time):
        """开始一个新的无车片段"""
        self.current_segment = {
            "start_frame": start_frame,
            "start_time": start_time,
            "end_frame": None,
            "end_time": None,
            "frame_count": 0,
            "segment_dir": None,
        }
        self.frame_buffer = []
        seg_idx = len(self.segments) + 1
        seg_name = f"{self.run_timestamp}_segment_{seg_idx:03d}_start{int(start_time):04d}s"
        seg_dir = os.path.join(self.output_dir, self.video_name, seg_name)
        os.makedirs(seg_dir, exist_ok=True)
        self.current_segment["segment_dir"] = seg_dir

    def add_frame(self, frame, frame_idx, timestamp):
        """向当前片段添加一帧"""
        if self.current_segment is None:
            return
        self.frame_buffer.append((frame.copy(), frame_idx, timestamp))

    def end_segment(self, end_frame, end_time):
        """结束当前片段，将帧保存到磁盘"""
        if self.current_segment is None:
            return

        self.current_segment["end_frame"] = end_frame
        self.current_segment["end_time"] = end_time
        self.current_segment["frame_count"] = len(self.frame_buffer)

        # 保存帧到磁盘
        seg_dir = self.current_segment["segment_dir"]
        seg_idx = len(self.segments) + 1
        for frame, fidx, ts in self.frame_buffer:
            fname = f"{self.run_timestamp}_{self.video_name}_seg{seg_idx:03d}_frame{fidx:06d}_t{ts:07.2f}s.{self.image_format}"
            fpath = os.path.join(seg_dir, fname)
            if self.image_format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            else:
                cv2.imwrite(fpath, frame)
            self.total_saved += 1

        self.segments.append(self.current_segment)
        self.current_segment = None
        self.frame_buffer = []

    def is_recording(self):
        return self.current_segment is not None

    def get_stats(self):
        return {
            "segment_count": len(self.segments),
            "total_frames_saved": self.total_saved,
            "segments": self.segments,
        }


# ============================================================
# 视频处理主函数
# ============================================================
def process_video_car_detect(model, processor, video_path, output_dir, device,
                             detector_config, pixel_points=None, frame_step=1,
                             run_timestamp=""):
    """
    处理视频：逐帧检测车辆状态，截取无车片段。

    Args:
        model: DA2 模型
        processor: DA2 processor
        video_path: 视频路径
        output_dir: 输出目录
        device: torch.device
        detector_config: dict, 检测器配置参数
        pixel_points: list[(x,y)], 监测点坐标
        frame_step: 帧采样间隔 (1=每帧, 2=隔帧)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(frame_step))
    effective_fps = fps / frame_step if fps > 0 else fps

    video_name = Path(video_path).stem
    print(f"\n[Video] {Path(video_path).name}")
    print(f"  分辨率: {w}x{h}, FPS: {fps:.1f}, 总帧数: {total_frames}, 时长: {duration:.1f}s, 采样间隔: {frame_step}")

    # 初始化深度检测器（使用有效帧率）—— 支持 YOLO 视觉 + 多规则权重
    yolo_cfg = detector_config.get("yolo", {})
    weights_cfg = detector_config.get("weights", {})
    rule_weights = None
    if weights_cfg:
        # 从配置映射到 detector 内部 key (R1..R5)
        rule_weights = {
            'R1': float(weights_cfg.get("R1_way4_depth_jump", 0.10)),
            'R2': float(weights_cfg.get("R2_way5_abs_depth", 0.25)),
            'R3': float(weights_cfg.get("R3_way1_baseline", 0.25)),
            'R4': float(weights_cfg.get("R4_vehicle_region", 0.10)),
            'R5': float(weights_cfg.get("R5_yolo_vision", 0.30)),
        }
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
        # YOLO 视觉检测参数
        yolo_enabled=bool(yolo_cfg.get("enabled", False)),
        yolo_model_path=str(yolo_cfg.get("model_path", "yolov8n.pt")),
        yolo_device=str(yolo_cfg.get("device", "") or ""),
        yolo_confidence=float(yolo_cfg.get("confidence", 0.40)),
        yolo_iou=float(yolo_cfg.get("iou", 0.45)),
        yolo_vehicle_classes=tuple(int(c) for c in yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
        yolo_enter_bottom_ratio=float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
        yolo_min_box_ratio=float(yolo_cfg.get("min_box_ratio", 0.02)),
        # 多规则融合权重
        rule_weights=rule_weights,
        car_confidence_threshold=float(detector_config.get("car_confidence_threshold", 0.470)),
    )
    if yolo_cfg.get("enabled", False):
        # 进入 tqdm 进度条之前先确保 YOLO 已加载（避免在进度条首帧 print 打断它）
        detector._ensure_yolo_loaded()
        print(f"  [YOLO] 视觉检测已启用 | 模型={yolo_cfg.get('model_path')} | 进入检测区阈值=下{float(yolo_cfg.get('enter_bottom_ratio',0.30))*100:.0f}% | 置信度≥{float(yolo_cfg.get('confidence',0.40)):.2f}")
        if rule_weights:
            print(f"  [权重] R1={rule_weights['R1']:.2f}(骤变) R2={rule_weights['R2']:.2f}(绝对深度) R3={rule_weights['R3']:.2f}(基线) R4={rule_weights['R4']:.2f}(统计) R5={rule_weights['R5']:.2f}(视觉) = {sum(rule_weights.values()):.2f}")
            print(f"  [阈值] car_confidence_threshold = {detector_config.get('car_confidence_threshold', 0.470):.3f}")

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
        print(f"  [MOG2] 运动检测已启用 | 检测线 x={detect_x}, 带宽={mog2_config.get('strip_width', 30)}px, 确认={mog2_config.get('confirm_frames', 3)}帧")
        if brightness_monitor:
            print(f"  [亮度] 动态权重已启用 | 白天≥{brightness_monitor.bright_high}→权重={brightness_monitor.min_weight:.1f} | 夜间<{brightness_monitor.bright_mid}→权重={brightness_monitor.max_weight:.1f}")

    # 初始化片段记录器
    recorder = NoCarSegmentRecorder(
        output_dir=output_dir,
        video_name=video_name,
        image_format=detector_config["image_format"],
        jpeg_quality=detector_config["jpeg_quality"],
        run_timestamp=run_timestamp,
    )

    # 创建可视化输出目录（可选）
    vis_dir = os.path.join(output_dir, video_name, "_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    save_vis = detector_config.get("save_visualization", True)

    # 状态统计
    state_counts = {"car": 0, "no_car": 0, "unknown": 0}
    frame_idx = 0
    processed_total = (total_frames + frame_step - 1) // frame_step
    pbar = tqdm(total=processed_total, desc=f"检测 {video_name[:15]}", ncols=130, leave=True)

    # ===== v8 融合 严格遵守“YOLO 检测到才加权；否则深度+运动占大头”原则 =====
    # 用户原则：
    #  ✅ YOLO 在检测线上检测到框（R5≥0.45 有框）→ 加权重 / HOLD 锁（仅当帧有效）
    #  ❌ YOLO 没检测到（R5=0 或 无框）→ 深度加权占大头，运动辅助；**不得**因“上一帧靠YOLO锁car”而跨帧续命（HOLD 续命帧完全删除）
    #  避免：车到检测线，YOLO 刚好漏检 → 卡在 no_car 一帧后后续帧即使 YOLO 回来也拉不上
    last_final_state = "unknown"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 隔帧采样：只处理 frame_step 倍数的帧
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps if fps > 0 else 0
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

        # 提取监测点深度
        pixel_depths = []
        if pixel_points:
            for px, py in pixel_points:
                if 0 <= px < w and 0 <= py < h:
                    pixel_depths.append(float(depth[py, px]))
                else:
                    pixel_depths.append(float('nan'))

        # 更新检测器（深度 + YOLO视觉协同 + MOG2 融合）
        car_state = None
        motion_state = None
        yolo_boxes = []
        if detector and pixel_depths:
            # 关键：传 frame_bgr 给 update()，YOLO 先跑得到 R5，再综合 R1-R4 深度加权
            # 视觉不是独立触发，只是 R5 规则
            car_state = detector.update(pixel_depths, frame)
            yolo_boxes = list(getattr(detector, '_last_yolo_boxes', []))
        if motion_detector:
            motion_state = motion_detector.update(frame)

        # ==================== 融合判定（v8: 严格 YOLO-only-boost + 深度主导 原则） ====================
        # 逻辑直接套用 fuse_v8_car_state() 共享函数，和 detect_no_car.py 100% 对齐
        state, last_final_state = fuse_v8_car_state(
            car_state=car_state,
            detector=detector,
            motion_detector=motion_detector,
            pixel_depths=pixel_depths,
            yolo_boxes=yolo_boxes,
            last_final_state=last_final_state,
        )

        state_counts[state] = state_counts.get(state, 0) + 1

        # 片段记录逻辑
        if state == "no_car":
            if not recorder.is_recording():
                recorder.start_segment(frame_idx, timestamp)
            recorder.add_frame(frame, frame_idx, timestamp)
        else:
            if recorder.is_recording():
                recorder.end_segment(frame_idx, timestamp)

        # 可视化（每N帧保存一张）
        if save_vis and (frame_idx % max(1, total_frames // 50) == 0):
            depth_vis, d_min, d_max = depth_to_colormap(depth)
            depth_vis = annotate_depth(
                depth_vis, d_min, d_max, h,
                pixel_points, pixel_depths, frame_time
            )
            # 添加状态标注
            state_color = {"car": (0, 0, 255), "no_car": (0, 255, 0), "unknown": (0, 165, 255)}
            state_text = {"car": "CAR DETECTED", "no_car": "NO CAR", "unknown": "UNKNOWN"}
            color = state_color.get(state, (255, 255, 255))
            text = state_text.get(state, state.upper())
            cv2.rectangle(depth_vis, (w - 220, 8), (w - 8, 48), (0, 0, 0), -1)
            cv2.putText(depth_vis, text, (w - 210, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            combined = np.hstack([frame, depth_vis])
            # 缩放
            max_w = 1920
            if combined.shape[1] > max_w:
                scale = max_w / combined.shape[1]
                combined = cv2.resize(combined, (int(combined.shape[1] * scale),
                                                  int(combined.shape[0] * scale)))
            vis_path = os.path.join(vis_dir, f"{run_timestamp}_frame_{frame_idx:06d}_t{timestamp:07.2f}s.jpg")
            cv2.imwrite(vis_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 进度条更新
        valid_depths = [d for d in pixel_depths if not (d != d)]  # 过滤NaN
        pts_str = "[" + ",".join(f"{d:.1f}" for d in valid_depths) + "]m" if valid_depths else "[]"
        pbar.set_postfix_str(
            f"state={state} pts={pts_str} {frame_time:.0f}ms segs={len(recorder.segments)}"
        )
        pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()

    # 结束最后一个未关闭的片段
    if recorder.is_recording():
        recorder.end_segment(frame_idx, timestamp)

    # 生成报告
    stats = recorder.get_stats()
    report = {
        "video_name": video_name,
        "video_path": str(video_path),
        "resolution": f"{w}x{h}",
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "state_counts": state_counts,
        "no_car_segment_count": stats["segment_count"],
        "total_frames_saved": stats["total_frames_saved"],
        "segments": [
            {
                "segment_idx": i + 1,
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "start_time_sec": round(s["start_time"], 2),
                "end_time_sec": round(s["end_time"], 2),
                "duration_sec": round(s["end_time"] - s["start_time"], 2),
                "frame_count": s["frame_count"],
            }
            for i, s in enumerate(stats["segments"])
        ],
        "detector_config": detector_config,
        "pixel_points": pixel_points,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 保存报告
    report_path = os.path.join(output_dir, video_name, f"{run_timestamp}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n  [报告] 视频时长: {duration:.1f}s | 总帧数: {total_frames}")
    print(f"  [报告] 状态统计: 有车={state_counts.get('car',0)}帧, "
          f"无车={state_counts.get('no_car',0)}帧, "
          f"未知={state_counts.get('unknown',0)}帧")
    print(f"  [报告] 无车片段: {stats['segment_count']} 个 | 截取帧数: {stats['total_frames_saved']} 张")
    for i, s in enumerate(stats["segments"]):
        print(f"    片段{i+1}: {s['start_time']:.1f}s ~ {s['end_time']:.1f}s "
              f"({s['end_time']-s['start_time']:.1f}s, {s['frame_count']}帧)")
    print(f"  [报告] 报告文件: {os.path.abspath(report_path)}")
    print(f"  [报告] 输出目录: {os.path.abspath(os.path.join(output_dir, video_name))}")

    # 保存亮度/权重日志
    if brightness_monitor and brightness_monitor.get_logs():
        bright_log_path = os.path.join(output_dir, video_name, f"{run_timestamp}_brightness_log.json")
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
        print(f"  [亮度] 日志已保存: {bright_log_path} ({len(brightness_monitor.get_logs())} 条记录)")

    return report


# ============================================================
# 主函数
# ============================================================
def main():
    cfg = _load_config()
    root = cfg.get("_root", os.getcwd())
    paths = cfg.get("paths", {})
    detect_cfg = cfg.get("detect", {})
    car_cfg = cfg.get("car_detect", {})

    parser = argparse.ArgumentParser(description="车辆检测测试脚本 - 基于深度变化的车辆存在判定")
    parser.add_argument("--source", type=str,
                        default=_abs_path(root, car_cfg.get("input_dir", "detect/input")),
                        help="输入视频文件或目录（支持子目录递归）")
    parser.add_argument("--output", type=str,
                        default=_abs_path(root, car_cfg.get("output_dir", "detect/car_output")),
                        help="输出根目录（每次运行在其下创建带时间戳的子目录）")
    parser.add_argument("--model-depth", type=str,
                        default=_abs_path(root, detect_cfg.get("model_depth", DEFAULT_MODEL)),
                        help="模型路径")
    parser.add_argument("--device", type=str,
                        default=detect_cfg.get("device", "cuda"))

    # 车辆检测参数
    parser.add_argument("--distance-drop", type=float,
                        default=float(car_cfg.get("distance_drop_threshold", 2.0)),
                        help="距离骤减阈值(m)，单个点深度减少超过此值视为被遮挡")
    parser.add_argument("--normal-fluctuation", type=float,
                        default=float(car_cfg.get("normal_fluctuation", 1.5)),
                        help="正常波动阈值(m)，偏差小于此值视为稳定")
    parser.add_argument("--min-car-points", type=int,
                        default=int(car_cfg.get("min_car_points", 1)),
                        help="判定有车所需的最少骤减点数(多点投票)")
    parser.add_argument("--min-stable-points", type=int,
                        default=int(car_cfg.get("min_stable_points", 3)),
                        help="判定无车所需的最少稳定点数(多数稳定)")
    parser.add_argument("--car-confirm-frames", type=int,
                        default=int(car_cfg.get("car_confirm_frames", 1)),
                        help="有车确认帧数(1=检测到就触发)")
    parser.add_argument("--stable-confirm-frames", type=int,
                        default=int(car_cfg.get("stable_confirm_frames", 3)),
                        help="无车确认帧数(需连续N帧稳定)")
    parser.add_argument("--baseline-warmup", type=int,
                        default=int(car_cfg.get("baseline_warmup", 5)),
                        help="基线预热帧数")
    parser.add_argument("--baseline-alpha", type=float,
                        default=float(car_cfg.get("baseline_alpha", 0.05)),
                        help="基线EMA更新系数(0~1)")
    parser.add_argument("--max-detect-depth", type=float,
                        default=float(car_cfg.get("max_detect_depth", 30.0)),
                        help="参与车辆检测的最大基线深度(m)，超过此值视为天空/远景")
    parser.add_argument("--relative-drop-threshold", type=float,
                        default=float(car_cfg.get("relative_drop_threshold", 0.3)),
                        help="相对跌幅阈值(0~1)，需同时满足绝对和相对跌幅")
    parser.add_argument("--image-format", type=str,
                        default=car_cfg.get("image_format", "jpg"),
                        choices=["jpg", "png"],
                        help="截取帧的图像格式")
    parser.add_argument("--jpeg-quality", type=int,
                        default=int(car_cfg.get("jpeg_quality", 95)),
                        help="JPEG保存质量(1-100)")
    parser.add_argument("--save-visualization", type=int, choices=[0, 1],
                        default=int(car_cfg.get("save_visualization", 1)),
                        help="是否保存可视化对比图(1=保存, 0=不保存)")
    parser.add_argument("--preview-count", type=int,
                        default=int(car_cfg.get("preview_count", 50)),
                        help="随机选取的视频数量 (0=处理全部)")
    parser.add_argument("--frame-step", type=int,
                        default=int(car_cfg.get("frame_step", 1)),
                        help="处理帧采样间隔 (1=每帧, 2=隔帧)")
    parser.add_argument("--seed", type=int,
                        default=car_cfg.get("seed"),
                        help="随机种子 (不指定=每次随机)")

    # 像素点配置
    parser.add_argument("--pixel-depth-points", type=str,
                        default=str(detect_cfg.get("pixel_depth_points", "") or ""),
                        help="监测像素点列表 (格式: x1,y1;x2,y2;... 或留空使用自动分布)")
    parser.add_argument("--pixel-depth-count", type=int,
                        default=int(detect_cfg.get("pixel_depth_count", 5)),
                        help="自动分布的监测点数量")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 每次运行创建带时间戳的唯一输出目录
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 日志收集（同时输出到控制台和日志文件）
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"[Output] 本次输出目录: {os.path.abspath(run_dir)}")
    log(f"[Run] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[Config] 输入: {args.source}")
    log(f"[Config] 帧采样间隔: {args.frame_step} | 随机种子: {args.seed}")

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

    # 随机选取视频
    import random
    if args.seed is not None:
        random.seed(args.seed)
    preview_count = args.preview_count
    if preview_count > 0 and len(files) > preview_count:
        random.shuffle(files)
        selected = files[:preview_count]
        log(f"[Detect] 找到 {len(files)} 个视频，随机选取 {preview_count} 个")
        for f in selected:
            log(f"  - {Path(f).name}")
        files = selected
    else:
        log(f"[Detect] 处理全部 {len(files)} 个视频")

    # 检测器配置（深度 + YOLO视觉 + MOG2 + 多规则权重）
    mog2_cfg = car_cfg.get("mog2", {})
    yolo_cfg = car_cfg.get("yolo", {})
    weights_cfg = car_cfg.get("weights", {})
    detector_config = {
        "distance_drop_threshold": args.distance_drop,
        "normal_fluctuation": args.normal_fluctuation,
        "min_car_points": args.min_car_points,
        "min_stable_points": args.min_stable_points,
        "car_confirm_frames": args.car_confirm_frames,
        "stable_confirm_frames": args.stable_confirm_frames,
        "baseline_warmup": args.baseline_warmup,
        "baseline_alpha": args.baseline_alpha,
        "max_detect_depth": args.max_detect_depth,
        "relative_drop_threshold": args.relative_drop_threshold,
        "image_format": args.image_format,
        "jpeg_quality": args.jpeg_quality,
        "save_visualization": bool(args.save_visualization),
        "frame_step": args.frame_step,
        "car_confidence_threshold": float(car_cfg.get("car_confidence_threshold", 0.470)),
        "yolo": {
            "enabled": bool(yolo_cfg.get("enabled", True)),
            # 转绝对路径：避免当前工作目录不对时找不着
            "model_path": _abs_path(root, str(yolo_cfg.get("model_path", "yolov8n.pt"))),
            "device": str(yolo_cfg.get("device", "") or ""),
            "confidence": float(yolo_cfg.get("confidence", 0.40)),
            "iou": float(yolo_cfg.get("iou", 0.45)),
            "vehicle_classes": list(yolo_cfg.get("vehicle_classes", [1, 2, 3, 5, 7])),
            "enter_bottom_ratio": float(yolo_cfg.get("enter_bottom_ratio", 0.30)),
            "min_box_ratio": float(yolo_cfg.get("min_box_ratio", 0.02)),
        },
        "weights": dict(weights_cfg) if weights_cfg else None,
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

    log(f"[Config] 骤减阈值: {args.distance_drop}m(≥{args.min_car_points}点) | "
        f"波动阈值: {args.normal_fluctuation}m(≥{args.min_stable_points}点) | "
        f"有车确认: {args.car_confirm_frames}帧, 无车确认: {args.stable_confirm_frames}帧")
    log(f"[Config] 天空过滤: 基线≤{args.max_detect_depth}m | 相对跌幅≥{args.relative_drop_threshold:.0%}")
    log(f"[Config] 图像格式: {args.image_format} | JPEG质量: {args.jpeg_quality}")
    log(f"[Config] 保存可视化: {'是' if args.save_visualization else '否'}")

    # 处理每个视频（带错误处理，单个视频失败不影响其他视频）
    all_reports = []
    errors = []
    for vi, video_path in enumerate(files, 1):
        print(f"\n{'=' * 70}")
        log(f"[Progress] 处理视频 {vi}/{len(files)}: {Path(video_path).name}")

        try:
            cap_test = cv2.VideoCapture(video_path)
            w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_test.release()

            # 解析监测点
            pixel_points = _parse_pixel_points(
                args.pixel_depth_points, args.pixel_depth_count, w, h
            )
            if pixel_points:
                reason = "自定义坐标" if args.pixel_depth_points else f"1/5线自动分布({len(pixel_points)}点)"
                coords_str = ", ".join(f"({x},{y})" for x, y in pixel_points)
                print(f"[Pixel] 监测点: {reason}")
                print(f"[Pixel] 坐标: {coords_str}")

            report = process_video_car_detect(
                model, processor, video_path, run_dir, device,
                detector_config, pixel_points, args.frame_step,
                run_timestamp=run_timestamp
            )
            if report:
                all_reports.append(report)
        except Exception as e:
            err_msg = f"[ERROR] 视频 {Path(video_path).name} 处理失败: {e}"
            print(err_msg)
            import traceback
            tb = traceback.format_exc()
            print(tb)
            errors.append({"video": Path(video_path).name, "error": str(e), "traceback": tb})

    # 汇总报告
    print(f"\n{'=' * 70}")
    fail_str = f", {len(errors)} 个失败" if errors else ""
    log(f"[汇总] 处理完成: {len(all_reports)}/{len(files)} 个视频成功{fail_str}")
    total_segments = sum(r["no_car_segment_count"] for r in all_reports)
    total_saved = sum(r["total_frames_saved"] for r in all_reports)
    log(f"[汇总] 无车片段总数: {total_segments} 个 | 截取总帧数: {total_saved} 张")

    # 保存汇总报告
    summary_path = os.path.join(run_dir, f"{run_timestamp}_summary.json")
    summary = {
        "run_timestamp": run_timestamp,
        "total_videos_requested": len(files),
        "total_videos_succeeded": len(all_reports),
        "total_videos_failed": len(errors),
        "total_no_car_segments": total_segments,
        "total_frames_saved": total_saved,
        "config": detector_config,
        "reports": all_reports,
        "errors": errors,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[汇总] 报告文件: {os.path.abspath(summary_path)}")

    # 保存运行日志（先收集完整日志再写入文件）
    log_path = os.path.join(run_dir, f"{run_timestamp}_run_log.txt")
    log(f"[汇总] 日志文件: {os.path.abspath(log_path)}")
    log(f"[Run] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        if errors:
            f.write("\n\n" + "=" * 70 + "\n错误详情:\n")
            for e in errors:
                f.write(f"\n视频: {e['video']}\n错误: {e['error']}\n{e['traceback']}\n")


if __name__ == "__main__":
    main()
