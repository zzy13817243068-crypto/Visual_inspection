"""
车辆检测切片脚本：靠监测点距离变化判断有没有车，按状态分类切片。

原理挺简单——
- 有车经过：车挡住背景，监测点深度值会骤降
- 没车：深度值基本贴在背景基线附近

跑起来大致是：
1. 读视频，用 DA2-Metric 模型抽每帧 5 个监测点的深度值
2. 维护背景基线（就是没车时各点的深度），深度骤减就判有车
3. 连续同状态的帧切成一段，所有状态的段都存标注图（左原视频+右深度图带米数状态条）：
   no_car 段 → run 目录下的 no_car/，car/unknown 等其他段 → other/
4. 结尾打印两个目录各自的切片数（只出图，不写任何 json 报告）

输出结构（每次运行一个 run 目录，一级子目录只有这两个）：
  run_YYYYMMDD_HHMMSS/
    ├─ no_car/   ← no_car 状态切片（所有视频平铺，标注图）
    ├─ other/    ← 其他状态切片（所有视频平铺，标注图）
    └─ {时间戳}_run_log.txt

用法：
  python detect/detect_car.py
  python detect/detect_car.py --source detect/input/xxx.mp4
  python detect/detect_car.py --distance-drop 4.0 --stable-time 3.0
"""

import os
import sys
import time
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
    load_baseline_depths, apply_clahe, build_detector_config,
    build_detector_pipeline, _infer_depth_map, _extract_pixel_depths,
)

# MLP 判定模型集成（实验性，可选）。import 失败不影响 v8 逻辑。
try:
    from judge_integration import get_judge_inference, judge_infer_frame
    _JUDGE_AVAILABLE = True
except Exception:
    get_judge_inference = None
    judge_infer_frame = None
    _JUDGE_AVAILABLE = False


# v8 融合判定（单帧，和 detect_sel_videos.py 共用这一份）
# 省得两边各维护一份、逻辑对不上；原则是 YOLO-only-boost——
# 只有当前帧 YOLO 检测到才加权，否则深度主导
def fuse_v8_car_state(car_state, detector, motion_detector, pixel_depths, yolo_boxes, last_final_state,
                      brightness_monitor=None, prev_d_min=None, prev_state_idx=2, judge=None):
    """
    单帧 v8 综合判定（策略 A：v8 为主，MLP 只在高置信度时纠正分歧帧）：
      - car_state: VehicleDetector.update() 内部加权 + 状态机的输出（car/no_car/unknown）
      - 其他检测器/上下文：detector(取 conf/R5)、motion_detector、pixel_depths(深度点)、
        yolo_boxes(当前帧 YOLO 框)、last_final_state(上一帧最终 state，当前帧 HOLD 用)
      - brightness_monitor/prev_d_min/prev_state_idx: MLP 推理要的上下文（可选，v8 用不上）
      - judge: JudgeMLPInference 实例（可选）；非 None 时只辅助纠正，不替代 v8。
    返回: (state 最终单帧状态, 新的 last_final_state)
    """
    mog2_confirmed = motion_detector.is_confirmed() if motion_detector else False
    depth_conf = detector.last_confidence if detector else 0.0
    vision_score = detector.last_vision_score if detector else 0.0
    yolo_boxes_n = len(yolo_boxes)
    vision_strong = vision_score >= 0.70 and yolo_boxes_n >= 1
    vision_weak_see = vision_score >= 0.45 and yolo_boxes_n >= 1
    yolo_blind = (yolo_boxes_n == 0) or (vision_score < 0.30)

    # 当前帧 HOLD：YOLO 真检测到才锁，防闪烁
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
        elif yolo_blind and depth_conf >= 0.45:
            # 漏检修复（深度主导）：YOLO 漏检但深度信号还挺强(≥0.45) → 放行 car
            # 门槛从 0.35 提到 0.45：车没到线只是靠近时，深度小波动 conf 常在 0.35~0.44，
            # 旧门槛会提前误报 car；真压到监测点了 conf 一般 ≥0.45
            state = "car"
        elif depth_conf > 0.30:
            # 已 car 状态深度兜底：状态机还判 car，深度也没明显坍塌(>0.30) → 保持 car
            # 门槛从 0.20 提到 0.30：纯靠状态惯性太松，"提前误报"会一直延续，0.30 在防误报和防闪断之间折中
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

    # 策略 A：MLP 辅助纠正，只在"v8 和 MLP 分歧"且"MLP 很确定"时才覆盖 v8
    if judge is not None:
        mlp_result = judge_infer_frame(
            judge, pixel_depths, detector, motion_detector, brightness_monitor,
            prev_d_min, prev_state_idx,
        )
        if mlp_result is not None:
            mlp_state, mlp_conf = mlp_result
            # 只有 v8 跟 MLP 判的不一样、且 MLP 置信度 ≥0.70 时才纠正
            if mlp_state != state and mlp_conf >= 0.70:
                state = mlp_state  # MLP 高置信度才纠正 v8

    last_final_state = state
    return state, last_final_state


# 状态片段记录器：连续同状态算一段，按状态分进 no_car / other 两个目录
class StateSegmentRecorder:
    """
    记片段并把帧截下来存盘，所有状态都切：
      - state == "no_car" 的片段帧 → no_car 目录
      - car / unknown 等其他状态  → other 目录
    帧平铺存放，文件名带时间戳+视频名+片信号+帧号，不会撞；
    真撞了就加 _xN 后缀，不覆盖不丢帧。
    """

    def __init__(self, no_car_dir, other_dir, video_name, image_format="jpg", jpeg_quality=95, run_timestamp=""):
        self.no_car_dir = no_car_dir
        self.other_dir = other_dir
        self.video_name = video_name
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.run_timestamp = run_timestamp

        self.segments = []           # 已完成的片段列表
        self.current_segment = None  # 正在记录的片段
        self.frame_buffer = []       # 当前片段的帧缓存
        self.saved_no_car = 0        # 存进 no_car/ 的帧数
        self.saved_other = 0         # 存进 other/ 的帧数
        self.save_errors = 0         # 写盘失败帧数

    def _start_segment(self, state, start_frame, start_time):
        self.current_segment = {
            "state": state,
            "start_frame": start_frame,
            "start_time": start_time,
            "end_frame": None,
            "end_time": None,
            "frame_count": 0,
        }
        self.frame_buffer = []

    def add_frame(self, frame, frame_idx, timestamp, state):
        """喂一帧；状态变了就结掉旧片段、开新片段"""
        if self.current_segment is None:
            self._start_segment(state, frame_idx, timestamp)
        elif state != self.current_segment["state"]:
            self.end_segment(frame_idx, timestamp)
            self._start_segment(state, frame_idx, timestamp)
        self.frame_buffer.append((frame.copy(), frame_idx, timestamp))

    def _unique_path(self, target_dir, fname):
        """防覆盖：同名已存在就加 _xN 后缀"""
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            return fpath
        stem, ext = os.path.splitext(fname)
        n = 1
        while os.path.exists(fpath):
            fpath = os.path.join(target_dir, f"{stem}_x{n}{ext}")
            n += 1
        return fpath

    def _save_one(self, fpath, frame):
        """写单帧。imencode+tofile，中文路径/文件名也不会静默失败"""
        ext = "." + self.image_format.lower()
        if ext in (".jpg", ".jpeg"):
            ok, buf = cv2.imencode(ext, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        else:
            ok, buf = cv2.imencode(ext, frame)
        if not ok:
            return False
        buf.tofile(fpath)
        return True

    def end_segment(self, end_frame, end_time):
        """结束当前片段，按状态写到对应目录"""
        if self.current_segment is None:
            return

        self.current_segment["end_frame"] = end_frame
        self.current_segment["end_time"] = end_time
        self.current_segment["frame_count"] = len(self.frame_buffer)

        seg_state = self.current_segment["state"]
        target_dir = self.no_car_dir if seg_state == "no_car" else self.other_dir
        seg_idx = len(self.segments) + 1
        for frame, fidx, ts in self.frame_buffer:
            fname = f"{self.run_timestamp}_{self.video_name}_seg{seg_idx:03d}_frame{fidx:06d}_t{ts:07.2f}s.{self.image_format}"
            fpath = self._unique_path(target_dir, fname)
            try:
                if self._save_one(fpath, frame):
                    if seg_state == "no_car":
                        self.saved_no_car += 1
                    else:
                        self.saved_other += 1
                else:
                    self.save_errors += 1
            except Exception:
                self.save_errors += 1

        self.segments.append(self.current_segment)
        self.current_segment = None
        self.frame_buffer = []

    def is_recording(self):
        return self.current_segment is not None

    def get_stats(self):
        return {
            "segment_count": len(self.segments),
            "total_frames_saved": self.saved_no_car + self.saved_other,
            "saved_no_car": self.saved_no_car,
            "saved_other": self.saved_other,
            "save_errors": self.save_errors,
            "segments": self.segments,
        }


# 视频处理主函数
def process_video_car_detect(model, processor, video_path, output_dir, no_car_dir, other_dir,
                             device, detector_config, pixel_points=None, frame_step=1,
                             run_timestamp=""):
    """处理视频：一帧帧判车辆状态，按状态切片分类存盘。

    参数：model / processor 是 DA2 的模型和 processor，video_path 视频路径，
    output_dir 是 run 根目录（日志放这），no_car_dir / other_dir 是两个切片目录，
    device 是 torch.device，detector_config 是检测器配置 dict，
    pixel_points 是监测点坐标 list[(x,y)]，frame_step 帧采样间隔 (1=每帧, 2=隔帧)。
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

    # 初始化检测流水线（深度检测器 + YOLO视觉 + 亮度监听 + MOG2 运动检测）
    detector, brightness_monitor, motion_detector = build_detector_pipeline(
        detector_config, pixel_points, effective_fps,
        with_vision=True, verbose="full", yolo_model_path_default="yolov8n.pt")

    # P1：CLAHE 夜间去眩光 + 监测点饱和屏蔽
    clahe_cfg = detector_config.get("clahe", {})
    clahe_enabled = bool(clahe_cfg.get("enabled", False))
    clahe_clip = float(clahe_cfg.get("clip_limit", 2.0))
    clahe_tile = int(clahe_cfg.get("tile_grid_size", 8))
    sat_threshold = int(detector_config.get("saturation_threshold", 0))
    # 过曝屏蔽夜间专属：白天白车身像素会顶到 255 被误杀成 nan，白天不屏蔽
    sat_night_only = bool(detector_config.get("sat_night_only", True))

    # 背景基线预设（可选）
    # 配置了 baseline_dir 时，用背景图/视频的深度作为基线，跳过视频预热
    baseline_dir = detector_config.get("baseline_dir", "") or ""
    if baseline_dir and detector and pixel_points:
        baseline_dir_abs = _abs_path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), baseline_dir)
        max_frames = int(detector_config.get("baseline_max_frames", 30))
        bg_depths = load_baseline_depths(
            baseline_dir_abs, model, processor, device, pixel_points, max_frames)
        detector.preset_baseline(bg_depths)

    # 初始化片段记录器（双目录分类）
    recorder = StateSegmentRecorder(
        no_car_dir=no_car_dir,
        other_dir=other_dir,
        video_name=video_name,
        image_format=detector_config["image_format"],
        jpeg_quality=detector_config["jpeg_quality"],
        run_timestamp=run_timestamp,
    )

    # 状态统计
    state_counts = {"car": 0, "no_car": 0, "unknown": 0}
    signal_times = []   # 每次有车信号的时间戳（一车一信号验收用）
    last_sig_n = 0
    frame_idx = 0
    processed_total = (total_frames + frame_step - 1) // frame_step
    pbar = tqdm(total=processed_total, desc=f"检测 {video_name[:15]}", ncols=130, leave=True)

    # v8 融合：YOLO 检测到才加权，检测不到就让深度 + 运动占大头（用户定的规则）
    # 用户定的规则：
    #  - YOLO 在检测线上有框（R5≥0.45）→ 加权 / HOLD 锁，只管当帧
    #  - YOLO 没框（R5=0 或 无框）→ 深度加权为主、运动辅助；不能因为"上一帧靠 YOLO 锁了 car"就跨帧续命（HOLD 续命帧直接删掉）
    #  不然：车到检测线、YOLO 刚好漏检 → 卡在 no_car 一帧，后面 YOLO 回来了也拉不上去
    last_final_state = "unknown"

    # MLP 判定模型集成（实验性；开关关了就是 judge=None，没影响）
    judge = get_judge_inference(detector_config) if _JUDGE_AVAILABLE else None
    if judge is not None:
        judge.reset()  # 每个视频清空时序窗口
    prev_d_min = None          # 上一帧深度 min（MLP 时序特征用）
    prev_state_idx = 2         # 上一帧状态索引（unknown=2）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 隔帧采样：只处理 frame_step 倍数的帧
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps if fps > 0 else 0

        # P1：CLAHE 夜间去眩光/提暗部，深度模型和 YOLO 共用这一张增强帧
        if clahe_enabled:
            infer_frame = apply_clahe(frame, clip_limit=clahe_clip,
                                      tile_grid_size=(clahe_tile, clahe_tile))
        else:
            infer_frame = frame

        depth, frame_time = _infer_depth_map(model, processor, infer_frame, device)

        # 提取监测点深度（P0-2 车灯过曝屏蔽：像素饱和了就认为这个点深度不可信 → nan）
        # 屏蔽夜间专属：白天白车身也会顶到 255，不分昼夜屏蔽会把关键监测点全变 nan 漏判车
        sat_now = sat_threshold if (not sat_night_only or getattr(detector, "_is_night", False)) else 0
        pixel_depths = _extract_pixel_depths(
            depth, pixel_points or [], frame, sat_now)

        # 更新检测器（深度 + YOLO视觉协同 + MOG2 融合）
        car_state = None
        motion_state = None
        yolo_boxes = []
        if detector and pixel_depths:
            # 关键是传 frame_bgr 给 update()：YOLO 先跑出 R5，再跟 R1-R4 的深度加权综合
            # 视觉不单独触发，只算 R5 这一条规则
            car_state = detector.update(pixel_depths, infer_frame)
            yolo_boxes = list(getattr(detector, '_last_yolo_boxes', []))
            # 有车信号沿：一辆车经过只发一次（检测器信号层保证）
            if detector.signal_count > last_sig_n:
                signal_times.append(round(timestamp, 2))
                print(f"\n  [信号] 第{detector.signal_count}次有车信号 t={timestamp:.2f}s")
                last_sig_n = detector.signal_count
        if motion_detector:
            motion_state = motion_detector.update(frame)

        # 融合判定：v8 是 YOLO-only-boost + 深度主导，逻辑直接走共享函数 fuse_v8_car_state()（跟 detect_sel_videos.py 一份逻辑）
        # judge 非 None 就顺带让 MLP 纠正一下，出错自动退回 v8
        state, last_final_state = fuse_v8_car_state(
            car_state=car_state,
            detector=detector,
            motion_detector=motion_detector,
            pixel_depths=pixel_depths,
            yolo_boxes=yolo_boxes,
            last_final_state=last_final_state,
            brightness_monitor=brightness_monitor,
            prev_d_min=prev_d_min,
            prev_state_idx=prev_state_idx,
            judge=judge,
        )

        state_counts[state] = state_counts.get(state, 0) + 1

        # 更新 MLP 时序上下文（judge=None 时这两行只是普通变量更新，无副作用）
        valid_depths = [d for d in (pixel_depths or []) if d is not None and not (isinstance(d, float) and d != d)]
        if valid_depths:
            prev_d_min = float(min(valid_depths))
        prev_state_idx = {"car": 0, "no_car": 1, "unknown": 2}.get(state, 2)

        # 标注图：左原视频 + 右深度图（带监测点米数、状态条），每帧都生成，切片存的就是它
        depth_vis, d_min, d_max = depth_to_colormap(depth)
        depth_vis = annotate_depth(
            depth_vis, d_min, d_max, h,
            pixel_points, pixel_depths, frame_time
        )
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

        # 片段记录：连续同状态为一段，状态切换即切段（全状态都切，存的是标注图）
        recorder.add_frame(combined, frame_idx, timestamp, state)

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
        recorder.end_segment(frame_idx, timestamp if total_frames > 0 else 0.0)

    # 统计（不落盘任何 json，只打印 + 返回给 main 汇总）
    stats = recorder.get_stats()
    no_car_seg_n = sum(1 for s in stats["segments"] if s["state"] == "no_car")
    other_seg_n = stats["segment_count"] - no_car_seg_n

    # 打印摘要
    print(f"\n  [摘要] 视频时长: {duration:.1f}s | 总帧数: {total_frames}")
    print(f"  [摘要] 状态统计: 有车={state_counts.get('car',0)}帧, "
          f"无车={state_counts.get('no_car',0)}帧, "
          f"未知={state_counts.get('unknown',0)}帧")
    print(f"  [摘要] 片段: no_car={no_car_seg_n}个 other={other_seg_n}个 | "
          f"切片: no_car={stats['saved_no_car']}张 other={stats['saved_other']}张"
          + (f" | 写盘失败 {stats['save_errors']} 张" if stats["save_errors"] else ""))
    sig_n = detector.signal_count if detector else 0
    print(f"  [摘要] 有车信号: {sig_n} 次" +
          (f" ({', '.join(f'{t}s' for t in signal_times)})" if signal_times else ""))
    for i, s in enumerate(stats["segments"]):
        print(f"    片段{i+1}[{s['state']}]: {s['start_time']:.1f}s ~ {s['end_time']:.1f}s "
              f"({s['end_time']-s['start_time']:.1f}s, {s['frame_count']}帧)")

    return {
        "video_name": video_name,
        "state_counts": state_counts,
        "no_car_segment_count": no_car_seg_n,
        "other_segment_count": other_seg_n,
        "total_frames_saved": stats["total_frames_saved"],
        "saved_no_car": stats["saved_no_car"],
        "saved_other": stats["saved_other"],
        "save_errors": stats["save_errors"],
        "car_signals": detector.signal_count if detector else 0,
        "signal_times": signal_times,
    }


# 主函数
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

    # 每次运行创建带时间戳的唯一输出目录，下面只建两个一级子目录
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output, f"run_{run_timestamp}")
    no_car_dir = os.path.join(run_dir, "no_car")
    other_dir = os.path.join(run_dir, "other")
    os.makedirs(no_car_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)

    # 日志收集（同时输出到控制台和日志文件）
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"[Output] 本次输出目录: {os.path.abspath(run_dir)}")
    log(f"[Output] no_car 切片目录: {os.path.abspath(no_car_dir)}")
    log(f"[Output] 其他状态切片目录: {os.path.abspath(other_dir)}")
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
    # 标量参数从 CLI(args.*) 来，yolo/mog2/weights/judge_mlp/night/saturation/clahe 这些子块走公共工厂
    detector_config = build_detector_config(
        car_cfg, root,
        {
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
            "frame_step": args.frame_step,
            "car_confidence_threshold": float(car_cfg.get("car_confidence_threshold", 0.470)),
            "baseline_dir": str(car_cfg.get("baseline_dir", "") or ""),
            "baseline_max_frames": int(car_cfg.get("baseline_max_frames", 30)),
        },
        yolo_model_default="yolov8n.pt",
    )

    log(f"[Config] 骤减阈值: {args.distance_drop}m(≥{args.min_car_points}点) | "
        f"波动阈值: {args.normal_fluctuation}m(≥{args.min_stable_points}点) | "
        f"有车确认: {args.car_confirm_frames}帧, 无车确认: {args.stable_confirm_frames}帧")
    log(f"[Config] 天空过滤: 基线≤{args.max_detect_depth}m | 相对跌幅≥{args.relative_drop_threshold:.0%}")
    log(f"[Config] 图像格式: {args.image_format} | JPEG质量: {args.jpeg_quality}")

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
                model, processor, video_path, run_dir, no_car_dir, other_dir,
                device, detector_config, pixel_points, args.frame_step,
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

    # 汇总统计
    print(f"\n{'=' * 70}")
    fail_str = f", {len(errors)} 个失败" if errors else ""
    log(f"[汇总] 处理完成: {len(all_reports)}/{len(files)} 个视频成功{fail_str}")
    total_no_car_segs = sum(r["no_car_segment_count"] for r in all_reports)
    total_other_segs = sum(r["other_segment_count"] for r in all_reports)
    total_saved_nc = sum(r["saved_no_car"] for r in all_reports)
    total_saved_ot = sum(r["saved_other"] for r in all_reports)
    total_save_err = sum(r.get("save_errors", 0) for r in all_reports)
    log(f"[汇总] 片段: no_car={total_no_car_segs} 个 | other={total_other_segs} 个")
    log(f"[汇总] 切片保存: no_car={total_saved_nc} 张 | other={total_saved_ot} 张"
        + (f" | 写盘失败 {total_save_err} 张" if total_save_err else ""))

    # 按磁盘实际文件数统计两个目录
    def _count_dir(d):
        try:
            return sum(1 for n in os.listdir(d)
                       if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png"))
        except Exception:
            return 0

    disk_nc = _count_dir(no_car_dir)
    disk_ot = _count_dir(other_dir)
    log(f"[统计] no_car/ 目录: 切片 {disk_nc} 个")
    log(f"[统计] other/  目录: 切片 {disk_ot} 个")

    # 一车一信号验收：每视频有车信号数分布 + 复核清单
    # 合格标准：连续100台以上车辆测试，无重复触发、无漏触发
    sig_dist = {}
    for r in all_reports:
        n = r.get("car_signals", 0)
        sig_dist[n] = sig_dist.get(n, 0) + 1
    log(f"[验收] 每视频有车信号数分布(信号数:视频个数): {dict(sorted(sig_dist.items()))}")
    zero_sig = [r["video_name"] for r in all_reports if r.get("car_signals", 0) == 0]
    if zero_sig:
        log(f"[验收] 信号=0 的视频 {len(zero_sig)} 个(复核是漏判还是本来无车): "
            f"{', '.join(zero_sig[:10])}{' ...' if len(zero_sig) > 10 else ''}")
    multi_sig = [(r["video_name"], r["car_signals"]) for r in all_reports
                 if r.get("car_signals", 0) >= 2]
    if multi_sig:
        log(f"[验收] 信号≥2 的视频 {len(multi_sig)} 个(复核是重复触发还是多车): "
            f"{', '.join(f'{n}({c}次)' for n, c in multi_sig[:10])}"
            f"{' ...' if len(multi_sig) > 10 else ''}")

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
