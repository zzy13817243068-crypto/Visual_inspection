# -*- coding: utf-8 -*-
# ============================================================================
# 车辆侧面图像拼接脚本 - Vehicle Side Image Stitching
# ============================================================================
# 功能: 基于固定机位视频，通过特征点匹配实现车辆侧面图像的拼接与展开
# 依赖: opencv-python, numpy, ultralytics
# 作者: Auto-generated
# 版本: 1.0.0
# ============================================================================

import os
import sys
import cv2
import numpy as np
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ============================================================================
# 参数配置区 - Parameter Configuration
# ============================================================================

CONFIG = {
    # ---------- 路径配置 ----------
    "video_path": "/media/fst/data/视频/夏各庄入2/夏各庄入2",              # 输入视频路径
    "model_path": "/home/fst/zzy work/yolov26/vehicle_stitcher/models",             # 车辆检测模型路径
    "output_dir": "/home/fst/zzy work/yolov26/vehicle_stitcher/output/result",             # 输出目录
    "output_format": "png",       # 输出格式: png / jpg
    "save_intermediate": False,   # 是否保存中间帧

    # ---------- 车辆检测配置 ----------
    "confidence_threshold": 0.3,  # 车辆检测置信度阈值 (0-1, 降低以提高检测率)
    "iou_threshold": 0.45,        # NMS IoU 阈值 (0-1, 推荐 0.4-0.5)
    "vehicle_classes": [0],       # 车辆类别ID: 自定义模型为0; COCO格式为2(car)/7(truck)
    "bbox_padding_ratio": 0.15,   # 检测框外扩比例 (0-1, 增加15%范围保留更多车辆细节)

    # ---------- 特征点检测配置 ----------
    "feature_method": "sift",     # 特征点方法: sift(推荐) / orb / surf
    "max_features": 4000,         # 最大特征点数量 (增大以提高匹配数)
    "good_match_ratio": 0.8,      # Lowe's ratio test 阈值 (放宽以保留更多匹配)
    "min_match_count": 8,         # 最少有效匹配点数 (降低以适应小目标)

    # ---------- 位移与拼接配置 ----------
    "min_displacement": 2.0,      # 最小位移阈值(像素)，低于此值视为停止
    "max_displacement": 800.0,    # 最大位移阈值(像素)，超过则跳过
    "overlap_threshold": 0.1,     # 车框最小重叠率(0-1)
    "max_overlap": 0.98,          # 车框最大重叠率，超过视为重复帧

    # ---------- 融合与输出配置 ----------
    "blend_strength": 0.5,        # 融合强度 (0-1)
    "scale_factor": 1.0,         # 输出缩放因子
    "padding": 80,                # 输出画布内边距(像素)，增大以保留更多上下文

    # ---------- 性能配置 ----------
    "frame_stride": 1,            # 帧间隔处理 (1=每帧, 2=隔帧, 推荐 1-3)
    "process_resolution": None,   # 处理分辨率 (None=原分辨率, 或 (宽,高))
    "show_progress": True,        # 显示处理进度
}


# ============================================================================
# 模块一: 视频读取 - Video Reader
# ============================================================================

class VideoReader:
    """视频读取器，支持常见视频格式"""

    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self._opened = False
        self.metadata = {}

    def open(self):
        """打开视频文件并获取元数据"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

        ext = os.path.splitext(self.video_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的视频格式: {ext}，支持: {self.SUPPORTED_FORMATS}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        self.metadata = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'duration': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) /
                       max(self.cap.get(cv2.CAP_PROP_FPS), 1)
        }
        self._opened = True
        return self.metadata

    def read_frame(self):
        """读取下一帧"""
        if not self._opened:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """释放视频资源"""
        if self.cap:
            self.cap.release()
            self._opened = False

    def get_frame_at(self, frame_idx):
        """跳转到指定帧"""
        if not self._opened:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return self.read_frame()

    def __del__(self):
        self.release()


# ============================================================================
# 模块二: 车辆检测 - Vehicle Detector
# ============================================================================

class VehicleDetector:
    """车辆检测器，基于 YOLO 模型"""

    def __init__(self, model_path, confidence=0.5, iou=0.45, vehicle_classes=None):
        self.confidence = confidence
        self.iou = iou
        self.vehicle_classes = vehicle_classes or [2]
        self.model = None

        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics 未安装，请执行: pip install ultralytics")

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"车辆检测模型不存在: {model_path}")

        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        检测图像中的车辆
        
        返回:
            list: 检测到的车辆信息列表
                  [{bbox: [x1,y1,x2,y2], confidence: float, class_id: int, center: (cx,cy)}]
        """
        if frame is None:
            return []

        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                })

        return detections

    def select_primary_vehicle(self, detections, prev_center=None):
        """
        从多个检测结果中选出主要车辆
        
        参数:
            detections: 检测结果列表
            prev_center: 上一帧车辆中心点，用于追踪
        
        返回:
            dict or None: 选中的车辆信息
        """
        if not detections:
            return None

        if len(detections) == 1:
            return detections[0]

        if prev_center is not None:
            # 选择距离上一帧车辆最近的检测结果
            best = None
            min_dist = float('inf')
            for det in detections:
                cx, cy = det['center']
                dist = np.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best = det
            return best
        else:
            # 选择面积最大的车辆
            return max(detections, key=lambda d: d['area'])


# ============================================================================
# 模块三: 特征点检测与匹配 - Feature Detection & Matching
# ============================================================================

class FeatureMatcher:
    """特征点检测器与匹配器"""

    def __init__(self, method='orb', max_features=2000, good_match_ratio=0.75,
                 min_match_count=15):
        self.method = method
        self.max_features = max_features
        self.good_match_ratio = good_match_ratio
        self.min_match_count = min_match_count
        self.detector = self._create_detector()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING if method == 'orb' else cv2.NORM_L2)

    def _create_detector(self):
        """创建特征点检测器"""
        if self.method == 'orb':
            return cv2.ORB_create(nfeatures=self.max_features, scaleFactor=1.2,
                                  nlevels=8, edgeThreshold=15)
        elif self.method == 'sift':
            return cv2.SIFT_create(nfeatures=self.max_features)
        elif self.method == 'surf':
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"未知的特征点方法: {self.method}")

    def detect_and_compute(self, image):
        """
        检测特征点并计算描述子
        
        返回:
            keypoints, descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match(self, desc1, desc2, k=2):
        """
        使用 KNN 匹配并应用 Lowe's ratio test
        
        返回:
            list: 有效匹配列表
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=k)
        except cv2.error:
            return []

        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.good_match_ratio * n.distance:
                    good_matches.append(m)

        return good_matches

    def compute_displacement(self, kp1, kp2, good_matches):
        """
        计算特征点匹配的位移向量集合
        
        返回:
            dict: 位移统计信息
        """
        if len(good_matches) < self.min_match_count:
            return {'valid': False, 'reason': '匹配点不足', 'displacement': None}

        displacements = []
        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            displacements.append((dx, dy))

        if not displacements:
            return {'valid': False, 'reason': '无有效位移', 'displacement': None}

        displacements = np.array(displacements)
        median_dx = np.median(displacements[:, 0])
        median_dy = np.median(displacements[:, 1])

        # 过滤异常值
        distances = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
        median_dist = np.median(distances)
        valid_mask = distances < (median_dist + 3 * np.std(distances))
        valid_displacements = displacements[valid_mask]

        if len(valid_displacements) < self.min_match_count:
            return {'valid': False, 'reason': '有效位移不足', 'displacement': None}

        mean_dx = np.mean(valid_displacements[:, 0])
        mean_dy = np.mean(valid_displacements[:, 1])
        std_dx = np.std(valid_displacements[:, 0])
        std_dy = np.std(valid_displacements[:, 1])

        return {
            'valid': True,
            'displacement': (mean_dx, mean_dy),
            'median_displacement': (median_dx, median_dy),
            'std_displacement': (std_dx, std_dy),
            'match_count': len(good_matches),
            'valid_match_count': len(valid_displacements),
            'mean_distance': np.mean(distances[valid_mask]) if np.any(valid_mask) else 0,
            'confidence': self._calculate_confidence(valid_displacements)
        }

    def _calculate_confidence(self, displacements):
        """计算位移估计的置信度 (0-1)"""
        if len(displacements) < 5:
            return 0.3

        std_dx = np.std(displacements[:, 0])
        std_dy = np.std(displacements[:, 1])
        mean_dist = np.mean(np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2))

        if mean_dist < 1:
            return 0.1

        cv_score = 1.0 / (1.0 + (std_dx + std_dy) / (mean_dist + 1))
        count_score = min(1.0, len(displacements) / 20.0)

        return min(1.0, max(0.0, 0.5 * cv_score + 0.5 * count_score))


# ============================================================================
# 模块四: 车辆拼接器 - Vehicle Stitcher
# ============================================================================

class VehicleStitcher:
    """车辆图像拼接器"""

    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("output_dir", "./output")
        self.output_format = config.get("output_format", "png")
        self.save_intermediate = config.get("save_intermediate", False)

        self.stitch_result = None
        self.history = []
        # 不预先创建输出目录，由调用方在成功时创建

    def stitch(self, frames_data):
        """
        执行车辆拼接流程（优化版）
        
        参数:
            frames_data: 帧数据列表，每个元素为 (frame, vehicle_bbox)
        
        返回:
            np.ndarray: 拼接完成的图像
        """
        if not frames_data:
            raise ValueError("没有可拼接的帧数据")

        if len(frames_data) == 1:
            return frames_data[0][0]

        print(f"\n开始拼接 {len(frames_data)} 帧车辆图像...")

        # 初始化画布 - 使用更大的初始画布
        first_frame, first_bbox = frames_data[0]
        roi = self._extract_roi(first_frame, first_bbox)
        h, w = roi.shape[:2]

        # 预估画布大小：基于车辆宽度和帧数
        estimated_width = w * len(frames_data) + self.config.get("padding", 80) * 2
        canvas_height = h + self.config.get("padding", 80) * 2

        # 创建画布
        stitched = np.zeros((canvas_height, estimated_width, 3), dtype=np.uint8)
        padding = self.config.get("padding", 80)
        
        # 放置第一帧
        stitched[padding:padding + h, :w] = roi
        current_end_x = w  # 当前已拼接区域的右边界
        prev_roi = roi
        prev_bbox = first_bbox
        stitch_count = 0  # 成功拼接的帧数

        # 逐帧拼接
        for i, (frame, bbox) in enumerate(frames_data[1:], start=1):
            roi = self._extract_roi(frame, bbox)

            # 计算位移
            displacement_info = self._calculate_frame_displacement(prev_roi, roi)

            if not displacement_info['valid']:
                continue

            dx, dy = displacement_info['displacement']
            abs_dx = abs(dx)

            # 判断位移有效性
            if abs_dx < self.config.get("min_displacement", 2.0):
                continue

            if abs_dx > self.config.get("max_displacement", 800.0):
                continue

            # 计算重叠率
            overlap_ratio = self._calculate_overlap(prev_bbox, bbox)

            # 重叠过大说明几乎是同一位置，跳过
            if overlap_ratio > self.config.get("max_overlap", 0.98):
                continue

            # 计算新ROI应放置的位置
            # 车辆向右移动，dx为正，新区域应放置在current_end_x - dx处
            roi_w = roi.shape[1]
            roi_h = roi.shape[0]
            
            # 新位置：从当前右边界回溯dx像素开始放置
            new_x_start = int(current_end_x - abs_dx)
            new_x_start = max(0, new_x_start)
            
            # 垂直位置
            y_start = padding + int(dy)
            y_start = max(0, min(y_start, canvas_height - roi_h - 1))

            # 确保画布足够大
            required_width = new_x_start + roi_w + self.config.get("padding", 80)
            if required_width > stitched.shape[1]:
                new_canvas = np.zeros((canvas_height, required_width * 2, 3), dtype=np.uint8)
                new_canvas[:, :stitched.shape[1]] = stitched
                stitched = new_canvas

            # 融合放置新ROI
            # 计算重叠区域
            overlap_start = max(new_x_start, 0)
            overlap_end = min(new_x_start + roi_w, current_end_x)

            if overlap_start < overlap_end and overlap_start >= 0:
                # 有重叠区域 - 平滑融合
                overlap_w = overlap_end - overlap_start
                src_overlap_start = overlap_start - new_x_start
                for col in range(overlap_w):
                    alpha = col / max(overlap_w, 1)
                    src_col = src_overlap_start + col
                    dst_col = overlap_start + col
                    if 0 <= src_col < roi_w and 0 <= dst_col < stitched.shape[1]:
                        # 线性加权融合
                        stitched[y_start:y_start + roi_h, dst_col] = (
                            alpha * roi[:, src_col] +
                            (1 - alpha) * stitched[y_start:y_start + roi_h, dst_col]
                        ).astype(np.uint8)

                # 非重叠区域直接复制
                non_overlap_start = overlap_end - new_x_start
                for col in range(non_overlap_start, roi_w):
                    dst_col = new_x_start + col
                    if 0 <= dst_col < stitched.shape[1]:
                        stitched[y_start:y_start + roi_h, dst_col] = roi[:, col]
            else:
                # 无重叠 - 直接放置
                stitched[y_start:y_start + roi_h, new_x_start:new_x_start + roi_w] = roi

            # 更新状态
            current_end_x = max(current_end_x, new_x_start + roi_w)
            prev_roi = roi
            prev_bbox = bbox
            stitch_count += 1

            if self.config.get("show_progress", True) and stitch_count % 10 == 0:
                pct = i / len(frames_data) * 100
                print(f"  进度: {pct:.0f}% | 已拼接: {stitch_count} 帧 | 位移: {abs_dx:.1f}px")

        # 裁剪有效区域
        result = self._crop_result(stitched)

        if result is None or result.size == 0:
            raise RuntimeError("拼接结果为空")

        # 调整输出尺寸
        scale = self.config.get("scale_factor", 1.0)
        if scale != 1.0:
            result = cv2.resize(result, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)

        self.stitch_result = result
        self._save_result(result)

        print(f"\n拼接完成！")
        print(f"  有效帧数: {stitch_count}")
        print(f"  输出尺寸: {result.shape[1]} x {result.shape[0]}")
        print(f"  保存路径: {self.output_dir}")

        return result

    def _extract_roi(self, frame, bbox):
        """从帧中提取车辆区域，增加外扩以保留更多上下文"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # 外扩检测框，保留更多车辆上下文
        pad_ratio = self.config.get("bbox_padding_ratio", 0.15)
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        roi = frame[y1:y2, x1:x2]
        return roi

    def _calculate_frame_displacement(self, prev_roi, curr_roi):
        """计算相邻帧车辆的位移"""
        matcher = FeatureMatcher(
            method=self.config.get("feature_method", "orb"),
            max_features=self.config.get("max_features", 2000),
            good_match_ratio=self.config.get("good_match_ratio", 0.75),
            min_match_count=self.config.get("min_match_count", 15)
        )

        kp1, desc1 = matcher.detect_and_compute(prev_roi)
        kp2, desc2 = matcher.detect_and_compute(curr_roi)

        if desc1 is None or desc2 is None:
            return {'valid': False, 'reason': '特征点提取失败', 'displacement': None}

        good_matches = matcher.match(desc1, desc2)
        return matcher.compute_displacement(kp1, kp2, good_matches)

    def _calculate_overlap(self, bbox1, bbox2):
        """计算两个边界框的重叠率 (IoU)"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

        overlap_area = x_overlap * y_overlap
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - overlap_area

        if union_area == 0:
            return 0.0

        return overlap_area / union_area

    def _blend_and_stitch(self, canvas, new_roi, current_x, new_x, dy):
        """
        融合拼接两帧图像
        
        保持原始比例，避免拉伸变形
        """
        h, w = new_roi.shape[:2]
        padding = self.config.get("padding", 50)
        canvas_h = canvas.shape[0]

        # 计算放置位置
        y_start = padding + int(dy)
        y_start = max(0, min(y_start, canvas_h - h - 1))

        new_w = w + new_x
        if new_w > canvas.shape[1]:
            # 扩展画布
            new_canvas = np.zeros((canvas_h, new_w * 2, 3), dtype=np.uint8)
            new_canvas[:, :canvas.shape[1]] = canvas
            canvas = new_canvas

        blend_strength = self.config.get("blend_strength", 0.5)

        # 重叠区域平滑融合
        overlap_start = max(current_x, new_x)
        overlap_end = min(current_x + w, new_x + w)

        if overlap_start < overlap_end:
            # 重叠区融合
            overlap_width = overlap_end - overlap_start
            for col in range(overlap_width):
                alpha = col / max(overlap_width, 1)
                src_x = new_x + col
                dst_x = current_x - (overlap_width - col) + current_x - new_x
                dst_x = new_x + col
                if 0 <= src_x < canvas.shape[1] and 0 <= dst_x < canvas.shape[1]:
                    canvas[:, src_x] = (
                        alpha * new_roi[:, col - (overlap_start - new_x)] +
                        (1 - alpha) * canvas[:, src_x]
                    ).astype(np.uint8)

        # 非重叠区域直接复制
        no_overlap_start = overlap_end
        if no_overlap_start < new_x + w:
            src_start = no_overlap_start - new_x
            src_end = w
            if src_start < src_end:
                roi_slice = new_roi[:, src_start:src_end]
                dst_start = no_overlap_start
                dst_end = no_overlap_start + (src_end - src_start)
                if dst_end <= canvas.shape[1]:
                    canvas[y_start:y_start + roi_slice.shape[0],
                           dst_start:dst_end] = roi_slice

        new_current_x = max(current_x, new_x + w)
        return canvas, new_current_x

    def _crop_result(self, canvas):
        """裁剪有效区域，移除黑边"""
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        non_zero = cv2.findNonZero(gray)

        if non_zero is None:
            return canvas

        x, y, w, h = cv2.boundingRect(non_zero)
        padding = self.config.get("padding", 50)

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(canvas.shape[1] - x, w + 2 * padding)
        h = min(canvas.shape[0] - y, h + 2 * padding)

        return canvas[y:y + h, x:x + w]

    def _save_result(self, result):
        """保存拼接结果"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vehicle_side_{timestamp}.{self.output_format}"
        filepath = os.path.join(self.output_dir, filename)

        cv2.imwrite(filepath, result)
        print(f"  结果已保存: {filepath}")

    def _save_intermediate(self, canvas, frame_idx):
        """保存中间结果"""
        filename = f"intermediate_frame_{frame_idx:04d}.{self.output_format}"
        filepath = os.path.join(self.output_dir, "intermediate", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, canvas)

    def _print_progress(self, current, total, displacement, overlap):
        """打印处理进度"""
        pct = current / total * 100
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  [{bar}] {pct:.0f}%  位移:{displacement:.1f}px  重叠:{overlap:.2f}", end='')


# ============================================================================
# 模块五: 主控制器 - Main Controller
# ============================================================================

class VehicleStitcherPipeline:
    """车辆拼接完整流程控制器"""

    def __init__(self, config):
        self.config = config
        self.video_reader = None
        self.detector = None
        self.stitcher = None

    def run(self):
        """执行完整拼接流程"""
        print("=" * 60)
        print("车辆侧面图像拼接系统")
        print("=" * 60)

        try:
            self._init_components()
            frames_data = self._process_video()

            if not frames_data:
                print("❌ 未检测到任何有效车辆帧")
                return None

            result = self.stitcher.stitch(frames_data)
            
            if result is None or result.size == 0:
                print("❌ 拼接结果为空")
                return None

            self._save_log(frames_data)
            return result

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if self.video_reader:
                self.video_reader.release()

    def _init_components(self):
        """初始化各组件"""
        # 1. 视频读取
        video_path = self.config.get("video_path", "")
        if not video_path:
            raise ValueError("未配置视频路径 (video_path)")

        self.video_reader = VideoReader(video_path)
        metadata = self.video_reader.open()
        print(f"\n视频信息:")
        print(f"  分辨率: {metadata['width']}x{metadata['height']}")
        print(f"  帧率: {metadata['fps']:.1f} FPS")
        print(f"  总帧数: {metadata['total_frames']}")
        print(f"  时长: {metadata['duration']:.1f} 秒")

        # 2. 车辆检测
        model_path = self.config.get("model_path", "")
        if not model_path:
            raise ValueError("未配置模型路径 (model_path)")

        self.detector = VehicleDetector(
            model_path=model_path,
            confidence=self.config.get("confidence_threshold", 0.5),
            iou=self.config.get("iou_threshold", 0.45),
            vehicle_classes=self.config.get("vehicle_classes", [2])
        )
        print(f"  检测模型: {model_path}")

        # 3. 拼接器
        self.stitcher = VehicleStitcher(self.config)

    def _process_video(self):
        """处理视频，提取车辆帧"""
        frames_data = []
        prev_center = None
        frame_idx = 0
        frame_stride = self.config.get("frame_stride", 1)
        total_processed = 0

        print(f"\n{'='*60}")
        print("视频处理中...")
        print(f"{'='*60}")

        while True:
            frame = self.video_reader.read_frame()
            if frame is None:
                break

            frame_idx += 1

            # 帧间隔处理
            if frame_idx % frame_stride != 0:
                continue

            total_processed += 1

            # 可选分辨率调整
            process_res = self.config.get("process_resolution")
            if process_res:
                frame = cv2.resize(frame, process_res)

            # 车辆检测
            detections = self.detector.detect(frame)
            vehicle = self.detector.select_primary_vehicle(detections, prev_center)

            if vehicle is not None:
                frames_data.append((frame.copy(), vehicle['bbox']))
                prev_center = vehicle['center']

            # 进度显示
            total_frames = self.video_reader.metadata.get('total_frames', 0)
            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                if self.config.get("show_progress", True) and frame_idx % 100 == 0:
                    print(f"  处理进度: {pct:.1f}% ({frame_idx}/{total_frames})")

        print(f"\n视频处理完成:")
        print(f"  处理帧数: {total_processed}")
        print(f"  检测到车辆帧数: {len(frames_data)}")

        return frames_data

    def _save_log(self, frames_data):
        """保存处理日志"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "video_path": self.config.get("video_path"),
            "model_path": self.config.get("model_path"),
            "frames_detected": len(frames_data),
            "config": self.config
        }

        log_path = os.path.join(self.output_dir(), "process_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def output_dir(self):
        return self.config.get("output_dir", "./output")


# ============================================================================
# 模块六: 命令行接口 - CLI Interface
# ============================================================================

def _resolve_model_path(path):
    """
    解析模型路径，兼容文件夹和文件路径
    如果是文件夹，自动查找第一个 .pt 文件
    """
    if not path:
        return None
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith('.pt'):
                return os.path.join(path, f)
    return None


def _resolve_video_paths(path):
    """
    解析视频路径，兼容文件夹和文件路径
    如果是文件夹，自动查找所有视频文件
    返回视频文件路径列表
    """
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.ts')
    
    if not path:
        return []
    
    if os.path.isfile(path):
        return [path]
    
    if os.path.isdir(path):
        videos = []
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(path, f))
        return videos
    
    return []


def build_arg_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='车辆侧面图像拼接系统 - Vehicle Side Image Stitching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用代码中配置的默认路径直接运行
  python vehicle_stitcher.py

  # 指定视频文件
  python vehicle_stitcher.py --video video.mp4

  # 指定视频文件夹（自动查找所有视频）
  python vehicle_stitcher.py --video ./videos/

  # 指定模型路径
  python vehicle_stitcher.py --model best.pt

  # 同时指定多个参数
  python vehicle_stitcher.py --video ./videos/ --model ./models/ --output ./result/

参数说明:
  --video             输入视频路径或文件夹 (可选，使用代码默认配置)
  --model             车辆检测模型路径或文件夹 (可选，使用代码默认配置)
  --output            输出目录 (默认: ./output)
  --confidence        检测置信度阈值 (默认: 0.5)
  --feature           特征点方法 orb/sift/surf (默认: orb)
  --min-displacement  最小位移阈值 (默认: 5.0)
  --save-intermediate 保存中间帧
        """
    )

    parser.add_argument('--video', type=str, default=None,
                        help='输入视频路径或文件夹 (可选，默认使用代码中配置的路径)')
    parser.add_argument('--model', type=str, default=None,
                        help='车辆检测模型路径或文件夹 (可选，默认使用代码中配置的路径)')
    parser.add_argument('--output', type=str, default='./output',
                        help='输出目录路径')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='检测置信度阈值 (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值 (0-1)')
    parser.add_argument('--feature', type=str, default='orb',
                        choices=['orb', 'sift', 'surf'],
                        help='特征点检测方法')
    parser.add_argument('--max-features', type=int, default=2000,
                        help='最大特征点数量')
    parser.add_argument('--min-match', type=int, default=15,
                        help='最少有效匹配点数')
    parser.add_argument('--min-displacement', type=float, default=5.0,
                        help='最小位移阈值(像素)')
    parser.add_argument('--frame-stride', type=int, default=1,
                        help='帧间隔')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'jpg'],
                        help='输出图像格式')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='保存中间处理结果')
    parser.add_argument('--no-progress', action='store_true',
                        help='不显示进度')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='输出缩放因子')

    return parser


def main():
    """主入口函数"""
    parser = build_arg_parser()
    args = parser.parse_args()

    # 解析模型路径（命令行参数优先，其次使用代码配置）
    model_path = _resolve_model_path(args.model) or _resolve_model_path(CONFIG.get("model_path", ""))
    if not model_path:
        print("❌ 未找到有效的模型文件，请通过 --model 参数或修改代码中 CONFIG['model_path'] 指定")
        sys.exit(1)

    # 解析视频路径（命令行参数优先，其次使用代码配置）
    video_input = args.video if args.video else CONFIG.get("video_path", "")
    video_paths = _resolve_video_paths(video_input)
    
    if not video_paths:
        print(f"❌ 未找到有效的视频文件，请通过 --video 参数或修改代码中 CONFIG['video_path'] 指定")
        print(f"   支持的格式: .mp4 .avi .mov .mkv .flv .wmv .ts")
        sys.exit(1)

    print("=" * 60)
    print("车辆侧面图像拼接系统")
    print("=" * 60)
    print(f"\n找到 {len(video_paths)} 个视频文件:")
    for i, v in enumerate(video_paths, 1):
        print(f"  {i}. {v}")
    print(f"\n模型路径: {model_path}")
    print(f"输出目录: {args.output}\n")

    # 逐个处理视频
    success_count = 0
    fail_count = 0
    
    for idx, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(video_paths)}] 正在处理: {os.path.basename(video_path)}")
        print(f"{'='*60}")

        # 先运行检测和拼接，不预先创建输出目录
        config = CONFIG.copy()
        config.update({
            "video_path": video_path,
            "model_path": model_path,
            "output_dir": args.output,  # 先指向根目录，成功后再创建子目录
            "output_format": args.format,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "feature_method": args.feature,
            "max_features": args.max_features,
            "min_match_count": args.min_match,
            "min_displacement": args.min_displacement,
            "frame_stride": args.frame_stride,
            "save_intermediate": args.save_intermediate,
            "show_progress": not args.no_progress,
            "scale_factor": args.scale,
        })

        # 运行流水线
        pipeline = VehicleStitcherPipeline(config)
        result = pipeline.run()

        if result is not None:
            # 成功时才创建输出子目录并保存结果
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(args.output, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # 移动已保存的结果到子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"vehicle_side_{timestamp}.{args.format}"
            result_filepath = os.path.join(video_output_dir, result_filename)
            cv2.imwrite(result_filepath, result)
            
            # 移动日志文件（如果存在）
            log_src = os.path.join(args.output, "process_log.json")
            if os.path.exists(log_src):
                log_dst = os.path.join(video_output_dir, "process_log.json")
                import shutil
                shutil.move(log_src, log_dst)
            
            print(f"✅ [{idx}/{len(video_paths)}] 拼接成功！图像尺寸: {result.shape[1]}x{result.shape[0]}")
            print(f"   结果保存在: {video_output_dir}")
            success_count += 1
        else:
            # 失败时不创建任何目录
            print(f"❌ [{idx}/{len(video_paths)}] 拼接失败: {os.path.basename(video_path)}")
            print(f"   未创建输出目录")
            fail_count += 1

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"处理完成汇总")
    print(f"{'='*60}")
    print(f"  总视频数: {len(video_paths)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  输出目录: {args.output}")


if __name__ == '__main__':
    main()