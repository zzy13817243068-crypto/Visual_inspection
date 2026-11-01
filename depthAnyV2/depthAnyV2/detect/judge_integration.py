"""
MLP 判定模型集成模块（实验性，能一键停用）。

就是把 train_judge.py 训练出来的判定 MLP 挂到 detect_car.py 的判定流程里。几条原则：
detect.py 的代码一行都不动；只在 detect_car.py 的 fuse_v8_car_state 顶部插一个分支；
开关关掉、模型加载失败或者推理出异常时，都自动退回原来的 v8 逻辑；
集成代码尽量都塞在这个文件，方便以后优化或直接删掉。

在 detect_car.py 里这么用：
    from judge_integration import get_judge_inference, judge_infer_frame

    judge = get_judge_inference(car_detect_config)   # 全局单例，开关关就返回 None
    state = judge_infer_frame(judge, ...)            # 返回 None 就走 v8
"""

import os
import numpy as np
import torch
import torch.nn as nn

# 常量跟 train_judge.py 保持一致
FEATURE_DIM = 25
STATE_LIST = ["car", "no_car", "unknown"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_LIST)}
IDX_TO_STATE = {i: s for s, i in STATE_TO_IDX.items()}
YOLO_CONF_THRESHOLD = 0.40


# 模型定义，和 train_judge.py 里的 JudgeMLP 完全一致。
class JudgeMLP(nn.Module):
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


# 特征构建，和 train_judge.py 的 build_feature 完全一致。
def car_crossed_line(yolo_boxes, detect_x=40, direction="any"):
    """用 YOLO 框坐标判断车是否越过检测线。"""
    if not yolo_boxes:
        return False, None
    for box in yolo_boxes:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        conf = float(box[4]) if len(box) > 4 else 0.0
        if conf < YOLO_CONF_THRESHOLD:
            continue
        cx = (x1 + x2) / 2
        if direction == "any":
            crossed = (x1 <= detect_x <= x2) or (cx <= detect_x)
        elif direction == "right":
            crossed = x1 > detect_x
        elif direction == "left":
            crossed = x2 >= detect_x
        else:
            crossed = (x1 <= detect_x <= x2) or (cx <= detect_x)
        if crossed:
            return True, {"x1": x1, "x2": x2, "conf": conf, "cx": cx}
    return False, None


def build_inference_feature(pixel_depths, detector, motion_detector, brightness_monitor,
                            prev_d_min, prev_state_idx):
    """
    构建单帧 25 维特征向量，和 train_judge.py 的 build_feature 完全一致。
    注意：这里不用 v8 的 depth_conf / r5，全用原始信号。
    """
    valid = [d for d in (pixel_depths or []) if d is not None and not (isinstance(d, float) and d != d)]
    if valid:
        d_min = float(min(valid))
        d_max = float(max(valid))
        d_mean = float(np.mean(valid))
        d_std = float(np.std(valid))
    else:
        d_min = d_max = d_mean = d_std = 0.0

    baseline = getattr(detector, "baseline", None) if detector else None
    if baseline is not None and len(baseline):
        bl_arr = np.asarray(baseline, dtype=np.float32)
        bl_min = float(np.nanmin(bl_arr))
        bl_mean = float(np.nanmean(bl_arr))
    else:
        bl_min = bl_mean = 0.0
    drop_abs = (bl_min - d_min) if bl_min > 0 else 0.0
    drop_rel = (drop_abs / bl_min) if bl_min > 0 else 0.0

    yolo_boxes = list(getattr(detector, "_last_yolo_boxes", []) or []) if detector else []
    yolo_max_conf = max((float(b[4]) for b in yolo_boxes), default=0.0)
    yolo_n = len(yolo_boxes)
    yolo_min_x1 = min((float(b[0]) for b in yolo_boxes), default=9999.0)
    yolo_max_x2 = max((float(b[2]) for b in yolo_boxes), default=0.0)
    yolo_nearest_cx = min((abs(((float(b[0]) + float(b[2])) / 2) - 40) for b in yolo_boxes), default=9999.0)

    motion_px = float(getattr(motion_detector, "motion_pixel_count", 0) if motion_detector else 0)
    mog2_confirmed = 1.0 if (motion_detector and motion_detector.is_confirmed()) else 0.0
    bright_val = 0.0
    if brightness_monitor:
        sb = getattr(brightness_monitor, "smoothed_brightness", None)
        if sb is not None:
            bright_val = float(sb)
        else:
            lvl_map = {"daytime": 2.0, "twilight": 1.0, "night": 0.0}
            lvl = brightness_monitor.get_brightness_level()
            bright_val = lvl_map.get(lvl, 0.0) if lvl else 0.0
    mog2_w = float(motion_detector.get_current_weight() if motion_detector and hasattr(motion_detector, "get_current_weight") else 1.0)

    depth_drop_flag = 1.0 if (drop_abs >= 0.4 and drop_rel >= 0.08) else 0.0
    yolo_crossed = 1.0 if car_crossed_line(yolo_boxes, detect_x=40, direction="any")[0] else 0.0

    p_dmin = float(prev_d_min) if prev_d_min is not None else d_min
    depth_diff_rate = (d_min - p_dmin) / p_dmin if p_dmin > 0 else 0.0

    prev_oh = [0.0, 0.0, 0.0]
    if 0 <= prev_state_idx < 3:
        prev_oh[prev_state_idx] = 1.0

    feat = [
        d_min, d_max, d_mean, d_std, bl_min, bl_mean, drop_abs,
        yolo_max_conf, yolo_n, yolo_min_x1, yolo_max_x2,
        motion_px, mog2_confirmed, bright_val, mog2_w,
        depth_drop_flag, yolo_crossed, yolo_nearest_cx,
        p_dmin, 0.0, drop_rel, depth_diff_rate,
        prev_oh[0], prev_oh[1], prev_oh[2],
    ]
    if len(feat) < FEATURE_DIM:
        feat += [0.0] * (FEATURE_DIM - len(feat))
    elif len(feat) > FEATURE_DIM:
        feat = feat[:FEATURE_DIM]
    return np.asarray(feat, dtype=np.float32)


# 推理管理器：负责加载模型、维护时序窗口。
class JudgeMLPInference:
    """
    MLP 推理管理器。
    加载 judge_mlp.pt，维护最近 W 帧的特征滑动窗口；
    每帧调 predict() 返回 car/no_car/unknown，或者 None（表示退回 v8）。
    """

    def __init__(self, model_path, window=5, device="cpu"):
        self.model_path = model_path
        self.window = window
        self.device = device
        self.model = None
        self.feature_buf = []  # 最近 window 帧的特征

        self._load()

    def _load(self):
        """加载模型。失败就让 self.model = None，调用方会走 v8。"""
        try:
            if not os.path.exists(self.model_path):
                print(f"[JudgeMLP] 模型文件不存在: {self.model_path}，回退 v8")
                return
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
            in_dim = meta.get("in_dim", self.window * FEATURE_DIM)
            self.model = JudgeMLP(in_dim=in_dim).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[JudgeMLP] 模型已加载: {self.model_path} (in_dim={in_dim}, window={self.window})")
        except Exception as e:
            print(f"[JudgeMLP] 模型加载失败: {e}，回退 v8")
            self.model = None

    def is_ready(self):
        return self.model is not None

    def predict(self, pixel_depths, detector, motion_detector, brightness_monitor,
                prev_d_min, prev_state_idx):
        """
        单帧推理。
        返回 (state 字符串, confidence 浮点数)，模型没就绪就返回 None（退回 v8）。
        confidence 是 softmax 之后预测类别的概率（0~1）。
        """
        if not self.is_ready():
            return None
        try:
            feat = build_inference_feature(
                pixel_depths, detector, motion_detector, brightness_monitor,
                prev_d_min, prev_state_idx,
            )
            self.feature_buf.append(feat)
            if len(self.feature_buf) > self.window:
                self.feature_buf = self.feature_buf[-self.window:]

            # 凑满 window 帧才推理，不足则用当前帧重复填充
            if len(self.feature_buf) < self.window:
                pad = [self.feature_buf[-1]] * (self.window - len(self.feature_buf))
                seq = self.feature_buf + pad
            else:
                seq = self.feature_buf

            x = np.concatenate(seq, axis=0).astype(np.float32)  # (window * FEATURE_DIM,)
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, in_dim)
            with torch.no_grad():
                logits = self.model(x_t)
                probs = torch.softmax(logits, dim=1)
                idx = int(probs.argmax(dim=1).item())
                conf = float(probs[0, idx].item())
            return IDX_TO_STATE.get(idx, "unknown"), conf
        except Exception as e:
            print(f"[JudgeMLP] 推理异常: {e}，回退 v8")
            return None

    def reset(self):
        """切换视频时调用，清空特征缓冲。"""
        self.feature_buf = []


# 全局单例和便捷接口，给 detect_car.py 用。
_judge_instance = None


def get_judge_inference(car_detect_config):
    """
    根据 config 里的 judge_mlp 开关返回 JudgeMLPInference 单例。
    judge_mlp.enabled=false 或配置缺了就返回 None（调用方走 v8），加载失败也返回 None。
    全局缓存，省得每个视频都重新加载一遍模型。
    """
    global _judge_instance
    jcfg = (car_detect_config or {}).get("judge_mlp", {}) if car_detect_config else {}
    enabled = bool(jcfg.get("enabled", False))
    if not enabled:
        return None
    model_path = jcfg.get("model_path", "checkpoints/judge_model/judge_mlp.pt")
    window = int(jcfg.get("window", 5))
    device = jcfg.get("device", "") or "cpu"
    # 转绝对路径
    if not os.path.isabs(model_path):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(root, model_path)
    # 若已有实例且路径/窗口一致，复用
    if _judge_instance is not None and _judge_instance.model_path == model_path and _judge_instance.window == window:
        return _judge_instance
    _judge_instance = JudgeMLPInference(model_path, window=window, device=device)
    return _judge_instance if _judge_instance.is_ready() else None


def judge_infer_frame(judge, pixel_depths, detector, motion_detector, brightness_monitor,
                      prev_d_min, prev_state_idx):
    """
    便捷函数，调一下 MLP 推理。
    judge=None 就返回 None（走 v8），返回 (state, confidence) 或 None。
    """
    if judge is None:
        return None
    return judge.predict(pixel_depths, detector, motion_detector, brightness_monitor,
                         prev_d_min, prev_state_idx)
