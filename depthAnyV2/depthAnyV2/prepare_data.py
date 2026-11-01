"""
Step 1: 数据准备脚本（Depth Anything V2 度量深度，输出真实米数）。

就干这几件事：先扫一遍原始图片统计信息；用预训练 DA2-Metric-Outdoor-Large 教师模型
生成"米数伪标签"（真实米数，0~80m）；再把深度图逐张存到磁盘，免得几万张图片一次性爆内存；
最后做增量处理，已经生成标签的图片直接跳过，有新增数据时只处理新图。

用法：python prepare_data.py

原始图片放这儿：datasets/my_dataset/images/  （所有 .jpg/.png 都丢进去）

输出长这样：
  datasets/my_dataset/processed/
    ├── image_files.npy          (图片路径列表，只含有深度文件的图片)
    └── depths/
        ├── 图片名_depth.npy     (逐张米数深度图, float32, 单位米)
        └── ...

几个要注意的：DA2-Metric-Outdoor 是用 Virtual KITTI 2（室外车载）训的，输出真实米数，max_depth=80m，
适合固定摄像头车辆监控这种场景；逐张存是为了避免几万张图片爆内存；
增量处理则保证新增图片时自动跳过已处理的，只跑新图。
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 默认教师模型：室外度量深度 Large（精度最高）
DEFAULT_TEACHER = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def _migrate_old_depths(depths_dir, image_files):
    """将旧格式深度文件 (00000.npy) 迁移为新格式 (图片名_depth.npy)"""
    migrated = 0
    for idx, fpath in enumerate(image_files):
        old_path = os.path.join(depths_dir, f"{idx:05d}.npy")
        stem = Path(fpath).stem
        new_path = os.path.join(depths_dir, f"{stem}_depth.npy")
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            migrated += 1
    if migrated > 0:
        print(f"[Data] 迁移旧格式深度文件: {migrated} 张 → 新格式 (文件名命名)")
    return migrated


def _inference_single(model, processor, img_rgb, device):
    """单次推理，返回与原图同尺寸的深度 (numpy float32, 单位米)"""
    inputs = processor(images=[img_rgb], return_tensors="pt").to(device)
    h, w = img_rgb.shape[:2]
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    if predicted_depth.dim() == 3:
        predicted_depth = predicted_depth.unsqueeze(1)
    depth = torch.nn.functional.interpolate(
        predicted_depth[0:1],
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)
    return depth


def _inference_tta(model, processor, img_rgb, device):
    """TTA 多尺度 + 水平翻转增强推理，取平均来降低伪标签噪声。

    一共增强 6 次：原图 + 原图翻转 + 3 个尺度（0.75/1.0/1.25）x（原图/翻转）。
    尺度靠 resize 实现，推理完再插值回原尺寸求平均。
    """
    h, w = img_rgb.shape[:2]
    scales = [0.75, 1.0, 1.25]
    depths_sum = np.zeros((h, w), dtype=np.float32)
    n_valid = 0

    for scale in scales:
        # 缩放
        if abs(scale - 1.0) < 1e-6:
            scaled = img_rgb
        else:
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 原图
        try:
            d = _inference_single(model, processor, scaled, device)
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            depths_sum += d
            n_valid += 1
        except Exception as e:
            print(f"\n  [TTA WARN] scale={scale} 推理失败: {e}")

        # 水平翻转
        try:
            flipped = cv2.flip(scaled, 1)
            d_flip = _inference_single(model, processor, flipped, device)
            # 深度图也要翻回来
            d_flip = cv2.flip(d_flip, 1)
            if d_flip.shape != (h, w):
                d_flip = cv2.resize(d_flip, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            depths_sum += d_flip
            n_valid += 1
        except Exception as e:
            print(f"\n  [TTA WARN] scale={scale} 翻转推理失败: {e}")

    if n_valid == 0:
        return None
    return (depths_sum / n_valid).astype(np.float32)


def _night_preprocess(img_rgb, bright_threshold=80.0, clip_limit=2.0, tile_grid_size=8):
    """夜间去眩光预处理（路线 C·夜间专用教师）：低亮度图用 CLAHE 压缩 L 通道高光

    车灯眩光使亮度通道过曝、暗部结构丢失，教师据此推理会对夜间区域估不准。
    CLAHE 局部对比度均衡压缩高光、增强暗部，让教师估出更可靠的夜间伪标签。
    非夜间图片原样返回（不改变白天标签）。
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (90, 90), interpolation=cv2.INTER_AREA)
    if float(small.mean()) >= bright_threshold:
        return img_rgb
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    l = clahe.apply(l)
    out_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _is_night_file(fpath, bright_threshold=80.0):
    """判断图片是否为夜间（低亮度），供 --force-night 只重跑夜间旧图"""
    img = cv2.imread(fpath)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (90, 90), interpolation=cv2.INTER_AREA)
    return float(small.mean()) < bright_threshold


def generate_depth_labels(image_files, output_dir, teacher_model=DEFAULT_TEACHER, use_tta=True,
                          night_enhance=False, force=False, force_night=False):
    """用 DA2-Metric 教师模型逐张推理，已经处理过的图片直接跳过。

    use_tta: 开不开 TTA 多尺度增强（默认开，伪标签噪声能降 ~15-20%）
    force: 强制把所有伪标签重新生成一遍（已存在的深度文件也不管）
    force_night: 只强制重跑「夜间旧图」，白天旧图照样跳过（配 night_enhance 用，省时间）
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths_dir = os.path.join(output_dir, "depths")
    os.makedirs(depths_dir, exist_ok=True)

    # 迁移旧格式深度文件（00000.npy → 图片名_depth.npy）
    _migrate_old_depths(depths_dir, image_files)

    # 筛选需要处理的图片（跳过已有深度文件的）
    to_process = []
    skipped = 0
    for fpath in image_files:
        stem = Path(fpath).stem
        depth_path = os.path.join(depths_dir, f"{stem}_depth.npy")
        if not os.path.exists(depth_path):
            to_process.append(fpath)                                # 新图：永远处理
        elif force:
            to_process.append(fpath)                                # 全部强制重跑
        elif force_night and _is_night_file(fpath):
            to_process.append(fpath)                                # 只重跑夜间旧图
        else:
            skipped += 1                                            # 已有标签且无需重跑 → 跳过

    total = len(image_files)
    tta_str = "TTA多尺度增强" if use_tta else "单次推理"
    print(f"[Data] 总计 {total} 张 | 已处理 {skipped} 张(跳过) | 待处理 {len(to_process)} 张 | 模式: {tta_str}")

    if len(to_process) > 0:
        print(f"\n[DA2] 加载教师模型: {teacher_model}")
        print(f"[DA2] 设备: {device}")
        processor = AutoImageProcessor.from_pretrained(teacher_model)
        model = AutoModelForDepthEstimation.from_pretrained(teacher_model).to(device)
        model.eval()

        pbar = tqdm(to_process, desc="生成深度", ncols=120)
        for fpath in pbar:
            img = cv2.imread(fpath)
            if img is None:
                print(f"\n  [WARN] 无法读取，跳过: {fpath}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 路线 C：夜间去眩光预处理（仅低亮度图，白天原样）
            if night_enhance:
                img_rgb = _night_preprocess(img_rgb)

            if use_tta:
                depth = _inference_tta(model, processor, img_rgb, device)
                if depth is None:
                    # TTA 全失败，回退单次
                    depth = _inference_single(model, processor, img_rgb, device)
            else:
                depth = _inference_single(model, processor, img_rgb, device)

            stem = Path(fpath).stem
            save_path = os.path.join(depths_dir, f"{stem}_depth.npy")
            np.save(save_path, depth)
        pbar.close()
    else:
        print("[DA2] 所有图片已处理完毕，无需重新生成")

    # 统计所有已处理深度的范围
    depth_min_all, depth_max_all = float("inf"), float("-inf")
    for fpath in image_files:
        stem = Path(fpath).stem
        depth_path = os.path.join(depths_dir, f"{stem}_depth.npy")
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
            valid = depth[depth > 0]
            if valid.size > 0:
                depth_min_all = min(depth_min_all, float(valid.min()))
                depth_max_all = max(depth_max_all, float(valid.max()))

    print(f"[DA2] 深度范围: {depth_min_all:.2f}m ~ {depth_max_all:.2f}m")
    return total, depth_min_all, depth_max_all


def _load_config():
    """加载项目根目录的 config.yaml"""
    import yaml
    root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(root, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
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


def main():
    cfg = _load_config()
    root = cfg.get("_root", os.getcwd())
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})

    parser = argparse.ArgumentParser(
        description="Step 1: 数据准备 - DA2-Metric 教师模型生成米数伪标签"
    )
    parser.add_argument("--image-dir", type=str,
                        default=_abs_path(root, paths.get("image_dir", "datasets/my_dataset/images")),
                        help="原始图片目录")
    parser.add_argument("--output-dir", type=str,
                        default=_abs_path(root, paths.get("processed_dir", "datasets/my_dataset/processed")),
                        help="输出目录")
    parser.add_argument("--teacher-model", type=str,
                        default=model_cfg.get("teacher", DEFAULT_TEACHER),
                        help="教师模型")
    parser.add_argument("--no-tta", action="store_true",
                        help="禁用 TTA 多尺度增强（默认开启，关闭后为单次推理）")
    parser.add_argument("--night-enhance", action="store_true",
                        default=bool(model_cfg.get("night_enhance", False)),
                        help="夜间去眩光教师（路线C）：低亮度图先 CLAHE 压高光再推理生成伪标签")
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成所有伪标签（忽略已存在的深度文件）")
    parser.add_argument("--force-night", action="store_true",
                        help="只强制重跑夜间旧图（白天旧图仍跳过），需配合 night_enhance 使用")
    args = parser.parse_args()
    use_tta = not args.no_tta

    os.makedirs(args.output_dir, exist_ok=True)

    # 扫描图片（递归搜索子目录，兼容嵌套文件夹）
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True))
    image_files = [f for f in image_files
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
                   and os.path.isfile(f)]

    if len(image_files) == 0:
        print("[ERROR] 未找到图片，请将 JPG/PNG 图片放入 image-dir 目录")
        sys.exit(1)

    print(f"[Data] 找到 {len(image_files)} 张图片")

    # 生成米数深度伪标签（增量处理，跳过已处理的）
    print(f"\n{'=' * 60}")
    print("Step 1: DA2-Metric 教师模型生成米数伪标签")
    print(f"{'=' * 60}")
    count, d_min, d_max = generate_depth_labels(
        image_files, args.output_dir, args.teacher_model, use_tta=use_tta,
        night_enhance=args.night_enhance, force=args.force,
        force_night=args.force_night
    )

    # 保存有效的图片路径列表（只包含有深度文件的）
    depths_dir = os.path.join(args.output_dir, "depths")
    valid_files = []
    for fpath in image_files:
        stem = Path(fpath).stem
        depth_path = os.path.join(depths_dir, f"{stem}_depth.npy")
        if os.path.exists(depth_path):
            valid_files.append(fpath)

    np.save(os.path.join(args.output_dir, "image_files.npy"),
            np.array(valid_files, dtype=object))

    print(f"\n{'=' * 60}")
    print(f"数据准备完成!")
    print(f"  图片数量: {len(valid_files)}")
    print(f"  深度范围: {d_min:.2f}m ~ {d_max:.2f}m (真实米数)")
    print(f"  输出目录: {args.output_dir}")
    print(f"  文件结构:")
    print(f"    - image_files.npy          (图片路径列表)")
    print(f"    - depths/图片名_depth.npy  (逐张米数深度图, float32)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
