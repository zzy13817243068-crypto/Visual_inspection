"""
Step 1: 数据准备脚本（Depth Anything V2 - 度量深度，输出真实米数）
功能：
  1. 扫描原始图片，统计基本信息
  2. 用预训练 DA2-Metric-Outdoor-Large 教师模型生成"米数伪标签"（真实米数，0~80m）
  3. 逐张保存深度图到磁盘（避免内存爆炸，支持几万张数据）
  4. 增量处理：跳过已生成深度标签的图片，新增数据时只处理新图片

用法：
  python prepare_data.py

原始图片放置位置：
  datasets/my_dataset/images/  （放入所有 .jpg/.png 图片）

输出结构：
  datasets/my_dataset/processed/
    ├── image_files.npy          (图片路径列表，只包含有深度文件的图片)
    └── depths/
        ├── 图片名_depth.npy     (逐张米数深度图, float32, 单位米)
        └── ...

注意：
  - DA2-Metric-Outdoor 用 Virtual KITTI 2（室外车载）训练，输出真实米数，max_depth=80m
  - 适合固定摄像头车辆监控场景
  - 逐张保存避免几万张图片爆内存
  - 增量处理：新增图片时自动跳过已处理的，只处理新图片
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


def generate_depth_labels(image_files, output_dir, teacher_model=DEFAULT_TEACHER):
    """用 DA2-Metric 教师模型逐张推理，跳过已处理的图片"""
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
        if os.path.exists(depth_path):
            skipped += 1
        else:
            to_process.append(fpath)

    total = len(image_files)
    print(f"[Data] 总计 {total} 张 | 已处理 {skipped} 张(跳过) | 待处理 {len(to_process)} 张")

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            inputs = processor(images=[img], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            if predicted_depth.dim() == 3:
                predicted_depth = predicted_depth.unsqueeze(1)

            h, w = img.shape[:2]
            depth = torch.nn.functional.interpolate(
                predicted_depth[0:1],
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy().astype(np.float32)

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
    args = parser.parse_args()

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
        image_files, args.output_dir, args.teacher_model
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
