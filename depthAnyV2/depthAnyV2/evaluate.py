"""
Step 3: 模型评估脚本（Depth Anything V2 度量深度）。

流程就三步：加载微调后的 DA2-Metric 模型；在测试图片上推理，输出真实米数；
再跟预训练模型对比，输出米数统计。

用法：
  python evaluate.py --checkpoint checkpoints/my_model/best_model.pt --image-dir datasets/my_dataset/images --output-dir eval_results --compare
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 默认基础模型：室外度量深度 Base
DEFAULT_BASE = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"


def load_model(checkpoint_path, base_model=DEFAULT_BASE):
    """加载微调后的 DA2-Metric 模型"""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] 加载基础模型: {base_model}")
    processor = AutoImageProcessor.from_pretrained(base_model)
    model = AutoModelForDepthEstimation.from_pretrained(base_model).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Model] 加载微调权重: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # 兼容新旧 checkpoint 字段名（train.py 已改用 train_loss/val_loss）
        loss_val = ckpt.get('val_loss', ckpt.get('loss', None))
        loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else "N/A"
        print(f"[Model] 训练阶段: {ckpt.get('stage', 'N/A')}, Val Loss: {loss_str}")

    model.eval()
    return model, processor, device


def run_inference(model, processor, image_files, device, desc="推理"):
    """逐张推理，返回米数深度图列表"""
    all_depths = []
    pbar = tqdm(image_files, desc=desc, ncols=120)
    for fpath in pbar:
        img = cv2.imread(fpath)
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
        ).squeeze().cpu().numpy()
        all_depths.append(depth)
    pbar.close()
    return all_depths


def visualize_depth(depth, save_path=None):
    """深度图可视化（米数）"""
    valid = depth > 0
    if valid.sum() == 0:
        vis = np.zeros_like(depth, dtype=np.uint8)
    else:
        d_min, d_max = depth[valid].min(), depth[valid].max()
        vis = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis[~valid] = 0
    if save_path:
        cv2.imwrite(save_path, vis)
    return vis


def print_depth_stats(depths, label=""):
    """打印米数深度统计"""
    all_valid = []
    for d in depths:
        v = d[(d > 0.1) & (d < 100)]
        if v.size > 0:
            all_valid.append(v)
    if not all_valid:
        print(f"  [{label}] 无有效深度")
        return
    all_valid = np.concatenate(all_valid)
    print(f"  [{label}] 深度统计 (米):")
    print(f"    最小: {all_valid.min():.2f}m | 最大: {all_valid.max():.2f}m | "
          f"均值: {all_valid.mean():.2f}m | 中位数: {np.median(all_valid):.2f}m")


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
    eval_cfg = cfg.get("eval", {})

    parser = argparse.ArgumentParser(description="Step 3: DA2-Metric 模型评估与可视化")
    parser.add_argument("--checkpoint", type=str,
                        default=_abs_path(root, eval_cfg.get("checkpoint")),
                        help="微调模型 checkpoint 路径")
    parser.add_argument("--image-dir", type=str,
                        default=_abs_path(root, paths.get("image_dir", "datasets/my_dataset/images")),
                        help="测试图片目录")
    parser.add_argument("--output-dir", type=str,
                        default=_abs_path(root, paths.get("eval_dir", "eval_results")))
    parser.add_argument("--base-model", type=str,
                        default=model_cfg.get("base", DEFAULT_BASE),
                        help="基础模型")
    parser.add_argument("--compare", action="store_true",
                        default=eval_cfg.get("compare", False),
                        help="同时运行预训练模型进行对比")
    parser.add_argument("--limit", type=int, default=100,
                        help="抽样评估的图片数量 (0=全部, 默认100避免太慢)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 优先从 processed/image_files.npy 读图片路径（与 train.py 一致，支持深层子目录）
    data_dir = os.path.join(os.path.dirname(args.image_dir), "processed")
    label_list_path = os.path.join(data_dir, "image_files.npy")
    if os.path.exists(label_list_path):
        image_files = [str(p) for p in np.load(label_list_path, allow_pickle=True)]
        print(f"[Eval] 从 image_files.npy 读取: {len(image_files)} 张")
    else:
        # fallback: 递归扫描所有子目录（支持多层嵌套）
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = sorted(
            str(p) for p in Path(args.image_dir).rglob('*')
            if p.suffix.lower() in exts
        )
        print(f"[Eval] 从目录递归扫描: {len(image_files)} 张")

    # 抽样限制（全量 23539 张跑两遍模型太慢）
    if args.limit > 0 and len(image_files) > args.limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(image_files), size=args.limit, replace=False)
        idx.sort()
        image_files = [image_files[i] for i in idx]
        print(f"[Eval] 抽样限制: {args.limit} 张")

    print(f"[Eval] 最终评估图片: {len(image_files)} 张")

    if len(image_files) == 0:
        print("[ERROR] 未找到测试图片")
        sys.exit(1)

    # 评估微调模型
    ft_depths = None
    if args.checkpoint:
        print(f"\n{'=' * 60}")
        print("评估微调模型")
        print(f"{'=' * 60}")
        model, processor, device = load_model(args.checkpoint, args.base_model)
        ft_depths = run_inference(model, processor, image_files, device, desc="微调推理")

        ft_dir = os.path.join(args.output_dir, "finetuned")
        os.makedirs(ft_dir, exist_ok=True)
        for i, f in enumerate(image_files):
            name = Path(f).stem
            visualize_depth(ft_depths[i], os.path.join(ft_dir, f"{name}_depth.png"))
            np.save(os.path.join(ft_dir, f"{name}_depth.npy"), ft_depths[i])
        print(f"\n微调模型深度图已保存到: {ft_dir}")
        print_depth_stats(ft_depths, "微调模型")

    # 和预训练模型对比
    base_depths = None
    if args.compare or not args.checkpoint:
        print(f"\n{'=' * 60}")
        print("评估预训练模型 (基准)")
        print(f"{'=' * 60}")
        model, processor, device = load_model(None, args.base_model)
        base_depths = run_inference(model, processor, image_files, device, desc="基准推理")

        base_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(base_dir, exist_ok=True)
        for i, f in enumerate(image_files):
            name = Path(f).stem
            visualize_depth(base_depths[i], os.path.join(base_dir, f"{name}_depth.png"))
            np.save(os.path.join(base_dir, f"{name}_depth.npy"), base_depths[i])
        print(f"\n预训练模型深度图已保存到: {base_dir}")
        print_depth_stats(base_depths, "预训练模型")

    # 对比统计（米数）
    if ft_depths is not None and base_depths is not None:
        print(f"\n{'=' * 60}")
        print("深度预测对比统计 (单位: 米)")
        print(f"{'=' * 60}")
        print(f"{'图片':<20} {'预训练均值(m)':<16} {'微调均值(m)':<16} {'差异(m)':<10}")
        print("-" * 62)
        for i, f in enumerate(image_files):
            name = os.path.basename(f)[:18]
            b = base_depths[i]
            ftd = ft_depths[i]
            b_mean = b[(b > 0.1) & (b < 100)].mean() if ((b > 0.1) & (b < 100)).any() else 0
            f_mean = ftd[(ftd > 0.1) & (ftd < 100)].mean() if ((ftd > 0.1) & (ftd < 100)).any() else 0
            diff = abs(b_mean - f_mean)
            print(f"{name:<20} {b_mean:<16.3f} {f_mean:<16.3f} {diff:<10.3f}")

    # 与训练标签对比，逐张读 depths/ 目录里的 .npy
    # data_dir / label_list_path 已在上方定义
    label_dir = os.path.join(data_dir, "depths")
    if os.path.exists(label_dir) and os.path.exists(label_list_path) and ft_depths is not None:
        print(f"\n{'=' * 60}")
        print("与训练标签对比 (微调模型, 单位: 米)")
        print(f"{'=' * 60}")
        label_files = list(np.load(label_list_path, allow_pickle=True))
        label_names = set(os.path.basename(lf) for lf in label_files)

        metrics = {"abs_rel": [], "rmse": [], "delta_1.25": []}
        for i, img_path in enumerate(image_files):
            img_name = os.path.basename(img_path)
            if img_name not in label_names:
                continue

            gt_path = os.path.join(label_dir, f"{Path(img_path).stem}_depth.npy")
            if not os.path.exists(gt_path):
                continue
            gt = np.load(gt_path).astype(np.float32)
            pred = ft_depths[i]
            if gt.shape != pred.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            valid = (gt > 0.1) & (gt < 100) & (pred > 0.1)
            if valid.sum() < 100:
                continue

            scale = np.median(gt[valid]) / (np.median(pred[valid]) + 1e-6)
            pred_aligned = pred * scale

            abs_rel = np.mean(np.abs(pred_aligned[valid] - gt[valid]) / gt[valid])
            rmse = np.sqrt(np.mean((pred_aligned[valid] - gt[valid]) ** 2))
            ratio = np.maximum(pred_aligned[valid] / gt[valid], gt[valid] / pred_aligned[valid])
            delta = np.mean(ratio < 1.25)

            metrics["abs_rel"].append(abs_rel)
            metrics["rmse"].append(rmse)
            metrics["delta_1.25"].append(delta)

        if metrics["abs_rel"]:
            print(f"  AbsRel:  {np.mean(metrics['abs_rel']):.4f}  (越低越好)")
            print(f"  RMSE:    {np.mean(metrics['rmse']):.4f} m  (越低越好)")
            print(f"  δ<1.25:  {np.mean(metrics['delta_1.25']):.4f}  (越高越好)")

    print(f"\n{'=' * 60}")
    print(f"评估完成! 结果保存在: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
