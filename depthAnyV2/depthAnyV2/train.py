"""
Step 2: 模型微调训练脚本（Depth Anything V2 - 度量深度，输出真实米数）
功能：
  1. 加载预训练 DA2-Metric-Outdoor-Base 学生模型
  2. 用教师模型生成的米数伪标签做自蒸馏微调
  3. 分阶段微调（先 Head 后全量），保存 checkpoint
  4. 流式数据加载 + DataLoader + 验证集早停

用法：
  python train.py --data-dir datasets/my_dataset/processed --output-dir checkpoints/my_model --stage 1 --epochs 100

参数说明：
  --stage 1: 仅训练 DPT head（快速适应数据分布）
  --stage 2: 全量微调（backbone + head 联合训练）

输出：
  - 输出深度为真实米数（0~80m），不是相对深度
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 默认学生模型：室外度量深度 Base（平衡精度与速度）
DEFAULT_STUDENT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"


class DepthTrainDataset(Dataset):
    """训练数据集：流式加载（按需读盘），支持 DataLoader 多进程并行

    设计要点：
    - 不在 __init__ 预加载全部数据到内存（23539 张会吃 ~107GB RAM，系统爆掉）
    - __getitem__ 每次调用时读单张图片 + 对应 .npy 深度文件
    - 完全可 pickle（无 lambda、无可读文件句柄），Windows spawn 多进程安全
    - image_files.npy 存为 Python str list，避免 numpy str pickling 问题
    """

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, data_dir, process_res=518, augment=True):
        self.data_dir = data_dir
        self.process_res = process_res
        self.augment = augment
        self.depths_dir = os.path.join(data_dir, "depths")

        # Python str list，pickle 安全
        self.image_files = [str(p) for p in np.load(
            os.path.join(data_dir, "image_files.npy"), allow_pickle=True
        )]

        n = len(self.image_files)
        # 预计算深度文件路径（省得每次 Path.stem）
        self._depth_paths = [
            os.path.join(self.depths_dir, f"{Path(p).stem}_depth.npy")
            for p in self.image_files
        ]

        print(f"[Dataset] 流式加载模式 | 共 {n} 张 | "
              f"process_res={process_res} | augment={augment}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读图片
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.process_res, self.process_res))

        # HWC uint8 -> CHW float32 -> ImageNet 归一化
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self._IMAGENET_MEAN) / self._IMAGENET_STD

        # 读深度
        depth = np.load(self._depth_paths[idx]).astype(np.float32)
        depth = cv2.resize(depth, (self.process_res, self.process_res),
                           interpolation=cv2.INTER_NEAREST)
        valid_mask = ~np.isnan(depth) & (depth > 0.001) & (depth < 100.0)
        depth[~valid_mask] = 0.0

        return (
            torch.from_numpy(img),
            torch.from_numpy(depth),
            torch.from_numpy(valid_mask),
        )


class DA2MetricLoss(nn.Module):
    """度量深度损失：SILog（官方实现，无 scale alignment）
    参照 https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/loss.py
    度量深度本身有绝对尺度，不需要 median scale 对齐"""

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred_depth, gt_depth, valid_mask):
        """ Args:
            pred_depth: (B, H, W) 模型预测的米数深度
            gt_depth: (B, H, W) 伪标签米数深度
            valid_mask: (B, H, W) bool，True=有效像素
        """
        valid_mask = valid_mask.detach()
        # clamp pred 防止 log(0/负数)，与官方一致用 min_depth=0.001
        pred_depth = torch.clamp(pred_depth, min=1e-3)
        gt_depth = torch.clamp(gt_depth, min=1e-3)

        n_valid = valid_mask.sum()
        if n_valid < 10:
            return None, {"silog": 0.0, "total": 0.0, "skipped": True}

        diff_log = torch.log(gt_depth[valid_mask]) - torch.log(pred_depth[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean()
                          - self.lambd * torch.pow(diff_log.mean(), 2) + 1e-8)
        return loss, {
            "silog": loss.item() if torch.isfinite(loss) else 0.0,
            "total": loss.item() if torch.isfinite(loss) else 0.0,
            "skipped": False,
        }


def build_optimizer(model, stage=1, base_lr=5e-6):
    """分层优化器（参照官方：pretrained lr=base_lr, 非pretrained lr=base_lr*10）"""
    inner = getattr(model, "model", model)
    backbone = getattr(inner, "backbone", None)
    head = getattr(inner, "head", None)

    if backbone is None or head is None:
        print("[Optimizer] 未找到 backbone/head 分层结构，整体优化")
        all_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(all_params, lr=base_lr * 10, weight_decay=0.01)
        return optimizer

    backbone_params = list(backbone.parameters())
    head_params = list(head.parameters())

    if stage == 1:
        for p in backbone_params:
            p.requires_grad = False
        lr_bb, lr_head = 0.0, base_lr * 10
        print(f"[Optimizer] Stage 1: backbone frozen, head only (head LR={lr_head:.1e})")
    else:
        for p in backbone_params:
            p.requires_grad = True
        lr_bb, lr_head = base_lr, base_lr * 10
        print(f"[Optimizer] Stage 2: full fine-tune (backbone LR={lr_bb:.1e}, head LR={lr_head:.1e})")

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_bb},
        {"params": head_params, "lr": lr_head},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    return optimizer


def _gpu_augment(images, gt_depth, valid_mask, step, flip_decisions, brightness_factors):
    """GPU 数据增强（翻转 + 亮度抖动）"""
    if flip_decisions[step] < 0.5:
        images = images.flip(-1)
        gt_depth = gt_depth.flip(-1)
        valid_mask = valid_mask.flip(-1)
    images = images * brightness_factors[step]
    return images, gt_depth, valid_mask


def train_one_epoch(model, dataloader, optimizer, loss_fn, device,
                    epoch, base_lr, total_iters, current_iter, scaler=None):
    """单 epoch 训练：DataLoader 自动预取 + GPU 增强 + poly LR decay

    简化：去掉了手动 CUDA stream prefetch（DataLoader 的 pin_memory=True
    已经在后台做异步 CPU→GPU 传输，再手动 stream 重叠反而冲突）。
    """
    model.train()
    total_losses = []
    skipped_batches = 0
    torch.backends.cudnn.benchmark = True
    amp_enabled = scaler is not None

    n_batches = len(dataloader)
    # 预生成所有增强决策（GPU 上，避免每 batch 生成）
    flip_decisions = torch.rand(n_batches, device=device)
    brightness_factors = 0.8 + 0.4 * torch.rand(n_batches, device=device)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:2d}", ncols=130, leave=True)

    for step, (images, gt_depth, valid_mask) in enumerate(pbar):
        # DataLoader 的 pin_memory=True + non_blocking=True 自动做异步传输
        images = images.to(device, non_blocking=True)
        gt_depth = gt_depth.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)

        # GPU 数据增强
        images, gt_depth, valid_mask = _gpu_augment(
            images, gt_depth, valid_mask, step, flip_decisions, brightness_factors
        )

        # 混合精度前向
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            outputs = model(pixel_values=images)
            pred_depth = outputs.predicted_depth
            if pred_depth.dim() == 4:
                pred_depth = pred_depth.squeeze(1)
            if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                pred_depth = F.interpolate(
                    pred_depth.unsqueeze(1),
                    size=gt_depth.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)

        # Loss 在 fp32 计算（AMP 不稳定）
        with torch.amp.autocast("cuda", enabled=False):
            loss, loss_dict = loss_fn(pred_depth.float(), gt_depth.float(), valid_mask)

        if loss is None or not torch.isfinite(loss):
            skipped_batches += 1
            pbar.set_postfix(loss="SKIP", skip=skipped_batches)
        else:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            total_losses.append(loss_dict["total"])

        # poly LR decay
        current_iter += 1
        lr = base_lr * (1 - current_iter / total_iters) ** 0.9
        if len(optimizer.param_groups) >= 2:
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10
        else:
            optimizer.param_groups[0]["lr"] = lr * 10

        pbar.set_postfix(loss=f"{loss_dict['total']:.4f}", skip=skipped_batches)

    pbar.close()
    valid_losses = [l for l in total_losses if not (np.isnan(l) or np.isinf(l))]
    if skipped_batches > 0:
        print(f"  [Info] 跳过 {skipped_batches} 个 batch (有效像素不足)")
    return np.mean(valid_losses) if valid_losses else float('inf'), current_iter


def validate(model, dataloader, loss_fn, device):
    """验证集评估：无增强、无梯度，返回平均 SILog loss"""
    model.eval()
    losses = []
    skipped = 0
    n_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="Val   ", ncols=130, leave=False)

    with torch.no_grad():
        for images, gt_depth, valid_mask in pbar:
            images = images.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=images)
                pred_depth = outputs.predicted_depth
                if pred_depth.dim() == 4:
                    pred_depth = pred_depth.squeeze(1)
                if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                    pred_depth = F.interpolate(
                        pred_depth.unsqueeze(1),
                        size=gt_depth.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(1)

            with torch.amp.autocast("cuda", enabled=False):
                loss, loss_dict = loss_fn(pred_depth.float(), gt_depth.float(), valid_mask)

            if loss is None or not torch.isfinite(loss):
                skipped += 1
            else:
                losses.append(loss_dict["total"])
            pbar.set_postfix(loss=f"{loss_dict['total']:.4f}", skip=skipped)

    pbar.close()
    valid_losses = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
    if skipped > 0:
        print(f"  [Val Info] 跳过 {skipped} 个 batch (有效像素不足)")
    return np.mean(valid_losses) if valid_losses else float('inf')


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
    train_cfg = cfg.get("train", {})

    parser = argparse.ArgumentParser(description="Step 2: DA2-Metric 模型微调训练")
    parser.add_argument("--data-dir", type=str,
                        default=_abs_path(root, paths.get("processed_dir", "datasets/my_dataset/processed")),
                        help="prepare_data.py 的输出目录")
    parser.add_argument("--output-dir", type=str,
                        default=_abs_path(root, paths.get("checkpoint_dir", "checkpoints/my_model")),
                        help="checkpoint 保存目录")
    parser.add_argument("--model-name", type=str,
                        default=model_cfg.get("student", DEFAULT_STUDENT),
                        help="预训练学生模型")
    parser.add_argument("--stage", type=int,
                        default=train_cfg.get("stage", 1), choices=[1, 2],
                        help="训练阶段: 1=Head微调, 2=全量微调")
    parser.add_argument("--epochs", type=int,
                        default=train_cfg.get("epochs", 5))
    parser.add_argument("--batch-size", type=int,
                        default=train_cfg.get("batch_size", 2),
                        help="批大小 (8GB显存建议1-2)")
    parser.add_argument("--process-res", type=int,
                        default=train_cfg.get("process_res", 518),
                        help="处理分辨率 (须为14的倍数)")
    parser.add_argument("--grad-accumulate", type=int,
                        default=train_cfg.get("grad_accumulate", 4))
    parser.add_argument("--resume", type=str,
                        default=_abs_path(root, train_cfg.get("resume")))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # === 加载模型 ===
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print(f"[INFO] 加载预训练学生模型: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name).to(device)

    if args.resume:
        print(f"[INFO] 从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        resume_info = {
            "epoch": ckpt.get("epoch", 0),
            "best_loss": ckpt.get("best_loss", float("inf")),
            "current_iter": ckpt.get("current_iter", 0),
        }
        print(f"[INFO] 恢复点: epoch={resume_info['epoch']}, "
              f"best_loss={resume_info['best_loss']:.4f}, "
              f"iter={resume_info['current_iter']}")
    else:
        resume_info = None

    # === 流式加载数据集（不预加载到内存！）===
    full_dataset = DepthTrainDataset(args.data_dir, args.process_res, augment=True)

    # === 划分训练/验证集（9:1，固定随机种子保证可复现）===
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    # Subset 包装原始 dataset，自动做索引映射
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    print(f"[Split] 训练集: {n_train} | 验证集: {n_val}")

    # === DataLoader 配置 ===
    # Windows 下 num_workers>0 用 spawn，DepthTrainDataset 必须可 pickle
    # persistent_workers 让 worker 跨 epoch 存活，避免重复启动开销
    num_workers = 4
    pin_memory = sys.platform != 'win32'  # Windows WDDM 下 pin_memory 有风险
    prefetch_factor = 2
    persistent_workers = num_workers > 0

    print(f"[DataLoader] num_workers={num_workers}, pin_memory={pin_memory}, "
          f"persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}")

    def _make_loader(subset, batch_size, shuffle, num_workers,
                     pin_memory, persistent_workers, prefetch_factor):
        """创建 DataLoader，Windows 下 num_workers=0 时跳过 persistent/prefetch"""
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(subset, **kwargs)

    train_loader = _make_loader(
        train_subset, args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )
    val_loader = _make_loader(
        val_subset, args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )

    n_batches = len(train_loader)  # DataLoader 自动 drop_last=False 覆盖所有样本

    # === 优化器 + 损失 ===
    base_lr = 5e-6
    optimizer = build_optimizer(model, stage=args.stage, base_lr=base_lr)
    loss_fn = DA2MetricLoss(lambd=0.5)

    # poly decay 的 total_iters
    total_iters = args.epochs * n_batches

    # === 恢复训练状态 ===
    # 关键：stage 切换时只加载模型权重，不加载 optimizer state（参数组结构已变）
    # 仅当 checkpoint 的 stage 与当前 stage 一致时才完整恢复 optimizer/计数器
    if resume_info is not None:
        ckpt_stage = ckpt.get("stage", 1)
        if ckpt_stage == args.stage:
            # 同阶段续训：完整恢复 optimizer + 计数器
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = resume_info["epoch"] + 1
            current_iter = resume_info["current_iter"]
            best_loss = resume_info["best_loss"]
            if start_epoch > args.epochs:
                print(f"[INFO] 已训练完成 (epoch {resume_info['epoch']}/{args.epochs})，无需继续")
                return
            print(f"[INFO] 同阶段续训: 从 epoch {start_epoch} 继续, current_iter={current_iter}")
        else:
            # stage 切换：只继承模型权重，重置 optimizer/计数器（新阶段从头训）
            start_epoch = 1
            current_iter = 0
            best_loss = float("inf")
            print(f"[INFO] Stage 切换 ({ckpt_stage}→{args.stage}): "
                  f"只继承模型权重，optimizer/计数器已重置，从头开始 Stage {args.stage}")
    else:
        start_epoch = 1
        current_iter = 0
        best_loss = float("inf")

    # === 混合精度 scaler ===
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    # stage 切换时不加载旧 scaler state（避免状态不一致）
    if resume_info is not None and ckpt.get("stage", 1) == args.stage and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # === 早停配置（基于验证集 val_loss）===
    patience = 15
    min_delta = 1e-4
    no_improve_count = 0
    if resume_info is not None:
        no_improve_count = ckpt.get("no_improve_count", 0)

    # === 训练循环 ===
    print(f"\n{'=' * 60}")
    print(f"开始训练: Stage {args.stage} | Epochs: {args.epochs} | "
          f"Batch: {args.batch_size} | Base LR: {base_lr:.1e}")
    print(f"输出: 真实米数 (度量深度) | AMP: bfloat16 + GradScaler")
    print(f"验证集: {n_val} 样本 | 早停: patience={patience}, min_delta={min_delta}")
    print(f"数据加载: 流式 + DataLoader(num_workers={num_workers})")
    if resume_info is not None:
        print(f"[Resume] 从 epoch {start_epoch} 恢复 (已训练 {resume_info['epoch']} epochs, "
              f"no_improve={no_improve_count}/{patience})")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        avg_loss, current_iter = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            base_lr=base_lr, total_iters=total_iters, current_iter=current_iter,
            scaler=scaler,
        )

        # 验证集评估（无增强、无梯度）
        val_loss = validate(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} | Train: {avg_loss:.4f} | "
              f"Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s")

        # 早停检查（基于 val_loss）
        is_best = val_loss < best_loss - min_delta
        if is_best:
            best_loss = val_loss
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "best_loss": best_loss,
                "current_iter": current_iter,
                "stage": args.stage,
                "model_name": args.model_name,
                "no_improve_count": no_improve_count,
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  >> 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  -- 验证集无改善 ({no_improve_count}/{patience})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "best_loss": best_loss,
            "current_iter": current_iter,
            "stage": args.stage,
            "model_name": args.model_name,
            "no_improve_count": no_improve_count,
        }, os.path.join(args.output_dir, "latest_model.pt"))

        if no_improve_count >= patience:
            print(f"\n[Early Stop] 验证集 {patience} 轮无改善，提前停止训练")
            break

    print(f"\n{'=' * 60}")
    print(f"训练完成! 最佳 Val Loss: {best_loss:.4f}")
    print(f"模型保存在: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
