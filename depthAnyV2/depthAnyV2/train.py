"""第二步：微调 Depth Anything V2，让模型输出真实米数的度量深度。

做法是拿教师模型生成的米数当伪标签，对学生做自蒸馏微调；分两阶段（先 Head 后全量），
checkpoint 存到 --output-dir。数据流式读盘 + DataLoader，带验证集早停。

跑法：
  python train.py --data-dir datasets/my_dataset/processed --output-dir checkpoints/my_model --stage 1 --epochs 100

--stage 1 只训 DPT head，上手快；--stage 2 才是 backbone + head 一起微调。
输出的是 0~80m 的真实米数，不是相对深度。
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


# LoRA 适配器：冻住 ViT 编码器，只在 DPT head 的 1x1 卷积上加低秩分解（W' = W + B@A）。
# 加在解码器上是因为分布差异主要出在解码器，而且参数量比全量微调小 10~100 倍，训得快、也不容易过拟合。
class LoRALayer(nn.Module):
    """A 用 Kaiming 初始化、B 初始化为 0，所以刚开训时 LoRA 分支输出是 0，不会破坏原模型。"""

    def __init__(self, original_layer, rank=8):
        super().__init__()
        self.original = original_layer
        for p in self.original.parameters():
            p.requires_grad = False

        in_dim = original_layer.in_channels if hasattr(original_layer, 'in_channels') else original_layer.in_features
        out_dim = original_layer.out_channels if hasattr(original_layer, 'out_channels') else original_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(in_dim, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_dim))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原卷积结果 + LoRA 扰动；1x1 卷积等价于线性变换，直接用矩阵乘更快
        out = self.original(x)
        # x: (B, C_in, H, W) → (B, H, W, C_in) @ A → (B, H, W, rank) @ B → (B, H, W, C_out) → (B, C_out, H, W)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C_in)
        lora_out = (x_flat @ self.lora_A) @ self.lora_B  # (B*H*W, C_out)
        lora_out = lora_out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out + lora_out


def apply_lora_to_dpt(model, rank=8):
    """将 LoRA patch 到 DPT head 的 project 卷积层（1x1 卷积）"""
    inner = getattr(model, "model", model)
    head = getattr(inner, "head", None)
    if head is None:
        print("[LoRA] 未找到 head，跳过")
        return 0

    patched = 0
    # DPT head 里的 projection 就是一堆 1x1 卷积
    for attr_name in dir(head):
        attr = getattr(head, attr_name, None)
        if isinstance(attr, nn.Conv2d) and attr.kernel_size == (1, 1):
            # 只 patch 1x1 卷积（适合低秩分解）
            setattr(head, attr_name, LoRALayer(attr, rank=rank))
            patched += 1

    # 也尝试 patch head 下的嵌套模块
    for name, module in head.named_children():
        for sub_name, sub_mod in module.named_children():
            if isinstance(sub_mod, nn.Conv2d) and sub_mod.kernel_size == (1, 1):
                setattr(module, sub_name, LoRALayer(sub_mod, rank=rank))
                patched += 1

    # 除了 LoRA 参数，其余全部冻住
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] patch 了 {patched} 个 1x1 卷积 | rank={rank} | "
          f"可训练参数: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    return patched


def load_state_dict_smart(model, state_dict):
    """加载 state_dict，顺便兼容 LoRA 包装前后键名对不上的问题。

    一旦用 LoRA 包了某一层，它的权重键就从 X.weight 变成 X.original.weight，
    直接 load 会静默丢权重。这里做双向重映射：旧 ckpt（没 LoRA）映射到 X.original.weight，
    LoRA ckpt 就直接匹配。
    """
    model_sd = model.state_dict()
    new_sd = {}
    n_matched, n_remapped, n_skipped = 0, 0, 0

    for k, v in state_dict.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                new_sd[k] = v
                n_matched += 1
            else:
                n_skipped += 1  # 形状不匹配（如分类头尺寸变化），跳过
            continue

        # 重映射: X.weight → X.original.weight（ckpt 无 LoRA、model 已包装）
        alt = k.replace(".weight", ".original.weight").replace(".bias", ".original.bias")
        if alt in model_sd and model_sd[alt].shape == v.shape:
            new_sd[alt] = v
            n_remapped += 1
            continue

        # 反向: X.original.weight → X.weight（ckpt 已包装、model 未包装）
        alt2 = k.replace(".original.", ".")
        if alt2 in model_sd and model_sd[alt2].shape == v.shape:
            new_sd[alt2] = v
            n_remapped += 1
            continue

        n_skipped += 1

    model.load_state_dict(new_sd, strict=False)
    print(f"[Load] 权重加载: 直接匹配 {n_matched} | 键名重映射 {n_remapped} | 跳过 {n_skipped}")
    return n_remapped


def merge_lora_for_save(model):
    """把 LoRA 合并回原权重（W' = W + B@A），返回标准键名的 state_dict（CPU 副本）。

    合并后键名和没包装过的 HF 模型一样，detect/evaluate 可以直接加载；
    同时把 lora_A/lora_B 键删掉，否则恢复训练时 delta 会被算两次
    （恢复时 W' 加载进 original.weight，LoRA 又从 B=0 开始，结果等价）。
    """
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_merged = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            W = module.original.weight.detach().cpu()      # (out, in, 1, 1)
            delta = module.lora_B.detach().cpu() @ module.lora_A.detach().cpu()  # (out, in)
            if W.dim() == 4:
                delta = delta.unsqueeze(-1).unsqueeze(-1)
            key = f"{name}.original.weight"
            if key in sd:
                sd[key] = (sd[key].float() + delta.float()).float()
                n_merged += 1
    # 删除 LoRA 参数键
    for k in list(sd.keys()):
        if "lora_A" in k or "lora_B" in k:
            del sd[k]
    if n_merged > 0:
        print(f"[Save] LoRA 已合并到原权重 ({n_merged} 层)，保存标准键名 checkpoint")
    return sd


# 时序梯度一致性 loss：约束相邻帧的深度差，压掉视频里那种一帧跳 6 米的抖动。
# 不用额外标签，只要相邻帧深度平滑过渡就行。
class TemporalGradientLoss(nn.Module):
    """约束相邻帧的深度差（L_temporal = mean|d_pred_t - d_pred_{t-1}|）要小。

    只有视频数据（有相邻帧对）才启用，单帧训练时这个 loss 不起作用。
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_t, pred_t_prev, valid_mask):
        """pred_t / pred_t_prev 是当前帧、上一帧的深度 (B,H,W)，valid_mask 是有效像素；
        返回 (loss, loss_dict)，没有上一帧时 loss 给 None。
        """
        if pred_t_prev is None:
            return None, {"temporal": 0.0}

        # 相邻帧深度相减
        temporal_grad = pred_t - pred_t_prev
        mask = valid_mask & (torch.abs(temporal_grad) < 50.0)  # 过滤异常跳变
        n = mask.sum()
        if n < 10:
            return None, {"temporal": 0.0}

        loss = torch.abs(temporal_grad[mask]).mean()
        if not torch.isfinite(loss):
            return None, {"temporal": 0.0}
        return loss, {"temporal": loss.item()}


class DepthTrainDataset(Dataset):
    """训练集：按需读盘的流式加载，能配合 DataLoader 多进程并行。

    故意不在 __init__ 里把数据全读进内存——23539 张会吃 ~107GB RAM，机器直接爆。
    改成 __getitem__ 现读单张图 + 对应 .npy 深度。
    Dataset 得能 pickle（不能塞 lambda、不能存打开的文件句柄），Windows 的 spawn 才安全；
    image_files.npy 存成 Python str list 就是为了绕开 numpy str 的 pickling 坑。
    temporal_pairs=True 时多返回上一帧，给时序梯度 loss 用。
    """

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, data_dir, process_res=518, temporal_pairs=False):
        self.data_dir = data_dir
        self.process_res = process_res
        self.temporal_pairs = temporal_pairs
        self.depths_dir = os.path.join(data_dir, "depths")

        # 转成 Python str list，pickle 才安全
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
              f"process_res={process_res} | temporal_pairs={temporal_pairs}")

    def __len__(self):
        return len(self.image_files)

    def _load_one(self, idx):
        """加载单张图片 + 深度"""
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.process_res, self.process_res))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self._IMAGENET_MEAN) / self._IMAGENET_STD

        depth = np.load(self._depth_paths[idx]).astype(np.float32)
        depth = cv2.resize(depth, (self.process_res, self.process_res),
                           interpolation=cv2.INTER_NEAREST)
        valid_mask = ~np.isnan(depth) & (depth > 0.001) & (depth < 100.0)
        depth[~valid_mask] = 0.0
        return img, depth, valid_mask

    def __getitem__(self, idx):
        img, depth, valid_mask = self._load_one(idx)

        if self.temporal_pairs:
            if idx > 0:
                # 返回上一帧（用于时序梯度 loss）
                img_prev, depth_prev, mask_prev = self._load_one(idx - 1)
                return (
                    torch.from_numpy(img), torch.from_numpy(depth), torch.from_numpy(valid_mask),
                    torch.from_numpy(img_prev), torch.from_numpy(depth_prev), torch.from_numpy(mask_prev),
                )
            # 首个样本无上一帧：全零占位（train 中 .sum()>0 检查会跳过）
            return (
                torch.from_numpy(img), torch.from_numpy(depth), torch.from_numpy(valid_mask),
                torch.zeros(3, self.process_res, self.process_res, dtype=torch.float32),
                torch.zeros(self.process_res, self.process_res, dtype=torch.float32),
                torch.zeros(self.process_res, self.process_res, dtype=torch.bool),
            )

        # 非时序模式：prev 返回 None，省掉每样本 ~3.4MB 全零张量
        return (
            torch.from_numpy(img), torch.from_numpy(depth), torch.from_numpy(valid_mask),
            None, None, None,
        )


def _collate_batch(batch):
    """自己拼 batch：非时序模式下 prev 三个字段是 None，default_collate 不认 None，
    所以直接透传；时序模式下就正常 stack。
    """
    img = torch.stack([b[0] for b in batch])
    depth = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    if batch[0][3] is None:
        return img, depth, mask, None, None, None
    return (
        img, depth, mask,
        torch.stack([b[3] for b in batch]),
        torch.stack([b[4] for b in batch]),
        torch.stack([b[5] for b in batch]),
    )


class DA2MetricLoss(nn.Module):
    """度量深度的 SILog loss，跟官方实现保持一致，不做 scale alignment。

    因为度量深度自带绝对尺度，不像相对深度那样需要 median scale 对齐。
    """

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred_depth, gt_depth, valid_mask):
        """pred_depth / gt_depth 都是 (B,H,W) 的米数，
        valid_mask 是 (B,H,W) 的 bool，True 表示这个像素有效。
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


def build_optimizer(model, stage=1, base_lr=5e-6, weight_decay=0.01):
    """分层给学习率：预训练部分用 base_lr，新加的 head 用 base_lr*10。

    LoRA 模式下 backbone 一直冻着（连 stage 2 也是），只训 LoRA 参数——
    不然 stage 2 一解冻 backbone，LoRA 的防过拟合就白做了。
    """
    inner = getattr(model, "model", model)
    backbone = getattr(inner, "backbone", None)
    head = getattr(inner, "head", None)

    lora_active = any("lora_" in n for n, _ in model.named_parameters())

    if backbone is None or head is None:
        print("[Optimizer] 未找到 backbone/head 分层结构，整体优化")
        all_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(all_params, lr=base_lr * 10, weight_decay=weight_decay)
        return optimizer

    backbone_params = list(backbone.parameters())
    head_params = list(head.parameters())

    if stage == 1:
        for p in backbone_params:
            p.requires_grad = False
        lr_bb, lr_head = 0.0, base_lr * 10
        print(f"[Optimizer] Stage 1: backbone frozen, head only (head LR={lr_head:.1e})")
    elif lora_active:
        # LoRA 模式：即使 stage 2 也保持 backbone 冻结
        lr_bb, lr_head = 0.0, base_lr * 10
        print(f"[Optimizer] Stage 2 + LoRA: backbone 保持冻结，只训练 LoRA 参数 (LR={lr_head:.1e})")
    else:
        for p in backbone_params:
            p.requires_grad = True
        lr_bb, lr_head = base_lr, base_lr * 10
        print(f"[Optimizer] Stage 2: full fine-tune (backbone LR={lr_bb:.1e}, head LR={lr_head:.1e})")

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_bb},
        {"params": [p for p in head_params if p.requires_grad], "lr": lr_head},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def _gpu_augment(images, gt_depth, valid_mask, step,
                 flip_decisions, brightness_factors, contrast_factors):
    """GPU 上做增强：随机翻转 + 亮度抖动 + 对比度抖动。

    都是冲着夜间光源移动导致深度剧变来的：亮度 0.5~1.5（±50%）模拟强弱光切换，
    对比度 0.6~1.4（±40%）模拟光源色温/远近变化。只改图片，深度标签不动（翻转时跟着翻）。
    """
    if flip_decisions[step] < 0.5:
        images = images.flip(-1)
        gt_depth = gt_depth.flip(-1)
        valid_mask = valid_mask.flip(-1)
    # 亮度：整体缩放
    images = images * brightness_factors[step]
    # 对比度：以每张图通道均值为中心缩放
    img_mean = images.mean(dim=[2, 3], keepdim=True)
    images = (images - img_mean) * contrast_factors[step] + img_mean
    return images, gt_depth, valid_mask


def _gpu_night_augment(images, do_night, dark_factor, bloom_s, bloom_kernel):
    """夜间模拟增强：低照度压暗 + 车灯眩光 bloom（光晕散射）。

    只有 do_night 为 True 才作用于整个 batch；深度标签不动，因为光照变化不改变真实深度。
    """
    if not do_night:
        return images
    images = images * dark_factor
    pad = bloom_kernel // 2
    blurred = F.avg_pool2d(images, kernel_size=bloom_kernel, stride=1, padding=pad)
    images = (1.0 - bloom_s) * images + bloom_s * blurred
    return images


def _forward_depth(model, images, target_size, amp_enabled=True):
    """统一深度推断块：amp 前向 + squeeze + bicubic 对齐目标分辨率"""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        outputs = model(pixel_values=images)
        pred = outputs.predicted_depth
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if pred.shape[-2:] != target_size:
            pred = F.interpolate(
                pred.unsqueeze(1), size=target_size,
                mode="bicubic", align_corners=False,
            ).squeeze(1)
    return pred


def train_one_epoch(model, dataloader, optimizer, loss_fn, device,
                    epoch, base_lr, total_iters, current_iter, scaler=None,
                    temporal_loss_fn=None, temporal_weight=0.0,
                    night_aug_prob=0.3, bloom_kernel=11, temporal_every=1):
    """跑一个 epoch：DataLoader 自己预取，增强在 GPU 上做，学习率按 poly 衰减。

    时序 loss 做了子采样——每 temporal_every 个 batch 才多前向一次上一帧，
    权重相应 ×temporal_every 补偿，期望梯度不变，大约省 40% 计算。
    """
    model.train()
    total_losses = []
    skipped_batches = 0
    torch.backends.cudnn.benchmark = True
    amp_enabled = scaler is not None
    use_temporal = temporal_loss_fn is not None and temporal_weight > 0
    temporal_every = max(1, int(temporal_every))
    temporal_scale = temporal_weight * temporal_every  # 子采样后按倍数补偿权重

    n_batches = len(dataloader)
    # 预生成所有增强决策（GPU 上，避免每 batch 生成）
    flip_decisions = torch.rand(n_batches, device=device)
    brightness_factors = 0.5 + 1.0 * torch.rand(n_batches, device=device)  # 0.5~1.5 (±50%)
    contrast_factors = 0.6 + 0.8 * torch.rand(n_batches, device=device)    # 0.6~1.4 (±40%)
    # 夜间模拟增强决策：每个 batch 以 night_aug_prob 概率触发
    night_flags = torch.rand(n_batches, device=device) < night_aug_prob
    night_dark = 0.25 + 0.30 * torch.rand(n_batches, device=device)      # 0.25~0.55 压暗系数
    bloom_strength = 0.30 + 0.40 * torch.rand(n_batches, device=device)  # 0.30~0.70 炫光强度

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:2d}", ncols=130, leave=True)

    for step, batch in enumerate(pbar):
        # 6 元组：(img, depth, mask, img_prev, depth_prev, mask_prev)
        images, gt_depth, valid_mask, images_prev, depth_prev, mask_prev = batch
        # 时序 loss 子采样：非触发 batch 不传 prev 数据（省 H2D 拷贝 + 第二次前向）
        do_temporal = use_temporal and (step % temporal_every == 0)
        if not do_temporal:
            images_prev = depth_prev = mask_prev = None

        images = images.to(device, non_blocking=True)
        gt_depth = gt_depth.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        if do_temporal:
            images_prev = images_prev.to(device, non_blocking=True)
            depth_prev = depth_prev.to(device, non_blocking=True)
            mask_prev = mask_prev.to(device, non_blocking=True)

        # 增强（含夜间模拟）
        images, gt_depth, valid_mask = _gpu_augment(
            images, gt_depth, valid_mask, step,
            flip_decisions, brightness_factors, contrast_factors
        )
        images = _gpu_night_augment(images, night_flags[step], night_dark[step], bloom_strength[step], bloom_kernel)
        if do_temporal:
            images_prev, depth_prev, mask_prev = _gpu_augment(
                images_prev, depth_prev, mask_prev, step,
                flip_decisions, brightness_factors, contrast_factors
            )
            images_prev = _gpu_night_augment(images_prev, night_flags[step], night_dark[step], bloom_strength[step], bloom_kernel)

        pred_depth = _forward_depth(model, images, gt_depth.shape[-2:], amp_enabled=amp_enabled)
        pred_prev = None
        if do_temporal and images_prev.abs().sum() > 0:  # 跳过第一个样本（全零占位）
            pred_prev = _forward_depth(model, images_prev, gt_depth.shape[-2:], amp_enabled=amp_enabled)

        # Loss 在 fp32 计算（AMP 不稳定）
        with torch.amp.autocast("cuda", enabled=False):
            loss, loss_dict = loss_fn(pred_depth.float(), gt_depth.float(), valid_mask)

            # 时序梯度 loss（子采样批次，权重已按倍数补偿）
            if do_temporal and pred_prev is not None:
                t_loss, t_dict = temporal_loss_fn(
                    pred_depth.float(), pred_prev.float(), valid_mask & mask_prev
                )
                if t_loss is not None:
                    loss = loss + temporal_scale * t_loss
                    loss_dict["temporal"] = t_dict["temporal"]
                else:
                    loss_dict["temporal"] = 0.0
            else:
                loss_dict["temporal"] = 0.0

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

        pbar.set_postfix(loss=f"{loss_dict['total']:.4f}",
                         tmp=f"{loss_dict.get('temporal', 0.0):.3f}",
                         skip=skipped_batches)

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
    pbar = tqdm(dataloader, desc="Val   ", ncols=130, leave=False)

    with torch.no_grad():
        for batch in pbar:
            # 验证只评纯 SILog，不计算时序 loss，仅取当前帧（prev 三字段丢弃）
            images, gt_depth, valid_mask, *_ = batch
            images = images.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            pred_depth = _forward_depth(model, images, gt_depth.shape[-2:])

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


def _build_checkpoint_dict(model, optimizer, scaler, epoch, avg_loss, val_loss,
                           best_loss, current_iter, stage, model_name,
                           no_improve_count, use_lora=False):
    """拼 checkpoint 要存的字典；LoRA 模式下先把 LoRA 合并回原权重（标准键名），
    这样 detect/evaluate 能直接加载，恢复训练时 LoRA 也会从零重新开始。"""
    if use_lora:
        model_state_dict = merge_lora_for_save(model)
    else:
        model_state_dict = model.state_dict()
    return {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": avg_loss,
        "val_loss": val_loss,
        "best_loss": best_loss,
        "current_iter": current_iter,
        "stage": stage,
        "model_name": model_name,
        "no_improve_count": no_improve_count,
    }


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
    parser.add_argument("--resume", type=str,
                        default=_abs_path(root, train_cfg.get("resume")))
    parser.add_argument("--lora", action="store_true",
                        default=train_cfg.get("lora", False),
                        help="启用 LoRA 微调（冻结 ViT，只调 DPT head 的 1x1 卷积，防过拟合）")
    parser.add_argument("--lora-rank", type=int,
                        default=train_cfg.get("lora_rank", 8),
                        help="LoRA 秩（越大容量越大，默认8）")
    parser.add_argument("--temporal-weight", type=float,
                        default=train_cfg.get("temporal_weight", 0.0),
                        help="时序梯度 loss 权重（0=禁用，0.1~0.3 推荐，约束相邻帧深度一致性）")
    parser.add_argument("--night-aug-prob", type=float,
                        default=train_cfg.get("night_aug_prob", 0.3),
                        help="夜间模拟增强触发概率（0=禁用，0~1，路线A：低照度+眩光bloom）")
    parser.add_argument("--bloom-kernel", type=int,
                        default=train_cfg.get("bloom_kernel", 11),
                        help="夜间眩光 bloom 模糊核大小（奇数，越大光晕越扩散）")
    parser.add_argument("--temporal-every", type=int,
                        default=train_cfg.get("temporal_every", 1),
                        help="时序 loss 子采样间隔（每 N 个 batch 算一次第二次前向，权重 ×N 补偿，N>1 省时）")
    parser.add_argument("--val-every", type=int,
                        default=train_cfg.get("val_every", 1),
                        help="验证频率（每 N 轮验证一次，N>1 省时；早停按验证次数计）")
    parser.add_argument("--patience", type=int,
                        default=train_cfg.get("patience", 15),
                        help="早停耐心（连续 N 次验证无改善则停止）")
    parser.add_argument("--base-lr", type=float,
                        default=train_cfg.get("base_lr", 5e-6),
                        help="基础学习率（backbone 用此值，head 自动 ×10）")
    parser.add_argument("--num-workers", type=int,
                        default=train_cfg.get("num_workers", 6),
                        help="DataLoader worker 进程数（0=主进程加载）")
    parser.add_argument("--prefetch-factor", type=int,
                        default=train_cfg.get("prefetch_factor", 4),
                        help="每个 worker 预取 batch 数（num_workers>0 时生效）")
    parser.add_argument("--seed", type=int,
                        default=train_cfg.get("seed", 42),
                        help="随机种子（数据集划分）")
    parser.add_argument("--min-delta", type=float,
                        default=train_cfg.get("min_delta", 1e-4),
                        help="早停最小改善幅度（验证 loss 改善小于此值视为无改善）")
    parser.add_argument("--weight-decay", type=float,
                        default=train_cfg.get("weight_decay", 0.01),
                        help="AdamW 权重衰减")
    parser.add_argument("--loss-lambd", type=float,
                        default=train_cfg.get("loss_lambd", 0.5),
                        help="SILog loss 方差惩罚系数 lambda（官方默认 0.5）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 加载模型
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print(f"[INFO] 加载预训练学生模型: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name).to(device)

    # LoRA 微调：必须在加载 checkpoint 之前包装，键名才对得上
    if args.lora:
        apply_lora_to_dpt(model, rank=args.lora_rank)

    if args.resume:
        print(f"[INFO] 从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        load_state_dict_smart(model, ckpt["model_state_dict"])
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

    # 时序梯度 loss
    temporal_loss_fn = TemporalGradientLoss() if args.temporal_weight > 0 else None
    use_temporal = args.temporal_weight > 0
    if use_temporal:
        print(f"[Temporal] 时序梯度 loss 已启用，权重={args.temporal_weight}")
    if args.night_aug_prob > 0:
        print(f"[NightAug] 夜间模拟增强已启用，概率={args.night_aug_prob}，bloom核={args.bloom_kernel}")

    # 流式加载数据集，不预加载进内存
    full_dataset = DepthTrainDataset(
        args.data_dir, args.process_res,
        temporal_pairs=use_temporal,
    )

    # 按 9:1 划训练/验证，固定种子方便复现
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    # Subset 包装原始 dataset，自动做索引映射
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    print(f"[Split] 训练集: {n_train} | 验证集: {n_val}")

    # DataLoader 配置
    # Windows 下 num_workers>0 用 spawn，DepthTrainDataset 必须可 pickle
    # persistent_workers 让 worker 跨 epoch 存活，避免重复启动开销
    num_workers = max(0, int(args.num_workers))
    pin_memory = sys.platform != 'win32'  # Windows WDDM 下 pin_memory 有风险
    prefetch_factor = max(1, int(args.prefetch_factor))
    persistent_workers = num_workers > 0
    # 验证集无反向传播，显存宽裕，可加大 batch 提速
    val_batch_size = min(8, max(4, args.batch_size * 2))
    args.val_every = max(1, int(args.val_every))

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
            collate_fn=_collate_batch,
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
        val_subset, val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )

    n_batches = len(train_loader)  # DataLoader 自动 drop_last=False 覆盖所有样本

    # 优化器 + 损失
    base_lr = args.base_lr
    optimizer = build_optimizer(model, stage=args.stage, base_lr=base_lr,
                                weight_decay=args.weight_decay)
    loss_fn = DA2MetricLoss(lambd=args.loss_lambd)

    # poly decay 的 total_iters
    total_iters = args.epochs * n_batches

    # 恢复训练状态
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

    # 混合精度 scaler
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    # stage 切换时不加载旧 scaler state（避免状态不一致）
    if resume_info is not None and ckpt.get("stage", 1) == args.stage and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # 早停配置：看验证集 val_loss，按验证次数计
    patience = max(3, int(args.patience))
    min_delta = float(args.min_delta)
    no_improve_count = 0
    if resume_info is not None:
        no_improve_count = ckpt.get("no_improve_count", 0)

    # 训练循环
    print(f"\n{'=' * 60}")
    print(f"开始训练: Stage {args.stage} | Epochs: {args.epochs} | "
          f"Batch: {args.batch_size} | Base LR: {base_lr:.1e}")
    print(f"输出: 真实米数 (度量深度) | AMP: bfloat16 + GradScaler")
    print(f"验证集: {n_val} 样本 | 早停: patience={patience} (每 {args.val_every} 轮验证) | min_delta={min_delta}")
    print(f"数据加载: 流式 + DataLoader(num_workers={num_workers})")
    print(f"时序子采样: 每 {args.temporal_every} batch | 验证 batch={val_batch_size}")
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
            temporal_loss_fn=temporal_loss_fn, temporal_weight=args.temporal_weight,
            night_aug_prob=args.night_aug_prob, bloom_kernel=args.bloom_kernel,
            temporal_every=args.temporal_every,
        )

        elapsed = time.time() - t0

        # 验证集评估（每 val_every 轮一次；验证不计算时序 loss，只评纯 SILog）
        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            val_loss = validate(model, val_loader, loss_fn, device)
            print(f"\nEpoch {epoch}/{args.epochs} | Train: {avg_loss:.4f} | "
                  f"Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")

            # 早停检查（基于 val_loss）
            is_best = val_loss < best_loss - min_delta
            if is_best:
                best_loss = val_loss
                no_improve_count = 0
                torch.save(_build_checkpoint_dict(
                    model, optimizer, scaler, epoch, avg_loss, val_loss,
                    best_loss, current_iter, args.stage, args.model_name,
                    no_improve_count, use_lora=args.lora,
                ), os.path.join(args.output_dir, "best_model.pt"))
                print(f"  >> 保存最佳模型 (val_loss={val_loss:.4f})")
            else:
                no_improve_count += 1
                print(f"  -- 验证集无改善 ({no_improve_count}/{patience})")
        else:
            val_loss = float("nan")
            print(f"\nEpoch {epoch}/{args.epochs} | Train: {avg_loss:.4f} | "
                  f"跳过验证 (每 {args.val_every} 轮一次) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s")

        torch.save(_build_checkpoint_dict(
            model, optimizer, scaler, epoch, avg_loss, val_loss,
            best_loss, current_iter, args.stage, args.model_name,
            no_improve_count, use_lora=args.lora,
        ), os.path.join(args.output_dir, "latest_model.pt"))

        if no_improve_count >= patience:
            print(f"\n[Early Stop] 验证集 {patience} 轮无改善，提前停止训练")
            break

    print(f"\n{'=' * 60}")
    print(f"训练完成! 最佳 Val Loss: {best_loss:.4f}")
    print(f"模型保存在: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
