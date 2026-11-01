# TRAIN_REFACTOR_CHANGELOG — 训练脚本路线优化记录

日期：2026-10-14
范围：`train.py`（933 行）+ `train_judge.py`（1219 行）+ `config.yaml`
原则：与 detect 会话完全相同的路线优化标准——**原有功能 / 输出结果 / 行为逻辑完全不变**，
每模块优化后立即做功能一致性验证（等价层 + 开关层）。
未改动：`train_classifier.py`（detect/training/ 下）全程不动。
备份：`train_backup_20261014/`（train.py / train_judge.py / config.yaml 原始副本 + 全部验证资产）。

---

## T1 — train.py 配置透传（打通 config → CLI → 使用链）

### 变更
- `build_optimizer(model, stage=1, base_lr=5e-6, weight_decay=0.01)` 签名显式化，
  两处 AdamW 统一 `weight_decay=weight_decay`（原硬编码 0.01 移到参数）
- 删除死参数 `--grad-accumulate`（从未被使用，batch_size=4 实际生效行为不变）
- 新增 CLI 参数（默认值全部 = 原硬编码值，零行为变化）：
  `--base-lr` `--num-workers` `--prefetch-factor` `--seed` `--min-delta`
  `--weight-decay` `--loss-lambd`（默认均从 `config.yaml` train 段读取）
- `--lora` 默认改读 `train_cfg.get("lora", False)`，`--lora-rank` 默认读 `lora_rank`
- main 内接线：`torch.Generator().manual_seed(args.seed)`（原硬编码 42）、
  `num_workers = max(0, int(args.num_workers))`、`prefetch_factor = max(1, ...)`、
  `DA2MetricLoss(lambd=args.loss_lambd)`、`min_delta = float(args.min_delta)`、
  dataset 实例化去掉从未起作用的 `augment=True` 实参
- config.yaml train 段：删 `grad_accumulate: 4`，新增 8 个键
  （base_lr / num_workers / prefetch_factor / seed / min_delta /
  weight_decay / loss_lambd / lora / lora_rank，共 9 个）

### 验证
- 等价层：新默认值 == 原硬编码值（静态核对 + 两环境同跑结果一致）
- 开关层：改 config 值后重跑，打印行与 checkpoint 均反映新值（见 T6-b）

## T2 — 统一深度推断块 `_forward_depth`

### 变更
- 新增模块级函数 `_forward_depth(model, images, target_size, amp_enabled)`：
  AMP 前向 → squeeze → bicubic 对齐目标分辨率（与原两处重复块逐行等价）
- `train_one_epoch` 中两处重复前向块替换为调用该函数；
  时序分支加 `images_prev.abs().sum() > 0` 全零占位跳过（语义与原版一致）

### 验证
- 等价层：1 epoch 训练/验证 loss 逐位一致（1.0472 / 1.0247）

## T3 — checkpoint 保存统一 `_build_checkpoint_dict` + LoRA 保存接线修复

### 变更
- 新增 `_build_checkpoint_dict(...)` 组装保存 dict，
  键顺序（epoch/train_loss/val_loss/best_loss/current_iter/stage/
  model_name/no_improve_count/model_state_dict/optimizer_state_dict）与原版完全一致
- 两处 `torch.save` 改为调用该函数
- **LoRA 接线修复**：`use_lora=True` 时保存前调用 `merge_lora_for_save(model)`
  合并为标准键名（避免保存带 `lora_` 前缀的中间权重）；当前 config `lora: false`，
  本次修复不产生任何行为变化，等未来启用 LoRA 时保存/加载才正确

### 验证
- 等价层：best/latest checkpoint 顶层键集合、全部标量、model_state_dict 一致

## T4 — 死代码清理 / 内存优化

### 变更
- `validate(model, dataloader, loss_fn, device)`：删除死时序分支
  （`use_temporal`/`prev`/`n_batches` 参数原版 validate 从未使用），
  调用点同步瘦身
- `DepthTrainDataset`：
  - 删 `augment` 参数（全链路从未生效，样本加载行为不变）
  - 非时序模式下 prev 三字段返回 `None`（原版每样本构造 ~3.4MB 全零张量纯浪费；
    新增 `_collate_batch` 透传 None，已实测 `default_collate` 不支持 None 会崩）
  - 时序模式（temporal_pairs=True）保持全零占位语义（train 的 `.sum()>0` 跳过逻辑依赖）
- `_make_loader` 接入 `collate_fn=_collate_batch`

### 验证
- 等价层：时序禁用路径 loss 逐位一致；非时序路径内存占用下降

## T5 — train_judge.py 通读优化

### 变更
- 删死常量 `LABELS = {"car": 0, ...}`（全程未引用，类别映射本就用 `STATE_TO_IDX`）
- `find_pairs_from_sample_root`：开头一次性建 `inp_index`（首个优先，与原逐个查找
  `if stem not in inp_index` 语义一致），内层循环 O(n) 查表替代反复扫描
- `train_model(...)` 增加 `dropout=0.3` 参数并传入 `JudgeMLP(in_dim, dropout=dropout)`
  （原版 Dropout 0.3 硬编码在类默认值里，行为不变，只是显式接线）
- `main()` 重构：
  - `root/cfg/judge_cfg/detect_cfg` 提前到 parser 之前
  - parser 默认值全部来自 `judge_cfg`（epochs/batch/window/lr/weight_decay/
    patience/dropout/seed/val_video_frac，默认 = 原硬编码值）
  - 新增 `--weight-decay`；`train_model` 调用加 `weight_decay=args.weight_decay`
- config.yaml 末尾新增 `judge:` 段（epochs/batch/window/lr/weight_decay/dropout/
  patience/seed/val_video_frac，值与原硬编码一致）

### 验证
- 等价层：报告/metrics/模型逐位一致（见 T6-a）

---

## T6 — 功能一致性验证

### T6-a 等价层（老/新两环境同跑，程序化比对产物）
环境：两侧同源补丁（root 硬编码 + seed 42 + cudnn deterministic），
共用真实 config，`--stage 1` 触发 stage 切换路径（两侧路径一致）。

| 比对项 | 结果 |
|---|---|
| 训练日志 loss（1 epoch） | 老/新均为 Train 1.0472 / Val 1.0247 |
| best/latest_model.pt 顶层键集合 | 完全一致 |
| checkpoint 标量（epoch/loss/iter/stage/…） | 一致（浮点相对差 <1e-6） |
| model_state_dict 数值 | max\|d\|=5.3e-6 < 2e-4（GPU 浮点噪声级） |
| optimizer_state_dict 数值 | max\|d\|=9.8e-5 < 1e-3 |
| judge classification_report.txt | 逐字节一致（仅耗时行归一化） |
| judge metrics.json | 一致（排除耗时/路径字段） |
| judge.pt | max\|d\|=0，逐位一致 |

### T6-b 开关层（临时改 config → 新版脚本不加 CLI 覆盖 → 验证生效）
- train 段：`base_lr: 0.0001 / num_workers: 0 / min_delta: 0.05 /
  weight_decay: 0.5 / loss_lambd: 0.9` →
  打印行 `Base LR: 1.0e-04`、`min_delta=0.05`、`num_workers=0` 生效；
  checkpoint optimizer param_groups weight_decay=0.5 生效；
  lambd=0.9（Train 0.6646/Val 0.6373）与 lambd=0.0（Train 0.8612/Val 0.6445）
  两次同种子运行的 loss 明显不同 → loss_lambd 生效
- judge 段：`epochs: 2 / batch: 64 / window: 3 / lr: 0.01 / dropout: 0.5 / patience: 1` →
  打印行 `参数: epochs=2 batch=64 window=3 lr=0.01 dropout=0.5 patience=1` 生效，
  完整训练 2 轮正常收敛（best epoch=2, Accuracy=0.30）
- 验后 config.yaml 已从备份恢复（与终态基线 MD5 一致）

---

## 结论
- 两个训练脚本全部优化完成，等价层 + 开关层双验证通过，
  原有功能/输出/行为逻辑完全不变
- 训练期间 `checkpoints/` 目录与已完成 Stage 1/2 模型未受影响（只读引用）
- 新增配置键均有默认值兜底，[conf] 修改不再需要动代码