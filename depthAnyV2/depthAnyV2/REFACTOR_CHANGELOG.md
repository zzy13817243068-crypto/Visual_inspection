# Detect 脚本执行路线优化 — 变更记录（REFACTOR_CHANGELOG）

- 日期：2026-10-12
- 范围：`detect/detect.py`、`detect/detect_car.py`、`detect/detect_sel_videos.py`
- 未动：`train.py` 与 config.yaml 的训练段完全搁置（Stage 2 训练运行中）。
  config.yaml 在 M6 验证期间临时改过 `judge_mlp.enabled`、`saturation_threshold` 两行；
  验证后 saturation_threshold 恢复 235，judge_mlp.enabled 按用户决策改为 false（详见 M6 节）。
  验证前快照：`detect_backup_20261012/config.yaml.m6bak`。

## 备份与回滚

M0 已将三个脚本完整备份到 `detect_backup_20261012/`：

| 文件 | 说明 |
|---|---|
| `detect.py`（116975 B） | 重构前原始版本 |
| `detect_car.py`（41777 B） | 重构前原始版本 |
| `detect_sel_videos.py` | 重构前原始版本 |
| `_env_old/` | 旧版可运行副本（detect.py + detect_car.py），供 M6 等价验证；其中 detect.py 的 `_load_config` 带指向项目根 config.yaml 的测试补丁（仅测试副本，不影响线上） |
| `config.yaml.m6bak` | M6 验证前 config.yaml 快照 |

回滚方式：把对应备份文件复制回 `detect/` 覆盖即可（`_env_old/` 里的 detect.py 多了一处 M6 测试补丁，
回滚请使用备份根目录下的原文件，勿用 `_env_old/` 副本）。

## M1 配置透传（已完成）

- 目标（用户选定方案 B）：检测参数统一由 config.yaml 驱动，命令行参数仍可覆盖，注重长期可维护性。
- 改动：`detect/detect.py` 新增 `build_detector_config` 工厂，集中读取 car_detect 各子块
  （judge_mlp / night / saturation_threshold / clahe / mog2 / weights / yolo）并统一默认值与路径解析；
  三个脚本的 main() 改为调用该工厂。
- 验证：`_verify_m1_config.py` 通过；M6 运行日志中新旧脚本的参数行完全一致
  （骤减阈值 0.4m(≥2点) / 相对跌幅 8% / 15 点坐标 / YOLO 路径均相同）。

## M2 深度推断块统一（已完成）

- `detect/detect.py` 新增 `_infer_depth_map` 与 `_extract_pixel_depths`（可选 frame_bgr/sat_threshold 过曝屏蔽）。
  detect_car.py 与 detect_sel_videos.py 的逐帧深度提取统一走该路径；过曝点语义统一为 nan
  （`sat_threshold<=0` 时完全不屏蔽，与原行为一致）。
- 验证：M6 等价层（sat=0 下新旧点值一致）+ 开关层合成单测
  （sat=0 → 12.5；sat=235/254 → nan，详见下）。

## M3 初始化块工厂化（已完成）

- `detect/detect.py` 新增 `build_detector_pipeline(detector_config, pixel_points, effective_fps,
  with_vision, verbose, yolo_model_path_default)`，一次性构建
  `(detector, brightness_monitor, motion_detector)`，替代三个脚本中重复的内联初始化块
  （含 YOLO 视觉、夜间自适应参数、MOG2、亮度监控）。
- 三个参数精确保持 4 个调用点原行为差异：

| 调用点 | with_vision | verbose | yolo 默认路径 |
|---|---|---|---|
| detect.py 主流水线 / 预览 | False | "basic" | （带视觉参数不再需要） |
| detect_car.py | True | "full" | "yolov8n.pt" |
| detect_sel_videos.py | True | "none" | "" |

- 验证：`_verify_m3_pipeline.py` 20/20 通过。

## M4 缓存/同步/内存优化（已完成）

- 合并双套亮度计算：新增模块级 `_frame_mean_brightness()`；
  `VehicleDetector._compute_frame_brightness` 与 `BrightnessMonitor.compute_brightness` 统一调用
  （`independent_scale` 参数分别保持"单一 scale"与"x/y 独立缩放"两种原语义）。
- 验证：`_verify_m4_brightness.py` 7 种分辨率下新旧亮度计算逐位一致。
- 评估后决定不改（备查）：
  1. `depths_list` 逐帧累积：`save_depth` 功能需求（默认关闭），保留；
  2. `NoCarSegmentRecorder.add_frame` 的 `frame.copy()`：防帧缓冲被后续原地修改，开销可忽略；
  3. 隐式 GPU 同步：代码中无逐帧 `.item()` 调用，唯一的 `.cpu().numpy()` 同步点属必要操作，无优化空间。

## M5 融合判定统一与死代码清理（已完成）

- 删除 `detect/detect.py` 中无任何调用者的死函数 `detect_vehicle_present`；
- detect.py 主流水线融合判定处注释改为"主流水线简易版：深度+MOG2 双重验证，无 YOLO 视觉协同……
  与 detect_car.py 的 fuse_v8_car_state() 是两套不同用途的判定，请勿混用"，防止后续误合并；
- detect_car.py 中两处陈旧注释 `detect_no_car.py` 更正为 `detect_sel_videos.py`；
- 三脚本 import 同步清洗（移除不再直接使用的类名，改导工厂函数）。

## M6 分层验证

### 等价层（judge=false、saturation_threshold=0，新旧同配置对比）

- 方法：旧版备份放入 `_env_old/` 运行（其 `_load_config` 带测试补丁硬指向项目根 config.yaml，
  保证与新读取完全相同的配置与路径解析）；新旧均以 `--device cpu --seed 42` 跑 3 个视频，
  对比 report.json 的行为字段（state_counts / segments / pixel_points 等）。
  验证期间 GPU 被 Stage 2 训练占满，故全程用 CPU。
- 结果：**3/3 全部一致**（`_verify_m6_compare.py`，见 `detect_backup_20261012/`）：

| 视频 | 帧数 | 新旧一致结果 |
|---|---|---|
| 南独乐河出4/1782525429_71.mp4 | 8 | car=0, no_car=4, unknown=4, 1 片段(4帧) |
| 北务出京出3/1784097456_642.mp4 | 7 | car=0, no_car=3, unknown=4, 1 片段(3帧) |
| 薛家庄进京出3/1783385230_57.mp4 | 23 | car=21, no_car=0, unknown=2, 0 片段 |

  （主视频为真实有车场景，覆盖了有车判定/fusion 全链路；逐帧进度 pts 数组亦完全一致；
   等价层日志中均无 [JudgeMLP] 输出，确认 judge 未参与。）

### 开关层（judge=true、saturation_threshold=235）

- 饱和屏蔽：合成单测通过——`sat=0` 过曝点返回原值 12.5；`sat=235/254` 返回 nan
  （`_extract_pixel_depths` 直接调用验证）。
- judge：开关开启时日志出现 `[JudgeMLP] 模型已加载: ...\judge_mlp.pt (in_dim=125, window=5)`，
  关闭时无任何 JudgeMLP 输出——开关接线正常。
- **重要发现（开关归因，主视频 23 帧有车场景）**：

| 组合 | 结果 |
|---|---|
| judge=off + sat=0（=旧版实际生效行为） | car=21, no_car=0, unknown=2 |
| judge=off + sat=235 | car=19, no_car=0, unknown=4 |
| judge=on  + sat=0 | car=0, no_car=23, unknown=0 |
| judge=on  + sat=235 | car=0, no_car=23, unknown=0 |

- 归因结论：judge 是主因——当前 `judge_mlp.pt` 把每帧 v8 判定都纠正为 no_car（差异帧的
  MLP 置信度均 ≥0.70 触发纠正）；sat=235 仅影响 2 帧（car→unknown）。
- 背景：旧版脚本 main() 从不把 `judge_mlp` / `saturation_threshold` 传入 detector_config，
  两者在旧版中是"死配置"（config 里写 true/235 也从不生效）。M1/M2 透传后首次真正生效，
  因此等价层特意关闭两开关进行对比。
- **用户决策（2026-10-12）**：`judge_mlp.enabled` 改为 false（透传能力保留，judge 模型
  训练靠谱后再开）；`saturation_threshold` 保持 235（接受实测 2 帧差异）。
  当前生产配置（judge=false + sat=235）行为已实测：car=19/no_car=0/unknown=4，
  日志无 [JudgeMLP] 输出。config.yaml 与 m6bak 快照相比仅 judge 一行不同（刻意为之）。

## 已知行为差异（预期内）

1. 越界/无效监测点的深度在可视化中显示为 nan（语义统一）；
2. `saturation_threshold>0` 时过曝像素点屏蔽为 nan——该功能的代码重构前已存在，
   但配置从未传入 detector_config（从未实际生效）；重构补全透传后正式生效。
   用户确认保持默认 235（实测仅 2 帧差异，见 M6）；设为 0 时与旧版完全一致；
3. `judge_mlp` 透传同理（旧版是死配置）。因当前 judge 模型不可靠，
   用户决策 config 中保持 `enabled: false`，透传能力保留待模型训好后启用。