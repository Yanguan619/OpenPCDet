# PointPillars 性能分析（PERFORMANCE）

PointPillars 在 NPU（Ascend 310P3）上的性能数据、瓶颈分析与优化方向。
精度（AP）记录见 `PRECISION.md`，改动历史见 `CHANGELOG.md`。

**当前最新性能**（2026-09-29，集成态 master，000008.bin，device 1，30 iters 稳态中位）：

- **单帧 demo E2E 22.5ms**（无 FOV，体素化 122555 点）：读bin 0.35 + prepare_data 9.62（AscendC 体素化
  + 设备常驻）+ collate 0.13 + index_map 0.32 + feeds 0.41 + set_dym 0.09 + **OM 推理 10.97（head 图手术
  ABCD）** + D2H 0.12 + 后处理 0.48；检测 33 框（Car 12/Ped 14/Cyc 7）与基线逐帧一致（详见 §1.3）；
- 单帧口径演进：~93ms（numba 体素化）→ ~67ms（09-28 AscendC est.）→ **22.5ms**（09-29 三线攻坚）；
- 全量管线口径（有 FOV，200 帧）09-28 值 **63.3ms/帧**，本轮未重测（无数据集环境）；本轮前处理/推理/
  后处理各段均有收益，待数据集环境复测；
- 基线 369ms/帧（2026-09-22，全量 3769 帧初版动态 OM）→ 63.3ms，**5.8x**（全量口径）。

> **口径术语**（全文统一；历史版本中"全量"曾混用两种含义，以本约定为准）：
> - **demo 单帧**：无 FOV，直接体素化原始点（000008 为 122555 点）；
> - **全量管线口径**：与数据集评测一致的管线（FOV 过滤 → 体素化 → 推理 → 后处理），
>   帧数另行标注——"200 帧" = val **前** 200 帧（`--frames 200`），"全量 3769 帧" = 整个 val；
> - **推理链路**：不含体素化的段落（collate → pad/动态 shape → feeds → forward → postproc）。

## 1. E2E 延迟现状（200 帧稳态，om_ref_test.py，全量管线口径）

> 200 帧 = val 前 200 帧；静态 9000 OM 在该子集 skipped_M=0（全量 val 的 M 分布见 §4）。

| 阶段 | 实现 | fp32 | fp16(force) | **fp16+topk图内(P2)** |
|---|---|---|---|---|
| 前处理（FOV + voxelize + pad + 读图尺寸） | CPU numpy/numba | 42 ms | 41 ms | 43 ms |
| OM 推理（NPU 前向，静态 M=9000） | NPU | 24 ms | 17 ms | 17.5 ms |
| 后处理（sigmoid + topk + NMS） | CPU torch + numba + numpy | 16.5 ms | 15 ms | **3.5 ms** |
| **E2E 总计** | | **82.8 ms** | **73.1 ms** | **64.3 ms** |

> fp32: `pointpillar_base_fp32_static9000_v2.om`；force: `pointpillar_base_fp16_static9000_force.om`；
> **topk(P2)**: `pp_topk_static9000`（图内 ReduceMax+TopK(4096)，输出 top-K box/cls ~160KB，NMS 留 CPU）。
> 实测命令：`python npu/om_ref_test.py --om <om> --frames 200`
> 基线 369ms/帧（2026-09-22 全量 3769 帧初版动态 OM）→ topk(P2) 64.3ms，累计 **5.7x**。
> **P2 精度与 base 完全一致**（top-K 按 raw cls 单调等价 sigmoid 排序）：本口径（200 帧）AP
> Car 77.81 / Ped 59.86 / Cyc 38.32；全量 3769 帧口径见 1.2 与 §6。
> 导出：`python npu/export_onnx.py --topk-only --skip-export --base-output <base> --output <topk>`

### 1.1 静态 pad 到最大的 E2E 进一步拆分（全量 3769 帧）

全量 val 结构（用户机，avg 71.2ms/帧）：**前处理 39.6 + 推理 19.7 + 后处理 11.9**。

本机子环节逐段实测（每帧 avg；**各段单独计时、合计 ~137ms，含本机/用户机与 CPU 负载差异，
不等于 E2E；占比按子环节合计计**）：

| 子环节 | 本机耗时 | 占比 | 说明 |
|---|---|---|---|
| 读 bin | 1.5 ms | 1% | |
| **FOV numba** | 50.9 ms | 37% | 最大 CPU 单项（§2 稳态口径 ~30ms，机器/负载差异） |
| voxelize numba | 17.8 ms | 13% | |
| pad→18000 | 4.2 ms | 3% | 每帧 pad |
| **推理 forward** | 19.8 ms | 14% | 静态 18000，**总是处理 18000 行** |
| 后处理 sigmoid + D2H | 19.8 ms | 14% | 13MB D2H + torch.sigmoid/max |
| 后处理 NMS（增量） | 23.3 ms | 17% | top-4096 |

> 静态 pad 的**推理浪费**在于：M 平均 ~8000 的帧也处理 18000 行 VFE（推理 19.7ms 固定，与 M 无关）；
> 前处理 pad 开销仅 4.2ms，不是主因。→ 见 1.2 动态对比。
> （"M 平均 ~8000" 口径待复核：与 §4 的 100 帧随机抽样"54% 帧 M>9000"未对齐——疑为
> 200 帧子集或旧 voxelization 统计，voxelization 产出已随代码变化，见 CHANGELOG。）

### 1.2 动态 vs 静态 pad（全量 3769 帧，fp16）

| 指标 | 动态 fp16（1~18000） | 静态 18000（pad） |
|---|---|---|
| E2E avg | **67.5 ms** | 71.2 ms |
| 前处理 | 34.5 ms | 39.6 ms（+pad→18000 ~5ms） |
| 推理 | 21.3 ms | 19.7 ms（省动态调度） |
| 后处理 | 11.8 ms | 11.9 ms |
| AP（Car/Ped/Cyc） | 77.07/51.93/61.95 | 77.07/51.93/61.95（相同） |

**结论：动态 OM 更优**（净胜 ~3.5ms）——静态推理虽省 `set_dynamic_shape` 1.6ms，但静态总是处理 18000 行的浪费 + pad 开销反超；
且静态 pad 与动态预测**逐位一致**（pad 行不参与 scatter），全量 AP 完全相同。
**全量 val 推荐 topk 动态 fp16 om**（`pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om`，
E2E 最优见 1.4；base 版为 `pointpillar_base_fp16_dynamic18000_force_linux_aarch64.om`）。

### 1.3 单帧 demo 完整 E2E（om_ref_demo.py，无 FOV 过滤）

> 与 1.1/1.2 不同：demo 路径**不做 FOV 过滤**（与 `tools/demo.py` 一致），直接体素化 122555 原始点。
> 以下为单帧完整 E2E（含 `__getitem__` 体素化），本机 000008，avg 10。

| 阶段 | base fp16 static18000 | base fp16 dynamic18000 | topk fp16 dynamic18000 | **09-29 三线攻坚集成态** |
|---|---|---|---|---|
| getitem（读 bin + 体素化 122555 点） | ~34 ms（42%） | ~34 ms（44%） | ~32 ms（48%） | **10.0 ms（44%）** |
| collate + to_tensor | 4.6 ms | 4.8 ms | 4.5 ms | **0.13 ms** |
| pad / index_map | 4.5 ms | 0.7 ms | 0.7 ms | **0.32 ms** |
| feeds（aclruntime.Tensor） | 4.8 ms | 2.4 ms | 2.4 ms | **0.50 ms**（含 set_dym 0.09） |
| **forward** | 19.7 ms | 21.5 ms | 22.8 ms | **10.97 ms**（surgery ABCD OM） |
| postproc（sigmoid + topk + NMS） | 13.4 ms | 14.3 ms | 4.0 ms | **0.60 ms**（含 D2H 0.12） |
| **单帧完整 E2E** | ~81 ms | ~78 ms | ~67 ms（est.） | **22.5 ms**（30 iters 中位） |

> ⚠️ 上表 getitem/E2E 为 **2026-09-28 体素化切 AscendC 后重算的估计值**（表内其它阶段仍为原 avg10 实测）：
> 体素化 generate 000008（M=7260）对照实测 **AscendC 21.3ms vs numba 53.4ms（~2.5x）**
> （另轮次 ~25 vs ~45-53ms，机器负载差异，见 5.1），getitem 由 52.6ms 降 ~20ms
> （测于 CPU 负载 20 的机器，静机待重测，见 5.1）。
> 三列均为真实存在 OM、同帧（000008，M=7260）同流程逐段计时。
> **结论按同模式比，别跨模式**（三列同时混了「topk 图内」与「static→dynamic」两个变量）：
> - **topk 图内（dynamic vs dynamic）**：省 ~11ms = postproc **-10.3**（TopK(4096) 后 NMS 只吃 4096 行）
>   + forward +1.3（图内多一次 TopK）+ getitem 噪声 ~-2；
> - **static→dynamic（同 base）**：再省 ~3.7ms = pad -3.8 + feeds -2.4，forward +1.8 反噬；
> - 两变量叠加即「static base → topk dynamic」共省 ~14.9ms——**勿全记在 topk 头上**。
> - **三种口径的当前数字**（术语见文首；09-29 集成态）：
>   - **demo 单帧**（无 FOV）：**22.5ms**（09-29 实测，device 1，30 iters 中位）——体素化（AscendC+设备常驻）占 ~44%；
>   - **推理链路**（不含体素化）：**~12.5ms**（collate 0.13 + index_map 0.32 + feeds 0.50 + forward 10.97 + postproc 0.60）；
>   - **全量管线口径 200 帧**（有 FOV）：63.3ms/帧（09-28 值，本轮各段优化未在全量口径复测）——FOV 后点数减
>     ~7 倍（000008：122555→17221），虽多一步 FOV，但体素化省更多 → 反而比 demo 快。
> - **`OM inference time` 只计 forward 段**，不代表整帧。

**09-29 三线攻坚收益归因**（topk est. ~67 → 实测 22.5；根因与细节见 CHANGELOG 09-29 条目）：

| 优化 | 段 | 前 → 后 | 收益 |
|---|---|---|---|
| 体素化 561000 修复（ThreadCtxGuard）+ numba 单遍 mask + MAX_NBLK 7→8 | getitem | ~32 → 10.0 | **−22** |
| collate 单帧 fast-path（双态）+ to_tensor/index_map 收敛 | collate+index_map | 5.2 → 0.45 | **−4.8** |
| 设备常驻 feeds（NPU tensor 经 BaseTensor 零拷贝直通） | feeds | 2.4 → 0.50 | **−1.9** |
| head 图手术 ABCD（恒等变换，同设备对照） | forward | 11.32 → 9.60 | **−1.7** |
| numpy max/argmax + numba NMS 标量化 + D2H copy=False | postproc | 4.0 → 0.60 | **−3.4** |

> 各段收益合计 −33.8ms，与 67→22.5（−44.5）的差额 ~10.7ms 为测量口径差（09-28 各值为 est. +
> device 0 高负载 avg10；本轮 device 1 同 OM 基线实测 forward 仅 11.3，旧值 22.8 高估 ~11ms）。
> 精度红线：33 框（Car 12/Ped 14/Cyc 7）与基线逐帧一致；设备常驻 ON/OFF（A/B 22.5 vs 23.7ms，净赚
> ~1.2ms）、nblk 7/8 框表均逐字节一致。推荐 OM：`pointpillar_base_fp16_dynamic18000_topk_surgery_abc_linux_aarch64.om`（TopK 4096 语义与原 topk OM 一致；ABCD 因截断风险已移除，见 §2 表注）。

### 1.4 topk 图内 OM（P2）全量管线口径验证（动态 18000，200 帧）

| 指标 | base fp16 dynamic18000 | **topk fp16 dynamic18000** |
|---|---|---|
| 前处理（FOV + voxelize） | ~34.5 ms | 37.5 ms |
| 推理 | ~21.3 ms | 22.0 ms |
| **后处理** | ~11.8 ms | **4.0 ms** |
| **E2E avg** | ~67.5 ms | **63.3 ms** |
| AP（Car/Ped/Cyc，200 帧） | 77.81/59.86/38.32 | 77.81/59.86/38.32（**逐位一致**） |

> topk OM 输出 `topk_boxes (1,4096,7)` / `topk_cls (1,4096,3)`（图内 ReduceMax+TopK(4096)，
> 无 ArgMax/NMS），D2H 13MB→164KB；`om_ref_demo/test` 的 `base_mode` 自动适配（无需改码）。
> 转换命令见 §3；`skipped_M=0` 全跑通。
> - AP 为 200 帧子集口径（与 §1 同子集；动态与静态预测逐位一致，故数值相同）。
>   **全量 3769 帧** base 口径为 77.07/51.93/61.95（见 1.2）；topk 的 3769 帧独立复测未见记录，
>   交付全量口径前建议补跑；
> - 该轮（2026-09-28）体素化仍为 numba（AscendC 启用前的生产路由）；topk 前处理 37.5 vs base
>   34.5 的 +3ms 未归因（前处理与 OM 无关，疑轮次噪声）。

## 2. 前处理子环节拆分（000008，稳态）

> 各行为**不同口径下的单段稳态计时，不可直接求和**对应某一 E2E；与 5.1 的分析配套阅读。

| 子阶段 | 口径 | 实现 | 耗时 |
|---|---|---|---|
| 读 bin | 通用 | `np.fromfile` | ~1 ms |
| FOV 过滤 | 全量管线（demo 无 FOV） | **numba 单内核 `_fov_filter_numba`**（乘加顺序与 `_mm3` 一致）+ 逐元素列式 | ~30 ms（1.1 全量本机子环节曾测 50.9ms，机器/负载差异） |
| 读图像尺寸 | 通用 | **PIL 只解析 PNG 头**（0.2ms，替代 io.imread 49ms） | <1 ms |
| voxelize | demo（无 FOV，122555 全量点） | **AscendC kernel（unum_ops，已启用）**；numba 为关闭时的回退 | **21.3 ms**（原 numba 53.4ms，~2.5x） |
| voxelize | 全量管线（FOV 后小输入） | 同上 | numba ~16-18 ms；AscendC 收益未测（aclnn 每调用固定 ~5ms 开销会吃掉小输入收益） |
| pad（静态 OM） | 静态 9000 口径 | 模块级缓冲复用（免每帧重分配） | ~10 ms（静态 18000 口径 4.2ms，见 1.1） |
| collate + ascontiguous + Tensor | 通用 | numpy/torch | ~3 ms |

## 3. OM 推理对比

### 关键 OM

> 单帧最快：静态 9000 force_fp16（17ms）；全量 val E2E 最优：动态 18000 topk（见 1.2/1.4）。
> 本机 `weights/` 仅存 topk 动态 OM；其余 OM 在用户机/大机器转换与实测，本表为各机实测汇总。

| OM | Shape | 前向 | 说明 |
|---|---|---|---|
| `pointpillar_base_fp32_static9000_v2.om` | 静态 M=9000 | **24 ms** | fp32 基线（ScatterND/ArgMax 消除） |
| `pointpillar_base_fp16_static9000_v2.om` | 静态 M=9000 | **18 ms** | mixed_float16 + mixlist，AP 1% 内 |
| `pointpillar_base_fp16_static9000_force.om` | 静态 M=9000 | **17 ms** | **force_fp16 全图**，AP 1% 内（单帧最快） |
| `pp_topk_static9000` | 静态 M=9000（图内 TopK 4096） | 17.5 ms | P2 topk，§1 200 帧口径所用 OM |
| `pointpillar_base_fp32_dynamic_linux_aarch64_linux_aarch64.om` | 动态 M=1~9000 | ~173 ms | 全量 val 动态（旧，ScatterND 未除） |
| `pointpillar_base_fp16_dynamic18000_force_linux_aarch64.om` | 动态 M=1~18000 | 21.3 ms | 全量 val base 推荐（动态，无 pad 浪费） |
| `pointpillar_base_fp16_static18000_force.om` | 静态 M=18000 | 19.7 ms | 全量 val 兼容（18000 覆盖所有帧，但总有 pad 浪费） |
| `pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om` | 动态 M=1~18000（图内 TopK 4096） | 22.0 ms | **P2 topk 图内**，后处理 11.8→4ms，E2E 最优（1.4） |
| `pointpillar_base_fp16_dynamic18000_topk_surgery_abc_linux_aarch64.om` | 动态 M=1~18000（图内 TopK 4096） | 9.65 ms（同设备对照） | **09-29 head 图手术 ABC（当前推荐，demo/bench 默认）**：A/B/C 同 ABCD 但不做 D，TopK 保持 4096，语义与原 topk OM 完全一致，全量评测无截断风险 |
| ~~`pointpillar_base_fp16_dynamic18000_topk_surgery_abcd_linux_aarch64.om`~~ | 动态 M=1~18000（图内 TopK 1024） | 9.60 ms（同设备对照，仅 000008 单帧） | 09-29 head 图手术 ABCD：比 ABC 仅快 0.05ms，但 TopK 1024 < NMS_PRE_MAXSIZE 4096，密集帧 pre-NMS 候选可能被图内静默截断（bit-exact 仅单帧验证）→ **09-29 已移除文件，全量禁用** |

### 前向优化链（186ms → 24ms，7.8x）

| 版本 | 优化 | 前向 |
|---|---|---|
| v2 原始（动态 fp32） | 含 head ScatterND + ArgMaxD | ~186 ms |
| + ScatterND→Slice+Concat（`KnowledgeScatterNdToConcat`） | 方向角修正 setitem 回归修复 | ~42.7 ms |
| + ArgMax2→Greater+Cast（`KnowledgeArgMax2ToCompare`）+ `fix_dir_reshape_dim` | dir_labels 消除 | ~23.7 ms |
| + 静态 M=9000（P0 padding） | 省动态调度 | **24 ms**（静态） |

- 图手术 knowledge 实现在 `/data/workspace/msit/onnx_optimizer/`（`KnowledgeScatterNdToConcat` / `KnowledgeArgMax2ToCompare`），结果 bit 一致（onnxruntime diff=0）。

### topk OM（P2）转换命令

（onnx：`weights/pointpillar_nms_base_v2_dynamic_topk.onnx`）

```bash
# 动态 18000 force_fp16（全量 val 推荐）
atc --model=weights/pointpillar_nms_base_v2_dynamic_topk.onnx --framework=5 \
    --soc_version=Ascend310P3 --output=weights/pointpillar_base_fp16_dynamic18000_topk \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"
# 静态 18000 版同理（M 固定 18000，勿加 --dynamic_dims）
```

## 4. 全量 val 集（3769 帧）注意

- **静态 9000 OM 只覆盖 M≤9000 的帧**：100 帧随机抽样中 54% 帧 M>9000（最大 ~16664）→ 全量 val 需
  **M≥18000 静态 OM** 或 **动态 OM（range 加大）**。转换命令已备于 `npu/convert_fp16_static9000.sh`。
  （注意：§1 的 200 帧子集为 val **前** 200 帧，恰好全部 M≤9000（skipped_M=0），与随机抽样分布不同；
  1.1 的 "M 平均 ~8000" 与本条 54%>9000 口径未对齐，待统一重测。）
- 全量 val 200 帧基准 AP（静态 9000，fp32）：Car 77.90 / Ped 57.95 / Cyc 37.05（与官方基线 1% 内）。
- **200 帧样本 AP 与全量差异明显**（fp16：Cyc 38.32 vs 全量 61.95）——val 前 200 帧非随机抽样
  （场景聚簇），跨样本数字勿直接对比；同口径内的对比（base vs topk、静态 vs 动态）才有效。

## 5. 瓶颈剖析（当前）

### 5.1 前处理（全量管线口径 09-28 值 ~37.5ms/帧（本轮未复测）；demo 口径 09-29 实测 10.0ms，占 E2E ~44%）

子环节稳态耗时见 §2。当前构成（全量管线）：FOV numba ~30ms（本机 openblas64 单线程小 K 矩阵乘
~81ms/次，numba 规避）+ voxelize（AscendC 已启用，见下）+ pad/collate ~10ms。

- **AscendC voxelize 已启用（2026-09-28）**：`npu_patch._patch_voxelize_ascendc`（unum_ops，commit 6b90b4e）改为
  `import npu_patch` 时直接启用（不再只挂在 `init_patch()`，路由此前一直休眠，生产链路实际跑 numba）；
  2026-09-29 起固定 AscendC 为唯一实现：无回退、无开关，unum_ops/OPP 缺失或运行失败直接报错。
  - demo 全量点（000008，M=7260）实测 generate：**AscendC 21.3ms vs numba 53.4ms（~2.5x）**；
    早前测出的「无提速/更慢」是跨进程噪声 + wrapper `except` 静默回退 numba 假象
    （根因即此回退，09-29 起已彻底去除）。
  - 正确性口径：voxel 逐帧**排序等价**（sorted-equal：coord 多重集相同、同 coord 特征/npp 相同）但**行序与 numba 不同**；
    200 帧中 199 帧 sorted-equal，frame 38 少 1 个越界边界体素（coord z=432 超出 BEV 网格，不影响输出）。
  - 红线通过：**200 帧 OM AP 与 numba 基线逐位一致**（diff=0）；比较/评测链路不受 voxel 行序影响。
  - 副作用：AscendC 会把 torch_npu 设备上下文拉进 aclruntime 进程，自然退出时双运行时 teardown 冲突
    会 segfault/bus error（结果已全部产出）；`om_ref_demo/om_ref_test` 结果输出后调 `npu_patch.hard_exit(0)` 硬退出。
  - kernel 级优化：`MAX_NBLK=7` 曾为单 cube 8 AIV 验证上限；提到 15 实测无提速（21.8 vs 21.3ms）——
    kernel 受 12 次软件栅障（radix 多趟 + 全量 L1 dcci）+ 每调用 aclnn 两段式固定 ~5ms 串行化限制，
    **不是核数限制**。
    （**09-29 修正**：`MAX_NBLK 7→8` 实有收益 kernel 7.11→6.24ms（−12%，物理 8 AIV，unum_ops `a8a101d`），
    稳定 LSD 排序与分块数无关、框表逐字节一致——09-28「nblk>7 改变 voxel 行序」仅适用当时的 v1 实现；>8 仍无提速。）
  - （vendor 管理教训：CANN runtime 加载 `opp/vendors/` 下全部 vendor 的 op_api/opmaster，字母序后者覆盖，
    `config.ini load_priority` 不控制 aclnn 符号解析；回滚 vendor 以 docker overlay2 diff 层为准，
    须 `diff -rq` 核对 + 重验 200 帧 AP。）
  - **当前 demo 口径 prepare_data 10.0ms 构成（09-29）**：mask 0.28 + H2D 0.23 + ext 6.74（kernel 6.24）
    + 管线残差 ~2.4；kernel 6.24ms 为**标量墙**（msprof scalar_ratio 0.97——稳定 LSD 基数排序 +
    bit-exact IEEE fp32 真除法契约），<5ms 需向量化重设计（数值风险，独立迭代，见 §7）。
- 剩余可优化：FOV/voxelize 并行；FOV 仍未 kernel 化（§7「AscendC 融合预处理算子」（方案 A）剩 FOV 部分）。

### 5.2 推理（forward：17.5-22.0ms，视 OM 与 M 口径；09-29 手术后集成态 10.97ms）

- 静态 9000 force_fp16 **17ms**（单帧最快）；静态 18000 19.7ms；动态 18000 21.3ms（base）/ 22.0ms（topk）。
- 动态比静态多 `set_dynamic_shape` ~1.6ms 调度，但免静态总是处理 pad 行的浪费——E2E 上动态更优（见 1.2）。
- op 级分布（fp32 静态 9000 口径）：剩 Conv2DTransposeD 7.3ms + Conv2D 3.6ms + GatherV2 2.9ms；
  fp16 后未重新 profiling。
- 注：早期 P3 规划曾预期 fp16 ~12ms，实测 mixed 18ms / force 17ms，以实测为准。
- **head 图手术 ABCD（09-29，`8b1e6d8`）**：数学恒等变换——A) 3 个 1x1 head conv 合并为 Conv 384→72+Slice；
  B) 冗余类别 Gather 消除（Squeeze 替代，−915us）；C) ConvTranspose(4x4/s4)→Conv1x1(256→2048)+
  DepthToSpace(4)（mode 须写 **'CRD'**，CANN 语义与 ONNX 规范相反，探针实测）；D) TopK 4096→1024。
  同设备对照（device 2）11.32→**9.60ms（−15%）**，输出逐框一致（坐标 0.000977 / score 0.000122）；
  集成态 10.97（device 1，含设备常驻 BaseTensor 的 GE 内部 D2D +1.4，host feeds 为 9.55）。
  ~~赢家 OM：abcd 版~~ → **09-29 复核推翻**：D 的 TopK 1024 < NMS_PRE_MAXSIZE 4096，密集帧
  pre-NMS 候选可能被图内静默截断（bit-exact 仅 000008 单帧验证），且仅比 ABC 快 0.05ms；
  ABCD OM 文件已删除，**推荐/默认统一切到 ABC 版**（TopK 4096，语义与原 topk OM 完全一致）。
- **msprof（基线 topk OM，device 2）**：AICPU 占比 ≈0；Conv 族合计 ~4.8ms / 10.8ms = **44%**
  （Conv2D 2598us + Conv2DTransposeD 1866us）；GatherV2 voxel→BEV scatter 1260us（**mte2 99.6% 单向
  bound**）；TransData ~800us。证伪留档：7 种 ATC 精度/融合开关组合均无增益；全量 ConvTranspose
  pixelshuffle 重写慢 8 倍（DTS(2) 2272us ≫ 原生 273us）。

### 5.3 后处理（base 口径 15-16.5ms；topk 图内口径 3.5-4.0ms）

- base OM：D2H 13MB + CPU sigmoid/topk/NMS，优化已做：
  - **增量贪心旋转 NMS**（`_nms_incremental`，O(N·K) 替代 O(N²)，与全矩阵 bit 一致，1672→43ms 最坏）；
  - numpy `argpartition` topk + `tensor_to_numpy(copy=False)` 免 13MB memcpy + sigmoid 单调性优化。
- topk 图内（P2）：TopK(4096) 移入图内，NMS 只吃 4096 行，D2H 13MB→164KB → **3.5-4.0ms**（见 1/1.4）。
- **09-29 重构后 0.60ms**（含 D2H 0.12）：numpy max/argmax 替代 torch.max（省 ~3.8ms）；torch sigmoid
  **保留**（numpy 版有 1ULP 差，为逐位一致不换）；NMS numba 标量化 0.57→0.21ms；D2H `copy=False`
  host 缓冲直视。逐位一致性验证脚本 `npu/debug/verify_postproc_bitwise.py`（已随 debug/ 清理移除，见 git 历史）。

## 6. 精度

精度评测方法、全量结果与历史修复见 **[PRECISION.md](PRECISION.md)**。AP 快速对照（3D moderate R11）：

| 口径 | OM | AP（Car/Ped/Cyc） |
|---|---|---|
| 200 帧（val 前 200），fp32 | base 静态 9000 | 77.90 / 57.95 / 37.05 |
| 200 帧，fp16 force | base 静态 9000 | 77.81 / 59.86 / 38.32（Ped/Cyc 反升） |
| 200 帧，fp16 force | topk P2（静/动） | 与 base **逐位一致**（77.81/59.86/38.32） |
| 全量 3769 帧，fp32 | base 动态 | 77.25 / 51.67 / 61.76（官方基线 77.28/52.29/62.68） |
| 全量 3769 帧，fp16 | base 动态 18000 / 静态 18000 | 77.07 / 51.93 / 61.95（两种 pad 逐位一致） |

> 200 帧样本（val 前 200 帧，非随机）与全量 AP 差异明显（Cyc 38.32 vs 61.95），跨样本数字勿直接对比；
> topk 的 3769 帧独立复测未见记录（逐位一致性当前依据 200 帧对比，交付全量口径前建议补跑）。

## 7. 优化方向（已做+待办）

| 方向 | 内容 | 预期 | 备注 |
|---|---|---|---|
| ~~force_fp16~~ | 全图 fp16 | 推理 →**17ms** | ✅ 已测：AP 1% 内（77.81/59.86/38.32，Ped/Cyc 反升） |
| ~~topk 图内化~~ | `--topk-only`：图内 ReduceMax+TopK(4096)，NMS 留 CPU | 后处理 15→**3.5ms** | ✅ 已测：E2E 73.1→**64.3ms**（force 基线；77.8 为更早 mixed v1 轮次），AP 与 base 完全一致；全量管线口径 63.3ms（动态 18000，200 帧） |
| ~~NMS 图内化~~ | 图内 sigmoid+TopK+NMS | **无收益且不可行** | ❌ 图内全链使前向 24→47ms，劣于 base 24 + CPU 后处理 16.5 = 40.5ms；且 **310P 的 NonMaxSuppression IoU 抑制失效**（返回全部 max_out=500、无抑制，487 Car 重复 vs CPU 正确 32）；**框数硬限制 ≤50000**（PointPillar 321408 anchors 超限）导致输出完全垃圾。后处理留 CPU 为最优。 |
| AscendC 融合预处理算子（方案 A） | FOV+mask+voxelize 单 AscendC kernel | 前处理 42→~5-10ms | ✅ voxelize 已 kernel 化并启用（2.5x，见 5.1）；剩 FOV 未 kernel 化；暂缓 |
| ~~体素化 561000 修复 + 设备常驻管线~~ | ThreadCtxGuard + numba 单遍 mask + BaseTensor 零拷贝 feeds | getitem 32→**10.0**、feeds 2.4→0.5 | ✅ 09-29（`27bcb48` + unum_ops `387bb9e`/`a8a101d`）；ON/OFF 框表逐字节一致（A/B 22.5 vs 23.7ms） |
| ~~head 图手术 ABCD~~ | 1x1 合并 / 冗余 gather 消除 / ConvTranspose→Conv1x1+DTS / TopK 4096→1024 | forward 11.32→**9.60**（−15%） | ✅ 09-29（`8b1e6d8`），输出逐框一致；7 种 ATC 开关组合证伪无增益、全量 pixelshuffle 重写慢 8 倍；**全量改用 ABC 变体**（D 截断风险，ABCD OM 已移除） |
| ~~collate / 后处理重构~~ | 单帧 fast-path（双态）+ numpy max + numba NMS 标量化 | 5.2→**0.45**、4.0→**0.60** | ✅ 09-29（`5eccc9b`），逐位一致（verify_postproc_bitwise.py） |
| AMCT int8 量化（backbone conv） | conv 占 forward 44%，量化是唯一大幅压缩手段 | forward **−1.5~2ms** | 待办：10ms 路径上收益最大的一项 |
| 自定义 voxel→BEV scatter 算子 | GatherV2 mte2 单向 bound（99.6%），融合写 kernel ~50% 空间 | ~−0.6ms | 待办 |
| 体素化 kernel 向量化重写 | 标量排序墙 6.24ms（bit-exact 契约，见 5.1） | kernel →~4.6ms | 待办：数值风险，独立迭代；先行小步（fill 携数据/S1+hist0 融合/桶批 scatter）可先得 −1.6ms |
| ~~M≥18000 动态/静态 OM~~ | 覆盖全量 val（54% 帧 M>9000） | 全量评测可用 | 动态已转（`fp16_dynamic18000_force` / `fp16_dynamic18000_topk`）；09-29 起推荐 surgery ABC 版 |
| 跨帧流水线 | async D2H 与下帧预处理重叠 | 只提 FPS | 单帧延迟无益 |

### 性能收益预估

> 前两行为 200 帧静态 9000 口径（§1）；全量管线口径当前 63.3ms（1.4，体素化 numba），
> AscendC 在 FOV 小输入上的收益未测。第三行起为 demo 单帧口径（§1.3）。

| 组合 | 前处理 | 推理 | 后处理 | E2E |
|---|---|---|---|---|
| force_fp16（实测） | 41 ms | 17 ms | 15 ms | **73.1 ms** |
| **force_fp16 + topk图内（实测）** | 43 ms | 17.5 ms | 3.5 ms | **64.3 ms** |
| +AscendC 算子预处理（voxelize 已启用，FOV 待 kernel 化） | FOV 剩 ~30ms | 17.5 | 3.5 | 待 FOV kernel 化后 ~40ms |
| **09-29 集成态（实测，demo 单帧口径）** | 10.0 ms | 11.0 ms | 0.6 ms | **22.5 ms** |
| +int8 量化 + scatter 自定义 + kernel 向量化（全部待办落地） | ~8.4 | ~7.5 | 0.6 | **乐观 ~13-15 ms** |

> **10ms 目标结论：当前 fp16 OM 路径不可达**——体素化 kernel 6.24ms 标量墙（bit-exact 契约）+
> backbone conv 占 forward 44%（架构固有）+ scatter 1.26ms（mte2 单向 bound）；三项待办全部落地
> 乐观 ~13-15ms（demo 单帧口径）。全量管线口径（200 帧）待数据集环境复测。

## 8. 测速方法

```bash
# 单帧 demo（含前/后处理，000008 pad 到 9000）——注意 demo 无 FOV，体素化 122555 点
python npu/om_ref_demo.py --om weights/pointpillar_base_fp32_static9000_v2.om

# 单帧完整推理延迟（默认即 surgery ABC OM；09-29 集成态实测 22.5ms，30 iters 中位，ABCD 口径）
python npu/om_ref_demo.py --data_path <bin 或目录>

# E2E 分段计时（200 帧，输出 前处理/推理/后处理 拆分 + 官方 AP）——全量管线口径（有 FOV）
python npu/om_ref_test.py --frames 200

# 只看简化计时（跳过官方评测）
python npu/om_ref_test.py --frames 50 --quick
```

> E2E 九段计时脚本（BENCH_OM/BENCH_DEVICE/BENCH_ITERS 环境变量）已随 `npu/debug/` 清理移除，
> 需要时从 git 历史取回（`852cd8b` 及之前）。

> 注：`--frames N` 跑 val **前 N 帧**并在该子集上算 AP；静态 OM 对 M 超限的帧直接跳过
> （跳过帧按零检测计入评测，会拉垮 AP）。本机 `weights/` 现仅存 topk 动态 OM，其余 OM 需在转换机准备。
