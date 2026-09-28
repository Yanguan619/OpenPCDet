# PointPillars 性能分析（PERFORMANCE）

PointPillars 在 NPU（Ascend 310P3）上的性能数据、瓶颈分析与优化方向。
**当前最新：单帧 demo 完整推理 ~67ms（fp16 force + topk 图内 + AscendC 体素化，无 FOV，静机待重测）；数据集全量 E2E 63.3ms/帧；AP 与 base 逐位一致（Car 77.81 / Ped 59.86 / Cyc 38.32）。**

## 1. E2E 延迟现状（200 帧稳态）

| 阶段 | 实现 | fp32 | fp16(force) | **fp16+topk图内(P2)** |
|---|---|---|---|---|
| 前处理（FOV + voxelize + pad + 读图尺寸） | CPU numpy/numba | 42 ms | 41 ms | 43 ms |
| OM 推理（NPU 前向，静态 M=9000） | NPU | 24 ms | 17 ms | 17.5 ms |
| 后处理（sigmoid + topk + NMS） | CPU torch + numba + numpy | 16.5 ms | 15 ms | **3.5 ms** |
| **E2E 总计** | | **82.8 ms** | **73.1 ms** | **64.3 ms** |

> fp32: `pointpillar_base_fp32_static9000_v2.om`；force: `pointpillar_base_fp16_static9000_force.om`；
> **topk(P2)**: `pp_topk_static9000`（图内 ReduceMax+TopK(4096)，输出 top-K box/cls ~160KB，NMS 留 CPU）。
> 实测命令：`python npu/om_ref_test.py --om <om> --frames 200`
> 基线 369ms（2026-09-22 全量）→ topk(P2) 64.3ms，累计 **5.7x**。
> **P2 精度与 base 完全一致**（top-K 按 raw cls 单调等价 sigmoid 排序）：Car 77.81 / Ped 59.86 / Cyc 38.32。
> 导出：`python npu/export_onnx.py --topk-only --skip-export --base-output <base> --output <topk>`

### 1.1 静态 pad-到-18000 fp16 的 E2E 进一步拆分（全量 val 3769 帧）

全量 val 结构（用户机，avg 71.2ms/帧）：**前处理 39.6 + 推理 19.7 + 后处理 11.9**。

本机子环节逐段实测（每帧 avg，弱 CPU，用于相对结构）：

| 子环节 | 本机耗时 | 占比 | 说明 |
|---|---|---|---|
| **FOV numba** | 50.9 ms | 37% | 最大 CPU 单项 |
| 后处理 sigmoid + D2H | 19.8 ms | 14% | 13MB D2H + torch.sigmoid/max |
| 后处理 NMS（增量） | 23.3 ms | 17% | top-4096 |
| **推理 forward** | 19.8 ms | 14% | 静态 18000，**总是处理 18000 行** |
| voxelize numba | 17.8 ms | 13% | |
| pad→18000 | 4.2 ms | 3% | 每帧 pad |
| 读 bin | 1.5 ms | 1% | |

> 静态 pad 的**推理浪费**在于：M 平均 ~8000 的帧也处理 18000 行 VFE（推理 19.7ms 固定，与 M 无关）；
> 前处理 pad 开销仅 4.2ms，不是主因。→ 见 1.2 动态对比。

### 1.2 动态 vs 静态 pad（全量 val 3769 帧，fp16）

| 指标 | 动态 fp16（1~18000） | 静态 18000（pad） |
|---|---|---|
| E2E avg | **67.5 ms** | 71.2 ms |
| 前处理 | 34.5 ms | 39.6 ms（+pad→18000 ~5ms） |
| 推理 | 21.3 ms | 19.7 ms（省动态调度） |
| 后处理 | 11.8 ms | 11.9 ms |
| AP（Car/Ped/Cyc） | 77.07/51.93/61.95 | 77.07/51.93/61.95（相同） |

**结论：动态 OM 更优**（净胜 ~3.5ms）——静态推理虽省 `set_dynamic_shape` 1.6ms，但静态总是处理 18000 行的浪费 + pad 开销反超；
且静态 pad 与动态预测**逐位一致**（pad 行不参与 scatter），全量 AP 完全相同。
**全量 val 推荐动态 fp16 om**（`pointpillar_base_fp16_dynamic18000_force_linux_aarch64.om`）。

### 1.3 单帧 demo 完整推理延迟（om_ref_demo，无 FOV 过滤）

> 与 1.1/1.2 不同：demo 路径**不做 FOV 过滤**（与 `tools/demo.py` 一致），直接体素化 122555 原始点。
> 以下为单帧完整 E2E（含 `__getitem__` 体素化），本机 000008，avg 10。

| 阶段 | base fp16 static18000 | base fp16 dynamic18000 | **topk fp16 dynamic18000** |
|---|---|---|---|
| getitem（读 bin + 体素化 122555 点） | ~34 ms（46%） | ~34 ms（48%） | ~32 ms（48%） |
| collate + to_tensor | 4.6 ms | 4.8 ms | 4.5 ms |
| pad / index_map | 4.5 ms | 0.7 ms | 0.7 ms |
| feeds（aclruntime.Tensor） | 4.8 ms | 2.4 ms | 2.4 ms |
| **forward** | 19.7 ms | 21.5 ms | 22.8 ms |
| postproc（sigmoid + topk + NMS） | 13.4 ms | 14.3 ms | **4.0 ms** |
| **单帧完整 E2E** | ~81 ms | ~78 ms | ~67 ms（est.） |

> ⚠️ 上表 getitem/E2E 为 **2026-09-28 体素化切 AscendC 后重算的估计值**（表内其它阶段仍为原 avg10 实测）：
> 体素化 generate 000008（M=7260）实测 AscendC ~25ms vs numba ~45-53ms（**~2.5x**），
> getitem 由 52.6ms 降 ~20ms（测于 CPU 负载 20 的机器，静机待重测，见 5.1）。
> 三列均为真实存在 OM、同帧（000008，M=7260）同流程逐段计时。
> **结论按同模式比，别跨模式**（三列同时混了「topk 图内」与「static→dynamic」两个变量）：
> - **topk 图内（dynamic vs dynamic）**：省 ~11ms = postproc **-10.3**（TopK(4096) 后 NMS 只吃 4096 行）
>   + forward +1.3（图内多一次 TopK）+ getitem 噪声 ~-2；
> - **static→dynamic（同 base）**：再省 ~3.7ms = pad -3.8 + feeds -2.4，forward +1.8 反噬；
> - 两变量叠加即「static base → topk dynamic」共省 ~14.9ms——**勿全记在 topk 头上**。
> - **三种口径的区别**（务必区分）：
> -   **单帧完整**（demo，无 FOV）：~67ms（topk，est.）——含体素化（AscendC 2.5x），`getitem` 占 ~48%；
>   - **推理链路**（不含体素化）：~34ms（collate 4.5 + pad 0.7 + feeds 2.4 + forward 22.8 + postproc 4.0）；
>   - **数据集全量**（test，有 FOV 过滤到 17221 点）：63.3ms/帧——FOV 让体素化输入减 7 倍，
>     虽然多一步 FOV，但体素化省更多 → 反而比 demo 快。
> - **`OM inference time` 只计 forward 段**，不代表整帧。

### 1.4 topk 图内 OM（P2）全量口径验证（动态 18000，200 帧）

| 指标 | base fp16 dynamic18000 | **topk fp16 dynamic18000** |
|---|---|---|
| 前处理（FOV + voxelize） | ~34.5 ms | 37.5 ms |
| 推理 | ~21.3 ms | 22.0 ms |
| **后处理** | ~11.8 ms | **4.0 ms** |
| **E2E avg** | ~67.5 ms | **63.3 ms** |
| AP（Car/Ped/Cyc） | 77.07/51.93/61.95 | 77.07/51.93/61.95（**逐位一致**） |

> topk OM 输出 `topk_boxes (1,4096,7)` / `topk_cls (1,4096,3)`（图内 ReduceMax+TopK(4096)，
> 无 ArgMax/NMS），D2H 13MB→164KB；`om_ref_demo/test` 的 `base_mode` 自动适配（无需改码）。
> 转换命令见下方；`skipped_M=0` 全跑通。

## 1b. 精度（200 帧 3D moderate R11，1% 容差）

| 类 | fp32 基线 | fp16(mixed) | fp16(force) |
|---|---|---|---|
| Car | 77.90 | 77.76 (-0.14 ✅) | **77.81 (-0.09 ✅)** |
| Pedestrian | 57.95 | 57.68 (-0.27 ✅) | **59.86 (+1.91 ✅)** |
| Cyclist | 37.05 | 37.42 (+0.37 ✅) | **38.32 (+1.27 ✅)** |

> force_fp16 的 Ped/Cyclist 反而高于 fp32 基线（fp16 数值波动恰好对齐部分边界），全部满足 1% 容差。

## 2. 前处理内部拆分（000008，稳态）

| 子阶段 | 实现 | 耗时 |
|---|---|---|
| 读 bin | `np.fromfile` | ~1 ms |
| FOV 过滤 | **numba 单内核 `_fov_filter_numba`**（乘加顺序与 `_mm3` 一致）+ 逐元素列式 | ~30 ms |
| 读图像尺寸 | **PIL 只解析 PNG 头**（0.2ms，替代 io.imread 49ms） | <1 ms |
| voxelize | **AscendC kernel（unum_ops，已启用）**；numba 核心循环为关闭时的回退 | ~21-25 ms（000008 全量点；原 numba ~45-53ms，2.5x） |
| pad（静态 9000） | 模块级缓冲复用（免每帧重分配） | ~10 ms |
| collate + ascontiguous + Tensor | numpy/torch | ~3 ms |

## 3. OM 推理对比

### 关键 OM（当前最佳为静态 9000 fp32）

| OM | Shape | fp32 前向 | fp16 前向 | 说明 |
|---|---|---|---|---|
| `pointpillar_base_fp32_static9000_v2.om` | 静态 M=9000 | **24 ms** | - | fp32 基线（ScatterND/ArgMax 消除） |
| `pointpillar_base_fp16_static9000_v2.om` | 静态 M=9000 | - | **18 ms** | mixed_float16 + mixlist，AP 1% 内 |
| `pointpillar_base_fp16_static9000_force.om` | 静态 M=9000 | - | **17 ms** | **force_fp16 全图**，AP 1% 内（最优） |
| `pointpillar_base_fp32_dynamic_linux_aarch64_linux_aarch64.om` | 动态 M=1~9000 | ~173 ms | - | 全量 val 动态（旧，ScatterND 未除） |
| `pointpillar_base_fp16_dynamic18000_force_linux_aarch64.om` | 动态 M=1~18000 | - | 21.3 ms | 全量 val 推荐（动态，无 pad 浪费） |
| `pointpillar_base_fp16_static18000_force.om` | 静态 M=18000 | - | 19.7 ms | 全量 val 兼容（18000 覆盖所有帧，但总有 pad 浪费） |
| `pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om` | 动态 M=1~18000（图内 TopK 4096） | - | 22.0 ms | **P2 topk 图内**，后处理 11.8→4ms，AP 与 base 逐位一致 |

### 前向优化链（186ms → 24ms，7.8x）

| 版本 | 优化 | 前向 |
|---|---|---|
| v2 原始（动态 fp32） | 含 head ScatterND + ArgMaxD | ~186 ms |
| + ScatterND→Slice+Concat（`KnowledgeScatterNdToConcat`） | 方向角修正 setitem 回归修复 | ~42.7 ms |
| + ArgMax2→Greater+Cast（`KnowledgeArgMax2ToCompare`）+ `fix_dir_reshape_dim` | dir_labels 消除 | ~23.7 ms |
| + 静态 M=9000（P0 padding） | 省动态调度 | **24 ms**（静态） |

- 图手术 knowledge 实现在 `/data/workspace/msit/onnx_optimizer/`（`KnowledgeScatterNdToConcat` / `KnowledgeArgMax2ToCompare`），结果 bit 一致（onnxruntime diff=0）。

## 4. 全量 val 集（3769 帧）注意

- **静态 9000 OM 只覆盖 M≤9000 的帧**：抽样 100 帧 val，54% 帧 M>9000（最大 ~16664）→ 全量 val 需
  **M≥18000 静态 OM** 或 **动态 OM（range 加大）**。转换命令已备于 `npu/convert_fp16_static9000.sh`。
- 全量 val 200 帧基准 AP（静态 9000）：Car 77.90 / Ped 57.95 / Cyc 37.05（与官方基线 1% 内）。

**topk OM（P2）转换命令**（onnx：`weights/pointpillar_nms_base_v2_dynamic_topk.onnx`）：
```bash
# 动态 18000 force_fp16（全量 val 推荐）
atc --model=weights/pointpillar_nms_base_v2_dynamic_topk.onnx --framework=5 \
    --soc_version=Ascend310P3 --output=weights/pointpillar_base_fp16_dynamic18000_topk \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"
# 静态 18000 版同理（M 固定 18000，勿加 --dynamic_dims）
```

## 5. 瓶颈剖析（当前）

### 5.1 前处理（42ms，占 E2E 51%）
- FOV numba 内核 ~30ms（本机 openblas64 单线程小 K 矩阵乘 ~81ms/次，numba 规避）。
- voxelize numba ~16ms（FOV 后小输入；demo 无 FOV 全量点时 ~53ms）。
- pad 复用 ~10ms。
- **AscendC voxelize 已启用（2026-09-28）**：`npu_patch._patch_voxelize_ascendc`（unum_ops，commit 6b90b4e）改为
  `import npu_patch` 时直接启用（不再只挂在 `init_patch()`，路由此前一直休眠，生产链路实际跑 numba）；
  `NPU_ASCENDC_VOXELIZE=0` 可关闭回退 numba。
  - demo 全量点（000008，M=7260）实测 generate：**AscendC 21.3ms vs numba 53.4ms（2.5x）**；
    早前测出的「无提速/更慢」是跨进程噪声 + wrapper `except` 静默回退 numba 假象（现已加回退日志，不再静默）。
  - 正确性口径：voxel 逐帧**排序等价**（sorted-equal：coord 多重集相同、同 coord 特征/npp 相同）但**行序与 numba 不同**；
    200 帧中 199 帧 sorted-equal，frame 38 少 1 个越界边界体素（coord z=432 超出 BEV 网格，不影响输出）。
  - 红线通过：**200 帧 OM AP 与 numba 基线逐位一致**（diff=0）；比较/评测链路不受 voxel 行序影响。
  - 副作用：AscendC 会把 torch_npu 设备上下文拉进 aclruntime 进程，自然退出时双运行时 teardown 冲突
    会 segfault/bus error（结果已全部产出）；`om_ref_demo/om_ref_test` 结果输出后调 `npu_patch.hard_exit(0)` 硬退出。
  - kernel 级优化**已实验为死路（2026-09-28）**：`MAX_NBLK=7` 是单 cube 8 AIV 验证上限；worktree 里把
    `MAX_NBLK` 提到 15 重建安装实测**无提速**（21.8 vs 21.3ms）——kernel 受 12 次软件栅障（radix 多趟 + 全量
    L1 dcci）+ 每调用 aclnn 两段式固定 ~5ms 串行化限制，**不是核数限制**；且 nblk>7 会改变 voxel 行序。
    （顺带教训：CANN runtime 加载 `opp/vendors/` 下全部 vendor 的 op_api/opmaster，字母序后者覆盖，
    `config.ini load_priority` 不控制 aclnn 符号解析；回滚 vendor 以 docker overlay2 diff 层为准，
    须 `diff -rq` 核对 + 重验 200 帧 AP。）
- 剩余可优化：FOV/voxelize 并行；FOV 仍未 kernel 化（方案 A 剩 FOV 部分）。

### 5.2 推理（24ms）
- 全量 NPU，fp32。剩 Conv2DTransposeD 7.3ms + Conv2D 3.6ms + GatherV2 2.9ms。
- fp16/mixed 静态 OM 预期 ~12ms（需 AP 门禁）。

### 5.3 后处理（16.5ms）
- **增量贪心旋转 NMS**（`_nms_incremental`，O(N·K) 替代 O(N²)，与全矩阵 bit 一致，1672→43ms 最坏）。
- numpy `argpartition` topk + `tensor_to_numpy(copy=False)` 免 13MB memcpy + sigmoid 单调性优化。

## 6. NMS 融合 OM 的精确结论（不可行，代码级原因）

对照 `cann/ops-cv/non_max_suppression_v6`（仅开源 aclnn 接口，Ascend C kernel 闭源）：

1. **框数硬限制 ≤50000**：PointPillar 321408 anchors 超限 → 输出完全垃圾。图手术加 pre-TopK(4096) 后**选择正确**。
2. **310P 的 NonMaxSuppression IoU 抑制失效**：返回全部 max_out=500、无抑制（487 Car 重复 vs CPU 正确 32），
   kernel 闭源无法修复。
3. **无收益**：图内 sigmoid+TopK+NMS 使前向 24→47ms，劣于 base 24ms + CPU 后处理 16.5ms = 40.5ms。

→ 后处理留 CPU（16.5ms）为最优；如需减 D2H 可把 sigmoid+TopK 放图内输出 top-4096（~140KB vs 13MB），NMS 留 Python。

## 7. 优化方向（待办）

| 方向 | 内容 | 预期 | 备注 |
|---|---|---|---|
| ~~fp16/mixed 静态 9000 OM~~ | `convert_fp16_static9000.sh` + mixlist | 推理 24→**18ms** | ✅ 已测：AP 1% 内（77.76/57.68/37.42） |
| ~~force_fp16~~ | 全图 fp16 | 推理 →**17ms** | ✅ 已测：AP 1% 内（77.81/59.86/38.32，Ped/Cyc 反升） |
| **~~P2 topk 图内化~~** | `--topk-only`：图内 ReduceMax+TopK(4096)，NMS 留 CPU | 后处理 15→**3.5ms** | ✅ 已测：E2E 77.8→**64.3ms**，AP 与 base 完全一致；全量口径 63.3ms（动态 18000） |
| ~~Ascend 融合预处理算子（方案 A）~~ | FOV+mask+voxelize 单 AscendC kernel | 前处理 42→~5-10ms | ✅ voxelize 已 kernel 化并启用（2.5x，见 5.1）；剩 FOV 未 kernel 化；暂缓 |
| **M≥18000 动态/静态 OM** | 覆盖全量 val（54% 帧 M>9000） | 全量评测可用 | 动态已转（`fp16_dynamic18000_force` / `fp16_dynamic18000_topk`） |
| 跨帧流水线 | async D2H 与下帧预处理重叠 | 只提 FPS | 单帧延迟无益 |

### 性能收益预估

| 组合 | 前处理 | 推理 | 后处理 | E2E |
|---|---|---|---|---|
| 现状 fp32（实测） | 42 ms | 24 ms | 16.5 ms | **82.8 ms** |
| force_fp16（实测） | 41 ms | 17 ms | 15 ms | **73.1 ms** |
| **force_fp16 + topk图内（实测）** | 43 ms | 17.5 ms | 3.5 ms | **64.3 ms** |
| +Ascend 预处理（voxelize 已启用，FOV 待 kernel 化） | FOV 剩 ~30ms | 17.5 | 3.5 | 待 FOV kernel 化后 ~40ms |

## 8. 测速方法

```bash
# 单帧 demo（含前/后处理，000008 pad 到 9000）——注意 demo 无 FOV，体素化 122555 点
python npu/om_ref_demo.py --om weights/pointpillar_base_fp32_static9000_v2.om

# 单帧完整推理延迟（topk 动态，~87ms；含 getitem 体素化）
python npu/om_ref_demo.py --om weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om

# E2E 分段计时（200 帧，输出 前处理/推理/后处理 拆分 + 官方 AP）——有 FOV 过滤口径
python npu/om_ref_test.py --om weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om --frames 200

# 只看简化计时（跳过官方评测）
python npu/om_ref_test.py --om weights/pointpillar_base_fp32_static9000_v2.om --frames 50 --quick
```

numba 首调（~1.5s）已在脚本内预热移出计时。
