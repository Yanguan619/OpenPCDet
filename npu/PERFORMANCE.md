# PointPillars 性能分析（PERFORMANCE）

PointPillars 在 NPU（Ascend 310P3）上的性能数据、瓶颈分析与优化方向。
**当前最新：E2E 73.1ms/帧（force_fp16，<100ms 达标），200 帧 AP 全部在基线 1% 容差内（Car 甚至持平）。**

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
| voxelize | **numba 核心循环**（分组+填充，与 numpy 输出逐位一致） | ~16 ms |
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

## 5. 瓶颈剖析（当前）

### 5.1 前处理（42ms，占 E2E 51%）
- FOV numba 内核 ~30ms（本机 openblas64 单线程小 K 矩阵乘 ~81ms/次，numba 规避）。
- voxelize numba ~16ms（与 numpy 输出逐位一致，BEV diff=0）。
- pad 复用 ~10ms。
- 剩余可优化：FOV/voxelize 并行或 **Ascend 融合预处理算子（方案 A）**。

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
| **~~P2 topk 图内化~~** | `--topk-only`：图内 ReduceMax+TopK(4096)，NMS 留 CPU | 后处理 15→**3.5ms** | ✅ 已测：E2E 77.8→**64.3ms**，AP 与 base 完全一致 |
| **Ascend 融合预处理算子（方案 A）** | FOV+mask+voxelize 单 AscendC kernel | 前处理 42→~5-10ms | 最可靠压 E2E；需写 kernel |
| **M≥18000 动态/静态 OM** | 覆盖全量 val（54% 帧 M>9000） | 全量评测可用 | 动态已转（`fp16_dynamic18000_force`） |
| 跨帧流水线 | async D2H 与下帧预处理重叠 | 只提 FPS | 单帧延迟无益 |

### 性能收益预估

| 组合 | 前处理 | 推理 | 后处理 | E2E |
|---|---|---|---|---|
| 现状 fp32（实测） | 42 ms | 24 ms | 16.5 ms | **82.8 ms** |
| force_fp16（实测） | 41 ms | 17 ms | 15 ms | **73.1 ms** |
| **force_fp16 + topk图内（实测）** | 43 ms | 17.5 ms | 3.5 ms | **64.3 ms** |
| +Ascend 融合预处理 | 5-10 | 17.5 | 3.5 | **~28 ms** |

## 8. 测速方法

```bash
# 单帧 demo（含前/后处理，000008 pad 到 9000）
python npu/om_ref_demo.py --om weights/pointpillar_base_fp32_static9000_v2.om

# E2E 分段计时（200 帧，输出 前处理/推理/后处理 拆分 + 官方 AP）
python npu/om_ref_test.py --om weights/pointpillar_base_fp32_static9000_v2.om --frames 200

# 只看简化计时（跳过官方评测）
python npu/om_ref_test.py --om weights/pointpillar_base_fp32_static9000_v2.om --frames 50 --quick
```

numba 首调（~1.5s）已在脚本内预热移出计时。
