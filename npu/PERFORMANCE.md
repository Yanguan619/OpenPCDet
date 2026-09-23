# PointPillars 性能分析（PERFORMANCE）

PointPillars 在 NPU（Ascend 310P3）上的性能数据、瓶颈分析与优化方向。

## 1. E2E 延迟现状（单帧，000008.bin）

| 阶段 | 实现 | 耗时 | 占比 |
|---|---|---|---|
| 前处理（读bin + FOV + mask + voxelize + index_map） | CPU numpy/torch | 187.7 ms | 80.4% |
| OM 推理（NPU 前向） | NPU (dyn9000 fp16 OM) | 22.0 ms | 9.4% |
| 后处理（sigmoid + topk + NMS） | CPU torch + numba | 23.8 ms | 10.2% |
| **E2E 总计** | | **233.5 ms** | 100% |

> **瓶颈在 host 侧 CPU 前处理（80%），NPU 推理仅占 ~9%** —— 流水严重倒挂
> （CPU 前处理约为 NPU 推理的 8.5 倍）。PyTorch 后端（非 OM）整帧 ~273 ms/帧（200 帧平均）。

## 2. 前处理内部拆分（000008，5 次均值）

| 子阶段 | 实现 | 耗时 | 前处理占比 |
|---|---|---|---|
| 读 bin | `np.fromfile` | ~3 ms | 1% |
| **FOV 过滤** | numpy 两次 `np.dot` + 比较 | **130~210 ms**（均值 ~150-175） | **~68%** |
| FOV 索引选择 | `points[fov]` | ~2 ms | 1% |
| range mask | `np.logical_and.reduce` | ~1-2 ms | 1% |
| **voxelize**（已向量化） | numpy `VoxelGeneratorV2.generate` | **66~80 ms** | **~31%** |
| collate + to_tensor + build_index_map | numpy/torch | ~1 ms | <1% |

## 3. OM 推理对比

### 三种 OM

| OM | Shape | Precision | 单帧前向 | 说明 |
|---|---|---|---|---|
| `pointpillar_mixed_dynamic` | 动态 (M=1~9000) | mixed_float16 | ~135 ms | 全量 val 评测用（每帧 set_dynamic_shape） |
| `pointpillar_fp32_dynamic` | 动态 (M=1~9000) | force_fp32 | ~134.9 ms | 全量 val 评测用（精度最高） |
| `pointpillar_fp16_static` | 静态 (M=3941) | force_fp16 | **15.4 ms** | 单帧 demo 用（延迟最优） |
| `pointpillar_mixed_float16_opt` | 静态 | mixed_float16+优化 | **15.7 ms** | 24/24 精度 |

### 优化链（累计 419.8 → 15.7 ms，26×）

| 版本 | 优化 | OM 推理 |
|---|---|---|
| V1 | 基线（force_fp32） | 419.8 ms |
| V2 | 图手术：Cast 链合并 + Head Conv 合并 + ATC 融合全开 | 150.5 ms |
| V3 | ScatterND → Gather（bev_index_map 第 4 输入） | 47.3 ms |
| V4 | ArgMax → `(x1>x0)` 比较 | 27.7 ms |
| V5 | onnxsim（无性能收益） | 27.8 ms |
| V6 | VFE BN1d → 逐通道 affine | 23.6 ms |
| **最终** | **mixed_float16 + mix_optimized** | **15.7 ms** |

详细原理见 `CHANGELOG.md` v1.0.1。

## 4. 全量 val 集（3769 帧）耗时

| 推理后端 | avg/帧 | 总耗时 |
|---|---|---|
| mixed_float16 OM | 302.6 ms | ~19 min |
| fp32 OM | 298.6 ms | ~19 min |
| PyTorch 原模型 (NPU) | 312.9 ms | ~20 min |

> 全量评测期间每帧需 `set_dynamic_shape`（动态 OM），且 host 前处理占大头，三种后端总耗时接近。

## 5. 瓶颈剖析

### 5.1 FOV 过滤（最大单项，~150ms）
`get_fov_flag`：`points_hom @ (V2C.T@R0.T)` → `points_rect_hom @ P2.T` → 4 次比较+逻辑与。
- 归因：numpy 小内维（3/4）GEMM 效率低 + 多趟内存拷贝。
- CPU 优化：预融合单一矩阵乘 `proj = pts@(V2C.T@R0.T)@P2.T[:3] + P2.T[3]` → 166.6→66.4 ms（2.5x）。
  ⚠️ 有 ~583 边界点舍入差异（0.48%），非逐位一致，需精度复核。

### 5.2 voxelize（66~80ms）
已向量化（`np.unique(return_counts)` + 一次性 scatter，原始 586ms → 67ms）。进一步 CPU 优化空间 ~20-30%。

### 5.3 后处理（~24ms）
`sigmoid+topk` 2-4ms；`class_agnostic_nms`（numba 旋转 BEV IoU）稳态 6~31ms。
NMS 数据依赖、串行裁剪多，且需把 321408×7≈9MB 框从 NPU 回传再 H2D，上 NPU 净收益为负。

### 5.4 OM 推理（15.7~22ms）
已全量在 NPU。动态 OM 比静态慢 ~8x（每帧 set_dynamic_shape + 算子编译/调度）。

## 6. Ascend 算子必要性评估

| 方案 | 内容 | 预估收益 | 结论 |
|---|---|---|---|
| **A（推荐）** | **FOV + range mask + voxelize 融合单 kernel**（输入原始点云 → voxels/coords/num_points） | 前处理 ~220ms → **5-10ms** | **值得写**。voxelization kernel 已开发（`ascend_ops/voxelization_direct/`，CPU 仿真 7 帧验证通过） |
| B | 仅 voxelize 单算子 | 省 ~66ms | 过渡方案 |
| C | 后处理 NMS 上 NPU | 负收益（9MB 重传） | 不建议 |
| D | OM 内部 VFE/Head 融合算子 | ~2ms | 优先级低 |

### 性能收益预估

| 方案 | 前处理 | OM 推理 | 后处理 | E2E | 相对现状 |
|---|---|---|---|---|---|
| 现状（实测） | 187.7 ms | 22.0 ms | 23.8 ms | **233.5 ms** | 1.0x |
| 仅 CPU 优化（FOV 融合） | ~110-130 ms | 22 ms | ~20 ms | **~155-175 ms** | ~1.4x |
| **Ascend 方案 A** | **5-10 ms** | 22 ms | ~20 ms | **~45-55 ms** | **~4.5x** |

## 7. 优化方向（待办）

1. **前处理并行化**：`multiprocessing.Pool` 把多帧 voxelization 分散到多核（当前单线程串行）。
2. **Ascend 前处理融合算子（方案 A）**：落地顺序 → ① 上板实测现有 voxelization kernel → ② kernel 向量化调优 → ③ 扩展 FOV+mask 为融合算子 → ④ ONNX-CPU 精度复核门禁 → ⑤ 接入 om_ref_demo。
3. **静态 shape**：固定 M 可省 `set_dynamic_shape` 开销，但仅适用于非空 pillar 数恒定的场景。
4. **后处理**：sigmoid + topk + NMS 纯 numpy/numba 化，减少张量往返。

## 8. 测速方法

```bash
# OM 单帧（分段计时，--num-iters 取平均）
python npu/om_ref_demo.py --bin data/kitti/training/velodyne/000008.bin \
    --om weights/pointpillar_fp16_static.om --num-iters 10

# E2E 性能测试
python npu/perf_e2e.py --om <om> --bin data/kitti/training/velodyne/000008.bin --num-iters 20
```

`--num-iters` 控制重复次数取平均，建议 10 次以上。numba 首调 ~1.5s 已移出计时。
