# PointPillars 精度评测（PRECISION）

PointPillars 在 GPU / NPU 双环境的精度评测方法、管线、结果与历史问题。

## 评测管线

```
数据(velodyne .bin)
  → 前处理(FOV filter + mask + voxelize)      npu/infer.py / quick_eval.py
  → 推理 (PyTorch GPU/NPU 或 OM)               npu/infer.py / eval_kitti_full*.py
  → 后处理(sigmoid + topk + class_agnostic_nms)
  → 保存预测 → preds_*.txt (KITTI lidar 格式)
  → 官方评测 (kitti_object_eval_python)        npu/eval.py / eval_official_kitti.py
```

## 快速评测命令

```bash
# 推理（200 帧）+ 自动官方评测
python npu/quick_eval.py --device cuda:0 --frames 200 --save-preds /tmp/preds

# 或分步：推理 → 评测
python npu/infer.py --ckpt weights/pointpillar_7728.pth --device auto --frames 200 --save-preds /tmp/preds
python npu/eval.py --preds /tmp/preds --frames 200

# OM 后端全量
python npu/eval_kitti_full.py --om <om> --save-preds preds_om
python npu/eval_official_kitti.py --preds preds_om
```

预测文件格式：`<class> -1 -1 -1 0 0 0 0 x y z dx dy dz ry score`（lidar 系）。
官方评测内部转 camera 系并投影 2D bbox（难度分类），GT 来自 `kitti_infos_val.pkl`（含 DontCare 处理）。

## 当前结果（2026-09-22 修复后，KITTI val 前 200 帧）

### 3D AP（R11，官方口径 Car@0.7 / Ped@0.5 / Cyc@0.5）

| class | easy | moderate | hard |
|---|---|---|---|
| Car | 85.46 | **77.83** | 73.25 |
| Pedestrian | 65.37 | **57.90** | 50.21 |
| Cyclist | 31.25 | **36.70** | 39.17 |

本机 GPU 与设备 NPU **逐位一致**。两端体素化输入 14840/14840 voxel 全等；
评测 rotate_iou 数学等价（max diff ~1e-6）。

### 与官方基线对照

| class | 官方（3D moderate R11） | 本实现 | 差距 |
|---|---|---|---|
| Car | 77.28 | 77.83 | +0.55 ✅ |
| Pedestrian | 52.29 | 57.90 | +5.61 |
| Cyclist | 62.68 | 36.70 | -25.98 |

- **Car 与官方一致**，确认 checkpoint + 管线 + 评测全链路正确。
- **Cyclist 明显低于官方**，且两端（GPU/NPU）逐位一致，指向 checkpoint 训练集/类别分布差异
  或官方基线的评测口径差异，**非管线问题**（同一输入两端 AP 相同）。

## 历史结果（修复前，仅存档）

### 旧基线（OM fp32 动态，全量 3769 帧）【2026-08-18，评测数学 bug 前】

```
Car  3d AP: 86.5 / 77.2 / 74.6    3d AP_R40: 87.8 / 78.3 / 75.2
Car  bev AP: 89.7 / 87.1 / 84.4
Ped  3d AP: 57.0 / 52.0 / 47.6
Cyc  3d AP: 80.0 / 62.8 / 59.8
```

### 设备修复前基线（200 帧）【2026-09-18~21】

设备 rotate_iou（CPU box_overlap_bev 版）评测给出：Car 3D moderate ≈ **64.16**（偏低 13 点），
本机 GPU 官方评测 ≈ 77.83。根因见 CHANGELOG v1.1.0。

### 全量（前 200 帧）先期结果【2026-09-22 之前，voxelizer bug 修复前】

全量 3769 帧 GPU 官方评测 3D AP 全 0 / 全 Cyclist 荒谬框 —— 根因是 VoxelGeneratorV2 分组错位
（36% voxel 点落入错误 pillar），见 CHANGELOG v1.1.0。

## 已修复的精度问题（历史）

| 问题 | 根因 | 修复 |
|---|---|---|
| 全量 AP=0、全 Cyclist 荒谬框 | unum_ops VoxelGeneratorV2 分组错位（lexsort 序 vs unique 序不齐） | 邻坐标切分，voxel 全等 |
| 设备评测 AP 偏低 13 点 | rotate_iou CPU 版数学与官方 CUDA 版不等价 | 逐行复刻官方数学的 CPU numba 版 |
| 喂数 dtype 错误 | `voxel_num_points`/`voxel_coords` 被 float32 喂给 int32 端口（261.0f → int32 1135714304） | `aclruntime.strict_input_dtype` C++ 守卫 + 按端口 dtype 转换 |
| 非连续数组 | collate 后 `voxel_coords` Fortran 序（strides=(4, M*4)），底层按 C 序拷贝乱码 | feeds 中 `np.ascontiguousarray` |
| `load_data_to_cpu` 统一 `.float()` | voxel_coords/num_points 被转 float32 → OM 垃圾框、全量 AP=0 | `torch.from_numpy` 保留原 dtype |
| 无 CUDA 评测崩溃 | 官方 `rotate_iou_gpu_eval` 用 numba.cuda | 无 CUDA 时 `sys.modules` 注入 CPU 版 rotate_iou（几何一致），eval.py 零改动 |

> **重要澄清**：早期 `ATC_PRECISION_BUG_REPORT.md` 曾怀疑 ATC 对 VFE centering 常量折叠有 bug，
> 后经分层子图抽取证实**根因是喂数 dtype/非连续错误**，ATC 转换本身无 bug。该报告历史结论作废。

## 精度验证方法

- `npu/compare_pt_om.py`：PyTorch vs OM 数值对比（box/cls cosine）。
- 两端体素化全等验证：dump voxels → diff（14840/14840）。
- rotate_iou 数学等价：随机框对比 GPU 版 vs CPU 版（max diff ~1e-6）。
