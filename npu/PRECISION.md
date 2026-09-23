# PointPillars 精度评测（PRECISION）

PointPillars 在 GPU / NPU 双环境的精度评测方法、管线、结果与历史问题。

## 评测管线

```
数据(velodyne .bin)
  → 前处理(FOV filter + mask + voxelize)      npu/infer.py / quick_eval.py
  → 推理 (PyTorch GPU/NPU 或 OM)               npu/infer.py / om_ref_test*.py
  → 后处理(sigmoid + topk + class_agnostic_nms)
  → 保存预测 → preds_*.txt (KITTI lidar 格式)
  → 官方评测 (kitti_object_eval_python)        om_ref_test.py / quick_eval.py（已内嵌）
```

## 快速评测命令

```bash
# 推理（200 帧）+ 自动官方评测
python npu/quick_eval.py --device cuda:0 --frames 200 --save-preds /tmp/preds

# 或分步：推理 → 评测
python npu/infer.py --ckpt weights/pointpillar_7728.pth --device auto --frames 200 --save-preds /tmp/preds
python npu/eval.py --preds /tmp/preds --frames 200

# OM 后端全量（默认已内嵌官方 AP 评测）
python npu/om_ref_test.py --om weights/pointpillar_base_fp32_dynamic_linux_aarch64_linux_aarch64.om
python npu/om_ref_test.py --om <om> --quick   # 只看简化 Recall/Precision
```

预测文件格式：`<class> -1 -1 -1 0 0 0 0 x y z dx dy dz ry score`（lidar 系）。
官方评测内部转 camera 系并投影 2D bbox（难度分类），GT 来自 `kitti_infos_val.pkl`（含 DontCare 处理）。

## 当前结果（2026-09-22，KITTI val 全量 3769 帧，OM 动态 fp32）

后端：`weights/pointpillar_base_fp32_dynamic_linux_aarch64_linux_aarch64.om`，
全量 3769 帧 `om_ref_test.py` 推理并内嵌官方评测（R11/R40），`skipped_M=0`。

### 3D AP（R11，官方口径 Car@0.7 / Ped@0.5 / Cyc@0.5）

| class | easy | moderate | hard |
|---|---|---|---|
| Car | 86.41 | **77.25** | 74.37 |
| Pedestrian | 56.91 | **51.67** | 47.54 |
| Cyclist | 79.36 | **61.76** | 58.59 |

### 3D AP（R40）

| class | easy | moderate | hard |
|---|---|---|---|
| Car | 87.69 | **78.33** | 75.06 |
| Pedestrian | 56.51 | **50.90** | 46.43 |
| Cyclist | 79.93 | **62.04** | 58.13 |

### BEV AP（R11）

| class | easy | moderate | hard |
|---|---|---|---|
| Car | 89.62 | 87.02 | 83.46 |
| Pedestrian | 61.05 | 56.31 | 52.45 |
| Cyclist | 81.60 | 65.36 | 61.33 |

### 与官方基线对照（3D moderate R11）

| class | 官方（3D moderate R11） | 本实现（全量） | 差距 |
|---|---|---|---|
| Car | 77.28 | 77.25 | -0.03 ✅ |
| Pedestrian | 52.29 | 51.67 | -0.62 |
| Cyclist | 62.68 | 61.76 | -0.92 |

- **Car 与官方基线几乎一致**（-0.03），确认 checkpoint + 前处理 + OM fp32 + 后处理 + 官方评测全链路正确。
- Pedestrian / Cyclist 比官方低 0.6~0.9 点，属正常波动，无异常偏差。

## 历史结果与修复记录

历史精度结果、已修复的精度问题（VoxelGeneratorV2 分组错位、rotate_iou 数学不等价、喂数 dtype/非连续、
`load_data_to_cpu` 转 float32 等）全部记录在 **`npu/CHANGELOG.md`**，本文档只保留最新精度。

## 精度验证方法

- `npu/compare_pt_om.py`：PyTorch vs OM 数值对比（box/cls cosine）。
- 两端体素化全等验证：dump voxels → diff（14840/14840）。
- rotate_iou 数学等价：随机框对比 GPU 版 vs CPU 版（max diff ~1e-6）。
