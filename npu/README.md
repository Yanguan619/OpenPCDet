# PointPillars (OpenPCDet) — NPU 移植版

基于 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 的 PointPillars 模型在**昇腾 NPU（Ascend 310P3）**上的推理移植。
支持 GPU / NPU 双环境运行，提供从 PyTorch checkpoint → ONNX → OM 的完整链路与官方 KITTI 评测。

## 环境

| 环境 | 设备 | 软件 |
|---|---|---|
| 本机（开发/基线） | NVIDIA GPU | torch 2.7.1+cu128，python 3.11 |
| 设备（推理） | Ascend 310P3 | torch 2.7.1 + torch_npu 2.7.1，CANN 9.0.0 |

NPU 环境通过 `unum_ops.spconv`（numpy 体素化 shim）替代原生 spconv，`pcdet` 代码无硬编码 `.npu()`，依赖 torch_npu 设备层，因此同一套代码可在 GPU 上直接运行做基线。

## 目录结构

```
npu/
├── infer.py            # 推理入口（build_data/pre_process/build_model/post_process/main）
├── eval.py             # 官方 KITTI 评测入口
├── npu_patch.py        # 设备检测/算子适配统一补丁（init_patch）
├── verify_npu.sh       # 环境→补丁→推理→评测 串联验证
├── quick_eval.py       # 快速评估（GPU/NPU 通用，生成 preds + 自动官方评测）
├── compute_kitti_ap.py # 快速 BEV AP（sanity check）
├── om_ref_test.py      # 全量 val 评测（OM 后端，内嵌官方 AP；--quick 简化口径）
├── om_ref_test_pt.py   # 全量 val 推理（PyTorch 后端）
├── export_onnx.py      # ONNX 导出 + 图手术（含 NMS）
├── export_full.py      # PyTorch 直接导出（含后处理，替代图手术方案）
├── export_postproc.py  # 导出 PPWrapper+后处理（无 NMS）
├── atc.py              # ATC 转 OM（动态/静态/fp32/fp16/mixed）
├── om_ref_demo.py          # 单帧 .bin 推理（OM，分段计时）
├── demo.py             # 单帧 demo（可视化）
├── perf_e2e.py         # E2E 性能测试
├── compare_pt_om.py    # PyTorch vs OM 数值对比
├── README.md           # 本文档
├── PRECISION.md        # 精度评测文档
├── PERFORMANCE.md      # 性能分析文档
└── CHANGELOG.md        # 变更记录
```

## 快速开始（PyTorch 推理 + 官方评测）

```bash
# 1. 推理（device 可选 auto/npu/cuda:0/cpu）
python npu/infer.py --ckpt weights/pointpillar_7728.pth --device auto \
    --frames 200 --save-preds preds

# 2. 官方 KITTI 评测（bbox/bev/3d AP，R11+R40）
python npu/eval.py --preds preds --frames 200

# 3. 一键验证（环境 + 补丁 + 推理 + 评测）
bash npu/verify_npu.sh --frames 5
```

### 快速评估（推荐调试用）

```bash
python npu/quick_eval.py --device cuda:0 --frames 200 --save-preds /tmp/preds
```

`quick_eval.py` 推理后自动运行官方评测，输出 Car/Pedestrian/Cyclist 的 bbox/bev/3d AP。

### 快速 BEV AP（不依赖官方 eval）

```bash
python npu/compute_kitti_ap.py --preds /tmp/preds
```

## ONNX 导出 → ATC 转 OM → NPU 推理

```bash
# 1. 导出 ONNX（纯 CPU，无需 NPU）
python npu/export_onnx.py --ckpt weights/pointpillar_7728.pth --sample-idx 000008 \
    --output weights/pointpillar_demo.onnx

# 2. ATC 转 OM（force_fp32 保证精度）
python npu/atc.py --fp32 --skip-export
#   或等价命令：--precision_mode=force_fp32 --soc_version=Ascend310P3

# 3. OM 单帧推理
python npu/om_ref_demo.py --bin data/kitti/training/velodyne/000008.bin \
    --om weights/pointpillar_fp32_linux_aarch64.om --num-iters 20

# 4. 全量 val 评测（OM 后端，内嵌官方 AP 评测，R11/R40 bbox/bev/3d）
python npu/om_ref_test.py --om weights/pointpillar_mixed_float16_dyn9000_linux_aarch64.om
python npu/om_ref_test.py --om <om> --quick     # 只看简化 Recall/Precision
```

### 输入输出规格（ONNX/OM）

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `voxels` | (M, 32, 4) | float32 | 非空 pillar 特征（M 每帧不同） |
| `voxel_num_points` | (M,) | **int32** | 每 pillar 点数（勿喂 float32） |
| `voxel_coords` | (M, 4) | **int32** | [batch, x, y, z] 网格坐标 |
| `bev_index_map` | (214272,) | **int64** | pillar→BEV 的 Gather 索引表（替代 ScatterND） |
| `batch_box_preds` | (1, 321408, 7) | float32 | 解码后 lidar 框 [x,y,z,dx,dy,dz,heading] |
| `batch_cls_preds` | (1, 321408, 3) | float32 | 分类 logits（未 sigmoid） |

> 321408 = 216×248（BEV 网格）× 2（rot）× 3（class）。
> 后处理（sigmoid + topk + NMS）在 Python 侧完成，不包含在 ONNX/OM 中。

### ATC 精度要点

- **必须 `--precision_mode=force_fp32`**：默认 force_fp16 逐元素偏差越过 NMS 阈值会丢框（24→23）。
- 静态导出 M 固定（000008 → 3941），换帧需重导；全量评测用动态 OM（`voxels:1~9000`）。
- `voxel_coords` 经 collate 后是 Fortran 序，喂 OM 前必须 `np.ascontiguousarray`。
- aclruntime 输出取数：`t.to_host()`（原地）后 `np.frombuffer(memoryview(t), dtype).reshape(t.shape).copy()`。

## 模型与权重

- 权重：`weights/pointpillar_7728.pth`（官方发布，127/127 keys 完全匹配）。
- 配置：`data/config.yaml`（MODEL 段与官方 `kitti_models/pointpillar.yaml` 字段一致）。
- 类别：`['Car', 'Pedestrian', 'Cyclist']`。

## 性能结论（摘要）

- PyTorch NPU 推理：~273 ms/帧（200 帧，含前/后处理）。
- OM（mixed_float16 + 优化）：NPU 前向 **15.7 ms**（单帧 demo）。
- 瓶颈在 host 侧 FOV 过滤 + voxelize + 后处理（CPU），NPU 推理仅占 ~10%。
- 详细见 `npu/PERFORMANCE.md`。

## 文档索引

- `npu/PRECISION.md` — 精度评测方法、管线、结果与历史问题。
- `npu/PERFORMANCE.md` — 性能数据、瓶颈分析与优化方向。
- `npu/CHANGELOG.md` — 全部代码改动与优化记录。
