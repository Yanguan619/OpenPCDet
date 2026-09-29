# PointPillars (OpenPCDet) — NPU 移植版

基于 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 的 PointPillars 模型在**昇腾 NPU（Ascend 310P3）**上的推理移植。
支持 GPU / NPU 双环境运行，提供从 PyTorch checkpoint → ONNX → OM → 官方 KITTI 评测的完整链路。

## 环境

| 环境 | 设备 | 软件 |
|---|---|---|
| 本机（开发/基线） | NVIDIA GPU | torch 2.7.1+cu128，python 3.11 |
| 设备（推理） | Ascend 310P3 | torch 2.7.1 + torch_npu 2.7.1，CANN 9.0.0 |

NPU 环境通过 `unum_ops.spconv`（numpy 体素化 shim）替代原生 spconv，`pcdet` 代码无硬编码 `.npu()`，依赖 torch_npu 设备层，因此同一套代码可在 GPU 上直接运行做基线。

## 依赖安装与数据准备（新设备）

> `weights/`（模型/OM 产物）与 `data/`（KITTI 数据集）**不入 git**，新设备需按下述步骤自行准备。以下步骤的前提是环境版本（见上表）就绪：CANN 9.0.0 + torch 2.7.1 + torch_npu 2.7.1 + aclruntime（安装步骤见昇腾文档）+ python 3.11。

### 1. NPU 依赖：unum_ops（含 AscendC 体素化 kernel）

`npu_patch` 依赖 `unum_ops` 提供 spconv 的 numpy shim 与 AscendC 体素化 kernel：

```bash
git clone https://github.com/Yanguan619/unum_ops.git
cd unum_ops && pip install -e .
```

- `pip install` 时 hatch_build 钩子会**自动构建并安装 AscendC OPP 包**（`csrc/ascend/voxelization_v2` → CANN `opp/vendors/`），需 CANN 环境可用（`ASCEND_HOME_PATH`）。
- 无 CANN / OPP 构建失败时不影响 Python 侧 spconv/numpy shim（体素化自动回退 CPU numba，功能可用）。
- 不装 unum_ops 则 `from spconv.utils import VoxelGeneratorV2` 会 ImportError（除非另装原生 spconv）——unum_ops 是必需依赖。

### 2. 权重与 OM（自行生成）

`weights/` 不入 git（本机 934MB 模型产物）。新设备流程：**下载权重 → 导出 ONNX → ATC 转 OM**。

```bash
mkdir -p weights

# (a) 官方 PointPillar 权重（OpenPCDet v0.6.0 Model Zoo，本仓库命名为 pointpillar_7728.pth）：
#     https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing
#     下载后存为 weights/pointpillar_7728.pth

# (b) base ONNX（batch_box_preds/batch_cls_preds，noscatter+noargmax）
python npu/export_onnx.py --ckpt weights/pointpillar_7728.pth --sample-idx 000008 \
    --output weights/pp_base.onnx --base-output weights/pp_base.onnx --fold-bn

# (c) ATC 转 base OM（动态 18000，force_fp16；需装有 CANN 的机器执行）
atc --model=weights/pp_base.onnx --framework=5 --soc_version=Ascend310P3 \
    --output=weights/pointpillar_base_fp16_dynamic18000_force \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"

# (d) 可选：topk OM（图内 TopK 4096，推荐，后处理 11.8→4ms；AP 与 base 逐位一致）
python npu/export_onnx.py --topk-only --skip-export --base-output weights/pp_base.onnx \
    --output weights/pp_topk.onnx --fold-bn
atc --model=weights/pp_topk.onnx --framework=5 --soc_version=Ascend310P3 \
    --output=weights/pointpillar_base_fp16_dynamic18000_topk \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"
```

> ATC 产物命名规则：`<名称>_linux_aarch64.om`（推理脚本按此后缀查找）。更多转换组合（fp32/静态 18000 等）见 `npu/PERFORMANCE.md`。

### 3. KITTI 数据（data/ 不入 git）

从 [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)（需注册）下载并组织：

```bash
mkdir -p data/kitti/training/{calib,image_2,label_2,velodyne,planes}
mkdir -p data/kitti/testing/{calib,image_2,velodyne}
# 解压 data_object_calib/label_2/image_2/velodyne.zip（训练集 ~12GB）到对应目录；
# ImageSets/{test,train,trainval,val}.txt 取自 OpenPCDet 上游仓库 data/kitti/ImageSets/

# 生成 infos（om_ref_test 依赖 kitti_infos_val.pkl）
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## 目录结构

```
npu/
├── om_ref_demo.py        # 单帧 .bin 推理（OM 后端，分段计时 + label 对比）
├── om_ref_test.py        # 全量 val 评测（OM 后端，内嵌官方 AP；--quick 简化口径）
├── om_ref_test_pt.py     # 全量 val 推理（PyTorch 后端）
├── export_onnx.py        # ONNX 导出 + 图手术（--fold-bn/--dynamic/--topk-only）
├── npu_patch.py          # 设备检测/算子适配统一补丁（init_patch）
├── verify_npu.sh         # 环境→补丁→推理→评测 串联验证
├── convert_fp16_static9000.sh  # fp16 OM 转换脚本（静态 9000 + 动态/静态 18000）
├── ops_native/           # 纯 numpy/numba 算子（voxelize、FOV、iou3d NMS）
├── debug/                # 历史/辅助脚本（见下）
├── README.md / PRECISION.md / PERFORMANCE.md / CHANGELOG.md
└── debug/
    ├── infer.py          # PyTorch 推理入口（旧，build_data/build_model/post_process）
    ├── eval.py           # 官方 KITTI 评测入口（旧）
    ├── quick_eval.py     # 快速评估（PyTorch，推理 + 自动官方评测）
    ├── atc.py            # ATC 转 OM 封装（动态/静态/fp32/fp16/mixed）
    ├── compare_pt_om.py  # PyTorch vs OM 数值对比
    ├── perf_e2e.py       # E2E 分段计时
    ├── demo.py / export_full.py / export_postproc.py / compute_kitti_ap.py / torch_infer.py / eval_om_nms.py
    └── profiler*/parse_msprof.sh   # msprof 性能剖析
```

## 快速开始

### OM 推理 + 官方评测（主路径）

```bash
# 单帧 demo（含前/后处理 + label 对比；demo 无 FOV 过滤，体素化 122555 点）
python npu/om_ref_demo.py \
    --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
    --data_path data/kitti/training/velodyne/000008.bin \
    --om weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om

# 全量 val 评测（200 帧：输出 前处理/推理/后处理 拆分 + 官方 R11/R40 AP）
python npu/om_ref_test.py \
    --om weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om --frames 200

# 只看简化计时（跳过官方评测）
python npu/om_ref_test.py --om <om> --frames 50 --quick
```

### PyTorch 后端推理 + 官方评测

```bash
python npu/om_ref_test_pt.py --ckpt weights/pointpillar_7728.pth --device npu \
    --frames 200 --save-preds preds_pt
```

（旧入口 `npu/debug/infer.py` + `npu/debug/eval.py` 仍可用；`npu/debug/quick_eval.py` 一键推理+评测。）

## ONNX 导出 → ATC 转 OM → NPU 推理

```bash
# 1. 导出 ONNX（纯 CPU，无需 NPU）：base（batch_box_preds/batch_cls_preds）+ 可选图手术
python npu/export_onnx.py --ckpt weights/pointpillar_7728.pth --sample-idx 000008 \
    --output weights/pp_base.onnx --base-output weights/pp_base.onnx --fold-bn

# 2. ATC 转 OM（推荐 fp16 force；见 npu/PERFORMANCE.md §4 的完整命令）
#    动态 18000 force_fp16（全量 val）：
atc --model=weights/pp_base.onnx --framework=5 --soc_version=Ascend310P3 \
    --output=weights/pointpillar_base_fp16_dynamic18000_force \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"

# 3. OM 单帧推理 / 全量评测
python npu/om_ref_demo.py --data_path <bin 或目录> --om weights/<你的 om>.om
python npu/om_ref_test.py --om weights/<你的 om>.om
```

> 也可用 `npu/debug/atc.py` 封装（`--fp32/--fp16/--mixed --dynamic/--static`）。
> 现成 OM（本机，不入 git）：`weights/pointpillar_base_fp16_dynamic18000_{force,topk}_linux_aarch64.om`、
> `..._static18000_force.om`；新设备按上文「依赖安装与数据准备」自行生成。

### 输入输出规格（base OM）

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `voxels` | (M, 32, 4) | float32 | 非空 pillar 特征（M 每帧不同） |
| `voxel_num_points` | (M,) | **int32** | 每 pillar 点数（勿喂 float32） |
| `voxel_coords` | (M, 4) | **int32** | [batch, x, y, z] 网格坐标 |
| `bev_index_map` | (214272,) | **int64** | pillar→BEV 的 Gather 索引表（替代 ScatterND） |
| `batch_box_preds` | (1, 321408, 7) | float32 | 解码后 lidar 框 [x,y,z,dx,dy,dz,heading] |
| `batch_cls_preds` | (1, 321408, 3) | float32 | 分类 logits（未 sigmoid） |

- 321408 = 216×248（BEV 网格）× 2（rot）× 3（class）。
- **base OM**：后处理（sigmoid + topk + NMS）在 Python 侧完成。
- **topk OM**（`--topk-only` 导出）：图内 ReduceMax+TopK(4096)+Gather，输出
  `topk_boxes (1,4096,7)` / `topk_cls (1,4096,3)`，D2H 13MB→164KB，NMS 仍留 CPU；脚本自动适配。

### ATC 精度要点

- 当前 base（noscatter+noargmax）下 **`force_fp16` 已验证 AP 1% 容差内**（Car 77.81 / Ped 59.86 / Cyc 38.32，
  200 帧 R11 3D moderate），无需强制 fp32。（旧的"force_fp16 必丢框"结论针对早期含 ScatterND/ArgMax 的导出。）
- 动态 shape 用 **range 记法** `voxels:1~18000`，**勿用 `-1`/`--dynamic_dims`**（避免 ATC mbatch 切分报错）。
- dtype：`voxel_num_points`/`voxel_coords` 为 int32、`bev_index_map` 为 int64，ATC 不可覆盖。
- `voxel_coords` 经 collate 后是 Fortran 序，喂 OM 前必须 `np.ascontiguousarray`。

## 模型与权重

- 权重：`weights/pointpillar_7728.pth`（官方发布，127/127 keys 完全匹配）。
- 配置：`data/config.yaml`（MODEL 段与官方 `kitti_models/pointpillar.yaml` 字段一致）。
- 类别：`['Car', 'Pedestrian', 'Cyclist']`。

## 性能结论（摘要）

- **单帧 demo 完整推理**（无 FOV，体素化 122555 点）：**22.5 ms**（2026-09-29 集成态实测，device 1
  稳态中位；推荐 OM：`pointpillar_base_fp16_dynamic18000_topk_surgery_abcd_linux_aarch64.om`）。
- 演进：~93ms（numba 体素化）→ ~67ms（AscendC 体素化）→ **22.5ms**（体素化 561000 修复 + 设备常驻
  管线 + head 图手术 ABCD + CPU 后处理重构）；检测输出与基线逐帧一致（33 框）。
- **数据集全量口径**（有 FOV，200 帧）：63.3 ms/帧（2026-09-28 测；本轮各段优化尚未在全量口径复测）。
- 基线 369ms（2026-09-22 全量）→ 63.3ms，累计 **5.8x**（全量口径）。
- 当前瓶颈：OM 推理 ~11ms（backbone conv 占 44% + scatter 1.26ms）与体素化 kernel 6.24ms（标量排序墙）；
  **10ms 目标在当前 fp16 OM 路径不可达**，后续需 int8 量化 / 自定义 scatter / kernel 向量化（详见
  `npu/PERFORMANCE.md` §1.5）。

## 文档索引

- `npu/PRECISION.md` — 精度评测方法、管线、结果与历史问题。
- `npu/PERFORMANCE.md` — 性能数据、瓶颈分析与优化方向。
- `npu/CHANGELOG.md` — 全部代码改动与优化记录。
