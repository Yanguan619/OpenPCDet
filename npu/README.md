# PointPillars (OpenPCDet) — NPU 移植版

基于 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 的 PointPillars 模型在**昇腾 NPU（Ascend 310P3）**上的推理移植，
提供从 PyTorch checkpoint → ONNX → OM → 官方 KITTI 评测的完整链路。

## 环境

| 环境 | 设备 | 软件 |
|---|---|---|
| 本机（开发/基线） | NVIDIA GPU | torch 2.7.1+cu128，python 3.11 |
| 设备（推理） | Ascend 310P3 | torch 2.7.1 + torch_npu 2.7.1，CANN 9.0.0 |

NPU 环境通过 `unum_ops.spconv`（numpy 体素化 shim）替代原生 spconv，`pcdet` 代码无硬编码 `.npu()`，依赖 torch_npu 设备层，因此同一套代码可在 GPU 上直接运行做基线。

## 新设备部署（按顺序走完即可）

> `weights/`（模型/OM 产物）与 `data/`（KITTI 数据集）**不入 git**，新设备需按下述步骤自行准备。
> 前提：CANN 9.0.0 + torch 2.7.1 + torch_npu 2.7.1 + aclruntime + python 3.11 就绪。

### 1. 安装 NPU 依赖（unum_ops + AscendC 体素化）

```bash
git clone https://github.com/Yanguan619/unum_ops.git
cd unum_ops && git checkout bevpool-kernel-opt && pip install -e .
```

- `pip install` 时 hatch_build 钩子会**自动构建并安装 AscendC OPP 包**（`csrc/ascend/voxelization_v2` → CANN `opp/vendors/`），需 CANN 环境可用（`ASCEND_HOME_PATH`）。
- 若 OPP 未随 pip 安装，手动重装：`bash build.sh --soc=ascend310p -j8 && cd build && bash custom_opp_openEuler_aarch64.run --install-path=/usr/local/Ascend/cann-9.0.0/opp`
- 体素化**固定走 AscendC NPU kernel**（无回退无开关）——unum_ops/OPP 缺失或版本过旧会在 `import npu_patch` 时直接 ImportError，运行期算子异常带栈抛出，**不会静默换 CPU numba**；启动时打印一行当前 voxelize 模式。
- 不装 unum_ops 则 `from spconv.utils import VoxelGeneratorV2` 会 ImportError——unum_ops 是必需依赖。

### 2. 准备 KITTI 数据（data/ 不入 git）

从 [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)（需注册）下载并组织：

```bash
mkdir -p data/kitti/training/{calib,image_2,label_2,velodyne,planes}
mkdir -p data/kitti/testing/{calib,image_2,velodyne}
# 解压 data_object_calib/label_2/image_2/velodyne.zip 到对应目录；
# ImageSets/{test,train,trainval,val}.txt 取自 OpenPCDet 上游仓库 data/kitti/ImageSets/

# 生成 infos（om_ref_test 依赖 kitti_infos_val.pkl）
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### 3. 生成 OM（ckpt → ONNX → 图手术 ABC → ATC）

```bash
mkdir -p weights

# (a) 官方 PointPillar 权重（OpenPCDet v0.6.0 Model Zoo），存为 weights/pointpillar_7728.pth
#     https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing

# (b) base ONNX（noscatter+noargmax，纯 CPU 可执行）
python npu/export_onnx.py --ckpt weights/pointpillar_7728.pth --sample-idx 000008 \
    --output weights/pp_base.onnx --base-output weights/pp_base.onnx --fold-bn

# (c) topk ONNX（图内 ReduceMax+TopK 4096，后处理 11.8→4ms）
python npu/export_onnx.py --topk-only --skip-export --base-output weights/pp_base.onnx \
    --output weights/pp_topk.onnx --fold-bn

# (d) head 图手术 ABC（数学恒等变换：1x1 head 合并 / 冗余 gather 消除 / ConvTranspose→Conv1x1+DTS；
#     不做 D，TopK 保持 4096，输出语义与 topk OM 完全一致）
python npu/surgery_heads.py --in weights/pp_topk.onnx --out weights/pp_surgery_abc.onnx --do ABC

# (e) ATC 转 OM（需装有 CANN 的机器；--output 名即推理脚本的默认 OM 名）
atc --model=weights/pp_surgery_abc.onnx --framework=5 --soc_version=Ascend310P3 \
    --output=weights/pointpillar_base_fp16_dynamic18000_topk_surgery_abc \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"
```

### 4. 推理

```bash
# 单帧 demo（含前/后处理 + label 对比；--om 缺省即上面生成的 surgery ABC OM）
python npu/om_ref_demo.py --data_path data/kitti/training/velodyne/000008.bin

# 全量 val 评测（官方 R11/R40 AP；默认 OM 同上）
python npu/om_ref_test.py

# 快速抽验（跳过官方评测，只看简化计时）
python npu/om_ref_test.py --frames 200 --quick
```

### 5. 环境自检（可选）

```bash
bash npu/verify_npu.sh
```

## 目录结构

```
npu/
├── om_ref_demo.py        # 单帧 .bin 推理（默认 OM：surgery ABC）
├── om_ref_test.py        # 全量 val 评测（内嵌官方 AP；--quick 简化口径）
├── om_ref_test_pt.py     # 全量 val 推理（PyTorch 后端，精度对照用）
├── export_onnx.py        # ONNX 导出（--fold-bn / --topk-only）
├── surgery_heads.py      # head 图手术 A/B/C（生成推荐 OM 的一步）
├── npu_patch.py          # 设备检测/算子适配统一补丁（体素化固定 AscendC）
├── verify_npu.sh         # 环境→补丁→推理→评测 串联验证
├── ops_native/           # 纯 numpy/numba 算子（voxelize、FOV、iou3d NMS）
├── convert_fp16_static9000.sh  # （历史）静态 9000 OM 转换
└── README.md / PRECISION.md / PERFORMANCE.md / CHANGELOG.md
```

## 输入输出规格（surgery ABC OM）

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `voxels` | (M, 32, 4) | float32 | 非空 pillar 特征（M 每帧不同，1~18000） |
| `voxel_num_points` | (M,) | **int32** | 每 pillar 点数（勿喂 float32） |
| `voxel_coords` | (M, 4) | **int32** | [batch, x, y, z] 网格坐标 |
| `bev_index_map` | (214272,) | **int64** | pillar→BEV 的 Gather 索引表（替代 ScatterND） |
| `topk_boxes` | (1, 4096, 7) | float32 | 图内 TopK 选出的框 [x,y,z,dx,dy,dz,heading] |
| `topk_cls` | (1, 4096, 3) | float32 | 对应 anchor 的三类 logits（未 sigmoid） |

- 321408 = 216×248（BEV 网格）× 2（rot）× 3（class）；图内已完成 ReduceMax+TopK(4096)。
- 后处理（sigmoid + score 阈值 + NMS）在 Python 侧完成（`postprocess_topk`）。

## ATC 精度要点

- **`force_fp16` 已验证 AP 1% 容差内**（Car 77.81 / Ped 59.86 / Cyc 38.32，200 帧 R11 3D moderate）。
- 动态 shape 用 **range 记法** `voxels:1~18000`，**勿用 `-1`/`--dynamic_dims`**（避免 ATC mbatch 切分报错）。
- dtype：`voxel_num_points`/`voxel_coords` 为 int32、`bev_index_map` 为 int64，ATC 不可覆盖。
- `voxel_coords` 经 collate 后是 Fortran 序，喂 OM 前必须 `np.ascontiguousarray`（推理脚本已处理）。

## 模型与权重

- 权重：`weights/pointpillar_7728.pth`（官方发布，127/127 keys 完全匹配）。
- 配置：`data/config.yaml`（MODEL 段与官方 `kitti_models/pointpillar.yaml` 字段一致）。
- 类别：`['Car', 'Pedestrian', 'Cyclist']`。

## 性能结论（摘要）

- **单帧 demo 完整推理**（无 FOV，体素化 122555 点）：**22.5 ms**（2026-09-29 集成态实测，device 1
  稳态中位；推荐 OM：`pointpillar_base_fp16_dynamic18000_topk_surgery_abc_linux_aarch64.om`——
  TopK 4096 语义与原 topk OM 一致；ABCD 版因密集帧截断风险已移除，前向仅差 0.05ms）。
- 演进：~93ms（numba 体素化）→ ~67ms（AscendC 体素化）→ **22.5ms**（体素化 561000 修复 + 设备常驻
  管线 + head 图手术 + CPU 后处理重构）；检测输出与基线逐帧一致（33 框）。
- **数据集全量口径**（有 FOV，200 帧）：63.3 ms/帧（2026-09-28 测；本轮各段优化尚未在全量口径复测）。
- 基线 369ms（2026-09-22 全量）→ 63.3ms，累计 **5.8x**（全量口径）。
- 当前瓶颈：OM 推理 ~11ms（backbone conv 占 44% + scatter 1.26ms）与体素化 kernel 6.24ms（标量排序墙）；
  **10ms 目标在当前 fp16 OM 路径不可达**，后续需 int8 量化 / 自定义 scatter / kernel 向量化（详见
  `npu/PERFORMANCE.md` §1.3）。

## 文档索引

- `npu/PRECISION.md` — 精度评测方法、管线、结果与历史问题。
- `npu/PERFORMANCE.md` — 性能数据、瓶颈分析与优化方向。
- `npu/CHANGELOG.md` — 全部代码改动与优化记录。
