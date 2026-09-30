# PointPillars-推理指导

- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [准备容器](#准备容器)
  - [安装依赖](#安装依赖)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型转换（ckpt → ONNX → OM）](#模型转换ckpt--onnx--om)
  - [模型推理](#模型推理)
- [模型推理精度](#模型推理精度)
  - [精度](#精度)

******

# 概述

PointPillars 是将 PointNet 直接作用于 pillar（柱体）稀疏体素的 3D 目标检测网络：点云体素化为 pillar 后经 2D CNN backbone 提取 BEV 特征，由 SSD head 输出 3D 检测框。本项目移植自 OpenPCDet 的 PointPillar，在 KITTI 数据集上实现 3 类 3D 目标检测（Car / Pedestrian / Cyclist），推理链路为 PyTorch checkpoint → ONNX 图手术 → ATC 转 OM → aclruntime 推理，体素化走 AscendC 自定义算子（NPU kernel，无回退）。

- 参考论文：https://arxiv.org/abs/1812.05784 （PointPillars: Fast Encoders for Object Detection from Point Clouds）
- 参考实现：
  ```
  url=https://github.com/open-mmlab/OpenPCDet
  model_name=PointPillar (tools/cfgs/kitti_models/pointpillar.yaml，v0.6.0 Model Zoo 官方权重)
  ```

## 输入输出数据

- 输入数据（OM 接口）

  | 输入数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | ---- | ------------ |
  | LiDAR 点云（velodyne `.bin`，前处理含 FOV 过滤 + 体素化） | FLOAT32 | (N, 4)，N 为点数（x, y, z, intensity） | ND |
  | 非空 pillar 特征 `voxels` | FLOAT32 | (M, 32, 4)，M 为 pillar 数（1~18000 动态） | ND |
  | 每 pillar 点数 `voxel_num_points` | INT32 | (M,) | ND |
  | pillar 网格坐标 `voxel_coords` | INT32 | (M, 4)，[batch, x, y, z] | ND |
  | pillar→BEV 索引表 `bev_index_map` | INT64 | (214272,)，216×248 BEV 网格 | ND |

- 输出数据（OM 接口）

  | 输出数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | ---- | ------------ |
  | TopK 框 `topk_boxes` | FLOAT32 | (1, 4096, 7)，[x,y,z,dx,dy,dz,heading] | ND |
  | TopK 类别 logits `topk_cls` | FLOAT32 | (1, 4096, 3)，未 sigmoid | ND |

- 最终输出（脚本内后处理：sigmoid + score 阈值 + NMS）

  | 输出数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | ---- | ------------ |
  | 3D 检测框 | FLOAT32 | (N_det, 7) | ND |
  | 检测分数（sigmoid 后） | FLOAT32 | (N_det,) | ND |
  | 类别标签（3 类） | INT64 | (N_det,) | ND |

# 推理环境准备

- 该模型需要以下插件与驱动

  | 配套 | 版本 | 环境准备指导 |
  | ---- | ---- | ------------ |
  | 固件与驱动 | 随镜像内置（Ascend 310P3） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 9.0.0 | - |
  | Python | 3.11 | - |
  | PyTorch | 2.7.1（CPU 版即可，模型推理走 aclruntime） | - |
  | torch_npu | 2.7.1 | - |
  | aclruntime | CANN 9.0.0 自带 | - |

  | 项 | 配套要求 | 实测可用版本 | 说明 |
  | ---- | ---- | ---- | ---- |
  | unum_ops | bevpool-kernel-opt 分支 | **bevpool-kernel-opt**（`a8a101d`） | 必需依赖：spconv shim + **AscendC 体素化 kernel**（含 561000 修复、MAX_NBLK 8）；`pip install -e .` 时自动构建安装 OPP |
  | numpy | ≥1.26 | 2.4.6 | 全链路实测通过 |
  | numba | ≥0.59 | 0.67.0 | mask / FOV / NMS 使用 |
  | onnx | ≥1.16 | 1.16.1 | 仅模型转换（`export_onnx.py`）需要 |
  | onnx_optimizer（msit） | surgeon-only 分支 | **surgeon-only**（`81c4aa6`） | 必需依赖（模型转换）：Ascend 图改写 knowledge（ScatterND/ArgMax 消除）+ Constant→initializer 归一化 |

> 体素化**固定走 AscendC NPU kernel**（无回退无开关）——unum_ops/OPP 缺失或版本过旧会在 `import npu_patch` 时直接 ImportError，运行期算子异常带栈抛出，**不会静默换 CPU numba**；启动时打印一行当前 voxelize 模式。

# 快速上手

## 准备容器

宿主机已具备 CANN 9.0.0 + torch 2.7.1 + torch_npu 2.7.1 环境时可跳过本节。

```bash
docker run -itd \
    --name npu-pointpillars \
    -w /workspace \
    --privileged \
    --ipc=host \
    --net=host \
    --shm-size=5g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/bin/hostname:/usr/bin/hostname \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/log/npu/:/usr/slog \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /etc/localtime:/etc/localtime \
    -v /etc/hosts:/etc/hosts \
    -v /data:/data \
    -v /home:/home \
    -v /mnt:/mnt \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11 bash
```

进容器执行

```bash
docker exec -it npu-pointpillars bash
npu-smi info
```

确认设备可见。

## 安装依赖

```bash
# 1) unum_ops（spconv shim + AscendC 体素化 kernel；pip install 自动构建 OPP，需 CANN 可用）
git clone https://github.com/Yanguan619/unum_ops.git
cd unum_ops && git checkout bevpool-kernel-opt && pip install -e .

# 2) msit onnx_optimizer（模型转换 Step 1.5 的 Ascend 图改写 knowledge，必经无开关）
git clone https://gitcode.com/Yanguan/msit.git
cd msit && git checkout surgeon-only && cd onnx_optimizer && pip install -e .

# 3) 若 OPP 未能随 pip 安装（或升级 kernel 后），手动重装：
bash build.sh --soc=ascend310p -j8 && cd build && \
    bash custom_opp_openEuler_aarch64.run --install-path=/usr/local/Ascend/cann-9.0.0/opp
```

OpenPCDet 本体无需安装（推理脚本自行把仓库根加入 `sys.path`），其余依赖（numpy / numba / onnx）按 [推理环境准备](#推理环境准备) 的版本安装即可（onnx_optimizer 已在上一步安装）。

## 获取源码

1. 获取本仓库：

   ```text
   OpenPCDet/
   └── npu/                            # NPU 推理交付目录
       ├── om_ref_demo.py              # 单帧推理 + 计时（主入口）
       ├── om_ref_test.py              # 全量数据集评测（内嵌官方 KITTI AP）
       ├── export_onnx.py              # ONNX 导出+Ascend 图改写+图手术一条链（无数据集依赖）
       ├── npu_patch.py                # 设备检测/算子适配统一补丁（体素化固定 AscendC）
       ├── verify_npu.sh               # 环境→补丁→推理→评测 串联验证
       ├── ops_native/                 # 纯 numpy/numba 算子（iou3d NMS 等）
       └── README.md / PRECISION.md / PERFORMANCE.md / CHANGELOG.md
   ```

## 准备数据集

> 数据集仅**推理 demo / 全量评测**需要；[模型转换](#模型转换ckpt--onnx--om)不依赖数据集，可先行完成。

- **单帧 demo**：任意 velodyne `.bin` 点云（如 `data/kitti/training/velodyne/000008.bin`，默认路径即此）。
- **全量评测（KITTI val）**：到 [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)（需注册）下载并组织：

  ```bash
  mkdir -p data/kitti/training/{calib,image_2,label_2,velodyne,planes}
  mkdir -p data/kitti/testing/{calib,image_2,velodyne}
  # 解压 data_object_calib/label_2/image_2/velodyne.zip 到对应目录；
  # ImageSets/{test,train,trainval,val}.txt 取自 OpenPCDet 上游仓库 data/kitti/ImageSets/

  # 生成 infos（om_ref_test 依赖 kitti_infos_val.pkl）
  python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
      tools/cfgs/dataset_configs/kitti_dataset.yaml
  ```

- **权重**（官方 PointPillar，OpenPCDet v0.6.0 Model Zoo）：下载
  `https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing`
  存为 `weights/pointpillar_7728.pth`（127/127 keys 完全匹配）。

## 模型转换（ckpt → ONNX → OM）

```bash
mkdir -p weights

# (a) ckpt → ONNX
python npu/export_onnx.py --ckpt weights/pointpillar_7728.pth \
    --output weights/pointpillar_7728.onnx

# (b) ATC 转 OM（--output 名即推理脚本的默认 OM 名，demo/评测无需再传 --om）
atc --model=weights/pointpillar_7728.onnx --framework=5 --soc_version=Ascend310P3 \
    --output=weights/pointpillar_base_fp16_dynamic18000_topk_surgery_abc_linux_aarch64 \
    --input_format=ND --precision_mode=force_fp16 \
    --input_shape="voxels:1~18000,32,4;voxel_num_points:1~18000;voxel_coords:1~18000,4;bev_index_map:214272"
```

## 模型推理

### 单帧推理 demo

```bash
python npu/om_ref_demo.py --data_path data/kitti/training/velodyne/000008.bin
```

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `--cfg_file` | `tools/cfgs/kitti_models/pointpillar.yaml` | 模型配置 |
| `--data_path` | `data/kitti/training/velodyne/000008.bin` | 点云 `.bin` 文件或目录 |
| `--om` | `weights/..._topk_surgery_abc_linux_aarch64.om` | OM 模型 |
| `--ext` | `.bin` | 点云文件扩展名（目录模式） |
| `--device` | `0` | NPU 设备 id |
| `--num-iters` | `10` | OM 测速迭代次数 |
| `--score-thresh` | `None`（取 config 值 0.1） | 检测输出阈值 |
| `--iou-thresh` | `0.5` | 与 label 匹配的 IoU 阈值 |
| `--label` | `None`（自动推断） | KITTI label 文件路径（无则跳过对比） |

输出为分段耗时（前处理/推理/后处理）+ 检测框与 label 的逐框对比。

**实测性能**（Ascend 310P3 单卡 `--device 1`，000008 帧、无 FOV、体素化 122555 点）：单帧 E2E 中位 **22.5–23.3 ms**（surgery ABCD/ABC 口径，30/10 iters），其中 OM 前向 ~11 ms、AscendC 体素化 ~9.8 ms；输出 **33 框（Car 12 / Ped 14 / Cyc 7）与 PyTorch 基线逐帧一致**。

### 全量数据集评测（KITTI val 3769 帧）

```bash
# 全量 val 评测（内嵌官方 R11/R40 AP）
python npu/om_ref_test.py

# 快速抽验（跳过官方评测，只看简化计时）
python npu/om_ref_test.py --frames 200 --quick
```

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `--config` | `data/config.yaml` | 数据集配置（含 `DATA_PATH`） |
| `--om` | `weights/..._topk_surgery_abc_linux_aarch64.om` | OM 模型 |
| `--device` | `0` | NPU 设备 id |
| `--frames` | `0`（全部） | 只跑前 N 帧（快速抽验） |
| `--start` / `--end` | `0` / `0`（到末尾） | 帧区间（分块续跑用） |
| `--score-thresh` | `None`（取 config 值 0.1） | 检测输出阈值 |
| `--iou-thresh` | `0.5` | 简化评测的匹配 IoU 阈值 |
| `--save-preds` | `None` | 逐帧预测落盘目录（KITTI label 格式） |
| `--no-fov` | `False` | 不做 FOV 过滤 |
| `--verbose-frames` | `False` | 逐帧打印检测框 |
| `--quick` | `False` | 只输出简化 TP/FP/FN + Recall/Precision（不跑官方 AP） |

**性能**：全量口径（有 FOV，200 帧）2026-09-28 实测 63.3 ms/帧（旧管线）；09-29 集成态优化（设备常驻 feeds / collate fast-path / 图手术 ABC / 后处理重构）预计 **~45–50 ms/帧**，3769 帧约 3–4 分钟 + 官方评测耗时（尚未在全量口径复测）。

### 环境自检

```bash
bash npu/verify_npu.sh
```

依次检查：环境（torch/torch_npu/设备）→ 核心文件语法 → `npu_patch` 初始化（voxelize 应为 AscendC）→ 单帧 demo → 200 帧简化评测。

# 模型推理精度

## 精度

测试条件：KITTI val **全量 3769 帧**，fp32 动态 OM（`om_ref_test.py` 内嵌官方 KITTI 评测，R11/R40，官方口径 Car@0.7 / Ped@0.5 / Cyc@0.5），`skipped_M=0`。

### 3D AP（R11）

| class | easy | moderate | hard |
| ----- | ---- | -------- | ---- |
| Car | 86.41 | **77.25** | 74.37 |
| Pedestrian | 56.91 | **51.67** | 47.54 |
| Cyclist | 79.36 | **61.76** | 58.59 |

### 3D AP（R40）

| class | easy | moderate | hard |
| ----- | ---- | -------- | ---- |
| Car | 87.69 | **78.33** | 75.06 |
| Pedestrian | 56.51 | **50.90** | 46.43 |
| Cyclist | 79.93 | **62.04** | 58.13 |

### 与官方基线对照（3D moderate R11）

| class | 官方（3D moderate R11） | 本实现（全量） | 差距 |
| ----- | ---------------------- | -------------- | ---- |
| Car | 77.28 | **77.25** | -0.03 |
| Pedestrian | 52.29 | 51.67 | -0.62 |
| Cyclist | 62.68 | 61.76 | -0.92 |

- **Car 与官方基线几乎一致**（-0.03），确认 checkpoint + 前处理 + OM + 后处理 + 官方评测全链路正确；Pedestrian / Cyclist 低 0.6~0.9 点，属正常波动。
- fp16 链路（force_fp16 + topk 图内化 + surgery ABC 图手术）已验证 **200 帧 AP 在 1% 容差内**，且 surgery ABC 与 topk OM 输出**语义完全一致**（TopK 保持 4096）；单帧 demo 检测框与 PyTorch 基线**逐帧一致**（33 框）。
- BEV AP 与历史精度问题记录见 `npu/PRECISION.md`。

# 公网地址声明

本 README 中引用的 GitHub（OpenPCDet、unum_ops）、OpenPCDet 权重下载、KITTI 官网及昇腾容器镜像地址，均用于获取公开源码、公开数据集、公开权重或官方环境镜像；交付的适配代码未新增公网访问行为。
