# PointPillars NPU 移植 / ONNX / OM 变更记录（CHANGELOG）

记录 NPU 移植 + ONNX 导出 + OM 推理 + 精度对齐全过程的代码改动、动机与验证结果，便于回溯。

涉及仓库：`OpenPCDet`（NPU 适配）与 `unum_ops`（spconv 兼容 shim）。

---

## v1.1.0（2026-09-22）：GPU/NPU 精度对齐 77.83

### 精度根因修复

| 问题 | 根因 | 修复 |
|---|---|---|
| GPU 推理 AP≈0、全 Cyclist 荒谬框 | `unum_ops/spconv/utils.py` VoxelGeneratorV2 分组错位：`np.lexsort((z,y,x))` 与 `np.unique(coords,axis=0)`（x,y,z 序）分组边界不对齐，36% voxel 内点落入错误 pillar | 邻坐标变化切分，14840/14840 voxel 与参考全等 |
| 设备 NPU 评测 AP 偏低 ~13 点 | 设备替换的 `rotate_iou.py`（CPU box_overlap_bev + camera→lidar 转换）与官方 numba CUDA `rbbox_iou` 数学不等价（相同朝向框官方 IoU=0 / 设备=1） | 逐行复刻官方 CUDA 数学的 CPU numba 版，与本机 GPU 版 max diff ~1e-6 |
| 纯 GPU torch build 下 `import unum_ops` 崩溃 | `infllm_v2/max_pooling_1d_varlen.py:22` 直接 `torch.npu.is_available()`，无 torch.npu 时抛 AttributeError（非 ImportError），try/except 捕获不到 | `getattr(torch,"npu",None)` 守卫 |
| 设备 NPU 上 anchors 设备不一致 | `anchor_head_template.py:31` 仅 `x.cuda() if torch.cuda.is_available() else x`，NPU 下 cuda 不可用 → anchors 留 CPU | anchors 设备感知（cuda/npu 分支） |
| 设备 NPU 逐帧编译卡顿 | torch_npu 默认 jit_compile | `torch.npu.set_compile_mode(jit_compile=False)` |

### 结果（KITTI val 前 200 帧，官方评测 3D moderate）

| class | 本机 GPU | 设备 NPU | 官方基线 |
|---|---|---|---|
| Car | 77.83 | 77.83 | 77.28 |
| Pedestrian | 57.90 | 57.90 | 52.29 |
| Cyclist | 36.70 | 36.70 | 62.68 |

Car 与官方一致；Ped 高于官方 5.6；Cyc 明显低于官方 26，指向 checkpoint 训练集/类别分布差异，非管线问题（两端逐位一致）。

### 新增 skill 交付物
`npu/infer.py`（build_data/pre_process/build_model/post_process/main）、`npu/eval.py`、
`npu/npu_patch.py`（设备检测/算子适配统一补丁）、`npu/verify_npu.sh`（环境→补丁→推理→评测串联验证）、
`CodeReview_Results_2026-09-22.md`。

---

## v1.0.3（2026-08-18）：单帧推理性能优化（1543ms → ~300ms）

### 优化 1：voxelize 向量化（8.7x，`unum_ops/src/unum_ops/spconv/utils.py`）
- 原实现：`np.unique(return_inverse)` 后逐体素 `np.nonzero(inv==v)` 逐点写循环。
- 新实现：`np.unique(coords, return_counts)` 后一次性向量化写入 `voxels[inverse[mask], pos[mask]]`。
- 效果：prepare_data 586→67ms；输出与旧实现逐位一致。

### 优化 2：numba 旋转 IoU NMS（~15x，`pcdet/ops/iou3d_nms/iou3d_nms_torch_native.py`）
- `nms_gpu` 在 numba 可用时走 `_nms_iou_matrix`（Sutherland–Hodgman，AABB 预筛 + 交点裁剪）。
- 效果：260×260 IoU 矩阵 881ms（torch）→ 59.8ms（numba，稳态 6~31ms）；keep 结果与 torch 一致。

### 优化 3：numba 首调开销移出计时
- run_bin.py 在计时前用 `_nms_iou_matrix(np.zeros((2,7),np.float32))` 预热（首调 ~1.5s）。

> 注意：2026-08-18 设备异常（npu-smi Health=Warning，dmesg `ascend_monitor dmp heart beat lost error`），
> OM 推理临时退化（0 框），ONNX(CPU) 复现 25 框正常，确认非代码回归。

---

## v1.0.2（2026-08-18）：全量 3769 帧评测对齐

- `tools/test_om.py` 重写：与 `tools/test.py` 同一评估方法（KittiDataset + generate_prediction_dicts + evaluation），仅替换推理后端为 OM。
- 关键 Bug：`load_data_to_cpu` 统一 `.float()` 把 voxel_coords/num_points 转 float32 → OM 输出垃圾框，全量 AP=0；改为 `torch.from_numpy` 保留原 dtype。
- 动态 OM range 不足：val 集 M 最大 8567 > 3941，重转 `pointpillar_mixed_float16_dyn9000`（voxels 1~9000）。
- 官方评估无 CUDA：注入 CPU 版 rotate_iou（Sutherland–Hodgman，与官方几何一致），eval.py 零改动。

全量 3769 帧（mixed_float16 + mix_optimized 动态 OM）：
```
Car  3d AP: 86.5 / 77.2 / 74.6     3d AP_R40: 87.8 / 78.3 / 75.2
Car  bev AP: 89.7 / 87.1 / 84.4
Ped  3d AP: 57.0 / 52.0 / 47.6
Cyc  3d AP: 80.0 / 62.8 / 59.8
```
与公开 PointPillars KITTI val 结果同量级（Car 3D moderate ~74-77）。

---

## v1.0.1（2026-08-14 ~ 08-17）：ONNX/OM 性能优化链（419.8 → 15.7ms）

| 版本 | 优化 | OM 推理 | 累计加速 |
|---|---|---|---|
| V1 | 基线（force_fp32） | 419.8 ms | 1× |
| V2 | 图手术：Cast 链合并 + Head Conv 合并 + ATC 融合规则全开 | 150.5 ms | 2.8× |
| V3 | ScatterND → Gather（`bev_index_map` 索引表，第 4 输入） | 47.3 ms | 9× |
| V4 | ArgMax → `(x1>x0)` 比较（NUM_DIR_BINS=2 等价） | 27.7 ms | 15× |
| V5 | onnxsim 简化（218→111 节点） | 27.8 ms | 持平 |
| V6 | VFE BatchNorm1d → 逐通道 affine（去 2 次 permute） | 23.6 ms | 17.8× |
| — | mixed_float16 混合精度 | 17.3 ms | 24/24 |
| **最终** | **mixed_float16 + mix_optimized（white-list 不含 Mul）** | **15.7 ms** | **24/24** |

### 关键诊断与决策
- msprof：5 个 ScatterNDUpdate 占算子耗时 ~96%，其中 4 个是 IndexPut 赋值被 trace 成伪 ScatterND；仅 map_to_bev 是核心 scatter。
- ScatterND 在 310P 单核标量循环（block=1, scalar 95%）；Gather 多核随机读，快 13×。
- ArgMax（dir_labels）block=8 但 scalar 92%；NUM_DIR_BINS=2 → `argmax ≡ (x1>x0)`。
- VFE BN1d 两次 permute 走 ConfusionTranspose 开销大；展开为 `x*scale+shift`（max|diff|=3.8e-6）。
- 精度边界：mixed_float16 自动保持 vfe 全段 + map_to_bev fp32；`--modify_mixlist` 排除法唯一必须 fp32 的是 vfe 的 `Mul`（拆散 AutomaticBufferFusionOp 使 vfe BN 链整体 fp16 化会丢框）。
- deconv 拆分实验（插零 Gather + 普通 Conv）慢 6×，不采用。

### ONNX 导出与 ATC 要点
- 纯 CPU 导出（`torch.onnx.export(dynamo=False)`），后处理留在 Python 侧。
- `PPWrapper` 必须保留 `self.model = model`（注册子模块），否则 tracer 报 `Cannot insert a Tensor that requires grad as a constant`。
- ATC 必须 `--precision_mode=force_fp32`（默认 force_fp16 丢框：24→23）。
- 输入：`voxels (M,32,4) f32`、`voxel_num_points (M,) i32`、`voxel_coords (M,4) i32`、`bev_index_map (G,) i64`。
- 输出：`batch_box_preds (1,321408,7)`、`batch_cls_preds (1,321408,3)`。

---

## v1.0.0（2026-08-14）：NPU 原生推理跑通

### 敏感算子 CPU 兜底（aicpu GatherElements 不稳定，errorCode 0x2a）
| 文件 | 改动 |
|---|---|
| `pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py` | scatter 移到 CPU |
| `pcdet/ops/iou3d_nms/iou3d_nms_torch_native.py` | NMS 整体移 CPU |
| `pcdet/models/model_utils/model_nms_utils.py` | class_agnostic_nms 整体移 CPU |
| `pcdet/models/detectors/detector3d_template.py` | post_processing 末端移 CPU |
| `pcdet/models/model_utils/model_nms_utils.py` | `scores_mask.nonzero().view(-1)` → `.reshape(-1)` |

效果：`demo.py` 端到端 exit=0，输出 23~24 框。

### spconv 兼容层（`unum_ops.spconv`）
- NPU 环境不能用原生 spconv，由 `unum_ops.spconv` 替代（`pcdet/utils/spconv_utils.py` 统一入口）。
- PointPillar 模型不依赖 spconv（backbone 为稠密 Conv），ONNX/OM 链路无稀疏算子；
  spconv 兼容层仅影响项目内其他模型（SECOND/VoxelNeX/PartA2 等）。

---

## 环境备忘

- 设备 SoC：Ascend 310P3；CANN 9.0.0（atc `/usr/local/Ascend/cann-9.0.0/bin/atc`）。
- 设备 venv：Python 3.11 + torch 2.7.1 + torch_npu 2.7.1。
- 本机 GPU：torch 2.7.1+cu128（无 torch_npu）。
- aclruntime wheel：`/data/workspace/qwen2onnx/aclruntime/wheel/aclruntime-0.0.3-cp311-cp311-linux_aarch64.whl`。
- OM 产物：`pointpillar_fp32_v6.om`（23.6ms）、`pointpillar_mixed_float16_opt.om`（15.7ms）、
  `pointpillar_mixed_float16_dyn9000_linux_aarch64.om`（动态 M 1~9000，全量评测用）。
- 数据：KITTI 训练/验证集，配置 `data/config.yaml`（MODEL 段与官方 pointpillar.yaml 字段一致）。
