# PointPillars NPU 移植 / ONNX / OM 变更记录（CHANGELOG）

记录 NPU 移植 + ONNX 导出 + OM 推理 + 精度对齐全过程的代码改动、动机与验证结果，便于回溯。

涉及仓库：`OpenPCDet`（NPU 适配）与 `unum_ops`（spconv 兼容 shim）。

---

## TODO v1.3.0（规划中）：性能优化（精度 bit 一致红线）

### ✅ 清理 npu/debug/ + README 单一路径化（2026-09-29）

- **删除 `npu/debug/`**（20 个历史/辅助脚本：infer/eval/quick_eval/atc 封装、compare_*/perf_*/
  om_loop、export_full/postproc、verify_* 等，含九段计时的 perf_e2e_topk.py；需要时从 git 历史取回）。
- **保留 `surgery_heads.py` 并提升到 `npu/` 顶层**：它是推荐 surgery ABC OM 生成链的一步
  （README §3(d)）；`--do` 默认改 ABC（D 已弃用：TopK 1024 截断风险）。
- `om_ref_test.py` 默认 `--om` 对齐 surgery ABC（原默认指向不存在的 static OM）；`verify_npu.sh`
  重写到新路径（环境检查 + 核心文件语法 + init_patch + om_ref_demo 单帧 + om_ref_test --quick）。
- README 重写为**单一路径**：依赖安装 → 数据准备 → OM 生成（ckpt→ONNX→手术 ABC→ATC 一条链）→
  推理（demo/test 默认 OM，零参数），移除可选 OM、旧入口、多后端并列等分支路径。
- README 进一步按 **ModelZoo 推理指导书规范**重构章节：概述（含输入输出数据表）→ 推理环境准备
  （配套/依赖版本表，实测版本）→ 快速上手（准备容器/安装依赖/获取源码/准备数据集/模型转换/模型推理，
  命令+参数表+实测性能）→ 模型推理精度（3769 帧全量 R11/R40 + 官方基线对照）→ 公网地址声明。

### ✅ 移除 surgery ABCD OM：全量截断风险，推荐/默认统一切到 ABC（2026-09-29）

- **根因**：D 手术把图内 TopK 4096→1024，而 config `NMS_PRE_MAXSIZE=4096`，且 TopK 作用在
  score 阈值之前的 raw 分数上——密集帧 pre-NMS 候选 >1024 时在图内被**静默截断**；
  bit-exact 仅 000008 单帧验证，3769 帧无保证。ABC 与 ABCD 前向仅差 0.05ms（9.65 vs 9.60ms），
  且 TopK 保持 4096，语义与原 topk OM 完全一致——0.05ms 换语义零风险，全量禁用 D。
- **改动**：删除 `weights/pointpillar_base_fp16_dynamic18000_topk_surgery_abcd_linux_aarch64.om`
  （及 pcdet-om/om_out 的 ABCD OM/ONNX 产物）；`om_ref_demo` / `perf_e2e_topk` 默认 OM 切到
  `..._surgery_abc_...om`；PERFORMANCE/README 推荐同步更新（ABCD 历史测量数据保留）。

### ✅ 三线性能攻坚 + 集成（2026-09-29，单帧 E2E ~67→22.5ms）

并行三线（各自 worktree/分支），全部合入 master（`3ef09d9`），检测输出与基线逐帧一致（33 框）：

- **体素化**（perf/voxelize-npu `27bcb48` + unum_ops `387bb9e`/`a8a101d`）：
  - **561000 根因**：aclruntime `session.run` 切换线程 device context 不还原 → torch_npu 流不在
    当前 context（107003）；ext 入口加 `ThreadCtxGuard`（SetDevice+退出还原）修复；
  - numba mask 单遍压缩 3.2→0.28ms（只比 x/y，与原版逐位一致）；
  - **设备常驻管线**：voxelization 输出保持 NPU tensor，`mkfeeds` 用 `aclruntime.BaseTensor`
    零拷贝直通，消除 D2H→numpy→H2D 往返；**GE 交接前 `torch.npu.synchronize()` 必需**
    （异步 op 不在 GE 流上，不同步 GE 读未写完内存 → 稳定 0 框）；
  - `MAX_NBLK 7→8`：kernel 7.11→6.24ms；demo/bench 补 `torch.npu.set_device` 对齐；
  - stage2 32.7→9.8ms；设备常驻 ON/OFF、nblk 7/8 框表均逐字节一致。
- **OM 图手术**（perf/om-opt `8b1e6d8`）：head 图手术 ABCD（1x1 conv 合并 / 冗余 gather 消除 /
  ConvTranspose→Conv1x1+DepthToSpace / TopK 4096→1024，数学恒等、ORT fp32 bit-exact），
  stage7 11.32→9.60ms（device 2 同设备对照），输出逐框一致（坐标 0.000977 / score 0.000122）。
  证伪：7 种 ATC 开关组合无增益、全量 pixelshuffle 重写慢 8 倍；CANN DepthToSpace mode 语义与
  ONNX 规范相反（规范 DCR 须写 'CRD'）。赢家 OM：`pointpillar_base_fp16_dynamic18000_topk_surgery_abcd_linux_aarch64.om`。
- **CPU 后处理**（perf/postproc `5eccc9b`）：collate 单帧 fast-path（0.15ms）+ numpy max/argmax
  （torch sigmoid 保留保 1ULP 一致）+ numba NMS 标量化（0.57→0.21ms）；stage3+5+9 14→3.35ms，逐位一致。
- **集成**（perf/integrated → master `3ef09d9`）：`collate_batch_fast` 升级双态（numpy / torch 设备
  常驻），bench `voxelize_tensors`/feeds 双态合并；最终单帧 E2E **22.5ms**（中位，device 1），33 框；
  `NPU_VOX_DEVICE_RESIDENT=0` 可整体回退 numpy 路径（A/B 23.7ms）。
- **10ms 结论**：当前 fp16 OM 路径不可达——体素化 kernel 6.24ms 标量排序墙 + backbone conv 44% +
  scatter 1.26ms；后续 int8 量化 / 自定义 scatter / kernel 向量化，乐观 ~13-15ms（PERFORMANCE.md §1.3/§7）。

### ✅ 体素化固定 AscendC：去除回退与开关（2026-09-29）

「用的哪个就是哪个」：AscendC 已验证全面不劣于 numba（demo 全量点 2.5x、FOV 小输入持平、
200 帧 OM AP 逐位一致），故 `_patch_voxelize_ascendc` 删除 `NPU_ASCENDC_VOXELIZE` 开关、
import try/except 与运行期回退——**唯一实现，任何帧都不换 numba**：

- unum_ops/OPP 缺失：`import npu_patch` 时直接 ImportError（unum_ops 为硬依赖，README §1）；
- 运行期异常（如旧 OPP 的 561000 context bug）：带原始栈抛出；
- 启动打印一行 `voxelize = AscendC NPU kernel`，日志可核对当前模式。

动机：静默回退曾把环境问题（旧 OPP、561000）伪装成「只是变慢」，RC/EP 新设备全量跑完才暴露。
改动：`npu/npu_patch.py`、`npu/README.md` §1、`npu/PERFORMANCE.md` §5.1。

### ✅ AscendC voxelize 启用（2026-09-28）

- **改动**：`npu_patch._patch_voxelize_ascendc`（unum_ops，commit 6b90b4e）从「仅 `init_patch()` 触发」改为
  **`import npu_patch` 时直接启用**——原路由一直休眠，生产链路实际跑 CPU numba（om_ref_demo/test/test_pt
  只 import npu_patch 从不调 init_patch）。09-29 起固定 AscendC、去除回退与开关（见下条）。
- **别名修复**：`_alias_spconv` 由 `setdefault` 改**强制赋值**——unum_ops 导入时会经 sparse_modules.py:229
  `import spconv.pytorch` 触发顶层 `import spconv`，产生与 `unum_ops.spconv` 同路径的**第二份拷贝**，
  setdefault 因拷贝已存在而静默失效，patch 落在 unum_ops 副本、下游拿到的是未 patch 的顶层副本（类不同名）。
- **性能**（demo 000008，M=7260，全量 122555 点）：generate **AscendC ~25ms vs numba ~45-53ms（2.5x）**。
  早前「无提速/更慢」结论为跨进程噪声 + wrapper `except Exception` **静默回退 numba** 假象（op 抛异常被吞掉），
  当时的修复是加回退日志；09-29 起彻底去除回退（见下条）。
- **精度**：voxel 逐帧**排序等价**（sorted-equal，coord 多重集相同、同 coord 特征/npp 相同）但行序与 numba 不同；
  200 帧中 199 帧 sorted-equal，frame 38 少 1 个越界边界体素（coord z=432 超出 BEV 网格，不影响输出）；
  **200 帧 OM AP 与 numba 基线逐位一致（diff=0）**。（早期「voxel diff=0 逐位一致」表述过于乐观，以本节口径为准。）
- **副作用修复**：AscendC 把 torch_npu 设备上下文拉进 aclruntime 进程，自然退出时双运行时 teardown 冲突
  segfault/bus error（结果已全部产出）；`om_ref_demo/om_ref_test` 结果输出后调 **`npu_patch.hard_exit(0)`** 硬退出。
- **kernel 级优化已实验为死路（2026-09-28）**：worktree（`omos/voxel-multicube`，/data/workspace/unum_ops）把
  `MAX_NBLK=7`→15 重建 + 安装 OPP 实测**无提速**（21.8 vs 21.3ms）——kernel 受 12 次软件栅障 + 全量 L1 dcci 串行化
  限制，**不是核数限制**（310P7 整颗 AIV 数小，nblk 被封顶）；且 nblk>7 会改变 voxel 行序。已回滚 vendor 并重验
  200 帧 AP 逐位一致。
- **OPP 安装/回滚教训**：CANN runtime 加载 `opp/vendors/` 下**全部** vendor 的 op_api/opmaster（字母序后者覆盖），
  `config.ini load_priority` 不控制 aclnn 符号解析；`.run` 安装会升级目标 vendor；回滚以 docker overlay2 diff 层为准，
  完成后须 `diff -rq` 核对 + 重验 200 帧 AP。
- PERFORMANCE.md 更新：5.1、1.3（getitem/E2E 重算为 est.）、7 方案 A、收益预估表。

### ✅ P2 topk 图内化已全量口径验证（2026-09-28）

- **topk OM**：`pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om`（图内 ReduceMax+TopK(4096)+
  Gather，输出 `topk_boxes (1,4096,7)`/`topk_cls (1,4096,3)`，D2H 13MB→164KB，无 ArgMax/NMS）。
- **精度**：200 帧 AP 与 base **逐位一致**（Car 77.81 / Ped 59.86 / Cyc 38.32；全量口径 77.07/51.93/61.95）。
- **性能**（全量口径 200 帧）：后处理 11.8→**4.0ms**，推理 21.3→22.0，E2E 67.5→**63.3ms/帧**。
- **单帧 demo 完整推理**（无 FOV，000008）：base 109ms → **topk 93ms**（getitem 体素化 122555 点占 58%）。
- **三种口径澄清**：demo 单帧完整（无 FOV）93ms / 推理链路（不含体素化）~39ms / 数据集全量（有 FOV）63.3ms。
- `om_ref_demo/test` 无需改码（2 输出 → base_mode 自动适配 4096）。
- PERFORMANCE.md 更新：新增 1.3 单帧完整延迟、1.4 topk 全量验证、topk OM 转换命令。

### ✅ P0' 已完成（2026-09-22）：msit 新增通用 knowledge `KnowledgeScatterNdToConcat`

- **根因**（profiler1 定位）：dense head 方向角修正 `box_preds[...,6] = dir_rot+...` 被 v2 导出追成
  **ScatterND（单核标量实现，单帧 125.1ms / 74.9%）**；旧导出（Sep 18 `pointpillar_nms.onnx`）用
  `Concat([Slice(:6), dir_corr], -1)`，无 ScatterND → 老静态 fp16 OM 才 15.7ms。v2 导出为回归。
- **实现**：`/data/workspace/msit/onnx_optimizer/src/onnx_optimizer/pattern/knowledges/
  knowledge_scatter_nd_to_concat.py`（已注册 `@KnowledgeFactory.register()`，`-k KnowledgeScatterNdToConcat` 可用）。
  - 匹配 ScatterND，**安全门禁**：迷你常量折叠 indices（Constant/Initializer→Unsqueeze/Concat/Reshape/
    Expand/Range/Cast/Gather/Shape/Where/Equal/ConstantOfShape），校验 = 各 leading 维 arange 网格（全覆盖、
    无碰撞）+ 尾部通道列常量 `k..k+K-1`；否则跳过（部分写入/动态索引不改写）。
  - 改写：`Concat([Slice(data,:k), updates, Slice(data,k+K:)], -1)`，结果 **bit 级一致**（onnxruntime 验证 max diff=0）。
- **产物**：`weights/pointpillar_nms_base_v2_dynamic_noscatter.onnx`（raw + 该 knowledge，ScatterND=0，223 节点）。
- 验证：正例 pointpillar bit 一致 ✅；负例（重复行/非常量通道/部分行）正确跳过 ✅。
- 注意：onnxsim / merge_convs 的 BN 折叠与 Conv 合并会引入 ~1e-4 浮点差（预存在行为），
  因此**精度门禁用 noscatter 版本（不叠加 onnxsim/merge）**，对已校验的 raw 基线保持 bit 一致。

### ✅ P1' 已完成（2026-09-23，profiler2 定位）：ArgMaxD 19.4ms → 消除

- profiler2（ScatterND 消除后）：前向 **185.75ms → 42.69ms**；host 调度仅 ~1ms；
  瓶颈变为 **ArgMaxD 19.4ms（46.6%）** = `dir_labels = torch.max(dir_cls, dim=-1)[1]`（输入 1×321408×2）。
- **根因**：ArgMaxD 在 310P 上单核标量实现；且导出时 `dir_cls.view(B, anchors, -1)` 用 `-1`，
  onnx shape inference 推不出 axis 维（head 空间维全未知）。
- **实现**：
  - msit 新增 `knowledge_argmax2_to_compare.py`（`KnowledgeArgMax2ToCompare`）：`ArgMax(x, axis)` 且
    `x.shape[axis]==2` → `Cast(Greater(x[...,1], x[...,0]), int64)`（元素级向量化；tie 语义与 argmax 一致）。
  - `export_onnx.py` 新增 `fix_dir_reshape_dim()`：导出后把 dir ArgMax 上游 Reshape 目标 `-1` 补成
    `cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS`，使 axis 维静态可证。
- **产物**：`weights/pointpillar_nms_base_v2_dynamic_noscatter_noargmax.onnx`（ScatterND=0、ArgMax=0，
  227 节点），与 raw 基线 **bit 一致**（onnxruntime max diff=0）。
- 预期：前向 42.7ms → ~25ms（ArgMaxD 19.4ms 消除）。

### ✅ P3 已就绪（2026-09-23）：fp16/mixed 静态 OM 转换脚本 + mixlist（推理 24 → ~12ms）

当前 fp32 静态 9000 OM 推理 **24ms**（200 帧 E2E 拆分，见上）。目标用 fp16/mixed 静态 OM
把推理降到 **~12ms**（精度容差 1% AP：Car 77.90 / Ped 57.95 / Cyc 37.05，3D moderate R11）。

- **产物（仅脚本 + mixlist，不在本机跑 ATC）**：
  - `npu/convert_fp16_static9000.sh`：生成 fp16/mixed 静态 OM + 全量 val 用 M≥17000 静态 OM 的命令。
  - `npu/mix_fp16_static9000.json`：white-list mixlist（参考老 mixed_optimized），脚本启动时
    `cp` 到 `weights/mix_fp16_static9000.json`（`weights` 是共享 symlink、不入 git，故 canonical 放 `npu/`）。
- **fp16 命令要点**（`weights/pointpillar_base_fp16_static9000_v2.om`，M=9000 静态）：
  ```
  atc --model=weights/pointpillar_nms_base_v2_dynamic_noscatter_noargmax.onnx --framework=5 \
      --soc_version=Ascend310P3 --output=weights/pointpillar_base_fp16_static9000_v2 \
      --input_format=ND --precision_mode_v2=mixed_float16 \
      --modify_mixlist=weights/mix_fp16_static9000.json \
      --input_shape="voxels:9000,32,4;voxel_num_points:9000;voxel_coords:9000,4;bev_index_map:214272"
  ```
- **mixlist 内容**（`white-list.to-add`，复用历史 15.7ms mixed_optimized 名单，
  VFE 关键 Mul/BN 不在名单内 → 保持 fp32，避免 force_fp16 丢框 24→23）：
  `StridedSliceD / ReduceSumD / ConcatD / GatherV2 / ConfusionTransposeD / AutomaticBufferFusionOp`
- **全量 val 用 M≥17000 静态 OM**（抽样 54% 帧 M>9000，最大 ~16664）：`pointpillar_base_fp32_static18000`
  （force_fp32，不动精度）+ `pointpillar_base_fp16_static18000`（mixed_float16 + 同 mixlist），
  `--input_shape="voxels:18000,32,4;voxel_num_points:18000;voxel_coords:18000,4;bev_index_map:214272"`。
- **验证命令**（在性能好的机器上）：
  `python npu/om_ref_test.py --om weights/pointpillar_base_fp16_static9000_v2.om --frames 200`
  全量 val：`python npu/om_ref_test.py --om weights/pointpillar_base_fp16_static18000.om`。
  门禁：3D moderate R11 三类与 fp32 基线差 ≤ ±1 AP。

> 备选：`--precision_mode=force_fp16` 更快但不保证精度（历史 24→23 丢框），仅对比测速用，不交付。

### 静态 OM（P0）待用户在大机器转换（noscatter onnx 已就绪）

```
# 动态（替换当前 173ms 的动态 om，ScatterND 消除后预计 ~60ms）
# 注意：必须用 range 记法 1~9000，勿用 -1 / --dynamic_dims（会把 M 当 batch 做 mbatch
# 切分，Concat_1 固定 batch=1 结构会报 E89999，见下"ATC 动态转换坑"）
atc --model=weights/pointpillar_nms_base_v2_dynamic_noscatter_noargmax.onnx --framework=5 \
    --soc_version=Ascend310P3 --output=weights/pointpillar_base_fp32_dynamic9000 \
    --input_format=ND --precision_mode=force_fp32 \
    --input_shape="voxels:1~9000,32,4;voxel_num_points:1~9000;voxel_coords:1~9000,4;bev_index_map:214272"

# 静态 M=9000（配合 P0 padding，省动态调度开销）
atc --model=weights/pointpillar_nms_base_v2_dynamic_noscatter_noargmax.onnx --framework=5 \
    --soc_version=Ascend310P3 --output=weights/pointpillar_base_fp32_static9000 \
    --input_format=ND --precision_mode=force_fp32 \
    --input_shape="voxels:9000,32,4;voxel_num_points:9000;voxel_coords:9000,4;bev_index_map:214272"
```

### ✅ P0/P1/P2 已完成（2026-09-23）——静态 9000 OM + 全 CPU 优化，demo 逐位一致

| 优化 | 内容 | 验证 |
|---|---|---|
| **P0** 静态 M=9000 + padding | `om_ref_demo.py`/`om_ref_test.py` 新增 `pad_to_static_m()`（pad 行 num_points=0、Gather 索引只引真实行、pad 值=9000，`build_index_map` 加 `pad` 参数） | 000008（M=7260→pad 9000）34 框与基线**逐位一致** |
| **P1** FOV 去 hstack | 新增 `fov_filter_fused()`：`points@A[:3]+A[3]` + `pts_rect@P2[:,:3].T+P2[:,3]`（同乘加顺序） | 20 帧 FOV mask **0 差异** |
| **P2** 后处理 numpy 化 | `1/(1+np.exp(-x))` + `argmax` 替代 torch.sigmoid/max | 34 框分数**逐位一致** |

**Demo 实测（000008，静态 9000 fp32 OM）**：推理 **173ms → 43ms**（P0' 去 ScatterND + P0 静态），检测结果与动态 fp32 基线逐位一致。

> 顺手修复：`ops_native` 重构后 `from npu.ops_native import boxes_iou_bev/_nms_iou_matrix` 失效 → 改为
> `npu.ops_native.iou3d_nms_torch_native`。

### ✅ 精度安全优化二轮（2026-09-23）：NMS 增量 / FOV 逐元素 / 后处理回退

| 优化 | 内容 | 验证 |
|---|---|---|
| **NMS 增量贪心** | `npu/ops_native/iou3d_nms_torch_native.py` 新增 `_nms_incremental()`：按 score 降序逐框只与**已保留框**算旋转 IoU（`_nms_inter` 逐对数学不变）→ O(N·K) 替代 O(N²) 全矩阵 | 6 组随机 + 真实框 **keep 集合完全一致**；N=4096 最坏 1672ms→**43ms（39x）** |
| **FOV 逐元素** | `fov_filter_fused` 改用 `_mm3()` 列式逐元素（本机 np.dot/@ 对小 K 走标量路径 ~81ms/次） | 200 帧 AP **77.90/57.95/37.05 与 bit-一致 FOV 完全相同** |
| **后处理回退 torch** | P2 的 `1/(1+np.exp(-x))` 在无有效 BLAS 机器上 **42ms（回归）**，回退 `torch.sigmoid`（~3ms） | 34 框与基线一致 |

**E2E 实测（200 帧，静态 9000 fp32 OM，本机）**：**186.8ms/帧**（基线 369ms → **2.0x**）
拆分：前处理 **113.5ms**（fov 59 + voxelize ~37 + pad ~22）+ 推理 **24ms** + 后处理 **49.5ms**。

### ✅ 并行优化三合一（2026-09-23，subagent × git worktree）——E2E **82.8ms/帧**（<100ms 达标）

| 任务 | 改动 | 实测 | AP 门禁 |
|---|---|---|---|
| A 前处理 | numba 体素化核心循环（unum_ops，voxel 全等）+ pad 缓冲复用 + FOV numba 内核（`_fov_filter_numba`）+ PIL 读图尺寸（49ms→0.2ms） | 前处理 113.5→**42ms** | 200帧 AP 与基线完全一致 |
| B 后处理 | numpy `argpartition` topk + `tensor_to_numpy(copy=False)` 免 13MB memcpy + sigmoid 单调性（`max sigmoid == sigmoid max`） | 后处理 49.5→**16.5ms** | 10 帧 preds 逐位一致 |
| C fp16 准备 | `npu/convert_fp16_static9000.sh`（mixed_float16 + mixlist 保 VFE）+ M≥17000 全量命令 | ✅ 已转已测：推理 24→**18ms**，AP 1% 内 | - |

### ✅ fp16/mixed 实测（2026-09-23）

- `weights/pointpillar_base_fp16_static9000_v2.om`（mixed_float16 + `weights/mix_fp16_static9000.json`，VFE 等关键层保 fp32）。
- 前向 **24 → 18ms**（mixlist 保关键层）；E2E 82.8 → **77.8ms/帧**。
- 200 帧 AP：**Car 77.76 / Ped 57.68 / Cyc 37.42**（fp32 基线 77.90/57.95/37.05，全部 1% 容差内 ✅）。
- 000008 检测 33 框 vs fp32 34 框（1 个低置信边界框被 fp16 变化吞掉，AP 无感）。
- **加白名单 Conv2D/Relu/Transpose/TransData（mix v2）无提速**（前向仍 18ms）：
  Conv2DTransposeD 已 fp16（7330→4234us），Conv2D 内存受限（3560→3335us，fp16 无益）；
  剩余 GatherV2/Transpose/TransData 为搬运瓶颈 → **fp16 空间已用尽**。

### ✅ force_fp16 实测（2026-09-23）——当前最优

- `weights/pointpillar_base_fp16_static9000_force.om`（`--precision_mode=force_fp16` 全图）。
- 前向 **17ms**，E2E **73.1ms/帧**（基线 369ms → **5.0x**）。
- 200 帧 AP：**Car 77.81 / Ped 59.86 / Cyc 38.32**（全部 1% 容差内 ✅；**Ped/Cyc 反升**，
  修正旧结论"force_fp16 必丢框"——旧记录针对带 ScatterND/ArgMax 的旧 onnx，当前 noscatter_noargmax base 无此问题）。
- `npu/convert_fp16_static9000.sh` 已把 force_fp16 升为主交付选项。

### ✅ P2：sigmoid+TopK 图内化（2026-09-23）——后处理 15→3.5ms

- **TopK 不调用 ArgMax**（ONNX/ATC 独立算子）；实测图内 `ReduceMax+TopK(321408→4096)+Gather` 仅 **+0.4ms**
  （ArgMaxD 才是慢的 ~19ms，已避开——类别标签留 CPU 从 top-4096 算）。
- `export_onnx.py` 新增 `--topk-only`：输出 `topk_boxes(1,4096,7)/topk_cls(1,4096,3)`（D2H 13MB→~160KB），**无 ArgMax/无 NMS**。
- `om_ref_demo/test` reshape 改 `(1,-1,7)` 兼容 top-K 输入。
- **实测（静态 9000 force_fp16，本机 200 帧）**：后处理 15.5→**3.5ms**，E2E 77.8→**64.3ms**；
  **AP 与 base force_fp16 完全一致（77.81/59.86/38.32）**（raw cls 单调排序 ≡ sigmoid 排序）。
- 全量 val 用：动态 18000 topk-only om（`weights/pointpillar_nms_base_v2_dynamic_topk.onnx` 转换，命令见对话）。

**合并后 200 帧官方评测（R11 3D moderate）**：**Car 77.90 / Ped 57.95 / Cyc 37.05** —— 与基线**完全一致**（bit 级精度保持）。
**E2E：82.8ms/帧**（前处理 42 + 推理 24 + 后处理 16.5），基线 369ms → **4.5x**，**达成 <100ms 目标**。

> 结论：NMS 内嵌 OM（图内 NonMaxSurppression/TopK）在 310P 上**输出垃圾**（top score 0.0046 vs 0.965、500 重复框），
> 动态/静态 shape 均复现 → 后处理**不能移入 OM**，CPU 后处理（16.5ms）为可靠路径。

#### NMS-in-OM 精确根因链（2026-09-23 复检，修正结论）

对照 `cann/ops-cv` 的 `non_max_suppression_v6`（其 README 明确：**该目录仅开源 aclnn host 接口，Ascend C kernel 闭源**，
最近提交记录了入参约束），定位到**两层代码级原因**：

1. **框数硬限制（已证实，修复有效）**：`aclnnNonMaxSuppression` 约束**每 batch 框数 ≤ 50000**。
   PointPillar NMS 输入 321408 anchors 远超限制 → 输出**完全垃圾**（全 0 索引、重复框、score ~0.005）。
   → `export_onnx.py` 图手术加 **pre-TopK(4096)** 后：**选择完全正确**（top5 score 0.9654/0.95/0.9284…、框 14.75/-1.07 等）。
2. **IoU 抑制失效（310P kernel 缺陷，无法绕过）**：框数合规后，NPU NonMaxSuppression **返回全部 max_out=500 框、无 IoU 抑制**
   （487 个 Car 重叠重复 vs CPU onnxruntime 正确 32 框）。即该算子 kernel 在 310P 上 IoU 计算失效；
   ops-cv 只开源 aclnn 接口，kernel 闭源且行为错误，无法修复。
   最小 NMS om（常量输入）同样执行失败。

**量化收益判断（即便抑制正常也不划算）**：图内 sigmoid+TopK+NMS 使前向 24ms → **47ms**（图内对 321408 做 sigmoid/topk 本身就贵），
加 ~2ms 后处理 ≈ 49ms，**劣于** base om 24ms + CPU 后处理 16.5ms = 40.5ms。→ **NMS 融合 OM 在此硬件上不可行且无收益**。
可行折中（如需减 D2H）：sigmoid+TopK 放图内输出 top-4096（~140KB vs 13MB），IoU NMS 留 Python。

> ⚠️ 注意：本机 openblas64 单线程对 (N,3)@(3,3) 小 K 矩阵乘走标量路径（~81ms/次），
> np.dot / np.einsum / torch.matmul 均非全 bit 一致；FOV 逐元素有 ~1 点/帧 边界翻转，
> 经 200 帧 AP 门禁判定**无精度影响**（预处理容忍微差，与 numba 体素化 4.5e-5 同性质）。
> 前向（模型输出）路径仍严格 bit 一致（ScatterND/ArgMax 消除，diff=0）。

### ⚠️ 全量 val 的 M 上限结论（静态 9000 覆盖不足）

抽样 100 帧 val：**54 帧 M>9000（最大 ~16664）**，与文档"val max 8567"（2026-08-18）不符
（当前 voxelization 产出更多 voxel）。→ **静态 9000 OM 不能用于全量 val**（会跳过 ~54%）。
全量 val 需转 **M≥17000 的静态 OM** 或 **动态 noscatter OM**（`1~9000` range 也覆盖不了 >9000 的帧，需加大 range）。
demo 单帧（M≤9000 的 bin）用静态 9000 即可。

#### ATC 动态转换坑（已解决）

`--input_shape` 用 `-1` + 多维 `--dynamic_dims`（如 `-1,32,4` + `1,1,1;500,500,500;...`）会把 M 当 **batch 维**
做 mbatch 切分；dense head `Concat_1` 是固定 batch=1 结构（axis=0 拼 `Squeeze(ReduceMax)` 的 `[64]` 与常量
`[1,64]`），按 batch 切分时报 `E89999: input shape dims should be equal except merge axis`。
解决：M 用 **range 记法 `1~9000`** 且不带 `--dynamic_dims`（与 08:23 已验证的动态 om 同格式）。

### 剩余待办

| 优先级 | 方案 | 内容 | 预期 | 精度 |
|---|---|---|---|---|
| （P0/P1/P2 已完成，见上） | - | - | - | - |
| 全量 val | 转 M≥17000 静态 OM 或 1~17000 动态 noscatter OM | val 有 54% 帧 M>9000，静态 9000 覆盖不足 | 全量推理 ~43ms/帧 | ✅ |
| （不做） | FOV 单矩阵乘融合 / fp16 | 有 0.48% 边界差异 / 量化误差 | - | ❌ 违反 bit 一致 |

验证门禁：`--frames 200` AP 与现全量逐位一致；`compare_pt_om` diff=0；`perf_e2e.py` 分段计时。

---

## v1.2.0（2026-09-22）：OM 动态 fp32 全量 3769 帧官方评测

### 变更
- `npu/run_bin.py` → `npu/om_ref_demo.py`（重构为 tools/demo.py 同构：DemoDataset + cfg_file + 逐样本打印）；
- `npu/eval_kitti_full.py` → `npu/om_ref_test.py`（tools/test.py 的 OM 版，全量 val 推理 + 官方评测对接）；
- 新增 `npu/export_onnx.py --dynamic`：导出 M 动态 base ONNX；图手术节点名不再与输出张量重名（auto_optimizer 兼容）；
- 新增 `weights/pointpillar_nms_base_v2_dynamic_opt.onnx`（onnxsim 426→105 + auto_optimizer 104 节点）；
- 新增动态 fp32 OM：`weights/pointpillar_base_fp32_dynamic_linux_aarch64_linux_aarch64.om`（142MB，M 动态 1~9000）。

### 结论（KITTI val 全量 3769 帧，OM 动态 fp32，官方评测）

| class | 3D moderate R11 | R40 | 官方基线（R11） | 差距 |
|---|---|---|---|---|
| Car | 77.25 | 78.33 | 77.28 | -0.03 ✅ |
| Pedestrian | 51.67 | 50.90 | 52.29 | -0.62 |
| Cyclist | 61.76 | 62.04 | 62.68 | -0.92 |

- 全量 `skipped_M=0`，avg 368.8ms/帧（本机，前处理/推理/后处理拆分见 PERFORMANCE.md）。
- **NMS 内嵌图（图内 NonMaxSurppression）经 ATC 转 OM 后 NPU 输出错误**（top score 0.024、大量重复框）；
  onnxruntime CPU 验证 raw/sim/opt 三版均正确 → 根因在 ATC/NPU 对 NMS 算子的执行，base OM + Python 侧 NMS 为可靠路径。

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

> 设备修复前基线【2026-09-18~21】：设备 rotate_iou（CPU box_overlap_bev 版）评测 Car 3D moderate ≈ **64.16**
> （偏低 13 点，官方同输入 ≈ 77.83），根因为 rotate_iou CPU 数学与官方 CUDA 不等价，见上表修复。

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
- om_ref_demo.py 在计时前用 `_nms_iou_matrix(np.zeros((2,7),np.float32))` 预热（首调 ~1.5s）。

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
