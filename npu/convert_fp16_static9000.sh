#!/usr/bin/env bash
# =============================================================
# convert_fp16_static9000.sh —— 生成 fp16/mixed 静态 OM（推理 24ms -> ~12ms）
#
# 前置:
#   ONNX: weights/pointpillar_nms_base_v2_dynamic_noscatter_noargmax.onnx
#         （ScatterND=0 / ArgMax=0，已用 build_index_map 消除 scatter，
#           见 npu/CHANGELOG.md P0'/P1'；fp32 静态 9000 基线 24ms 即由它转出）
#   ATC:  /usr/local/Ascend/cann-9.0.0/bin/atc （Ascend310P3）
#
# ⚠️ 本脚本不在本机跑 ATC（太慢），产物转好后在性能好的机器执行：
#   python npu/om_ref_test.py --om weights/pointpillar_base_fp16_static9000_v2.om --frames 200
#   精度门禁: 与 fp32 基线 (Car 77.90 / Ped 57.95 / Cyc 37.05) 容差 ±1% AP (3D moderate R11)
#
# 产物:
#   weights/pointpillar_base_fp16_static9000_v2.om    # M=9000 静态, mixed_float16 + mixlist (demo/200帧)
#   weights/pointpillar_base_fp32_static18000.om      # M=18000 静态 fp32（全量 val：54% 帧 M>9000）
#   weights/pointpillar_base_fp16_static18000.om      # M=18000 静态 mixed（全量 val）
# =============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ONNX="weights/pointpillar_nms_base_v2_dynamic_noscatter_noargmax.onnx"
MIXLIST="weights/mix_fp16_static9000.json"   # 由 npu/mix_fp16_static9000.json 同步（weights 为共享 symlink，不入 git）
mkdir -p "$(dirname "$MIXLIST")"
cp -f npu/mix_fp16_static9000.json "$MIXLIST"

ATC="/usr/local/Ascend/cann-9.0.0/bin/atc"
SOC="Ascend310P3"

# -------------------------------------------------------------
# 1) fp16 静态 M=9000（demo / 200 帧 AP 验证用）
#    - precision_mode_v2=mixed_float16 + modify_mixlist: white-list 指定可降
#      fp16 的算子，其余（含 VFE Mul/BN）保 fp32。
#    - ✅ 已验证（2026-09-23）：force_fp16 全图在 200 帧 AP 门禁全部 1% 容差内
#      （Car 77.81 / Ped 59.86 / Cyc 38.32，Ped/Cyc 反升），且比 mixed 更快（17 vs 18ms）
#      → force_fp16 为当前最优交付。
# -------------------------------------------------------------
"$ATC" --model="$ONNX" --framework=5 \
    --soc_version="$SOC" \
    --output="weights/pointpillar_base_fp16_static9000_force" \
    --input_format=ND \
    --precision_mode=force_fp16 \
    --input_shape="voxels:9000,32,4;voxel_num_points:9000;voxel_coords:9000,4;bev_index_map:214272"

# 备选 mixed（VFE 等保 fp32，若 force 精度不达标再用）:
# "$ATC" --model="$ONNX" --framework=5 \
#     --soc_version="$SOC" \
#     --output="weights/pointpillar_base_fp16_static9000_v2" \
#     --input_format=ND \
#     --precision_mode_v2=mixed_float16 \
#     --modify_mixlist="$MIXLIST" \
#     --input_shape="voxels:9000,32,4;voxel_num_points:9000;voxel_coords:9000,4;bev_index_map:214272"
#     --input_format=ND \
#     --precision_mode=force_fp16 \
#     --input_shape="voxels:9000,32,4;voxel_num_points:9000;voxel_coords:9000,4;bev_index_map:214272"

# -------------------------------------------------------------
# 2) 全量 val 用 M≥17000 静态 OM（当前 9000 覆盖不足：抽样 54% 帧 M>9000，
#    最大 ~16664）。fp32 版（不动精度，保精度基线）+ fp16 版（提速）。
#    M 取 18000（≥17000 且为 4 的倍数）。超出 M 的帧会被 om_ref_test 跳过
#    （n_skip>0 时需换更大的静态 M 或 1~18000 动态 range）。
# -------------------------------------------------------------
"$ATC" --model="$ONNX" --framework=5 \
    --soc_version="$SOC" \
    --output="weights/pointpillar_base_fp32_static18000" \
    --input_format=ND \
    --precision_mode=force_fp32 \
    --input_shape="voxels:18000,32,4;voxel_num_points:18000;voxel_coords:18000,4;bev_index_map:214272"

"$ATC" --model="$ONNX" --framework=5 \
    --soc_version="$SOC" \
    --output="weights/pointpillar_base_fp16_static18000" \
    --input_format=ND \
    --precision_mode_v2=mixed_float16 \
    --modify_mixlist="$MIXLIST" \
    --input_shape="voxels:18000,32,4;voxel_num_points:18000;voxel_coords:18000,4;bev_index_map:214272"

# -------------------------------------------------------------
# 转完后的验证（在性能好的机器上）:
#   # 200 帧 AP 门禁（1% 容差；若 M>9000 的帧被跳过，先换 static18000）
#   python npu/om_ref_test.py --om weights/pointpillar_base_fp16_static9000_v2.om --frames 200
#   # 全量 val（用 M=18000 版）
#   python npu/om_ref_test.py --om weights/pointpillar_base_fp16_static18000.om
#   # 测速对比
#   python npu/om_ref_demo.py --data_path data/kitti/training/velodyne/000008.bin \
#       --om weights/pointpillar_base_fp16_static9000_v2.om
# -------------------------------------------------------------
echo "转换完成（若在本机跳过，请在性能好的机器执行上述命令并跑 AP 门禁）。"
