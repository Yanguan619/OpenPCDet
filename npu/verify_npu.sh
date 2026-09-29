#!/bin/bash
# NPU 适配验证脚本：环境检查 -> 补丁检查 -> 推理验证 -> 简化评测
# 用法: bash npu/verify_npu.sh [--frames N] [--bin <demo.bin>]
# 前提: 已按 npu/README.md 完成依赖安装（unum_ops + OPP）、OM 生成、KITTI 数据准备。

set -e
cd "$(dirname "$0")/.."

FRAMES=5
BIN=data/kitti/training/velodyne/000008.bin
OM=weights/pointpillar_base_fp16_dynamic18000_topk_surgery_abc_linux_aarch64.om
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames) FRAMES="$2"; shift 2;;
        --bin) BIN="$2"; shift 2;;
        *) echo "未知参数: $1"; exit 1;;
    esac
done

echo "=== 1. 环境检查 ==="
python - <<'PY'
import sys
print("python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__)
if getattr(torch, "npu", None) is not None and torch.npu.is_available():
    import torch_npu
    print("torch_npu:", torch_npu.__version__, "| npu devices:", torch.npu.device_count())
else:
    print("torch_npu: 未安装 / NPU 不可用（当前设备: cuda=%s, cpu）" % torch.cuda.is_available())
PY

echo
echo "=== 2. 核心文件存在性 / 语法检查 ==="
for f in npu/npu_patch.py npu/om_ref_demo.py npu/om_ref_test.py npu/surgery_heads.py; do
    if [ ! -f "$f" ]; then
        echo "FAIL: $f 缺失"; exit 1
    fi
    python -m py_compile "$f" && echo "OK  语法通过: $f"
done

echo
echo "=== 3. npu_patch 初始化（voxelize 应为 AscendC，无回退） ==="
python - <<'PY'
from npu.npu_patch import init_patch, get_device
d1 = init_patch()
d2 = init_patch()
print("device =", get_device(), "| init 可重复调用 OK")
assert d1 == d2
PY

echo
echo "=== 4. 单帧 demo 推理 ==="
if [ ! -f "$BIN" ]; then
    echo "SKIP: $BIN 不存在（先按 README 准备 KITTI 数据，或 --bin 指定 .bin 路径）"
    exit 0
fi
if [ ! -f "$OM" ]; then
    echo "SKIP: $OM 不存在（先按 README §生成 OM 完成 ckpt→ONNX→手术→ATC）"
    exit 0
fi
python npu/om_ref_demo.py --data_path "$BIN"

echo
echo "=== 5. 简化评测（$FRAMES 帧，--quick） ==="
python npu/om_ref_test.py --frames "$FRAMES" --quick

echo
echo "=== 完成 ==="
