#!/bin/bash
# NPU 适配验证脚本：环境检查 -> 补丁检查 -> 推理验证 -> 任务评测
# 用法: bash npu/verify_npu.sh [--frames N] [--preds DIR]

set -e
cd "$(dirname "$0")/.."

FRAMES=5
PREDS_DIR=/tmp/verify_preds
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames) FRAMES="$2"; shift 2;;
        --preds) PREDS_DIR="$2"; shift 2;;
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
echo "=== 2. 必需交付物存在性 / 语法检查 ==="
for f in npu/infer.py npu/eval.py npu/npu_patch.py; do
    if [ ! -f "$f" ]; then
        echo "FAIL: $f 缺失"; exit 1
    fi
    python -m py_compile "$f" && echo "OK  语法通过: $f"
done

echo
echo "=== 3. npu_patch.py 初始化入口可重复调用 ==="
python - <<'PY'
from npu.npu_patch import init_patch, get_device
d1 = init_patch()
d2 = init_patch()
print("device =", get_device(), "| init 可重复调用 OK")
assert d1 == d2
PY

echo
echo "=== 4. 推理验证: infer.py (frames=%d) ===" % "$FRAMES"
rm -rf "$PREDS_DIR"
python npu/infer.py --device auto --frames "$FRAMES" --save-preds "$PREDS_DIR"
N_FILES=$(ls "$PREDS_DIR" | wc -l)
echo "OK  推理产出 $N_FILES 个预测文件"
[ "$N_FILES" -gt 0 ] || { echo "FAIL: 无预测文件"; exit 1; }

echo
echo "=== 5. 任务评测: eval.py ==="
python npu/eval.py --preds "$PREDS_DIR" --frames "$FRAMES" | tail -40

echo
echo "=== 完成 ==="
