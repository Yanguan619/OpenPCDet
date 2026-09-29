"""perf/postproc 优化逐位一致性验证：collate fast-path vs 通用 collate；numpy 后处理 vs torch 参考。

验证点：
1. collate_batch_fast 与 DatasetTemplate.collate_batch 单帧输出：voxels / voxel_num_points /
   voxel_coords（+batch 列）逐位一致（points 为文档化偏差，跳过对比）。
2. postprocess_topk（numpy max/argmax + torch sigmoid）与 torch 参考
   （torch.max + label+1 + torch.sigmoid + nms_topk_numpy）：框数、每框 class/coords/score
   完全一致（行序按 label+score 排序后对比，兼容并列分数次序差异）。

用法:
    BENCH_DEVICE=3 python3 npu/debug/verify_postproc_bitwise.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import npu.npu_patch  # noqa: E402,F401

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

from npu.om_ref_demo import (DemoDataset, build_index_map, tensor_to_numpy, nms_topk_numpy,
                             collate_batch_fast, postprocess_topk)

BIN = ROOT / '000008.bin'
CFG = ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'
DEVICE = int(os.environ.get('BENCH_DEVICE', '3'))

os.chdir(str(ROOT / 'tools'))
cfg_from_yaml_file(str(CFG), cfg)
logger = common_utils.create_logger()
demo_dataset = DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=BIN, logger=logger
)
nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
score_thresh = cfg.MODEL.POST_PROCESSING.SCORE_THRESH

points = np.fromfile(str(BIN), dtype=np.float32).reshape(-1, 4)
dd = demo_dataset.prepare_data(data_dict={'points': points.copy(), 'frame_id': 0, 'use_lead_xyz': True})

ok = True


def check(name, a, b, allow_order=False):
    global ok
    if a.shape != b.shape:
        print('  [FAIL] %s: shape %s != %s' % (name, a.shape, b.shape))
        ok = False
        return
    if allow_order:
        # 按字典序排序后对比（兼容行序/并列分数差异）
        if np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0)):
            print('  [PASS] %s（排序后逐位一致，shape=%s）' % (name, a.shape))
        else:
            print('  [FAIL] %s: 排序后存在不一致' % name)
            ok = False
    else:
        if np.array_equal(a, b):
            print('  [PASS] %s（逐位一致，shape=%s）' % (name, a.shape))
        else:
            print('  [FAIL] %s: 逐位不一致，maxdiff=%r' % (name, np.abs(a - b).max() if np.issubdtype(a.dtype, np.number) else 'n/a'))
            ok = False


print('=== 1. collate fast-path vs 通用 collate ===')
d_fast = collate_batch_fast([dd])
d_generic = demo_dataset.collate_batch([dd])
for k in ['voxels', 'voxel_num_points', 'voxel_coords']:
    check('collate[%s]' % k, d_fast[k], d_generic[k])
check('batch_size', np.asarray(d_fast['batch_size']), np.asarray(d_generic['batch_size']))
print('  注: points 为文档化偏差（fast 返回 (N,4) 原引用，通用返回 (N,5)），OM 路径不消费，跳过对比')

print('=== 2. OM 推理（取一次输出做后处理对比） ===')
import aclruntime
from aclruntime import InferenceSession
OM = ROOT / 'weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om'
session = InferenceSession(str(OM), DEVICE, aclruntime.session_options())
out_names = [d.name for d in session.get_outputs()]

voxels = d_fast['voxels']
vnp = d_fast['voxel_num_points']
vco = d_fast['voxel_coords']
bim = build_index_map(vco, M=voxels.shape[0]).numpy()
M = voxels.shape[0]
fs = [aclruntime.Tensor(np.ascontiguousarray(voxels)),
      aclruntime.Tensor(np.ascontiguousarray(vnp)),
      aclruntime.Tensor(np.ascontiguousarray(vco)),
      aclruntime.Tensor(np.ascontiguousarray(bim))]
for t in fs:
    t.to_device(DEVICE)
dym = ['%s:%s' % (inp.name, ','.join(str(M if d <= 0 else d) for d in inp.shape))
       for inp in session.get_inputs()]
session.set_dynamic_shape(';'.join(dym))
out_size = []
for o in session.get_outputs():
    n = 1
    for d in o.shape:
        n *= max(d, 1)
    out_size.append(n * 4 * 4)
session.set_custom_outsize(out_size)
out = session.run(out_names, fs)
om_box = tensor_to_numpy(out[0], np.float32, copy=False).reshape(1, -1, 7)
om_cls = tensor_to_numpy(out[1], np.float32, copy=False).reshape(1, -1, 3)

# 新路径
boxes_new, labels_new, scores_new = postprocess_topk(om_box, om_cls, score_thresh, nms_config)

# torch 参考（优化前原写法）
c = om_cls.reshape(1, -1, 3)[0]
cls_max, label = torch.max(torch.from_numpy(c), dim=-1)
label = label + 1
scores = torch.sigmoid(cls_max)
selected, s2 = nms_topk_numpy(om_box.reshape(1, -1, 7)[0], scores, score_thresh, nms_config)
boxes_ref = om_box.reshape(1, -1, 7)[0][selected]
labels_ref = label[selected].numpy()

print('  框数 new=%d ref=%d' % (len(boxes_new), len(boxes_ref)))
check('boxes', boxes_new, boxes_ref, allow_order=True)
check('labels', labels_new, labels_ref, allow_order=True)
check('scores', scores_new, s2, allow_order=True)

# 严格版：新路径与参考按 (label, score) 对齐行序后逐位对比
if boxes_new.shape == boxes_ref.shape:
    order_new = np.lexsort((scores_new, labels_new))
    order_ref = np.lexsort((s2, labels_ref))
    aligned = (np.array_equal(boxes_new[order_new], boxes_ref[order_ref])
               and np.array_equal(labels_new[order_new], labels_ref[order_ref])
               and np.array_equal(scores_new[order_new], s2[order_ref]))
    print('  [%s] 行序对齐后逐位一致（按 label+score 排序）' % ('PASS' if aligned else 'FAIL'))
    ok = ok and aligned

# 校验 sigmoid 中间量本身：postprocess_topk 内部 cls_max 与 torch.max 值一致
cls_max_new = c.max(axis=-1)
print('  [%s] numpy.max 与 torch.max 值逐位一致' % ('PASS' if np.array_equal(cls_max_new, cls_max.numpy()) else 'FAIL'))
print('  [%s] numpy.argmax+1 与 torch label 逐位一致' % ('PASS' if np.array_equal(c.argmax(axis=-1) + 1, label.numpy()) else 'FAIL'))

print()
print('RESULT:', 'ALL PASS' if ok else 'FAILED')
sys.exit(0 if ok else 1)
