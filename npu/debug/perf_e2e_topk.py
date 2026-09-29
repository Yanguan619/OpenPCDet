"""E2E 分段计时（topk OM + DemoDataset，demo-only 环境无需 KittiDataset/data 目录）。

用法:
    BENCH_DEVICE=1 BENCH_ITERS=30 python3 npu/debug/perf_e2e_topk.py
    # 静态 OM（expect_m>0）自动走 pad 分支
    # NPU_VOX_DEVICE_RESIDENT=0 可关闭设备常驻管线（回退 numpy 路径做 A/B）
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# 设备常驻管线：voxelization 输出保持 NPU tensor（collate/index_map/feeds 全程
# device 侧），消除 D2H→numpy→collate→H2D 往返。须在 import npu_patch 前设置。
os.environ.setdefault('NPU_VOX_DEVICE_RESIDENT', '1')
import npu.npu_patch  # noqa: E402,F401  预注入 stub，必须在 import pcdet 之前

import aclruntime
import numpy as np
import torch
from aclruntime import InferenceSession
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from npu.om_ref_demo import (DemoDataset, to_tensor, build_index_map, pad_to_static_m,
                             tensor_to_numpy, nms_topk_numpy, mkfeeds, coords_int32)

OM = ROOT / 'weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om'
BIN = ROOT / '000008.bin'
CFG = ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'
DEVICE = int(os.environ.get('BENCH_DEVICE', '0'))
ITERS = int(os.environ.get('BENCH_ITERS', '30'))
SKIP = 3  # 预热帧不计入统计


def main():
    # 必修：patched generate() 的 .npu() 默认走 device 0，必须与 BENCH_DEVICE
    # 对齐（否则算子跑在 device 0、OM 跑在 device 1，跨设备错位）。
    if getattr(torch, 'npu', None) is not None and torch.npu.is_available():
        torch.npu.set_device(DEVICE)
    os.chdir(str(ROOT / 'tools'))  # 与 demo 一致：_BASE_CONFIG_ 相对 tools/ 解析
    cfg_from_yaml_file(str(CFG), cfg)
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=BIN, logger=logger
    )
    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = cfg.MODEL.POST_PROCESSING.SCORE_THRESH

    # numba 预热（移出计时区间）
    try:
        from npu.ops_native.iou3d_nms_torch_native import _nms_iou_matrix, _nms_incremental
        _nms_iou_matrix(np.zeros((2, 7), dtype=np.float32))
        _nms_incremental(np.zeros((2, 7), dtype=np.float32), 0.01)
    except Exception:
        pass

    session = InferenceSession(str(OM), DEVICE, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    is_dynamic = expect_m is None or expect_m <= 0

    stage = {}

    def timed(name, fn, record=True):
        t0 = time.perf_counter()
        r = fn()
        if record:
            stage[name] = stage.get(name, 0.0) + (time.perf_counter() - t0)
        return r

    e2e_times = []
    n_boxes = 0
    for i in range(ITERS):
        rec = i >= SKIP
        frame_t0 = time.perf_counter()

        points = timed('1.读bin', lambda: np.fromfile(str(BIN), dtype=np.float32).reshape(-1, 4), rec)
        dd = timed('2.prepare_data(mask+体素化)',
                   lambda: demo_dataset.prepare_data(data_dict={
                       'points': points.copy(), 'frame_id': 0, 'use_lead_xyz': True}), rec)
        data_dict = timed('3.collate_batch', lambda: demo_dataset.collate_batch([dd]), rec)

        def voxelize_tensors():
            voxels = to_tensor(data_dict['voxels'])
            vnp = to_tensor(data_dict['voxel_num_points'])
            vco = to_tensor(coords_int32(data_dict['voxel_coords']))
            if is_dynamic:
                bim = build_index_map(vco, M=voxels.shape[0])
            else:
                if voxels.shape[0] > expect_m:
                    raise RuntimeError('M=%d 超过静态 OM M=%d' % (voxels.shape[0], expect_m))
                voxels, vnp, vco, bim = pad_to_static_m(voxels, vnp, vco, expect_m)
            return voxels, vnp, vco, bim
        voxels, vnp, vco, bim = timed('4.tensor+index_map/pad', voxelize_tensors, rec)
        M = voxels.shape[0]

        feeds = timed('5.feeds构造+to_device(H2D)',
                      lambda: mkfeeds(voxels, vnp, vco, bim, DEVICE), rec)

        if is_dynamic:
            def setdym():
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
            timed('6.set_dynamic_shape', setdym, rec)

        out = timed('7.OM推理', lambda: session.run(out_names, feeds), rec)

        def d2h():
            return (tensor_to_numpy(out[0], np.float32, copy=False),
                    tensor_to_numpy(out[1], np.float32, copy=False))
        om_box, om_cls = timed('8.输出D2H', d2h, rec)

        def post():
            b = om_box.reshape(1, -1, 7)
            c = om_cls.reshape(1, -1, 3)
            cls_max, label = torch.max(torch.from_numpy(c[0]), dim=-1)
            label = label + 1
            scores = torch.sigmoid(cls_max)
            selected, s2 = nms_topk_numpy(b[0], scores, score_thresh, nms_config)
            return b[0][selected], label[selected].numpy(), s2
        boxes, labels, scores = timed('9.后处理(max+sigmoid+NMS)', post, rec)

        e2e_times.append((time.perf_counter() - frame_t0) * 1000)
        if i == SKIP:
            n_boxes = len(boxes)
            base_boxes = (boxes.copy(), labels.copy(), scores.copy())

    n_stat = ITERS - SKIP
    order = ['1.读bin', '2.prepare_data(mask+体素化)', '3.collate_batch', '4.tensor+index_map/pad',
             '5.feeds构造+to_device(H2D)', '6.set_dynamic_shape', '7.OM推理', '8.输出D2H',
             '9.后处理(max+sigmoid+NMS)']
    print('=' * 62)
    print('E2E 分段计时  frame=%s  M=%d  device=%d  %s  iters=%d(计%d)'
          % (BIN.name, M, DEVICE, 'dynamic' if is_dynamic else 'static%d' % expect_m, ITERS, n_stat))
    print('=' * 62)
    tot = 0.0
    for k in order:
        v = stage.get(k, 0.0) / n_stat * 1000
        tot += v
        print('%-34s %9.3f ms' % (k, v))
    es = sorted(e2e_times[SKIP:])
    print('-' * 62)
    print('%-34s %9.3f ms' % ('稳态 E2E 均值', sum(es) / len(es)))
    print('%-34s %9.3f ms' % ('稳态 E2E 中位', es[len(es) // 2]))
    print('%-34s %9.3f ms' % ('分段累加', tot))
    print('检测框数(1帧):', n_boxes)


if __name__ == '__main__':
    main()
