"""OM 纯推理循环脚本（供 msprof 剖析 stage7 OM 推理热点）。

只循环 session.run（stage7 口径），前处理/feeds 构造只做一次（移出剖析窗口）。
feeds 必须为 list[aclruntime.Tensor] 且每个 .to_device()（dict 不带 to_device 会 MTE DDR 越界崩溃）。

用法:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 npu/debug/om_loop.py [--om <model.om>] [--device 2] [--iters 20]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import npu.npu_patch  # noqa: E402,F401

import aclruntime
import numpy as np
import torch
from aclruntime import InferenceSession
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from npu.om_ref_demo import (DemoDataset, to_tensor, build_index_map, tensor_to_numpy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--om', default=str(ROOT / 'weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om'))
    ap.add_argument('--device', type=int, default=2)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--bin', default=str(ROOT / '000008.bin'))
    args = ap.parse_args()

    os.chdir(str(ROOT / 'tools'))
    cfg_from_yaml_file(str(ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'), cfg)
    logger = common_utils.create_logger()

    session = InferenceSession(args.om, args.device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    is_dynamic = expect_m is None or expect_m <= 0

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.bin), logger=logger
    )
    dd = demo_dataset.prepare_data(data_dict={
        'points': np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4).copy(),
        'frame_id': 0, 'use_lead_xyz': True})
    data_dict = demo_dataset.collate_batch([dd])

    voxels = to_tensor(data_dict['voxels'])
    vnp = to_tensor(data_dict['voxel_num_points'])
    vco = to_tensor(data_dict['voxel_coords'].astype(np.int32))
    M = voxels.shape[0]
    bim = build_index_map(vco, M=M) if is_dynamic else None
    print('M=%d dynamic=%s expect_m=%s' % (M, is_dynamic, expect_m))

    feeds = [aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
             aclruntime.Tensor(np.ascontiguousarray(vnp.numpy())),
             aclruntime.Tensor(np.ascontiguousarray(vco.numpy())),
             aclruntime.Tensor(np.ascontiguousarray(bim.numpy()))]
    for t in feeds:
        t.to_device(args.device)

    if is_dynamic:
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

    # 预热 2 次（编译/首跑抖动移出）
    for _ in range(2):
        session.run(out_names, feeds)

    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        out = session.run(out_names, feeds)
        times.append((time.perf_counter() - t0) * 1000)
        # 轻量校验：输出 shape 稳定
        if i == 0:
            print('out shapes:', [o.shape for o in out])

    times = sorted(times)
    print('OM run: iters=%d  mean=%.3f ms  median=%.3f ms  min=%.3f ms  max=%.3f ms'
          % (len(times), sum(times) / len(times), times[len(times) // 2], times[0], times[-1]))


if __name__ == '__main__':
    main()
