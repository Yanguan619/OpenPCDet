"""NPU OM 输出对比验证：跑一次 OM，dump 原始输出 + 后处理结果，与基线引用逐框对比。

用途：fp16 变体（ATC 开关 / 图手术版）与基线 OM 在同一 device 上的输出一致性验证。
红线：框数一致（≈33）、类别分布一致、坐标偏差 < 0.05；score 允许微小抖动但记录最大偏差。

用法:
    # 生成基线引用（只需一次）
    python3 npu/debug/compare_om_npu.py --om weights/pointpillar_base_fp16_dynamic18000_topk_linux_aarch64.om --tag base
    # 待验变体（与 base 引用对比）
    python3 npu/debug/compare_om_npu.py --om om_out/dyn_fp16_surgery_abc_linux_aarch64.om --tag surgery_abc --ref base
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import npu.npu_patch  # noqa: E402,F401

import aclruntime  # noqa: E402
from aclruntime import InferenceSession  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from npu.om_ref_demo import (DemoDataset, to_tensor, build_index_map,  # noqa: E402
                             tensor_to_numpy, nms_topk_numpy)

OUTDIR = Path('/tmp/opencode/omcmp')
CFG = ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'


def run_once(om_path, device, bin_path):
    os.chdir(str(ROOT / 'tools'))
    cfg_from_yaml_file(str(CFG), cfg)
    logger = common_utils.create_logger()
    session = InferenceSession(str(om_path), device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    is_dynamic = expect_m is None or expect_m <= 0

    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                               training=False, root_path=Path(bin_path), logger=logger)
    dd = demo_dataset.prepare_data(data_dict={
        'points': np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4).copy(),
        'frame_id': 0, 'use_lead_xyz': True})
    data_dict = demo_dataset.collate_batch([dd])
    voxels = to_tensor(data_dict['voxels'])
    vnp = to_tensor(data_dict['voxel_num_points'])
    vco = to_tensor(data_dict['voxel_coords'].astype(np.int32))
    M = voxels.shape[0]
    bim = build_index_map(vco, M=M) if is_dynamic else None

    feeds = [aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
             aclruntime.Tensor(np.ascontiguousarray(vnp.numpy())),
             aclruntime.Tensor(np.ascontiguousarray(vco.numpy()))]
    if bim is not None:
        feeds.append(aclruntime.Tensor(np.ascontiguousarray(bim.numpy())))
    for t in feeds:
        t.to_device(device)

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

    out = session.run(out_names, feeds)
    om_box = tensor_to_numpy(out[0], np.float32, copy=False)
    om_cls = tensor_to_numpy(out[1], np.float32, copy=False)
    return om_box, om_cls


def post(om_box, om_cls):
    b = om_box.reshape(1, -1, 7)
    c = om_cls.reshape(1, -1, 3)
    cls_max, label = torch.max(torch.from_numpy(c[0]), dim=-1)
    label = label + 1
    scores = torch.sigmoid(cls_max)
    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    selected, s2 = nms_topk_numpy(b[0], scores, score_thresh, nms_config)
    return b[0][selected], label[selected].numpy(), s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--om', required=True)
    ap.add_argument('--tag', required=True, help='本次运行标签（输出 npy 前缀）')
    ap.add_argument('--ref', default=None, help='对比引用标签（须已生成）')
    ap.add_argument('--device', type=int, default=2)
    ap.add_argument('--bin', default=str(ROOT / '000008.bin'))
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    om_path = str(Path(args.om).resolve())
    om_box, om_cls = run_once(om_path, args.device, args.bin)
    boxes, labels, scores = post(om_box, om_cls)
    np.save(OUTDIR / ('%s_boxes_raw.npy' % args.tag), om_box)
    np.save(OUTDIR / ('%s_cls_raw.npy' % args.tag), om_cls)
    np.save(OUTDIR / ('%s_det.npy' % args.tag),
            np.concatenate([boxes, labels[:, None].astype(np.float32), scores[:, None]], axis=1))
    cls_dist = np.bincount(labels, minlength=4).tolist()
    print('[%s] 原始输出 box%s cls%s -> 检测框 %d 个, 类别分布(0bg/1car/2ped/3cyc) %s'
          % (args.tag, om_box.shape, om_cls.shape, len(boxes), cls_dist))

    if args.ref:
        ref = np.load(OUTDIR / ('%s_det.npy' % args.ref))
        rb, rl, rs = ref[:, :7], ref[:, 7].astype(int), ref[:, 8]
        print('[ref %s] 检测框 %d 个, 类别分布 %s' % (args.ref, len(rb), np.bincount(rl, minlength=4).tolist()))
        ok = True
        if len(boxes) != len(rb):
            print('!! 框数不一致: %d vs %d' % (len(boxes), len(rb)))
            ok = False
        elif not np.array_equal(labels, rl):
            print('!! 类别分布/顺序不一致')
            ok = False
        else:
            dcoord = np.abs(boxes - rb).max()
            dscore = np.abs(scores - rs).max()
            print('坐标最大偏差 %.6f | score 最大偏差 %.6f' % (dcoord, dscore))
            if dcoord >= 0.05:
                print('!! 坐标偏差 >= 0.05 红线')
                ok = False
        print('结论:', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
