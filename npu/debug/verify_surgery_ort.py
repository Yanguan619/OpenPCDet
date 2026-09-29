"""整模型等价验证：原始 ONNX vs pixel-shuffle 手术版 ONNX（ORT fp32 逐位对比）。

用法: python3 npu/debug/verify_surgery_ort.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from npu.om_ref_demo import DemoDataset, to_tensor, build_index_map

BIN = ROOT / '000008.bin'
CFG = ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'
ORIG = ROOT / 'weights/pointpillar_nms_base_v2_dynamic_topk.onnx'
MOD = os.environ.get('MOD_ONNX', ROOT / 'om_out/pp_topk_pixelshuffle.onnx')


def main():
    os.chdir(str(ROOT / 'tools'))
    cfg_from_yaml_file(str(CFG), cfg)
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                               training=False, root_path=BIN, logger=logger)
    dd = demo_dataset.prepare_data(data_dict={
        'points': np.fromfile(str(BIN), dtype=np.float32).reshape(-1, 4).copy(),
        'frame_id': 0, 'use_lead_xyz': True})
    data_dict = demo_dataset.collate_batch([dd])
    voxels = to_tensor(data_dict['voxels'])
    vnp = to_tensor(data_dict['voxel_num_points'])
    vco = to_tensor(data_dict['voxel_coords'].astype(np.int32))
    M = voxels.shape[0]
    bim = build_index_map(vco, M=M)
    feeds = {
        'voxels': np.ascontiguousarray(voxels.numpy()),
        'voxel_num_points': np.ascontiguousarray(vnp.numpy()),
        'voxel_coords': np.ascontiguousarray(vco.numpy()),
        'bev_index_map': np.ascontiguousarray(bim.numpy()),
    }
    print('M=%d' % M)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for tag, path in (('orig', ORIG), ('pixelshuffle', MOD)):
        sess = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])
        outs = sess.run(None, feeds)
        np.save('/tmp/opencode/%s_boxes.npy' % tag, outs[0])
        np.save('/tmp/opencode/%s_cls.npy' % tag, outs[1])
        print('%s: boxes %s cls %s' % (tag, outs[0].shape, outs[1].shape))

    bo = np.load('/tmp/opencode/orig_boxes.npy')
    co_ = np.load('/tmp/opencode/orig_cls.npy')
    bm = np.load('/tmp/opencode/pixelshuffle_boxes.npy')
    cm = np.load('/tmp/opencode/pixelshuffle_cls.npy')
    print('boxes max abs diff:', np.abs(bo - bm).max())
    print('cls   max abs diff:', np.abs(co_ - cm).max())
    print('boxes bit-equal:', np.array_equal(bo, bm))
    print('cls   bit-equal:', np.array_equal(co_, cm))


if __name__ == '__main__':
    main()
