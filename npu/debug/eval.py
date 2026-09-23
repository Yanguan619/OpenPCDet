"""NPU 评测入口：对 infer.py 产出的预测执行官方 KITTI 评测。

复用项目原有评测逻辑（kitti_object_eval_python，官方评测已内嵌 om_ref_test/quick_eval），
计算 bbox / bev / 3d AP（R11 与 R40，easy/moderate/hard）。

用法:
    python npu/eval.py --preds /tmp/infer_preds [--frames 200]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

from npu.npu_patch import init_patch, patch_rotate_iou

import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.utils import box_utils, common_utils

patch_rotate_iou()
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # noqa: E402

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def load_preds(pred_dir, dataset, frames=0):
    """从 pred_dir 读取 KITTI 格式预测，组装为官方 eval 需要的 dt_annos."""
    sample_ids = dataset.sample_id_list
    if frames > 0:
        sample_ids = sample_ids[:frames]
    pred_dir = Path(pred_dir)
    dt_annos = []
    for fid in sample_ids:
        pred_path = pred_dir / ("%s.txt" % fid)
        names, scores, boxes_lidar = [], [], []
        if pred_path.exists():
            with open(pred_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 16 or p[0] not in CLASS_NAMES:
                        continue
                    names.append(p[0])
                    scores.append(float(p[15]))
                    boxes_lidar.append([float(p[8]), float(p[9]), float(p[10]),
                                        float(p[11]), float(p[12]), float(p[13]), float(p[14])])
        n = len(names)
        dt = {
            'name': np.array(names, dtype=object) if n else np.array([], dtype=object),
            'truncated': np.zeros(n), 'occluded': np.zeros(n), 'alpha': np.zeros(n),
            'bbox': np.zeros([n, 4]), 'dimensions': np.zeros([n, 3]),
            'location': np.zeros([n, 3]), 'rotation_y': np.zeros(n),
            'score': np.array(scores, dtype=np.float32) if n else np.zeros(0),
            'boxes_lidar': np.array(boxes_lidar, dtype=np.float32) if n else np.zeros([0, 7]),
        }
        if n > 0:
            try:
                calib = dataset.get_calib(fid)
                img_shape = dataset.get_image_shape(fid)
                boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(
                    np.array(boxes_lidar, dtype=np.float32), calib)
                boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    boxes_cam, calib, image_shape=img_shape)
                dt['alpha'] = (-np.arctan2(-np.array(boxes_lidar)[:, 1], np.array(boxes_lidar)[:, 0])
                               + boxes_cam[:, 6])
                dt['bbox'] = boxes_img
                dt['dimensions'] = boxes_cam[:, 3:6]
                dt['location'] = boxes_cam[:, 0:3]
                dt['rotation_y'] = boxes_cam[:, 6]
            except Exception:
                pass
        dt['frame_id'] = fid
        dt_annos.append(dt)
    return dt_annos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--preds", required=True, help="预测目录")
    parser.add_argument("--frames", type=int, default=0, help="0=全部")
    args = parser.parse_args()

    device = init_patch()
    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, logger=logger,
    )

    gt_annos = [info['annos'] for info in dataset.kitti_infos[:args.frames or len(dataset.kitti_infos)]]
    dt_annos = load_preds(args.preds, dataset, frames=args.frames)
    assert len(gt_annos) == len(dt_annos), "GT/DT 帧数不一致"

    result_str, _ = kitti_eval.get_official_eval_result(gt_annos, dt_annos, CLASS_NAMES)
    print("device = %s  frames = %d" % (device, len(dt_annos)))
    print(result_str)


if __name__ == "__main__":
    main()
