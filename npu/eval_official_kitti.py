"""用官方 KITTI 评测（kitti_object_eval_python，OpenPCDet 自带）评估已保存的预测。

用法:
    python npu/eval_official_kitti.py --preds preds_kitti
    python npu/eval_official_kitti.py --preds preds_kitti --config data/config.yaml

说明:
    - 预测文件格式: 每帧一个 txt, 行 = <class> 8个占位 x y z dx dy dz ry score (lidar 系)
    - 内部: lidar box -> camera box -> 2D bbox, 构造与 OpenPCDet generate_prediction_dicts
      完全一致的 det_annos, 再调官方 get_official_eval_result(gt, dt, class_names)。
    - 输出: bbox / bev / 3d AP, 每类 R11 与 R40, easy/moderate/hard 三难度。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.utils import common_utils, box_utils

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def load_preds_lidar(path):
    """读取预测 txt -> (name[], score[], boxes_lidar (N,7) [x,y,z,dx,dy,dz,ry])"""
    names, scores, boxes = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16:
                continue
            name = parts[0]
            if name not in CLASS_NAMES:
                continue
            names.append(name)
            scores.append(float(parts[15]))
            boxes.append([float(parts[8]), float(parts[9]), float(parts[10]),
                          float(parts[11]), float(parts[12]), float(parts[13]),
                          float(parts[14])])
    if not boxes:
        return (np.array([], dtype=object), np.zeros((0,), dtype=np.float32),
                np.zeros((0, 7), dtype=np.float32))
    return np.array(names), np.array(scores, dtype=np.float32), np.array(boxes, dtype=np.float32)


def build_det_annos(preds_dir, dataset, sample_ids):
    """构造与 OpenPCDet generate_prediction_dicts 一致的 det_annos 列表"""
    annos = []
    for fid in sample_ids:
        pred_path = preds_dir / ("%s.txt" % fid)
        num = 0
        if pred_path.exists():
            names, scores, boxes_lidar = load_preds_lidar(pred_path)
            num = len(names)
        else:
            names, scores, boxes_lidar = (np.array([], dtype=object), np.zeros((0,), dtype=np.float32),
                                          np.zeros((0, 7), dtype=np.float32))

        pred_dict = {
            'name': np.zeros(num, dtype=object) if num else np.array([], dtype=object),
            'truncated': np.zeros(num), 'occluded': np.zeros(num), 'alpha': np.zeros(num),
            'bbox': np.zeros([num, 4]), 'dimensions': np.zeros([num, 3]),
            'location': np.zeros([num, 3]), 'rotation_y': np.zeros(num),
            'score': np.zeros(num), 'boxes_lidar': np.zeros([num, 7])
        }
        pred_dict['name'] = names
        pred_dict['score'] = scores
        pred_dict['boxes_lidar'] = boxes_lidar
        if num > 0:
            try:
                calib = dataset.get_calib(fid)
                image_shape = dataset.get_image_shape(fid)
            except Exception:
                calib = None
                image_shape = None
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes_lidar, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            pred_dict['alpha'] = -np.arctan2(-boxes_lidar[:, 1], boxes_lidar[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        pred_dict['frame_id'] = fid
        annos.append(pred_dict)
    return annos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="preds_kitti/ 目录")
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    preds_dir = Path(args.preds)

    # 官方 GT: info['annos'] (与 OpenPCDet evaluation() 一致)
    gt_annos = [info['annos'] for info in demo_dataset.kitti_infos]
    sample_ids = [info['point_cloud']['lidar_idx'] for info in demo_dataset.kitti_infos]

    print("[build det_annos] ...", flush=True)
    dt_annos = build_det_annos(preds_dir, demo_dataset, sample_ids)

    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    print("[eval official] ...", flush=True)
    result_str, ret_dict = kitti_eval.get_official_eval_result(gt_annos, dt_annos, CLASS_NAMES)
    print(result_str)
    print("\n===== 数值摘要 (R40) =====")
    for k, v in ret_dict.items():
        print("%-30s %.4f" % (k, v))


if __name__ == "__main__":
    main()