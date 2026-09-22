"""用已保存的 KITTI 预测计算官方 11 点 3D AP（Car@0.7 / Pedestrian@0.5 / Cyclist@0.5）。

用法:
    python npu/compute_kitti_ap.py --preds preds_kitti
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
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_torch_native import boxes_iou3d_gpu

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
IOU_THRESH = {1: 0.7, 2: 0.5, 3: 0.5}


def load_preds_file(path):
    """读取预测 txt，返回 [(class_id, box7, score), ...]"""
    items = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16:
                continue
            cls_name = parts[0]
            if cls_name not in CLASS_NAMES:
                continue
            cid = CLASS_NAMES.index(cls_name) + 1
            x, y, z = float(parts[8]), float(parts[9]), float(parts[10])
            dx, dy, dz = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            score = float(parts[15])
            items.append((cid, np.array([x, y, z, dx, dy, dz, ry], dtype=np.float32), score))
    return items


def load_gt_file(path, calib=None):
    """读取 GT txt，返回 [(class_id, box7), ...]（坐标系转为 lidar）"""
    items = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            name = parts[0]
            if name not in CLASS_NAMES:
                continue
            cid = CLASS_NAMES.index(name) + 1
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            if calib is not None and hasattr(calib, "rect_to_lidar"):
                pts_lidar = calib.rect_to_lidar(np.array([[x_cam, y_cam, z_cam]]))[0]
                x, y, z = float(pts_lidar[0]), float(pts_lidar[1]), float(pts_lidar[2])
                z = z + h / 2.0
                heading = -ry - np.pi / 2
            else:
                x, y, z = x_cam, y_cam, z_cam
                heading = ry
            items.append((cid, np.array([x, y, z, l, w, h, heading], dtype=np.float32)))
    return items


def compute_ap(scores, is_tp, n_gt):
    """11 点插值 AP"""
    order = np.argsort(-scores)
    tp = is_tp[order].astype(np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    prec = tp_cum / (tp_cum + fp_cum)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at = prec[recall >= t]
        if len(prec_at):
            ap += np.max(prec_at)
    return ap / 11.0


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
    label_dir = ROOT / "data" / "kitti" / "training" / "label_2"

    # {cid: {"scores":[], "is_tp":[], "n_gt":int}}
    stats = {1: {"scores": [], "is_tp": [], "n_gt": 0},
             2: {"scores": [], "is_tp": [], "n_gt": 0},
             3: {"scores": [], "is_tp": [], "n_gt": 0}}

    n_total = len(demo_dataset.sample_id_list)
    for i, fid in enumerate(demo_dataset.sample_id_list):
        pred_path = preds_dir / ("%s.txt" % fid)
        gt_path = label_dir / ("%s.txt" % fid)
        if not pred_path.exists():
            continue
        preds = load_preds_file(pred_path)  # [(cid, box7, score)]
        try:
            calib = demo_dataset.get_calib(fid)
        except Exception:
            calib = None
        gts = load_gt_file(str(gt_path), calib)  # [(cid, box7)]

        # 按类别分组
        gt_by_cid = {c: [] for c in range(1, 4)}
        for cid, box in gts:
            gt_by_cid[cid].append(box)

        # 预测按类别分组，每类内按 score 降序
        preds_by_cid = {c: [] for c in range(1, 4)}
        for cid, box, score in preds:
            preds_by_cid[cid].append((box, score))

        for cid in range(1, 4):
            n_gt = len(gt_by_cid[cid])
            stats[cid]["n_gt"] += n_gt
            if n_gt == 0 and len(preds_by_cid[cid]) == 0:
                continue

            # 构建 IoU 矩阵并做贪心匹配
            if n_gt > 0 and len(preds_by_cid[cid]) > 0:
                gt_boxes = np.stack(gt_by_cid[cid], axis=0)  # (M, 7)
                det_boxes = np.stack([p[0] for p in preds_by_cid[cid]], axis=0)  # (N, 7)
                det_scores = np.array([p[1] for p in preds_by_cid[cid]], dtype=np.float32)
                iou = boxes_iou3d_gpu(torch.from_numpy(det_boxes), torch.from_numpy(gt_boxes)).numpy()  # (N, M)
                order = np.argsort(-det_scores)
                matched_gt = set()
                for idx in order:
                    if len(iou[idx]) == 0:
                        stats[cid]["scores"].append(det_scores[idx])
                        stats[cid]["is_tp"].append(0)
                        continue
                    best = int(iou[idx].argmax())
                    is_tp = 1 if iou[idx, best] >= IOU_THRESH[cid] and best not in matched_gt else 0
                    if is_tp:
                        matched_gt.add(best)
                    stats[cid]["scores"].append(det_scores[idx])
                    stats[cid]["is_tp"].append(is_tp)
            else:
                # 无 GT 或 无预测：所有预测为 FP，GT 仅计数
                for box, score in preds_by_cid[cid]:
                    stats[cid]["scores"].append(score)
                    stats[cid]["is_tp"].append(0)

        if (i + 1) % 500 == 0 or i + 1 == n_total:
            sys.stdout.write("\r[%d/%d]" % (i + 1, n_total))
            sys.stdout.flush()
    sys.stdout.write("\n")

    print("\n%s  (3D AP, IoU thresholds: Car@0.7, Ped@0.5, Cyc@0.5)" % ("=" * 60))
    print("%-12s %6s %8s" % ("class", "GT", "AP"))
    print("-" * 60)
    aps = {}
    for cid in range(1, 4):
        s = stats[cid]
        scores = np.array(s["scores"], dtype=np.float32)
        is_tp = np.array(s["is_tp"], dtype=np.float32)
        n_gt = s["n_gt"]
        ap = compute_ap(scores, is_tp, n_gt) if n_gt > 0 and len(scores) > 0 else 0.0
        aps[cid] = ap
        print("%-12s %6d %8.3f" % (CLASS_NAMES[cid - 1], n_gt, ap * 100))
    mAP = np.mean(list(aps.values())) * 100
    print("-" * 60)
    print("%-12s %6s %8.3f" % ("mAP", "", mAP))
    print("=" * 60)


if __name__ == "__main__":
    main()