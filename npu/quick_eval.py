"""快速评估: 指定帧数跑 PyTorch -> 官方 KITTI AP。

在单卡 GPU 上快速验证 checkpoint 基线精度。

用法:
    python npu/quick_eval.py --frames 200 --save-preds /tmp/gpu_preds
    # 之后:
    python npu/eval_official_kitti.py --preds /tmp/gpu_preds
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import numpy as np
import torch

if getattr(torch, "npu", None) is not None and torch.npu.is_available():
    torch.npu.set_compile_mode(jit_compile=False)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.models.detectors.pointpillar import PointPillar
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None):
    coords = voxel_coords.cpu().numpy()
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    index_map = np.full(G, M, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map).to(voxel_coords.device)


class PPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module_list = model.module_list

    def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
        batch_dict = {
            "voxels": voxels, "voxel_num_points": voxel_num_points,
            "voxel_coords": voxel_coords, "bev_index_map": bev_index_map,
            "batch_size": 1,
        }
        for m in self.module_list:
            batch_dict = m(batch_dict)
        return batch_dict["batch_box_preds"], batch_dict["batch_cls_preds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--save-preds", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, logger=logger,
    )
    sample_ids = demo_dataset.sample_id_list[:args.frames]
    out_dir = Path(args.save_preds)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[init] 加载模型 %s -> %s" % (args.ckpt, args.device), flush=True)
    model = PointPillar(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.to(args.device)
    model.eval()
    wrapper = PPWrapper(model)

    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = cfg.MODEL.POST_PROCESSING.SCORE_THRESH

    t0 = time.time()
    for i, fid in enumerate(sample_ids):
        data_dict = demo_dataset[demo_dataset.sample_id_list.index(fid)]
        data_dict = demo_dataset.collate_batch([data_dict])
        for key, val in data_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ["frame_id", "metadata", "calib"]:
                continue
            else:
                data_dict[key] = torch.from_numpy(val)

        voxels = data_dict["voxels"]
        voxel_num_points = data_dict["voxel_num_points"]
        voxel_coords = data_dict["voxel_coords"]
        M = voxels.shape[0]
        bev_index_map = build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=M)

        if "cuda" in args.device or "npu" in args.device:
            voxels = voxels.to(args.device)
            voxel_num_points = voxel_num_points.to(args.device)
            voxel_coords = voxel_coords.to(args.device)
            bev_index_map = bev_index_map.to(args.device)

        with torch.no_grad():
            om_box, om_cls = wrapper(voxels, voxel_num_points, voxel_coords, bev_index_map)
        om_box = om_box.cpu().numpy().reshape(1, -1, 7)
        om_cls = om_cls.cpu().numpy().reshape(1, -1, 3)

        cls = torch.sigmoid(torch.from_numpy(om_cls[0]).to(args.device))
        cls, label = torch.max(cls, dim=-1)
        label = label + 1
        selected, scores = model_nms_utils.class_agnostic_nms(
            box_scores=cls.reshape(-1),
            box_preds=torch.from_numpy(om_box[0]).to(args.device),
            nms_config=nms_config,
            score_thresh=score_thresh,
        )
        # boxes = om_box[0][selected.numpy()]
        # labels = label[selected].numpy()
        # scores = scores.numpy()
        sel = selected.cpu().numpy()
        sc = scores.cpu().numpy()
        boxes = om_box[0][sel]
        labels = label.cpu().numpy()[sel]
        scores = sc


        with open(out_dir / ("%s.txt" % fid), "w") as f:
            for b, l, s in zip(boxes, labels, scores):
                f.write(("%s -1 -1 -1 0 0 0 0 "
                         "%.2f %.2f %.2f %.2f %.2f %.2f %.3f %.6f\n")
                        % (CLASS_NAMES[int(l) - 1],
                           b[0], b[1], b[2], b[3], b[4], b[5], b[6], s))

        if (i + 1) % 50 == 0 or i + 1 == len(sample_ids):
            print("[%d/%d] %.1fs (%.1f ms/frame)"
                  % (i + 1, len(sample_ids), time.time() - t0,
                     (time.time() - t0) / (i + 1) * 1000), flush=True)

    print("预测已保存 -> %s  (%d 帧, %.1fs)" % (out_dir, len(sample_ids), time.time() - t0), flush=True)

    # 自动跑官方评测
    print("\n运行官方评测 ...", flush=True)
    sys.path.insert(0, str(ROOT))
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    gt_annos = [info['annos'] for info in demo_dataset.kitti_infos[:args.frames]]
    dt_annos = []
    for info in demo_dataset.kitti_infos[:args.frames]:
        fid = info['point_cloud']['lidar_idx']
        pred_path = out_dir / ("%s.txt" % fid)
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
            from pcdet.utils import box_utils
            try:
                calib = demo_dataset.get_calib(fid)
                img_shape = demo_dataset.get_image_shape(fid)
                boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(np.array(boxes_lidar, dtype=np.float32), calib)
                boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes_cam, calib, image_shape=img_shape)
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

    result_str, _ = kitti_eval.get_official_eval_result(gt_annos, dt_annos, CLASS_NAMES)
    print(result_str)


if __name__ == "__main__":
    main()