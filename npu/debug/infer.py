"""NPU 推理入口：加载 PointPillar 权重，对 KITTI val 帧做端到端推理并保存预测。

按 NPU 适配 skill 规范提供 build_data / pre_process / build_model / post_process / main，
统一复用 npu/npu_patch.py 完成设备检测与 NPU 初始化。

用法:
    python npu/infer.py --ckpt weights/pointpillar_7728.pth --device auto \
        --frames 200 --save-preds /tmp/infer_preds
    # --device 可选: auto(默认) / npu / cuda:0 / cpu
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

from npu.npu_patch import (
    init_patch, get_device, to_device, build_index_map, PPWrapper,
)

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.models.detectors.pointpillar import PointPillar
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def build_data(args):
    """加载 KITTI 验证集 dataset，返回 (dataset, sample_ids, cfg, logger)."""
    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, logger=logger,
    )
    sample_ids = demo_dataset.sample_id_list[: args.frames]
    return demo_dataset, sample_ids, cfg, logger


def pre_process(demo_dataset, fid, device):
    """取单帧数据 -> collate -> 转 tensor 并搬运到 device，返回前向输入."""
    data_dict = demo_dataset[demo_dataset.sample_id_list.index(fid)]
    data_dict = demo_dataset.collate_batch([data_dict])
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ["frame_id", "metadata", "calib"]:
            continue
        data_dict[key] = torch.from_numpy(val)

    voxels = data_dict["voxels"]
    voxel_num_points = data_dict["voxel_num_points"]
    voxel_coords = data_dict["voxel_coords"]
    M = voxels.shape[0]
    bev_index_map = build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=M)

    voxels = to_device(voxels, device)
    voxel_num_points = to_device(voxel_num_points, device)
    voxel_coords = to_device(voxel_coords, device)
    bev_index_map = to_device(bev_index_map, device)
    return voxels, voxel_num_points, voxel_coords, bev_index_map


def build_model(args, dataset, logger):
    """构建 PointPillar 并加载权重，返回 (model, wrapper)."""
    model = PointPillar(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    device = get_device() if args.device == "auto" else args.device
    if device == "npu":
        model.npu()
    elif device == "cpu":
        model.cpu()
    else:
        model.to(device)
    model.eval()
    return model, PPWrapper(model)


def post_process(wrapper, voxels, voxel_num_points, voxel_coords, bev_index_map,
                 nms_config, score_thresh, device):
    """前向 + sigmoid + topk + class_agnostic_nms，返回 (boxes, labels, scores)."""
    with torch.no_grad():
        om_box, om_cls = wrapper(voxels, voxel_num_points, voxel_coords, bev_index_map)
    om_box = om_box.cpu().numpy().reshape(1, -1, 7)
    om_cls = om_cls.cpu().numpy().reshape(1, -1, 3)

    cls = torch.sigmoid(torch.from_numpy(om_cls[0]).to(device))
    cls, label = torch.max(cls, dim=-1)
    label = label + 1
    selected, scores = model_nms_utils.class_agnostic_nms(
        box_scores=cls.reshape(-1),
        box_preds=torch.from_numpy(om_box[0]).to(device),
        nms_config=nms_config,
        score_thresh=score_thresh,
    )
    boxes = om_box[0][selected.cpu().numpy()]
    labels = label.cpu().numpy()[selected.cpu().numpy()]
    scores = scores.cpu().numpy()
    return boxes, labels, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    parser.add_argument("--device", default="auto",
                        help="auto / npu / cuda:0 / cpu")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--save-preds", required=True)
    parser.add_argument("--score-thresh", type=float, default=None)
    args = parser.parse_args()

    device = init_patch() if args.device == "auto" else args.device
    print("[init] device = %s" % device, flush=True)

    dataset, sample_ids, cfg, logger = build_data(args)
    model, wrapper = build_model(args, dataset, logger)
    print("[init] 加载模型 %s -> %s" % (args.ckpt, device), flush=True)

    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = (
        args.score_thresh
        if args.score_thresh is not None
        else cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    )
    out_dir = Path(args.save_preds)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, fid in enumerate(sample_ids):
        voxels, voxel_num_points, voxel_coords, bev_index_map = pre_process(
            dataset, fid, device
        )
        boxes, labels, scores = post_process(
            wrapper, voxels, voxel_num_points, voxel_coords, bev_index_map,
            nms_config, score_thresh, device,
        )
        with open(out_dir / ("%s.txt" % fid), "w") as f:
            for b, l, s in zip(boxes, labels, scores):
                f.write(
                    "%s -1 -1 -1 0 0 0 0 %.2f %.2f %.2f %.2f %.2f %.2f %.3f %.6f\n"
                    % (CLASS_NAMES[int(l) - 1], b[0], b[1], b[2], b[3], b[4], b[5], b[6], s)
                )
        if (i + 1) % 50 == 0 or i + 1 == len(sample_ids):
            print("[%d/%d] %.1fs (%.1f ms/frame)"
                  % (i + 1, len(sample_ids), time.time() - t0,
                     (time.time() - t0) / (i + 1) * 1000), flush=True)

    print("预测已保存 -> %s  (%d 帧, %.1fs)"
          % (out_dir, len(sample_ids), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
