"""PyTorch 原模型全量 val 推理 -> 保存预测（与 om_ref_test.py 完全同管线，仅推理后端不同）。

用法:
    python npu/om_ref_test_pt.py --save-preds preds_pt
    python npu/om_ref_test_pt.py --save-preds preds_pt --frames 3   # 抽验
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import npu.npu_patch  # noqa: E402,F401  预注入 CUDA ops 降级 stub，必须在 import pcdet 之前

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.models.detectors.pointpillar import PointPillar
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

from npu.om_ref_demo import to_tensor, build_index_map
from npu.export_onnx import PPWrapper

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--frames", type=int, default=0, help="0=全部")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--save-preds", default="preds_pt")
    parser.add_argument("--no-fov", action="store_true")
    args = parser.parse_args()

    if args.device == "npu":
        from torch_npu.contrib import transfer_to_npu
        torch.npu.set_compile_mode(jit_compile=False)

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    sample_ids = demo_dataset.sample_id_list
    n_total = len(sample_ids)
    end = args.end if args.end > 0 else n_total
    if args.frames > 0 and end > args.start + args.frames:
        end = args.start + args.frames
    sample_ids = sample_ids[args.start:end]
    fx = Path(cfg.DATA_CONFIG.DATA_PATH) / "training"
    velodyne_dir = fx / "velodyne"

    print("[init] 加载 PyTorch 模型 (device=%s) ..." % args.device, flush=True)
    model = PointPillar(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    if args.device == "npu":
        model.npu()
    model.eval()
    wrapper = PPWrapper(model)

    if args.save_preds:
        out_dir = Path(args.save_preds)
        out_dir.mkdir(parents=True, exist_ok=True)

    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = (
        args.score_thresh
        if args.score_thresh is not None
        else cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    )

    t0 = time.perf_counter()
    to_dev = (lambda t: t.npu()) if args.device == "npu" else (lambda t: t)
    for i, fid in enumerate(sample_ids):
        data_dict = demo_dataset[demo_dataset.sample_id_list.index(fid)]
        data_dict = demo_dataset.collate_batch([data_dict])
        for key, val in data_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ["frame_id", "metadata", "calib"]:
                continue
            elif key in ["images"]:
                data_dict[key] = to_dev(torch.from_numpy(val).float().contiguous())
            elif key in ["image_shape"]:
                data_dict[key] = to_dev(torch.from_numpy(val).int())
            else:
                data_dict[key] = to_dev(torch.from_numpy(val))

        voxels = data_dict["voxels"]
        voxel_num_points = data_dict["voxel_num_points"]
        voxel_coords = data_dict["voxel_coords"]
        M = voxels.shape[0]
        bev_index_map = to_dev(build_index_map(voxel_coords.cpu(), nx=432, ny=496, nz=1, M=M))

        with torch.no_grad():
            om_box, om_cls = wrapper(voxels, voxel_num_points, voxel_coords, bev_index_map)
        om_box = om_box.cpu().numpy().reshape(1, 321408, 7)
        om_cls = om_cls.cpu().numpy().reshape(1, 321408, 3)

        cls = torch.sigmoid(torch.from_numpy(om_cls[0]))
        cls, label = torch.max(cls, dim=-1)
        label = label + 1
        selected, scores = model_nms_utils.class_agnostic_nms(
            box_scores=cls.reshape(-1),
            box_preds=torch.from_numpy(om_box[0]),
            nms_config=nms_config,
            score_thresh=score_thresh,
        )
        boxes = om_box[0][selected.numpy()]
        labels = label[selected].numpy()
        scores = scores.numpy()

        with open(out_dir / ("%s.txt" % fid), "w") as f:
            for b, l, s in zip(boxes, labels, scores):
                f.write(
                    "%s -1 -1 -1 0 0 0 0 %.2f %.2f %.2f %.2f %.2f %.2f %.3f %.6f\n"
                    % (CLASS_NAMES[int(l) - 1], b[0], b[1], b[2], b[3], b[4], b[5], b[6], s)
                )

        if (i + 1) % 50 == 0 or i + 1 == len(sample_ids):
            el = time.perf_counter() - t0
            print("[%d/%d] elapsed=%.1fs avg=%.1fms/frame"
                  % (i + 1, len(sample_ids), el, el / (i + 1) * 1000), flush=True)

    print("DONE -> %s (%.1fs)" % (out_dir, time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()