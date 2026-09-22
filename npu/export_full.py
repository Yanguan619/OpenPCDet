"""直接用 PyTorch 导出 PointPillars + 后处理(sigmoid + topk + NMS)，不用图手术。

用法:
    python npu/export_full.py [--opset 16]
"""

import argparse, sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))
from torch_npu.contrib import transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.utils import common_utils
from pcdet.models.detectors.pointpillar import PointPillar

import torchvision


class PPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module_list = model.module_list

    def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
        batch_dict = {"voxels": voxels, "voxel_num_points": voxel_num_points,
                      "voxel_coords": voxel_coords, "bev_index_map": bev_index_map, "batch_size": 1}
        for m in self.module_list:
            batch_dict = m(batch_dict)
        return batch_dict["batch_box_preds"], batch_dict["batch_cls_preds"]


class FullWrapper(nn.Module):
    """PPWrapper + Sigmoid + Max + class-agnostic NMS（axis-aligned BEV）。"""
    def __init__(self, model, iou_thresh=0.01, score_thresh=0.1, max_det=500):
        super().__init__()
        self.base = PPWrapper(model)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.max_det = max_det

    def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
        box_preds, cls_preds = self.base(voxels, voxel_num_points, voxel_coords, bev_index_map)
        # (1,N,7) (1,N,3)
        cls_sig = torch.sigmoid(cls_preds)                    # (1,N,3)
        scores, labels = torch.max(cls_sig, dim=-1)           # (1,N)
        scores = scores.squeeze(0)                            # (N,)
        labels = labels.squeeze(0) + 1                        # (N,) 1-indexed
        box = box_preds.squeeze(0)                            # (N,7) lidar

        # 3D box -> axis-aligned BEV 2D box [x1,y1,x2,y2]
        x, y = box[:, 0], box[:, 1]
        dx, dy = box[:, 3], box[:, 4]
        boxes2d = torch.stack([x - dx/2, y - dy/2, x + dx/2, y + dy/2], dim=1)  # (N,4)

        # score 过滤
        keep_mask = scores >= self.score_thresh
        boxes2d = boxes2d[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]
        box = box[keep_mask]

        # NMS
        keep = torchvision.ops.nms(boxes2d, scores, self.iou_thresh)
        keep = keep[:self.max_det]

        final_boxes = box[keep]          # (K,7)
        final_scores = scores[keep]      # (K,)
        final_labels = labels[keep]      # (K,)
        final_count = final_boxes.shape[0]

        return final_boxes, final_scores, final_labels, final_count


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None):
    coords = voxel_coords.cpu().numpy()
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    index_map = np.full(G, M, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    p.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    p.add_argument("--sample-idx", default="000008")
    p.add_argument("--output", default=str(ROOT / "weights/pointpillar_full.onnx"))
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--score-thresh", type=float, default=0.1)
    p.add_argument("--iou-thresh", type=float, default=0.01)
    p.add_argument("--max-det", type=int, default=500)
    args = p.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                training=False, logger=logger)

    model = PointPillar(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.npu(); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    data_dict = demo_dataset[demo_dataset.sample_id_list.index(args.sample_idx)]
    data_dict = demo_dataset.collate_batch([data_dict])
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray): continue
        elif key in ["frame_id", "metadata", "calib"]: continue
        elif key in ["images"]: data_dict[key] = torch.from_numpy(val).float().npu().contiguous()
        elif key in ["image_shape"]: data_dict[key] = torch.from_numpy(val).int().npu()
        else: data_dict[key] = torch.from_numpy(val).npu()

    voxels, vnp, vc = data_dict["voxels"], data_dict["voxel_num_points"], data_dict["voxel_coords"]
    M = voxels.shape[0]
    bim = build_index_map(vc.cpu(), nx=432, ny=496, nz=1, M=M).npu()

    wrapper = FullWrapper(model, iou_thresh=args.iou_thresh,
                          score_thresh=args.score_thresh, max_det=args.max_det)
    with torch.no_grad():
        b, s, l, c = wrapper(voxels, vnp, vc, bim)
    print("ref outputs:", b.shape, s.shape, l.shape, "count=", c.item(), flush=True)
    print("ref top scores:", torch.sort(s, descending=True)[0][:5].tolist(), flush=True)
    print("ref labels:", l.tolist(), flush=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (voxels, vnp, vc, bim),
            args.output,
            input_names=["voxels", "voxel_num_points", "voxel_coords", "bev_index_map"],
            output_names=["final_boxes", "final_scores", "final_labels", "final_count"],
            opset_version=args.opset,
            dynamo=False,
        )
    print("EXPORT DONE -> %s" % args.output, flush=True)


if __name__ == "__main__":
    main()