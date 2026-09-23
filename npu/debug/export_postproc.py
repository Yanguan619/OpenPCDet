"""合并导出: PPWrapper + 后处理(sigmoid/ReduceMax/topk/Gather)，没有 NMS。

输出: boxes(1,M,7), scores(M), labels(M) — 保留所有 anchor, NMS 在 Python 侧做。
"""

import argparse, sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))
import npu.npu_patch  # noqa: E402,F401  预注入 CUDA ops 降级 stub，必须在 import pcdet 之前
from torch_npu.contrib import transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.utils import common_utils
from pcdet.models.detectors.pointpillar import PointPillar

NUM_ANCHORS = 321408


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


class PostprocWrapper(nn.Module):
    """PPWrapper + Sigmoid + ReduceMax + 无 NMS，输出精简张量。"""
    def __init__(self, model):
        super().__init__()
        self.base = PPWrapper(model)

    def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
        box_preds, cls_preds = self.base(voxels, voxel_num_points, voxel_coords, bev_index_map)
        # cls_preds: (1, N, 3) f32, 未归一化
        cls_sig = torch.sigmoid(cls_preds)  # (1, N, 3)
        scores, labels = torch.max(cls_sig, dim=-1)  # (1, N)
        labels = labels + 1  # 1-indexed: 1=Car,2=Ped,3=Cyc
        # 去掉 batch 维
        scores = scores.squeeze(0)  # (N,)
        labels = labels.squeeze(0)  # (N,)
        box_preds = box_preds.squeeze(0)  # (N, 7)
        return box_preds, scores, labels


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
    p.add_argument("--output", default=str(ROOT / "weights/pointpillar_postproc.onnx"))
    p.add_argument("--opset", type=int, default=16)
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

    wrapper = PostprocWrapper(model)
    with torch.no_grad():
        box_ref, scores_ref, labels_ref = wrapper(voxels, vnp, vc, bim)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (voxels, vnp, vc, bim),
            args.output,
            input_names=["voxels", "voxel_num_points", "voxel_coords", "bev_index_map"],
            output_names=["batch_box_preds", "batch_scores", "batch_labels"],
            opset_version=args.opset,
            dynamo=False,
        )
    print("EXPORT DONE -> %s" % args.output, flush=True)

    # ATC 命令
    print("\nATC:", flush=True)
    print("  atc --model=%s --framework=5 --soc_version=Ascend310P3 \\" % args.output, flush=True)
    print("      --output=%s --input_format=ND --log=error" % args.output.replace('.onnx', ''), flush=True)
    print("      --input_shape='voxels:%d,32,4;voxel_num_points:%d;voxel_coords:%d,4;bev_index_map:214272'" % (M, M, M), flush=True)
    print("      --precision_mode=force_fp16", flush=True)


if __name__ == "__main__":
    main()