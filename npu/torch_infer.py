"""单帧 PyTorch 推理 -> 保存 batch_box_preds / batch_cls_preds 到 .npy。

支持 device=cpu 或 npu（npu 时自动加载 torch_npu）。

用法:
    python torch_infer.py --device cpu  --out-box /tmp/cpu_box.npy --out-cls /tmp/cpu_cls.npy
    python torch_infer.py --device npu  --out-box /tmp/npu_box.npy --out-cls /tmp/npu_cls.npy
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import numpy as np
import torch


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None):
    coords = voxel_coords.cpu().numpy()
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    index_map = np.full(G, M, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    parser.add_argument("--sample-idx", default="000008")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--out-box", required=True)
    parser.add_argument("--out-cls", required=True)
    args = parser.parse_args()

    if args.device == "npu":
        from torch_npu.contrib import transfer_to_npu
        torch.npu.set_compile_mode(jit_compile=False)

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import KittiDataset
    from pcdet.utils import common_utils
    from pcdet.models.detectors.pointpillar import PointPillar

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, logger=logger,
    )

    idx = demo_dataset.sample_id_list.index(args.sample_idx)
    data_dict = demo_dataset[idx]
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
    bev_index_map = build_index_map(voxel_coords.cpu(), nx=432, ny=496, nz=1, M=M)

    model = PointPillar(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    if args.device == "npu":
        model.npu()
        voxels = voxels.npu()
        voxel_num_points = voxel_num_points.npu()
        voxel_coords = voxel_coords.npu()
        bev_index_map = bev_index_map.npu()

    model.eval()

    class PPWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module_list = m.module_list
        def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
            batch_dict = {
                "voxels": voxels, "voxel_num_points": voxel_num_points,
                "voxel_coords": voxel_coords, "bev_index_map": bev_index_map,
                "batch_size": 1,
            }
            for m in self.module_list:
                batch_dict = m(batch_dict)
            return batch_dict["batch_box_preds"], batch_dict["batch_cls_preds"]

    wrapper = PPWrapper(model)
    with torch.no_grad():
        box, cls = wrapper(voxels, voxel_num_points, voxel_coords, bev_index_map)

    box_np = box.cpu().numpy().reshape(1, -1, 7)
    cls_np = cls.cpu().numpy().reshape(1, -1, 3)
    np.save(args.out_box, box_np)
    np.save(args.out_cls, cls_np)
    print("DONE  device=%s  box range=[%.4f, %.4f]  cls range=[%.4f, %.4f]"
          % (args.device, box_np.min(), box_np.max(), cls_np.min(), cls_np.max()), flush=True)


if __name__ == "__main__":
    main()