"""用 NMS 图手术版 OM 逐帧推理，生成 KITTI 格式 preds（与 quick_eval 基线可比）。

用法:
    python npu/eval_om_nms.py --om weights/pointpillar_nms_v2_fp32_linux_aarch64.om \
        --frames 200 --save-preds /tmp/preds_om_check
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))
import npu.npu_patch  # noqa: E402,F401  预注入 CUDA ops 降级 stub，必须在 import pcdet 之前

import aclruntime
import numpy as np
import torch
from aclruntime import InferenceSession
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.utils import common_utils

from npu.om_ref_demo import to_tensor, build_index_map, tensor_to_numpy

NUM_ANCHORS = 321408
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--om", required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--save-preds", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )

    preds_dir = Path(args.save_preds)
    preds_dir.mkdir(parents=True, exist_ok=True)

    session = InferenceSession(args.om, args.device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    print("OM outputs:", out_names, flush=True)
    for out in session.get_outputs():
        try:
            out_dtype = out.datatype()
        except Exception:
            out_dtype = "?"
        print("  %s shape=%s dtype=%s" % (out.name, out.shape, out_dtype), flush=True)

    expect_m = session.get_inputs()[0].shape[0]
    print("expect_m:", expect_m, flush=True)

    frames = demo_dataset.sample_id_list[: args.frames]
    skipped = 0
    t0 = time.perf_counter()
    for i, fid in enumerate(frames):
        points = demo_dataset.get_lidar(fid)
        calib = demo_dataset.get_calib(fid)
        img_shape = demo_dataset.get_image_shape(fid)
        pts_rect = calib.lidar_to_rect(points[:, :3])
        fov_flag = demo_dataset.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]

        input_dict = {"frame_id": fid, "points": points, "use_lead_xyz": True}
        data_dict = demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = demo_dataset.collate_batch([data_dict])

        voxels = to_tensor(data_dict["voxels"])
        voxel_num_points = to_tensor(data_dict["voxel_num_points"])
        voxel_coords = to_tensor(data_dict["voxel_coords"].astype(np.int32))
        bev_index_map = build_index_map(voxel_coords, M=voxels.shape[0])

        M = voxels.shape[0]
        if expect_m is not None and expect_m > 0 and M != expect_m:
            skipped += 1
            continue

        dym = []
        for inp in session.get_inputs():
            dims = ",".join(str(M if d <= 0 else d) for d in inp.shape)
            dym.append("%s:%s" % (inp.name, dims))
        session.set_dynamic_shape(";".join(dym))
        out_size = []
        for out in session.get_outputs():
            n = 1
            for d in out.shape:
                n *= max(d, 1)
            out_size.append(n * 4 * 4)
        session.set_custom_outsize(out_size)

        feeds = {
            "voxels": aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
            "voxel_num_points": aclruntime.Tensor(np.ascontiguousarray(voxel_num_points.numpy())),
            "voxel_coords": aclruntime.Tensor(np.ascontiguousarray(voxel_coords.numpy())),
            "bev_index_map": aclruntime.Tensor(np.ascontiguousarray(bev_index_map.numpy())),
        }
        for _key, _t in feeds.items():
            if hasattr(_t, "to_device"):
                _t.to_device(args.device)
        out = session.run(out_names, feeds)

        # NMS 图手术版输出: boxes (1,N,7) f32 / scores (N,) f32 / labels (N,) i64 / count (1,) i64
        nms_count = int(tensor_to_numpy(out[3], np.int64).reshape(-1)[0])
        if nms_count <= 0:
            open(preds_dir / ("%s.txt" % fid), "w").close()
            continue
        boxes = tensor_to_numpy(out[0], np.float32).reshape(-1, 7)[:nms_count]
        scores = tensor_to_numpy(out[1], np.float32).reshape(-1)[:nms_count]
        labels = tensor_to_numpy(out[2], np.int64).reshape(-1)[:nms_count]

        lines = []
        for b, s, l in zip(boxes, scores, labels):
            cid = int(l) + 1
            if cid not in (1, 2, 3):
                continue
            name = CLASS_NAMES[cid - 1]
            x, y, z, dx, dy, dz, ry = b
            lines.append(
                "%s -1 -1 -1 0 0 0 0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
                % (name, x, y, z, dx, dy, dz, ry, s)
            )
        with open(preds_dir / ("%s.txt" % fid), "w") as f:
            f.write("\n".join(lines))
        if (i + 1) % 50 == 0:
            dt = time.perf_counter() - t0
            print("  [%d/%d] %.1fs elapsed" % (i + 1, len(frames), dt), flush=True)

    dt = time.perf_counter() - t0
    print("完成: %d frames, skipped=%d, %.1fs (avg %.0f ms/frame)"
          % (len(frames), skipped, dt, dt / max(len(frames) - skipped, 1) * 1000), flush=True)


if __name__ == "__main__":
    main()
