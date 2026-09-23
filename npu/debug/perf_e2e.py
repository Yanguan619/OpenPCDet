"""E2E 性能统计: 单帧从 bin 读取到检测框输出的分阶段耗时。

用法:
    python npu/perf_e2e.py --om weights/pointpillar_fp16_static.om
    python npu/perf_e2e.py --om weights/pointpillar_fp32_dynamic_linux_aarch64.om --iters 10
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
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

from npu.om_ref_demo import to_tensor, build_index_map, tensor_to_numpy

NUM_ANCHORS = 321408


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--om", default=str(ROOT / "weights/pointpillar_fp16_static.om"))
    parser.add_argument("--bin", default="data/kitti/training/velodyne/000008.bin")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20, help="推理计时迭代次数")
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    frame_id = Path(args.bin).stem

    times = {}

    def mark(name, fn):
        t0 = time.perf_counter()
        r = fn()
        times[name] = (time.perf_counter() - t0) * 1000
        return r

    points = mark("读bin", lambda: np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4))

    def fov():
        if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
            try:
                calib = demo_dataset.get_calib(frame_id)
                img_shape = demo_dataset.get_image_shape(frame_id)
                pts_rect = calib.lidar_to_rect(points[:, :3])
                fov_flag = demo_dataset.get_fov_flag(pts_rect, img_shape, calib)
                return points[fov_flag]
            except Exception:
                return points
        return points

    points = mark("FOV过滤", fov)

    def voxelize():
        dd = demo_dataset.prepare_data(data_dict={"frame_id": frame_id, "points": points, "use_lead_xyz": True})
        return demo_dataset.collate_batch([dd])

    data_dict = mark("voxelize(mask+体素化)", voxelize)

    def tensors():
        voxels = to_tensor(data_dict["voxels"])
        voxel_num_points = to_tensor(data_dict["voxel_num_points"])
        voxel_coords = to_tensor(data_dict["voxel_coords"].astype(np.int32))
        bev_index_map = build_index_map(voxel_coords, M=voxels.shape[0])
        return voxels, voxel_num_points, voxel_coords, bev_index_map

    voxels, voxel_num_points, voxel_coords, bev_index_map = mark("tensor转换+index_map", tensors)
    M = voxels.shape[0]

    t0 = time.perf_counter()
    session = InferenceSession(args.om, args.device, aclruntime.session_options())
    times["session初始化+模型加载"] = (time.perf_counter() - t0) * 1000

    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    if expect_m is not None and expect_m <= 0:
        dym = ["%s:%s" % (inp.name, ",".join(str(M if d <= 0 else d) for d in inp.shape))
               for inp in session.get_inputs()]
        t0 = time.perf_counter()
        session.set_dynamic_shape(";".join(dym))
        out_size = []
        for out in session.get_outputs():
            n = 1
            for d in out.shape:
                n *= max(d, 1)
            out_size.append(n * 4 * 4)
        session.set_custom_outsize(out_size)
        times["set_dynamic_shape"] = (time.perf_counter() - t0) * 1000

    # 热启动（numba 预热，不计入）
    try:
        from npu.ops_native import _nms_iou_matrix
        _nms_iou_matrix(np.zeros((2, 7), dtype=np.float32))
    except Exception:
        pass

    feeds = mark("构造feeds", lambda: {
        "voxels": aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
        "voxel_num_points": aclruntime.Tensor(np.ascontiguousarray(voxel_num_points.numpy())),
        "voxel_coords": aclruntime.Tensor(np.ascontiguousarray(voxel_coords.numpy())),
        "bev_index_map": aclruntime.Tensor(np.ascontiguousarray(bev_index_map.numpy())),
    })
    for _t in feeds.values():
        if hasattr(_t, "to_device"):
            _t.to_device(args.device)
    session.run(out_names, feeds)  # warmup 1
    session.run(out_names, feeds)  # warmup 2 (稳定 device 状态)

    t_inf = time.perf_counter()
    for _ in range(args.iters):
        out = session.run(out_names, feeds)
    times["NPU推理"] = (time.perf_counter() - t_inf) / args.iters * 1000

    om_box = tensor_to_numpy(out[0], np.float32).reshape(1, NUM_ANCHORS, 7)
    om_cls = tensor_to_numpy(out[1], np.float32).reshape(1, NUM_ANCHORS, 3)

    def sigmoid():
        cls = torch.sigmoid(torch.from_numpy(om_cls[0]))
        cls, label = torch.max(cls, dim=-1)
        return cls, label + 1

    cls, label = mark("sigmoid+topk", sigmoid)

    def nms():
        nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
        selected, scores = model_nms_utils.class_agnostic_nms(
            box_scores=cls.reshape(-1),
            box_preds=torch.from_numpy(om_box[0]),
            nms_config=nms_config,
            score_thresh=cfg.MODEL.POST_PROCESSING.SCORE_THRESH,
        )
        return selected, scores

    mark("NMS", nms)

    # 打印报告
    print("\n" + "=" * 56)
    print("E2E 性能统计 (frame=%s  M=%d  OM=%s)" % (frame_id, M, Path(args.om).name))
    print("=" * 56)
    order = ["读bin", "FOV过滤", "voxelize(mask+体素化)", "tensor转换+index_map",
             "session初始化+模型加载", "set_dynamic_shape", "构造feeds", "NPU推理", "sigmoid+topk", "NMS"]
    total_e2e = 0
    print("%-28s %10s" % ("阶段", "耗时(ms)"))
    print("-" * 56)
    for k in order:
        if k in times:
            v = times[k]
            total_e2e += v
            print("%-28s %10.2f" % (k, v))
    print("-" * 56)
    print("%-28s %10.2f" % ("E2E 合计(含session)", total_e2e))
    steady = sum(times[k] for k in ["读bin", "FOV过滤", "voxelize(mask+体素化)", "tensor转换+index_map",
                                    "构造feeds", "NPU推理", "sigmoid+topk", "NMS"] if k in times)
    print("%-28s %10.2f   (不含session加载)" % ("稳态单帧", steady))
    print("=" * 56)
    print("注: NPU推理为 %d 次平均; 其余为单次。" % args.iters)


if __name__ == "__main__":
    main()