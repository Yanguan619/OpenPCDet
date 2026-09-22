"""对比 PyTorch 原模型与 OM 的 raw 输出（batch_box_preds / batch_cls_preds），找出精度丢失步骤。

用法:
    python npu/compare_pt_om.py
    python npu/compare_pt_om.py --sample-idx 000008
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import aclruntime
import numpy as np
import torch
from aclruntime import InferenceSession
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import KittiDataset
from pcdet.utils import common_utils
from pcdet.models.detectors.pointpillar import PointPillar

from npu.run_bin import to_tensor, build_index_map, tensor_to_numpy
from npu.export_onnx import PPWrapper

NUM_ANCHORS = 321408


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    parser.add_argument("--om", default=str(ROOT / "weights/pointpillar_fp32_dynamic_linux_aarch64.om"))
    parser.add_argument("--sample-idx", default="000008")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, logger=logger,
    )

    # ======== 前处理（两路共用） ========
    idx = demo_dataset.sample_id_list.index(args.sample_idx)
    data_dict = demo_dataset[idx]
    data_dict = demo_dataset.collate_batch([data_dict])

    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ["frame_id", "metadata", "calib"]:
            continue
        elif key in ["images"]:
            data_dict[key] = torch.from_numpy(val).float().npu().contiguous()
        elif key in ["image_shape"]:
            data_dict[key] = torch.from_numpy(val).int().npu()
        else:
            data_dict[key] = torch.from_numpy(val).npu()

    voxels = data_dict["voxels"]
    voxel_num_points = data_dict["voxel_num_points"]
    voxel_coords = data_dict["voxel_coords"]
    M = voxels.shape[0]
    bev_index_map = build_index_map(voxel_coords.cpu(), nx=432, ny=496, nz=1, M=M).npu()

    # ======== PyTorch 前向 ========
    print("--- PyTorch forward ---", flush=True)
    model = PointPillar(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.npu()
    model.eval()

    wrapper = PPWrapper(model)
    with torch.no_grad():
        ref_box, ref_cls = wrapper(voxels, voxel_num_points, voxel_coords, bev_index_map)
    ref_box_np = ref_box.cpu().numpy().reshape(1, NUM_ANCHORS, 7)
    ref_cls_np = ref_cls.cpu().numpy().reshape(1, NUM_ANCHORS, 3)
    print("  ref_box:     shape=%s  range=[%.4f, %.4f]  mean=%.4f"
          % (ref_box_np.shape, ref_box_np.min(), ref_box_np.max(), ref_box_np.mean()), flush=True)
    print("  ref_cls:     shape=%s  range=[%.4f, %.4f]  mean=%.4f"
          % (ref_cls_np.shape, ref_cls_np.min(), ref_cls_np.max(), ref_cls_np.mean()), flush=True)

    # ======== OM 前向 ========
    print("--- OM forward ---", flush=True)
    session = InferenceSession(args.om, args.device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]

    feeds = {
        "voxels": aclruntime.Tensor(np.ascontiguousarray(voxels.cpu().numpy())),
        "voxel_num_points": aclruntime.Tensor(np.ascontiguousarray(voxel_num_points.cpu().numpy())),
        "voxel_coords": aclruntime.Tensor(np.ascontiguousarray(voxel_coords.cpu().numpy())),
        "bev_index_map": aclruntime.Tensor(np.ascontiguousarray(bev_index_map.cpu().numpy())),
    }
    # 动态 shape
    expect_m = session.get_inputs()[0].shape[0]
    if expect_m is not None and expect_m <= 0:
        dym = ["%s:%s" % (inp.name, ",".join(str(M if d <= 0 else d) for d in inp.shape))
               for inp in session.get_inputs()]
        session.set_dynamic_shape(";".join(dym))
        out_size = []
        for out in session.get_outputs():
            n = 1
            for d in out.shape:
                n *= max(d, 1)
            out_size.append(n * 4 * 4)
        session.set_custom_outsize(out_size)

    for _t in feeds.values():
        if hasattr(_t, "to_device"):
            _t.to_device(args.device)
    session.run(out_names, feeds)  # warmup
    out = session.run(out_names, feeds)

    om_box = tensor_to_numpy(out[0], np.float32).reshape(1, NUM_ANCHORS, 7)
    om_cls = tensor_to_numpy(out[1], np.float32).reshape(1, NUM_ANCHORS, 3)
    print("  om_box:      shape=%s  range=[%.4f, %.4f]  mean=%.4f"
          % (om_box.shape, om_box.min(), om_box.max(), om_box.mean()), flush=True)
    print("  om_cls:      shape=%s  range=[%.4f, %.4f]  mean=%.4f"
          % (om_cls.shape, om_cls.min(), om_cls.max(), om_cls.mean()), flush=True)

    # ======== 对比 ========
    print("\n--- 对比 ---", flush=True)
    d_box = ref_box_np - om_box
    d_cls = ref_cls_np - om_cls
    cos_box = (ref_box_np.flatten() @ om_box.flatten()) / (np.linalg.norm(ref_box_np.flatten()) * np.linalg.norm(om_box.flatten()) + 1e-12)
    cos_cls = (ref_cls_np.flatten() @ om_cls.flatten()) / (np.linalg.norm(ref_cls_np.flatten()) * np.linalg.norm(om_cls.flatten()) + 1e-12)

    print("  box maxdiff:  %.6f" % np.abs(d_box).max(), flush=True)
    print("  cls maxdiff:  %.6f" % np.abs(d_cls).max(), flush=True)
    print("  box cosine:   %.6f" % cos_box, flush=True)
    print("  cls cosine:   %.6f" % cos_cls, flush=True)

    if cos_box > 0.999:
        print("  box: 上层无精度丢失", flush=True)
    else:
        print("  box: 上层有精度丢失 (cos=%.6f)" % cos_box, flush=True)

    # 后处理对比
    print("\n--- 后处理对比 ---", flush=True)
    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    from pcdet.models.model_utils import model_nms_utils

    cls = torch.sigmoid(torch.from_numpy(ref_cls_np[0]))
    cls, label = torch.max(cls, dim=-1)
    label = label + 1
    selected, scores = model_nms_utils.class_agnostic_nms(
        box_scores=cls.reshape(-1), box_preds=torch.from_numpy(ref_box_np[0]),
        nms_config=nms_config, score_thresh=score_thresh,
    )
    ref_boxes = ref_box_np[0][selected.numpy()]
    ref_labels = label[selected].numpy()
    ref_scores = scores.numpy()
    print("  ref:  %d boxes, class dist: Car=%d Ped=%d Cyc=%d"
          % (len(ref_boxes), (ref_labels == 1).sum(), (ref_labels == 2).sum(), (ref_labels == 3).sum()), flush=True)

    cls2 = torch.sigmoid(torch.from_numpy(om_cls[0]))
    cls2, label2 = torch.max(cls2, dim=-1)
    label2 = label2 + 1
    selected2, scores2 = model_nms_utils.class_agnostic_nms(
        box_scores=cls2.reshape(-1), box_preds=torch.from_numpy(om_box[0]),
        nms_config=nms_config, score_thresh=score_thresh,
    )
    om_boxes = om_box[0][selected2.numpy()]
    om_labels = label2[selected2.numpy()]
    om_scores = scores2.numpy()
    print("  om:   %d boxes, class dist: Car=%d Ped=%d Cyc=%d"
          % (len(om_boxes), (om_labels == 1).sum(), (om_labels == 2).sum(), (om_labels == 3).sum()), flush=True)


if __name__ == "__main__":
    main()