"""全量 KITTI val 集精度评估（tools/test.py 的 OM 版，内嵌官方评测）。

逐帧 前处理(FOV+voxelize) -> OM 推理 -> sigmoid/NMS，默认直接跑官方 KITTI 评测
（R11/R40 bbox/bev/3d AP，与 tools/test.py 的 eval_one_epoch 同口径）。

用法:
    python npu/om_ref_test.py --om weights/pointpillar_fp16_static.om
    python npu/om_ref_test.py --om weights/pointpillar_fp16_static.om --frames 5   # 快速抽验
    python npu/om_ref_test.py --om weights/pointpillar_fp16_static.om --start 0 --end 1000
    python npu/om_ref_test.py --om weights/pointpillar_fp16_static.om --save-preds preds_kitti
    python npu/om_ref_test.py --om <om> --quick   # 只输出简化 TP/FP/FN + Recall/Precision

说明:
    - 复用同一个 InferenceSession，静态/动态 shape OM 均支持（动态 OM 无 M 跳过）。
    - 默认输出官方 AP；--quick 走简化 BEV IoU(0.5) 匹配（快速 sanity check）。
    - 运行时间估计: 3769 帧 * ~370ms ≈ 23 分钟（单进程，含官方评测）。
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
from pcdet.utils import common_utils, box_utils

from npu.om_ref_demo import (
    to_tensor,
    build_index_map,
    pad_to_static_m,
    fov_filter_fused,
    tensor_to_numpy,
    load_kitti_labels,
)

NUM_ANCHORS = 321408
ID2NAME = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def build_det_annos(preds_by_frame, dataset, sample_ids, class_names):
    """由内存中的逐帧预测构造 det_annos（与 OpenPCDet generate_prediction_dicts 一致）。"""
    annos = []
    for fid in sample_ids:
        boxes, labels, scores = preds_by_frame.get(
            fid,
            (np.zeros((0, 7), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32)),
        )
        num = len(boxes)
        pred_dict = {
            'name': np.array([class_names[int(l) - 1] for l in labels], dtype=object),
            'truncated': np.zeros(num), 'occluded': np.zeros(num), 'alpha': np.zeros(num),
            'bbox': np.zeros([num, 4]), 'dimensions': np.zeros([num, 3]),
            'location': np.zeros([num, 3]), 'rotation_y': np.zeros(num),
            'score': np.asarray(scores, np.float32), 'boxes_lidar': np.asarray(boxes, np.float32),
            'frame_id': fid,
        }
        if num > 0:
            try:
                calib = dataset.get_calib(fid)
                image_shape = dataset.get_image_shape(fid)
            except Exception:
                calib, image_shape = None, None
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            pred_dict['alpha'] = -np.arctan2(-boxes[:, 1], boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        annos.append(pred_dict)
    return annos


def run_official_eval(preds_by_frame, dataset, sample_ids, class_names):
    """官方 KITTI 评测（R11/R40 AP，与 tools/test.py 的 eval_one_epoch 同口径）。"""
    try:
        from npu.npu_patch import patch_rotate_iou
    except ImportError:
        from npu.debug.npu_patch import patch_rotate_iou
    patch_rotate_iou()  # 必须先于 kitti 评测模块 import（CPU rotate_iou 注入）
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    info_by_id = {info['point_cloud']['lidar_idx']: info for info in dataset.kitti_infos}
    gt_annos = [info_by_id[fid]['annos'] for fid in sample_ids]

    print("\n[build det_annos] ...", flush=True)
    dt_annos = build_det_annos(preds_by_frame, dataset, sample_ids, class_names)

    print("[eval official] ...", flush=True)
    result_str, ret_dict = kitti_eval.get_official_eval_result(gt_annos, dt_annos, class_names)
    print(result_str)
    print("\n===== 数值摘要 (R40) =====")
    for k, v in ret_dict.items():
        print("%-30s %.4f" % (k, v))


def match_one_frame(boxes, labels, scores, gt_objs, iou_thresh):
    """单帧匹配，返回 class_id -> (tp, fp, fn) 与 每类 GT/检测 数量。"""
    from npu.ops_native.iou3d_nms_torch_native import boxes_iou_bev

    gt_by, det_by = {}, {}
    for obj in gt_objs:
        cid = int(obj[0])
        gt_by.setdefault(cid, []).append(obj[1:])
    for i in range(len(boxes)):
        cid = int(labels[i])
        det_by.setdefault(cid, []).append(np.concatenate([boxes[i], [scores[i]]]))

    result = {}
    for cid in sorted(set(list(gt_by.keys()) + list(det_by.keys()))):
        gb_raw = np.asarray(gt_by.get(cid, []), dtype=np.float32)
        gb = gb_raw[:, :7].reshape(-1, 7) if gb_raw.ndim == 2 else np.zeros((0, 7), dtype=np.float32)
        di = np.array(det_by.get(cid, []), dtype=np.float32)  # (N,8)
        db = di[:, :7] if len(di) else np.zeros((0, 7), dtype=np.float32)
        ds = di[:, 7] if len(di) else np.zeros((0,), dtype=np.float32)

        n_gt, n_det = len(gb), len(db)
        if n_det == 0:
            tp, fp, fn = 0, 0, n_gt
        elif n_gt == 0:
            tp, fp, fn = 0, n_det, 0
        else:
            iou = boxes_iou_bev(torch.from_numpy(db), torch.from_numpy(gb)).numpy()  # (N, M)
            order = np.argsort(-ds)
            matched, tp = set(), 0
            for idx in order:
                if len(iou[idx]) == 0:
                    continue
                best = int(iou[idx].argmax())
                if iou[idx, best] >= iou_thresh and best not in matched:
                    tp += 1
                    matched.add(best)
            fp, fn = n_det - tp, n_gt - tp
        result[cid] = (tp, fp, fn)
    return result, gt_by, det_by


def main():
    parser = argparse.ArgumentParser(description="Full KITTI val precision eval via OM")
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--om", default=str(ROOT / "weights/pointpillar_fp16_static.om"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="0=全部; >0 只跑前 N 帧(抽验)")
    parser.add_argument("--start", type=int, default=0, help="从第 start 帧开始")
    parser.add_argument("--end", type=int, default=0, help="0=到末尾")
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--save-preds", default=None, help="保存逐帧预测到目录(KITTI label 格式)")
    parser.add_argument("--no-fov", action="store_true", help="不做 FOV 过滤")
    parser.add_argument("--verbose-frames", action="store_true", help="逐帧打印检测框")
    parser.add_argument("--quick", action="store_true", help="只输出简化 TP/FP/FN+Recall/Precision（不跑官方 AP）")
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    sys.stdout.write("[init] 加载 dataset ...\n")
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    sample_ids = demo_dataset.sample_id_list or [i["point_cloud"]["lidar_idx"] for i in demo_dataset.kitti_infos]
    n_total = len(sample_ids)
    end = args.end if args.end > 0 else n_total
    if args.frames > 0 and end > args.start + args.frames:
        end = args.start + args.frames
    sample_ids = sample_ids[args.start:end]
    fx = Path(cfg.DATA_CONFIG.DATA_PATH) / "training"
    velodyne_dir = fx / "velodyne"
    label_dir = fx / "label_2"

    sys.stdout.write(
        "[init] session 加载 %s (frames %d 起始=%d 终止=%d)\n" % (args.om, len(sample_ids), args.start, end)
    )
    sys.stdout.flush()
    session = InferenceSession(args.om, args.device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    is_dynamic = expect_m is not None and expect_m <= 0
    if is_dynamic:
        sys.stdout.write("[init] 动态 shape OM\n")

    def _set_dynamic_shape(M):
        if not is_dynamic:
            return
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
            out_size.append(n * 4 * 4)  # 每元素 4B * 4 倍余量
        session.set_custom_outsize(out_size)
    try:
        from npu.ops_native.iou3d_nms_torch_native import _nms_iou_matrix, _nms_incremental

        _nms_iou_matrix(np.zeros((2, 7), dtype=np.float32))
        _nms_incremental(np.zeros((2, 7), dtype=np.float32), 0.01)  # numba 预热
    except Exception:
        pass

    if args.save_preds:
        out_dir = Path(args.save_preds)
        out_dir.mkdir(parents=True, exist_ok=True)

    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = (
        args.score_thresh
        if args.score_thresh is not None
        else cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    )

    agg = {}          # class_id -> [tp, fp, fn]
    preds_by_frame = {}
    n_det_total = 0
    n_gt_total = 0
    n_skip = 0
    skip_log = []
    t_inf, t_pre, t_post = 0.0, 0.0, 0.0
    t0 = time.perf_counter()

    for i, fid in enumerate(sample_ids):
        t_frame = time.perf_counter()
        # ---------------- 前处理 ----------------
        t_a = time.perf_counter()
        points = np.fromfile(velodyne_dir / ("%s.bin" % fid), dtype=np.float32).reshape(-1, 4)
        if not args.no_fov and cfg.DATA_CONFIG.FOV_POINTS_ONLY:
            try:
                calib = demo_dataset.get_calib(fid)
                img_shape = demo_dataset.get_image_shape(fid)
                fov_flag = fov_filter_fused(points, calib, img_shape)
                points = points[fov_flag]
            except Exception:
                pass
        data_dict = demo_dataset.prepare_data(data_dict={"frame_id": fid, "points": points, "use_lead_xyz": True})
        data_dict = demo_dataset.collate_batch([data_dict])

        voxels = to_tensor(data_dict["voxels"])
        voxel_coords = to_tensor(data_dict["voxel_coords"].astype(np.int32))
        M = voxels.shape[0]
        if expect_m is not None and expect_m > 0:
            # 静态 shape OM：M < 固定值时 pad，M 超限则跳过
            if M > expect_m:
                n_skip += 1
                if len(skip_log) < 10:
                    skip_log.append("%s (M=%d)" % (fid, M))
                continue
            voxel_num_points = to_tensor(data_dict["voxel_num_points"])
            voxels, voxel_num_points, voxel_coords, bev_index_map = pad_to_static_m(
                voxels, voxel_num_points, voxel_coords, expect_m
            )
        else:
            bev_index_map = build_index_map(voxel_coords, M=M)
            voxel_num_points = to_tensor(data_dict["voxel_num_points"])
        feeds = {
            "voxels": aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
            "voxel_num_points": aclruntime.Tensor(np.ascontiguousarray(voxel_num_points.numpy())),
            "voxel_coords": aclruntime.Tensor(np.ascontiguousarray(voxel_coords.numpy())),
            "bev_index_map": aclruntime.Tensor(np.ascontiguousarray(bev_index_map.numpy())),
        }
        _set_dynamic_shape(M)
        if i == 0:
            for _key, _t in feeds.items():
                if hasattr(_t, "to_device"):
                    _t.to_device(args.device)
            session.run(out_names, feeds)  # warmup
        t_pre += time.perf_counter() - t_a

        # ---------------- OM 推理 ----------------
        t_b = time.perf_counter()
        out = session.run(out_names, feeds)
        t_inf += time.perf_counter() - t_b

        # ---------------- 后处理 ----------------
        t_c = time.perf_counter()
        om_box = tensor_to_numpy(out[0], np.float32).reshape(1, NUM_ANCHORS, 7)
        om_cls = tensor_to_numpy(out[1], np.float32).reshape(1, NUM_ANCHORS, 3)
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
        t_post += time.perf_counter() - t_c
        preds_by_frame[fid] = (boxes, labels, scores)

        # ---------------- GT 匹配（仅 --quick 简化口径） ----------------
        if args.quick:
            gt_objs = np.zeros((0, 8), dtype=np.float32)
            label_file = label_dir / ("%s.txt" % fid)
            if label_file.exists():
                try:
                    calib = demo_dataset.get_calib(fid)
                except Exception:
                    calib = None
                gt_objs = load_kitti_labels(str(label_file), calib)
            res, gt_by, det_by = match_one_frame(boxes, labels, scores, gt_objs, args.iou_thresh)
            for cid, (tp, fp, fn) in res.items():
                a = agg.setdefault(cid, [0, 0, 0])
                a[0] += tp
                a[1] += fp
                a[2] += fn
            n_det_total += len(boxes)
            n_gt_total += len(gt_objs)

        if args.save_preds:
            with open(out_dir / ("%s.txt" % fid), "w") as f:
                for b, l, s in zip(boxes, labels, scores):
                    f.write(
                        "%s -1 -1 -1 0 0 0 0 %.2f %.2f %.2f %.2f %.2f %.2f %.3f %.6f\n"
                        % (
                            CLASS_NAMES[int(l) - 1],
                            b[0], b[1], b[2], b[3], b[4], b[5], b[6], s,
                        )
                    )
        if args.verbose_frames:
            print("frame=%s det=%d" % (fid, len(boxes)), flush=True)
        elif (i + 1) % 50 == 0 or i + 1 == len(sample_ids):
            el = time.perf_counter() - t0
            print(
                "[%d/%d] elapsed=%.1fs avg=%.1fms/frame skipped_M=%d"
                % (i + 1, len(sample_ids), el, el / (i + 1) * 1000, n_skip),
                flush=True,
            )
    total = time.perf_counter() - t0

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 66)
    print("全量评估 (frames=%d skipped_M=%d total_time=%.1fs avg=%.1fms/frame)"
          % (len(sample_ids), n_skip, total, total / max(len(sample_ids) - n_skip, 1) * 1000))
    print("拆分耗时: 前处理=%.1fs 推理=%.1fs 后处理=%.1fs" % (t_pre, t_inf, t_post))
    if skip_log:
        print("M 不匹配被跳过的帧(前10): %s" % ", ".join(skip_log))

    if args.quick:
        print("GT 总数=%d  检测总数=%d" % (n_gt_total, n_det_total))
        print("%-12s %6s %6s %6s %6s %6s %8s %8s" % ("class", "GT", "Det", "TP", "FP", "FN", "Recall", "Prec"))
        print("-" * 66)
        gtp = gfp = gfn = 0
        for cid in sorted(agg.keys()):
            tp, fp, fn = agg[cid]
            n_gt = tp + fn
            n_det = tp + fp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            gtp += tp
            gfp += fp
            gfn += fn
            print("%-12s %6d %6d %6d %6d %6d %8.3f %8.3f" % (ID2NAME.get(cid, cid), n_gt, n_det, tp, fp, fn, recall, prec))
        tr = gtp / (gtp + gfn) if (gtp + gfn) > 0 else 0
        tp_ = gtp / (gtp + gfp) if (gtp + gfp) > 0 else 0
        print("-" * 66)
        print("%-12s %6d %6d %6d %6d %6d %8.3f %8.3f" % ("合计", gtp + gfn, gtp + gfp, gtp, gfp, gfn, tr, tp_))
        print("=" * 66)
    else:
        # 官方 KITTI 评测（与 tools/test.py 的 eval_one_epoch 同口径，R11/R40 AP）
        run_official_eval(preds_by_frame, demo_dataset, sample_ids, CLASS_NAMES)


if __name__ == "__main__":
    main()