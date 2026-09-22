"""单跑一个原始 KITTI velodyne .bin 点云文件 -> OM 推理 -> 打印检测结果。

读入任意 (x,y,z,intensity) 的 velodyne bin，复用 dataset 的 data_processor 做
mask 范围 + voxelization，交给 aclruntime 加载的 OM 前向，最后在 Python 侧做
后处理（sigmoid/topk/class_agnostic_nms）并打印检测框。

用法:
    python npu/run_bin.py --bin data/kitti/training/velodyne/000008.bin \
                               --om ./pointpillar_fp32.om

依赖: pip install aclruntime-0.0.3-cp311-cp311-linux_aarch64.whl
"""

import argparse
import sys
import time
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
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

NUM_ANCHORS = 321408  # 216 * 248 * 2(rot) * 3(class)


def to_tensor(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x.cpu()


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None):
    """构造 PointPillarScatter 的 Gather 索引表 (G,) int64（pad 行号 = M）。"""
    coords = voxel_coords.numpy() if hasattr(voxel_coords, 'numpy') else voxel_coords
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    index_map = np.full(G, M, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map)


def tensor_to_numpy(t, dtype):
    """aclruntime.Tensor -> numpy（先 to_host 搬到 host 内存）。"""
    t.to_host()
    return np.frombuffer(memoryview(t), dtype=dtype).reshape(t.shape).copy()


def load_kitti_labels(label_path, calib=None):
    """加载 KITTI label_2/<id>.txt，返回 (N, 8) 数组 [class_id, x, y, z, dx, dy, dz, heading]。
    class_id: 1=Car, 2=Pedestrian, 3=Cyclist（与 OpenPCDet CLASS_NAMES 对齐）。
    若提供 calib，将 label 从相机坐标系转换到 velodyne（lidar）坐标系。
    """
    name_to_id = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
    objs = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            name = parts[0]
            if name not in name_to_id:
                continue
            # KITTI label: class truncated occlusion alpha x1 y1 x2 y2 h w l x y z ry
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])

            if calib is not None and hasattr(calib, "rect_to_lidar"):
                # 相机坐标系 -> velodyne 坐标系
                pts_rect = np.array([[x_cam, y_cam, z_cam]])
                pts_lidar = calib.rect_to_lidar(pts_rect)[0]
                x_l, y_l, z_l = float(pts_lidar[0]), float(pts_lidar[1]), float(pts_lidar[2])
                # 旋转角：相机绕 y 轴 ry -> lidar 绕 z 轴 heading
                heading = -ry - np.pi / 2
            else:
                x_l, y_l, z_l = x_cam, y_cam, z_cam
                heading = ry

            # 尺寸：相机 (h, w, l) -> lidar (dx=l, dy=w, dz=h)
            objs.append([name_to_id[name], x_l, y_l, z_l, l, w, h, heading])
    return np.array(objs, dtype=np.float32) if objs else np.zeros((0, 8), dtype=np.float32)


def boxes_iou_bev(boxes_a, boxes_b):
    """计算两组 3D box 的 BEV IoU。
    boxes_a/b: (N, 7) [x, y, z, dx, dy, dz, heading]
    返回: (N_a, N_b)
    """
    from pcdet.ops.iou3d_nms.iou3d_nms_torch_native import boxes_iou_bev

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    return boxes_iou_bev(torch.from_numpy(boxes_a), torch.from_numpy(boxes_b)).numpy()


def evaluate_against_labels(boxes, labels, scores, gt_objs, class_names, iou_thresh=0.5):
    """检测结果与 KITTI label 对比，打印匹配结果。
    boxes: (N, 7)  检测框
    labels: (N,)   检测类别 (1-indexed)
    scores: (N,)   置信度
    gt_objs: (M, 8) [class_id, x, y, z, dx, dy, dz, ry]
    """
    id_to_name = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}

    print(f"\n{'=' * 60}")
    print(f"Label 对比 (IoU 阈值={iou_thresh})")
    print(f"{'=' * 60}")
    print(f"检测框: {len(boxes)}   GT: {len(gt_objs)}")

    if len(gt_objs) == 0:
        print("无 GT label，跳过对比")
        return

    # 按类别统计 GT
    gt_by_class = {}
    for obj in gt_objs:
        cid = int(obj[0])
        gt_by_class.setdefault(cid, []).append(obj[1:])  # [x,y,z,dx,dy,dz,ry]
    print(
        f"GT 分布: {', '.join(f'{id_to_name.get(c, c)}={len(v)}' for c, v in sorted(gt_by_class.items()))}"
    )

    # 按类别统计检测
    det_by_class = {}
    for i in range(len(boxes)):
        cid = int(labels[i])
        det_by_class.setdefault(cid, []).append(np.concatenate([boxes[i], [scores[i]]]))
    print(
        f"检测分布: {', '.join(f'{id_to_name.get(c, c)}={len(v)}' for c, v in sorted(det_by_class.items()))}"
    )

    # 逐类别匹配
    total_tp, total_fp, total_fn = 0, 0, 0
    print(
        f"\n{'class':<12} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'Recall':>8} {'Prec':>8}"
    )
    print("-" * 60)
    for cid in sorted(set(list(gt_by_class.keys()) + list(det_by_class.keys()))):
        gt_boxes = np.array(gt_by_class.get(cid, []), dtype=np.float32)  # (M, 7)
        det_items = np.array(det_by_class.get(cid, []), dtype=np.float32)  # (N, 8)
        det_boxes = det_items[:, :7] if len(det_items) else np.zeros((0, 7), dtype=np.float32)
        det_scores = det_items[:, 7] if len(det_items) else np.zeros((0,), dtype=np.float32)

        n_gt = len(gt_boxes)
        n_det = len(det_boxes)
        if n_det == 0:
            tp, fp, fn = 0, 0, n_gt
        elif n_gt == 0:
            tp, fp, fn = 0, n_det, 0
        else:
            iou = boxes_iou_bev(det_boxes, gt_boxes)  # (N_det, N_gt)
            # 按置信度降序匹配
            order = np.argsort(-det_scores)
            matched_gt = set()
            tp = 0
            for idx in order:
                best_iou = iou[idx].max() if len(iou[idx]) else 0
                best_gt = iou[idx].argmax() if len(iou[idx]) else -1
                if best_iou >= iou_thresh and best_gt not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt)
            fp = n_det - tp
            fn = n_gt - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        name = id_to_name.get(cid, str(cid))
        print(
            f"{name:<12} {n_gt:>4} {n_det:>4} {tp:>4} {fp:>4} {fn:>4} {recall:>8.2f} {prec:>8.2f}"
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    print("-" * 60)
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    print(
        f"{'合计':<12} {'':>4} {'':>4} {total_tp:>4} {total_fp:>4} {total_fn:>4} {total_recall:>8.2f} {total_prec:>8.2f}"
    )
    print(f"{'=' * 60}")

    # 打印未匹配的 GT（漏检）
    if len(gt_objs) > 0 and len(boxes) > 0:
        for cid in sorted(gt_by_class.keys()):
            gt_boxes = np.array(gt_by_class[cid], dtype=np.float32)
            det_items = np.array(det_by_class.get(cid, []), dtype=np.float32)
            det_boxes = det_items[:, :7] if len(det_items) else np.zeros((0, 7), dtype=np.float32)
            if len(det_boxes) == 0:
                iou = np.zeros((0, len(gt_boxes)))
            else:
                iou = boxes_iou_bev(det_boxes, gt_boxes)
            for j in range(len(gt_boxes)):
                if iou.shape[0] == 0 or iou[:, j].max() < iou_thresh:
                    g = gt_boxes[j]
                    print(
                        f"  漏检: {id_to_name.get(cid, cid)} at ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})"
                    )


def main():
    parser = argparse.ArgumentParser(description="Run PointPillars OM on a single velodyne .bin")
    parser.add_argument("--bin", default="data/kitti/training/velodyne/000008.bin", help="path to velodyne .bin (x,y,z,intensity)")
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--om", default=str(ROOT / "./pointpillar.om"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=10, help="OM 测速迭代次数")
    parser.add_argument("--frame-id", default=None, help="自定义 frame_id（默认用 bin 文件名）")
    parser.add_argument(
        "--no-fov", action="store_true", help="不做 FOV 过滤（用于无 calib/image 的任意 bin）"
    )
    parser.add_argument(
        "--label",
        default=None,
        help="KITTI label 文件路径（默认自动推断 data/kitti/training/label_2/<frame>.txt）",
    )
    parser.add_argument("--score-thresh", type=float, default=None, help="置信度阈值（默认使用 cfg 中的 SCORE_THRESH=0.1）")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="与 label 匹配的 IoU 阈值")
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )

    # ==================== 前处理：读 bin + FOV/mask + voxelize ====================
    t_pre = time.perf_counter()
    points = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4)
    print("points: %d (x,y,z,intensity)" % points.shape[0], flush=True)
    frame_id = args.frame_id if args.frame_id else Path(args.bin).stem

    # FOV 过滤（与 demo/__getitem__ 路径一致，减少非相机视角点）。
    # 仅当该 frame 有 calib/image（训练集内 bin）时可用；任意 bin 可 --no-fov 关闭。
    if not args.no_fov and cfg.DATA_CONFIG.FOV_POINTS_ONLY:
        n_orig = points.shape[0]
        try:
            calib = demo_dataset.get_calib(frame_id)
            img_shape = demo_dataset.get_image_shape(frame_id)
            pts_rect = calib.lidar_to_rect(points[:, :3])
            fov_flag = demo_dataset.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]
            print("FOV filter: %d -> %d points" % (n_orig, points.shape[0]), flush=True)
        except Exception as e:
            print("WARN: 跳过 FOV 过滤（无 calib/image）: %s" % e, flush=True)

    # use_lead_xyz: bin 列为 (x,y,z,intensity)，voxel 前 3 维保留 xyz 作前导特征（与 config 一致）
    input_dict = {"frame_id": frame_id, "points": points, "use_lead_xyz": True}
    data_dict = demo_dataset.prepare_data(data_dict=input_dict)
    data_dict = demo_dataset.collate_batch([data_dict])

    voxels = to_tensor(data_dict["voxels"])
    voxel_num_points = to_tensor(data_dict["voxel_num_points"])
    voxel_coords = to_tensor(data_dict["voxel_coords"].astype(np.int32))
    bev_index_map = build_index_map(voxel_coords, M=voxels.shape[0])
    print(
        "voxels: %s (M=%d)  coords: %s"
        % (tuple(voxels.shape), voxels.shape[0], tuple(voxel_coords.shape)),
        flush=True,
    )
    t_pre = time.perf_counter() - t_pre
    print("[前处理] 读bin + mask + voxelize: %.1f ms" % (t_pre * 1000), flush=True)

    # ==================== OM 推理 ====================
    session = InferenceSession(args.om, args.device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    M = voxels.shape[0]
    if expect_m is not None and expect_m > 0 and M != expect_m:
        sys.exit(
            "ERROR: 该帧非空 pillar 数 M=%d 与 OM 静态 M=%d 不匹配。\n"
            "       解决：a) 对训练集内 bin 不要用 --no-fov（FOV 过滤后应回到 3941）；\n"
            "       b) 换帧需重新导出/转换 OM。"
            % (M, expect_m)
        )
    if expect_m is not None and expect_m <= 0:
        # 动态 shape OM（ATC --input_shape 用 a~b 范围转换）：运行时指定实际 shape + 输出缓存
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
        print("[动态 shape] set_dynamic_shape: %s" % ";".join(dym), flush=True)
    # numba 预热：后处理旋转 NMS 的 numba 首次初始化约 1.5s，预热移出计时区间
    try:
        from pcdet.ops.iou3d_nms.iou3d_nms_torch_native import _nms_iou_matrix
        _nms_iou_matrix(np.zeros((2, 7), dtype=np.float32))
    except Exception:
        pass

    # 预处理产物 dtype 已匹配 OM 端口（voxels f32 / voxel_num_points,
    # voxel_coords int32 / bev_index_map int64），无需再转 dtype；但
    # voxel_coords 经 collate 后是非连续（Fortran 序, strides=(4, M*4)），
    # aclruntime.Tensor 是原样字节拷贝，必须显式 ascontiguousarray，
    # 否则把 F 序内存按线性搬运得到乱码（见 ATC_PRECISION_BUG_REPORT 根因）。
    feeds = {
        "voxels": aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
        "voxel_num_points": aclruntime.Tensor(np.ascontiguousarray(voxel_num_points.numpy())),
        "voxel_coords": aclruntime.Tensor(np.ascontiguousarray(voxel_coords.numpy())),
        "bev_index_map": aclruntime.Tensor(np.ascontiguousarray(bev_index_map.numpy())),
    }
    # 输入一次性放到 device，避免每次 run 重复 H2D 传输
    for _key, _t in feeds.items():
        if hasattr(_t, "to_device"):
            _t.to_device(args.device)
    for _ in range(2):  # warmup
        session.run(out_names, feeds)

    n = args.num_iters
    t_inf = time.perf_counter()
    for _ in range(n):
        out = session.run(out_names, feeds)
    t_inf = (time.perf_counter() - t_inf) / n
    print("[OM 推理] NPU 前向 (avg %d iters): %.1f ms" % (n, t_inf * 1000), flush=True)

    om_box = tensor_to_numpy(out[0], np.float32).reshape(1, NUM_ANCHORS, 7)
    om_cls = tensor_to_numpy(out[1], np.float32).reshape(1, NUM_ANCHORS, 3)

    # ==================== 后处理：sigmoid + topk + class_agnostic_nms ====================
    t_post = time.perf_counter()
    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = args.score_thresh if args.score_thresh is not None else cfg.MODEL.POST_PROCESSING.SCORE_THRESH

    cls = torch.sigmoid(torch.from_numpy(om_cls[0]))  # (NUM_ANCHORS, 3)
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
    t_post = time.perf_counter() - t_post
    print("[后处理] sigmoid + NMS: %.1f ms" % (t_post * 1000), flush=True)

    # ==================== E2E 汇总 ====================
    e2e = t_pre + t_inf + t_post
    print(f"\n{'=' * 50}", flush=True)
    print(f"E2E 汇总:", flush=True)
    print(f"  前处理:   {t_pre * 1000:7.1f} ms  ({t_pre / e2e * 100:5.1f}%)", flush=True)
    print(f"  OM 推理:   {t_inf * 1000:7.1f} ms  ({t_inf / e2e * 100:5.1f}%)", flush=True)
    print(f"  后处理:   {t_post * 1000:7.1f} ms  ({t_post / e2e * 100:5.1f}%)", flush=True)
    print(f"  总计:     {e2e * 1000:7.1f} ms", flush=True)
    print(f"{'=' * 50}", flush=True)

    # ==================== 输出 ====================
    print("\n%s 检测结果 (%d 个框):" % (frame_id, len(selected)))
    print(
        "  %-10s %8s %8s %8s %6s %6s %6s %8s %8s"
        % ("class", "x", "y", "z", "dx", "dy", "dz", "r", "score")
    )
    class_names = cfg.CLASS_NAMES
    for b, l, s in zip(boxes, labels, scores):
        print(
            "  %-10s %8.2f %8.2f %8.2f %6.2f %6.2f %6.2f %8.3f %8.3f"
            % (class_names[int(l) - 1], b[0], b[1], b[2], b[3], b[4], b[5], b[6], s)
        )

    # ==================== Label 对比 ====================
    label_path = args.label
    if label_path is None:
        # 自动推断: data/kitti/training/label_2/<frame>.txt
        auto = ROOT / "data" / "kitti" / "training" / "label_2" / f"{frame_id}.txt"
        if auto.exists():
            label_path = str(auto)
    if label_path and Path(label_path).exists():
        # 加载 calib 做坐标系转换（相机 -> velodyne）
        calib = None
        try:
            calib = demo_dataset.get_calib(frame_id)
        except Exception as e:
            print("WARN: 无法加载 calib，label 将不做坐标系转换: %s" % e, flush=True)
        gt_objs = load_kitti_labels(label_path, calib)
        evaluate_against_labels(boxes, labels, scores, gt_objs, class_names, args.iou_thresh)
    else:
        print("\n(未找到 label 文件，跳过对比。可用 --label 指定)")


if __name__ == "__main__":
    main()
