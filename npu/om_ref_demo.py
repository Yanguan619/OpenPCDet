"""om_ref_demo.py —— tools/demo.py 的 OM 版（参照 demo 结构）。

与 tools/demo.py 同构：
  - 同样用 DemoDataset 直接读 .bin/.npy 点云（--data_path 支持目录或单文件，--ext 指定后缀），
  - 同样 parse_config(--cfg_file) + 遍历样本 + collate_batch + 打印检测结果，
唯一区别是前向由 aclruntime 加载的 OM 完成（代替 build_network + load_params_from_file），
后处理（sigmoid + class_agnostic_nms）在 Python 侧完成（对应模型内建 post_processing）。

用法:
    python npu/om_ref_demo.py \
        --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
        --data_path data/kitti/training/velodyne/000008.bin \
        --om weights/pointpillar_base_fp32_linux_aarch64.om
    python npu/om_ref_demo.py --data_path demo_data --om <动态fp32.om>

说明:
    - 静态 M OM 只接受固定 pillar 数的帧（如 000008 的 M=3941），
      其他帧会打印警告并跳过；用动态 OM（--input_shape 范围转出）可处理任意 M。
    - 依赖: pip install aclruntime-0.0.3-cp311-cp311-linux_aarch64.whl
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import npu.npu_patch  # noqa: E402,F401  预注入 CUDA ops 降级 stub，必须在 import pcdet 之前

import aclruntime
import numpy as np
import torch
from aclruntime import InferenceSession

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils
NUM_ANCHORS = 321408  # 216 * 248 * 2(rot) * 3(class)


class DemoDataset(DatasetTemplate):
    """与 tools/demo.py 的 DemoDataset 完全一致：读任意 .bin/.npy 点云做推理。"""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


# ============================================================
# OM 推理 helpers（被 npu/om_ref_test.py 等复用，勿删）
# ============================================================

def to_tensor(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x.cpu()


def _mm3(pts, m):
    """pts(N,3) @ m(3,3)：列式逐元素（与 np.dot 同乘加顺序，仅 ~1 点/帧 浮点边界翻转）。

    这台机器 numpy(openblas64 单线程) 对 (N,3)@(3,3) 走标量路径 ~81ms/次；
    逐元素约 5ms。FOV 是预处理过滤，1 点边界翻转对 AP 无影响（AP 门禁验证）。
    """
    x = pts[:, 0] * m[0, 0] + pts[:, 1] * m[1, 0] + pts[:, 2] * m[2, 0]
    y = pts[:, 0] * m[0, 1] + pts[:, 1] * m[1, 1] + pts[:, 2] * m[2, 1]
    z = pts[:, 0] * m[0, 2] + pts[:, 1] * m[1, 2] + pts[:, 2] * m[2, 2]
    return np.stack([x, y, z], axis=1)


def fov_filter_fused(points, calib, img_shape):
    """FOV 过滤（去 cart_to_hom 的 hstack + 慢速 np.dot/@）。

    与 calib.lidar_to_rect + get_fov_flag 同数学（乘加顺序一致）：
      A = V2C.T @ R0.T (4,3)；pts_rect = points @ A[:3] + A[3]
      pts_2d = pts_rect @ P2[:,:3].T + P2[:,3]；pts_img = pts_2d[:,:2]/pts_rect[:,2:3]
      depth = pts_2d[:,2] - P2.T[3,2]
    返回与 get_fov_flag 一致的布尔 mask。
    """
    pts = points[:, :3]
    a = calib.V2C.T @ calib.R0.T  # (4, 3)
    pts_rect = _mm3(pts, a[:3]) + a[3]  # (N, 3)
    pts_2d = _mm3(pts_rect, calib.P2[:, :3].T) + calib.P2[:, 3]  # (N, 3)
    pts_img = pts_2d[:, :2] / pts_rect[:, 2:3]
    pts_rect_depth = pts_2d[:, 2] - calib.P2.T[3, 2]
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    return np.logical_and(val_flag_merge, pts_rect_depth >= 0)


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None, pad=None):
    """构造 PointPillarScatter 的 Gather 索引表 (G,) int64（pad 行号 = pad，默认 = M）。"""
    coords = voxel_coords.numpy() if hasattr(voxel_coords, 'numpy') else voxel_coords
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    pad = M if pad is None else pad
    index_map = np.full(G, pad, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map)


def pad_to_static_m(voxels, voxel_num_points, voxel_coords, m_target):
    """把非空 pillar 张量 pad 到静态 OM 的固定 M（P0）。

    pad 行 num_points=0 → VFE 掩码掉、特征恒 0；Gather 索引表只引用真实行，
    空网格单元指向 pad 行（零特征）。与动态版输出 bit 一致（见 pointpillar_scatter 的
    bev_index_map Gather 路径）。返回 (voxels, num_points, coords, index_map)。
    """
    M = voxels.shape[0]
    if M > m_target:
        raise ValueError("M=%d > 静态 OM M=%d，超出范围" % (M, m_target))
    if M < m_target:
        pad = m_target - M
        voxels = torch.cat([voxels, voxels.new_zeros(pad, *voxels.shape[1:])], dim=0)
        voxel_num_points = torch.cat([voxel_num_points, voxel_num_points.new_zeros(pad)], dim=0)
        voxel_coords = torch.cat([voxel_coords, voxel_coords.new_zeros(pad, voxel_coords.shape[1])], dim=0)
    # 索引表只由真实行（前 M 个）构造，pad 值 = m_target
    index_map = build_index_map(voxel_coords[:M], M=M, pad=m_target)
    return voxels, voxel_num_points, voxel_coords, index_map


def tensor_to_numpy(t, dtype, copy=True):
    """aclruntime.Tensor -> numpy（先 to_host 搬到 host 内存）。

    copy=False 时返回 aclruntime host 缓冲的直接视图（writable，可喂 torch），
    省去整块 memcpy；调用方须在下次 session.run 前消费完（后处理路径满足）。
    """
    t.to_host()
    arr = np.frombuffer(memoryview(t), dtype=dtype).reshape(t.shape)
    return arr.copy() if copy else arr


def nms_topk_numpy(boxes, scores, score_thresh, nms_config):
    """numpy 版 top-K 预筛 + 增量贪心旋转 NMS（与 class_agnostic_nms 同结果，避开 torch 小算子开销）。

    NPU 310P 的 aicpu 对 topk/nonzero 等不稳定（见 model_nms_utils.py 注释），因此 top-K
    筛选放到 host 侧 numpy 完成：
      - score_thresh 掩码（>=，与 class_agnostic_nms 一致）
      - 超出 NMS_PRE_MAXSIZE 时用 np.argpartition（O(N) 选 top-K，与 torch.topk 同集合，
        并列分数次序差异允许）+ np.argsort 对 top-K 排序
    NMS 直接复用 npu/ops_native/iou3d_nms_torch_native.py 的 numba 增量贪心 _nms_incremental
    （要求输入已按 score 降序）；numba 不可用时回退 class_agnostic_nms。

    Args:
        boxes: (N, 7) float32 numpy，anchor 回归框（原始顺序）
        scores: (N,) torch CPU，sigmoid 后单类分数
        score_thresh: float 或 None
        nms_config: NMS_PRE_MAXSIZE / NMS_POST_MAXSIZE / NMS_THRESH
    Returns:
        sel_idx: numpy int64，原始下标（score 降序）
        sel_scores: numpy float32，对应分数
    """
    pre_max = int(getattr(nms_config, "NMS_PRE_MAXSIZE", 4096))
    post_max = int(getattr(nms_config, "NMS_POST_MAXSIZE", 500))
    thresh = float(getattr(nms_config, "NMS_THRESH", 0.01))
    snp = scores.numpy()
    if score_thresh is not None:
        keep = np.nonzero(snp >= score_thresh)[0]
    else:
        keep = np.arange(snp.shape[0])
    if keep.size > pre_max:
        part = np.argpartition(snp[keep], keep.size - pre_max)
        keep = keep[part[keep.size - pre_max:]]
    order = np.argsort(-snp[keep])
    keep = keep[order]
    try:
        from npu.ops_native.iou3d_nms_torch_native import _nms_incremental
    except ImportError:
        from pcdet.models.model_utils import model_nms_utils
        selected, sub_scores = model_nms_utils.class_agnostic_nms(
            box_scores=torch.from_numpy(snp[keep]),
            box_preds=torch.from_numpy(boxes[keep]),
            nms_config=nms_config,
            score_thresh=None,
        )
        orig = keep[selected.numpy()]
        return orig, sub_scores.numpy()
    boxes_nms = np.ascontiguousarray(boxes[keep][:, :7])
    kept = _nms_incremental(boxes_nms, thresh)[:post_max]
    orig = keep[kept]
    return orig, snp[orig]


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
    from npu.ops_native.iou3d_nms_torch_native import boxes_iou_bev

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


# ============================================================
# 与 tools/demo.py 同构的 parse_config / main
# ============================================================

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default=str(ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'),
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str,
                        default=str(ROOT / 'data/kitti/training/velodyne/000008.bin'),
                        help='specify the point cloud data file or directory')
    parser.add_argument('--om', type=str,
                        default=str(ROOT / 'weights/pointpillar_base_fp32_linux_aarch64.om'),
                        help='specify the OM model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-iters', type=int, default=10, help='OM 测速迭代次数')
    parser.add_argument('--score-thresh', type=float, default=None)
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='与 label 匹配的 IoU 阈值')
    parser.add_argument('--label', default=None, help='KITTI label 文件路径（默认自动推断）')

    args = parser.parse_args()

    # kitti_models 配置的 _BASE_CONFIG_ 相对 tools/ 解析，与 demo.py 一致
    cfg_path = Path(args.cfg_file)
    if 'tools/cfgs' in str(cfg_path) and os.getcwd() != str(ROOT / 'tools'):
        os.chdir(str(ROOT / 'tools'))

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def resolve_path(p):
    """相对路径先按 cwd 找，找不到再按仓库根目录 ROOT 兜底（chdir 到 tools/ 后仍可用）。"""
    p = Path(p)
    if not p.exists():
        alt = ROOT / p
        if alt.exists():
            return alt
    return p


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet (OM)-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=resolve_path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # 构建 OM session（代替 demo 里的 build_network + load_params_from_file）
    session = InferenceSession(str(resolve_path(args.om)), args.device, aclruntime.session_options())
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    base_mode = len(out_names) == 2  # base OM 输出 box+cls；否则为内嵌 NMS 的 OM

    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = (
        args.score_thresh if args.score_thresh is not None
        else cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    )

    # numba 预热：旋转 NMS 首次初始化 ~1.5s，移出计时区间
    try:
        from npu.ops_native.iou3d_nms_torch_native import _nms_iou_matrix, _nms_incremental
        _nms_iou_matrix(np.zeros((2, 7), dtype=np.float32))
        _nms_incremental(np.zeros((2, 7), dtype=np.float32), 0.01)
    except Exception:
        pass

    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        frame_id = Path(demo_dataset.sample_file_list[idx]).stem
        data_dict = demo_dataset.collate_batch([data_dict])

        voxels = to_tensor(data_dict['voxels'])
        voxel_num_points = to_tensor(data_dict['voxel_num_points'])
        voxel_coords = to_tensor(data_dict['voxel_coords'].astype(np.int32))
        M = voxels.shape[0]
        if expect_m is not None and expect_m > 0:
            # 静态 shape OM：M < 固定值时 pad，M 超限则跳过
            if M > expect_m:
                logger.warning(
                    f'frame {frame_id}: M={M} 超过静态 OM M={expect_m}，跳过（请用动态 OM 或更大的静态 OM）'
                )
                continue
            voxels, voxel_num_points, voxel_coords, bev_index_map = pad_to_static_m(
                voxels, voxel_num_points, voxel_coords, expect_m
            )
        else:
            bev_index_map = build_index_map(voxel_coords, M=M)

        feeds = {
            "voxels": aclruntime.Tensor(np.ascontiguousarray(voxels.numpy())),
            "voxel_num_points": aclruntime.Tensor(np.ascontiguousarray(voxel_num_points.numpy())),
            "voxel_coords": aclruntime.Tensor(np.ascontiguousarray(voxel_coords.numpy())),
            "bev_index_map": aclruntime.Tensor(np.ascontiguousarray(bev_index_map.numpy())),
        }
        if expect_m is not None and expect_m <= 0:
            # 动态 shape OM：运行时指定实际 shape + 输出缓存
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

        time_start = time.time()
        out = session.run(out_names, feeds)
        print('OM inference time: {:.4f}s'.format(time.time() - time_start))

        if base_mode:
            om_box = tensor_to_numpy(out[0], np.float32, copy=False).reshape(1, NUM_ANCHORS, 7)
            om_cls = tensor_to_numpy(out[1], np.float32, copy=False).reshape(1, NUM_ANCHORS, 3)
            # sigmoid 单调递增：max(sigmoid(x)) == sigmoid(max(x))，先取类间 max 再 sigmoid（省 2/3 逐元素）
            cls_max, label = torch.max(torch.from_numpy(om_cls[0]), dim=-1)
            label = label + 1
            scores = torch.sigmoid(cls_max)
            selected, scores = nms_topk_numpy(om_box[0], scores, score_thresh, nms_config)
            boxes = om_box[0][selected]
            labels = label[selected].numpy()
        else:
            # 内嵌 NMS 的 OM：输出 nms_final_boxes/scores/labels/count
            boxes = tensor_to_numpy(out[0], np.float32).reshape(-1, 7)
            scores = tensor_to_numpy(out[1], np.float32)
            labels = tensor_to_numpy(out[2], np.int64) + 1
            n_count = tensor_to_numpy(out[3], np.int64).reshape(-1)[0] if len(out) >= 4 else len(boxes)
            boxes, scores, labels = boxes[:n_count], scores[:n_count], labels[:n_count]

        print('\n%s 检测结果 (%d 个框):' % (frame_id, len(boxes)))
        print('  %-10s %8s %8s %8s %6s %6s %6s %8s %8s'
              % ('class', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'r', 'score'))
        class_names = cfg.CLASS_NAMES
        for b, l, s in zip(boxes, labels, scores):
            print('  %-10s %8.2f %8.2f %8.2f %6.2f %6.2f %6.2f %8.3f %8.3f'
                  % (class_names[int(l) - 1], b[0], b[1], b[2], b[3], b[4], b[5], b[6], s))

        # Label 对比（可选）
        label_path = args.label
        if label_path is None:
            auto = ROOT / 'data' / 'kitti' / 'training' / 'label_2' / f'{frame_id}.txt'
            if auto.exists():
                label_path = str(auto)
        if label_path and Path(label_path).exists():
            gt_objs = load_kitti_labels(label_path)
            evaluate_against_labels(boxes, labels, scores, gt_objs, class_names, args.iou_thresh)
        else:
            print('\n(未找到 label 文件，跳过对比。可用 --label 指定)')

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
