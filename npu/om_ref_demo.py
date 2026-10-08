"""om_ref_demo.py —— tools/demo.py 的 OM 版（参照 demo 结构）。

与 tools/demo.py 同构：
  - 同样用 DemoDataset 直接读 .bin/.npy 点云（--data_path 支持目录或单文件，--ext 指定后缀），
  - 同样 parse_config(--cfg_file) + 遍历样本 + collate_batch + 打印检测结果，
唯一区别是前向由 aclruntime 加载的 OM 完成（代替 build_network + load_params_from_file），
后处理（sigmoid + topk_nms）在 host 侧完成（对应模型内建 post_processing）。

用法:
    python npu/om_ref_demo.py \
        --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
        --data_path data/kitti/training/velodyne/000008.bin \
        --om weights/pointpillar_base_fp32_linux_aarch64.om
    python npu/om_ref_demo.py --data_path demo_data --om <动态fp32.om>

说明:
    - 静态 M OM 只接受固定 pillar 数的帧（如 000008 的 M=3941），
      其他帧会打印警告并跳过；用动态 OM（--input_shape 范围转出）可处理任意 M。
    - 依赖: pip install aclruntime-0.0.3-cp311-cp311-linux_aarch64.whl；
      unum_ops 为硬依赖（AscendC 体素化 + cv.topk_nms 后处理，安装见 npu/README.md §1）。
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 设备常驻管线（voxelization 输出保持 NPU tensor 直通 feeds）：默认开启，
# NPU_VOX_DEVICE_RESIDENT=0 可回退 numpy 路径做 A/B。必须在 import npu_patch 前设置。
os.environ.setdefault("NPU_VOX_DEVICE_RESIDENT", "1")
import aclruntime
import numpy as np
import torch
from aclruntime import InferenceSession
from unum_ops.cv import topk_nms

import npu.npu_patch  # noqa: E402,F401  spconv alias + AscendC 体素化 + numba mask，须在 import pcdet 之前

# OM 推理共享 helper 的唯一实现在 npu_patch（含 NPU 常驻 tensor 路径与 pad 语义）；
# 此处转引以维持 `from npu.om_ref_demo import build_index_map`（om_ref_test）。
from npu.npu_patch import build_index_map
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

# numba 为硬依赖（上面 import 的 npu_patch 顶层无条件 import numba），不做回退守卫。
import numba


@numba.jit(nopython=True, cache=True)
def _fov_filter_numba(pts, a, p2, p2_c, p2_t32, img_w, img_h):
    """FOV 投影 + 过滤单内核（逐点保持与 get_fov_flag 的列式 numpy 数学同乘加顺序）。

    与 get_fov_flag 同数学：
      pts_rect = pts @ A[:3] + A[3]   （A = V2C.T @ R0.T）
      pts_2d   = pts_rect @ P2[:,:3].T + P2[:,3]
      pts_img  = pts_2d[:,:2] / pts_rect[:,2] ；depth = pts_2d[:,2] - P2.T[3,2]
    逐元素的乘加链顺序与列式 numpy 版完全一致，故 mask 与 numpy 版逐位相同。
    """
    n = pts.shape[0]
    out = np.empty(n, dtype=np.bool_)
    for i in range(n):
        px = pts[i, 0]
        py = pts[i, 1]
        pz = pts[i, 2]
        # pts @ a[:3] + a[3]
        xr = (px * a[0, 0] + py * a[1, 0] + pz * a[2, 0]) + a[3, 0]
        yr = (px * a[0, 1] + py * a[1, 1] + pz * a[2, 1]) + a[3, 1]
        zr = (px * a[0, 2] + py * a[1, 2] + pz * a[2, 2]) + a[3, 2]
        # pts_rect @ P2[:,:3].T + P2[:,3]
        x2 = (xr * p2[0, 0] + yr * p2[1, 0] + zr * p2[2, 0]) + p2_c[0]
        y2 = (xr * p2[0, 1] + yr * p2[1, 1] + zr * p2[2, 1]) + p2_c[1]
        z2 = (xr * p2[0, 2] + yr * p2[1, 2] + zr * p2[2, 2]) + p2_c[2]
        ui = x2 / zr
        vi = y2 / zr
        depth = z2 - p2_t32
        out[i] = (
            (ui >= 0.0) and (ui < img_w) and (vi >= 0.0) and (vi < img_h) and (depth >= 0.0)
        )
    return out


class DemoDataset(DatasetTemplate):
    """与 tools/demo.py 的 DemoDataset 完全一致：读任意 .bin/.npy 点云做推理。"""

    def __init__(
        self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext=".bin"
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            "points": points,
            "frame_id": index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


# ============================================================
# OM 推理 helpers（被 npu/om_ref_test.py 等复用，勿删）
# ============================================================


def to_tensor(x):
    """ndarray → CPU tensor；tensor 原样返回（设备常驻模式下为 NPU tensor，不再强制 .cpu()）。"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    return x


def _is_npu_tensor(x):
    return isinstance(x, torch.Tensor) and getattr(x, "is_npu", False)


def collate_batch_fast(batch_list):
    """单帧 collate fast-path：跳过 list-of-dict + 通用 collate_batch 的拷贝开销。

    与 DatasetTemplate.collate_batch 对单帧的输出保持同格式：
      - voxels / voxel_num_points 直接引用原数组（OM 路径只读消费，不拷贝）；
      - voxel_coords 补 batch 列 -> (M, 4) int32（col0=0，与 np.pad 一致）；
      - 标量/矩阵 key 与通用 collate 相同 np.stack。
    唯一偏差：'points' 不补 batch 列（返回 (N, 4) 原引用）——OM 推理路径不消费
    points，省去 ~1ms 的 (N,5) 填充拷贝；需要标准 collate 输出时请用
    DatasetTemplate.collate_batch。
    多帧 batch（len>1）回退到通用 collate_batch，保证兼容。
    """
    if len(batch_list) == 1:
        dd = batch_list[0]
        if any(_is_npu_tensor(v) for v in dd.values()):
            # 设备常驻模式：voxels/coords/num_points 为 NPU tensor —— torch 单帧
            # fast path（引用 + F.pad 补 batch 列，全 device 侧无 host 往返；
            # 单帧 batch idx=0，语义与 numpy 版同格式）。
            ret = {}
            for key, val in dd.items():
                if key in ("voxels", "voxel_num_points", "points"):
                    ret[key] = val  # 单帧无 batch 维，直接引用（只读消费）
                elif key == "voxel_coords":
                    ret[key] = torch.nn.functional.pad(val, (1, 0), mode="constant", value=0)
                else:
                    ret[key] = np.stack([val], axis=0)
            ret["batch_size"] = 1
            return ret
        ret = {}
        for key, val in dd.items():
            if key in ("voxels", "voxel_num_points"):
                ret[key] = val  # 单帧无 batch 维，直接引用（只读消费）
            elif key == "voxel_coords":
                out = np.zeros((val.shape[0], val.shape[1] + 1), dtype=val.dtype)
                out[:, 1:] = val
                ret[key] = out
            elif key == "points":
                ret[key] = val  # 不补 batch 列，见 docstring
            else:
                ret[key] = np.stack([val], axis=0)
        ret["batch_size"] = 1
        return ret
    return DatasetTemplate.collate_batch(batch_list)


def fov_filter_fused(points, calib, img_shape):
    """FOV 过滤（numba 单内核，与 get_fov_flag 同乘加顺序）。

    与 calib.lidar_to_rect + get_fov_flag 同数学（乘加顺序一致）：
      A = V2C.T @ R0.T (4,3)；pts_rect = points @ A[:3] + A[3]
      pts_2d = pts_rect @ P2[:,:3].T + P2[:,3]；pts_img = pts_2d[:,:2]/pts_rect[:,2:3]
      depth = pts_2d[:,2] - P2.T[3,2]
    返回与 get_fov_flag 一致的布尔 mask。
    """
    pts = points[:, :3]
    a = calib.V2C.T @ calib.R0.T  # (4, 3)
    return _fov_filter_numba(
        np.ascontiguousarray(pts, dtype=np.float32),
        np.ascontiguousarray(a, dtype=np.float32),
        np.ascontiguousarray(calib.P2[:, :3].T, dtype=np.float32),
        np.ascontiguousarray(calib.P2[:, 3], dtype=np.float32),
        calib.P2.T[3, 2],
        int(img_shape[1]),
        int(img_shape[0]),
    )


# 模块级 pad 缓冲复用（按 (m_target, *shape) 缓存，避免每帧 torch.cat + new_zeros 重分配）。
# 注意：返回的 tensor 与下一帧共享底层缓冲，调用方须在下一帧 pad 前完成消费（现有调用点均满足）。
_PAD_BUFS = {}


def pad_to_static_m(voxels, voxel_num_points, voxel_coords, m_target):
    """把非空 pillar 张量 pad 到静态 OM 的固定 M（P0）。

    pad 行 num_points=0 → VFE 掩码掉、特征恒 0；Gather 索引表只引用真实行，
    空网格单元指向 pad 行（零特征）。与动态版输出 bit 一致（见 pointpillar_scatter 的
    bev_index_map Gather 路径）。返回 (voxels, num_points, coords, index_map)。

    缓冲复用：pad 目标固定时复用模块级预分配缓冲（原地清零 + 拷贝），
    输出数值与 torch.cat 版逐位一致。输入为 NPU tensor 时缓冲也在 NPU
    （设备常驻模式，清零/拷贝在 device 侧完成，无 host 往返）。
    """
    M = voxels.shape[0]
    if M > m_target:
        raise ValueError("M=%d > 静态 OM M=%d，超出范围" % (M, m_target))
    if M < m_target:
        key = (
            m_target,
            *tuple(voxels.shape[1:]),
            voxel_coords.shape[1],
            str(voxels.device) if isinstance(voxels, torch.Tensor) else "cpu",
        )
        bufs = _PAD_BUFS.get(key)
        if bufs is None:
            if isinstance(voxels, torch.Tensor):
                bufs = (
                    torch.zeros(
                        (m_target, *voxels.shape[1:]), dtype=voxels.dtype, device=voxels.device
                    ),
                    torch.zeros(
                        (m_target,), dtype=voxel_num_points.dtype, device=voxel_num_points.device
                    ),
                    torch.zeros(
                        (m_target, voxel_coords.shape[1]),
                        dtype=voxel_coords.dtype,
                        device=voxel_coords.device,
                    ),
                )
            else:
                bufs = (
                    torch.zeros((m_target, *voxels.shape[1:]), dtype=voxels.dtype),
                    torch.zeros((m_target,), dtype=voxel_num_points.dtype),
                    torch.zeros((m_target, voxel_coords.shape[1]), dtype=voxel_coords.dtype),
                )
            _PAD_BUFS[key] = bufs
        bv, bn, bc = bufs
        bv.zero_()
        bv[:M] = voxels
        bn.zero_()
        bn[:M] = voxel_num_points
        bc.zero_()
        bc[:M] = voxel_coords
        voxels, voxel_num_points, voxel_coords = bv, bn, bc
    # 索引表只由真实行（前 M 个）构造，pad 值 = m_target
    index_map = build_index_map(voxel_coords[:M], M=M, pad=m_target)
    return voxels, voxel_num_points, voxel_coords, index_map


def coords_int32(x):
    """voxel_coords → int32（tensor / ndarray 双态）。"""
    if isinstance(x, torch.Tensor):
        return x.to(torch.int32)
    return x.astype(np.int32)


def mkfeeds(voxels, voxel_num_points, voxel_coords, bev_index_map, device):
    """构造 OM 输入 feeds。

    NPU 常驻 tensor（设备常驻模式）用 aclruntime.BaseTensor 直接包设备内存
    （零拷贝，实测与 Tensor(numpy)+to_device 路径输出 bit 一致）；CPU
    tensor / ndarray 走原 Tensor(host)+to_device 路径。

    ⚠️ 交给 aclruntime 前必须 torch.npu.synchronize()：collate/index_map 的
    torch NPU op 是异步 enqueue，GE 不在 torch 流上、不会等它——不同步则 GE
    读到未写完的 device 内存（demo 实测稳定 0 框的根因）。同步约 0.05ms，
    这些等待本来就要发生，只是显式化。
    """
    fs = []
    dev_resident = any(
        isinstance(t, torch.Tensor) and getattr(t, "is_npu", False)
        for t in (voxels, voxel_num_points, voxel_coords, bev_index_map)
    )
    if dev_resident:
        torch.npu.synchronize()
    for t in (voxels, voxel_num_points, voxel_coords, bev_index_map):
        if dev_resident:
            fs.append(aclruntime.BaseTensor(t.data_ptr(), t.numel() * t.element_size()))
        else:
            arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
            ft = aclruntime.Tensor(np.ascontiguousarray(arr))
            ft.to_device(device)
            fs.append(ft)
    return fs


def tensor_to_numpy(t, dtype, copy=True):
    """aclruntime.Tensor -> numpy（先 to_host 搬到 host 内存）。

    copy=False 时返回 aclruntime host 缓冲的直接视图（writable，可喂 torch），
    省去整块 memcpy；调用方须在下次 session.run 前消费完（后处理路径满足）。
    """
    t.to_host()
    arr = np.frombuffer(memoryview(t), dtype=dtype).reshape(t.shape)
    return arr.copy() if copy else arr


def postprocess_topk(om_box, om_cls, score_thresh, nms_config):
    """OM topk 输出的 host 后处理（numpy max/argmax + torch sigmoid + topk_nms）。

    与 torch 版（torch.max + sigmoid + class_agnostic_nms）**逐位一致**：
      - numpy .max/.argmax 与 torch.max 的值/下标一致（已验证）
      - sigmoid 保留 torch（numpy 1/(1+exp(-x)) 与 torch.sigmoid 有 1ULP 差异，不能替换）
      - NMS 用 unum_ops.cv.topk_nms（numba 增量贪心，与 class_agnostic_nms 同结果）
    省掉 torch.max 在 (4096,3) 小张量上的调度开销（~3.8ms -> ~0.05ms）。

    Args:
        om_box: (1, N, 7) float32 numpy（可为 aclruntime host 视图）
        om_cls: (1, N, 3) float32 numpy
        score_thresh / nms_config: 同 unum_ops.cv.topk_nms
    Returns:
        boxes: (K, 7) float32 numpy
        labels: (K,) int64 numpy（1-indexed class id）
        scores: (K,) float32 numpy（sigmoid 后）
    """
    b = om_box.reshape(1, -1, 7)[0]
    c0 = om_cls.reshape(1, -1, 3)[0]
    cls_max = c0.max(axis=-1)
    label = c0.argmax(axis=-1) + 1
    scores = torch.sigmoid(torch.from_numpy(cls_max))
    selected, sel_scores = topk_nms(b, scores, score_thresh, nms_config)
    return b[selected], label[selected], sel_scores


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
    from unum_ops.cv import boxes_iou_bev

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
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default=str(ROOT / "tools/cfgs/kitti_models/pointpillar.yaml"),
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT / "data/kitti/training/velodyne/000008.bin"),
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--om",
        type=str,
        default=str(
            ROOT / "weights/pointpillar_base_fp16_dynamic18000_topk_surgery_abc_linux_aarch64.om"
        ),
        help="specify the OM model",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="与 label 匹配的 IoU 阈值")
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=30,
        help="每帧性能计时执行遍数（各段取中位；默认 30 与 PERFORMANCE.md §1.3 同口径，"
        "1=预热后单发，大批量冒烟可调小）",
    )
    parser.add_argument("--label", default=None, help="KITTI label 文件路径（默认自动推断）")

    args = parser.parse_args()

    # kitti_models 配置的 _BASE_CONFIG_ 相对 tools/ 解析，与 demo.py 一致
    cfg_path = Path(args.cfg_file)
    if "tools/cfgs" in str(cfg_path) and os.getcwd() != str(ROOT / "tools"):
        os.chdir(str(ROOT / "tools"))

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
    logger.info("-----------------Quick Demo of OpenPCDet (OM)-------------------------")
    # 必修：demo/bench 此前从不调用 torch.npu.set_device，patched generate() 的
    # .npu() 走默认 device 0，与 session 的 args.device 错位（跨设备访问/错设备占用）。
    if getattr(torch, "npu", None) is not None and torch.npu.is_available():
        torch.npu.set_device(args.device)
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=resolve_path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    # 构建 OM session（代替 demo 里的 build_network + load_params_from_file）
    session = InferenceSession(
        str(resolve_path(args.om)), args.device, aclruntime.session_options()
    )
    out_names = [d.name for d in session.get_outputs()]
    expect_m = session.get_inputs()[0].shape[0]
    base_mode = len(out_names) == 2  # base OM 输出 box+cls；否则为内嵌 NMS 的 OM

    nms_config = cfg.MODEL.POST_PROCESSING.NMS_CONFIG
    score_thresh = (
        args.score_thresh
        if args.score_thresh is not None
        else cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    )

    # numba 预热：topk_nms 旋转 NMS 首次 JIT ~1.5s，移出首帧
    try:
        topk_nms(
            np.zeros((2, 7), dtype=np.float32),
            np.array([0.9, 0.8], dtype=np.float32),
            0.1,
            nms_config,
        )
    except Exception:
        pass

    def run_frame(idx, warm_om=True):
        """单帧完整流水线：getitem → collate/to_tensor → pad/index_map → feeds → forward → postproc。

        返回 (boxes, labels, scores, stage_ms)。stage_ms 为六段耗时(ms)，口径与
        PERFORMANCE.md §1.3 对齐。warm_om=False 时不做 OM 预热执行（多遍取中位时
        首遍冷启动由中位吸收，省一遍执行）。M 超静态 OM 上限返回 None。
        """
        t0 = time.perf_counter()
        data_dict = demo_dataset[idx]  # getitem：读 bin + processors（AscendC 体素化）
        t1 = time.perf_counter()
        data_dict = collate_batch_fast([data_dict])

        voxels = to_tensor(data_dict["voxels"])
        voxel_num_points = to_tensor(data_dict["voxel_num_points"])
        voxel_coords = to_tensor(coords_int32(data_dict["voxel_coords"]))
        t2 = time.perf_counter()
        M = voxels.shape[0]
        if expect_m is not None and expect_m > 0:
            # 静态 shape OM：M < 固定值时 pad，M 超限则跳过
            if M > expect_m:
                logger.warning(
                    f"frame {Path(demo_dataset.sample_file_list[idx]).stem}: M={M} "
                    f"超过静态 OM M={expect_m}，跳过（请用动态 OM 或更大的静态 OM）"
                )
                return None
            voxels, voxel_num_points, voxel_coords, bev_index_map = pad_to_static_m(
                voxels, voxel_num_points, voxel_coords, expect_m
            )
        else:
            bev_index_map = build_index_map(voxel_coords, M=M)
        t3 = time.perf_counter()

        feeds = mkfeeds(voxels, voxel_num_points, voxel_coords, bev_index_map, args.device)
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
        t4 = time.perf_counter()

        # OM 预热：首次 aclmdlExecute 含内核装载/workspace 分配等一次性开销
        # （实测冷 17-20ms vs 稳态 10.7ms），单发计时前先跑一次不计时的同 shape 执行
        if warm_om:
            session.run(out_names, feeds)
            t4w = time.perf_counter()
        else:
            t4w = t4
        out = session.run(out_names, feeds)
        t5 = time.perf_counter()

        if base_mode:
            om_box = tensor_to_numpy(out[0], np.float32, copy=False).reshape(1, -1, 7)
            om_cls = tensor_to_numpy(out[1], np.float32, copy=False).reshape(1, -1, 3)
            # numpy max/argmax + torch sigmoid + topk_nms（与 torch 版逐位一致）
            boxes, labels, scores = postprocess_topk(om_box, om_cls, score_thresh, nms_config)
        else:
            # 内嵌 NMS 的 OM：输出 nms_final_boxes/scores/labels/count
            boxes = tensor_to_numpy(out[0], np.float32).reshape(-1, 7)
            scores = tensor_to_numpy(out[1], np.float32)
            labels = tensor_to_numpy(out[2], np.int64) + 1
            n_count = (
                tensor_to_numpy(out[3], np.int64).reshape(-1)[0] if len(out) >= 4 else len(boxes)
            )
            boxes, scores, labels = boxes[:n_count], scores[:n_count], labels[:n_count]

        t6 = time.perf_counter()
        st = {
            "getitem": (t1 - t0) * 1e3,
            "collate": (t2 - t1) * 1e3,
            "idxmap": (t3 - t2) * 1e3,
            "feeds": (t4 - t3) * 1e3,
            "forward": (t5 - t4w) * 1e3,
            "postproc": (t6 - t5) * 1e3,
        }
        return boxes, labels, scores, st

    for idx in range(len(demo_dataset)):
        logger.info(f"Visualized sample index: \t{idx + 1}")
        frame_id = Path(demo_dataset.sample_file_list[idx]).stem
        n_perf = max(1, args.perf_iters)
        if n_perf == 1:
            # 单发模式：先冷跑一遍吸收一次性开销（numba 装载 / AscendC 算子首次加载 /
            # NPU allocator 首分配，实测冷态 E2E ~270ms vs 稳态 ~22.5ms），结果丢弃
            run_frame(idx)
        # 多遍取各段中位：整链路有 ~8-10 遍的时钟爬坡（DVFS，25.3→22.8ms 才进稳态），
        # 采样窗太短会落在衰减带；默认 30 遍与 PERFORMANCE.md §1.3 基线同口径
        rets = []
        for _ in range(n_perf):
            r = run_frame(idx, warm_om=(n_perf == 1))
            if r is None:
                break
            rets.append(r)
        if not rets:
            continue
        boxes, labels, scores, _ = rets[0]
        med = {k: sorted(r[3][k] for r in rets)[len(rets) // 2] for k in rets[0][3]}
        print(
            "[perf] E2E {e:.2f}ms (n={n} 中位) | getitem(读bin+体素化) {g:.2f} | "
            "collate→tensor {c:.2f} | pad/index_map {im:.2f} | feeds(含set_dym) {f:.2f} | "
            "forward {fw:.2f} | postproc(含D2H) {p:.2f}".format(
                n=len(rets),
                e=sum(med.values()),
                g=med["getitem"],
                c=med["collate"],
                im=med["idxmap"],
                f=med["feeds"],
                fw=med["forward"],
                p=med["postproc"],
            )
        )

        print("\n%s 检测结果 (%d 个框):" % (frame_id, len(boxes)))
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

        # Label 对比（可选）
        label_path = args.label
        if label_path is None:
            auto = ROOT / "data" / "kitti" / "training" / "label_2" / f"{frame_id}.txt"
            if auto.exists():
                label_path = str(auto)
        if label_path and Path(label_path).exists():
            gt_objs = load_kitti_labels(label_path)
            evaluate_against_labels(boxes, labels, scores, gt_objs, class_names, args.iou_thresh)
        else:
            print("\n(未找到 label 文件，跳过对比。可用 --label 指定)")

    logger.info("Demo done.")


if __name__ == "__main__":
    main()
