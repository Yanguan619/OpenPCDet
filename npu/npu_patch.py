"""NPU 适配补丁：统一管理设备检测、算子适配与可重复调用的初始化入口。

被 npu/infer.py 与 npu/eval.py 复用，集中处理：
1. 设备检测（npu / cuda / cpu）
2. torch_npu 初始化（关闭 jit_compile 以避免逐帧编译）
3. anchors / tensor 的跨设备搬运
4. voxelization 相关适配（build_index_map 等）

用法:
    from npu.npu_patch import init_patch, patch_rotate_iou, get_device, to_tensor, build_index_map, PPWrapper
    device = init_patch()
"""

import math
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import numba
import numpy as np
import torch


def _alias_spconv():
    """将 spconv / spconv.utils 别名到 unum_ops 的 spconv shim（sys.modules 注入）。

    使 `from spconv.utils import VoxelGeneratorV2` 无需仓库根目录的 spconv 符号链接，
    任何机器（含无 unum_ops 相邻部署）都能通过本补丁解析。必须在 import pcdet 之前调用。
    """
    try:
        import unum_ops.spconv
        import unum_ops.spconv.utils
    except ImportError:
        return
    sys.modules.setdefault('spconv', sys.modules['unum_ops.spconv'])
    sys.modules.setdefault('spconv.utils', sys.modules['unum_ops.spconv.utils'])


def _patch_voxelize_ascendc():
    """将 VoxelGeneratorV2.generate 路由到 unum_ops 的 AscendC 硬体素化（NPU kernel）。

    输出与 CPU numba 版逐位一致（voxels/npp/coords 已验证 2306/2306），仅 coords 序
    为 (x,y,z) → 转回 spconv 的 (z,y,x)。全量点（demo 无 FOV）~2x 快，FOV 后小输入
    持平。任何异常自动回退 CPU 原路径。可用环境变量 NPU_ASCENDC_VOXELIZE=0 关闭。
    """
    if os.environ.get('NPU_ASCENDC_VOXELIZE', '1') != '1':
        return
    try:
        from unum_ops.voxelization.voxelization_ascendc_v2 import voxelization
        from unum_ops.spconv.utils import VoxelGeneratorV2
    except Exception:
        return
    _orig_generate = VoxelGeneratorV2.generate

    def generate(self, points):
        try:
            p = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).npu()
            out = voxelization(
                p,
                voxel_size=[float(v) for v in self.voxel_size],
                pcr=[float(v) for v in self.point_cloud_range],
                max_num_points=self.max_num_points,
                max_voxels=self.max_voxels,
            )
            return {
                'voxels': out.voxels.cpu().numpy(),
                'coordinates': out.coords.cpu().numpy()[:, [2, 1, 0]],
                'num_points_per_voxel': out.num_points.cpu().numpy(),
            }
        except Exception:
            return _orig_generate(self, points)

    VoxelGeneratorV2.generate = generate

_F32 = np.float32


# ---------------------------------------------------------------------------
# CPU-only rotated box IoU (BEV)：逐行复刻官方 numba CUDA 数学，数值一致。
# 输入格式: [x, z, w, l, ry] (camera BEV 坐标, ry 绕 y 轴)
# 输出: (N, K) IoU / 重叠矩阵
# ---------------------------------------------------------------------------


@numba.njit(nopython=True)
def _cpu_trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
            (b[0] - c[0])) / _F32(2.0)


@numba.njit(nopython=True)
def _cpu_area(int_pts, num_of_inter):
    area_val = _F32(0.0)
    for i in range(num_of_inter - 2):
        area_val += abs(
            _cpu_trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                              int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@numba.njit(nopython=True)
def _cpu_sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.zeros((2, ), dtype=_F32)
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.zeros((2, ), dtype=_F32)
        vs = np.zeros((16, ), dtype=_F32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1
                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@numba.njit(nopython=True)
def _cpu_line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = np.zeros((2, ), dtype=_F32)
    B = np.zeros((2, ), dtype=_F32)
    C = np.zeros((2, ), dtype=_F32)
    D = np.zeros((2, ), dtype=_F32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@numba.njit(nopython=True)
def _cpu_point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@numba.njit(nopython=True)
def _cpu_quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if _cpu_point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if _cpu_point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.zeros((2, ), dtype=_F32)
    for i in range(4):
        for j in range(4):
            has_pts = _cpu_line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@numba.njit(nopython=True)
def _cpu_rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.zeros((4, ), dtype=_F32)
    corners_y = np.zeros((4, ), dtype=_F32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@numba.njit(nopython=True)
def _cpu_inter(rbbox1, rbbox2):
    corners1 = np.zeros((8, ), dtype=_F32)
    corners2 = np.zeros((8, ), dtype=_F32)
    intersection_corners = np.zeros((16, ), dtype=_F32)

    _cpu_rbbox_to_corners(corners1, rbbox1)
    _cpu_rbbox_to_corners(corners2, rbbox2)

    num_intersection = _cpu_quadrilateral_intersection(corners1, corners2,
                                                       intersection_corners)
    _cpu_sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    return _cpu_area(intersection_corners, num_intersection)


@numba.njit(nopython=True)
def _cpu_dev_rotate_iou_eval(rbox1, rbox2, criterion):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = _cpu_inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@numba.njit(nopython=True, parallel=False)
def _cpu_rotate_iou_loop(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion):
    for tx in range(N):
        for i in range(K):
            dev_iou[tx * K + i] = _cpu_dev_rotate_iou_eval(
                dev_query_boxes[i * 5:i * 5 + 5],
                dev_boxes[tx * 5:tx * 5 + 5], criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou. CPU numba 版，逐行复刻官方 CUDA 数学。

    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims,
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): 兼容接口保留.

    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    _cpu_rotate_iou_loop(N, K, boxes.reshape([-1]), query_boxes.reshape([-1]),
                         iou.reshape([-1]), criterion)
    return iou.astype(box_dtype)


_ROTATE_IOU_MODULE_NAMES = [
    "pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou",
]


# ---------------------------------------------------------------------------
# CUDA ops 的 sys.modules 预注入 stub：把「CUDA 扩展导入失败→native 降级」逻辑
# 从 pcdet 源码抽离，集中到 npu_patch.py。必须在 `import pcdet` 之前调用，
# 否则 pcdet 源码（已恢复 HEAD 的无 try/except 版本）会在 `from . import xxx_cuda`
# 处抛 ImportError。
# ---------------------------------------------------------------------------


def _cuda_ext_available():
    """真实 CUDA 扩展是否可导入（可导入则保留 pcdet 原模块，不注入 stub）。"""
    try:
        import pcdet.ops.iou3d_nms.iou3d_nms_cuda  # noqa: F401
        return True
    except ImportError:
        return False


def _make_stub(fullname, attrs):
    mod = types.ModuleType(fullname)
    mod.__name__ = fullname
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[fullname] = mod
    return mod


def patch_cuda_ops():
    """预注入 CUDA 扩展缺失时的 native 降级 stub（幂等，可重复调用）。

    必须在任意 `import pcdet` 之前调用一次。注入后 pcdet 的：
      pcdet.ops.iou3d_nms.iou3d_nms_utils
      pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils
      pcdet.ops.ingroup_inds.ingroup_inds_op
      pcdet.ops.bev_pool.bev_pool
      pcdet.ops.roipoint_pool3d.roipoint_pool3d_utils
      pcdet.models.dense_heads.target_assigner.hungarian_assigner
    会命中 sys.modules 中的 stub（提供与 HEAD 源码一致的公开接口），
    从而不再执行真实的 try/except 缺失逻辑。CUDA 环境不注入。
    """
    if _cuda_ext_available():
        return
    _patch_iou3d_nms_utils()
    _patch_roiaware_pool3d_utils()
    _patch_ingroup_inds()
    _patch_bev_pool()
    _patch_roipoint_pool3d()
    _patch_hungarian_assigner()
    try:
        from npu.npu_patch_pointnet2 import patch_pointnet2_ops
        patch_pointnet2_ops()
    except Exception:
        import traceback
        traceback.print_exc()


def _patch_iou3d_nms_utils():
    from npu.ops_native import iou3d_nms_torch_native as n
    _make_stub("pcdet.ops.iou3d_nms.iou3d_nms_utils", {
        "boxes_bev_iou_cpu": n.boxes_bev_iou_cpu,
        "boxes_iou_bev": n.boxes_iou_bev,
        "boxes_iou3d_gpu": n.boxes_iou3d_gpu,
        "boxes_aligned_iou3d_gpu": n.boxes_aligned_iou3d_gpu,
        "paired_boxes_iou3d_gpu": n.paired_boxes_iou3d_gpu,
        "nms_gpu": n.nms_gpu,
        "nms_normal_gpu": n.nms_normal_gpu,
    })


def _patch_roiaware_pool3d_utils():
    from npu.ops_native import roiaware_pool3d_torch_native as n
    from pcdet.utils import common_utils

    def points_in_boxes_cpu(points, boxes):
        points, is_numpy = common_utils.check_numpy_to_torch(points)
        boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)
        indices = n.points_in_boxes_cpu(
            points.float().contiguous(), boxes.float().contiguous())
        return indices.numpy() if is_numpy else indices

    def points_in_boxes_gpu(points, boxes):
        return n.points_in_boxes_gpu(points.contiguous(), boxes.contiguous())

    _make_stub("pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils", {
        "points_in_boxes_cpu": points_in_boxes_cpu,
        "points_in_boxes_gpu": points_in_boxes_gpu,
        "RoIAwarePool3d": n.RoIAwarePool3d,
        "RoIAwarePool3dFunction": n.RoIAwarePool3dFunction,
    })


def _patch_ingroup_inds():
    def ingroup_inds(group_inds):
        return _ingroup_inds_native(group_inds)

    _make_stub("pcdet.ops.ingroup_inds.ingroup_inds_op", {
        "ingroup_inds": ingroup_inds,
        "ingroup_inds_native": _ingroup_inds_native,
    })


def _ingroup_inds_native(group_inds):
    """ingroup_inds 的 torch-native 实现（排序求组内 0-based 序号）。"""
    out_inds = torch.full_like(group_inds, -1)
    N = group_inds.numel()
    if N == 0:
        return out_inds
    sorted_groups, order = torch.sort(group_inds)
    is_new = torch.ones(N, dtype=torch.bool, device=group_inds.device)
    is_new[1:] = sorted_groups[1:] != sorted_groups[:-1]
    seq = torch.arange(N, dtype=torch.long, device=group_inds.device)
    first_pos = torch.where(is_new, seq, torch.zeros_like(seq))
    first_pos = torch.cummax(first_pos, dim=0).values
    within = seq - first_pos
    out_inds[order] = within
    return out_inds


def _patch_bev_pool():
    def bev_pool(feats, coords, B, D, H, W):
        assert feats.shape[0] == coords.shape[0]
        ranks = (
            coords[:, 0] * (W * D * B)
            + coords[:, 1] * (D * B)
            + coords[:, 2] * B
            + coords[:, 3]
        )
        indices = ranks.argsort()
        feats, coords, ranks = feats[indices], coords[indices], ranks[indices]
        return _bev_pool_native(feats, coords, ranks, B, D, H, W)

    _make_stub("pcdet.ops.bev_pool.bev_pool", {
        "bev_pool": bev_pool,
        "bev_pool_native": _bev_pool_native,
        "QuickCumsum": _QuickCumsum,
    })


class _QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        ctx.save_for_backward(kept)
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None


def _bev_pool_native(feats, coords, ranks, B, D, H, W):
    """bev_pool 的 torch-native 实现。feats/coords 已按 ranks 排序。"""
    x, geom_feats = _QuickCumsum.apply(feats, coords, ranks)
    batch_ix = geom_feats[:, 3]
    z_ix = geom_feats[:, 2]
    y_ix = geom_feats[:, 1]
    x_ix = geom_feats[:, 0]
    flat_ix = batch_ix * (D * H * W) + z_ix * (H * W) + y_ix * W + x_ix
    out = feats.new_zeros((B * D * H * W, feats.shape[1]))
    out[flat_ix.long()] = x
    out = out.view(B, D, H, W, feats.shape[1])
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out


def _patch_roipoint_pool3d():
    from pcdet.utils import box_utils

    def roipoint_pool3d_native(points, pooled_boxes3d, point_features, pooled_features, pooled_empty_flag):
        batch_size, pts_num, _ = points.shape
        boxes_num = pooled_boxes3d.shape[1]
        sampled_pts_num = pooled_features.shape[2]
        feature_len = point_features.shape[2]
        S = sampled_pts_num
        MARGIN = 1e-5
        for b in range(batch_size):
            pts = points[b]
            feats = point_features[b]
            boxes = pooled_boxes3d[b]
            cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
            dx, dy, dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]
            rz = boxes[:, 6]
            shift = pts[None, :, :] - boxes[:, None, :3]
            cosa = torch.cos(-rz)
            sina = torch.sin(-rz)
            local_x = shift[..., 0] * cosa[:, None] - shift[..., 1] * sina[:, None]
            local_y = shift[..., 0] * sina[:, None] + shift[..., 1] * cosa[:, None]
            local_z = shift[..., 2]
            in_x = local_x.abs() < (dx[:, None] / 2.0 + MARGIN)
            in_y = local_y.abs() < (dy[:, None] / 2.0 + MARGIN)
            in_z = local_z.abs() <= (dz[:, None] / 2.0)
            mask = in_x & in_y & in_z
            for m in range(boxes_num):
                in_pts = torch.nonzero(mask[m]).flatten()
                cnt = in_pts.numel()
                if cnt == 0:
                    pooled_empty_flag[b, m] = 1
                    continue
                if cnt >= S:
                    idx = in_pts[:S]
                else:
                    pad = in_pts[torch.arange(S - cnt, device=in_pts.device) % cnt]
                    idx = torch.cat([in_pts, pad])
                pooled_features[b, m, :, :3] = pts[idx]
                pooled_features[b, m, :, 3:] = feats[idx]

    class RoIPointPool3dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
            assert points.shape.__len__() == 3 and points.shape[2] == 3
            batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[1], point_features.shape[2]
            pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)
            pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
            pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()
            roipoint_pool3d_native(points, pooled_boxes3d, point_features, pooled_features, pooled_empty_flag)
            return pooled_features, pooled_empty_flag

        @staticmethod
        def backward(ctx, grad_out):
            raise NotImplementedError

    class RoIPointPool3d(torch.nn.Module):
        def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
            super().__init__()
            self.num_sampled_points = num_sampled_points
            self.pool_extra_width = pool_extra_width

        def forward(self, points, point_features, boxes3d):
            return RoIPointPool3dFunction.apply(
                points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points)

    _make_stub("pcdet.ops.roipoint_pool3d.roipoint_pool3d_utils", {
        "RoIPointPool3d": RoIPointPool3d,
        "RoIPointPool3dFunction": RoIPointPool3dFunction,
        "roipoint_pool3d_native": roipoint_pool3d_native,
    })


def _patch_hungarian_assigner():
    from scipy.optimize import linear_sum_assignment
    from npu.ops_native import iou3d_nms_torch_native as iou3d_nms_native

    def height_overlaps(boxes1, boxes2):
        boxes1_top_height = (boxes1[:, 2] + boxes1[:, 5]).view(-1, 1)
        boxes1_bottom_height = boxes1[:, 2].view(-1, 1)
        boxes2_top_height = (boxes2[:, 2] + boxes2[:, 5]).view(1, -1)
        boxes2_bottom_height = boxes2[:, 2].view(1, -1)
        heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    def overlaps(boxes1, boxes2):
        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.new(rows, cols)
        overlaps_h = height_overlaps(boxes1, boxes2)
        boxes1_bev = boxes1[:, :7]
        boxes2_bev = boxes2[:, :7]
        overlaps_bev = iou3d_nms_native.box_overlap_bev(
            boxes1_bev.contiguous(), boxes2_bev.contiguous()).to(boxes1.device)
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h
        volume1 = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]).view(-1, 1)
        volume2 = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]).view(1, -1)
        iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)
        return iou3d

    class HungarianAssigner3D:
        def __init__(self, cls_cost, reg_cost, iou_cost):
            self.cls_cost = cls_cost
            self.reg_cost = reg_cost
            self.iou_cost = iou_cost

        def focal_loss_cost(self, cls_pred, gt_labels):
            weight = self.cls_cost.get('weight', 0.15)
            alpha = self.cls_cost.get('alpha', 0.25)
            gamma = self.cls_cost.get('gamma', 2.0)
            eps = self.cls_cost.get('eps', 1e-12)
            cls_pred = cls_pred.sigmoid()
            neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
            pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
            cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
            return cls_cost * weight

        def bevbox_cost(self, bboxes, gt_bboxes, point_cloud_range):
            weight = self.reg_cost.get('weight', 0.25)
            pc_start = bboxes.new(point_cloud_range[0:2])
            pc_range = bboxes.new(point_cloud_range[3:5]) - bboxes.new(point_cloud_range[0:2])
            normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
            normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
            reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
            return reg_cost * weight

        def iou3d_cost(self, bboxes, gt_bboxes):
            iou = overlaps(bboxes, gt_bboxes)
            weight = self.iou_cost.get('weight', 0.25)
            return -iou * weight, iou

        def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range):
            num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
            assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            if num_gts == 0 or num_bboxes == 0:
                if num_gts == 0:
                    assigned_gt_inds[:] = 0
                return num_gts, assigned_gt_inds, None, assigned_labels
            cls_cost = self.focal_loss_cost(cls_pred[0].T, gt_labels)
            reg_cost = self.bevbox_cost(bboxes, gt_bboxes, point_cloud_range)
            iou_cost, iou = self.iou3d_cost(bboxes, gt_bboxes)
            cost = cls_cost + reg_cost + iou_cost
            cost = cost.detach().cpu()
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)
            assigned_gt_inds[:] = 0
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
            max_overlaps = torch.zeros_like(iou.max(1).values)
            max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
            return assigned_gt_inds, max_overlaps

    _make_stub("pcdet.models.dense_heads.target_assigner.hungarian_assigner", {
        "HungarianAssigner3D": HungarianAssigner3D,
        "height_overlaps": height_overlaps,
        "overlaps": overlaps,
    })


def patch_rotate_iou():
    """将 CUDA 版 rotate_iou 替换为 CPU numba 实现（sys.modules 预注入）。

    必须在 import kitti 评测模块之前调用，否则原 CUDA 版
    （from numba import cuda）会因缺少 CUDA 驱动而导入失败。

    注意: 仅替换 KITTI 的 rotate_iou；ONCE 评测（once_eval/iou_utils.py）
    的 point_in_quadrilateral 用叉积法，数值语义不同，不在本补丁范围内。
    """
    for name in _ROTATE_IOU_MODULE_NAMES:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__name__ = name
            mod.rotate_iou_gpu_eval = rotate_iou_gpu_eval
            sys.modules[name] = mod


def get_device():
    """自动检测运行设备: npu > cuda > cpu."""
    if getattr(torch, "npu", None) is not None and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_patch(jit_compile=False):
    """NPU 适配初始化入口，可重复调用。

    - 检测并返回运行设备
    - 在 NPU 上关闭 jit_compile，避免模型逐帧编译拖慢推理
    - 补充 unum_ops spconv shim 依赖路径

    Returns:
        str: "npu" / "cuda" / "cpu"
    """
    device = get_device()
    if device == "npu":
        torch.npu.set_compile_mode(jit_compile=jit_compile)
    if device != "cuda":
        patch_rotate_iou()
    _alias_spconv()
    _patch_voxelize_ascendc()
    return device


def to_device(tensor, device):
    """将 tensor 搬运到指定设备（npu/cuda/cpu）。"""
    if device == "npu":
        return tensor.npu()
    if device == "cuda":
        return tensor.cuda()
    return tensor.cpu()


def to_tensor(data_dict, device, keys=None):
    """将 data_dict 中的 numpy 数组转为 tensor 并搬运到 device。

    跳过非 ndarray 及 ['frame_id', 'metadata', 'calib'] 元数据字段。
    """
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ["frame_id", "metadata", "calib"]:
            continue
        data_dict[key] = to_device(torch.from_numpy(val), device)
    return data_dict


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None):
    """由 voxel_coords 构造 BEV index map（scatter 用）。

    Args:
        voxel_coords: (M, 4) [batch, x, y, z]（或 [batch, z, y, x]）
        nx, ny, nz: 体素网格尺寸
        M: voxel 数（默认取 voxel_coords 行数）

    Returns:
        torch.LongTensor: 展平 BEV 网格 -> voxel 索引，空位为 M。
    """
    coords = voxel_coords.cpu().numpy()
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    if M is None:
        M = coords.shape[0]
    index_map = np.full(G, M, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map)


# 在 npu_patch 被 import 时即注入 CUDA ops 降级 stub，
# 确保任意 `from pcdet...` 之前 stub 已就位（幂等）。
patch_cuda_ops()


class PPWrapper(torch.nn.Module):
    """PointPillar module_list 前向封装：喂入 voxel 输入，输出 (box_preds, cls_preds)。

    等价于原模型 forward，但显式组装 batch_dict 并逐模块执行，
    便于直接拿到解码后的 batch_box_preds / batch_cls_preds。
    """

    def __init__(self, model):
        super().__init__()
        self.module_list = model.module_list

    def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
        batch_dict = {
            "voxels": voxels,
            "voxel_num_points": voxel_num_points,
            "voxel_coords": voxel_coords,
            "bev_index_map": bev_index_map,
            "batch_size": 1,
        }
        for m in self.module_list:
            batch_dict = m(batch_dict)
        return batch_dict["batch_box_preds"], batch_dict["batch_cls_preds"]
