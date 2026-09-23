"""
IoU3D NMS: Torch-native (纯 PyTorch) 实现
完全对应 OpenPCDet pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu 的 CUDA kernel 逻辑
支持 CPU / GPU，无需编译 CUDA

对应 iou3d_nms_utils.py 中全部 7 个公开函数:
  boxes_bev_iou_cpu, boxes_iou_bev, boxes_iou3d_gpu,
  boxes_aligned_iou3d_gpu, nms_gpu, nms_normal_gpu, paired_boxes_iou3d_gpu
"""

import torch
import torch.nn as nn
import numpy as np

EPS = 1e-8

# ============================================================
# CPU 旋转 IoU 矩阵（numba 快速路径，用于 nms_gpu）
# 与 _box_overlap_bev_pairs 相同几何（角点 + 凸多边形裁剪），但比
# torch 逐对向量化实现快 ~15x（000008 后处理 747ms -> ~60ms）。
# ============================================================
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    numba = None
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @numba.jit(nopython=True, cache=True)
    def _nms_corners(r, c):
        a_cos = np.cos(r[6])
        a_sin = np.sin(r[6])
        cx, cy, xd, yd = r[0], r[1], r[3], r[4]
        lx = np.array([-xd / 2, -xd / 2, xd / 2, xd / 2])
        ly = np.array([-yd / 2, yd / 2, yd / 2, -yd / 2])
        for i in range(4):
            c[i, 0] = a_cos * lx[i] + a_sin * ly[i] + cx
            c[i, 1] = -a_sin * lx[i] + a_cos * ly[i] + cy

    @numba.jit(nopython=True)
    def _nms_inter(c1, c2):
        ref = np.array(
            [c2[0, 0] + c2[1, 0] + c2[2, 0] + c2[3, 0],
             c2[0, 1] + c2[1, 1] + c2[2, 1] + c2[3, 1]],
            dtype=np.float32,
        )
        poly = np.empty((8, 2), dtype=np.float32)
        poly[:4] = c1
        n = 4
        out = np.empty((8, 2), dtype=np.float32)
        for e in range(4):
            ax, ay = c2[e, 0], c2[e, 1]
            bx, by = c2[(e + 1) % 4, 0], c2[(e + 1) % 4, 1]
            dx, dy = bx - ax, by - ay
            cref = dx * (ref[1] - 4 * ay) - dy * (ref[0] - 4 * ax)
            m = 0
            for k in range(n):
                px, py = poly[k, 0], poly[k, 1]
                qx, qy = poly[(k + 1) % n, 0], poly[(k + 1) % n, 1]
                pin = dx * (py - ay) - dy * (px - ax)
                qin = dx * (qy - ay) - dy * (qx - ax)
                ps = (pin >= 0) == (cref >= 0)
                qs = (qin >= 0) == (cref >= 0)
                if ps:
                    if qs:
                        out[m, 0], out[m, 1] = qx, qy
                        m += 1
                    else:
                        t = pin / (pin - qin)
                        out[m, 0] = px + t * (qx - px)
                        out[m, 1] = py + t * (qy - py)
                        m += 1
                elif qs:
                    t = pin / (pin - qin)
                    out[m, 0] = px + t * (qx - px)
                    out[m, 1] = py + t * (qy - py)
                    m += 1
                    out[m, 0], out[m, 1] = qx, qy
                    m += 1
                if m > 8:
                    break
            poly, n = out.copy(), m
            if n == 0:
                return 0.0
        area = 0.0
        for k in range(n):
            x1, y1 = poly[k, 0], poly[k, 1]
            x2, y2 = poly[(k + 1) % n, 0], poly[(k + 1) % n, 1]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    @numba.jit(nopython=True, cache=True)
    def _nms_iou_matrix(boxes):
        """boxes: (N,7) [x,y,z,dx,dy,dz,heading] -> (N,N) 上三角 IoU 矩阵"""
        N = boxes.shape[0]
        out = np.zeros((N, N), dtype=np.float32)
        corn = np.empty((N, 4, 2), dtype=np.float32)
        amin = np.empty((N, 2), dtype=np.float32)
        amax = np.empty((N, 2), dtype=np.float32)
        for i in range(N):
            _nms_corners(boxes[i], corn[i])
            amin[i, 0] = corn[i, :, 0].min()
            amax[i, 0] = corn[i, :, 0].max()
            amin[i, 1] = corn[i, :, 1].min()
            amax[i, 1] = corn[i, :, 1].max()
        for i in range(N):
            a1 = boxes[i, 3] * boxes[i, 4]
            for j in range(i + 1, N):
                if (amin[i, 0] > amax[j, 0] or amax[i, 0] < amin[j, 0]
                        or amin[i, 1] > amax[j, 1] or amax[i, 1] < amin[j, 1]):
                    continue
                a2 = boxes[j, 3] * boxes[j, 4]
                ai = _nms_inter(corn[i], corn[j])
                if ai <= 0:
                    continue
                out[i, j] = ai / (a1 + a2 - ai)
        return out

    @numba.jit(nopython=True, cache=True)
    def _nms_incremental(boxes, thresh):
        """增量贪心旋转 NMS：按输入序（须已按 score 降序），每框只与已保留框算 IoU。

        与 _nms_iou_matrix + 逐行抑制的贪心**严格等价**（被抑制的框不会成为 kept，
        每对 (i, kept) 的 IoU 用同一个 _nms_inter 数学计算 → bit 一致），
        复杂度 O(N * K)（K = 实际保留数，通常远小于 N）替代 O(N^2)。
        返回保留框在输入数组中的下标（升序）。
        """
        N = boxes.shape[0]
        kept = np.empty(N, dtype=np.int64)
        kcorn = np.empty((N, 4, 2), dtype=np.float32)
        kamin = np.empty((N, 2), dtype=np.float32)
        kamax = np.empty((N, 2), dtype=np.float32)
        karea = np.empty(N, dtype=np.float32)
        n_kept = 0
        corn = np.empty((4, 2), dtype=np.float32)
        for i in range(N):
            _nms_corners(boxes[i], corn)
            a1 = boxes[i, 3] * boxes[i, 4]
            keep = True
            for k in range(n_kept):
                if (corn[:, 0].min() > kamax[k, 0] or corn[:, 0].max() < kamin[k, 0]
                        or corn[:, 1].min() > kamax[k, 1] or corn[:, 1].max() < kamin[k, 1]):
                    continue
                ai = _nms_inter(corn, kcorn[k])
                if ai <= 0:
                    continue
                if ai / (a1 + karea[k] - ai) > thresh:
                    keep = False
                    break
            if keep:
                kcorn[n_kept] = corn
                kamin[n_kept, 0] = corn[:, 0].min()
                kamax[n_kept, 0] = corn[:, 0].max()
                kamin[n_kept, 1] = corn[:, 1].min()
                kamax[n_kept, 1] = corn[:, 1].max()
                karea[n_kept] = a1
                kept[n_kept] = i
                n_kept += 1
        return kept[:n_kept]


# ============================================================
# 辅助函数
# ============================================================

def _check_numpy_to_torch(x):
    """与 OpenPCDet common_utils.check_numpy_to_torch 等价"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x), True
    return x, False


def _cross_2d(a, b):
    """2D 叉积: a.x * b.y - a.y * b.x
    对应 CUDA: cross(const Point &a, const Point &b)
    Args: a, b: (..., 2)
    Returns: (...,)
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _cross_3pts(p1, p2, p0):
    """cross(p1-p0, p2-p0)
    对应 CUDA: cross(const Point &p1, const Point &p2, const Point &p0)
    Args: p0, p1, p2: (..., 2)
    Returns: (...,)
    """
    return (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1]) - \
           (p2[..., 0] - p0[..., 0]) * (p1[..., 1] - p0[..., 1])


def _get_box_corners(boxes):
    """计算 BEV 旋转框的 4 个角点
    对应 CUDA box_overlap 中构造 corners + rotate_around_center 的逻辑

    Args:
        boxes: (P, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        corners: (P, 4, 2) 顺序: [左下, 右下, 右上, 左上]（旋转后）
    """
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    dx = boxes[:, 3]
    dy = boxes[:, 4]
    heading = boxes[:, 6]

    dx_half = dx / 2
    dy_half = dy / 2

    # 旋转前的 4 个角点
    x1 = cx - dx_half;  x2 = cx + dx_half
    y1 = cy - dy_half;  y2 = cy + dy_half

    corners = torch.stack([
        torch.stack([x1, y1], dim=-1),  # 左下
        torch.stack([x2, y1], dim=-1),  # 右下
        torch.stack([x2, y2], dim=-1),  # 右上
        torch.stack([x1, y2], dim=-1),  # 左上
    ], dim=1)  # (P, 4, 2)

    # 绕中心旋转
    cos_a = torch.cos(heading)
    sin_a = torch.sin(heading)
    center = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # (P, 1, 2)
    rel = corners - center  # (P, 4, 2)

    rot_x = rel[..., 0] * cos_a.unsqueeze(1) - rel[..., 1] * sin_a.unsqueeze(1)
    rot_y = rel[..., 0] * sin_a.unsqueeze(1) + rel[..., 1] * cos_a.unsqueeze(1)
    rot = torch.stack([rot_x, rot_y], dim=-1)  # (P, 4, 2)

    return rot + center


def _check_in_box2d(box, pts):
    """判断点是否在 2D 旋转框内
    对应 CUDA: check_in_box2d(const float *box, const Point &p)

    Args:
        box: (P, 7) 单个框
        pts: (P, K, 2) 待检测点
    Returns:
        inside: (P, K) bool
    """
    MARGIN = 1e-2
    cx = box[:, 0:1]   # (P, 1)
    cy = box[:, 1:2]
    dx = box[:, 3:4]
    dy = box[:, 4:5]
    heading = box[:, 6:7]  # (P, 1)

    px = pts[..., 0]  # (P, K)
    py = pts[..., 1]

    # 将点旋转到框的局部坐标系（旋转 -heading）
    angle_cos = torch.cos(-heading)  # (P, 1)
    angle_sin = torch.sin(-heading)

    rot_x = (px - cx) * angle_cos + (py - cy) * (-angle_sin)  # (P, K)
    rot_y = (px - cx) * angle_sin + (py - cy) * angle_cos

    inside = (rot_x.abs() < dx / 2 + MARGIN) & (rot_y.abs() < dy / 2 + MARGIN)
    return inside


def _check_rect_cross(p1, p2, q1, q2):
    """快速排除：两条线段的 AABB 是否重叠
    对应 CUDA: check_rect_cross

    Args: p1, p2, q1, q2: (..., 2)
    Returns: (...,) bool
    """
    return (
        (torch.min(p1[..., 0], p2[..., 0]) <= torch.max(q1[..., 0], q2[..., 0])) &
        (torch.min(q1[..., 0], q2[..., 0]) <= torch.max(p1[..., 0], p2[..., 0])) &
        (torch.min(p1[..., 1], p2[..., 1]) <= torch.max(q1[..., 1], q2[..., 1])) &
        (torch.min(q1[..., 1], q2[..., 1]) <= torch.max(p1[..., 1], p2[..., 1]))
    )


def _edge_intersection(p0, p1, q0, q1):
    """计算两条线段的交点
    对应 CUDA: intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans)

    Args:
        p0, p1: (P, K, K, 2) 框 A 的边 (p0 -> p1)
        q0, q1: (P, K, K, 2) 框 B 的边 (q0 -> q1)
    Returns:
        inter: (P, K, K, 2) 交点坐标
        valid: (P, K, K) bool 是否存在交点
    """
    # 1) AABB 快速排除
    rect_cross = _check_rect_cross(p1, p0, q1, q0)

    # 2) 跨立实验（叉积同号判定）
    s1 = _cross_3pts(q0, p1, p0)  # cross(q0-p0, p1-p0)
    s2 = _cross_3pts(p1, q1, p0)  # cross(p1-p0, q1-p0)
    s3 = _cross_3pts(p0, q1, q0)  # cross(p0-q0, q1-q0)
    s4 = _cross_3pts(q1, p1, q0)  # cross(q1-q0, p1-q0)

    cross_valid = (s1 * s2 > 0) & (s3 * s4 > 0)
    valid = rect_cross & cross_valid

    # 3) 计算交点坐标
    s5 = _cross_3pts(q1, p1, p0)  # cross(q1-p0, p1-p0)
    diff_s = s5 - s1
    normal_case = diff_s.abs() > EPS

    # 情况 A: |s5 - s1| > EPS
    denom = torch.where(normal_case, diff_s, torch.ones_like(diff_s))
    inter_x = (s5 * q0[..., 0] - s1 * q1[..., 0]) / denom
    inter_y = (s5 * q0[..., 1] - s1 * q1[..., 1]) / denom

    # 情况 B: 退化情况（共线），用直线方程
    a0 = p0[..., 1] - p1[..., 1]
    b0 = p1[..., 0] - p0[..., 0]
    c0 = p0[..., 0] * p1[..., 1] - p1[..., 0] * p0[..., 1]
    a1 = q0[..., 1] - q1[..., 1]
    b1 = q1[..., 0] - q0[..., 0]
    c1 = q0[..., 0] * q1[..., 1] - q1[..., 0] * q0[..., 1]
    D = a0 * b1 - a1 * b0
    D_safe = torch.where(D.abs() > EPS, D, torch.ones_like(D))
    degen_x = (b0 * c1 - b1 * c0) / D_safe
    degen_y = (a1 * c0 - a0 * c1) / D_safe

    inter_x = torch.where(normal_case, inter_x, degen_x)
    inter_y = torch.where(normal_case, inter_y, degen_y)

    inter = torch.stack([inter_x, inter_y], dim=-1)
    return inter, valid


# ============================================================
# 核心：BEV 旋转框重叠面积（Sutherland-Hodgman 多边形相交）
# ============================================================

def _box_overlap_bev_pairs(boxes_a, boxes_b):
    """计算 P 对框的 BEV 重叠面积
    对应 CUDA: box_overlap(const float *box_a, const float *box_b) — 向量化版本

    算法:
      1. 计算 4 个角点（旋转后）
      2. 求 4×4=16 条边的交点
      3. 检查角点包含关系（A 的角在 B 内，B 的角在 A 内）
      4. 将所有交点按角度排序（围绕质心）
      5. 用鞋带公式计算多边形面积

    Args:
        boxes_a: (P, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (P, 7)
    Returns:
        overlap: (P,) BEV 重叠面积
    """
    P = boxes_a.shape[0]
    device = boxes_a.device

    # 1) 角点: (P, 4, 2)
    corners_a = _get_box_corners(boxes_a)
    corners_b = _get_box_corners(boxes_b)

    # 边: edge_i = corners[i] -> corners[(i+1)%4]
    corners_a_next = torch.roll(corners_a, -1, dims=1)  # (P, 4, 2)
    corners_b_next = torch.roll(corners_b, -1, dims=1)

    # 2) 4×4 边对求交: (P, 4, 4, 2)
    p0 = corners_a.unsqueeze(2).expand(P, 4, 4, 2)       # A 的边起点
    p1 = corners_a_next.unsqueeze(2).expand(P, 4, 4, 2)   # A 的边终点
    q0 = corners_b.unsqueeze(1).expand(P, 4, 4, 2)       # B 的边起点
    q1 = corners_b_next.unsqueeze(1).expand(P, 4, 4, 2)   # B 的边终点

    inter_pts, inter_valid = _edge_intersection(p0, p1, q0, q1)
    inter_pts = inter_pts.reshape(P, 16, 2)
    inter_valid = inter_valid.reshape(P, 16)

    # 3) 角点包含检测: (P, 4)
    a_in_b = _check_in_box2d(boxes_b, corners_a)  # A 的角在 B 内
    b_in_a = _check_in_box2d(boxes_a, corners_b)  # B 的角在 A 内

    # 合并所有交点: (P, 24, 2)
    all_pts = torch.cat([inter_pts, corners_a, corners_b], dim=1)
    all_valid = torch.cat([inter_valid, a_in_b, b_in_a], dim=1)

    # 4) 计算质心，按角度排序
    valid_pts_sum = (all_pts * all_valid.unsqueeze(-1).float()).sum(dim=1)  # (P, 2)
    cnt = all_valid.sum(dim=1, keepdim=True).clamp(min=1)  # (P, 1)
    centroid = valid_pts_sum / cnt  # (P, 2)

    diff = all_pts - centroid.unsqueeze(1)  # (P, 24, 2)
    angles = torch.atan2(diff[..., 1], diff[..., 0])  # (P, 24)
    angles = torch.where(all_valid, angles,
                         torch.full_like(angles, float('inf')))

    sorted_idx = angles.argsort(dim=1)  # (P, 24)
    sorted_pts = torch.gather(all_pts, 1,
                              sorted_idx.unsqueeze(-1).expand(P, 24, 2))
    sorted_valid = torch.gather(all_valid, 1, sorted_idx)

    # 5) 鞋带公式（三角形扇分解）
    #   对应 CUDA: area += cross(p[k]-p[0], p[k+1]-p[0]); area = |area|/2
    p0_ref = sorted_pts[:, 0:1, :]  # (P, 1, 2)
    vec_k = sorted_pts[:, :-1, :] - p0_ref   # (P, 23, 2)
    vec_k1 = sorted_pts[:, 1:, :] - p0_ref    # (P, 23, 2)
    cross_vals = _cross_2d(vec_k, vec_k1)     # (P, 23)

    valid_pairs = sorted_valid[:, :-1] & sorted_valid[:, 1:]  # (P, 23)
    cross_vals = cross_vals * valid_pairs.to(cross_vals.dtype)

    area = cross_vals.sum(dim=1).abs() / 2.0  # (P,)
    area = torch.where(cnt.squeeze(1) > 0, area, torch.zeros_like(area))
    return area


def box_overlap_bev(boxes_a, boxes_b, max_pairs=200000):
    """N×M BEV 重叠面积
    对应 CUDA: boxes_overlap_kernel

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7)
        max_pairs: 分块大小，控制内存
    Returns:
        overlap: (N, M)
    """
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    device = boxes_a.device
    dtype = boxes_a.dtype

    if N == 0 or M == 0:
        return torch.zeros(N, M, device=device, dtype=dtype)

    total = N * M
    if total <= max_pairs:
        a = boxes_a.unsqueeze(1).expand(N, M, 7).reshape(-1, 7)
        b = boxes_b.unsqueeze(0).expand(N, M, 7).reshape(-1, 7)
        return _box_overlap_bev_pairs(a, b).reshape(N, M)

    # 分块处理，防止内存溢出
    result = torch.zeros(N, M, device=device, dtype=dtype)
    chunk_n = max(1, int((max_pairs ** 0.5)))
    chunk_m = max(1, max_pairs // chunk_n)

    for i in range(0, N, chunk_n):
        for j in range(0, M, chunk_m):
            ni = min(i + chunk_n, N)
            nj = min(j + chunk_m, M)
            ci, cj = ni - i, nj - j
            a = boxes_a[i:ni].unsqueeze(1).expand(ci, cj, 7).reshape(-1, 7)
            b = boxes_b[j:nj].unsqueeze(0).expand(ci, cj, 7).reshape(-1, 7)
            result[i:ni, j:nj] = _box_overlap_bev_pairs(a, b).reshape(ci, cj)
    return result


def _box_iou_normal_pairs(boxes_a, boxes_b):
    """轴对齐 BEV IoU（忽略 heading）
    对应 CUDA: iou_normal()

    Args:
        boxes_a: (P, 7)
        boxes_b: (P, 7)
    Returns:
        iou: (P,)
    """
    a = boxes_a
    b = boxes_b

    left = torch.max(a[:, 0] - a[:, 3] / 2, b[:, 0] - b[:, 3] / 2)
    right = torch.min(a[:, 0] + a[:, 3] / 2, b[:, 0] + b[:, 3] / 2)
    top = torch.max(a[:, 1] - a[:, 4] / 2, b[:, 1] - b[:, 4] / 2)
    bottom = torch.min(a[:, 1] + a[:, 4] / 2, b[:, 1] + b[:, 4] / 2)

    width = torch.clamp(right - left, min=0.0)
    height = torch.clamp(bottom - top, min=0.0)
    interS = width * height

    Sa = a[:, 3] * a[:, 4]
    Sb = b[:, 3] * b[:, 4]
    return interS / torch.clamp(Sa + Sb - interS, min=EPS)


def _boxes_iou_normal(boxes_a, boxes_b, max_pairs=200000):
    """N×M 轴对齐 BEV IoU"""
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    device = boxes_a.device
    dtype = boxes_a.dtype

    if N == 0 or M == 0:
        return torch.zeros(N, M, device=device, dtype=dtype)

    total = N * M
    if total <= max_pairs:
        a = boxes_a.unsqueeze(1).expand(N, M, 7).reshape(-1, 7)
        b = boxes_b.unsqueeze(0).expand(N, M, 7).reshape(-1, 7)
        return _box_iou_normal_pairs(a, b).reshape(N, M)

    result = torch.zeros(N, M, device=device, dtype=dtype)
    chunk_n = max(1, int((max_pairs ** 0.5)))
    chunk_m = max(1, max_pairs // chunk_n)
    for i in range(0, N, chunk_n):
        for j in range(0, M, chunk_m):
            ni = min(i + chunk_n, N)
            nj = min(j + chunk_m, M)
            ci, cj = ni - i, nj - j
            a = boxes_a[i:ni].unsqueeze(1).expand(ci, cj, 7).reshape(-1, 7)
            b = boxes_b[j:nj].unsqueeze(0).expand(ci, cj, 7).reshape(-1, 7)
            result[i:ni, j:nj] = _box_iou_normal_pairs(a, b).reshape(ci, cj)
    return result


# ============================================================
# 公开 API（与 OpenPCDet iou3d_nms_utils.py 接口完全一致）
# ============================================================

def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """BEV IoU（CPU）
    对应 iou3d_nms_utils.boxes_bev_iou_cpu

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7)
    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = _check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = _check_numpy_to_torch(boxes_b)
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7

    overlap = box_overlap_bev(boxes_a.float(), boxes_b.float())
    sa = (boxes_a[:, 3] * boxes_a[:, 4]).unsqueeze(1)  # (N, 1)
    sb = (boxes_b[:, 3] * boxes_b[:, 4]).unsqueeze(0)  # (1, M)
    ans_iou = overlap / torch.clamp(sa + sb - overlap, min=EPS)
    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """BEV IoU
    对应 iou3d_nms_utils.boxes_iou_bev

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7)
    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    overlap = box_overlap_bev(boxes_a, boxes_b)
    sa = (boxes_a[:, 3] * boxes_a[:, 4]).unsqueeze(1)
    sb = (boxes_b[:, 3] * boxes_b[:, 4]).unsqueeze(0)
    return overlap / torch.clamp(sa + sb - overlap, min=EPS)


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """3D IoU
    对应 iou3d_nms_utils.boxes_iou3d_gpu

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7)
    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7

    # 高度重叠
    a_hmax = (boxes_a[:, 2] + boxes_a[:, 5] / 2).unsqueeze(1)  # (N, 1)
    a_hmin = (boxes_a[:, 2] - boxes_a[:, 5] / 2).unsqueeze(1)
    b_hmax = (boxes_b[:, 2] + boxes_b[:, 5] / 2).unsqueeze(0)  # (1, M)
    b_hmin = (boxes_b[:, 2] - boxes_b[:, 5] / 2).unsqueeze(0)

    max_of_min = torch.max(a_hmin, b_hmin)
    min_of_max = torch.min(a_hmax, b_hmax)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # BEV 重叠
    overlaps_bev = box_overlap_bev(boxes_a, boxes_b)

    # 3D 重叠 = BEV 面积 × 高度重叠
    overlaps_3d = overlaps_bev * overlaps_h

    # 体积
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).unsqueeze(1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).unsqueeze(0)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    return iou3d


def boxes_aligned_iou3d_gpu(boxes_a, boxes_b):
    """对齐 3D IoU（1-to-1）
    对应 iou3d_nms_utils.boxes_aligned_iou3d_gpu

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7)
    Returns:
        ans_iou: (N,)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7

    # 高度重叠
    a_hmax = (boxes_a[:, 2] + boxes_a[:, 5] / 2).unsqueeze(1)
    a_hmin = (boxes_a[:, 2] - boxes_a[:, 5] / 2).unsqueeze(1)
    b_hmax = (boxes_b[:, 2] + boxes_b[:, 5] / 2).unsqueeze(1)
    b_hmin = (boxes_b[:, 2] - boxes_b[:, 5] / 2).unsqueeze(1)

    max_of_min = torch.max(a_hmin, b_hmin)
    min_of_max = torch.min(a_hmax, b_hmax)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # BEV 重叠（1-to-1）
    overlaps_bev = _box_overlap_bev_pairs(boxes_a, boxes_b)  # (N,)

    # 3D 重叠
    overlaps_3d = overlaps_bev * overlaps_h.squeeze(1)

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5])
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5])

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    return iou3d


def paired_boxes_iou3d_gpu(boxes_a, boxes_b):
    """配对 3D IoU（1-to-1）
    对应 iou3d_nms_utils.paired_boxes_iou3d_gpu

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7)
    Returns:
        ans_iou: (N,)
    """
    return boxes_aligned_iou3d_gpu(boxes_a, boxes_b)


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """旋转 NMS（BEV IoU）
    对应 iou3d_nms_utils.nms_gpu

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        scores: (N,)
        thresh: NMS 阈值
        pre_maxsize: 预筛选最大数量
    Returns:
        keep: (K,) 保留的索引（对应原始输入顺序）
        None
    """
    assert boxes.shape[1] == 7
    device = boxes.device

    # NPU 310P 的 aicpu GatherElements 对 `tensor[index_tensor]`、布尔掩码索引、
    # 元组索引等不稳定（errorCode 0x2a），因此整个旋转 NMS 放到 CPU 完成，
    # 只把结果索引搬回设备。张量很小（N <= pre_maxsize），CPU 开销可忽略。
    boxes = boxes.detach().cpu()
    scores = scores.detach().cpu()

    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    N = boxes.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device), None

    # 计算保留框（增量贪心，等价于全矩阵 + 逐行抑制，但 O(N*K)）
    if _HAS_NUMBA:
        # numba 快速路径：增量版只对已保留框算 IoU，省去 O(N^2) 全矩阵，逐对数学不变 → bit 一致
        keep_np = _nms_incremental(
            np.ascontiguousarray(boxes.detach().cpu().numpy(), dtype=np.float32),
            float(thresh),
        )
        keep = torch.from_numpy(keep_np).long()
    else:
        iou_np = boxes_iou_bev(boxes, boxes).detach().cpu().numpy()  # (N, N)
        keep = []
        suppressed = np.zeros(N, dtype=bool)
        for i in range(N):
            if suppressed[i]:
                continue
            keep.append(i)
            suppressed[i + 1:] |= iou_np[i, i + 1:] > thresh
        keep = torch.tensor(keep, dtype=torch.long)

    return order[keep].contiguous().to(device), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """轴对齐 NMS（忽略 heading）
    对应 iou3d_nms_utils.nms_normal_gpu

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        scores: (N,)
        thresh: NMS 阈值
    Returns:
        keep: (K,) 保留的索引
        None
    """
    assert boxes.shape[1] == 7
    device = boxes.device

    # 同 nms_gpu：整个轴对齐 NMS 放到 CPU 完成，规避 NPU aicpu GatherElements 崩溃
    boxes = boxes.detach().cpu()
    scores = scores.detach().cpu()

    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()

    N = boxes.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device), None

    # 轴对齐 IoU 矩阵
    iou_matrix = _boxes_iou_normal(boxes, boxes)  # (N, N)
    iou_np = iou_matrix.detach().cpu().numpy()

    keep = []
    suppressed = np.zeros(N, dtype=bool)

    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed[i + 1:] |= iou_np[i, i + 1:] > thresh

    keep = torch.tensor(keep, dtype=torch.long)
    return order[keep].contiguous().to(device), None


# ============================================================
# 验证测试
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # --- 测试 1: BEV IoU ---
    print("=" * 60)
    print("测试 1: boxes_iou_bev")
    boxes_a = torch.tensor([
        [0, 0, 0, 4, 4, 4, 0],       # 中心在原点，无旋转
        [5, 5, 0, 4, 4, 4, 0.5],     # 旋转 0.5 rad
    ]).float()
    boxes_b = torch.tensor([
        [1, 1, 0, 4, 4, 4, 0],       # 与 box_a[0] 部分重叠
        [0, 0, 0, 4, 4, 4, 0.3],     # 与 box_a[0] 旋转重叠
    ]).float()
    iou_bev = boxes_iou_bev(boxes_a, boxes_b)
    print(f"BEV IoU:\n{iou_bev}")
    # box_a[0] 与 box_b[0]: 两个轴对齐框，偏移 (1,1)，重叠 3×3=9，面积各 16
    # IoU = 9 / (16+16-9) = 9/23 ≈ 0.391

    # --- 测试 2: 3D IoU ---
    print("\n测试 2: boxes_iou3d_gpu")
    iou3d = boxes_iou3d_gpu(boxes_a, boxes_b)
    print(f"3D IoU:\n{iou3d}")

    # --- 测试 3: 对齐 3D IoU ---
    print("\n测试 3: boxes_aligned_iou3d_gpu")
    iou_aligned = boxes_aligned_iou3d_gpu(boxes_a, boxes_a)
    print(f"Self IoU (应全为 1.0): {iou_aligned}")  # 应为 [1.0, 1.0]

    # --- 测试 4: NMS ---
    print("\n测试 4: nms_gpu")
    boxes = torch.tensor([
        [0, 0, 0, 2, 2, 2, 0],
        [0.5, 0.5, 0, 2, 2, 2, 0],   # 与 box[0] 重叠度高
        [10, 10, 0, 2, 2, 2, 0],      # 远离，不重叠
        [0.3, 0.3, 0, 2, 2, 2, 0],   # 与 box[0] 重叠度高
    ]).float()
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    keep, _ = nms_gpu(boxes, scores, thresh=0.3)
    print(f"NMS keep indices: {keep}")  # 应保留 0 (最高分) 和 2 (不重叠)

    # --- 测试 5: NMS normal ---
    print("\n测试 5: nms_normal_gpu")
    keep_normal, _ = nms_normal_gpu(boxes, scores, thresh=0.3)
    print(f"NMS normal keep: {keep_normal}")

    # --- 测试 6: 完全重叠 ---
    print("\n测试 6: 完全重叠框")
    box1 = torch.tensor([[0, 0, 0, 4, 4, 4, 0.0]]).float()
    box2 = torch.tensor([[0, 0, 0, 4, 4, 4, 0.0]]).float()
    print(f"完全重叠 IoU: {boxes_iou_bev(box1, box2)}")  # 应为 1.0

    # --- 测试 7: 旋转框 ---
    print("\n测试 7: 旋转 45 度的正方形")
    # 两个相同大小的正方形，一个旋转 45 度
    box_axis = torch.tensor([[0, 0, 0, 2, 2, 1, 0.0]]).float()
    box_rot45 = torch.tensor([[0, 0, 0, 2, 2, 1, 0.7853981633974483]]).float()  # π/4
    overlap = box_overlap_bev(box_axis, box_rot45)
    print(f"轴对齐面积: {2*2}, 旋转45重叠: {overlap[0,0]:.4f}")
    # 旋转 45 度后，重叠面积 = 2√2 × 2√2 - 2×(2√2-2)²/2 ≈ 2.343
    iou = boxes_iou_bev(box_axis, box_rot45)
    print(f"旋转45 IoU: {iou[0,0]:.4f}")