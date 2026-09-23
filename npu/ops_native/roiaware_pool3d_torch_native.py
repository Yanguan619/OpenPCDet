"""
RoIAwarePool3d: Torch-native (纯 PyTorch) 实现
- 完全对应 OpenPCDet pcdet/ops/roiaware_pool3d/src/ 的 CUDA kernel 逻辑
- 支持 CPU / GPU，支持 autograd（max_pool + avg_pool 前后向）
"""

import torch
import torch.nn as nn
from torch.autograd import Function

# ============================================================
# 基础工具：点在 3D 框内判定（向量化版）
# ============================================================

def lidar_to_local_coords(shift_x, shift_y, rot_angle):
    """
    将世界系偏移 (shift_x, shift_y) 按 -rot_angle 旋转到 box 局部系
    对应 CUDA kernel 中的 lidar_to_local_coords()
    Args:
        shift_x, shift_y: (..., )  任意形状
        rot_angle: (..., )         广播一致
    Returns:
        local_x, local_y: 与输入同形状
    """
    cosa = torch.cos(-rot_angle)
    sina = torch.sin(-rot_angle)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    return local_x, local_y


def check_pt_in_box3d(pts, box3d):
    """
    批量判定：每个点是否在每个 3D box 内，并返回局部坐标
    Args:
        pts:   (N, 3)   [x, y, z]
        box3d: (M, 7)   [x, y, z, dx, dy, dz, heading]
    Returns:
        in_flag:   (M, N)  bool
        local_x:   (M, N)  局部 x（未在框内值无意义）
        local_y:   (M, N)
        local_z:   (M, N)
    """
    N = pts.shape[0]
    M = box3d.shape[0]

    px = pts[:, 0].view(1, N)          # (1, N)
    py = pts[:, 1].view(1, N)
    pz = pts[:, 2].view(1, N)

    cx = box3d[:, 0].view(M, 1)        # (M, 1)
    cy = box3d[:, 1].view(M, 1)
    cz = box3d[:, 2].view(M, 1)
    dx = box3d[:, 3].view(M, 1)
    dy = box3d[:, 4].view(M, 1)
    dz = box3d[:, 5].view(M, 1)
    rz = box3d[:, 6].view(M, 1)

    # 1) Z 方向快速过滤
    z_ok = (pz - cz).abs() <= dz / 2.0  # (M, N)

    # 2) XY 旋转到局部系
    local_x, local_y = lidar_to_local_coords(px - cx, py - cy, rz)  # (M, N)
    MARGIN = 1e-5
    x_ok = local_x.abs() < dx / 2.0 + MARGIN
    y_ok = local_y.abs() < dy / 2.0 + MARGIN

    in_flag = z_ok & x_ok & y_ok
    local_z = pz - cz  # (M, N)  局部 z = 点 z - 框中心 z

    return in_flag, local_x, local_y, local_z


def _clip(val, lo, hi):
    return val.clamp(min=lo, max=hi)


# ============================================================
# RoIAwarePool3dFunction: 自定义 autograd Function
# ============================================================

class RoIAwarePool3dFunction(Function):
    """
    纯 PyTorch 版，完全等价于 CUDA kernel:
      generate_pts_mask_for_box3d
      collect_inside_pts_for_box3d
      roiaware_maxpool3d / roiaware_avgpool3d
      + 对应 backward kernel
    """

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        Args:
            rois:        (M, 7)    [x, y, z, dx, dy, dz, heading]
            pts:         (N, 3)    [x, y, z]
            pts_feature: (N, C)
            out_size:    int or (3,)  e.g. 7  or (7, 7, 7)
            max_pts_each_voxel: int
            pool_method: 'max' | 'avg'
        Returns:
            pooled_features: (M, ox, oy, oz, C)
        """
        if isinstance(out_size, int):
            ox = oy = oz = out_size
        else:
            ox, oy, oz = out_size

        M = rois.shape[0]
        N = pts.shape[0]
        C = pts_feature.shape[-1]

        device = pts_feature.device
        dtype = pts_feature.dtype

        # ------------------------------------------------------------
        # Step 1: 判定每个 (box, pt) 是否在框内，计算体素索引
        #   对应 generate_pts_mask_for_box3d
        # ------------------------------------------------------------
        in_flag, lx, ly, lz = check_pt_in_box3d(pts, rois)  # (M,N), (M,N)*3

        dx = rois[:, 3].view(M, 1)
        dy = rois[:, 4].view(M, 1)
        dz = rois[:, 5].view(M, 1)

        x_res = dx / ox
        y_res = dy / oy
        z_res = dz / oz

        x_idx = ((lx + dx / 2.0) / x_res).long()          # (M, N)
        y_idx = ((ly + dy / 2.0) / y_res).long()
        z_idx = ((lz + dz / 2.0) / z_res).long()

        x_idx = _clip(x_idx, 0, ox - 1)
        y_idx = _clip(y_idx, 0, oy - 1)
        z_idx = _clip(z_idx, 0, oz - 1)

        # 未在框内的置为特殊标记 (-1)
        x_idx = torch.where(in_flag, x_idx, torch.tensor(-1, device=device, dtype=torch.long))
        y_idx = torch.where(in_flag, y_idx, torch.tensor(-1, device=device, dtype=torch.long))
        z_idx = torch.where(in_flag, z_idx, torch.tensor(-1, device=device, dtype=torch.long))

        # ------------------------------------------------------------
        # Step 2: 收集每个体素中的点索引
        #   对应 collect_inside_pts_for_box3d
        #   pts_idx_of_voxels: (M, ox, oy, oz, max_pts)  [0]=计数, [1..cnt]=点索引
        # ------------------------------------------------------------
        pts_idx_of_voxels = torch.full(
            (M, ox, oy, oz, max_pts_each_voxel), -1,
            device=device, dtype=torch.long
        )

        # 计算每个 (box, pt) 对应的 voxel 位置
        voxel_idx_flat = x_idx * (oy * oz) + y_idx * oz + z_idx  # (M,N)
        valid = in_flag  # (M,N)

        # 遍历每个 box 执行收集（M 较小，几十~几百）
        for bi in range(M):
            v_flat_b = voxel_idx_flat[bi]  # (N,)
            valid_b = valid[bi]            # (N,)
            pts_b_idx = torch.arange(N, device=device)

            # 对每个体素取最多 max_pts 个点（上限截断）
            for v in range(ox * oy * oz):
                # 找出属于该体素且有效的点索引
                mask = valid_b & (v_flat_b == v)
                if not mask.any():
                    continue
                candidate_idx = pts_b_idx[mask]
                take_cnt = min(candidate_idx.numel(), max_pts_each_voxel - 1)
                take_idx = candidate_idx[:take_cnt]
                xi = v // (oy * oz)
                yi = (v - xi * oy * oz) // oz
                zi = v % oz
                pts_idx_of_voxels[bi, xi, yi, zi, 0] = take_cnt
                pts_idx_of_voxels[bi, xi, yi, zi, 1:1 + take_cnt] = take_idx

        # ------------------------------------------------------------
        # Step 3: 池化 (max / avg)
        #   对应 roiaware_maxpool3d / roiaware_avgpool3d
        # ------------------------------------------------------------
        pooled_features = torch.zeros(M, ox, oy, oz, C, device=device, dtype=dtype)
        argmax = torch.full(
            (M, ox, oy, oz, C), -1,
            device=device, dtype=torch.long
        )

        # 展开便于索引
        # pts_idx_of_voxels_v: (M, ox*oy*oz, max_pts)
        pfl = pts_idx_of_voxels.view(M, ox * oy * oz, max_pts_each_voxel)
        pool_flat = pooled_features.view(M, ox * oy * oz, C)     # (M, V, C)
        argmax_flat = argmax.view(M, ox * oy * oz, C)

        # 展开体素计数与点索引
        cnt = pfl[:, :, 0].long()                        # (M, V)  每个体素点数
        # idx_of_pts: (M, V, max_pts-1)  点索引
        idx_of_pts = pfl[:, :, 1:]                      # (M, V, K)

        # 构造取特征用的索引：(M, V, K, C)
        max_cnt = cnt.max().item() if cnt.numel() > 0 else 0
        if max_cnt > 0:
            # 裁剪 K 到实际最大，减少计算
            K = idx_of_pts.shape[2]
            # 掩码：只有前 cnt 个点才是有效的
            # range_of_k: (1, 1, K)
            range_k = torch.arange(K, device=device).view(1, 1, K)
            valid_mask = range_k < cnt.unsqueeze(-1)      # (M, V, K)

            # gather 点特征: (N, C) -> (M, V, K, C)
            safe_idx = idx_of_pts.clone()
            safe_idx[~valid_mask] = 0  # 非法位置置 0 避免 index error
            # (M, V, K, C)
            gathered = pts_feature[safe_idx]
            # 非法位置用极小值（max）或 0（avg）填充
            valid_mask_c = valid_mask.unsqueeze(-1)       # (M, V, K, C)

            if pool_method == 'max':
                neg_inf = torch.full_like(gathered, -1e50)
                gathered = torch.where(valid_mask_c, gathered, neg_inf)
                # pool over K
                vals, k_idx = gathered.max(dim=2)         # (M, V, C)
                pool_flat.copy_(vals)
                # argmax_flat: 将 k_idx 映射回原始点索引
                # k_idx: (M, V, C), idx_of_pts: (M, V, K)
                # 对每个 C 取 idx_of_pts[m, v, k_idx[m,v,c]]
                m_idx = torch.arange(M, device=device).view(M, 1, 1)
                v_idx = torch.arange(ox * oy * oz, device=device).view(1, ox * oy * oz, 1)
                c_idx = torch.arange(C, device=device).view(1, 1, C)
                original_idx = idx_of_pts[m_idx, v_idx, k_idx]
                argmax_flat.copy_(original_idx)

                # 空体素（cnt==0）全部是 -1e50，需要手动置 0 & argmax=-1
                empty_mask = (cnt == 0).unsqueeze(-1)     # (M, V, 1)
                pool_flat.masked_fill_(empty_mask, 0.0)
                argmax_flat.masked_fill_(empty_mask, -1)

            else:  # avg
                gathered = torch.where(valid_mask_c, gathered, torch.zeros_like(gathered))
                sum_val = gathered.sum(dim=2)             # (M, V, C)
                divisor = cnt.unsqueeze(-1).clamp(min=1)  # (M, V, 1)  避免除 0
                avg_val = sum_val / divisor
                pool_flat.copy_(avg_val)
                # 空体素置 0（本身就是 0，可不处理）

        # 保存反向用
        ctx.save_for_backward = (pts_idx_of_voxels, argmax, pool_method, N, C, (ox, oy, oz))

        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: (M, ox, oy, oz, C)
        Returns:
            grad_in: (N, C)  对应 pts_feature 的梯度
        """
        pts_idx_of_voxels, argmax, pool_method, N, C, (ox, oy, oz) = ctx.save_for_backward
        device = grad_out.device
        dtype = grad_out.dtype

        grad_in = torch.zeros(N, C, device=device, dtype=dtype)

        M = grad_out.shape[0]
        go_flat = grad_out.view(M, ox * oy * oz, C)

        if pool_method == 'max':
            # max 反向：argmax 指示梯度流向
            am_flat = argmax.view(M, ox * oy * oz, C)      # (M, V, C)
            valid_am = am_flat >= 0
            if valid_am.any():
                valid_idx = am_flat[valid_am]               # (K_total,)
                valid_grad = go_flat[valid_am]              # (K_total,) 的对应 C 值？
                # 展开对应：对每 (m,v,c) 是独立条目
                # 用 scatter_add_ 更高效
                # am_flat 合法索引是 (M*V*C,) 中的 C 维度展开？逐个 C 更安全
                for c in range(C):
                    src = go_flat[:, :, c].reshape(-1)     # (M*V,)
                    idx = am_flat[:, :, c].reshape(-1)     # (M*V,)
                    mask = idx >= 0
                    src_valid = src[mask]
                    idx_valid = idx[mask]
                    if idx_valid.numel() > 0:
                        grad_in[:, c].scatter_add_(0, idx_valid, src_valid)

        else:  # avg
            # avg 反向：每个体素内所有点均分梯度
            pfl = pts_idx_of_voxels.view(M, ox * oy * oz, -1)  # (M, V, max_pts)
            cnt = pfl[:, :, 0].long()                           # (M, V)
            idx_of_pts = pfl[:, :, 1:]                          # (M, V, K)
            K = idx_of_pts.shape[2]
            range_k = torch.arange(K, device=device).view(1, 1, K)
            valid_mask = range_k < cnt.unsqueeze(-1)            # (M, V, K)
            safe_idx = idx_of_pts.clone()
            safe_idx[~valid_mask] = 0
            divisor = cnt.unsqueeze(-1).clamp(min=1).float()    # (M, V, 1)

            for c in range(C):
                # go_flat[...,c]: (M, V)
                g = go_flat[:, :, c]                            # (M, V)
                per_pt_grad = g.unsqueeze(-1) / divisor         # (M, V, 1) / (M,V,1) = (M,V,1)
                # (M, V, K)：每个合法位置放 per_pt_grad
                per_pt_grad_bc = per_pt_grad.expand(-1, -1, K)   # (M, V, K)
                # 合法位置
                flat_idx = safe_idx.reshape(-1)                  # (M*V*K,)
                flat_val = per_pt_grad_bc.reshape(-1)            # (M*V*K,)
                flat_mask = valid_mask.reshape(-1)               # (M*V*K,)
                if flat_mask.any():
                    grad_in[:, c].scatter_add_(
                        0, flat_idx[flat_mask], flat_val[flat_mask]
                    )

        # rois, pts, out_size, max_pts, pool_method 无梯度
        return None, None, grad_in, None, None, None


# ============================================================
# nn.Module 封装（与 OpenPCDet RoIAwarePool3d 接口一致）
# ============================================================

class RoIAwarePool3d(nn.Module):
    """
    与 OpenPCDet RoIAwarePool3d 接口完全一致。
    Args:
        out_size:           int or (3,)  e.g. 7 -> (7,7,7)
        max_pts_each_voxel: 每个体素最多取多少个点（默认 128 与 OpenPCDet 对齐）
    """
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        """
        Args:
            rois:        (M, 7)  [x, y, z, dx, dy, dz, heading]
            pts:         (N, 3)  [x, y, z]
            pts_feature: (N, C)
            pool_method: 'max' | 'avg'
        Returns:
            pooled_features: (M, ox, oy, oz, C)
        """
        assert pool_method in ('max', 'avg'), f"pool_method={pool_method} not in (max,avg)"
        return RoIAwarePool3dFunction.apply(
            rois, pts, pts_feature, self.out_size,
            self.max_pts_each_voxel, pool_method
        )


# ============================================================
# points_in_boxes_gpu / points_in_boxes_cpu（OpenPCDet 工具等价版）
# ============================================================

def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (N, 3)
        boxes:  (M, 7)  各 box 不重叠
    Returns:
        point_indices: (M, N) int64  [1=in, 0=out]
    """
    in_flag, _, _, _ = check_pt_in_box3d(points, boxes)  # (M, N)
    return in_flag.long()


def points_in_boxes_gpu(points, boxes):
    """
    Args:
        points: (B, N, 3)
        boxes:  (B, T, 7)
    Returns:
        box_idxs_of_pts: (B, N) int64  每个点属于的 box 索引（不重叠时 first match），否则 -1
    """
    B, N, _ = points.shape
    _, T, _ = boxes.shape
    device = points.device
    out = torch.full((B, N), -1, device=device, dtype=torch.long)

    for b in range(B):
        pts_b = points[b]            # (N,3)
        boxes_b = boxes[b]           # (T,7)
        in_flag, _, _, _ = check_pt_in_box3d(pts_b, boxes_b)  # (T,N)
        # 取第一个命中的 box（与 CUDA kernel break 语义一致）
        hit = in_flag.int().argmax(dim=0)                 # (N,)  全 0 时默认 0
        any_hit = in_flag.any(dim=0)                      # (N,)
        out[b] = torch.where(any_hit, hit, torch.tensor(-1, device=device, dtype=torch.long))
    return out


# ============================================================
# 用法示例 / 基本验证
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) 构造数据：1 个 box，若干点（box 中心在原点，2x3x4，heading=0）
    rois = torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0]])  # (1,7)
    N = 1000
    pts = torch.randn(N, 3) * 3
    feat = torch.randn(N, 16).requires_grad_(True)

    pool = RoIAwarePool3d(out_size=7, max_pts_each_voxel=128)

    # --- max pool ---
    out_max = pool(rois, pts, feat, 'max')
    print("[max] out shape:", out_max.shape)            # (1,7,7,7,16)
    (out_max.sum()).backward()
    g1 = feat.grad.clone()
    feat.grad = None

    # --- avg pool ---
    out_avg = pool(rois, pts, feat, 'avg')
    print("[avg] out shape:", out_avg.shape)            # (1,7,7,7,16)
    (out_avg.sum()).backward()
    g2 = feat.grad.clone()
    feat.grad = None

    print("grad max nonzero:", (g1 != 0).sum().item())
    print("grad avg nonzero:", (g2 != 0).sum().item())

    # 2) points_in_boxes_gpu / cpu 测试
    boxes = torch.tensor([
        [0, 0, 0, 2, 2, 2, 0],
        [5, 5, 5, 2, 2, 2, 0],
    ]).unsqueeze(0)                                   # (1, 2, 7)
    p = torch.tensor([[0, 0, 0], [5, 5, 5], [10, 10, 10]]).unsqueeze(0)  # (1,3,3)
    print("points_in_boxes_gpu:", points_in_boxes_gpu(p, boxes))   # [[0,1,-1]]