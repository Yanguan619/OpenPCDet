import torch
import torch.nn as nn

"""
Torch-native implementation of the pointnet2 stack operators
(see ``pointnet2_utils.py`` for the CUDA-compiled counterparts).

Every function exposes the same call signature as the CUDA version, so the
models can run on CPU / non-CUDA accelerators (e.g. NPU) without compiling
the ``pointnet2_stack_cuda`` extension. Gradients are provided through native
PyTorch autograd ops wherever the CUDA kernel supports them.
"""

_CHUNK = 2048


def _ball_query_slots(valid_mask, nsample):
    """
    Given a per-row boolean mask (M, N) of valid points (in scan order),
    produce idx (M, nsample) with exactly the fill semantics of the CUDA
    ball-query kernels:

      * slot ``l`` holds the ``(l+1)``-th valid point in scan order while it
        exists, otherwise it repeats the first valid point of the row;
      * rows with no valid point get ``-1`` everywhere.

    Returns:
        idx: (M, nsample) long
        empty: (M,) bool, whether the row has no valid point
    """
    M, N = valid_mask.shape
    order = torch.cumsum(valid_mask.long(), dim=1)        # 1-based scan rank of valid pts
    cnt = order[:, -1]                                    # number of valid points per row
    first_valid = valid_mask.int().argmax(dim=1)          # position of first valid pt (0 if none)

    idx = torch.zeros(M, nsample, dtype=torch.long, device=valid_mask.device)
    for l in range(nsample):
        pos = torch.argmax((order == (l + 1)).int(), dim=1)      # position of the (l+1)-th valid pt
        present = torch.gather(order, 1, pos[:, None])[:, 0] == (l + 1)
        idx[:, l] = torch.where(present, pos, first_valid)
    empty = cnt == 0
    idx[empty] = -1
    return idx, empty


# ---------------------------------------------------------------------------
# ball query
# ---------------------------------------------------------------------------
def ball_query(radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt):
    """
    Args:
        xyz: (N1 + N2 ..., 3)
        xyz_batch_cnt: (batch_size)
        new_xyz: (M1 + M2 ..., 3) centers of the ball query
        new_xyz_batch_cnt: (batch_size)

    Returns:
        idx: (M1 + M2, nsample) int32, local (within batch) point indices
        empty_ball_mask: (M1 + M2,) bool
    """
    xyz = xyz.contiguous()
    new_xyz = new_xyz.contiguous()
    device = new_xyz.device
    B = xyz_batch_cnt.shape[0]
    radius2 = radius * radius

    starts_n = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(xyz_batch_cnt.long(), dim=0)])
    starts_m = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(new_xyz_batch_cnt.long(), dim=0)])

    idx_list, empty_list = [], []
    for b in range(B):
        n_b = int(xyz_batch_cnt[b])
        m_b = int(new_xyz_batch_cnt[b])
        if m_b == 0:
            continue
        xyz_b = xyz[starts_n[b]:starts_n[b] + n_b]
        new_xyz_b = new_xyz[starts_m[b]:starts_m[b] + m_b]
        for s in range(0, m_b, _CHUNK):
            e = min(s + _CHUNK, m_b)
            centers = new_xyz_b[s:e]                                 # (c, 3)
            dist2 = ((centers[:, None, :] - xyz_b[None, :, :]) ** 2).sum(-1)
            mask = dist2 < radius2
            idx_b, empty_b = _ball_query_slots(mask, nsample)
            idx_list.append(idx_b)
            empty_list.append(empty_b)

    if len(idx_list) == 0:
        idx = torch.zeros(0, nsample, dtype=torch.int32, device=device)
        empty_ball_mask = torch.zeros(0, dtype=torch.bool, device=device)
        return idx, empty_ball_mask

    idx = torch.cat(idx_list, dim=0).to(torch.int32)
    empty_ball_mask = torch.cat(empty_list, dim=0)
    idx[empty_ball_mask] = 0
    return idx, empty_ball_mask


# ---------------------------------------------------------------------------
# grouping operation (gather by index with per-batch offset)
# ---------------------------------------------------------------------------
def grouping_operation(features, features_batch_cnt, idx, idx_batch_cnt):
    """
    Args:
        features: (N1 + N2 ..., C)
        idx: (M1 + M2 ..., nsample), local (within batch) point indices
        idx_batch_cnt: (batch_size) number of grouped centers per batch

    Returns:
        output: (M1 + M2, C, nsample)
    """
    features = features.contiguous()
    idx = idx.contiguous()
    if idx.numel() == 0:
        return features.new_zeros((0, features.shape[1], idx.shape[1] if idx.dim() == 2 else 0))

    M, nsample = idx.shape
    device = features.device
    features_batch_cnt = features_batch_cnt.long()
    idx_batch_cnt = idx_batch_cnt.long()
    B = idx_batch_cnt.shape[0]

    start_feat = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(features_batch_cnt, dim=0)])
    bs_idx_m = torch.arange(B, device=device).repeat_interleave(idx_batch_cnt)
    global_idx = start_feat[bs_idx_m][:, None] + idx.long()          # (M, nsample)
    out = features[global_idx].permute(0, 2, 1)                      # (M, C, nsample)
    return out


def grouping_operation_grad(features_grad):
    # not used: gather/permute above are already differentiable through autograd
    raise NotImplementedError


# ---------------------------------------------------------------------------
# farthest point sampling
# ---------------------------------------------------------------------------
def farthest_point_sample(xyz, npoint):
    """
    Args:
        xyz: (B, N, 3)
        npoint: int

    Returns:
        output: (B, npoint) long
    """
    xyz = xyz.contiguous()
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    for b in range(B):
        cur_xyz = xyz[b]                                            # (N, 3)
        dist_to_set = torch.full((N,), 1e10, dtype=xyz.dtype, device=device)
        old = 0
        for j in range(npoint):
            idx[b, j] = old
            d = ((cur_xyz - cur_xyz[old]) ** 2).sum(-1)
            dist_to_set = torch.minimum(dist_to_set, d)
            old = int(torch.argmax(dist_to_set))
    return idx


def stack_farthest_point_sample(xyz, xyz_batch_cnt, npoint):
    """
    Args:
        xyz: (N1 + N2 + ..., 3)
        xyz_batch_cnt: (batch_size)
        npoint: int / list / tensor of numbers of sampled points per batch

    Returns:
        output: (npoint.sum(),) long global point indices
    """
    xyz = xyz.contiguous()
    batch_size = xyz_batch_cnt.shape[0]
    if not isinstance(npoint, torch.Tensor):
        if not isinstance(npoint, list):
            npoint = [npoint for _ in range(batch_size)]
        npoint = torch.tensor(npoint, device=xyz.device).int()

    device = xyz.device
    starts_n = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(xyz_batch_cnt.long(), dim=0)])
    idxs = []
    for b in range(batch_size):
        n = int(xyz_batch_cnt[b])
        m = int(npoint[b])
        if m == 0:
            continue
        cur = xyz[starts_n[b]:starts_n[b] + n]                     # (n, 3)
        dist = torch.full((n,), 1e10, dtype=xyz.dtype, device=device)
        out = torch.zeros(m, dtype=torch.long, device=device)
        old = 0
        for j in range(m):
            out[j] = old + int(starts_n[b])
            d = ((cur - cur[old]) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            old = int(torch.argmax(dist))
        idxs.append(out)
    if len(idxs) == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(idxs)


# ---------------------------------------------------------------------------
# three nearest neighbors + interpolation
# ---------------------------------------------------------------------------
def three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt):
    """
    Args:
        unknown: (N1 + N2 ..., 3)
        known: (M1 + M2 ..., 3)

    Returns:
        dist: (N1 + N2 ..., 3) distance (sqrt of squared-dist) of the 3 nearest neighbors
        idx: (N1 + N2 ..., 3) global indices of the 3 nearest neighbors
    """
    unknown = unknown.contiguous()
    known = known.contiguous()
    device = unknown.device
    B = unknown_batch_cnt.shape[0]

    starts_u = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(unknown_batch_cnt.long(), dim=0)])
    starts_k = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(known_batch_cnt.long(), dim=0)])

    dist_list, idx_list = [], []
    for b in range(B):
        n_u = int(unknown_batch_cnt[b])
        n_k = int(known_batch_cnt[b])
        u = unknown[starts_u[b]:starts_u[b] + n_u]                # (n_u, 3)
        k = known[starts_k[b]:starts_k[b] + n_k]                  # (n_k, 3)
        for s in range(0, n_u, _CHUNK):
            e = min(s + _CHUNK, n_u)
            uc = u[s:e]
            dist2 = ((uc[:, None, :] - k[None, :, :]) ** 2).sum(-1)  # (c, n_k)
            pick = min(3, n_k)
            if pick == 0:
                d2 = torch.full((e - s, 3), 1e40, dtype=unknown.dtype, device=device)
                ind = torch.zeros(e - s, 3, dtype=torch.long, device=device)
            else:
                d2, ind = torch.topk(dist2, pick, dim=1, largest=False)
                if pick < 3:
                    d2 = torch.cat([d2, torch.full((e - s, 3 - pick), 1e40, dtype=unknown.dtype, device=device)], dim=1)
                    ind = torch.cat([ind, torch.zeros(e - s, 3 - pick, dtype=torch.long, device=device)], dim=1)
            idx_list.append(ind + int(starts_k[b]))
            dist_list.append(d2)
    dist = torch.sqrt(torch.cat(dist_list, dim=0))
    idx = torch.cat(idx_list, dim=0).to(torch.int32)
    return dist, idx


def three_interpolate(features, idx, weight):
    """
    Args:
        features: (M1 + M2 ..., C)
        idx: (N1 + N2 ..., 3) global feature indices
        weight: (N1 + N2 ..., 3)

    Returns:
        out: (N1 + N2 ..., C)
    """
    features = features.contiguous()
    idx = idx.contiguous()
    weight = weight.contiguous()
    gathered = features[idx.long()]                          # (N, 3, C)
    out = (gathered * weight.unsqueeze(-1)).sum(dim=1)       # (N, C)
    return out


# ---------------------------------------------------------------------------
# three-NN for vector pool (two step: local neighbor query + three-NN)
# ---------------------------------------------------------------------------
def three_nn_for_vector_pool_by_two_step(support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers,
                                         new_xyz_batch_cnt, max_neighbour_distance, nsample, neighbor_type,
                                         avg_length_of_neighbor_idxs, num_total_grids, neighbor_distance_multiplier):
    """
    Args:
        support_xyz: (N1 + N2 ..., 3)
        xyz_batch_cnt: (batch_size)
        new_xyz: (M1 + M2 ..., 3) centers of the ball query
        new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3)
        new_xyz_batch_cnt: (batch_size)
        max_neighbour_distance: float
        nsample: find all (-1) or limited number (>0)
        neighbor_type: 1: ball, others: cube

    Returns:
        new_xyz_grid_dist: (M1 + M2 ..., num_total_grids, 3)
        new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) int32 (global indices, -1 for empty)
        num_avg_length_of_neighbor_idxs: int tensor
    """
    support_xyz = support_xyz.contiguous()
    new_xyz = new_xyz.contiguous()
    new_xyz_grid_centers = new_xyz_grid_centers.contiguous()
    device = support_xyz.device
    B = xyz_batch_cnt.shape[0]
    query_dist = max_neighbour_distance * neighbor_distance_multiplier
    radius2 = query_dist * query_dist

    starts_n = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(xyz_batch_cnt.long(), dim=0)])
    starts_m = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(new_xyz_batch_cnt.long(), dim=0)])

    M = new_xyz.shape[0]
    dist_all = []
    idx_all = []
    total_neighbors = 0
    for b in range(B):
        n_b = int(xyz_batch_cnt[b])
        m_b = int(new_xyz_batch_cnt[b])
        if m_b == 0:
            continue
        sup = support_xyz[starts_n[b]:starts_n[b] + n_b]                    # (n, 3)
        centers = new_xyz[starts_m[b]:starts_m[b] + m_b]                    # (m, 3)
        grid_centers = new_xyz_grid_centers[starts_m[b]:starts_m[b] + m_b]  # (m, T, 3)
        for mi in range(m_b):
            diff = sup - centers[mi][None, :]                               # (n, 3)
            if neighbor_type == 1:
                mask = (diff ** 2).sum(-1) <= radius2
            else:
                mask = (diff.abs() <= query_dist).all(dim=1)
            valid_indices = torch.nonzero(mask).flatten()                   # local scan-order indices
            cnt = valid_indices.numel()
            if nsample > 0 and cnt > nsample:
                valid_indices = valid_indices[:nsample]
                cnt = nsample
            if cnt > 1000:
                valid_indices = valid_indices[:1000]
                cnt = 1000
            total_neighbors += cnt

            d = torch.zeros(num_total_grids, 3, dtype=support_xyz.dtype, device=device)
            i2 = torch.full((num_total_grids, 3), -1, dtype=torch.long, device=device)
            if cnt > 0:
                nb_xyz = sup[valid_indices]                                 # (cnt, 3)
                gc = grid_centers[mi]                                       # (T, 3)
                sd = ((gc[:, None, :] - nb_xyz[None, :, :]) ** 2).sum(-1)  # (T, cnt)
                pick = min(cnt, 3)
                sd_top, ind_top = torch.topk(sd, pick, dim=1, largest=False)
                d[:, :pick] = sd_top
                i2[:, :pick] = valid_indices[ind_top]
                if pick < 3:
                    d[:, pick:] = sd_top[:, 0:1].expand(-1, 3 - pick)
                    i2[:, pick:] = i2[:, 0:1].expand(-1, 3 - pick)
            dist_all.append(d)
            idx_all.append(i2)

    if len(idx_all) == 0:
        dist_out = torch.zeros(0, num_total_grids, 3, dtype=support_xyz.dtype, device=device)
        idx_out = torch.zeros(0, num_total_grids, 3, dtype=torch.int32, device=device)
    else:
        dist_out = torch.sqrt(torch.stack(dist_all, dim=0))
        idx_out = torch.stack(idx_all, dim=0)

    avg = total_neighbors // M + (1 if total_neighbors % M > 0 else 0) if M > 0 else 0
    return dist_out, idx_out.to(torch.int32), torch.tensor(avg, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# vector pool (with voxel query) + its backward
# ---------------------------------------------------------------------------
class VectorPoolWithVoxelQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, support_xyz, xyz_batch_cnt, support_features, new_xyz, new_xyz_batch_cnt,
                num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance, num_c_out_each_grid,
                use_xyz, num_mean_points_per_grid=100, nsample=-1, neighbor_type=0, pooling_type=0):
        """
        Args:
            support_xyz: (N1 + N2 ..., 3)
            support_features: (N1 + N2 ..., C_in)
            new_xyz: (M1 + M2 ..., 3)
            neighbor_type: 1: ball, others: cube
            pooling_type: 0: avg_pool, 1: random choice (first point of each grid)

        Returns:
            new_features: (M1 + M2 ..., num_c_out_each_grid * num_total_grids)
            new_local_xyz: (M1 + M2 ..., 3 * num_total_grids)
            num_mean_points_per_grid: int tensor
            point_cnt_of_grid: (M1 + M2 ..., num_total_grids) int
        """
        support_xyz = support_xyz.contiguous()
        support_features = support_features.contiguous()
        new_xyz = new_xyz.contiguous()

        device = support_features.device
        B = xyz_batch_cnt.shape[0]
        num_total_grids = num_grid_x * num_grid_y * num_grid_z
        num_c_in = support_features.shape[1]
        num_c_out = num_c_out_each_grid * num_total_grids
        M = new_xyz.shape[0]

        grid_size_x = max_neighbour_distance * 2 / num_grid_x
        grid_size_y = max_neighbour_distance * 2 / num_grid_y
        grid_size_z = max_neighbour_distance * 2 / num_grid_z
        radius2 = max_neighbour_distance * max_neighbour_distance

        starts_n = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(xyz_batch_cnt.long(), dim=0)])
        starts_m = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(new_xyz_batch_cnt.long(), dim=0)])

        new_features = support_features.new_zeros((M, num_c_out))
        new_local_xyz = support_features.new_zeros((M, 3 * num_total_grids))
        point_cnt_of_grid = torch.zeros((M, num_total_grids), dtype=xyz_batch_cnt.dtype, device=device)

        grouped_idxs_list = []
        total_recorded = 0

        for b in range(B):
            n_b = int(xyz_batch_cnt[b])
            m_b = int(new_xyz_batch_cnt[b])
            if m_b == 0 or n_b == 0:
                continue
            sup = support_xyz[starts_n[b]:starts_n[b] + n_b]                    # (n, 3)
            feats = support_features[starts_n[b]:starts_n[b] + n_b]             # (n, C_in)
            centers = new_xyz[starts_m[b]:starts_m[b] + m_b]                    # (m, 3)
            # fold input channels into num_c_out_each_grid buckets by (i % each_grid)
            folded = feats.view(n_b, num_c_out_each_grid, -1).sum(dim=2)        # (n, each)

            for mi in range(m_b):
                diff = sup - centers[mi][None, :]                               # (n, 3)
                if neighbor_type == 1:
                    mask = (diff ** 2).sum(-1) <= radius2
                else:
                    mask = (diff.abs() <= max_neighbour_distance).all(dim=1)
                valid_indices = torch.nonzero(mask).flatten()
                n_valid = valid_indices.numel()
                if n_valid == 0:
                    continue
                local = diff[valid_indices]                                     # (nv, 3)
                gx = torch.floor((local[:, 0] + max_neighbour_distance) / grid_size_x) \
                    .clamp(0, num_grid_x - 1).long()
                gy = torch.floor((local[:, 1] + max_neighbour_distance) / grid_size_y) \
                    .clamp(0, num_grid_y - 1).long()
                gz = torch.floor((local[:, 2] + max_neighbour_distance) / grid_size_z) \
                    .clamp(0, num_grid_z - 1).long()
                grid = (gx * (num_grid_y * num_grid_z) + gy * num_grid_z + gz)  # (nv,)

                if pooling_type == 0:
                    # avg pooling over all valid points
                    cnt_ones = torch.ones_like(grid, dtype=point_cnt_of_grid.dtype)
                    point_cnt_of_grid[starts_m[b] + mi].scatter_add_(0, grid, cnt_ones)
                    new_features[starts_m[b] + mi].view(num_total_grids, num_c_out_each_grid) \
                        .scatter_add_(0, grid[:, None].expand(-1, num_c_out_each_grid),
                                      folded[valid_indices])
                    if use_xyz:
                        new_local_xyz[starts_m[b] + mi].view(num_total_grids, 3) \
                            .scatter_add_(0, grid[:, None].expand(-1, 3), local)

                    # record up to `nsample` assignments per center for the backward
                    rec = n_valid if nsample <= 0 else min(n_valid, nsample)
                    if rec > 0:
                        rec_grid = grid[:rec]
                        rec_feat = valid_indices[:rec] + int(starts_n[b])
                        rec_center = torch.full_like(rec_grid, int(starts_m[b]) + mi)
                        grouped_idxs_list.append(torch.stack([rec_feat, rec_center, rec_grid], dim=1))
                        total_recorded += rec
                else:
                    # pooling_type == 1: keep the first point of each grid (scan order)
                    claimed = torch.zeros(num_total_grids, dtype=torch.bool, device=device)
                    sample_cnt = 0
                    for k in range(n_valid):
                        g = int(grid[k])
                        if claimed[g]:
                            continue
                        claimed[g] = True
                        point_cnt_of_grid[int(starts_m[b]) + mi, g] = 1
                        # the cuda kernel assigns, for each input channel i,
                        #   new_features[g * each + i % each] = support_features[i]
                        # so the last input channel in each residue class wins.
                        cur_feat = feats[valid_indices[k]]
                        new_features[int(starts_m[b]) + mi, g * num_c_out_each_grid:(g + 1) * num_c_out_each_grid] \
                            = cur_feat[(num_c_in - num_c_out_each_grid):]
                        if use_xyz:
                            new_local_xyz[int(starts_m[b]) + mi, g * 3:(g + 1) * 3] = local[k]
                        grouped_idxs_list.append(
                            torch.tensor([[int(valid_indices[k]) + int(starts_n[b]),
                                           int(starts_m[b]) + mi, g]],
                                         dtype=torch.long, device=device))
                        total_recorded += 1
                        sample_cnt += 1
                        if (nsample > 0 and sample_cnt >= nsample) or sample_cnt >= num_total_grids:
                            break

        normalizer = torch.clamp_min(point_cnt_of_grid[:, :, None].float(), min=1e-6)
        new_features = (new_features.view(M, num_total_grids, num_c_out_each_grid) / normalizer).view(M, num_c_out)
        if use_xyz:
            new_local_xyz = (new_local_xyz.view(M, num_total_grids, 3) / normalizer).view(M, num_total_grids * 3)

        num_mean_points_per_grid = torch.tensor(
            [total_recorded // M + (1 if total_recorded % M > 0 else 0) if M > 0 else 0],
            dtype=torch.int32, device=device)

        grouped_idxs = torch.cat(grouped_idxs_list, dim=0) if len(grouped_idxs_list) else None
        ctx.save_for_backward(point_cnt_of_grid, grouped_idxs)
        ctx.vector_pool_for_backward = (num_c_in, num_c_out_each_grid, num_total_grids, M, support_features.shape[0])
        ctx.mark_non_differentiable(new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid)
        return new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid

    @staticmethod
    def backward(ctx, grad_new_features, grad_local_xyz, grad_num_mean, grad_point_cnt):
        point_cnt_of_grid, grouped_idxs = ctx.saved_tensors
        num_c_in, num_c_each_grid, num_total_grids, M, N = ctx.vector_pool_for_backward

        ret_none = (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)

        if grouped_idxs is None or grouped_idxs.numel() == 0:
            grad_support_features = grad_new_features.new_zeros((N, num_c_in))
            return (None, None, grad_support_features) + ret_none[3:]

        grad_support_features = grad_new_features.new_zeros((N, num_c_in))

        feat_idx = grouped_idxs[:, 0]                                    # (A,)
        center_idx = grouped_idxs[:, 1]                                  # (A,) global center row index
        grid = grouped_idxs[:, 2]                                        # (A,)
        denom = 1.0 / torch.clamp_min(point_cnt_of_grid[center_idx, grid].float(), 1.0)

        # flattened index into new_features (M, num_c_out) for each (assignment, input channel)
        base = center_idx * num_total_grids * num_c_each_grid + grid * num_c_each_grid   # (A,)
        ch = torch.arange(num_c_in, device=grad_new_features.device) % num_c_each_grid
        src_idx = base[:, None] + ch[None, :]                            # (A, num_c_in)
        contrib = grad_new_features.contiguous().view(-1)[src_idx] * denom[:, None]
        grad_support_features.scatter_add_(
            0, feat_idx[:, None].expand(-1, num_c_in), contrib)

        return (None, None, grad_support_features) + ret_none[3:]


vector_pool_with_voxel_query_op = VectorPoolWithVoxelQuery.apply


# ---------------------------------------------------------------------------
# voxel query (group points by neighboring voxels, used by VoxelQueryAndGrouping)
# ---------------------------------------------------------------------------
def voxel_query(max_range, radius, nsample, xyz, new_xyz, new_coords, point_indices):
    """
    Args:
        max_range: (z_range, y_range, x_range) numbers of neighbor voxels
        radius: float
        nsample: int
        xyz: (N1 + N2 ..., 3)
        new_xyz: (M1 + M2 ..., 3) centers of the voxel query
        new_coords: (M1 + M2, 4) [batch_id, z, y, x] coords of keypoints
        point_indices: (batch_size, Z, Y, X) int, point index of each voxel (-1 for empty)

    Returns:
        idx: (M1 + M2, nsample) int32, global point indices
        empty_ball_mask: (M1 + M2,) bool
    """
    device = new_xyz.device
    M, nsample = new_coords.shape[0], nsample
    B, Z, Y, X = point_indices.shape
    z_range, y_range, x_range = max_range

    idx = torch.zeros(M, nsample, dtype=torch.long, device=device)
    empty_mask = torch.zeros(M, dtype=torch.bool, device=device)

    point_indices_flat = point_indices.reshape(-1)
    for b in range(B):
        rows_b = torch.nonzero(new_coords[:, 0] == b).flatten()
        if rows_b.numel() == 0:
            continue
        coords_b = new_coords[rows_b]                                  # (m_b, 4)
        centers = new_xyz[rows_b]                                      # (m_b, 3)
        m_b = rows_b.numel()
        zc0 = coords_b[:, 1]
        yc0 = coords_b[:, 2]
        xc0 = coords_b[:, 3]

        cand_idx = []
        cand_mask = []
        for dz in range(-z_range, z_range + 1):
            for dy in range(-y_range, y_range + 1):
                for dx in range(-x_range, x_range + 1):
                    zc, yc, xc = zc0 + dz, yc0 + dy, xc0 + dx
                    inb = (zc >= 0) & (zc < Z) & (yc >= 0) & (yc < Y) & (xc >= 0) & (xc < X)
                    fzc = zc.clamp(0, Z - 1)
                    fyc = yc.clamp(0, Y - 1)
                    fxc = xc.clamp(0, X - 1)
                    flat = b * (Z * Y * X) + fzc * (Y * X) + fyc * X + fxc
                    nbr = point_indices_flat[flat]                     # (m_b,)
                    nbr_safe = nbr.clamp(min=0)
                    nb_xyz = xyz[nbr_safe]                             # (m_b, 3)
                    d2 = ((centers - nb_xyz) ** 2).sum(-1)
                    valid = inb & (nbr >= 0) & (d2 <= radius * radius)
                    cand_idx.append(nbr_safe)
                    cand_mask.append(valid)

        cand_idx = torch.stack(cand_idx, dim=1)                        # (m_b, num_off)
        cand_mask = torch.stack(cand_mask, dim=1)
        idx_b, empty_b = _ball_query_slots(cand_mask, nsample)
        idx_b = torch.gather(cand_idx, 1, idx_b.clamp(min=0))          # map slot position -> neighbor index
        idx_b[empty_b] = -1
        idx[rows_b] = idx_b.long()
        empty_mask[rows_b] = empty_b

    idx = idx.to(torch.int32)
    idx[empty_mask] = 0
    return idx, empty_mask


# ---------------------------------------------------------------------------
# QueryAndGroup used by StackSAModuleMSG etc.
# ---------------------------------------------------------------------------
class QueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None):
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)   # (M, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)
        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        return new_features, idx