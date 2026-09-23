import torch


def farthest_point_sample(xyz, npoint):
    """
    Iterative farthest point sampling (batch version).
    :param xyz: (B, N, 3)
    :param npoint: int
    :return: (B, npoint) int64 indices
    """
    B, N, _ = xyz.shape
    if npoint <= 0 or N == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=xyz.device)
    if N < npoint:
        npoint = N

    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    temp = torch.full((B, N), 1e10, dtype=xyz.dtype, device=xyz.device)
    for j in range(npoint):
        if j == 0:
            besti = torch.zeros(B, dtype=torch.long, device=xyz.device)
        else:
            besti = temp.argmax(dim=1)
        idx[:, j] = besti
        cur = xyz.gather(1, besti[:, None, None].expand(B, 1, 3)).squeeze(1)
        d2 = ((xyz - cur[:, None, :]) ** 2).sum(-1)
        temp = torch.minimum(temp, d2)
    return idx


def gather_operation(features, idx):
    """
    :param features: (B, C, N)
    :param idx: (B, npoint) int64
    :return: (B, C, npoint)
    """
    B, C, N = features.shape
    npoint = idx.shape[1]
    idx_exp = idx.reshape(B, 1, -1).expand(B, C, -1)
    return features.gather(2, idx_exp)


def three_nn(unknown, known):
    """
    :param unknown: (B, N, 3)
    :param known: (B, M, 3)
    :return: dist (B, N, 3), idx (B, N, 3) int64
    """
    B, N, _ = unknown.shape
    M = known.shape[1]
    if M == 0:
        dist2 = torch.full((B, N, 3), 1e40, dtype=unknown.dtype, device=unknown.device)
        idx = torch.zeros(B, N, 3, dtype=torch.long, device=unknown.device)
        return torch.sqrt(dist2), idx
    d2 = ((unknown[:, :, None, :] - known[:, None, :, :]) ** 2).sum(-1)  # (B, N, M)
    k = min(3, M)
    topk_d2, topk_i = torch.topk(d2, k, dim=-1, largest=False)
    if k < 3:
        pad_d2 = torch.full((B, N, 3 - k), 1e40, dtype=d2.dtype, device=d2.device)
        pad_i = torch.zeros(B, N, 3 - k, dtype=torch.long, device=d2.device)
        topk_d2 = torch.cat([topk_d2, pad_d2], dim=-1)
        topk_i = torch.cat([topk_i, pad_i], dim=-1)
    return torch.sqrt(topk_d2), topk_i.long()


def three_interpolate(features, idx, weight):
    """
    :param features: (B, C, M)
    :param idx: (B, N, 3) int64
    :param weight: (B, N, 3)
    :return: (B, C, N)
    """
    B, C, N = features.shape[0], features.shape[1], idx.shape[1]
    idx_exp = idx.reshape(B, 1, -1).expand(B, C, -1)
    gathered = features.gather(2, idx_exp)  # (B, C, N*3)
    gathered = gathered.view(B, C, N, 3)
    out = (gathered * weight.unsqueeze(1)).sum(-1)
    return out


def grouping_operation(features, idx):
    """
    :param features: (B, C, N)
    :param idx: (B, npoint, nsample) int64
    :return: (B, C, npoint, nsample)
    """
    B, C, _ = features.shape
    npoint, nsample = idx.shape[1], idx.shape[2]
    idx_exp = idx.reshape(B, 1, -1).expand(B, C, -1)
    gathered = features.gather(2, idx_exp)
    return gathered.view(B, C, npoint, nsample)


def ball_query(radius, nsample, xyz, new_xyz):
    """
    :param radius: float
    :param nsample: int
    :param xyz: (B, N, 3)
    :param new_xyz: (B, M, 3)
    :return: idx (B, M, nsample) int64
    """
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    idx = torch.zeros(B, M, nsample, dtype=torch.long, device=xyz.device)
    d2 = ((new_xyz[:, :, None, :] - xyz[:, None, :, :]) ** 2).sum(-1)  # (B, M, N)
    mask = d2 < radius * radius
    for b in range(B):
        for m in range(M):
            hits = torch.nonzero(mask[b, m]).flatten()
            cnt = hits.numel()
            if cnt == 0:
                continue
            if cnt >= nsample:
                idx[b, m] = hits[:nsample]
            else:
                idx[b, m, :cnt] = hits[:cnt]
                idx[b, m, cnt:] = hits[0]
    return idx
