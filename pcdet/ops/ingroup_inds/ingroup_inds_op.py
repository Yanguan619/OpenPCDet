import torch
from torch.autograd import Function

try:
    from . import ingroup_inds_cuda
    INGROUP_CUDA_ENABLED = True
except ImportError:
    ingroup_inds_cuda = None
    INGROUP_CUDA_ENABLED = False


def ingroup_inds_native(group_inds):
    """
    Torch-native implementation of in-group indices.

    Args:
        group_inds: (N,) int64 tensor of group ids (e.g. contiguous window ids).

    Returns:
        out_inds: (N,) int64 tensor, 0-based sequential index of each point
        inside its own group. The order inside a group is deterministic
        (points are processed in group-id then point-id order).
    """
    out_inds = torch.full_like(group_inds, -1)
    N = group_inds.numel()
    if N == 0:
        return out_inds

    sorted_groups, order = torch.sort(group_inds)
    is_new = torch.ones(N, dtype=torch.bool, device=group_inds.device)
    is_new[1:] = sorted_groups[1:] != sorted_groups[:-1]

    seq = torch.arange(N, dtype=torch.long, device=group_inds.device)
    first_pos = torch.where(is_new, seq, torch.zeros_like(seq))
    # running max of the first-occurrence positions gives, for every element,
    # the position of the first element of its contiguous group.
    first_pos = torch.cummax(first_pos, dim=0).values
    within = seq - first_pos
    out_inds[order] = within
    return out_inds


class IngroupIndicesFunction(Function):

    @staticmethod
    def forward(ctx, group_inds):

        if INGROUP_CUDA_ENABLED:
            out_inds = torch.zeros_like(group_inds) - 1
            ingroup_inds_cuda.forward(group_inds, out_inds)
        else:
            out_inds = ingroup_inds_native(group_inds)

        ctx.mark_non_differentiable(out_inds)

        return out_inds

    @staticmethod
    def backward(ctx, g):

        return None


ingroup_inds = IngroupIndicesFunction.apply