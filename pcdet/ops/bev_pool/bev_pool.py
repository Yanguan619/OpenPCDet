import torch

try:
    from . import bev_pool_ext
    BEV_POOL_CUDA_ENABLED = True
except ImportError:
    bev_pool_ext = None
    BEV_POOL_CUDA_ENABLED = False

__all__ = ["bev_pool"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def bev_pool_native(feats, coords, ranks, B, D, H, W):
    """Torch-native implementation of the BEV pooling.

    Both ``feats`` and ``coords`` are assumed to be already sorted by ``ranks``,
    so that points mapping to the same BEV position appear consecutively.
    """
    x, geom_feats = QuickCumsum.apply(feats, coords, ranks)  # (num_unique, C), (num_unique, 4)

    # scatter the per-position sums into a dense volume whose layout is
    # coords = [x, y, z, batch]  ->  out[batch, z, y, x, channel]
    batch_ix = geom_feats[:, 3]
    z_ix = geom_feats[:, 2]
    y_ix = geom_feats[:, 1]
    x_ix = geom_feats[:, 0]
    flat_ix = batch_ix * (D * H * W) + z_ix * (H * W) + y_ix * W + x_ix

    out = feats.new_zeros((B * D * H * W, feats.shape[1]))
    out[flat_ix.long()] = x  # differentiable scatter
    out = out.view(B, D, H, W, feats.shape[1])
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out


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

    if BEV_POOL_CUDA_ENABLED:
        x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    else:
        x = bev_pool_native(feats, coords, ranks, B, D, H, W)
        return x