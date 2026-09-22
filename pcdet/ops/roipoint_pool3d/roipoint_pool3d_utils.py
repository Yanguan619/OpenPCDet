import torch
import torch.nn as nn
from torch.autograd import Function

from ...utils import box_utils

try:
    from . import roipoint_pool3d_cuda
    ROIPOOL3D_CUDA_ENABLED = True
except ImportError:
    roipoint_pool3d_cuda = None
    ROIPOOL3D_CUDA_ENABLED = False


def roipoint_pool3d_native(points, pooled_boxes3d, point_features, pooled_features, pooled_empty_flag):
    """
    Torch-native implementation of the ROI-point pooling kernel.

    Args:
        points: (B, N, 3)
        pooled_boxes3d: (B, M, 7) enlarged boxes [x, y, z, dx, dy, dz, heading]
        point_features: (B, N, C)
        pooled_features: (B, M, num_sampled_points, 3 + C) pre-allocated output
        pooled_empty_flag: (B, M) pre-allocated int output (1 if a box has no point)
    """
    batch_size, pts_num, _ = points.shape
    boxes_num, sampled_pts_num, feature_len = pooled_features.shape[1], pooled_features.shape[2], point_features.shape[2]
    if pts_num == 0 or boxes_num == 0:
        return

    S = sampled_pts_num
    MARGIN = 1e-5

    for b in range(batch_size):
        pts = points[b]                 # (N, 3)
        feats = point_features[b]       # (N, C)
        boxes = pooled_boxes3d[b]       # (M, 7)
        if points.shape[0] == 0:
            pooled_empty_flag[b] = 1
            continue

        cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
        dx, dy, dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]
        rz = boxes[:, 6]

        shift = pts[None, :, :] - boxes[:, None, :3]                    # (M, N, 3)
        cosa = torch.cos(-rz)
        sina = torch.sin(-rz)
        local_x = shift[..., 0] * cosa[:, None] - shift[..., 1] * sina[:, None]
        local_y = shift[..., 0] * sina[:, None] + shift[..., 1] * cosa[:, None]
        local_z = shift[..., 2]

        in_x = local_x.abs() < (dx[:, None] / 2.0 + MARGIN)
        in_y = local_y.abs() < (dy[:, None] / 2.0 + MARGIN)
        in_z = local_z.abs() <= (dz[:, None] / 2.0)
        mask = in_x & in_y & in_z                                        # (M, N)

        for m in range(boxes_num):
            in_pts = torch.nonzero(mask[m]).flatten()                    # (num_in,)
            cnt = in_pts.numel()
            if cnt == 0:
                pooled_empty_flag[b, m] = 1
                continue
            if cnt >= S:
                idx = in_pts[:S]
            else:
                # duplicate the same points for sampling like the cuda kernel
                pad = in_pts[torch.arange(S - cnt, device=in_pts.device) % cnt]
                idx = torch.cat([in_pts, pad])
            pooled_features[b, m, :, :3] = pts[idx]
            pooled_features[b, m, :, 3:] = feats[idx]


class RoIPointPool3d(nn.Module):
    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dFunction.apply(
            points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points
        )


class RoIPointPool3dFunction(Function):
    @staticmethod
    def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[1], point_features.shape[2]
        pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)

        pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()

        if ROIPOOL3D_CUDA_ENABLED:
            roipoint_pool3d_cuda.forward(
                points.contiguous(), pooled_boxes3d.contiguous(),
                point_features.contiguous(), pooled_features, pooled_empty_flag
            )
        else:
            roipoint_pool3d_native(
                points, pooled_boxes3d, point_features, pooled_features, pooled_empty_flag
            )

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


if __name__ == '__main__':
    pass