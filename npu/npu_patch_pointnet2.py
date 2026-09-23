"""pointnet2 系列算子的 sys.modules 预注入 stub。

把「CUDA 扩展导入失败 → torch-native 降级」逻辑从 pcdet 源码抽离到本文件，
与 npu/npu_patch.py 分开管理（pointnet2 模块较多，单独成文件便于维护）。

被替换的 3 个模块（源码已恢复 HEAD，无 try/except）:
    pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils
    pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils
    pcdet.ops.pointnet2.pointnet2_stack.voxel_query_utils

必须在任意 `import pcdet` 之前调用一次 `patch_pointnet2_ops()`。
CUDA 环境（扩展可导入）不注入，直接使用原生 CUDA 实现。
"""

import sys
import types

import torch
import torch.nn as nn


def _cuda_ext_available():
    """真实 CUDA 扩展是否可导入（可导入则保留 pcdet 原模块，不注入 stub）。"""
    try:
        import pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda  # noqa: F401
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


def patch_pointnet2_ops():
    """预注入 pointnet2 三个模块的 native 降级 stub（幂等，可重复调用）。"""
    if _cuda_ext_available():
        return
    _patch_pointnet2_batch_utils()
    _patch_pointnet2_stack_utils()
    _patch_voxel_query_utils()


# ---------------------------------------------------------------------------
# pointnet2_batch.pointnet2_utils
# ---------------------------------------------------------------------------
def _patch_pointnet2_batch_utils():
    from npu.ops_native import pointnet2_batch_native as n

    # --- QueryAndGroup / GroupAll（HEAD 源码里基于 ball_query/grouping_operation 的 nn.Module）---
    class QueryAndGroup(nn.Module):
        def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
            super().__init__()
            self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

        def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
            idx = n.ball_query(self.radius, self.nsample, xyz, new_xyz)
            xyz_trans = xyz.transpose(1, 2).contiguous()
            grouped_xyz = n.grouping_operation(xyz_trans, idx)
            grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
            if features is not None:
                grouped_features = n.grouping_operation(features, idx)
                if self.use_xyz:
                    new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
                else:
                    new_features = grouped_features
            else:
                assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
                new_features = grouped_xyz
            return new_features

    class GroupAll(nn.Module):
        def __init__(self, use_xyz: bool = True):
            super().__init__()
            self.use_xyz = use_xyz

        def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
            grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
            if features is not None:
                grouped_features = features.unsqueeze(2)
                if self.use_xyz:
                    new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
                else:
                    new_features = grouped_features
            else:
                new_features = grouped_xyz
            return new_features

    _make_stub("pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils", {
        "farthest_point_sample": n.farthest_point_sample,
        "gather_operation": n.gather_operation,
        "three_nn": n.three_nn,
        "three_interpolate": n.three_interpolate,
        "grouping_operation": n.grouping_operation,
        "ball_query": n.ball_query,
        "QueryAndGroup": QueryAndGroup,
        "GroupAll": GroupAll,
    })


# ---------------------------------------------------------------------------
# pointnet2_stack.pointnet2_utils
# ---------------------------------------------------------------------------
def _patch_pointnet2_stack_utils():
    from npu.ops_native import pointnet2_utils_native as n

    _make_stub("pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils", {
        "ball_query": n.ball_query,
        "grouping_operation": n.grouping_operation,
        "farthest_point_sample": n.farthest_point_sample,
        "stack_farthest_point_sample": n.stack_farthest_point_sample,
        "three_nn": n.three_nn,
        "three_interpolate": n.three_interpolate,
        "three_nn_for_vector_pool_by_two_step": n.three_nn_for_vector_pool_by_two_step,
        "vector_pool_with_voxel_query_op": n.vector_pool_with_voxel_query_op,
        "voxel_query": n.voxel_query,
        "QueryAndGroup": n.QueryAndGroup,
    })


# ---------------------------------------------------------------------------
# pointnet2_stack.voxel_query_utils
# ---------------------------------------------------------------------------
def _patch_voxel_query_utils():
    from npu.ops_native import pointnet2_utils_native as n

    def voxel_query(max_range, radius, nsample, xyz, new_xyz, new_coords, point_indices):
        return n.voxel_query(max_range, radius, nsample, xyz, new_xyz, new_coords, point_indices)

    class VoxelQueryAndGrouping(nn.Module):
        def __init__(self, max_range: int, radius: float, nsample: int):
            super().__init__()
            self.max_range, self.radius, self.nsample = max_range, radius, nsample

        def forward(self, new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                    features, voxel2point_indices):
            assert xyz.shape[0] == xyz_batch_cnt.sum(), \
                'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(xyz_batch_cnt))
            assert new_coords.shape[0] == new_xyz_batch_cnt.sum(), \
                'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
            batch_size = xyz_batch_cnt.shape[0]

            idx1, empty_ball_mask1 = voxel_query(
                self.max_range, self.radius, self.nsample, xyz, new_xyz, new_coords, voxel2point_indices)

            idx1 = idx1.view(batch_size, -1, self.nsample)
            count = 0
            for bs_idx in range(batch_size):
                idx1[bs_idx] -= count
                count += xyz_batch_cnt[bs_idx]
            idx1 = idx1.view(-1, self.nsample)
            idx1[empty_ball_mask1] = 0

            idx = idx1
            empty_ball_mask = empty_ball_mask1

            grouped_xyz = n.grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
            grouped_features = n.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)
            return grouped_features, grouped_xyz, empty_ball_mask

    _make_stub("pcdet.ops.pointnet2.pointnet2_stack.voxel_query_utils", {
        "voxel_query": voxel_query,
        "VoxelQueryAndGrouping": VoxelQueryAndGrouping,
    })
