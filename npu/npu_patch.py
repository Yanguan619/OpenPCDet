"""NPU 适配补丁：统一管理设备检测、算子适配与可重复调用的初始化入口。

被 npu/infer.py 与 npu/eval.py 复用，集中处理：
1. 设备检测（npu / cuda / cpu）
2. torch_npu 初始化（关闭 jit_compile 以避免逐帧编译）
3. anchors / tensor 的跨设备搬运
4. voxelization 相关适配（build_index_map 等）

用法:
    from npu.npu_patch import init_patch, get_device, to_tensor, build_index_map, PPWrapper
    device = init_patch()
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import numpy as np
import torch


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


if __name__ == "__main__":
    print("device =", init_patch())
