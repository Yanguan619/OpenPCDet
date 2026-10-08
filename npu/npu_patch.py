"""NPU 适配补丁：统一管理设备检测、算子适配与可重复调用的初始化入口。

被 npu/om_ref_demo.py、npu/om_ref_test.py、npu/export_onnx.py 等推理/转换脚本复用，
集中处理：
1. 设备检测（npu / cuda / cpu）
2. torch_npu 初始化（关闭 jit_compile 以避免逐帧编译）
3. spconv 别名 / AscendC 体素化 / numba mask 等 monkey patch
4. CPU rotate IoU（KITTI 评测用）
5. voxelization 共享工具（build_index_map，含 NPU 常驻 tensor 路径）

用法:
    from npu.npu_patch import init_patch, patch_rotate_iou, get_device, build_index_map
    device = init_patch()
"""

import math
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT)
)  # unum_ops 一律走 pip 安装（editable）解析，不做相邻 checkout 的 sys.path 兜底

import numba
import numpy as np
import torch


def _alias_spconv():
    """将 spconv / spconv.utils 别名到 unum_ops 的 spconv shim（sys.modules 注入）。

    使 `from spconv.utils import VoxelGeneratorV2` 无需仓库根目录的 spconv 符号链接，
    任何机器（含无 unum_ops 相邻部署）都能通过本补丁解析。必须在 import pcdet 之前调用。

    注意：必须用强制赋值而非 setdefault——unum_ops 自身在导入时会经 sparse_modules 触发
    顶层 `import spconv`，产生与 unum_ops.spconv 同路径的**第二份拷贝**；setdefault 会因
    该拷贝已存在而静默失效，导致下游拿到的类与这里 patch 的不是同一个。

    unum_ops 为硬依赖（同 _patch_voxelize_ascendc），缺失时直接 ImportError，不回退。
    """
    import unum_ops.spconv
    import unum_ops.spconv.utils  # noqa: F401
    sys.modules["spconv"] = sys.modules["unum_ops.spconv"]
    sys.modules["spconv.utils"] = sys.modules["unum_ops.spconv.utils"]


def _patch_voxelize_ascendc():
    """将 VoxelGeneratorV2.generate 路由到 unum_ops 的 AscendC 硬体素化（NPU kernel）。

    输出与 CPU numba 版**排序等价**（coord 多重集相同、同 coord 特征/npp 相同）但行序不同，
    coords 序为 (x,y,z) → 转回 spconv 的 (z,y,x)；端到端 200 帧 OM AP 与 numba 基线
    逐位一致（红线通过）。全量点（demo 无 FOV）~2.5x 快，FOV 后小输入持平——
    AscendC 全面不劣于 numba，故为**唯一实现**：无回退、无开关，
    unum_ops/OPP 缺失在 import 时直接 ImportError，运行期异常（如旧 OPP 的
    561000 context bug）带栈抛出，任何帧都不换 numba。
    幂等，可重复调用。
    """
    global _VOXELIZE_PATCHED
    if _VOXELIZE_PATCHED:
        return
    _VOXELIZE_PATCHED = True
    from unum_ops.spconv.utils import VoxelGeneratorV2
    from unum_ops.voxelization.voxelization_ascendc_v2 import voxelization

    def generate(self, points):
        p = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).npu()
        out = voxelization(
            p,
            voxel_size=[float(v) for v in self.voxel_size],
            pcr=[float(v) for v in self.point_cloud_range],
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )
        if _VOX_DEVICE_RESIDENT:
            # 设备常驻：输出保持 NPU tensor（下游 collate/index_map/feeds 由
            # 配套 patch 走 torch/aclruntime.BaseTensor 路径），消除
            # D2H→numpy→H2D 往返。coords 的 (x,y,z)→(z,y,x) 换列在 device
            # 侧完成，值与 numpy 版逐位一致。
            return {
                "voxels": out.voxels,
                "coordinates": out.coords[:, [2, 1, 0]].contiguous(),
                "num_points_per_voxel": out.num_points,
            }
        return {
            "voxels": out.voxels.cpu().numpy(),
            "coordinates": out.coords.cpu().numpy()[:, [2, 1, 0]],
            "num_points_per_voxel": out.num_points.cpu().numpy(),
        }

    VoxelGeneratorV2.generate = generate


_VOXELIZE_PATCHED = False

# 设备常驻模式：voxelization 输出以 NPU tensor 直通 collate/index_map/feeds，
# 消除 D2H→numpy→collate 拷贝→H2D 往返（省 ~4-5ms）。由调用脚本在 import 本模块
# 前设置 NPU_VOX_DEVICE_RESIDENT=1 开启（默认关：其余脚本保持 numpy 契约不变）。
_VOX_DEVICE_RESIDENT = os.environ.get("NPU_VOX_DEVICE_RESIDENT", "0") == "1"


# ---------------------------------------------------------------------------
# mask_points_and_boxes_outside_range 的 numba 单遍实现（值与 numpy 版逐位一致）。
#
# OpenPCDet 原版（common_utils.mask_points_by_range 生成 bool mask + 调用方
# fancy index gather）在本机 aarch64 上 ~3.2ms/帧：6 次全数组比较 + 5 次 and
# 产生多份临时，再布尔 gather；numba 单遍「判界 + 原地压缩」~0.3ms（-2.9ms）。
# 比较语义与 common_utils.mask_points_by_range 完全一致：只比 x/y（原版不含
# z），>= / <= 边界一致，行序保持原始顺序 → 输出逐位一致。gt_boxes 分支保持
# 原逻辑（demo 无 gt_boxes，训练路径不受影响）。NPU_NUMBA_MASK=0 关闭。
# ---------------------------------------------------------------------------
def _patch_mask_points_by_range():
    if os.environ.get("NPU_NUMBA_MASK", "1") != "1":
        return
    try:
        from functools import partial

        import numba

        from pcdet.datasets.processor import data_processor as dp_mod
        from pcdet.utils import box_utils, common_utils
    except Exception:
        return

    @numba.njit(cache=True)
    def _mask_compact(pts, lim, out):
        n = pts.shape[0]
        c = pts.shape[1]
        cnt = 0
        for i in range(n):
            x = pts[i, 0]
            y = pts[i, 1]
            if (x >= lim[0] and x <= lim[3]) and (y >= lim[1] and y <= lim[4]):
                for j in range(c):
                    out[cnt, j] = pts[i, j]
                cnt += 1
        return cnt

    def _npy_mask_compact(points, point_cloud_range):
        try:
            pts = np.ascontiguousarray(points, dtype=np.float32)
            lim = np.asarray(point_cloud_range, dtype=np.float32)
            out = np.empty_like(pts)
            cnt = _mask_compact(pts, lim, out)
            return out[:cnt]
        except Exception:
            return points[common_utils.mask_points_by_range(points, point_cloud_range)]

    def _mask_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        if data_dict.get("points", None) is not None:
            data_dict["points"] = _npy_mask_compact(data_dict["points"], self.point_cloud_range)
        if (
            data_dict.get("gt_boxes", None) is not None
            and config.REMOVE_OUTSIDE_BOXES
            and self.training
        ):
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict["gt_boxes"],
                self.point_cloud_range,
                min_num_corners=config.get("min_num_corners", 1),
                use_center_to_filter=config.get("USE_CENTER_TO_FILTER", True),
            )
            data_dict["gt_boxes"] = data_dict["gt_boxes"][mask]
        return data_dict

    dp_mod.DataProcessor.mask_points_and_boxes_outside_range = _mask_and_boxes_outside_range


_F32 = np.float32


# ---------------------------------------------------------------------------
# CPU-only rotated box IoU (BEV)：逐行复刻官方 numba CUDA 数学，数值一致。
# 输入格式: [x, z, w, l, ry] (camera BEV 坐标, ry 绕 y 轴)
# 输出: (N, K) IoU / 重叠矩阵
# ---------------------------------------------------------------------------


@numba.njit(nopython=True)
def _cpu_trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / _F32(2.0)


@numba.njit(nopython=True)
def _cpu_area(int_pts, num_of_inter):
    area_val = _F32(0.0)
    for i in range(num_of_inter - 2):
        area_val += abs(
            _cpu_trangle_area(
                int_pts[:2], int_pts[2 * i + 2 : 2 * i + 4], int_pts[2 * i + 4 : 2 * i + 6]
            )
        )
    return area_val


@numba.njit(nopython=True)
def _cpu_sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.zeros((2,), dtype=_F32)
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.zeros((2,), dtype=_F32)
        vs = np.zeros((16,), dtype=_F32)
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
    A = np.zeros((2,), dtype=_F32)
    B = np.zeros((2,), dtype=_F32)
    C = np.zeros((2,), dtype=_F32)
    D = np.zeros((2,), dtype=_F32)

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
    temp_pts = np.zeros((2,), dtype=_F32)
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
    corners_x = np.zeros((4,), dtype=_F32)
    corners_y = np.zeros((4,), dtype=_F32)
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
    corners1 = np.zeros((8,), dtype=_F32)
    corners2 = np.zeros((8,), dtype=_F32)
    intersection_corners = np.zeros((16,), dtype=_F32)

    _cpu_rbbox_to_corners(corners1, rbbox1)
    _cpu_rbbox_to_corners(corners2, rbbox2)

    num_intersection = _cpu_quadrilateral_intersection(corners1, corners2, intersection_corners)
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
                dev_query_boxes[i * 5 : i * 5 + 5], dev_boxes[tx * 5 : tx * 5 + 5], criterion
            )


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
    _cpu_rotate_iou_loop(
        N, K, boxes.reshape([-1]), query_boxes.reshape([-1]), iou.reshape([-1]), criterion
    )
    return iou.astype(box_dtype)


_ROTATE_IOU_MODULE_NAMES = [
    "pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou",
]


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


def build_index_map(voxel_coords, nx=432, ny=496, nz=1, M=None, pad=None):
    """构造 PointPillarScatter 的 Gather 索引表 (G,) int64（空位 = pad，默认 = M）。

    唯一实现，被 om_ref_demo / om_ref_test / export_onnx 共用
    （demo 侧 `from npu.om_ref_demo import build_index_map` 为本函数的转引）。

    NPU 常驻 tensor 走纯 torch 路径（device 侧构造，省 D2H→numpy scatter→H2D
    往返，实测 0.27ms vs numpy 路径含 host 往返）；整数算术与 numpy 路径
    逐位一致（voxel 坐标唯一，无重复索引写入）。注意 CPU tensor 必须走 numpy
    路径：本机 aarch64 的 torch CPU index_put_ 慢路径 ~9ms（numpy 仅 0.2ms）。

    Args:
        voxel_coords: (M, 4) [batch, x, y, z]（或 [batch, z, y, x]）tensor / ndarray
        nx, ny, nz: 体素网格尺寸
        M: voxel 数（默认取 voxel_coords 行数）
        pad: 空位填充值（默认 = M，静态 OM pad 行语义）

    Returns:
        torch.LongTensor: 展平 BEV 网格 -> voxel 索引，空位为 pad。
    """
    if M is None:
        M = voxel_coords.shape[0]
    if isinstance(voxel_coords, torch.Tensor) and getattr(voxel_coords, "is_npu", False):
        coords = voxel_coords.to(torch.int64)
        indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
        G = nx * ny * nz
        pad = M if pad is None else pad
        index_map = torch.full((G,), pad, dtype=torch.int64, device=coords.device)
        index_map[indices] = torch.arange(M, dtype=torch.int64, device=coords.device)
        return index_map
    coords = voxel_coords.numpy() if isinstance(voxel_coords, torch.Tensor) else voxel_coords
    indices = coords[:, 1] + coords[:, 2] * nx + coords[:, 3]
    G = nx * ny * nz
    pad = M if pad is None else pad
    index_map = np.full(G, pad, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    return torch.from_numpy(index_map)


# 生产链路（om_ref_demo/om_ref_test 等）只 import 本模块、不调 init_patch()，
# 故在此 import 时即：① alias spconv→unum_ops shim（否则顶层 spconv 与 unum_ops.spconv 是两份拷贝，
# patch 打不到消费者用的类）；② 体素化固定路由 AscendC（唯一实现，无回退无开关，
# unum_ops/OPP 为硬依赖，缺失 import 即报错）。
# ③ numba 单遍 mask（值与原版逐位一致，NPU_NUMBA_MASK=0 关闭）。
# 设备常驻输出的 collate 由 demo 的 collate_batch_fast 单帧 fast-path 承担
# （全链路单帧；多帧+tensor 场景当前无调用方，原 tensor-aware collate 补丁已删）。
# ①②③ 必须在任意 DataProcessor/DemoDataset 构造之前生效（processor 以 partial
# 捕获 bound method，晚 patch 不会命中已创建的实例）。
_alias_spconv()
_patch_voxelize_ascendc()
_patch_mask_points_by_range()
