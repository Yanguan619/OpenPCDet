"""NPU 纯 numpy/numba 算子（无编译扩展依赖）。

目前仅 iou3d_nms_torch_native（BEV IoU / rotated NMS / box_overlap_bev），
被 om_ref_demo / om_ref_test 的后处理直接 import 使用。
pcdet 侧 ops 的 native 降级副本位于 pcdet/ops 各目录内（try/except fallback），
不再经本目录重复注入。
"""
