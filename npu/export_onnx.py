"""导出 PointPillars ONNX，含后处理（sigmoid → topk → NMS → Gather），
ATC 后可直接输出最终检测框，无需 Python 后处理。

用法:
    python npu/export_onnx.py                          # 完整流程: 导出 base → 图手术
    python npu/export_onnx.py --output weights/pp_nms.onnx
    python npu/export_onnx.py --score-thresh 0.1 --iou-thresh 0.01 --max-det 500
    python npu/export_onnx.py --skip-export            # 复用已有 base ONNX，只做图手术
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unum_ops" / "src" / "unum_ops"))

import npu.npu_patch  # noqa: E402,F401  预注入 CUDA ops 降级 stub，必须在 import pcdet 之前

NUM_ANCHORS = 321408


# ============================================================
# Step 0: ONNX helpers
# ============================================================

def make_value(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def make_const(name, arr):
    dtype = mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist()), name=name)


# ============================================================
# Step 1: Export base ONNX (PPWrapper without postproc)
# ============================================================

def export_base_onnx(output_path, args):
    """导出 base ONNX（PPWrapper: 4 输入 → batch_box_preds, batch_cls_preds）。"""
    import torch
    import torch.nn as nn
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import KittiDataset
    from pcdet.utils import common_utils
    from pcdet.models.detectors.pointpillar import PointPillar

    cfg_from_yaml_file(args.config, cfg)
    logger = common_utils.create_logger()
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        training=False, logger=logger,
    )

    model = PointPillar(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.npu()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    data_dict = demo_dataset[demo_dataset.sample_id_list.index(args.sample_idx)]
    data_dict = demo_dataset.collate_batch([data_dict])
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ["frame_id", "metadata", "calib"]:
            continue
        elif key in ["images"]:
            data_dict[key] = torch.from_numpy(val).float().npu().contiguous()
        elif key in ["image_shape"]:
            data_dict[key] = torch.from_numpy(val).int().npu()
        else:
            data_dict[key] = torch.from_numpy(val).npu()

    voxels = data_dict["voxels"]
    voxel_num_points = data_dict["voxel_num_points"]
    voxel_coords = data_dict["voxel_coords"]
    M = voxels.shape[0]
    coords_np = voxel_coords.cpu().numpy()
    indices = coords_np[:, 1] + coords_np[:, 2] * 432 + coords_np[:, 3]
    index_map = np.full(432 * 496, M, dtype=np.int64)
    index_map[indices.astype(np.int64)] = np.arange(M, dtype=np.int64)
    bev_index_map = torch.from_numpy(index_map).npu()

    class PPWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module_list = model.module_list
        def forward(self, voxels, voxel_num_points, voxel_coords, bev_index_map):
            batch_dict = {"voxels": voxels, "voxel_num_points": voxel_num_points,
                          "voxel_coords": voxel_coords, "bev_index_map": bev_index_map, "batch_size": 1}
            for m in self.module_list:
                batch_dict = m(batch_dict)
            return batch_dict["batch_box_preds"], batch_dict["batch_cls_preds"]

    wrapper = PPWrapper(model)
    dynamic_axes = None
    if getattr(args, "dynamic", False):
        dynamic_axes = {
            "voxels": {0: "M"},
            "voxel_num_points": {0: "M"},
            "voxel_coords": {0: "M"},
            "batch_box_preds": {0: "batch"},
            "batch_cls_preds": {0: "batch"},
        }
    with torch.no_grad():
        torch.onnx.export(
        wrapper,
        (voxels, voxel_num_points, voxel_coords, bev_index_map),
        output_path,
        input_names=["voxels", "voxel_num_points", "voxel_coords", "bev_index_map"],
        output_names=["batch_box_preds", "batch_cls_preds"],
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=False,
        dynamic_axes=dynamic_axes,
    )
    fix_dir_reshape_dim(output_path, num_dir_bins=cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS)
    print("base ONNX -> %s" % output_path, flush=True)


def fix_dir_reshape_dim(output_path, num_dir_bins):
    """把 dir argmax 上游 Reshape 目标里的 -1 补成 NUM_DIR_BINS。

    否则 onnx shape inference 推不出 ArgMax 输入的通道数（head 空间维全未知），
    KnowledgeArgMax2ToCompare 无法静态证明 axis 维 == 2。此处直接用配置值补全。
    """
    import onnx as _onnx
    from onnx import numpy_helper as _nh

    model = _onnx.load(output_path)
    g = model.graph
    nodes = list(g.node)
    prods = {o: n for n in nodes for o in n.output}

    def target_arr(name):
        for init in g.initializer:
            if init.name == name:
                return _nh.to_array(init)
        n = prods.get(name)
        if n is not None and n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    return _nh.to_array(a.t)
        return None

    changed = False
    for am in nodes:
        if am.op_type != "ArgMax":
            continue
        axis = int([a.i for a in am.attribute if a.name == "axis"][0])
        resh = prods.get(am.input[0])
        if resh is None or resh.op_type != "Reshape":
            continue
        tgt = target_arr(resh.input[1])
        if tgt is None or -1 not in tgt:
            continue
        n = tgt.shape[0]
        ax = axis + n if axis < 0 else axis
        if not (0 <= ax < n and tgt[ax] == -1):
            continue
        tgt[ax] = int(num_dir_bins)
        new_init = _nh.from_array(tgt, resh.input[1])
        for i, init in enumerate(g.initializer):
            if init.name == resh.input[1]:
                del g.initializer[i]
                g.initializer.insert(i, new_init)
                changed = True
                break
    if changed:
        _onnx.save(model, output_path)
        print("  fix_dir_reshape_dim: patched ArgMax reshape -1 -> %s" % num_dir_bins, flush=True)


# ============================================================
# Step 2: Graph surgery — add post-processing nodes
# ============================================================

def add_postproc(base_onnx_path, final_onnx_path, args):
    """在 base ONNX 后追加 sigmoid + topk + NMS + Gather。"""
    model = onnx.load(base_onnx_path)
    graph = model.graph
    nodes = list(graph.node)
    vi = {v.name: v for v in graph.value_info}
    for v in graph.input: vi[v.name] = v
    for v in graph.output: vi[v.name] = v

    # 从 base 输出实际名字中识别 box/cls 输出
    base_out_names = [o.name for o in graph.output]
    print("base outputs:", base_out_names, flush=True)
    if len(base_out_names) != 2:
        raise RuntimeError("期望 2 个 base 输出, 实际 %d" % len(base_out_names))
    BOX = base_out_names[0]  # (1, N, 7) f32
    CLS = base_out_names[1]  # (1, N, 3) f32

    # 确保输出 TensorProto 信息
    def ensure_vi(name, shape, dtype=TensorProto.FLOAT):
        if name not in vi:
            t = helper.make_tensor_value_info(name, dtype, list(shape))
            graph.value_info.append(t)
            vi[name] = t

    ensure_vi(CLS, [1, NUM_ANCHORS, 3])
    ensure_vi(BOX, [1, NUM_ANCHORS, 7])

    prefix = "nms_"
    new_nodes = []
    mid = {}  # intermediate results

    def add(node):
        new_nodes.append(node)
        for out in node.output:
            mid[out] = True

    # ---------- Sigmoid ----------
    # cls_sigmoid = Sigmoid(CLS)
    cls_sig = prefix + "cls_sigmoid"
    ensure_vi(cls_sig, [1, NUM_ANCHORS, 3])
    add(helper.make_node("Sigmoid", [CLS], [cls_sig], name=cls_sig))

    # ---------- ReduceMax: scores (1,N,1) ----------
    # scores_t = ReduceMax(cls_sig, axes=[2], keepdims=1)
    scores_t = prefix + "scores_trans"
    ensure_vi(scores_t, [1, NUM_ANCHORS, 1])
    add(helper.make_node("ReduceMax", [cls_sig], [scores_t],
                         axes=[2], keepdims=1, name=scores_t))

    # ---------- ArgMax: labels (1,N,1) ----------
    labels_t = prefix + "labels_trans"
    ensure_vi(labels_t, [1, NUM_ANCHORS, 1], TensorProto.INT64)
    add(helper.make_node("ArgMax", [cls_sig], [labels_t],
                         axis=2, keepdims=1, name=labels_t))

    # ---------- Squeeze (1,N,1) -> (N,) ----------
    scores_1d = prefix + "scores_1d"
    ensure_vi(scores_1d, [NUM_ANCHORS])
    add(make_const(prefix + "axes_02", np.array([0, 2], dtype=np.int64)))
    add(helper.make_node("Squeeze", [scores_t, prefix + "axes_02"], [scores_1d], name=scores_1d))

    labels_1d = prefix + "labels_1d"
    ensure_vi(labels_1d, [NUM_ANCHORS], TensorProto.INT64)
    add(helper.make_node("Squeeze", [labels_t, prefix + "axes_02"], [labels_1d], name=labels_1d))

    # ---------- Cast labels to int64 ----------
    labels_i64 = prefix + "labels_i64"
    ensure_vi(labels_i64, [NUM_ANCHORS], TensorProto.INT64)
    add(helper.make_node("Cast", [labels_1d], [labels_i64],
                         to=TensorProto.INT64, name=labels_i64))

    # ---------- Pre-TopK: 把 NMS 输入框数压到 <=50000 ----------
    # aclnnNonMaxSuppression 硬限制：每 batch 框数 <= 50000（PointPillar 321408 直接超限 → 输出垃圾）。
    # 先按分数 TopK 选 top-k（对齐 Python 侧 score-mask + topk(NMS_PRE_MAXSIZE)），再进 NMS。
    topk_k = getattr(args, "pre_topk", 4096)
    topk_vals = prefix + "topk_vals"
    topk_idx = prefix + "topk_idx"
    ensure_vi(topk_vals, [topk_k])
    ensure_vi(topk_idx, [topk_k], TensorProto.INT64)
    add(make_const(prefix + "topk_k", np.array([topk_k], dtype=np.int64)))
    add(helper.make_node("TopK", [scores_1d, prefix + "topk_k"],
                         [topk_vals, topk_idx],
                         axis=0, largest=1, sorted=1, name=topk_vals))

    labels_top = prefix + "labels_top"
    ensure_vi(labels_top, [topk_k], TensorProto.INT64)
    add(helper.make_node("Gather", [labels_i64, topk_idx], [labels_top],
                         axis=0, name=labels_top))

    # ---------- Unsqueeze top scores to (1,1,K) for NMS ----------
    scores_nms_mid = prefix + "scores_nms_mid"
    ensure_vi(scores_nms_mid, [1, topk_k])
    add(make_const(prefix + "ax0", np.array([0], dtype=np.int64)))
    add(helper.make_node("Unsqueeze", [topk_vals, prefix + "ax0"], [scores_nms_mid], name=scores_nms_mid))

    scores_nms = prefix + "scores_nms"
    ensure_vi(scores_nms, [1, 1, topk_k])
    add(helper.make_node("Unsqueeze", [scores_nms_mid, prefix + "ax0"], [scores_nms], name=scores_nms))

    # ---------- Convert 3D box [x,y,z,dx,dy,dz,r] to 2D BEV [x1,y1,x2,y2] ----------
    # Slice box_preds
    # x = BOX[:,:,0:1]; y = BOX[:,:,1:2]; dx = BOX[:,:,3:4]; dy = BOX[:,:,4:5]

    def slice_last_dim(inp, start, end, name_prefix):
        out = prefix + name_prefix
        ensure_vi(out, [1, NUM_ANCHORS, 1])
        c_starts = prefix + name_prefix + "_starts"
        c_ends = prefix + name_prefix + "_ends"
        c_axes = prefix + name_prefix + "_axes"
        ensure_vi(c_starts, [3], TensorProto.INT64)
        ensure_vi(c_ends, [3], TensorProto.INT64)
        ensure_vi(c_axes, [3], TensorProto.INT64)
        add(make_const(c_starts, np.array([0, 0, start], dtype=np.int64)))
        add(make_const(c_ends, np.array([1, NUM_ANCHORS, end], dtype=np.int64)))
        add(make_const(c_axes, np.array([0, 1, 2], dtype=np.int64)))
        add(helper.make_node("Slice", [inp, c_starts, c_ends, c_axes], [out], name=out))
        return out

    box_x = slice_last_dim(BOX, 0, 1, "box_x")      # (1,N,1)
    box_y = slice_last_dim(BOX, 1, 2, "box_y")
    box_dx = slice_last_dim(BOX, 3, 4, "box_dx")
    box_dy = slice_last_dim(BOX, 4, 5, "box_dy")

    # half = [0.5]
    half = prefix + "half"
    ensure_vi(half, [1], TensorProto.FLOAT)
    add(make_const(half, np.array([0.5], dtype=np.float32)))

    # dx_half = dx * half, dy_half = dy * half
    dx_half = prefix + "dx_half"
    ensure_vi(dx_half, [1, NUM_ANCHORS, 1])
    add(helper.make_node("Mul", [box_dx, half], [dx_half], name=dx_half))

    dy_half = prefix + "dy_half"
    ensure_vi(dy_half, [1, NUM_ANCHORS, 1])
    add(helper.make_node("Mul", [box_dy, half], [dy_half], name=dy_half))

    # x1 = x - dx_half, x2 = x + dx_half, y1 = y - dy_half, y2 = y + dy_half
    x1 = prefix + "x1"
    ensure_vi(x1, [1, NUM_ANCHORS, 1])
    add(helper.make_node("Sub", [box_x, dx_half], [x1], name=x1))

    x2 = prefix + "x2"
    ensure_vi(x2, [1, NUM_ANCHORS, 1])
    add(helper.make_node("Add", [box_x, dx_half], [x2], name=x2))

    y1 = prefix + "y1"
    ensure_vi(y1, [1, NUM_ANCHORS, 1])
    add(helper.make_node("Sub", [box_y, dy_half], [y1], name=y1))

    y2 = prefix + "y2"
    ensure_vi(y2, [1, NUM_ANCHORS, 1])
    add(helper.make_node("Add", [box_y, dy_half], [y2], name=y2))

    # Concat(x1, y1, x2, y2) -> boxes_2d (1,N,4)
    boxes_2d = prefix + "boxes_2d"
    ensure_vi(boxes_2d, [1, NUM_ANCHORS, 4])
    add(helper.make_node("Concat", [x1, y1, x2, y2], [boxes_2d],
                         axis=2, name=boxes_2d))

    # ---------- 用 top-k 的 BEV 框进 NMS ----------
    boxes_2d_top = prefix + "boxes_2d_top"
    ensure_vi(boxes_2d_top, [1, topk_k, 4])
    add(helper.make_node("Gather", [boxes_2d, topk_idx], [boxes_2d_top],
                         axis=1, name=boxes_2d_top))

    # ---------- NonMaxSuppression ----------
    max_out = prefix + "max_out"
    ensure_vi(max_out, [1], TensorProto.INT64)
    add(make_const(max_out, np.array([args.max_det], dtype=np.int64)))

    iou_thr = prefix + "iou_thr"
    ensure_vi(iou_thr, [1], TensorProto.FLOAT)
    add(make_const(iou_thr, np.array([args.iou_thresh], dtype=np.float32)))

    score_thr = prefix + "score_thr"
    ensure_vi(score_thr, [1], TensorProto.FLOAT)
    add(make_const(score_thr, np.array([args.score_thresh], dtype=np.float32)))

    # NMS output: selected_indices (num_detected, 3) int64
    sel = prefix + "selected_indices"
    out_shape = [None, 3]  # dynamic
    ensure_vi(sel, out_shape, TensorProto.INT64)
    add(helper.make_node("NonMaxSuppression",
                         [boxes_2d_top, scores_nms, max_out, iou_thr, score_thr],
                         [sel], name=sel))

    # ---------- Squeeze batch & class dims: (num, 3) -> (num,) ----------
    # sel: [batch_id, class_id, box_id] -> take column 2 (box_id)（相对 top-k）
    box_ids = prefix + "box_ids"
    ensure_vi(box_ids, [None, 1], TensorProto.INT64)
    c_starts = prefix + "sel_starts"
    c_ends = prefix + "sel_ends"
    c_axes = prefix + "sel_axes"
    ensure_vi(c_starts, [2], TensorProto.INT64)
    ensure_vi(c_ends, [2], TensorProto.INT64)
    ensure_vi(c_axes, [2], TensorProto.INT64)
    add(make_const(c_starts, np.array([0, 2], dtype=np.int64)))
    add(make_const(c_ends, np.array([args.max_det, 3], dtype=np.int64)))
    add(make_const(c_axes, np.array([0, 1], dtype=np.int64)))
    add(helper.make_node("Slice", [sel, c_starts, c_ends, c_axes], [box_ids], name=box_ids))

    box_ids_1d = prefix + "box_ids_1d"
    ensure_vi(box_ids_1d, [None], TensorProto.INT64)
    add(make_const(prefix + "axes_1", np.array([1], dtype=np.int64)))
    add(helper.make_node("Squeeze", [box_ids, prefix + "axes_1"], [box_ids_1d], name=box_ids_1d))

    # ---------- 把 top-k 索引映射回原始 anchor 索引 ----------
    orig_ids_1d = prefix + "orig_ids_1d"
    ensure_vi(orig_ids_1d, [None], TensorProto.INT64)
    add(helper.make_node("Gather", [topk_idx, box_ids_1d], [orig_ids_1d],
                         axis=0, name=orig_ids_1d))

    # ---------- Gather final boxes, scores, labels ----------
    # final_boxes = Gather(BOX, orig_ids_1d, axis=1)
    final_boxes = prefix + "final_boxes"
    ensure_vi(final_boxes, [1, None, 7])
    add(helper.make_node("Gather", [BOX, orig_ids_1d], [final_boxes],
                         axis=1, name=final_boxes))

    final_scores = prefix + "final_scores"
    ensure_vi(final_scores, [None])
    add(helper.make_node("Gather", [topk_vals, box_ids_1d], [final_scores],
                         axis=0, name=final_scores))

    final_labels = prefix + "final_labels"
    ensure_vi(final_labels, [None], TensorProto.INT64)
    add(helper.make_node("Gather", [labels_top, box_ids_1d], [final_labels],
                         axis=0, name=final_labels))

    final_count = prefix + "final_count"
    ensure_vi(final_count, [1], TensorProto.INT64)
    add(helper.make_node("Size", [box_ids_1d], [final_count], name=final_count))

    # ---------- Build new graph ----------
    new_outputs = [
        helper.make_tensor_value_info(final_boxes, TensorProto.FLOAT, [1, None, 7]),
        helper.make_tensor_value_info(final_scores, TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info(final_labels, TensorProto.INT64, [None]),
        helper.make_tensor_value_info(final_count, TensorProto.INT64, [1]),
    ]
    for old_out in list(graph.output):
        graph.output.remove(old_out)
    for out_info in new_outputs:
        graph.output.extend([out_info])

    all_nodes = list(nodes) + new_nodes
    # auto_optimizer 要求节点名与张量名全局唯一：把与自身输出张量同名的节点重命名
    for _n in all_nodes:
        if _n.name and _n.name in _n.output:
            _n.name = _n.name + "_op"
    graph.ClearField("node")
    graph.node.extend(all_nodes)

    # Clean isolated value_infos (keep only referenced)
    # (optional optimization — skip for now)

    onnx.checker.check_model(model)
    onnx.save(model, final_onnx_path)
    print("postproc ONNX -> %s" % final_onnx_path, flush=True)

    # Print structure summary
    print("\n输入:", [i.name + str(i.type.tensor_type.shape.dim) for i in graph.input], flush=True)
    print("输出:", [o.name + " " + str([d.dim_value for d in o.type.tensor_type.shape.dim if d.dim_value > 0]) for o in graph.output], flush=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export PointPillars ONNX with NMS post-processing")
    parser.add_argument("--config", default=str(ROOT / "data/config.yaml"))
    parser.add_argument("--ckpt", default=str(ROOT / "weights/pointpillar_7728.pth"))
    parser.add_argument("--sample-idx", default="000008")
    parser.add_argument("--output", default=str(ROOT / "weights/pointpillar_nms.onnx"))
    parser.add_argument("--base-output", default=str(ROOT / "weights/pointpillar_nms_base.onnx"))
    parser.add_argument("--skip-export", action="store_true", help="复用已有 base ONNX，只做图手术")
    parser.add_argument("--dynamic", action="store_true", help="导出时把 voxels 的 M 维度标为动态")
    parser.add_argument("--topk-only", action="store_true",
                        help="只做 sigmoid-等价的 ReduceMax + TopK + Gather（不叠加 NMS），"
                             "输出 top-K box/cls，后处理（NMS）留 Python（P2 方案，D2H 13MB->~160KB）")
    parser.add_argument("--topk-k", type=int, default=4096, help="topk-only 的 K（须 <=50000）")
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--iou-thresh", type=float, default=0.01)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset (推荐 16/17)")
    args = parser.parse_args()

    base_path = Path(args.base_output)
    final_path = Path(args.output)

    if not args.skip_export:
        print("===== Step 1: 导出 base ONNX =====", flush=True)
        export_base_onnx(base_path, args)

    if getattr(args, "topk_only", False):
        print("===== Step 2: 图手术 - topk-only（无 NMS）=====", flush=True)
        add_topk_only(base_path, final_path, args)
    else:
        print("===== Step 2: 图手术 - 追加后处理 =====", flush=True)
        add_postproc(base_path, final_path, args)

    print("\n完成! 后续 ATC 转换:", flush=True)
    print("  atc --model=%s --framework=5 --soc_version=Ascend310P3 --output=weights/pointpillar_nms --input_format=ND" % final_path, flush=True)


def add_topk_only(base_onnx_path, final_onnx_path, args):
    """P2：只加 sigmoid-等价的 ReduceMax + TopK + Gather（**无 NMS**）。

    图内: ReduceMax(cls,axis=2)->(1,N)；Squeeze->(N,)；TopK(k)->vals/idx；
    Gather(boxes, idx, axis=1)->(1,K,7)；Gather(cls, idx, axis=1)->(1,K,3)。
    输出 top-K box + raw cls（D2H ~160KB vs 13MB）；类别标签与 NMS 留在 CPU。

    - 不叠加 ArgMax（310P ArgMaxD ~19ms 慢）；TopK 本身 ~0.4ms（实测）。
    - K 必须 <=50000（aclnnNonMaxSuppression 硬限制，这里无 NMS 但仍保守）。
    """
    import onnx as _onnx
    from onnx import helper as _h, TensorProto as _T

    model = _onnx.load(base_onnx_path)
    g = model.graph
    nodes = list(g.node)
    base_out = [o.name for o in g.output]
    if len(base_out) != 2:
        raise RuntimeError("期望 2 个 base 输出, 实际 %d" % len(base_out))
    box, cls = base_out[0], base_out[1]
    k = getattr(args, "topk_k", 4096)

    def add(n):
        nodes.append(n)

    scores = "topk_scores"
    add(_h.make_node("ReduceMax", [cls], [scores], axes=[2], keepdims=0, name="topk_reducemax"))
    scores1 = "topk_scores_1d"
    ax = _h.make_tensor("topk_sq_ax", _T.INT64, [1], np.array([0], np.int64))
    g.initializer.append(ax)
    add(_h.make_node("Squeeze", [scores, "topk_sq_ax"], [scores1], name="topk_squeeze"))
    tv, ti = "topk_vals", "topk_idx"
    kk = _h.make_tensor("topk_k", _T.INT64, [1], np.array([k], np.int64))
    g.initializer.append(kk)
    add(_h.make_node("TopK", [scores1, "topk_k"], [tv, ti], axis=0, largest=1, sorted=1, name="topk_topk"))
    tb, tc = "topk_boxes", "topk_cls"
    add(_h.make_node("Gather", [box, ti], [tb], axis=1, name="topk_gather_box"))
    add(_h.make_node("Gather", [cls, ti], [tc], axis=1, name="topk_gather_cls"))
    del g.output[:]
    g.output.extend([
        _h.make_tensor_value_info(tb, _T.FLOAT, [1, k, 7]),
        _h.make_tensor_value_info(tc, _T.FLOAT, [1, k, 3]),
    ])
    g.ClearField("node")
    g.node.extend(nodes)
    _onnx.checker.check_model(model)
    _onnx.save(model, final_onnx_path)
    print("topk-only ONNX -> %s  (K=%d)" % (final_onnx_path, k), flush=True)


if __name__ == "__main__":
    main()