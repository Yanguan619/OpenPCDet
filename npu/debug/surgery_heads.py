"""ONNX 图手术合集（每项可独立开关，输出单独文件便于对比）。

A. merge_head_convs: 三个 1x1 head conv（/Conv_16=18ch, /Conv_17=42ch, /Conv_18=12ch）
   共用同一 384ch 输入，合并为一个 384→72 卷积 + 3 个 Slice（数学恒等，省 2 次重复读）。
B. bypass_class_gather: /Gather_5 与 /Gather_6 都是 axis=2 从 /Concat_6 取 index 6
   （Concat_6 第 7 个输入就是 /Add_5_output_0），等价于 Squeeze(/Add_5_output_0)。
   省掉两个 321408 长度 gather（gatherv2_110 单 kernel 915us）。
C. convtranspose2_pixelshuffle: 仅把 4x4/s4 的 /ConvTranspose_2 重写为
   Conv1x1(256→2048) + DepthToSpace(blocksize=4)。注意 CANN 的 mode 语义与规范相反，
   必须用 mode='CRD' 才是规范 DCR（已用 310P 探针 OM 实测）。
D. topk_k_1024: TopK 的 K 常量 4096→1024（topk_topk 输入 'topk_k'）。

用法: python3 npu/debug/surgery_heads.py --out om_out/pp_merged.onnx --do ABC
"""
import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_output_edge(g, old, new):
    """把图上所有引用 old 输出名的节点输入改到 new。"""
    for n in g.node:
        for i in range(len(n.input)):
            if n.input[i] == old:
                n.input[i] = new


def merge_head_convs(model):
    """A: 合并 /Conv_16/17/18 → 单 Conv 384→72 + Slice 回三份。"""
    g = model.graph
    init_map = {i.name: i for i in g.initializer}
    node_by_name = {n.name: n for n in g.node}
    convs = [node_by_name[nm] for nm in ('/Conv_16', '/Conv_17', '/Conv_18')]
    ins = [n.input[1] for n in convs]
    biases = [n.input[2] for n in convs]
    Ws = [numpy_helper.to_array(init_map[i]) for i in ins]     # (18,384,1,1) (42,384,1,1) (12,384,1,1)
    bs = [numpy_helper.to_array(init_map[b]) for b in biases]
    assert all(w.shape[1] == 384 for w in Ws)
    Wcat = np.concatenate(Ws, axis=0).astype(np.float32)       # (72,384,1,1)
    bcat = np.concatenate(bs, axis=0).astype(np.float32)       # (72,)
    w_name, b_name = '/MergedHead_w', '/MergedHead_b'
    g.initializer.append(numpy_helper.from_array(Wcat, w_name))
    g.initializer.append(numpy_helper.from_array(bcat, b_name))

    out = '/MergedHead_out'
    conv = helper.make_node('Conv', ['/Concat_5_output_0', w_name, b_name], [out],
                            name='/MergedHead', kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    # 插到 /Concat_5 之后（/Concat_5 是它的输入生产者），保证拓扑序
    concat_idx = list(g.node).index(node_by_name['/Concat_5'])
    g.node.insert(concat_idx + 1, conv)
    insert_at = concat_idx + 2

    # 三个原始 conv 的输出名，改成从合并输出 Slice
    starts, ends = (0, 18, 60), (18, 60, 72)
    for i, (start, end, orig) in enumerate(zip(starts, ends, ('/Conv_16_output_0', '/Conv_17_output_0', '/Conv_18_output_0'))):
        s_name = '/MergedSlice%d_start' % i
        e_name = '/MergedSlice%d_end' % i
        a_name = '/MergedSlice%d_axes' % i
        g.initializer.append(numpy_helper.from_array(np.array([start], dtype=np.int64), s_name))
        g.initializer.append(numpy_helper.from_array(np.array([end], dtype=np.int64), e_name))
        g.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), a_name))  # 通道维
        sl = helper.make_node('Slice', [out, s_name, e_name, a_name], [orig],
                              name='/MergedSlice%d' % i)
        g.node.insert(insert_at, sl)
        insert_at += 1

    # 删除原始三个 conv
    remove = {c.name for c in convs}
    nodes = [n for n in g.node if n.name not in remove]
    del g.node[:]
    g.node.extend(nodes)
    print('A: merged /Conv_16/17/18 -> 384x72 conv + slices')
    return model


def bypass_class_gather(model):
    """B: /Gather_5、/Gather_6（axis=2, index=6 on /Concat_6）→ Squeeze(/Add_5_output_0)。"""
    g = model.graph
    node_by_name = {n.name: n for n in g.node}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n.name)

    for nm in ('/Gather_5', '/Gather_6'):
        n = node_by_name[nm]
        if n.op_type != 'Gather':
            print('skip B: %s op=%s' % (nm, n.op_type))
            continue
        idx = [i for i in n.input if i.endswith('Constant_18_output_0')]
        axis = [a.i for a in n.attribute if a.name == 'axis']
        if not idx or not axis or axis[0] != 2:
            print('skip B: %s unexpected attrs' % nm)
            continue
        # 替换为 Squeeze，保持输出名不变；插到 /Add_5 之后满足拓扑序
        # Squeeze 的 axes 以 input（tensor）形式传入，兼容旧 opset
        axes_name = nm[1:].replace('/', '_') + '_squeeze_axes'
        g.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), axes_name))
        sq = helper.make_node('Squeeze', ['/Add_5_output_0', axes_name], [n.output[0]],
                              name=nm + '_squeeze')
        g.node.remove(n)
        add_idx = list(g.node).index(node_by_name['/Add_5'])
        g.node.insert(add_idx + 1, sq)
        print('B: %s -> Squeeze(/Add_5_output_0) axes=[2]' % nm)
    return model


def convtranspose2_pixelshuffle(model):
    """C: 仅重写 /ConvTranspose_2 (4x4/s4)。"""
    g = model.graph
    init_map = {i.name: i for i in g.initializer}
    n = next(nn for nn in g.node if nn.name == '/ConvTranspose_2')
    strides = [list(a.ints) for a in n.attribute if a.name == 'strides'][0]
    pads = [list(a.ints) for a in n.attribute if a.name == 'pads'][0]
    if any(pads):
        raise RuntimeError('ConvTranspose_2 has pads!')
    s = strides[0]
    W = numpy_helper.to_array(init_map[n.input[1]])           # (256,128,4,4) = [C_in, C_out, s, s]
    c_in, c_out = W.shape[0], W.shape[1]
    W2 = np.transpose(W, (1, 2, 3, 0)).reshape(c_out * s * s, c_in)[:, :, None, None].astype(np.float32)
    w2_name = n.name + '_ps_w'
    g.initializer.append(numpy_helper.from_array(W2, w2_name))
    conv_out = n.output[0] + '_ps_conv'
    conv = helper.make_node('Conv', [n.input[0], w2_name], [conv_out],
                            name=n.name + '_ps_conv', kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    dts = helper.make_node('DepthToSpace', [conv_out], [n.output[0]],
                           name=n.name + '_ps_dts', blocksize=s, mode='CRD')
    idx = list(g.node).index(n)
    del g.node[idx]
    g.node.insert(idx, dts)
    g.node.insert(idx, conv)
    print('C: /ConvTranspose_2 -> Conv1x1(256->2048) + DepthToSpace(CRD, b=4)')
    return model


def topk_k_1024(model):
    """D: TopK K 常量 4096→1024。"""
    g = model.graph
    for i in g.initializer:
        if i.name == 'topk_k':
            arr = numpy_helper.to_array(i)
            arr[:] = 1024
            i.CopyFrom(numpy_helper.from_array(arr, i.name))
            print('D: topk_k -> 1024')
            return model
    raise RuntimeError('topk_k initializer not found')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=str(ROOT / 'weights/pointpillar_nms_base_v2_dynamic_topk.onnx'))
    ap.add_argument('--out', default=str(ROOT / 'om_out/pp_surgery.onnx'))
    ap.add_argument('--do', default='ABCD', help='要做的手术，如 ABC / ABCD / C')
    args = ap.parse_args()

    model = onnx.load(args.inp)
    if 'A' in args.do:
        merge_head_convs(model)
    if 'B' in args.do:
        bypass_class_gather(model)
    if 'C' in args.do:
        convtranspose2_pixelshuffle(model)
    if 'D' in args.do:
        topk_k_1024(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
