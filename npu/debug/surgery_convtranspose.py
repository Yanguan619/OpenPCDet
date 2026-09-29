"""ONNX 图手术：把 kernel==stride、pad=0 的 ConvTranspose 重写为 Conv(1x1) + DepthToSpace（pixel-shuffle 等价）。

数学依据：ConvTranspose(stride=s, kernel=s, pad=0) 输出
    out[co, ih*s+kh, iw*s+kw] = Σ_ci in[ci, ih, iw] * W[ci, co, kh, kw]
恰好等于
    Y = Conv1x1(in, W2)  # W2[co*s²+kh*s+kw, ci] = W[ci, co, kh, kw]，输出 C_out*s² 通道
    out = DepthToSpace(Y, blocksize=s, mode='CRD')   # 见下方 ⚠️ 说明

⚠️ 实现注意：ONNX 规范里 DCR 才是「depth-column-row」（out[co, ih*s+kh, iw*s+kw] =
Y[co*s²+kh*s+kw, ih, iw]）；但 CANN 与 ORT 1.15.1 的实际实现把 DCR/CRD 对调了
（已用探针 OM 在 310P 实测：mode=DCR→规范CRD、mode=CRD→规范DCR）。
本重写需要规范 DCR 语义 → ONNX 里必须写 mode='CRD' 才能在 CANN/ORT 上得到规范 DCR。


适用本模型三个 head 上采样 ConvTranspose（1x1/s1、2x2/s2、4x4/s4，均无 pad 无 bias）。
1x1/s1 的块大小 s=1 时 DepthToSpace 退化为恒等，直接省略。

用法:
    python3 npu/debug/surgery_convtranspose.py \
        --in weights/pointpillar_nms_base_v2_dynamic_topk.onnx \
        --out om_out/pp_topk_pixelshuffle.onnx
"""
import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def convtranspose_to_pixelshuffle(model, node_names=('/ConvTranspose', '/ConvTranspose_1', '/ConvTranspose_2')):
    """把指定 ConvTranspose 节点替换为 Conv(1x1)+DepthToSpace，就地修改 model 的 graph。"""
    g = model.graph
    init_map = {i.name: i for i in g.initializer}
    nodes = list(g.node)
    removals = []
    for n in nodes:
        if n.name not in node_names:
            continue
        if n.op_type != 'ConvTranspose':
            print('skip %s (op=%s)' % (n.name, n.op_type))
            continue
        strides = [list(a.ints) for a in n.attribute if a.name == 'strides'][0]
        pads = [list(a.ints) for a in n.attribute if a.name == 'pads'][0]
        if pads and any(pads):
            raise RuntimeError('%s has nonzero pads, not supported' % n.name)
        kshape = [list(a.ints) for a in n.attribute if a.name == 'kernel_shape'][0]
        if kshape[0] != kshape[1] or strides[0] != strides[1] or kshape[0] != strides[0]:
            raise RuntimeError('%s kernel!=stride (%s vs %s), not a no-overlap deconv' % (n.name, kshape, strides))
        s = strides[0]
        if len(n.input) > 2:
            raise RuntimeError('%s has bias, not supported' % n.name)

        W = numpy_helper.to_array(init_map[n.input[1]])  # [C_in, C_out, s, s]
        c_in, c_out = W.shape[0], W.shape[1]
        # W2[co*s²+kh*s+kw, ci] = W[ci, co, kh, kw]  →  weight layout [C_out*s², C_in, 1, 1]
        W2 = np.transpose(W, (1, 2, 3, 0))  # [C_out, s, s, C_in]
        W2 = W2.reshape(c_out * s * s, c_in)[:, :, None, None].astype(np.float32)
        w2_init = numpy_helper.from_array(W2, name=n.name + '_pixelshuffle_w')
        g.initializer.append(w2_init)

        conv_out = n.output[0] + '_ps_conv'
        conv_node = helper.make_node(
            'Conv', [n.input[0], w2_init.name], [conv_out],
            name=n.name + '_ps_conv', kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
        repl = [conv_node]
        if s > 1:
            dts = helper.make_node(
                'DepthToSpace', [conv_out], [n.output[0]],
                name=n.name + '_ps_dts', blocksize=s, mode='CRD')
            repl.append(dts)
        else:
            # s=1：DepthToSpace 恒等，直接让 conv 输出指向原输出名
            conv_node.output[0] = n.output[0]
        idx = nodes.index(n)
        nodes[idx:idx + 1] = repl
        removals.append(n.name)
        print('replaced %-22s k=%s s=%s W[%s]->W2[%s]' % (n.name, kshape, strides, W.shape, W2.shape))

    del g.node[:]
    g.node.extend(nodes)
    for n in list(g.node):
        if n.name in removals:
            g.node.remove(n)
    # 清理孤儿 initializer（保险起见只移除被替换节点的旧权重，仍被引用则保留）
    used = set()
    for n in g.node:
        used.update(n.input)
    for i in list(g.initializer):
        if i.name not in used:
            g.initializer.remove(i)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=str(ROOT / 'weights/pointpillar_nms_base_v2_dynamic_topk.onnx'))
    ap.add_argument('--out', default=str(ROOT / 'om_out/pp_topk_pixelshuffle.onnx'))
    args = ap.parse_args()

    model = onnx.load(args.inp)
    convtranspose_to_pixelshuffle(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
