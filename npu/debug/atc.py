#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointPillars 一键转换: PyTorch -> ONNX -> 简化 -> 图手术 -> OM (Ascend310P3).

用法:
  python npu/atc_convert.py                                    # 全流程(动态 shape, mixed_float16)
  python npu/atc_convert.py --static                          # 静态 shape(force_fp16, 15.19ms 最优)
  python npu/atc_convert.py /tmp/opencode/out                  # 指定 OM 输出前缀
  python npu/atc_convert.py --skip-export                      # 跳过 ONNX 导出
  python npu/atc_convert.py --ckpt weights/other.pth           # 指定 checkpoint
  python npu/atc_convert.py --sample-idx 000008                # 指定样本
  python npu/atc_convert.py --onnx /path/to/model.onnx        # 指定已有 ONNX(跳过导出)
  python npu/atc_convert.py --fp32                             # force_fp32(精度最高)
  python npu/atc_convert.py --mixlist path/to/mix.json        # 自定义混合精度名单
  python npu/atc_convert.py --no-graphopt                    # 跳过图手术(仅 onnxsim+onnxslim)
  python npu/atc_convert.py --no-onnxslim                    # 跳过 onnxslim(仅 onnxsim)

Pipeline 步骤:
  [0] ONNX 导出 (export_onnx.py, 动态 shape)
  [1] ONNX 简化 (auto_optimizer simplify = onnxsim)
  [2] ONNX 深度简化 (auto_optimizer slim = onnxslim)
  [3] 图手术: auto_optimizer opt (Cast 合并 + Conv 合并)
  [4] ATC 转换
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from shlex import join as sh_join
except ImportError:  # py < 3.8
    def sh_join(argv):
        import shlex
        return " ".join(shlex.quote(a) for a in argv)


ROOT = Path(__file__).resolve().parent.parent  # 脚本在 npu/ 下

DEFAULT_MIXLIST = {
    "white-list": {
        "to-add": [
            "StridedSliceD",
            "ReduceSumD",
            "ConcatD",
            "GatherV2",
            "ConfusionTransposeD",
            "AutomaticBufferFusionOp",
        ]
    }
}


def run(cmd, **kw):
    """运行外部命令,失败立即抛出异常."""
    print("  $", sh_join(cmd))
    subprocess.run(cmd, check=True, **kw)


def onnx_input_dim(onnx_path: Path, input_name: str) -> int:
    """从简化后 ONNX 读取指定输入的首维(固定值)."""
    import onnx
    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.name == input_name:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            return int(dims[0])
    raise RuntimeError(f"ONNX 输入 '{input_name}' 未找到")


def write_mixlist(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def step0_export_onnx(ckpt: Path, sample_idx: str, output: Path, skip: bool):
    if skip:
        print("[0/4] 跳过 ONNX 导出 (--skip-export)")
        return
    if output.exists():
        print(f"[0/4] ONNX 已存在,跳过导出: {output}")
        return
    print(f"[0/4] ONNX 导出(动态 shape): {ckpt} -> {output} (sample={sample_idx})")
    run([
        sys.executable, str(ROOT / "examples" / "export_onnx.py"),
        "--ckpt", str(ckpt),
        "--sample-idx", sample_idx,
        "--output", str(output),
    ])


def step1_simplify(onnx_path: Path, simplified: Path):
    print(f"[1/5] ONNX 简化 (auto_optimizer simplify / onnxsim): {onnx_path} -> {simplified}")
    if simplified.exists():
        print(f"  简化结果已存在,跳过: {simplified}")
        return
    run(["auto_optimizer", "simplify", str(onnx_path), str(simplified), "--no-large-tensor"])


def step1b_onnxslim(simplified: Path, slimmed: Path):
    print(f"[2/5] ONNX 深度简化 (auto_optimizer slim / onnxslim): {simplified} -> {slimmed}")
    if slimmed.exists():
        print(f"  onnxslim 结果已存在,跳过: {slimmed}")
        return
    run(["auto_optimizer", "slim", str(simplified), str(slimmed)])


def step2_graph_opt(slimmed: Path, final_onnx: Path, no_graphopt: bool):
    if no_graphopt:
        print("[3/5] 跳过图手术 (--no-graphopt)")
        return
    print("[3/5] 图手术: auto_optimizer (Cast 合并 + Conv 合并)")
    if final_onnx.exists():
        print(f"  图手术结果已存在,跳过: {final_onnx}")
        return
    run([
        "auto_optimizer", "opt",
        "-k", "KnowledgeMergeCasts,KnowledgeMergeConvs",
        str(slimmed), str(final_onnx),
    ])


def build_atc_cmd(final_onnx: Path, out_prefix: Path, mode: str,
                  static: bool, g: int, mixlist: Path) -> list:
    base = [
        "/usr/local/Ascend/cann/bin/atc",
        f"--model={final_onnx}",
        "--framework=5",
        "--soc_version=Ascend310P3",
        f"--output={out_prefix}",
        "--input_format=ND",
        "--log=error",
    ]
    if static:
        m = 3941
        shape = f"voxels:{m},32,4;voxel_num_points:{m};voxel_coords:{m},4;bev_index_map:{g}"
        base.append(f"--input_shape={shape}")
        if mode == "fp32":
            base.append("--precision_mode=force_fp32")
        else:  # fp16
            base += [
                "--precision_mode=force_fp16",
                "--op_select_implmode=high_performance",
                "--buffer_optimize=l1_optimize",
                "--enable_single_stream=true",
                f"--fusion_switch_file={ROOT/'tools'/'fusion_all_on.cfg'}",
            ]
    else:
        shape = f"voxels:1~9000,32,4;voxel_num_points:1~9000;voxel_coords:1~9000,4;bev_index_map:{g}"
        base.append(f"--input_shape={shape}")
        if mode == "fp32":
            base.append("--precision_mode=force_fp32")
        else:
            base += [
                "--precision_mode_v2=mixed_float16",
                f"--modify_mixlist={mixlist}",
            ]
    return base


def main():
    p = argparse.ArgumentParser(
        description="PointPillars PyTorch->ONNX->简化->图手术->OM 一键转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("out_prefix", nargs="?", default=None, help="OM 输出前缀")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--ckpt", default=str(ROOT / "weights" / "pointpillar_7728.pth"))
    p.add_argument("--sample-idx", default="000008")
    p.add_argument("--onnx", default=None)
    p.add_argument("--fp32", action="store_true", help="force_fp32(精度最高)")
    p.add_argument("--static", action="store_true", help="静态 shape M=3941, force_fp16+全融合")
    p.add_argument("--no-graphopt", action="store_true")
    p.add_argument("--no-onnxslim", action="store_true", help="跳过 onnxslim(仅 onnxsim)")
    p.add_argument("--mixlist", default=None)
    args = p.parse_args()

    static = args.static
    mode = "fp32" if args.fp32 else ("fp16" if static else "mixed")

    if static:
        out_prefix = Path(args.out_prefix or (ROOT / "weights" / f"pointpillar_{mode}_static"))
    else:
        out_prefix = Path(args.out_prefix or (ROOT / "weights" / f"pointpillar_{mode}_dynamic"))

    onnx_path = Path(args.onnx) if args.onnx else (ROOT / "weights" / "pointpillar_demo.onnx")
    simplified = onnx_path.with_name(onnx_path.stem + "_sim.onnx")
    slimmed = (simplified if args.no_onnxslim
               else onnx_path.with_name(onnx_path.stem + "_slim.onnx"))
    final_onnx = (onnx_path.with_name(onnx_path.stem + "_opt.onnx")
                  if not args.no_graphopt else slimmed)

    # mixlist: 默认写到临时位置;若用户指定则使用用户指定
    if args.mixlist:
        mixlist = Path(args.mixlist)
    else:
        mixlist = Path("/tmp/mix_optimized.json")
        write_mixlist(mixlist, DEFAULT_MIXLIST)

    ckpt = Path(args.ckpt)

    # ---- Step 0: ONNX 导出 ----
    step0_export_onnx(ckpt, args.sample_idx, onnx_path, args.skip_export)
    if not onnx_path.exists():
        print(f"ERROR: ONNX 文件不存在: {onnx_path}", file=sys.stderr)
        print("       请先运行: python npu/export_onnx.py", file=sys.stderr)
        sys.exit(1)

    # ---- Step 1: ONNX 简化 (onnxsim) ----
    step1_simplify(onnx_path, simplified)

    # 从 ONNX 推断 G (bev_index_map 长度,固定值)
    g = onnx_input_dim(simplified, "bev_index_map")
    print(f"  bev_index_map G={g}")

    # ---- Step 2: ONNX 深度简化 (onnxslim) ----
    step1b_onnxslim(simplified, slimmed)

    # ---- Step 3: 图手术 ----
    step2_graph_opt(slimmed, final_onnx, args.no_graphopt)
    print(f"  最终 ONNX: {final_onnx}")

    # ---- Step 4: ATC 转换 ----
    atc_cmd = build_atc_cmd(final_onnx, out_prefix, mode, static, g, mixlist)
    if static:
        print(f"[4/5] ATC 静态 shape 转换 (mode={mode}): {final_onnx} -> {out_prefix}")
        print(f"  M=3941 (静态), G={g}")
    else:
        print(f"[4/5] ATC 动态 shape 转换 (mode={mode}): {final_onnx} -> {out_prefix}")
    run(atc_cmd)

    # 查找生成的 OM 文件
    om_files = sorted(out_prefix.parent.glob(out_prefix.name + "*.om"))
    om_file = om_files[0] if om_files else None

    print("=" * 44)
    if static:
        print("转换完成(静态 shape M=3941, mode=force_fp16 + 全融合)")
    else:
        print(f"转换完成(动态 shape M=1~9000, mode={mode})")
    print(f"  ONNX (final):     {final_onnx}")
    print(f"  OM:                {om_file}")
    print("=" * 44)
    print("验证:")
    if static:
        print(f"  python npu/om_ref_demo.py --bin data/kitti/training/velodyne/000008.bin --om {om_file} --score-thresh 0.3")
    else:
        print(f"  python tools/test_om.py --om {om_file} --max-samples 100 --detail")


if __name__ == "__main__":
    main()
