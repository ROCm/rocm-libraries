#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_grouped_emit.py -- Python reference emitter for the
# direct grouped convolution parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectConv16cSpec / DirectConv4cSpec /
# DirectConv8cSpec / DirectConv32cSpec / DirectDepthwiseSpec /
# DirectDepthwiseColSpec / DirectDepthwiseSpatialSpec /
# DirectConvDgradSpec / DirectDepthwiseDgradSpec, builds the kernel via the
# matching build_direct_conv_* function (arch=<cfg arch>) and prints
# _native_lower(arch=<cfg arch>) to stdout so it can be byte-compared with
# the C emitter conv_direct_grouped_emit.c.
import sys

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConv8cSpec,
    DirectConv32cSpec,
    DirectDepthwiseSpec,
    DirectDepthwiseColSpec,
    DirectDepthwiseSpatialSpec,
    DirectConvDgradSpec,
    DirectDepthwiseDgradSpec,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_direct_conv_8c,
    build_direct_conv_32c,
    build_direct_depthwise,
    build_direct_depthwise_col,
    build_direct_depthwise_spatial,
    build_direct_conv_dgrad,
    build_direct_depthwise_dgrad,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _spec(idx: int):
    """Return (kind, spec, arch) for config index `idx`."""
    if idx == 0:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=4, fold_k32=True),
            "gfx950",
        )
    if idx == 1:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=True),
            "gfx950",
        )
    if idx == 2:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=4, block_groups=16), "gfx950")
    if idx == 3:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=8, block_groups=16), "gfx950")
    if idx == 4:
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=1, fold_k32=False),
            "gfx942",
        )
    if idx == 5:
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=16, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=4, block_groups=16), "gfx950")
    if idx == 6:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=8, kpg=8, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "8c",
            DirectConv8cSpec(problem=p, block_q=16, block_groups=8, double_buffer=True),
            "gfx950",
        )
    if idx == 7:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=8, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "32c",
            DirectConv32cSpec(
                problem=p, block_q=32, block_groups=4, double_buffer=True
            ),
            "gfx950",
        )
    if idx == 8:
        # groups must be divisible by block_ch = block_waves * wave_size (2 * 64 = 128)
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "depthwise",
            DirectDepthwiseSpec(problem=p, block_w=16, block_waves=2),
            "gfx950",
        )
    if idx == 9:
        # depthwise with stride=2: exercises Ho/Wo output descriptors and
        # stride-aware flush (p_flush_val % stride == 0 guard)
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "depthwise",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 10:
        # spatial layout: groups=3 (non-power-of-two, exercises partial wave)
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=2),
            "gfx950",
        )
    if idx == 11:
        # spatial layout with stride=2: exercises Ho/Wo + spatial thread mapping
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=1),
            "gfx950",
        )
    if idx == 12:
        # column-streamed depthwise, stride=1 fp16, both tile guards elided
        # (groups % block_ch == 0 and Wo % block_w == 0): addr() must emit a
        # bare mul with no select at all.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="fp16"),
            "gfx950",
        )
    if idx == 13:
        # col stride=2 bf16 with BOTH guards live (groups=70 % 64, Wo=5 % 4)
        p = DirectConvProblem(
            N=1, H=9, W=9, groups=70, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=1, dtype="bf16"),
            "gfx950",
        )
    if idx == 14:
        # col stride=3 fp32: exercises the (y - r) % stride tap pruning and the
        # f32 load/store forms; ch guard elided, w guard live.
        p = DirectConvProblem(
            N=1, H=16, W=16, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=3
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=1, dtype="fp32"),
            "gfx950",
        )
    if idx == 15:
        # col with a large filter (31x31): the regime the variant exists for --
        # KW rides the runtime loop so only KH weights are live.
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=64, cpg=1, kpg=1, KH=31, KW=31, PAD=15, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=1, dtype="fp16"),
            "gfx950",
        )
    if idx == 16:
        # col 1x1 / PAD=0 degenerate with a non-power-of-two group count
        p = DirectConvProblem(
            N=2, H=6, W=6, groups=3, cpg=1, kpg=1, KH=1, KW=1, PAD=0, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=2, block_waves=1, dtype="fp32"),
            "gfx950",
        )
    if idx == 17:
        # col with KH != KW (5x3): separates the unrolled axis from the runtime one
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=128, cpg=1, kpg=1, KH=5, KW=3, PAD=2, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="bf16"),
            "gfx950",
        )
    if idx == 18:
        # col stride=2 with valid padding (PAD=0), both guards elided
        p = DirectConvProblem(
            N=1, H=13, W=13, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=0, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=6, block_waves=1, dtype="fp16"),
            "gfx950",
        )
    if idx == 19:
        # col with block_w=1 and a channel tail (groups=100 % 128)
        p = DirectConvProblem(
            N=1, H=10, W=10, groups=100, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=1, block_waves=2, dtype="fp32"),
            "gfx950",
        )
    if idx == 20:
        # dgrad: baseline grouped dgrad stride=1
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx950",
        )
    if idx == 21:
        # dgrad: larger groups / different block_groups
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=4),
            "gfx950",
        )
    if idx == 22:
        # dgrad: gfx942 target
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx942",
        )
    if idx == 23:
        # depthwise_dgrad: stride=1
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 24:
        # depthwise_dgrad: stride=2 exercises divisibility checks
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 25:
        # dwcol PAD-overhang: PAD=2 > (KH-1)/2=1 with stride=2.
        # n_iters = (Ho-1)*stride + KH formula is cross-verified by the C/Python byte-identity gate.
        p = DirectConvProblem(
            N=1, H=10, W=10, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=2, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=1, dtype="fp16"),
            "gfx950",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: conv_direct_grouped_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    kind, spec, arch = _spec(idx)
    if kind == "16c":
        kernel = build_direct_conv_16c(spec, arch=arch)
    elif kind == "4c":
        kernel = build_direct_conv_4c(spec, arch=arch)
    elif kind == "8c":
        kernel = build_direct_conv_8c(spec, arch=arch)
    elif kind == "32c":
        kernel = build_direct_conv_32c(spec, arch=arch)
    elif kind == "spatial":
        kernel = build_direct_depthwise_spatial(spec, arch=arch)
    elif kind == "dwcol":
        kernel = build_direct_depthwise_col(spec, arch=arch)
    elif kind == "dgrad":
        kernel = build_direct_conv_dgrad(spec, arch=arch)
    elif kind == "dw_dgrad":
        kernel = build_direct_depthwise_dgrad(spec, arch=arch)
    else:
        kernel = build_direct_depthwise(spec, arch=arch)
    if mode == "ll":
        text = _native_lower(kernel, arch=arch)
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
