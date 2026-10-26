#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_gen_emit.py -- Python reference emitter for the
# generalized Winograd forward transform kernels (conv_winograd_gen). Selects
# one config by argv[1]; a config is (WinogradGenSpec, stage, arch) with stage
# K1 (filter), K2 (input) or K4 (output). Prints the LLVM IR so it can be
# byte-compared with conv_winograd_gen_emit.c.
#
# Config index map (stage K1, K2, K4 at index 3 * case + 0, 1, 2):
#   case 0  1x11 6s2 (2 x F(6,6)), fp16, pad 5                 gfx1250
#   case 1  1x11 4s3 (3 x F(4,4)), bf16 I/O, vec 8             gfx1250
#   case 2  1x7 F(4,7), dilation 2, nt_stores off              gfx1250
#   case 3  1x7 4s2 (2 x F(4,4)), block_size 128               gfx950
#   case 4  1x3 F(6,3), H > 1 (two rows)                       gfx1250
#   case 5  7x1 F(4,7) (r x 1 filter)                          gfx1250
#   case 6  5x5 F(2x2,5x5), groups 2                           gfx1250
#   case 7  5x5 F(3x3,5x5), groups 4, dilation 2, bf16 I/O     gfx950
#   case 8  1x11 6s2, C = K = 33 (odd: vec 1), dilation 3      gfx1250
#   case 9  1x3 F(2,3), vec 2, N = 3                           gfx1250
#   30      rejected by both engines: stride 2
#   31      rejected by both engines: F(4,11) single segment (no default points)
#   32      rejected by both engines: segments along H
from _emit_common import run_emit
from kernels.common._conv_implicit_gemm_common import ConvProblem as P
from kernels.common.conv_winograd_gen import (
    WinoAxis,
    WinogradGenSpec,
    build_filter_transform,
    build_input_transform,
    build_output_transform,
)

A = WinoAxis
ONE = A(1, 1)
STAGES = ("K1", "K2", "K4")
BUILD = {
    "K1": build_filter_transform,
    "K2": build_input_transform,
    "K4": build_output_transform,
}


def _case(i: int):
    """(problem, ax_y, ax_x, spec kwargs, arch) of case ``i``."""
    if i == 0:
        p = P(N=2, Hi=1, Wi=200, C=64, K=64, Y=1, X=11, pW=5)
        return p, ONE, A(6, 11, 2), {}, "gfx1250"
    if i == 1:
        p = P(N=2, Hi=1, Wi=150, C=64, K=96, Y=1, X=11, pW=5)
        return p, ONE, A(4, 11, 3), dict(io_dtype="bf16", vec=8), "gfx1250"
    if i == 2:
        p = P(N=1, Hi=1, Wi=100, C=32, K=64, Y=1, X=7, pW=6, dW=2)
        return p, ONE, A(4, 7), dict(nt_stores=False), "gfx1250"
    if i == 3:
        p = P(N=1, Hi=1, Wi=90, C=128, K=64, Y=1, X=7, pW=3)
        return p, ONE, A(4, 7, 2), dict(block_size=128), "gfx950"
    if i == 4:
        p = P(N=1, Hi=2, Wi=100, C=128, K=128, Y=1, X=3, pW=1)
        return p, ONE, A(6, 3), {}, "gfx1250"
    if i == 5:
        p = P(N=1, Hi=50, Wi=4, C=64, K=64, Y=7, X=1, pH=3)
        return p, A(4, 7), ONE, {}, "gfx1250"
    if i == 6:
        p = P(N=1, Hi=20, Wi=20, C=64, K=64, Y=5, X=5, pH=2, pW=2, groups=2)
        return p, A(2, 5), A(2, 5), {}, "gfx1250"
    if i == 7:
        p = P(
            N=1, Hi=24, Wi=22, C=128, K=64, Y=5, X=5, pH=4, pW=4, dH=2, dW=2, groups=4
        )
        return p, A(3, 5), A(3, 5), dict(io_dtype="bf16"), "gfx950"
    if i == 8:
        p = P(N=1, Hi=1, Wi=120, C=33, K=33, Y=1, X=11, pW=15, dW=3)
        return p, ONE, A(6, 11, 2), {}, "gfx1250"
    if i == 9:
        p = P(N=3, Hi=1, Wi=40, C=64, K=32, Y=1, X=3, pW=1)
        return p, ONE, A(2, 3), dict(vec=2), "gfx1250"
    return None


def _spec(idx: int):
    if idx == 30:
        p = P(N=1, Hi=1, Wi=64, C=64, K=64, Y=1, X=3, pW=1, sW=2)
        return (WinogradGenSpec(p, ONE, A(2, 3)), "K1"), "gfx1250"
    if idx == 31:
        p = P(N=1, Hi=1, Wi=64, C=64, K=64, Y=1, X=11, pW=5)
        return (WinogradGenSpec(p, ONE, A(4, 11)), "K2"), "gfx1250"
    if idx == 32:
        p = P(N=1, Hi=20, Wi=20, C=64, K=64, Y=5, X=5, pH=2, pW=2)
        return (WinogradGenSpec(p, A(3, 5, 2), A(2, 5)), "K4"), "gfx1250"
    case = _case(idx // 3)
    if case is None:
        raise SystemExit(f"unknown config index {idx}")
    p, ay, ax, kw, arch = case
    return (WinogradGenSpec(problem=p, ax_y=ay, ax_x=ax, **kw), STAGES[idx % 3]), arch


def _build(sel, *, arch):
    spec, stage = sel
    return BUILD[stage](spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: conv_winograd_gen_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
