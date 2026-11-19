#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_wgrad_workspace_reduce_emit.py -- Python reference emitter
# for the wgrad workspace-reduce (Stage 2) parity harness.  Selects one of N
# sampled spec configs by argv[1], builds the WgradReduceSpec, builds the
# kernel via build_conv_wgrad_workspace_reduce(spec, arch=<cfg arch>) and
# prints the LLVM IR to stdout so it can be byte-compared with the C emitter
# conv_wgrad_workspace_reduce_emit.c.
#
# Config index map:
#   0  -- fp16 output, wg_M=64, wg_N=576  (3x3 filter, C=64, K=64), gfx950
#   1  -- bf16 output, wg_M=64, wg_N=576, gfx950
#   2  -- fp32 output, wg_M=64, wg_N=576, gfx950
#   3  -- fp16 output, wg_M=32, wg_N=72   (3x3 filter, C=8, K=32), gfx950
#   4  -- fp16 output, wg_M=64, wg_N=576, gfx942
#   5  -- bf16 output, wg_M=32, wg_N=72,  gfx950
#   6  -- fp16 output, custom tile_m=8, tile_n=32, gfx950
#   7  -- ws_replicas=1 (the fold elided), gfx950
#   8  -- bf16 output, ws_replicas=4, gfx950
#   9-17 -- the vector / looped folds (vec, rezero, ws_bf16, fold_loop,
#         slab_par) on the bf16 64x576 problem, gfx942; 17 (fold_loop on a
#         grouped fold) both engines must reject
#   18-21 -- both engines must reject: ws_bf16 on the scalar fold, slab_par
#         without fold_loop, vec=32 and fold_loop=128 (past the widest vector
#         and the deepest loop the folds hold)
from kernels.common.conv_wgrad_workspace_reduce import (
    WgradReduceSpec,
    build_conv_wgrad_workspace_reduce,
)
from kernels.common._conv_implicit_gemm_common import ConvProblem
from _emit_common import run_emit


def _make_problem(wg_M: int, wg_N: int) -> ConvProblem:
    """Build a minimal ConvProblem that yields the requested wg_M and wg_N.

    wg_M = K, wg_N = Y * X * C.  Use Y=3, X=3 so wg_N = 9 * C = wg_N.
    Requires wg_N divisible by 9.  Fallback: Y=1, X=1, C=wg_N.
    """
    if wg_N % 9 == 0:
        C = wg_N // 9
        return ConvProblem(N=1, Hi=4, Wi=4, C=C, K=wg_M, Y=3, X=3)
    return ConvProblem(N=1, Hi=4, Wi=4, C=wg_N, K=wg_M, Y=1, X=1)


def _spec(idx: int):
    if idx == 0:
        return (
            WgradReduceSpec(
                problem=_make_problem(64, 576),
                dtype_d="fp16",
            ),
            "gfx950",
        )
    if idx == 1:
        return (
            WgradReduceSpec(
                problem=_make_problem(64, 576),
                dtype_d="bf16",
            ),
            "gfx950",
        )
    if idx == 2:
        return (
            WgradReduceSpec(
                problem=_make_problem(64, 576),
                dtype_d="fp32",
            ),
            "gfx950",
        )
    if idx == 3:
        return (
            WgradReduceSpec(
                problem=_make_problem(32, 72),
                dtype_d="fp16",
            ),
            "gfx950",
        )
    if idx == 4:
        return (
            WgradReduceSpec(
                problem=_make_problem(64, 576),
                dtype_d="fp16",
            ),
            "gfx942",
        )
    if idx == 5:
        return (
            WgradReduceSpec(
                problem=_make_problem(32, 72),
                dtype_d="bf16",
            ),
            "gfx950",
        )
    if idx == 6:
        return (
            WgradReduceSpec(
                problem=_make_problem(64, 576),
                dtype_d="fp16",
                tile_m=8,
                tile_n=32,
            ),
            "gfx950",
        )
    if idx == 7:
        # ws_replicas=1: the fold is elided entirely and the body collapses to
        # a single load.  A distinct emission path from the default.
        return (
            WgradReduceSpec(
                problem=_make_problem(64, 576),
                dtype_d="fp16",
                ws_replicas=1,
            ),
            "gfx950",
        )
    if idx == 8:
        # A non-default replica count, so the factor is proven to reach the
        # emitted body rather than only the kernel name.
        return (
            WgradReduceSpec(
                problem=_make_problem(32, 72),
                dtype_d="bf16",
                ws_replicas=4,
            ),
            "gfx950",
        )
    # 9-21: the vector / looped folds (vec, rezero, ws_bf16, fold_loop,
    # slab_par) on the bf16 64x576 problem; 17-21 both engines must reject.
    # Keep in lockstep with the C emitter.
    _v = {
        9: dict(vec=4),
        10: dict(vec=4, rezero=1),
        11: dict(vec=4, ws_bf16=True),
        12: dict(vec=4, ws_replicas=48, fold_loop=4, slab_par=4, tile_m=1),
        13: dict(vec=4, ws_replicas=50, fold_loop=4, slab_par=4, tile_m=1),
        14: dict(vec=4, ws_replicas=10, fold_loop=4, ws_bf16=True),
        15: dict(vec=4, ws_replicas=2, dtype_d="fp32"),
        16: dict(rezero=1),
        17: dict(vec=4, fold_loop=4, groups=2),
        18: dict(ws_bf16=True),
        19: dict(vec=4, slab_par=4),
        20: dict(vec=32),
        21: dict(vec=4, ws_replicas=256, fold_loop=128, tile_m=1),
    }
    if idx in _v:
        kw = dict(dtype_d="bf16")
        kw.update(_v[idx])
        return WgradReduceSpec(problem=_make_problem(64, 576), **kw), "gfx942"
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_conv_wgrad_workspace_reduce,
        usage="usage: conv_wgrad_workspace_reduce_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
