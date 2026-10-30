#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_fused_1d_wmma_emit.py -- Python reference emitter
# for the fused 1-D Winograd forward parity harness. Selects one of N sampled
# spec configs by argv[1], builds the FusedWino1DSpec, builds the kernel via
# build_fused_wino_1d(spec, arch=<cfg arch>) and prints the LLVM IR to stdout
# so it can be byte-compared with the C emitter
# conv_winograd_fused_1d_wmma_emit.c.
#
# Config index map (gfx1250 unless noted; "sel" = the selector's config:
# q2 fk4 4 waves, WMMA transform batch 16, prefetch, fp16 M, merge; "sel32" =
# the C = 32 one: q4 fk2 8 waves, WMMA transform batch 8, prefetch, fp16 M,
# merge):
#   0  1x11 3xF(4,4), C=128, sel
#   1  1x11 3xF(4,4), C=32, sel32, N=2
#   2  1x7 2xF(4,4), C=128, sel
#   3  1x3 F(6,3), C=64, sel, dilation 2
#   4  1x11 3xF(4,4), C=64, sel, dilation 3, pad 15
#   5  1x11 3xF(4,4), C=128, defaults (VALU transform, no prefetch)
#   6  1x11 3xF(4,4), C=128, VALU transform, prefetch, fp16 M
#   7  1x11 3xF(4,4), C=128, VALU transform, prefetch, fp32 M, q2 fk2
#   8  1x11 3xF(4,4), C=128, VALU transform, prefetch + merge, q2 fk2
#   9  1x11 3xF(4,4), C=128, sel with waves_per_eu 2 (WMMA batch 8)
#  10  1x7 F(4,7) single segment, C=64, VALU, prefetch, fp16 M, merge, 4 waves
#  11  1x11 2xF(6,6), C=64, VALU transform, no prefetch, q2 fk2
#  12  1x3 F(2,3), C=64, WMMA transform batch 4, prefetch, fp16 M, no merge
#  13  rejected by both engines: H != 1
#  14  rejected by both engines: B^T of F(6,6) is not fp16-exact (WMMA transform)
#  15  rejected by both engines: gfx950 (no wave32 WMMA atom)
from _emit_common import run_emit
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fused_1d_wmma import (
    FusedWino1DSpec,
    build_fused_wino_1d,
)
from kernels.common.conv_winograd_gen import WinoAxis

SEL = dict(
    q=2, fk=4, waves=4, xf="wmma", xf_batch=16, prefetch=True, m_f16=True, merge=True
)
SEL32 = dict(q=4, fk=2, xf="wmma", xf_batch=8, prefetch=True, m_f16=True, merge=True)


def _p(*, C=128, K=128, X=11, Wi=600, N=1, pW=None, dW=1, Hi=1):
    if pW is None:
        pW = dW * (X // 2)
    return ConvProblem(N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=1, X=X, pW=pW, dW=dW)


def _spec(idx: int):
    S = FusedWino1DSpec
    ax11 = WinoAxis(4, 11, 3)
    cfgs = {
        0: lambda: S(problem=_p(), ax=ax11, **SEL),
        1: lambda: S(problem=_p(C=32, K=64, N=2, Wi=300), ax=ax11, **SEL32),
        2: lambda: S(problem=_p(X=7, K=128), ax=WinoAxis(4, 7, 2), **SEL),
        3: lambda: S(
            problem=_p(C=64, K=64, X=3, Wi=300, dW=2), ax=WinoAxis(6, 3), **SEL
        ),
        4: lambda: S(problem=_p(C=64, K=64, Wi=517, dW=3), ax=ax11, **SEL),
        5: lambda: S(problem=_p(), ax=ax11),
        6: lambda: S(problem=_p(), ax=ax11, prefetch=True, m_f16=True),
        7: lambda: S(problem=_p(C=64, K=64), ax=ax11, q=2, fk=2, prefetch=True),
        8: lambda: S(
            problem=_p(), ax=ax11, q=2, fk=2, prefetch=True, m_f16=True, merge=True
        ),
        9: lambda: S(problem=_p(), ax=ax11, **dict(SEL, xf_batch=8, waves_per_eu=2)),
        10: lambda: S(
            problem=_p(C=64, K=64, X=7, Wi=517, dW=5, pW=15),
            ax=WinoAxis(4, 7, 1),
            q=2,
            fk=4,
            prefetch=True,
            m_f16=True,
            merge=True,
            waves=4,
        ),
        11: lambda: S(problem=_p(C=64, K=64), ax=WinoAxis(6, 11, 2), q=2, fk=2),
        12: lambda: S(
            problem=_p(C=64, K=64, X=3, Wi=300),
            ax=WinoAxis(2, 3),
            xf="wmma",
            xf_batch=4,
            prefetch=True,
            m_f16=True,
        ),
        13: lambda: S(problem=_p(Hi=2), ax=ax11),
        14: lambda: S(problem=_p(C=64, K=64), ax=WinoAxis(6, 11, 2), xf="wmma"),
        15: lambda: (S(problem=_p(), ax=ax11, **SEL), "gfx950"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    out = cfgs[idx]()
    return out if isinstance(out, tuple) else (out, "gfx1250")


def main() -> int:
    return run_emit(
        _spec,
        build_fused_wino_1d,
        usage="usage: conv_winograd_fused_1d_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
