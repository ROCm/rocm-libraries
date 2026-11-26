#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_fwd_tap1d_mfma_emit.py -- Python reference emitter
# for the gfx950 1-D tap-reuse MFMA direct forward parity harness. Selects one
# of N sampled spec configs by argv[1], builds the FwdTap1dMfmaSpec, builds the
# kernel via build_fwd_tap1d_mfma(spec, arch=<cfg arch>) and prints the LLVM IR
# to stdout so it can be byte-compared with the C emitter
# conv_direct_fwd_tap1d_mfma_emit.c.
#
# The problem is N=2, 4x50, 1 x S filter, stride 1, no height padding. Rows
# 0-7 are fp16, 8-15 the same rows in bf16 (row i + 8); the (bk, nwv, ck, one)
# knobs are the AOT grid's eight tuples (seg is the default 9):
#   0  1x3  pad 0  dil 1, C 128, K 256, bk 128 nwv 8 ck 64
#   1  1x3  pad 1  dil 1, C 192, K 128, bk 128 nwv 4 ck 64
#   2  1x7  pad 9  dil 3, C 96,  K 128, bk 128 nwv 4 ck 32
#   3  1x11 pad 15 dil 3, C 128, K 64,  bk 64  nwv 4 ck 64
#   4  1x11 pad 25 dil 5, C 96,  K 128, bk 64  nwv 4 ck 32
#   5  1x7  pad 0  dil 1, C 64,  K 128, bk 64  nwv 4 ck 64 one
#   6  1x3  pad 5  dil 5, C 64,  K 96,  bk 32  nwv 4 ck 64 one
#   7  1x11 pad 5  dil 1, C 32,  K 64,  bk 32  nwv 4 ck 32 one
# Rejected by both engines:
#  16  gfx942 (no 16-byte LDS DMA)
#  17  3x3 filter
#  18  bk 96
#  19  one with C != ck
#  20  C 96 with ck 64
#  21  width padding past dilation * (S - 1)
#  22  stride 2
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_tap1d_mfma import (
    FwdTap1dMfmaSpec,
    build_fwd_tap1d_mfma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(*, S=3, pw=0, dil=1, C=128, K=128, KH=1, stride=1, dtype="fp16"):
    return DirectConvProblem(
        N=2,
        H=4,
        W=50,
        groups=1,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=S,
        PAD=0,
        pad_w=pw,
        stride=stride,
        dil_w=dil,
        dtype=dtype,
    )


# (S, pad_w, dil, C, K, bk, nwv, ck, one) of rows 0-7
_ROWS = (
    (3, 0, 1, 128, 256, 128, 8, 64, False),
    (3, 1, 1, 192, 128, 128, 4, 64, False),
    (7, 9, 3, 96, 128, 128, 4, 32, False),
    (11, 15, 3, 128, 64, 64, 4, 64, False),
    (11, 25, 5, 96, 128, 64, 4, 32, False),
    (7, 0, 1, 64, 128, 64, 4, 64, True),
    (3, 5, 5, 64, 96, 32, 4, 64, True),
    (11, 5, 1, 32, 64, 32, 4, 32, True),
)

_REJECT = {
    16: ((dict(), dict()), "gfx942"),
    17: ((dict(KH=3), dict()), "gfx950"),
    18: ((dict(K=192), dict(bk=96)), "gfx950"),
    19: ((dict(), dict(one=True)), "gfx950"),
    20: ((dict(C=96), dict()), "gfx950"),
    21: ((dict(pw=3), dict()), "gfx950"),
    22: ((dict(stride=2), dict()), "gfx950"),
}


def _spec(idx: int):
    n = len(_ROWS)
    if 0 <= idx < 2 * n:
        S, pw, dil, C, K, bk, nwv, ck, one = _ROWS[idx % n]
        p = _p(S=S, pw=pw, dil=dil, C=C, K=K, dtype="bf16" if idx >= n else "fp16")
        return FwdTap1dMfmaSpec(problem=p, bk=bk, nwv=nwv, ck=ck, one=one), "gfx950"
    if idx in _REJECT:
        (pkw, skw), arch = _REJECT[idx]
        return FwdTap1dMfmaSpec(problem=_p(**pkw), **skw), arch
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_fwd_tap1d_mfma,
        usage="usage: conv_direct_fwd_tap1d_mfma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
