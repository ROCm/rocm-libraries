#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_fwd_rr_mfma_emit.py -- Python reference emitter for
# the gfx950 rows-in-registers MFMA direct forward parity harness (the conv and
# its weight-format helper). Selects one of N sampled spec configs by argv[1],
# builds the FwdRrMfmaSpec (or FwdRrWfmtSpec), builds the kernel via
# build_fwd_rr_mfma / build_fwd_rr_wfmt(spec, arch=<cfg arch>) and prints the
# LLVM IR to stdout so it can be byte-compared with the C emitter
# conv_direct_fwd_rr_mfma_emit.c.
#
# The base problem is N=2, 20x40, C=128, K=128, 3x3 padding 1, stride 1; each
# row overrides what it changes. Rows 0-20 are fp16, 21-41 the same rows in
# bf16 (row i + 21); every knob class of the AOT grid is covered per dtype.
#   0  3x3 p1 (defaults: kw16 4)
#   1  3x3 p0, K 192, kw16 3
#   2  3x3 p2, C 192, kw16 2, odd_c
#   3  3x3 p1, kw16 2, c64
#   4  3x3 p2, C 192, kw16 2, c64, odd_c
#   5  3x3 p1, K 256, k4w
#   6  3x3 p0, C 72, K 256, k4w, c_tail
#   7  3x3 p0, C 64, one_c
#   8  3x3 p1, C 40, kw16 2, one_c, c_tail
#   9  3x3 p1, groups 2, grp
#  10  3x3 p1, groups 4, C 64, K 64, grp, kw16 2, one_c
#  11  3x3 p1, xcd
#  12  3x3 p1, W 20, K 256, k4w, chunk
#  13  3x1 p1x0
#  14  3x1 p1x0, W 1, N 40, unfold
#  15  3x1 p2x0, W 1, N 40, K 256, unfold, k4w, xcd
#  16  3x1 p1x0, W 1, N 40, C 64, unfold, kw16 2, one_c
#  17  3x1 p2x0, K 192, kw16 3
#  18  3x1 p0x0, C 192, kw16 2, odd_c
#  19  3x3 p2, kw16 2, c64, xcd
#  20  3x3 p1, C 64, K 256, k4w, one_c, xcd
# Weight-format helper (C 72: the format zero-pads C to a whole chunk); rows
# 42-47 fp16, 48-53 the same rows in bf16:
#  42  kw16 4, fw 3, nwk 2 (defaults)
#  43  kw16 3, fw 1, nwk 2
#  44  kw16 2, fw 3, nwk 2
#  45  kw16 4, fw 3, nwk 4 (k4w)
#  46  kw16 2, fw 1, nwk 4
#  47  kw16 4, fw 3, nwk 2, dg (a dgrad's own filter, flipped)
# Rejected by both engines:
#  54  gfx942 (no 16-byte LDS DMA)
#  55  C 96 (not a multiple of 64, no c_tail)
#  56  3x1 with c64
#  57  chunk without k4w
#  58  C 192 (odd chunk count) without odd_c
#  59  kw16 5
#  60  padding 3
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_rr_mfma import (
    FwdRrMfmaSpec,
    FwdRrWfmtSpec,
    build_fwd_rr_mfma,
    build_fwd_rr_wfmt,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(*, N=2, W=40, groups=1, C=128, K=128, KW=3, PAD=1, pad_w=-1, dtype="fp16"):
    return DirectConvProblem(
        N=N,
        H=20,
        W=W,
        groups=groups,
        cpg=C,
        kpg=K,
        KH=3,
        KW=KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=1,
        dtype=dtype,
    )


# (problem overrides, spec knobs) of rows 0-20
_CONV = (
    ({}, {}),
    (dict(PAD=0, K=192), dict(kw16=3)),
    (dict(PAD=2, C=192), dict(kw16=2, odd_c=True)),
    ({}, dict(kw16=2, c64=True)),
    (dict(PAD=2, C=192), dict(kw16=2, c64=True, odd_c=True)),
    (dict(K=256), dict(k4w=True)),
    (dict(PAD=0, C=72, K=256), dict(k4w=True, c_tail=True)),
    (dict(PAD=0, C=64), dict(one_c=True)),
    (dict(C=40), dict(kw16=2, one_c=True, c_tail=True)),
    (dict(groups=2), dict(grp=True)),
    (dict(groups=4, C=64, K=64), dict(grp=True, kw16=2, one_c=True)),
    ({}, dict(xcd=True)),
    (dict(W=20, K=256), dict(k4w=True, chunk=True)),
    (dict(KW=1, pad_w=0), {}),
    (dict(KW=1, pad_w=0, W=1, N=40), dict(unfold=True)),
    (
        dict(KW=1, PAD=2, pad_w=0, W=1, N=40, K=256),
        dict(unfold=True, k4w=True, xcd=True),
    ),
    (dict(KW=1, pad_w=0, W=1, N=40, C=64), dict(unfold=True, kw16=2, one_c=True)),
    (dict(KW=1, PAD=2, pad_w=0, K=192), dict(kw16=3)),
    (dict(KW=1, PAD=0, pad_w=0, C=192), dict(kw16=2, odd_c=True)),
    (dict(PAD=2), dict(kw16=2, c64=True, xcd=True)),
    (dict(C=64, K=256), dict(k4w=True, one_c=True, xcd=True)),
)

# (kw16, fw, nwk, dg) of rows 42-47
_WFMT = (
    (4, 3, 2, False),
    (3, 1, 2, False),
    (2, 3, 2, False),
    (4, 3, 4, False),
    (2, 1, 4, False),
    (4, 3, 2, True),
)

_REJECT = {
    54: (({}, {}), "gfx942"),
    55: ((dict(C=96), {}), "gfx950"),
    56: ((dict(KW=1, pad_w=0), dict(kw16=2, c64=True)), "gfx950"),
    57: (({}, dict(chunk=True)), "gfx950"),
    58: ((dict(C=192), {}), "gfx950"),
    59: (({}, dict(kw16=5)), "gfx950"),
    60: ((dict(PAD=3), {}), "gfx950"),
}


def _spec(idx: int):
    nc, nw = len(_CONV), len(_WFMT)
    if 0 <= idx < 2 * nc:
        pkw, skw = _CONV[idx % nc]
        dtype = "bf16" if idx >= nc else "fp16"
        return FwdRrMfmaSpec(problem=_p(dtype=dtype, **pkw), **skw), "gfx950"
    if 2 * nc <= idx < 2 * nc + 2 * nw:
        j = idx - 2 * nc
        kw16, fw, nwk, dg = _WFMT[j % nw]
        dtype = "bf16" if j >= nw else "fp16"
        p = _p(dtype=dtype, C=72, K=192, KW=fw)
        return FwdRrWfmtSpec(problem=p, kw16=kw16, fw=fw, nwk=nwk, dg=dg), "gfx950"
    if idx in _REJECT:
        (pkw, skw), arch = _REJECT[idx]
        return FwdRrMfmaSpec(problem=_p(**pkw), **skw), arch
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, *, arch):
    if isinstance(spec, FwdRrWfmtSpec):
        return build_fwd_rr_wfmt(spec, arch=arch)
    return build_fwd_rr_mfma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_direct_fwd_rr_mfma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
