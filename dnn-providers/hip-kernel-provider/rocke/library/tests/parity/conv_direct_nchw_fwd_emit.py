#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_nchw_fwd_emit.py -- Python reference emitter for the
# native-NCHW direct forward convolution parity harness. Selects one of N
# sampled spec configs by argv[1], builds the DirectNchwFwdSpec, calls
# build_direct_conv_nchw_fwd(spec, arch="gfx950") and prints
# lower_kernel_to_llvm to stdout so it can be byte-compared with the C
# emitter conv_direct_nchw_fwd_emit.c. Keep the config table in lockstep with
# that file's make_cfg().
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nchw_fwd import DirectNchwFwdSpec, build_direct_conv_nchw_fwd
from _emit_common import run_emit


def _p(KH, KW, stride, PAD, dtype="fp16"):
    return DirectConvProblem(
        N=2,
        H=32,
        W=32,
        groups=1,
        cpg=64,
        kpg=64,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
    )


# idx -> (problem kwargs, spec kwargs). Keep this table in lockstep with
# make_cfg() in conv_direct_nchw_fwd_emit.c (same numbering, same coverage:
# fp16/bf16, 3x3/5x5, stride 1/2, pad 0/1/2, every knob touched at least once).
_CONFIGS = [
    # 0: baseline fp16 3x3 s1 p1, default knobs.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16"),
    # 1: bf16 3x3 s1 p1, default knobs.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="bf16"),
    # 2: fp16 3x3 s2 p1, col_split off.
    dict(KH=3, KW=3, stride=2, PAD=1, dtype="fp16", col_split=False),
    # 3: fp16 3x3 s2 p1, col_split on.
    dict(KH=3, KW=3, stride=2, PAD=1, dtype="fp16", col_split=True),
    # 4: fp16 5x5 s1 p2, default knobs (pad=2).
    dict(KH=5, KW=5, stride=1, PAD=2, dtype="fp16"),
    # 5: fp16 5x5 s2 p2, col_split on.
    dict(KH=5, KW=5, stride=2, PAD=2, dtype="fp16", col_split=True),
    # 6: fp16 3x3 s1 p0 (pad=0).
    dict(KH=3, KW=3, stride=1, PAD=0, dtype="fp16"),
    # 7: fp16 3x3 s1 p1, lds_pad=0.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", lds_pad=0),
    # 8: fp16 3x3 s1 p1, lds_wpad=8.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", lds_wpad=8),
    # 9: fp16 3x3 s1 p1, waves_per_eu=2.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", waves_per_eu=2),
    # 10: fp16 3x3 s1 p1, pin_prefetch on.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", pin_prefetch=True),
    # 11: bf16 5x5 s2 p2, col_split + waves_per_eu + pin_prefetch combined.
    dict(
        KH=5,
        KW=5,
        stride=2,
        PAD=2,
        dtype="bf16",
        col_split=True,
        waves_per_eu=2,
        pin_prefetch=True,
    ),
]


def _spec(idx: int):
    if idx < 0 or idx >= len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = dict(_CONFIGS[idx])
    KH = cfg.pop("KH")
    KW = cfg.pop("KW")
    stride = cfg.pop("stride")
    PAD = cfg.pop("PAD")
    dtype = cfg.pop("dtype")
    p = _p(KH, KW, stride, PAD, dtype)
    return DirectNchwFwdSpec(problem=p, **cfg), "gfx950"


def main() -> int:
    return run_emit(
        _spec,
        build_direct_conv_nchw_fwd,
        usage="usage: conv_direct_nchw_fwd_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
