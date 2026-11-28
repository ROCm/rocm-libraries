#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_nchw_dgrad_emit.py -- Python reference emitter for the
# native-NCHW direct dgrad convolution parity harness. Selects one of N
# sampled spec configs by argv[1], builds the DirectNchwDgradSpec, calls
# build_direct_conv_nchw_dgrad(spec, arch="gfx950") and prints
# lower_kernel_to_llvm to stdout so it can be byte-compared with the C
# emitter conv_direct_nchw_dgrad_emit.c. Keep the config table in lockstep with
# that file's make_cfg().
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nchw_dgrad import (
    DirectNchwDgradSpec,
    build_direct_conv_nchw_dgrad,
)
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
# make_cfg() in conv_direct_nchw_dgrad_emit.c (same numbering, same coverage:
# fp16/bf16, 3x3/5x5, stride 1/2, pad 0/1/2, every stage_group value touched).
_CONFIGS = [
    # 0: baseline fp16 3x3 s1 p1, default knobs.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16"),
    # 1: bf16 3x3 s1 p1, default knobs.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="bf16"),
    # 2: fp16 3x3 s2 p1.
    dict(KH=3, KW=3, stride=2, PAD=1, dtype="fp16"),
    # 3: fp16 5x5 s1 p2, default knobs (pad=2).
    dict(KH=5, KW=5, stride=1, PAD=2, dtype="fp16"),
    # 4: fp16 5x5 s2 p2.
    dict(KH=5, KW=5, stride=2, PAD=2, dtype="fp16"),
    # 5: fp16 3x3 s1 p0 (pad=0).
    dict(KH=3, KW=3, stride=1, PAD=0, dtype="fp16"),
    # 6: fp16 3x3 s1 p1, lds_pad=0.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", lds_pad=0),
    # 7: fp16 3x3 s1 p1, stage_group=1 (per-element ds_write_b16 scatter).
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", stage_group=1),
    # 8: fp16 3x3 s1 p1, stage_group=2.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", stage_group=2),
    # 9: fp16 3x3 s1 p1, stage_group=4.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", stage_group=4),
    # 10: fp16 3x3 s2 p1, stage_group=1 (halo + stride-2 interleaved store).
    dict(KH=3, KW=3, stride=2, PAD=1, dtype="fp16", stage_group=1),
    # 11: bf16 5x5 s2 p2, stage_group=2 + lds_pad=0 combined.
    dict(
        KH=5,
        KW=5,
        stride=2,
        PAD=2,
        dtype="bf16",
        stage_group=2,
        lds_pad=0,
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
    return DirectNchwDgradSpec(problem=p, **cfg), "gfx950"


def main() -> int:
    return run_emit(
        _spec,
        build_direct_conv_nchw_dgrad,
        usage="usage: conv_direct_nchw_dgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
