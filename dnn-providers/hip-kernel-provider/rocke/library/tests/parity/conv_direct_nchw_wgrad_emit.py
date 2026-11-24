#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_nchw_wgrad_emit.py -- Python reference emitter for the
# native-NCHW direct wgrad convolution parity harness. Selects one of N
# sampled spec configs by argv[1], builds the DirectNchwWgradSpec, calls
# build_direct_conv_nchw_wgrad(spec, arch="gfx950") and prints
# lower_kernel_to_llvm to stdout so it can be byte-compared with the C
# emitter conv_direct_nchw_wgrad_emit.c. Keep the config table in lockstep with
# that file's make_cfg().
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nchw_wgrad import (
    DirectNchwWgradSpec,
    build_direct_conv_nchw_wgrad,
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
# make_cfg() in conv_direct_nchw_wgrad_emit.c (same numbering, same coverage:
# fp16/bf16, 3x3/5x5, stride 1/2, every knob touched at least once).
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
    # 5: fp16 3x3 s1 p1, lds_pad=0, stage_group=8.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", lds_pad=0, stage_group=8),
    # 6: fp16 3x3 s1 p1, mfma_order="ks".
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", mfma_order="ks"),
    # 7: fp16 3x3 s1 p1, prefetch=False.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", prefetch=False),
    # 8: fp16 3x3 s1 p1, xcd_remap=False.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", xcd_remap=False),
    # 9: fp16 3x3 s1 p1, waves_per_eu=2.
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", waves_per_eu=2),
    # 10: fp16 3x3 s1 p1, epilogue="direct" + y_groups=3.
    dict(
        KH=3,
        KW=3,
        stride=1,
        PAD=1,
        dtype="fp16",
        epilogue="direct",
        y_groups=3,
        tile_m=32,
        tile_c=32,
        waves_m=1,
        waves_c=2,
    ),
    # 11: fp16 3x3 s1 p1, atomic=False (plain-store epilogue, lds).
    dict(KH=3, KW=3, stride=1, PAD=1, dtype="fp16", atomic=False),
    # 12: fp16 3x3 s2 p1, lds_buffers=2, tile_h=2 (prefetch stays True).
    dict(KH=3, KW=3, stride=2, PAD=1, dtype="fp16", lds_buffers=2, tile_h=2),
    # 13: fp16 3x3 s1 p1, kmap="2row" with the matching pads.
    dict(
        KH=3,
        KW=3,
        stride=1,
        PAD=1,
        dtype="fp16",
        kmap="2row",
        tile_h=2,
        tile_w=16,
        lds_pad=8,
        lds_wpad=2,
    ),
    # 14: bf16 5x5 s2 p2, waves_t=5 (KH/y_groups * KW = 25 taps, waves_t | 25).
    dict(
        KH=5,
        KW=5,
        stride=2,
        PAD=2,
        dtype="bf16",
        tile_m=32,
        tile_c=32,
        waves_m=1,
        waves_c=1,
        waves_t=5,
    ),
    # 15: fp16 3x3 s1 p0 (pad=0), small tiles / wave split.
    dict(
        KH=3,
        KW=3,
        stride=1,
        PAD=0,
        dtype="fp16",
        tile_m=32,
        tile_c=32,
        waves_m=1,
        waves_c=1,
    ),
    # 16: fp16 3x3 s1 p1, kmap="2row" + lds_buffers=2 combined.
    dict(
        KH=3,
        KW=3,
        stride=1,
        PAD=1,
        dtype="fp16",
        kmap="2row",
        tile_h=2,
        lds_pad=8,
        lds_buffers=2,
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
    return DirectNchwWgradSpec(problem=p, **cfg), "gfx950"


def main() -> int:
    return run_emit(
        _spec,
        build_direct_conv_nchw_wgrad,
        usage="usage: conv_direct_nchw_wgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
