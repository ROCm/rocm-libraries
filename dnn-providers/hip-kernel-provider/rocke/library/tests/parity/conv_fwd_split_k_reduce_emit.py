#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_fwd_split_k_reduce_emit.py -- Python reference emitter for
# the split-K forward Stage 2 (conv_fwd_split_k_reduce) parity harness. Selects
# one of N sampled spec configs by argv[1], builds the FwdSplitKReduceSpec,
# builds the kernel via build_conv_fwd_split_k_reduce(spec, arch=<cfg arch>)
# and prints the LLVM IR to stdout so it can be byte-compared with the C
# emitter conv_fwd_split_k_reduce_emit.c.
#
# Config index map:
#   0  -- bf16, vec 4, runtime slab loop, gfx1250
#   1  -- fp16, vec 4, runtime slab loop, gfx1250
#   2  -- bf16, vec 2, runtime slab loop, gfx1250
#   3  -- fp16, vec 1 (scalar store), runtime slab loop, gfx1250
#   4  -- bf16, vec 4, unrolled fold, degree 4, gfx1250
#   5  -- fp16, vec 2, unrolled fold, degree 3, gfx1250
#   6  -- bf16, vec 1, unrolled fold, degree 16, gfx1250
#   7  -- fp16, vec 4, block 128, unrolled fold, degree 8, gfx1250
#   8  -- bf16, vec 4, runtime slab loop, gfx950
#   9  -- vec 3 (rejected by both engines)
#   10 -- dtype_d None / NULL (rejected by both engines, not defaulted)
from kernels.common.conv_fwd_split_k_reduce import (
    FwdSplitKReduceSpec,
    build_conv_fwd_split_k_reduce,
)
from _emit_common import run_emit

_CONFIGS = {
    0: (dict(dtype_d="bf16", vec=4), "gfx1250"),
    1: (dict(dtype_d="fp16", vec=4), "gfx1250"),
    2: (dict(dtype_d="bf16", vec=2), "gfx1250"),
    3: (dict(dtype_d="fp16", vec=1), "gfx1250"),
    4: (dict(dtype_d="bf16", vec=4, degree=4), "gfx1250"),
    5: (dict(dtype_d="fp16", vec=2, degree=3), "gfx1250"),
    6: (dict(dtype_d="bf16", vec=1, degree=16), "gfx1250"),
    7: (dict(dtype_d="fp16", vec=4, block_size=128, degree=8), "gfx1250"),
    8: (dict(dtype_d="bf16", vec=4), "gfx950"),
    9: (dict(dtype_d="bf16", vec=3), "gfx1250"),
    10: (dict(dtype_d=None, vec=4), "gfx1250"),
}


def _spec(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    kw, arch = _CONFIGS[idx]
    return FwdSplitKReduceSpec(**kw), arch


def main() -> int:
    return run_emit(
        _spec,
        build_conv_fwd_split_k_reduce,
        usage="usage: conv_fwd_split_k_reduce_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
