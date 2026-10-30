#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_modes_emit.py -- Python reference emitter of the gemm_vp_modes parity
# family: vgpr_prefetch modes 1-4 with depth 1-3, vp_unroll, the mode-4 main-loop levers singly and combined, pads (tight resources), the K tail, split-K (atomic, aligned slices, in-launch serial reduction), swizzle / schedule / epilogue variants, gfx950, and the out_f32 prologue.
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_modes"] (see
# _gemm_vp_common.py); gemm_vp_modes_emit.c builds the same specs through the C API.
# `python gemm_vp_modes_emit.py --c-cases` prints the case rows of gemm_vp_modes_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_modes")


if __name__ == "__main__":
    sys.exit(main())
