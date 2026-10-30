#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_pro2_emit.py -- Python reference emitter of the gemm_vp_pro2 parity
# family: the prologue combinations across the axes: recipe x split_mode x complex form on every body, split modes with batched / wide_addr, f16 / bf16 h16 on every mode, psub / cap / unary operators, pads, lds_swizzle_l 2 / 3 / 4 and the main-loop levers with a prologue.
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_pro2"] (see
# _gemm_vp_common.py); gemm_vp_pro2_emit.c builds the same specs through the C API.
# `python gemm_vp_pro2_emit.py --c-cases` prints the case rows of gemm_vp_pro2_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_pro2")


if __name__ == "__main__":
    sys.exit(main())
