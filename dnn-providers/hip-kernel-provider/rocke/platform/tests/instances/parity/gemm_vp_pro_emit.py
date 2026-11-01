#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_pro_emit.py -- Python reference emitter of the gemm_vp_pro parity
# family: the screened prologue set on the register-staged body: every recipe (h16 out_f32, f32 cvt / x3 / x4 / x6 / x9) x body (modes 3 / 4, main-loop levers, unroll) x complex form (cplx_b, conj) x batched x geometry, plus wide_addr, split modes, lds_swizzle_l, split_at load and modes 1 / 2.
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_pro"] (see
# _gemm_vp_common.py); gemm_vp_pro_emit.c builds the same specs through the C API.
# `python gemm_vp_pro_emit.py --c-cases` prints the case rows of gemm_vp_pro_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_pro")


if __name__ == "__main__":
    sys.exit(main())
