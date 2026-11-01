#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_h16_emit.py -- Python reference emitter of the gemm_vp_h16 parity
# family: the same picks as the hipTensor h16 route builds them: tile-multiple operands (no pads, no split-K) with the out_f32 operand prologue attached.
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_h16"] (see
# _gemm_vp_common.py); gemm_vp_h16_emit.c builds the same specs through the C API.
# `python gemm_vp_h16_emit.py --c-cases` prints the case rows of gemm_vp_h16_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_h16")


if __name__ == "__main__":
    sys.exit(main())
