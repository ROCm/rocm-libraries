#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_pick_emit.py -- Python reference emitter of the gemm_vp_pick parity
# family: the gfx942 dispatcher's register-staged (vgpr_prefetch) RCR picks over a log grid of f16 / bf16 GEMM shapes (incl. the x6 and cf32 route shapes), deduplicated.
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_pick"] (see
# _gemm_vp_common.py); gemm_vp_pick_emit.c builds the same specs through the C API.
# `python gemm_vp_pick_emit.py --c-cases` prints the case rows of gemm_vp_pick_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_pick")


if __name__ == "__main__":
    sys.exit(main())
