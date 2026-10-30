#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_sk_emit.py -- Python reference emitter of the gemm_vp_sk parity
# family: split_k_aligned and split_k_serial outside the register-staged body: the global-load compv4 body (whole-tile slices with pad_k) and the direct_to_lds ring.
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_sk"] (see
# _gemm_vp_common.py); gemm_vp_sk_emit.c builds the same specs through the C API.
# `python gemm_vp_sk_emit.py --c-cases` prints the case rows of gemm_vp_sk_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_sk")


if __name__ == "__main__":
    sys.exit(main())
