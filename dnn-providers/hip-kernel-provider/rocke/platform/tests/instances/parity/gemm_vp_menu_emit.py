#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_vp_menu_emit.py -- Python reference emitter of the gemm_vp_menu parity
# family: every gfx942 g5_plan menu geometry once without and once with the G5P_MAINLOOP main-loop levers (vp_pair_reads, vp_release_atoms, vp_tail_pack, vp_mfma_first).
# The configurations are the frozen table _gemm_vp_cases.CASES["gemm_vp_menu"] (see
# _gemm_vp_common.py); gemm_vp_menu_emit.c builds the same specs through the C API.
# `python gemm_vp_menu_emit.py --c-cases` prints the case rows of gemm_vp_menu_emit.c.
import sys

from _gemm_vp_common import main as _main


def main() -> int:
    return _main("gemm_vp_menu")


if __name__ == "__main__":
    sys.exit(main())
