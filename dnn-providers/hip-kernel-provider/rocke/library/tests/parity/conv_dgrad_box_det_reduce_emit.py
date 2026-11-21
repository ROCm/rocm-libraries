#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_dgrad_box_det_reduce_emit.py -- Python reference emitter
# for the dgrad pp_box_det Stage 2 (conv_dgrad_box_det_reduce) parity harness.
# Selects one of N sampled spec configs by argv[1], builds the kernel via
# build_dgrad_box_det_reduce and prints the LLVM IR, to be byte-compared with
# the C emitter conv_dgrad_box_det_reduce_emit.c.
#
# Config index map (gfx942; keep in lockstep with the C emitter):
#   0  t128x256 w2x4, f32 slots, msplit 1
#   1  t256x256 w4x2, msplit 4, pix_pre (the AOT default form)
#   2  t256x256 w4x2, bf16 slots (pp_box_det 2), msplit 2, pix_pre
#   3  t256x256 w4x2, m_chunks 2
#   4  t128x256 w2x4, wave_wg, unroll 2, load_aux 2
#   5  t128x256 w2x4, msplit 4, pix_pre, bf16 slots
#   6  t128x256 w2x4, unroll 8 (the most slices per trip)
#   7  t128x256 w2x4, unroll 9: rejected by both engines
from kernels.common.conv_dgrad_box_det import (
    DgradBoxDetReduceSpec,
    build_dgrad_box_det_reduce,
)
from _emit_common import run_emit

_T128 = dict(tile_m=128, tile_n=256, warp_m=2, warp_n=4)
_T256 = dict(tile_m=256, tile_n=256, warp_m=4, warp_n=2)
_CFGS = {
    0: dict(_T128),
    1: dict(_T256, msplit=4, pix_pre=True),
    2: dict(_T256, slot_bytes=2, msplit=2, pix_pre=True),
    3: dict(_T256, m_chunks=2),
    4: dict(_T128, wave_wg=True, unroll=2, load_aux=2),
    5: dict(_T128, msplit=4, pix_pre=True, slot_bytes=2),
    6: dict(_T128, unroll=8),
    7: dict(_T128, unroll=9),
}


def _spec(idx: int):
    if idx not in _CFGS:
        raise SystemExit(f"unknown config index {idx}")
    return DgradBoxDetReduceSpec(**_CFGS[idx]), "gfx942"


def main() -> int:
    return run_emit(
        _spec,
        build_dgrad_box_det_reduce,
        usage="usage: conv_dgrad_box_det_reduce_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
