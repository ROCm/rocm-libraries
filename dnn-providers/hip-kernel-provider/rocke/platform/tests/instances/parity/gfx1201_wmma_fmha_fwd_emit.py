#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1201_wmma_fmha_fwd_emit.py -- Python reference emitter for the
# gfx1201 (RDNA4 / Navi 48) WMMA FMHA forward instance parity harness. Same
# sampled WmmaFmhaFwdSpec configs as the gfx1151 harness, but built and lowered
# at arch='gfx1201' so the RDNA4 split-K WMMA attention path (the
# wmma_gfx12_f32_16x16x16_f16 atom, <8 x half> fragments) is byte-compared
# C-vs-Python. Selects a config by argv[1] (0..5).
from rocke.instances.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, build_wmma_fmha_fwd
from _emit_common import run_emit


def _spec(idx: int) -> WmmaFmhaFwdSpec:
    if idx == 0:
        return WmmaFmhaFwdSpec(
            head_size=64,
            num_query_heads=4,
            num_kv_heads=0,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 1:
        return WmmaFmhaFwdSpec(
            head_size=128,
            num_query_heads=8,
            num_kv_heads=0,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 2:
        return WmmaFmhaFwdSpec(
            head_size=64,
            num_query_heads=4,
            num_kv_heads=0,
            mask_mode="causal",
            v_lds_stage=False,
        )
    if idx == 3:
        return WmmaFmhaFwdSpec(
            head_size=256,
            num_query_heads=8,
            num_kv_heads=2,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 4:
        return WmmaFmhaFwdSpec(
            head_size=128,
            num_query_heads=4,
            num_kv_heads=4,
            mask_mode="causal",
            v_lds_stage=False,
        )
    if idx == 5:
        return WmmaFmhaFwdSpec(
            head_size=64,
            num_query_heads=6,
            num_kv_heads=0,
            mask_mode="none",
            v_lds_stage=True,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_wmma_fmha_fwd,
        usage="usage: gfx1201_wmma_fmha_fwd_emit.py <config_index 0..5>\n",
        arch="gfx1201",
    )


if __name__ == "__main__":
    raise SystemExit(main())
