#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1201_wmma_fmha_fwd_emit.py -- Python reference emitter for the
# gfx1201 (RDNA4) WMMA FMHA forward instance parity harness. The first 12 of the
# gfx1151 harness's sampled WmmaFmhaFwdSpec configs, built and lowered at
# arch='gfx1201' so the RDNA4 split-K WMMA attention path is byte-compared
# C-vs-Python. Configs 0-5 cover the fp16 atom (wmma_gfx12_f32_16x16x16_f16,
# <8 x half> fragments); configs 6-11 repeat the same six specs as bf16, which
# covers the bf16 atom (wmma_gfx12_f32_16x16x16_bf16, <8 x i16> fragments) --
# including bf16 crossed with v_lds_stage (config 11), the one knob whose LDS
# allocation, stores and loads are typed by dtype.
# Selects a config by argv[1] (0..11).
from dataclasses import replace

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, build_wmma_fmha_fwd
from _emit_common import run_emit


def _spec(idx: int) -> WmmaFmhaFwdSpec:
    if 6 <= idx < 12:
        return replace(_spec(idx - 6), dtype="bf16")
    if idx == 0:
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 1:
        return WmmaFmhaFwdSpec(
            head_size=128,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 2:
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="causal",
            v_lds_stage=False,
        )
    if idx == 3:
        return WmmaFmhaFwdSpec(
            head_size=256,
            mask_mode="none",
            v_lds_stage=False,
        )
    if idx == 4:
        return WmmaFmhaFwdSpec(
            head_size=128,
            mask_mode="causal",
            v_lds_stage=False,
        )
    if idx == 5:
        return WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="none",
            v_lds_stage=True,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_wmma_fmha_fwd,
        usage="usage: gfx1201_wmma_fmha_fwd_emit.py <config_index 0..11>\n",
        arch="gfx1201",
    )


if __name__ == "__main__":
    raise SystemExit(main())
