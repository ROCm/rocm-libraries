#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_fmha_fwd_emit.py -- Python reference emitter for the
# gfx1151 (RDNA3.5 / Strix Halo) WMMA FMHA forward instance parity harness.
# Selects one of 86 sampled configurations by argv[1] (0..85), builds it
# via build_wmma_fmha_fwd(arch='gfx1151') and prints
# lower_kernel_to_llvm(kernel, arch='gfx1151') to stdout so it can be
# byte-compared with the C emitter gfx1151_wmma_fmha_fwd_emit.c.
from dataclasses import replace

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, build_wmma_fmha_fwd
from _emit_common import run_emit


def _spec(idx: int) -> WmmaFmhaFwdSpec:
    if 70 <= idx < 86:
        variant = idx - 70
        return WmmaFmhaFwdSpec(
            head_size=64 if variant < 8 else 128, num_query_heads=8, num_kv_heads=2,
            mask_mode="none" if variant % 8 < 4 else "causal",
            transposed_qk=True, block_n=32 if variant % 4 < 2 else 64,
            num_waves=1 + variant % 2,
        )
    if 58 <= idx < 70:
        bases = (0, 7, 14, 15, 22, 23, 24, 25, 44, 47, 51, 57)
        return replace(_spec(bases[idx - 58]), kv_dtype="fp8e4m3")
    if 42 <= idx < 58:
        bases = (0, 7, 12, 15, 16, 39, 0, 7, 12, 15, 16, 17, 26, 33, 38, 39)
        pages = (0, 0, 0, 0, 0, 0, 16, 64, 16, 32, 32, 64, 64, 32, 16, 64)
        page = pages[idx - 42]
        return replace(
            _spec(bases[idx - 42]), layout="paged" if page else "ragged",
            page_block_size=page,
        )
    if 26 <= idx < 42:
        feature_cases = (
            (12, dict(sliding_window=128)), (13, dict(sliding_window=128)),
            (2, dict(use_softcap=True)), (8, dict(use_softcap=True)),
            (2, dict(use_sinks=True)), (8, dict(use_sinks=True)),
            (12, dict(use_alibi=True)), (13, dict(use_alibi=True)),
            (0, dict(use_qq_bias=True)), (6, dict(use_qq_bias=True)),
            (18, dict(sliding_window=128, use_sinks=True)),
            (19, dict(sliding_window=128, use_sinks=True)),
            (24, dict(sliding_window=64, use_softcap=True, use_sinks=True, use_alibi=True, use_qq_bias=True)),
            (25, dict(sliding_window=64, use_softcap=True, use_sinks=True, use_alibi=True, use_qq_bias=True)),
            (2, dict(use_softcap=True, use_alibi=True)),
            (8, dict(use_softcap=True, use_alibi=True)),
        )
        base, features = feature_cases[idx - 26]
        return replace(_spec(base), **features)
    if 18 <= idx < 26:
        bases = (12, 13, 3, 9, 12, 13, 16, 17)
        return replace(
            _spec(bases[idx - 18]),
            query_tail=idx < 20 or idx >= 22,
            kv_tail=idx >= 20,
        )
    if 12 <= idx < 18:
        return replace(
            _spec((2, 8, 3, 9, 5, 11)[idx - 12]),
            mask_mode="causal", causal_bottom_right=True,
        )
    if 6 <= idx < 12:
        return replace(_spec(idx - 6), dtype="bf16")
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
        usage="usage: gfx1151_wmma_fmha_fwd_emit.py <config_index 0..85>\n",
        arch="gfx1151",
    )


if __name__ == "__main__":
    raise SystemExit(main())
