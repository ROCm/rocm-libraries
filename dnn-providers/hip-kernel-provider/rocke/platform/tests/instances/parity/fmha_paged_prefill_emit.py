#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_paged_prefill_emit.py -- Python reference emitter for the
# paged-KV prefill FMHA-fwd parity harness. Selects one of 6 sampled
# FmhaFwdPagedPrefillSpec configs by argv[1] (0..5), builds via
# build_fmha_fwd_paged_prefill and prints lower_kernel_to_llvm(arch='gfx950')
# to stdout so it can be byte-compared with the C emitter
# fmha_paged_prefill_emit.c.
from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.instances.common.fmha_paged_prefill import (
    FmhaFwdPagedPrefillSpec,
    build_fmha_fwd_paged_prefill,
)
from _emit_common import run_emit


def _spec(idx: int) -> FmhaFwdPagedPrefillSpec:
    if idx == 0:
        return FmhaFwdPagedPrefillSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
                dtype="f16",
                mask_mode="none",
            ),
            page_block_size=16,
            max_blocks_per_seq=32,
            batch=2,
            use_mfma_body=False,
        )
    if idx == 1:
        return FmhaFwdPagedPrefillSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=8),
                dtype="f16",
                mask_mode="causal",
            ),
            page_block_size=32,
            max_blocks_per_seq=64,
            batch=4,
            use_mfma_body=False,
        )
    if idx == 2:
        return FmhaFwdPagedPrefillSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=256, num_query_heads=8, num_kv_heads=2),
                dtype="bf16",
                mask_mode="sliding_window",
                sliding_window=2048,
            ),
            page_block_size=64,
            max_blocks_per_seq=128,
            batch=8,
            use_mfma_body=True,
        )
    if idx == 3:
        return FmhaFwdPagedPrefillSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=32, num_kv_heads=8),
                dtype="f16",
                mask_mode="causal",
            ),
            page_block_size=128,
            max_blocks_per_seq=256,
            batch=16,
            use_mfma_body=False,
        )
    if idx == 4:
        return FmhaFwdPagedPrefillSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=16, num_kv_heads=2),
                dtype="bf16",
                mask_mode="none",
            ),
            page_block_size=256,
            max_blocks_per_seq=512,
            batch=1,
            use_mfma_body=True,
        )
    if idx == 5:
        return FmhaFwdPagedPrefillSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=192, num_query_heads=12, num_kv_heads=12),
                dtype="f16",
                mask_mode="sliding_window",
                sliding_window=1024,
            ),
            page_block_size=32,
            max_blocks_per_seq=64,
            batch=4,
            use_mfma_body=False,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_fmha_fwd_paged_prefill,
        usage="usage: fmha_paged_prefill_emit.py <config_index 0..5>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
