#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_attention_tiled_3d_emit.py -- Python reference emitter
# for the gfx1250 WMMA split-KV 3D decode attention *segment* kernel parity.
from rocke.instances.gfx1250.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    build_unified_attention_3d_tiled,
)
from _emit_common import run_emit


def _spec(idx: int) -> UnifiedAttention3DTiledSpec:
    base = dict(
        head_size=64,
        num_query_heads=32,
        num_kv_heads=4,
        dtype="bf16",
        has_softcap=False,
    )
    if idx == 0:
        return UnifiedAttention3DTiledSpec(
            block_size=16,
            use_sinks=True,
            sliding_window=0,
            num_segments=16,
            num_seqs=2,
            kv_storage_dtype="fp8e4m3",
            **base,
        )
    if idx == 1:
        return UnifiedAttention3DTiledSpec(
            block_size=32,
            use_sinks=True,
            sliding_window=128,
            num_segments=8,
            num_seqs=2,
            kv_storage_dtype="fp8e4m3",
            **base,
        )
    if idx == 2:
        return UnifiedAttention3DTiledSpec(
            block_size=16,
            use_sinks=False,
            sliding_window=0,
            num_segments=4,
            num_seqs=4,
            kv_storage_dtype="bf16",
            **base,
        )
    if idx == 3:
        return UnifiedAttention3DTiledSpec(
            block_size=32,
            use_sinks=True,
            sliding_window=0,
            num_segments=16,
            num_seqs=2,
            kv_storage_dtype="bf16",
            **base,
        )
    if idx == 4:
        return UnifiedAttention3DTiledSpec(
            block_size=32,
            use_sinks=False,
            sliding_window=64,
            num_segments=8,
            num_seqs=1,
            kv_storage_dtype="fp8e4m3",
            **base,
        )
    if idx == 5:
        return UnifiedAttention3DTiledSpec(
            block_size=16,
            use_sinks=True,
            sliding_window=128,
            num_segments=4,
            num_seqs=8,
            kv_storage_dtype="fp8e4m3",
            **base,
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_unified_attention_3d_tiled(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_attention_tiled_3d_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
