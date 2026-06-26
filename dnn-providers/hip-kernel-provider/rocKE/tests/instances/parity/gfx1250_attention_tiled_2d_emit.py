#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_attention_tiled_2d_emit.py -- Python reference emitter for
# the gfx1250 WMMA tiled-2D unified-attention parity harness.
from rocke.instances.gfx1250.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from _emit_common import run_emit


def _spec(idx: int) -> UnifiedAttention2DTiledSpec:
    base = dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
        tile_size=32,
    )
    if idx == 0:
        return UnifiedAttention2DTiledSpec(
            use_sinks=True, sliding_window=128, num_seqs=4, **base
        )
    if idx == 1:
        return UnifiedAttention2DTiledSpec(
            use_sinks=False, sliding_window=0, num_seqs=4, **base
        )
    if idx == 2:
        return UnifiedAttention2DTiledSpec(
            use_sinks=True, sliding_window=0, num_seqs=8, **base
        )
    if idx == 3:
        return UnifiedAttention2DTiledSpec(
            use_sinks=True, sliding_window=256, num_seqs=16, **base
        )
    if idx == 4:
        return UnifiedAttention2DTiledSpec(
            use_sinks=False, sliding_window=64, num_seqs=1, **base
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_unified_attention_2d_tiled(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_attention_tiled_2d_emit.py <config_index 0..4>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
