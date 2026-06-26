#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_attention_reduce_tiled_emit.py -- Python reference
# emitter for the gfx1250 split-KV reduce kernel parity harness.
from rocke.instances.gfx1250.attention_tiled_3d import (
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_reduce_tiled,
)
from _emit_common import run_emit


def _spec(idx: int) -> UnifiedAttentionReduceTiledSpec:
    base = dict(
        head_size=64,
        dtype="bf16",
    )
    if idx == 0:
        return UnifiedAttentionReduceTiledSpec(
            num_query_heads=32, num_kv_heads=4, num_segments=16, **base
        )
    if idx == 1:
        return UnifiedAttentionReduceTiledSpec(
            num_query_heads=32, num_kv_heads=4, num_segments=8, **base
        )
    if idx == 2:
        return UnifiedAttentionReduceTiledSpec(
            num_query_heads=32, num_kv_heads=4, num_segments=4, **base
        )
    if idx == 3:
        return UnifiedAttentionReduceTiledSpec(
            num_query_heads=64, num_kv_heads=8, num_segments=16, **base
        )
    if idx == 4:
        return UnifiedAttentionReduceTiledSpec(
            num_query_heads=32, num_kv_heads=4, num_segments=2, **base
        )
    if idx == 5:
        return UnifiedAttentionReduceTiledSpec(
            num_query_heads=16, num_kv_heads=2, num_segments=32, **base
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_unified_attention_reduce_tiled(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_attention_reduce_tiled_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
