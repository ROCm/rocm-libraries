#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx942_attention_tiled_2d_emit.py -- Python reference emitter for
# the gfx942 narrow-atom tiled-2D unified-attention kernel parity harness.
# Selects one of the sampled configs by argv[1], builds a
# UnifiedAttention2DTiledSpec, emits the kernel via
# build_unified_attention_2d_tiled(spec, arch="gfx942") and prints
# lower_kernel_to_llvm(kernel, arch="gfx942") to stdout so it can be
# byte-compared with the C emitter gfx942_attention_tiled_2d_emit.c.
from ck_dsl.instances.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from _emit_common import run_emit


_CONFIGS = {
    0: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=True,
        sliding_window=2048,
        has_softcap=False,
    ),
    1: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=True,
    ),
    2: dict(
        head_size=64,
        block_size=128,
        num_query_heads=16,
        num_kv_heads=4,
        dtype="bf16",
        use_sinks=False,
        sliding_window=4096,
        has_softcap=False,
    ),
    3: dict(
        head_size=128,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=True,
        sliding_window=2048,
        has_softcap=True,
    ),
    4: dict(
        head_size=256,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
}


def _kernel(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    spec = UnifiedAttention2DTiledSpec(**_CONFIGS[idx])
    return build_unified_attention_2d_tiled(spec, arch="gfx942")


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _kernel,
        _emit_build,
        usage="usage: gfx942_attention_tiled_2d_emit.py <config_index>\n",
        arch="gfx942",
    )


if __name__ == "__main__":
    raise SystemExit(main())
