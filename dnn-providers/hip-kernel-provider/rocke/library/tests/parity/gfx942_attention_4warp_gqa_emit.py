#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx942_attention_4warp_gqa_emit.py -- Python reference emitter for
# the gfx942 4-warp GQA paged-attention builder. Selects one config by argv[1],
# builds a UnifiedAttention2DTiledSpec, emits the kernel via
# build_gfx942_4warp_gqa(spec, arch="gfx942") and prints the .ll so it can be
# byte-compared with the C emitter gfx942_attention_4warp_gqa_emit.c.
#
# The specs are the ones _tiled_spec_from_problem produces for the routed
# cohorts (num_warps=1, tile_size=max(32, block_size), block_m_per_warp=32,
# waves_per_eu=2), covering the D256 lean body, the D128 double-buffer pipe with
# and without the GQA head-fold, and the D128 block_size=64 single-buffer body.
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_gfx942_4warp_gqa,
)
from _emit_common import run_emit

_ROUTE = dict(
    use_sinks=False,
    has_softcap=False,
    num_warps=1,
    block_m_per_warp=32,
    waves_per_eu=2,
)


_CONFIGS = {
    # D256 lean, the rocke_ir_parity_harness D256 pin.
    0: dict(
        _ROUTE,
        head_size=256,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=2,
        dtype="bf16",
        sliding_window=0,
        num_seqs=1,
        tile_size=32,
        use_register_pv=True,
    ),
    # D256 lean, bs32 (one block per 32-key half-tile), multi-sequence search.
    1: dict(
        _ROUTE,
        head_size=256,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        sliding_window=0,
        num_seqs=4,
        tile_size=32,
    ),
    # D128 pipe + head-fold, the rocke_ir_parity_harness fold pin.
    2: dict(
        _ROUTE,
        head_size=128,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        sliding_window=4096,
        num_seqs=1,
        tile_size=32,
    ),
    3: dict(
        _ROUTE,
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        sliding_window=1024,
        num_seqs=4,
        tile_size=32,
    ),
    # D128 pipe, no fold: fp16 (the fold is bf16-only), f16 MFMA atom.
    4: dict(
        _ROUTE,
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        sliding_window=4096,
        num_seqs=1,
        tile_size=32,
    ),
    # D128 pipe, no fold: 5:1 GQA.
    5: dict(
        _ROUTE,
        head_size=128,
        block_size=16,
        num_query_heads=40,
        num_kv_heads=8,
        dtype="bf16",
        sliding_window=4096,
        num_seqs=1,
        tile_size=32,
    ),
    # D128 bs64: single-buffer BN=64 body with the window mask.
    6: dict(
        _ROUTE,
        head_size=128,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        sliding_window=4096,
        num_seqs=1,
        tile_size=64,
    ),
    7: dict(
        _ROUTE,
        head_size=128,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        sliding_window=2048,
        num_seqs=2,
        tile_size=64,
    ),
    # D128 pipe, causal and no waves_per_eu: not routed today, but the builder
    # admits it and it is the only config reaching the window == 0 branches of
    # the general body.
    8: dict(
        _ROUTE,
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        sliding_window=0,
        num_seqs=1,
        tile_size=32,
        waves_per_eu=None,
    ),
}


def _kernel(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    spec = UnifiedAttention2DTiledSpec(**_CONFIGS[idx])
    return build_gfx942_4warp_gqa(spec, arch="gfx942")


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _kernel,
        _emit_build,
        usage="usage: gfx942_attention_4warp_gqa_emit.py <config_index>\n",
        arch="gfx942",
    )


if __name__ == "__main__":
    raise SystemExit(main())
