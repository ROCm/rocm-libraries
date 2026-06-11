#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx950_attention_tiled_2d_emit.py -- Python reference emitter for
# the gfx950 WIDE-ATOM tiled-2D unified-attention kernel parity harness.
#
# Selects one of the sampled configs by argv[1], builds a
# UnifiedAttention2DTiledSpec, emits the kernel via
# build_unified_attention_2d_tiled(spec, arch="gfx950") and prints
# lower_kernel_to_llvm(kernel, arch="gfx950") to stdout so it can be
# byte-compared with the C emitter gfx950_attention_tiled_2d_emit.c.
#
# The configs are kept IN LOCKSTEP with the C make_spec() switch. The configs
# specify num_queries_per_kv (= num_query_heads // num_kv_heads); each entry
# below sets num_query_heads / num_kv_heads to realise that ratio identically on
# both sides.
import sys

from ck_dsl.instances.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> UnifiedAttention2DTiledSpec:
    if idx == 0:
        # head_size=128 block_size=64 nqpkv=8 bf16 sw=0 no-softcap nw=1
        return UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=64,
            num_query_heads=8,
            num_kv_heads=1,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_warps=1,
        )
    if idx == 1:
        # head_size=64 block_size=32 nqpkv=8 fp16 sw=0 no-softcap nw=4
        # mfma32 + transposed_qk_32x32, tile_size=64 (block_m_per_warp=32)
        return UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_warps=4,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            tile_size=64,
        )
    if idx == 2:
        # head_size=128 block_size=32 nqpkv=4 bf16 sw=2048 softcap nw=2
        # mfma32, tile_size=64 (block_m_per_warp=32)
        return UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=32,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=False,
            sliding_window=2048,
            has_softcap=True,
            num_warps=2,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            tile_size=64,
        )
    if idx == 3:
        # head_size=64 block_size=64 nqpkv=1 bf16 kv=fp8e4m3 sw=0 nw=1
        # use_fp8_mfma_qk
        return UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=64,
            num_query_heads=8,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_warps=1,
            kv_storage_dtype="fp8e4m3",
            use_fp8_mfma_qk=True,
        )
    if idx == 4:
        # head_size=256 block_size=64 nqpkv=16 fp16 sw=0 no-softcap nw=4
        # block_m_per_warp=32, tile_size=128 (no mfma32)
        return UnifiedAttention2DTiledSpec(
            head_size=256,
            block_size=64,
            num_query_heads=32,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_warps=4,
            block_m_per_warp=32,
            tile_size=128,
        )
    if idx == 5:
        # head_size=128 block_size=16 nqpkv=2 bf16 sw=512 no-softcap nw=1
        # use_register_pv
        return UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=False,
            sliding_window=512,
            has_softcap=False,
            num_warps=1,
            use_register_pv=True,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx950_attention_tiled_2d_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
