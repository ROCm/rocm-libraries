#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx942_attention_tiled_3d_emit.py -- Python reference emitter for
# the gfx942 narrow-atom tiled split-KV 3D attention kernels parity harness.
#
# Selects one of the sampled configs by argv[1] and emits, in Python execution
# order, BOTH kernels the module exposes:
#   1. the per-segment SEGMENT kernel via
#      build_unified_attention_3d_tiled(spec, arch="gfx942")
#   2. the REDUCE kernel via
#      build_unified_attention_reduce_tiled(reduce_spec, arch="gfx942")
# whose UnifiedAttentionReduceTiledSpec is derived from the same config
# (head_size / num_query_heads / num_kv_heads / dtype / num_segments).
#
# Each kernel is lowered with _native_lower(kernel, arch="gfx942") and the two
# .ll texts are concatenated (segment first, then reduce) to stdout so they can
# be byte-compared with the C emitter gfx942_attention_tiled_3d_emit.c.
import sys

from kernels.gfx942.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


_CONFIGS = {
    0: dict(
        head_size=64,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=8,
        dtype="fp16",
        num_segments=8,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
    ),
    1: dict(
        head_size=128,
        block_size=32,
        num_query_heads=16,
        num_kv_heads=16,
        dtype="bf16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
    ),
    2: dict(
        head_size=256,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        num_segments=8,
        use_sinks=False,
        sliding_window=4096,
        has_softcap=False,
        kv_storage_dtype=None,
    ),
    3: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=16,
        use_sinks=True,
        sliding_window=0,
        has_softcap=True,
        kv_storage_dtype="fp8e4m3",
    ),
    4: dict(
        head_size=64,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=4,
        dtype="fp16",
        num_segments=8,
        use_sinks=False,
        sliding_window=2048,
        has_softcap=False,
        use_alibi=True,
        use_qq_bias=False,
        kv_storage_dtype=None,
    ),
    5: dict(
        # All-decode grid (seq_idx = block_id_x, no search). Decode-shaped GQA.
        head_size=128,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_decode_grid=True,
    ),
    6: dict(
        # All-decode grid at NQK=16 (BLOCK_Q=1) with fp8 KV and sinks.
        head_size=128,
        block_size=32,
        num_query_heads=16,
        num_kv_heads=1,
        dtype="bf16",
        num_segments=32,
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
        use_decode_grid=True,
    ),
    7: dict(
        # D256 at block 64 (dispatch tile T=32): fits gfx942's 64 KiB LDS only
        # with one V slot.
        head_size=256,
        block_size=64,
        num_query_heads=16,
        num_kv_heads=2,
        dtype="bf16",
        num_segments=8,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        tile_size_override=32,
    ),
    8: dict(
        # Decode K/V streamed (nt) through the wide loader.
        head_size=128,
        block_size=16,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_wide_kv_load=True,
        use_decode_grid=True,
        kv_cache_policy="stream",
    ),
    9: dict(
        # Decode K/V with SC0 + NT through the 1-DWORD async loader, with sinks.
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        num_segments=16,
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_decode_grid=True,
        kv_cache_policy="nt",
    ),
    10: dict(
        # Decode K/V streamed through the wide loader into swizzled LDS tiles.
        head_size=128,
        block_size=16,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_wide_kv_load=True,
        use_decode_grid=True,
        kv_cache_policy="stream",
        use_kv_lds_swizzle=True,
    ),
    11: dict(
        # Swizzled LDS tiles at head size 64, with sinks and softcap.
        head_size=64,
        block_size=32,
        num_query_heads=16,
        num_kv_heads=4,
        dtype="fp16",
        num_segments=8,
        use_sinks=True,
        sliding_window=0,
        has_softcap=True,
        kv_storage_dtype=None,
        use_wide_kv_load=True,
        use_kv_lds_swizzle=True,
    ),
    12: dict(
        # Decode K/V streamed into swizzled LDS tiles, addressed from the wave's
        # page-id window.
        head_size=128,
        block_size=16,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_wide_kv_load=True,
        use_decode_grid=True,
        kv_cache_policy="stream",
        use_kv_lds_swizzle=True,
        use_wave_page_id_load=True,
    ),
    13: dict(
        # The wave's page-id window without the swizzle, at head size 64 with a
        # sliding window.
        head_size=64,
        block_size=32,
        num_query_heads=16,
        num_kv_heads=4,
        dtype="fp16",
        num_segments=8,
        use_sinks=False,
        sliding_window=64,
        has_softcap=False,
        kv_storage_dtype=None,
        use_wide_kv_load=True,
        use_wave_page_id_load=True,
    ),
    14: dict(
        # Decode K/V streamed by the async copies into swizzled LDS tiles.
        head_size=256,
        block_size=16,
        num_query_heads=64,
        num_kv_heads=64,
        dtype="bf16",
        num_segments=32,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_decode_grid=True,
        kv_cache_policy="stream",
        use_kv_lds_swizzle=True,
    ),
    15: dict(
        # Decode K/V streamed by the async copies into swizzled LDS tiles, addressed
        # from the wave's page-id window.
        head_size=256,
        block_size=16,
        num_query_heads=64,
        num_kv_heads=64,
        dtype="bf16",
        num_segments=32,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_decode_grid=True,
        kv_cache_policy="stream",
        use_kv_lds_swizzle=True,
        use_wave_page_id_load=True,
    ),
    16: dict(
        # The wave's page-id window with half-page tiles on the wide loader.
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        tile_size_override=16,
        use_wide_kv_load=True,
        use_wave_page_id_load=True,
    ),
    17: dict(
        # The wave's page-id window with half-page tiles on the async copies.
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=8,
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        tile_size_override=16,
        use_wave_page_id_load=True,
    ),
}


def _kernels(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]
    seg_kernel = build_unified_attention_3d_tiled(
        UnifiedAttention3DTiledSpec(**cfg), arch="gfx942"
    )
    red_spec = UnifiedAttentionReduceTiledSpec(
        head_size=cfg["head_size"],
        num_query_heads=cfg["num_query_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        dtype=cfg["dtype"],
        num_segments=cfg["num_segments"],
    )
    red_kernel = build_unified_attention_reduce_tiled(red_spec, arch="gfx942")
    return seg_kernel, red_kernel


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx942_attention_tiled_3d_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    for kernel in _kernels(idx):
        if mode == "ll":
            sys.stdout.write(_native_lower(kernel, arch="gfx942"))
        elif mode == "ir":
            sys.stdout.write(serialize(kernel))
        else:
            sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
