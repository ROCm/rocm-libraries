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
# The config table is kept IN LOCKSTEP with the C make_spec() switch (same index
# -> same UnifiedAttention2DTiledSpec). This is the "edge / feature-flag" cluster:
# minimal dims, GQA ratios, and every feature-flag path (qq_bias, fp8 KV,
# register-PV, i64 KV addressing, the transposed 32x32 + grouped-KV2 softmax
# stack, early-V schedule, and the fast paged-KV descriptor).
import sys

from ck_dsl.instances.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


_CONFIGS = {
    # --- idx0-4: minimal dims, block_size=16, head_size {64,128,256}, GQA ratios
    0: dict(
        head_size=64,
        block_size=16,
        num_query_heads=1,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    1: dict(
        head_size=128,
        block_size=16,
        num_query_heads=1,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    2: dict(
        head_size=256,
        block_size=16,
        num_query_heads=1,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    3: dict(
        head_size=64,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    4: dict(
        head_size=64,
        block_size=16,
        num_query_heads=2,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    # --- idx5-14: baseline dtype / mask-feature / head-size / block-size variety
    5: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    6: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=2048,
        has_softcap=True,
    ),
    7: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=True,
        sliding_window=1,
        has_softcap=False,
    ),
    8: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=True,
    ),
    9: dict(
        head_size=256,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    10: dict(
        head_size=64,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    11: dict(
        head_size=64,
        block_size=32,
        num_query_heads=7,
        num_kv_heads=7,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    12: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    13: dict(
        head_size=64,
        block_size=32,
        num_query_heads=40,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    14: dict(
        head_size=64,
        block_size=32,
        num_query_heads=128,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    # --- idx15: QQ-bias feature flag
    15: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_qq_bias=True,
    ),
    # --- idx16,17: ALiBi / composite mask features
    16: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_alibi=True,
    ),
    17: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=512,
        has_softcap=True,
        use_alibi=True,
        use_qq_bias=True,
    ),
    # --- idx18: num_warps=8 (BLOCK_M=128), no tile_size
    18: dict(
        head_size=64,
        block_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=8,
    ),
    # --- idx19-26: num_warps / tile_size / waves_per_eu / num_seqs variety
    19: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=2,
    ),
    20: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
    ),
    21: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=64,
    ),
    22: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=128,
    ),
    23: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=2,
        tile_size=128,
    ),
    24: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        waves_per_eu=2,
    ),
    25: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
    ),
    26: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=257,
    ),
    # --- idx27: fp8 KV cache with native fp8 PV MFMA
    27: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
        use_fp8_mfma_pv=True,
    ),
    # --- idx28: 64-bit paged-KV addressing
    28: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_i64_kv_addr=True,
    ),
    # --- idx29: register-PV bf16 path
    29: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_register_pv=True,
    ),
    # --- idx30-32: fp8 KV (dequant), fp8 QK MFMA, mfma_32x32 base
    30: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
    ),
    31: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
        use_fp8_mfma_qk=True,
    ),
    32: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx33-35: transposed 32x32 + scalar-state + invariant-hoist + mask-once
    #               + grouped-KV2 softmax stack (bf16, BLOCK_M=128 warp slice)
    33: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_grouped_kv2_softmax=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    34: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_grouped_kv2_softmax=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    35: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_grouped_kv2_softmax=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx36: early-V schedule
    36: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_early_v_schedule=True,
    ),
    # --- idx37: fast paged-KV descriptor (bf16 h64kv8 HD=64 BS=32 T=64 nw=4)
    37: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=16,
        tile_size=64,
        use_fast_paged_kv_desc=True,
    ),
}


def _spec(idx: int) -> UnifiedAttention2DTiledSpec:
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    return UnifiedAttention2DTiledSpec(**_CONFIGS[idx])


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx950_attention_tiled_2d_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx950")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
