#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx950_attention_tiled_3d_emit.py -- Python reference emitter for
# the gfx950 (CDNA4) WIDE-K tiled split-KV 3D attention kernels parity harness.
#
# Selects one of the sampled configs by argv[1] and emits, in Python execution
# order, BOTH kernels the module exposes:
#   1. the per-segment SEGMENT kernel via
#      build_unified_attention_3d_tiled(spec, arch="gfx950")
#   2. the arch-neutral REDUCE kernel via
#      build_unified_attention_reduce_tiled(reduce_spec, arch="gfx950")
# whose UnifiedAttentionReduceTiledSpec is derived from the same config
# (head_size / num_query_heads / num_kv_heads / dtype / num_segments).
#
# Each kernel is lowered with _native_lower(kernel, arch="gfx950") and
# the two .ll texts are concatenated (segment first, then reduce) to stdout so
# they can be byte-compared with the C emitter gfx950_attention_tiled_3d_emit.c.
import sys

from rocke.instances.gfx950.attention_tiled_3d import (
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
        # 64-bit paged-KV addressing (caches > 2 GiB). Decode-shaped MHA.
        head_size=128,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=16,
        dtype="fp16",
        num_segments=16,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype=None,
        use_i64_kv_addr=True,
    ),
}


def _emit(idx: int) -> str:
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]

    seg_spec = UnifiedAttention3DTiledSpec(**cfg)
    seg_kernel = build_unified_attention_3d_tiled(seg_spec, arch="gfx950")
    seg_ll = _native_lower(seg_kernel, arch="gfx950")

    red_spec = UnifiedAttentionReduceTiledSpec(
        head_size=cfg["head_size"],
        num_query_heads=cfg["num_query_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        dtype=cfg["dtype"],
        num_segments=cfg["num_segments"],
    )
    red_kernel = build_unified_attention_reduce_tiled(red_spec, arch="gfx950")
    red_ll = _native_lower(red_kernel, arch="gfx950")

    return seg_ll + red_ll


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx950_attention_tiled_3d_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode == "ll":
        sys.stdout.write(_emit(idx))
    elif mode == "ir":
        cfg = _CONFIGS[idx] if idx in _CONFIGS else None
        if cfg is None:
            raise SystemExit(f"unknown config index {idx}")
        seg_spec = UnifiedAttention3DTiledSpec(**cfg)
        seg_kernel = build_unified_attention_3d_tiled(seg_spec, arch="gfx950")
        red_spec = UnifiedAttentionReduceTiledSpec(
            head_size=cfg["head_size"],
            num_query_heads=cfg["num_query_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            dtype=cfg["dtype"],
            num_segments=cfg["num_segments"],
        )
        red_kernel = build_unified_attention_reduce_tiled(red_spec, arch="gfx950")
        sys.stdout.write(serialize(seg_kernel))
        sys.stdout.write(serialize(red_kernel))
    elif mode == "verify":
        cfg = _CONFIGS[idx] if idx in _CONFIGS else None
        if cfg is None:
            raise SystemExit(f"unknown config index {idx}")
        seg_spec = UnifiedAttention3DTiledSpec(**cfg)
        seg_kernel = build_unified_attention_3d_tiled(seg_spec, arch="gfx950")
        red_spec = UnifiedAttentionReduceTiledSpec(
            head_size=cfg["head_size"],
            num_query_heads=cfg["num_query_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            dtype=cfg["dtype"],
            num_segments=cfg["num_segments"],
        )
        red_kernel = build_unified_attention_reduce_tiled(red_spec, arch="gfx950")
        sys.stdout.write("".join(str(d) + "\n" for d in verify(seg_kernel)))
        sys.stdout.write("".join(str(d) + "\n" for d in verify(red_kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
