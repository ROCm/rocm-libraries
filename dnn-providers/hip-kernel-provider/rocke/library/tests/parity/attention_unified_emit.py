#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_unified_emit.py -- Python reference emitter for the
# unified scalar attention parity harness (2D, 3D segment and reduce kernels).
# Selects one of the sampled configs by argv[1], builds it and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter attention_unified_emit.c.
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
)
from _emit_common import run_emit

# num_seqs is a runtime kernel param (not a build-time constant) in the scalar
# builders, so its value does not affect the emitted IR. Configs 0-5 and 6-10 are
# 2D kernels, 11-15 3D segment kernels (num_segments from _SEGMENTS), 16-17
# reduce kernels.
_CONFIGS = {
    0: dict(
        head_size=64,
        block_size=32,
        dtype="bf16",
        num_query_heads=64,
        num_kv_heads=8,
        total_q=128,
        max_seqlen_q=512,
        max_seqlen_k=512,
        use_sinks=True,
        softcap=0.0,
        num_seqs=1,
    ),
    1: dict(
        head_size=128,
        block_size=32,
        dtype="bf16",
        num_query_heads=64,
        num_kv_heads=8,
        total_q=256,
        max_seqlen_q=1024,
        max_seqlen_k=1024,
        use_sinks=True,
        softcap=0.0,
        num_seqs=1,
    ),
    2: dict(
        head_size=256,
        block_size=16,
        dtype="fp16",
        num_query_heads=32,
        num_kv_heads=4,
        total_q=64,
        max_seqlen_q=256,
        max_seqlen_k=256,
        use_sinks=False,
        softcap=0.0,
        num_seqs=1,
    ),
    3: dict(
        head_size=64,
        block_size=16,
        dtype="bf16",
        num_query_heads=16,
        num_kv_heads=2,
        total_q=512,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        use_sinks=True,
        sliding_window=128,
        softcap=0.0,
        num_seqs=1,
    ),
    4: dict(
        head_size=128,
        block_size=32,
        dtype="bf16",
        num_query_heads=64,
        num_kv_heads=8,
        total_q=1,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        use_sinks=False,
        softcap=0.0,
        num_seqs=1,
    ),
    5: dict(
        head_size=64,
        block_size=32,
        dtype="bf16",
        num_query_heads=128,
        num_kv_heads=16,
        total_q=1024,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        use_sinks=True,
        softcap=50.0,
        num_seqs=1,
    ),
    6: dict(
        head_size=128,
        block_size=32,
        dtype="bf16",
        num_query_heads=32,
        num_kv_heads=8,
        total_q=64,
        max_seqlen_q=64,
        max_seqlen_k=1024,
        use_fp8=True,
        num_seqs=1,
    ),
    7: dict(
        head_size=64,
        block_size=16,
        dtype="fp16",
        num_query_heads=16,
        num_kv_heads=4,
        total_q=256,
        max_seqlen_q=256,
        max_seqlen_k=512,
        sliding_window=64,
        causal_top_left=True,
        num_seqs=1,
    ),
    8: dict(
        head_size=128,
        block_size=32,
        dtype="bf16",
        num_query_heads=64,
        num_kv_heads=8,
        total_q=128,
        max_seqlen_q=128,
        max_seqlen_k=1024,
        use_sinks=True,
        right_bound=5,
        num_seqs=1,
    ),
    9: dict(
        head_size=64,
        block_size=64,
        dtype="bf16",
        num_query_heads=8,
        num_kv_heads=8,
        total_q=512,
        max_seqlen_q=512,
        max_seqlen_k=512,
        softcap=30.0,
        right_bound=-1,
        num_seqs=1,
    ),
    10: dict(
        head_size=256,
        block_size=16,
        dtype="fp16",
        num_query_heads=16,
        num_kv_heads=2,
        total_q=32,
        max_seqlen_q=32,
        max_seqlen_k=2048,
        use_fp8=True,
        causal_top_left=True,
        right_bound=3,
        num_seqs=1,
    ),
    11: dict(
        head_size=128,
        block_size=16,
        dtype="bf16",
        num_query_heads=32,
        num_kv_heads=8,
        total_q=4,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        num_seqs=4,
    ),
    12: dict(
        head_size=64,
        block_size=32,
        dtype="fp16",
        num_query_heads=16,
        num_kv_heads=16,
        total_q=8,
        max_seqlen_q=2,
        max_seqlen_k=8192,
        sliding_window=256,
        num_seqs=4,
    ),
    13: dict(
        head_size=64,
        block_size=16,
        dtype="bf16",
        num_query_heads=64,
        num_kv_heads=8,
        total_q=4,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        sliding_window=128,
        causal_top_left=True,
        num_seqs=4,
    ),
    14: dict(
        head_size=128,
        block_size=32,
        dtype="fp16",
        num_query_heads=32,
        num_kv_heads=4,
        total_q=16,
        max_seqlen_q=4,
        max_seqlen_k=4096,
        use_sinks=True,
        right_bound=-1,
        num_seqs=4,
    ),
    15: dict(
        head_size=256,
        block_size=64,
        dtype="bf16",
        num_query_heads=16,
        num_kv_heads=2,
        total_q=2,
        max_seqlen_q=1,
        max_seqlen_k=8192,
        right_bound=7,
        num_seqs=2,
    ),
    16: dict(
        head_size=128,
        block_size=16,
        dtype="bf16",
        num_query_heads=32,
        num_kv_heads=8,
        total_q=4,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        num_seqs=4,
    ),
    17: dict(
        head_size=64,
        block_size=32,
        dtype="fp16",
        num_query_heads=16,
        num_kv_heads=16,
        total_q=8,
        max_seqlen_q=2,
        max_seqlen_k=8192,
        num_seqs=4,
    ),
}

_SEGMENTS = {11: 8, 12: 4, 13: 8, 14: 16, 15: 8, 16: 8, 17: 16}


def _kernel(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    problem = UnifiedAttentionProblem(**_CONFIGS[idx])
    if idx >= 16:
        spec = UnifiedAttentionReduceSpec(problem=problem, num_segments=_SEGMENTS[idx])
        return build_unified_attention_reduce(spec)
    if idx >= 11:
        spec = UnifiedAttention3DSpec(problem=problem, num_segments=_SEGMENTS[idx])
        return build_unified_attention_3d(spec)
    spec = UnifiedAttention2DSpec(problem=problem)
    return build_unified_attention_2d(spec)


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _kernel,
        _emit_build,
        usage="usage: attention_unified_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
