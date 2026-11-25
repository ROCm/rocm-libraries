#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_unified_emit.py -- Python reference emitter for the
# unified scalar attention parity harness. Selects one of the sampled configs
# by argv[1]: configs 0-5 build a UnifiedAttention2DSpec via
# build_unified_attention_2d; configs 6-7 build the scalar split-3D reduce
# (build_unified_attention_reduce, 8 segments) of the problem named in
# _REDUCE_CONFIGS; configs 8-10 build the scalar split-3D segment kernel
# (build_unified_attention_3d, 8 segments) of the problem named in
# _SEGMENT_CONFIGS (config 10 with a sliding window); configs 11-22 are 2D,
# segment and reduce kernels of their own problems (see _SEGMENTS). Prints
# _SEGMENT_CONFIGS (config 10 with a sliding window). Configs 23 and 24 are the
# scalar 2D and split-3D segment kernels given an additive-bias problem, which
# both engines refuse (_EXPECTED_REJECTIONS). Prints
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
# 2D builder, so its value does not affect the emitted IR.
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
}

# Configs 11-15 are further 2D kernels, 16-20 split-3D segment kernels and
# 21-22 split-3D reduces, each of its own problem (segment count from _SEGMENTS).
_CONFIGS[11] = dict(
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
)

_CONFIGS[12] = dict(
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
)

_CONFIGS[13] = dict(
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
)

_CONFIGS[14] = dict(
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
)

_CONFIGS[15] = dict(
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
)

_CONFIGS[16] = dict(
    head_size=128,
    block_size=16,
    dtype="bf16",
    num_query_heads=32,
    num_kv_heads=8,
    total_q=4,
    max_seqlen_q=1,
    max_seqlen_k=4096,
    num_seqs=4,
)

_CONFIGS[17] = dict(
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
)

_CONFIGS[18] = dict(
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
)

_CONFIGS[19] = dict(
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
)

_CONFIGS[20] = dict(
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
)

_CONFIGS[21] = dict(
    head_size=128,
    block_size=16,
    dtype="bf16",
    num_query_heads=32,
    num_kv_heads=8,
    total_q=4,
    max_seqlen_q=1,
    max_seqlen_k=4096,
    num_seqs=4,
)

_CONFIGS[22] = dict(
    head_size=64,
    block_size=32,
    dtype="fp16",
    num_query_heads=16,
    num_kv_heads=16,
    total_q=8,
    max_seqlen_q=2,
    max_seqlen_k=8192,
    num_seqs=4,
)

_SEGMENTS = {16: 8, 17: 4, 18: 8, 19: 16, 20: 8, 21: 8, 22: 16}
# Additive bias on the scalar kernels: they have no bias score site, so both
# engines refuse it instead of building a kernel that drops the bias.
_CONFIGS[23] = dict(_CONFIGS[2], use_additive_bias=True)

# Configs both engines must reject, with the reason each one pins: run_diff
# requires the regex in both engines' stderr (BOTH_REJECTED alone accepts any
# reason, a typo included).
_EXPECTED_REJECTIONS = {
    23: r"scalar 2D kernel: additive bias runs on the gfx950 tiled kernels only",
    24: r"scalar 3D kernel: additive bias runs on the gfx950 tiled kernels only",
}

# Reduce config index -> the 2D config whose problem it reduces.
_REDUCE_CONFIGS = {6: 2, 7: 1}
_NUM_SEGMENTS = 8

# Segment config index -> the 2D config whose problem the split-3D segment kernel
# runs. Both engines build the segment kernel for the default (bottom-right
# causal) mask, which every 2D config uses; config 10 (problem 3) adds a sliding
# window.
_SEGMENT_CONFIGS = {8: 2, 9: 1, 10: 3, 24: 23}


def _kernel(idx: int):
    if idx in _REDUCE_CONFIGS:
        problem = UnifiedAttentionProblem(**_CONFIGS[_REDUCE_CONFIGS[idx]])
        spec = UnifiedAttentionReduceSpec(problem, num_segments=_NUM_SEGMENTS)
        return build_unified_attention_reduce(spec)
    if idx in _SEGMENT_CONFIGS:
        problem = UnifiedAttentionProblem(**_CONFIGS[_SEGMENT_CONFIGS[idx]])
        spec = UnifiedAttention3DSpec(problem=problem, num_segments=_NUM_SEGMENTS)
        return build_unified_attention_3d(spec)
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    problem = UnifiedAttentionProblem(**_CONFIGS[idx])
    if idx in _SEGMENTS and idx >= 21:
        spec = UnifiedAttentionReduceSpec(problem=problem, num_segments=_SEGMENTS[idx])
        return build_unified_attention_reduce(spec)
    if idx in _SEGMENTS:
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
