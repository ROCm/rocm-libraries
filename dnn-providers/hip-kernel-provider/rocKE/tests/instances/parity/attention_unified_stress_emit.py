#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Stress parity emitter for unified attention scalar 2D (Python reference side).
from rocke.instances.common.attention_unified import (
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    build_unified_attention_2d,
)
from _emit_common import run_emit


def C(
    head_size,
    block_size,
    dtype,
    num_query_heads,
    num_kv_heads,
    total_q,
    max_seqlen_q,
    max_seqlen_k,
    use_sinks=False,
    sliding_window=0,
    softcap=0.0,
    num_seqs=1,
):
    return dict(
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        total_q=total_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        use_sinks=use_sinks,
        sliding_window=sliding_window,
        softcap=softcap,
        num_seqs=num_seqs,
    )


_CONFIGS = {
    0: C(64, 32, "bf16", 64, 8, 128, 512, 512, use_sinks=True),
    1: C(128, 32, "bf16", 64, 8, 256, 1024, 1024, use_sinks=True),
    2: C(256, 16, "fp16", 32, 4, 64, 256, 256),
    3: C(64, 16, "bf16", 16, 2, 512, 2048, 2048, use_sinks=True, sliding_window=128),
    4: C(128, 32, "bf16", 64, 8, 1, 1, 4096),
    5: C(64, 32, "bf16", 128, 16, 1024, 4096, 4096, use_sinks=True, softcap=50.0),
    6: C(64, 64, "bf16", 8, 1, 64, 128, 128),
    7: C(256, 64, "fp16", 8, 8, 32, 64, 64, use_sinks=True),
    8: C(128, 16, "fp16", 32, 8, 200, 300, 300, use_sinks=True, softcap=30.0),
    9: C(64, 32, "fp16", 40, 5, 77, 128, 128, softcap=10.0),
    10: C(64, 16, "bf16", 1, 1, 1, 1, 1),
    11: C(128, 32, "fp16", 1, 1, 1, 1, 16, use_sinks=True),
    12: C(256, 64, "bf16", 2, 1, 1, 1, 1, softcap=5.0),
    13: C(64, 16, "bf16", 7, 7, 13, 17, 19),
    14: C(128, 32, "fp16", 11, 11, 101, 103, 107, use_sinks=True),
    15: C(64, 32, "bf16", 13, 1, 251, 257, 263, sliding_window=31),
    16: C(256, 64, "bf16", 128, 8, 65536, 32768, 131072, use_sinks=True),
    17: C(128, 16, "fp16", 96, 12, 100000, 8192, 262144),
    18: C(64, 16, "bf16", 32, 4, 300, 512, 512, sliding_window=1),
    19: C(128, 32, "fp16", 64, 8, 300, 512, 512, use_sinks=True, sliding_window=4096),
    20: C(256, 16, "bf16", 16, 16, 128, 256, 256, sliding_window=63),
    21: C(64, 32, "bf16", 8, 8, 64, 128, 128, softcap=0.0001),
    22: C(64, 32, "bf16", 8, 8, 64, 128, 128, softcap=0.0),
    23: C(128, 32, "bf16", 128, 1, 512, 1024, 1024),
    24: C(64, 16, "fp16", 256, 64, 256, 512, 512, use_sinks=True),
    25: C(64, 16, "fp16", 32, 4, 1, 1, 8192, use_sinks=True),
    26: C(256, 32, "bf16", 16, 2, 1, 1, 2048, softcap=20.0),
    27: C(64, 32, "bf16", 64, 8, 128, 512, 512, use_sinks=True, num_seqs=16),
    28: C(128, 64, "fp16", 32, 8, 64, 1, 65535),
    29: C(64, 16, "bf16", 8, 8, 2, 2, 3, use_sinks=True, sliding_window=2, softcap=7.0),
    30: C(
        128,
        16,
        "bf16",
        40,
        8,
        333,
        1000,
        2000,
        use_sinks=True,
        sliding_window=256,
        softcap=42.0,
    ),
    31: C(
        256,
        64,
        "fp16",
        64,
        64,
        997,
        1009,
        1013,
        use_sinks=True,
        sliding_window=511,
        softcap=99.0,
    ),
}


def _kernel(idx):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    problem = UnifiedAttentionProblem(**_CONFIGS[idx])
    spec = UnifiedAttention2DSpec(problem=problem)
    return build_unified_attention_2d(spec)


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _kernel,
        _emit_build,
        usage="usage: attention_unified_stress_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
