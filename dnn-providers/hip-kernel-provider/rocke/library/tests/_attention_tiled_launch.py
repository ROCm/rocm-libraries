# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One-sequence launch of the tiled 2D unified attention path for the
torch-free paged attention numeric tests."""

from __future__ import annotations

import numpy as np

from dispatch.attention.common import AttentionTuningSpec
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from rocke.runtime.hip_module import Runtime

from ._attention_device_array import DeviceArray
from ._attention_paged_reference import decode_half, encode_half

__all__ = ["launch_tiled_one_sequence"]


def launch_tiled_one_sequence(
    rt: Runtime,
    problem: UnifiedAttentionProblem,
    spec: AttentionTuningSpec | None,
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_table: np.ndarray,
    scale: float,
    dtype: str,
) -> np.ndarray:
    """Run the tiled 2D path on one sequence and return O as float32.

    ``q``, ``k_cache`` and ``v_cache`` are float32 arrays of values exact in
    ``dtype``; ``spec`` is the tuning spec to pin, or ``None`` for the
    production route. The output buffer starts with every byte 0xFF, which is
    NaN in fp16 and bf16, so an element the kernel skips fails the caller's
    checks instead of passing on stale memory.
    """
    q_len, kv_len = problem.total_q, problem.max_seqlen_k
    host_q = encode_half(q, dtype)
    out_host = np.empty_like(host_q)
    out_host.view(np.uint8)[...] = 0xFF
    out = DeviceArray(rt, out_host)
    run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(rt, host_q),
        k=DeviceArray(rt, encode_half(k_cache, dtype)),
        v=DeviceArray(rt, encode_half(v_cache, dtype)),
        out=out,
        cu_seqlens_q=DeviceArray(rt, np.array([0, q_len], dtype=np.int32)),
        seqused_k=DeviceArray(rt, np.array([kv_len], dtype=np.int32)),
        softmax_scale=scale,
        block_table=DeviceArray(rt, block_table),
        softcap=0.0,
        backend="tiled",
        tuning_spec=spec,
    )
    return decode_half(out.read(), dtype)
