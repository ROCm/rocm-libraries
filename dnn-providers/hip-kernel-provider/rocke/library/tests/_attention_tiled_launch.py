# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launches of the tiled 2D unified attention path for the torch-free paged
attention numeric tests."""

from __future__ import annotations

import itertools

import numpy as np
from dispatch.attention.common import AttentionTuningSpec
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from rocke.runtime.device_array import DeviceArray

from ._attention_paged_reference import decode_half, encode_half

__all__ = ["launch_tiled", "launch_tiled_one_sequence"]


def launch_tiled(
    problem: UnifiedAttentionProblem,
    spec: AttentionTuningSpec | None,
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_table: np.ndarray,
    scale: float,
    *,
    q_lens: list[int],
    kv_lens: list[int],
    softcap: float = 0.0,
    sinks: np.ndarray | None = None,
    alibi_slopes: np.ndarray | None = None,
    qq_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Run the tiled 2D path on a batch of sequences and return O as float32.

    ``q`` (``[sum(q_lens), num_query_heads, head_size]``), ``k_cache``,
    ``v_cache`` and ``sinks`` are float32 arrays of values exact in
    ``problem.dtype``; the kernel reads all four in that dtype. ``alibi_slopes``
    (``[num_query_heads]``) and ``qq_bias`` (2D; its row length is the stride
    the kernel indexes with) are read as float32. ``spec`` is the tuning spec
    to pin, or ``None`` for the production route. The output buffer starts with
    every byte 0xFF, which is NaN in fp16 and bf16, so an element the kernel
    skips fails the caller's checks instead of passing on stale memory.
    """
    dtype = problem.dtype

    def in_kernel_dtype(values):
        return (
            None if values is None else DeviceArray(encode_half(values, dtype), dtype)
        )

    def as_numpy_dtype(values, numpy_dtype):
        if values is None:
            return None
        return DeviceArray(np.ascontiguousarray(values, dtype=numpy_dtype))

    host_q = encode_half(q, dtype)
    out_host = np.empty_like(host_q)
    out_host.view(np.uint8)[...] = 0xFF
    out = DeviceArray(out_host, dtype)
    run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(host_q, dtype),
        k=in_kernel_dtype(k_cache),
        v=in_kernel_dtype(v_cache),
        out=out,
        cu_seqlens_q=as_numpy_dtype([0, *itertools.accumulate(q_lens)], np.int32),
        seqused_k=as_numpy_dtype(kv_lens, np.int32),
        softmax_scale=scale,
        block_table=DeviceArray(block_table),
        softcap=softcap,
        sinks=in_kernel_dtype(sinks),
        alibi_slopes=as_numpy_dtype(alibi_slopes, np.float32),
        qq_bias=as_numpy_dtype(qq_bias, np.float32),
        qq_bias_stride_0=0 if qq_bias is None else int(np.shape(qq_bias)[1]),
        backend="tiled",
        tuning_spec=spec,
    )
    return decode_half(out.read(), dtype)


def launch_tiled_one_sequence(
    problem: UnifiedAttentionProblem,
    spec: AttentionTuningSpec | None,
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_table: np.ndarray,
    scale: float,
    dtype: str,
) -> np.ndarray:
    """:func:`launch_tiled` for one sequence of ``problem.total_q`` queries
    and ``problem.max_seqlen_k`` keys, with no score modifiers. ``dtype`` must
    be ``problem.dtype``."""
    if dtype != problem.dtype:
        raise ValueError(
            f"one-sequence tiled launch failed - dtype={dtype!r}, "
            f"expected problem.dtype={problem.dtype!r}"
        )
    return launch_tiled(
        problem,
        spec,
        q,
        k_cache,
        v_cache,
        block_table,
        scale,
        q_lens=[problem.total_q],
        kv_lens=[problem.max_seqlen_k],
    )
