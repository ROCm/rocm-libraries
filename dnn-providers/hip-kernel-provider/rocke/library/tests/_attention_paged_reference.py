# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side pieces of the torch-free paged attention numeric tests.

The kernels read fp16 or bf16 caches; the tests build float32 values, upload
them as the kernel's dtype, and compare the output with an FP32 numpy
reference over the same rounded values.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference

__all__ = [
    "alibi_score_bias",
    "decode_half",
    "encode_half",
    "paged_attention_reference",
    "qq_score_bias",
    "round_to_half",
]


_HALF_DTYPES = ("fp16", "bf16")


def _require_half_dtype(dtype: str) -> None:
    """Refuse a ``dtype`` these helpers have no host encoding for, so an fp8 or
    fp32 caller fails here instead of being read as bf16."""
    if dtype not in _HALF_DTYPES:
        raise ValueError(
            f"paged attention host encoding failed - dtype={dtype!r}, "
            f"expected one of {_HALF_DTYPES}"
        )


def encode_half(values: npt.ArrayLike, dtype: str) -> np.ndarray:
    """float32 ``values`` as the kernel's ``dtype`` host array: float16 for
    ``"fp16"``, raw uint16 bit patterns for ``"bf16"``. Any other ``dtype``
    raises ``ValueError``."""
    _require_half_dtype(dtype)
    values = np.asarray(values, dtype=np.float32)
    return values.astype(np.float16) if dtype == "fp16" else f32_to_bf16(values)


def decode_half(array: np.ndarray, dtype: str) -> np.ndarray:
    """The float32 values of an ``encode_half`` array. Any ``dtype`` other than
    ``"fp16"`` or ``"bf16"`` raises ``ValueError``."""
    _require_half_dtype(dtype)
    return array.astype(np.float32) if dtype == "fp16" else bf16_to_f32(array)


def round_to_half(values: npt.ArrayLike, dtype: str) -> np.ndarray:
    """float32 ``values`` rounded to ``dtype``: what the kernel reads back. Any
    ``dtype`` other than ``"fp16"`` or ``"bf16"`` raises ``ValueError``."""
    return decode_half(encode_half(values, dtype), dtype)


def alibi_score_bias(slopes: npt.ArrayLike, q_len: int, kv_len: int) -> np.ndarray:
    """ALiBi as an additive score bias ``[num_query_heads, q_len, kv_len]``.

    The unified attention kernels add ``slopes[h] * (k - context_len)`` to every
    score of head ``h`` (AITER's semantics), with the bottom-right offset
    ``context_len = kv_len - q_len``; the term does not depend on the query row.
    """
    key_offsets = np.arange(kv_len, dtype=np.float64) - (kv_len - q_len)
    slopes = np.asarray(slopes, dtype=np.float64)
    term = slopes[:, None] * key_offsets[None, :]
    return np.broadcast_to(term[:, None, :], (slopes.size, q_len, kv_len))


def qq_score_bias(qq_bias: npt.ArrayLike, q_len: int, kv_len: int) -> np.ndarray:
    """The query-to-query bias as an additive score bias ``[q_len, kv_len]``.

    The kernels add ``qq_bias[p, k - context_len]`` to the score of query row
    ``p`` and key ``k`` when ``0 <= k - context_len < qq_bias.shape[1]`` (the
    bias row stride), and nothing otherwise; ``context_len = kv_len - q_len``.
    The same matrix serves every head and sequence.
    """
    qq_bias = np.asarray(qq_bias, dtype=np.float64)
    query_keys = np.arange(kv_len) - (kv_len - q_len)
    in_range = (query_keys >= 0) & (query_keys < qq_bias.shape[1])
    out = np.zeros((q_len, kv_len))
    out[:, in_range] = qq_bias[:q_len, query_keys[in_range]]
    return out


def paged_attention_reference(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    kv_len: int,
    block_ids: np.ndarray,
    scale: float,
    *,
    sliding_window: int = 0,
    top_left: bool = False,
    score_bias: npt.ArrayLike | None = None,
    softcap: float = 0.0,
    sinks: npt.ArrayLike | None = None,
) -> np.ndarray:
    """FP32 output of one sequence, ``[q_len, num_query_heads, head_size]``.

    ``q`` is ``[q_len, num_query_heads, head_size]`` and the caches are
    ``[num_blocks, block_size, num_kv_heads, head_size]``, all float32;
    ``block_ids`` is the sequence's block-table row. Key ``k`` is visible to
    query row ``p`` when ``k <= diag``, where ``diag`` is ``p`` for the top-left
    causal mask and ``p + kv_len - q_len`` for the bottom-right one, and, with a
    ``sliding_window`` above 0, also ``diag - k < sliding_window``. A row that
    sees no key gets an output of 0.

    The score modifiers apply in the kernels' order: the scaled score is capped
    by ``softcap`` (0 = off), then ``score_bias`` (broadcast against
    ``[num_query_heads, q_len, kv_len]``; see :func:`alibi_score_bias` and
    :func:`qq_score_bias`) is added, and ``sinks`` (one logit per query head)
    join the softmax denominator. Pass the sinks rounded to the dtype the kernel
    reads them in.

    The sequence's keys and values are gathered from their pages and passed to
    ``sdpa_reference``, the shared SDPA oracle.
    """
    q_len = q.shape[0]
    block_size, num_kv_heads, head_size = k_cache.shape[1:]
    blocks = block_ids[: (kv_len + block_size - 1) // block_size]
    keys = k_cache[blocks].reshape(-1, num_kv_heads, head_size)[:kv_len]
    values = v_cache[blocks].reshape(-1, num_kv_heads, head_size)[:kv_len]
    # sdpa_reference's left bound L keeps the diagonal and the L keys to its
    # left, so a window of W keys is L = W - 1.
    out, _lse = sdpa_reference(
        q,
        keys,
        values,
        scale=scale,
        bias=score_bias,
        left_bound=sliding_window - 1 if sliding_window > 0 else None,
        right_bound=0,
        top_left=top_left,
        q_offsets=[0, q_len],
        kv_offsets=[0, kv_len],
        softcap=softcap,
        sinks=sinks,
    )
    return out.astype(np.float32)
