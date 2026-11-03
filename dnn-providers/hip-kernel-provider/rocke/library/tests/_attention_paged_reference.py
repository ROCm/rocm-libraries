# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side pieces of the torch-free paged attention numeric tests.

The kernels read fp16 or bf16 caches; the tests build float32 values, upload
them as the kernel's dtype, and compare the output with an FP32 numpy
reference over the same rounded values.
"""

import numpy as np
import numpy.typing as npt

from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference

__all__ = [
    "decode_half",
    "encode_half",
    "paged_attention_reference",
    "round_to_half",
]


def encode_half(values: npt.ArrayLike, dtype: str) -> np.ndarray:
    """float32 ``values`` as the kernel's ``dtype`` host array: float16 for
    ``"fp16"``, raw uint16 bit patterns for ``"bf16"``."""
    values = np.asarray(values, dtype=np.float32)
    return values.astype(np.float16) if dtype == "fp16" else f32_to_bf16(values)


def decode_half(array: np.ndarray, dtype: str) -> np.ndarray:
    """The float32 values of an ``encode_half`` array."""
    return array.astype(np.float32) if dtype == "fp16" else bf16_to_f32(array)


def round_to_half(values: npt.ArrayLike, dtype: str) -> np.ndarray:
    """float32 ``values`` rounded to ``dtype``: what the kernel reads back."""
    return decode_half(encode_half(values, dtype), dtype)


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
) -> tuple[np.ndarray, np.ndarray]:
    """FP32 output of one sequence and which of its query rows see a key.

    ``q`` is ``[q_len, num_query_heads, head_size]`` and the caches are
    ``[num_blocks, block_size, num_kv_heads, head_size]``, all float32;
    ``block_ids`` is the sequence's block-table row. Key ``k`` is visible to
    query row ``p`` when ``k <= diag``, where ``diag`` is ``p`` for the top-left
    causal mask and ``p + kv_len - q_len`` for the bottom-right one, and, with a
    ``sliding_window`` above 0, also ``diag - k < sliding_window``. A row that
    sees no key gets an output of 0.

    The sequence's keys and values are gathered from their pages and passed to
    ``sdpa_reference``, the shared SDPA oracle. A row "sees a key" when some
    head of it has a finite log-sum-exp, which is exact for finite inputs.
    """
    q_len = q.shape[0]
    block_size, num_kv_heads, head_size = k_cache.shape[1:]
    blocks = block_ids[: (kv_len + block_size - 1) // block_size]
    keys = k_cache[blocks].reshape(-1, num_kv_heads, head_size)[:kv_len]
    values = v_cache[blocks].reshape(-1, num_kv_heads, head_size)[:kv_len]
    # sdpa_reference's left bound L keeps the diagonal and the L keys to its
    # left, so a window of W keys is L = W - 1.
    out, lse = sdpa_reference(
        q,
        keys,
        values,
        scale=scale,
        left_bound=sliding_window - 1 if sliding_window > 0 else None,
        right_bound=0,
        top_left=top_left,
        q_offsets=[0, q_len],
        kv_offsets=[0, kv_len],
    )
    return out.astype(np.float32), np.isfinite(lse).any(axis=-1)
