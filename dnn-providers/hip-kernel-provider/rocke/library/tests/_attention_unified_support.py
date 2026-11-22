# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side pieces of the torch-free unified attention numeric tests.

A decode test builds a logical K/V cache ``[batch, num_kv_heads, capacity,
head_size]`` of float32 values exact in fp16 or bf16, lays it out on the device
as a paged cache or as a strided (BSHD storage) view, and compares the kernel's
output with an FP64 numpy decode reference over the same values. A test on a
cache of GiBs fills the pages no sequence reads with cheap random bits
(:func:`sparse_paged_cache`) and checks long sequences against the paged
reference in pieces (:func:`chunked_paged_reference`).
"""

from __future__ import annotations

import itertools

import numpy as np
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.numeric.storage import host_dtype
from rocke.runtime.device_array import DeviceArray

from ._attention_paged_reference import (
    encode_half,
    paged_attention_reference,
    round_to_half,
)

__all__ = [
    "chunked_paged_reference",
    "decode_reference",
    "paged_cache",
    "paged_layout",
    "random_half",
    "sparse_paged_cache",
    "strided_bshd",
]


def random_half(rng: np.random.Generator, shape, dtype: str) -> np.ndarray:
    """Standard-normal float32 values rounded to ``dtype`` (fp16 or bf16)."""
    return round_to_half(rng.standard_normal(shape, dtype=np.float32), dtype)


def paged_layout(
    logical: np.ndarray, block_size: int, *, poison_after=None
) -> tuple[np.ndarray, np.ndarray]:
    """``logical`` ``[B, H, capacity, D]`` as host pages ``[num_pages,
    block_size, H, D]`` (float32) and their int32 block table ``[B, pages]``.

    Pages are stored in reverse order, so neighbouring table entries are not
    neighbouring pages; the last page of a sequence is zero-padded. With
    ``poison_after`` (the per-sequence lengths), an extra all-NaN page is
    appended and every table entry past a sequence's last page points at it,
    so a kernel that reads one returns NaN."""
    batch, heads, capacity, dim = logical.shape
    blocks = -(-capacity // block_size)
    backing = np.zeros((batch, blocks * block_size, heads, dim), np.float32)
    backing[:, :capacity] = logical.transpose(0, 2, 1, 3)
    pages = backing.reshape(batch * blocks, block_size, heads, dim)[::-1]
    table = np.arange(batch * blocks - 1, -1, -1, dtype=np.int32).reshape(batch, blocks)
    if poison_after is not None:
        nan_page = np.full((1, block_size, heads, dim), np.nan, np.float32)
        pages = np.concatenate([pages, nan_page])
        for row, length in zip(table, poison_after):
            row[-(-length // block_size) :] = len(pages) - 1
    return np.ascontiguousarray(pages), table


def paged_cache(
    logical: np.ndarray, block_size: int, dtype: str, *, poison_after=None
) -> tuple[DeviceArray, np.ndarray]:
    """:func:`paged_layout` with the pages as a ``dtype`` device array."""
    pages, table = paged_layout(logical, block_size, poison_after=poison_after)
    return DeviceArray(encode_half(pages, dtype), dtype), table


def strided_bshd(logical: np.ndarray, dtype: str) -> DeviceArray:
    """``logical`` ``[B, H, S, D]`` stored as ``[B, S, H, D]`` and viewed as
    ``[B, H, S, D]``: the non-paged layout of strided-KV decode."""
    _, heads, seq, dim = logical.shape
    storage = DeviceArray(encode_half(logical.transpose(0, 2, 1, 3), dtype), dtype)
    return storage.view(logical.shape, (seq * heads * dim, dim, heads * dim, 1))


def decode_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    lengths,
    scale: float,
    *,
    sliding_window: int = 0,
) -> np.ndarray:
    """FP64 decode of one query per sequence, as float32 ``[B, Hq, D]``.

    ``q`` is ``[B, Hq, D]``; ``k`` and ``v`` are logical ``[B, Hkv, capacity, D]``
    of which sequence ``b`` sees its first ``lengths[b]`` tokens, or with a
    ``sliding_window`` above 0 only the last ``sliding_window`` of them (none:
    output 0)."""
    windows = [
        (max(0, n - sliding_window) if sliding_window else 0, n) for n in lengths
    ]
    keys = np.concatenate(
        [k[b, :, lo:hi].transpose(1, 0, 2) for b, (lo, hi) in enumerate(windows)]
    )
    values = np.concatenate(
        [v[b, :, lo:hi].transpose(1, 0, 2) for b, (lo, hi) in enumerate(windows)]
    )
    out, _ = sdpa_reference(
        q,
        keys,
        values,
        scale=scale,
        q_offsets=np.arange(len(lengths) + 1),
        kv_offsets=[0, *itertools.accumulate(hi - lo for lo, hi in windows)],
    )
    return out.astype(np.float32)


def sparse_paged_cache(
    rng: np.random.Generator,
    shape,
    used_ids: np.ndarray,
    used_values: np.ndarray,
    dtype: str,
) -> DeviceArray:
    """A device page cache of ``shape`` ``[num_blocks, block_size, H, D]`` whose
    pages ``used_ids`` hold ``used_values`` and every other page random ``dtype``
    bit patterns of magnitude in [1, 2).

    The filler is cheap to draw for caches of GiBs and unlike the used values,
    so a kernel that reads a page it should not is caught."""
    bits = rng.integers(0, 1 << 16, size=shape, dtype=np.uint16)
    if dtype == "bf16":
        bits = (bits & 0x807F) | 0x3F80
    else:
        bits = (bits & 0x83FF) | 0x3C00
    # Copy the used pages as bits: assigning fp16 values to the uint16 array
    # would convert them to integers.
    bits[used_ids] = encode_half(used_values, dtype).view(np.uint16)
    return DeviceArray(bits.view(host_dtype(dtype)), dtype)


def chunked_paged_reference(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    kv_len: int,
    block_ids: np.ndarray,
    scale: float,
    *,
    sliding_window: int = 0,
    rows_per_chunk: int = 1024,
) -> np.ndarray:
    """:func:`paged_attention_reference` of one bottom-right-causal sequence,
    computed one KV head and ``rows_per_chunk`` query rows at a time so a long
    sequence's score matrix stays small.

    A chunk of rows ``[r0, r1)`` sees the keys a full sequence's rows see when
    it is given ``kv_len - q_len + r1`` keys: its bottom-right diagonal then
    lands on the same keys."""
    q_len, num_query_heads, _ = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_query_heads // num_kv_heads
    out = np.empty(q.shape, np.float32)
    for head in range(num_kv_heads):
        heads = slice(head * group, (head + 1) * group)
        for start in range(0, q_len, rows_per_chunk):
            stop = min(start + rows_per_chunk, q_len)
            out[start:stop, heads] = paged_attention_reference(
                q[start:stop, heads],
                k_cache[:, :, head : head + 1],
                v_cache[:, :, head : head + 1],
                kv_len - q_len + stop,
                block_ids,
                scale,
                sliding_window=sliding_window,
            )
    return out
