# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pure-torch Philox4x32-10 and the FlyDSL SDPA backward's dropout mask.

The map reproduced here is the one every shipped arch uses (lane width 32):
``dnn-providers/hip-kernel-provider/flydsl/kernels_src/kernels/attention_bwd_cdna/philox.py``
(``philox_4x``, ``philox_u32``, ``Philox.grid_plane`` / ``grid_offset``,
``dropout_threshold``) and ``attention_bwd_rdna/fmha_bwd_dkdv_gfx11.py`` (keep
rule and counter layout).

Element ``(q, kv)`` of sequence ``z`` and query head ``h`` draws counter::

    offset + (z * Hq + h) * max_sq * ceil(max_sk / 4) + q * ceil(max_sk / 4) + kv // 4

at slot ``kv % 4``. The counter words are ``(lo32, hi32, 0, 0)``, the key words
``(seed_lo32, seed_hi32)``, and the element is kept iff ``int32(u32) > threshold``.
Counters of consecutive ``(z, h, q, kv // 4)`` are consecutive integers, so the
mask is the Philox stream of ``offset + arange`` laid out row by row.

Arithmetic runs in int64 tensors holding 32-bit values. The 32x32 products are
assembled from 16-bit halves, so nothing overflows and no unsigned dtype is needed.
"""

from __future__ import annotations

import torch

_MASK32 = 0xFFFFFFFF
_MUL_A = 0xD2511F53
_MUL_B = 0xCD9E8D57
_KEY_A = 0x9E3779B9
_KEY_B = 0xBB67AE85
_ROUNDS = 10

# Counters per chunk: bounds the int64 temporaries (about a hundred bytes per counter).
_CHUNK_COUNTERS = 1 << 21


def _mul_hi_lo(x: torch.Tensor, c: int) -> tuple[torch.Tensor, torch.Tensor]:
    """High and low 32 bits of ``x * c`` for ``x`` in ``[0, 2**32)`` and a 32-bit constant."""
    xl = x & 0xFFFF
    xh = x >> 16
    cl = c & 0xFFFF
    ch = c >> 16
    ll = xl * cl
    mid = xl * ch + xh * cl + (ll >> 16)
    hi = xh * ch + (mid >> 16)
    lo = ((mid & 0xFFFF) << 16) | (ll & 0xFFFF)
    return hi, lo


def philox4x32_10(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    k0: int,
    k1: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Philox4x32-10 over int64 tensors of 32-bit counter words and a scalar key.

    Returns the four output words as int64 tensors in ``[0, 2**32)``.
    """
    for _ in range(_ROUNDS):
        hi0, lo0 = _mul_hi_lo(c0, _MUL_A)
        hi1, lo1 = _mul_hi_lo(c2, _MUL_B)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + _KEY_A) & _MASK32
        k1 = (k1 + _KEY_B) & _MASK32
    return c0, c1, c2, c3


def _to_int64(v: int) -> int:
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v


def dropout_threshold(p: float) -> int:
    """The int32 keep threshold for drop probability ``p``: ``keep = int32(u32) > threshold``.

    ``p = 0`` gives ``-2**31 + 1`` and ``p = 1`` gives ``2**31 - 1`` (keep nothing).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    t = int((p - 0.5) * 0xFFFFFFFF)
    return max(-(2**31), min(2**31 - 1, t))


def philox_mask(
    B: int,
    Hq: int,
    max_sq: int,
    max_sk: int,
    p: float,
    seed: int,
    offset: int,
    device,
) -> torch.Tensor:
    """Dropout keep mask ``[B, Hq, max_sq, max_sk]`` (``True`` = keep), on ``device``.

    ``max_sq`` / ``max_sk`` are the S extents of the Q and K tensors declared to
    hipDNN. ``seed`` and ``offset`` are 64-bit; ``offset`` is ``offset1 + offset2``.
    """
    threshold = dropout_threshold(p)
    mask = torch.empty((B, Hq, max_sq, max_sk), dtype=torch.bool, device=device)
    if mask.numel() == 0:
        return mask

    row_counters = (max_sk + 3) // 4
    rows = B * Hq * max_sq
    flat = mask.view(rows, max_sk)
    seed &= 0xFFFFFFFFFFFFFFFF
    key0, key1 = seed & _MASK32, seed >> 32
    base = offset

    rows_per_chunk = max(1, _CHUNK_COUNTERS // row_counters)
    for row0 in range(0, rows, rows_per_chunk):
        n_rows = min(rows_per_chunk, rows - row0)
        n = n_rows * row_counters
        start = _to_int64(base + row0 * row_counters)  # int64 wraps exactly like u64
        ctr = torch.arange(n, dtype=torch.int64, device=device) + start
        zero = torch.zeros_like(ctr)
        words = philox4x32_10(
            ctr & _MASK32, (ctr >> 32) & _MASK32, zero, zero, key0, key1
        )
        u32 = torch.stack(words, dim=1).view(n_rows, row_counters * 4)[:, :max_sk]
        signed = (u32 ^ 0x80000000) - 0x80000000
        flat[row0 : row0 + n_rows] = signed > threshold
    return mask
