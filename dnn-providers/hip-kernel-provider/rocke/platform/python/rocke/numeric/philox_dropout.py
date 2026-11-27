# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host oracle of the SDPA dropout mask (SdpaSemanticsRegister SEM-015).

SEM-015 fixes the mask a forward and a backward kernel must both regenerate, so
that a rocKE forward can pair with any other engine's backward (and the
reverse). It is the mask hipDNN's conformance reference
``hipdnn_sdpa._philox.philox_mask`` draws: element ``(q, kv)`` of batch ``z`` and query head ``h`` takes word ``kv % 4`` of
Philox4x32-10 at counter::

    offset + ((z * Hq + h) * max_sq + q) * ceil(max_sk / 4) + kv // 4

with counter words ``(lo32, hi32, 0, 0)``, key words ``(seed_lo32, seed_hi32)``,
and is kept iff ``int32(word) > dropout_threshold(p)``. All arithmetic wraps as
u64 / u32. The kernels compare against the same host-computed threshold, so the
mask is bit-exact, not statistical.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PHILOX_KEY_A",
    "PHILOX_KEY_B",
    "PHILOX_MUL_A",
    "PHILOX_MUL_B",
    "PHILOX_ROUNDS",
    "dropout_scale",
    "dropout_threshold",
    "philox4x32_10",
    "philox_counter_row_base",
    "philox_keep_mask",
]

PHILOX_MUL_A = 0xD2511F53
PHILOX_MUL_B = 0xCD9E8D57
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUNDS = 10

_M32 = np.uint64(0xFFFFFFFF)
_M64 = 0xFFFFFFFFFFFFFFFF
_SH32 = np.uint64(32)

# Counters per chunk: bounds the uint64 temporaries.
_CHUNK_COUNTERS = 1 << 20


def philox4x32_10(c0, c1, c2, c3, k0: int, k1: int):
    """Philox4x32-10 over arrays of 32-bit counter words (any int dtype) and a scalar key.

    Returns the four output words as ``uint64`` arrays holding values in ``[0, 2**32)``.
    A 32x32 product is below ``2**64``, so uint64 holds it exactly.
    """
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) & _M32 for c in (c0, c1, c2, c3))
    k0 &= 0xFFFFFFFF
    k1 &= 0xFFFFFFFF
    mul_a = np.uint64(PHILOX_MUL_A)
    mul_b = np.uint64(PHILOX_MUL_B)
    for _ in range(PHILOX_ROUNDS):
        p0 = c0 * mul_a
        p1 = c2 * mul_b
        c0, c1, c2, c3 = (
            (p1 >> _SH32) ^ c1 ^ np.uint64(k0),
            p1 & _M32,
            (p0 >> _SH32) ^ c3 ^ np.uint64(k1),
            p0 & _M32,
        )
        k0 = (k0 + PHILOX_KEY_A) & 0xFFFFFFFF
        k1 = (k1 + PHILOX_KEY_B) & 0xFFFFFFFF
    return c0, c1, c2, c3


def dropout_threshold(p: float) -> int:
    """The int32 keep threshold of drop probability ``p``: keep iff ``int32(word) > threshold``.

    ``int32(f32(p - 0.5) * 2**32)`` evaluated in f32, as SEM-015 states: the
    same expression in double flips the keep bit of a few words. The host passes
    it to the kernel, which then needs one integer compare per element. ``p = 0`` keeps all but the one word ``0x80000000``; ``p = 1`` keeps
    nothing.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    t = int(np.float32(np.float32(p) - np.float32(0.5)) * np.float32(4294967296.0))
    return max(-(2**31), min(2**31 - 1, t))


def dropout_scale(p: float) -> float:
    """The factor ``1 / (1 - p)`` on kept probabilities (``0`` for ``p = 1``, nothing is kept)."""
    return 0.0 if p >= 1.0 else 1.0 / (1.0 - p)


def philox_counter_row_base(
    z: int, h: int, q: int, *, Hq: int, max_sq: int, max_sk: int
) -> int:
    """Counter offset of row ``(z, h, q)`` relative to ``offset``: the counter of key ``kv`` adds ``kv // 4``."""
    return ((z * Hq + h) * max_sq + q) * ((max_sk + 3) // 4)


def philox_keep_mask(
    B: int, Hq: int, max_sq: int, max_sk: int, p: float, seed: int, offset: int
) -> np.ndarray:
    """Keep mask ``[B, Hq, max_sq, max_sk]`` of bool (``True`` = keep).

    The mask a kernel launched with ``p`` draws, for any ``p`` (as hipDNN's
    reference): a request with ``p = 0`` asks for no dropout and launches no
    mask, so it is no caller of this at ``p = 0``. ``max_sq`` / ``max_sk`` are
    the declared S extents of Q and K (the padded extents for a padded or
    ragged batch); ``seed`` and ``offset`` are u64.
    """
    threshold = dropout_threshold(p)
    mask = np.empty((B, Hq, max_sq, max_sk), dtype=bool)
    if mask.size == 0:
        return mask
    row_counters = (max_sk + 3) // 4
    rows = B * Hq * max_sq
    flat = mask.reshape(rows, max_sk)
    seed &= _M64
    key0, key1 = seed & 0xFFFFFFFF, seed >> 32
    rows_per_chunk = max(1, _CHUNK_COUNTERS // row_counters)
    for row0 in range(0, rows, rows_per_chunk):
        n_rows = min(rows_per_chunk, rows - row0)
        n = n_rows * row_counters
        start = np.uint64((offset + row0 * row_counters) & _M64)
        # uint64 addition wraps modulo 2**64, as the counter does.
        with np.errstate(over="ignore"):
            ctr = np.arange(n, dtype=np.uint64) + start
        zero = np.zeros_like(ctr)
        words = philox4x32_10(ctr & _M32, ctr >> _SH32, zero, zero, key0, key1)
        u32 = np.stack(words, axis=1).reshape(n_rows, row_counters * 4)[:, :max_sk]
        flat[row0 : row0 + n_rows] = u32.astype(np.uint32).view(np.int32) > threshold
    return mask
