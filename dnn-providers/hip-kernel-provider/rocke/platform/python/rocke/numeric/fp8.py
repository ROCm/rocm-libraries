# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free OCP fp8 host codecs: e4m3 (``fn``) and e5m2, carried as ``uint8``.

The decode tables are derived from the formats rather than typed in. The
``*_encode`` functions accept only values the format holds exactly: the verify
harnesses that use them compare a kernel against a reference computed on exactly
the numbers the device sees, and a silent round would demote that to a tolerance
check. ``fp8e4m3_round`` rounds to the nearest code instead, for random operand
data whose reference is computed from the rounded bytes. Sibling of
:mod:`rocke.numeric.bf16`; harnesses import the codec from here instead of
growing a second copy.
"""

from __future__ import annotations

__all__ = [
    "bf8e5m2_decode_table",
    "bf8e5m2_encode",
    "fp8e4m3_decode_table",
    "fp8e4m3_encode",
    "fp8e4m3_round",
]


def fp8e4m3_decode_table():
    """All 256 OCP e4m3 (``fn``) codes decoded to f32, built from the format.

    ``s.eeee.mmm``, bias 7. ``e == 0`` is subnormal — ``(-1)^s * 2^-6 * m/8``;
    otherwise ``(-1)^s * 2^(e-7) * (1 + m/8)``. The format carries no
    infinities, and ``e == 0xF, m == 0x7`` (``0x7f`` / ``0xff``) is NaN, so the
    largest finite magnitude is 448.

    Derived rather than tabulated on purpose: a hand-typed table is a second
    definition of the wire format that can drift from the hardware's.
    """
    import numpy as np

    code = np.arange(256, dtype=np.uint32)
    sign = np.where((code >> 7) & 1, np.float32(-1.0), np.float32(1.0))
    exp = ((code >> 3) & 0xF).astype(np.int32)
    man = (code & 0x7).astype(np.float32)
    sub = np.float32(2.0**-6) * (man / np.float32(8.0))
    nor = np.ldexp(np.float32(1.0) + man / np.float32(8.0), exp - 7)
    val = (sign * np.where(exp == 0, sub, nor)).astype(np.float32)
    val[(exp == 0xF) & ((code & 0x7) == 0x7)] = np.float32("nan")
    return val


def bf8e5m2_decode_table():
    """All 256 OCP e5m2 codes decoded to f32, built from the format.

    ``s.eeeee.mm``, bias 15. ``e == 0`` is subnormal — ``(-1)^s * 2^-14 * m/4``;
    otherwise ``(-1)^s * 2^(e-15) * (1 + m/4)``. Unlike e4m3 the format is
    IEEE-like: ``e == 0x1F`` holds the infinities (``m == 0``) and NaNs, so the
    largest finite magnitude is 57344. Those codes decode to NaN here, which
    keeps them out of the encoder's table.
    """
    import numpy as np

    code = np.arange(256, dtype=np.uint32)
    sign = np.where((code >> 7) & 1, np.float32(-1.0), np.float32(1.0))
    exp = ((code >> 2) & 0x1F).astype(np.int32)
    man = (code & 0x3).astype(np.float32)
    sub = np.float32(2.0**-14) * (man / np.float32(4.0))
    nor = np.ldexp(np.float32(1.0) + man / np.float32(4.0), exp - 15)
    val = (sign * np.where(exp == 0, sub, nor)).astype(np.float32)
    val[exp == 0x1F] = np.float32("nan")
    return val


def fp8e4m3_encode(x):
    """Encode f32 values to OCP e4m3 bytes, rejecting anything inexact."""
    return _encode_exact(x, fp8e4m3_decode_table(), "fp8e4m3")


def bf8e5m2_encode(x):
    """Encode f32 values to OCP e5m2 bytes, rejecting anything inexact."""
    return _encode_exact(x, bf8e5m2_decode_table(), "bf8e5m2")


def fp8e4m3_round(x):
    """Round f32 values to the nearest OCP e4m3 bytes, ties to even, saturating.

    For inputs that need not be representable (random operand data), unlike
    :func:`fp8e4m3_encode`. Ties go to the even code, which in e4m3 is the
    even mantissa (the IEEE rule). Magnitudes beyond the largest finite value
    (448) clamp to it, since the format has no infinity. Zero, and a negative
    value that rounds to zero, give +0 (code 0x00): the code table holds one
    zero, as for the exact encoders. NaN inputs are rejected, naming the first
    one's element.
    """
    return _round_nearest(x, fp8e4m3_decode_table(), "fp8e4m3")


def _sorted_codes(table):
    """The finite values of ``table`` sorted ascending, each once, with their codes.

    -0.0 and +0.0 compare equal; ``np.unique`` folds them and ``return_index``
    keeps the first occurrence, which the stable sort leaves as +0.0 (code 0x00).
    """
    import numpy as np

    keep = np.isfinite(table)
    vals = table[keep]
    codes = np.arange(256, dtype=np.uint8)[keep]
    order = np.argsort(vals, kind="stable")
    vals, codes = vals[order], codes[order]
    uniq, first = np.unique(vals, return_index=True)
    return uniq, codes[first]


def _encode_exact(x, table, name):
    """Encode f32 values to the 8-bit codes of ``table``, rejecting anything
    inexact.

    An exact verify harness rests on the device seeing exactly the numbers the
    reference multiplies, so a silent round here would quietly demote its gate
    to a tolerance check. Vectorised via the inverted decode
    table — a Python loop over a 4096x8192 operand is not viable.
    """
    import numpy as np

    vals, codes = _sorted_codes(table)
    xf = np.ascontiguousarray(x, dtype=np.float32).ravel()
    idx = np.minimum(np.searchsorted(vals, xf), vals.size - 1)
    bad = vals[idx] != xf
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{name} encode: {float(xf[first])!r} (element {first}) is not exactly "
            f"representable in {name}"
        )
    return codes[idx].reshape(np.shape(x))


def _round_nearest(x, table, name):
    """Round f32 values to the nearest code of ``table``: ties to the even code,
    saturating at the table's largest finite magnitude; NaN is rejected."""
    import numpy as np

    vals, codes = _sorted_codes(table)
    xf = np.ascontiguousarray(x, dtype=np.float32)
    nan = np.isnan(xf.ravel())
    if nan.any():
        raise ValueError(f"{name} round: NaN at element {int(np.flatnonzero(nan)[0])}")
    flat = np.clip(xf.ravel(), vals[0], vals[-1])
    hi = np.clip(np.searchsorted(vals, flat), 1, vals.size - 1)
    lo = hi - 1
    d_lo = flat - vals[lo]
    d_hi = vals[hi] - flat
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (codes[hi] % 2 == 0))
    return np.where(pick_hi, codes[hi], codes[lo]).astype(np.uint8).reshape(xf.shape)
