# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side tests for ``rocke.numeric.bf16``.

numpy has no native bfloat16, so every verify harness that touches bf16 goes
through this hand-rolled codec. It is the one piece of the bf16 path that no
GPU test can cover -- a bug here corrupts the tensor before the device ever
sees it, and the kernel takes the blame. These run anywhere; no device needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16


def _bits(u32: list[int]) -> np.ndarray:
    return np.array(u32, dtype=np.uint32).view(np.float32)


def test_roundtrip_is_exact_for_representable_values():
    """Anything already expressible in bf16 must survive encode+decode intact."""
    rng = np.random.default_rng(0xB16)
    f = bf16_to_f32(rng.integers(0, 1 << 16, size=4096, dtype=np.uint16))
    finite = f[np.isfinite(f)]
    assert np.array_equal(bf16_to_f32(f32_to_bf16(finite)), finite)


def test_rounding_is_nearest_even_not_truncation():
    """Truncation would bias every value toward zero; RNE must not."""
    # 1.0 + half a bf16 ULP, exactly on the tie. RNE breaks toward the even
    # mantissa (0x3F80), truncation would also give 0x3F80 -- so pair it with
    # the next tie up (0x3F81 is odd, rounds away to 0x3F82).
    assert f32_to_bf16(_bits([0x3F808000]))[0] == 0x3F80  # tie -> even (down)
    assert f32_to_bf16(_bits([0x3F818000]))[0] == 0x3F82  # tie -> even (up)
    # Above the tie: must round up, which truncation would never do.
    assert f32_to_bf16(_bits([0x3F808001]))[0] == 0x3F81


def test_rounding_can_carry_into_the_exponent():
    """A mantissa carry is legitimate for finite values and must be kept."""
    # Largest mantissa just below 2.0 rounds up to exactly 2.0 (0x4000).
    assert f32_to_bf16(_bits([0x3FFFFFFF]))[0] == 0x4000


def test_relative_error_meets_the_round_to_nearest_bound():
    """RNE is bounded by half an ULP, so this also tells it from truncation."""
    r = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
    rel = float((np.abs(bf16_to_f32(f32_to_bf16(r)) - r) / np.abs(r)).max())
    # bf16 carries 8 significand bits: RNE stays within 2**-8, while plain
    # truncation would reach 2**-7.
    assert rel <= 2**-8 + 1e-9


@pytest.mark.parametrize(
    "pattern",
    [
        0x7F800001,  # +NaN, payload only in the low 16 bits -- the bias would
        0xFF800001,  # -NaN, carry into the exponent and produce +/-inf.
        0x7FC00000,  # +quiet NaN
        0xFFFFFFFF,  # -NaN, all payload bits set
    ],
)
def test_nan_stays_nan(pattern):
    """NaN must never round into an infinity.

    ``0x7F800001`` is the sharp case: adding the RNE bias carries straight into
    the exponent and yields ``0x7F80`` (+inf). Sending an infinity where the
    caller meant a NaN would mask a special-value handling bug in the kernel.
    """
    enc = f32_to_bf16(_bits([pattern]))
    assert np.isnan(bf16_to_f32(enc))[0]
    # Sign is part of the observable value; it must survive too.
    assert (enc[0] >> 15) == (pattern >> 31)


def test_infinity_stays_infinity_with_sign():
    """Genuine infinities must pass through unchanged -- not become NaN."""
    enc = f32_to_bf16(_bits([0x7F800000, 0xFF800000]))
    dec = bf16_to_f32(enc)
    assert dec[0] == np.inf
    assert dec[1] == -np.inf


def test_signed_zero_is_preserved():
    enc = f32_to_bf16(np.array([0.0, -0.0], dtype=np.float32))
    assert enc[0] == 0x0000
    assert enc[1] == 0x8000


def test_encode_returns_uint16_storage():
    """bf16 is carried as raw uint16; a float dtype here means a silent upcast."""
    enc = f32_to_bf16(np.zeros(4, dtype=np.float32))
    assert enc.dtype == np.uint16
    assert bf16_to_f32(enc).dtype == np.float32


def test_shape_is_preserved():
    a = np.zeros((2, 3, 4), dtype=np.float32)
    assert f32_to_bf16(a).shape == a.shape
    assert bf16_to_f32(f32_to_bf16(a)).shape == a.shape


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (0x7FD30000, 0x7FD3),  # payload in the retained mantissa survives
        (0x7FA50000, 0x7FE5),  # ... and the quiet bit is forced on
        (0xFFD30000, 0xFFD3),  # the sign is kept
        (0x7F800001, 0x7FC0),  # payload only in the low 16 bits -> canonical
    ],
)
def test_nan_payload_policy(pattern, expected):
    """Pin the payload policy the module docstring promises.

    ``ml_dtypes`` canonicalises every NaN, so the oracle tests below cannot
    cover this -- it is the one place the two deliberately differ.
    """
    assert f32_to_bf16(_bits([pattern]))[0] == expected


# -- independent oracle --------------------------------------------------------
#
# Everything above checks the codec against properties someone wrote down. These
# check it against ml_dtypes.bfloat16, which shares no code with it. ml_dtypes is
# the ``dev`` extra, so they skip (rather than fail) on a bare toolchain image;
# the codec itself must never depend on it.


def _fp32_probe_bits() -> np.ndarray:
    """fp32 bit patterns that reach every round-to-nearest-even decision.

    RNE depends only on the low 16 bits relative to ``0x8000`` and on the parity
    of the retained half. Every high half crossed with the boundary low halves
    below -- under the tie, on it, just over it, and the maximum -- therefore
    reaches each case for every sign and exponent, including subnormals, the
    overflow edge and the infinities. A fixed-seed random tail adds interior
    points.
    """
    hi = np.arange(1 << 16, dtype=np.uint32) << np.uint32(16)
    lo = np.array([0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=np.uint32)
    edges = (hi[:, None] | lo[None, :]).ravel()
    rng = np.random.default_rng(0xB16)
    tail = rng.integers(0, 1 << 32, size=1 << 20, dtype=np.uint64).astype(np.uint32)
    return np.concatenate([edges, tail])


def _ml_dtypes_encode(ml_dtypes, f: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):  # numpy flags NaN inputs on this cast
        return f.astype(ml_dtypes.bfloat16).view(np.uint16)


def test_encode_matches_ml_dtypes_bit_for_bit():
    """Every non-NaN input must encode to the identical 16 bits."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    f = _fp32_probe_bits().view(np.float32)
    keep = ~np.isnan(f)
    assert np.array_equal(f32_to_bf16(f)[keep], _ml_dtypes_encode(ml_dtypes, f)[keep])


def test_nan_encoding_matches_ml_dtypes_on_nanness_and_sign():
    """NaN stays NaN with the same sign on both sides; the payload is not compared."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    f = _fp32_probe_bits().view(np.float32)
    f = f[np.isnan(f)]
    assert f.size  # the probe set must actually contain NaNs
    ours = f32_to_bf16(f)
    theirs = _ml_dtypes_encode(ml_dtypes, f)
    assert np.isnan(bf16_to_f32(ours)).all()
    assert np.isnan(bf16_to_f32(theirs)).all()
    assert np.array_equal(ours >> 15, theirs >> 15)


def test_decode_matches_ml_dtypes_for_every_pattern():
    """All 65536 bf16 patterns -- NaN payloads included -- widen identically."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    u16 = np.arange(1 << 16, dtype=np.uint16)
    theirs = u16.view(ml_dtypes.bfloat16).astype(np.float32)
    assert np.array_equal(bf16_to_f32(u16).view(np.uint32), theirs.view(np.uint32))
