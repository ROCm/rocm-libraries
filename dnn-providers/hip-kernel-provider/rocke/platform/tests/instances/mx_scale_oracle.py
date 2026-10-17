# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent oracle for the E8M0 scale decoder and the MX GEMM that uses it.

Shared by ``test_mx_scale_decode.py`` (host) and ``test_mx_gemm_scale_numeric.py``
(GPU). Nothing here calls the implementation under test. The E8M0 reference is
``2^(code - 127)`` for codes 0..254 and NaN for code 255, as the OCP MX E8M0
format defines. ``math.ldexp`` and ``np.ldexp`` compute the powers of two.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARITY = HERE / "parity"

KERNEL = "rocke_mx_scale_decode"
NUM_CODES = 256
NAN_CODE = 255
SUBNORMAL_BITS = 0x00400000  # f32 bits of 2^-127, the value for code 0


@cache
def emitter():
    """Load the decoder parity emitter by path. Its directory is not a package."""
    path = PARITY / "mx_scale_decode_emit.py"
    sys.path.insert(0, str(PARITY))  # the emitter imports _emit_common
    try:
        spec = importlib.util.spec_from_file_location("mx_scale_decode_emit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PARITY))
    return module


def expected_bits(code: int, *, flush_subnormals: bool = False) -> int | None:
    """f32 bits of ``2^(code - 127)``, or ``None`` for the NaN code.

    ``math.ldexp`` is exact in f64. Every finite result is exact in f32:
    ``2^-127`` is a subnormal f32. With ``flush_subnormals`` the subnormal
    result for code 0 reads as +0.0, as on a device that flushes f32 denormals.
    """
    if code == NAN_CODE:
        return None
    value = np.float32(math.ldexp(1.0, code - 127))
    if flush_subnormals and 0.0 < value < np.finfo(np.float32).tiny:
        return 0
    return int(value.view(np.uint32))


def mismatches(
    bits: np.ndarray, *, flush_subnormals: bool = False
) -> list[tuple[int, str, str]]:
    """Codes whose f32 bits differ from the oracle, as (code, got, want)."""
    assert bits.shape == (NUM_CODES,) and bits.dtype == np.uint32
    values = bits.view(np.float32)
    bad = []
    for code in range(NUM_CODES):
        want = expected_bits(code, flush_subnormals=flush_subnormals)
        if want is None:
            if not np.isnan(values[code]):
                bad.append((code, f"{bits[code]:#010x}", "NaN"))
        elif int(bits[code]) != want:
            bad.append((code, f"{bits[code]:#010x}", f"{want:#010x}"))
    return bad


def check_all_codes(bits: np.ndarray, *, flush_subnormals: bool = False) -> None:
    """Assert the f32 bit patterns for codes 0..255 against the oracle."""
    bad = mismatches(bits, flush_subnormals=flush_subnormals)
    assert not bad, f"{len(bad)} codes differ (code, got, want): {bad[:8]}"
    values = bits.view(np.float32)

    # Explicit NaN and finiteness checks, separate from the table above.
    assert np.isnan(values[NAN_CODE])
    assert not np.isnan(values[:NAN_CODE]).any()
    assert np.isfinite(values[:NAN_CODE]).all()

    # Code 0 is 2^-127, a subnormal. E8M0 has no zero code, but a device that
    # flushes f32 denormals reads this value as +0.0.
    if flush_subnormals:
        assert int(bits[0]) == 0
    else:
        assert int(bits[0]) == SUBNORMAL_BITS
        assert 0.0 < values[0] < np.finfo(np.float32).tiny

    # Codes 128..254 stay positive. A signed extension would zero them.
    assert (values[128:NAN_CODE] > 1.0).all()

    # For codes 1..254 the f32 exponent field equals the code.
    codes = np.arange(1, NAN_CODE, dtype=np.uint32)
    assert (bits[1:NAN_CODE] == (codes << 23)).all()

    # Each step of the code doubles the value, exactly. Skip the step from a
    # flushed code 0.
    start = 1 if flush_subnormals else 0
    assert (values[start + 1 : NAN_CODE] == 2.0 * values[start : NAN_CODE - 1]).all()


def dequantized_gemm(
    a_mantissa: np.ndarray,
    b_mantissa: np.ndarray,
    a_codes: np.ndarray,
    b_codes: np.ndarray,
    group_k: int = 32,
) -> np.ndarray:
    """Reference f32 GEMM from dequantized operands.

    ``a_mantissa`` is (M, K) and ``b_mantissa`` is (K, N), both already decoded
    to numbers. ``a_codes`` is (M, K / group_k) and ``b_codes`` is
    (K / group_k, N), both raw E8M0 bytes. The math runs in f64 and rounds once
    to f32. A NaN scale makes every output that sums over its group NaN.
    """

    def scale(codes: np.ndarray) -> np.ndarray:
        values = np.ldexp(1.0, codes.astype(np.int64) - 127)
        values[codes == NAN_CODE] = np.nan
        return values

    a = a_mantissa.astype(np.float64) * np.repeat(scale(a_codes), group_k, axis=1)
    b = b_mantissa.astype(np.float64) * np.repeat(scale(b_codes), group_k, axis=0)
    return (a @ b).astype(np.float32)
