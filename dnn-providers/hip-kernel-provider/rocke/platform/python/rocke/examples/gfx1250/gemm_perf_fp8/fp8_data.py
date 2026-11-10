# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""fp8 (OCP e4m3) operand data for the gfx1250 fp8 GEMM measurements.

Two input sets, because the WMMA rate itself depends on the operand bits (see
the case study, "The ceiling depends on the operand data"):

``int``
    Integers in -5..5. Each is exact in e4m3 and every fp32 partial sum stays
    an integer below 2^24 for K below ~670k, so the kernel's output must equal
    the reference bit for bit.
``rand``
    ``normal(0, 0.1)`` rounded to e4m3. Closer to real operand data and
    measurably slower for the same kernel; checked against a tolerance (see
    :func:`rand_rel_error_bound`).

Encoding and rounding come from :mod:`rocke.numeric.fp8`, so there is one
definition of the wire format.
"""

from __future__ import annotations

from rocke.numeric.fp8 import fp8e4m3_decode_table, fp8e4m3_encode, fp8e4m3_round

#: Seeds of the two input sets, fixed so every run sees the same bytes.
INT_SEED = 0xC0FFEE
RAND_SEED = 42


def e4m3_decode(np, codes):
    """e4m3 bytes -> f32 values."""
    return fp8e4m3_decode_table()[np.asarray(codes, dtype=np.uint8)]


def int_operands(np, m: int, n: int, k: int) -> tuple[object, object, object, object]:
    """``(a_bytes, b_bytes, a_f32, b_f32)`` for the exact integer set."""
    rng = np.random.default_rng(INT_SEED)
    a = rng.integers(-5, 6, size=(m, k), dtype=np.int16).astype(np.float32)
    b = rng.integers(-5, 6, size=(n, k), dtype=np.int16).astype(np.float32)
    return fp8e4m3_encode(a), fp8e4m3_encode(b), a, b


def rand_operands(np, m: int, n: int, k: int) -> tuple[object, object, object, object]:
    """``(a_bytes, b_bytes, a_f32, b_f32)`` for the random set.

    The f32 arrays are the decoded bytes, i.e. exactly what the kernel
    multiplies, so the reference differs from the kernel only by summation
    order and the final rounding to bf16.
    """
    rng = np.random.default_rng(RAND_SEED)
    a8 = fp8e4m3_round(rng.standard_normal((m, k), dtype=np.float32) * 0.1)
    b8 = fp8e4m3_round(rng.standard_normal((n, k), dtype=np.float32) * 0.1)
    return a8, b8, e4m3_decode(np, a8), e4m3_decode(np, b8)


def rand_rel_error_bound() -> float:
    """Tolerance for ``max|C - ref| / max|ref|`` on the random set: 2^-7.

    The bf16 output alone rounds by up to 2^-8 of an element (half an ulp just
    above a power of two); the rest is headroom for fp32 summation order over a
    long K. A layout or synchronisation bug is orders of magnitude above it.
    """
    return 2.0**-7
