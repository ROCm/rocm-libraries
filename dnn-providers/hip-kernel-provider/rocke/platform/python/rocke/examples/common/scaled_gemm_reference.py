# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent finite low-bit GEMM fixtures and references shared by examples."""

from __future__ import annotations

import numpy as np

from ...core.dtypes import normalize_dtype
from ...core.storage import BitPacking


def _canon_lowbit(dtype):
    return {
        "fp8e4m3": "fp8",
        "bf8e5m2": "bf8",
        "fp6e2m3": "fp6",
        "fp6e3m2": "bf6",
        "fp4e2m1": "fp4",
    }[normalize_dtype(dtype)]


def decode_e8m0(encoded: np.ndarray) -> np.ndarray:
    """Decode finite unsigned E8M0 bytes without using kernel packing logic."""
    if encoded.dtype != np.uint8 or np.any(encoded == 0xFF):
        raise ValueError("expected finite uint8 E8M0 scales (0xff encodes NaN)")
    return np.ldexp(np.ones(encoded.shape), encoded.astype(np.int32) - 127)


def pack_fp6_codes(codes: np.ndarray) -> np.ndarray:
    """Pack four consecutive six-bit codes into three little-endian bytes."""
    if (
        codes.dtype != np.uint8
        or codes.ndim != 2
        or codes.shape[1] % 4
        or np.any(codes > 63)
    ):
        raise ValueError(
            "expected rank-2 uint8 FP6 codes in [0, 63] and K divisible by 4"
        )
    packed = b"".join(BitPacking(6).pack(row.tolist()) for row in codes)
    return np.frombuffer(packed, dtype=np.uint8).reshape(codes.shape[0], -1)


def decode_fp6(packed: np.ndarray, dtype: str = "fp6") -> np.ndarray:
    """Decode packed E2M3 or E3M2, including finite extrema and signed zero."""
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] % 3:
        raise ValueError("expected rank-2 uint8 FP6 bytes with row size divisible by 3")
    if dtype not in ("fp6", "fp6e2m3", "bf6", "fp6e3m2"):
        raise ValueError("FP6 dtype must be fp6/E2M3 or bf6/E3M2")
    # Bit addressing is independent of the packer's four-code groups.
    bits = np.unpackbits(packed, axis=1, bitorder="little").reshape(
        packed.shape[0], -1, 6
    )
    codes = (bits * (1 << np.arange(6))).sum(axis=-1)
    mantissa_bits, bias = (3, 1) if _canon_lowbit(dtype) == "fp6" else (2, 3)
    fraction = (codes & ((1 << mantissa_bits) - 1)) / (1 << mantissa_bits)
    exponent = (codes & 31) >> mantissa_bits
    magnitude = np.where(
        exponent == 0,
        np.ldexp(fraction, 1 - bias),
        np.ldexp(1 + fraction, exponent - bias),
    )
    return np.copysign(magnitude, np.where(codes & 32, -1.0, 1.0))


def _output_dtype(dtype: str):
    import ml_dtypes

    if dtype == "bf16":
        return ml_dtypes.bfloat16
    if dtype in ("fp16", "f16"):
        return np.float16
    raise ValueError(f"verifier output must be bf16/fp16 (got {dtype!r})")


def decode_fp4(packed: np.ndarray) -> np.ndarray:
    """Decode low-nibble-first E2M1 pairs by the format's exponent formula."""
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("expected a rank-2 uint8 packed FP4 matrix")
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    exponent = ((codes >> 1) & 3).astype(np.int32)
    mantissa = (codes & 1).astype(np.float64)
    magnitude = np.where(
        exponent == 0, mantissa * 0.5, np.ldexp(1.0 + mantissa * 0.5, exponent - 1)
    )
    return np.copysign(magnitude, np.where(codes & 8, -1.0, 1.0))


def pack_fp4_codes(codes: np.ndarray) -> np.ndarray:
    """Pack E2M1 codes, not arbitrary floating values; no quantization policy."""
    if (
        codes.dtype != np.uint8
        or codes.ndim != 2
        or codes.shape[1] % 2
        or np.any(codes > 15)
    ):
        raise ValueError("expected rank-2 uint8 FP4 codes in [0, 15] and even K")
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def decode_scale(encoded: np.ndarray, dtype: str) -> np.ndarray:
    """Decode finite nonnegative scale fixtures, independently of lowering."""
    if dtype in ("e8m0", "i8"):
        return decode_e8m0(encoded)
    if dtype not in ("e4m3", "e5m3"):
        raise ValueError("scale dtype must be e8m0, e4m3, or e5m3")
    maximum, bias = (126, 7) if dtype == "e4m3" else (254, 15)
    if encoded.dtype != np.uint8 or np.any(encoded > maximum):
        raise ValueError("expected finite nonnegative encoded scale bytes")
    exponent = (encoded >> 3).astype(np.int32)
    fraction = (encoded & 7).astype(np.float64) / 8
    return np.where(
        exponent == 0,
        np.ldexp(fraction, 1 - bias),
        np.ldexp(1 + fraction, exponent - bias),
    )


def reference_result(
    a: np.ndarray,
    b: np.ndarray,
    a_scale: np.ndarray,
    b_scale: np.ndarray,
    block_k: int,
    *,
    native: bool,
    dtype_a: str | None = None,
    dtype_b: str | None = None,
    dtype_c: str = "bf16",
    scale_dtype_a: str = "e8m0",
    scale_dtype_b: str = "e8m0",
) -> np.ndarray:
    """Expand scales onto logical A/B elements, multiply, then round the output.

    The native fixtures use quarter-integer inputs of magnitude <=1 and
    scales 2**[-2,3]. At K<=256, even the sum of absolute products fits in
    2**22 units of 2**-8, so every FP32 partial sum is exact. Float64 host
    arithmetic and a single output-type rounding provide an independent oracle.
    Alternative scale fixtures use the bounded dyadic values 0.5, 0.75, 1,
    1.25, 1.5, and 1.75 with the same bounded matrix inputs.
    FP6 all-code fixtures isolate one K element, avoiding accumulation error.
    Input decoding preserves signed zero, but this reference and its numerical
    comparison do not establish output zero-sign or accumulation sign semantics.
    FP4 fixtures cover all E2M1 values (magnitude <=6), scales 2**[-2,1],
    and K<=256: absolute partial sums are below 2**22 units of 2**-6,
    so FP32 accumulation is also exact before the final output-type rounding.
    """
    output_dtype = _output_dtype(dtype_c)

    sa = decode_scale(a_scale, scale_dtype_a) if native else a_scale.astype(np.float64)
    sb = decode_scale(b_scale, scale_dtype_b) if native else b_scale.astype(np.float64)

    def matrix_values(data, dtype):
        if dtype in ("fp6", "fp6e2m3", "bf6", "fp6e3m2"):
            return decode_fp6(data, dtype)
        if dtype in ("fp4", "fp4e2m1") or (dtype is None and data.dtype == np.uint8):
            return decode_fp4(data)
        return data.astype(np.float64)

    a_values = matrix_values(a, dtype_a)
    b_values = matrix_values(b, dtype_b)
    scaled_a = a_values * np.repeat(sa, block_k, axis=1)
    scaled_b = b_values * np.repeat(sb.T, block_k, axis=1)
    ref = scaled_a @ scaled_b.T
    if native:
        ref = ref.astype(output_dtype)
    return ref.astype(np.float32)


def make_case_inputs(
    spec, case: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create asymmetric mantissas and scales, optionally isolating a K group."""
    import ml_dtypes

    native = spec.resolved_matrix_path() != "wmma"
    if native and spec.K not in (64, 128, 256):
        raise ValueError("exact native fixtures require K=64, K=128 or K=256")
    groups = spec.K // spec.block_k
    lowbit_types = {
        "fp8": ml_dtypes.float8_e4m3fn,
        "fp8e4m3": ml_dtypes.float8_e4m3fn,
        "bf8": ml_dtypes.float8_e5m2,
        "bf8e5m2": ml_dtypes.float8_e5m2,
    }
    rng = np.random.default_rng(0xB10C)

    def operand(dtype, rows):
        kind = _canon_lowbit(dtype)
        if kind == "fp4":
            # Keep E2M1 codes unpacked until group masking is finished.
            return rng.integers(0, 16, size=(rows, spec.K), dtype=np.uint8)
        if kind in ("fp6", "bf6"):
            # Small dyadic values ensure exact FP32 partial sums in these tests.
            codes = rng.integers(
                0, 9 if kind == "fp6" else 13, size=(rows, spec.K), dtype=np.uint8
            )
            return codes | (rng.integers(0, 2, size=codes.shape, dtype=np.uint8) << 5)
        magnitude = 0.25 if native else 0.5
        return (rng.integers(-4, 5, size=(rows, spec.K)) * magnitude).astype(
            lowbit_types[dtype]
        )

    a, b = operand(spec.dtype_a, spec.M), operand(spec.dtype_b, spec.N)
    if native:
        small = any(
            _canon_lowbit(d) in ("fp6", "bf6", "fp4")
            for d in (spec.dtype_a, spec.dtype_b)
        )

        da, db = (
            spec.resolved_scale_dtypes()
            if hasattr(spec, "resolved_scale_dtypes")
            else ("e8m0", "e8m0")
        )

        def scales(dtype, shape):
            if dtype in ("e8m0", "i8"):
                return (
                    rng.integers(
                        125, 129 if small else 131, size=shape, dtype=np.uint8
                    ),
                    127,
                )
            if dtype not in ("e4m3", "e5m3"):
                raise ValueError("scale dtype must be e8m0, e4m3, or e5m3")
            bias = 7 if dtype == "e4m3" else 15
            # Include non-power-of-two scales to distinguish these from E8M0.
            codes = np.array(
                [
                    (bias - 1) * 8,
                    (bias - 1) * 8 + 4,
                    bias * 8,
                    bias * 8 + 2,
                    bias * 8 + 4,
                    bias * 8 + 6,
                ],
                dtype=np.uint8,
            )
            return rng.choice(codes, size=shape), bias * 8

        sa, neutral_a = scales(da, (spec.M, groups))
        sb, neutral_b = scales(db, (groups, spec.N))
    else:
        sa = rng.uniform(0.5, 1.5, size=(spec.M, groups)).astype(np.float32)
        sb = rng.uniform(0.5, 1.5, size=(groups, spec.N)).astype(np.float32)
        neutral_a = neutral_b = 1.0
    if case in ("neutral", "b-only"):
        sa.fill(neutral_a)
    if case in ("neutral", "a-only"):
        sb.fill(neutral_b)
    if case.startswith("codes-"):
        if (
            _canon_lowbit(spec.dtype_a) not in ("fp6", "bf6")
            or _canon_lowbit(spec.dtype_b) not in ("fp6", "bf6")
            or min(spec.M, spec.N) < 64
        ):
            raise ValueError("codes-N requires FP6/BF6 operands and M/N >= 64")
        k = int(case.removeprefix("codes-"))
        if not 0 <= k < spec.K:
            raise ValueError("codes-N K index outside the matrix")
        a.fill(0)
        b.fill(0)
        a[:, k] = np.arange(spec.M, dtype=np.uint8) % 64
        b[:, k] = np.arange(spec.N, dtype=np.uint8) % 64
        sa.fill(neutral_a)
        sb.fill(neutral_b)
    elif case.startswith("group-"):
        group = int(case.removeprefix("group-"))
        if not 0 <= group < groups:
            raise ValueError(f"scale group {group} outside [0, {groups})")
        active = (np.arange(spec.K) // spec.block_k) == group
        a[:, ~active] = 0
        b[:, ~active] = 0
    elif case not in ("neutral", "a-only", "b-only", "mixed"):
        raise ValueError(f"unknown verification case {case!r}")

    def pack(data, dtype):
        kind = _canon_lowbit(dtype)
        if kind == "fp4":
            return pack_fp4_codes(data)
        return pack_fp6_codes(data) if kind in ("fp6", "bf6") else data

    a, b = pack(a, spec.dtype_a), pack(b, spec.dtype_b)
    return a, b, sa, sb


def check_result(
    got: np.ndarray, expected: np.ndarray, *, exact: bool, tol: float = 2e-2
) -> None:
    """Fail on incomplete/non-finite output or a numerical mismatch.

    Exact mode requires numerical equality after output-type rounding, with
    positive and negative zero treated as equal. Tolerance applies only to the
    legacy software-scaled WMMA path.
    """
    if got.shape != expected.shape:
        raise AssertionError(f"output shape {got.shape} != {expected.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(got).all():
        raise AssertionError("unexpected non-finite output or reference")
    diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    bad = got != expected if exact else diff > tol * np.maximum(np.abs(expected), 1)
    if np.any(bad):
        worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"bad={np.count_nonzero(bad)}/{diff.size}; worst={worst}: "
            f"got={got[worst]}, expected={expected[worst]}, diff={diff[worst]}"
        )
