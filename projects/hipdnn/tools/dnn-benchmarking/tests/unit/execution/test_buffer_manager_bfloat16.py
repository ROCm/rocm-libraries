# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for bfloat16 byte conversion helpers in buffer_manager.

These tests lock in the numpy bit-manipulation behaviour of the bf16
helpers. The forward conversion is plain truncation of the low 16
mantissa bits, NOT round-to-nearest-even (which is what
``torch.Tensor.bfloat16()`` does). Outputs can therefore differ from a
torch round-trip by up to 1 ULP. See ``_f32_to_bf16_bytes`` docstring.
"""

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import (
    _bfloat16_bytes_to_ndarray,
    _f32_to_bf16_bytes,
    _generate_bfloat16_bytes,
)


def _truncate_f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Reference implementation: zero the low 16 bits of each f32 word."""
    bits = values.astype(np.float32).view(np.uint32)
    truncated_bits = bits & np.uint32(0xFFFF0000)
    return truncated_bits.view(np.float32).copy()


class TestF32ToBf16Roundtrip:
    """Roundtrip identity tests for the truncation conversion."""

    def test_roundtrip_matches_truncated_f32(self) -> None:
        """f32 -> bf16 bytes -> f32 equals f32 with low 16 mantissa bits zeroed."""
        x = np.array([1.0, -1.0, 0.0, 3.14, 0.5, 100.0, 1e-30, -7.5], dtype=np.float32)
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, [8])

        expected = _truncate_f32_to_bf16(x)
        np.testing.assert_array_equal(result, expected)

    def test_zero_and_negative_zero_preserved(self) -> None:
        """Both +0.0 and -0.0 roundtrip with sign bit intact."""
        x = np.array([0.0, -0.0], dtype=np.float32)
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, [2])

        assert result[0] == 0.0
        assert result[1] == 0.0
        assert not np.signbit(result[0])
        assert np.signbit(result[1])


class TestReversePathSpecialValues:
    """Tests for the reverse (bf16 bytes -> f32) path on special values.

    These craft raw uint16 buffers directly rather than going through
    ``_f32_to_bf16_bytes`` because truncation can corrupt NaN payloads
    (a quiet NaN's signalling bit may sit in the low 16 mantissa bits).
    """

    def test_positive_infinity_decoded(self) -> None:
        """uint16 0x7F80 decodes to +inf."""
        buf = np.array([0x7F80], dtype=np.uint16).tobytes()
        result = _bfloat16_bytes_to_ndarray(buf, [1])
        assert np.isinf(result[0])
        assert result[0] > 0

    def test_negative_infinity_decoded(self) -> None:
        """uint16 0xFF80 decodes to -inf."""
        buf = np.array([0xFF80], dtype=np.uint16).tobytes()
        result = _bfloat16_bytes_to_ndarray(buf, [1])
        assert np.isinf(result[0])
        assert result[0] < 0

    def test_nan_decoded(self) -> None:
        """uint16 0x7FC0 (quiet NaN) decodes to NaN."""
        buf = np.array([0x7FC0], dtype=np.uint16).tobytes()
        result = _bfloat16_bytes_to_ndarray(buf, [1])
        assert np.isnan(result[0])


class TestSubnormalTruncation:
    """Behaviour-locking test for subnormal/tiny-value flush behaviour.

    Picks an f32 value far below the bf16 normal range. Plain
    truncation flushes the value to zero. If someone later switches the
    conversion to round-to-nearest-even, this test will fail and force
    them to consider the behavioural change.
    """

    def test_tiny_f32_flushes_to_zero_under_truncation(self) -> None:
        # Smallest positive normal f32 (~1.175e-38) — well below the
        # smallest bf16 normal (~1.175e-38 has top 16 bits 0x0080,
        # but a value just under it has top 16 bits 0x0000).
        x = np.array([1e-40], dtype=np.float32)  # subnormal f32
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, [1])

        # Truncation: subnormal f32 has exponent bits all zero, and the
        # significant bits live in the low 23. Top 16 bits are 0 -> 0.0.
        assert result[0] == 0.0


class TestGenerateBfloat16BytesRngDeterminism:
    """Generator is deterministic when given a seeded RandomState."""

    def test_same_seed_produces_same_bytes(self) -> None:
        rng_a = np.random.RandomState(42)
        rng_b = np.random.RandomState(42)
        bytes_a = _generate_bfloat16_bytes([16], rng=rng_a)
        bytes_b = _generate_bfloat16_bytes([16], rng=rng_b)
        assert bytes_a == bytes_b

    def test_output_size_matches_dims(self) -> None:
        rng = np.random.RandomState(0)
        raw = _generate_bfloat16_bytes([16], rng=rng)
        # 16 elements * 2 bytes per bf16 = 32
        assert len(raw) == 32


class TestTorchParity:
    """Optional parity check against torch.bfloat16() for inputs where
    truncation and round-to-nearest-even agree.

    A bf16 truncation matches torch's RNE whenever bit 15 of the f32
    mantissa is 0 (no rounding needed). Values like 1.0, 2.0, 4.0, 0.5
    have all-zero low mantissa bits and are safe.
    """

    def test_truncation_matches_torch_when_low_bits_zero(self) -> None:
        torch = pytest.importorskip("torch")

        x = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float32)
        numpy_bytes = _f32_to_bf16_bytes(x)

        torch_bf16 = torch.from_numpy(x).bfloat16().contiguous()
        # View as uint16 to get raw bytes (bf16 is 2 bytes per element).
        torch_bytes = torch_bf16.view(torch.uint8).numpy().tobytes()

        assert numpy_bytes == torch_bytes
