# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for bfloat16 byte conversion helpers in buffer_manager.

These tests lock in the numpy bit-manipulation behaviour of the bf16
helpers. The forward conversion is round-to-nearest, ties-to-even
(RNE), matching ``torch.Tensor.bfloat16()``. NaN inputs are preserved
as NaN with the bf16 quiet bit forced on.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dnn_benchmarking.execution.buffer_manager import (
    BufferManager,
    _bfloat16_bytes_to_ndarray,
    _f32_to_bf16_bytes,
    _generate_bfloat16_bytes,
)
from dnn_benchmarking.graph.tensor_info import TensorInfo


def _bf16_uint16(values: np.ndarray) -> np.ndarray:
    """Decode raw bf16 bytes from ``_f32_to_bf16_bytes(values)`` into a uint16 array."""
    raw = _f32_to_bf16_bytes(values)
    return np.frombuffer(raw, dtype=np.uint16).reshape(values.shape)


class TestF32ToBf16RoundtripRNE:
    """Roundtrip and exactness tests for the RNE conversion."""

    def test_values_with_zero_low_bits_roundtrip_exactly(self) -> None:
        """Values whose f32 low 16 bits are already zero roundtrip exactly."""
        x = np.array([1.0, -1.0, 0.0, 0.5, 2.0, 4.0, 100.0, -7.5], dtype=np.float32)
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, list(x.shape))
        np.testing.assert_array_equal(result, x)

    def test_round_up_when_low_bits_above_half(self) -> None:
        """Low 16 bits > 0x8000 round up (toward larger magnitude)."""
        # f32 bit pattern 0x3F8080FF: low half-word 0x80FF > 0x8000 -> round up.
        # Result bf16 = 0x3F81.
        bits = np.array([0x3F8080FF], dtype=np.uint32)
        x = bits.view(np.float32)
        assert _bf16_uint16(x)[0] == 0x3F81

    def test_round_down_when_low_bits_below_half(self) -> None:
        """Low 16 bits < 0x8000 round down (toward zero)."""
        bits = np.array([0x3F807FFF], dtype=np.uint32)
        x = bits.view(np.float32)
        assert _bf16_uint16(x)[0] == 0x3F80

    def test_tie_rounds_to_even_lsb_zero(self) -> None:
        """Exact tie (low bits == 0x8000) with even bf16 LSB stays put."""
        # 0x3F800000 has bf16 word 0x3F80 (LSB=0, even). Tie -> stay at 0x3F80.
        bits = np.array([0x3F808000], dtype=np.uint32)
        x = bits.view(np.float32)
        assert _bf16_uint16(x)[0] == 0x3F80

    def test_tie_rounds_to_even_lsb_one(self) -> None:
        """Exact tie with odd bf16 LSB rounds up to even."""
        # 0x3F810000 has bf16 word 0x3F81 (LSB=1, odd). Tie -> round up to 0x3F82.
        bits = np.array([0x3F818000], dtype=np.uint32)
        x = bits.view(np.float32)
        assert _bf16_uint16(x)[0] == 0x3F82

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
    ``_f32_to_bf16_bytes`` to exercise the decoder in isolation.
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


class TestForwardPathNaNPreservation:
    """NaN inputs must encode to a bf16 NaN (not overflow to infinity).

    The forward path forces the bf16 quiet bit on for NaN inputs and
    skips rounding so the exponent cannot overflow.
    """

    def test_quiet_nan_input_encodes_to_nan(self) -> None:
        x = np.array([np.float32(np.nan)], dtype=np.float32)
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, [1])
        assert np.isnan(result[0])

    def test_signaling_nan_payload_still_nan(self) -> None:
        # Craft a signalling NaN whose payload is entirely in the low
        # 16 mantissa bits — naive truncation would corrupt this to inf.
        bits = np.array([0x7F800001], dtype=np.uint32)
        x = bits.view(np.float32)
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, [1])
        assert np.isnan(result[0])


class TestRNETinyValues:
    """Subnormal-input behaviour under RNE.

    f32 values smaller than half the smallest bf16 subnormal round to
    zero because the RNE bias cannot lift their bit pattern past the
    bf16 LSB.
    """

    def test_smallest_f32_subnormal_rounds_to_zero(self) -> None:
        # 0x00000001 is the smallest positive f32 subnormal (~1.4e-45).
        # +bias 0x7FFF = 0x00008000; lsb=0 so no even-up; >>16 -> 0x0000.
        bits = np.array([0x00000001], dtype=np.uint32)
        x = bits.view(np.float32)
        raw = _f32_to_bf16_bytes(x)
        result = _bfloat16_bytes_to_ndarray(raw, [1])
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


def _make_bf16_input_tensor(uid: int = 1) -> TensorInfo:
    return TensorInfo(
        uid=uid,
        name=f"input_{uid}",
        dims=[2, 4],
        strides=[],
        data_type="bfloat16",
        is_virtual=False,
        is_output=False,
    )


class TestFillInputsRandomReproducibility:
    """``fill_inputs_random(seed=...)`` must be reproducible for all dtypes,
    including the bfloat16 path. Regression test: earlier the bf16 branch
    created its own unseeded ``RandomState``, ignoring the caller's seed.
    """

    def test_bfloat16_input_reproducible_across_runs(self) -> None:
        tensor = _make_bf16_input_tensor(uid=42)

        bm_a = BufferManager([tensor])
        bm_b = BufferManager([tensor])
        # Bypass allocate_all() (needs hipdnn). Seed buffer dict with mocks
        # so fill_inputs_random does not raise and copy_from_host is callable.
        bm_a._buffers[tensor.uid] = MagicMock()
        bm_b._buffers[tensor.uid] = MagicMock()

        bm_a.fill_inputs_random(seed=1234)
        bm_b.fill_inputs_random(seed=1234)

        data_a = bm_a.get_input_data(tensor.uid)
        data_b = bm_b.get_input_data(tensor.uid)
        np.testing.assert_array_equal(data_a, data_b)

        # Sanity: a different seed produces different bytes.
        bm_c = BufferManager([tensor])
        bm_c._buffers[tensor.uid] = MagicMock()
        bm_c.fill_inputs_random(seed=5678)
        data_c = bm_c.get_input_data(tensor.uid)
        assert not np.array_equal(data_a, data_c)

    def test_bfloat16_copy_from_host_bytes_match_across_runs(self) -> None:
        tensor = _make_bf16_input_tensor(uid=7)

        bm_a = BufferManager([tensor])
        bm_b = BufferManager([tensor])
        mock_a = MagicMock()
        mock_b = MagicMock()
        bm_a._buffers[tensor.uid] = mock_a
        bm_b._buffers[tensor.uid] = mock_b

        bm_a.fill_inputs_random(seed=99)
        bm_b.fill_inputs_random(seed=99)

        bytes_a = mock_a.copy_from_host.call_args[0][0]
        bytes_b = mock_b.copy_from_host.call_args[0][0]
        assert bytes_a == bytes_b
