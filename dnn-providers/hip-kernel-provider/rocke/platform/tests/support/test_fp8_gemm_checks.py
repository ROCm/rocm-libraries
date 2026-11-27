# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fp8 / bf8 operand bytes the GPU checks of the scaled GEMMs feed an MFMA,
the host codec of each element format, and the exact-operand output bound.

Each arch's code for every exact operand value is checked against the code
written out from the format: sign, biased exponent, mantissa. The OCP formats
(gfx950) have biases 7 (E4M3) and 15 (E5M2); the FNUZ formats (gfx942) the
same layouts with the bias one higher.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.helpers.fp8_encoding import (
    BF8_E5M2_FNUZ,
    BF8_E5M2_OCP,
    FP8_E4M3_FNUZ,
    FP8_E4M3_OCP,
    Fp8ElementFormat,
)
from rocke.tests.fp8_gemm_checks import (
    EXACT_OPERAND_VALUES,
    exact_operand_tolerance,
    fp8_host_codec,
    mfma_operand_bytes,
)

# (arch, mantissa) -> (exponent bias, mantissa bits) of the encoding its MFMA reads.
_FORMATS = {
    ("gfx950", "fp8e4m3"): (7, 3),
    ("gfx950", "bf8e5m2"): (15, 2),
    ("gfx942", "fp8e4m3"): (8, 3),
    ("gfx942", "bf8e5m2"): (16, 2),
}


def _code(value: float, bias: int, mantissa_bits: int) -> int:
    """The code of zero or a normal ``value`` whose mantissa fits."""
    if value == 0:
        return 0
    exponent = int(np.floor(np.log2(abs(value))))
    field = (abs(value) / 2.0**exponent - 1.0) * 2**mantissa_bits
    sign = 0x80 if value < 0 else 0
    return sign | ((exponent + bias) << mantissa_bits) | int(field)


@pytest.mark.parametrize("arch,mantissa", sorted(_FORMATS))
def test_every_exact_value_gets_the_code_of_the_arch_format(arch, mantissa):
    bias, mantissa_bits = _FORMATS[(arch, mantissa)]
    got = mfma_operand_bytes(arch, mantissa, EXACT_OPERAND_VALUES)
    want = [_code(value, bias, mantissa_bits) for value in EXACT_OPERAND_VALUES]
    assert got.dtype == np.uint8
    assert got.tolist() == want


def test_a_value_the_format_does_not_hold_is_refused():
    with pytest.raises(ValueError, match="not exactly representable in fp8e4m3"):
        mfma_operand_bytes("gfx950", "fp8e4m3", np.array([0.3]))


def test_a_fnuz_value_whose_double_ocp_does_not_hold_is_refused():
    # 240 is the largest FNUZ E4M3 value, but its code is looked up as OCP E4M3's code of 480,
    # beyond OCP's largest value (448).
    with pytest.raises(
        ValueError, match="480.0 .* not exactly representable in fp8e4m3"
    ):
        mfma_operand_bytes("gfx942", "fp8e4m3", np.array([240.0]))


def test_an_arch_without_an_mfma_fp8_format_is_refused():
    with pytest.raises(ValueError, match="no fp8e4m3 element format known for gfx90a"):
        mfma_operand_bytes("gfx90a", "fp8e4m3", EXACT_OPERAND_VALUES)


_ENCODINGS = (FP8_E4M3_OCP, BF8_E5M2_OCP, FP8_E4M3_FNUZ, BF8_E5M2_FNUZ)


@pytest.mark.parametrize("encoding", _ENCODINGS, ids=lambda value: value.name)
def test_host_codec_decodes_the_format_largest_value(encoding):
    """Each encoding's largest_value is the largest magnitude its host codec
    decodes, so the quantization bound and the host reference cannot drift
    apart; and the codec is FNUZ exactly for the FNUZ encodings."""
    codec = fp8_host_codec(encoding)
    assert encoding.largest_value == float(np.nanmax(np.abs(codec.decode_table())))
    assert codec.is_fnuz == (encoding in (FP8_E4M3_FNUZ, BF8_E5M2_FNUZ))


@pytest.mark.parametrize(
    "encoding", (FP8_E4M3_OCP, FP8_E4M3_FNUZ), ids=lambda value: value.name
)
def test_host_codec_rounds_e4m3_to_codes_it_decodes(encoding):
    """An E4M3 codec rounds each finite value it decodes back to that value."""
    codec = fp8_host_codec(encoding)
    table = codec.decode_table()
    finite = table[np.isfinite(table)]
    np.testing.assert_array_equal(table[codec.round_to_codes(finite)], finite)


@pytest.mark.parametrize(
    "encoding", (BF8_E5M2_OCP, BF8_E5M2_FNUZ), ids=lambda value: value.name
)
def test_host_codec_of_e5m2_has_no_rounding(encoding):
    assert fp8_host_codec(encoding).round_to_codes is None


def test_host_codec_of_an_unknown_format_is_refused():
    with pytest.raises(KeyError, match="fp8_host_codec failed - element_format="):
        fp8_host_codec(Fp8ElementFormat("E3M4", 15.5))


def test_the_exact_operand_bound_is_the_f32_rounding_of_each_output():
    """``groups + 2`` f32 roundings in the kernel and as many in the reference,
    each at most a unit roundoff (2**-24) of the output's term magnitude."""
    magnitude = np.array([[0.0, 1.0], [512.0, 3.5]])
    tolerance = exact_operand_tolerance(magnitude, 7)
    np.testing.assert_array_equal(tolerance.atol, 18 * 2.0**-24 * magnitude)
    assert tolerance.rtol == 0.0


@pytest.mark.parametrize("groups", [0, -1, 2.0, True])
def test_the_exact_operand_bound_refuses_a_group_count_that_is_not_positive(groups):
    with pytest.raises(
        ValueError,
        match=rf"exact_operand_tolerance failed - groups={groups!r}, expected an int",
    ):
        exact_operand_tolerance(np.ones(2), groups)
