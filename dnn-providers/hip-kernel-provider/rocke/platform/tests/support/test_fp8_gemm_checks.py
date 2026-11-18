# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fp8 / bf8 operand bytes the GPU checks of the scaled GEMMs feed an MFMA.

Each arch's code for every exact operand value is checked against the code
written out from the format: sign, biased exponent, mantissa. The OCP formats
(gfx950) have biases 7 (E4M3) and 15 (E5M2); the FNUZ formats (gfx942) the
same layouts with the bias one higher.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.tests.fp8_gemm_checks import EXACT_OPERAND_VALUES, mfma_operand_bytes

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
