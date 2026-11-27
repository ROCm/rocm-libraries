# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: what the GPU checks of the fp8 / bf8 GEMM kernels share, and
the fp8 / bf8 host codec every test uses.

A GPU check of a scaled fp8 / bf8 GEMM fills its operands with numbers the
arch's MFMA reads exactly (:data:`EXACT_OPERAND_VALUES`, encoded by
:func:`mfma_operand_bytes`), so the reference is computed on the values the
device sees. It then holds the output to one of two bounds, by what the
output depends on:

* :func:`exact_operand_tolerance`: an output computed from those operands and
  f32 scales only (the block-scaled GEMM). Every group's dot product is exact
  in f32, so the output is exact up to the f32 rounding of its scaling and
  sums, and the bound is that rounding, element by element.
* :data:`GEMM_TOLERANCE`: an output that also passes through a rounding to an
  fp8 type the reference cannot reproduce bit for bit (the fused MoE, which
  requantizes its hidden activation: an f32 value one ulp from a rounding
  boundary takes a different code in the kernel than in the reference). That
  is the runbook's bound for random fp8 data, relative to the largest
  reference output.

These facts are kept here so the checks cannot drift apart.

:func:`fp8_host_codec` is the home of the fp8 / bf8 host codec for all tests,
GEMM or not: the one map from an element format (as
:func:`~rocke.helpers.fp8_encoding.fp8_element_format` gives it for a dtype on
an arch) to the codec that decodes and rounds its codes. A test that needs to
decode or round device fp8 / bf8 codes takes the codec from here rather than
choosing between the FNUZ and OCP codecs itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..helpers.fp8_encoding import (
    BF8_E5M2_FNUZ,
    BF8_E5M2_OCP,
    FP8_E4M3_FNUZ,
    FP8_E4M3_OCP,
    Fp8ElementFormat,
)
from ..instances.common.mx_gemm import mx_gemm_element_format
from ..numeric.compare import Tolerance
from ..numeric.fp8 import (
    bf8e5m2_decode_table,
    bf8e5m2_encode,
    bf8e5m2fnuz_decode_table,
    fp8e4m3_decode_table,
    fp8e4m3_encode,
    fp8e4m3_round,
    fp8e4m3fnuz_decode_table,
    fp8e4m3fnuz_round,
)

__all__ = [
    "EXACT_OPERAND_VALUES",
    "GEMM_TOLERANCE",
    "Fp8HostCodec",
    "exact_operand_tolerance",
    "fp8_host_codec",
    "mfma_operand_bytes",
]

# The runbook's GEMM tolerance (optimization_runbook.md section 1.4), as an
# error relative to the largest reference output: for outputs that pass
# through an fp8 requantization (see the module docstring), not for outputs
# of exact operands.
GEMM_TOLERANCE = 7e-2

# Operand values every fp8 / bf8 encoding an MFMA reads (OCP and FNUZ, E4M3
# and E5M2) holds exactly.
EXACT_OPERAND_VALUES = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])

_OCP_ENCODERS = {"fp8e4m3": fp8e4m3_encode, "bf8e5m2": bf8e5m2_encode}

# The unit roundoff of f32.
_F32_UNIT_ROUNDOFF = 2.0**-24


def exact_operand_tolerance(term_magnitude: npt.ArrayLike, groups: int) -> Tolerance:
    """The element-wise bound of a scaled fp8 / bf8 GEMM output whose operands
    are :data:`EXACT_OPERAND_VALUES` and whose ``groups`` K groups are each
    scaled by an f32 factor and summed in f32.

    ``term_magnitude`` holds, per output, the sum over its K groups of
    ``|dot_g * scale_g|``. A group's products are multiples of 1/4 of at most
    4 in magnitude, so a group of fewer than 2**20 of them sums exactly in f32;
    what is left is the f32 rounding of the scale product, of the scaled group
    term and of the running sum, each at most one unit roundoff of a value no
    larger than ``term_magnitude``: ``groups + 2`` of them in the kernel and as
    many in an f32 reference, whatever order either takes.
    """
    if isinstance(groups, bool) or not (isinstance(groups, int) and groups >= 1):
        raise ValueError(
            f"exact_operand_tolerance failed - groups={groups!r}, expected an "
            "int >= 1"
        )
    rounding = 2 * (groups + 2) * _F32_UNIT_ROUNDOFF
    return Tolerance(atol=rounding * np.asarray(term_magnitude, dtype=np.float64))


@dataclass(frozen=True)
class Fp8HostCodec:
    """The host codec of one fp8 / bf8 element format, for a reference that
    works on the codes the device writes or reads.

    ``is_fnuz`` says whether the format is a FNUZ one (the OCP bit layout with
    the exponent bias one higher). ``decode_table`` builds the f32 value of
    each of the 256 codes; ``round_to_codes`` rounds f32 values to the nearest
    codes, and is ``None`` for the E5M2 formats, which have no rounding codec.
    """

    is_fnuz: bool
    decode_table: Callable[[], np.ndarray]
    round_to_codes: Callable[[np.ndarray], np.ndarray] | None


_HOST_CODECS = {
    FP8_E4M3_OCP: Fp8HostCodec(False, fp8e4m3_decode_table, fp8e4m3_round),
    BF8_E5M2_OCP: Fp8HostCodec(False, bf8e5m2_decode_table, None),
    FP8_E4M3_FNUZ: Fp8HostCodec(True, fp8e4m3fnuz_decode_table, fp8e4m3fnuz_round),
    BF8_E5M2_FNUZ: Fp8HostCodec(True, bf8e5m2fnuz_decode_table, None),
}


def fp8_host_codec(element_format: Fp8ElementFormat) -> Fp8HostCodec:
    """The host codec of ``element_format``, as
    :func:`~rocke.helpers.fp8_encoding.fp8_element_format` gives it for a dtype
    on an arch. Raises ``KeyError`` for a format that is not one of the four
    fp8 / bf8 encodings."""
    if element_format not in _HOST_CODECS:
        raise KeyError(
            f"fp8_host_codec failed - element_format={element_format!r}, "
            f"expected one of {[known.name for known in _HOST_CODECS]}"
        )
    return _HOST_CODECS[element_format]


def mfma_operand_bytes(
    arch: str, mantissa_dtype: str, values: np.ndarray
) -> np.ndarray:
    """The ``uint8`` codes ``arch``'s MFMA reads as ``values`` for the
    ``mantissa_dtype`` element type (``"fp8e4m3"`` or ``"bf8e5m2"``), in the
    encoding that MFMA reads
    (:func:`~rocke.instances.common.mx_gemm.mx_gemm_element_format`, which
    refuses an arch without one).

    A FNUZ format has the OCP bit layout with an exponent bias one higher, so a
    value's FNUZ code is the OCP code of twice the value (zero is 0x00 in
    both). Every value must be held exactly, and for FNUZ its double too, by
    the OCP format; otherwise the OCP encoder raises ``ValueError``. So the
    largest FNUZ E4M3 value, 240, is refused: OCP E4M3 does not hold 480.
    """
    element_format = mx_gemm_element_format(mantissa_dtype, arch)
    is_fnuz = fp8_host_codec(element_format).is_fnuz
    return _OCP_ENCODERS[mantissa_dtype](2.0 * values if is_fnuz else values)
