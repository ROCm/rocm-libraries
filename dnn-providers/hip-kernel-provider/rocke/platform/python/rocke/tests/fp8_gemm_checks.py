# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: what the GPU checks of the fp8 / bf8 GEMM kernels share.

A GPU check of a scaled fp8 / bf8 GEMM fills its operands with numbers the
arch's MFMA reads exactly, so the reference is computed on the values the
device sees, and accepts the kernel's output within the runbook's GEMM
tolerance. :data:`GEMM_TOLERANCE`, :data:`EXACT_OPERAND_VALUES` and
:func:`mfma_operand_bytes` are those three facts, kept here so the checks
cannot drift apart. :func:`fp8_host_codec` is the one map from an element
format to the host codec that decodes and rounds its codes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..helpers.fp8_encoding import (
    BF8_E5M2_FNUZ,
    BF8_E5M2_OCP,
    FP8_E4M3_FNUZ,
    FP8_E4M3_OCP,
    Fp8ElementFormat,
)
from ..instances.common.mx_gemm import mx_gemm_element_format
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
    "fp8_host_codec",
    "mfma_operand_bytes",
]

# The runbook's GEMM tolerance (optimization_runbook.md section 1.4), as an
# error relative to the largest reference output.
GEMM_TOLERANCE = 7e-2

# Operand values every fp8 / bf8 encoding an MFMA reads (OCP and FNUZ, E4M3
# and E5M2) holds exactly.
EXACT_OPERAND_VALUES = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])

_OCP_ENCODERS = {"fp8e4m3": fp8e4m3_encode, "bf8e5m2": bf8e5m2_encode}


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
