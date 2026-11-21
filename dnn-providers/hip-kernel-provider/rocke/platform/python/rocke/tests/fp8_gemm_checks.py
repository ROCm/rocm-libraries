# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: what the GPU checks of the fp8 / bf8 GEMM kernels share.

A GPU check of a scaled fp8 / bf8 GEMM fills its operands with numbers the
arch's MFMA reads exactly, so the reference is computed on the values the
device sees, and accepts the kernel's output within the runbook's GEMM
tolerance. :data:`GEMM_TOLERANCE`, :data:`EXACT_OPERAND_VALUES` and
:func:`mfma_operand_bytes` are those three facts, kept here so the checks
cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from ..helpers.mx_scale import BF8_E5M2_FNUZ, FP8_E4M3_FNUZ
from ..instances.common.mx_gemm import mx_gemm_element_format
from ..numeric.fp8 import bf8e5m2_encode, fp8e4m3_encode

__all__ = ["EXACT_OPERAND_VALUES", "GEMM_TOLERANCE", "mfma_operand_bytes"]

# The runbook's GEMM tolerance (optimization_runbook.md section 1.4), as an
# error relative to the largest reference output.
GEMM_TOLERANCE = 7e-2

# Operand values every fp8 / bf8 encoding an MFMA reads (OCP and FNUZ, E4M3
# and E5M2) holds exactly.
EXACT_OPERAND_VALUES = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])

_OCP_ENCODERS = {"fp8e4m3": fp8e4m3_encode, "bf8e5m2": bf8e5m2_encode}


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
    is_fnuz = element_format in (FP8_E4M3_FNUZ, BF8_E5M2_FNUZ)
    return _OCP_ENCODERS[mantissa_dtype](2.0 * values if is_fnuz else values)
