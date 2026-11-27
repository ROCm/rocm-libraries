# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The element-wise bounds the attention numeric tests judge a kernel by.

* :func:`attention_tolerance`: an attention output stored as fp16 or bf16,
  against the float64 oracle on the stored inputs.
* :data:`LSE_TOLERANCE`: a natural-log LSE side output.

Both are :class:`rocke.numeric.compare.Tolerance` bounds,
``|out - ref| <= atol + rtol * |ref|`` per element. A bound on the largest
error alone would be sized by the largest outputs and accept a zeroed or
mis-scaled output wherever the output is small (a flat softmax, a short
window); an element-wise one holds every output to its own size.
"""

from __future__ import annotations

from rocke.numeric.compare import Tolerance

__all__ = ["LSE_TOLERANCE", "attention_tolerance"]

#: Per stored element type: about two output ulps at unit magnitude (fp16 keeps
#: 10 mantissa bits, bf16 7), for the rounding of the output and of the softmax
#: probabilities to that type. A correct bf16 kernel at a large softmax scale
#: comes within a factor of two of the bf16 bound, so it cannot be tighter.
_ATTENTION_TOLERANCE = {
    "fp16": Tolerance(atol=2e-3, rtol=2e-3),
    "bf16": Tolerance(atol=1e-2, rtol=1e-2),
}

#: The kernels keep the row maximum and sum in fp32 whatever the output type,
#: so one bound serves fp16 and bf16.
LSE_TOLERANCE = Tolerance(atol=1e-3, rtol=1e-3)


def attention_tolerance(dtype: str, *, value_scale: float = 1.0) -> Tolerance:
    """The element-wise bound of an attention output stored as ``dtype`` (fp16
    or bf16), against the float64 oracle.

    The absolute part is for values (V) of unit size, as the tests' standard
    normal draws are: an output is a weighted mean of V rows, so it and its
    rounding scale with V, and a launch whose V is ``value_scale`` times larger
    gets that much more absolute room."""
    if dtype not in _ATTENTION_TOLERANCE:
        raise ValueError(
            f"attention_tolerance failed - dtype={dtype!r}, expected one of "
            f"{tuple(_ATTENTION_TOLERANCE)}"
        )
    if not value_scale > 0:
        raise ValueError(
            f"attention_tolerance failed - value_scale={value_scale!r}, expected > 0"
        )
    unit = _ATTENTION_TOLERANCE[dtype]
    return Tolerance(atol=unit.atol * value_scale, rtol=unit.rtol)
