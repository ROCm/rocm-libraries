# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How large an element-wise bound a kernel output is held to.

* :data:`EPSILON`: the machine epsilon (the gap from 1.0 to the next value) of
  each floating type a kernel rounds a result to.
* :func:`sum_of_products_tolerance`: the :class:`~rocke.numeric.compare.Tolerance`
  of outputs that each sum products of operands (a GEMM, a convolution and its
  gradients), from the output type, the type the kernel keeps its partial sums
  in, and each output's magnitude: the same sum over the products' absolute
  values, which is the reference evaluated on the operands' absolute values.

Why a magnitude per element rather than one bound for the tensor: the error a
kernel's rounding leaves in an output scales with the products it sums, not
with the output itself, which can be small through cancellation. A bound
sized by the tensor's largest output accepts a zeroed, mis-scaled or
missing-term result on every smaller output; one sized by each output's own
products does not.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .compare import Tolerance

__all__ = ["EPSILON", "sum_of_products_tolerance"]

#: Machine epsilon per element type. ``tf32`` is a matrix-core operand format
#: with fp16's 10 mantissa bits.
EPSILON = {"fp32": 2.0**-23, "tf32": 2.0**-10, "fp16": 2.0**-10, "bf16": 2.0**-7}

#: The smallest positive (subnormal) value of each type an output is stored as:
#: the fixed step a store rounds to below the smallest normal value, where the
#: rounding error is no longer relative to the value.
_SUBNORMAL_STEP = {"fp32": 2.0**-149, "fp16": 2.0**-24, "bf16": 2.0**-133}


def _lookup(table: dict, dtype: str, name: str) -> float:
    if dtype not in table:
        raise ValueError(
            f"sum_of_products_tolerance failed - {name}={dtype!r}, expected one "
            f"of {tuple(table)}"
        )
    return table[dtype]


def sum_of_products_tolerance(
    output_dtype: str,
    magnitude: npt.ArrayLike,
    *,
    partial_dtype: str = "fp32",
    partial_roundings: int = 2,
) -> Tolerance:
    """The element-wise bound of outputs that each sum products of operands,
    stored as ``output_dtype``.

    ``magnitude`` holds, per output, the sum of its products' absolute values
    (the reference on the absolute values of the operands); ``partial_dtype``
    is the type the kernel keeps partial sums in (fp32 for a kernel that
    accumulates in fp32 and rounds once on store, the output type for one that
    accumulates or stages partial results in it).

    * ``rtol = EPSILON[output_dtype]``: the store rounds the result by at most
      half an ulp of itself; twice that. Below the smallest normal value the
      ulp stops shrinking, so ``atol`` also holds one subnormal step of the
      output type.
    * ``atol += partial_roundings * EPSILON[partial_dtype] / 2 * magnitude``:
      rounding a partial sum to ``partial_dtype`` moves it by at most half an
      ulp, and a partial sum is at most the output's magnitude, so the bound
      allows ``partial_roundings`` such roundings at full magnitude. With
      operands of random sign the roundings of many partial sums partly cancel
      and the default 2 covers them; a sum of same-sign partials (a split-K
      reduction of positive data) keeps every rounding and needs one per
      partial.

    A missing product moves its output by that product's size, which this
    bound sees whenever the product exceeds the output's ``atol``.
    """
    subnormal_step = _lookup(_SUBNORMAL_STEP, output_dtype, "output_dtype")
    output_epsilon = EPSILON[output_dtype]
    partial_epsilon = _lookup(EPSILON, partial_dtype, "partial_dtype")
    if isinstance(partial_roundings, bool) or not (
        isinstance(partial_roundings, int) and partial_roundings >= 1
    ):
        raise ValueError(
            f"sum_of_products_tolerance failed - partial_roundings="
            f"{partial_roundings!r}, expected an int >= 1"
        )
    partial_rounding = partial_roundings * partial_epsilon / 2
    atol = subnormal_step + partial_rounding * np.asarray(magnitude, np.float64)
    return Tolerance(atol=atol, rtol=output_epsilon)
