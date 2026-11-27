# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.numeric.tolerance``: the element-wise bound of sums of products.

The bound's formula, its refusals, and on a real convolution that it accepts
the reference rounded the way a kernel rounds it and rejects a zeroed output,
a scale error and a dropped filter tap.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import within_tolerance
from rocke.numeric.conv import conv_fwd
from rocke.numeric.tolerance import EPSILON, sum_of_products_tolerance


def test_rtol_is_the_output_epsilon_and_atol_the_partial_roundings():
    magnitude = np.array([0.0, 2.0, 100.0])
    tolerance = sum_of_products_tolerance(
        "bf16", magnitude, partial_dtype="fp16", partial_roundings=4
    )
    assert tolerance.rtol == EPSILON["bf16"] == 2.0**-7
    np.testing.assert_array_equal(
        tolerance.atol, 2.0**-133 + 4 * 2.0**-10 / 2 * magnitude
    )
    default = sum_of_products_tolerance("fp16", magnitude)
    np.testing.assert_array_equal(default.atol, 2.0**-24 + 2.0**-23 * magnitude)


def test_a_subnormal_output_is_held_to_one_subnormal_step():
    """Below fp16's smallest normal value (2**-14) the store rounds to a fixed
    step of 2**-24, so a correct output there can miss by more than its
    relative bound; one step of room accepts it, two steps do not."""
    reference = np.array([1.181e-6])
    tolerance = sum_of_products_tolerance("fp16", np.abs(reference))
    stored = storage.round_to(reference, "fp16").astype(np.float64)
    assert abs(stored - reference)[0] > tolerance.rtol * reference[0]
    assert within_tolerance(stored, reference, tolerance)
    assert not within_tolerance(stored + 2 * 2.0**-24, reference, tolerance)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"output_dtype": "fp8e4m3"}, r"output_dtype='fp8e4m3', expected one of"),
        ({"output_dtype": "tf32"}, r"output_dtype='tf32', expected one of"),
        ({"partial_dtype": "f16"}, r"partial_dtype='f16', expected one of"),
        ({"partial_roundings": 0}, r"partial_roundings=0, expected an int >= 1"),
        ({"partial_roundings": 2.0}, r"partial_roundings=2.0, expected an int"),
        ({"partial_roundings": True}, r"partial_roundings=True, expected an int"),
    ],
)
def test_refuses_an_unknown_type_or_a_bad_rounding_count(kwargs, message):
    arguments = {"output_dtype": "fp16", "magnitude": np.ones(3), **kwargs}
    with pytest.raises(
        ValueError, match=r"sum_of_products_tolerance failed - " + message
    ):
        sum_of_products_tolerance(**arguments)


def _conv_case(dtype):
    """A 3x3, pad 1 forward convolution of 64 channels on U(-1, 1) operands
    stored as ``dtype``: ``(x, w, reference, magnitude)`` in float64."""
    rng = np.random.default_rng(5)
    x = storage.round_to(rng.uniform(-1, 1, (2, 14, 14, 64)), dtype)
    w = storage.round_to(rng.uniform(-1, 1, (64, 3, 3, 64)), dtype)
    x, w = x.astype(np.float64), w.astype(np.float64)
    reference = conv_fwd(x, w, padding=1)
    return x, w, reference, conv_fwd(np.abs(x), np.abs(w), padding=1)


# (output type, partial-sum type, scale error that must fail). With fp32
# partial sums the bound is the output rounding, so four epsilons of scale
# error fail (0.4% in fp16, 3.1% in bf16; the peak-relative bounds this
# replaced accepted 4% and 9%). With fp16 partial sums the bound also holds the
# partial-sum rounding of each output's magnitude, and 2% fails.
_CONV_CASES = [
    ("fp16", "fp32", 1 + 4 * 2.0**-10),
    ("bf16", "fp32", 1 + 4 * 2.0**-7),
    ("fp16", "fp16", 1.02),
]


@pytest.mark.parametrize("dtype,partial_dtype,scale", _CONV_CASES)
def test_accepts_the_rounded_reference_and_rejects_known_bugs(
    dtype, partial_dtype, scale
):
    """The reference rounded to the output type passes; the same output with
    every element ``scale`` times too large, with its median-magnitude output
    zeroed, or with the centre filter tap dropped from every output, fails."""
    x, w, reference, magnitude = _conv_case(dtype)
    tolerance = sum_of_products_tolerance(dtype, magnitude, partial_dtype=partial_dtype)
    kernel_like = storage.round_to(reference, dtype).astype(np.float64)
    assert within_tolerance(kernel_like, reference, tolerance)

    assert not within_tolerance(kernel_like * scale, reference, tolerance), "scale"

    zeroed = kernel_like.copy()
    median = np.argsort(np.abs(reference), axis=None)[reference.size // 2]
    zeroed.flat[median] = 0.0
    assert not within_tolerance(zeroed, reference, tolerance), "zeroed output"

    no_centre = w.copy()
    no_centre[:, 1, 1, :] = 0.0
    dropped = storage.round_to(conv_fwd(x, no_centre, padding=1), dtype)
    assert not within_tolerance(dropped, reference, tolerance), "dropped tap"
