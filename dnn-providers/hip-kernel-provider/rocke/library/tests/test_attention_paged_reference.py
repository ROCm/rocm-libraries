# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The host encoding of the torch-free paged attention numeric tests accepts
only fp16 and bf16, so a test written for another dtype fails loudly instead
of having its values read as bf16.

No GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._attention_paged_reference import decode_half, encode_half, round_to_half


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_half_dtypes_round_trip_exact_values(dtype):
    values = np.array([0.0, 1.0, -2.5, 0.125], dtype=np.float32)
    np.testing.assert_array_equal(round_to_half(values, dtype), values)


@pytest.mark.parametrize("helper", [encode_half, decode_half, round_to_half])
@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp32", "float16", ""])
def test_non_half_dtype_is_refused(helper, dtype):
    values = np.zeros(4, dtype=np.float32)
    with pytest.raises(
        ValueError, match=rf"dtype={dtype!r}, expected one of \('fp16', 'bf16'\)"
    ):
        helper(values, dtype)
