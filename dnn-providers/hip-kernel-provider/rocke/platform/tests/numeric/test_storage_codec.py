# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the storage-dtype codec in ``rocke.numeric.storage``.

The bf16 bit codec itself is held to ``ml_dtypes`` in ``test_bf16_codec.py``;
these tests check the dtype-name dispatch on top of it and the XF32 truncation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.numeric.bf16 import f32_to_bf16
from rocke.numeric.fp8 import fp8e4m3_decode_table, fp8e4m3_round
from rocke.numeric.storage import (
    STORAGE_DTYPES,
    decode,
    element_type_of,
    encode,
    host_dtype,
    round_to,
    xf32_truncate,
)

VALUES = np.array([0.0, -0.0, 1.0, -2.5, 1.0 + 2.0**-9, 3.14159265, 65504.0, 1e-8])


def test_host_dtype_is_numpy_native_or_raw_bits_of_the_same_width():
    assert STORAGE_DTYPES == ("fp32", "fp16", "bf16", "fp8e4m3")
    assert host_dtype("fp16") == np.float16 and host_dtype("half") == np.float16
    assert host_dtype("i32") == np.int32 and host_dtype("i64") == np.int64
    assert host_dtype("bf16") == np.uint16
    assert host_dtype("fp8e4m3") == np.uint8 and host_dtype("bf8e5m2") == np.uint8
    with pytest.raises(ValueError, match=r"element_bytes failed - dtype='fp4e2m1'"):
        host_dtype("fp4e2m1")


def test_element_type_of_names_numpy_native_hosts_only():
    assert element_type_of(np.float16) == "fp16"
    assert element_type_of(np.int32) == "i32"
    assert element_type_of(np.uint8) == "u8"
    with pytest.raises(ValueError, match=r"element type failed - host dtype=uint16"):
        element_type_of(np.uint16)


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
def test_encode_returns_the_host_dtype_and_round_trips(dtype):
    stored = encode(VALUES, dtype)
    assert stored.dtype == host_dtype(dtype)
    assert stored.flags.c_contiguous
    values = decode(stored, dtype)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(encode(values, dtype), stored)


def test_bf16_dispatches_to_the_bit_codec():
    np.testing.assert_array_equal(encode(VALUES, "bf16"), f32_to_bf16(VALUES))


def test_fp8e4m3_dispatches_to_the_fp8_codec():
    stored = encode(VALUES, "fp8e4m3")
    np.testing.assert_array_equal(stored, fp8e4m3_round(VALUES))
    np.testing.assert_array_equal(
        decode(stored, "fp8e4m3"), fp8e4m3_decode_table()[stored]
    )


def test_round_to_is_what_the_kernel_reads():
    np.testing.assert_array_equal(
        round_to(VALUES, "fp16"), VALUES.astype(np.float16).astype(np.float32)
    )
    # bf16 keeps 7 mantissa bits: 1 + 2^-9 rounds down to 1, pi to 3.140625.
    assert round_to(VALUES, "bf16")[4] == 1.0
    assert round_to(VALUES, "bf16")[5] == 3.140625
    np.testing.assert_array_equal(round_to(VALUES, "fp32"), VALUES.astype(np.float32))
    # e4m3 keeps 3 mantissa bits: pi rounds to 3.25 and 65504 saturates to 448.
    assert round_to(VALUES, "fp8e4m3")[5] == 3.25
    assert round_to(VALUES, "fp8e4m3")[6] == 448.0


def test_decode_refuses_a_buffer_of_another_host_dtype():
    with pytest.raises(
        ValueError, match=r"storage decode failed - array dtype=float16"
    ):
        decode(np.zeros(2, np.float16), "bf16")


def test_unknown_dtype_is_refused():
    with pytest.raises(ValueError, match=r"storage codec failed - dtype='fp8'"):
        encode(VALUES, "fp8")


def test_xf32_truncate_clears_the_low_13_mantissa_bits():
    bits = np.array([0x3F001FFF, 0xBF7FFFFF, 0x7F800000], dtype=np.uint32)
    got = xf32_truncate(bits.view(np.float32)).view(np.uint32)
    np.testing.assert_array_equal(got, [0x3F000000, 0xBF7FE000, 0x7F800000])
