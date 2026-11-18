# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DeviceArray: validated host transfers, checked in C++ so calling _core directly is safe too.

The validation tests need no GPU, because every check runs before hipMalloc or hipMemcpy. The
round trips need one.
"""

import ml_dtypes
import numpy as np
import pytest

import hipblaslt

c = hipblaslt._core
rng = np.random.default_rng(20261118)


def _sample(dtype: np.dtype, shape=(3, 5)) -> np.ndarray:
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)
    # Random bytes cover every bit pattern, NaNs included; comparisons below are bytewise.
    raw = rng.integers(
        0, 256, size=int(np.prod(shape)) * dtype.itemsize, dtype=np.uint8
    )
    return raw.view(dtype).reshape(shape)


TYPES = sorted(hipblaslt.DTYPE_TO_NUMPY.items(), key=lambda kv: int(kv[0]))


def test_mapping_covers_the_types_hipblaslt_uses():
    names = {np.dtype(v).name for v in hipblaslt.DTYPE_TO_NUMPY.values()}
    assert {
        "float32",
        "float64",
        "float16",
        "bfloat16",
        "int8",
        "int32",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    } <= names
    assert hipblaslt.DTYPE_TO_NUMPY[c.DataType.R_16BF] == np.dtype(ml_dtypes.bfloat16)


def test_mapping_itemsizes_match_the_cpp_side():
    for value, (name, itemsize) in c._host_types().items():
        assert hipblaslt.DTYPE_TO_NUMPY[c.DataType(value)].itemsize == itemsize, name


# --- validation, no GPU needed -------------------------------------------------------------------


def test_dtype_mismatch_is_refused_through_core():
    # The prototype checked only in its Python wrapper, so this call went straight through.
    with pytest.raises(
        ValueError, match="float16 array can't go into a DataType.R_32F"
    ):
        c.DeviceArray.from_numpy(np.zeros(4, np.float16), c.DataType.R_32F)


def test_dtype_mismatch_is_refused_through_the_wrapper():
    with pytest.raises(
        ValueError, match="float16 array can't go into a DataType.R_32F"
    ):
        hipblaslt.from_numpy(np.zeros(4, np.float16), c.DataType.R_32F)


def test_a_raw_wire_view_without_its_name_is_refused():
    # A uint16 array isn't bfloat16 data unless the caller says so.
    with pytest.raises(
        ValueError, match="uint16 array can't go into a DataType.R_16BF"
    ):
        c.DeviceArray.from_numpy(np.zeros(4, np.uint16), c.DataType.R_16BF)


def test_host_dtype_needs_the_matching_wire_view():
    with pytest.raises(ValueError, match="must be its uint16 view"):
        c.DeviceArray.from_numpy(np.zeros(4, np.uint8), c.DataType.R_16BF, "bfloat16")


def test_non_contiguous_arrays_are_refused_not_copied():
    strided = np.zeros((4, 4), np.float32)[:, ::2]
    with pytest.raises(TypeError):
        c.DeviceArray.from_numpy(strided, c.DataType.R_32F)


def test_unsupported_types_say_so():
    # fp4 and fp6 are packed differently from ml_dtypes' one value per byte; they arrive later.
    with pytest.raises(ValueError, match="doesn't support DataType.R_4F_E2M1 yet"):
        c.DeviceArray.from_numpy(np.zeros(4, np.uint8), c.DataType.R_4F_E2M1)


def test_unknown_numpy_type_has_no_default():
    with pytest.raises(ValueError, match="no hipBLASLt data type holds complex64"):
        hipblaslt.from_numpy(np.zeros(4, np.complex64))


# --- round trips, GPU ----------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,np_dtype", TYPES, ids=lambda x: getattr(x, "name", str(x))
)
def test_round_trip_is_bytewise_exact(dtype, np_dtype):
    host = _sample(np.dtype(np_dtype))
    dev = hipblaslt.from_numpy(host, dtype)
    assert dev.dtype == dtype and dev.shape == host.shape and dev.nbytes == host.nbytes
    back = dev.to_numpy()
    assert back.dtype == host.dtype and back.shape == host.shape
    assert back.tobytes() == host.tobytes()


@pytest.mark.gpu
def test_uint8_stands_in_for_int8_buffers():
    raw = np.arange(256, dtype=np.uint8)
    dev = c.DeviceArray.from_numpy(raw, c.DataType.R_8I)
    out = np.empty(256, np.uint8)
    dev.copy_to_host(out)
    assert out.tobytes() == raw.tobytes()


@pytest.mark.gpu
def test_size_mismatch_is_refused_both_ways():
    dev = hipblaslt.from_numpy(np.zeros(6, np.float32))
    with pytest.raises(ValueError, match="has 20 bytes, and this DeviceArray has 24"):
        dev.copy_from_host(np.zeros(5, np.float32))
    with pytest.raises(ValueError, match="has 28 bytes, and this DeviceArray has 24"):
        dev.copy_to_host(np.zeros(7, np.float32))


@pytest.mark.gpu
def test_copy_to_host_checks_the_dtype_too():
    dev = hipblaslt.from_numpy(np.zeros(6, np.float32))
    with pytest.raises(ValueError, match="int32 array can't go into a DataType.R_32F"):
        dev.copy_to_host(np.zeros(6, np.int32))


@pytest.mark.gpu
def test_copy_from_host_reuses_the_buffer():
    dev = hipblaslt.from_numpy(np.zeros((2, 3), np.float32))
    new = np.arange(6, dtype=np.float32).reshape(2, 3)
    dev.copy_from_host(new)
    np.testing.assert_array_equal(dev.to_numpy(), new)


@pytest.mark.gpu
def test_freed_array_raises_instead_of_crashing():
    dev = hipblaslt.from_numpy(np.zeros(4, np.float32))
    dev.free()
    dev.free()
    assert dev.freed
    with pytest.raises(hipblaslt.HipblasLtError, match="freed"):
        dev.ptr
    with pytest.raises(hipblaslt.HipblasLtError, match="freed"):
        dev.to_numpy()
