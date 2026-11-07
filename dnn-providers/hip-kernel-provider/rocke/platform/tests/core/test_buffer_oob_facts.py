# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Buffer range-check facts, asserted against the GPU that is present.

Safe access through buffer resources rests on how the hardware range-checks a
raw buffer access (stride 0, offset mode). Each test here runs real accesses
on the local GPU and compares every byte with one model of that check:

- The checked offset is ``voffset + soffset``, both unsigned 32-bit, summed
  without wrapping. A "negative" voffset is a large unsigned one.
- An access of 4 bytes or more is checked per 4-byte component, counted from
  the access start relative to the descriptor base: component ``k`` is kept iff
  ``offset + 4 * (k + 1) <= num_records``. A 1- or 2-byte access is kept only
  whole. The base address's alignment does not matter.
- A dropped load component reads 0; a dropped store component writes nothing,
  and the bytes around a store are never touched.

The model is ``_kept_components``; the tests choose offsets that straddle the
end, sit below zero, pass 2^31, or reach the end only through ``soffset``.
Each test skips without a GPU.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from rocke.core.ir import I8, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_LANE_STRIDE = 16  # bytes per lane in the load results and the store sources
_SOURCE_BYTES = 4096


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


@pytest.fixture(scope="module")
def arch():
    found = _gpu_arch()
    if not found:
        pytest.skip("no GPU")
    return found


def _i32(value: int) -> int:
    """``value`` (0 .. 2^32 - 1, or already signed) as a signed i32 constant."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _kept_components(offset: int, soffset: int, nbytes: int, num_records: int):
    """The model: ``(start byte, length)`` of each component the hardware keeps
    for an ``nbytes`` access at ``voffset = offset`` (read as unsigned 32-bit)
    and ``soffset``."""
    start = (offset & 0xFFFFFFFF) + (soffset & 0xFFFFFFFF)  # no 32-bit wrap
    if nbytes < 4:
        return [(start, nbytes)] if start + nbytes <= num_records else []
    return [
        (start + 4 * k, 4)
        for k in range(nbytes // 4)
        if start + 4 * (k + 1) <= num_records
    ]


def _source_bytes() -> np.ndarray:
    """Distinct non-zero bytes, so a 0 can only be a dropped component."""
    return (np.arange(_SOURCE_BYTES, dtype=np.int64) * 37 % 251 + 1).astype(np.uint8)


def _descriptor(b, ptr, base_shift, num_records):
    if base_shift:
        ptr = b.global_ptr_add(ptr, b.const_i64(base_shift))
    return b.buffer_rsrc(ptr, b.const_i32(_i32(num_records)))


def _load_kernel(nbytes, num_records, soffset, base_shift):
    """Lane ``l`` loads ``nbytes`` at ``voffset = OFF[l]`` through a descriptor
    on ``X + base_shift`` and writes them to ``Y[l * 16 ...]``."""
    b = IRBuilder(f"buffer_fact_load_{nbytes}")
    x = b.param("X", PtrType(I8, "global"))
    y = b.param("Y", PtrType(I8, "global"))
    offsets = b.param("OFF", PtrType(I32, "global"))
    lane = b.thread_id_x()
    rsrc = _descriptor(b, x, base_shift, num_records)
    voffset = b.global_load(offsets, lane, I32)
    value = b.buffer_load_vN(rsrc, voffset, b.const_i32(_i32(soffset)), I8, nbytes)
    row = b.mul(lane, b.const_i32(_LANE_STRIDE))
    for j in range(nbytes):
        b.global_store(y, b.add(row, b.const_i32(j)), b.vec_extract(value, j))
    b.ret()
    return b.kernel


def _store_kernel(nbytes, num_records, soffset, base_shift):
    """Lane ``l`` stores ``X[l * 16 ...]`` (``nbytes`` of them) at ``voffset =
    OFF[l]`` through a descriptor on ``Y + base_shift``."""
    b = IRBuilder(f"buffer_fact_store_{nbytes}")
    x = b.param("X", PtrType(I8, "global"))
    y = b.param("Y", PtrType(I8, "global"))
    offsets = b.param("OFF", PtrType(I32, "global"))
    lane = b.thread_id_x()
    source = b.buffer_rsrc(x, b.const_i32(_SOURCE_BYTES))
    row = b.mul(lane, b.const_i32(_LANE_STRIDE))
    value = b.buffer_load_vN(source, row, b.const_i32(0), I8, nbytes)
    rsrc = _descriptor(b, y, base_shift, num_records)
    voffset = b.global_load(offsets, lane, I32)
    b.buffer_store_vN(rsrc, voffset, b.const_i32(_i32(soffset)), value)
    b.ret()
    return b.kernel


def _run(arch, kernel, offsets, out_init):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    blob, _ = build_hsaco_from_llvm_ir(
        _lower_kernel_to_llvm_python(kernel, arch=arch),
        isa=f"amdgcn-amd-amdhsa--{arch}",
    )
    runtime = Runtime()
    module = runtime.load_module(blob)
    arrays = [
        _source_bytes(),
        np.ascontiguousarray(out_init).copy(),
        np.array([_i32(o) for o in offsets], dtype=np.int32),
    ]
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            runtime.memcpy_h2d(
                ptr, (ctypes.c_ubyte * array.nbytes).from_buffer(array), array.nbytes
            )
        runtime.launch_blocking(
            module.get_function(kernel.name),
            (1, 1, 1),
            (len(offsets), 1, 1),
            struct.pack("3Q", *ptrs),
        )
        out = arrays[1]
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * out.nbytes).from_buffer(out), ptrs[1], out.nbytes
        )
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()
    return out


def _check_loads(arch, nbytes, offsets, *, num_records, soffset=0, base_shift=0):
    kernel = _load_kernel(nbytes, num_records, soffset, base_shift)
    got = _run(
        arch, kernel, offsets, np.full(len(offsets) * _LANE_STRIDE, 0xEE, np.uint8)
    )
    source = _source_bytes()
    want = np.full_like(got, 0xEE)
    for lane, offset in enumerate(offsets):
        row = lane * _LANE_STRIDE
        want[row : row + nbytes] = 0  # dropped components read 0
        for start, length in _kept_components(offset, soffset, nbytes, num_records):
            access = start - ((offset & 0xFFFFFFFF) + soffset)
            src = base_shift + start
            want[row + access : row + access + length] = source[src : src + length]
    np.testing.assert_array_equal(got, want, err_msg=f"loads at {offsets}")


def _check_stores(arch, nbytes, offsets, *, num_records, soffset=0, base_shift=0):
    kernel = _store_kernel(nbytes, num_records, soffset, base_shift)
    sentinel = np.full(_SOURCE_BYTES, 0xEE, np.uint8)
    got = _run(arch, kernel, offsets, sentinel)
    source = _source_bytes()
    want = sentinel.copy()  # every byte not kept stays untouched
    for lane, offset in enumerate(offsets):
        for start, length in _kept_components(offset, soffset, nbytes, num_records):
            access = start - ((offset & 0xFFFFFFFF) + soffset)
            src = lane * _LANE_STRIDE + access
            dst = base_shift + start
            want[dst : dst + length] = source[src : src + length]
    np.testing.assert_array_equal(got, want, err_msg=f"stores at {offsets}")


# A dwordx4 access straddling the end keeps exactly its in-range components.
@pytest.mark.parametrize("num_records", [20, 18])
def test_straddling_dwordx4_keeps_whole_in_range_components(arch, num_records):
    offsets = [0, 4, 8, 12, 16, 20, 24]
    _check_loads(arch, 16, offsets, num_records=num_records)
    _check_stores(arch, 16, [0, 32, 64], num_records=num_records + 64)
    _check_stores(arch, 16, [8], num_records=num_records)


# Components count from the descriptor base, whatever that address's alignment.
@pytest.mark.parametrize("base_shift", [1, 2])
def test_components_count_from_an_unaligned_descriptor_base(arch, base_shift):
    _check_loads(arch, 16, [0, 4, 8, 12, 16], num_records=18, base_shift=base_shift)
    _check_stores(arch, 16, [8], num_records=18, base_shift=base_shift)


# 1- and 2-byte accesses are kept only whole.
@pytest.mark.parametrize("nbytes,offsets", [(1, [5, 6, 7, 8]), (2, [4, 5, 6, 7])])
def test_narrow_accesses_are_kept_only_whole(arch, nbytes, offsets):
    _check_loads(arch, nbytes, offsets, num_records=7)
    _check_stores(arch, nbytes, offsets[::2], num_records=7)
    _check_stores(arch, nbytes, offsets[1::2], num_records=7)


# A negative voffset is a large unsigned one: out of range.
def test_negative_voffset_reads_zero_and_drops_stores(arch):
    _check_loads(arch, 4, [-16, -4, -1, 0], num_records=64)
    _check_stores(arch, 4, [-4, 8, -1], num_records=64)


# An offset of 2^31 or more is out of range when num_records is at most 2^31.
@pytest.mark.parametrize("num_records", [64, 1 << 31])
def test_offsets_from_two_to_the_31_are_out_of_range(arch, num_records):
    offsets = [0, 1 << 31, (1 << 31) + 16, 0xFFFFFFF0]
    _check_loads(arch, 16, offsets, num_records=num_records)
    _check_stores(arch, 16, offsets, num_records=num_records)


# soffset counts in the range check, alone and in the sum with voffset.
@pytest.mark.parametrize(
    "nbytes,soffset,offsets",
    [(4, 64, [0, 4]), (4, 48, [0, 16, 12]), (16, 48, [0, 4])],
)
def test_soffset_counts_in_the_range_check(arch, nbytes, soffset, offsets):
    _check_loads(arch, nbytes, offsets, num_records=64, soffset=soffset)
    _check_stores(arch, nbytes, offsets[:1], num_records=64, soffset=soffset)
    _check_stores(arch, nbytes, offsets[1:], num_records=64, soffset=soffset)


# voffset + soffset near 2^32 does not wrap back into range.
@pytest.mark.parametrize(
    "soffset,offsets",
    [(32, [0xFFFFFFF0, 0xFFFFFFE0]), (0x80000010, [0x7FFFFFF0, 0x7FFFFFFF])],
)
def test_a_sum_near_two_to_the_32_does_not_wrap(arch, soffset, offsets):
    _check_loads(arch, 4, offsets, num_records=256, soffset=soffset)
    _check_stores(arch, 4, offsets, num_records=256, soffset=soffset)
