# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Accesses whose element offset does not fit i32 reach the right address.

A global instruction adds an i32 element index to its pointer, sign-extended. An offset at or past
``2^31`` elements given as that index wraps by ``2^32`` elements: a tile ``2^32`` elements into
its tensor would read and write the tensor's first elements instead. So the one-index form of a
proven window is taken only when every offset its accesses can reach fits i32, and the placed form's
constant is the instruction's index only when it fits i32; any other offset goes on the 64-bit
pointer.

These tests pin, through the public verbs only, that a ``65536 x 65536`` tensor builds for every
kind of window (proven by guards, proven at int origins, checked) and never emits an i32 index
outside i32; that the one-index form is taken exactly up to the last offset that fits; and, on the
GPU, that a row ``2^32`` elements into its tensor is read and written where it lies.
"""

from __future__ import annotations

import ctypes
import re
import struct

import numpy as np
import pytest

from rocke.core.ir import F16, I8, IRBuilder, PtrType, Value
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.tiling import (
    AccessPath,
    exit_if_tile_outside,
    explain_bounds_checks,
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.clipping import GlobalForm

WAVE = 64
TM = TK = 16
N = 65536  # a 65536 x 65536 tensor: its last element is 2^32 - 1 elements from its first

# (M, K) tile, K-contiguous: 4 consecutive K elements per lane (one vector), 16 lanes on M, 4 on K.
TILE = make_tile_desc(
    shape=[TM, TK],
    thread_tile=[1, 4],
    thread_dist=[16, 4],
    thread_order=[1, 0],
    wave_size=WAVE,
)

_I32_LITERAL = re.compile(r"\bi32 (-?\d+)\b")


def _builder(name: str) -> tuple[IRBuilder, Value]:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = WAVE
    return b, b.param("A", PtrType(F16, "global"), align=16)


def _square():
    return make_tensor_desc((N, N), (N, 1), F16)


def _guarded(b: IRBuilder):
    """Block origins guarded in the prologue: every side proven, any tile of the tensor."""
    m0 = exit_if_tile_outside(b, b.mul(b.block_id_y(), b.const_i32(TM)), N)
    k0 = exit_if_tile_outside(b, b.mul(b.block_id_x(), b.const_i32(TK)), N)
    return make_window(_square(), (m0, k0))


def _unguarded(b: IRBuilder):
    """Block origins with no guard: their upper sides stay checked."""
    m0 = b.mul(b.block_id_y(), b.const_i32(TM))
    k0 = b.mul(b.block_id_x(), b.const_i32(TK))
    return make_window(_square(), (m0, k0))


def _windows():
    """(name, window builder, whether a side is checked) for every kind of window."""
    return [
        ("guarded block origins", _guarded, False),
        (
            "int origin 2^31 elements in",
            lambda b: make_window(_square(), (N // 2, 0)),
            False,
        ),
        (
            "int origin of the last tile",
            lambda b: make_window(_square(), (N - TM, N - TK)),
            False,
        ),
        ("int origin past the tensor", lambda b: make_window(_square(), (N, 0)), True),
        (
            "int origin before the tensor",
            lambda b: make_window(_square(), (-N // 2, 0)),
            True,
        ),
        ("unguarded block origins", _unguarded, True),
    ]


def _i32_literals(llvm: str) -> list[int]:
    return [int(value) for value in _I32_LITERAL.findall(llvm)]


@pytest.mark.parametrize("verb", ["load", "store"])
@pytest.mark.parametrize(
    "name,window_of,checked", _windows(), ids=[case[0] for case in _windows()]
)
def test_every_window_of_a_large_tensor_builds_with_no_i32_index_outside_i32(
    verb, name, window_of, checked
):
    b, ptr = _builder("large_tensor")
    window = window_of(b)
    plan = explain_bounds_checks(b, ptr, window, TILE)
    assert plan.path is AccessPath.GLOBAL
    assert (plan.form is GlobalForm.MASKED) == checked
    lane = b.thread_id_x()
    if verb == "load":
        load_fragment(b, ptr, window, TILE, lane)
    else:
        source = make_window(make_tensor_desc((TM, TK), (TK, 1), F16), (0, 0))
        store_fragment(b, ptr, window, load_fragment(b, ptr, source, TILE, lane), lane)
    b.ret()
    llvm = lower_kernel_to_llvm(b.kernel, arch="gfx90a")
    assert all(-(1 << 31) <= value < (1 << 31) for value in _i32_literals(llvm))


def test_the_one_index_form_is_taken_exactly_while_every_offset_fits_i32():
    # The tile at row 32752 ends at row 32767, column 65535: offset 2^31 - 1, the last i32.
    b, ptr = _builder("last_fit")
    plan = explain_bounds_checks(
        b, ptr, make_window(_square(), (N // 2 - TM, N - TK)), TILE
    )
    assert plan.form is GlobalForm.INDEX32
    # One row further reaches offset 2^31 + 65535: the placed form, its tile start (offset
    # 2^31 - 65536 - 16) still the instruction's i32 index, the lanes' part a 32-bit byte offset.
    b, ptr = _builder("first_past")
    plan = explain_bounds_checks(
        b, ptr, make_window(_square(), (N // 2 - TM + 1, N - TK)), TILE
    )
    assert plan.form is GlobalForm.PLACED


def test_a_tile_start_past_i32_goes_on_the_pointer_as_one_byte_offset():
    # The tile at row 32768 starts at offset 2^31, which is no i32 index: it is added to the
    # pointer in 64 bits (2^32 bytes of f16), and the instruction's index is 0.
    b, ptr = _builder("start_past")
    window = make_window(_square(), (N // 2, 0))
    assert explain_bounds_checks(b, ptr, window, TILE).form is GlobalForm.PLACED
    load_fragment(b, ptr, window, TILE, b.thread_id_x())
    b.ret()
    llvm = lower_kernel_to_llvm(b.kernel, arch="gfx90a")
    assert re.search(
        r"getelementptr inbounds i8, ptr addrspace\(1\) %A, i64 4294967296\b", llvm
    )
    assert re.search(r"getelementptr inbounds half, .* i32 0\n", llvm)


# --- on the GPU: a row 2^32 elements into its tensor -------------------------------------------------

FAR = 1 << 32  # the far row's offset in elements (i8: bytes); as an i32 index it is 0
ROW = 64

ROW_TILE = make_tile_desc(
    shape=[ROW], thread_tile=[1], thread_dist=[ROW], thread_order=[0], wave_size=WAVE
)


def _gpu_arch() -> str | None:
    try:
        from rocke.runtime.hip_module import current_device, get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch(current_device())
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _far_row_kernel():
    """``Y = X[1, 0:64]`` and ``X[1, 64:128] = Z`` on an i8 tensor ``X`` of shape ``(2, 128)``
    with row stride ``2^32``: every access of row 1 is ``2^32`` elements or more from ``X``.
    """
    b = IRBuilder("far_row")
    b.kernel.attrs["max_workgroup_size"] = WAVE
    x = b.param("X", PtrType(I8, "global"), align=16)
    y = b.param("Y", PtrType(I8, "global"), align=16)
    z = b.param("Z", PtrType(I8, "global"), align=16)
    far = make_tensor_desc((2, 2 * ROW), (FAR, 1), I8)
    near = make_tensor_desc((ROW,), (1,), I8)
    lane = b.thread_id_x()
    row = load_fragment(b, x, make_window(far, (1, 0)), ROW_TILE, lane)
    store_fragment(b, y, make_window(near, (0,)), row, lane)
    source = load_fragment(b, z, make_window(near, (0,)), ROW_TILE, lane)
    store_fragment(b, x, make_window(far, (1, ROW)), source, lane)
    b.ret()
    return b.kernel


def _to_device(runtime, address: int, host: np.ndarray) -> None:
    runtime.memcpy_h2d(
        address, (ctypes.c_ubyte * host.nbytes).from_buffer(host), host.nbytes
    )


def _from_device(runtime, address: int, nbytes: int) -> np.ndarray:
    host = np.empty(nbytes, dtype=np.int8)
    runtime.memcpy_d2h((ctypes.c_ubyte * nbytes).from_buffer(host), address, nbytes)
    return host


def test_a_row_two_to_the_32_elements_in_is_read_and_written_where_it_lies():
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    kernel = _far_row_kernel()
    runtime = Runtime()
    function = runtime.load_module(
        compile_kernel(kernel, arch=arch).hsaco
    ).get_function(kernel.name)
    first = np.arange(2 * ROW, dtype=np.int8)  # row 0: where a wrapped index would land
    far_row = (np.arange(ROW, dtype=np.int8) + 50).astype(np.int8)
    sentinel = np.full(ROW, -7, dtype=np.int8)
    stored = (np.arange(ROW, dtype=np.int8) - 100).astype(np.int8)
    x = runtime.alloc(FAR + 2 * ROW)
    y = runtime.alloc(ROW)
    z = runtime.alloc(ROW)
    try:
        _to_device(runtime, x, first)
        _to_device(runtime, x + FAR, np.concatenate([far_row, sentinel]))
        _to_device(runtime, y, sentinel.copy())
        _to_device(runtime, z, stored)
        runtime.launch_blocking(
            function, (1, 1, 1), (WAVE, 1, 1), struct.pack("QQQ", x, y, z)
        )
        assert np.array_equal(_from_device(runtime, y, ROW), far_row)
        assert np.array_equal(
            _from_device(runtime, x + FAR, 2 * ROW),
            np.concatenate([far_row, stored]),
        )
        assert np.array_equal(_from_device(runtime, x, 2 * ROW), first)
    finally:
        for address in (x, y, z):
            runtime.free(address)
