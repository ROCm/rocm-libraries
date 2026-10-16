# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Magic division stays exact for dividends near 2**31.

``do_magic_division`` and ``do_magic_division_dynamic`` compute
``(umul_hi(n, multiplier) + n) >> shift``. For a dividend below 2**31 the sum
can pass the signed i32 maximum (from 1,610,612,736 for divisor 3) while
staying below 2**32, so it must be a no-unsigned-wrap add: a no-signed-wrap add
would make the quotient poison inside the range the helpers document.

These tests check the emitted LLVM text (the sum is ``add nuw``, never
``add nsw``) and, when a GPU is present, the quotients the hardware computes
for dividends on both sides of that point against numpy.
"""

from __future__ import annotations

import ctypes
import re
import struct

import numpy as np
import pytest

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.magic_division import (
    calculate_magic_numbers,
    do_magic_division,
    do_magic_division_dynamic,
)

FLAVORS = ("llvm20", "llvm22", "llvm23")
DIVIDENDS = (2**30 - 1, 2**30, 1_610_612_736, 2**31 - 1)
DIVISORS = (3, 9, 1025, 64)


def _static_kernel(divisor):
    """Lane i writes ``n[i] // divisor`` with the divisor's constants baked in."""
    b = IRBuilder(f"magic_div_static_{divisor}")
    dividends = b.param("n", PtrType(I32, "global"))
    out = b.param("q", PtrType(I32, "global"))
    lane = b.thread_id_x()
    multiplier, shift = calculate_magic_numbers(divisor)
    quotient = do_magic_division(
        b, b.global_load(dividends, lane, I32), multiplier, shift
    )
    b.store(quotient, out, lane)
    b.ret()
    return b.kernel


def _dynamic_kernel():
    """Lane i writes ``n[i] // d[i]`` from the runtime magic pair of ``d[i]``."""
    b = IRBuilder("magic_div_dynamic")
    ptrs = [b.param(name, PtrType(I32, "global")) for name in ("n", "mult", "shift")]
    out = b.param("q", PtrType(I32, "global"))
    lane = b.thread_id_x()
    dividend, multiplier, shift = (b.global_load(p, lane, I32) for p in ptrs)
    b.store(do_magic_division_dynamic(b, dividend, multiplier, shift), out, lane)
    b.ret()
    return b.kernel


def _sum_lines(text):
    """The ``add`` lines of the lowered magic-division sequence."""
    return [line.strip() for line in text.splitlines() if re.search(r"= add ", line)]


# --------------------------------------------------------------------------
# IR text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel", [_static_kernel(3), _dynamic_kernel()], ids=["static", "dynamic"]
)
def test_the_sum_is_a_no_unsigned_wrap_add(kernel):
    for flavor in FLAVORS:
        text = _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)
        lines = _sum_lines(text)
        assert lines, text
        assert all(
            line.split("=", 1)[1].startswith(" add nuw ") for line in lines
        ), lines
        assert "add nsw" not in text


@pytest.mark.parametrize(
    "kernel", [_static_kernel(3), _dynamic_kernel()], ids=["static", "dynamic"]
)
def test_both_engines_emit_the_same_text(kernel):
    native = pytest.importorskip("rocke_engine")
    ir = serialize(kernel)
    for flavor in FLAVORS:
        python_text = _lower_kernel_to_llvm_python(
            kernel, arch="gfx942", llvm_flavor=flavor
        )
        assert (
            native.lower_serialized_ir(ir, arch="gfx942", flavor=flavor) == python_text
        )


# --------------------------------------------------------------------------
# GPU values
# --------------------------------------------------------------------------


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run(arch, kernel, inputs):
    """Launch ``kernel`` on one wave over ``inputs``; return its i32 output."""
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    blob, _ = build_hsaco_from_llvm_ir(
        _lower_kernel_to_llvm_python(kernel, arch=arch),
        isa=f"amdgcn-amd-amdhsa--{arch}",
    )
    runtime = Runtime()
    module = runtime.load_module(blob)
    out = np.full_like(inputs[0], -7)
    arrays = [np.ascontiguousarray(a).copy() for a in inputs] + [out]
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            buf = (ctypes.c_ubyte * array.nbytes).from_buffer(array)
            runtime.memcpy_h2d(ptr, buf, array.nbytes)
        runtime.launch_blocking(
            module.get_function(kernel.name),
            (1, 1, 1),
            (len(out), 1, 1),
            struct.pack(f"{len(ptrs)}Q", *ptrs),
        )
        buf = (ctypes.c_ubyte * out.nbytes).from_buffer(out)
        runtime.memcpy_d2h(buf, ptrs[-1], out.nbytes)
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()
    return out


@pytest.fixture(scope="module")
def gpu_arch():
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    return arch


@pytest.mark.parametrize("divisor", DIVISORS)
def test_static_quotients_match_numpy(gpu_arch, divisor):
    dividends = np.array(DIVIDENDS, dtype=np.int32)
    got = _run(gpu_arch, _static_kernel(divisor), [dividends])
    np.testing.assert_array_equal(got, dividends // divisor)


def test_dynamic_quotients_match_numpy(gpu_arch):
    cases = [(n, d) for d in DIVISORS for n in DIVIDENDS]
    dividends = np.array([n for n, _ in cases], dtype=np.int32)
    magic = [calculate_magic_numbers(d) for _, d in cases]
    multipliers = np.array([m for m, _ in magic], dtype=np.uint32).view(np.int32)
    shifts = np.array([s for _, s in magic], dtype=np.int32)
    divisors = np.array([d for _, d in cases], dtype=np.int32)
    got = _run(gpu_arch, _dynamic_kernel(), [dividends, multipliers, shifts])
    np.testing.assert_array_equal(got, dividends // divisors)
