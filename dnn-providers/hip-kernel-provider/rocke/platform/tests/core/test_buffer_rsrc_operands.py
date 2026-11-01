# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The operands ``buffer_rsrc`` refuses in the LLVM and HIP lowerings of both engines.

``buffer_rsrc(ptr, num_bytes)`` lowers to ``@llvm.amdgcn.make.buffer.rsrc``,
whose ``p1`` overload names ``ptr addrspace(1)``:

- a pointer parameter declared ``addr_space="constant"`` is
  ``ptr addrspace(4)`` in the kernel header, so the call would not match the
  header. It is refused at every flavor, with the text the global-pointer ops
  use.
- llvm20's intrinsic takes an i32 ``num_records``; an i64 ``num_bytes`` would be
  printed as an i32 operand that does not match its own definition. It is
  refused at llvm20, with the fix in the message. llvm22 and later take an i64
  and zero-extend an i32, so both widths still lower there.

Before these refusals both engines emitted the call anyway, and an LLVM 20
``llc`` rejects the module ("'%X' defined with type 'ptr addrspace(4)' but
expected 'ptr addrspace(1)'"; "'%n' defined with type 'i64' but expected
'i32'").

The HIP lowering calls ``__builtin_amdgcn_make_buffer_rsrc``, whose
``num_records`` is an ``int``: it cast an i64 ``num_bytes`` to ``(int)``, which
truncates the byte count at every LLVM version, so it refuses any width but i32.
It keeps a constant-space pointer: the HIP signature declares every pointer
parameter the same way, and the builtin takes a ``void*``.
"""

from __future__ import annotations

import pytest
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)

FLAVORS = ("llvm20", "llvm22", "llvm23", "llvm24")
ARCH = "gfx942"


def _rsrc_kernel(name, *, constant, num_bytes_type):
    """A buffer_rsrc over ``X`` with a ``num_bytes`` parameter of
    ``num_bytes_type``, and one load through it; ``X`` is declared
    ``addr_space="constant"`` when ``constant`` is set."""
    b = IRBuilder(name)
    space = {"addr_space": "constant"} if constant else {}
    x = b.param("X", PtrType(F32, "global"), **space)
    y = b.param("Y", PtrType(F32, "global"))
    n = b.param("n", num_bytes_type)
    rsrc = b.buffer_rsrc(x, n)
    value = b.buffer_load_vN(rsrc, b.const_i32(0), b.const_i32(0), F32, 1)
    b.global_store(y, b.const_i32(0), b.vec_extract(value, 0))
    b.ret()
    return b.kernel


@pytest.mark.parametrize("flavor", FLAVORS)
def test_both_llvm_lowerings_refuse_a_constant_space_pointer(flavor):
    kernel = _rsrc_kernel("rsrc_constant", constant=True, num_bytes_type=I32)
    assert_both_engines_reject(
        kernel,
        ARCH,
        TypeError,
        "buffer_rsrc ptr must be a global pointer, got ptr addrspace(4) "
        "(its parameter's addr_space)",
        llvm_flavor=flavor,
    )


def test_both_llvm_lowerings_refuse_an_i64_num_bytes_at_llvm20():
    kernel = _rsrc_kernel("rsrc_i64", constant=False, num_bytes_type=I64)
    assert_both_engines_reject(
        kernel,
        ARCH,
        ValueError,
        "tile.buffer_rsrc num_bytes must be i32 at llvm20, got i64; the llvm20 "
        "make.buffer.rsrc takes an i32 num_records, so pass the byte count as i32",
        llvm_flavor="llvm20",
    )


@pytest.mark.parametrize("flavor", FLAVORS)
def test_an_i32_num_bytes_lowers_at_every_flavor(flavor):
    kernel = _rsrc_kernel("rsrc_i32", constant=False, num_bytes_type=I32)
    ir = lower_with_both_engines(kernel, ARCH, llvm_flavor=flavor)
    if flavor == "llvm20":
        assert (
            "@llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) %X, i16 0, i32 %n," in ir
        )
    else:
        assert "= zext i32 %n to i64" in ir
        assert (
            "@llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %X, i16 0, i64 %"
            in ir
        )


@pytest.mark.parametrize("flavor", FLAVORS[1:])
def test_an_i64_num_bytes_lowers_after_llvm20(flavor):
    kernel = _rsrc_kernel("rsrc_i64", constant=False, num_bytes_type=I64)
    ir = lower_with_both_engines(kernel, ARCH, llvm_flavor=flavor)
    assert (
        "@llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %X, i16 0, i64 %n," in ir
    )


def test_both_hip_lowerings_refuse_an_i64_num_bytes():
    kernel = _rsrc_kernel("rsrc_i64", constant=False, num_bytes_type=I64)
    assert_both_engines_reject_hip(
        kernel,
        ARCH,
        ValueError,
        "tile.buffer_rsrc num_bytes must be i32 in the HIP lowering, got i64; "
        "the HIP builtin takes an int num_records, so the byte count would be "
        "truncated at every LLVM version; pass it as i32",
    )


@pytest.mark.parametrize("constant", (False, True))
def test_the_hip_lowering_keeps_an_i32_num_bytes_and_a_constant_space_pointer(
    constant,
):
    kernel = _rsrc_kernel("rsrc_hip", constant=constant, num_bytes_type=I32)
    hip = lower_hip_with_both_engines(kernel, ARCH)
    assert "void rsrc_hip(float* __restrict__ X, float* __restrict__ Y, int n)" in hip
    assert (
        "__builtin_amdgcn_make_buffer_rsrc((void*)X, /*stride=*/(short)0, "
        "/*num_records=*/(int)n, " in hip
    )
