# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU lowering checks for ``IRBuilder.fdot2`` (``arith.fdot2``).

``fdot2(a, b, c)`` takes two ``vec<f16x2>`` or two ``vec<bf16x2>`` operands and
an f32 accumulator, and lowers to ``llvm.amdgcn.fdot2`` (f16) or
``llvm.amdgcn.fdot2.f32.bf16`` (bf16) with clamp off.
"""

from __future__ import annotations

import pytest
from rocke.core.ir import BF16, F16, F32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm

_EXPECTED = {
    "f16": (
        "call float @llvm.amdgcn.fdot2(<2 x half> ",
        "declare float @llvm.amdgcn.fdot2(<2 x half>, <2 x half>, float, i1 immarg)",
    ),
    "bf16": (
        "call float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat> ",
        (
            "declare float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat>, <2 x bfloat>, "
            "float, i1 immarg)"
        ),
    ),
}


def _build(dtype: str):
    elem = BF16 if dtype == "bf16" else F16
    b = IRBuilder(f"fdot2_{dtype}_probe")
    A = b.param("A", PtrType(elem, "global"), align=4)
    D = b.param("D", PtrType(F32, "global"), align=4)
    tid = b.thread_id_x()
    a = b.global_load_vN(A, b.mul(tid, b.const_i32(2)), elem, 2)
    acc = b.global_load(D, tid, F32, align=4)
    b.global_store(D, tid, b.fdot2(a, a, acc), align=4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_fdot2_lowers_to_the_dot2_intrinsic(dtype):
    ll = lower_kernel_to_llvm(_build(dtype), arch="gfx950")
    call, decl = _EXPECTED[dtype]
    assert call in ll
    assert decl in ll
    # Clamp off: the immarg is the constant false.
    assert ll.count(", i1 false)") >= 1


def test_fdot2_rejects_mismatched_operands():
    b = IRBuilder("fdot2_bad")
    A = b.param("A", PtrType(F16, "global"), align=4)
    B = b.param("B", PtrType(BF16, "global"), align=4)
    zero = b.const_i32(0)
    a16 = b.global_load_vN(A, zero, F16, 2)
    abf = b.global_load_vN(B, zero, BF16, 2)
    a4 = b.global_load_vN(A, zero, F16, 4)
    acc = b.const_f32(0.0)
    with pytest.raises(ValueError, match="fdot2 expects"):
        b.fdot2(a16, abf, acc)  # element types differ
    with pytest.raises(ValueError, match="fdot2 expects"):
        b.fdot2(a4, a4, acc)  # not two-wide
    with pytest.raises(ValueError, match="fdot2 expects"):
        b.fdot2(a16, a16, b.const_i32(0))  # accumulator is not f32
