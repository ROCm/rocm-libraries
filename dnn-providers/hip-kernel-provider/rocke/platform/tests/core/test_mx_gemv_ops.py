# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 ``arith.cvt_scalef32_pk_bf16_fp4`` and ``tile.permlane16_swap``.

CPU-only, Python engine (both ops are in ``PYTHON_ONLY_OPCODES``): builder
validation, the emitted declaration and call at every LLVM flavor, and the
python-only bookkeeping. The gfx950 numerics of both ops are covered by the
MX decode GEMV family test (``library/tests/test_mx_decode_gemv.py``: bitwise
against an exact host reference).
"""

import pytest
from rocke.core.ir import (
    BF16,
    F32,
    I32,
    PURE_OP_NAMES,
    PYTHON_ONLY_OPCODES,
    IRBuilder,
    PtrType,
    VectorType,
    uses_python_only_ops,
)
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python


def _cvt_kernel(byte):
    b = IRBuilder(f"cvt_bf16_fp4_{byte}")
    ps = b.param("S", PtrType(I32, "global"))
    po = b.param("O", PtrType(I32, "global"))
    src = b.global_load(ps, b.const_i32(0), I32, align=4)
    sc = b.bitcast(b.global_load(ps, b.const_i32(1), I32, align=4), F32)
    v = b.cvt_scalef32_pk_bf16_fp4(src, sc, byte)
    assert v.type == VectorType(BF16, 2)
    b.global_store(po, b.const_i32(0), b.bitcast(v, I32))
    return b.kernel


def _swap_kernel():
    b = IRBuilder("p16swap")
    ps = b.param("S", PtrType(I32, "global"))
    lo = b.global_load(ps, b.lane_id(), I32, align=4)
    hi = b.global_load(ps, b.add(b.lane_id(), b.const_i32(64)), I32, align=4)
    x, y = b.permlane16_swap(lo, hi)
    b.global_store(ps, b.lane_id(), b.add(x, y))
    return b.kernel


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("byte", [0, 1, 2, 3])
def test_cvt_emission(flavor, byte):
    ll = _lower_kernel_to_llvm_python(_cvt_kernel(byte), arch="gfx950", llvm_flavor=flavor)
    assert (
        "declare <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp4(i32, float, i32 immarg)"
        in ll
    )
    assert f"@llvm.amdgcn.cvt.scalef32.pk.bf16.fp4(i32 " in ll
    assert ll.count(f", i32 {byte})") >= 1


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_permlane16_swap_emission(flavor):
    ll = _lower_kernel_to_llvm_python(_swap_kernel(), arch="gfx950", llvm_flavor=flavor)
    assert "declare { i32, i32 } @llvm.amdgcn.permlane16.swap(i32, i32, i1 immarg, i1 immarg)" in ll
    assert "call { i32, i32 } @llvm.amdgcn.permlane16.swap(" in ll
    assert ll.count("extractvalue { i32, i32 }") == 2


def test_python_only_bookkeeping():
    for name in ("arith.cvt_scalef32_pk_bf16_fp4", "tile.permlane16_swap"):
        assert name in PYTHON_ONLY_OPCODES
        assert name in PURE_OP_NAMES
    assert uses_python_only_ops(_cvt_kernel(0)) == ["arith.cvt_scalef32_pk_bf16_fp4"]
    assert uses_python_only_ops(_swap_kernel()) == ["tile.permlane16_swap"]


def test_validation():
    b = IRBuilder("bad")
    ps = b.param("S", PtrType(I32, "global"))
    i = b.global_load(ps, b.const_i32(0), I32, align=4)
    f = b.bitcast(i, F32)
    with pytest.raises(ValueError, match="byte must be 0..3"):
        b.cvt_scalef32_pk_bf16_fp4(i, f, 4)
    with pytest.raises(ValueError, match="byte must be 0..3"):
        b.cvt_scalef32_pk_bf16_fp4(i, f, True)
    with pytest.raises(ValueError, match="expects"):
        b.cvt_scalef32_pk_bf16_fp4(f, f, 0)
    with pytest.raises(ValueError, match="expects"):
        b.cvt_scalef32_pk_bf16_fp4(i, i, 0)
    with pytest.raises(ValueError, match="i32"):
        b.permlane16_swap(f, i)
