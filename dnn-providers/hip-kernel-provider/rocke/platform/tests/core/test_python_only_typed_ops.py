# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dual-engine opcodes with a Python-engine-only operand type.

``math.sqrt`` is lowered by both engines for f32; only the Python lowering
emits the f64 form (``llvm.sqrt.f64``, the FlyDSL mono's square root, used
by the Qwen3.8 ``stage_q2`` body). ``uses_python_only_ops`` must report it
(``math.sqrt.f64``) so a kernel carrying it is routed to the Python engine
up front instead of failing in the C++ engine (NOTIMPL). CPU only.
"""

import pytest
from rocke.core.ir import (
    F32,
    F64,
    PYTHON_ONLY_OPCODES,
    PYTHON_ONLY_TYPED_OPS,
    IRBuilder,
    PtrType,
    uses_python_only_ops,
)
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python


def _sqrt_kernel(f64: bool, nested: bool = False):
    b = IRBuilder(f"sqrt_{'f64' if f64 else 'f32'}{'_nested' if nested else ''}")
    ps = b.param("S", PtrType(F32, "global"))
    po = b.param("O", PtrType(F64 if f64 else F32, "global"))
    x = b.global_load(ps, b.const_i32(0), F32, align=4)

    def body():
        v = b.sqrt(b.cvt_f32_to_f64(x) if f64 else x)
        b.global_store(po, b.const_i32(0), v)

    if nested:
        with b.scf_if(b.cmp_eq(b.thread_id_x(), b.const_i32(0))):
            body()
    else:
        body()
    return b.kernel


def test_f64_sqrt_is_reported_python_only():
    assert "math.sqrt" not in PYTHON_ONLY_OPCODES  # dual-engine for f32
    assert PYTHON_ONLY_TYPED_OPS[("math.sqrt", "f64")] == "math.sqrt.f64"
    assert uses_python_only_ops(_sqrt_kernel(False)) == []
    assert uses_python_only_ops(_sqrt_kernel(True)) == ["math.sqrt.f64"]
    assert uses_python_only_ops(_sqrt_kernel(True, nested=True)) == ["math.sqrt.f64"]


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_f64_sqrt_emission(flavor):
    ll = _lower_kernel_to_llvm_python(_sqrt_kernel(True), arch="gfx950", llvm_flavor=flavor)
    assert "call double @llvm.sqrt.f64(double " in ll
    ll32 = _lower_kernel_to_llvm_python(_sqrt_kernel(False), arch="gfx950", llvm_flavor=flavor)
    assert "llvm.sqrt.f64" not in ll32 and "call float @llvm.sqrt.f32(float " in ll32
