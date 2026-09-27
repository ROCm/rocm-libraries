# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value barriers preserve types and survive serialization through both engines."""

import pytest

from rocke.core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I1,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
    Type,
    VectorType,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm


@pytest.mark.parametrize(
    "dtype", [I1, I8, I16, I32, I64, BF16, F16, F32, FP8E4M3, BF8E5M2]
)
def test_scalar_barrier_roundtrip(dtype):
    b = IRBuilder("barrier")
    ptr = b.param("p", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, dtype)
    result = b.optimization_barrier(value)
    assert result.type == value.type
    b.global_store(ptr, tid, result)
    b.ret()
    copy = parse(serialize(b.kernel))
    for arch in ("gfx950", "gfx1250"):
        llvm = lower_kernel_to_llvm(b.kernel, arch=arch, llvm_flavor="llvm23")
        assert llvm == lower_kernel_to_llvm(copy, arch=arch, llvm_flavor="llvm23")
        assert 'asm "", "=v,0"' in llvm
        assert "asm sideeffect" not in llvm
        hip = lower_kernel_to_hip(copy, arch=arch)
        assert 'asm ("" : "=v"' in hip
        assert '"memory"' not in hip


@pytest.mark.parametrize(
    "dtype", [PtrType(F32, "global"), VectorType(F32, 2), Type("unknown")]
)
def test_rejects_non_numeric_scalar(dtype):
    b = IRBuilder("invalid_barrier")
    value = b.param("value", dtype)
    with pytest.raises(ValueError, match="numeric scalar"):
        b.optimization_barrier(value)
