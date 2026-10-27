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


@pytest.mark.parametrize(
    "dtype", [PtrType(F32, "global"), VectorType(F32, 2), Type("unknown")]
)
def test_rejects_non_numeric_scalar(dtype):
    b = IRBuilder("invalid_barrier")
    value = b.param("value", dtype)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="directly lowerable scalar"):
        b.optimization_barrier(value)
    assert serialize(b.kernel) == before


@pytest.mark.parametrize("dtype", ["fp4", "fp6", "bf6", "e8m0", "e5m3", "tf32"])
def test_logical_types_use_storage_barriers(dtype):
    from rocke.core.ir import dtype_to_ir_type
    from rocke.helpers.mma_io import storage_ir_type

    logical = dtype_to_ir_type(dtype)
    b = IRBuilder("logical_barrier")
    value = b.param("value", logical)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="directly lowerable scalar"):
        b.optimization_barrier(value)
    assert serialize(b.kernel) == before
    storage = storage_ir_type(dtype)
    assert storage == (I32 if dtype == "tf32" else I8)
    assert b.optimization_barrier(b.param("bits", storage)).type == storage


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_predicate_producer(arch):
    b = IRBuilder("barrier_predicate")
    ptr = b.param("p", PtrType(I32, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, I32)
    predicate = b.cmp_lt(value, b.const_i32(0))
    result = b.optimization_barrier(predicate)
    b.global_store(ptr, tid, b.zext(result, I32))
    b.ret()
    copy = parse(serialize(b.kernel))
    llvm = lower_kernel_to_llvm(b.kernel, arch=arch, llvm_flavor="llvm23")
    assert llvm == lower_kernel_to_llvm(copy, arch=arch, llvm_flavor="llvm23")
    assert 'asm "", "=v,0"' in llvm
