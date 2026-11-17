# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic scalar global accesses preserve index widths in both lowerers."""

from __future__ import annotations

import pytest
from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _kernel(dtype, index_type):
    builder = IRBuilder("typed_global_indices")
    builder.kernel.attrs["max_workgroup_size"] = 64
    source = builder.param("X", PtrType(dtype, "global"))
    destination = builder.param("Y", PtrType(dtype, "global"))
    index = builder.param("index", index_type)
    value = builder.global_load(source, index, dtype=dtype, align=2)
    builder.global_store(destination, index, value, align=2)
    return builder.kernel


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize(
    "dtype, llvm_dtype", [(F16, "half"), (BF16, "bfloat"), (F32, "float"), (I32, "i32")]
)
@pytest.mark.parametrize("index_type", [I32, I64])
def test_native_scalar_geps_use_declared_index_width(
    dtype, llvm_dtype, index_type, flavor
) -> None:
    llvm = _lower_kernel_to_llvm_python(
        _kernel(dtype, index_type), arch="gfx950", llvm_flavor=flavor
    )
    geps = [line for line in llvm.splitlines() if "getelementptr" in line]
    assert len(geps) == 2
    assert all(
        f"inbounds {llvm_dtype}" in line and f", {index_type.name} %index" in line
        for line in geps
    )


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("dtype", [F16, BF16, F32, I32])
@pytest.mark.parametrize("index_type", [I32, I64])
def test_cpp_scalar_memory_indices_match_native(dtype, index_type, flavor) -> None:
    engine = pytest.importorskip("rocke_engine")
    kernel = _kernel(dtype, index_type)
    expected = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    assert (
        engine.lower_serialized_ir(serialize(kernel), arch="gfx950", flavor=flavor)
        == expected
    )
