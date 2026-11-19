# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Publication primitives: order/scope contracts, effects and paired LLVM bytes.

These are CPU proofs, not evidence of remote publication on a GPU. Set
ROCKE_ORDERED_ATOMICS_EMIT to the built rocke_ordered_atomics_test executable
for an independent C-builder proof (not merely serialized-Python lowering).
"""

from __future__ import annotations

import itertools
import os
import subprocess

import pytest
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.passes import optimize_kernel
from rocke.core.verify import verify

ORDERS = list(
    itertools.product(
        ("monotonic", "acquire", "seq_cst"),
        ("monotonic", "release", "seq_cst"),
        ("acquire", "release", "acq_rel", "seq_cst"),
    )
)


def _kernel(index_type=I32, scope="system", orders=("acquire", "release", "acq_rel")):
    b = IRBuilder("ordered_atomics")
    b.kernel.attrs["max_workgroup_size"] = 64
    flag = b.param("flag", PtrType(I32, "global"))
    index = b.param("index", index_type)
    value = b.param("value", I32)
    b.global_atomic_load_i32(flag, index, ordering=orders[0], scope=scope)
    b.global_atomic_store_i32(flag, index, value, ordering=orders[1], scope=scope)
    b.global_fence(ordering=orders[2], scope=scope)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("index_type", [I32, I64])
@pytest.mark.parametrize("scope", ["system", "agent"])
@pytest.mark.parametrize("orders", ORDERS)
def test_ordered_atomic_llvm_contracts(flavor, arch, index_type, scope, orders):
    kernel = _kernel(index_type, scope, orders)
    assert verify(kernel) == []
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)
    suffix = "" if scope == "system" else ' syncscope("agent")'
    geps = [
        line.split(" = ")[0].strip()
        for line in llvm.splitlines()
        if "getelementptr" in line
    ]
    assert len(geps) == 2
    assert (
        f"load atomic i32, ptr addrspace(1) {geps[0]}{suffix} {orders[0]}, align 4"
        in llvm
    )
    assert (
        f"store atomic i32 %value, ptr addrspace(1) {geps[1]}{suffix} {orders[1]}, align 4"
        in llvm
    )
    assert f"fence{suffix} {orders[2]}" in llvm
    assert all(
        f", {index_type.name} %index" in line
        for line in llvm.splitlines() if "getelementptr" in line
    )
    assert "no.remote.memory" not in llvm
    assert "no.fine.grained.memory" not in llvm
    assert ('syncscope("agent")' in llvm) == (scope == "agent")


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("index_type", [I32, I64])
@pytest.mark.parametrize("scope", ["system", "agent"])
@pytest.mark.parametrize("orders", ORDERS)
def test_serialized_atomic_cpp_lowering_matches(
    flavor, arch, index_type, scope, orders
):
    engine = pytest.importorskip("rocke_engine")
    kernel = _kernel(index_type, scope, orders)
    assert engine.lower_serialized_ir(serialize(kernel), arch=arch, flavor=flavor) == (
        _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)
    )


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("index_type", [I32, I64])
@pytest.mark.parametrize("scope", ["system", "agent"])
@pytest.mark.parametrize("orders", ORDERS)
def test_independent_cpp_atomic_builder_matches(
    flavor, arch, index_type, scope, orders
):
    executable = os.environ.get("ROCKE_ORDERED_ATOMICS_EMIT")
    if not executable:
        pytest.skip("independent C-builder executable not configured")
    result = subprocess.run(
        [executable, "--emit", flavor, arch, index_type.name, scope, *orders],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = _lower_kernel_to_llvm_python(
        _kernel(index_type, scope, orders),
        arch=arch,
        llvm_flavor=flavor,
    )
    assert result.stdout == expected


def test_ordered_atomic_reads_are_not_csed_or_removed():
    kernel = _kernel()
    b = IRBuilder("effectful_reads")
    flag = b.param("flag", PtrType(I32, "global"))
    index = b.param("index", I32)
    b.global_atomic_load_i32(flag, index)
    b.global_atomic_load_i32(flag, index)
    b.ret()
    for candidate in (kernel, b.kernel):
        before = serialize(candidate)
        assert all(not op.is_pure for op in candidate.body.ops)
        stats = optimize_kernel(candidate)
        assert stats.dead_ops_removed == 0
        assert stats.common_subexpressions == 0
        assert serialize(candidate) == before


@pytest.mark.parametrize(
    "method, bad",
    [
        ("global_atomic_load_i32", "release"),
        ("global_atomic_load_i32", "acq_rel"),
        ("global_atomic_store_i32", "acquire"),
        ("global_atomic_store_i32", "acq_rel"),
        ("global_fence", "monotonic"),
        ("global_fence", "unordered"),
    ],
)
def test_invalid_order_rejected_before_emitting(method, bad):
    b = IRBuilder("bad_order")
    flag = b.param("flag", PtrType(I32, "global"))
    index = b.param("index", I32)
    value = b.param("value", I32)
    operands = () if method == "global_fence" else (flag, index)
    if method == "global_atomic_store_i32":
        operands += (value,)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="ordering"):
        getattr(b, method)(*operands, ordering=bad)
    assert serialize(b.kernel) == before


@pytest.mark.parametrize("scope", ["", "workgroup", "singlethread", None])
@pytest.mark.parametrize(
    "method", ["global_atomic_load_i32", "global_atomic_store_i32", "global_fence"]
)
def test_invalid_scope_rejected_before_emitting(scope, method):
    b = IRBuilder("bad_scope")
    flag = b.param("flag", PtrType(I32, "global"))
    index = b.param("index", I32)
    value = b.param("value", I32)
    operands = () if method == "global_fence" else (flag, index)
    if method == "global_atomic_store_i32":
        operands += (value,)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="scope"):
        getattr(b, method)(*operands, scope=scope)
    assert serialize(b.kernel) == before


@pytest.mark.parametrize(
    "pointer_type, index_type, value_type, message",
    [
        (I32, I32, I32, "ptr<i32,global>"),
        (PtrType(F32, "global"), I32, I32, "ptr<i32,global>"),
        (PtrType(I32, "shared"), I32, I32, "ptr<i32,global>"),
        (PtrType(I32, "constant"), I32, I32, "ptr<i32,global>"),
        (PtrType(I32, "global"), F32, I32, "index"),
        (PtrType(I32, "global"), I32, F32, "value"),
    ],
)
def test_invalid_atomic_operands_rejected_before_emitting(
    pointer_type, index_type, value_type, message
):
    b = IRBuilder("bad_operands")
    flag = b.param("flag", pointer_type)
    index = b.param("index", index_type)
    value = b.param("value", value_type)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match=message):
        b.global_atomic_store_i32(flag, index, value)
    assert serialize(b.kernel) == before


@pytest.mark.parametrize("op_index", [0, 1, 2])
@pytest.mark.parametrize(
    "key, value, message",
    [
        ("ordering", "unordered", "invalid ordering"),
        ("scope", "workgroup", "invalid scope"),
        ("ordering", None, "missing required attr"),
        ("scope", None, "missing required attr"),
    ],
)
def test_verifier_rejects_mutated_atomic_attributes(op_index, key, value, message):
    kernel = _kernel()
    op = kernel.body.ops[op_index]
    if value is None:
        del op.attrs[key]
    else:
        op.attrs[key] = value
    assert any(message in diagnostic.message for diagnostic in verify(kernel))


def test_default_order_and_scope_are_explicit_in_ir():
    kernel = _kernel()
    assert kernel.body.ops[0].attrs == {"ordering": "acquire", "scope": "system"}
    assert kernel.body.ops[1].attrs == {"ordering": "release", "scope": "system"}
    assert kernel.body.ops[2].attrs == {"ordering": "acq_rel", "scope": "system"}
