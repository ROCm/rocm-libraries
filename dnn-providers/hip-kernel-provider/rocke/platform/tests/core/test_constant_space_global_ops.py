# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Global-memory ops on a pointer parameter declared ``addr_space="constant"``.

Such a parameter keeps its IR type (a global pointer), but the kernel header
types it ``ptr addrspace(4)``. An instruction that names it
``ptr addrspace(1)`` does not match the header, and LLVM rejects the module.

- The global loads (``global_load_f16``, ``global_load``, ``global_load_vN``)
  name the pointer's own type, so they read through a constant-space parameter
  and the module compiles. They refuse a pointer in any other space (flat or
  LDS), which they used to name ``ptr addrspace(1)`` as well.
- Every other op that addresses global memory through the pointer -- stores,
  atomics, flag ops, the cooperative store, prefetch, the transposed and
  cluster loads, and the copies between global memory and LDS -- is refused by
  the LLVM lowering of both engines, with one text that names the op's builder
  and the operand. ``flat_prefetch`` refuses a parameter whose override moved
  it out of the flat space the same way.
- Every one of these refusals is a ``TypeError`` (``ROCKE_ERR_TYPE`` in the
  C++ engine). The text blames the parameter's ``addr_space`` only when the
  override moved its header type.

Each case runs per engine, so the Python engine is checked in every lane; the
C++ cases skip where the ``rocke_engine`` binding is not built. There,
``future_intrinsic_lowering``'s ``constant_space_global_ops`` CTest case
checks a sample of the C++ engine's side: the three loads and the
``store_f16``, ``global_store`` and ``global_atomic_add`` refusals.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import ENGINES, assert_engine_rejects, lower_with_engine
from rocke.tests.ir_validity import assert_target_accepts

ARCH = "gfx942"


def _constant_param(b: IRBuilder, dtype: Type) -> Value:
    return b.param("X", PtrType(dtype, "global"), addr_space="constant")


# --------------------------------------------------------------------------
# Loads read through a constant-space parameter
# --------------------------------------------------------------------------


def _load_kernel(
    name: str,
    load: Callable[[IRBuilder, Value], Value],
    dtype: Type,
    x_space: str = "constant",
) -> KernelDef:
    """Load from ``X`` with ``load`` and store the value to the global ``Y``.

    ``X`` is a global pointer parameter declared ``addr_space="constant"``, or,
    for any other ``x_space``, a pointer parameter in that space."""
    b = IRBuilder(name)
    if x_space == "constant":
        x = _constant_param(b, dtype)
    else:
        x = b.param("X", PtrType(dtype, x_space))
    y = b.param("Y", PtrType(dtype, "global"))
    value = load(b, x)
    if value.type == dtype:
        b.global_store(y, b.const_i32(0), value)
    else:
        b.global_store_vN(y, b.const_i32(0), value, 4)
    b.ret()
    return b.kernel


# builder -> (the load applied to X, element type of X, GEP element, loaded type)
LOADS = {
    "global_load_f16": (
        lambda b, x: b.global_load_f16(x, b.const_i32(1)),
        F16,
        "half",
        "half",
    ),
    "global_load": (
        lambda b, x: b.global_load(x, b.const_i32(1), F32, align=4),
        F32,
        "float",
        "float",
    ),
    "global_load_vN": (
        lambda b, x: b.global_load_vN(x, b.const_i32(4), F16, 4),
        F16,
        "half",
        "<4 x half>",
    ),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("builder", sorted(LOADS))
def test_a_load_reads_through_a_constant_space_parameter(builder, flavor, engine):
    load, dtype, elem_ty, loaded_ty = LOADS[builder]
    ir = lower_with_engine(
        _load_kernel(f"{builder}_constant", load, dtype),
        ARCH,
        engine=engine,
        llvm_flavor=flavor,
    )
    assert f"@{builder}_constant(ptr addrspace(4) nocapture %X, " in ir
    assert f"getelementptr inbounds {elem_ty}, ptr addrspace(4) %X, i32 " in ir
    assert f"= load {loaded_ty}, ptr addrspace(4) %gep" in ir
    assert "ptr addrspace(1) %X" not in ir


@pytest.mark.parametrize("builder", sorted(LOADS))
def test_a_load_through_a_constant_space_parameter_compiles(builder):
    """The IR is the same for both engines (the test above), so one compile
    judges it."""
    load, dtype, _, _ = LOADS[builder]
    ir = lower_with_engine(
        _load_kernel(f"{builder}_compiles", load, dtype), ARCH, engine="python"
    )
    assert_target_accepts(ir, ARCH)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "x_space, header_type", [("private", "ptr"), ("lds", "ptr addrspace(3)")]
)
@pytest.mark.parametrize("builder", sorted(LOADS))
def test_a_load_refuses_a_pointer_in_another_space(
    builder, x_space, header_type, engine
):
    load, dtype, _, _ = LOADS[builder]
    kernel = _load_kernel(f"{builder}_{x_space}", load, dtype, x_space)
    assert_engine_rejects(
        kernel,
        ARCH,
        engine=engine,
        error_type=TypeError,
        message=f"{builder}: pointer operand is {header_type}, but the op accepts only "
        "ptr addrspace(1), ptr addrspace(4)",
    )


# --------------------------------------------------------------------------
# Every other global-memory op refuses a constant-space parameter
# --------------------------------------------------------------------------


def _header_refusal(builder: str, operand: str = "ptr") -> str:
    return (
        f"{builder} {operand} must be a global pointer, got ptr addrspace(4) "
        "(its parameter's addr_space)"
    )


def _lds_address(b: IRBuilder, dtype: Type) -> Value:
    return b.smem_addr_of(b.smem_alloc(dtype, [64], name_hint="stage"))


# builder -> (element type of X, the op applied to X, arch, flavor, message)
REFUSALS = {
    "store_f16": (
        F16,
        lambda b, x: b.store_f16(
            x, b.const_i32(0), b.global_load_f16(x, b.const_i32(1))
        ),
        ARCH,
        None,
        _header_refusal("store_f16"),
    ),
    "global_store": (
        F32,
        lambda b, x: b.global_store(x, b.const_i32(0), b.const_f32(1.0)),
        ARCH,
        None,
        _header_refusal("global_store"),
    ),
    "global_store_vN": (
        F16,
        lambda b, x: b.global_store_vN(x, b.const_i32(0), b.zero_vec(F16, 4), 4),
        ARCH,
        None,
        _header_refusal("global_store_vN"),
    ),
    "global_atomic_add": (
        I32,
        lambda b, x: b.global_atomic_add(x, b.const_i32(0), b.const_i32(1)),
        ARCH,
        None,
        _header_refusal("global_atomic_add"),
    ),
    "global_atomic_add_f32": (
        F32,
        lambda b, x: b.global_atomic_add_f32(x, b.const_i32(0), b.const_f32(1.0)),
        ARCH,
        None,
        _header_refusal("global_atomic_add_f32"),
    ),
    "global_atomic_add_pk_bf16": (
        BF16,
        lambda b, x: b.global_atomic_add_pk_bf16(
            x, b.const_i32(0), b.zero_vec(BF16, 2)
        ),
        ARCH,
        None,
        _header_refusal("global_atomic_add_pk_bf16"),
    ),
    "global_atomic_add_pk_f16": (
        F16,
        lambda b, x: b.global_atomic_add_pk_f16(x, b.const_i32(0), b.zero_vec(F16, 2)),
        ARCH,
        None,
        _header_refusal("global_atomic_add_pk_f16"),
    ),
    "global_flag_store": (
        I32,
        lambda b, x: b.global_flag_store(x, b.const_i32(0), b.const_i32(1)),
        ARCH,
        None,
        _header_refusal("global_flag_store"),
    ),
    "global_flag_wait_eq": (
        I32,
        lambda b, x: b.global_flag_wait_eq(x, b.const_i32(0), b.const_i32(1)),
        ARCH,
        None,
        _header_refusal("global_flag_wait_eq"),
    ),
    "cooperative_global_store": (
        F32,
        lambda b, x: b.cooperative_global_store(
            x, b.zero_vec(I32, 4), b.zero_vec(F32, 4)
        ),
        ARCH,
        None,
        _header_refusal("cooperative_global_store"),
    ),
    "global_load_lds": (
        F32,
        lambda b, x: b.global_load_lds(x, b.const_i32(0), _lds_address(b, F32), 4),
        ARCH,
        None,
        _header_refusal("global_load_lds", "src_ptr"),
    ),
    "global_load_async_to_lds": (
        I32,
        lambda b, x: b.global_load_async_to_lds(
            x,
            b.const_i32(0),
            b.smem_alloc(I32, [64], name_hint="stage"),
            [b.const_i32(0)],
            width_bytes=4,
        ),
        "gfx1250",
        "llvm23",
        _header_refusal("global_load_async_to_lds", "src_ptr"),
    ),
    "global_store_async_from_lds": (
        I16,
        lambda b, x: b.global_store_async_from_lds(
            x, _lds_address(b, I16), width_bytes=16
        ),
        "gfx1250",
        "llvm23",
        _header_refusal("global_store_async_from_lds", "dst_ptr"),
    ),
    "global_load_tr16_b128": (
        I16,
        lambda b, x: b.global_load_tr16_b128(x, dtype=I16),
        "gfx1250",
        "llvm23",
        _header_refusal("global_load_tr16_b128", "src_ptr"),
    ),
    "global_prefetch": (
        I32,
        lambda b, x: b.global_prefetch(x),
        "gfx1250",
        "llvm23",
        _header_refusal("global_prefetch"),
    ),
    "cluster_load": (
        I32,
        lambda b, x: b.cluster_load(x, b.const_i32(1)),
        "gfx1250",
        "llvm23",
        _header_refusal("cluster_load"),
    ),
    "cluster_load_async_to_lds": (
        I32,
        lambda b, x: b.cluster_load_async_to_lds(
            x, _lds_address(b, I32), b.const_i32(1), width_bytes=4
        ),
        "gfx1250",
        "llvm23",
        _header_refusal("cluster_load_async_to_lds", "src_ptr"),
    ),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("builder", sorted(REFUSALS))
def test_an_op_that_names_a_global_pointer_refuses_a_constant_space_parameter(
    builder, engine
):
    dtype, apply_op, arch, flavor, message = REFUSALS[builder]
    b = IRBuilder(f"{builder}_constant")
    apply_op(b, _constant_param(b, dtype))
    b.ret()
    assert_engine_rejects(
        b.kernel,
        arch,
        engine=engine,
        error_type=TypeError,
        message=message,
        llvm_flavor=flavor,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_the_refusal_blames_addr_space_only_when_the_override_moved_it(engine):
    """An LDS parameter without an ``addr_space`` override reaches the store as
    ``ptr addrspace(3)``; its own type is the cause, not an override."""
    b = IRBuilder("store_lds_param")
    x = b.param("X", PtrType(F32, "lds"))
    b.global_store(x, b.const_i32(0), b.const_f32(1.0))
    b.ret()
    assert_engine_rejects(
        b.kernel,
        ARCH,
        engine=engine,
        error_type=TypeError,
        message="global_store ptr must be a global pointer, got ptr addrspace(3)",
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_flat_prefetch_refuses_a_parameter_moved_to_constant_space(engine):
    b = IRBuilder("flat_prefetch_constant")
    x = b.param("X", PtrType(I32, "private"), addr_space="constant")
    b.flat_prefetch(x)
    b.ret()
    assert_engine_rejects(
        b.kernel,
        "gfx1250",
        engine=engine,
        error_type=TypeError,
        message="flat_prefetch ptr must be a flat pointer, got ptr addrspace(4) "
        "(its parameter's addr_space)",
        llvm_flavor="llvm23",
    )
