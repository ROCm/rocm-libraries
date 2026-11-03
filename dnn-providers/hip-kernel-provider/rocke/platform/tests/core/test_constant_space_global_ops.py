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
  atomics, flag ops, the cooperative store, the transposed load and the copies
  between global memory and LDS -- is refused by the LLVM lowering of both
  engines. Most use the text the ``global_ptr_*`` ops use; the two ops that
  already refused a non-global pointer keep their own text.
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
from rocke.tests.engines import assert_both_engines_reject, lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts

FLAVORS = ("llvm20", "llvm22", "llvm23", "llvm24")
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


LOADS = {
    "global_load": (
        lambda b, x: b.global_load_f16(x, b.const_i32(1)),
        F16,
        "half",
        "half",
    ),
    "global_load_typed": (
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


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("op_name", sorted(LOADS))
def test_both_llvm_lowerings_load_through_a_constant_space_parameter(op_name, flavor):
    load, dtype, elem_ty, loaded_ty = LOADS[op_name]
    ir = lower_with_both_engines(
        _load_kernel(f"{op_name}_constant", load, dtype), ARCH, llvm_flavor=flavor
    )
    assert f"@{op_name}_constant(ptr addrspace(4) nocapture %X, " in ir
    assert f"getelementptr inbounds {elem_ty}, ptr addrspace(4) %X, i32 " in ir
    assert f"= load {loaded_ty}, ptr addrspace(4) %gep" in ir
    assert "ptr addrspace(1) %X" not in ir


@pytest.mark.parametrize("op_name", sorted(LOADS))
def test_a_load_through_a_constant_space_parameter_compiles(op_name):
    load, dtype, _, _ = LOADS[op_name]
    ir = lower_with_both_engines(_load_kernel(f"{op_name}_compiles", load, dtype), ARCH)
    assert_target_accepts(ir, ARCH)


@pytest.mark.parametrize(
    "x_space, header_type", [("private", "ptr"), ("lds", "ptr addrspace(3)")]
)
@pytest.mark.parametrize("op_name", sorted(LOADS))
def test_both_llvm_lowerings_refuse_a_load_through_another_space(
    op_name, x_space, header_type
):
    load, dtype, _, _ = LOADS[op_name]
    kernel = _load_kernel(f"{op_name}_{x_space}", load, dtype, x_space)
    assert_both_engines_reject(
        kernel,
        ARCH,
        ValueError,
        f"{op_name}: pointer operand is {header_type}, but the op accepts only "
        "ptr addrspace(1), ptr addrspace(4)",
    )


# --------------------------------------------------------------------------
# Every other global-memory op refuses a constant-space parameter
# --------------------------------------------------------------------------


def _header_refusal(op_name: str) -> str:
    return (
        f"{op_name} ptr must be a global pointer, got ptr addrspace(4) "
        "(its parameter's addr_space)"
    )


def _lds_address(b: IRBuilder, dtype: Type) -> Value:
    return b.smem_addr_of(b.smem_alloc(dtype, [64], name_hint="stage"))


# op name -> (element type of X, the op applied to X, arch, flavor, message)
REFUSALS = {
    "global_store": (
        F16,
        lambda b, x: b.store_f16(
            x, b.const_i32(0), b.global_load_f16(x, b.const_i32(1))
        ),
        ARCH,
        None,
        _header_refusal("global_store"),
    ),
    "global_store_typed": (
        F32,
        lambda b, x: b.global_store(x, b.const_i32(0), b.const_f32(1.0)),
        ARCH,
        None,
        _header_refusal("global_store_typed"),
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
        _header_refusal("global_load_lds"),
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
        _header_refusal("global_load_async_to_lds"),
    ),
    "global_store_async_from_lds": (
        I16,
        lambda b, x: b.global_store_async_from_lds(
            x, _lds_address(b, I16), width_bytes=16
        ),
        "gfx1250",
        "llvm23",
        "global_store_async_from_lds dst_ptr must be a global pointer",
    ),
    "global_load_tr16_b128": (
        I16,
        lambda b, x: b.global_load_tr16_b128(x, dtype=I16),
        "gfx1250",
        "llvm23",
        "global_load_tr16_b128 src_ptr must be a global pointer",
    ),
}


@pytest.mark.parametrize("op_name", sorted(REFUSALS))
def test_both_llvm_lowerings_refuse_a_constant_space_parameter(op_name):
    dtype, apply_op, arch, flavor, message = REFUSALS[op_name]
    b = IRBuilder(f"{op_name}_constant")
    apply_op(b, _constant_param(b, dtype))
    b.ret()
    assert_both_engines_reject(b.kernel, arch, TypeError, message, llvm_flavor=flavor)
