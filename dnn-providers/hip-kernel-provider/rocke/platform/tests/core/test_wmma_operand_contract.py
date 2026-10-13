# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the fixed unscaled WMMA ABI independently of catalog result metadata."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType, Value, VectorType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

# Explicit physical shapes: a change to the catalog must not change the oracle.
CASES = [
    ("gfx1151", "wmma_i32_16x16x16_iu8", I32, 4, I32),
    ("gfx1151", "wmma_i32_16x16x16_iu4", I32, 2, I32),
    ("gfx1151", "wmma_f32_16x16x16_f16", F16, 16, F32),
    ("gfx1151", "wmma_f32_16x16x16_bf16", BF16, 16, F32),
    ("gfx1201", "wmma_gfx12_f32_16x16x16_f16", F16, 8, F32),
    ("gfx1201", "wmma_gfx12_f32_16x16x16_bf16", BF16, 8, F32),
    ("gfx1250", "wmma_gfx1250_f32_16x16x32_f16", F16, 16, F32),
    ("gfx1250", "wmma_gfx1250_f32_16x16x32_bf16", BF16, 16, F32),
    *[
        ("gfx1250", f"wmma_gfx1250_f32_16x16x64_{ab}", I32, 8, F32)
        for ab in ("fp8_fp8", "fp8_bf8", "bf8_fp8", "bf8_bf8")
    ],
]
FLAVORS = ["llvm20", "llvm22", "llvm23"]
ARITY_ERROR = "unscaled WMMA expects 3 operands and 1 result"


def _kernel(case, *, atom=None, use_result=False):
    arch, op_id, elem, width, accum = case
    b = IRBuilder("wmma_operand_contract")
    a = b.param("a", VectorType(elem, width))
    bb = b.param("b", VectorType(elem, width))
    c = b.param("c", VectorType(accum, 8))
    d = b.mma(atom or ArchTarget.from_gfx(arch).mma.by_op_id(op_id), a, bb, c)
    op = b.kernel.body.ops[-1]
    if use_result:
        out = b.param("out", PtrType(accum, "global"))
        b.global_store(out, b.const_i32(0), b.vec_extract(d, 7))
    b.ret()
    return b.kernel, op


def _type_error(case):
    return f"unscaled WMMA requires src2 and dst to be {VectorType(case[4], 8).name}"


def _assert_python_rejects(lower, error):
    with pytest.raises(ValueError) as caught:
        lower()
    assert str(caught.value) == error


def _assert_llvm_rejects(kernel, arch, flavor, error, engine):
    ir = serialize(kernel)
    if engine == "python":
        for candidate in (kernel, parse(ir)):
            _assert_python_rejects(
                lambda: _lower_kernel_to_llvm_python(
                    candidate, arch=arch, llvm_flavor=flavor
                ),
                error,
            )
        return
    native = pytest.importorskip("rocke_engine")
    with pytest.raises(RuntimeError) as caught:
        native.lower_serialized_ir(ir, arch=arch, flavor=flavor)
    assert str(caught.value) == (
        f"rocke_engine.lower_serialized_ir: lower failed for arch '{arch}' "
        f"(status 1): {error}"
    )


def _corrupt_type(kernel, op, case, role, kind):
    elem = case[4]
    if kind == "scalar":
        bad_type = elem
    elif kind == "dtype":
        bad_type = VectorType(F32 if elem == I32 else I32, 8)
    else:
        bad_type = VectorType(elem, kind)
    if role == "src2":
        op.operands[2].type = bad_type
        next(p for p in kernel.params if p.name == "c").type = bad_type
    else:
        op.result.type = bad_type


def _corrupt_arity(op, kind):
    if kind == "missing_operand":
        op.operands.pop()
    elif kind == "extra_operand":
        op.operands.append(op.operands[0])
    elif kind == "missing_result":
        op.results.clear()
    else:
        op.results.append(Value("%extra", op.result.type))


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
def test_wmma_valid_llvm_bytes(case, flavor):
    native = pytest.importorskip("rocke_engine")
    kernel, _ = _kernel(case, use_result=True)
    ir = serialize(kernel)
    expected = _lower_kernel_to_llvm_python(kernel, arch=case[0], llvm_flavor=flavor)
    assert native.lower_serialized_ir(ir, arch=case[0], flavor=flavor) == expected
    assert (
        _lower_kernel_to_llvm_python(parse(ir), arch=case[0], llvm_flavor=flavor)
        == expected
    )


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("role", ["src2", "dst"])
@pytest.mark.parametrize("kind", [4, 7, 16, "scalar", "dtype"])
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_llvm_rejects_actual_types(case, flavor, role, kind, engine):
    kernel, op = _kernel(case)
    _corrupt_type(kernel, op, case, role, kind)
    _assert_llvm_rejects(kernel, case[0], flavor, _type_error(case), engine)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize(
    "kind", ["missing_operand", "extra_operand", "missing_result", "extra_result"]
)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_llvm_rejects_arity(case, flavor, kind, engine):
    kernel, op = _kernel(case)
    _corrupt_arity(op, kind)
    _assert_llvm_rejects(kernel, case[0], flavor, ARITY_ERROR, engine)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("kind", ["width", "dtype"])
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_rejects_custom_destination(case, flavor, kind, engine):
    atom = ArchTarget.from_gfx(case[0]).mma.by_op_id(case[1])
    dst = (
        replace(atom.dst, frag_len=7)
        if kind == "width"
        else replace(atom.dst, dtype="fp32" if case[4] == I32 else "i32")
    )
    kernel, _ = _kernel(case, atom=replace(atom, dst=dst), use_result=True)
    _assert_llvm_rejects(kernel, case[0], flavor, _type_error(case), engine)


# Python HIP has no gfx11/gfx12 bf16 handlers. Native HIP has no unscaled WMMA
# handlers; its unsupported status is covered separately in the C++ suite.
HIP_CASES = [c for c in CASES if c[2] != BF16 or c[0] == "gfx1250"]


@pytest.mark.parametrize("case", HIP_CASES, ids=[c[1] for c in HIP_CASES])
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "valid",
        4,
        7,
        16,
        "scalar",
        "dtype",
        "missing_operand",
        "extra_operand",
        "missing_result",
        "extra_result",
    ],
)
@pytest.mark.parametrize("role", ["src2", "dst"])
def test_wmma_hip_contract(case, concrete, kind, role):
    kernel, op = _kernel(case, use_result=kind == "valid")
    if concrete:
        op.name = f"tile.{op.attrs.pop('op_id')}"
    if kind == "valid":
        assert lower_kernel_to_hip(kernel, arch=case[0]) == lower_kernel_to_hip(
            parse(serialize(kernel)), arch=case[0]
        )
        return
    if kind in ("missing_operand", "extra_operand", "missing_result", "extra_result"):
        _corrupt_arity(op, kind)
        error = ARITY_ERROR
    else:
        _corrupt_type(kernel, op, case, role, kind)
        error = _type_error(case)
    for candidate in (kernel, parse(serialize(kernel))):
        _assert_python_rejects(
            lambda: lower_kernel_to_hip(candidate, arch=case[0]), error
        )


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.parametrize("kind", ["scalar", "dtype", "short", "long"])
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_llvm_rejects_multiplicands(case, flavor, role, kind, engine):
    kernel, op = _kernel(case)
    elem, width = case[2:4]
    bad_type = (
        elem
        if kind == "scalar"
        else (
            VectorType(F32 if elem != F32 else I32, width)
            if kind == "dtype"
            else VectorType(elem, width + (-1 if kind == "short" else 1))
        )
    )
    op.operands[role].type = bad_type
    next(p for p in kernel.params if p.name == ("a", "b")[role]).type = bad_type
    error = f"unscaled WMMA requires src{role} to be {VectorType(elem, width).name}"
    _assert_llvm_rejects(kernel, case[0], flavor, error, engine)


@pytest.mark.parametrize("case", HIP_CASES, ids=[c[1] for c in HIP_CASES])
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.parametrize("kind", ["scalar", "dtype", "short", "long"])
def test_wmma_hip_rejects_multiplicands(case, concrete, role, kind):
    kernel, op = _kernel(case)
    if concrete:
        op.name = f"tile.{op.attrs.pop('op_id')}"
    elem, width = case[2:4]
    bad_type = (
        elem
        if kind == "scalar"
        else (
            VectorType(F32, width)
            if kind == "dtype"
            else VectorType(elem, width + (-1 if kind == "short" else 1))
        )
    )
    op.operands[role].type = bad_type
    next(p for p in kernel.params if p.name == ("a", "b")[role]).type = bad_type
    error = f"unscaled WMMA requires src{role} to be {VectorType(elem, width).name}"
    for candidate in (kernel, parse(serialize(kernel))):
        _assert_python_rejects(
            lambda: lower_kernel_to_hip(candidate, arch=case[0]), error
        )
