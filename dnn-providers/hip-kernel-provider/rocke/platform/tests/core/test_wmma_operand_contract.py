# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the fixed unscaled WMMA ABI independently of catalog result metadata."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from unittest.mock import patch

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    FP8E4M3,
    I8,
    I32,
    IRBuilder,
    PtrType,
    Value,
    VectorType,
    type_bits,
)
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


@pytest.mark.parametrize("gfx", ["gfx1100", "gfx1101", "gfx1151", "gfx1250"])
@pytest.mark.parametrize("concrete", [False, True])
def test_hip_validation_uses_the_actual_target(gfx, concrete):
    from rocke.core import mma_operand

    case = (
        (gfx, "wmma_gfx1250_f32_16x16x64_fp8_fp8", I32, 8, F32)
        if gfx == "gfx1250"
        else (gfx, "wmma_i32_16x16x16_iu8", I32, 4, I32)
    )
    kernel, op = _kernel(case)
    if concrete:
        op.name = f"tile.{case[1]}"
        op.attrs.pop("op_id")
    seen = []
    real = mma_operand.mma_operand_needs_reinterpret

    def check_target(*args, **kwargs):
        seen.append(args[4].gfx)
        return real(*args, **kwargs)

    with patch.object(
        mma_operand, "mma_operand_needs_reinterpret", side_effect=check_target
    ):
        text = lower_kernel_to_hip(kernel, arch=gfx)
    assert "__builtin_amdgcn_wmma" in text
    assert seen == [gfx, gfx]


def _type_error(case):
    return f"unscaled WMMA requires src2 and dst to be {VectorType(case[4], 8).name}"


def _llvm_type_error(case, op, role):
    if case[4] != I32:
        return _type_error(case)
    actual = op.operands[2].type if role == "src2" else op.result.type
    element = actual.elem if isinstance(actual, VectorType) else actual
    label = (
        f"MMA operand c of {case[1]!r}"
        if role == "src2"
        else f"MMA result of {case[1]!r}"
    )
    verb = "takes" if role == "src2" else "returns"
    if element != I32:
        hint = (
            " -- convert the values first"
            if role == "src2"
            else " -- declare the result in an element type it returns"
        )
        return f"{label} holds {element.name} values; the instruction {verb} i32{hint}"
    hint = (
        " -- pass a 256-bit value" if role == "src2" else " -- declare a 256-bit result"
    )
    return (
        f"{label} is {actual.name} ({type_bits(actual)} bits); the instruction "
        f"{verb} vec<i32x8> (256 bits){hint}"
    )


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
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_llvm_rejects_actual_types(case, flavor, engine, subtests):
    if engine == "native":
        pytest.importorskip("rocke_engine")
    # Keep subtest capture/reporting off the successful path.
    for role, kind in product(("src2", "dst"), (4, 7, 16, "scalar", "dtype")):
        try:
            kernel, op = _kernel(case)
            _corrupt_type(kernel, op, case, role, kind)
            _assert_llvm_rejects(
                kernel, case[0], flavor, _llvm_type_error(case, op, role), engine
            )
        except (Exception, pytest.fail.Exception, pytest.skip.Exception):
            with subtests.test(role=role, kind=kind):
                raise


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize(
    "kind", ["missing_operand", "extra_operand", "missing_result", "extra_result"]
)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_llvm_rejects_arity(case, flavor, kind, engine):
    kernel, op = _kernel(case)
    _corrupt_arity(op, kind)
    error = ARITY_ERROR
    if case[4] == I32:
        name = f"tile.{case[1]}"
        error = (
            f"{name} expects 3 operands, got {len(op.operands)}"
            if kind in ("missing_operand", "extra_operand")
            else f"{name}: expected exactly one result, got {len(op.results)}"
        )
    _assert_llvm_rejects(kernel, case[0], flavor, error, engine)


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
    kernel, op = _kernel(case, atom=replace(atom, dst=dst), use_result=True)
    _assert_llvm_rejects(
        kernel, case[0], flavor, _llvm_type_error(case, op, "dst"), engine
    )


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


@pytest.mark.parametrize(
    "case,elem,count",
    [
        (CASES[0], I32, 4.0),
        (CASES[0], I8, 16.0),
        (CASES[1], I32, 2.0),
        (CASES[8], I32, 8.0),
        (CASES[8], FP8E4M3, 32.0),
        (CASES[11], BF8E5M2, 32.0),
    ],
)
@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.parametrize(
    "flavor,concrete",
    [(flavor, False) for flavor in FLAVORS] + [("hip", False), ("hip", True)],
)
def test_wmma_rejects_non_integer_carrier_counts(
    case, elem, count, role, concrete, flavor
):
    from rocke.core.lower_hip import _Lowerer as HipLowerer
    from rocke.core.lower_llvm import _Lowerer as LlvmLowerer

    kernel, op = _kernel(case)
    malformed = VectorType(elem, count)
    op.operands[role].type = malformed
    kernel.params[role].type = malformed
    if concrete:
        op.name = f"tile.{op.attrs.pop('op_id')}"
    # LLVM WMMA dispatch uses tile.mma; HIP also supports concrete operations.
    # Parsing rejects decimal counts, so exercise actual SSA at the final gate.
    lowerer = (
        HipLowerer(kernel, arch=case[0])
        if flavor == "hip"
        else LlvmLowerer(kernel, arch=case[0], llvm_flavor=flavor)
    )
    with pytest.raises(ValueError) as caught:
        lowerer.lower_op(op)
    assert str(caught.value) == (
        f"MMA operand {('a', 'b')[role]} of {case[1]!r} requires an integer "
        f"operand vector count, got {count!r}"
    )
    if flavor == "hip":
        assert lowerer.lines == []
    else:
        assert lowerer._needs_intrin == {}
        assert all(block.lines == [] for block in lowerer._blocks)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_wmma_llvm_rejects_multiplicands(case, flavor, engine, subtests):
    if engine == "native":
        pytest.importorskip("rocke_engine")
    for role, kind in product((0, 1), ("scalar", "dtype", "short", "long")):
        try:
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
            error = (
                f"unscaled WMMA requires src{role} to be {VectorType(elem, width).name}"
            )
            if elem == I32:
                from rocke.core.arch import ArchTarget
                from rocke.core.mma_operand import mma_operand_needs_reinterpret

                with pytest.raises(ValueError) as bridge:
                    mma_operand_needs_reinterpret(
                        case[1],
                        "a" if role == 0 else "b",
                        bad_type,
                        VectorType(elem, width),
                        ArchTarget.from_gfx(case[0]),
                    )
                error = str(bridge.value)
            _assert_llvm_rejects(kernel, case[0], flavor, error, engine)
        except (Exception, pytest.fail.Exception, pytest.skip.Exception):
            with subtests.test(role=role, kind=kind):
                raise


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
    if elem == I32:
        from rocke.core.arch import ArchTarget
        from rocke.core.mma_operand import mma_operand_needs_reinterpret

        with pytest.raises(ValueError) as bridge:
            mma_operand_needs_reinterpret(
                case[1],
                "a" if role == 0 else "b",
                bad_type,
                VectorType(elem, width),
                ArchTarget.from_gfx(case[0]),
            )
        error = str(bridge.value)
    for candidate in (kernel, parse(serialize(kernel))):
        _assert_python_rejects(
            lambda: lower_kernel_to_hip(candidate, arch=case[0]), error
        )


# Independent register widths, including the scalar gfx12 iu4 register (one
# word, accepted either as i32 or a same-bit vec<i32x1>). Signedness changes the
# call's flags, not the integer accumulator/result or the preflight boundary.
_INTEGER_WMMA_SHAPES = [
    ("gfx1151", "wmma_i32_16x16x16_iu8", 4, 8),
    ("gfx1151", "wmma_i32_16x16x16_iu4", 2, 4),
    ("gfx1201", "wmma_gfx12_i32_16x16x16_iu8", 2, 8),
    ("gfx1201", "wmma_gfx12_i32_16x16x16_iu4", 1, 4),
    ("gfx1201", "wmma_gfx12_i32_16x16x32_iu4", 2, 4),
    ("gfx1250", "wmma_gfx1250_i32_16x16x64_iu8", 8, 8),
]
_INTEGER_WMMA_CASES = [
    (gfx, op_id + suffix, I32, words, I32)
    for gfx, op_id, words, bits in _INTEGER_WMMA_SHAPES
    for suffix in ("", f"_u{bits}_i{bits}", f"_i{bits}_u{bits}", f"_u{bits}_u{bits}")
]


@pytest.mark.parametrize(
    "case", _INTEGER_WMMA_CASES, ids=[c[1] for c in _INTEGER_WMMA_CASES]
)
@pytest.mark.parametrize("flavor", [*FLAVORS, "llvm24"])
def test_integer_registry_preserves_valid_carriers_and_signedness(case, flavor):
    kernel, _ = _kernel(case, use_result=True)
    native = pytest.importorskip("rocke_engine")
    text = _lower_kernel_to_llvm_python(kernel, arch=case[0], llvm_flavor=flavor)
    assert (
        native.lower_serialized_ir(serialize(kernel), arch=case[0], flavor=flavor)
        == text
    )
    if case[3] == 1:
        # The logical vector is reinterpreted as the instruction's scalar word.
        assert "bitcast <1 x i32>" in text


@pytest.mark.parametrize(
    "case", _INTEGER_WMMA_CASES, ids=[c[1] for c in _INTEGER_WMMA_CASES]
)
@pytest.mark.parametrize("flavor", [*FLAVORS, "llvm24"])
@pytest.mark.parametrize("role", ["src2", "dst"])
@pytest.mark.parametrize("engine", ["python", "native"])
def test_integer_registry_preflights_fixed_source_and_result(
    case, flavor, role, engine
):
    kernel, op = _kernel(case)
    _corrupt_type(kernel, op, case, role, 7)
    error = _llvm_type_error(case, op, role)
    _assert_llvm_rejects(kernel, case[0], flavor, error, engine)
    if engine == "python":
        from rocke.core.lower_llvm import _Lowerer

        lowerer = _Lowerer(kernel, arch=case[0], llvm_flavor=flavor)
        _assert_python_rejects(lambda: lowerer.lower_op(op), error)
        assert lowerer._needs_intrin == {}
        assert all(block.lines == [] for block in lowerer._blocks)


@pytest.mark.parametrize(
    "case", _INTEGER_WMMA_CASES, ids=[c[1] for c in _INTEGER_WMMA_CASES]
)
def test_integer_registry_refuses_non_integer_counts_before_emission(case):
    from rocke.core.lower_llvm import _Lowerer

    kernel, op = _kernel(case)
    count = float(case[3])
    op.operands[0].type = VectorType(I32, count)
    lowerer = _Lowerer(kernel, arch=case[0], llvm_flavor="llvm24")
    with pytest.raises(ValueError) as caught:
        lowerer.lower_op(op)
    assert str(caught.value) == (
        f"MMA operand a of {case[1]!r} requires an integer operand vector count, got {count!r}"
    )
    assert lowerer._needs_intrin == {}
    assert all(block.lines == [] for block in lowerer._blocks)
