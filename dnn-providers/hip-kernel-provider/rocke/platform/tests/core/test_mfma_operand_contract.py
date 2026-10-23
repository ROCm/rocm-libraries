# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check actual MFMA SSA types against the selected fixed instruction ABI."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    F64,
    FP8E4M3,
    I8,
    I16,
    I32,
    I64,
    TF32,
    U8,
    IRBuilder,
    PtrType,
    Value,
    VectorType,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python, _Lowerer

# Explicit emitter signatures: catalog metadata cannot change the test oracle.
# Fields are target, op_id, actual source type, accumulator width, packed bits.
CASES = [
    ("gfx942", "mfma_f32_16x16x4_f32", F32, 4, None),
    ("gfx942", "mfma_f32_32x32x2_f32", F32, 16, None),
    ("gfx942", "mfma_f32_16x16x16_f16", VectorType(F16, 4), 4, None),
    ("gfx950", "mfma_f32_16x16x32_f16", VectorType(F16, 8), 4, None),
    ("gfx942", "mfma_f32_32x32x8_f16", VectorType(F16, 4), 16, None),
    ("gfx950", "mfma_f32_32x32x16_f16", VectorType(F16, 8), 16, None),
    ("gfx942", "mfma_f32_4x4x4_f16", VectorType(F16, 4), 4, None),
    ("gfx942", "mfma_f32_16x16x16_bf16", VectorType(BF16, 4), 4, None),
    ("gfx950", "mfma_f32_16x16x32_bf16", VectorType(BF16, 8), 4, None),
    ("gfx942", "mfma_f32_32x32x8_bf16", VectorType(BF16, 4), 16, None),
    ("gfx950", "mfma_f32_32x32x16_bf16", VectorType(BF16, 8), 16, None),
    ("gfx942", "mfma_f32_16x16x8_xf32", VectorType(TF32, 2), 4, None),
    ("gfx942", "mfma_f32_32x32x4_xf32", VectorType(TF32, 2), 16, None),
    *[
        ("gfx942", f"mfma_f32_{shape}_{suffix}", VectorType(elem, 8), count, None)
        for shape, count in (("16x16x32", 4), ("32x32x16", 16))
        for suffix, elem in (("fp8", FP8E4M3), ("bf8", BF8E5M2))
    ],
    ("gfx950", "mfma_f32_16x16x128_fp4", VectorType(I8, 8), 4, 64),
    ("gfx950", "mfma_f32_16x16x96_fp6", VectorType(I8, 12), 4, 96),
    ("gfx950", "mfma_f32_16x16x128_fp8", VectorType(FP8E4M3, 32), 4, 256),
    ("gfx950", "mfma_scale_f32_16x16x128_f8f6f4", VectorType(I32, 8), 4, 256),
]
FLAVORS = ["llvm20", "llvm22", "llvm23"]
ARITIES = ["missing_operand", "extra_operand", "missing_result", "extra_result"]
TF32_SOURCE_ERROR = (
    "XF32 MMA requires two vec<tf32x2> operands and its FP32 accumulator"
)
TF32_RESULT_ERROR = "XF32 MMA result must match its FP32 accumulator"


def _kernel(case, *, concrete=False, atom=None, source=None, use_result=False):
    _arch, op_id, ty, count, _ = case
    b = IRBuilder("mfma_operand_contract")
    a = b.param("a", ty if source is None else source)
    bb = b.param("b", ty if source is None else source)
    c = b.param("c", VectorType(F32, count))
    extra = (
        [b.param("a_scale", I32), b.param("b_scale", I32)]
        if op_id.startswith("mfma_scale_")
        else []
    )
    d = b.mma(op_id if atom is None else atom, a, bb, c, *extra)
    op = b.kernel.body.ops[-1]
    if concrete:
        op.name = f"tile.{op.attrs.pop('op_id')}"
    if use_result:
        out = b.param("out", PtrType(F32, "global"))
        b.global_store(out, b.const_i32(0), b.vec_extract(d, count - 1))
    b.ret()
    return b.kernel, op


def _assert_rejects(kernel, case, flavor, error, engine):
    ir = serialize(kernel)
    if engine == "native":
        native = pytest.importorskip("rocke_engine")
        with pytest.raises(RuntimeError) as caught:
            native.lower_serialized_ir(ir, arch=case[0], flavor=flavor)
        assert str(caught.value) == (
            f"rocke_engine.lower_serialized_ir: lower failed for arch '{case[0]}' "
            f"(status 1): {error}"
        )
        return
    for candidate in (kernel, parse(ir)):
        with pytest.raises(ValueError) as caught:
            if engine == "hip":
                lower_kernel_to_hip(candidate, arch=case[0])
            else:
                _lower_kernel_to_llvm_python(
                    candidate, arch=case[0], llvm_flavor=flavor
                )
        assert str(caught.value) == error


def _corrupt_type(kernel, op, role, ty):
    if role == "dst":
        op.result.type = ty
    else:
        index = int(role.removeprefix("src"))
        op.operands[index].type = ty
        name = ("a", "b", "c", "a_scale", "b_scale")[index]
        next(p for p in kernel.params if p.name == name).type = ty


def _corrupt_arity(op, kind):
    if kind == "missing_operand":
        op.operands.pop()
    elif kind == "extra_operand":
        op.operands.append(op.operands[0])
    elif kind == "missing_result":
        op.results.clear()
    else:
        op.results.append(Value("%extra", op.result.type))


def _type_error(case, role):
    if "xf32" in case[1]:
        return TF32_RESULT_ERROR if role == "dst" else TF32_SOURCE_ERROR
    if role in ("src2", "dst"):
        return f"MFMA requires src2 and dst to be vec<f32x{case[3]}>"
    if case[4] is not None:
        return f"MFMA requires src0 and src1 to have {case[4]} bits"
    source = (
        VectorType(I8, 8)
        if isinstance(case[2], VectorType) and case[2].elem in (FP8E4M3, BF8E5M2)
        else case[2]
    )
    return f"MFMA requires src0 and src1 to be {source.name}"


def _bad_type(case, role, kind):
    original = case[2] if role in ("src0", "src1") else VectorType(F32, case[3])
    elem = original.elem if isinstance(original, VectorType) else original
    count = original.count if isinstance(original, VectorType) else 1
    if kind == "dtype":
        # Same-sized integer carrier is invalid for typed ordinary arithmetic.
        return VectorType(I32 if elem != I32 else F32, count)
    if kind == "scalar":
        return F32 if case[4] is not None or original == F32 else elem
    if kind == "pointer":
        return PtrType(I8, "global")
    return VectorType(elem, max(1, count - 1) if kind == "short" else count + 1)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_mfma_valid_llvm_signature_and_bytes(case, flavor, engine):
    kernel, _ = _kernel(case, use_result=True)
    expected = _lower_kernel_to_llvm_python(kernel, arch=case[0], llvm_flavor=flavor)
    assert f"call <{case[3]} x float> @llvm.amdgcn.mfma." in expected
    assert f"extractelement <{case[3]} x float>" in expected
    if case[2] == F32:
        assert "float %a, float %b" in expected
    elif isinstance(case[2], VectorType) and case[2].elem == F16:
        assert f"<{case[2].count} x half> %a" in expected
        assert f"<{case[2].count} x half> %b" in expected
    elif isinstance(case[2], VectorType) and case[2].elem == BF16:
        assert f"<{case[2].count} x bfloat> %a" in expected
        assert f"<{case[2].count} x bfloat> %b" in expected
    for concrete in (False, True):
        candidate, _ = _kernel(case, concrete=concrete, use_result=True)
        assert (
            _lower_kernel_to_llvm_python(
                parse(serialize(candidate)), arch=case[0], llvm_flavor=flavor
            )
            == expected
        )
    if engine == "native":
        native = pytest.importorskip("rocke_engine")
        assert (
            native.lower_serialized_ir(serialize(kernel), arch=case[0], flavor=flavor)
            == expected
        )
    else:
        assert (
            _lower_kernel_to_llvm_python(
                parse(serialize(kernel)), arch=case[0], llvm_flavor=flavor
            )
            == expected
        )


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("role", ["src0", "src1", "src2", "dst"])
@pytest.mark.parametrize("kind", ["short", "long", "dtype", "scalar", "pointer"])
@pytest.mark.parametrize(
    "engine,concrete", [("python", False), ("python", True), ("native", False)]
)
def test_mfma_llvm_rejects_actual_ssa_types(case, flavor, concrete, role, kind, engine):
    # Some packed dtype/scalar alternatives are legitimate equal-width carriers.
    if role in ("src0", "src1") and case[4] is not None and kind == "dtype":
        return
    if role in ("src0", "src1") and case[2] == F32 and kind in ("scalar", "short"):
        return
    kernel, op = _kernel(case, concrete=concrete)
    _corrupt_type(kernel, op, role, _bad_type(case, role, kind))
    _assert_rejects(kernel, case, flavor, _type_error(case, role), engine)


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("kind", ARITIES)
@pytest.mark.parametrize(
    "engine,concrete", [("python", False), ("python", True), ("native", False)]
)
def test_mfma_llvm_rejects_arity(case, flavor, concrete, kind, engine):
    kernel, op = _kernel(case, concrete=concrete)
    _corrupt_arity(op, kind)
    error = (
        TF32_SOURCE_ERROR
        if "xf32" in case[1] and "operand" in kind
        else (
            TF32_RESULT_ERROR
            if "xf32" in case[1]
            else f"MFMA expects {5 if case[1].startswith('mfma_scale_') else 3} operands and 1 result"
        )
    )
    _assert_rejects(kernel, case, flavor, error, engine)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("engine", ["python", "native", "hip"])
def test_mfma_custom_destination_keeps_authoring_metadata_independent(flavor, engine):
    case = CASES[0]
    atom = ArchTarget.from_gfx(case[0]).mma.by_op_id(case[1])
    custom = replace(atom, dst=replace(atom.dst, dtype="i32", frag_len=7))
    kernel, op = _kernel(case, atom=custom)
    assert op.operands[2].type == VectorType(F32, 4)
    assert op.result.type == VectorType(I32, 7)
    _assert_rejects(kernel, case, flavor, _type_error(case, "dst"), engine)


# Python HIP has no concrete XF32 handler or wide fp8 hero handler. Keep that
# unsupported surface explicit while exercising every implemented HIP sibling.
HIP_CASES = [c for c in CASES if c[1] != "mfma_f32_16x16x128_fp8"]


@pytest.mark.parametrize("case", HIP_CASES, ids=[c[1] for c in HIP_CASES])
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize("role", ["src0", "src1", "src2", "dst"])
@pytest.mark.parametrize(
    "kind", ["valid", "short", "long", "dtype", "scalar", "pointer", *ARITIES]
)
def test_mfma_supported_hip_handlers(case, concrete, role, kind):
    if concrete and "xf32" in case[1]:
        return
    if role in ("src0", "src1") and (
        (case[4] is not None and kind == "dtype")
        or (case[2] == F32 and kind in ("scalar", "short"))
    ):
        return
    kernel, op = _kernel(case, concrete=concrete, use_result=kind == "valid")
    if kind == "valid":
        assert lower_kernel_to_hip(kernel, arch=case[0]) == lower_kernel_to_hip(
            parse(serialize(kernel)), arch=case[0]
        )
        return
    if kind in ARITIES:
        _corrupt_arity(op, kind)
        error = (
            TF32_SOURCE_ERROR
            if "xf32" in case[1] and "operand" in kind
            else (
                TF32_RESULT_ERROR
                if "xf32" in case[1]
                else f"MFMA expects {5 if case[1].startswith('mfma_scale_') else 3} operands and 1 result"
            )
        )
    else:
        _corrupt_type(kernel, op, role, _bad_type(case, role, kind))
        error = _type_error(case, role)
    _assert_rejects(kernel, case, "llvm23", error, "hip")


PACKED_CARRIERS = (
    [
        (CASES[-4], ty)
        for ty in (
            I64,
            F64,
            VectorType(I32, 2),
            VectorType(I16, 4),
            VectorType(F32, 2),
            VectorType(FP8E4M3, 8),
            VectorType(U8, 8),
        )
    ]
    + [
        (CASES[-3], ty)
        for ty in (
            VectorType(I32, 3),
            VectorType(I16, 6),
            VectorType(BF8E5M2, 12),
            VectorType(U8, 12),
        )
    ]
    + [
        (case, ty)
        for case in CASES[-2:]
        for ty in (
            VectorType(I64, 4),
            VectorType(F64, 4),
            VectorType(I32, 8),
            VectorType(I8, 32),
            VectorType(U8, 32),
            VectorType(F16, 16),
        )
    ]
)


@pytest.mark.parametrize("case,carrier", PACKED_CARRIERS)
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_mfma_preserves_equal_width_packed_carriers(case, carrier, flavor, engine):
    kernel, _ = _kernel(case, source=carrier, use_result=True)
    expected = _lower_kernel_to_llvm_python(kernel, arch=case[0], llvm_flavor=flavor)
    if engine == "native":
        native = pytest.importorskip("rocke_engine")
        assert (
            native.lower_serialized_ir(serialize(kernel), arch=case[0], flavor=flavor)
            == expected
        )
    else:
        assert (
            _lower_kernel_to_llvm_python(
                parse(serialize(kernel)), arch=case[0], llvm_flavor=flavor
            )
            == expected
        )


@pytest.mark.parametrize("case", [*CASES[13:17], *CASES[-4:]])
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize(
    "engine,concrete", [("python", False), ("python", True), ("native", False)]
)
def test_mfma_unsigned_byte_carriers_match_signed_bytes(case, flavor, concrete, engine):
    count = case[4] // 8 if case[4] else 8
    unsigned, _ = _kernel(case, source=VectorType(U8, count), concrete=concrete)
    signed, _ = _kernel(case, source=VectorType(I8, count), concrete=concrete)
    expected = _lower_kernel_to_llvm_python(signed, arch=case[0], llvm_flavor=flavor)
    if engine == "native":
        native = pytest.importorskip("rocke_engine")
        actual = native.lower_serialized_ir(
            serialize(unsigned), arch=case[0], flavor=flavor
        )
    else:
        actual = _lower_kernel_to_llvm_python(
            unsigned, arch=case[0], llvm_flavor=flavor
        )
        assert actual == _lower_kernel_to_llvm_python(
            parse(serialize(unsigned)), arch=case[0], llvm_flavor=flavor
        )
    assert actual == expected


@pytest.mark.parametrize("case", [*CASES[13:17], CASES[-4], CASES[-3], CASES[-1]])
@pytest.mark.parametrize("concrete", [False, True])
def test_mfma_unsigned_byte_carriers_preserve_unsupported_hip(case, concrete):
    count = case[4] // 8 if case[4] else 8
    kernel, _ = _kernel(case, source=VectorType(U8, count), concrete=concrete)
    # U8 vectors already lower to i8 in LLVM, but have no HIP type mapping.
    for candidate in (kernel, parse(serialize(kernel))):
        with pytest.raises(
            NotImplementedError, match="no HIP mapping for type vec<u8x"
        ):
            lower_kernel_to_hip(candidate, arch=case[0])


@pytest.mark.parametrize("case", CASES[-1:])
@pytest.mark.parametrize("role", ["src3", "src4"])
@pytest.mark.parametrize("ty", [F32, I64, VectorType(I32, 1), VectorType(I8, 4)])
@pytest.mark.parametrize(
    "engine,concrete",
    [
        ("python", False),
        ("python", True),
        ("native", False),
        ("hip", False),
        ("hip", True),
    ],
)
def test_mfma_scales_require_actual_i32(case, role, ty, concrete, engine):
    kernel, op = _kernel(case, concrete=concrete)
    _corrupt_type(kernel, op, role, ty)
    _assert_rejects(
        kernel, case, "llvm23", "MFMA requires scale operands to be i32", engine
    )


@pytest.mark.parametrize("case", CASES, ids=[c[1] for c in CASES])
def test_mfma_rejects_before_intrinsic_declarations_or_instructions(case):
    kernel, op = _kernel(case, concrete=True)
    op.result.type = VectorType(I32, 7)
    lowerer = _Lowerer(kernel, arch=case[0], llvm_flavor="llvm23")
    with pytest.raises(ValueError) as caught:
        lowerer.lower_op(op)
    assert str(caught.value) == _type_error(case, "dst")
    assert lowerer._needs_intrin == {}
    assert all(block.lines == [] for block in lowerer._blocks)


def test_mfma_preserves_unsupported_hip_handlers():
    kernel, _ = _kernel(CASES[-2])
    with pytest.raises(NotImplementedError, match="no HIP lowering"):
        lower_kernel_to_hip(kernel, arch="gfx950")
