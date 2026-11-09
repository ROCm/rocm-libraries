# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value barriers preserve types and survive serialization through both engines."""

from pathlib import Path
import runpy

import pytest

from rocke.core.backend import _cpp_strict, resolve_backend
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
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
)


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


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("clobber", [None, ""])
def test_hip_barrier_valid_clobber(arch, roundtrip, clobber):
    b = IRBuilder("barrier_clobber")
    result = b.optimization_barrier(b.param("value", F32))
    baseline = lower_kernel_to_hip(b.kernel, arch=arch)
    if clobber is not None:
        result.op.attrs["clobber"] = clobber
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    assert lower_kernel_to_hip(kernel, arch=arch) == baseline
    assert 'asm ("" : "=v"' in baseline
    assert "asm volatile" not in baseline


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("clobber", [False, True, 0, 1, 0.0, 1.0, "memory"])
def test_hip_barrier_rejects_invalid_clobber(roundtrip, clobber):
    b = IRBuilder("invalid_barrier_clobber")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs["clobber"] = clobber
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    with pytest.raises(ValueError, match="clobber must be an empty string"):
        lower_kernel_to_hip(kernel, arch="gfx1250")


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("sideeffect", [0, 1, 0.0, 1.0, "", "false"])
def test_hip_inline_asm_rejects_non_bool_sideeffect(roundtrip, sideeffect):
    b = IRBuilder("invalid_asm_sideeffect")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs["sideeffect"] = sideeffect
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    with pytest.raises(ValueError, match="sideeffect must be a boolean"):
        lower_kernel_to_hip(kernel, arch="gfx1250")


@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("sideeffect", [None, False, True])
@pytest.mark.parametrize("template", ["", "v_mov_b32 $0, $1"])
def test_hip_inline_asm_preserves_bool_and_default_sideeffect(
    roundtrip, sideeffect, template
):
    b = IRBuilder("asm_sideeffect")
    result = b.inline_asm(template, "=v,0", [b.param("value", F32)], F32)
    if sideeffect is None:
        del result.op.attrs["sideeffect"]
    else:
        result.op.attrs["sideeffect"] = sideeffect
    kernel = parse(serialize(b.kernel)) if roundtrip else b.kernel
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert ("asm volatile" in hip) == (sideeffect is not False)
    assert ('"memory"' in hip) == (sideeffect is not False)
    if template:
        assert "v_mov_b32 %0, %1" in hip


def test_hip_general_pure_asm_preserves_explicit_memory_clobber():
    b = IRBuilder("asm_clobber")
    result = b.inline_asm(
        "v_mov_b32 $0, $1", "=v,0", [b.param("value", F32)], F32, sideeffect=False
    )
    result.op.attrs["clobber"] = "memory"
    hip = lower_kernel_to_hip(parse(serialize(b.kernel)), arch="gfx1250")
    assert 'asm ("v_mov_b32 %0, %1"' in hip
    assert '"memory"' in hip


@pytest.fixture
def barrier_emitter():
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "instances"
            / "parity"
            / "optimization_barrier_emit.py"
        )
    )


@pytest.mark.parametrize("index", range(8))
def test_hip_barrier_paired_emitters(index, barrier_emitter):
    arch, dtype, clobber = barrier_emitter["HIP_CONFIGS"][index]
    kernel = barrier_emitter["build_hip_epilogue"](dtype, clobber)
    source = lower_hip_with_both_engines(kernel, arch)
    assert "factor_a * factor_b" in source
    assert 'asm ("" : "=v"' in source
    assert "asm volatile" not in source
    assert ' : "memory"' not in source


@pytest.mark.parametrize(
    "field,value",
    [("clobber", v) for v in (False, True, 0, 1, 0.0, 1.0, "memory")]
    + [("sideeffect", v) for v in (0, 1, 0.0, 1.0, "", "false")],
)
def test_hip_native_rejects_malformed_barrier_attributes(field, value):
    b = IRBuilder("invalid_native_asm")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs[field] = value
    diagnostic = (
        "HIP optimization barrier clobber must be an empty string"
        if field == "clobber"
        else "HIP inline asm sideeffect must be a boolean"
    )
    assert_both_engines_reject_hip(b.kernel, "gfx1250", ValueError, diagnostic)


@pytest.mark.parametrize("sideeffect", [None, False, True])
@pytest.mark.parametrize("output", [None, F32])
def test_hip_general_assembly_matches_native(sideeffect, output):
    b = IRBuilder("general_asm")
    value = b.param("value", F32)
    b.inline_asm(
        "v_mov_b32 $0, $1" if output else "s_nop 0",
        "=v,0" if output else "",
        [value] if output else [],
        output,
    )
    op = b.kernel.body.ops[-1]
    if sideeffect is None:
        del op.attrs["sideeffect"]
    else:
        op.attrs["sideeffect"] = sideeffect
    op.attrs["clobber"] = "memory"
    lower_hip_with_both_engines(b.kernel, "gfx1250")


@pytest.mark.parametrize("clobber", [False, True, 0, 1, 0.0, 1.0])
def test_hip_general_assembly_rejects_non_string_clobber(clobber):
    b = IRBuilder("invalid_general_clobber")
    result = b.inline_asm("v_mov_b32 $0, $1", "=v,0", [b.param("value", F32)], F32)
    result.op.attrs["clobber"] = clobber
    assert_both_engines_reject_hip(
        b.kernel, "gfx1250", ValueError, "HIP inline asm clobber must be a string"
    )


@pytest.mark.parametrize("index", range(4))
def test_hip_assembly_paired_emitters(index, barrier_emitter):
    arch, output = barrier_emitter["HIP_STRING_CONFIGS"][index]
    kernel = barrier_emitter["build_hip_assembly"](output)
    source = lower_hip_with_both_engines(kernel, arch)
    assert r"\n\ts_nop 0 ; \"quoted\" \\path" in source


def _constraint_diagnostic(entry):
    return (
        f'HIP inline asm constraint failed - constraint="{entry}", '
        "expected a supported output, input, tie, or clobber"
    )


_HIP_ASM_INVALID_CONSTRAINTS = [
    ("=v", 1, 1, "HIP inline asm inputs failed - constraints=0, expected operands=1"),
    (
        "=v,0,v",
        1,
        1,
        "HIP inline asm inputs failed - constraints=2, expected operands=1",
    ),
    ("", 0, 1, "HIP inline asm inputs failed - constraints=0, expected operands=1"),
    ("v", 0, 0, "HIP inline asm inputs failed - constraints=1, expected operands=0"),
    ("=v", 0, 0, "HIP inline asm outputs failed - constraints=1, expected results=0"),
    (
        "=v,=s",
        1,
        0,
        "HIP inline asm outputs failed - constraints=2, expected results=1",
    ),
    ("v", 1, 1, "HIP inline asm requires an output constraint"),
    ("", 1, 0, "HIP inline asm requires an output constraint"),
    ("=v,,0", 1, 1, _constraint_diagnostic("")),
    ("=v,0,", 1, 1, _constraint_diagnostic("")),
    (", ", 0, 0, _constraint_diagnostic("")),
    ("=v,~m0", 1, 0, _constraint_diagnostic("~m0")),
    ("~{m0", 0, 0, _constraint_diagnostic("~{m0")),
    ("~{}", 0, 0, _constraint_diagnostic("~{}")),
    ("~{s0}", 0, 0, _constraint_diagnostic("~{s0}")),
    ("~{memory},v", 0, 1, _constraint_diagnostic("v")),
    ("v,=v", 1, 1, _constraint_diagnostic("=v")),
    ("0", 0, 1, _constraint_diagnostic("0")),
    ("=v,1", 1, 1, _constraint_diagnostic("1")),
    ("+v", 1, 0, _constraint_diagnostic("+v")),
    ("=r", 1, 0, _constraint_diagnostic("=r")),
    ("=v,0,0", 1, 2, _constraint_diagnostic("0")),
]


def _asm_constraint_kernel(constraints, outputs, operands):
    b = IRBuilder("asm_constraints")
    values = [b.param(f"value{index}", F32) for index in range(operands)]
    b.inline_asm("s_nop 0", "=v" if outputs else "", values, F32 if outputs else None)
    b.kernel.body.ops[-1].attrs["constraints"] = constraints
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "constraints,outputs,operands,diagnostic", _HIP_ASM_INVALID_CONSTRAINTS
)
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_inline_asm_constraint_admission(
    constraints, outputs, operands, diagnostic, roundtrip
):
    kernel = _asm_constraint_kernel(constraints, outputs, operands)
    if roundtrip:
        kernel = parse(serialize(kernel))
    with pytest.raises(ValueError) as error:
        lower_kernel_to_hip(kernel, arch="gfx950")
    assert str(error.value) == diagnostic


@pytest.mark.parametrize(
    "constraints,outputs,operands,diagnostic", _HIP_ASM_INVALID_CONSTRAINTS
)
@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_inline_asm_native_constraint_rejection(
    constraints, outputs, operands, diagnostic, arch, roundtrip
):
    kernel = _asm_constraint_kernel(constraints, outputs, operands)
    if roundtrip:
        kernel = parse(serialize(kernel))
    assert_both_engines_reject_hip(kernel, arch, ValueError, diagnostic)


@pytest.mark.parametrize("constraints", [False, True, 0, 1.0])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_inline_asm_constraint_type(constraints, roundtrip):
    kernel = _asm_constraint_kernel(constraints, 0, 0)
    if roundtrip:
        kernel = parse(serialize(kernel))
    diagnostic = "HIP inline asm constraints must be a string"
    with pytest.raises(TypeError, match=diagnostic):
        lower_kernel_to_hip(kernel, arch="gfx950")


@pytest.mark.parametrize("constraints", [False, True, 0, 1.0])
@pytest.mark.parametrize("clobber", [None, False])
def test_hip_inline_asm_native_constraint_type(constraints, clobber):
    kernel = _asm_constraint_kernel(constraints, 0, 0)
    if clobber is not None:
        kernel.body.ops[0].attrs["clobber"] = clobber
    assert_both_engines_reject_hip(
        kernel,
        "gfx950",
        TypeError,
        "HIP inline asm constraints must be a string",
    )


@pytest.mark.parametrize("kind", ["void", "output", "ring"])
@pytest.mark.parametrize("sideeffect", [False, True])
@pytest.mark.parametrize("explicit", [None, "", "memory", "m0"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_inline_asm_clobbers_preserved(
    kind, sideeffect, explicit, roundtrip, barrier_emitter
):
    kernel = barrier_emitter["build_hip_clobbers"](kind)
    op = next(op for op in kernel.body.ops if op.name == "tile.inline_asm")
    op.attrs["sideeffect"] = sideeffect
    if explicit is not None:
        op.attrs["clobber"] = explicit
    if roundtrip:
        kernel = parse(serialize(kernel))
    source = lower_kernel_to_hip(kernel, arch="gfx950")
    clause = next(
        line for line in source.splitlines() if line.strip().startswith("asm ")
    )
    assert clause.endswith(' : "m0", "memory");')
    assert ("asm volatile" in clause) == sideeffect
    for operand in op.operands:
        assert f"({operand.name.lstrip('%')})" in clause


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_clobber_paired_emitters(index, roundtrip, barrier_emitter):
    arch, kind = barrier_emitter["HIP_CLOBBER_CONFIGS"][index]
    kernel = barrier_emitter["build_hip_clobbers"](kind)
    if roundtrip:
        kernel = parse(serialize(kernel))
    source = lower_hip_with_both_engines(kernel, arch)
    assert ' : "m0", "memory");' in source


@pytest.mark.parametrize("clobber", ["memory", "m0", "scc", "vcc", "exec"])
@pytest.mark.parametrize("output", [False, True])
def test_hip_inline_asm_clobber_deduplication(clobber, output):
    prefix = "=v,0," if output else ""
    kernel = _asm_constraint_kernel(
        prefix + f"~{{{clobber}}},~{{{clobber}}}", int(output), int(output)
    )
    op = kernel.body.ops[0]
    op.attrs["sideeffect"] = False
    op.attrs["clobber"] = clobber
    source = lower_hip_with_both_engines(kernel, "gfx950")
    clause = next(
        line for line in source.splitlines() if line.strip().startswith("asm ")
    )
    assert clause.count(f'"{clobber}"') == 1


@pytest.mark.parametrize("clobber", ["s0", "m0,memory", 'm0"', "~{m0}"])
def test_hip_inline_asm_rejects_unsupported_explicit_clobber(clobber):
    kernel = _asm_constraint_kernel("", 0, 0)
    kernel.body.ops[0].attrs["clobber"] = clobber
    escaped = clobber.replace('"', '\\"')
    assert_both_engines_reject_hip(
        kernel, "gfx950", ValueError, _constraint_diagnostic(escaped)
    )


_HIP_ASM_STRING_CASES = [
    ("\n\ts_nop 0", r"\n\ts_nop 0"),
    (' ; "quoted" \\path', r" ; \"quoted\" \\path"),
    ("\r\x017\x7f", r"\r\0017\177"),
]


def _asm_string_kernel(suffix, output):
    b = IRBuilder("asm_string")
    prefix = "v_mov_b32 $0, $1" if output else "s_nop 0"
    value = b.param("value", F32) if output else None
    b.inline_asm(
        prefix + suffix, "=v,0" if output else "", [value] if output else [], output
    )
    b.ret()
    return b.kernel


@pytest.mark.parametrize("suffix,escaped", _HIP_ASM_STRING_CASES)
@pytest.mark.parametrize("output", [None, F32])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_hip_assembly_string_literal(suffix, escaped, output, roundtrip):
    kernel = _asm_string_kernel(suffix, output)
    if roundtrip:
        kernel = parse(serialize(kernel))
    source = lower_kernel_to_hip(kernel, arch="gfx950")
    prefix = "v_mov_b32 %0, %1" if output else "s_nop 0"
    assert f'asm volatile ("{prefix}{escaped}"' in source


@pytest.mark.parametrize("suffix,escaped", _HIP_ASM_STRING_CASES)
@pytest.mark.parametrize("output", [None, F32])
def test_hip_assembly_string_matches_native(suffix, escaped, output):
    kernel = _asm_string_kernel(suffix, output)
    source = lower_hip_with_both_engines(kernel, "gfx950")
    prefix = "v_mov_b32 %0, %1" if output else "s_nop 0"
    assert f'asm volatile ("{prefix}{escaped}"' in source


def _sideeffect_kernel(sideeffect, outputs):
    b = IRBuilder("asm_sideeffect")
    value = b.param("value", F32)
    b.inline_asm_multi(
        "" if outputs else "s_nop 0",
        ("=v,0" if outputs == 1 else "=v,=v,0,1") if outputs else "",
        [value] * outputs,
        result_types=[F32] * outputs,
    )
    op = b.kernel.body.ops[-1]
    if sideeffect is None:
        del op.attrs["sideeffect"]
    else:
        op.attrs["sideeffect"] = sideeffect
    b.ret()
    return b.kernel


@pytest.mark.parametrize("sideeffect", [0, 1, 0.0, 1.0, "", "false"])
@pytest.mark.parametrize("outputs", [0, 1, 2])
@pytest.mark.parametrize("roundtrip", [False, True])
@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23", "llvm24"])
def test_llvm_inline_asm_rejects_non_bool_sideeffect(
    sideeffect, outputs, roundtrip, arch, flavor
):
    kernel = _sideeffect_kernel(sideeffect, outputs)
    if roundtrip:
        kernel = parse(serialize(kernel))
    exception = (
        RuntimeError if resolve_backend() == "cpp" and _cpp_strict() else ValueError
    )
    with pytest.raises(exception, match="LLVM inline asm sideeffect must be a boolean"):
        lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=flavor)


@pytest.mark.parametrize("sideeffect", [0, 1, 0.0, 1.0, "", "false"])
@pytest.mark.parametrize("outputs", [0, 1, 2])
@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23", "llvm24"])
def test_llvm_native_rejects_non_bool_sideeffect(sideeffect, outputs, arch, flavor):
    kernel = _sideeffect_kernel(sideeffect, outputs)
    assert_both_engines_reject(
        kernel, arch, ValueError, "LLVM inline asm sideeffect must be a boolean", flavor
    )


@pytest.mark.parametrize("sideeffect", [None, False, True])
@pytest.mark.parametrize("outputs", [0, 1, 2])
@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23", "llvm24"])
def test_llvm_inline_asm_preserves_boolean_and_default(
    sideeffect, outputs, arch, flavor
):
    kernel = _sideeffect_kernel(sideeffect, outputs)
    llvm = lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=flavor)
    assert ("asm sideeffect" in llvm) == (sideeffect is not False)
    assert llvm == lower_kernel_to_llvm(
        parse(serialize(kernel)), arch=arch, llvm_flavor=flavor
    )


def test_hip_assembly_requires_output_constraint():
    b = IRBuilder("missing_output_constraint")
    result = b.inline_asm("", "=v,0", [b.param("value", F32)], F32)
    result.op.attrs["constraints"] = " , "
    assert_both_engines_reject_hip(
        b.kernel, "gfx1250", ValueError, "HIP inline asm requires an output constraint"
    )
