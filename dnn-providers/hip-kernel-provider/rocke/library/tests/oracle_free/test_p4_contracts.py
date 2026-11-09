# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the structured-logit (P4) oracle-free check.

The P4 prediction is derived from a pinned model of the dense kernel's
floating-point path: the score scale, the masked value, ``v_exp_f32``, the FP32
denominator reduction, one cast of the probabilities, the PV MFMA, an IEEE
reciprocal and one output cast. The IR contract here lowers the two quick specs
to LLVM IR on the CPU and classifies every floating-point instruction against
that model. Any instruction outside it fails with "re-derive the path model":
the prediction is re-derived for the new path, never widened to absorb it.

The IR is not the whole story: affected AMDGPU compilers select some FP16
output casts as single-rounding mixed-precision FMAs (assumption A-F). The ISA
contract disassembles the quick code objects and checks that this epilogue is
the only fused site.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import json
import re
import struct
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from rocke.numeric.bf16 import f32_to_bf16

from ._p4_shared_analytical import (
    UNIFORM_CALIBRATION,
    UNIFORM_QUICK_CASES,
    UniformCase,
    UniformObservation,
    encode_bits,
    legal_interval,
    target_bits,
    uniform_expected_bits,
)
from ._p4_shared_subject import (
    _resolve_tuning_spec,
    _shape,
    _validate_supported_case,
    resolve_dense_subject,
    run_dense_case,
)
from ._p4_analytical import (
    EPILOGUE_ASSUMPTION,
    F32_SMALLEST_SUBNORMAL,
    LOG2E_F32,
    MASKED_SCORE,
    P4_ASSUMPTIONS,
    P4_CALIBRATION,
    P4_EPILOGUE_ARTIFACTS,
    P4_EXPOSURES,
    P4_FAILURE_LABELS,
    P4_HEAD_LOGITS,
    P4_HEAD_SCALE_MARGIN,
    P4_HEAD_SCALES,
    P4_POLICY,
    P4_QUICK_CASES,
    P4_SEARCH_BASES,
    P4_SEPARATION_ROWS,
    P4_UNSEPARATED,
    SCORE_SCALE,
    SEPARATION_LABELS,
    EmulationMode,
    F32Interval,
    P4CalibrationEntry,
    P4Case,
    P4Fixture,
    P4Label,
    P4Mutation,
    add_faithful,
    add_rn,
    analyze_p4_policy,
    applicable_mutations,
    cast_bits_hull,
    cast_single_or_double_hull,
    certified_target_bits,
    coordinate_class,
    emulate_row,
    epilogue_alternatives,
    epilogue_artifact_for,
    exp2_candidates,
    f32,
    f32_quantum,
    format_p4_result,
    format_separation,
    gamma,
    head_scale_margin,
    is_f32,
    mul_rn,
    mutation_separation,
    next_down,
    next_up,
    p4_diagnostics,
    p4_inputs,
    p4_query_key,
    p4_records,
    rcp_rn,
    require_p4_calibration,
    round_f32,
    row_logit,
    row_trace,
    score,
    score_numerator,
    sum_any_order,
    target_value,
    tier_a_exposure,
    tier_b_exposure,
    tier_b_row,
    visited_tiles,
)
from ._p4_capture_audit import audit_captures, format_audit

_REDERIVE = "re-derive the P4 path model"

# LLVM spells FP32 constants as the hex of the equivalent double.
_F32_LOG2E = "0x3FF7154760000000"
_F32_ONE = "0x3FF0000000000000"
_F32_ZERO = "0x0000000000000000"
_F32_MASKED = "0xC6293E5940000000"  # -1e30, the masked score and initial row max

_MFMA_ATOM = {
    "fp16": "llvm.amdgcn.mfma.f32.32x32x8f16",
    "bf16": "llvm.amdgcn.mfma.f32.32x32x8bf16.1k",
}
_CAST_TYPE = {"fp16": "half", "bf16": "bfloat"}
_EXP2 = "llvm.amdgcn.exp2.f32"
_MAXNUM = "llvm.maxnum.f32"
_BPERMUTE = "llvm.amdgcn.ds.bpermute"

_FP_FLAGS = ("fast", "contract", "afn", "reassoc", "nnan", "ninf", "nsz", "arcp")
_FP_OPCODES = ("fadd", "fsub", "fmul", "fdiv", "frem", "fneg", "fptrunc", "fpext")
_FP_TYPE = re.compile(r"\b(float|half|bfloat|double)\b")
_FP_MODE_ATTRIBUTES = (
    "denormal-fp-math",
    "unsafe-fp-math",
    "no-nans-fp-math",
    "no-infs-fp-math",
    "approx-func-fp-math",
    "no-signed-zeros-fp-math",
)

_INSTRUCTION = re.compile(r"^\s*(%[-\w.]+) = (.+)$")
_VALUE = re.compile(r"%[-\w.]+")
_CALLEE = re.compile(r"@([-\w.]+)\(")
_DECLARE = re.compile(r"^declare (.+?) @([-\w.]+)\((.*)\)")
_NO_OP_BITCAST = re.compile(r"bitcast (\S+(?: x \S+>)?) %[-\w.]+ to (\S+(?: x \S+>)?)$")


@dataclass(frozen=True)
class _Instruction:
    name: str
    opcode: str
    text: str
    operands: tuple[str, ...]


def _parse(ir: str) -> dict[str, _Instruction]:
    """Map each SSA name to its defining instruction.

    ``opcode`` is the first word, or the callee for a call. ``operands`` lists
    the SSA values the instruction reads, in order, excluding the callee."""
    instructions: dict[str, _Instruction] = {}
    for line in ir.splitlines():
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        name, text = match.groups()
        if re.match(r"(?:tail |musttail |notail )?call ", text):
            callee = _CALLEE.search(text)
            opcode = callee.group(1) if callee else "call"
            arguments = text[callee.end() :] if callee else text
        else:
            opcode = text.split()[0]
            arguments = text[len(opcode) :]
        instructions[name] = _Instruction(
            name, opcode, text, tuple(_VALUE.findall(arguments))
        )
    return instructions


@dataclass
class PathModel:
    """Classified floating-point steps of one lowered kernel."""

    violations: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    chain_lengths: list[int] = field(default_factory=list)


def classify_dense_path(ir: str, dtype: str) -> PathModel:
    """Classify every floating-point instruction against the pinned path model.

    The model, per visited KV tile and row (gfx942 dense, fast exp2):

    - ``qk_scale = fmul(scale, LOG2E)`` once; ``s = fmul(acc, qk_scale)`` with
      ``acc`` an element of a QK MFMA result; masking selects ``-1e30``.
    - ``m_new = maxnum(...)``; every ``exp2`` reads ``fsub(x, m_new)``.
      A probability ``exp2`` feeds one denominator ``fadd`` and one ``fptrunc``
      to the subject dtype; a rescale ``exp2`` (``alpha``) feeds ``fmul`` only.
    - The lane denominator is a sequential ``fadd`` chain starting at ``+0``,
      then one ``fadd`` with the shuffle-xor partner's sum (``ds.bpermute``),
      then ``l = fadd(fmul(l_i, alpha), l_tile)``.
    - The accumulator rescale is ``fmul(o, alpha)``; PV is the dtype's
      32x32x8 MFMA with ``cbsz = abid = blgp = 0``.
    - The epilogue is one ``fdiv 1.0, l`` and ``fptrunc(fmul(o, rcp))``.

    No floating-point instruction may carry fast-math flags or ``!fpmath``, and
    the function may set no floating-point mode attribute. This pins the IR
    only; what an affected compiler selects for the FP16 epilogue is A-F and
    the ISA contract below."""
    model = PathModel()
    violations = model.violations
    counts = model.counts
    instructions = _parse(ir)
    uses: dict[str, list[_Instruction]] = {}
    for instruction in instructions.values():
        for operand in instruction.operands:
            uses.setdefault(operand, []).append(instruction)

    def source(name: str) -> str:
        """Follow no-op ``bitcast float -> float`` (and vector) copies."""
        while name in instructions:
            item = instructions[name]
            no_op = _NO_OP_BITCAST.match(item.text)
            if not (no_op and no_op.group(1) == no_op.group(2)):
                break
            name = item.operands[0]
        return name

    def opcode(name: str) -> str | None:
        item = instructions.get(source(name))
        return item.opcode if item else None

    def users(name: str) -> list[_Instruction]:
        """Uses of ``name``, looking through no-op copies."""
        found: list[_Instruction] = []
        for user in uses.get(name, []):
            if user.opcode == "bitcast" and source(user.name) == source(name):
                found.extend(users(user.name))
            else:
                found.append(user)
        return found

    # Fast-math flags, !fpmath and FP mode attributes.
    for line in ir.splitlines():
        stripped = line.strip()
        for op in (*_FP_OPCODES, "fcmp", "call", "select", "phi"):
            match = re.search(rf"= {op} ((?:\w+ )*)", stripped)
            if not match:
                continue
            flags = set(match.group(1).split()) & set(_FP_FLAGS)
            if flags:
                violations.append(f"fast-math flags {sorted(flags)}: {stripped}")
        if "!fpmath" in stripped:
            violations.append(f"!fpmath metadata: {stripped}")
        if stripped.startswith("attributes #"):
            for attribute in _FP_MODE_ATTRIBUTES:
                if f'"{attribute}' in stripped:
                    violations.append(f"FP mode attribute {attribute}: {stripped}")

    # Floating-point intrinsics must be the modeled ones.
    allowed = {_EXP2, _MAXNUM, _MFMA_ATOM[dtype]}
    for line in ir.splitlines():
        match = _DECLARE.match(line.strip())
        if not match:
            continue
        returned, callee, arguments = match.groups()
        if callee in allowed or not callee.startswith("llvm."):
            continue
        if _FP_TYPE.search(returned) or _FP_TYPE.search(arguments):
            violations.append(f"unmodeled floating-point intrinsic @{callee}")

    # Score scale: exactly one fmul(scale, LOG2E).
    scale_products = [
        item
        for item in instructions.values()
        if item.opcode == "fmul"
        and re.fullmatch(
            rf"fmul float (%scale, {_F32_LOG2E}|{_F32_LOG2E}, %scale)", item.text
        )
    ]
    if len(scale_products) != 1:
        violations.append(
            f"expected one fmul(scale, LOG2E), found {len(scale_products)}"
        )
    qk_scale = scale_products[0].name if scale_products else None

    # Masked value and loop-carried initial values.
    for item in instructions.values():
        if item.opcode == "select" and _FP_TYPE.search(item.text.split(",")[1]):
            if item.text.startswith("select i1 %") and ", float %" in item.text:
                if not item.text.endswith(f"float {_F32_MASKED}"):
                    violations.append(
                        f"float select without the -1e30 mask: {item.text}"
                    )
                counts["mask_select"] += 1
            elif item.text.count("zeroinitializer") != 2:
                # Accumulator zero-initialization is the only other float select.
                violations.append(f"unmodeled float select: {item.text}")
        if item.opcode == "phi" and item.text.startswith("phi float"):
            for constant in re.findall(r"\[ (0x[0-9A-F]+),", item.text):
                if constant not in (_F32_ZERO, _F32_MASKED):
                    violations.append(f"float phi starts at {constant}: {item.text}")

    # exp2: probability or rescale, each reading fsub(x, maxnum).
    probability_exp2: set[str] = set()
    alpha_exp2: set[str] = set()
    for item in instructions.values():
        if item.opcode != _EXP2:
            continue
        argument = item.operands[0]
        if opcode(argument) != "fsub":
            violations.append(f"exp2 argument is not an fsub: {item.text}")
        elif opcode(instructions[source(argument)].operands[1]) != _MAXNUM:
            violations.append(
                f"exp2 argument does not subtract the row max: {item.text}"
            )
        kinds = Counter(user.opcode for user in users(item.name))
        if kinds == Counter({"fadd": 1, "fptrunc": 1}):
            probability_exp2.add(item.name)
        elif kinds and set(kinds) == {"fmul"}:
            alpha_exp2.add(item.name)
        else:
            violations.append(f"exp2 with unmodeled uses {dict(kinds)}: {item.text}")
    for item in instructions.values():
        if item.opcode == "fsub" and {u.opcode for u in users(item.name)} != {_EXP2}:
            violations.append(f"fsub not consumed by exp2 only: {item.text}")
    counts["exp2_probability"] = len(probability_exp2)
    counts["exp2_alpha"] = len(alpha_exp2)

    def is_alpha(name: str) -> bool:
        return source(name) in alpha_exp2

    def is_mfma_element(name: str) -> bool:
        item = instructions.get(source(name))
        return (
            item is not None
            and item.opcode == "extractelement"
            and opcode(item.operands[0]) == _MFMA_ATOM[dtype]
        )

    rcp = [item for item in instructions.values() if item.opcode == "fdiv"]
    reciprocal = rf"fdiv float {_F32_ONE}, %[-\w.]+"
    if len(rcp) != 1 or not re.fullmatch(reciprocal, rcp[0].text):
        violations.append(f"expected one fdiv 1.0, l; found {[i.text for i in rcp]}")
    rcp_name = rcp[0].name if rcp else None

    # fmul: scale, score, rescale (l or o), epilogue product.
    for item in instructions.values():
        if item.opcode != "fmul" or item in scale_products:
            continue
        left, right = (*item.operands, None, None)[:2]
        product = left if right == qk_scale else right
        if qk_scale in (left, right) and is_mfma_element(product):
            counts["score"] += 1
        elif is_alpha(right) or is_alpha(left):
            counts["rescale"] += 1
        elif rcp_name is not None and rcp_name in (left, right):
            if {user.opcode for user in users(item.name)} != {"fptrunc"}:
                violations.append(f"output product not cast once: {item.text}")
            counts["output_product"] += 1
        else:
            violations.append(f"unmodeled fmul: {item.text}")

    # fadd: lane chain (sequential, from +0), shuffle partner, rescale combine.
    chain_length: dict[str, int] = {}
    shuffles: set[str] = set()
    for item in instructions.values():
        if item.opcode != "fadd":
            continue
        left, right = item.operands[0], item.operands[-1]
        if source(right) in probability_exp2 and (
            item.text.startswith(f"fadd float {_F32_ZERO}, ")
            or source(left) in chain_length
        ):
            chain_length[item.name] = chain_length.get(source(left), 0) + 1
        elif (
            source(left) in chain_length
            and opcode(right) == "bitcast"
            and opcode(instructions[source(right)].operands[0]) == _BPERMUTE
        ):
            shuffles.add(item.name)
            model.chain_lengths.append(chain_length[source(left)])
        elif (
            opcode(left) == "fmul"
            and any(is_alpha(value) for value in instructions[source(left)].operands)
            and source(right) in shuffles
        ):
            counts["combine"] += 1
        else:
            violations.append(f"unmodeled fadd: {item.text}")
    counts["shuffle"] = len(shuffles)
    heads = sum(1 for length in chain_length.values() if length == 1)
    if heads != len(shuffles):
        violations.append(f"{heads} lane chains but {len(shuffles)} shuffle adds")

    # Casts: probability (exp2 -> dtype) or output (fmul by rcp -> dtype).
    cast_type = _CAST_TYPE[dtype]
    for item in instructions.values():
        if item.opcode == "fpext":
            violations.append(f"unmodeled fpext: {item.text}")
        if item.opcode != "fptrunc":
            continue
        if not item.text.endswith(f" to {cast_type}"):
            violations.append(f"cast to a type other than {cast_type}: {item.text}")
        operand = item.operands[0]
        if source(operand) in probability_exp2:
            counts["cast_probability"] += 1
        elif (
            opcode(operand) == "fmul"
            and rcp_name in instructions[source(operand)].operands
        ):
            counts["cast_output"] += 1
        else:
            violations.append(f"unmodeled cast: {item.text}")

    # MFMA: dtype atom only, no broadcast or lane-group controls.
    for item in instructions.values():
        if "mfma" not in item.opcode:
            continue
        if item.opcode != _MFMA_ATOM[dtype]:
            violations.append(f"unmodeled MFMA {item.opcode}")
        elif not item.text.rstrip().endswith("i32 0, i32 0, i32 0)"):
            violations.append(f"MFMA with nonzero cbsz/abid/blgp: {item.text}")
        counts["mfma"] += 1
    return model


@cache
def _lowered_quick_case(case: UniformCase) -> str:
    """Lower one quick case's dispatch-resolved spec to LLVM IR text on the CPU."""
    from kernels.gfx942.attention_dense import build_attention_dense
    from rocke.core.lower_llvm import (
        _lower_kernel_to_llvm_python,
        _resolve_llvm_flavor,
    )

    spec = _resolve_tuning_spec(case).kernel_spec
    kernel = build_attention_dense(spec, arch="gfx942")
    return _lower_kernel_to_llvm_python(
        kernel, arch="gfx942", llvm_flavor=_resolve_llvm_flavor()
    )


# Per tile body: 64 keys over two lane halves give one 32-term lane chain, one
# shuffle, one rescale combine and one alpha. The FP16 causal kernel has one KV
# loop; the BF16 window-64 kernel splits it into three phases.
_TILE_BODIES: Mapping[str, int] = {
    UNIFORM_QUICK_CASES[0].case_id: 1,
    UNIFORM_QUICK_CASES[1].case_id: 3,
}
_LANE_CHAIN = 32
_OUTPUT_CASTS = 32  # D = 64 over two 32-wide output tiles, 16 per lane


@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_ir_contract_pins_the_dense_path_model(case: UniformCase) -> None:
    model = classify_dense_path(_lowered_quick_case(case), case.dtype)
    assert not model.violations, f"{_REDERIVE}: " + "; ".join(model.violations)

    bodies = _TILE_BODIES[case.case_id]
    assert model.counts["exp2_alpha"] == bodies
    assert model.counts["combine"] == bodies
    assert model.counts["shuffle"] == bodies
    assert model.chain_lengths == [_LANE_CHAIN] * bodies
    assert model.counts["exp2_probability"] == _LANE_CHAIN * bodies
    assert model.counts["cast_probability"] == _LANE_CHAIN * bodies
    assert model.counts["cast_output"] == _OUTPUT_CASTS
    assert model.counts["output_product"] == _OUTPUT_CASTS


def _first_line(ir: str, marker: str) -> str:
    return next(line for line in ir.splitlines() if marker in line)


def _replace_once(old: str, new: str):
    def edit(ir: str) -> str:
        assert old in ir, f"edit target {old!r} not in the lowered IR"
        return ir.replace(old, new, 1)

    return edit


def _append_to_line(marker: str, suffix: str):
    def edit(ir: str) -> str:
        line = _first_line(ir, marker)
        return ir.replace(line, line + suffix, 1)

    return edit


# Each edit applies to the real lowered IR and must be rejected for its own
# reason, so the contract above is shown able to fail on every assumption it pins.
_IR_EDITS = {
    "contract-flag": (
        _replace_once("= fadd float", "= fadd contract float"),
        "fast-math flags ['contract']",
    ),
    "fast-fmul": (
        _replace_once("= fmul float", "= fmul fast float"),
        "fast-math flags ['fast']",
    ),
    "afn-exp2": (
        _replace_once(
            "= call float @llvm.amdgcn.exp2", "= call afn float @llvm.amdgcn.exp2"
        ),
        "fast-math flags ['afn']",
    ),
    "fpmath": (_append_to_line(" fdiv float ", ", !fpmath !0"), "!fpmath metadata"),
    "generic-exp2": (
        lambda ir: ir.replace("@llvm.amdgcn.exp2.f32", "@llvm.exp2.f32"),
        "unmodeled floating-point intrinsic @llvm.exp2.f32",
    ),
    "fma-intrinsic": (
        _replace_once(
            "declare float @llvm.maxnum.f32(float, float)",
            "declare float @llvm.maxnum.f32(float, float)\n"
            "declare float @llvm.fma.f32(float, float, float)",
        ),
        "unmodeled floating-point intrinsic @llvm.fma.f32",
    ),
    "mask-value": (
        _replace_once(f"float {_F32_MASKED}\n", "float 0xFFF0000000000000\n"),
        "float select without the -1e30 mask",
    ),
    "mfma-broadcast": (
        lambda ir: ir.replace(
            (line := _first_line(ir, "= call <16 x float> @llvm.amdgcn.mfma")),
            line.replace("i32 0, i32 0, i32 0)", "i32 1, i32 0, i32 0)"),
            1,
        ),
        "MFMA with nonzero cbsz/abid/blgp",
    ),
    "denormal-mode": (
        _replace_once(
            '"uniform-work-group-size"="true"',
            '"uniform-work-group-size"="true" "denormal-fp-math-f32"="preserve-sign"',
        ),
        "FP mode attribute denormal-fp-math",
    ),
    "second-scale": (
        _append_to_line(
            f"= fmul float %scale, {_F32_LOG2E}",
            f"\n  %p4.extra = fmul float %scale, {_F32_LOG2E}",
        ),
        "expected one fmul(scale, LOG2E), found 2",
    ),
    "stray-fsub": (
        _append_to_line(
            f"= fmul float %scale, {_F32_LOG2E}",
            "\n  %p4.extra = fsub float %scale, %scale",
        ),
        "fsub not consumed by exp2 only",
    ),
}


@pytest.mark.parametrize("edit", sorted(_IR_EDITS))
@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_ir_contract_rejects_each_unmodeled_change(
    case: UniformCase, edit: str
) -> None:
    apply, reason = _IR_EDITS[edit]
    edited = apply(_lowered_quick_case(case))
    violations = classify_dense_path(edited, case.dtype).violations
    assert any(reason in item for item in violations), violations[:3]


def test_ir_contract_rejects_an_unmodeled_reciprocal_intrinsic() -> None:
    case = UNIFORM_QUICK_CASES[0]
    ir = _lowered_quick_case(case)
    line = next(line for line in ir.splitlines() if " fdiv float " in line)
    name = line.split("=")[0].strip()
    divisor = line.rsplit(",", 1)[1].strip()
    edited = ir.replace(
        line, f"  {name} = call float @llvm.amdgcn.rcp.f32(float {divisor})", 1
    ).replace(
        "declare float @llvm.amdgcn.exp2.f32(float)",
        "declare float @llvm.amdgcn.exp2.f32(float)\n"
        "declare float @llvm.amdgcn.rcp.f32(float)",
        1,
    )
    violations = classify_dense_path(edited, case.dtype).violations
    assert any("@llvm.amdgcn.rcp.f32" in item for item in violations)
    assert any("fdiv" in item for item in violations)


# ---------------------------------------------------------------------------
# ISA contract (A-F): the FP16 output epilogue is the only fused site.
#
# Affected AMDGPU LLVM revisions select ``fptrunc(fmul float)`` without
# ``contract`` as ``v_fma_mix{lo,hi}_f16 R, o, 0`` (one rounding) at some sites
# and as an FP32 multiply then ``v_cvt_f16_f32`` (two) at others; LLVM PR
# #236759 makes the fold require ``contract``. The classifier follows each
# 16-bit half of the epilogue registers into the four-half output stores, so a
# store element is "single", "double" or "unknown". FP32 FMAs are allowed only
# inside the one correctly rounded division ``fdiv 1.0, l``.
# ---------------------------------------------------------------------------

_ISA_REGISTER = re.compile(r"^v(\d+)$|^v\[(\d+):(\d+)\]$")
_ISA_MODIFIER = re.compile(r"\s\S+:\[[^]]*\]")
_ISA_FP32_FMA = re.compile(
    r"^v_(?:pk_)?fma(?:c|ak|mk)?_f32|^v_mac_f32|^v_mad_f32|^v_div_fmas_f32"
)
_ISA_MIX = ("v_fma_mix", "v_mad_mix")
# v_perm_b32 selector that packs src1.lo into the low half and src0.lo above it.
_PACK_LOW_HALVES = "0x5040100"


def _isa_registers(operand: str) -> list[int] | None:
    match = _ISA_REGISTER.match(operand)
    if not match:
        return None
    if match.group(1) is not None:
        return [int(match.group(1))]
    return list(range(int(match.group(2)), int(match.group(3)) + 1))


def _isa_instructions(text: str) -> list[tuple[str, list[str]]]:
    """``(mnemonic, operands)`` of every instruction in ``llvm-objdump -d`` text."""
    instructions = []
    for raw in text.splitlines():
        code = raw.split("//")[0].strip()
        if not code or code.endswith(":") or code.startswith(("<", ".")):
            continue
        mnemonic, _, rest = code.partition(" ")
        operands = [
            item.split()[0]
            for item in _ISA_MODIFIER.sub("", rest).split(",")
            if item.strip()
        ]
        instructions.append((mnemonic, operands))
    return instructions


@dataclass
class EpilogueIsa:
    """The fused sites of one disassembled quick kernel.

    ``stores`` lists, per output store, how each 16-bit element was rounded:
    "single" (``v_fma_mix*``), "double" (FP32 multiply then ``v_cvt_f16_f32``)
    or "unknown"."""

    divisions: int = 0
    fma_outside_division: list[str] = field(default_factory=list)
    mix_sites: int = 0
    unmodeled_mix: list[str] = field(default_factory=list)
    stores: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def single_lanes(self) -> frozenset[int] | None:
        """``d mod 4`` of the single-rounded elements, if every store agrees and
        every element is classified."""
        patterns = set(self.stores)
        if len(patterns) != 1 or "unknown" in next(iter(patterns)):
            return None
        (pattern,) = patterns
        return frozenset(index for index, kind in enumerate(pattern) if kind == "single")


def classify_epilogue_isa(text: str) -> EpilogueIsa:
    """Classify the FP16 output epilogue and every FP32 FMA of one kernel."""
    instructions = _isa_instructions(text)
    model = EpilogueIsa()
    starts = [i for i, (name, _) in enumerate(instructions) if name == "v_div_scale_f32"]
    ends = [i for i, (name, _) in enumerate(instructions) if name == "v_div_fixup_f32"]
    model.divisions = len(ends)
    if len(ends) != 1 or not starts:
        return model
    start, end = min(starts), ends[0]
    for index, (name, operands) in enumerate(instructions):
        if _ISA_FP32_FMA.match(name) and not start <= index <= end:
            model.fma_outside_division.append(f"{name} {', '.join(operands)}")
    reciprocal = (_isa_registers(instructions[end][1][0]) or [None])[0]

    halves: dict[tuple[int, int], str] = {}  # (VGPR, 0 low | 1 high) -> rounding
    products: set[int] = set()  # VGPRs holding the FP32 product o * R
    scalars: dict[str, str] = {}
    for name, operands in instructions[start:]:
        if name.startswith(_ISA_MIX):
            model.mix_sites += 1
        if name == "s_mov_b32" and len(operands) == 2:
            scalars[operands[0]] = operands[1]
            continue
        if name.startswith("global_store"):
            data = _isa_registers(operands[1]) or []
            model.stores.append(
                tuple(halves.get((reg, half), "unknown") for reg in data for half in (0, 1))
            )
            continue
        if not name.startswith("v_") or not operands:
            continue
        destination = _isa_registers(operands[0]) or []
        sources = [_isa_registers(item) for item in operands[1:]]
        reads_reciprocal = any(reciprocal in (item or []) for item in sources)
        update: dict[tuple[int, int], str] = {}
        if name.startswith(_ISA_MIX):
            if (
                name in ("v_fma_mixlo_f16", "v_fma_mixhi_f16")
                and operands[-1] == "0"
                and reciprocal in (sources[0] or []) + (sources[1] or [])
            ):
                half = 0 if name == "v_fma_mixlo_f16" else 1
                update[(destination[0], half)] = "single"
            else:
                model.unmodeled_mix.append(f"{name} {', '.join(operands)}")
        elif name in ("v_mul_f32_e32", "v_mul_f32_e64", "v_pk_mul_f32") and (
            reads_reciprocal
        ):
            for register in destination:
                products.add(register)
                halves.pop((register, 0), None)
                halves.pop((register, 1), None)
            continue
        elif name.startswith("v_cvt_f16_f32") and (sources[0] or [None])[0] in products:
            update[(destination[0], 0)] = "double"
            update[(destination[0], 1)] = "zero"
        elif name == "v_pack_b32_f16":
            update[(destination[0], 0)] = halves.get((sources[0][0], 0), "unknown")
            update[(destination[0], 1)] = halves.get((sources[1][0], 0), "unknown")
        elif name == "v_perm_b32" and scalars.get(operands[3]) == _PACK_LOW_HALVES:
            update[(destination[0], 0)] = halves.get((sources[1][0], 0), "unknown")
            update[(destination[0], 1)] = halves.get((sources[0][0], 0), "unknown")
        elif name.startswith("v_mov_b32") and sources[0]:
            for half in (0, 1):
                update[(destination[0], half)] = halves.get((sources[0][0], half), "unknown")
            if sources[0][0] in products:
                products.add(destination[0])
        else:
            update = {(register, half): "unknown" for register in destination for half in (0, 1)}
        for register in destination:
            if not (name.startswith("v_mov_b32") and register in products):
                products.discard(register)
        halves.update(update)
    return model


def _isa_excerpt(*lines: str) -> str:
    return "\n".join(f"\t{line}" for line in lines)


_ISA_DIVISION = (
    "v_div_scale_f32 v32, s[0:1], v84, v84, 1.0",
    "v_rcp_f32_e32 v33, v32",
    "v_div_scale_f32 v34, vcc, 1.0, v84, 1.0",
    "s_mov_b32 s0, 0x5040100",
    "v_fma_f32 v35, -v32, v33, 1.0",
    "v_fmac_f32_e32 v33, v35, v33",
    "v_mul_f32_e32 v35, v34, v33",
    "v_div_fmas_f32 v32, v32, v33, v35",
    "v_div_fixup_f32 v32, v32, v84, 1.0",
)
# One four-half store as the ROCm 7.13 code object emits it (elements 0 and 1
# single), as ROCm 7.1 emits it (elements 0 and 3 single), and as a compiler
# that requires ``contract`` would (all double).
_ISA_STORES = {
    frozenset({0, 1}): (
        "v_pk_mul_f32 v[2:3], v[2:3], v[32:33] op_sel_hi:[1,0]",
        "v_cvt_f16_f32_e32 v33, v2",
        "v_cvt_f16_f32_e32 v3, v3",
        "v_fma_mixlo_f16 v2, v32, v0, 0",
        "v_fma_mixhi_f16 v2, v32, v1, 0",
        "v_pack_b32_f16 v3, v33, v3",
        "v_lshl_add_u64 v[0:1], v[66:67], 1, s[66:67]",
        "global_store_dwordx2 v[0:1], v[2:3], off",
    ),
    frozenset({0, 3}): (
        "v_mov_b32_e32 v34, v17",
        "v_mov_b32_e32 v35, v18",
        "v_pk_mul_f32 v[34:35], v[34:35], v[32:33] op_sel_hi:[1,0]",
        "v_fma_mixlo_f16 v16, v32, v16, 0",
        "v_cvt_f16_f32_e32 v17, v34",
        "v_cvt_f16_f32_e32 v18, v35",
        "v_fma_mixlo_f16 v19, v32, v19, 0",
        "v_pack_b32_f16 v16, v16, v17",
        "v_perm_b32 v17, v19, v18, s0",
        "v_lshl_add_u64 v[18:19], v[64:65], 1, s[72:73]",
        "global_store_dwordx2 v[18:19], v[16:17], off",
    ),
    frozenset(): (
        "v_pk_mul_f32 v[0:1], v[32:33], v[0:1] op_sel_hi:[0,1]",
        "v_pk_mul_f32 v[2:3], v[32:33], v[2:3] op_sel_hi:[0,1]",
        "v_cvt_f16_f32_e32 v0, v0",
        "v_cvt_f16_f32_e32 v1, v1",
        "v_cvt_f16_f32_e32 v2, v2",
        "v_cvt_f16_f32_e32 v3, v3",
        "v_pack_b32_f16 v0, v0, v1",
        "v_pack_b32_f16 v1, v2, v3",
        "global_store_dwordx2 v[4:5], v[0:1], off",
    ),
}


@pytest.mark.parametrize(
    "lanes",
    sorted(_ISA_STORES, key=sorted),
    ids=lambda lanes: "single" + "".join(map(str, sorted(lanes))),
)
def test_isa_classifier_reads_each_epilogue_form(lanes: frozenset[int]) -> None:
    model = classify_epilogue_isa(_isa_excerpt(*_ISA_DIVISION, *_ISA_STORES[lanes]))
    assert model.divisions == 1
    assert not model.fma_outside_division and not model.unmodeled_mix
    assert model.single_lanes == lanes
    assert model.mix_sites == (2 if lanes else 0)


@pytest.mark.parametrize(
    "extra,finding",
    [
        ("v_fmac_f32_e32 v40, v41, v42", "fma_outside_division"),
        ("v_pk_fma_f32 v[40:41], v[42:43], v[44:45], v[46:47]", "fma_outside_division"),
        ("v_fma_mixlo_f16 v2, v32, v0, v5", "unmodeled_mix"),
        ("v_mad_mixlo_f16 v2, v32, v0, 0", "unmodeled_mix"),
    ],
)
def test_isa_classifier_rejects_an_unmodeled_fusion(extra: str, finding: str) -> None:
    """A denominator FMA after the division, a mix FMA with an addend or the
    other mix family is not the A-F epilogue and must be reported."""
    lines = (*_ISA_DIVISION, extra, *_ISA_STORES[frozenset({0, 1})])
    assert getattr(classify_epilogue_isa(_isa_excerpt(*lines)), finding)


def test_isa_classifier_does_not_guess_an_unknown_element() -> None:
    store = list(_ISA_STORES[frozenset({0, 1})])
    store[1] = "v_cvt_f16_f32_e32 v33, v40"  # not a product of the reciprocal
    model = classify_epilogue_isa(_isa_excerpt(*_ISA_DIVISION, *store))
    assert model.stores == [("single", "single", "unknown", "double")]
    assert model.single_lanes is None


def _disassemble(hsaco: bytes, directory) -> str:
    from rocke.tests.llc import find_llvm_tool

    objdump = find_llvm_tool("llvm-objdump")
    if objdump is None:
        pytest.skip("llvm-objdump not found: the ISA contract needs a disassembler")
    path = directory / "kernel.hsaco"
    path.write_bytes(hsaco)
    completed = subprocess.run(
        [objdump, "-d", str(path)], capture_output=True, text=True, check=True
    )
    return completed.stdout


@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_quick_code_objects_fuse_only_the_fp16_epilogue(
    case: UniformCase, tmp_path
) -> None:
    """ISA contract (A-F): compile the quick spec as ``launch_attention_dense``
    does, with the loaded COMGR, and disassemble it.

    - One division, and every FP32 FMA lies inside it: the denominator combine
      ``fadd(fmul(l, alpha), l_tile)`` is not fused.
    - Every ``v_fma_mix*`` is ``R * o + 0`` feeding an output store; BF16 has
      none. FP16 stores are classified element by element. No mix instruction
      at all is the fixed compiler (LLVM PR #236759) and is accepted.
    - A named artifact (by HSACO digest) must show its recorded lanes."""
    import hashlib

    from kernels.gfx942.attention_dense import build_attention_dense
    from rocke.helpers.compile import compile_kernel

    spec = _resolve_tuning_spec(case).kernel_spec
    artifact = compile_kernel(
        build_attention_dense(spec, arch="gfx942"),
        arch="gfx942",
        backend="python",
        capture_ir_text=False,
    )
    model = classify_epilogue_isa(_disassemble(artifact.hsaco, tmp_path))
    digest = hashlib.sha256(artifact.hsaco).hexdigest()
    named = epilogue_artifact_for(case.dtype, digest)
    print(
        f"{case.case_id}: sha256={digest} artifact={named} mix={model.mix_sites} "
        f"single_lanes={model.single_lanes}"
    )
    assert model.divisions == 1, f"{_REDERIVE}: {model.divisions} divisions"
    assert not model.fma_outside_division, f"{_REDERIVE}: {model.fma_outside_division}"
    assert not model.unmodeled_mix, f"{_REDERIVE}: {model.unmodeled_mix}"
    assert len(model.stores) == _OUTPUT_CASTS // 4
    if case.dtype == "bf16":
        assert model.mix_sites == 0
        return
    assert model.single_lanes is not None, model.stores
    assert model.mix_sites == len(model.single_lanes) * len(model.stores)
    if named is not None:
        assert model.single_lanes == P4_EPILOGUE_ARTIFACTS[named].single_lanes


# ---------------------------------------------------------------------------
# Reachable-set primitives: exact FP32 rounding, intervals, exp2 candidates.
# ---------------------------------------------------------------------------

_RNG_SEED = 20261105


def _f32_bits(value: Fraction) -> int:
    return int(np.float32(float(value)).view(np.uint32))


def _f32_from_bits(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _members(
    interval: F32Interval, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Random FP32 members of a nonnegative interval, endpoints included."""
    low, high = _f32_bits(interval.lo), _f32_bits(interval.hi)
    drawn = rng.integers(low, high, size=count, endpoint=True, dtype=np.uint32)
    endpoints = _f32_from_bits(np.array([low, high], dtype=np.uint32))
    return np.concatenate([endpoints, _f32_from_bits(drawn)])


def _random_f32_values(rng: np.random.Generator, count: int) -> np.ndarray:
    """Positive finite FP32 values over every binade, subnormals included, below
    the largest value so a successor exists."""
    bits = rng.integers(1, 0x7F7FFFFE, size=count, endpoint=True, dtype=np.uint32)
    return _f32_from_bits(bits)


def test_round_f32_matches_ieee_float32_round_to_nearest() -> None:
    rng = np.random.default_rng(_RNG_SEED)
    exponents = rng.integers(-160, 120, size=4000)
    doubles = rng.uniform(1.0, 2.0, size=4000) * np.ldexp(1.0, exponents)
    for value in np.concatenate([doubles, -doubles]):
        expected = Fraction(float(np.float32(value)))
        assert round_f32(Fraction(float(value)), "rn") == expected, value


def test_round_f32_breaks_ties_to_even_including_subnormals() -> None:
    rng = np.random.default_rng(_RNG_SEED)
    for value in _random_f32_values(rng, 2000):
        low = Fraction(float(value))
        midpoint = (low + next_up(low)) / 2
        expected = Fraction(float(np.float32(float(midpoint))))
        assert round_f32(midpoint, "rn") == expected
        assert _f32_bits(expected) & 1 == 0


def test_directed_rounding_brackets_with_adjacent_values() -> None:
    rng = np.random.default_rng(_RNG_SEED)
    exponents = rng.integers(-160, 120, size=2000)
    doubles = rng.uniform(1.0, 2.0, size=2000) * np.ldexp(1.0, exponents)
    for value in np.concatenate([doubles, -doubles]):
        exact = Fraction(float(value))
        down, up = round_f32(exact, "rd"), round_f32(exact, "ru")
        assert is_f32(down) and is_f32(up)
        assert down <= exact <= up
        assert up == down if is_f32(exact) else up == next_up(down)


def test_f32_grid_steps_across_a_binade_and_into_subnormals() -> None:
    one = Fraction(1)
    assert f32_quantum(one) == Fraction(1, 1 << 23)
    assert next_up(one) == one + Fraction(1, 1 << 23)
    assert next_down(one) == one - Fraction(1, 1 << 24)
    assert next_up(Fraction(0)) == F32_SMALLEST_SUBNORMAL
    smallest_normal = Fraction(1, 1 << 126)
    assert next_down(smallest_normal) == smallest_normal - F32_SMALLEST_SUBNORMAL
    assert f32_quantum(smallest_normal / 2) == F32_SMALLEST_SUBNORMAL


def test_gamma_is_higham_bound() -> None:
    assert gamma(0) == 0
    assert gamma(1) == Fraction(1, (1 << 24) - 1)
    with pytest.raises(ValueError):
        gamma(1 << 24)


def _sum_in_order(values: np.ndarray, order: np.ndarray) -> np.float32:
    total = np.float32(0.0)
    for index in order:
        total = np.float32(total + values[index])
    return total


def _sum_as_tree(values: list[np.float32], rng: np.random.Generator) -> np.float32:
    items = list(values)
    while len(items) > 1:
        left = items.pop(int(rng.integers(len(items))))
        right = items.pop(int(rng.integers(len(items))))
        items.append(np.float32(left + right))
    return items[0]


@pytest.mark.parametrize("count", [1, 2, 33, 64, 256])
def test_sum_any_order_encloses_every_order_and_tree(count: int) -> None:
    rng = np.random.default_rng(_RNG_SEED + count)
    for _ in range(40):
        centers = _random_f32_values(rng, count).astype(np.float64)
        centers = np.ldexp(centers / np.max(centers), -int(rng.integers(0, 40)))
        terms = []
        for center in centers:
            low = round_f32(Fraction(float(center)), "rd")
            widen = 1 + Fraction(int(rng.integers(0, 8)), 1 << 22)
            high = round_f32(low * widen, "rd")
            terms.append(F32Interval(low, high))
        enclosure = sum_any_order(terms)
        members = np.array(
            [_members(term, rng, 1)[int(rng.integers(3))] for term in terms],
            dtype=np.float32,
        )
        for _ in range(4):
            ordered = _sum_in_order(members, rng.permutation(count))
            assert Fraction(float(ordered)) in enclosure
        assert Fraction(float(_sum_as_tree(list(members), rng))) in enclosure


def test_sum_any_order_admits_flushed_masked_terms() -> None:
    tiny = F32Interval(Fraction(0), F32_SMALLEST_SUBNORMAL)
    one = F32Interval.point(Fraction(1))
    enclosure = sum_any_order([one, tiny, tiny])
    # Flushed (0), kept (2**-149) and absorbed masked terms all sum to 1 in
    # FP32. The order-agnostic bound gamma_2 ~ 2**-23 widens that to
    # [1 - 2**-23, 1 + 2**-23] (two FP32 steps below 1, where the spacing
    # halves, one above); that width is the price of not pinning the order.
    for realization in (Fraction(0), F32_SMALLEST_SUBNORMAL):
        total = np.float32(np.float32(1.0) + np.float32(float(realization)))
        total = np.float32(total + np.float32(float(realization)))
        assert Fraction(float(total)) in enclosure
    step = Fraction(1, 1 << 23)
    assert enclosure == F32Interval(1 - step, 1 + step)


@pytest.mark.parametrize("operation", ["add_rn", "mul_rn", "rcp_rn", "add_faithful"])
def test_interval_operations_enclose_ieee_float32_results(operation: str) -> None:
    rng = np.random.default_rng(_RNG_SEED)
    for _ in range(200):
        base = Fraction(float(_random_f32_values(rng, 1)[0]))
        base = round_f32(base / f32_quantum(base) * Fraction(1, 1 << 23), "rn")
        scale = Fraction(2) ** int(rng.integers(-20, 20))
        left = F32Interval(
            base * scale, round_f32(base * scale * Fraction(65, 64), "rn")
        )
        right = F32Interval(base, round_f32(base * Fraction(33, 32), "rn"))
        for a, b in zip(_members(left, rng, 20), _members(right, rng, 20)):
            if operation == "add_rn":
                result, enclosure = np.float32(a + b), add_rn(left, right)
            elif operation == "mul_rn":
                result, enclosure = np.float32(a * b), mul_rn(left, right)
            elif operation == "rcp_rn":
                result, enclosure = np.float32(np.float32(1.0) / a), rcp_rn(left)
            else:
                exact = Fraction(float(a)) + Fraction(float(b))
                enclosure = add_faithful(left, right)
                for mode in ("rd", "ru", "rn"):
                    assert round_f32(exact, mode) in enclosure
                continue
            assert Fraction(float(result)) in enclosure


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cast_bits_hull_contains_every_member_cast(dtype: str) -> None:
    rng = np.random.default_rng(_RNG_SEED)
    for _ in range(200):
        exponent = int(rng.integers(-30, 10))
        mantissa = Fraction(float(rng.uniform(1.0, 2.0)))
        low = round_f32(mantissa * Fraction(2) ** exponent)
        interval = F32Interval(low, round_f32(low * Fraction(9, 8), "rn"))
        low_bits, high_bits = cast_bits_hull(interval, dtype)
        members = _members(interval, rng, 30)
        if dtype == "fp16":
            bits = members.astype(np.float16).view(np.uint16)
        else:
            bits = f32_to_bf16(members).reshape(members.shape)
        assert np.all((bits >= low_bits) & (bits <= high_bits))


# A-F: the output epilogue. Double rounding is the IR, ``q_T(RN32(o R))``;
# single rounding, FP16 only, is ``q_T(o R)``. The product of two FP32 values
# is exact in FP64, and NumPy rounds FP64 to FP16 once, so both are computed
# here without the module's rational rounding.


def _f32_from_bit(bits: int) -> np.float32:
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def _double_rounded(o: np.float32, r: np.float32, dtype: str) -> int:
    product = np.array([np.float32(o * r)], dtype=np.float32)
    if dtype == "fp16":
        return int(product.astype(np.float16).view(np.uint16)[0])
    return int(f32_to_bf16(product).reshape(-1)[0])


def _single_rounded_fp16(o: np.float32, r: np.float32) -> int:
    return int(np.float16(np.float64(o) * np.float64(r)).view(np.uint16))


# The Phase 4 capture: row 207, head 0, folded, background dimensions 16..63.
_COUNTEREXAMPLE = {"o": 0x3A83E800, "r": 0x3E7BB1A4, "single": 0x0C0D, "double": 0x0C0E}


def test_epilogue_hull_holds_both_roundings_of_the_captured_counterexample() -> None:
    o = _f32_from_bit(_COUNTEREXAMPLE["o"])
    r = _f32_from_bit(_COUNTEREXAMPLE["r"])
    assert _single_rounded_fp16(o, r) == _COUNTEREXAMPLE["single"]
    assert _double_rounded(o, r, "fp16") == _COUNTEREXAMPLE["double"]
    points = F32Interval.point(Fraction(float(o))), F32Interval.point(Fraction(float(r)))
    assert cast_single_or_double_hull(*points, "fp16") == (
        _COUNTEREXAMPLE["single"],
        _COUNTEREXAMPLE["double"],
    )
    # BF16 keeps the IR's double rounding (no BF16 mixed instruction on gfx942).
    bf16 = _double_rounded(o, r, "bf16")
    assert cast_single_or_double_hull(*points, "bf16") == (bf16, bf16)
    assert cast_single_or_double_hull(*points, "bf16") == cast_bits_hull(
        mul_rn(*points), "bf16"
    )


def _epilogue_probes(dtype: str) -> list[tuple[np.float32, np.float32]]:
    """FP32 pairs whose exact product lies within a few FP32 steps of a target
    midpoint, at subnormal, binade-boundary and ordinary target values."""
    if dtype == "fp16":
        anchors = [0x0000, 0x0001, 0x0002, 0x01FF, 0x03FE, 0x03FF, 0x0400, 0x07FF]
        anchors += [0x0C0D, 0x0C0E, 0x2BFF, 0x3BFF, 0x3C00, 0x3C01, 0x5BFF]
    else:
        anchors = [0x0000, 0x0001, 0x007F, 0x0080, 0x3F7F, 0x3F80, 0x3F81, 0x4B7F]
    factors = [
        np.float32(1.0) + np.float32(2.0**-23) * step for step in (1, 3, 5, 11)
    ]
    factors += [np.float32(1.0) - np.float32(2.0**-24) * step for step in (1, 3, 7)]
    probes = []
    for bits in anchors:
        low, high = target_value(bits, dtype), target_value(bits + 1, dtype)
        midpoint = (low + high) / 2
        for factor in factors:
            o = round_f32(midpoint / Fraction(float(factor)), "rn")
            for step in (-2, -1, 0, 1, 2):
                value = o + step * f32_quantum(o)
                if value > 0 and is_f32(value):
                    probes.append((np.float32(float(value)), factor))
    assert len(probes) > 100
    return probes


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_epilogue_hull_at_midpoints_binade_edges_and_subnormals(dtype: str) -> None:
    """At every probe the helper's hull is exactly the allowed epilogue
    results: both roundings for FP16, the double rounding for BF16. The FP16
    probes must include products where the two roundings differ, or the test
    would not tell them apart."""
    differing = 0
    for o, r in _epilogue_probes(dtype):
        points = F32Interval.point(Fraction(float(o))), F32Interval.point(Fraction(float(r)))
        double = _double_rounded(o, r, dtype)
        allowed = {double}
        if dtype == "fp16":
            allowed.add(_single_rounded_fp16(o, r))
            differing += len(allowed) == 2
        assert cast_single_or_double_hull(*points, dtype) == (min(allowed), max(allowed))
    if dtype == "fp16":
        assert differing > 10, differing


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_epilogue_hull_encloses_every_member_of_its_intervals(dtype: str) -> None:
    rng = np.random.default_rng(_RNG_SEED)
    for _ in range(200):
        o = round_f32(Fraction(float(rng.uniform(1.0, 2.0))) * Fraction(2) ** int(rng.integers(-28, 4)))
        r = round_f32(Fraction(float(rng.uniform(1.0, 2.0))) * Fraction(2) ** int(rng.integers(-9, 0)))
        accumulator = F32Interval(o, round_f32(o * Fraction(1025, 1024), "rn"))
        reciprocal = F32Interval(r, round_f32(r * Fraction(129, 128), "rn"))
        low, high = cast_single_or_double_hull(accumulator, reciprocal, dtype)
        for a, b in zip(_members(accumulator, rng, 12), _members(reciprocal, rng, 12)):
            assert low <= _double_rounded(a, b, dtype) <= high
            if dtype == "fp16":
                assert low <= _single_rounded_fp16(a, b) <= high


def test_epilogue_hull_rejects_negative_intervals() -> None:
    """The endpoint formulas hold on nonnegative intervals only."""
    one = F32Interval.point(Fraction(1))
    signed = F32Interval(Fraction(-1), Fraction(1))
    for accumulator, reciprocal in ((signed, one), (one, signed)):
        for dtype in ("fp16", "bf16"):
            with pytest.raises(ValueError, match="nonnegative"):
                cast_single_or_double_hull(accumulator, reciprocal, dtype)


# DT9 / invariant I4: the v_exp_f32 candidate set from the documented bound.


def test_exp2_candidates_at_zero_admit_four_values() -> None:
    """``2**0`` is a power of two: the spacing below 1 is half the spacing
    above, so the 1-ULP bound admits two values below 1, 1 itself and one
    above. A "nearest plus two neighbors" model would miss one of them."""
    one = Fraction(1)
    assert exp2_candidates(Fraction(0)) == (
        one - Fraction(1, 1 << 23),
        one - Fraction(1, 1 << 24),
        one,
        one + Fraction(1, 1 << 23),
    )


def test_exp2_candidates_at_the_masked_argument() -> None:
    """``2**(-1e30)`` is far below FP32's range: the bound admits ``+0`` and the
    smallest subnormal, and both cast to ``+0`` in FP16 and BF16."""
    masked = f32(-1e30)
    assert exp2_candidates(masked) == (Fraction(0), F32_SMALLEST_SUBNORMAL)
    assert exp2_candidates(round_f32(masked - 8, "rn")) == exp2_candidates(masked)


def _exp2_reference_value(argument: Fraction) -> float:
    return float(2.0 ** float(argument))


@pytest.mark.parametrize("logit", [0.5, 2.0, 6.0, 7.5, 8.0, 10.0, 16.0])
def test_exp2_candidates_off_powers_of_two_are_consecutive_and_complete(
    logit: float,
) -> None:
    log2e = f32(np.log2(np.e))
    argument = round_f32(-Fraction(logit) * log2e, "rn")
    candidates = exp2_candidates(argument)
    assert 1 <= len(candidates) <= 3
    assert all(next_up(left) == right for left, right in itertools.pairwise(candidates))
    nearest = round_f32(Fraction(_exp2_reference_value(argument)), "rn")
    assert nearest in candidates
    # Completeness: the FP32 values just outside lie more than one quantum
    # from 2**x (here checked against a double evaluation far from the edge).
    quantum = f32_quantum(Fraction(_exp2_reference_value(argument)))
    for outside in (next_down(candidates[0]), next_up(candidates[-1])):
        distance = abs(Fraction(_exp2_reference_value(argument)) - outside)
        assert distance > quantum


def test_exp2_candidates_refuse_non_f32_arguments() -> None:
    with pytest.raises(ValueError, match="FP32"):
        exp2_candidates(Fraction(1, 3))


# ---------------------------------------------------------------------------
# Fixtures, targets, Tier A and Tier B over the quick matrix.
# ---------------------------------------------------------------------------


def _quick_id(case: P4Case) -> str:
    return f"{case.dtype}-{case.mask}-{case.fixture.value}"


def _llvm_hex(value: Fraction) -> str:
    return "0x" + struct.pack(">d", float(value)).hex().upper()


def test_score_constants_match_the_kernel_and_the_ir_contract() -> None:
    from kernels.gfx942 import attention_dense

    # The kernel's constant, once in FP32, is log2 e correctly rounded: the
    # model's value is derived from the definition, never copied from rocKE.
    log2e = f32(attention_dense.LOG2E)
    assert log2e == LOG2E_F32
    assert _llvm_hex(log2e) == _F32_LOG2E
    assert _llvm_hex(MASKED_SCORE) == _F32_MASKED
    # scale = RN(1/sqrt(64)) = 1/8 exactly, so the product is exact too.
    assert SCORE_SCALE == LOG2E_F32 / 8


def test_quick_matrix_covers_every_fixture_on_both_kernels() -> None:
    ids = [case.case_id for case in P4_QUICK_CASES]
    assert len(ids) == len(set(ids)) == 2 * len(P4Fixture)
    for case in P4_QUICK_CASES:
        family = "walsh" if case.fixture is P4Fixture.WALSH else "periodic"
        # The distinguished key of this row is 64 (even): the Walsh sign is +1.
        query = case.residue_shift + 64
        logits = tuple(row_logit(case, head, query) for head in range(4))
        assert logits == P4_HEAD_LOGITS[family]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"q_scales": (3, 1, 1, 1)}, "power of two"),
        ({"fixture": "cyclic", "shift": 0}, "0 < shift"),
        ({"shift": 5}, "cyclic fixture only"),
        ({"fixture": "late_max"}, "0 < late"),
        ({"block_n": 32}, "block_n == 64"),
    ],
)
def test_p4_case_rejects_inexact_or_unsupported_configurations(kwargs, match) -> None:
    arguments = {
        "base": UNIFORM_QUICK_CASES[0],
        "fixture": "identity",
        "q_scales": (1, 1, 1, 1),
        "k_scales": (1, 1, 1, 1),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        P4Case(**arguments)


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_generators_are_exact_and_match_the_score_model(case: P4Case) -> None:
    q, k = p4_query_key(case)
    for values in (q, k):
        encode_bits(values, case.dtype)  # raises if any value would round
    dots = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64))
    for head in range(case.base.nhead_q):
        expected = np.array(
            [
                [float(score_numerator(case, head, row, key)) for key in range(256)]
                for row in range(256)
            ]
        )
        np.testing.assert_array_equal(dots[0, head], expected)
    for exposure in P4_EXPOSURES:
        encode_bits(p4_inputs(case, exposure)[2], case.dtype)


def test_visited_tiles_follow_the_kernel_block_schedule() -> None:
    causal = next(case for case in P4_QUICK_CASES if case.mask == "causal")
    window = next(case for case in P4_QUICK_CASES if case.mask == "window")
    assert [tuple(visited_tiles(causal, q)) for q in (0, 127, 128, 255)] == [
        (0, 1),
        (0, 1),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
    ]
    assert [tuple(visited_tiles(window, q)) for q in (0, 63, 64, 127, 255)] == [
        (0,),
        (0,),
        (0, 1),
        (0, 1),
        (2, 3),
    ]
    for case in (causal, window):
        for row in range(256):
            first, last = legal_interval(case.base, row)
            tiles = visited_tiles(case, row)
            assert tiles[0] * 64 <= first and last < (tiles[-1] + 1) * 64
    # Row 64t+63 of the window kernel starts on a fully masked tile (case B).
    first, _ = legal_interval(window.base, 127)
    assert first == 64 == visited_tiles(window, 127)[1] * 64


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_target_rounding_geometry_is_exhaustive(dtype: str) -> None:
    """DT1a: format-only rounding geometry over every positive finite value.

    Just inside each midpoint rounds to the near side; the midpoint itself
    rounds to the even pattern. Subnormals and binade edges are included."""
    largest = 0x7BFF if dtype == "fp16" else 0x7F7F
    for bits in range(largest):
        low, high = target_value(bits, dtype), target_value(bits + 1, dtype)
        midpoint = (low + high) / 2
        nudge = (high - low) / 1024
        for value, expected in (
            (midpoint - nudge, bits),
            (midpoint + nudge, bits + 1),
            (midpoint, bits if bits % 2 == 0 else bits + 1),
        ):
            assert target_bits(value.numerator, value.denominator, dtype) == expected


def _within(bits: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (bits >= low) & (bits <= high)


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_tier_b_is_nonempty_and_reports_its_widths(case: P4Case) -> None:
    """DT6 / I1: Tier B is nonempty everywhere. Its width is reported, not
    assumed: A-F lets an FP16 set reach a value either side of the IR's, so
    ``pytest -s`` prints the complete ``|B|`` histogram of each case."""
    histogram: Counter = Counter()
    for exposure in P4_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        width = high.astype(np.int64) - low.astype(np.int64) + 1
        assert np.all(width >= 1), exposure
        histogram.update(width.ravel().tolist())
    print(f"{_quick_id(case)} |B| histogram: {dict(sorted(histogram.items()))}")


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_tier_b_lies_inside_tier_a(case: P4Case) -> None:
    """DT1b / T3: every Tier B value is within 1 target ULP of the certified
    target at row-max coordinates, within 2 elsewhere, and exactly the target
    (+0) at structural zeros. Values are nonnegative, so bit distance is ULP
    distance."""
    for exposure in P4_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        center, radius, _ = tier_a_exposure(case, exposure)
        center = center.astype(np.int64)
        for edge in (low, high):
            distance = np.abs(edge.astype(np.int64) - center)
            assert np.all(distance <= radius), exposure


# Block, tile, window and subnormal-threshold edges of both quick kernels.
_EMULATED_ROWS = (
    *(0, 5, 6, 7, 31, 62, 63, 64, 65, 100),
    *(126, 127, 128, 129, 190, 191, 192, 230, 254, 255),
)


_LANES = np.arange(64) % 4


def _artifact_bits(artifact: str, double: np.ndarray, single: np.ndarray) -> np.ndarray:
    lanes = sorted(P4_EPILOGUE_ARTIFACTS[artifact].single_lanes)
    return np.where(np.isin(_LANES, lanes), single, double)


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_emulated_paths_lie_inside_tier_b(case: P4Case) -> None:
    """DT2 / I2: every emulation lands inside Tier B: the IR-exact path, each
    named artifact's epilogue, single rounding at every FP16 site, and
    randomized-semantic members of the modeled path (summation tree, E
    candidate per call, MFMA rounding direction, epilogue rounding per FP16
    site). The emulator is a test device, never a verdict."""
    rng = np.random.default_rng(_RNG_SEED)
    for exposure in P4_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        for row in _EMULATED_ROWS:
            for head in range(case.base.nhead_q):
                bounds = (low[0, row, head], high[0, row, head])
                double, single = epilogue_alternatives(case, exposure, row, head)
                realizations = [double, single]
                realizations += [
                    _artifact_bits(name, double, single) for name in P4_EPILOGUE_ARTIFACTS
                ]
                realizations.append(
                    emulate_row(
                        case,
                        exposure,
                        row,
                        head,
                        mode=EmulationMode.RANDOMIZED_SEMANTIC,
                        rng=rng,
                    )
                )
                for bits in realizations:
                    assert np.all(_within(bits, *bounds)), (exposure, row, head)


def test_emulation_modes_differ_only_in_the_fp16_epilogue() -> None:
    """``IR_EXACT`` is the double-rounded alternative and ``ARTIFACT_EXACT`` the
    named artifact's per-site choice; BF16 has one alternative. At the
    captured counterexample the two FP16 roundings differ, so the artifact
    rule is observable there."""
    fp16 = next(
        c for c in P4_QUICK_CASES if c.dtype == "fp16" and c.fixture is P4Fixture.IDENTITY
    )
    double, single = epilogue_alternatives(fp16, "folded", 207, 0)
    assert int(double[16]) == _COUNTEREXAMPLE["double"]
    assert int(single[16]) == _COUNTEREXAMPLE["single"]
    np.testing.assert_array_equal(emulate_row(fp16, "folded", 207, 0), double)
    for name in P4_EPILOGUE_ARTIFACTS:
        bits = emulate_row(
            fp16, "folded", 207, 0, mode=EmulationMode.ARTIFACT_EXACT, artifact=name
        )
        np.testing.assert_array_equal(bits, _artifact_bits(name, double, single))
    bf16 = next(c for c in P4_QUICK_CASES if c.dtype == "bf16")
    for row in (0, 127, 255):
        double, single = epilogue_alternatives(bf16, "folded", row, 1)
        np.testing.assert_array_equal(double, single)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mode": "randomized_semantic"}, "rng is required"),
        ({"rng": np.random.default_rng(0)}, "rng is required"),
        ({"mode": "artifact_exact"}, "artifact is required"),
        ({"artifact": "rocm-7.13.0"}, "artifact is required"),
        ({"mode": "artifact_exact", "artifact": "rocm-0"}, "no epilogue artifact"),
    ],
)
def test_emulation_modes_refuse_ambiguous_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        emulate_row(P4_QUICK_CASES[0], "basis@0", 0, 0, **kwargs)


def test_epilogue_artifacts_name_their_code_objects() -> None:
    """Each named artifact gives a compiler, both quick code objects by digest
    and its single-rounded store elements; digests resolve back to it."""
    digests = set()
    for name, artifact in P4_EPILOGUE_ARTIFACTS.items():
        assert artifact.compiler and artifact.evidence
        assert artifact.single_lanes <= {0, 1, 2, 3}
        assert sorted(dtype for dtype, _ in artifact.code_objects) == ["bf16", "fp16"]
        for dtype, digest in artifact.code_objects:
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
            assert epilogue_artifact_for(dtype, digest) == name
            digests.add(digest)
    assert len(digests) == 2 * len(P4_EPILOGUE_ARTIFACTS)
    assert epilogue_artifact_for("fp16", "0" * 64) is None


def test_targets_are_certified_identically_at_50_and_70_digits() -> None:
    """DT3: every distinct target of the quick matrix rounds the same at 50 and
    70 digits, each with its midpoint margin."""
    classes = {
        (case.dtype, exponent, item)
        for case in P4_QUICK_CASES
        for exposure in P4_EXPOSURES
        for exponent, item in tier_a_exposure(case, exposure)[2]
    }
    assert len(classes) > 100
    for dtype, exponent, item in classes:
        assert certified_target_bits(dtype, exponent, item, 50) == (
            certified_target_bits(dtype, exponent, item, 70)
        )


@pytest.mark.parametrize("base", UNIFORM_QUICK_CASES, ids=lambda base: base.case_id)
def test_zero_logits_reproduce_the_uniform_targets(base: UniformCase) -> None:
    """DT5: with every logit zero, P4's targets are P3's, and Tier B holds them."""
    quick = next(case for case in P4_QUICK_CASES if case.base == base)
    zero = P4Case(base, "identity", (0,) * 4, (0,) * 4, block_m=quick.block_m)
    for exposure in P4_EXPOSURES:
        center = tier_a_exposure(zero, exposure)[0]
        np.testing.assert_array_equal(center, uniform_expected_bits(base, exposure))
        low, high = tier_b_exposure(zero, exposure)
        assert np.all(_within(center, low, high)), exposure


def test_fp16_head_scale_does_not_commute_with_the_cast() -> None:
    """DT7: in the FP16 subnormal range ``q_T(2**h p) != 2**h q_T(p)`` for some
    quick-matrix coordinate, so head predictions must not be shifted copies
    of head 0. One counterexample is kept here."""
    case = next(
        c for c in P4_QUICK_CASES if c.dtype == "fp16" and c.fixture is P4Fixture.IDENTITY
    )
    found = []
    for row in range(256):
        item = coordinate_class(case, "basis@0", row, 1, row % 64)
        if item.m_c and item.logit == -8:
            scaled = target_value(certified_target_bits("fp16", 1, item), "fp16")
            shifted = 2 * target_value(certified_target_bits("fp16", 0, item), "fp16")
            if scaled != shifted:
                found.append(row)
    assert found, "no FP16 coordinate where the head scale fails to commute"


def _decimal_exp(value: int) -> Decimal:
    with localcontext() as context:
        context.prec = 40
        return Decimal(value).exp()


def test_fp16_subnormal_threshold_for_one_distinguished_key() -> None:
    """T1: with one distinguished key at c = -8 in a basis exposure, the first
    row length whose stored value is FP16-subnormal is 7, 12, 23, 45 for head
    exponents 0..3 (not 6: at N = 6 the value is 6.71e-5 > 2**-14)."""
    tiny = _decimal_exp(-8)
    limit = Decimal(2) ** -14
    thresholds = []
    for exponent in range(4):
        rows = 1
        while Decimal(2) ** exponent * tiny / (rows - 1 + tiny) >= limit:
            rows += 1
        thresholds.append(rows)
    assert thresholds == [7, 12, 23, 45]


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_range_lemma_holds_on_the_quick_matrix(case: P4Case) -> None:
    """DT8 / T1: every weight and rescale factor is either masked-small
    (at most 2**-149) or at least 2**-12; every nonstructural stored value is
    at least 2**-20; FP16-subnormal outputs occur only at c = -8
    distinguished coordinates."""
    floor_weight, ceiling_masked = Fraction(1, 1 << 12), F32_SMALLEST_SUBNORMAL
    for row in range(256):
        for head in range(case.base.nhead_q):
            for step in row_trace(case, head, row).steps:
                for item in (step.alpha, *step.weights):
                    assert item.hi <= ceiling_masked or item.lo >= floor_weight
    for exposure in P4_EXPOSURES:
        low, _ = tier_b_exposure(case, exposure)
        _, radius, _ = tier_a_exposure(case, exposure)
        nonstructural = radius > 0
        assert np.all(low[nonstructural] > 0)
        smallest = min(target_value(int(bits), case.dtype) for bits in low[nonstructural])
        assert smallest >= Fraction(1, 1 << 20)
        if case.dtype != "fp16":
            continue
        for index in zip(*np.nonzero(nonstructural & (low < 0x0400))):
            _, row, head, dim = (int(value) for value in index)
            item = coordinate_class(case, exposure, row, head, dim)
            assert item.m_c > 0 and item.logit == -8, (exposure, row, head, dim)


@pytest.mark.parametrize("family", sorted(P4_HEAD_SCALES))
def test_head_scales_keep_their_committed_separation(family: str) -> None:
    """DT10: every wrong single Q-head or K-head read moves some coordinate by
    at least the committed margin in target ULPs, on both quick kernels."""
    q_scales, k_scales = P4_HEAD_SCALES[family]
    margin = head_scale_margin(P4_SEARCH_BASES, family, q_scales, k_scales)
    assert margin == P4_HEAD_SCALE_MARGIN[family] > 0


# ---------------------------------------------------------------------------
# Mutation separation (T5, DT4, invariant I5).
# ---------------------------------------------------------------------------


def test_mutation_applicability_follows_fixture_and_mask() -> None:
    covered = set()
    for case in P4_QUICK_CASES:
        mutations = applicable_mutations(case)
        assert len(mutations) == len(set(mutations))
        covered.update(mutations)
        assert (P4Mutation.INVERSE_SHIFT in mutations) == (
            case.fixture is P4Fixture.CYCLIC
        )
        assert (P4Mutation.WALSH_SIGN in mutations) == (
            case.fixture is P4Fixture.WALSH
        )
        assert (P4Mutation.EXCLUDE_LAST_KEY in mutations) == (case.mask == "causal")
        assert (P4Mutation.INCLUDE_OLD_WINDOW_KEY in mutations) == (
            case.mask == "window"
        )
    assert covered == set(P4Mutation)


def _mutated_query_key(
    case: P4Case, mutation: P4Mutation
) -> tuple[np.ndarray, np.ndarray, range]:
    """The Q and K a mutated kernel reads, and the keys they are defined for."""
    q, k = p4_query_key(case)
    keys = range(case.base.seqlen_k)
    if mutation is P4Mutation.Q_HEAD_ROTATE:
        q = np.roll(q, -1, axis=2)
    elif mutation is P4Mutation.K_HEAD_ROTATE:
        k = np.roll(k, -1, axis=2)
    elif mutation is P4Mutation.K_ROW_PLUS_ONE:
        k, keys = k[:, 1:], range(case.base.seqlen_k - 1)
    elif mutation is P4Mutation.K_ROW_MINUS_ONE:
        k, keys = k[:, :-1], range(1, case.base.seqlen_k)
    elif mutation is P4Mutation.INVERSE_SHIFT:
        # Transpose each 64 x 64 key-by-dimension block of the periodic K.
        block = np.swapaxes(k[:, :64], 1, 3)
        k = np.tile(block, (1, case.base.seqlen_k // 64, 1, 1))
    elif mutation is P4Mutation.WALSH_SIGN:
        signs = np.where(np.arange(case.base.seqlen_k) & 1, -1.0, 1.0)
        k = k * signs[None, :, None, None].astype(np.float32)
    elif mutation.value.startswith("omit_walsh_block_"):
        block = int(mutation.value.rsplit("_", 1)[1])
        q = q.copy()
        q[..., 16 * block : 16 * (block + 1)] = 0.0
    return q, k, keys


_SCORE_MUTATIONS = (
    P4Mutation.Q_HEAD_ROTATE,
    P4Mutation.K_HEAD_ROTATE,
    P4Mutation.K_ROW_PLUS_ONE,
    P4Mutation.K_ROW_MINUS_ONE,
    P4Mutation.INVERSE_SHIFT,
    P4Mutation.WALSH_SIGN,
    P4Mutation.OMIT_WALSH_BLOCK_0,
    P4Mutation.OMIT_WALSH_BLOCK_1,
    P4Mutation.OMIT_WALSH_BLOCK_3,
)


@pytest.mark.parametrize(
    "case",
    [case for case in P4_QUICK_CASES if case.dtype == "fp16"],
    ids=lambda case: case.fixture.value,
)
def test_mutated_scores_are_the_mutated_dot_products(case: P4Case) -> None:
    """The closed-form mutated scores equal the exact QK dot products of the
    inputs a mutated kernel would read (heads rotated, K rows off by one, the
    cyclic K transposed, the Walsh sign dropped or one 16-dimension block of
    the contraction skipped). Scale mutations change only the score scale."""
    for mutation in set(_SCORE_MUTATIONS) & set(applicable_mutations(case)):
        q, k, keys = _mutated_query_key(case, mutation)
        dots = np.einsum(
            "bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)
        )[0]
        for head in range(case.base.nhead_q):
            expected = np.array(
                [
                    [
                        float(score_numerator(case, head, row, key, mutation))
                        for key in keys
                    ]
                    for row in range(case.base.seqlen_q)
                ]
            )
            np.testing.assert_array_equal(dots[head], expected, err_msg=mutation.value)

    from kernels.gfx942 import attention_dense

    scales = {
        P4Mutation.OMIT_SQRT_D: round_f32(f32(1.0) * f32(attention_dense.LOG2E)),
        P4Mutation.EXP_ENCODING: f32(1.0 / 8.0),
    }
    row = 64 + case.residue_shift
    key = row - case.residue_shift
    for mutation, scale in scales.items():
        numerator = score_numerator(case, 0, row, key)
        assert numerator != 0
        assert score(case, 0, row, key, mutation) == round_f32(numerator * scale)


def _dot_products(q: np.ndarray, k: np.ndarray) -> set[Fraction]:
    dots = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64))
    return {Fraction(float(value)) for value in np.unique(dots)}


def test_score_products_are_exact_so_a_fused_score_is_unchanged() -> None:
    """DT11: every QK dot product the correct kernel forms on the quick matrix
    is zero or a signed power of two, so its product with every modeled score
    scale is exact in FP32. Then ``fma(acc, qk_scale, -m)`` equals the IR's
    ``RN(RN(acc qk_scale) - m)`` for every row max ``m``: contracting the score
    site could not change a score, and the model needs no score alternative.
    The quick ISA does not fuse it (ISA contract); this is defensive.

    The omitted-Walsh-block mutants are the exception: their dot products are
    ``3 * 2**k`` and their products round. Their mutated Tier B keeps the IR's
    unfused score, which is what an uncontracted multiply-subtract executes."""
    scales = {
        SCORE_SCALE,
        round_f32(f32(1.0) * LOG2E_F32),  # omit_sqrt_d
        f32(1.0 / 8.0),  # exp_encoding
    }
    correct: set[Fraction] = set()
    rounded = set()
    for case in P4_QUICK_CASES:
        correct |= _dot_products(*p4_query_key(case))
        for mutation in set(_SCORE_MUTATIONS) & set(applicable_mutations(case)):
            values = _dot_products(*_mutated_query_key(case, mutation)[:2])
            if any(round_f32(v * SCORE_SCALE) != v * SCORE_SCALE for v in values):
                rounded.add(mutation)
    assert len(correct) > 8
    products = {numerator * scale for numerator in correct for scale in scales}
    for product in products:
        assert round_f32(product) == product, float(product)
    for product in products:
        for row_max in (MASKED_SCORE, *products):
            assert round_f32(product - row_max) == round_f32(round_f32(product) - row_max)
    assert rounded == {
        P4Mutation.OMIT_WALSH_BLOCK_0,
        P4Mutation.OMIT_WALSH_BLOCK_1,
        P4Mutation.OMIT_WALSH_BLOCK_3,
    }


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_every_mutation_is_separated_or_classified(case: P4Case) -> None:
    """DT4 / I5: every applicable mutation's Tier B is disjoint from the correct
    Tier B at some coordinate (``G(m) > 0``), or the cell is listed in
    ``P4_UNSEPARATED`` with its owner. ``G`` is taken over the sampled rows,
    a lower bound of the full gap; ``pytest -s`` prints each ``G`` and its
    coordinate, including the P3-owned denominator and one-key boundary gaps."""
    unexplained = []
    for mutation in applicable_mutations(case):
        separation = mutation_separation(case, mutation)
        print(format_separation(case, separation))
        label = P4_UNSEPARATED.get((case.case_id, mutation))
        if (separation.gap > 0) == (label is not None):
            unexplained.append(f"{mutation.value}: G={separation.gap}, label={label}")
    assert not unexplained, unexplained


def test_unseparated_cells_are_proven_and_owned() -> None:
    """Each listed cell has a proven zero gap at every coordinate and, when it
    names another P4 fixture as owner, that fixture separates the mutation on
    the same kernel.

    For ``OMIT_RESCALE`` the zero is proven, not sampled: if every ``alpha``
    after the first visited tile admits 1, the mutated recurrence
    (``alpha = 1``) is a member of the correct one, and interval propagation is
    inclusion-monotone, so the mutated Tier B lies inside the correct Tier B."""
    assert P4_UNSEPARATED
    cases = {case.case_id: case for case in P4_QUICK_CASES}
    for (case_id, mutation), label in P4_UNSEPARATED.items():
        assert label in SEPARATION_LABELS
        case = cases[case_id]
        assert mutation in applicable_mutations(case)
        if mutation is P4Mutation.OMIT_RESCALE:
            for row in range(case.base.seqlen_q):
                for head in range(case.base.nhead_q):
                    for step in row_trace(case, head, row).steps[1:]:
                        assert Fraction(1) in step.alpha, (case_id, row, head)
        else:
            assert mutation_separation(case, mutation, rows=None).gap == 0
        if label == "owned by another P4 fixture":
            owners = [
                other
                for other in P4_QUICK_CASES
                if other.base == case.base
                and mutation in applicable_mutations(other)
                and (other.case_id, mutation) not in P4_UNSEPARATED
            ]
            assert any(
                mutation_separation(other, mutation).gap > 0 for other in owners
            ), (case_id, mutation.value)


_MUTANT_EXPOSURES = ("basis@0", "folded")
_MUTANT_ROWS = (0, 63, 64, 127, 128, 255)


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_mutant_paths_lie_inside_their_tier_b(case: P4Case) -> None:
    """I2 for the mutated semantics: a mutated kernel's IR-exact and
    randomized-semantic realizations (A-F epilogue choices included) lie in the
    mutated Tier B. Without this, a disjoint pair of enclosures would not imply
    that the mutant's output is rejected."""
    rng = np.random.default_rng(_RNG_SEED)
    modes = (
        {"mode": EmulationMode.IR_EXACT},
        {"mode": EmulationMode.RANDOMIZED_SEMANTIC, "rng": rng},
    )
    for mutation in applicable_mutations(case):
        for exposure in _MUTANT_EXPOSURES:
            for row in _MUTANT_ROWS:
                for head in range(case.base.nhead_q):
                    low, high = tier_b_row(case, exposure, row, head, mutation=mutation)
                    for mode in modes:
                        bits = emulate_row(
                            case, exposure, row, head, mutation=mutation, **mode
                        )
                        assert np.all(_within(bits, low, high)), (
                            mutation.value,
                            exposure,
                            row,
                            head,
                        )


# ---------------------------------------------------------------------------
# Analyzer policy and failure taxonomy (Phase 3).
# ---------------------------------------------------------------------------


def _tier_b_observation(case: P4Case, end: int = 0) -> UniformObservation:
    """Synthetic outputs at one end (0 low, 1 high) of every Tier B set."""
    return UniformObservation(
        tuple(
            (exposure, tier_b_exposure(case, exposure)[end])
            for exposure in P4_EXPOSURES
        )
    )


def _with_bits(
    observation: UniformObservation,
    exposure: str,
    coordinate: tuple[int, int, int, int],
    bits: int,
) -> UniformObservation:
    exposures = []
    for name, values in observation.exposures:
        values = values.copy()
        if name == exposure:
            values[coordinate] = bits
        exposures.append((name, values))
    return UniformObservation(tuple(exposures))


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_structural_zeros_have_tier_b_plus_zero(case: P4Case) -> None:
    """I3 / T4: every structural coordinate (masked, no target) has Tier B
    ``{+0}``. The quick matrix has no trailing basis coordinate and no keyless
    row, so those sentinels cannot occur here."""
    for exposure in P4_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        radius = tier_a_exposure(case, exposure)[1]
        structural = radius == 0
        assert np.all(low[structural] == 0) and np.all(high[structural] == 0)
    assert all(offset + 63 < case.base.seqlen_k for offset in (0, 64, 128, 192))
    assert all(
        legal_interval(case.base, row)[1] >= legal_interval(case.base, row)[0]
        for row in range(case.base.seqlen_q)
    )


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_every_tier_b_value_passes(case: P4Case) -> None:
    """Synthetic outputs at either end of every Tier B set pass, and every
    two-value set is counted with the end that was observed."""
    for end in (0, 1):
        result = analyze_p4_policy(case, _tier_b_observation(case, end))
        assert not result.violations, format_p4_result(result)
        assert result.labels[P4Label.PASS.value] == 5 * int(np.prod(case.output_shape))
        assert result.ambiguous_high == (result.ambiguous if end else 0)


_LABEL_CASE = next(
    case
    for case in P4_QUICK_CASES
    if case.dtype == "fp16" and case.fixture is P4Fixture.IDENTITY
)


def _find(
    exposure: str, rows: range, predicate
) -> tuple[tuple[int, int, int, int], int, int, int, int]:
    """The first coordinate whose ``(low, high, center, radius)`` satisfies
    ``predicate(query, head, dim, low, high, center, radius)``."""
    low, high = tier_b_exposure(_LABEL_CASE, exposure)
    center, radius, _ = tier_a_exposure(_LABEL_CASE, exposure)
    for query in rows:
        for head in range(4):
            for dim in range(64):
                values = tuple(
                    int(array[0, query, head, dim])
                    for array in (low, high, center, radius)
                )
                if predicate(query, head, dim, *values):
                    return ((0, query, head, dim), *values)
    raise AssertionError(f"no {exposure} coordinate satisfies the predicate")


def _relaxed(exposure: str, query: int, head: int) -> tuple[np.ndarray, np.ndarray]:
    return tier_b_row(_LABEL_CASE, exposure, query, head, e_ulps=2)


def _normal(center: int, radius: int) -> bool:
    return radius > 0 and center >= 0x0400


def _label_input(label: P4Label) -> tuple[str, tuple[int, int, int, int], int]:
    """An exposure, coordinate and stored bits that must carry ``label``."""
    from ._p4_analytical import _genuine_addition

    if label is P4Label.STRUCTURAL:
        where, *_ = _find("basis@0", range(256), lambda q, h, d, lo, hi, c, r: r == 0)
        return "basis@0", where, 0x0001
    if label is P4Label.NEGATIVE:
        where, low, *_ = _find(
            "basis@0", range(256), lambda q, h, d, lo, hi, c, r: r > 0
        )
        return "basis@0", where, low | 0x8000
    if label is P4Label.INVALID:
        where, *_ = _find("basis@0", range(256), lambda q, h, d, lo, hi, c, r: r > 0)
        return "basis@0", where, 0x7E00  # the adapter's unwritten-output prefill
    if label is P4Label.E2:
        where, _, high, *_ = _find(
            "basis@0", range(256), lambda q, h, d, lo, hi, c, r: r > 0 and c < 0x0400
        )
        return "basis@0", where, high + 1
    if label is P4Label.A_MFMA:
        where, _, high, *_ = _find(
            "folded",
            range(64, 256),
            lambda q, h, d, lo, hi, c, r: _normal(c, r)
            and _genuine_addition(_LABEL_CASE, q, h, d),
        )
        return "folded", where, high + 1
    if label is P4Label.A_E:

        def widened(q, h, d, lo, hi, c, r):
            relaxed_low, relaxed_high = _relaxed("basis@0", q, h)
            return _normal(c, r) and (relaxed_low[d] < lo or relaxed_high[d] > hi)

        where, low, _, _, _ = _find("basis@0", range(128, 160), widened)
        relaxed_low, relaxed_high = _relaxed("basis@0", where[1], where[2])
        dim = where[3]
        return "basis@0", where, int(
            relaxed_low[dim] if relaxed_low[dim] < low else relaxed_high[dim]
        )

    def outside(step: int):
        def predicate(q, h, d, lo, hi, c, r):
            bits = c + (1 if step == 1 else r + 1)
            return (
                _normal(c, r)
                and lo == hi == c
                and bits > _relaxed("basis@0", q, h)[1][d]
            )

        return predicate

    if label is P4Label.PATH_MODEL:  # one ULP outside an unambiguous Tier B
        where, _, _, center, _ = _find("basis@0", range(1, 64), outside(1))
        return "basis@0", where, center + 1
    assert label is P4Label.NUMERICAL
    where, _, _, center, radius = _find("basis@0", range(1, 64), outside(2))
    return "basis@0", where, center + radius + 1


@pytest.mark.parametrize("label", P4_FAILURE_LABELS, ids=lambda label: label.name)
def test_each_failure_label_is_produced_and_fails(label: P4Label) -> None:
    """Policy: one synthetic coordinate per taxonomy label (§5) fails with that
    label and nothing else, starting from outputs that pass. One ULP outside an
    unambiguous Tier B fails as "path model"; the exact-zero sentinels (a
    nonzero masked coordinate, a negative value, NaN, the unwritten prefill)
    fail as structural, negative and invalid."""
    exposure, coordinate, bits = _label_input(label)
    observation = _with_bits(
        _tier_b_observation(_LABEL_CASE), exposure, coordinate, bits
    )
    result = analyze_p4_policy(_LABEL_CASE, observation, rows=(coordinate[1],))
    failures = {name: count for name, count in result.labels.items() if name != "pass"}
    assert failures == {**dict.fromkeys(failures, 0), label.value: 1}, failures
    assert result.first[label.value] == (exposure, coordinate)
    assert result.violations and result.violations[0].startswith(label.value)


def test_negative_zero_and_infinity_are_not_zero_or_finite() -> None:
    structural, *_ = _find("basis@0", range(256), lambda q, h, d, lo, hi, c, r: r == 0)
    for bits, label in ((0x8000, P4Label.NEGATIVE), (0x7C00, P4Label.INVALID)):
        observation = _with_bits(
            _tier_b_observation(_LABEL_CASE), "basis@0", structural, bits
        )
        result = analyze_p4_policy(_LABEL_CASE, observation, rows=(structural[1],))
        assert result.labels[label.value] == 1


def test_mode_labels_need_a_value_the_mode_can_produce() -> None:
    """At an FP16-subnormal target a flushed ``+0`` or a nearby value is E2, but
    a far value (such as a wrong scale stores) is numerical; a far folded
    value is numerical, not A-MFMA."""
    exposure, coordinate, _ = _label_input(P4Label.E2)
    _, _, _, center, radius = _find(
        exposure,
        range(coordinate[1], coordinate[1] + 1),
        lambda q, h, d, lo, hi, c, r: (0, q, h, d) == coordinate,
    )
    folded, folded_at, _ = _label_input(P4Label.A_MFMA)
    folded_center = int(tier_a_exposure(_LABEL_CASE, folded)[0][folded_at])
    for where, at, bits, label in (
        (exposure, coordinate, 0x0000, P4Label.E2),
        (exposure, coordinate, center + radius + 64, P4Label.NUMERICAL),
        (folded, folded_at, folded_center + 64, P4Label.NUMERICAL),
    ):
        observation = _with_bits(_tier_b_observation(_LABEL_CASE), where, at, bits)
        result = analyze_p4_policy(_LABEL_CASE, observation, rows=(at[1],))
        assert result.labels[label.value] == 1, (where, bits, dict(result.labels))


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in P4_QUICK_CASES
        if (case.dtype, case.fixture)
        in {("fp16", P4Fixture.LATE_MAX), ("bf16", P4Fixture.WALSH)}
    ],
    ids=_quick_id,
)
def test_mutated_analysis_rejects_correct_outputs(case: P4Case) -> None:
    """The analyzer-side mutations the GPU lane runs reject a correct output on
    the sampled rows, for every mutation not listed as inseparable."""
    observation = _tier_b_observation(case)
    for mutation in applicable_mutations(case):
        if (case.case_id, mutation) in P4_UNSEPARATED:
            continue
        result = analyze_p4_policy(
            case, observation, mutation=mutation, rows=P4_SEPARATION_ROWS
        )
        assert result.violations, mutation.value


def test_records_cover_every_coordinate_and_serialize() -> None:
    case = _LABEL_CASE
    observation = _tier_b_observation(case)
    control = analyze_p4_policy(case, observation)
    mutated = analyze_p4_policy(
        case, observation, mutation=P4Mutation.OMIT_SQRT_D, rows=P4_SEPARATION_ROWS
    )
    separation = mutation_separation(case, P4Mutation.OMIT_SQRT_D)
    code_object = {"compiler": "test", "hsaco_sha256": "0" * 64}
    records = p4_records(
        case,
        "attention_gfx942_dense",
        {"none": control, "omit_sqrt_d": mutated},
        deterministic=True,
        diagnostics=p4_diagnostics(case, observation, artifact="rocm-7.13.0"),
        separations={"omit_sqrt_d": separation},
        code_object=code_object,
    )
    for record in records:
        json.dumps(record)
    groups = [r for r in records if r["record_type"] == "rocke_p4_group"]
    control_groups = [r for r in groups if r["mutation"] == "none"]
    per_exposure = Counter()
    for record in control_groups:
        per_exposure[record["exposure"]] += record["coordinates"]
    assert per_exposure == dict.fromkeys(P4_EXPOSURES, int(np.prod(case.output_shape)))
    summaries = {
        r["mutation"]: r for r in records if r["record_type"] == "rocke_p4_summary"
    }
    assert not summaries["none"]["rejected"]
    assert summaries["omit_sqrt_d"]["rejected"]
    assert summaries["omit_sqrt_d"]["cpu_gap"] == separation.gap > 0
    (diagnostics,) = [r for r in records if r["record_type"] == "rocke_p4_diagnostics"]
    assert diagnostics["assumptions"] == list(P4_ASSUMPTIONS)
    assert diagnostics["deterministic"] is True
    assert diagnostics["code_object"] == code_object
    assert diagnostics["epilogue_artifact"] == "rocm-7.13.0"
    assert diagnostics["mfma_rn"] <= diagnostics["mfma_additions"]
    assert 0 < diagnostics["ir_exact"] <= diagnostics["path_compared"]
    assert 0 < diagnostics["artifact_exact"] <= diagnostics["path_compared"]
    unnamed = p4_diagnostics(case, observation, rows=(0,))
    assert unnamed.artifact is None and unnamed.artifact_exact is None


# ---------------------------------------------------------------------------
# Replayable capture audit (exact epilogue union, position rule, Tier B).
# ---------------------------------------------------------------------------

_AUDIT_CASE = _LABEL_CASE  # FP16 identity: holds the captured counterexample
_AUDIT_ROWS = (0, 64, 207)
_AUDIT_ARTIFACT = "rocm-7.13.0"


def _write_capture(directory, case: P4Case = _AUDIT_CASE) -> dict[str, np.ndarray]:
    """A synthetic capture whose audited rows store what the named artifact
    stores; the other rows are not audited."""
    arrays = {
        exposure: np.zeros(case.output_shape, dtype=np.uint16)
        for exposure in P4_EXPOSURES
    }
    for exposure, array in arrays.items():
        for row in _AUDIT_ROWS:
            for head in range(case.base.nhead_q):
                double, single = epilogue_alternatives(case, exposure, row, head)
                array[0, row, head] = _artifact_bits(_AUDIT_ARTIFACT, double, single)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{case.case_id}.npz", **arrays)
    return arrays


def _audit(*directories):
    return audit_captures(
        directories, artifact=_AUDIT_ARTIFACT, cases=(_AUDIT_CASE,), rows=_AUDIT_ROWS
    )


def test_capture_audit_explains_an_artifact_capture(tmp_path) -> None:
    """The audit reproduces the captured finding on an artifact-consistent
    capture: row 207, head 0, folded, background dimensions 16..63 with
    ``d mod 4 in {0, 1}`` differ from the IR-exact emulation (24 here, per
    fixture), every coordinate is in the exact union and in Tier B, and the
    single/double split per lane is the artifact's rule."""
    _write_capture(tmp_path / "a")
    _write_capture(tmp_path / "b")
    audit = _audit(tmp_path / "a", tmp_path / "b")
    print(format_audit(audit))
    assert audit.explained and not audit.problems
    (item,) = audit.cases.values()
    assert item.coordinates == len(_AUDIT_ROWS) * len(P4_EXPOSURES) * 4 * 64
    assert set(item.ir_mismatches) == {
        ("folded", 207, 0, dim) for dim in range(16, 64) if dim % 4 in (0, 1)
    }
    single_lanes = P4_EPILOGUE_ARTIFACTS[_AUDIT_ARTIFACT].single_lanes
    for lane, (single, double) in item.lane_rule.items():
        assert (single > 0, double > 0) == (
            (True, False) if lane in single_lanes else (False, True)
        ), item.lane_rule


def test_capture_audit_reports_rather_than_absorbs(tmp_path) -> None:
    """A rule violation inside the union, a value outside it, drift between
    captures and a malformed capture are each reported; none is explained."""
    arrays = _write_capture(tmp_path / "a")
    double_bits = _COUNTEREXAMPLE["double"]
    arrays["folded"][0, 207, 0, 16] = double_bits  # union member, wrong lane rule
    outside = int(arrays["basis@0"][0, 0, 1, 0]) + 2
    arrays["basis@0"][0, 0, 1, 0] = outside
    np.savez_compressed(tmp_path / "a" / f"{_AUDIT_CASE.case_id}.npz", **arrays)
    drifted = _write_capture(tmp_path / "b")
    drifted["folded"][0, 64, 2, 5] ^= 1
    np.savez_compressed(tmp_path / "b" / f"{_AUDIT_CASE.case_id}.npz", **drifted)
    malformed = tmp_path / "c"
    malformed.mkdir()
    np.savez_compressed(
        malformed / f"{_AUDIT_CASE.case_id}.npz", folded=arrays["folded"]
    )

    audit = _audit(tmp_path / "a", tmp_path / "b", malformed, tmp_path / "missing")
    assert not audit.explained
    (item,) = audit.cases.values()
    assert ("folded", 207, 0, 16) in item.artifact_mismatches
    assert ("folded", 207, 0, 16) not in {where for where, _ in item.unexplained}
    assert (("basis@0", 0, 1, 0), outside) in item.unexplained
    problems = " | ".join(audit.problems)
    assert "3 coordinates differ" in problems  # the two edits and the drift
    assert "exposures ['folded']" in problems
    assert "missing" in problems


def _p4_entry(dtype: str, **overrides) -> P4CalibrationEntry:
    spec = UNIFORM_CALIBRATION[("gfx942", "attention_gfx942_dense", dtype)].spec
    arguments = {
        "policy": P4_POLICY,
        "assumptions": P4_ASSUMPTIONS,
        "case_ids": tuple(c.case_id for c in P4_QUICK_CASES if c.dtype == dtype),
        "spec": spec,
    }
    arguments.update(overrides)
    return P4CalibrationEntry(**arguments)


def test_calibration_entry_has_no_numeric_width() -> None:
    """I6 / D8: an entry names the policy, the assumption versions, the cases
    and the fingerprint; there is no field a measurement could widen."""
    assert [item.name for item in dataclasses.fields(P4CalibrationEntry)] == [
        "policy",
        "assumptions",
        "case_ids",
        "spec",
    ]
    _p4_entry("fp16")
    with pytest.raises(ValueError, match="policy"):
        _p4_entry("fp16", policy="ulp_budget")
    with pytest.raises(ValueError, match="assumptions"):
        _p4_entry("fp16", assumptions=("A-E v2", "A-MFMA v1"))
    with pytest.raises(ValueError, match="case_ids"):
        _p4_entry("fp16", case_ids=())
    with pytest.raises(TypeError):
        _p4_entry("fp16", ulp_budget=1)


def test_calibration_requires_the_epilogue_assumption_version() -> None:
    """Tier B is sound under A-E, A-MFMA and A-F (FP16 epilogue rounding);
    an entry ratified without A-F v1, or under another version, is refused."""
    assert P4_ASSUMPTIONS == ("A-E v1", "A-MFMA v1", "A-F v1")
    assert EPILOGUE_ASSUMPTION.startswith("A-F v1:")
    for assumptions in (
        ("A-E v1", "A-MFMA v1"),
        ("A-E v1", "A-MFMA v1", "A-F v2"),
        ("A-F v1", "A-E v1", "A-MFMA v1"),
    ):
        with pytest.raises(ValueError, match="assumptions"):
            _p4_entry("fp16", assumptions=assumptions)


def test_observations_never_change_the_enclosures() -> None:
    """I6: judging hostile outputs leaves Tier B and the exponential candidate
    sets exactly as derived, and the cached enclosures are read-only."""
    case = _LABEL_CASE
    before = {
        exposure: [bits.copy() for bits in tier_b_exposure(case, exposure)]
        for exposure in P4_EXPOSURES
    }
    candidates = {x: exp2_candidates(x) for x in (Fraction(0), MASKED_SCORE)}
    for bits in (0x3C00, 0x7E00, 0x0000, 0x8001):
        observation = UniformObservation(
            tuple(
                (exposure, np.full(case.output_shape, bits, dtype=np.uint16))
                for exposure in P4_EXPOSURES
            )
        )
        result = analyze_p4_policy(case, observation, rows=P4_SEPARATION_ROWS)
        assert result.violations
    for exposure in P4_EXPOSURES:
        for cached, saved in zip(tier_b_exposure(case, exposure), before[exposure]):
            assert not cached.flags.writeable
            np.testing.assert_array_equal(cached, saved)
    assert {x: exp2_candidates(x) for x in candidates} == candidates


def test_calibration_scope_is_all_quick_cases_or_none() -> None:
    """No partial ratification: the calibrated case set is empty (Phase 4
    capture, failing closed) or exactly the quick matrix, with the same
    dispatch fingerprint as the uniform (P3) entries."""
    quick = {case.case_id for case in P4_QUICK_CASES}
    calibrated = {
        case_id for entry in P4_CALIBRATION.values() for case_id in entry.case_ids
    }
    assert calibrated in (set(), quick)
    for key, entry in P4_CALIBRATION.items():
        assert entry.spec == UNIFORM_CALIBRATION[key].spec
        assert entry.policy == P4_POLICY and entry.assumptions == P4_ASSUMPTIONS


@pytest.mark.parametrize("case", P4_QUICK_CASES, ids=_quick_id)
def test_require_p4_calibration_fails_closed(case: P4Case) -> None:
    spec = UNIFORM_CALIBRATION[("gfx942", "attention_gfx942_dense", case.dtype)].spec
    key = ("gfx942", "attention_gfx942_dense", case.dtype)
    if key not in P4_CALIBRATION:
        with pytest.raises(AssertionError, match="no P4 calibration") as raised:
            require_p4_calibration("gfx942", "attention_gfx942_dense", case, spec)
        assert "ROCKE_ORACLE_FREE_CALIBRATION_OUT" in str(raised.value)
    table = {key: _p4_entry(case.dtype)}
    entry = require_p4_calibration(
        "gfx942", "attention_gfx942_dense", case, spec, table=table
    )
    assert entry is table[key]
    drifted = dataclasses.replace(spec, block_m=spec.block_m // 2)
    with pytest.raises(AssertionError, match="block_m: calibrated"):
        require_p4_calibration(
            "gfx942", "attention_gfx942_dense", case, drifted, table=table
        )
    narrow = {key: _p4_entry(case.dtype, case_ids=("other",))}
    with pytest.raises(AssertionError, match=f"case {case.case_id} not in"):
        require_p4_calibration(
            "gfx942", "attention_gfx942_dense", case, spec, table=narrow
        )


def _p4_gpu_lane_tree() -> ast.Module:
    from pathlib import Path

    path = Path(__file__).with_name("test_p4_analytical_gfx942.py")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_p4_gpu_lane_scope_and_gate() -> None:
    """The P4 GPU lane runs exactly ``P4_QUICK_CASES``, gates once through the
    require-aware ``current_device_arch_in(("gfx942",), ...)``, never skips
    otherwise, and ends in ``require_p4_calibration``."""
    tree = _p4_gpu_lane_tree()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [_dotted(call.func) for call in calls]
    parametrized = [
        call for call, name in zip(calls, names) if name.endswith("mark.parametrize")
    ]
    assert parametrized
    for call in parametrized:
        argvalues = call.args[1]
        assert isinstance(argvalues, ast.Name) and argvalues.id == "P4_QUICK_CASES"
    skip_calls = {"skip", "skipif", "importorskip", "xfail"}
    skips = [name for name in names if name.split(".")[-1] in skip_calls]
    assert not skips
    gates = [
        call for call, name in zip(calls, names) if name == "current_device_arch_in"
    ]
    assert len(gates) == 1 and ast.literal_eval(gates[0].args[0]) == ("gfx942",)
    assert "require_p4_calibration" in names


# ---------------------------------------------------------------------------
# Subject adapter widening.
# ---------------------------------------------------------------------------


def test_adapter_accepts_p4_cases_through_the_shape_protocol() -> None:
    for case in P4_QUICK_CASES:
        _validate_supported_case(case)
        shape = _shape(case)
        assert (shape.batch, shape.seqlen_q, shape.nhead_q, shape.head_dim) == (
            case.output_shape
        )
    parameter = inspect.signature(run_dense_case).parameters["scale_override"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("base", UNIFORM_QUICK_CASES, ids=lambda base: base.case_id)
def test_p4_cases_resolve_to_the_uniform_dispatch_identity(base: UniformCase) -> None:
    """P4 reuses the ratified P3 dispatch identity: every P4 case on a quick
    shape resolves to the same candidate and fingerprint as its P3 case."""
    uniform = resolve_dense_subject(base)
    for case in (c for c in P4_QUICK_CASES if c.base == base):
        subject = resolve_dense_subject(case)
        assert subject.case_id == case.case_id
        assert subject.candidate_name == uniform.candidate_name
        assert subject.fingerprint == uniform.fingerprint


def test_subject_side_wrong_scale_is_the_omitted_sqrt_d_mutation() -> None:
    """The GPU lane's ``scale_override`` launch is the kernel twin of the
    analyzer's ``omit_sqrt_d``: both give ``qk_scale = RN(scale * LOG2E)``."""
    from kernels.gfx942 import attention_dense

    from .test_p4_analytical_gfx942 import _WRONG_SCALE

    case = _LABEL_CASE
    row, key = 64, 64
    numerator = score_numerator(case, 0, row, key)
    qk_scale = round_f32(f32(_WRONG_SCALE) * f32(attention_dense.LOG2E))
    assert score(case, 0, row, key, P4Mutation.OMIT_SQRT_D) == round_f32(
        numerator * qk_scale
    )


_P4_TORCH_BLOCKED_PROBE = """
import importlib
import importlib.abc
import json
import sys


class _RefuseTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "torch":
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, _RefuseTorch())
config = json.loads(sys.argv[1])
sys.path[:0] = config["path"]
package = config["package"]
p4 = importlib.import_module(package + "._p4_analytical")
subject = importlib.import_module(package + "._p4_shared_subject")
importlib.import_module(package + ".test_p4_analytical_gfx942")
for case in p4.P4_QUICK_CASES:
    subject.resolve_dense_subject(case)
loaded = sorted(name for name in sys.modules if name.split(".")[0] == "torch")
assert not loaded, loaded
print("torch-free P4 import and resolution ok")
"""


def test_p4_suite_imports_and_resolves_with_torch_blocked() -> None:
    from .test_contracts import _package_root

    package, root = _package_root()
    config = {"package": package, "path": [str(root), *(e for e in sys.path if e)]}
    completed = subprocess.run(
        [sys.executable, "-c", _P4_TORCH_BLOCKED_PROBE, json.dumps(config)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "torch-free P4 import and resolution ok" in completed.stdout
