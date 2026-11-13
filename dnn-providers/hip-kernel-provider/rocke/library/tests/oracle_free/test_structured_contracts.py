# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the structured-logit oracle-free check.

The prediction is derived from a pinned model of the dense kernel's
floating-point path: the score scale, the masked value, ``v_exp_f32``, the FP32
denominator reduction, one cast of the probabilities, the PV MFMA, one
``fdiv 1.0, l`` reciprocal and one output cast. The IR contract here lowers
the two quick specs
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
import os
import re
import struct
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from rocke.numeric.bf16 import f32_to_bf16

from ._structured_analytical import (
    A_DIV_DOMAIN,
    A_QK_EXPONENTS,
    A_QK_MAGNITUDE,
    EPILOGUE_ASSUMPTION,
    F32_SMALLEST_NORMAL,
    F32_SMALLEST_SUBNORMAL,
    LOG2E_F32,
    MASKED_SCORE,
    STRUCTURED_ASSUMPTION_STATEMENTS,
    STRUCTURED_ASSUMPTIONS,
    STRUCTURED_CALIBRATION,
    STRUCTURED_EPILOGUE_ARTIFACTS,
    STRUCTURED_EXPOSURES,
    STRUCTURED_FAILURE_LABELS,
    STRUCTURED_HEAD_LOGITS,
    STRUCTURED_HEAD_SCALE_MARGIN,
    STRUCTURED_HEAD_SCALES,
    STRUCTURED_POLICY,
    STRUCTURED_QUICK_CASES,
    STRUCTURED_QUICK_SPECS,
    STRUCTURED_SEARCH_BASES,
    STRUCTURED_SEPARATION_ROWS,
    STRUCTURAL_ZERO_RECIPROCAL_BOUND,
    STRUCTURED_UNOBSERVED_VIOLATIONS,
    STRUCTURED_UNSEPARATED,
    SCORE_SCALE,
    SEPARATION_LABELS,
    EmulationMode,
    F32Interval,
    StructuredCalibrationEntry,
    StructuredCase,
    StructuredFixture,
    StructuredLabel,
    StructuredMutation,
    StructuredScopeError,
    StructuredViolation,
    TileClass,
    accumulator_steps,
    add_faithful,
    add_rn,
    analyze_structured_policy,
    applicable_mutations,
    applicable_violations,
    cast_bits_hull,
    cast_single_or_double_hull,
    certified_target_bits,
    certify_target,
    coordinate_class,
    d_ulp,
    derived_rows,
    emulate_row,
    epilogue_alternatives,
    epilogue_artifact_for,
    epilogue_artifacts_for_llvm,
    exp2_candidates,
    exp2_interval,
    exposure_key,
    f32,
    f32_quantum,
    fixture_family,
    floor_log2,
    format_structured_result,
    format_separation,
    gamma,
    head_scale_margin,
    is_f32,
    masked_prefix,
    matching_violation,
    mul_rn,
    mutation_separation,
    next_down,
    next_up,
    ordered_distance,
    structured_diagnostics,
    structured_inputs,
    structured_query_key,
    structured_quick_case,
    structured_records,
    prefix_selected_keys,
    rcp_a_div,
    rcp_rn,
    require_structured_calibration,
    round_f32,
    rounding_boundaries,
    row_logit,
    row_scope,
    row_trace,
    score,
    score_numerator,
    sum_any_order,
    target_rank,
    target_value,
    tier_a_exposure,
    tier_b_exposure,
    tier_b_row,
    tile_classes,
    tile_sum,
    visited_tiles,
)
from ._structured_capture_audit import audit_case, audit_captures, format_audit
from ._structured_shapes import (
    QUICK_SHAPES,
    DenseShape,
    Observation,
    encode_bits,
    exposure_values,
    legal_interval,
    target_bits,
)
from ._structured_subject import (
    _resolve_tuning_spec,
    _shape,
    _validate_supported_case,
    resolve_dense_subject,
    run_dense_case,
)

_REDERIVE = "re-derive the path model"

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
# What may touch a floating-point value: the arithmetic the model has a step
# for, and the instructions that only move its bits (loads, vector packing,
# selects, phis, same-width casts). Each arithmetic opcode has its own rule in
# ``classify_dense_path``; anything else on a floating-point type (a
# conversion, a comparison, ``fneg``, ``frem``) has no step in the model.
_FP_MODELED_ARITHMETIC = ("fadd", "fsub", "fmul", "fdiv", "fptrunc")
_FP_BIT_MOVES = (
    "bitcast",
    "extractelement",
    "getelementptr",
    "insertelement",
    "load",
    "phi",
    "select",
)
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
    # Scores folded into each lane's row-max chain before the shuffle.
    max_chain_lengths: list[int] = field(default_factory=list)
    # One ``(kind, calls)`` per chain of MFMA calls on one accumulator: "score"
    # starts from zero, "pv" from the accumulator rescaled by alpha.
    mfma_chains: list[tuple[str, int]] = field(default_factory=list)


def classify_dense_path(ir: str, dtype: str) -> PathModel:
    """Classify every floating-point instruction against the pinned path model.

    The model, per visited KV tile and row (gfx942 dense, fast exp2):

    - ``qk_scale = fmul(scale, LOG2E)`` once; ``s = fmul(acc, qk_scale)`` with
      ``acc`` an element of a QK MFMA result; masking selects ``-1e30``.
    - The lane maximum is a sequential ``maxnum`` chain starting at ``-1e30``
      over the tile's masked scores, then one ``maxnum`` with the shuffle-xor
      partner's chain (``ds.bpermute``), then ``m_new = maxnum(m, ...)`` with
      ``m`` the maximum carried from the previous tiles (a phi that starts at
      ``-1e30`` and is fed by ``m_new`` only).
    - Every ``exp2`` reads ``fsub(x, m_new)``. A probability ``exp2`` has ``x``
      one of the scores ``m_new`` covers and feeds one denominator ``fadd`` and
      one ``fptrunc`` to the subject dtype; a rescale ``exp2`` (``alpha``) has
      ``x = m`` and feeds ``fmul`` only.
    - The lane denominator is a sequential ``fadd`` chain starting at ``+0``,
      then one ``fadd`` with the shuffle-xor partner's sum (``ds.bpermute``),
      then ``l = fadd(fmul(l_i, alpha), l_tile)``.
    - The accumulator rescale is ``fmul(o, alpha)``; PV is the dtype's
      32x32x8 MFMA with ``cbsz = abid = blgp = 0``.
    - Every MFMA belongs to a chain of eight calls on one accumulator. A score
      chain starts from zero and multiplies two stored operands (K and Q, as
      loaded); a PV chain starts from the rescaled accumulator and multiplies
      a stored operand (V) by the probability cast. Operands reach the call
      through vector packing and bit-preserving casts only.
    - The epilogue is one ``fdiv 1.0, l`` and ``fptrunc(fmul(o, rcp))``.

    No floating-point instruction may carry fast-math flags or ``!fpmath``, and
    the function may set no floating-point mode attribute. Any other
    instruction on a floating-point type must only move bits (a load, vector
    packing, a select, a phi, a ``bitcast``); a conversion, a comparison,
    ``fneg`` or ``frem`` is outside the model. This pins the IR only; what an
    affected compiler selects for the FP16 epilogue is A-F and the ISA contract
    below."""
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

    # Every other instruction on a floating-point type only moves bits. An
    # ``fpext`` and another MFMA atom are reported below, for their own reason.
    modeled = {
        *_FP_MODELED_ARITHMETIC,
        *_FP_BIT_MOVES,
        _EXP2,
        _MAXNUM,
        _MFMA_ATOM[dtype],
    }
    for item in instructions.values():
        if item.opcode in modeled or item.opcode == "fpext" or "mfma" in item.opcode:
            continue
        if _FP_TYPE.search(_VALUE.sub("", item.text)):
            violations.append(f"unmodeled floating-point instruction: {item.text}")

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

    def is_mfma_element(name: str) -> bool:
        item = instructions.get(source(name))
        return (
            item is not None
            and item.opcode == "extractelement"
            and opcode(item.operands[0]) == _MFMA_ATOM[dtype]
        )

    def is_score(name: str) -> bool:
        """``fmul(acc, qk_scale)`` with ``acc`` an element of an MFMA result."""
        item = instructions.get(source(name))
        return (
            item is not None
            and item.opcode == "fmul"
            and qk_scale is not None
            and qk_scale in item.operands
            and any(is_mfma_element(value) for value in item.operands)
        )

    def is_masked_score(name: str) -> bool:
        """A score, or the mask select that keeps it or replaces it by -1e30."""
        item = instructions.get(source(name))
        if item is not None and _MASK_SELECT.match(item.text):
            return is_score(item.operands[1])
        return is_score(name)

    def shuffled_from(name: str) -> str | None:
        """The value whose bits the shuffle partner sends: ``name`` is the
        ``ds.bpermute`` of that value, cast back to float."""
        received = instructions.get(source(name))
        if received is None or received.opcode != "bitcast":
            return None
        permute = instructions.get(received.operands[0])
        if permute is None or permute.opcode != _BPERMUTE:
            return None
        if len(permute.operands) != 2:
            return None
        sent = instructions.get(permute.operands[1])
        if sent is None or sent.opcode != "bitcast":
            return None
        return source(sent.operands[0])

    def carried_phi(name: str) -> str | None:
        item = instructions.get(source(name))
        if item is not None and item.text.startswith("phi float "):
            return item.name
        return None

    # Row max: a sequential lane chain from -1e30 over the tile's masked
    # scores, one maxnum with the shuffle partner's chain, and one with the
    # maximum carried from the previous tiles. Each maps to the scores it
    # covers, so an exp2 below can be held to a score of its own row max.
    lane_max: dict[str, tuple[str, ...]] = {}
    shuffle_max: dict[str, tuple[str, ...]] = {}
    row_max: dict[str, tuple[str, ...]] = {}
    carried_into: dict[str, str] = {}  # row max -> the phi it updates
    for item in instructions.values():
        if item.opcode != _MAXNUM:
            continue
        left, right = (*item.operands, None, None)[:2]
        two = len(item.operands) == 2
        chain = lane_max.get(source(left)) if two else None
        if (
            len(item.operands) == 1
            and f"(float {_F32_MASKED}, float %" in item.text
            and is_masked_score(left)
        ):
            lane_max[item.name] = (source(left),)
        elif chain is not None and is_masked_score(right):
            lane_max[item.name] = (*chain, source(right))
        elif chain is not None and shuffled_from(right) == source(left):
            shuffle_max[item.name] = chain
            model.max_chain_lengths.append(len(chain))
        elif two and carried_phi(left) and source(right) in shuffle_max:
            row_max[item.name] = shuffle_max[source(right)]
            carried_into[item.name] = carried_phi(left)
        else:
            violations.append(f"unmodeled maxnum: {item.text}")
    counts["row_max"] = len(row_max)
    carried = set(carried_into.values())
    for name in sorted(carried):
        for incoming in re.findall(r"\[ (\S+), %[-\w.]+ \]", instructions[name].text):
            if incoming == _F32_MASKED:
                continue
            if source(incoming) not in row_max and source(incoming) not in carried:
                violations.append(
                    "row max carried from an unmodeled value: "
                    f"{instructions[name].text}"
                )

    # exp2: probability or rescale, each reading fsub(x, row max).
    probability_exp2: set[str] = set()
    alpha_exp2: set[str] = set()
    for item in instructions.values():
        if item.opcode != _EXP2:
            continue
        argument = instructions.get(source((*item.operands, None)[0]))
        terms = () if argument is None else argument.operands
        minuend = maximum = None
        if argument is None or argument.opcode != "fsub":
            violations.append(f"exp2 argument is not an fsub: {item.text}")
        elif len(terms) != 2 or source(terms[1]) not in row_max:
            violations.append(
                f"exp2 argument does not subtract the row max: {item.text}"
            )
        else:
            minuend, maximum = source(terms[0]), source(terms[1])
        kinds = Counter(user.opcode for user in users(item.name))
        if kinds == Counter({"fadd": 1, "fptrunc": 1}):
            probability_exp2.add(item.name)
            if maximum is not None and minuend not in row_max[maximum]:
                violations.append(
                    f"probability exp2 of a score outside its row max: {item.text}"
                )
        elif kinds and set(kinds) == {"fmul"}:
            alpha_exp2.add(item.name)
            if maximum is not None and minuend != carried_into[maximum]:
                violations.append(
                    f"rescale exp2 does not read the carried row max: {item.text}"
                )
        else:
            violations.append(f"exp2 with unmodeled uses {dict(kinds)}: {item.text}")
    for item in instructions.values():
        if item.opcode == "fsub" and {u.opcode for u in users(item.name)} != {_EXP2}:
            violations.append(f"fsub not consumed by exp2 only: {item.text}")
    counts["exp2_probability"] = len(probability_exp2)
    counts["exp2_alpha"] = len(alpha_exp2)

    def is_alpha(name: str) -> bool:
        return source(name) in alpha_exp2

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

    # MFMA chains: what each call accumulates onto and what it multiplies.
    storage = _CAST_TYPE[dtype]
    stored_load = re.compile(rf"load (?:<\d+ x {storage}>|{storage}),")

    def assembled_from(name: str, through: tuple[str, ...]) -> list:
        """The instructions ``name`` is built from, looking through ``through``."""
        found, stack, seen = [], [name], set()
        while stack:
            value = stack.pop()
            if value in seen:
                continue
            seen.add(value)
            item = instructions.get(value)
            if item is not None and item.opcode in through:
                stack.extend(item.operands)
            else:
                found.append(item)
        return found

    def operand_kind(name: str) -> str:
        parts = assembled_from(name, ("insertelement", "extractelement", "bitcast"))
        if all(p is not None and stored_load.match(p.text) for p in parts):
            return "stored"
        if all(
            p is not None
            and p.opcode == "fptrunc"
            and source(p.operands[0]) in probability_exp2
            for p in parts
        ):
            return "probability"
        return "unmodeled"

    def accumulator_kind(name: str) -> str:
        item = instructions.get(source(name))
        if item is None:
            return "unmodeled"
        if item.opcode == _MFMA_ATOM[dtype]:
            return "chained"
        if item.opcode == "select" and item.text.count("zeroinitializer") == 2:
            return "zero"
        scalars = assembled_from(item.name, ("insertelement",))
        if len(scalars) == 16 and all(
            part is not None
            and part.opcode == "fmul"
            and any(is_alpha(value) for value in part.operands)
            and any(opcode(value) == "extractelement" for value in part.operands)
            for part in scalars
        ):
            return "rescaled"
        return "unmodeled"

    calls = [i for i in instructions.values() if i.opcode == _MFMA_ATOM[dtype]]
    successor: dict[str, _Instruction] = {}
    heads = []
    for item in calls:
        if len(item.operands) != 3:
            violations.append(f"MFMA without three value operands: {item.text}")
            continue
        if accumulator_kind(item.operands[2]) != "chained":
            heads.append(item)
            continue
        previous = source(item.operands[2])
        if previous in successor:
            violations.append(f"MFMA result accumulated twice: {item.text}")
        successor[previous] = item
    chain_operands = {
        "zero": ["stored", "stored"],
        "rescaled": ["probability", "stored"],
    }
    chain_name = {"zero": "score", "rescaled": "pv"}
    for head in heads:
        kind = accumulator_kind(head.operands[2])
        if kind not in chain_operands:
            violations.append(
                f"MFMA chain starts from an unmodeled accumulator: {head.text}"
            )
            continue
        length, item = 0, head
        while item is not None:
            length += 1
            operands = sorted(operand_kind(value) for value in item.operands[:2])
            if operands != chain_operands[kind]:
                violations.append(
                    f"{chain_name[kind]} MFMA operands are {operands}, "
                    f"expected {chain_operands[kind]}: {item.text}"
                )
            item = successor.get(item.name)
        if length != _MFMA_CHAIN:
            violations.append(
                f"{chain_name[kind]} MFMA chain of {length} calls, "
                f"expected {_MFMA_CHAIN}"
            )
        model.mfma_chains.append((chain_name[kind], length))
    return model


@cache
def _lowered_quick_case(case: DenseShape) -> str:
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
    QUICK_SHAPES[0].case_id: 1,
    QUICK_SHAPES[1].case_id: 3,
}
_LANE_CHAIN = 32
_OUTPUT_CASTS = 32  # D = 64 over two 32-wide output tiles, 16 per lane
# Eight products per call: D / 8 calls per score, block_n / 8 calls per PV tile.
_MFMA_CHAIN = 8
# Per tile body: two 32-key score accumulators and two 32-dim PV accumulators.
_MFMA_CHAINS_PER_BODY = (("pv", _MFMA_CHAIN),) * 2 + (("score", _MFMA_CHAIN),) * 2


@pytest.mark.parametrize("case", QUICK_SHAPES, ids=lambda case: case.case_id)
def test_ir_contract_pins_the_dense_path_model(case: DenseShape) -> None:
    model = classify_dense_path(_lowered_quick_case(case), case.dtype)
    assert not model.violations, f"{_REDERIVE}: " + "; ".join(model.violations)

    bodies = _TILE_BODIES[case.case_id]
    assert model.counts["exp2_alpha"] == bodies
    assert model.counts["combine"] == bodies
    assert model.counts["shuffle"] == bodies
    assert model.chain_lengths == [_LANE_CHAIN] * bodies
    assert model.counts["row_max"] == bodies
    assert model.max_chain_lengths == [_LANE_CHAIN] * bodies
    assert model.counts["exp2_probability"] == _LANE_CHAIN * bodies
    assert model.counts["cast_probability"] == _LANE_CHAIN * bodies
    assert model.counts["cast_output"] == _OUTPUT_CASTS
    assert model.counts["output_product"] == _OUTPUT_CASTS
    assert sorted(model.mfma_chains) == sorted(_MFMA_CHAINS_PER_BODY * bodies)
    assert model.counts["mfma"] == _MFMA_CHAIN * len(_MFMA_CHAINS_PER_BODY) * bodies


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


_MFMA_CALL = re.compile(
    r"= call <16 x float> @llvm\.amdgcn\.mfma[^(]*\("
    r"(<4 x \w+>) (%[-\w.]+), <4 x \w+> %[-\w.]+, <16 x float> (%[-\w.]+),"
)


def _mfma_calls(ir: str) -> list[tuple[str, re.Match]]:
    return [
        (line, match)
        for line in ir.splitlines()
        if (match := _MFMA_CALL.search(line)) is not None
    ]


def _unrescaled_pv_accumulator(ir: str) -> str:
    """The first PV chain accumulates onto the carried ``o``, skipping alpha."""
    line, match = next(
        (line, match)
        for line, match in _mfma_calls(ir)
        if not match.group(3).startswith(("%cz", "%acc"))
    )
    carried = "<16 x float> %o0,"
    assert "%o0 = phi <16 x float>" in ir and carried not in line
    return ir.replace(line, line.replace(f"<16 x float> {match.group(3)},", carried), 1)


def _restarted_mfma_chain(ir: str) -> str:
    """The second call of the first score chain restarts from zero."""
    (_, first), (line, second) = _mfma_calls(ir)[:2]
    assert second.group(3) != first.group(3)
    restarted = line.replace(
        f"<16 x float> {second.group(3)},", f"<16 x float> {first.group(3)},"
    )
    return ir.replace(line, restarted, 1)


def _converted_mfma_operand(ir: str) -> str:
    """An MFMA operand passes through an instruction that is not a load, a
    vector pack or a bit-preserving cast."""
    line, match = _mfma_calls(ir)[0]
    kind, name = match.group(1), match.group(2)
    extra = f"  %probe.operand = select i1 true, {kind} {name}, {kind} {name}\n"
    edited = line.replace(f"{kind} {name},", f"{kind} %probe.operand,", 1)
    return ir.replace(line, extra + edited, 1)


_MAX_CALL = "= call float @llvm.maxnum.f32("
_MAX_HEAD = re.compile(
    rf"= call float @llvm\.maxnum\.f32\(float {_F32_MASKED}, float (%[-\w.]+)\)"
)
_MAX_CARRIED = re.compile(r"\(float (%[-\w.]+), float ")


def _row_max_lines(ir: str) -> tuple[str, str, str, str]:
    """Of the first tile body: the head of the lane-max chain, the shuffle
    ``maxnum``, the row ``maxnum`` and the first ``fsub`` (the rescale's)."""
    lines = ir.splitlines()
    first = next(i for i, line in enumerate(lines) if " = fsub float " in line)
    maxima = [line for line in lines[:first] if _MAX_CALL in line]
    assert len(maxima) == _LANE_CHAIN + 2 and _MAX_HEAD.search(maxima[0])
    return maxima[0], maxima[-2], maxima[-1], lines[first]


def _negated_max_operand(ir: str) -> str:
    """The lane-max chain starts from a negated score."""
    head = _row_max_lines(ir)[0]
    score = _MAX_HEAD.search(head).group(1)
    extra = f"  %probe.negated = fneg float {score}\n"
    edited = head.replace(f"float {score})", "float %probe.negated)")
    return ir.replace(head, extra + edited, 1)


def _max_over_another_value(ir: str) -> str:
    """The lane-max chain starts from the raw scale instead of a score."""
    head = _row_max_lines(ir)[0]
    score = _MAX_HEAD.search(head).group(1)
    return ir.replace(head, head.replace(f"float {score})", "float %scale)"), 1)


def _max_skips_a_score(ir: str) -> str:
    """The second link of the lane-max chain folds the first score again, so
    the row max no longer covers the second."""
    head, second = [line for line in ir.splitlines() if _MAX_CALL in line][:2]
    score = _MAX_HEAD.search(head).group(1)
    edited = re.sub(r"float %[-\w.]+\)$", f"float {score})", second)
    assert edited != second
    return ir.replace(second, edited, 1)


def _max_carried_from_another_value(ir: str) -> str:
    """The carried row max is fed by the raw scale instead of the row max."""
    row = _row_max_lines(ir)[2]
    carried = _MAX_CARRIED.search(row).group(1)
    line = _first_line(ir, f"  {carried} = phi float ")
    edited = re.sub(r"\[ %[-\w.]+, (%[-\w.]+) \]$", r"[ %scale, \1 ]", line)
    assert edited != line
    return ir.replace(line, edited, 1)


def _exp2_subtracts_the_lane_max(ir: str) -> str:
    """The rescale exponent subtracts the shuffle maximum, not the row max."""
    _, shuffle, row, fsub = _row_max_lines(ir)
    lane, updated = (line.split(" = ")[0].strip() for line in (shuffle, row))
    edited = fsub.replace(f", {updated}", f", {lane}")
    assert edited != fsub
    return ir.replace(fsub, edited, 1)


def _alpha_reads_a_score(ir: str) -> str:
    """The rescale exponent starts from a score instead of the carried max."""
    head, _, row, fsub = _row_max_lines(ir)
    score = _MAX_HEAD.search(head).group(1)
    carried = _MAX_CARRIED.search(row).group(1)
    edited = fsub.replace(f"fsub float {carried},", f"fsub float {score},")
    assert edited != fsub
    return ir.replace(fsub, edited, 1)


# Each edit applies to the real lowered IR and must be rejected for its own
# reason, so the contract above is shown able to fail on every assumption it pins.
_IR_EDITS = {
    "pv-unrescaled-accumulator": (
        _unrescaled_pv_accumulator,
        "MFMA chain starts from an unmodeled accumulator",
    ),
    "mfma-restarted-chain": (_restarted_mfma_chain, "MFMA chain of 1 calls"),
    "mfma-converted-operand": (_converted_mfma_operand, "MFMA operands are"),
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
            f"\n  %probe.extra = fmul float %scale, {_F32_LOG2E}",
        ),
        "expected one fmul(scale, LOG2E), found 2",
    ),
    "stray-fsub": (
        _append_to_line(
            f"= fmul float %scale, {_F32_LOG2E}",
            "\n  %probe.extra = fsub float %scale, %scale",
        ),
        "fsub not consumed by exp2 only",
    ),
    "fcmp": (
        _append_to_line(
            f"= fmul float %scale, {_F32_LOG2E}",
            "\n  %probe.extra = fcmp olt float %scale, %scale",
        ),
        "unmodeled floating-point instruction: fcmp",
    ),
    "int-to-float": (
        _append_to_line(
            f"= fmul float %scale, {_F32_LOG2E}",
            "\n  %probe.extra = uitofp i32 %batch to float",
        ),
        "unmodeled floating-point instruction: uitofp",
    ),
    "max-of-negated-score": (
        _negated_max_operand,
        "unmodeled floating-point instruction: fneg",
    ),
    "max-over-another-value": (_max_over_another_value, "unmodeled maxnum"),
    "max-skips-a-score": (
        _max_skips_a_score,
        "probability exp2 of a score outside its row max",
    ),
    "max-carried-from-another-value": (
        _max_carried_from_another_value,
        "row max carried from an unmodeled value",
    ),
    "exp2-subtracts-the-lane-max": (
        _exp2_subtracts_the_lane_max,
        "exp2 argument does not subtract the row max",
    ),
    "alpha-reads-a-score": (
        _alpha_reads_a_score,
        "rescale exp2 does not read the carried row max",
    ),
}


@pytest.mark.parametrize("edit", sorted(_IR_EDITS))
@pytest.mark.parametrize("case", QUICK_SHAPES, ids=lambda case: case.case_id)
def test_ir_contract_rejects_each_unmodeled_change(case: DenseShape, edit: str) -> None:
    apply, reason = _IR_EDITS[edit]
    edited = apply(_lowered_quick_case(case))
    violations = classify_dense_path(edited, case.dtype).violations
    assert any(reason in item for item in violations), violations[:3]


def test_ir_contract_rejects_an_unmodeled_reciprocal_intrinsic() -> None:
    case = QUICK_SHAPES[0]
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
# Tile loop and mask predicate, evaluated from the IR.
#
# The masked-prefix topology is derived from two premises about the
# kernel: which KV tiles a query block visits, and which keys the mask keeps
# (``q - W < k <= q``). Both are integer arithmetic in the lowered IR, so they
# are evaluated here for every work item and lane of the quick shape instead of
# being matched as text: the loop limits give the visited tiles, and each mask
# select gives a key, a row and whether the key is kept.
# ---------------------------------------------------------------------------

_BLOCK_LABEL = re.compile(r"^([-\w.]+):\s*$")
_LOOP_PHI = re.compile(r"^phi i32 \[ (\S+), %[-\w.]+ \], \[ (%[-\w.]+), %([-\w.]+) \]$")
_INT_BINARY = re.compile(
    r"^(add|sub|mul|sdiv|srem)(?: nsw| nuw| exact)* i32 (\S+), (\S+)$"
)
_INT_COMPARE = re.compile(r"^icmp (\w+) i32 (\S+), (\S+)$")
_INT_SELECT = re.compile(r"^select i1 (\S+), i32 (\S+), i32 (\S+)$")
_ID_CALL = re.compile(
    r"^call i32 @llvm\.amdgcn\.(workitem|workgroup)\.id\.([xyz])\(\)$"
)
_MASK_SELECT = re.compile(
    rf"^select i1 (%[-\w.]+), float %[-\w.]+, float {_F32_MASKED}$"
)
_COMPARE = {
    "slt": lambda a, b: a < b,
    "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b,
    "sge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


class _Unmodeled(Exception):
    """The loop schedule uses an instruction the evaluator does not model."""


@dataclass(frozen=True)
class _TileLoop:
    """One KV loop: its tile index, limits and the mask selects of its body."""

    induction: str
    start: str
    bound: str
    masks: tuple[str, ...]  # the ``icmp`` each mask select tests


def _function_blocks(ir: str) -> dict[str, list[str]]:
    """The basic blocks of the one kernel function, in order."""
    blocks: dict[str, list[str]] = {}
    current = None
    for line in ir.splitlines():
        if line.startswith("define "):
            current = None
            continue
        label = _BLOCK_LABEL.match(line)
        if label:
            current = blocks.setdefault(label.group(1), [])
        elif current is not None and line.strip() and line.strip() != "}":
            current.append(line.strip())
    return blocks


def _tile_loops(ir: str) -> tuple[list[_TileLoop], list[str]]:
    """Parse the KV loops. The function must be straight-line code around
    them: ``entry``, then ``header, body, latch, exit`` per loop, each body one
    block (a wave-skip or any other inner control flow is another path)."""
    blocks = _function_blocks(ir)
    names = list(blocks)
    problems: list[str] = []
    loops: list[_TileLoop] = []
    if not names or names[0] != "entry" or (len(names) - 1) % 4:
        return loops, [f"unmodeled control flow: blocks {names}"]
    definitions = {
        match.group(1): match.group(2)
        for lines in blocks.values()
        for line in lines
        if (match := _INSTRUCTION.match(line))
    }
    for index in range(1, len(names), 4):
        header, body, latch, _ = names[index : index + 4]
        phi = compare = None
        for line in blocks[header]:
            match = _INSTRUCTION.match(line)
            if match and phi is None and (found := _LOOP_PHI.match(match.group(2))):
                phi = (match.group(1), *found.groups())
            elif match and (found := _INT_COMPARE.match(match.group(2))):
                compare = (match.group(1), *found.groups())
        if phi is None or compare is None:
            problems.append(f"{header}: no tile index or no trip test")
            continue
        induction, start, step, back_edge = phi
        test, predicate, left, bound = compare
        if predicate != "slt" or left != induction:
            problems.append(f"{header}: trip test is not `{induction} < bound`")
        branch = f"br i1 {test}, label %{body}, label %{names[index + 3]}"
        if blocks[header][-1] != branch:
            problems.append(f"{header}: unmodeled branch {blocks[header][-1]}")
        if back_edge != latch or definitions.get(step) != f"add nsw i32 {induction}, 1":
            problems.append(f"{latch}: the tile index does not advance by one")
        if blocks[latch][-1] != f"br label %{header}":
            problems.append(f"{latch}: unmodeled branch {blocks[latch][-1]}")
        if blocks[body][-1] != f"br label %{latch}":
            problems.append(f"{body}: the tile body is not one block")
        masks = []
        for line in blocks[body]:
            match = _INSTRUCTION.match(line)
            if match and (found := _MASK_SELECT.match(match.group(2))):
                masks.append(found.group(1))
        loops.append(_TileLoop(induction, start, bound, tuple(masks)))
    return loops, problems


def _evaluate(
    definitions: Mapping[str, str], token: str, env: Mapping[str, int], memo: dict
) -> int:
    """The value of an integer SSA token under ``env`` (kernel arguments, the
    work-item and work-group ids and, inside a body, the tile index)."""
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if token in env:
        return env[token]
    if token in memo:
        return memo[token]
    text = definitions.get(token)
    if text is None:
        raise _Unmodeled(f"{token} has no integer definition")
    if found := _INT_BINARY.match(text):
        operation, left, right = found.groups()
        a = _evaluate(definitions, left, env, memo)
        b = _evaluate(definitions, right, env, memo)
        if operation in ("sdiv", "srem"):
            if b == 0:
                raise _Unmodeled(f"division by zero in {token}")
            quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
            value = quotient if operation == "sdiv" else a - b * quotient
        else:
            value = {"add": a + b, "sub": a - b, "mul": a * b}[operation]
    elif found := _INT_COMPARE.match(text):
        predicate, left, right = found.groups()
        value = int(
            _COMPARE[predicate](
                _evaluate(definitions, left, env, memo),
                _evaluate(definitions, right, env, memo),
            )
        )
    elif found := _INT_SELECT.match(text):
        condition, left, right = found.groups()
        chosen = left if _evaluate(definitions, condition, env, memo) else right
        value = _evaluate(definitions, chosen, env, memo)
    elif found := _ID_CALL.match(text):
        value = env[".".join(found.groups())]
    else:
        raise _Unmodeled(f"{token} = {text}")
    memo[token] = value
    return value


@dataclass
class IrSchedule:
    """What the lowered kernel does per query row: the tiles it visits, the
    keys each visited tile scores, and the keys its mask selects keep."""

    tiles: dict[int, tuple[int, ...]] = field(default_factory=dict)
    scored: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    masked: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)


def ir_tile_schedule(ir: str, case: DenseShape, block_m: int) -> IrSchedule:
    """Evaluate the KV loop limits and the mask selects for every work item
    and lane of ``case``. A work item is a ``block_m``-row query block; each
    of its ``block_m / 32`` waves holds 32 rows in 64 lanes, and each lane
    scores 32 of a tile's 64 keys."""
    schedule = IrSchedule()
    loops, problems = _tile_loops(ir)
    schedule.violations.extend(problems)
    if problems:
        return schedule
    definitions = {
        match.group(1): match.group(2)
        for line in ir.splitlines()
        if (match := _INSTRUCTION.match(line))
    }
    arguments = {
        "%batch": case.batch,
        "%seqlen_q": case.seqlen_q,
        "%seqlen_kv": case.seqlen_k,
        "%num_query_heads": case.nhead_q,
        "%num_kv_heads": case.nhead_k,
    }
    blocks = case.seqlen_q // block_m
    # The query block is one work-group axis; which one is the kernel's grid
    # layout, so both candidates are walked and every row must agree.
    groups = {
        (block if axis == "x" else 0, block if axis == "z" else 0)
        for axis in "xz"
        for block in range(blocks)
    }
    problems_seen: set[str] = set()
    for x, z in sorted(groups):
        for thread in range(block_m // 32 * 64):
            env = {
                **arguments,
                "workitem.x": thread,
                "workgroup.x": x,
                "workgroup.y": 0,
                "workgroup.z": z,
            }
            try:
                _evaluate_lane(schedule, definitions, loops, env, case.head_dim)
            except _Unmodeled as error:
                problems_seen.add(f"unmodeled loop schedule: {error}")
    schedule.violations.extend(sorted(problems_seen))
    return schedule


def _evaluate_lane(
    schedule: IrSchedule,
    definitions: Mapping[str, str],
    loops: list[_TileLoop],
    env: Mapping[str, int],
    block_n: int,
) -> None:
    outer: dict = {}
    visited: list[int] = []
    rows: set[int] = set()
    scored: dict[int, set[int]] = {}
    masked: dict[int, set[int]] = {}
    for loop in loops:
        start = _evaluate(definitions, loop.start, env, outer)
        bound = _evaluate(definitions, loop.bound, env, outer)
        for tile in range(start, bound):
            visited.append(tile)
            if not loop.masks:
                raise _Unmodeled(f"tile {tile} of {loop.induction} runs without a mask")
            inner: dict = {}
            tile_env = {**env, loop.induction: tile}
            for condition in loop.masks:
                found = _INT_COMPARE.match(definitions.get(condition, ""))
                if not found or found.group(1) not in ("sle", "sgt"):
                    raise _Unmodeled(f"mask test {definitions.get(condition)}")
                predicate, left, right = found.groups()
                key = _evaluate(definitions, left, tile_env, inner)
                limit = _evaluate(definitions, right, env, outer)  # tile-free
                if predicate == "sle":
                    rows.add(limit)
                else:
                    window = _INT_BINARY.match(definitions.get(right, ""))
                    if not window or window.group(1) != "sub":
                        raise _Unmodeled(f"window bound {definitions.get(right)}")
                    rows.add(_evaluate(definitions, window.group(2), env, outer))
                if not tile * block_n <= key < (tile + 1) * block_n:
                    raise _Unmodeled(f"mask key {key} is outside tile {tile}")
                scored.setdefault(tile, set()).add(key)
                if not _COMPARE[predicate](key, limit):
                    masked.setdefault(tile, set()).add(key)
    if len(rows) != 1:
        raise _Unmodeled(f"a lane masks against rows {sorted(rows)}")
    (row,) = rows
    if schedule.tiles.setdefault(row, tuple(visited)) != tuple(visited):
        raise _Unmodeled(f"row {row} visits different tiles in different lanes")
    for tile in visited:
        schedule.scored.setdefault((row, tile), set()).update(scored.get(tile, ()))
        schedule.masked.setdefault((row, tile), set()).update(masked.get(tile, ()))


def schedule_violations(ir: str, case: DenseShape) -> list[str]:
    """Compare the IR's tile loop and mask with the model's
    :func:`visited_tiles` and ``legal_interval`` on every row of ``case``."""
    quick = structured_quick_case(case, StructuredFixture.IDENTITY)
    schedule = ir_tile_schedule(ir, case, quick.block_m)
    violations = list(schedule.violations)
    if violations:
        return violations
    if sorted(schedule.tiles) != list(range(case.seqlen_q)):
        return [f"the lanes cover rows {sorted(schedule.tiles)[:4]}..., not every row"]
    for row in range(case.seqlen_q):
        tiles = visited_tiles(quick, row)
        if schedule.tiles[row] != tiles:
            violations.append(
                f"row {row} visits tiles {schedule.tiles[row]}, modeled {tiles}"
            )
            continue
        first, last = legal_interval(case, row)
        for tile in tiles:
            keys = set(range(tile * quick.block_n, (tile + 1) * quick.block_n))
            if schedule.scored[(row, tile)] != keys:
                violations.append(f"row {row} tile {tile} does not score 64 keys")
            kept = keys - schedule.masked[(row, tile)]
            legal = {key for key in keys if first <= key <= last}
            if kept != legal:
                violations.append(
                    f"row {row} tile {tile} keeps keys {sorted(kept ^ legal)[:4]} "
                    "differently from the modeled mask"
                )
    return violations


@pytest.mark.parametrize("case", QUICK_SHAPES, ids=lambda case: case.case_id)
def test_ir_tile_loop_and_mask_are_the_modeled_ones(case: DenseShape) -> None:
    """The premises of the tile classes, read from the lowered IR: for every
    query row the kernel visits exactly the modeled tiles, scores all 64 keys of each
    and masks exactly the keys outside ``q - W < k <= q``. The resolved spec
    does not skip fully masked tiles per wave, which would drop their rescale."""
    assert not _resolve_tuning_spec(case).kernel_spec.window_wave_skip
    violations = schedule_violations(_lowered_quick_case(case), case)
    assert not violations, f"{_REDERIVE}: " + "; ".join(violations[:6])


def _first_mask_test(ir: str, predicate: str) -> str:
    conditions = {
        match.group(1)
        for line in ir.splitlines()
        if (found := _INSTRUCTION.match(line))
        and (match := _MASK_SELECT.match(found.group(2)))
    }
    return next(
        line
        for line in ir.splitlines()
        if (found := _INSTRUCTION.match(line))
        and found.group(1) in conditions
        and found.group(2).startswith(f"icmp {predicate} ")
    )


def _strict_causal_mask(ir: str) -> str:
    """One mask select keeps ``k < q`` instead of ``k <= q``."""
    line = _first_mask_test(ir, "sle")
    return ir.replace(line, line.replace("icmp sle", "icmp slt"), 1)


def _extra_tile(ir: str) -> str:
    """The first KV loop runs one tile past its bound."""
    line = next(
        line for line in ir.splitlines() if "%cmp." in line and "icmp slt" in line
    )
    return ir.replace(line, line.replace("icmp slt", "icmp sle"), 1)


def _double_step(ir: str) -> str:
    line = next(
        line for line in ir.splitlines() if "%iv.next." in line and " = add" in line
    )
    return ir.replace(line, line.removesuffix(", 1") + ", 2", 1)


_SCHEDULE_EDITS = {
    "strict-causal-mask": (_strict_causal_mask, "mask test icmp slt"),
    "extra-tile": (_extra_tile, "trip test is not"),
    "double-step": (_double_step, "does not advance by one"),
}


@pytest.mark.parametrize("edit", sorted(_SCHEDULE_EDITS))
@pytest.mark.parametrize("case", QUICK_SHAPES, ids=lambda case: case.case_id)
def test_ir_schedule_contract_rejects_each_change(case: DenseShape, edit: str) -> None:
    apply, reason = _SCHEDULE_EDITS[edit]
    ir = _lowered_quick_case(case)
    edited = apply(ir)
    assert edited != ir
    violations = schedule_violations(edited, case)
    assert any(reason in item for item in violations), violations[:3]


def test_ir_schedule_contract_rejects_a_wrong_bound_or_window() -> None:
    """Changes the evaluator must catch by value, not by shape: a causal loop
    that stops one tile early, and a window one key too narrow."""
    causal, window = QUICK_SHAPES
    ir = _lowered_quick_case(causal)
    (limit,) = re.findall(r"= add nsw i32 %mul\d+, 2\n\s*%lt\d+ = icmp slt", ir)
    early = ir.replace(limit, limit.replace(", 2\n", ", 1\n"), 1)
    assert any("visits tiles" in item for item in schedule_violations(early, causal))

    ir = _lowered_quick_case(window)
    condition = _first_mask_test(ir, "sgt")
    bound = condition.rsplit(", ", 1)[1]
    line = next(
        line for line in ir.splitlines() if line.strip().startswith(f"{bound} = sub")
    )
    assert line.endswith(f", {window.window}")
    narrower = line.removesuffix(str(window.window)) + str(window.window - 1)
    narrow = ir.replace(line, narrower, 1)
    assert any("keeps keys" in item for item in schedule_violations(narrow, window))


# ---------------------------------------------------------------------------
# ISA contract (A-F): the FP16 output epilogue is the only fused site.
#
# Affected AMDGPU LLVM revisions select ``fptrunc(fmul float)`` without
# ``contract`` as ``v_fma_mix{lo,hi}_f16 R, o, 0`` (one rounding) at some sites
# and as an FP32 multiply then ``v_cvt_f16_f32`` (two) at others; a compiler
# that requires ``contract`` for the fold selects none. The classifier follows
# each 16-bit half of the epilogue registers into the four-half output stores,
# so a store element is "single", "double" or "unknown". FP32 FMAs are allowed
# only inside the expansion of the one division ``fdiv 1.0, l``; that the
# expansion returns ``RN(1 / l)`` is assumption A-DIV, which this contract
# does not show: it pins which expansion the kernel holds
# (``division_dataflow``).
# ---------------------------------------------------------------------------

_ISA_REGISTER = re.compile(r"^v(\d+)$|^v\[(\d+):(\d+)\]$")
_ISA_MODIFIER = re.compile(r"\s\S+:\[[^]]*\]")
_ISA_FP32_FMA = re.compile(
    r"^v_(?:pk_)?fma(?:c|ak|mk)?_f32|^v_mac_f32|^v_mad_f32|^v_div_fmas_f32"
)
_ISA_MIX = ("v_fma_mix", "v_mad_mix")
_ISA_SCALAR = re.compile(r"^s(\d+)$|^s\[(\d+):(\d+)\]$")
_ISA_LOADS = ("global_load", "ds_read", "buffer_load", "scratch_load", "flat_load")
# VALU instructions whose second operand is a destination too: the scale flag
# of a division, a carry or borrow, the high part of a 64-bit multiply-add.
_ISA_SECOND_DESTINATION = (
    "v_div_scale",
    "v_add_co",
    "v_addc_co",
    "v_sub_co",
    "v_subb_co",
    "v_subrev_co",
    "v_subbrev_co",
    "v_mad_u64",
    "v_mad_i64",
)
# v_perm_b32 selector that packs src1.lo into the low half and src0.lo above it.
_PACK_LOW_HALVES = "0x5040100"


def _isa_registers(operand: str) -> list[int] | None:
    match = _ISA_REGISTER.match(operand)
    if not match:
        return None
    if match.group(1) is not None:
        return [int(match.group(1))]
    return list(range(int(match.group(2)), int(match.group(3)) + 1))


def _isa_scalar_registers(operand: str) -> list[str]:
    """The scalar registers an operand names, spelled as single registers."""
    match = _ISA_SCALAR.match(operand)
    if not match:
        return []
    if match.group(1) is not None:
        return [operand]
    return [f"s{n}" for n in range(int(match.group(2)), int(match.group(3)) + 1)]


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
    or "unknown". ``mix_sites`` counts every mixed-precision FMA of the kernel,
    the KV loop included; ``unmodeled_mix`` lists those that are not the
    epilogue's ``R * o + 0``."""

    divisions: int = 0
    division: tuple[str, ...] | None = None  # see ``division_dataflow``
    fma_outside_division: list[str] = field(default_factory=list)
    mode_writes: list[str] = field(default_factory=list)
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
        return frozenset(
            index for index, kind in enumerate(pattern) if kind == "single"
        )


def classify_epilogue_isa(text: str) -> EpilogueIsa:
    """Classify the FP16 output epilogue and every FP32 FMA of one kernel."""
    instructions = _isa_instructions(text)
    model = EpilogueIsa()
    starts = [
        i for i, (name, _) in enumerate(instructions) if name == "v_div_scale_f32"
    ]
    ends = [i for i, (name, _) in enumerate(instructions) if name == "v_div_fixup_f32"]
    model.divisions = len(ends)
    model.division = division_dataflow(text)
    # A write to the MODE register changes the rounding or denormal mode the
    # wave started with (A-MODE32, A-CVT-F16, and the mode the division was
    # evaluated in), and the change outlives the instruction that needed it.
    # None is accepted, not even inside the division expansion: nothing here
    # proves that a write there is undone before the epilogue.
    model.mode_writes = [
        f"{name} {', '.join(operands)}"
        for name, operands in instructions
        if name.startswith("s_setreg")
    ]
    model.mix_sites = sum(name.startswith(_ISA_MIX) for name, _ in instructions)
    if len(ends) != 1 or not starts:
        return model
    start, end = min(starts), ends[0]
    for index, (name, operands) in enumerate(instructions):
        site = f"{name} {', '.join(operands)}"
        if _ISA_FP32_FMA.match(name) and not start <= index <= end:
            model.fma_outside_division.append(site)
        # The reciprocal does not exist before the division, so a
        # mixed-precision FMA there (the KV loop) is not the epilogue's.
        if name.startswith(_ISA_MIX) and index < start:
            model.unmodeled_mix.append(site)
    reciprocal = (_isa_registers(instructions[end][1][0]) or [None])[0]

    halves: dict[tuple[int, int], str] = {}  # (VGPR, 0 low | 1 high) -> rounding
    products: set[int] = set()  # VGPRs holding the FP32 product o * R
    scalars: dict[str, str] = {}
    for name, operands in instructions[start:]:
        if name == "s_mov_b32" and len(operands) == 2:
            scalars[operands[0]] = operands[1]
            continue
        # Any other write to a scalar register ends what ``s_mov_b32`` put
        # there, and a load replaces whatever its destination held.
        written = operands[: 2 if name.startswith(_ISA_SECOND_DESTINATION) else 1]
        for operand in written:
            for register in _isa_scalar_registers(operand):
                scalars.pop(register, None)
        if name.startswith(_ISA_LOADS) and operands:
            for register in _isa_registers(operands[0]) or []:
                products.discard(register)
                halves[(register, 0)] = halves[(register, 1)] = "unknown"
            continue
        if name.startswith("global_store"):
            data = _isa_registers(operands[1]) or []
            model.stores.append(
                tuple(
                    halves.get((reg, half), "unknown")
                    for reg in data
                    for half in (0, 1)
                )
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
                update[(destination[0], half)] = halves.get(
                    (sources[0][0], half), "unknown"
                )
            if sources[0][0] in products:
                products.add(destination[0])
        else:
            update = {
                (register, half): "unknown"
                for register in destination
                for half in (0, 1)
            }
        for register in destination:
            if not (name.startswith("v_mov_b32") and register in products):
                products.discard(register)
        halves.update(update)
    return model


# ---------------------------------------------------------------------------
# The division expansion (A-DIV).
#
# gfx942 has no FP32 divide instruction: the backend expands ``fdiv 1.0, l``
# into a scaled reciprocal refined by FMAs. A-DIV says that expansion returns
# ``RN(1 / l)``. The stored bits cannot show whether it does, so the evidence
# is a device evaluation of one particular sequence on every FP32 denominator.
# This contract ties a compiled kernel to that evidence: its expansion must be
# the evaluated one, the same operations on the same values in the same order,
# whatever registers and operand order the compiler chose.
# ---------------------------------------------------------------------------

_ISA_SUFFIX = re.compile(r"_e(?:32|64)$")
# Operation and number of operands read; ``v_fmac_f32`` (destination += a * b)
# is the FMA it computes.
_ISA_DIVISION_OPERATIONS = {
    "v_div_scale_f32": ("div_scale", 3),
    "v_rcp_f32": ("rcp", 1),
    "v_mul_f32": ("mul", 2),
    "v_fma_f32": ("fma", 3),
    "v_fmac_f32": ("fma", 3),
    "v_div_fmas_f32": ("div_fmas", 3),
    "v_div_fixup_f32": ("div_fixup", 3),
}
# The expansion both recorded compilers emit, and the one evaluated on the
# device. ``t0`` and ``t2`` are the scaled denominator and numerator, ``t4`` a
# Newton refinement of ``rcp(t0)``, ``t7`` two FMA corrections of the quotient
# estimate ``t2 * t4``, ``t8`` its residual; the last two instructions undo the
# scaling and fix the special cases.
_ISA_DIVISION_DATAFLOW = (
    "t0 = div_scale(x0, x0, 1.0)",
    "t1 = rcp(t0)",
    "t2 = div_scale(1.0, x0, 1.0)",
    "t3 = fma(-t0*t1 + 1.0)",
    "t4 = fma(t1*t3 + t1)",
    "t5 = mul(t2*t4)",
    "t6 = fma(-t0*t5 + t2)",
    "t7 = fma(t4*t6 + t5)",
    "t8 = fma(-t0*t7 + t2)",
    "t9 = div_fmas(t8, t4, t7)",
    "t10 = div_fixup(t9, x0, 1.0)",
)


def division_dataflow(text: str) -> tuple[str, ...] | None:
    """The division expansion of one disassembled kernel, as value dataflow.

    Every floating-point instruction of the expansion that the quotient
    depends on becomes one line ``tN = operation(operands)``. An operand is an
    earlier ``tN``, a value the expansion reads from outside (``x0`` is the
    denominator) or a constant. Registers do not appear, a multiply-accumulate
    is written as the FMA it computes and the two factors of a product are
    ordered, so two register allocations or operand orders of one computation
    give the same lines. An instruction the scheduler interleaves does not
    appear either; if it overwrites a register of the expansion, the value
    read from that register afterwards is an outside value and the lines
    differ.

    ``None`` unless the kernel holds exactly one expansion of that form."""
    instructions = _isa_instructions(text)
    starts = [
        i for i, (name, _) in enumerate(instructions) if name == "v_div_scale_f32"
    ]
    ends = [i for i, (name, _) in enumerate(instructions) if name == "v_div_fixup_f32"]
    if len(ends) != 1 or not starts or starts[0] > ends[0]:
        return None

    # A value is ("step", index into ``steps``), ("outside", serial number) or
    # ("constant", its text); an operand adds whether it is negated.
    holds: dict[int, tuple[str, int]] = {}  # VGPR -> the value it holds
    steps: list[tuple[str, list[tuple[bool, str, int | str]]]] = []
    outside = itertools.count()

    def read(operand: str) -> tuple[bool, str, int | str]:
        negative, name = operand.startswith("-"), operand.lstrip("-")
        registers = _isa_registers(name)
        if registers is None:
            return negative, "constant", name
        if registers[0] not in holds:
            holds[registers[0]] = ("outside", next(outside))
        return (negative, *holds[registers[0]])

    for name, operands in instructions[starts[0] : ends[0] + 1]:
        mnemonic = _ISA_SUFFIX.sub("", name)
        if mnemonic not in _ISA_DIVISION_OPERATIONS:
            if operands and name.startswith(("v_", *_ISA_LOADS)):
                for register in _isa_registers(operands[0]) or []:
                    holds.pop(register, None)
            continue
        operation, arity = _ISA_DIVISION_OPERATIONS[mnemonic]
        # ``v_div_scale_f32`` writes its scale flag as a second destination.
        sources = operands[2:] if operation == "div_scale" else operands[1:]
        values = [read(item) for item in sources]
        if mnemonic == "v_fmac_f32":
            values.append(read(operands[0]))
        destination = _isa_registers(operands[0]) if operands else None
        if len(values) != arity or not destination:
            return None
        steps.append((operation, values))
        holds[destination[0]] = ("step", len(steps) - 1)

    # Keep what the quotient (the last step, the fixup) depends on.
    needed = {len(steps) - 1}
    for index in range(len(steps) - 1, -1, -1):
        if index in needed:
            needed.update(key for _, kind, key in steps[index][1] if kind == "step")
    order = {index: rank for rank, index in enumerate(sorted(needed))}
    inputs: dict[int | str, int] = {}

    def spell(operand: tuple[bool, str, int | str]) -> tuple[bool, str]:
        negative, kind, key = operand
        if kind == "step":
            return negative, f"t{order[key]}"
        if kind == "outside":
            return negative, f"x{inputs.setdefault(key, len(inputs))}"
        return negative, str(key)

    lines = []
    for index in sorted(needed):
        operation, values = steps[index]
        spelled = [spell(item) for item in values]
        if operation in ("fma", "mul"):
            (left_negative, left), (right_negative, right) = spelled[:2]
            factors = sorted((left, right), key=lambda item: (len(item), item))
            body = ("-" if left_negative != right_negative else "") + "*".join(factors)
            if operation == "fma":
                body += f" + {'-' if spelled[2][0] else ''}{spelled[2][1]}"
        else:
            body = ", ".join(
                ("-" if negative else "") + item for negative, item in spelled
            )
        lines.append(f"t{order[index]} = {operation}({body})")
    return tuple(lines)


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
    "v_fma_f32 v36, -v32, v35, v34",
    "v_fmac_f32_e32 v35, v36, v33",
    "v_fma_f32 v32, -v32, v35, v34",
    "v_div_fmas_f32 v32, v32, v33, v35",
    "v_div_fixup_f32 v32, v32, v84, 1.0",
)
# The same expansion as the other recorded compiler allocates it in a small
# kernel: other registers, and the quotient written over the denominator.
_ISA_DIVISION_OTHER_REGISTERS = (
    "v_div_scale_f32 v3, s[0:1], v2, v2, 1.0",
    "v_rcp_f32_e32 v4, v3",
    "v_div_scale_f32 v5, vcc, 1.0, v2, 1.0",
    "v_fma_f32 v6, -v3, v4, 1.0",
    "v_fmac_f32_e32 v4, v6, v4",
    "v_mul_f32_e32 v6, v5, v4",
    "v_fma_f32 v7, -v3, v6, v5",
    "v_fmac_f32_e32 v6, v7, v4",
    "v_fma_f32 v3, -v3, v6, v5",
    "v_div_fmas_f32 v3, v3, v4, v6",
    "v_div_fixup_f32 v2, v3, v2, 1.0",
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


def test_division_dataflow_is_the_evaluated_expansion() -> None:
    """A-DIV: the expansion reduces to the evaluated dataflow under either
    register allocation, with the factors of a product in either order, with
    a multiply-accumulate written as an FMA and with unrelated instructions
    interleaved. An expansion that computes anything else does not."""
    for lines in (_ISA_DIVISION, _ISA_DIVISION_OTHER_REGISTERS):
        assert division_dataflow(_isa_excerpt(*lines)) == _ISA_DIVISION_DATAFLOW
    kernel = _isa_excerpt(*_ISA_DIVISION, *_ISA_STORES[frozenset()])
    assert classify_epilogue_isa(kernel).division == _ISA_DIVISION_DATAFLOW

    def edited(old: str, *new: str) -> tuple[str, ...] | None:
        assert old in _ISA_DIVISION
        lines: list[str] = []
        for line in _ISA_DIVISION:
            lines.extend(new if line == old else (line,))
        return division_dataflow(_isa_excerpt(*lines))

    multiply = "v_mul_f32_e32 v35, v34, v33"
    residual = "v_fma_f32 v36, -v32, v35, v34"
    same = (
        (multiply, "v_mul_f32_e32 v35, v33, v34"),
        (residual, "v_fma_f32 v36, v35, -v32, v34"),
        ("v_fmac_f32_e32 v33, v35, v33", "v_fma_f32 v33, v33, v35, v33"),
        (
            multiply,
            "v_mul_f32_e32 v60, v61, v62",
            multiply,
            "v_add_u32_e32 v63, v60, v61",
        ),
    )
    for old, *new in same:
        assert edited(old, *new) == _ISA_DIVISION_DATAFLOW, new
    numerator = "v_div_scale_f32 v34, vcc, 1.0, v84, 1.0"
    different = (
        (residual,),  # one refinement step fewer
        (residual, "v_fma_f32 v36, v32, v35, v34"),  # a residual of the wrong sign
        ("v_rcp_f32_e32 v33, v32", "v_rcp_f32_e32 v33, v84"),  # unscaled reciprocal
        (numerator, "v_div_scale_f32 v34, vcc, 2.0, v84, 1.0"),  # another numerator
        (multiply, multiply, "v_mov_b32_e32 v33, v40"),  # a value overwritten
    )
    for old, *new in different:
        found = edited(old, *new)
        assert found is not None and found != _ISA_DIVISION_DATAFLOW, new
    # No expansion, two expansions, an instruction of the wrong shape.
    assert division_dataflow(_isa_excerpt(*_ISA_STORES[frozenset()])) is None
    assert division_dataflow(_isa_excerpt(*_ISA_DIVISION, *_ISA_DIVISION)) is None
    assert edited("v_rcp_f32_e32 v33, v32", "v_rcp_f32_e32 v33, v32, v32") is None


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


@pytest.mark.parametrize(
    "extra", ["v_fma_mixlo_f16 v2, v32, v0, 0", "v_mad_mixhi_f16 v7, v8, v9, v10"]
)
def test_isa_classifier_reports_a_mix_before_the_division(extra: str) -> None:
    """A mixed-precision FMA in the KV loop comes before the division. It is
    counted and reported whatever its operands: the reciprocal the epilogue
    form reads does not exist yet."""
    lanes = frozenset({0, 1})
    lines = (extra, *_ISA_DIVISION, *_ISA_STORES[lanes])
    model = classify_epilogue_isa(_isa_excerpt(*lines))
    assert model.unmodeled_mix == [extra]
    assert model.mix_sites == 3
    assert model.single_lanes == lanes


def test_isa_classifier_drops_what_a_later_write_replaces() -> None:
    """A pack selector is trusted only while its scalar register still holds
    it, and a store element only while no load has replaced its register."""
    store = _ISA_STORES[frozenset({0, 3})]
    perm = next(i for i, line in enumerate(store) if line.startswith("v_perm_b32"))
    for write in ("s_add_u32 s0, s0, 4", "v_cmp_lt_u32_e64 s[0:1], v40, v41"):
        lines = (*_ISA_DIVISION, *store[:perm], write, *store[perm:])
        model = classify_epilogue_isa(_isa_excerpt(*lines))
        assert model.stores == [("single", "double", "unknown", "unknown")], write
        assert model.single_lanes is None
    load = "global_load_dwordx2 v[16:17], v[18:19], off"
    lines = (*_ISA_DIVISION, *store[:-1], load, store[-1])
    model = classify_epilogue_isa(_isa_excerpt(*lines))
    assert model.stores == [("unknown",) * 4]


def test_isa_classifier_reports_a_mode_register_write() -> None:
    """A-MODE32 and A-CVT-F16: the wave keeps the modes of its kernel
    descriptor. Every write to the MODE register is reported, wherever it is.
    One inside the division expansion (a compiler that enables denormals
    around the refinement) is reported too, alone or followed by a second
    write: the classifier does not prove that the second restores the mode
    the wave started with."""
    store = _ISA_STORES[frozenset({0, 1})]
    write = "s_setreg_imm32_b32 0x3, 0xf0"
    restore = "s_setreg_b32 0x3, s4"
    clean = classify_epilogue_isa(_isa_excerpt(*_ISA_DIVISION, *store))
    assert not clean.mode_writes and clean.divisions == 1
    outside = classify_epilogue_isa(_isa_excerpt(*_ISA_DIVISION, write, *store))
    assert outside.mode_writes == [write]
    early = classify_epilogue_isa(_isa_excerpt(write, *_ISA_DIVISION, *store))
    assert early.mode_writes == [write]
    division = (*_ISA_DIVISION[:3], write, *_ISA_DIVISION[3:])
    inside = classify_epilogue_isa(_isa_excerpt(*division, *store))
    assert inside.mode_writes == [write] and inside.divisions == 1
    division = (*_ISA_DIVISION[:3], write, *_ISA_DIVISION[3:-1], restore)
    paired = classify_epilogue_isa(_isa_excerpt(*division, _ISA_DIVISION[-1], *store))
    assert paired.mode_writes == [write, restore] and paired.divisions == 1


# A-CVT-BF16: gfx942 has no BF16 conversion instruction, so ``fptrunc float to
# bfloat`` is an integer sequence the compiler chooses. From LLVM 19 it rounds
# to nearest even: ``x + 0x7fff + ((x >> 16) & 1)``, upper 16 bits kept; LLVM
# 18 and earlier kept the upper 16 bits of ``x`` unrounded. The ROCm 7.1.0
# compiler emits, per conversion,
#
#     v_bfe_u32     t, x, 16, 1
#     v_add3_u32    t, t, x, s[0x7fff]
#     v_or_b32      n, 0x400000, x        (quieted NaN)
#     v_cmp_u_f32   vcc, x, x
#     v_cndmask_b32 r, t, n, vcc
#
# and packs the upper halves of two results with ``v_perm_b32 .., s[0x7060302]``.
_ROUND_BIAS = "0x7fff"
_PACK_HIGH_HALVES = "0x7060302"


@dataclass
class Bf16Isa:
    """The FP32-to-BF16 conversions of one disassembled kernel: how many
    values go through the rounding sequence, and whether each pack of two
    upper halves takes rounded values ("rounded") or something else
    ("unrounded": a truncating or unrecognized conversion)."""

    rounded: int = 0
    packs: list[str] = field(default_factory=list)


def classify_bf16_conversions(text: str) -> Bf16Isa:
    model = Bf16Isa()
    scalars: dict[str, str] = {}
    low_bit: dict[int, int] = {}  # VGPR holding (x >> 16) & 1 -> VGPR x
    rounded: set[int] = set()
    for name, operands in _isa_instructions(text):
        if name in ("s_mov_b32", "s_movk_i32") and len(operands) == 2:
            scalars[operands[0]] = operands[1]
            continue
        if not name.startswith("v_") or not operands:
            continue
        destination = _isa_registers(operands[0]) or []
        sources = [(_isa_registers(item) or [None])[0] for item in operands[1:]]
        keep: set[int] = set()
        if name == "v_bfe_u32" and operands[2:] == ["16", "1"]:
            for register in destination:
                rounded.discard(register)
            low_bit[destination[0]] = sources[0]
            continue
        if name == "v_add3_u32":
            bias = [item for item in operands[1:] if scalars.get(item) == _ROUND_BIAS]
            values = [item for item in sources if item is not None]
            if len(bias) == 1 and any(
                low_bit.get(bit) == value
                for bit in values
                for value in values
                if bit != value
            ):
                model.rounded += 1
                keep.update(destination)
        elif name.startswith("v_cndmask_b32") and any(
            item in rounded for item in sources[:2]
        ):
            keep.update(destination)
        elif name.startswith("v_mov_b32") and sources[0] in rounded:
            keep.update(destination)
        elif name == "v_perm_b32" and scalars.get(operands[3]) == _PACK_HIGH_HALVES:
            both = all(item in rounded for item in sources[:2])
            model.packs.append("rounded" if both else "unrounded")
        for register in destination:
            low_bit.pop(register, None)
            if register in keep:
                rounded.add(register)
            else:
                rounded.discard(register)
    return model


_ISA_BF16_ROUNDED = (
    "s_movk_i32 s15, 0x7fff",
    "s_mov_b32 s17, 0x7060302",
    "v_bfe_u32 v36, v33, 16, 1",
    "v_add3_u32 v36, v36, v33, s15",
    "v_or_b32_e32 v37, 0x400000, v33",
    "v_cmp_u_f32_e32 vcc, v33, v33",
    "v_cndmask_b32_e32 v33, v36, v37, vcc",
    "v_bfe_u32 v34, v35, 16, 1",
    "v_add3_u32 v34, v34, v35, s15",
    "v_or_b32_e32 v38, 0x400000, v35",
    "v_cmp_u_f32_e32 vcc, v35, v35",
    "v_cndmask_b32_e32 v35, v34, v38, vcc",
    "v_perm_b32 v50, v35, v33, s17",
)


def test_bf16_classifier_reads_the_rounding_sequence_and_rejects_a_bare_shift() -> None:
    model = classify_bf16_conversions(_isa_excerpt(*_ISA_BF16_ROUNDED))
    assert (model.rounded, model.packs) == (2, ["rounded"])
    # The same pack of two values that never went through the sequence: the
    # truncating conversion of older compilers.
    truncating = [
        line
        for line in _ISA_BF16_ROUNDED
        if not line.startswith(("v_bfe", "v_add3", "v_or", "v_cmp", "v_cndmask"))
    ]
    model = classify_bf16_conversions(_isa_excerpt(*truncating))
    assert (model.rounded, model.packs) == (0, ["unrounded"])
    # A bias other than 0x7fff, or a low bit taken from another value, is not
    # the rounding sequence.
    for old, new in (
        ("s_movk_i32 s15, 0x7fff", "s_movk_i32 s15, 0x8000"),
        ("v_bfe_u32 v36, v33, 16, 1", "v_bfe_u32 v36, v40, 16, 1"),
        ("v_bfe_u32 v36, v33, 16, 1", "v_bfe_u32 v36, v33, 15, 1"),
    ):
        lines = [new if line == old else line for line in _ISA_BF16_ROUNDED]
        model = classify_bf16_conversions(_isa_excerpt(*lines))
        assert model.packs == ["unrounded"], new
    # A rounded value overwritten before the pack is not rounded any more.
    lines = list(_ISA_BF16_ROUNDED)
    lines.insert(-1, "v_mul_f32_e32 v33, v64, v48")
    assert classify_bf16_conversions(_isa_excerpt(*lines)).packs == ["unrounded"]


@dataclass(frozen=True)
class KernelModes:
    """The floating-point modes a wave starts with, from ``compute_pgm_rsrc1``
    of the kernel descriptor: FP32 and FP16/FP64 rounding (0 is
    round-to-nearest-even) and denormal handling (3 is no flush)."""

    round_32: int
    round_16_64: int
    denorm_32: int
    denorm_16_64: int


_ELF_SECTION = struct.Struct("<IIQQQQIIQQ")
_ELF_SYMBOL = struct.Struct("<IBBHQQ")
_ELF_SYMBOL_TABLES = (2, 11)  # SHT_SYMTAB, SHT_DYNSYM
_KERNEL_DESCRIPTOR_SIZE = 64
_COMPUTE_PGM_RSRC1_OFFSET = 48


def kernel_descriptor_modes(hsaco: bytes, kernel_name: str) -> KernelModes:
    """Read the mode fields of ``<kernel_name>.kd`` from a 64-bit little-endian
    ELF code object."""
    if hsaco[:6] != b"\x7fELF\x02\x01":
        raise ValueError("not a 64-bit little-endian ELF code object")
    section_offset = struct.unpack_from("<Q", hsaco, 0x28)[0]
    entry_size, count = struct.unpack_from("<HH", hsaco, 0x3A)
    sections = [
        _ELF_SECTION.unpack_from(hsaco, section_offset + index * entry_size)
        for index in range(count)
    ]
    wanted = f"{kernel_name}.kd".encode()
    for _, kind, _, _, offset, size, link, _, _, _ in sections:
        if kind not in _ELF_SYMBOL_TABLES:
            continue
        strings = sections[link][4]
        for at in range(offset, offset + size, _ELF_SYMBOL.size):
            name, _, _, index, value, length = _ELF_SYMBOL.unpack_from(hsaco, at)
            start = strings + name
            if hsaco[start : hsaco.index(b"\0", start)] != wanted:
                continue
            if length != _KERNEL_DESCRIPTOR_SIZE or not 0 < index < count:
                raise ValueError(f"malformed kernel descriptor symbol {wanted!r}")
            address, file_offset = sections[index][3], sections[index][4]
            descriptor = file_offset + value - address
            (rsrc1,) = struct.unpack_from(
                "<I", hsaco, descriptor + _COMPUTE_PGM_RSRC1_OFFSET
            )
            return KernelModes(
                round_32=rsrc1 >> 12 & 3,
                round_16_64=rsrc1 >> 14 & 3,
                denorm_32=rsrc1 >> 16 & 3,
                denorm_16_64=rsrc1 >> 18 & 3,
            )
    raise ValueError(f"no kernel descriptor {wanted!r} in the code object")


def _synthetic_code_object(kernel_name: str, rsrc1: int) -> bytes:
    """A minimal ELF64 holding only ``<kernel_name>.kd`` with ``rsrc1``."""
    strings = b"\0" + f"{kernel_name}.kd".encode() + b"\0"
    descriptor = bytearray(_KERNEL_DESCRIPTOR_SIZE)
    struct.pack_into("<I", descriptor, _COMPUTE_PGM_RSRC1_OFFSET, rsrc1)
    address = 0x800
    symbols = _ELF_SYMBOL.pack(0, 0, 0, 0, 0, 0) + _ELF_SYMBOL.pack(
        1, 0x11, 0, 1, address, _KERNEL_DESCRIPTOR_SIZE
    )
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    body = bytes(descriptor) + symbols + strings
    offsets = (64, 64 + len(descriptor), 64 + len(descriptor) + len(symbols))
    sections = (
        _ELF_SECTION.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        + _ELF_SECTION.pack(0, 1, 2, address, offsets[0], len(descriptor), 0, 0, 64, 0)
        + _ELF_SECTION.pack(0, 2, 0, 0, offsets[1], len(symbols), 3, 1, 8, 24)
        + _ELF_SECTION.pack(0, 3, 0, 0, offsets[2], len(strings), 0, 0, 1, 0)
    )
    struct.pack_into("<Q", header, 0x28, 64 + len(body))
    struct.pack_into("<HH", header, 0x3A, _ELF_SECTION.size, 4)
    return bytes(header) + body + sections


def test_kernel_descriptor_modes_decode_compute_pgm_rsrc1() -> None:
    """The four mode fields sit at bits 13:12, 15:14, 17:16 and 19:18. The
    first value is what the ROCm 7.1.0 compiler writes for a kernel without a
    denormal attribute: round-to-nearest, denormals kept, in both groups."""
    hsaco = _synthetic_code_object("kernel", 0x00AF031E)
    assert kernel_descriptor_modes(hsaco, "kernel") == KernelModes(0, 0, 3, 3)
    fields = 1 << 12 | 2 << 14 | 3 << 16 | 0 << 18
    hsaco = _synthetic_code_object("kernel", fields)
    assert kernel_descriptor_modes(hsaco, "kernel") == KernelModes(1, 2, 3, 0)
    with pytest.raises(ValueError, match="no kernel descriptor"):
        kernel_descriptor_modes(hsaco, "other")
    with pytest.raises(ValueError, match="ELF"):
        kernel_descriptor_modes(b"not an object", "kernel")


def test_isa_classifier_does_not_guess_an_unknown_element() -> None:
    store = list(_ISA_STORES[frozenset({0, 1})])
    store[1] = "v_cvt_f16_f32_e32 v33, v40"  # not a product of the reciprocal
    model = classify_epilogue_isa(_isa_excerpt(*_ISA_DIVISION, *store))
    assert model.stores == [("single", "single", "unknown", "double")]
    assert model.single_lanes is None


def _toolchain_unavailable(reason: str):
    """Skip the code-object contract for want of a tool, or fail where gfx942 is
    required: that lane must not lose the only mechanical check of the compiled
    code objects."""
    if os.environ.get("ROCKE_REQUIRE_GFX942") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_own_compiler() -> None:
    """Skip, or fail where gfx942 is required, unless COMGR loads a compiler of
    a known LLVM version and the kernel IR is lowered in that compiler's own
    flavor (``ROCKE_LLVM_FLAVOR`` unset, or naming it).

    Only these make a code object unavailable. Past them a compile error is
    the kernel's, and it fails the contract instead of skipping it."""
    from rocke.core.lower_llvm import _flavor_for_llvm, _resolve_llvm_flavor
    from rocke.runtime.comgr import loaded_compiler_info

    info = loaded_compiler_info()
    if info is None or info.llvm_version is None:
        found = "COMGR could not be loaded" if info is None else info.describe()
        _toolchain_unavailable(f"comgr toolchain unavailable: {found}")
    try:
        own = _flavor_for_llvm(info.llvm_version[0])
    except ValueError as error:
        _toolchain_unavailable(f"comgr toolchain unavailable: {error}")
    lowered = _resolve_llvm_flavor()
    if lowered != own:
        _toolchain_unavailable(
            f"the kernel IR is lowered for {lowered} but the loaded compiler "
            f"takes {own}: {info.describe()}"
        )


@cache
def _compiled_quick_case(case: DenseShape):
    """Compile one quick case's dispatch-resolved spec with the loaded COMGR,
    as ``run_attention_dense_torch`` does."""
    from kernels.gfx942.attention_dense import build_attention_dense
    from rocke.helpers.compile import compile_kernel

    _require_own_compiler()
    spec = _resolve_tuning_spec(case).kernel_spec
    return compile_kernel(
        build_attention_dense(spec, arch="gfx942"),
        arch="gfx942",
        backend="python",
        capture_ir_text=False,
    )


@cache
def _amdgpu_objdump() -> str | None:
    """An ``llvm-objdump`` that lists the AMDGPU backend, or None.

    ``find_llvm_tool`` falls back to ``PATH``, where a distro LLVM may lack the
    target. Such a tool cannot read these code objects, so the ISA contract
    treats it as missing instead of failing on its output. Looked up once per
    process."""
    from rocke.tests.llc import find_llvm_tool, lists_amdgpu_backend

    objdump = find_llvm_tool("llvm-objdump")
    if objdump is None:
        return None
    try:
        version = subprocess.run(
            [objdump, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if version.returncode != 0 or not lists_amdgpu_backend(version.stdout):
        return None
    return objdump


def _disassemble(hsaco: bytes, directory) -> str:
    objdump = _amdgpu_objdump()
    if objdump is None:
        _toolchain_unavailable(
            "no llvm-objdump with the AMDGPU backend: the ISA contract needs a "
            "disassembler"
        )
    path = directory / "kernel.hsaco"
    path.write_bytes(hsaco)
    completed = subprocess.run(
        [objdump, "-d", str(path)], capture_output=True, text=True, check=False
    )
    # A tool that predates the processor exits nonzero or prints words it
    # cannot decode; neither is a statement about the kernel.
    if completed.returncode != 0 or "<unknown>" in completed.stdout:
        _toolchain_unavailable(
            f"{objdump} cannot decode the gfx942 code object: "
            f"{completed.stderr.strip()[-400:] or 'undecoded instructions'}"
        )
    return completed.stdout


@pytest.mark.parametrize("case", QUICK_SHAPES, ids=lambda case: case.case_id)
def test_quick_code_objects_start_waves_in_the_modeled_modes(case: DenseShape) -> None:
    """A-MODE32 and A-CVT-F16, from the kernel descriptor of the compiled
    quick spec: every wave starts in round-to-nearest-even for FP32 and FP16,
    with FP16 denormals kept. No enclosure depends on the FP32 denormal mode
    (every FP32 subnormal of the path is enclosed with its flushed value), but
    the division expansion was evaluated with FP32 denormals kept, so the
    evidence for A-DIV applies only to a kernel that starts in that mode.

    The descriptor is read from the ELF itself, so this needs COMGR but no
    disassembler and still runs where ``llvm-objdump`` is missing."""
    artifact = _compiled_quick_case(case)
    modes = kernel_descriptor_modes(artifact.hsaco, artifact.kernel_name)
    print(f"{case.case_id}: modes={modes}")
    assert (modes.round_32, modes.round_16_64) == (0, 0), f"{_REDERIVE}: {modes}"
    assert modes.denorm_16_64 == 3, f"{_REDERIVE}: {modes}"
    assert modes.denorm_32 == 3, f"{_REDERIVE}: evaluate the division again: {modes}"


@pytest.mark.parametrize("case", QUICK_SHAPES, ids=lambda case: case.case_id)
def test_quick_code_objects_fuse_only_the_fp16_epilogue(
    case: DenseShape, tmp_path
) -> None:
    """ISA contract (A-F): compile the quick spec as ``run_attention_dense_torch``
    does, with the loaded COMGR, and disassemble it.

    - One division, and every FP32 FMA lies inside it: the denominator combine
      ``fadd(fmul(l, alpha), l_tile)`` is not fused.
    - Every ``v_fma_mix*`` of the kernel, the KV loop included, is
      ``R * o + 0`` feeding an output store; BF16 has none. FP16 stores are
      classified element by element. No mix instruction at all is a compiler
      that requires ``contract`` for the fold, and is accepted.
    - A named artifact (by HSACO digest) must show its recorded lanes. A digest
      no artifact names, from a compiler version an artifact was recorded
      under, raises a warning: the recorded digests are out of date for it.
    - A-DIV: the division expansion is the one evaluated on the device
      (``division_dataflow``). Another expansion has no evidence behind it.
    - A-MODE32, A-CVT-F16: no instruction, the division expansion included,
      writes the MODE register. The modes the waves start with are the
      descriptor contract above.
    - A-CVT-BF16: every ``fptrunc float to bfloat`` of the BF16 kernel is the
      round-to-nearest-even integer sequence, and every pack of upper halves
      takes rounded values. A compiler with another sequence is not guessed."""
    import hashlib

    from rocke.runtime.comgr import loaded_compiler_info

    artifact = _compiled_quick_case(case)
    disassembly = _disassemble(artifact.hsaco, tmp_path)
    model = classify_epilogue_isa(disassembly)
    digest = hashlib.sha256(artifact.hsaco).hexdigest()
    named = epilogue_artifact_for(case.dtype, digest)
    print(
        f"{case.case_id}: sha256={digest} artifact={named} mix={model.mix_sites} "
        f"single_lanes={model.single_lanes}"
    )
    if named is None:
        major = loaded_compiler_info().llvm_version[0]
        recorded = epilogue_artifacts_for_llvm(major)
        if recorded:
            warnings.warn(
                f"{case.case_id}: LLVM {major} compiled HSACO {digest}, which "
                f"none of {recorded} records; STRUCTURED_EPILOGUE_ARTIFACTS is "
                "out of date for this kernel and no record can name its artifact",
                stacklevel=1,
            )
    assert not model.mode_writes, f"{_REDERIVE}: {model.mode_writes}"
    assert model.divisions == 1, f"{_REDERIVE}: {model.divisions} divisions"
    assert model.division == _ISA_DIVISION_DATAFLOW, (
        f"{_REDERIVE}: the division expansion is not the evaluated one: "
        f"{model.division}"
    )
    assert not model.fma_outside_division, f"{_REDERIVE}: {model.fma_outside_division}"
    assert not model.unmodeled_mix, f"{_REDERIVE}: {model.unmodeled_mix}"
    assert len(model.stores) == _OUTPUT_CASTS // 4
    if case.dtype == "bf16":
        assert model.mix_sites == 0
        path = classify_dense_path(_lowered_quick_case(case), case.dtype)
        casts = path.counts["cast_probability"] + path.counts["cast_output"]
        conversions = classify_bf16_conversions(disassembly)
        print(f"{case.case_id}: bf16 conversions={conversions.rounded} of {casts}")
        assert conversions.rounded == casts, f"{_REDERIVE}: BF16 conversions"
        assert conversions.packs.count("rounded") * 2 == casts, conversions.packs
        assert "unrounded" not in conversions.packs, f"{_REDERIVE}: BF16 pack"
        return
    assert model.single_lanes is not None, model.stores
    assert model.mix_sites == len(model.single_lanes) * len(model.stores)
    if named is not None:
        assert model.single_lanes == STRUCTURED_EPILOGUE_ARTIFACTS[named].single_lanes


# ---------------------------------------------------------------------------
# Reachable-set primitives: exact FP32 rounding, intervals, exp2 candidates.
# ---------------------------------------------------------------------------

_RNG_SEED = 20261105


def _f32_bits(value: Fraction) -> int:
    return int(np.float32(float(value)).view(np.uint32))


def _f32_from_bits(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _members(interval: F32Interval, rng: np.random.Generator, count: int) -> np.ndarray:
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


def test_upward_spacing_is_defined_for_real_arguments() -> None:
    """A-E measures its bound in ``f32_quantum(2**x)``, and ``2**x`` is in
    general not an FP32 value. For a real ``r > 0`` the spacing is that of the
    FP32 cell ``[2**e, 2**(e+1))`` holding ``r``: the step from the largest
    FP32 value not above ``r`` to its successor. At a power of two it is the
    spacing above, twice the spacing below."""
    one = Fraction(1)
    reals = [
        Fraction(1, 3),
        one - Fraction(1, 1 << 30),
        one,
        one + Fraction(1, 1 << 30),
        Fraction(7, 5),
        Fraction(3, 1 << 130),
        F32_SMALLEST_SUBNORMAL / 3,
        Fraction(2981) + Fraction(1, 7),
    ]
    for value in reals:
        below = round_f32(value, "rd")
        assert f32_quantum(value) == next_up(below) - below, value
        exponent = max(floor_log2(value), -126)
        assert f32_quantum(value) == Fraction(2) ** (exponent - 23), value
    assert f32_quantum(Fraction(0)) == F32_SMALLEST_SUBNORMAL
    assert f32_quantum(one) == 2 * f32_quantum(one - Fraction(1, 1 << 30))


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


# From a gfx942 capture: row 207, head 0, folded, background dimensions 16..63.
_COUNTEREXAMPLE = {"o": 0x3A83E800, "r": 0x3E7BB1A4, "single": 0x0C0D, "double": 0x0C0E}


def test_epilogue_hull_holds_both_roundings_of_the_captured_counterexample() -> None:
    o = _f32_from_bit(_COUNTEREXAMPLE["o"])
    r = _f32_from_bit(_COUNTEREXAMPLE["r"])
    assert _single_rounded_fp16(o, r) == _COUNTEREXAMPLE["single"]
    assert _double_rounded(o, r, "fp16") == _COUNTEREXAMPLE["double"]
    points = F32Interval.point(Fraction(float(o))), F32Interval.point(
        Fraction(float(r))
    )
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
    factors = [np.float32(1.0) + np.float32(2.0**-23) * step for step in (1, 3, 5, 11)]
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
        points = F32Interval.point(Fraction(float(o))), F32Interval.point(
            Fraction(float(r))
        )
        double = _double_rounded(o, r, dtype)
        allowed = {double}
        if dtype == "fp16":
            allowed.add(_single_rounded_fp16(o, r))
            differing += len(allowed) == 2
        assert cast_single_or_double_hull(*points, dtype) == (
            min(allowed),
            max(allowed),
        )
    if dtype == "fp16":
        assert differing > 10, differing


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_epilogue_hull_encloses_every_member_of_its_intervals(dtype: str) -> None:
    rng = np.random.default_rng(_RNG_SEED)
    for _ in range(200):
        o = round_f32(
            Fraction(float(rng.uniform(1.0, 2.0)))
            * Fraction(2) ** int(rng.integers(-28, 4))
        )
        r = round_f32(
            Fraction(float(rng.uniform(1.0, 2.0)))
            * Fraction(2) ** int(rng.integers(-9, 0))
        )
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


# Series enclosures of e**x, ln 2 and 2**x.


def _decimal_power(value: Fraction, *, base: int | None = None) -> Fraction:
    """``e**value``, or ``base**value``, from a 110-digit decimal evaluation."""
    with localcontext() as context:
        context.prec = 110
        exponent = Decimal(value.numerator) / Decimal(value.denominator)
        if base is not None:
            exponent *= Decimal(base).ln()
        return Fraction(exponent.exp())


def test_series_enclosures_bracket_the_decimal_evaluations() -> None:
    """Both transcendental quantities of the model are enclosed by a series
    with an explicit tail bound, not by a count of digits. The enclosures
    hold the 110-digit decimal evaluations (a regression, not the proof), are
    narrower than ``2**-190`` relative, and are exact where the value is
    rational."""
    from ._structured_analytical import _LN2, _exp2_enclosure, _exp_enclosure

    narrow = Fraction(1, 1 << 190)
    assert _exp_enclosure(Fraction(0)) == (1, 1)
    logits = [Fraction(n, d) for n in (1, 2, 4, 8, 25) for d in (1, 2, 3, 7)]
    for logit in [*logits, *(-value for value in logits)]:
        low, high = _exp_enclosure(logit)
        assert 0 < low < _decimal_power(logit) < high, logit
        assert high - low < narrow * low, logit
        inverse_low, inverse_high = _exp_enclosure(-logit)
        assert low * inverse_low < 1 < high * inverse_high

    with localcontext() as context:
        context.prec = 110
        ln2 = Fraction(Decimal(2).ln())
    assert _LN2[0] < ln2 < _LN2[1] and _LN2[1] - _LN2[0] < narrow
    assert LOG2E_F32 == round_f32(1 / ln2, "rn")

    for power in (-149, -12, 0, 3):
        assert _exp2_enclosure(Fraction(power)) == (Fraction(2) ** power,) * 2
    arguments = [-Fraction(logit) * LOG2E_F32 for logit in (8, 4, 2, 1)]
    arguments += [Fraction(-1, 1 << 20), Fraction(-599, 2), Fraction(1, 3)]
    for argument in arguments:
        low, high = _exp2_enclosure(argument)
        decimal = _decimal_power(argument, base=2)
        # The decimal value carries its own rounding, far below the bracket.
        slack = decimal / 10**100
        assert low - slack < decimal < high + slack, argument
        assert 0 < low < high and high - low < narrow * low, argument
    low, high = _exp2_enclosure(f32(-1e30))
    assert 0 < low < high < F32_SMALLEST_SUBNORMAL


# The v_exp_f32 candidate set from the documented bound.


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


def _quick_id(case: StructuredCase) -> str:
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
    ids = [case.case_id for case in STRUCTURED_QUICK_CASES]
    assert len(ids) == len(set(ids)) == 2 * len(StructuredFixture)
    for case in STRUCTURED_QUICK_CASES:
        family = "walsh" if case.fixture is StructuredFixture.WALSH else "periodic"
        # The distinguished key of this row is 64 (even): the Walsh sign is +1.
        query = case.residue_shift + 64
        logits = tuple(row_logit(case, head, query) for head in range(4))
        assert logits == STRUCTURED_HEAD_LOGITS[family]


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
def test_structured_case_rejects_inexact_or_unsupported_configurations(
    kwargs, match
) -> None:
    arguments = {
        "base": QUICK_SHAPES[0],
        "fixture": "identity",
        "q_scales": (1, 1, 1, 1),
        "k_scales": (1, 1, 1, 1),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        StructuredCase(**arguments)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_generators_are_exact_and_match_the_score_model(case: StructuredCase) -> None:
    q, k = structured_query_key(case)
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
    for exposure in STRUCTURED_EXPOSURES:
        encode_bits(structured_inputs(case, exposure)[2], case.dtype)


def test_visited_tiles_follow_the_kernel_block_schedule() -> None:
    causal = next(case for case in STRUCTURED_QUICK_CASES if case.mask == "causal")
    window = next(case for case in STRUCTURED_QUICK_CASES if case.mask == "window")
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
    """Format-only rounding geometry over every positive finite value.

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


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_tier_b_is_nonempty_and_reports_its_widths(case: StructuredCase) -> None:
    """Tier B is nonempty everywhere. Its width is reported, not
    assumed: A-F lets an FP16 set reach a value either side of the IR's, so
    ``pytest -s`` prints the complete ``|B|`` histogram of each case."""
    histogram: Counter = Counter()
    for exposure in STRUCTURED_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        width = high.astype(np.int64) - low.astype(np.int64) + 1
        assert np.all(width >= 1), exposure
        histogram.update(width.ravel().tolist())
    print(f"{_quick_id(case)} |B| histogram: {dict(sorted(histogram.items()))}")


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_tier_b_lies_inside_tier_a(case: StructuredCase) -> None:
    """Tier A: every Tier B value is within 1 target ULP of the certified
    target at row-max coordinates, within 2 elsewhere, and exactly the target
    (+0) at structural zeros. Values are nonnegative, so bit distance is ULP
    distance."""
    for exposure in STRUCTURED_EXPOSURES:
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
    lanes = sorted(STRUCTURED_EPILOGUE_ARTIFACTS[artifact].single_lanes)
    return np.where(np.isin(_LANES, lanes), single, double)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_emulated_paths_lie_inside_tier_b(case: StructuredCase) -> None:
    """Every emulation lands inside Tier B: the IR-exact path, each
    named artifact's epilogue, single rounding at every FP16 site, and
    randomized-semantic members of the modeled path (summation tree, E
    candidate per call, MFMA rounding direction, epilogue rounding per FP16
    site). The emulator is a test device, never a verdict."""
    rng = np.random.default_rng(_RNG_SEED)
    for exposure in STRUCTURED_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        for row in _EMULATED_ROWS:
            for head in range(case.base.nhead_q):
                bounds = (low[0, row, head], high[0, row, head])
                double, single = epilogue_alternatives(case, exposure, row, head)
                realizations = [double, single]
                realizations += [
                    _artifact_bits(name, double, single)
                    for name in STRUCTURED_EPILOGUE_ARTIFACTS
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
        c
        for c in STRUCTURED_QUICK_CASES
        if c.dtype == "fp16" and c.fixture is StructuredFixture.IDENTITY
    )
    double, single = epilogue_alternatives(fp16, "folded", 207, 0)
    assert int(double[16]) == _COUNTEREXAMPLE["double"]
    assert int(single[16]) == _COUNTEREXAMPLE["single"]
    np.testing.assert_array_equal(emulate_row(fp16, "folded", 207, 0), double)
    for name in STRUCTURED_EPILOGUE_ARTIFACTS:
        bits = emulate_row(
            fp16, "folded", 207, 0, mode=EmulationMode.ARTIFACT_EXACT, artifact=name
        )
        np.testing.assert_array_equal(bits, _artifact_bits(name, double, single))
    bf16 = next(c for c in STRUCTURED_QUICK_CASES if c.dtype == "bf16")
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
        emulate_row(STRUCTURED_QUICK_CASES[0], "basis@0", 0, 0, **kwargs)


def test_epilogue_artifacts_name_their_code_objects() -> None:
    """Each named artifact gives a compiler, both quick code objects by digest
    and its single-rounded store elements; digests resolve back to it, and its
    LLVM major (the one its compiler string states) finds it again, so an
    unrecorded digest of a recorded compiler can be told apart."""
    digests = set()
    for name, artifact in STRUCTURED_EPILOGUE_ARTIFACTS.items():
        assert artifact.compiler and artifact.evidence
        assert f" {artifact.llvm_major}.0.0git " in artifact.compiler
        assert name in epilogue_artifacts_for_llvm(artifact.llvm_major)
        assert artifact.single_lanes <= {0, 1, 2, 3}
        assert sorted(dtype for dtype, _ in artifact.code_objects) == ["bf16", "fp16"]
        for dtype, digest in artifact.code_objects:
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
            assert epilogue_artifact_for(dtype, digest) == name
            digests.add(digest)
    assert len(digests) == 2 * len(STRUCTURED_EPILOGUE_ARTIFACTS)
    assert epilogue_artifact_for("fp16", "0" * 64) is None
    assert epilogue_artifacts_for_llvm(0) == ()


def _decimal_target_bits(dtype: str, exponent: int, item, digits: int) -> int:
    """The target rounded from a ``digits``-digit decimal evaluation: what
    certified a target before the interval certificate."""
    with localcontext() as context:
        context.prec = digits
        growth = (Decimal(item.logit.numerator) / Decimal(item.logit.denominator)).exp()
        value = Fraction(
            (item.m_c * growth + item.m_0) / (item.n_c * growth + item.n_0)
        ) * Fraction(2) ** exponent
    return target_bits(value.numerator, value.denominator, dtype)


def test_decimal_evaluations_reproduce_the_certified_targets() -> None:
    """Regression, not proof: every distinct target of the quick matrix,
    certified by interval containment, has the bits a 50-digit and a 70-digit
    decimal evaluation round to."""
    classes = {
        (case.dtype, exponent, item)
        for case in STRUCTURED_QUICK_CASES
        for exposure in STRUCTURED_EXPOSURES
        for exponent, item in tier_a_exposure(case, exposure)[2]
    }
    irrational = 0
    for dtype, exponent, item in classes:
        record = certify_target(dtype, exponent, item)
        assert record.bits == certified_target_bits(dtype, exponent, item)
        if record.rational:
            continue
        irrational += 1
        for digits in (50, 70):
            assert record.bits == _decimal_target_bits(dtype, exponent, item, digits)
    assert irrational > 100


@pytest.mark.parametrize("base", QUICK_SHAPES, ids=lambda base: base.case_id)
def test_zero_logits_give_the_uniform_distribution(base: DenseShape) -> None:
    """With every logit zero each legal key has probability ``1 / N``: a
    coordinate that selects ``m`` of a row's ``N`` legal keys has the target
    ``q_T(2**h m / N)``, counted here key by key, and Tier B holds it."""
    quick = next(case for case in STRUCTURED_QUICK_CASES if case.base == base)
    zero = StructuredCase(base, "identity", (0,) * 4, (0,) * 4, block_m=quick.block_m)
    for exposure in STRUCTURED_EXPOSURES:
        selected = exposure_values(base, exposure)[0, :, 0, :] != 0  # [Sk, D]
        expected = np.zeros(base.output_shape, dtype=np.uint16)
        for row in range(base.seqlen_q):
            first, last = legal_interval(base, row)
            counts = selected[first : last + 1].sum(axis=0)
            for head, exponent in enumerate(base.head_scale_exponents):
                expected[0, row, head] = [
                    target_bits(int(count) << exponent, last - first + 1, base.dtype)
                    for count in counts
                ]
        center = tier_a_exposure(zero, exposure)[0]
        np.testing.assert_array_equal(center, expected)
        low, high = tier_b_exposure(zero, exposure)
        assert np.all(_within(center, low, high)), exposure


def test_fp16_head_scale_does_not_commute_with_the_cast() -> None:
    """In the FP16 subnormal range ``q_T(2**h p) != 2**h q_T(p)`` for some
    quick-matrix coordinate, so head predictions must not be shifted copies
    of head 0. One counterexample is kept here."""
    case = next(
        c
        for c in STRUCTURED_QUICK_CASES
        if c.dtype == "fp16" and c.fixture is StructuredFixture.IDENTITY
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
    """With one distinguished key at c = -8 in a basis exposure, the first
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


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_weight_and_output_ranges_hold_on_the_quick_matrix(
    case: StructuredCase,
) -> None:
    """Range result: every weight and rescale factor is either masked-small
    (at most 2**-149) or at least 2**-12; every nonstructural stored value is
    at least 2**-20; FP16-subnormal outputs occur only at c = -8
    distinguished coordinates."""
    floor_weight, ceiling_masked = Fraction(1, 1 << 12), F32_SMALLEST_SUBNORMAL
    for row in range(256):
        for head in range(case.base.nhead_q):
            for step in row_trace(case, head, row).steps:
                for item in (step.alpha, *step.weights):
                    assert item.hi <= ceiling_masked or item.lo >= floor_weight
    for exposure in STRUCTURED_EXPOSURES:
        low, _ = tier_b_exposure(case, exposure)
        _, radius, _ = tier_a_exposure(case, exposure)
        nonstructural = radius > 0
        assert np.all(low[nonstructural] > 0)
        smallest = min(
            target_value(int(bits), case.dtype) for bits in low[nonstructural]
        )
        assert smallest >= Fraction(1, 1 << 20)
        if case.dtype != "fp16":
            continue
        for index in zip(*np.nonzero(nonstructural & (low < 0x0400))):
            _, row, head, dim = (int(value) for value in index)
            item = coordinate_class(case, exposure, row, head, dim)
            assert item.m_c > 0 and item.logit == -8, (exposure, row, head, dim)


@pytest.mark.parametrize("family", sorted(STRUCTURED_HEAD_SCALES))
def test_head_scales_keep_their_committed_separation(family: str) -> None:
    """Every wrong single Q-head or K-head read moves some coordinate by
    at least the committed margin in target ULPs, on both quick kernels."""
    q_scales, k_scales = STRUCTURED_HEAD_SCALES[family]
    margin = head_scale_margin(STRUCTURED_SEARCH_BASES, family, q_scales, k_scales)
    assert margin == STRUCTURED_HEAD_SCALE_MARGIN[family] > 0


# ---------------------------------------------------------------------------
# Premises of the Tier B theorem: the rank distance, target certification, the
# exactness ranges, the tile classes, the masked prefix and the reciprocal
# domain. Each is a finite property of the committed quick matrix.
# ---------------------------------------------------------------------------

_INFINITY_BITS = {"fp16": 0x7C00, "bf16": 0x7F80}
_FRACTION_MASK = {"fp16": 0x03FF, "bf16": 0x007F}
# Per format: (largest subnormal, smallest normal) and (last value of a binade,
# the power of two above it).
_FORMAT_EDGES = {
    "fp16": ((0x03FF, 0x0400), (0x3BFF, 0x3C00)),
    "bf16": ((0x007F, 0x0080), (0x3F7F, 0x3F80)),
}


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_target_ulp_distance_is_the_rank_difference(dtype: str) -> None:
    """The encoding of a nonnegative finite value is its rank: values increase
    with it, adjacent values are at distance one across a binade edge and
    across the subnormal/normal transition, and a negative value, an infinity
    or a NaN has no rank."""
    infinity = _INFINITY_BITS[dtype]
    values = [target_value(bits, dtype) for bits in range(infinity)]
    assert values[0] == 0
    assert all(low < high for low, high in itertools.pairwise(values))
    assert target_rank(0, dtype) == 0 and target_rank(infinity - 1, dtype) == (
        infinity - 1
    )
    for low, high in _FORMAT_EDGES[dtype]:
        assert d_ulp(low, high, dtype) == d_ulp(high, low, dtype) == 1
    assert d_ulp(0, 1, dtype) == 1 and d_ulp(5, 5, dtype) == 0
    assert ordered_distance((3, 7), (9, 12)) == d_ulp(7, 9, dtype) == 2
    assert ordered_distance((9, 12), (3, 7)) == 2
    assert ordered_distance((3, 9), (9, 12)) == 0
    for bits in (0x8000, 0x8001, infinity, infinity + 1, 0x10000, -1):
        with pytest.raises(ValueError, match="nonnegative finite"):
            target_rank(bits, dtype)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_every_enclosure_endpoint_has_a_rank(case: StructuredCase) -> None:
    """Every Tier B endpoint and Tier A center is a nonnegative finite value,
    so each bit difference taken on them (the Tier A radius, the separation
    gaps, the head-scale margin) is the rank distance."""
    infinity = _INFINITY_BITS[case.dtype]
    for exposure in STRUCTURED_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        center = tier_a_exposure(case, exposure)[0]
        for bits in (low, high, center):
            assert int(bits.max()) < infinity, exposure
        assert np.all(low <= high), exposure


def test_irrational_targets_lie_inside_one_rounding_cell() -> None:
    """An irrational target is certified by a rational interval that holds it
    and lies strictly between the two rounding boundaries adjacent to its
    bits: no count of digits and no library function enters. The boundaries
    are the midpoints to the two neighbors: twice as far above a power of two
    as below it, equidistant at the smallest normal (subnormals share the
    spacing of the first normal binade), and only the upper one exists at
    ``+0``. A rational target is rounded exactly and has no interval."""
    for dtype, ((subnormal, normal), (_, power)) in _FORMAT_EDGES.items():
        below, above = rounding_boundaries(power, dtype)
        center = target_value(power, dtype)
        assert above - center == 2 * (center - below)
        below, above = rounding_boundaries(normal, dtype)
        center = target_value(normal, dtype)
        assert below == (target_value(subnormal, dtype) + center) / 2
        assert above - center == center - below
        assert rounding_boundaries(0, dtype) == (None, target_value(1, dtype) / 2)
    # The FP16 normal/subnormal rounding boundary.
    assert rounding_boundaries(0x0400, "fp16")[0] == (
        Fraction(1, 1 << 14) - Fraction(1, 1 << 25)
    )

    classes = {
        (case.dtype, exponent, item)
        for case in STRUCTURED_QUICK_CASES
        for exposure in STRUCTURED_EXPOSURES
        for exponent, item in tier_a_exposure(case, exposure)[2]
    }
    irrational = binade_edges = subnormals = 0
    for dtype, exponent, item in classes:
        record = certify_target(dtype, exponent, item)
        if record.rational:
            assert record.margin is None and record.lower_boundary is None
            assert record.lower is None and record.upper is None
            proportional = item.m_c * item.n_0 == item.m_0 * item.n_c
            assert item.is_zero or item.logit == 0 or proportional
            continue
        irrational += 1
        assert record.bits > 0
        below, above = rounding_boundaries(record.bits, dtype)
        assert (record.lower_boundary, record.upper_boundary) == (below, above)
        assert below < record.lower < record.upper < above
        assert record.margin == min(record.lower - below, above - record.upper) > 0
        # The interval is far narrower than its cell, so the margin says how
        # close the target itself comes to a boundary.
        assert (record.upper - record.lower) * 10**50 < above - below
        fraction = record.bits & _FRACTION_MASK[dtype]
        binade_edges += fraction == 0 and record.bits > _FORMAT_EDGES[dtype][0][1]
        subnormals += dtype == "fp16" and record.bits < 0x0400
    # The matrix exercises both asymmetric geometries.
    assert irrational > 100 and binade_edges and subnormals


# Per family: the range of ``log2 |a_h b_h'|`` over every pairing of a Q head
# with a K head (wrong-head reads included), and what the 64-term contraction
# adds to it (64 equal products for Walsh, one product otherwise).
_SCORE_EXPONENTS = {"periodic": ((2, 8), 0), "walsh": ((-4, 0), 6)}


@pytest.mark.parametrize("family", sorted(STRUCTURED_HEAD_SCALES))
def test_score_exactness_ranges_of_the_committed_scales(family: str) -> None:
    """Score exactness ranges: every Q and K entry is ``2**e`` with
    ``-2 <= e <= 6``; every operand product is ``2**r`` with ``r`` in the
    family's range; every partial sum ``z 2**r``, ``|z| <= 64``, of a Walsh
    score is zero or a normal FP32 number; every accumulator is ``2**rho``
    with ``2 <= rho <= 8``; and its product with each modeled score scale is exact."""
    q_scales, k_scales = STRUCTURED_HEAD_SCALES[family]
    for value in (*q_scales, *k_scales):
        exponent = floor_log2(abs(value))
        assert abs(value) == Fraction(2) ** exponent and -2 <= exponent <= 6
    (lowest, highest), contraction = _SCORE_EXPONENTS[family]
    exponents = {floor_log2(abs(a * b)) for a in q_scales for b in k_scales}
    assert (min(exponents), max(exponents)) == (lowest, highest)
    if family == "walsh":
        # A-QK is stated for exactly this range and this many products.
        assert (lowest, highest) == A_QK_EXPONENTS
        assert A_QK_MAGNITUDE == 1 << contraction

    assert SCORE_SCALE == LOG2E_F32 / 8
    assert Fraction(1, 8) <= SCORE_SCALE < Fraction(1, 4)
    scales = (SCORE_SCALE, round_f32(f32(1.0) * LOG2E_F32), f32(1.0 / 8.0))
    for exponent in exponents:
        unit = Fraction(2) ** exponent
        if family == "walsh":
            for count in range(-A_QK_MAGNITUDE, A_QK_MAGNITUDE + 1):
                partial = count * unit
                assert is_f32(partial)
                assert count == 0 or abs(partial) >= F32_SMALLEST_NORMAL
        rho = exponent + contraction
        assert 2 <= rho <= 8
        accumulator = Fraction(2) ** rho
        for scale in scales:
            assert round_f32(accumulator * scale) == accumulator * scale
            assert round_f32(-accumulator * scale) == -accumulator * scale
        magnitude = accumulator * SCORE_SCALE
        assert Fraction(2) ** (rho - 3) <= magnitude < Fraction(2) ** (rho - 2)

    # The generated tensors realize exactly these accumulators, for the correct
    # heads and for each wrong-head read.
    for case in STRUCTURED_QUICK_CASES:
        if case.dtype != "fp16" or fixture_family(case.fixture) != family:
            continue
        inputs = [structured_query_key(case)]
        inputs += [
            _mutated_query_key(case, mutation)[:2]
            for mutation in (
                StructuredMutation.Q_HEAD_ROTATE,
                StructuredMutation.K_HEAD_ROTATE,
            )
        ]
        for q, k in inputs:
            for value in _dot_products(q, k) - {Fraction(0)}:
                rho = floor_log2(abs(value))
                assert abs(value) == Fraction(2) ** rho
                assert lowest + contraction <= rho <= highest + contraction


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_pv_products_are_exact_and_in_range(case: StructuredCase) -> None:
    """PV product range: a legal weight lies in ``[2**-12, 1 + 2**-23]`` and its
    PV product
    ``q_T(p) 2**h`` in ``[2**-12, 2**3]``, a normal FP32 number. A masked
    weight gives the product ``+0``, or ``2**h`` in a fully masked leading
    tile, where the weight is ``E(0)`` and casts to one."""
    from ._structured_analytical import _scaled_cast

    unit_hull = exp2_interval(Fraction(0))
    floor_weight = Fraction(1, 1 << 12)
    zero = F32Interval.point(Fraction(0))
    for row in range(case.base.seqlen_q):
        classes = tile_classes(case, row)
        first, last = legal_interval(case.base, row)
        for head in range(case.base.nhead_q):
            exponent = case.head_scale_exponents[head]
            for tile_class, step in zip(classes, row_trace(case, head, row).steps):
                for offset, weight in enumerate(step.weights):
                    product = _scaled_cast(case.dtype, weight, exponent)
                    if first <= step.tile * case.block_n + offset <= last:
                        assert floor_weight <= weight.lo and weight.hi <= unit_hull.hi
                        assert floor_weight <= product.lo and product.hi <= 8
                    elif tile_class is TileClass.MASKED_PREFIX:
                        assert weight == unit_hull
                        assert product == F32Interval.point(Fraction(2) ** exponent)
                    else:
                        assert weight.hi <= F32_SMALLEST_SUBNORMAL and product == zero


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_mutated_pv_products_stay_fp32_normal(case: StructuredCase) -> None:
    """A mutated kernel's weights leave the correct range (a wrong scale gives
    weights near ``2**-93``), so its enclosure asserts what exactness needs:
    every nonzero PV product endpoint is a normal FP32 number."""
    from ._structured_analytical import _scaled_cast

    for mutation in applicable_mutations(case):
        for row in derived_rows(case, STRUCTURED_SEPARATION_ROWS, mutation):
            for head in range(case.base.nhead_q):
                exponent = case.head_scale_exponents[head]
                for step in row_trace(case, head, row, mutation).steps:
                    for weight in set(step.weights):
                        product = _scaled_cast(case.dtype, weight, exponent)
                        for value in (product.lo, product.hi):
                            assert value == 0 or value >= F32_SMALLEST_NORMAL


_L = TileClass.LEGAL
_M0 = TileClass.MASKED_PREFIX
_M1 = TileClass.MASKED_AFTER_LEGAL


def _expected_tiles(case: StructuredCase, row: int) -> tuple[tuple[int, ...], tuple]:
    """The visited tiles of ``row`` and their classes, written out for
    the two quick kernels (causal ``block_m = 128``; window 64, ``block_m = 64``)."""
    block = row // 64
    if case.mask == "causal":
        tiles = (0, 1) if block < 2 else (0, 1, 2, 3)
        return tiles, tuple(_L if tile <= block else _M1 for tile in tiles)
    if block == 0:
        return (0,), (_L,)
    return (block - 1, block), (_M0 if row % 64 == 63 else _L, _L)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_tile_classes_match_the_masked_prefix_topology(case: StructuredCase) -> None:
    """Masked-prefix topology over every row and exposure: at most one fully
    masked leading
    tile, present exactly on rows 127, 191 and 255 of the window kernel; there
    every folded coordinate, and every coordinate of the basis exposure on
    that tile, selects one of its keys, and no other coordinate does."""
    prefix_rows = []
    for row in range(case.base.seqlen_q):
        tiles, classes = _expected_tiles(case, row)
        assert visited_tiles(case, row) == tiles, row
        assert tile_classes(case, row) == classes, row
        assert row_scope(case, row, None) is None
        prefix = masked_prefix(case, row)
        assert len(prefix) <= 1
        if not prefix:
            continue  # nothing to select: the count is zero by construction
        prefix_rows.append(row)
        for exposure in STRUCTURED_EXPOSURES:
            selected = {
                prefix_selected_keys(case, exposure, row, dim)
                for dim in range(case.base.head_dim)
            }
            on_prefix = exposure in ("folded", f"basis@{64 * prefix[0]}")
            assert selected == ({1} if on_prefix else {0}), (row, exposure)
    assert prefix_rows == ([127, 191, 255] if case.mask == "window" else [])


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_tile_sums_follow_the_tile_class(case: StructuredCase) -> None:
    """The ``gamma`` bound is applied only to tiles with normal-scale terms. A
    legal tile holds a weight of at least ``2**-12``; a fully masked leading
    tile holds 64 copies of ``E(0)``; a fully masked tile after a legal one
    holds only weights in ``[0, 2**-149]``, its sum is enclosed by
    ``[0, 2**-143]`` without the bound, and the combine returns ``RN(l alpha)``."""
    masked = F32Interval(Fraction(0), F32_SMALLEST_SUBNORMAL)
    unit_hull = exp2_interval(Fraction(0))
    seen: Counter = Counter()
    for row in range(case.base.seqlen_q):
        classes = tile_classes(case, row)
        for head in range(case.base.nhead_q):
            trace = row_trace(case, head, row)
            for index, (tile_class, step) in enumerate(zip(classes, trace.steps)):
                seen[tile_class] += 1
                total = tile_sum(tile_class, step.weights)
                if tile_class is _M1:
                    assert set(step.weights) == {masked} and step.alpha == unit_hull
                    assert total == F32Interval(Fraction(0), Fraction(1, 1 << 143))
                    assert trace.denominators[index] == mul_rn(
                        trace.denominators[index - 1], step.alpha
                    )
                    continue
                assert total == sum_any_order(step.weights)
                if tile_class is _M0:
                    assert set(step.weights) == {unit_hull}
                else:
                    assert any(w.lo >= Fraction(1, 1 << 12) for w in step.weights)
    causal = case.mask == "causal"
    assert seen[_M1] == (128 * case.base.nhead_q if causal else 0)
    assert seen[_M0] == (0 if causal else 3 * case.base.nhead_q)


def test_tile_sum_refuses_a_normal_weight_in_a_masked_tile() -> None:
    masked = F32Interval(Fraction(0), F32_SMALLEST_SUBNORMAL)
    one = F32Interval.point(Fraction(1))
    assert tile_sum(_M1, [masked] * 64) == F32Interval(
        Fraction(0), Fraction(1, 1 << 143)
    )
    with pytest.raises(StructuredScopeError, match="normal-scale weight"):
        tile_sum(_M1, [masked, one])
    # The looser exponential used only for attribution: still explicit.
    loose = F32Interval(Fraction(0), 2 * F32_SMALLEST_SUBNORMAL)
    assert tile_sum(_M1, [loose] * 64) == F32Interval(
        Fraction(0), Fraction(1, 1 << 142)
    )
    # A mutation's extra normal-scale key puts the tile back under the bound.
    assert tile_sum(_M1, [masked] * 64, [one]) == sum_any_order([*[masked] * 64, one])


@pytest.mark.parametrize(
    "case",
    [case for case in STRUCTURED_QUICK_CASES if case.mask == "window"],
    ids=_quick_id,
)
def test_masked_prefix_leaves_only_a_subnormal_residue(case: StructuredCase) -> None:
    """The masked-prefix residue, recomputed from the propagated endpoints on
    the three prefix
    rows: (0) after the prefix ``l`` is the ``gamma_63`` sum of 64 ``E(0)``
    terms and a selecting coordinate holds ``2**h``; (1) the rescale by
    ``alpha`` in ``[0, 2**-149]`` leaves less than ``2**-142``, which the legal
    tile absorbs; (2) a coordinate with no legal product keeps a residue of at
    most ``2**(h-149)`` and stores ``+0``; (3) a later legal product ``pi``
    gives ``pi`` or its FP32 successor."""
    from ._structured_analytical import _scaled_cast

    step = Fraction(1, 1 << 23)
    prefix_sum = F32Interval(
        round_f32(64 * (1 - step) * (1 - gamma(63)), "ru"),
        round_f32(64 * (1 + step) * (1 + gamma(63)), "rd"),
    )
    assert 63 < prefix_sum.lo and prefix_sum.hi < 65
    masked = F32Interval(Fraction(0), F32_SMALLEST_SUBNORMAL)
    for row in (127, 191, 255):
        assert tile_classes(case, row) == (_M0, _L)
        for head in range(case.base.nhead_q):
            exponent = case.head_scale_exponents[head]
            scale = Fraction(2) ** exponent
            trace = row_trace(case, head, row)
            legal = trace.steps[1]
            assert trace.denominators[0] == prefix_sum
            assert legal.alpha == masked
            assert mul_rn(prefix_sum, legal.alpha).hi < Fraction(1, 1 << 142)
            assert trace.denominator == tile_sum(_L, legal.weights)
            residue = F32Interval(Fraction(0), scale * F32_SMALLEST_SUBNORMAL)
            for exposure in STRUCTURED_EXPOSURES:
                low, high = tier_b_exposure(case, exposure)
                for dim in range(case.base.head_dim):
                    if not prefix_selected_keys(case, exposure, row, dim):
                        continue
                    (_, after_prefix), (rescaled, final) = accumulator_steps(
                        case, exposure, row, head, dim
                    )
                    assert after_prefix == F32Interval.point(scale)
                    assert rescaled == residue
                    if exposure == "folded":
                        product = _scaled_cast(case.dtype, legal.weights[dim], exponent)
                        assert product.lo >= Fraction(1, 1 << 12)
                        assert final == F32Interval(product.lo, next_up(product.hi))
                    else:
                        assert final == residue
                        assert low[0, row, head, dim] == high[0, row, head, dim] == 0


def test_reciprocal_is_modeled_only_on_the_a_div_domain() -> None:
    low, high = A_DIV_DOMAIN
    assert (low, high) == (Fraction(1, 2), Fraction(512))
    inside = F32Interval(low, next_down(high))
    assert rcp_a_div(inside) == rcp_rn(inside)
    for outside in (
        F32Interval(next_down(low), Fraction(1)),
        F32Interval(Fraction(1), high),
        F32Interval.point(Fraction(2981)),
    ):
        with pytest.raises(StructuredScopeError, match="A-DIV domain"):
            rcp_a_div(outside)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_denominators_stay_in_the_a_div_domain(case: StructuredCase) -> None:
    """From the first legal tile on, every denominator of the correct
    quick matrix lies in ``[1/2, 2**9)``, the domain of A-DIV, so the
    reciprocal is at most 2."""
    low, high = A_DIV_DOMAIN
    for row in range(case.base.seqlen_q):
        first_legal = tile_classes(case, row).index(_L)
        for head in range(case.base.nhead_q):
            trace = row_trace(case, head, row)
            for denominator in trace.denominators[first_legal:]:
                assert low <= denominator.lo and denominator.hi < high
            assert trace.reciprocal == rcp_rn(trace.denominator)
            assert trace.reciprocal.hi <= 2


def _underived_rows(
    case: StructuredCase, mutation: StructuredMutation
) -> dict[int, str]:
    """The sampled rows whose mutated enclosure is not derived, and why.

    Excluding the last key leaves causal row 0 without a key, which the
    masked-prefix argument does not cover. A phantom logit-0 key added to a row
    whose only key has
    logit -8 has weight ``e**8``, which takes the denominator out of the A-DIV
    domain."""
    if mutation is StructuredMutation.EXCLUDE_LAST_KEY:
        return {0: "fully masked leading tiles"}
    if (
        mutation is StructuredMutation.DENOMINATOR_PLUS_ONE
        and case.fixture is StructuredFixture.IDENTITY
    ):
        return {0: "A-DIV domain"}
    return {}


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_mutated_rows_outside_the_derivation_make_no_claim(
    case: StructuredCase,
) -> None:
    """A mutated enclosure must establish the masked-prefix bound and the A-DIV
    domain for its
    own recurrence. On the sampled rows every mutation does, except the two
    rows listed above: those raise instead of producing a set, and the
    separation and the mutated analysis leave them out."""
    for mutation in applicable_mutations(case):
        expected = _underived_rows(case, mutation)
        refused = {
            row: reason
            for row in STRUCTURED_SEPARATION_ROWS
            if (reason := row_scope(case, row, mutation)) is not None
        }
        assert set(refused) == set(expected), mutation.value
        for row, reason in refused.items():
            assert expected[row] in reason
            with pytest.raises(StructuredScopeError):
                for head in range(case.base.nhead_q):
                    tier_b_row(case, "basis@0", row, head, mutation=mutation)
        kept = tuple(row for row in STRUCTURED_SEPARATION_ROWS if row not in refused)
        assert derived_rows(case, STRUCTURED_SEPARATION_ROWS, mutation) == kept
        if not refused:
            continue
        observation = _tier_b_observation(case)
        with pytest.raises(ValueError, match="at least one query"):
            analyze_structured_policy(
                case, observation, mutation=mutation, rows=tuple(refused)
            )
        result = analyze_structured_policy(
            case, observation, mutation=mutation, rows=(*refused, 1)
        )
        judged = sum(group.coordinates for group in result.groups["basis@0"].values())
        assert judged == case.base.nhead_q * case.base.head_dim


def test_row_trace_fails_closed_on_two_masked_leading_tiles() -> None:
    """The correct semantics are never filtered: a shape with two fully masked
    leading tiles has no enclosure until its residue bound is derived."""
    window = next(
        case
        for case in STRUCTURED_QUICK_CASES
        if case.mask == "window" and case.fixture is StructuredFixture.IDENTITY
    )
    wide = dataclasses.replace(window, block_m=128)
    assert visited_tiles(wide, 255) == (1, 2, 3)
    assert tile_classes(wide, 255) == (_M0, _M0, _L)
    assert len(masked_prefix(wide, 255)) == 2
    assert derived_rows(wide, (254, 255), None) == (254, 255)
    with pytest.raises(StructuredScopeError, match="2 fully masked leading tiles"):
        row_trace(wide, 0, 255)
    assert tile_classes(wide, 254) == (_M0, _L, _L)
    assert row_scope(wide, 254, None) is None


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_exposed_atoms_hold_at_most_one_nonzero_product(case: StructuredCase) -> None:
    """What reduces an MFMA call to one faithful addition, and what makes the
    scores exact.

    PV: for every exposure, KV head and output dimension, V is nonzero at no
    more than one key of a tile (hence of each group of eight keys an MFMA
    call covers), that key is the one the model reads, and its value is
    ``2**h``. No V entry has its sign bit set, so every zero PV product is
    ``+0`` and a structural coordinate keeps the accumulator ``+0`` (A-MFMA).
    QK: a periodic score has at most one nonzero product over its eight
    atoms; every atom of a Walsh score has eight, each ``+-2**r`` with ``r``
    in the A-QK range, 64 per score (the A-QK magnitude)."""
    tiles = case.base.seqlen_k // case.block_n
    scales = np.ldexp(1.0, np.array(case.head_scale_exponents)).astype(np.float32)
    for exposure in STRUCTURED_EXPOSURES:
        v = structured_inputs(case, exposure)[2][0]  # [Sk, Hkv, D]
        assert not np.signbit(v).any(), exposure
        nonzero = v != 0
        per_tile = nonzero.reshape(tiles, case.block_n, *nonzero.shape[1:]).sum(axis=1)
        per_call = nonzero.reshape(tiles * 8, 8, *nonzero.shape[1:]).sum(axis=1)
        assert per_tile.max() <= 1 and per_call.max() <= 1, exposure
        for tile in range(tiles):
            for dim in range(case.base.head_dim):
                key = exposure_key(case, exposure, dim, tile)
                selected = nonzero[tile * 64 : (tile + 1) * 64, :, dim]
                if key is None:
                    assert not selected.any()
                else:
                    assert selected[key - tile * 64].all() and selected.sum() == len(
                        scales
                    )
                    np.testing.assert_array_equal(v[key, :, dim], scales)

    q, k = structured_query_key(case)
    for head in range(case.base.nhead_q):
        products = q[0, :, head, None, :] * k[0, None, :, head, :]  # [Sq, Sk, D]
        per_atom = (products != 0).reshape(*products.shape[:2], 8, 8).sum(axis=-1)
        if case.fixture is StructuredFixture.WALSH:
            assert np.all(per_atom == 8)
            assert products.shape[-1] == A_QK_MAGNITUDE
            (magnitude,) = np.unique(np.abs(products))
            exponent = floor_log2(Fraction(float(magnitude)))
            assert Fraction(float(magnitude)) == Fraction(2) ** exponent
            assert A_QK_EXPONENTS[0] <= exponent <= A_QK_EXPONENTS[1]
        else:
            assert per_atom.max() <= 1 and per_atom.sum(axis=-1).max() <= 1


# ---------------------------------------------------------------------------
# Mutation separation.
# ---------------------------------------------------------------------------


def test_mutation_applicability_follows_fixture_and_mask() -> None:
    covered = set()
    for case in STRUCTURED_QUICK_CASES:
        mutations = applicable_mutations(case)
        assert len(mutations) == len(set(mutations))
        covered.update(mutations)
        assert (StructuredMutation.INVERSE_SHIFT in mutations) == (
            case.fixture is StructuredFixture.CYCLIC
        )
        assert (StructuredMutation.WALSH_SIGN in mutations) == (
            case.fixture is StructuredFixture.WALSH
        )
        assert (StructuredMutation.EXCLUDE_LAST_KEY in mutations) == (
            case.mask == "causal"
        )
        assert (StructuredMutation.INCLUDE_OLD_WINDOW_KEY in mutations) == (
            case.mask == "window"
        )
    assert covered == set(StructuredMutation)


def _mutated_query_key(
    case: StructuredCase, mutation: StructuredMutation
) -> tuple[np.ndarray, np.ndarray, range]:
    """The Q and K a mutated kernel reads, and the keys they are defined for."""
    q, k = structured_query_key(case)
    keys = range(case.base.seqlen_k)
    if mutation is StructuredMutation.Q_HEAD_ROTATE:
        q = np.roll(q, -1, axis=2)
    elif mutation is StructuredMutation.K_HEAD_ROTATE:
        k = np.roll(k, -1, axis=2)
    elif mutation is StructuredMutation.K_ROW_PLUS_ONE:
        k, keys = k[:, 1:], range(case.base.seqlen_k - 1)
    elif mutation is StructuredMutation.K_ROW_MINUS_ONE:
        k, keys = k[:, :-1], range(1, case.base.seqlen_k)
    elif mutation is StructuredMutation.INVERSE_SHIFT:
        # Transpose each 64 x 64 key-by-dimension block of the periodic K.
        block = np.swapaxes(k[:, :64], 1, 3)
        k = np.tile(block, (1, case.base.seqlen_k // 64, 1, 1))
    elif mutation is StructuredMutation.WALSH_SIGN:
        signs = np.where(np.arange(case.base.seqlen_k) & 1, -1.0, 1.0)
        k = k * signs[None, :, None, None].astype(np.float32)
    elif mutation.value.startswith("omit_walsh_block_"):
        block = int(mutation.value.rsplit("_", 1)[1])
        q = q.copy()
        q[..., 16 * block : 16 * (block + 1)] = 0.0
    return q, k, keys


_SCORE_MUTATIONS = (
    StructuredMutation.Q_HEAD_ROTATE,
    StructuredMutation.K_HEAD_ROTATE,
    StructuredMutation.K_ROW_PLUS_ONE,
    StructuredMutation.K_ROW_MINUS_ONE,
    StructuredMutation.INVERSE_SHIFT,
    StructuredMutation.WALSH_SIGN,
    StructuredMutation.OMIT_WALSH_BLOCK_0,
    StructuredMutation.OMIT_WALSH_BLOCK_1,
    StructuredMutation.OMIT_WALSH_BLOCK_3,
)


@pytest.mark.parametrize(
    "case",
    [case for case in STRUCTURED_QUICK_CASES if case.dtype == "fp16"],
    ids=lambda case: case.fixture.value,
)
def test_mutated_scores_are_the_mutated_dot_products(case: StructuredCase) -> None:
    """The closed-form mutated scores equal the exact QK dot products of the
    inputs a mutated kernel would read (heads rotated, K rows off by one, the
    cyclic K transposed, the Walsh sign dropped or one 16-dimension block of
    the contraction skipped). Scale mutations change only the score scale."""
    for mutation in set(_SCORE_MUTATIONS) & set(applicable_mutations(case)):
        q, k, keys = _mutated_query_key(case, mutation)
        dots = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64))[
            0
        ]
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
        StructuredMutation.OMIT_SQRT_D: round_f32(
            f32(1.0) * f32(attention_dense.LOG2E)
        ),
        StructuredMutation.EXP_ENCODING: f32(1.0 / 8.0),
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
    """Every QK dot product the correct kernel forms on the quick matrix
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
    for case in STRUCTURED_QUICK_CASES:
        correct |= _dot_products(*structured_query_key(case))
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
            assert round_f32(product - row_max) == round_f32(
                round_f32(product) - row_max
            )
    assert rounded == {
        StructuredMutation.OMIT_WALSH_BLOCK_0,
        StructuredMutation.OMIT_WALSH_BLOCK_1,
        StructuredMutation.OMIT_WALSH_BLOCK_3,
    }


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_every_mutation_is_separated_or_classified(case: StructuredCase) -> None:
    """Every applicable mutation's Tier B is disjoint from the correct
    Tier B at some coordinate (``G(m) > 0``), or the cell is listed in
    ``STRUCTURED_UNSEPARATED`` with its owner. ``G`` is taken over the sampled rows,
    a lower bound of the full gap; ``pytest -s`` prints each ``G`` and its
    coordinate."""
    unexplained = []
    for mutation in applicable_mutations(case):
        separation = mutation_separation(case, mutation)
        print(format_separation(case, separation))
        label = STRUCTURED_UNSEPARATED.get((case.case_id, mutation))
        if (separation.gap > 0) == (label is not None):
            unexplained.append(f"{mutation.value}: G={separation.gap}, label={label}")
    assert not unexplained, unexplained


def test_unseparated_cells_are_proven_and_owned() -> None:
    """Each listed cell has a proven zero gap at every coordinate and, when it
    names another fixture as owner, that fixture separates the mutation on
    the same kernel.

    For ``OMIT_RESCALE`` the zero is proven, not sampled: if every ``alpha``
    after the first visited tile admits 1, the mutated recurrence
    (``alpha = 1``) is a member of the correct one, and interval propagation is
    inclusion-monotone, so the mutated Tier B lies inside the correct Tier B."""
    assert STRUCTURED_UNSEPARATED
    cases = {case.case_id: case for case in STRUCTURED_QUICK_CASES}
    for (case_id, mutation), label in STRUCTURED_UNSEPARATED.items():
        assert label in SEPARATION_LABELS
        case = cases[case_id]
        assert mutation in applicable_mutations(case)
        if mutation is StructuredMutation.OMIT_RESCALE:
            for row in range(case.base.seqlen_q):
                for head in range(case.base.nhead_q):
                    for step in row_trace(case, head, row).steps[1:]:
                        assert Fraction(1) in step.alpha, (case_id, row, head)
        else:
            assert mutation_separation(case, mutation, rows=None).gap == 0
        if label == "owned by another fixture":
            owners = [
                other
                for other in STRUCTURED_QUICK_CASES
                if other.base == case.base
                and mutation in applicable_mutations(other)
                and (other.case_id, mutation) not in STRUCTURED_UNSEPARATED
            ]
            assert any(
                mutation_separation(other, mutation).gap > 0 for other in owners
            ), (case_id, mutation.value)


_MUTANT_EXPOSURES = ("basis@0", "folded")
_MUTANT_ROWS = (0, 63, 64, 127, 128, 255)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_mutant_paths_lie_inside_their_tier_b(case: StructuredCase) -> None:
    """Containment for the mutated semantics: a mutated kernel's IR-exact and
    randomized-semantic realizations (A-F epilogue choices included) lie in the
    mutated Tier B. Without this, a disjoint pair of enclosures would not imply
    that the mutant's output is rejected. A row whose mutated enclosure is not
    derived makes no claim and is left out."""
    rng = np.random.default_rng(_RNG_SEED)
    modes = (
        {"mode": EmulationMode.IR_EXACT},
        {"mode": EmulationMode.RANDOMIZED_SEMANTIC, "rng": rng},
    )
    for mutation in applicable_mutations(case):
        for exposure in _MUTANT_EXPOSURES:
            for row in derived_rows(case, _MUTANT_ROWS, mutation):
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
# Analyzer policy and failure taxonomy.
# ---------------------------------------------------------------------------


def _tier_b_observation(case: StructuredCase, end: int = 0) -> Observation:
    """Synthetic outputs at one end (0 low, 1 high) of every Tier B set."""
    return Observation(
        tuple(
            (exposure, tier_b_exposure(case, exposure)[end])
            for exposure in STRUCTURED_EXPOSURES
        )
    )


def _with_bits(
    observation: Observation,
    exposure: str,
    coordinate: tuple[int, int, int, int],
    bits: int,
) -> Observation:
    exposures = []
    for name, values in observation.exposures:
        values = values.copy()
        if name == exposure:
            values[coordinate] = bits
        exposures.append((name, values))
    return Observation(tuple(exposures))


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_structural_zeros_have_tier_b_plus_zero(case: StructuredCase) -> None:
    """Every structural coordinate (masked, no target) has Tier B
    ``{+0}``. The quick matrix has no trailing basis coordinate and no keyless
    row, so those sentinels cannot occur here."""
    for exposure in STRUCTURED_EXPOSURES:
        low, high = tier_b_exposure(case, exposure)
        radius = tier_a_exposure(case, exposure)[1]
        structural = radius == 0
        assert np.all(low[structural] == 0) and np.all(high[structural] == 0)
    assert all(offset + 63 < case.base.seqlen_k for offset in (0, 64, 128, 192))
    assert all(
        legal_interval(case.base, row)[1] >= legal_interval(case.base, row)[0]
        for row in range(case.base.seqlen_q)
    )


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_structural_zeros_need_only_a_coarse_reciprocal_bound(
    case: StructuredCase,
) -> None:
    """A structural zero does not rest on A-DIV. On the block, tile and window
    edge rows, which include the three rows with a fully masked leading tile,
    the accumulator of every structural coordinate is exactly ``+0`` or, after
    that tile, the residue of at most ``2**(h-149)``. Either stores ``+0`` for
    every reciprocal below ``STRUCTURAL_ZERO_RECIPROCAL_BOUND``, correctly
    rounded or not. FP16 has no such tile here and would allow ``2**120``."""
    bf16 = case.dtype == "bf16"
    bound = STRUCTURAL_ZERO_RECIPROCAL_BOUND if bf16 else Fraction(1 << 120)
    coarse = F32Interval(Fraction(0), next_down(bound))
    scales = [Fraction(2) ** exponent for exponent in case.head_scale_exponents]
    residues = 0
    for exposure in STRUCTURED_EXPOSURES:
        radius = tier_a_exposure(case, exposure)[1][0]
        for row in STRUCTURED_SEPARATION_ROWS:
            for head, dim in np.argwhere(radius[row] == 0):
                head, dim = int(head), int(dim)
                steps = accumulator_steps(case, exposure, row, head, dim)
                final = steps[-1][1]
                assert final.lo == 0
                assert final.hi <= scales[head] * F32_SMALLEST_SUBNORMAL
                selected = prefix_selected_keys(case, exposure, row, dim)
                assert (final.hi > 0) == (selected == 1)
                residues += selected
                stored = cast_single_or_double_hull(final, coarse, case.dtype)
                assert stored == (0, 0), (exposure, row, head, dim)
    prefix_rows = 3 if case.mask == "window" else 0
    assert residues == prefix_rows * case.base.nhead_q * case.base.head_dim


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_every_tier_b_value_passes(case: StructuredCase) -> None:
    """Synthetic outputs at either end of every Tier B set pass, and every
    two-value set is counted with the end that was observed."""
    for end in (0, 1):
        result = analyze_structured_policy(case, _tier_b_observation(case, end))
        assert not result.violations, format_structured_result(result)
        assert result.labels[StructuredLabel.PASS.value] == 5 * int(
            np.prod(case.output_shape)
        )
        assert result.ambiguous_high == (result.ambiguous if end else 0)


_LABEL_CASE = next(
    case
    for case in STRUCTURED_QUICK_CASES
    if case.dtype == "fp16" and case.fixture is StructuredFixture.IDENTITY
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


def _unmatched(exposure: str, query: int, head: int, dim: int, bits: int) -> bool:
    """No single violated assumption stores ``bits`` at this coordinate, so its
    label comes from the heuristics."""
    return matching_violation(_LABEL_CASE, exposure, query, head, dim, bits) is None


def _label_input(label: StructuredLabel) -> tuple[str, tuple[int, int, int, int], int]:
    """An exposure, coordinate and stored bits that must carry ``label``."""
    from ._structured_analytical import _genuine_addition

    if label is StructuredLabel.STRUCTURAL:
        where, *_ = _find("basis@0", range(256), lambda q, h, d, lo, hi, c, r: r == 0)
        return "basis@0", where, 0x0001
    if label is StructuredLabel.NEGATIVE:
        where, low, *_ = _find(
            "basis@0", range(256), lambda q, h, d, lo, hi, c, r: r > 0
        )
        return "basis@0", where, low | 0x8000
    if label is StructuredLabel.INVALID:
        where, *_ = _find("basis@0", range(256), lambda q, h, d, lo, hi, c, r: r > 0)
        return "basis@0", where, 0x7E00  # the adapter's unwritten-output prefill
    if label is StructuredLabel.A_CVT_F16:  # heuristic: near an FP16-subnormal target
        where, _, high, *_ = _find(
            "basis@0",
            range(256),
            lambda q, h, d, lo, hi, c, r: r > 0
            and c < 0x0400
            and _unmatched("basis@0", q, h, d, hi + 1),
        )
        return "basis@0", where, high + 1
    if label is StructuredLabel.A_F:  # the store of the exact product, truncated
        for query in range(1, 64):
            for head in range(4):
                low, _ = tier_b_row(_LABEL_CASE, "basis@0", query, head)
                stored, _ = epilogue_alternatives(
                    _LABEL_CASE, "basis@0", query, head, StructuredViolation.F_TRUNCATED
                )
                for dim in np.nonzero(stored < low)[0]:
                    return "basis@0", (0, query, head, int(dim)), int(stored[dim])
        raise AssertionError("no truncated FP16 store leaves Tier B")
    if label is StructuredLabel.A_MFMA:
        where, _, high, *_ = _find(
            "folded",
            range(64, 256),
            lambda q, h, d, lo, hi, c, r: _normal(c, r)
            and _genuine_addition(_LABEL_CASE, q, h, d)
            and _unmatched("folded", q, h, d, hi + 1),
        )
        return "folded", where, high + 1
    if label is StructuredLabel.A_E:

        def loosened(q, h, d, lo):
            relaxed_low, relaxed_high = _relaxed("basis@0", q, h)
            return int(relaxed_low[d] if relaxed_low[d] < lo else relaxed_high[d])

        def widened(q, h, d, lo, hi, c, r):
            relaxed_low, relaxed_high = _relaxed("basis@0", q, h)
            return (
                _normal(c, r)
                and (relaxed_low[d] < lo or relaxed_high[d] > hi)
                and _unmatched("basis@0", q, h, d, loosened(q, h, d, lo))
            )

        where, low, _, _, _ = _find("basis@0", range(128, 160), widened)
        return "basis@0", where, loosened(where[1], where[2], where[3], low)

    def outside(step: int):
        def predicate(q, h, d, lo, hi, c, r):
            bits = c + (1 if step == 1 else r + 1)
            return (
                _normal(c, r)
                and lo == hi == c
                and bits > _relaxed("basis@0", q, h)[1][d]
                and _unmatched("basis@0", q, h, d, bits)
            )

        return predicate

    if label is StructuredLabel.PATH_MODEL:  # one ULP outside an unambiguous Tier B
        where, _, _, center, _ = _find("basis@0", range(1, 64), outside(1))
        return "basis@0", where, center + 1
    assert label is StructuredLabel.NUMERICAL
    where, _, _, center, radius = _find("basis@0", range(1, 64), outside(2))
    return "basis@0", where, center + radius + 1


# No single-step violation of the first three assumptions leaves Tier B on the
# quick matrix (``STRUCTURED_UNOBSERVED_VIOLATIONS``), so no stored value can
# carry them; the BF16 conversion label belongs to the BF16 kernel, and the
# label case is FP16 (the assumption-failure contract produces it there).
_UNPRODUCIBLE_LABELS = (
    StructuredLabel.A_DIV,
    StructuredLabel.A_QK,
    StructuredLabel.A_MODE32,
    StructuredLabel.A_CVT_BF16,
)


@pytest.mark.parametrize(
    "label",
    [label for label in STRUCTURED_FAILURE_LABELS if label not in _UNPRODUCIBLE_LABELS],
    ids=lambda label: label.name,
)
def test_each_failure_label_is_produced_and_fails(label: StructuredLabel) -> None:
    """Policy: one synthetic coordinate per taxonomy label fails with that
    label and nothing else, starting from outputs that pass. One ULP outside an
    unambiguous Tier B fails as "path model"; the exact-zero sentinels (a
    nonzero masked coordinate, a negative value, NaN, the unwritten prefill)
    fail as structural, negative and invalid; a truncated FP16 store fails as
    assumption A-F."""
    exposure, coordinate, bits = _label_input(label)
    observation = _with_bits(
        _tier_b_observation(_LABEL_CASE), exposure, coordinate, bits
    )
    result = analyze_structured_policy(_LABEL_CASE, observation, rows=(coordinate[1],))
    failures = {name: count for name, count in result.labels.items() if name != "pass"}
    assert failures == {**dict.fromkeys(failures, 0), label.value: 1}, failures
    assert result.first[label.value] == (exposure, coordinate)
    assert result.violations and result.violations[0].startswith(label.value)


def test_negative_zero_and_infinity_are_not_zero_or_finite() -> None:
    structural, *_ = _find("basis@0", range(256), lambda q, h, d, lo, hi, c, r: r == 0)
    for bits, label in (
        (0x8000, StructuredLabel.NEGATIVE),
        (0x7C00, StructuredLabel.INVALID),
    ):
        observation = _with_bits(
            _tier_b_observation(_LABEL_CASE), "basis@0", structural, bits
        )
        result = analyze_structured_policy(
            _LABEL_CASE, observation, rows=(structural[1],)
        )
        assert result.labels[label.value] == 1


def test_mode_labels_need_a_value_the_mode_can_produce() -> None:
    """At an FP16-subnormal target a flushed ``+0`` or a nearby value is
    A-CVT-F16 (a conversion that does not keep subnormals), but a far value
    (such as a wrong scale stores) is numerical; a far folded value is
    numerical, not A-MFMA."""
    exposure, coordinate, _ = _label_input(StructuredLabel.A_CVT_F16)
    _, _, _, center, radius = _find(
        exposure,
        range(coordinate[1], coordinate[1] + 1),
        lambda q, h, d, lo, hi, c, r: (0, q, h, d) == coordinate,
    )
    folded, folded_at, _ = _label_input(StructuredLabel.A_MFMA)
    folded_center = int(tier_a_exposure(_LABEL_CASE, folded)[0][folded_at])
    for where, at, bits, label in (
        (exposure, coordinate, 0x0000, StructuredLabel.A_CVT_F16),
        (exposure, coordinate, center + radius + 64, StructuredLabel.NUMERICAL),
        (folded, folded_at, folded_center + 64, StructuredLabel.NUMERICAL),
    ):
        observation = _with_bits(_tier_b_observation(_LABEL_CASE), where, at, bits)
        result = analyze_structured_policy(_LABEL_CASE, observation, rows=(at[1],))
        assert result.labels[label.value] == 1, (where, bits, dict(result.labels))


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in STRUCTURED_QUICK_CASES
        if (case.dtype, case.fixture)
        in {("fp16", StructuredFixture.LATE_MAX), ("bf16", StructuredFixture.WALSH)}
    ],
    ids=_quick_id,
)
def test_mutated_analysis_rejects_correct_outputs(case: StructuredCase) -> None:
    """The analyzer-side mutations the GPU lane runs reject a correct output on
    the sampled rows, for every mutation not listed as inseparable."""
    observation = _tier_b_observation(case)
    for mutation in applicable_mutations(case):
        if (case.case_id, mutation) in STRUCTURED_UNSEPARATED:
            continue
        result = analyze_structured_policy(
            case, observation, mutation=mutation, rows=STRUCTURED_SEPARATION_ROWS
        )
        assert result.violations, mutation.value


def test_records_cover_every_coordinate_and_serialize() -> None:
    case = _LABEL_CASE
    observation = _tier_b_observation(case)
    control = analyze_structured_policy(case, observation)
    mutated = analyze_structured_policy(
        case,
        observation,
        mutation=StructuredMutation.OMIT_SQRT_D,
        rows=STRUCTURED_SEPARATION_ROWS,
    )
    separation = mutation_separation(case, StructuredMutation.OMIT_SQRT_D)
    code_object = {"compiler": "test", "hsaco_sha256": "0" * 64}
    records = structured_records(
        case,
        "attention_gfx942_dense",
        {"none": control, "omit_sqrt_d": mutated},
        deterministic=True,
        diagnostics=structured_diagnostics(case, observation, artifact="rocm-7.13.0"),
        separations={"omit_sqrt_d": separation},
        code_object=code_object,
    )
    for record in records:
        json.dumps(record)
    groups = [r for r in records if r["record_type"] == "rocke_structured_group"]
    control_groups = [r for r in groups if r["mutation"] == "none"]
    per_exposure = Counter()
    for record in control_groups:
        per_exposure[record["exposure"]] += record["coordinates"]
    assert per_exposure == dict.fromkeys(
        STRUCTURED_EXPOSURES, int(np.prod(case.output_shape))
    )
    summaries = {
        r["mutation"]: r
        for r in records
        if r["record_type"] == "rocke_structured_summary"
    }
    assert not summaries["none"]["rejected"]
    assert summaries["omit_sqrt_d"]["rejected"]
    assert summaries["omit_sqrt_d"]["cpu_gap"] == separation.gap > 0
    (diagnostics,) = [
        r for r in records if r["record_type"] == "rocke_structured_diagnostics"
    ]
    assert diagnostics["assumptions"] == list(STRUCTURED_ASSUMPTIONS)
    assert diagnostics["deterministic"] is True
    assert diagnostics["code_object"] == code_object
    assert diagnostics["epilogue_artifact"] == "rocm-7.13.0"
    assert diagnostics["mfma_rn"] <= diagnostics["mfma_additions"]
    assert 0 < diagnostics["ir_exact"] <= diagnostics["path_compared"]
    assert 0 < diagnostics["artifact_exact"] <= diagnostics["path_compared"]
    unnamed = structured_diagnostics(case, observation, rows=(0,))
    assert unnamed.artifact is None and unnamed.artifact_exact is None


# ---------------------------------------------------------------------------
# Assumption-failure contract: a violated assumption is reported, never
# absorbed.
# ---------------------------------------------------------------------------

# Block, tile and window edges, and row 207, where the two FP16 epilogue
# roundings differ. Setting the environment variable below sweeps every row.
_VIOLATION_ROWS = (0, 63, 64, 127, 128, 207, 255)
_FULL_VIOLATION_SWEEP = "ROCKE_STRUCTURED_FULL_VIOLATION_SWEEP"


def _violation_observation(
    case: StructuredCase, violation: StructuredViolation, rows: tuple[int, ...]
) -> Observation:
    """What the IR-exact path stores on ``rows`` with one assumption violated."""
    exposures = []
    for exposure in STRUCTURED_EXPOSURES:
        bits = np.zeros(case.output_shape, dtype=np.uint16)
        for row in rows:
            for head in range(case.base.nhead_q):
                bits[0, row, head] = epilogue_alternatives(
                    case, exposure, row, head, violation
                )[0]
        exposures.append((exposure, bits))
    return Observation(tuple(exposures))


def test_every_violation_names_its_assumption() -> None:
    from ._structured_analytical import _VIOLATION_LABEL

    assert set(_VIOLATION_LABEL) == set(StructuredViolation)
    named = set()
    for violation, label in _VIOLATION_LABEL.items():
        # "assumption A-CVT-F16" names the violations "a_cvt_f16_...".
        assumption = label.value.removeprefix("assumption ")
        assert assumption.startswith("A-"), label.value
        assert violation.value.startswith(
            assumption.lower().replace("-", "_") + "_"
        ), violation.value
        named.add(f"{assumption} v1")
    # Every named assumption has a violating variant, and only those.
    assert named == set(STRUCTURED_ASSUMPTIONS)
    assert STRUCTURED_UNOBSERVED_VIOLATIONS < set(StructuredViolation)
    covered = set()
    for case in STRUCTURED_QUICK_CASES:
        violations = applicable_violations(case)
        assert len(violations) == len(set(violations))
        assert (StructuredViolation.QK_UP in violations) == (
            case.fixture is StructuredFixture.WALSH
        )
        fp16 = case.dtype == "fp16"
        assert (StructuredViolation.F_TRUNCATED in violations) == fp16
        assert (StructuredViolation.CVT_F16_TRUNCATING in violations) == fp16
        assert (StructuredViolation.CVT_BF16_TRUNCATING in violations) == (not fp16)
        covered.update(violations)
    assert covered == set(StructuredViolation)


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_violated_assumptions_fail_and_are_never_absorbed(case: StructuredCase) -> None:
    """For each named assumption, the IR-exact path with that one primitive
    replaced by an excluded value (``StructuredViolation``):

    - Tier B and the exponential candidate sets are the same before and after
      the violating output is judged: no tolerance absorbs it;
    - every coordinate where the stored value leaves Tier B fails, labeled
      with a named assumption, and the capture audit reports the same
      coordinates as unexplained;
    - a violation that leaves Tier B nowhere is listed in
      ``STRUCTURED_UNOBSERVED_VIOLATIONS``: the stored bits cannot show it, which says
      nothing about the assumption.

    ``pytest -s`` prints the number of coordinates each violation moves out of
    Tier B."""
    from ._structured_analytical import _VIOLATION_LABEL, _VIOLATION_REACH

    every_row = tuple(range(case.base.seqlen_q))
    rows = every_row if os.environ.get(_FULL_VIOLATION_SWEEP) else _VIOLATION_ROWS
    before = {
        exposure: [bits.copy() for bits in tier_b_exposure(case, exposure)]
        for exposure in STRUCTURED_EXPOSURES
    }
    candidates = {x: exp2_candidates(x) for x in (Fraction(0), MASKED_SCORE)}
    conversion_labels = {
        StructuredLabel.A_F,
        StructuredLabel.A_CVT_F16,
        StructuredLabel.A_CVT_BF16,
    } & {_VIOLATION_LABEL[violation] for violation in applicable_violations(case)}
    judged = (
        len(rows) * len(STRUCTURED_EXPOSURES) * case.base.nhead_q * case.base.head_dim
    )
    for violation in applicable_violations(case):
        observation = _violation_observation(case, violation, rows)
        outside = set()
        reach = 0
        for exposure, stored in observation.exposures:
            low, high = (bits.astype(np.int64) for bits in before[exposure])
            for row in rows:
                bits = stored[0, row].astype(np.int64)
                distance = np.maximum(low[0, row] - bits, bits - high[0, row])
                reach = max(reach, int(distance.max()))
                outside.update(
                    (exposure, row, int(head), int(dim))
                    for head, dim in zip(*np.nonzero(distance > 0))
                )
        result = analyze_structured_policy(case, observation, rows=rows)
        failures = {
            name: count
            for name, count in result.labels.items()
            if count and name != StructuredLabel.PASS.value
        }
        print(
            f"{_quick_id(case)} {violation.value}: {len(outside)} of {judged} "
            f"coordinates leave Tier B (reach {reach}); labels {failures}"
        )
        assert reach <= _VIOLATION_REACH, violation.value
        assert (not outside) == (
            violation in STRUCTURED_UNOBSERVED_VIOLATIONS
        ), violation.value
        assert sum(failures.values()) == len(outside), violation.value
        assert bool(result.violations) == bool(outside)
        if outside:
            # A truncated FP16 store is A-F; where a truncating conversion
            # stores the same bits it is attributed to A-F as well, else to
            # the conversion assumption of the kernel's format.
            assert _VIOLATION_LABEL[violation].value in failures, violation.value
            assert set(failures) <= {label.value for label in conversion_labels}
        audit = audit_case(
            case, dict(observation.exposures), artifact=_AUDIT_ARTIFACT, rows=rows
        )
        assert set(audit.outside_tier_b) == outside, violation.value
        assert outside <= {where for where, _ in audit.unexplained}, violation.value
    for exposure in STRUCTURED_EXPOSURES:
        for cached, saved in zip(tier_b_exposure(case, exposure), before[exposure]):
            np.testing.assert_array_equal(cached, saved)
    assert {x: exp2_candidates(x) for x in candidates} == candidates


# ---------------------------------------------------------------------------
# Replayable capture audit (exact epilogue union, position rule, Tier B).
# ---------------------------------------------------------------------------

_AUDIT_CASE = _LABEL_CASE  # FP16 identity: holds the captured counterexample
_AUDIT_ROWS = (0, 64, 207)
_AUDIT_ARTIFACT = "rocm-7.13.0"


def _write_capture(
    directory, case: StructuredCase = _AUDIT_CASE
) -> dict[str, np.ndarray]:
    """A synthetic capture whose audited rows store what the named artifact
    stores; the other rows are not audited."""
    arrays = {
        exposure: np.zeros(case.output_shape, dtype=np.uint16)
        for exposure in STRUCTURED_EXPOSURES
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
    assert item.coordinates == len(_AUDIT_ROWS) * len(STRUCTURED_EXPOSURES) * 4 * 64
    assert set(item.ir_mismatches) == {
        ("folded", 207, 0, dim) for dim in range(16, 64) if dim % 4 in (0, 1)
    }
    single_lanes = STRUCTURED_EPILOGUE_ARTIFACTS[_AUDIT_ARTIFACT].single_lanes
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


def _structured_entry(dtype: str, **overrides) -> StructuredCalibrationEntry:
    spec = STRUCTURED_QUICK_SPECS[dtype]
    arguments = {
        "policy": STRUCTURED_POLICY,
        "assumptions": STRUCTURED_ASSUMPTIONS,
        "case_ids": tuple(
            c.case_id for c in STRUCTURED_QUICK_CASES if c.dtype == dtype
        ),
        "spec": spec,
    }
    arguments.update(overrides)
    return StructuredCalibrationEntry(**arguments)


def test_calibration_entry_has_no_numeric_width() -> None:
    """An entry names the policy, the assumption versions, the cases
    and the fingerprint; there is no field a measurement could widen."""
    assert [item.name for item in dataclasses.fields(StructuredCalibrationEntry)] == [
        "policy",
        "assumptions",
        "case_ids",
        "spec",
    ]
    _structured_entry("fp16")
    with pytest.raises(ValueError, match="policy"):
        _structured_entry("fp16", policy="ulp_budget")
    with pytest.raises(ValueError, match="assumptions"):
        _structured_entry("fp16", assumptions=("A-E v2", "A-MFMA v1"))
    with pytest.raises(ValueError, match="case_ids"):
        _structured_entry("fp16", case_ids=())
    with pytest.raises(TypeError):
        _structured_entry("fp16", ulp_budget=1)


def test_calibration_requires_every_assumption_version() -> None:
    """Tier B is sound under eight named assumptions, each an antecedent of
    the soundness theorem. An entry that omits one, lists another version of
    one, or lists them in another order is refused; so is an entry that still
    names the rounding and conversion assumptions as one."""
    assert STRUCTURED_ASSUMPTIONS == (
        "A-E v1",
        "A-MFMA v1",
        "A-QK v1",
        "A-DIV v1",
        "A-F v1",
        "A-MODE32 v1",
        "A-CVT-F16 v1",
        "A-CVT-BF16 v1",
    )
    assert len(STRUCTURED_ASSUMPTION_STATEMENTS) == len(STRUCTURED_ASSUMPTIONS)
    for version, statement in zip(
        STRUCTURED_ASSUMPTIONS, STRUCTURED_ASSUMPTION_STATEMENTS
    ):
        assert statement.startswith(f"{version}: ")
    assert EPILOGUE_ASSUMPTION.startswith("A-F v1:")
    _structured_entry("fp16", assumptions=STRUCTURED_ASSUMPTIONS)
    refused = [
        STRUCTURED_ASSUMPTIONS[:index] + STRUCTURED_ASSUMPTIONS[index + 1 :]
        for index in range(len(STRUCTURED_ASSUMPTIONS))
    ]
    refused += [
        STRUCTURED_ASSUMPTIONS[:index]
        + (version.replace(" v1", " v2"),)
        + STRUCTURED_ASSUMPTIONS[index + 1 :]
        for index, version in enumerate(STRUCTURED_ASSUMPTIONS)
    ]
    refused.append(tuple(reversed(STRUCTURED_ASSUMPTIONS)))
    # The entry as ratified before A-QK, A-DIV and the mode assumptions were
    # named, and the one that named a single A-MODE.
    refused.append(("A-E v1", "A-MFMA v1", "A-F v1"))
    refused.append(
        ("A-E v1", "A-MFMA v1", "A-QK v1", "A-DIV v1", "A-F v1", "A-MODE v1")
    )
    for assumptions in refused:
        with pytest.raises(ValueError, match="assumptions"):
            _structured_entry("fp16", assumptions=assumptions)
    for entry in STRUCTURED_CALIBRATION.values():
        assert entry.assumptions == STRUCTURED_ASSUMPTIONS


def test_observations_never_change_the_enclosures() -> None:
    """Judging hostile outputs leaves Tier B and the exponential candidate
    sets exactly as derived, and the cached enclosures are read-only."""
    case = _LABEL_CASE
    before = {
        exposure: [bits.copy() for bits in tier_b_exposure(case, exposure)]
        for exposure in STRUCTURED_EXPOSURES
    }
    candidates = {x: exp2_candidates(x) for x in (Fraction(0), MASKED_SCORE)}
    for bits in (0x3C00, 0x7E00, 0x0000, 0x8001):
        observation = Observation(
            tuple(
                (exposure, np.full(case.output_shape, bits, dtype=np.uint16))
                for exposure in STRUCTURED_EXPOSURES
            )
        )
        result = analyze_structured_policy(
            case, observation, rows=STRUCTURED_SEPARATION_ROWS
        )
        assert result.violations
    for exposure in STRUCTURED_EXPOSURES:
        for cached, saved in zip(tier_b_exposure(case, exposure), before[exposure]):
            assert not cached.flags.writeable
            np.testing.assert_array_equal(cached, saved)
    assert {x: exp2_candidates(x) for x in candidates} == candidates


def test_calibration_scope_is_all_quick_cases_or_none() -> None:
    """No partial ratification: the calibrated case set is exactly the quick
    matrix, one entry per dtype holding that dtype's cases and that dtype's
    dispatch fingerprint."""
    quick = {case.case_id for case in STRUCTURED_QUICK_CASES}
    calibrated = {
        case_id
        for entry in STRUCTURED_CALIBRATION.values()
        for case_id in entry.case_ids
    }
    assert calibrated == quick
    assert set(STRUCTURED_CALIBRATION) == {
        ("gfx942", "attention_gfx942_dense", base.dtype) for base in QUICK_SHAPES
    }
    for (_, _, dtype), entry in STRUCTURED_CALIBRATION.items():
        assert set(entry.case_ids) == {
            case.case_id for case in STRUCTURED_QUICK_CASES if case.dtype == dtype
        }
    for (_, _, dtype), entry in STRUCTURED_CALIBRATION.items():
        assert entry.spec == STRUCTURED_QUICK_SPECS[dtype]
        assert (
            entry.policy == STRUCTURED_POLICY
            and entry.assumptions == STRUCTURED_ASSUMPTIONS
        )


@pytest.mark.parametrize("case", STRUCTURED_QUICK_CASES, ids=_quick_id)
def test_require_structured_calibration_fails_closed(case: StructuredCase) -> None:
    spec = STRUCTURED_QUICK_SPECS[case.dtype]
    key = ("gfx942", "attention_gfx942_dense", case.dtype)
    if key not in STRUCTURED_CALIBRATION:
        with pytest.raises(
            AssertionError, match="no structured-logit calibration"
        ) as raised:
            require_structured_calibration(
                "gfx942", "attention_gfx942_dense", case, spec
            )
        assert "ROCKE_ORACLE_FREE_CALIBRATION_OUT" in str(raised.value)
    table = {key: _structured_entry(case.dtype)}
    entry = require_structured_calibration(
        "gfx942", "attention_gfx942_dense", case, spec, table=table
    )
    assert entry is table[key]
    drifted = dataclasses.replace(spec, block_m=spec.block_m // 2)
    with pytest.raises(AssertionError, match="block_m: calibrated"):
        require_structured_calibration(
            "gfx942", "attention_gfx942_dense", case, drifted, table=table
        )
    # The adapter resolves dispatch's default spec and pins no knob, so a
    # retune onto the persistent grid arrives here as a fingerprint mismatch.
    regridded = dataclasses.replace(spec, persistent=True)
    with pytest.raises(AssertionError, match="persistent: calibrated False"):
        require_structured_calibration(
            "gfx942", "attention_gfx942_dense", case, regridded, table=table
        )
    narrow = {key: _structured_entry(case.dtype, case_ids=("other",))}
    with pytest.raises(AssertionError, match=f"case {case.case_id} not in"):
        require_structured_calibration(
            "gfx942", "attention_gfx942_dense", case, spec, table=narrow
        )


def _structured_gpu_lane_tree() -> ast.Module:
    path = Path(__file__).with_name("test_structured_analytical_gfx942.py")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_structured_gpu_lane_scope_and_gate() -> None:
    """The structured-logit GPU lane runs exactly ``STRUCTURED_QUICK_CASES``, gates once
    through the
    require-aware ``current_device_arch_in(("gfx942",), ...)``, never skips
    otherwise, and ends in ``require_structured_calibration``."""
    tree = _structured_gpu_lane_tree()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [_dotted(call.func) for call in calls]
    parametrized = [
        call for call, name in zip(calls, names) if name.endswith("mark.parametrize")
    ]
    assert parametrized
    for call in parametrized:
        argvalues = call.args[1]
        assert (
            isinstance(argvalues, ast.Name) and argvalues.id == "STRUCTURED_QUICK_CASES"
        )
    skip_calls = {"skip", "skipif", "importorskip", "xfail"}
    skips = [name for name in names if name.split(".")[-1] in skip_calls]
    assert not skips
    gates = [
        call for call, name in zip(calls, names) if name == "current_device_arch_in"
    ]
    assert len(gates) == 1 and ast.literal_eval(gates[0].args[0]) == ("gfx942",)
    assert "require_structured_calibration" in names


# ---------------------------------------------------------------------------
# Subject adapter widening.
# ---------------------------------------------------------------------------


def test_adapter_accepts_structured_cases_through_the_shape_protocol() -> None:
    for case in STRUCTURED_QUICK_CASES:
        _validate_supported_case(case)
        shape = _shape(case)
        assert (shape.batch, shape.seqlen_q, shape.nhead_q, shape.head_dim) == (
            case.output_shape
        )
    parameter = inspect.signature(run_dense_case).parameters["scale_override"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("base", QUICK_SHAPES, ids=lambda base: base.case_id)
def test_every_fixture_resolves_to_the_calibrated_dispatch_identity(
    base: DenseShape,
) -> None:
    """Dispatch sees only the shape, dtype and mask: every fixture on a quick
    shape resolves to the gfx942 dense candidate and to the fingerprint the
    calibration entry records for that dtype."""
    for case in (c for c in STRUCTURED_QUICK_CASES if c.base == base):
        subject = resolve_dense_subject(case)
        assert subject.case_id == case.case_id
        assert subject.candidate_name == "attention_gfx942_dense"
        assert subject.fingerprint == STRUCTURED_QUICK_SPECS[base.dtype]


def test_adapter_resolves_the_default_spec_with_no_knob_pinned() -> None:
    """The check must launch the kernel dispatch ships, so the adapter may not
    choose a configuration itself: it resolves ``gfx942_dense`` from the
    request and the spec id alone, and a retune reaches the calibration gate."""
    tree = ast.parse(inspect.getsource(_resolve_tuning_spec))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [_dotted(call.func) for call in calls]
    assert "tuning_spec_with_knobs" not in names
    assert names.count("attention_tuning_spec") == 1
    resolve = calls[names.index("attention_tuning_spec")]
    assert len(resolve.args) == 2 and not resolve.keywords


def test_subject_side_wrong_scale_is_the_omitted_sqrt_d_mutation() -> None:
    """The GPU lane's ``scale_override`` launch is the kernel twin of the
    analyzer's ``omit_sqrt_d``: both give ``qk_scale = RN(scale * LOG2E)``."""
    from kernels.gfx942 import attention_dense

    from .test_structured_analytical_gfx942 import _WRONG_SCALE

    case = _LABEL_CASE
    row, key = 64, 64
    numerator = score_numerator(case, 0, row, key)
    qk_scale = round_f32(f32(_WRONG_SCALE) * f32(attention_dense.LOG2E))
    assert score(case, 0, row, key, StructuredMutation.OMIT_SQRT_D) == round_f32(
        numerator * qk_scale
    )


_STRUCTURED_TORCH_BLOCKED_PROBE = """
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
lane = importlib.import_module(package + "._structured_analytical")
subject = importlib.import_module(package + "._structured_subject")
importlib.import_module(package + ".test_structured_analytical_gfx942")
for case in lane.STRUCTURED_QUICK_CASES:
    subject.resolve_dense_subject(case)
loaded = sorted(name for name in sys.modules if name.split(".")[0] == "torch")
assert not loaded, loaded
print("torch-free import and resolution ok")
"""


def _package_root() -> tuple[str, Path]:
    """This suite's package name and the directory that makes it importable."""
    package = __package__ or ""
    assert package, "the contracts must be imported as part of their package"
    root = Path(__file__).resolve().parent
    for _ in package.split("."):
        root = root.parent
    return package, root


def test_structured_suite_imports_and_resolves_with_torch_blocked() -> None:
    package, root = _package_root()
    config = {"package": package, "path": [str(root), *(e for e in sys.path if e)]}
    completed = subprocess.run(
        [sys.executable, "-c", _STRUCTURED_TORCH_BLOCKED_PROBE, json.dumps(config)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "torch-free import and resolution ok" in completed.stdout
