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
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from rocke.numeric.bf16 import f32_to_bf16

from ._p4_shared_analytical import UNIFORM_QUICK_CASES, UniformCase
from ._p4_shared_subject import _resolve_tuning_spec
from ._p4_analytical import (
    F32_SMALLEST_SUBNORMAL,
    F32Interval,
    add_faithful,
    add_rn,
    cast_bits_hull,
    exp2_candidates,
    f32,
    f32_quantum,
    gamma,
    is_f32,
    mul_rn,
    next_down,
    next_up,
    rcp_rn,
    round_f32,
    sum_any_order,
)

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
    the function may set no floating-point mode attribute."""
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
    assert all(
        next_up(left) == right for left, right in zip(candidates, candidates[1:])
    )
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
