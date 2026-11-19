# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks of the compiled artifact a value-shift launch runs.

The tolerance of the relation is derived for one arithmetic path, and its
two-launch argument needs one more fact about that path: nothing computed from V
reaches a score, a weight, the normalizer, a branch or an address. These checks
read the LLVM IR a kernel is compiled from and report every way it leaves the
path or that fact. They are host-only text analysis: the CPU contracts apply
them to the lowered quick specs, and the GPU lane applies them, before any
launch, to the artifact of the kernel under test.

The path is the one the key-permutation bound is derived for, so its
classifier, its MFMA operand trace and its modeled counts are those of
``_permutation_path``. This module adds the V cone and the gate that requires
all three.
"""

from __future__ import annotations

import re
from collections import Counter

from ._permutation_path import (
    _BUFFER,
    _CAST_TYPE,
    _DECLARE,
    _EXP2,
    _INSTRUCTION,
    _LDS_ARRAY,
    _LDS_BASE,
    _LDS_FILL,
    _LDS_OFFSET,
    _MFMA_ATOM,
    _VALUE,
    _Instruction,
    _parse,
    classify_unmasked_path,
    expected_chain_counts,
    expected_path_counts,
    trace_mfma_operands,
)

REDERIVE = "re-derive the tolerance"

_STORE = re.compile(
    r"^\s*store .+ (%[-\w.]+|undef|poison|zeroinitializer), "
    r"ptr addrspace\((\d+)\) (%[-\w.]+|@[-\w.]+),"
)
_VECTOR_INDEX = re.compile(r", i\d+ (%[-\w.]+)$")
# Intrinsics that read or write the floating-point environment.
_FP_ENVIRONMENT = re.compile(
    r"(?:^|\.)(?:set\.rounding|get\.rounding|flt\.rounds|set\.fpenv|get\.fpenv|"
    r"reset\.fpenv|set\.fpmode|get\.fpmode|reset\.fpmode|setreg|getreg)(?:\.|$)"
)


def trace_value_cone(ir: str, dtype: str) -> tuple[list, Counter]:
    """Follow everything one lowered unmasked kernel computes from V and report
    whatever leaves the cone the relation's two-launch argument allows.

    The relation runs the kernel on ``V`` and on ``V + c`` with the same Q and K
    and takes the weights, the rescales and the normalizer of the two launches
    to be the same numbers. That holds when V cannot reach them. Allowed, and
    counted: ``v_ptr`` is read only by the fills of its LDS region; that region
    is read only as the first operand of the value-chain MFMA calls, whose
    second operand, the cast weights, is not computed from V; their results are
    the loop-carried accumulator, which is multiplied only by a rescale
    ``alpha`` or by the reciprocal of the normalizer; and the reciprocal
    products are cast and stored through ``o_ptr``. Everything else computed
    from V is a violation: a maximum, an exponential or its argument, a
    normalizer term, a probability cast, a comparison or branch condition, an
    address or an LDS offset.

    The analysis is on the SSA text and on LDS regions as a whole. Which row of
    its own region a fill writes is index arithmetic and is not traced."""
    violations: list[str] = []
    counts: Counter = Counter()
    instructions = _parse(ir)
    uses: dict[str, list[_Instruction]] = {}
    for instruction in instructions.values():
        for operand in instruction.operands:
            uses.setdefault(operand, []).append(instruction)

    def through(name: str) -> str:
        """Follow every ``bitcast``."""
        while name in instructions and instructions[name].opcode == "bitcast":
            name = instructions[name].operands[0]
        return name

    def users(name: str) -> list[_Instruction]:
        found: list[_Instruction] = []
        for user in uses.get(name, []):
            found.extend(users(user.name) if user.opcode == "bitcast" else [user])
        return found

    def lds_offset(base: str) -> int | None:
        """Byte offset of an LDS base inside the pool: the pool itself, or an
        ``i8`` offset from it."""
        if base.startswith("@"):
            return 0
        item = instructions.get(base)
        match = _LDS_OFFSET.match(item.text) if item else None
        return int(match.group(1)) if match else None

    def fill_offset(pointer: str) -> int | None:
        """LDS region a fill writes: follow its address arithmetic to the
        ``ptrtoint`` of the region's base."""
        item = instructions.get(through(pointer))
        while item is not None:
            base = _LDS_BASE.match(item.text)
            if base:
                return lds_offset(base.group(1))
            item = instructions.get(item.operands[0]) if item.operands else None
        return None

    for line in ir.splitlines():
        declared = _DECLARE.match(line.strip())
        if declared and _FP_ENVIRONMENT.search(declared.group(2)):
            violations.append(
                f"floating-point environment intrinsic @{declared.group(2)}"
            )

    # V enters through buffer resources of v_ptr, and only through them.
    resources = {
        item.name
        for item in instructions.values()
        if item.opcode.startswith(_BUFFER) and item.operands[:1] == ("%v_ptr",)
    }
    body = ir[ir.find("{") :] if "define " in ir else ir
    for line in body.splitlines():
        if "%v_ptr" not in _VALUE.findall(line):
            continue
        match = _INSTRUCTION.match(line)
        if match is None or match.group(1) not in resources:
            violations.append(f"v_ptr is read outside an LDS fill: {line.strip()}")
    for name in resources:
        for user in uses.get(name, []):
            violations.append(f"the V buffer resource is not only filled: {user.text}")

    # LDS regions the fills from V write.
    value_regions: set[int] = set()
    for match in _LDS_FILL.finditer(ir):
        if match.group(1) not in resources:
            continue
        counts["value_fills"] += 1
        offset = fill_offset(match.group(2))
        if offset is None:
            violations.append(f"V fill into an untraced LDS address {match.group(2)}")
        else:
            value_regions.add(offset)
    if not counts["value_fills"]:
        violations.append("no LDS fill from v_ptr")

    # Every LDS load reads one traced region; the loads of a V region carry V.
    tainted: set[str] = set(resources)
    loads: set[str] = set()
    for item in instructions.values():
        if item.opcode != "load" or "ptr addrspace(3)" not in item.text:
            continue
        address = instructions.get(item.operands[0]) if item.operands else None
        array = _LDS_ARRAY.match(address.text) if address is not None else None
        offset = lds_offset(array.group(3)) if array else None
        if offset is None:
            violations.append(f"LDS load from an untraced region: {item.text}")
            continue
        size = 2 * int(array.group(1)) * int(array.group(2))
        if offset in value_regions:
            loads.add(item.name)
        elif any(offset < region < offset + size for region in value_regions):
            violations.append(
                f"an LDS array of {size} bytes at offset {offset} spans a V "
                f"region at {sorted(value_regions)}: {item.text}"
            )
    tainted |= loads
    pending = sorted(loads)
    while pending:
        for user in uses.get(pending.pop(), []):
            if user.name not in tainted:
                tainted.add(user.name)
                pending.append(user.name)

    alphas = {
        item.name
        for item in instructions.values()
        if item.opcode == _EXP2
        and {user.opcode for user in users(item.name)} == {"fmul"}
    }
    reciprocals = {item.name for item in instructions.values() if item.opcode == "fdiv"}
    cast_type = _CAST_TYPE[dtype]
    atom = _MFMA_ATOM[dtype]
    for name in sorted(tainted - resources):
        item = instructions[name]
        positions = [
            index for index, operand in enumerate(item.operands) if operand in tainted
        ]
        if item.opcode == "load":
            counts["value_loads"] += 1
        elif item.opcode == "bitcast":
            continue
        elif item.opcode in ("insertelement", "extractelement"):
            index = _VECTOR_INDEX.search(item.text)
            if index and index.group(1) in tainted:
                violations.append(f"V selects a vector element: {item.text}")
        elif item.opcode == "phi":
            if item.text.startswith("phi <16 x float>"):
                counts["accumulator_phi"] += 1
            else:
                violations.append(
                    f"V reaches a phi that is not the accumulator: {item.text}"
                )
        elif item.opcode == atom:
            if positions in ([0], [0, 2]):
                counts["value_atoms"] += 1
            else:
                violations.append(
                    f"V reaches an MFMA call outside its value operand and "
                    f"accumulator: {item.text}"
                )
        elif item.opcode == "fmul":
            factor = through(item.operands[1]) if len(item.operands) == 2 else None
            if positions == [0] and factor in alphas:
                counts["accumulator_rescales"] += 1
            elif positions == [0] and factor in reciprocals:
                counts["output_products"] += 1
            else:
                violations.append(
                    f"V reaches an fmul that is neither the accumulator rescale "
                    f"nor the output product: {item.text}"
                )
        elif item.opcode == "fptrunc":
            product = instructions.get(through(item.operands[0]))
            if (
                product is not None
                and product.opcode == "fmul"
                and len(product.operands) == 2
                and through(product.operands[1]) in reciprocals
                and item.text.endswith(f" to {cast_type}")
            ):
                counts["output_casts"] += 1
            else:
                violations.append(
                    f"V reaches a cast that is not the output store: {item.text}"
                )
        else:
            violations.append(f"V reaches {item.opcode}: {item.text}")

    # Statements without a result: stores, branches and void calls.
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or _INSTRUCTION.match(line) or stripped.endswith(":"):
            continue
        operands = [name for name in _VALUE.findall(stripped) if name in tainted]
        if stripped.startswith("store "):
            match = _STORE.match(stripped)
            pointer = instructions.get(match.group(3)) if match else None
            if (
                match is None
                or match.group(2) != "1"
                or pointer is None
                or pointer.opcode != "getelementptr"
                or "ptr addrspace(1) %o_ptr," not in pointer.text
            ):
                violations.append(f"store outside the output tensor: {stripped}")
            elif match.group(3) in tainted:
                violations.append(f"V reaches a store address: {stripped}")
            elif match.group(1) not in tainted:
                violations.append(f"stored value is not computed from V: {stripped}")
            else:
                counts["stores"] += 1
        elif _LDS_FILL.match(stripped):
            fill = _LDS_FILL.match(stripped)
            if [name for name in operands if name != fill.group(1)]:
                violations.append(f"V reaches an LDS fill address: {stripped}")
        elif operands:
            violations.append(f"V reaches {stripped.split()[0]}: {stripped}")
    return violations, counts


def expected_cone_counts(block_n: int, head_size: int) -> dict[str, int]:
    """Instructions of the modeled V cone, per lane and key tile: one value
    chain per 32 output columns, ``block_n / 8`` MFMA calls long, each reading
    four V rows; 16 accumulator elements per chain, each rescaled once and
    normalized, cast and stored once, four to a store."""
    chains = head_size // 32
    atoms = chains * (block_n // 8)
    elements = 16 * chains
    return {
        "value_loads": 4 * atoms,
        "value_atoms": atoms,
        "accumulator_phi": chains,
        "accumulator_rescales": elements,
        "output_products": elements,
        "output_casts": elements,
        "stores": elements // 4,
    }


def unmasked_path_violations(
    ir: str, dtype: str, *, block_n: int, head_size: int
) -> list[str]:
    """Every way one lowered unmasked kernel leaves the path the tolerance
    models: an unmodeled instruction, a misrouted MFMA operand, a value computed
    from V outside its cone, or a count that differs from the modeled
    structure."""
    violations, counts = classify_unmasked_path(ir, dtype, block_n)
    traced, chains = trace_mfma_operands(ir, dtype)
    escaped, cone = trace_value_cone(ir, dtype)
    violations = [*violations, *traced, *escaped]
    fills = {name: chains.pop(name, 0) for name in ("k_fills", "v_fills")}
    fills["value_fills"] = cone.pop("value_fills", 0)
    for name, found, expected in (
        ("instruction", counts, expected_path_counts(block_n, head_size)),
        ("MFMA chain", chains, expected_chain_counts(block_n, head_size)),
        ("V cone", cone, expected_cone_counts(block_n, head_size)),
    ):
        if dict(found) != expected:
            violations.append(f"{name} counts {dict(found)} != {expected}")
    if not all(fills.values()):
        violations.append(f"LDS fills {fills}")
    return violations


def verified_key_tiles(
    ir: str, dtype: str, *, seqlen_k: int, block_n: int, head_size: int
) -> int:
    """Return the number of key tiles one query row visits in the kernel this
    IR is compiled to, after checking that the IR is the modeled path.

    The tolerance takes this count from the artifact, not from the requested
    shape: the path check establishes that the one KV loop runs
    ``seqlen_kv / block_n`` times, each iteration scoring, weighting and
    accumulating one whole tile, with no mask and no early exit. Raises
    ``AssertionError`` otherwise; the caller must not launch."""
    violations = unmasked_path_violations(
        ir, dtype, block_n=block_n, head_size=head_size
    )
    if seqlen_k <= 0 or seqlen_k % block_n:
        violations.append(f"{seqlen_k} keys are not whole tiles of {block_n}")
    if violations:
        raise AssertionError(f"{REDERIVE}: " + "; ".join(violations))
    return seqlen_k // block_n
