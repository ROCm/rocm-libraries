# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks of the compiled artifact a masked-token poisoning launch runs.

The relation's bitwise verdict holds for one masked arithmetic path: every score
is replaced by the mask sentinel before any use, the row maximum and both
exponentials read only masked scores, one rescale multiplies the whole running
state, and nothing computed from Q, K or V steers the kernel. These checks read
the LLVM IR a kernel is compiled from and report every way it leaves that path.
They also execute the kernel's own position arithmetic for every thread of the
launch and compare the keys it leaves unmasked with the case's legal keys.

Host-only text analysis: the CPU contracts apply it to the lowered cases, and
the GPU lanes apply it, before any launch, to the artifact of the kernel under
test.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ._analytical import _legal_bounds
from ._permutation_path import (
    _BPERMUTE,
    _CALLEE,
    _CAST_TYPE,
    _DECLARE,
    _EXP2,
    _F32_INITIAL_MAX,
    _F32_LOG2E,
    _F32_ONE,
    _F32_ZERO,
    _FP_FLAGS,
    _FP_MODE_ATTRIBUTES,
    _FP_TYPE,
    _MAXNUM,
    _MFMA_ATOM,
    _NO_OP_BITCAST,
    _VALUE,
    REDERIVE,
    expected_chain_counts,
    trace_mfma_operands,
)

_LABEL = re.compile(r"^([-\w.]+):")
_ASSIGNMENT = re.compile(r"^\s+(?:(%[-\w.]+) = )?(\S.*)$")
_PHI_VALUE = re.compile(r"\[ ([^,\]]+), %[-\w.]+ \]")
_BRANCH_CONDITION = re.compile(r"^br i1 (%[-\w.]+),")
_BRANCH_TARGETS = re.compile(r"label %([-\w.]+)")
_MASK_SELECT = re.compile(
    rf"^select i1 (%[-\w.]+), float (%[-\w.]+), float {_F32_INITIAL_MAX}$"
)
_MASK_COMPARE = re.compile(r"^icmp (sle|sgt) i32 (%[-\w.]+), (%[-\w.]+)$")
_ZERO_VECTOR = re.compile(
    r"^select i1 true, <\d+ x float> zeroinitializer, <\d+ x float> zeroinitializer$"
)
_BINARY = re.compile(
    r"^(add|sub|mul|sdiv|srem|udiv|urem|shl|lshr|ashr|and|or|xor)"
    r"(?: nsw| nuw| exact| disjoint)* i\d+ (\S+), (\S+)$"
)
_COMPARE = re.compile(r"^icmp (\w+) i\d+ (\S+), (\S+)$")
_INTEGER_SELECT = re.compile(r"^select i1 (\S+), i\d+ (\S+), i\d+ (\S+)$")
_INTEGER_CAST = re.compile(r"^(?:zext|sext|trunc) i\d+ (\S+) to i\d+$")
_INTEGER_ARGUMENT = re.compile(r"i\d+ (%[-\w.]+|-?\d+)")

# Integer instructions position arithmetic is made of.
_INDEX_OPCODES = frozenset(
    {
        "add",
        "sub",
        "mul",
        "sdiv",
        "srem",
        "udiv",
        "urem",
        "shl",
        "lshr",
        "ashr",
        "and",
        "or",
        "xor",
        "icmp",
        "zext",
        "sext",
        "trunc",
        "ptrtoint",
        "inttoptr",
        "getelementptr",
    }
)
# Calls position arithmetic may contain, by name prefix: thread and work-group
# coordinates, the lane index, integer min/max and the buffer descriptor.
_INDEX_CALLS = (
    "llvm.amdgcn.workitem.id.",
    "llvm.amdgcn.workgroup.id.",
    "llvm.amdgcn.mbcnt.",
    "llvm.smin.",
    "llvm.smax.",
    "llvm.amdgcn.make.buffer.rsrc.",
)
_BUFFER_LOAD = "llvm.amdgcn.raw.ptr.buffer.load"
# Instructions every operand of which steers the kernel.
_STEERING_OPCODES = frozenset({"fcmp", "icmp", "br", "getelementptr"})


@dataclass(frozen=True)
class _Op:
    """One instruction: its SSA name (empty when it defines none), opcode (the
    callee for a call), text without the assignment, and basic block."""

    name: str
    opcode: str
    text: str
    block: str


@dataclass(frozen=True)
class MaskLane:
    """One score lane of a key loop: the score, the value the lane passes on
    (the last mask select, or the score in a mask-free loop), the key and query
    positions its mask compares, and the compare instructions."""

    score: str
    masked: str
    key: str | None
    query: str | None
    compares: tuple[str, ...]


@dataclass(frozen=True)
class KeyLoop:
    """One key-tile loop: its counter and bounds, the mask its lanes apply
    (``none``, ``upper``, ``lower`` or ``both``), and its lanes."""

    counter: str
    start: str
    bound: str
    mask: str
    lanes: tuple[MaskLane, ...]


@dataclass(frozen=True)
class MaskedPath:
    """What :func:`classify_masked_path` found: the violations, the key loops in
    execution order with their per-loop instruction counts, and the counts of
    the epilogue."""

    violations: tuple[str, ...]
    loops: tuple[KeyLoop, ...]
    loop_counts: tuple[Counter, ...]
    epilogue_counts: Counter


def _parse_function(ir: str) -> tuple[list[_Op], dict[str, _Op], frozenset[str]]:
    """Parse the one function of a lowered kernel into instructions in order, a
    name-to-instruction map, and the names of its integer and pointer
    arguments."""
    ops: list[_Op] = []
    named: dict[str, _Op] = {}
    arguments: set[str] = set()
    block = ""
    inside = False
    for line in ir.splitlines():
        if line.startswith("define "):
            inside = True
            signature = line[line.index("(") + 1 : line.rindex(")")]
            for argument in signature.split(","):
                words = argument.split()
                if words and words[-1].startswith("%") and words[0] != "float":
                    arguments.add(words[-1])
            block = "entry"
            continue
        if not inside:
            continue
        if line.startswith("}"):
            break
        label = _LABEL.match(line)
        if label:
            block = label.group(1)
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        name, text = match.groups()
        if re.match(r"(?:tail |musttail |notail )?call ", text):
            callee = _CALLEE.search(text)
            opcode = callee.group(1) if callee else "call"
        else:
            opcode = text.split()[0]
        op = _Op(name or "", opcode, text, block)
        ops.append(op)
        if name:
            named[name] = op
    return ops, named, frozenset(arguments)


def _values(op: _Op) -> tuple[str, ...]:
    """The SSA values one instruction reads; block labels are not values."""
    if op.opcode == "phi":
        return tuple(
            item for item in _PHI_VALUE.findall(op.text) if item.startswith("%")
        )
    if op.opcode == "br":
        condition = _BRANCH_CONDITION.match(op.text)
        return (condition.group(1),) if condition else ()
    if op.text.split()[0] == op.opcode:
        return tuple(_VALUE.findall(op.text[len(op.opcode) :]))
    callee = _CALLEE.search(op.text)
    return tuple(_VALUE.findall(op.text[callee.end() :] if callee else op.text))


def classify_masked_path(
    ir: str, dtype: str, *, block_n: int, head_size: int
) -> MaskedPath:
    """Classify every floating-point instruction of one lowered masked kernel
    against the path the relation's bit equality is derived for.

    The path, per key-tile loop and lane: scores ``fmul(acc, scale * LOG2E)``;
    each score read only by its mask selects, whose false value is the sentinel
    ``-1e30`` and whose conditions are ``icmp sle`` (causal bound) and
    ``icmp sgt`` (window bound) of integers; the tile maximum, the running
    maximum and ``exp2(fsub(x, m_new))`` read the masked scores; each weight
    feeds the normalizer chain uncast and the PV operand through one
    ``fptrunc``; one ``alpha = exp2(fsub(m, m_new))`` rescales the normalizer
    and every accumulator element. The running state starts at
    ``(-1e30, 0, 0)`` and is handed from one loop to the next unchanged. The
    epilogue is one ``fdiv 1.0, l`` and ``fptrunc(fmul(o, rcp))``. No fast-math
    flag, ``!fpmath`` or floating-point mode attribute may appear.

    A loop whose lanes carry no mask select is reported with ``mask="none"``;
    whether it may run is decided by :func:`legal_set_violations`."""
    violations: list[str] = []
    ops, named, _ = _parse_function(ir)
    uses: dict[str, list[_Op]] = {}
    for op in ops:
        for value in _values(op):
            uses.setdefault(value, []).append(op)

    def source(name: str) -> str:
        """Follow type-preserving ``bitcast`` copies."""
        while name in named:
            op = named[name]
            no_op = _NO_OP_BITCAST.match(op.text)
            if not (no_op and no_op.group(1) == no_op.group(2)) or not _values(op):
                break
            name = _values(op)[0]
        return name

    def opcode(name: str) -> str | None:
        op = named.get(source(name))
        return op.opcode if op else None

    def users(name: str) -> list[_Op]:
        found: list[_Op] = []
        for user in uses.get(name, []):
            if user.opcode == "bitcast" and source(user.name) == source(name):
                found.extend(users(user.name))
            else:
                found.append(user)
        return found

    for line in ir.splitlines():
        stripped = line.strip()
        match = re.search(r"= (?:f\w+|call|select|phi) ((?:\w+ )*)", stripped)
        if match and set(match.group(1).split()) & set(_FP_FLAGS):
            violations.append(f"fast-math flags: {stripped}")
        if "!fpmath" in stripped:
            violations.append(f"!fpmath metadata: {stripped}")
        if stripped.startswith("attributes #"):
            for attribute in _FP_MODE_ATTRIBUTES:
                if f'"{attribute}' in stripped:
                    violations.append(f"FP mode attribute {attribute}: {stripped}")
        declared = _DECLARE.match(stripped)
        if declared:
            returned, callee, arguments = declared.groups()
            modeled = callee in (_EXP2, _MAXNUM, _MFMA_ATOM[dtype])
            if not modeled and (
                _FP_TYPE.search(returned) or _FP_TYPE.search(arguments)
            ):
                violations.append(f"unmodeled floating-point intrinsic @{callee}")

    scale = [
        op.name
        for op in ops
        if op.text
        in (f"fmul float %scale, {_F32_LOG2E}", f"fmul float {_F32_LOG2E}, %scale")
    ]
    if len(scale) != 1:
        violations.append(f"expected one fmul(scale, LOG2E), found {len(scale)}")

    def element_of(name: str, producer: str) -> bool:
        op = named.get(source(name))
        return (
            op is not None
            and op.opcode == "extractelement"
            and bool(_values(op))
            and opcode(_values(op)[0]) == producer
        )

    def shuffled(name: str) -> bool:
        """Whether a float is the partner lane's value: a bitcast of a lane
        shuffle."""
        op = named.get(source(name))
        return (
            op is not None
            and op.opcode == "bitcast"
            and bool(_values(op))
            and opcode(_values(op)[0]) == _BPERMUTE
        )

    # Key-tile loops, in the order control reaches them.
    headers: list[dict[str, str]] = []
    block_loop: dict[str, int] = {}
    by_block: dict[str, list[_Op]] = {}
    for op in ops:
        by_block.setdefault(op.block, []).append(op)
    for block, members in by_block.items():
        branch = members[-1]
        targets = _BRANCH_TARGETS.findall(branch.text)
        if branch.opcode != "br" or not _BRANCH_CONDITION.match(branch.text):
            continue
        compare = named.get(_values(branch)[0])
        match = compare and re.fullmatch(
            r"icmp slt i32 (%[-\w.]+), (%[-\w.]+)", compare.text
        )
        counter = named.get(match.group(1)) if match else None
        if counter is None or counter.opcode != "phi" or counter.block != block:
            violations.append(f"unmodeled conditional branch: {branch.text}")
            continue
        incoming = _PHI_VALUE.findall(counter.text)
        step = named.get(incoming[1]) if len(incoming) == 2 else None
        if step is None or step.text != f"add nsw i32 {counter.name}, 1":
            violations.append(f"key-tile counter does not step by one: {counter.text}")
            continue
        index = len(headers)
        headers.append(
            {
                "header": block,
                "body": targets[0],
                "latch": step.block,
                "counter": counter.name,
                "start": incoming[0],
                "bound": match.group(2),
            }
        )
        for member in (block, targets[0], step.block):
            block_loop[member] = index
    if not headers:
        violations.append("no key-tile loop")

    # Loop-carried state: the running maximum, the normalizer and the
    # accumulators, started at (-1e30, 0, 0) and handed from loop to loop.
    running_max: list[str | None] = []
    normalizer: list[str | None] = []
    accumulators: list[set[str]] = []
    for index, header in enumerate(headers):
        found_max = found_normalizer = None
        found_accumulators: set[str] = set()
        for op in by_block[header["header"]]:
            if op.opcode != "phi" or op.name == header["counter"]:
                continue
            initial = _PHI_VALUE.findall(op.text)[0]
            origin = source(initial) if initial.startswith("%") else initial
            if op.text.startswith("phi float"):
                first = (
                    (_F32_INITIAL_MAX, _F32_ZERO)
                    if index == 0
                    else (running_max[index - 1], normalizer[index - 1])
                )
                if origin == first[0] and found_max is None:
                    found_max = op.name
                elif origin == first[1] and found_normalizer is None:
                    found_normalizer = op.name
                else:
                    violations.append(f"unmodeled float phi: {op.text}")
            elif "x float>" in op.text:
                started = named.get(origin)
                if index == 0:
                    valid = started is not None and _ZERO_VECTOR.match(started.text)
                else:
                    valid = origin in accumulators[index - 1]
                if valid:
                    found_accumulators.add(op.name)
                else:
                    violations.append(f"unmodeled accumulator phi: {op.text}")
            else:
                violations.append(f"unmodeled phi: {op.text}")
        if found_max is None or found_normalizer is None or not found_accumulators:
            violations.append(
                f"loop {header['counter']} has no loop-carried maximum, "
                "normalizer or accumulator"
            )
        running_max.append(found_max)
        normalizer.append(found_normalizer)
        accumulators.append(found_accumulators)
    for op in ops:
        if op.opcode == "phi" and op.block not in {h["header"] for h in headers}:
            violations.append(f"phi outside a key-tile loop header: {op.text}")

    def loop_of(op: _Op) -> int | None:
        return block_loop.get(op.block)

    # Scores and their mask selects.
    scores: dict[str, int | None] = {}
    for op in ops:
        if op.opcode != "fmul" or op.name in scale or len(_values(op)) != 2:
            continue
        left, right = _values(op)
        if right in scale and element_of(left, _MFMA_ATOM[dtype]):
            scores[op.name] = loop_of(op)
    mask_selects: dict[str, tuple[str, str, str, str]] = {}
    for op in ops:
        if op.opcode != "select" or not _FP_TYPE.search(op.text):
            continue
        if _ZERO_VECTOR.match(op.text):
            continue
        match = _MASK_SELECT.match(op.text)
        compare = named.get(match.group(1)) if match else None
        bound = _MASK_COMPARE.match(compare.text) if compare else None
        if not bound:
            violations.append(f"unmodeled float select: {op.text}")
            continue
        kind = "upper" if bound.group(1) == "sle" else "lower"
        mask_selects[op.name] = (kind, match.group(2), compare.name, bound.group(2))

    lanes: list[list[MaskLane]] = [[] for _ in headers]
    lane_masks: list[set[str]] = [set() for _ in headers]
    masked_scores: dict[str, int] = {}
    for score, index in scores.items():
        if index is None:
            violations.append(f"score outside a key-tile loop: {named[score].text}")
            continue
        kinds: list[str] = []
        compares: list[str] = []
        keys: set[str] = set()
        current = score
        while True:
            readers = users(current)
            selects = [
                user
                for user in readers
                if user.name in mask_selects and mask_selects[user.name][1] == current
            ]
            if not selects:
                break
            if len(readers) != 1:
                violations.append(
                    f"a score is read before its mask: {named[current].text} by "
                    f"{[user.text for user in readers if user not in selects]}"
                )
                break
            kind, _, compare, key = mask_selects[selects[0].name]
            kinds.append(kind)
            compares.append(compare)
            keys.add(key)
            current = selects[0].name
        if kinds not in ([], ["upper"], ["lower"], ["upper", "lower"]):
            violations.append(f"unmodeled mask chain {kinds} on {named[score].text}")
        if len(keys) > 1:
            violations.append(f"mask bounds of one lane compare different keys: {keys}")
        mask = {(): "none", ("upper",): "upper", ("lower",): "lower"}.get(
            tuple(kinds), "both"
        )
        query = None
        for compare in compares:
            bound = _MASK_COMPARE.match(named[compare].text)
            limit = bound.group(3)
            if bound.group(1) == "sgt":
                # The window bound compares against query - window.
                lower = named.get(limit)
                window = lower and re.fullmatch(
                    r"sub nsw i32 (%[-\w.]+), \d+", lower.text
                )
                limit = window.group(1) if window else None
                if limit is None:
                    violations.append(
                        f"window bound is not query - window: {named[compare].text}"
                    )
            if query not in (None, limit):
                violations.append(
                    f"mask bounds of one lane compare different queries: "
                    f"{query}, {limit}"
                )
            query = query or limit
        lanes[index].append(
            MaskLane(
                score=score,
                masked=current,
                key=next(iter(keys)) if len(keys) == 1 else None,
                query=query,
                compares=tuple(compares),
            )
        )
        lane_masks[index].add(mask)
        masked_scores[current] = index
    used_selects = {
        compare
        for index_lanes in lanes
        for lane in index_lanes
        for compare in lane.compares
    }
    for name, (_, _, compare, _) in mask_selects.items():
        if compare not in used_selects:
            violations.append(f"mask select on a value that is no score: {name}")

    # Per-loop instruction roles.
    counts: list[Counter] = [Counter() for _ in headers]
    epilogue: Counter = Counter()
    for index, index_lanes in enumerate(lanes):
        counts[index]["score"] = len(index_lanes)
        if len(lane_masks[index]) > 1:
            violations.append(
                f"lanes of loop {headers[index]['counter']} carry different masks: "
                f"{sorted(lane_masks[index])}"
            )

    # Maximum: a lane chain from the sentinel over the masked scores, the
    # partner lane's chain, then the running maximum.
    tile_chain: dict[str, int] = {}
    new_max: dict[int, str] = {}
    for op in ops:
        if op.opcode != _MAXNUM:
            continue
        index = loop_of(op)
        if index is None:
            violations.append(f"maximum outside a key-tile loop: {op.text}")
            continue
        arguments = re.findall(
            r"float (%[-\w.]+|0x[0-9A-F]+)", op.text[op.text.index("(") :]
        )
        if len(arguments) != 2 or not arguments[1].startswith("%"):
            violations.append(f"unmodeled maximum: {op.text}")
            continue
        left, right = arguments
        left_source = source(left) if left.startswith("%") else left
        if masked_scores.get(source(right)) == index and (
            left == _F32_INITIAL_MAX or left_source in tile_chain
        ):
            tile_chain[op.name] = tile_chain.get(left_source, 0) + 1
            counts[index]["max_chain"] += 1
        elif left_source in tile_chain and opcode(right) == "bitcast":
            if not shuffled(right):
                violations.append(f"partner maximum is not a lane shuffle: {op.text}")
            tile_chain[op.name] = tile_chain[left_source]
            counts[index]["max_partner"] += 1
        elif left_source == running_max[index] and source(right) in tile_chain:
            if index in new_max:
                violations.append(f"second running maximum in one loop: {op.text}")
            new_max[index] = op.name
            counts[index]["max_running"] += 1
        else:
            violations.append(f"unmodeled maximum: {op.text}")
    for index, header in enumerate(headers):
        latch = [
            source(_values(op)[0])
            for op in by_block[header["latch"]]
            if op.opcode == "bitcast" and _values(op)
        ]
        if new_max.get(index) not in latch:
            violations.append(
                f"loop {header['counter']} does not carry its new maximum forward"
            )

    # exp2: a weight (normalizer fadd + one cast) or the rescale alpha.
    weights: dict[str, int] = {}
    alphas: dict[str, int] = {}
    for op in ops:
        if op.opcode == "fsub" and {user.opcode for user in users(op.name)} != {_EXP2}:
            violations.append(f"fsub not consumed by exp2 only: {op.text}")
        if op.opcode != _EXP2:
            continue
        index = loop_of(op)
        argument = named.get(source(_values(op)[0])) if _values(op) else None
        if (
            index is None
            or argument is None
            or argument.opcode != "fsub"
            or len(_values(argument)) != 2
        ):
            violations.append(f"unmodeled exp2: {op.text}")
            continue
        minuend, subtrahend = (source(value) for value in _values(argument))
        if subtrahend != new_max.get(index):
            violations.append(f"exp2 does not subtract the new maximum: {op.text}")
            continue
        kinds = Counter(user.opcode for user in users(op.name))
        if masked_scores.get(minuend) == index and kinds == Counter(
            {"fadd": 1, "fptrunc": 1}
        ):
            weights[op.name] = index
            counts[index]["exp2_weight"] += 1
        elif minuend == running_max[index] and kinds and set(kinds) == {"fmul"}:
            alphas[op.name] = index
            counts[index]["exp2_alpha"] += 1
        else:
            violations.append(f"exp2 with unmodeled operands or uses: {op.text}")
    for name, index in masked_scores.items():
        kinds = Counter(user.opcode for user in users(name))
        if kinds != Counter({_MAXNUM: 1, "fsub": 1}):
            violations.append(
                f"masked score with unmodeled uses {dict(kinds)}: {named[name].text}"
            )

    reciprocal = [op for op in ops if op.opcode == "fdiv"]
    last = len(headers) - 1
    if (
        len(reciprocal) != 1
        or not reciprocal[0].text.startswith(f"fdiv float {_F32_ONE}, %")
        or last < 0
        or source(_values(reciprocal[0])[0]) != normalizer[last]
        or loop_of(reciprocal[0]) is not None
    ):
        violations.append(
            f"expected one fdiv 1.0, l after the last loop; found "
            f"{[op.text for op in reciprocal]}"
        )
    rcp = reciprocal[0].name if reciprocal else None
    epilogue["reciprocal"] = len(reciprocal)

    def accumulator_element(name: str, index: int) -> bool:
        op = named.get(source(name))
        return (
            op is not None
            and op.opcode == "extractelement"
            and bool(_values(op))
            and source(_values(op)[0]) in accumulators[index]
        )

    rescaled: dict[str, int] = {}
    for op in ops:
        if op.opcode != "fmul" or op.name in scale or op.name in scores:
            continue
        if len(_values(op)) != 2:
            violations.append(f"unmodeled fmul: {op.text}")
            continue
        left, right = _values(op)
        index = loop_of(op)
        if index is not None and alphas.get(source(right)) == index:
            if source(left) == normalizer[index]:
                rescaled[op.name] = index
                counts[index]["rescale_normalizer"] += 1
            elif accumulator_element(left, index):
                counts[index]["rescale_accumulator"] += 1
            else:
                violations.append(f"unmodeled rescale: {op.text}")
        elif (
            index is None
            and right == rcp
            and last >= 0
            and accumulator_element(left, last)
            and {user.opcode for user in users(op.name)} == {"fptrunc"}
        ):
            epilogue["output_product"] += 1
        else:
            violations.append(f"unmodeled fmul: {op.text}")

    # Normalizer: a lane chain of uncast weights from +0, the partner lane's
    # sum, then l * alpha + tile sum, carried to the next iteration.
    chain: dict[str, int] = {}
    shuffles: dict[str, int] = {}
    for op in ops:
        if op.opcode != "fadd":
            continue
        index = loop_of(op)
        if not _values(op):
            violations.append(f"unmodeled fadd: {op.text}")
            continue
        left, right = _values(op)[0], _values(op)[-1]
        from_zero = op.text.startswith(f"fadd float {_F32_ZERO}, ")
        if (
            index is not None
            and weights.get(source(right)) == index
            and (from_zero or source(left) in chain)
        ):
            chain[op.name] = 1 if from_zero else chain[source(left)] + 1
            counts[index]["lane_chain"] += 1
        elif index is not None and source(left) in chain and opcode(right) == "bitcast":
            shuffles[op.name] = index
            counts[index]["partner_sum"] += 1
            if not shuffled(right):
                violations.append(f"partner sum is not a lane shuffle: {op.text}")
        elif (
            index is not None
            and rescaled.get(source(left)) == index
            and shuffles.get(source(right)) == index
        ):
            counts[index]["combine"] += 1
            latch = [
                source(_values(item)[0])
                for item in by_block[headers[index]["latch"]]
                if item.opcode == "bitcast" and _values(item)
            ]
            if op.name not in latch:
                violations.append(f"combine is not the next normalizer: {op.text}")
        else:
            violations.append(f"unmodeled fadd: {op.text}")

    cast_type = _CAST_TYPE[dtype]
    for op in ops:
        if op.opcode in ("fpext", "fcmp", "frem", "fneg"):
            violations.append(f"unmodeled floating-point instruction: {op.text}")
        if op.opcode != "fptrunc":
            continue
        if not op.text.endswith(f" to {cast_type}"):
            violations.append(f"cast to a type other than {cast_type}: {op.text}")
        operand = named.get(source(_values(op)[0])) if _values(op) else None
        index = loop_of(op)
        if index is not None and operand and weights.get(operand.name) == index:
            counts[index]["cast_weight"] += 1
        elif (
            operand is not None
            and index is None
            and operand.opcode == "fmul"
            and rcp in _values(operand)
        ):
            epilogue["cast_output"] += 1
        else:
            violations.append(f"unmodeled cast: {op.text}")

    for op in ops:
        if op.opcode == _MFMA_ATOM[dtype]:
            index = loop_of(op)
            if index is None:
                violations.append(f"MFMA outside a key-tile loop: {op.text}")
                continue
            counts[index]["mfma"] += 1
            if not op.text.endswith("i32 0, i32 0, i32 0)"):
                violations.append(f"MFMA with nonzero controls: {op.text}")

    expected = expected_loop_counts(block_n, head_size)
    for index, header in enumerate(headers):
        if dict(counts[index]) != expected:
            violations.append(
                f"loop {header['counter']} counts {dict(counts[index])} != {expected}"
            )
    expected_end = expected_epilogue_counts(head_size)
    if dict(epilogue) != expected_end:
        violations.append(f"epilogue counts {dict(epilogue)} != {expected_end}")

    loops = tuple(
        KeyLoop(
            counter=header["counter"],
            start=header["start"],
            bound=header["bound"],
            mask=next(iter(lane_masks[index])) if len(lane_masks[index]) == 1 else "?",
            lanes=tuple(lanes[index]),
        )
        for index, header in enumerate(headers)
    )
    return MaskedPath(tuple(violations), loops, tuple(counts), epilogue)


def expected_loop_counts(block_n: int, head_size: int) -> dict[str, int]:
    """Instruction counts of one key-tile loop of the modeled path, per lane.

    Two lane halves share a key tile; a lane holds 16 accumulator elements per
    32 output columns; an MFMA call takes eight products."""
    lane_keys = block_n // 2
    return {
        "score": lane_keys,
        "max_chain": lane_keys,
        "max_partner": 1,
        "max_running": 1,
        "exp2_weight": lane_keys,
        "exp2_alpha": 1,
        "rescale_normalizer": 1,
        "rescale_accumulator": 16 * (head_size // 32),
        "lane_chain": lane_keys,
        "partner_sum": 1,
        "combine": 1,
        "cast_weight": lane_keys,
        "mfma": (block_n // 32) * (head_size // 8) + (head_size // 32) * (block_n // 8),
    }


def expected_epilogue_counts(head_size: int) -> dict[str, int]:
    """Instruction counts of the modeled epilogue, per lane."""
    output_elements = 16 * (head_size // 32)
    return {
        "reciprocal": 1,
        "output_product": output_elements,
        "cast_output": output_elements,
    }


def steering_violations(ir: str) -> list[str]:
    """Every place where a value that is not position arithmetic steers the
    kernel: an operand of a comparison, a branch or select condition, an
    address, or an argument of a buffer load or of a lane shuffle's address.

    Position arithmetic is integer arithmetic on the kernel's integer
    arguments, the thread and work-group coordinates and loop counters. A value
    loaded from memory or computed in floating point is not."""
    ops, named, arguments = _parse_function(ir)
    state: dict[str, bool] = {}

    def index_only(value: str) -> bool:
        if not value.startswith("%"):
            return True
        if value in state:
            return state[value]
        op = named.get(value)
        if op is None:
            return value in arguments
        # A loop counter reads its own increment: assume it while checking it.
        state[value] = True
        if op.opcode in ("getelementptr", "ptrtoint", "inttoptr"):
            allowed = True
        elif op.opcode in _INDEX_OPCODES:
            allowed = not _FP_TYPE.search(op.text)
        elif op.opcode == "select":
            allowed = bool(_INTEGER_SELECT.match(op.text))
        elif op.opcode == "phi":
            allowed = bool(re.match(r"phi i\d+ ", op.text))
        else:
            allowed = op.opcode.startswith(_INDEX_CALLS)
        state[value] = allowed and all(index_only(item) for item in _values(op))
        return state[value]

    violations: list[str] = []
    for op in ops:
        values = _values(op)
        if op.opcode in _STEERING_OPCODES or op.opcode.startswith(_BUFFER_LOAD):
            checked = values
        elif op.opcode == "select":
            checked = values[:1] if op.text.startswith("select i1 %") else ()
        elif op.opcode in ("load", "store"):
            # The address is the last operand of both.
            checked = values[-1:]
        elif op.opcode == _BPERMUTE:
            checked = values[:1]
        else:
            continue
        steered = [item for item in checked if not index_only(item)]
        if op.opcode == "fcmp" or steered:
            violations.append(f"data-dependent {op.opcode}: {op.text}")
    return violations


class _IndexEvaluator:
    """Evaluates position arithmetic for every thread of a launch at once.

    Values are ``int64`` (or boolean) arrays broadcastable to ``[work groups,
    threads]``. Values of the entry block are evaluated once; values inside a
    loop are evaluated again whenever a loop counter changes."""

    def __init__(
        self,
        named: dict[str, _Op],
        arguments: dict[str, int],
        grid: tuple[int, int, int],
        threads: int,
    ) -> None:
        self._named = named
        self._arguments = arguments
        coordinates = np.indices(grid).reshape(3, -1)
        self.groups = coordinates.shape[1]
        self.threads = threads
        self._coordinates = {
            "llvm.amdgcn.workgroup.id.x": coordinates[0][:, None],
            "llvm.amdgcn.workgroup.id.y": coordinates[1][:, None],
            "llvm.amdgcn.workgroup.id.z": coordinates[2][:, None],
            "llvm.amdgcn.workitem.id.x": np.arange(threads, dtype=np.int64)[None, :],
        }
        self._fixed: dict[str, np.ndarray] = {}
        self._scoped: dict[str, np.ndarray] = {}
        self._counters: dict[str, int] = {}

    def set_counter(self, name: str, value: int) -> None:
        self._counters[name] = value
        self._scoped.clear()

    def value(self, token: str) -> np.ndarray:
        if token == "true":
            return np.array(True)
        if token == "false":
            return np.array(False)
        if not token.startswith("%"):
            return np.array(int(token), dtype=np.int64)
        op = self._named.get(token)
        if op is None:
            if token not in self._arguments:
                raise ValueError(f"no value for kernel argument {token}")
            return np.array(self._arguments[token], dtype=np.int64)
        if op.opcode == "phi":
            if token not in self._counters:
                raise ValueError(f"phi that is not an active loop counter: {op.text}")
            return np.array(self._counters[token], dtype=np.int64)
        cache = self._fixed if op.block == "entry" else self._scoped
        if token not in cache:
            cache[token] = self._evaluate(op)
        return cache[token]

    def full(self, token: str) -> np.ndarray:
        return np.broadcast_to(self.value(token), (self.groups, self.threads))

    def _evaluate(self, op: _Op) -> np.ndarray:
        binary = _BINARY.match(op.text)
        if binary:
            left, right = self.value(binary.group(2)), self.value(binary.group(3))
            return self._binary(binary.group(1), left, right)
        compare = _COMPARE.match(op.text)
        if compare:
            left, right = self.value(compare.group(2)), self.value(compare.group(3))
            predicate = compare.group(1)
            if predicate[0] == "u" and (np.any(left < 0) or np.any(right < 0)):
                raise ValueError(f"unsigned compare of a negative value: {op.text}")
            relation = predicate.lstrip("su")
            return {
                "lt": np.less,
                "le": np.less_equal,
                "gt": np.greater,
                "ge": np.greater_equal,
                "eq": np.equal,
                "ne": np.not_equal,
            }[relation](left, right)
        select = _INTEGER_SELECT.match(op.text)
        if select:
            return np.where(
                self.value(select.group(1)),
                self.value(select.group(2)),
                self.value(select.group(3)),
            )
        cast = _INTEGER_CAST.match(op.text)
        if cast:
            return self.value(cast.group(1))
        if op.opcode in self._coordinates:
            return self._coordinates[op.opcode]
        if op.opcode.startswith(("llvm.smin.", "llvm.smax.")):
            left, right = (
                self.value(item)
                for item in _INTEGER_ARGUMENT.findall(op.text[op.text.index("(") :])
            )
            return (np.minimum if ".smin." in op.opcode else np.maximum)(left, right)
        raise ValueError(f"position arithmetic the evaluator does not model: {op.text}")

    @staticmethod
    def _binary(opcode: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if opcode in ("sdiv", "srem", "udiv", "urem"):
            if np.any(right == 0):
                raise ValueError("division by zero in position arithmetic")
            # LLVM integer division truncates toward zero.
            quotient = np.abs(left) // np.abs(right) * np.sign(left) * np.sign(right)
            return quotient if opcode.endswith("div") else left - right * quotient
        return {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "shl": np.left_shift,
            "lshr": np.right_shift,
            "ashr": np.right_shift,
            "and": np.bitwise_and,
            "or": np.bitwise_or,
            "xor": np.bitwise_xor,
        }[opcode](left, right)


def _legal_set_run(
    ir: str,
    path: MaskedPath,
    *,
    first: np.ndarray,
    last: np.ndarray,
    window: int,
    arguments: dict[str, int],
    grid: tuple[int, int, int],
    threads: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Execute the kernel's position arithmetic for every thread of the launch
    and compare the keys each query row leaves unmasked with its legal keys.
    Returns the violations and, per query row, the first and one past the last
    key tile its work groups visit.

    ``first``/``last`` are the inclusive legal bounds of every query row, from
    the case's mask rule. ``arguments`` holds the kernel's integer arguments,
    ``grid`` and ``threads`` the launch geometry. For every work group, thread,
    key-tile loop iteration and lane, the conjunction of the lane's mask
    compares must equal membership of the lane's key in the row's legal set;
    over a work group, every legal key of every row it serves must be left
    unmasked exactly once; every row must be served by the same number of work
    groups. A loop without mask selects must not run at all."""
    violations: list[str] = []
    ops, named, _ = _parse_function(ir)
    del ops
    seqlen_q, seqlen_k = int(first.shape[0]), int(arguments["seqlen_kv"])
    evaluator = _IndexEvaluator(
        named, {f"%{name}": value for name, value in arguments.items()}, grid, threads
    )
    groups = evaluator.groups
    shape = (groups, threads)
    group_index = np.broadcast_to(np.arange(groups)[:, None], shape)
    unmasked = np.zeros((groups, seqlen_q, seqlen_k), dtype=np.int32)
    served = np.zeros((groups, seqlen_q), dtype=bool)
    tile_low = np.full((groups, seqlen_q), np.iinfo(np.int64).max, dtype=np.int64)
    tile_high = np.zeros((groups, seqlen_q), dtype=np.int64)
    no_rows = np.zeros(seqlen_q, dtype=np.int64)

    widths = {
        int(named[limit.group(3)].text.rsplit(", ", 1)[1])
        for loop in path.loops
        for lane in loop.lanes
        for compare in lane.compares
        if (limit := _MASK_COMPARE.match(named[compare].text)).group(1) == "sgt"
    }
    if widths - {window}:
        violations.append(f"window bounds use {sorted(widths)} keys, the case {window}")
    if window and not widths:
        violations.append("a window case without a window bound in the kernel")

    try:
        for loop in path.loops:
            start = evaluator.full(loop.start)
            bound = evaluator.full(loop.bound)
            for iteration in range(int(start.min()), int(bound.max())):
                active = (start <= iteration) & (iteration < bound)
                if not active.any():
                    continue
                if loop.mask == "none":
                    violations.append(
                        f"mask-free key loop {loop.counter} runs at tile {iteration} "
                        f"in {int(active.any(axis=1).sum())} work group(s)"
                    )
                    break
                evaluator.set_counter(loop.counter, iteration)
                for lane in loop.lanes:
                    if lane.key is None or lane.query is None:
                        continue
                    key = evaluator.full(lane.key)
                    query = evaluator.full(lane.query)
                    if np.any(active & ((query < 0) | (query >= seqlen_q))):
                        violations.append(
                            f"loop {loop.counter} tile {iteration}: a lane's query "
                            "row is outside the query range"
                        )
                        continue
                    kept = active.copy()
                    for compare in lane.compares:
                        kept &= evaluator.full(compare)
                    row = np.clip(query, 0, seqlen_q - 1)
                    legal = (first[row] <= key) & (key <= last[row])
                    wrong = active & (kept != legal)
                    if wrong.any():
                        group, thread = np.argwhere(wrong)[0]
                        violations.append(
                            f"loop {loop.counter} tile {iteration}: "
                            f"{int(wrong.sum())} lane(s) disagree with the legal "
                            f"set, first query {int(query[group, thread])} key "
                            f"{int(key[group, thread])} kept="
                            f"{bool(kept[group, thread])}"
                        )
                    inside = kept & (key >= 0) & (key < seqlen_k)
                    np.add.at(
                        unmasked, (group_index[inside], row[inside], key[inside]), 1
                    )
                    where = (group_index[active], row[active])
                    served[where] = True
                    np.minimum.at(tile_low, where, iteration)
                    np.maximum.at(tile_high, where, iteration + 1)
    except (ValueError, KeyError) as error:
        violations.append(f"position arithmetic could not be evaluated: {error}")
        return violations, no_rows, no_rows

    keys = np.arange(seqlen_k)
    legal_matrix = (first[:, None] <= keys[None, :]) & (keys[None, :] <= last[:, None])
    expected = served[:, :, None] & legal_matrix[None, :, :]
    differs = unmasked != expected
    if differs.any():
        group, row, key = np.argwhere(differs)[0]
        violations.append(
            f"{int(differs.sum())} (work group, row, key) triples are not left "
            f"unmasked exactly once, first work group {group} row {row} key {key}: "
            f"{int(unmasked[group, row, key])} time(s), legal="
            f"{bool(legal_matrix[row, key])}"
        )
    multiplicity = served.sum(axis=0)
    if multiplicity.min() == 0 or multiplicity.min() != multiplicity.max():
        violations.append(
            "query rows are not served by the same number of work groups: "
            f"between {int(multiplicity.min())} and {int(multiplicity.max())}"
        )
    low = np.where(served, tile_low, np.iinfo(np.int64).max).min(axis=0)
    high = np.where(served, tile_high, 0).max(axis=0)
    uneven = served & ((tile_low != low[None, :]) | (tile_high != high[None, :]))
    if uneven.any():
        violations.append(
            f"work groups visit different key tiles for {int(uneven.any(0).sum())} "
            "query row(s)"
        )
    return violations, low, high


def legal_set_violations(ir: str, path: MaskedPath, **scope) -> list[str]:
    """Every way the keys a launch leaves unmasked differ from the case's legal
    keys.

    ``scope`` holds ``first``/``last`` (the inclusive legal bounds of every
    query row, from the case's mask rule), ``window``, ``arguments`` (the
    kernel's integer arguments) and ``grid``/``threads`` (the launch geometry).
    The kernel's position arithmetic is executed for every work group, thread,
    key-tile loop iteration and lane."""
    return _legal_set_run(ir, path, **scope)[0]


def visited_tiles(ir: str, path: MaskedPath, **scope) -> tuple[np.ndarray, np.ndarray]:
    """Per query row, the first and one past the last key tile the launch
    visits for it, from the kernel's own loop bounds. ``scope`` as in
    :func:`legal_set_violations`."""
    _, low, high = _legal_set_run(ir, path, **scope)
    return low, high


def _loop_slices(ir: str, path: MaskedPath) -> list[str]:
    """One IR text per key-tile loop: the function with every other loop's
    blocks removed, so a single-loop check can read each loop on its own."""
    ops, named, _ = _parse_function(ir)
    del ops
    counters = [loop.counter for loop in path.loops]
    if len(counters) <= 1:
        return [ir]
    blocks: list[set[str]] = []
    for counter in counters:
        header = named[counter].block
        latch = named[_PHI_VALUE.findall(named[counter].text)[1]].block
        branch = [
            line
            for line in ir.splitlines()
            if line.strip().startswith("br i1 ")
            and named[_BRANCH_CONDITION.match(line.strip()).group(1)].block == header
        ]
        body = _BRANCH_TARGETS.findall(branch[0])[0]
        blocks.append({header, body, latch})
    everything = set().union(*blocks)
    slices = []
    for keep in blocks:
        dropped = everything - keep
        lines: list[str] = []
        block = ""
        inside = False
        for line in ir.splitlines():
            if line.startswith("define "):
                inside, block = True, "entry"
            elif inside and line.startswith("}"):
                inside = False
            elif inside and _LABEL.match(line):
                block = _LABEL.match(line).group(1)
            if not (inside and block in dropped):
                lines.append(line)
        slices.append("\n".join(lines))
    return slices


def masked_path_violations(
    ir: str,
    dtype: str,
    *,
    block_n: int,
    head_size: int,
    first: np.ndarray,
    last: np.ndarray,
    window: int,
    arguments: dict[str, int],
    grid: tuple[int, int, int],
    threads: int,
) -> list[str]:
    """Every way one lowered masked kernel leaves the path the relation's bit
    equality is derived for: an unmodeled floating-point instruction, a score
    read before its mask, a value that is not position arithmetic steering the
    kernel, a misrouted MFMA operand, or an unmasked key set that differs from
    the case's legal keys."""
    path = classify_masked_path(ir, dtype, block_n=block_n, head_size=head_size)
    violations = [*path.violations, *steering_violations(ir)]
    expected_chains = expected_chain_counts(block_n, head_size)
    for loop, text in zip(path.loops, _loop_slices(ir, path)):
        try:
            traced, chains = trace_mfma_operands(text, dtype)
        except (IndexError, KeyError) as error:
            # The trace assumes well-formed operands; IR it cannot read is not
            # the modeled path.
            violations.append(
                f"loop {loop.counter}: MFMA operand trace failed: {error!r}"
            )
            continue
        violations.extend(f"loop {loop.counter}: {item}" for item in traced)
        fills = {name: chains.pop(name, 0) for name in ("k_fills", "v_fills")}
        if dict(chains) != expected_chains:
            violations.append(
                f"loop {loop.counter}: MFMA chain counts {dict(chains)} != "
                f"{expected_chains}"
            )
        if not all(fills.values()):
            violations.append(f"loop {loop.counter}: LDS fills {fills}")
    if not path.violations:
        violations.extend(
            legal_set_violations(
                ir,
                path,
                first=first,
                last=last,
                window=window,
                arguments=arguments,
                grid=grid,
                threads=threads,
            )
        )
    return violations


def path_scope(case, *, block_n: int, head_size: int, grid, threads: int) -> dict:
    """The scope arguments of :func:`masked_path_violations` for one case and
    one launch geometry: the case's legal bounds by its own mask rule, the
    kernel's integer arguments, the tile and the grid."""
    first, last = _legal_bounds(case, None)
    return {
        "block_n": int(block_n),
        "head_size": int(head_size),
        "first": first,
        "last": last,
        "window": int(case.window),
        "arguments": {
            "batch": case.batch,
            "seqlen_q": case.seqlen_q,
            "seqlen_kv": case.seqlen_k,
            "num_query_heads": case.nhead_q,
            "num_kv_heads": case.nhead_k,
        },
        "grid": tuple(int(item) for item in grid),
        "threads": int(threads),
    }


def verified_masked_path(ir: str, dtype: str, **scope) -> MaskedPath:
    """Return the classified path of the kernel this IR is compiled to, after
    checking that it is the modeled one and that it leaves exactly the case's
    legal keys unmasked. Raises ``AssertionError`` otherwise; the caller must
    not launch. ``scope`` is passed to :func:`masked_path_violations`."""
    violations = masked_path_violations(ir, dtype, **scope)
    if violations:
        raise AssertionError(f"{REDERIVE}: " + "; ".join(violations[:12]))
    return classify_masked_path(
        ir, dtype, block_n=scope["block_n"], head_size=scope["head_size"]
    )
