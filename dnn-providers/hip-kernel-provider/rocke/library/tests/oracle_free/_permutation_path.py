# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks of the compiled artifact a key-permutation launch runs.

The bound of the relation is derived for one arithmetic path. These checks read
the LLVM IR a kernel is compiled from and the code object it is compiled to, and
report every way they leave that path. They are host-only text and byte
analysis: the CPU contracts apply them to the lowered quick specs, and the GPU
lane applies them, before any launch, to the artifact of the kernel under test.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass

REDERIVE = "re-derive the bound"

# LLVM spells FP32 constants as the hex of the equivalent double.
_F32_LOG2E = "0x3FF7154760000000"
_F32_ONE = "0x3FF0000000000000"
_F32_ZERO = "0x0000000000000000"
_F32_INITIAL_MAX = "0xC6293E5940000000"  # -1e30

_MFMA_ATOM = {
    "fp16": "llvm.amdgcn.mfma.f32.32x32x8f16",
    "bf16": "llvm.amdgcn.mfma.f32.32x32x8bf16.1k",
}
_CAST_TYPE = {"fp16": "half", "bf16": "bfloat"}
_EXP2 = "llvm.amdgcn.exp2.f32"
_MAXNUM = "llvm.maxnum.f32"
_BPERMUTE = "llvm.amdgcn.ds.bpermute"
_FP_FLAGS = ("fast", "contract", "afn", "reassoc", "nnan", "ninf", "nsz", "arcp")
_FP_MODE_ATTRIBUTES = (
    "denormal-fp-math",
    "unsafe-fp-math",
    "no-nans-fp-math",
    "no-infs-fp-math",
    "approx-func-fp-math",
    "no-signed-zeros-fp-math",
)
_FP_TYPE = re.compile(r"\b(float|half|bfloat|double)\b")
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
    """Map each SSA name to its defining instruction. ``opcode`` is the first
    word, or the callee for a call; ``operands`` are the SSA values it reads."""
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


def classify_unmasked_path(ir: str, dtype: str, block_n: int) -> tuple[list, Counter]:
    """Classify every floating-point instruction of one lowered unmasked kernel
    against the path the bound is derived for, and return the violations and
    the per-step instruction counts.

    The path, per key tile and lane: scores ``fmul(acc, scale * LOG2E)`` with no
    mask; ``exp2(fsub(x, m_new))``; each probability feeds the normalizer chain
    uncast and the PV operand through one ``fptrunc``; one ``alpha`` rescales
    the normalizer and every accumulator element; the key-tile loop runs
    ``seqlen_kv / block_n`` times; the epilogue is one ``fdiv 1.0, l`` and
    ``fptrunc(fmul(o, rcp))``. No fast-math flag, ``!fpmath`` or floating-point
    mode attribute may appear."""
    violations: list[str] = []
    counts: Counter = Counter()
    instructions = _parse(ir)
    uses: dict[str, list[_Instruction]] = {}
    for instruction in instructions.values():
        for operand in instruction.operands:
            uses.setdefault(operand, []).append(instruction)

    def source(name: str) -> str:
        """Follow type-preserving ``bitcast`` copies."""
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
        found: list[_Instruction] = []
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

    # Loop-carried state: the running maximum and the normalizer. No mask.
    running_max = normalizer = None
    for item in instructions.values():
        zeroed = item.text.count("zeroinitializer") == 2
        if item.opcode == "select" and _FP_TYPE.search(item.text) and not zeroed:
            violations.append(f"float select on an unmasked path: {item.text}")
        if item.opcode == "phi" and item.text.startswith("phi float"):
            initial = re.findall(r"\[ (0x[0-9A-F]+),", item.text)
            if initial == [_F32_INITIAL_MAX] and running_max is None:
                running_max = item.name
            elif initial == [_F32_ZERO] and normalizer is None:
                normalizer = item.name
            else:
                violations.append(f"unmodeled float phi: {item.text}")
    if running_max is None or normalizer is None:
        violations.append("no loop-carried running maximum or normalizer")

    # The key-tile loop visits every tile: kt in [0, seqlen_kv / block_n).
    compares = [item for item in instructions.values() if item.opcode == "icmp"]
    bound = None
    if len(compares) == 1:
        match = re.fullmatch(r"icmp slt i32 (%[-\w.]+), (%[-\w.]+)", compares[0].text)
        if match:
            counter, bound = instructions.get(match.group(1)), match.group(2)
            start = re.match(
                r"phi i32 \[ 0, %[-\w.]+ \], \[ (%[-\w.]+), ", counter.text
            )
            step = instructions.get(start.group(1)) if start else None
            if step is None or step.text != f"add nsw i32 {counter.name}, 1":
                violations.append(f"key-tile counter is not 0, 1, ...: {counter.text}")
    if bound is None or instructions[bound].text != f"sdiv i32 %seqlen_kv, {block_n}":
        violations.append(
            f"key-tile loop bound is not seqlen_kv / {block_n}: "
            f"{[item.text for item in compares]}"
        )

    scale = [
        item.name
        for item in instructions.values()
        if item.text
        in (f"fmul float %scale, {_F32_LOG2E}", f"fmul float {_F32_LOG2E}, %scale")
    ]
    if len(scale) != 1:
        violations.append(f"expected one fmul(scale, LOG2E), found {len(scale)}")
    reciprocal = [item for item in instructions.values() if item.opcode == "fdiv"]
    if (
        len(reciprocal) != 1
        or not reciprocal[0].text.startswith(f"fdiv float {_F32_ONE}, %")
        or source(reciprocal[0].operands[0]) != normalizer
    ):
        violations.append(
            f"expected one fdiv 1.0, l; found {[item.text for item in reciprocal]}"
        )
    rcp = reciprocal[0].name if reciprocal else None

    # exp2: a probability (normalizer fadd + one cast) or the rescale alpha.
    probabilities: set[str] = set()
    alphas: set[str] = set()
    for item in instructions.values():
        if item.opcode == "fsub" and {u.opcode for u in users(item.name)} != {_EXP2}:
            violations.append(f"fsub not consumed by exp2 only: {item.text}")
        if item.opcode != _EXP2:
            continue
        argument = instructions.get(source(item.operands[0]))
        if argument is None or argument.opcode != "fsub":
            violations.append(f"exp2 argument is not an fsub: {item.text}")
        elif opcode(argument.operands[1]) != _MAXNUM:
            violations.append(f"exp2 does not subtract the row maximum: {item.text}")
        kinds = Counter(user.opcode for user in users(item.name))
        if kinds == Counter({"fadd": 1, "fptrunc": 1}):
            probabilities.add(item.name)
        elif kinds and set(kinds) == {"fmul"}:
            alphas.add(item.name)
        else:
            violations.append(f"exp2 with unmodeled uses {dict(kinds)}: {item.text}")
    counts["exp2_probability"] = len(probabilities)
    counts["exp2_alpha"] = len(alphas)

    def element_of(name: str, producer: str) -> bool:
        item = instructions.get(source(name))
        return (
            item is not None
            and item.opcode == "extractelement"
            and opcode(item.operands[0]) == producer
        )

    for item in instructions.values():
        if item.opcode != "fmul" or item.name in scale:
            continue
        left, right = item.operands
        if right in scale and element_of(left, _MFMA_ATOM[dtype]):
            counts["score"] += 1
        elif source(right) in alphas and source(left) == normalizer:
            counts["rescale_normalizer"] += 1
        elif source(right) in alphas and element_of(left, "phi"):
            counts["rescale_accumulator"] += 1
        elif right == rcp and {u.opcode for u in users(item.name)} == {"fptrunc"}:
            counts["output_product"] += 1
        else:
            violations.append(f"unmodeled fmul: {item.text}")

    # Normalizer: a lane chain of uncast probabilities from +0, the partner
    # lane's sum, then l * alpha + tile sum.
    chain: dict[str, int] = {}
    shuffles: set[str] = set()
    for item in instructions.values():
        if item.opcode != "fadd":
            continue
        left, right = item.operands[0], item.operands[-1]
        rescaled = instructions.get(source(left))
        if source(right) in probabilities and (
            item.text.startswith(f"fadd float {_F32_ZERO}, ") or source(left) in chain
        ):
            chain[item.name] = chain.get(source(left), 0) + 1
        elif source(left) in chain and opcode(right) == "bitcast":
            shuffles.add(item.name)
            counts["lane_chain"] = chain[source(left)]
            if opcode(instructions[source(right)].operands[0]) != _BPERMUTE:
                violations.append(f"partner sum is not a lane shuffle: {item.text}")
        elif (
            rescaled is not None
            and rescaled.opcode == "fmul"
            and source(rescaled.operands[0]) == normalizer
            and source(right) in shuffles
        ):
            counts["combine"] += 1
            if normalizer not in {user.name for user in users(item.name)}:
                violations.append(f"combine is not the next normalizer: {item.text}")
        else:
            violations.append(f"unmodeled fadd: {item.text}")
    counts["shuffle"] = len(shuffles)

    cast_type = _CAST_TYPE[dtype]
    for item in instructions.values():
        if item.opcode == "fpext":
            violations.append(f"unmodeled fpext: {item.text}")
        if item.opcode != "fptrunc":
            continue
        if not item.text.endswith(f" to {cast_type}"):
            violations.append(f"cast to a type other than {cast_type}: {item.text}")
        operand = instructions.get(source(item.operands[0]))
        if operand is not None and operand.name in probabilities:
            counts["cast_probability"] += 1
        elif (
            operand is not None and operand.opcode == "fmul" and rcp in operand.operands
        ):
            counts["cast_output"] += 1
        else:
            violations.append(f"unmodeled cast: {item.text}")

    for item in instructions.values():
        if item.opcode == _MFMA_ATOM[dtype]:
            counts["mfma"] += 1
            if not item.text.endswith("i32 0, i32 0, i32 0)"):
                violations.append(f"MFMA with nonzero controls: {item.text}")
    return violations, counts


_INSERT = re.compile(
    r"insertelement <(\d+) x \w+> (undef|%[-\w.]+), \w+ (%[-\w.]+), i32 (\d+)$"
)
_EXTRACT = re.compile(r"extractelement <\d+ x \w+> (%[-\w.]+), i32 (\d+)$")
_LDS_FILL = re.compile(
    r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds\("
    r"ptr addrspace\(8\) (%[-\w.]+), ptr addrspace\(3\) (%[-\w.]+),"
)
_LDS_ARRAY = re.compile(
    r"getelementptr inbounds \[1 x \[(\d+) x \[(\d+) x \w+\]\]\], "
    r"ptr addrspace\(3\) (@[-\w.]+|%[-\w.]+),"
)
_LDS_OFFSET = re.compile(
    r"getelementptr inbounds i8, ptr addrspace\(3\) @[-\w.]+, i32 (\d+)$"
)
_LDS_BASE = re.compile(r"ptrtoint ptr addrspace\(3\) (@[-\w.]+|%[-\w.]+) to i64$")
# The intrinsic's suffix names its pointer types and differs between LLVM
# versions (``.p1``, ``.p8.p1``).
_BUFFER = "llvm.amdgcn.make.buffer.rsrc."


def trace_mfma_operands(ir: str, dtype: str) -> tuple[list, Counter]:
    """Trace every MFMA operand of one lowered unmasked kernel back to the
    tensor it is loaded from, and measure every chain of MFMA calls.

    The score chains start from a zero vector, multiply K tiles read from the K
    region of the LDS by Q read from ``q_ptr``, and end in the 16 scores of a
    lane. The value chains start from the accumulator rescaled by ``alpha``,
    multiply V read from the V region of the LDS by the cast probabilities,
    each of which enters every chain once, and end in the accumulator the loop
    carries. The K region is filled from ``k_ptr`` only and the V region from
    ``v_ptr`` only. Which token lands in which LDS row is index arithmetic and
    is not traced; the relation itself tests it on the device."""
    violations: list[str] = []
    counts: Counter = Counter()
    instructions = _parse(ir)
    uses: dict[str, list[_Instruction]] = {}
    for instruction in instructions.values():
        for operand in instruction.operands:
            uses.setdefault(operand, []).append(instruction)

    def through(name: str) -> str:
        """Follow every ``bitcast``, type-changing ones included."""
        while name in instructions and instructions[name].opcode == "bitcast":
            name = instructions[name].operands[0]
        return name

    def users(name: str) -> list[_Instruction]:
        found: list[_Instruction] = []
        for user in uses.get(name, []):
            found.extend(users(user.name) if user.opcode == "bitcast" else [user])
        return found

    def elements(name: str) -> list[str] | None:
        """The values at positions ``0..n-1`` of an ``insertelement`` chain."""
        found: dict[int, str] = {}
        width = None
        name = through(name)
        while name != "undef":
            item = instructions.get(name)
            match = _INSERT.match(item.text) if item else None
            if match is None:
                return None
            width, index = int(match.group(1)), int(match.group(4))
            if index in found:
                return None
            found[index] = through(match.group(3))
            name = match.group(2)
            if name != "undef":
                name = through(name)
        if width is None or sorted(found) != list(range(width)):
            return None
        return [found[index] for index in range(width)]

    def scalar(name: str) -> str:
        """Follow ``extractelement`` of an ``insertelement`` chain to the value."""
        name = through(name)
        while name in instructions:
            match = _EXTRACT.match(instructions[name].text)
            packed = elements(match.group(1)) if match else None
            if packed is None or int(match.group(2)) >= len(packed):
                break
            name = packed[int(match.group(2))]
        return name

    def lds_offset(base: str) -> int | None:
        """Byte offset of an LDS base inside the pool: the pool itself, or an
        ``i8`` offset from it."""
        if base.startswith("@"):
            return 0
        item = instructions.get(base)
        match = _LDS_OFFSET.match(item.text) if item else None
        return int(match.group(1)) if match else None

    def loaded_from(name: str) -> tuple[str, object]:
        """``("lds", (offset, bytes of the array))`` or ``("global", pointer)``
        for a ``load``; ``("other", text)`` otherwise."""
        item = instructions.get(through(name))
        if item is None or item.opcode != "load" or not item.operands:
            return ("other", name if item is None else item.text)
        address = instructions.get(item.operands[0])
        if address is None or address.opcode != "getelementptr":
            return ("other", item.text)
        if "ptr addrspace(3)" in item.text:
            array = _LDS_ARRAY.match(address.text)
            offset = lds_offset(array.group(3)) if array else None
            if offset is None:
                return ("other", address.text)
            return ("lds", (offset, 2 * int(array.group(1)) * int(array.group(2))))
        match = re.search(r"ptr addrspace\(1\) (%[-\w.]+),", address.text)
        return ("global", match.group(1)) if match else ("other", address.text)

    # Chains: an MFMA call continues the chain of its accumulator operand.
    atom = _MFMA_ATOM[dtype]
    calls = [item for item in instructions.values() if item.opcode == atom]
    successor: dict[str, list[str]] = {}
    starts: list[_Instruction] = []
    for call in calls:
        previous = through(call.operands[2])
        if previous in instructions and instructions[previous].opcode == atom:
            successor.setdefault(previous, []).append(call.name)
        else:
            starts.append(call)
    chains: list[list[_Instruction]] = []
    for start in starts:
        chain = [start]
        while len(successor.get(chain[-1].name, [])) == 1:
            chain.append(instructions[successor[chain[-1].name][0]])
        if successor.get(chain[-1].name):
            violations.append(f"MFMA result feeds two MFMA calls: {chain[-1].text}")
        chains.append(chain)
    if sum(len(chain) for chain in chains) != len(calls):
        violations.append("an MFMA call is in no chain that starts outside the MFMA")

    probabilities = {
        item.name
        for item in instructions.values()
        if item.opcode == "fptrunc"
        and through(item.operands[0]) in instructions
        and instructions[through(item.operands[0])].opcode == _EXP2
    }
    alphas = {
        item.name
        for item in instructions.values()
        if item.opcode == _EXP2
        and {user.opcode for user in users(item.name)} == {"fmul"}
    }
    k_region: set[tuple[int, int]] = set()
    v_region: set[tuple[int, int]] = set()
    packs_by_step: dict[int, set[tuple[str, ...]]] = {}
    for chain in chains:
        start = instructions.get(through(chain[0].operands[2]))
        if start is not None and start.text.count("zeroinitializer") == 2:
            # Score chain: K from the LDS times Q from q_ptr, from zero.
            counts["score_chains"] += 1
            counts[f"score_chain_length_{len(chain)}"] += 1
            for call in chain:
                kind, where = loaded_from(call.operands[0])
                if kind == "lds":
                    k_region.add(where)
                else:
                    violations.append(
                        f"score MFMA does not read K from the LDS: {where}"
                    )
                if loaded_from(call.operands[1]) != ("global", "%q_ptr"):
                    violations.append(
                        f"score MFMA does not read Q from q_ptr: {call.text}"
                    )
            ends = users(chain[-1].name)
            indices = sorted(
                int(match.group(2))
                for match in (_EXTRACT.match(user.text) for user in ends)
                if match
            )
            if len(ends) != 16 or indices != list(range(16)):
                violations.append(
                    f"score chain does not end in the 16 scores of a lane: {chain[-1].text}"
                )
            continue

        # Value chain: the rescaled accumulator, then V times cast probabilities.
        carried = elements(chain[0].operands[2])
        state = None
        if carried is not None and len(carried) == 16:
            sources = set()
            for index, name in enumerate(carried):
                product = instructions.get(name)
                element = (
                    _EXTRACT.match(instructions[through(product.operands[0])].text)
                    if product is not None
                    and product.opcode == "fmul"
                    and through(product.operands[0]) in instructions
                    else None
                )
                if (
                    element is None
                    or int(element.group(2)) != index
                    or through(product.operands[1]) not in alphas
                ):
                    sources.add(None)
                else:
                    sources.add(through(element.group(1)))
            if len(sources) == 1:
                state = sources.pop()
        loop_state = instructions.get(state) if state else None
        if loop_state is None or not loop_state.text.startswith("phi <16 x float>"):
            violations.append(
                f"value chain does not start from the rescaled accumulator: {chain[0].text}"
            )
        elif chain[-1].name not in {through(name) for name in loop_state.operands}:
            violations.append(
                f"value chain does not end in the accumulator it started from: {chain[-1].text}"
            )
        counts["value_chains"] += 1
        counts[f"value_chain_length_{len(chain)}"] += 1
        for step, call in enumerate(chain):
            rows = elements(call.operands[0])
            if rows is None or len(rows) != 4:
                violations.append(
                    f"value MFMA operand is not four V elements: {call.text}"
                )
            else:
                for row in rows:
                    kind, where = loaded_from(scalar(row))
                    if kind == "lds":
                        v_region.add(where)
                    else:
                        violations.append(
                            f"value MFMA does not read V from the LDS: {where}"
                        )
            pack = elements(call.operands[1])
            if pack is None or len(pack) != 4 or not set(pack) <= probabilities:
                violations.append(
                    f"value MFMA operand is not four cast probabilities: {call.text}"
                )
            else:
                packs_by_step.setdefault(step, set()).add(tuple(pack))

    # Every cast probability enters every value chain exactly once, at the
    # same step in each.
    packed: list[str] = []
    for step, packs in sorted(packs_by_step.items()):
        if len(packs) != 1:
            violations.append(
                f"value chains disagree on the probabilities of atom {step}"
            )
        packed.extend(next(iter(packs)))
    counts["probability_packs"] = len(packs_by_step)
    if sorted(packed) != sorted(probabilities):
        violations.append(
            f"the value chains use {len(set(packed))} of {len(probabilities)} cast "
            f"probabilities, {len(packed) - len(set(packed))} of them twice"
        )

    # The two LDS regions are disjoint: V starts where the K array ends.
    k_offsets = {offset for offset, _ in k_region}
    v_offsets = {offset for offset, _ in v_region}
    k_bytes = {size for _, size in k_region}
    if k_offsets != {0} or len(k_bytes) != 1 or v_offsets != k_bytes:
        violations.append(
            f"K is read at LDS offsets {sorted(k_offsets)} (array bytes "
            f"{sorted(k_bytes)}) and V at {sorted(v_offsets)}"
        )

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

    # LDS fills: the K region from k_ptr only, the V region from v_ptr only.
    for match in _LDS_FILL.finditer(ir):
        buffer = instructions.get(match.group(1))
        source = None
        if buffer is not None and buffer.opcode.startswith(_BUFFER):
            source = buffer.operands[0]
        offset = fill_offset(match.group(2))
        if source == "%k_ptr" and offset in k_offsets:
            counts["k_fills"] += 1
        elif source == "%v_ptr" and offset in v_offsets and offset not in k_offsets:
            counts["v_fills"] += 1
        else:
            violations.append(
                f"LDS fill from {source} into the region at offset {offset}"
            )
    if not counts["k_fills"] or not counts["v_fills"]:
        violations.append("no LDS fill of the K or of the V region")
    return violations, counts


def expected_path_counts(block_n: int, head_size: int) -> dict[str, int]:
    """Instruction counts of the modeled path, per lane and key tile.

    Two lane halves share a key tile; a lane holds 16 accumulator elements per
    32 output columns; an MFMA call takes eight products."""
    lane_keys = block_n // 2
    output_elements = 16 * (head_size // 32)
    return {
        "exp2_probability": lane_keys,
        "exp2_alpha": 1,
        "score": lane_keys,
        "lane_chain": lane_keys,
        "shuffle": 1,
        "combine": 1,
        "rescale_normalizer": 1,
        "rescale_accumulator": output_elements,
        "cast_probability": lane_keys,
        "output_product": output_elements,
        "cast_output": output_elements,
        "mfma": (block_n // 32) * (head_size // 8) + (head_size // 32) * (block_n // 8),
    }


def expected_chain_counts(block_n: int, head_size: int) -> dict[str, int]:
    """MFMA chains of the modeled path: one score chain per 32-key half of a
    tile, ``head_size / 8`` atoms long, and one value chain per 32 output
    columns, ``block_n / 8`` atoms long, fed by ``block_n / 8`` packs of cast
    probabilities."""
    return {
        "score_chains": block_n // 32,
        f"score_chain_length_{head_size // 8}": block_n // 32,
        "value_chains": head_size // 32,
        f"value_chain_length_{block_n // 8}": head_size // 32,
        "probability_packs": block_n // 8,
    }


def unmasked_path_violations(
    ir: str, dtype: str, *, block_n: int, head_size: int
) -> list[str]:
    """Every way one lowered unmasked kernel leaves the path the bound models:
    an unmodeled instruction, a misrouted MFMA operand, or a count that differs
    from the modeled structure."""
    violations, counts = classify_unmasked_path(ir, dtype, block_n)
    traced, chains = trace_mfma_operands(ir, dtype)
    violations = [*violations, *traced]
    fills = {name: chains.pop(name, 0) for name in ("k_fills", "v_fills")}
    for name, found, expected in (
        ("instruction", counts, expected_path_counts(block_n, head_size)),
        ("MFMA chain", chains, expected_chain_counts(block_n, head_size)),
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

    The bound takes this count from the artifact, not from the requested shape:
    the path check establishes that the one KV loop runs
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


# compute_pgm_rsrc1 fields of an AMDGPU kernel descriptor.
_ROUND_NEAREST_EVEN = 0
_DENORM_FLUSH_NONE = 3
_MODE_FIELDS = (
    ("float_round_mode_32", 12, _ROUND_NEAREST_EVEN),
    ("float_round_mode_16_64", 14, _ROUND_NEAREST_EVEN),
    ("float_denorm_mode_32", 16, _DENORM_FLUSH_NONE),
    ("float_denorm_mode_16_64", 18, _DENORM_FLUSH_NONE),
)
_SHT_SYMTAB, _SHT_DYNSYM = 2, 11


def kernel_descriptor_words(code_object: bytes) -> dict[str, int]:
    """``compute_pgm_rsrc1`` of every kernel descriptor of an ELF64 code object.

    A kernel descriptor is the 64-byte object named ``<kernel>.kd``; the word
    is little-endian at offset 48."""
    if code_object[:6] != b"\x7fELF\x02\x01":
        raise ValueError("not a little-endian ELF64 code object")
    (section_offset,) = struct.unpack_from("<Q", code_object, 0x28)
    entry_size, count = struct.unpack_from("<HH", code_object, 0x3A)
    sections = []
    for index in range(count):
        _name, kind, _flags, address, offset, size, link, _info, _align, entries = (
            struct.unpack_from(
                "<IIQQQQIIQQ", code_object, section_offset + index * entry_size
            )
        )
        sections.append((kind, address, offset, size, link, entries))
    words: dict[str, int] = {}
    for kind, _address, offset, size, link, entries in sections:
        if kind not in (_SHT_SYMTAB, _SHT_DYNSYM) or not entries:
            continue
        strings = sections[link][2]
        for at in range(offset, offset + size, entries):
            name_at, _info, _other, section, value, _size = struct.unpack_from(
                "<IBBHQQ", code_object, at
            )
            end = code_object.index(b"\0", strings + name_at)
            name = code_object[strings + name_at : end].decode()
            if not name.endswith(".kd") or not 0 < section < count:
                continue
            _kind, base, start, *_ = sections[section]
            (word,) = struct.unpack_from("<I", code_object, start + value - base + 48)
            words[name] = word
    return words


def fp_mode_violations(code_object: bytes, kernel_name: str) -> list[str]:
    """Ways the kernel's descriptor departs from the mode the bound assumes:
    FP32 and FP16/FP64 rounding to nearest even, with subnormals kept."""
    word = kernel_descriptor_words(code_object).get(f"{kernel_name}.kd")
    if word is None:
        return [f"no kernel descriptor {kernel_name}.kd in the code object"]
    return [
        f"{name}={(word >> shift) & 3}, expected {expected} "
        f"(compute_pgm_rsrc1=0x{word:08x})"
        for name, shift, expected in _MODE_FIELDS
        if (word >> shift) & 3 != expected
    ]
