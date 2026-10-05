# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lint AMDGPU assembly for 64-bit address updates that drop the carry.

A 64-bit address held in a register pair must be advanced with a carry chain: a low add that
produces a carry (s_add_u32, v_add_co_u32) followed by a high add that consumes it (s_addc_u32,
v_addc_co_u32). An add to the low dword with no carry into the high dword is correct only until
the low dword wraps, which happens when the buffer crosses a 4 GiB boundary: the access then
lands 4 GiB away (ROCM-32046, class C on AIHPBLAS-4988).

The lint finds the registers used as the low dword of an address: the base of a buffer resource
descriptor (s[N:N+3] in a buffer instruction), the address pair of a scalar load (s[A:A+1]), the
address pair or saddr of a global or flat access, and any register copied into one of those. For
each write to such a register by an add or subtract, it requires either a carry-producing
instruction followed by a carry-consuming write to the next register, or a full redefinition of
the pair. A scalar carry is followed until SCC changes, since the scheduler can move the carry-in
far from the carry-out; a vector carry, until its carry register is written again. Anything else
is reported if the updated value can reach a use as an address before it is overwritten,
following branches and loop back-edges for up to FLOW_STEPS instructions. The search does not
know which branches go together, so a register reused for an integer can look like an address
along a path that never runs. An add of two constants sets the register rather than advancing an
address, and an add to the result of a bit operation such as xor is integer arithmetic, so
neither is reported. A finding is a lead to read, not a proof, and a clean result is not one
either: the lint does not see addresses held in other forms, such as TDM descriptor groups.

It is meant for Tensile-generated and hand-written kernels, which keep a 64-bit value in an
adjacent register pair. Compiler-generated code may keep the two halves of a sum in unrelated
registers, so the lint reports false findings there and is not meant for it.

    python -m Tensile.Utilities.address_carry_lint kernel.s
    python -m Tensile.Utilities.address_carry_lint --disassemble TensileLibrary_gfx942.co
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Low adds that set a carry, and the high adds that must consume it.
# gfx12 renamed the scalar forms (s_add_co_u32, s_add_co_ci_u32, s_add_co_i32, ...).
CARRY_OUT = {
    "s_add_u32",
    "s_sub_u32",
    "s_add_co_u32",
    "s_sub_co_u32",
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
}
CARRY_IN = {
    "s_addc_u32",
    "s_subb_u32",
    "s_add_co_ci_u32",
    "s_sub_co_ci_u32",
    "v_addc_co_u32",
    "v_subb_co_u32",
    "v_subbrev_co_u32",
    "v_add_co_ci_u32",
    "v_sub_co_ci_u32",
    "v_subrev_co_ci_u32",
}
# A carry-in add sets a carry out as well, so on a low dword it must be carried on too.
CARRY_SETTERS = CARRY_OUT | CARRY_IN
# Adds that update one dword and drop any carry.
NO_CARRY = {
    "s_add_i32",
    "s_sub_i32",
    "s_add_co_i32",
    "s_sub_co_i32",
    "s_addk_i32",
    "v_add_u32",
    "v_sub_u32",
    "v_subrev_u32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_add_i32",
    "v_sub_i32",
    "v_add_nc_i32",
    "v_sub_nc_i32",
}
# How a global or flat atomic asks for the old value: a _rtn mnemonic, glc (gfx90a), sc0
# (gfx94x and gfx950) or th:TH_ATOMIC_RETURN (gfx12 and later).
_ATOMIC_RETURN = re.compile(r"_rtn\b|\bglc\b|\bsc0\b|\bTH_ATOMIC_RETURN\b")
# Instructions that rewrite a whole pair, which ends any obligation on its low dword.
PAIR_DEFS = {"s_mov_b64", "v_mov_b64", "v_lshlrev_b64", "s_lshl_b64", "s_add_u64", "v_add_nc_u64"}
# Unconditional transfers: the next instruction in the listing is not the next one executed.
JUMPS = {"s_branch", "s_setpc_b64", "s_endpgm"}
# 32-bit copies, through which an updated low dword can reach the register used as the address.
COPIES = {"s_mov_b32", "v_mov_b32"}
# Bit operations whose result is a plain integer, never an address: an add that updates their
# result in place, such as Tensile's sign extension of the workgroup mapping (xor, then subtract
# the same constant), is integer arithmetic even when the register is an address elsewhere.
# s_and_b32 is left out, since aligning an address and then advancing it needs a carry.
INTEGER_BIT_OPS = {
    "s_xor_b32",
    "s_xnor_b32",
    "s_not_b32",
    "s_bfe_u32",
    "s_bfe_i32",
    "v_xor_b32",
    "v_not_b32",
    "v_bfe_u32",
    "v_bfe_i32",
}
# Scalar instructions that write SCC, which ends a scalar carry chain.
_SCC_WRITERS = re.compile(
    r"^s_(add|sub|addc|subb|addk|cmp|cmpk|bitcmp|and|or|xor|andn[12]|orn[12]|nand|nor|xnor|not|"
    r"lshl|lshr|ashr|bfe|abs|absdiff|min|max|bcnt|quadmask|wqm)"
)

# How many instructions a written low dword, or a carry, is followed through, along every branch
# and loop back-edge. A main loop body fits comfortably.
FLOW_STEPS = 4000

_REG = re.compile(r"^(?P<kind>[sv])(?:(?P<num>\d+)|\[(?P<lo>[^\]:]+)(?::(?P<hi>[^\]]+))?\])")


@dataclass(frozen=True)
class Reg:
    kind: str  # "s" or "v"
    base: str  # number as text, or a symbol such as sgprSrdA
    offset: int

    def plus(self, n: int) -> "Reg":
        return Reg(self.kind, self.base, self.offset + n)

    def __str__(self) -> str:
        if self.base.isdigit():
            return f"{self.kind}{int(self.base) + self.offset}"
        return f"{self.kind}[{self.base}+{self.offset}]"


def _single(text: str, kind: str) -> Reg:
    text = text.strip()
    if text.isdigit():
        return Reg(kind, "0", int(text))
    m = re.match(r"^([A-Za-z_]\w*)(?:\s*\+\s*(\d+))?$", text)
    if m:
        return Reg(kind, m.group(1), int(m.group(2) or 0))
    return Reg(kind, text, 0)


def parse_regs(operand: str) -> list[Reg]:
    """The registers an operand names, low to high; empty for a non-register operand."""
    m = _REG.match(operand.strip())
    if not m:
        return []
    kind = m.group("kind")
    if m.group("num") is not None:
        return [Reg(kind, "0", int(m.group("num")))]
    lo = _single(m.group("lo"), kind)
    if m.group("hi") is None:
        return [lo]
    hi = _single(m.group("hi"), kind)
    if hi.base != lo.base or hi.offset < lo.offset:
        return [lo]
    return [lo.plus(i) for i in range(hi.offset - lo.offset + 1)]


_VCC_PARTS = {"vcc": {"vcc_lo", "vcc_hi"}, "vcc_lo": {"vcc_lo"}, "vcc_hi": {"vcc_hi"}}


def _carry_parts(operand: str) -> set:
    """What a carry operand occupies: its registers, or the halves of vcc, so that a write to
    any part of a carry register is seen as changing it."""
    operand = operand.strip()
    if operand in _VCC_PARTS:
        return set(_VCC_PARTS[operand])
    regs = parse_regs(operand)
    return set(regs) if regs else ({operand} if operand else set())


@dataclass
class Instruction:
    line: int
    text: str
    mnemonic: str
    operands: list[str]
    labels: tuple[str, ...] = ()  # labels that name this instruction


def parse(asm: str, first_line: int = 1) -> list[Instruction]:
    out = []
    pending: list[str] = []
    for n, raw in enumerate(asm.splitlines(), first_line):
        text = re.split(r"//|;", raw)[0].strip()
        if text.endswith(":"):
            # label_X: in assembly, <label_X>: in disassembly.
            pending.append(text[:-1].strip().strip("<>"))
            continue
        if not text or text.startswith("."):
            continue
        # Hand-written kernels separate the mnemonic from its operands with a tab.
        mnemonic, _, rest = text.replace("\t", " ").partition(" ")
        if not re.match(r"^[sv]_|^buffer_|^global_|^flat_|^scratch_", mnemonic):
            continue
        mnemonic = re.sub(r"_e(32|64)$", "", mnemonic)
        operands = [o.strip() for o in rest.split(",")] if rest.strip() else []
        out.append(Instruction(n, text, mnemonic, operands, tuple(pending)))
        pending = []
    return out


def address_operands(inst: Instruction) -> list[Reg]:
    """The registers this instruction uses as the low dword of an address or descriptor base."""
    regs = [parse_regs(o) for o in inst.operands]
    m = inst.mnemonic
    if m.startswith(("buffer_", "s_buffer_")):
        # The descriptor is the SGPR quadruple; a VGPR quadruple is data.
        return [r[0] for r in regs if len(r) == 4 and r[0].kind == "s"]
    if m.startswith(("s_load_", "s_store_", "s_scratch_")):
        return [regs[1][0]] if len(regs) > 1 and len(regs[1]) == 2 else []
    if m.startswith(("global_", "flat_")):
        # Leave out the data and destination operands: a load writes operand 0, a store reads
        # its data from operand 1, and a returning atomic does both.
        if "_atomic" in m:
            data = {0, 2} if _ATOMIC_RETURN.search(inst.text) else {1}
        elif "_store" in m:
            data = {1}
        else:
            data = {0}
        return [r[0] for k, r in enumerate(regs) if k not in data and len(r) == 2]
    return []


def written(inst: Instruction) -> list[Reg]:
    """The registers this instruction writes: its first operand, unless it has no destination."""
    if not inst.operands or inst.mnemonic.startswith(
        ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_setpc", "s_waitcnt", "s_nop")
    ):
        return []
    if re.search(r"_store|_atomic", inst.mnemonic) and not (
        "_atomic" in inst.mnemonic and _ATOMIC_RETURN.search(inst.text)
    ):
        return []
    return parse_regs(inst.operands[0])


def address_lows(insts: Iterable[Instruction]) -> set[Reg]:
    """Registers used as the low dword of a 64-bit address or a descriptor base."""
    lows: set[Reg] = set()
    for inst in insts:
        lows.update(address_operands(inst))
    return lows


@dataclass
class Finding:
    line: int
    text: str
    reason: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.text}\n    {self.reason}"


# A kernel starts at Tensile's label_ASM_Start, at an assembler kernel directive, or, in
# disassembly, at a function symbol header such as <Dijk_SS_MT256x1_VW1_Reduction>:.
_KERNEL_START = re.compile(
    r"label_ASM_Start>?:|^\s*\.amdgpu_hsa_kernel\b|^\s*\.globl\b|^<(?!label_)[^>\s]+>:\s*$",
    re.M,
)


def _kernel_chunks(asm: str) -> list[tuple[int, str]]:
    """One (first line number, text) chunk per kernel."""
    starts = [m.start() for m in _KERNEL_START.finditer(asm)]
    if not starts:
        return [(1, asm)]
    bounds = [0] + starts[1:] + [len(asm)]
    chunks, line, pos = [], 1, 0
    for a, b in zip(bounds, bounds[1:]):
        line += asm.count("\n", pos, a)
        pos = a
        chunks.append((line, asm[a:b]))
    return chunks


def kernels(asm: str) -> list[str]:
    """Splits assembly into one chunk per kernel, so registers are judged within their kernel."""
    return [text for _, text in _kernel_chunks(asm)]


def lint(asm: str) -> list[Finding]:
    findings = []
    for first_line, chunk in _kernel_chunks(asm):
        findings += _lint_kernel(chunk, first_line)
    return findings


def _lint_kernel(asm: str, first_line: int = 1) -> list[Finding]:
    insts = parse(asm, first_line)
    uses = [address_operands(inst) for inst in insts]
    writes = [written(inst) for inst in insts]
    lows = {r for u in uses for r in u}
    # A register copied into an address's low dword holds an address too (two copies deep).
    for _ in range(2):
        lows |= {
            parse_regs(inst.operands[-1])[0]
            for inst, w in zip(insts, writes)
            if inst.mnemonic in COPIES
            and w[:1]
            and w[0] in lows
            and len(parse_regs(inst.operands[-1])) == 1
        }

    target = {label: k for k, inst in enumerate(insts) for label in inst.labels}

    def successors(j: int) -> list[int]:
        """The instructions that can run after j: the next one unless j always jumps, and the
        target of a branch to a label in this kernel."""
        m = insts[j].mnemonic
        out = [] if m in JUMPS else [j + 1]
        if m == "s_branch" or m.startswith("s_cbranch"):
            dest = target.get(insts[j].operands[0].strip("<>")) if insts[j].operands else None
            if dest is not None:
                out.append(dest)
        return [k for k in out if k < len(insts)]

    def flows_to_address(low: Reg, start: int, depth: int = 0) -> bool:
        """Whether the value written at start can reach a use as an address before it is
        overwritten, along any path (loop back-edges included), directly or through a copy."""
        seen, todo, steps = set(), successors(start), 0
        while todo and steps < FLOW_STEPS:
            j = todo.pop()
            if j in seen:
                continue
            seen.add(j)
            steps += 1
            if low in uses[j]:
                return True
            inst = insts[j]
            if depth < 2 and inst.mnemonic in COPIES and parse_regs(inst.operands[-1]) == [low]:
                copy = writes[j][:1]
                if copy and copy[0] in lows and flows_to_address(copy[0], j, depth + 1):
                    return True
            if low in writes[j]:
                continue
            todo.extend(successors(j))
        return False

    def consumes(j: int, low: Reg, high: Reg) -> bool:
        """Whether instruction j adds a carry into high, or redefines the whole pair."""
        return (insts[j].mnemonic in CARRY_IN and writes[j][:1] == [high]) or (
            insts[j].mnemonic in PAIR_DEFS and writes[j][:1] == [low]
        )

    def carry_writes(j: int) -> set:
        """The carry-register parts instruction j writes: its destination, including vcc, and
        the carry-out of another add."""
        ops = insts[j].operands
        out = set(writes[j])
        if ops and ops[0] in _VCC_PARTS:
            out |= _carry_parts(ops[0])
        if insts[j].mnemonic in CARRY_SETTERS and len(ops) > 1:
            out |= _carry_parts(ops[1])
        return out

    def vector_carried(start: int, low: Reg, high: Reg) -> bool:
        """Whether a vector carry reaches high: a carry-in to high that reads the carry register
        the low add wrote, before anything writes any part of that register."""
        ops = insts[start].operands
        carry = _carry_parts(ops[1]) if len(ops) > 1 else set()
        for j in range(start + 1, min(len(insts), start + 1 + FLOW_STEPS)):
            ops = insts[j].operands
            if insts[j].mnemonic in PAIR_DEFS and writes[j][:1] == [low]:
                return True
            if insts[j].mnemonic in CARRY_IN and writes[j][:1] == [high]:
                return bool(ops) and bool(carry) and _carry_parts(ops[-1]) == carry
            if carry_writes(j) & carry or insts[j].mnemonic in JUMPS:
                return False
        return False

    def scalar_carried(start: int, low: Reg, high: Reg) -> bool:
        """Whether a scalar carry reaches high. The scheduler may move the carry-in well away
        from the carry-out, so it is followed until SCC changes rather than for a fixed count.
        """
        for j in range(start + 1, min(len(insts), start + 1 + FLOW_STEPS)):
            if consumes(j, low, high):
                return True
            if insts[j].mnemonic in JUMPS or _SCC_WRITERS.match(insts[j].mnemonic):
                return False
        return False

    def updates_bit_op_result(i: int, low: Reg) -> bool:
        """Whether instruction i updates low in place, and the last write to low before it, in
        the same basic block, is a bit operation that makes a plain integer."""
        if not any(parse_regs(o) == [low] for o in insts[i].operands[1:]):
            return False
        j = i
        while j > 0 and not insts[j].labels:
            j -= 1
            if low in writes[j]:
                return insts[j].mnemonic in INTEGER_BIT_OPS
        return False

    findings = []
    for i, inst in enumerate(insts):
        dst = writes[i]
        if len(dst) != 1 or dst[0] not in lows:
            continue
        low, high = dst[0], dst[0].plus(1)
        if updates_bit_op_result(i, low):
            continue
        if inst.mnemonic in NO_CARRY:
            if not any(parse_regs(o) for o in inst.operands[1:]):
                continue
            if flows_to_address(low, i):
                findings.append(
                    Finding(
                        inst.line,
                        inst.text,
                        f"{inst.mnemonic} updates {low}, the low dword of an address, with no "
                        f"carry into {high}",
                    )
                )
        elif inst.mnemonic in CARRY_SETTERS:
            if inst.mnemonic.startswith("s_"):
                carried, where = scalar_carried(i, low, high), "before SCC changes"
            else:
                carried = vector_carried(i, low, high)
                where = "from the carry it wrote, before that carry changes,"
            if not carried and flows_to_address(low, i):
                findings.append(
                    Finding(
                        inst.line,
                        inst.text,
                        f"{inst.mnemonic} sets a carry out of {low}, the low dword of an "
                        f"address, but nothing {where} adds it into {high}",
                    )
                )
    return findings


def disassemble(code_object: Path) -> str:
    objdump = shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
    return subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", "--no-leading-addr", str(code_object)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path, help="assembly files or code objects")
    parser.add_argument(
        "--disassemble",
        action="store_true",
        help="treat the paths as code objects and disassemble them with llvm-objdump",
    )
    args = parser.parse_args(argv)
    total = 0
    for path in args.paths:
        asm = disassemble(path) if args.disassemble else path.read_text(errors="replace")
        findings = lint(asm)
        total += len(findings)
        for f in findings:
            print(f"{path}: {f}")
    print(f"{total} finding(s) in {len(args.paths)} file(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
