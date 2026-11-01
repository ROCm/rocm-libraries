"""Exec-join ISA lint of gfx950 disassembly (FlyTape ``scripts/lint_isa.py``'s
rule, on ``llvm-objdump -d`` output instead of LLVM's ``.s``).

    python r2_lint.py OBJDUMP.s ...

A divergent ``if`` or loop ends in a join block that restores EXEC
(``s_or_b64 exec, exec, s[..]`` after an ``if``, ``s_mov_b64 exec, s[..]``
after an EXEC-masked loop or poll). A VGPR or AGPR written in that block
before the restore runs under the narrowed EXEC (an empty one when the
branch was skipped), so the other lanes keep stale values: a spill or
rematerialization placed there corrupts them. ``v_writelane`` ignores EXEC
and is exempt. Blocks are cut at every branch target and after every branch
or ``s_endpgm``. Exits non-zero on a finding.
"""

from __future__ import annotations

import re
import sys

INST = re.compile(r"^\s+([a-z_0-9]+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):")
TARGET = re.compile(r"<[^>]*\+0x([0-9a-f]+)>")
RESTORE = re.compile(r"^s_(or|mov)_b64$")
VECTOR_DST = re.compile(r"^(v\d+|v\[\d+:\d+\]|a\d+|a\[\d+:\d+\])\s*(,|$)")
NO_VECTOR_DST = re.compile(
    r"^(s_|v_writelane|v_readlane|v_readfirstlane|v_cmp|ds_write|ds_add|global_store|buffer_store|global_atomic(?!.*_rtn))"
)


def parse(lines: list[str]):
    """``[(addr, op, operands, line)]`` and the kernel start address."""
    out, base = [], None
    for number, line in enumerate(lines, 1):
        m = INST.match(line)
        if not m:
            continue
        op, operands, addr = m.group(1), m.group(2), int(m.group(3), 16)
        if base is None:
            base = addr
        out.append((addr, op, operands, number, line))
    return out, base or 0


def findings(lines: list[str]):
    insts, base = parse(lines)
    starts = {insts[0][0]} if insts else set()
    for i, (addr, op, operands, _, line) in enumerate(insts):
        if op.startswith("s_branch") or op.startswith("s_cbranch") or op == "s_endpgm":
            if i + 1 < len(insts):
                starts.add(insts[i + 1][0])
            t = TARGET.search(line)
            if t:
                starts.add(base + int(t.group(1), 16))
    out, early, block = [], [], None
    for addr, op, operands, number, _ in insts:
        if addr in starts:
            block, early = addr, []
        if block is None:
            continue
        dst = operands.split(",")[0].strip()
        if RESTORE.match(op) and dst == "exec" and "s[" in operands.split(",")[-1]:
            if early:
                out.append((block, number, early))
            block = None
        elif "exec" in dst or "saveexec" in op:
            block = None
        elif not NO_VECTOR_DST.match(op) and VECTOR_DST.match(operands.strip()):
            early.append((number, f"{op} {operands}"))
    return out


def main(paths: list[str]) -> int:
    total = 0
    for path in paths:
        with open(path) as f:
            found = findings(f.read().splitlines())
        for block, restore, early in found:
            print(f"{path}:{restore}: block 0x{block:x}: {len(early)} vector write(s) before the exec restore")
            for number, text in early:
                print(f"    {number}: {text}")
        total += len(found)
    print(f"{total} offending join block(s) in {len(paths)} file(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
