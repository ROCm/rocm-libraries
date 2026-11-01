"""Classify the sites where ``hazard_corpus`` finds the direct compiler's
hazard recognizer waiting less than COMGR.

    python r2_hazard_classify.py REPORT.tsv OBJDUMP.s [...pairs]

``hazard_corpus`` strips every ``s_nop`` from COMGR's disassembly and puts
back the ones the recognizer asks for; it cannot tell a hazard wait from an
``s_nop`` that is no wait. Each UNDER row is classified from the
disassembly, by what precedes the stripped ``s_nop`` run:

  padding   the run follows an unconditional ``s_branch`` or ``s_endpgm`` (the
            bytes are never executed: LLVM's loop-header alignment, or the
            256-byte kernarg preload entry, which ends in ``s_branch`` to +256);
  asm       the run sits next to an inline-asm statement's instruction:
            ``s_sleep``, an explicit ``s_waitcnt vmcnt(0)``, a lone ``s_nop 0``
            that is the ``tagged_poll`` pause (LLVM lowers the pause as inline
            asm, and puts a conservative wait in front of inline asm), i.e. the
            run is followed by, or ends with, one of those statements; or one
            of the run's three predecessors is the shipped builds' pointer asm
            (``s_mov_b64 s[a:b], s[a:b]``, ``device_ops.ptr_from_i64`` after
            LLVM coalesced it): LLVM counts an inline-asm statement as no wait
            state, the recognizer counts the instruction it is;
  hazard    anything else: a recognizer gap (the gate fails).

Exits non-zero on any ``hazard`` row. Prints every row with its context.
"""

from __future__ import annotations

import re
import sys

INST = re.compile(r"^\s+([a-z_0-9]+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):")


def parse(path: str):
    out = []
    for line in open(path):
        m = INST.match(line)
        if m:
            out.append((int(m.group(3), 16), m.group(1), m.group(2)))
    return out


def classify(insts, addr: int):
    at = next(i for i, x in enumerate(insts) if x[0] == addr)
    j = at - 1
    while j >= 0 and insts[j][1] == "s_nop":
        j -= 1
    run = insts[j + 1 : at]
    before = insts[j] if j >= 0 else None
    cur = insts[at]
    ctx = [f"{a:#x} {o} {r}" for a, o, r in insts[max(0, j - 2) : at + 2]]
    if before is not None and before[1] in ("s_branch", "s_endpgm"):
        return "padding", ctx
    asm_like = {"s_sleep", "s_waitcnt"}
    if cur[1] in asm_like or (before is not None and before[1] in asm_like):
        return "asm", ctx
    # the tagged_poll pause: one `s_nop 0` (an asm statement) closing the retry
    # path, then the back edge or the loop head's poll load
    for a, o, r in insts[max(0, j - 2) : j + 1]:
        ops = [x.strip() for x in r.split(",")]
        if o == "s_mov_b64" and len(ops) == 2 and ops[0] == ops[1]:
            return "asm", ctx
    pause = len(run) >= 1 and run[-1][2].strip() == "0"
    if pause and (cur[1] == "s_branch" or (cur[1].startswith("buffer_load") and "sc1" in cur[2])):
        return "asm", ctx
    return "hazard", ctx


def main(args: list[str]) -> int:
    bad = 0
    for tsv, dis in zip(args[::2], args[1::2]):
        insts = parse(dis)
        for line in open(tsv):
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[5] != "UNDER":
                continue
            addr = int(f[1], 16)
            kind, ctx = classify(insts, addr)
            bad += kind == "hazard"
            print(f"{tsv}: {f[1]} {f[2]} comgr={f[3]} ours={f[4]} -> {kind}")
            for c in ctx:
                print(f"      {c}")
    print(f"{bad} unexplained site(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
