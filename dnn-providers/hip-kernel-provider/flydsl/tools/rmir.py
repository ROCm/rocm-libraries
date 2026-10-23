# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""R-MIR gate: scan the SGPR greedy-allocation MIR of a build for the `InlineSpiller::rmRedundantSpills` bug (a tracked LLVM issue).

    rmir.py --mir FILE           scan one printed MIR dump

Rule: after SGPR greedy RA, every virtual-SGPR lane read must have a definition on all paths, tracked through SI_SPILL_S*
SAVE/RESTORE per stack slot. A finding on a register that is the target of an SI_SPILL_S*_RESTORE is the bug's signature
(the spill findings, the gate value). Findings on other registers do not fail the gate (merged-kernel undefined
descriptor lanes, a different mechanism present before register allocation).

`split_dumps` and `scan_text` take the text LLVM printed under `print-after=greedy`; `sdpa_bwd_gate.py` drives them.

Source: the sweep's `tools/rmir.py` at the sweep snapshot (sweep commit e196d0d), without its `build_and_scan` (the generators dump and scan in
their own compile). The lane scanner `mirlanescan2` (`parse`, `ops`, `scan`) is copied verbatim.
"""

import json, os, re, sys
from pathlib import Path
from collections import defaultdict

BB = re.compile(r"^(?:\d+B)?\t?bb\.(\d+)[.: ]")
SUCC = re.compile(r"successors: (.*?)(?:;|$)")
OPND = re.compile(
    r"((?:(?:dead|undef|early-clobber|implicit-def|implicit|renamable|internal|killed|debug-use)\s+)*)%(\d+)(?:\.([A-Za-z0-9_]+))?:(\w+)"
)


def cls_lanes(c):
    m = re.match(r"sgpr_(\d+)$", c)
    if m:
        return int(m.group(1)) // 32
    m = re.match(r"sreg_(\d+)", c)
    if m:
        return int(m.group(1)) // 32
    return 0


def sub_mask(sub, n):
    if sub is None:
        return (1 << n) - 1
    toks = sub.split("_")
    m = 0
    if toks[0] in ("lo16", "hi16"):
        m |= 1
    for t in toks:
        mm = re.match(r"sub(\d+)$", t)
        if mm:
            m |= 1 << int(mm.group(1))
    return m or ((1 << n) - 1)


def parse(path):
    blocks = {}
    order = []
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        m = BB.match(line)
        if m:
            cur = int(m.group(1))
            blocks[cur] = {"succ": [], "ins": []}
            order.append(cur)
            continue
        if cur is None:
            continue
        s = SUCC.search(line)
        if s and "%bb." in line and "successors" in line:
            blocks[cur]["succ"] = [
                int(x) for x in re.findall(r"%bb\.(\d+)", s.group(1))
            ]
            continue
        if (
            line.startswith("\t;")
            or not line.strip()
            or line.strip().startswith("liveins")
        ):
            continue
        if line.strip() in ("}",):
            continue
        blocks[cur]["ins"].append(line)
    return blocks, order


def ops(line):
    body = re.sub(r"^\d+B\t", "", line).strip()
    if body.startswith("BUNDLE"):
        return [], []
    if body.startswith("INLINEASM"):
        # operands are `[regdef:..], def [flags] %v[.sub]:cls` / `[reguse:..], %v:cls`; the plain operand parser would read defs as uses
        d, u = [], []
        for m in re.finditer(
            r"(def\s+)?" + OPND.pattern, body.split("$0:", 1)[-1] if False else body
        ):
            t = (m.group(2), int(m.group(3)), m.group(4), m.group(5))
            (d if m.group(1) else u).append(t)
        return d, u
    if " = " in body:
        lhs, rhs = body.split(" = ", 1)
    else:
        lhs, rhs = "", body
    rhs = rhs.split("::")[0]
    d = [
        (m.group(1), int(m.group(2)), m.group(3), m.group(4))
        for m in OPND.finditer(lhs)
    ]
    u = [
        (m.group(1), int(m.group(2)), m.group(3), m.group(4))
        for m in OPND.finditer(rhs)
    ]
    return d, u


def scan(path, verbose=True):
    blocks, order = parse(path)
    pred = defaultdict(list)
    for b, v in blocks.items():
        for s in v["succ"]:
            pred[s].append(b)
    # per-block transfer; findings computed after fixpoint
    IN = {b: None for b in blocks}
    OUT = {b: None for b in blocks}
    entry = order[0]

    def transfer(b, st, report=None):
        st = dict(st)
        for ln in blocks[b]["ins"]:
            d, u = ops(ln)
            msv = re.search(
                r"SI_SPILL_S\d+_SAVE %(\d+)(?:\.(\w+))?:(\w+), %stack\.(\d+)", ln
            )
            msr = re.search(r"%(\d+):(\w+) = SI_SPILL_S\d+_RESTORE %stack\.(\d+)", ln)
            if msv:
                v0 = int(msv.group(1))
                n0 = cls_lanes(msv.group(3))
                st[("s", int(msv.group(4)))] = st.get(v0, 0) & ((1 << n0) - 1)
                continue
            if msr:
                v0 = int(msr.group(1))
                n0 = cls_lanes(msr.group(2))
                st[v0] = st.get(("s", int(msr.group(3))), 0) & ((1 << n0) - 1)
                continue
            if "V_PK_" in ln:
                u = []
            for fl, v, sub, c in u:
                n = cls_lanes(c)
                if not n or "undef" in fl.split():
                    continue
                need = sub_mask(sub, n)
                have = st.get(v, 0)
                if need & ~have and report is not None:
                    report.append((b, ln.strip()[:200], v, sub, bin(need & ~have)))
            for fl, v, sub, c in d:
                n = cls_lanes(c)
                if not n:
                    continue
                if sub is None and "undef" in fl.split():
                    pass
                st[v] = st.get(v, 0) | sub_mask(sub, n)
        return st

    work = [entry]
    IN[entry] = {}
    changed = True
    while changed:
        changed = False
        for b in order:
            if b == entry:
                ins = {}
            else:
                ps = [OUT[p] for p in pred[b] if OUT[p] is not None]
                if not ps:
                    continue
                keys = set().union(*(o.keys() for o in ps))
                ins = {}
                for k in keys:
                    if isinstance(k, tuple):
                        vals = [o[k] for o in ps if k in o]
                    else:
                        vals = [o.get(k, 0) for o in ps]
                    m = vals[0]
                    for x in vals[1:]:
                        m &= x
                    if m or isinstance(k, tuple):
                        ins[k] = m
            if IN[b] != ins or OUT[b] is None:
                IN[b] = ins
                OUT[b] = transfer(b, ins)
                changed = True
    rep = []
    for b in order:
        if IN[b] is not None:
            transfer(b, IN[b], rep)
    restored = {
        int(m.group(1))
        for v in blocks.values()
        for ln in v["ins"]
        for m in [re.search(r"%(\d+):\w+ = SI_SPILL_S\d+_RESTORE", ln)]
        if m
    }
    return [x + (x[2] in restored,) for x in rep]


GREEDY_BANNER = (
    r"(?m)^# \*\*\* IR Dump After Greedy Register Allocator \(greedy\) \*\*\*:\n"
)


def split_dumps(txt):
    """The MIR function bodies of every greedy dump in `txt` (per gpu-module-to-binary target, per allocator instance: sgpr, wwm, vgpr; the first is the SGPR instance)."""
    out = []
    for part in re.split(GREEDY_BANNER, txt)[1:]:
        m = re.search(r"(?m)^}\s*$", part)
        out.append(part[: m.end()] if m else part)
    return out


def scan_text(dump, tmp):
    """(spill findings, other findings, raw findings) of one printed MIR function."""
    Path(tmp).write_text(dump)
    r = scan(str(tmp))
    return sum(1 for x in r if x[-1]), sum(1 for x in r if not x[-1]), r


if __name__ == "__main__":
    if sys.argv[1:2] != ["--mir"] or len(sys.argv) != 3:
        sys.exit(__doc__)
    r = scan(sys.argv[2])
    print(
        len(r),
        "findings;",
        sum(1 for x in r if x[-1]),
        "on a restored-from-slot register",
    )
