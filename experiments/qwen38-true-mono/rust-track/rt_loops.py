"""Loops of a disassembled code object (``llvm-objdump -d`` text): each backward branch's body with its
instruction mix, so two compilations of one kernel can be compared loop by loop.

    python rt_loops.py A.s [B.s]
"""

import collections
import re
import sys

LINE = re.compile(r"^\s+([a-z0-9_]+)\b(.*?)//\s*([0-9A-Fa-f]+):")
CLASSES = [
    ("mfma", r"^v_mfma"), ("vload", r"^(buffer|global)_load|^global_atomic"), ("vstore", r"^(buffer|global)_store"),
    ("smem", r"^s_(load|buffer_load|memtime|memrealtime)"), ("lds", r"^ds_"), ("wait", r"^s_waitcnt"),
    ("nop", r"^s_nop"), ("branch", r"^s_(cbranch|branch)"), ("salu", r"^s_"), ("valu", r"^v_"),
]  # fmt: skip


def parse(path):
    insts = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(ln)
        if m:
            insts.append((int(m.group(3), 16), m.group(1), m.group(2)))
    return insts


def klass(mn):
    for name, pat in CLASSES:
        if re.match(pat, mn):
            return name
    return "other"


def loops(insts):
    """(start address, end address, Counter) per backward branch, by size."""
    out = []
    addrs = [a for a, _, _ in insts]
    for i, (a, mn, rest) in enumerate(insts):
        if not mn.startswith("s_cbranch") and mn != "s_branch":
            continue
        m = re.match(r"\s*(-?\d+)", rest)
        if not m:
            continue
        simm = int(m.group(1))
        simm = simm - 65536 if simm >= 32768 else simm
        size = 8 if "s_branch" in mn and False else 4
        target = a + 4 + 4 * simm
        if target <= a:
            body = [x for x in insts if target <= x[0] <= a]
            c = collections.Counter(klass(x[1]) for x in body)
            c["total"] = len(body)
            out.append((target, a, c))
    return out


def main():
    files = sys.argv[1:]
    for f in files:
        ls = loops(parse(f))
        print(f"== {f}: {len(ls)} loops")
        for s, e, c in sorted(ls, key=lambda x: -x[2]["total"])[:12]:
            print(f"  [{s:#x}, {e:#x}] " + " ".join(f"{k}={c[k]}" for k in ["total", "mfma", "vload", "lds", "valu", "salu", "wait", "nop", "branch"]))


if __name__ == "__main__":
    main()
