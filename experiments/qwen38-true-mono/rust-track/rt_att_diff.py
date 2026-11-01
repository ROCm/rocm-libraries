"""RT: compare two decoded ATT dispatches (``code.json`` of WaveScope) per mnemonic: stall and latency shares and
per-hit cycles, so a direct object and COMGR's can be put side by side.

    python rt_att_diff.py A_code.json B_code.json [--top 30]

``code.json`` rows: [isa, ?, ?, source, codeobj, vaddr, hit, latency, stall, ...] (hit-weighted totals). Prints ratios
of B to A per mnemonic (stall per launch-normalized hit) and the top mnemonics by stall share in each.
"""

from __future__ import annotations

import argparse
import collections
import json


def load(path):
    rows = json.load(open(path))
    if isinstance(rows, dict):
        rows = rows.get("code") or rows.get("rows") or next(iter(rows.values()))
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # hits, latency, stall, static count
    for r in rows:
        if not isinstance(r, list) or len(r) < 9 or not isinstance(r[0], str):
            continue
        isa = r[0].strip()
        if not isa:
            continue
        m = isa.split()[0]
        if m == "s_waitcnt":
            m += " " + ("vm" if "vmcnt" in isa else "") + ("lgkm" if "lgkmcnt" in isa else "")
        a = agg[m]
        a[0] += int(r[6] or 0)
        a[1] += int(r[7] or 0)
        a[2] += int(r[8] or 0)
        a[3] += 1
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=30)
    x = ap.parse_args(argv)
    A, B = load(x.a), load(x.b)
    ta = sum(v[2] for v in A.values()) or 1
    tb = sum(v[2] for v in B.values()) or 1
    la = sum(v[1] for v in A.values()) or 1
    lb = sum(v[1] for v in B.values()) or 1
    print(f"total stall B/A {tb / ta:.3f}  total latency B/A {lb / la:.3f}")
    keys = sorted(set(A) | set(B), key=lambda k: -(B.get(k, [0, 0, 0])[2] + A.get(k, [0, 0, 0])[2]))
    print(f"{'mnemonic':32s} {'stall% A':>9s} {'stall% B':>9s} {'lat% A':>8s} {'lat% B':>8s} {'hits A':>8s} {'hits B':>8s} {'n A':>4s} {'n B':>4s}")
    for k in keys[: x.top]:
        a, b = A.get(k, [0, 0, 0, 0]), B.get(k, [0, 0, 0, 0])
        print(f"{k:32s} {100 * a[2] / ta:9.2f} {100 * b[2] / tb:9.2f} {100 * a[1] / la:8.2f} {100 * b[1] / lb:8.2f} {a[0]:8d} {b[0]:8d} {a[3]:4d} {b[3]:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
