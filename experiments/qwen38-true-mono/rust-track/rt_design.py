"""RT lever combination screening (decision 13: every lever, losers included, stays in the pool).

    python rt_design.py design [--runs 32] [--pool a,b,...]     the screening arms (rt_gpu.sh arms string)
    python rt_design.py fit DIR [DIR ...] [--pool ...]          surrogate over measured arms -> next arms

Design: a regular two-level fractional factorial of resolution IV over the lever pool (2^(12-7), 32 runs for
12 levers): every main effect is clear of every two-lever interaction, interactions are aliased only with
each other (in chains printed by ``design``). Its generators are odd-length products of the base columns
(any product of three columns is then odd-length, never the identity: resolution >= IV), picked among all
such sets (a fixed-seed sample of 4000 when there are more than 20000) for the fewest length-4 words. Run 0 is all levers off,
the plain direct object (arm D); arm ``Sxx`` is run xx.

Fit: per kernel, the measured median ratio of every arm (vs the pooled COMGR arms where the harness reports
it) is regressed on main effects plus every pairwise interaction with ridge shrinkage (more columns than
runs: aliased pairs share their weight; the chains say which). The model ranks configurations over the
full 2^12 space; ``fit`` prints the predicted best, the pairs carrying weight and the next arms: the
predicted top configurations plus the fold-over runs that separate the strongest alias chains. Only
measured data prunes: a lever leaves the pool only when its main effect and every interaction it appears in
are measured at or above neutral with the chains separated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys

POOL = ["storewait", "nopause", "nofence", "dppmirror", "dppcombine", "dppcombinef",
        "ldsquads", "ldspairs", "vmemfirst", "minmax", "xorcmp", "wgfence",
        "litreuse", "litshare", "bitop3", "selxor", "subadd", "loadorder", "scalesep", "dsrebase"]  # fmt: skip


def _cols(k_base: int):
    """Base columns as bit masks over the base factors; candidates: odd-weight products of >= 3 base factors."""
    base = [1 << i for i in range(k_base)]
    cand = [m for m in range(1, 1 << k_base) if bin(m).count("1") % 2 == 1 and bin(m).count("1") >= 3]
    return base, cand


def _words(cols: list[int], length: int) -> int:
    return sum(1 for c in itertools.combinations(cols, length) if _xor(c) == 0)


def _xor(c) -> int:
    x = 0
    for m in c:
        x ^= m
    return x


def build(n_factors: int, runs: int):
    k_base = runs.bit_length() - 1
    assert 1 << k_base == runs, "runs must be a power of two"
    base, cand = _cols(k_base)
    need = n_factors - k_base
    if need > len(cand):
        raise SystemExit(f"{n_factors} factors do not fit a resolution IV design in {runs} runs")
    best = None
    import math
    import random

    if math.comb(len(cand), need) <= 20000:
        pool = itertools.combinations(cand, need)
    else:  # too many generator sets: a fixed-seed sample of them
        rng = random.Random(13)
        pool = (tuple(sorted(rng.sample(cand, need))) for _ in range(4000))
    for gens in pool:
        cols = base + list(gens)
        w4 = _words(cols, 4)
        if best is None or w4 < best[0]:
            best = (w4, cols)
    w4, cols = best
    for length in (1, 2, 3):
        assert _words(cols, length) == 0, "resolution below IV"
    rows = []
    for r in range(runs):
        rows.append([bin(r & c).count("1") % 2 for c in cols])  # 1 = lever on; run 0 is all off
    return cols, rows, w4


def alias_chains(cols: list[int], names: list[str]):
    chains = {}
    for i, j in itertools.combinations(range(len(cols)), 2):
        chains.setdefault(cols[i] ^ cols[j], []).append(f"{names[i]}*{names[j]}")
    return sorted(chains.values(), key=len, reverse=True)


def arms_string(rows, names) -> str:
    out = []
    for k, r in enumerate(rows):
        on = [n for n, b in zip(names, r) if b]
        out.append("D" if not on else f"S{k:02d}=" + "+".join(on))
    return ",".join(out)


def cmd_design(a) -> int:
    names = a.pool.split(",")
    cols, rows, w4 = build(len(names), a.runs)
    print(f"# 2^({len(names)}-{len(names) - (a.runs.bit_length() - 1)}) resolution IV, {a.runs} runs, {w4} length-4 words")
    for ch in alias_chains(cols, names):
        if len(ch) > 1:
            print("# chain:", " = ".join(ch))
    for k, r in enumerate(rows):
        print(f"# S{k:02d}", "".join(str(b) for b in r))
    print(arms_string(rows, names))
    return 0


def _measured(dirs):
    """{kernel/view: {arm: median ratio}} from harness ratios.json files (fmoe: median; b3/head: vs_pooled)."""
    out = {}
    for d in dirs:
        j = json.load(open(os.path.join(d, "ratios.json")))
        lab = j.get("label", os.path.basename(d.rstrip("/")))
        t = j.get("timing")
        if not isinstance(t, dict):
            continue
        for key, s in t.items():
            arm, _, view = key.partition("|")
            out.setdefault(f"{lab}{'|' + view if view else ''}", {})[arm] = s.get("vs_pooled", s["median"])
    return out


def _levers_of(arm: str, arms: dict[str, str]) -> set[str]:
    return set(filter(None, arms.get(arm, "").split("+")))


def _ridge(X, y, lam):
    import numpy as np

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    A = X.T @ X + lam * np.eye(p)
    A[0, 0] -= lam  # the intercept is not shrunk
    return np.linalg.solve(A, X.T @ y)


def _features(on: set[str], names):
    x = [1.0] + [1.0 if n in on else -1.0 for n in names]
    s = x[1:]
    x += [s[i] * s[j] for i, j in itertools.combinations(range(len(names)), 2)]
    return x


def _optimize(b, names, starts, top):
    """The model's best configurations: the whole 2^n space when n <= 14, else a hill climb (flip the lever
    that lowers the prediction most, until none does) from every measured configuration and from all-off and
    all-on; returns ``top`` distinct (prediction, levers) pairs, lowest first."""
    import numpy as np

    pred = lambda on: float(np.dot(_features(on, names), b))
    if len(names) <= 14:
        space = []
        for bits in itertools.product((0, 1), repeat=len(names)):
            on = {n for n, bt in zip(names, bits) if bt}
            space.append((pred(on), tuple(sorted(on))))
        space.sort()
        return space[:top]
    found = {}
    for s0 in list(starts) + [set(), set(names)]:
        on = set(s0) & set(names)
        cur = pred(on)
        while True:
            best = None
            for n in names:
                t = on ^ {n}
                v = pred(t)
                if v < cur - 1e-9 and (best is None or v < best[0]):
                    best = (v, t)
            if best is None:
                break
            cur, on = best
        found[tuple(sorted(on))] = cur
    return sorted((v, k) for k, v in found.items())[:top]


def cmd_fit(a) -> int:
    import numpy as np

    names = a.pool.split(",")
    arm_levers = dict(s.split("=", 1) for s in a.arms.split(",") if "=" in s) if a.arms else {}
    data = _measured(a.dirs)
    feat_names = ["1"] + names + [f"{names[i]}*{names[j]}" for i, j in itertools.combinations(range(len(names)), 2)]
    report = {}
    for kern, arms in sorted(data.items()):
        pts = [(arm, r) for arm, r in arms.items() if arm == "D" or arm in arm_levers]
        if len(pts) < 8:
            continue
        X = [_features(_levers_of(arm, arm_levers), names) for arm, _ in pts]
        y = [r for _, r in pts]
        b = _ridge(X, y, a.lam)
        # leave-one-out residual spread: the model's own noise estimate
        loo = []
        for k in range(len(pts)):
            bk = _ridge([x for i, x in enumerate(X) if i != k], [v for i, v in enumerate(y) if i != k], a.lam)
            loo.append(float(np.dot(X[k], bk) - y[k]))
        space = _optimize(b, names, [_levers_of(arm, arm_levers) for arm, _ in pts], a.top)
        mains = sorted(((float(b[1 + i]) * 2, n) for i, n in enumerate(names)))
        pairs = sorted(((float(v) * 2, feat_names[1 + len(names) + k]) for k, v in enumerate(b[1 + len(names):])),
                       key=lambda t: -abs(t[0]))[: a.top]  # fmt: skip
        best_meas = min(pts, key=lambda t: t[1])
        report[kern] = {"n": len(pts), "loo_rms": round(statistics.pstdev(loo) if len(loo) > 1 else 0.0, 4),
                        "measured_best": best_meas, "D": arms.get("D"),
                        "main_effects(on-off)": [(round(v, 4), n) for v, n in mains],
                        "top_pairs(on-off)": [(round(v, 4), n) for v, n in pairs],
                        "predicted_top": [(round(v, 4), "+".join(c)) for v, c in space[: a.top]]}  # fmt: skip
        print(f"== {kern}  n={len(pts)}  loo_rms={report[kern]['loo_rms']}  D={arms.get('D')}  best measured {best_meas}")
        print("   main (on-off):", ", ".join(f"{n} {v:+.4f}" for v, n in mains))
        print("   pairs:", ", ".join(f"{n} {v:+.4f}" for v, n in pairs))
        for v, c in space[: a.top]:
            print(f"   predicted {v:.4f}  {'+'.join(c) or 'D'}")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("design")
    d.add_argument("--runs", type=int, default=32)
    d.add_argument("--pool", default=",".join(POOL))
    f = sp.add_parser("fit")
    f.add_argument("dirs", nargs="+")
    f.add_argument("--pool", default=",".join(POOL))
    f.add_argument("--arms", default="", help="the arms string the dirs were measured with (NAME=LEVERS,...)")
    f.add_argument("--lam", type=float, default=0.5)
    f.add_argument("--top", type=int, default=8)
    f.add_argument("--json", default="")
    a = ap.parse_args(argv)
    return cmd_design(a) if a.cmd == "design" else cmd_fit(a)


if __name__ == "__main__":
    sys.exit(main())
