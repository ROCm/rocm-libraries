# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Vendored verbatim (CRLF->LF) from hackathon-prep/qwen38/truemono/WP0a-fix/selftest_isa_dist.py.
"""CPU self-test of isa_dist.py on synthetic pools (no cluster): the gate passes a same-key copy whose variants are
sampled from the same distribution, passes a key-only mutant, FAILs exactly the changed kinds of an ISA mutant,
FAILs a copy whose keys differ under KEYS=same, turns a rare unseen variant into PENDING (not FAIL) before
escalation, and its expectation checker reports violations.

    python selftest_isa_dist.py             (exit 0 = every check as expected)
    python selftest_isa_dist.py --mutants   (each mutant of isa_dist.py's rules must fail some check)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDS = {
    "delta_r1": ("delta", [("a" * 12, 1.0)]),
    "qsa_r9g3": ("qsa", [("b" * 12, 0.8), ("c" * 12, 0.2)]),
    "delta_ple_r2s": ("delta_ple", [("d" * 12, 0.9), ("e" * 12, 0.1)]),
}


def write_pool(d, rng):
    def pick(vs):
        x, acc = rng.random(), 0.0
        for v, p in vs:
            acc += p
            if x < acc:
                return v
        return vs[-1][0]

    for tree, key, mut, n in (("H", "k1", False, 24), ("T", "k1", False, 24), ("MI", "k2", True, 8),
                              ("MK", "k2", False, 8), ("TK", "k3", False, 8)):
        for rep in range(n):
            for ctx in ("solo-x", "fix-x"):
                bs = {}
                for b, (k, vs) in BUILDS.items():
                    v = pick(vs)
                    if mut and k != "delta_ple":
                        v = "f" * 12
                    bs[b] = {"ok": True, "kind": k, "keyed_sha": key, "objects": [{"text_sha": v * 5}]}
                with open(os.path.join(d, f"{tree}__base__{ctx}__{rep}.json"), "w") as f:
                    json.dump({"builds": bs, "env": {}}, f)
    # a TEST that only ever shows each build's majority variant (a subset of REF's): must PASS
    for rep in range(8):
        bs = {b: {"ok": True, "kind": k, "keyed_sha": "k1", "objects": [{"text_sha": vs[0][0] * 5}]}
              for b, (k, vs) in BUILDS.items()}
        with open(os.path.join(d, f"TS__base__solo-x__{rep}.json"), "w") as f:
            json.dump({"builds": bs, "env": {}}, f)
    # a rare variant only TEST shows (one compile), REF small: PENDING without --final, FAIL with it
    bs = {"qsa_r9g3": {"ok": True, "kind": "qsa", "keyed_sha": "k1", "objects": [{"text_sha": "9" * 60}]}}
    with open(os.path.join(d, "TR__base__solo-x__0.json"), "w") as f:
        json.dump({"builds": bs, "env": {}}, f)


SCRIPT = os.environ.get("ISA_DIST", os.path.join(HERE, "isa_dist.py"))
MUTANTS = [
    ("outside reversed", "outside = sorted(set(T) - set(R))", "outside = sorted(set(R) - set(T))"),
    ("--final ignored", "if nR >= a.escalate or a.final:", "if nR >= a.escalate:"),
    ("no key check (same)", 'if kmode == "same" and not (len(kr | kt) == 1):', "if False:"),
    ("no key check (differ)", 'if kmode == "differ" and kr & kt:', "if False:"),
    ("no minimum", "if nR < a.min_ref or nT < a.min:", "if False:"),
    ("fail-kinds expectation always ok", 'ok = all((r["verdict"] == "FAIL") == (r["kind"] in fk)', "ok = True or all((r"),
    ("strict ignored", "if strict and not (len(R) == 1", "if False and not (len(R) == 1"),
    ("pending counted as pass", '"PENDING" if verdicts["PENDING"] else "PASS"', '"PASS"'),
]


def mutants():
    src = open(os.path.join(HERE, "isa_dist.py")).read()
    survived = 0
    with tempfile.TemporaryDirectory() as d:
        for name, old, new in MUTANTS:
            assert src.count(old) == 1, name
            m = os.path.join(d, "isa_dist_mut.py")
            open(m, "w").write(src.replace(old, new))
            p = subprocess.run([sys.executable, os.path.abspath(__file__)], env={**os.environ, "ISA_DIST": m},
                               capture_output=True, text=True)
            killed = p.returncode != 0
            survived += not killed
            print(f"{'killed  ' if killed else 'SURVIVED'} {name}")
    print(f"MUTANTS {len(MUTANTS) - survived}/{len(MUTANTS)} killed")
    return 1 if survived else 0


def run(d, *args):
    p = subprocess.run([sys.executable, SCRIPT, d, "--out", os.path.join(d, "s.json"),
                        *args], capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    checks = []
    with tempfile.TemporaryDirectory() as d:
        write_pool(d, random.Random(1))
        rc, out = run(d, "--gate", "H:T:base:same", "--gate", "H:MK:base:differ", "--gate", "H:MI:base:any",
                      "--expect", "H:T:base:same=pass", "--expect", "H:MK:base:differ=pass",
                      "--expect", "H:MI:base:any=fail:delta+qsa", "--final", "--bootstrap", "base:H+T")
        checks.append(("copy PASS, key-only mutant PASS, ISA mutant FAILs exactly delta+qsa", rc == 0, out))
        rc, out = run(d, "--gate", "H:TS:base:same", "--final")
        checks.append(("TEST showing a subset of REF's variants PASSes (REF-only variants are info)", rc == 0, out))
        rc, out = run(d, "--gate", "H:T:base:same:strict", "--final")
        checks.append(("strict FAILs multi-variant builds", rc == 1 and "strict: not one identical binary" in out, out))
        rc, out = run(d, "--gate", "TS:TS:base:same:strict", "--final")
        checks.append(("strict PASSes one identical binary", rc == 0, out))
        rc, out = run(d, "--gate", "H:MI:base:any", "--expect", "H:MI:base:any=pass", "--final")
        checks.append(("expectation 'pass' on the ISA mutant is reported VIOLATED", rc == 1 and "VIOLATED" in out, out))
        rc, out = run(d, "--gate", "H:TK:base:same", "--final")
        checks.append(("KEYS=same FAILs a tree with another build key", rc == 1 and "build keys" in out, out))
        rc, out = run(d, "--gate", "H:T:base:differ", "--final")
        checks.append(("KEYS=differ FAILs an unchanged key", rc == 1 and "key unchanged" in out, out))
        rc, out = run(d, "--gate", "H:TR:base:any", "--min", "1", "--escalate", "1000")
        checks.append(("unseen variant, small REF pool: PENDING, not FAIL",
                       "base:any: PENDING" in out and "resample" in out, out))
        rc, out = run(d, "--gate", "H:TR:base:any", "--min", "1", "--final")
        checks.append(("unseen variant with --final: FAIL", rc == 1 and ": FAIL" in out, out))
        rc, out = run(d, "--gate", "H:T:base:same", "--min", "100", "--final")
        checks.append(("too few compiles FAIL", rc == 1 and "too few" in out, out))
    bad = 0
    for name, ok, out in checks:
        print(f"{'ok  ' if ok else 'BAD '} {name}")
        if not ok:
            bad += 1
            print(out)
    print(f"SELFTEST {'PASS' if not bad else 'FAIL'} ({len(checks) - bad}/{len(checks)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(mutants() if "--mutants" in sys.argv else main())
