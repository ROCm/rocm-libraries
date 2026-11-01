# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Vendored verbatim (CRLF->LF) from hackathon-prep/qwen38/truemono/WP0a-fix/isa_dist.py (WP0a fix round, tested by
# its selftest and 8 rule mutants there; selftest copy: isa_dist_selftest.py). The pooled ISA distribution gate.
"""WP0a fix round: pooled ISA variant distributions of the Qwen3.8 mono layer builds and the distribution-level ISA
gate that replaces WP0a's variant-SET equality (unsound at 3-8 compiles per tree, WP0a verification).

Input: keys_isa.py JSON files named ``<tree>__<arm>__<ctx>__<rep>.json`` (isa_pool.sh), ``ctx`` = ``solo-<build>``
(one build per fresh process), ``fix-<chunk>`` (WP0a's gate chunks, gate order), ``shuf-<n>`` (a random 6-build
chunk in random order), ``srv`` (the served builds in one process). ``arm`` = ``base`` (plain env, as served: random
PYTHONHASHSEED, ASLR on), ``all`` (the hybrid ALL env), ``hs0`` / ``norand`` / ``both`` (PYTHONHASHSEED=0 /
``setarch -R`` / both), ``dumpir`` (FLYDSL_DUMP_IR=1).

A variant = the build's code objects' ``.text`` sha256 (12 hex) joined. Per build and tree: the pooled counts of one
arm over every context, and the counts per context.

Gate ``REF:TEST:ARM:KEYS[:strict]`` (e.g. ``H:T:base:same``): per build
  - at least ``--min`` compiles of the build in each tree (pooled over contexts; both trees must be sampled in the
    same contexts, which isa_pool.sh does);
  - KEYS ``same``: exactly one build key over both trees (a host-only edit; THE decisive gate there), ``differ``:
    TEST's key differs from REF's (a kernel-side knob: the sources digest changes by design), ``any``: not checked;
  - every TEST variant is among REF's pooled variants. A TEST variant REF never showed is PENDING while REF's pool is
    smaller than ``--escalate`` (isa_pool.sh then compiles REF more and runs this again with ``--final``) and FAIL
    after that. A genuine code change (new instructions) never appears in REF however many compiles: it FAILs.
  - REF variants TEST never showed are info only, with the chance of that by sampling, (1 - p_REF)^n_TEST.
  - ``strict``: additionally one single binary in REF and the same one in TEST (for a deterministic compile
    protocol: PYTHONHASHSEED=0 + ``setarch -R``, one build per process; arm ``both``).
``--expect GATE=pass`` (every build PASS) or ``GATE=fail:<kinds>`` (exactly the builds of these '+'-joined kinds FAIL,
the rest PASS: a mutant's teeth). Exit 0 when every expectation holds (or, with no --expect, every gate PASSes).

Also reported (``--bootstrap``): from the pooled H+T compiles of each build (same keys = one distribution), the
chance that set(H) != set(T) with k compiles per tree (the old gate's false-FAIL rate) and the chance the within rule
false-FAILs with m REF and k TEST compiles, per build and for "any of the builds".
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys


def variant(rec):
    return "+".join(sorted(o["text_sha"][:12] for o in rec["objects"]))


def load(dirs):
    rows = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))))  # build -> tree -> arm -> ctxtype -> variant -> n
    seq = collections.defaultdict(list)  # (build, tree, arm) -> [(ctxtype, variant)]
    keys = collections.defaultdict(lambda: collections.defaultdict(set))  # (build, arm) -> tree -> keys
    kinds, failed, files, envs = {}, [], 0, collections.defaultdict(set)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            parts = os.path.basename(f)[:-5].split("__")
            if len(parts) != 4:
                continue
            tree, arm, ctx, rep = parts
            ctxtype = ctx.split("-")[0]
            try:
                data = json.load(open(f))
            except (OSError, ValueError) as e:
                failed.append(f"{os.path.basename(f)}: {e}")
                continue
            files += 1
            envs[(tree, arm)].add((data.get("env", {}).get("PYTHONHASHSEED", "unset"), data.get("personality")))
            for b, r in data["builds"].items():
                if not r.get("ok"):
                    failed.append(f"{os.path.basename(f)} {b}: {r.get('error', '')[:200]}")
                    continue
                v = variant(r)
                rows[b][tree][arm][ctxtype][v] += 1
                seq[(b, tree, arm)].append((ctxtype, v))
                keys[(b, arm)][tree].add(r["keyed_sha"])
                kinds[b] = r["kind"]
    return rows, seq, keys, kinds, failed, files, envs


def pooled(rows, b, tree, arm):
    c = collections.Counter()
    for cnt in rows[b][tree][arm].values():
        c.update(cnt)
    return c


def gate(spec, rows, keys, kinds, a):
    ref, test, arm, kmode, *mode = spec.split(":")
    strict = mode == ["strict"]
    out, problems, pending = {}, [], []
    builds = sorted(b for b in rows if rows[b][test][arm])
    if not builds:
        problems.append(f"{spec}: no {test} compiles in arm {arm}")
    for b in builds:
        R, T = pooled(rows, b, ref, arm), pooled(rows, b, test, arm)
        nR, nT = sum(R.values()), sum(T.values())
        r = {"kind": kinds[b], "n_ref": nR, "n_test": nT, "ref": dict(R), "test": dict(T), "verdict": "PASS",
             "why": []}
        if nR < a.min_ref or nT < a.min:
            r["verdict"] = "FAIL"
            r["why"].append(f"too few compiles ({ref} {nR} < {a.min_ref} or {test} {nT} < {a.min})")
        kr, kt = keys[(b, arm)][ref], keys[(b, arm)][test]
        if kmode == "same" and not (len(kr | kt) == 1):
            r["verdict"] = "FAIL"
            r["why"].append(f"build keys: {ref} {len(kr)}, {test} {len(kt)}, union {len(kr | kt)} (want 1)")
        if kmode == "differ" and kr & kt:
            r["verdict"] = "FAIL"
            r["why"].append("build key unchanged (a kernel-side knob changes the sources digest)")
        outside = sorted(set(T) - set(R))
        if outside:
            if nR >= a.escalate or a.final:
                r["verdict"] = "FAIL"
                r["why"].append(f"{test} variants never seen in {nR} {ref} compiles: {outside}")
            elif r["verdict"] == "PASS":
                r["verdict"] = "PENDING"
                r["why"].append(f"{test} variants not (yet) in {nR} {ref} compiles: {outside}; resample {ref}")
                pending.append((ref, arm, b))
        r["ref_only"] = {v: round((1 - R[v] / nR) ** nT, 4) for v in set(R) - set(T)} if nR else {}
        if strict and not (len(R) == 1 and set(T) == set(R)):
            r["verdict"] = "FAIL"
            r["why"].append(f"strict: not one identical binary ({ref} {len(R)} variants, {test} {len(T)}, equal "
                            f"{set(T) == set(R)})")
        r["set_equal"] = set(R) == set(T)
        out[b] = r
    verdicts = collections.Counter(r["verdict"] for r in out.values())
    overall = "FAIL" if problems or verdicts["FAIL"] else ("PENDING" if verdicts["PENDING"] else "PASS")
    return {"spec": spec, "verdict": overall, "counts": dict(verdicts), "problems": problems,
            "fail": sorted(b for b, r in out.items() if r["verdict"] == "FAIL"),
            "pending": sorted(b for b, r in out.items() if r["verdict"] == "PENDING"),
            "set_equal_false": sorted(b for b, r in out.items() if not r["set_equal"]), "builds": out}, pending


def bootstrap(seq, rows, arm, trees, ks, ms, trials, rng):
    """Per build: the pooled compiles of ``trees`` in ``arm`` as one population (same build key)."""
    res = {}
    for b in sorted(rows):
        pop = [v for t in trees for _, v in seq.get((b, t, arm), [])]
        if len(pop) < 2 * max(ks) or len(set(pop)) < 2:
            continue
        r = {"n": len(pop), "variants": dict(collections.Counter(pop))}
        for k in ks:
            bad = 0
            for _ in range(trials):
                s = rng.sample(pop, 2 * k)
                bad += set(s[:k]) != set(s[k:])
            r[f"set_neq_k{k}"] = bad / trials
        for m in ms:
            for k in ks:
                if m + k > len(pop):
                    continue
                bad = 0
                for _ in range(trials):
                    s = rng.sample(pop, m + k)
                    bad += bool(set(s[m:]) - set(s[:m]))
                r[f"within_m{m}_k{k}"] = bad / trials
        res[b] = r
    agg = {}
    for key in sorted({k for r in res.values() for k in r if k.startswith(("set_neq", "within"))}):
        p_none = 1.0
        for r in res.values():
            p_none *= 1 - r.get(key, 0.0)
        agg[key] = round(1 - p_none, 4)  # any build false-fails (the stable builds never do)
    return {"per_build": res, "any_build": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--gate", action="append", default=[], help="REF:TEST:ARM:same|differ|any")
    ap.add_argument("--expect", action="append", default=[], help="GATE=pass | GATE=fail:<kind>+<kind>")
    ap.add_argument("--min", type=int, default=8, help="TEST compiles per build")
    ap.add_argument("--min-ref", type=int, default=8)
    ap.add_argument("--escalate", type=int, default=64, help="REF pool size past which a missing variant FAILs")
    ap.add_argument("--final", action="store_true", help="no PENDING: a TEST variant outside REF FAILs")
    ap.add_argument("--bootstrap", help="ARM:TREE+TREE (e.g. base:H+T)")
    ap.add_argument("--resample-out", help="write the PENDING (ref, arm, build) lines here")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, seq, keys, kinds, failed, files, envs = load(a.dirs)
    summary = {"files": files, "failed": failed[:50], "n_failed": len(failed),
               "envs": {f"{t}/{arm}": sorted(map(str, v)) for (t, arm), v in envs.items()}, "rows": {}}
    for b in sorted(rows):
        summary["rows"][b] = {
            t: {arm: {"pooled": dict(pooled(rows, b, t, arm)), "by_ctx": {c: dict(n) for c, n in ctxs.items()}}
                for arm, ctxs in rows[b][t].items()}
            for t in rows[b]}
        # keys_cmp.py --variants reads rows[b]["H"]["pooled"] (the base arm)
        if "H" in rows[b] and "base" in rows[b]["H"]:
            summary["rows"][b]["H"]["pooled"] = dict(pooled(rows, b, "H", "base"))
    multi = {}
    for b in sorted(rows):
        for t in rows[b]:
            for arm in rows[b][t]:
                p = pooled(rows, b, t, arm)
                if len(p) > 1:
                    multi.setdefault(arm, {}).setdefault(b, {})[t] = dict(p)
    summary["multi_variant"] = multi
    rc, pend_all = 0, []
    summary["gates"] = {}
    for g in a.gate:
        res, pend = gate(g, rows, keys, kinds, a)
        summary["gates"][g] = res
        pend_all += pend
        print(f"ISA_GATE {g}: {res['verdict']} {res['counts']} fail={res['fail']} pending={res['pending']} "
              f"(set-equality would flag {len(res['set_equal_false'])}: {res['set_equal_false']})")
        for b in res["fail"] + res["pending"]:
            print(f"   {b}: {res['builds'][b]['why']}")
    exp_ok = True
    for e in a.expect:
        g, _, want = e.rpartition("=")
        res = summary["gates"].get(g)
        if res is None:
            print(f"EXPECT {e}: no such gate"); exp_ok = False; continue
        if want == "pass":
            ok = res["verdict"] == "PASS"
        else:
            fk = set(want.split(":", 1)[1].split("+"))
            ok = all((r["verdict"] == "FAIL") == (r["kind"] in fk) for r in res["builds"].values()) and not res["pending"]
        print(f"EXPECT {e}: {'OK' if ok else 'VIOLATED'}")
        exp_ok &= ok
    if a.expect:
        rc = 0 if exp_ok else 1
    elif any(r["verdict"] != "PASS" for r in summary["gates"].values()):
        rc = 1
    if a.resample_out:
        with open(a.resample_out, "w") as f:
            for ref, arm, b in sorted(set(pend_all)):
                f.write(f"{ref} {arm} {b}\n")
    if a.bootstrap:
        arm, trees = a.bootstrap.split(":")
        summary["bootstrap"] = bootstrap(seq, rows, arm, trees.split("+"), (3, 4, 8), (16, 24, 40), 4000, random.Random(7))
        print("BOOTSTRAP any-build false-FAIL:", summary["bootstrap"]["any_build"])
    for arm, bs in multi.items():
        print(f"MULTI {arm}: {len(bs)} builds with >1 variant: " +
              "; ".join(f"{b} " + " ".join(f"{t}{dict(sorted(c.items(), key=lambda x: -x[1]))}" for t, c in v.items())
                        for b, v in bs.items()))
    print(f"files {files}, failed compiles {len(failed)}")
    json.dump(summary, open(a.out, "w"), indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
