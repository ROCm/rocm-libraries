"""RT: the W2 probe kernels P4 / P5, direct code objects vs COMGR, paired in one process (gfx950, container/venv).

    python rt_probe_ab.py --probe p4 --kind a2a_ir --arms D,Dl=litreuse --rounds 24 --bench-dir $PB/P4 --out DIR

The kernel: ``rt_builds`` label ``<probe>_<kind>`` (p4: counter, a2a, a2a_ir, ticket, xl, xl_ir, boundary;
p5: kv, sc). Arms, all the SAME ``KernelDef``: ``C`` COMGR, ``C2`` C's bytes loaded as a second function, ``R`` a
fresh-process COMGR compile, ``D`` the direct compiler, ``NAME=LEVERS`` direct with ``ROCKE_DIRECT_LEVERS``.

Each arm runs through the probe bench's OWN cell (``p4_bench.py`` / ``p5_bench.py`` from ``--bench-dir``: CtrArm,
A2AArm, TicketArm, XlArm, BoundaryArm, KvArm, ScArm), on buffer sets shared by every arm (ROT sets on calibrated
2 MiB pages, rotated per round, so arms see the same addresses). Numerics are the bench's checks of every launch
(``m["ok"]``: mismatches, timeouts, ordering violations, payload sums, ticket ids, stale words): two numerics
launches per arm before timing, then every timed launch is re-checked. A failing arm stops the run, or with
``RT_DROP_FAILING=1`` leaves the timing (``rt_tune.py``).

Timing: ``--rounds`` rounds, arm order shuffled per round, the per-round delay table (``uni`` producer skew, the
bench's section A / C default) written before each round. The metric is the probe's own device-tick latency
(counter / a2a: ``lat_max``; ticket: ``cost``; xl / boundary: ``hop_max``; kv / sc: ``hop_max`` of the stamped
build), ratio vs ``C`` per round, win rule against both controls. The device ticks go to ``DIR/raw.json`` under
``raw_us`` (private; the name is shared with the other harnesses for ``rt_tune.py``); ratios in ``ratios.json``.
P5 ``host`` / ``hostw`` (host-in-the-loop rounds) have no paired timing here: ``rt_asm.sh`` gates them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

P4_KINDS = ("counter", "a2a", "a2a_ir", "ticket", "xl", "xl_ir", "boundary")
P5_KINDS = ("kv", "sc")
METRIC = {"counter": "lat_max", "a2a": "lat_max", "ticket": "cost", "xl": "hop_max", "boundary": "hop_max",
          "kv": "hop_max", "sc": "hop_max"}  # fmt: skip


def _ir(knobs):
    import kernels.gfx950.qwen38_mono.mailbox_ir  # noqa: F401  (registers "ir")
    import dataclasses

    return dataclasses.replace(knobs, impl="ir", fence_impl="ir")


def spec_of(probe: str, kind: str):
    """The spec ``rt_builds`` builds the label ``<probe>_<kind>`` from."""
    if probe == "p4":
        from rocke.helpers import mailbox as MB

        from kernels.gfx950.qwen38_mono.probes import p4_sync as S

        base, ir = kind.removesuffix("_ir"), kind.endswith("_ir")
        kw = {"knobs": _ir(MB.MailboxKnobs())} if ir else {}
        return base, {"counter": S.CounterSpec, "a2a": S.A2ASpec, "ticket": S.TicketSpec, "xl": S.XlSpec,
                      "boundary": S.BoundarySpec}[base](**kw)  # fmt: skip
    from kernels.gfx950.qwen38_mono.probes import p5_coherence as P

    return kind, {"kv": P.KvSpec, "sc": P.ScSpec}[kind]()


def p4_cells(B, cx, fns, base, spec, phases, rng):
    """{arm: callable(r) -> metrics} for a P4 family, every arm on shared buffers; ``prep(r)`` per round."""
    S = B.S
    pool = B.SyncPool(cx)
    dtab = None
    prep = None
    if base in ("counter", "a2a"):
        P = getattr(spec, "producers", 160)
        dtab = cx.dev.alloc(phases * B.G * 4)

        def prep(r):
            cx.write(dtab, B.delays("uni", phases, P, rng))

        if base == "counter":
            arms = {n: B.CtrArm(cx, f, n, 8, "local", P, phases, pool) for n, f in fns.items()}
        else:
            boxes = B.BoxSet(cx, P, spec.knobs.numa_copies == 2, phases)
            arms = {n: B.A2AArm(cx, f, spec, n, phases, boxes, "cal") for n, f in fns.items()}
        return {n: (lambda r, a=a: a.timed(dtab, r)[0]) for n, a in arms.items()}, prep, []
    if base == "ticket":
        arms = {n: B.TicketArm(cx, f, n, 1) for n, f in fns.items()}
        return {n: (lambda r, a=a: a.timed(None, r)[0]) for n, a in arms.items()}, None, []
    cell = {c[0]: c for c in B.XL_CELLS}[spec.rows]
    _R, C, ph = cell
    dtab = cx.dev.alloc(ph * B.G * 4)
    cx.write(dtab, B.delays("idle", ph, spec.producers, rng))
    if base == "xl":
        ctabs = [cx.put(B.ctr_table(8, B.G, "local", f)) for f in pool.flags]
        bufs = B.XlBufs(cx, spec, ph)
        arms = {n: B.XlArm(cx, f, spec, n, ph, C, ctabs, pool, bufs, "cal") for n, f in fns.items()}
        return {n: (lambda r, a=a: a.timed(dtab, r)[0]) for n, a in arms.items()}, None, []
    arms = {n: B.BoundaryArm(cx, f, spec.rows, n, ph, C, dtab) for n, f in fns.items()}
    return {n: (lambda r, a=a: a.timed(dtab, r)[0]) for n, a in arms.items()}, None, list(arms.values())


def p5_cells(B, cx, fns, base, spec):
    if base == "kv":
        pools = B.Pools(cx, True, B.KV_PH)
        arms = {n: B.KvArm(cx, f, spec, n, B.KV_PH, pools) for n, f in fns.items()}
        return {n: (lambda r, a=a: a.timed(r, 0)) for n, a in arms.items()}
    pools = B.Pools(cx, False, B.SC_PH)
    arms = {n: B.ScArm(cx, f, spec, n, B.SC_PH, pools) for n, f in fns.items()}
    return {n: (lambda r, a=a: a.timed(r)) for n, a in arms.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=("p4", "p5"), required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--arms", default="D")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--phases", type=int, default=100, help="P4 counter / a2a epochs per launch")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--bench-dir", default=os.environ.get("RT_PROBE_BENCH", ""),
                    help="directory of p4_bench.py / p5_bench.py (the rt-probes stage's P4 / P5)")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.kind not in (P4_KINDS if a.probe == "p4" else P5_KINDS):
        raise SystemExit(f"{a.probe} kind {a.kind!r}: one of {P4_KINDS if a.probe == 'p4' else P5_KINDS}")
    os.makedirs(a.out, exist_ok=True)

    import rt_builds
    from rocke.helpers.compile import compile_kernel
    from rt_b3 import arm_specs, compile_arms, dedupe, drop_failing, fresh_compile, win_rule

    label = f"{a.probe}_{a.kind}"
    base, spec = spec_of(a.probe, a.kind)
    t0 = time.time()
    # ---- every code object before HIP is touched
    kern = rt_builds.build(label)
    arts = {}
    c_art = compile_kernel(kern, arch="gfx950", backend="python", code_object_backend="comgr")
    kname = c_art.kernel_name
    arts["C"] = c_art.hsaco
    arts["C2"] = bytes(c_art.hsaco)
    arts["R"] = fresh_compile(label, "comgr", "", os.path.join(a.out, "R.hsaco"))
    d_art = compile_kernel(rt_builds.build(label), arch="gfx950", backend="python", code_object_backend="direct")
    if getattr(d_art, "code_object_backend", "") != "direct":
        print("DIRECT REFUSED", getattr(d_art, "direct_fallback", ""), flush=True)
        return 2
    specs = arm_specs(a.arms)
    built = compile_arms(label, [(n, lv) for n, lv in specs if n != "D"], a.out)
    for n, _ in specs:
        arts[n] = d_art.hsaco if n == "D" else built[n]
    for n, h in arts.items():
        with open(os.path.join(a.out, f"{label}.{n}.hsaco"), "wb") as fh:
            fh.write(h)
    alias = dedupe(arts)
    for n, t in alias.items():
        del arts[n]
        print("ALIAS", label, n, "=", t, flush=True)

    if not a.bench_dir:
        raise SystemExit("--bench-dir (or RT_PROBE_BENCH): the rt-probes stage's P4 / P5 directory")
    sys.path.insert(0, a.bench_dir)
    import importlib
    from types import SimpleNamespace

    B = importlib.import_module(f"{a.probe}_bench")
    cx = B.Cx()
    fns = {n: cx.dev.function(SimpleNamespace(kernel_name=kname, hsaco=bytes(h))) for n, h in arts.items()}
    rng = __import__("numpy").random.default_rng(a.seed)
    closers = []
    if a.probe == "p4":
        cells, prep, closers = p4_cells(B, cx, fns, base, spec, a.phases, rng)
    else:
        cells, prep = p5_cells(B, cx, fns, base, spec), None
    metric = METRIC[base]
    res = {"label": label, "kernel": kname, "node": os.uname().nodename, "metric": metric, "alias": alias}
    rc = 0
    try:
        # ---- numerics: two launches per arm through the bench's own checks
        checks, failed, ok = {}, set(), True
        for n, cell in cells.items():
            if prep:
                prep(0)
            ms = [cell(k) for k in range(2)]
            e = {"ok": [bool(m["ok"]) for m in ms],
                 "detail": {k: ms[0][k] for k in ("mismatch", "timeouts", "viol", "sink_bad", "bad", "status_ok")
                            if k in ms[0]}}  # fmt: skip
            e["pass"] = all(e["ok"])
            ok &= e["pass"]
            if not e["pass"]:
                failed.add(n)
            checks[n] = e
            print("CHECK", label, n, "PASS" if e["pass"] else "FAIL", json.dumps(e["detail"], default=str), flush=True)
        for n, t in alias.items():
            checks[n] = {"same_as": t, "pass": checks[t]["pass"]}
        res["checks"] = checks
        go, dropped = drop_failing(failed)
        for n in dropped:
            print("DROPPED", label, n, "(numerics failed)", flush=True)
            for t in [n] + [x for x, y in alias.items() if y == n]:
                alias.pop(t, None)
            cells.pop(n, None)
        res["dropped"] = dropped
        if a.no_time or not go:
            res["timing"] = "skipped" if a.no_time else "numerics failed"
            print("NUMERICS", "PASS" if ok else "FAIL", flush=True)
            json.dump(res, open(os.path.join(a.out, "ratios.json"), "w"), indent=1, default=str)
            return 0 if ok else 1
        # ---- timing: shuffled rounds, every launch re-checked
        raw = {n: [] for n in cells}
        bad_timed = {n: 0 for n in cells}
        order = list(cells)
        rr = random.Random(a.seed + 1)
        for r in range(a.rounds):
            if prep:
                prep(r)
            rr.shuffle(order)
            for n in order:
                m = cells[n](r)
                bad_timed[n] += 0 if m["ok"] else 1
                raw[n].append(float(m[metric]))
        ratios = {n: [x / c for x, c in zip(raw[n], raw["C"])] for n in raw if n != "C"}
        controls = [ratios["C2"], ratios["R"]]
        summary = {}
        for n, rs in ratios.items():
            v = win_rule(rs, controls) if n not in ("C2", "R") else "control"
            if bad_timed[n]:
                v = "NUMERICS-FAIL-IN-TIMING"
                rc = 1
            summary[n] = {"median": round(statistics.median(rs), 4), "min": round(min(rs), 4),
                          "max": round(max(rs), 4), "below1": f"{sum(x < 1 for x in rs)}/{len(rs)}", "verdict": v}  # fmt: skip
            print("TIME", label, n, json.dumps(summary[n]), flush=True)
        for n, t in alias.items():
            summary[n] = dict(summary[t], same_as=t)
            print("TIME", label, n, json.dumps(summary[n]), flush=True)
        res["timing"] = summary
        res["bad_timed"] = bad_timed
        res["rounds"] = a.rounds
        res["wall_s"] = round(time.time() - t0, 1)
        json.dump(res, open(os.path.join(a.out, "ratios.json"), "w"), indent=1, default=str)
        json.dump({"raw_us": raw, "unit": f"device ticks of {metric}"}, open(os.path.join(a.out, "raw.json"), "w"))
    finally:
        for c in closers:
            c.close()
        cx.dev.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
