"""RT: the combination lookback over every Rust direct-compiler lever of one kernel family (PLAN decision 13).

    python rt_tune.py --family b3:r1:tuned --start <levers> --budget 96 --batch 8 --rounds 12 --out DIR

Driver of ``tune.strategies.SignedLookback`` (the mono tuner, ``kernels/gfx950/qwen38_mono/tune``) inside one GPU
job. The space is ``tune.catalog.direct_levers()``: every lever of ``ROCKE_DIRECT_LEVERS`` a bool, losers included
(nothing is pruned from solo results; the lookback's covering screen and interaction-aware GP pick the combinations).
Each ask batch is one run of the family's harness (``rt_b3.py`` / ``rt_fmoe.py`` / ``rt_head.py``: fresh-process
compiles, numerics of every arm against COMGR / the reference, then shuffled interleaved timing with the COMGR arm
``C`` and its controls ``C2`` / ``R`` in every batch); a config's value is its median per-round time ratio vs ``C``
(for the head: the mean over ``m`` of those medians). Arms failing numerics are dropped (``RT_DROP_FAILING``) and
told as ``numeric_fail``. ``--family`` accepts ``b3:<variant>:<which>``, ``fmoe:<rows>``, ``head:<epi>:<rows>``, ``q2:<variant>``,
``probe:<p4|p5>:<kind>`` (bench dir: ``RT_PROBE_BENCH``, else ``$Q38_STAGES/$RT_PROBES/P4|P5``);
``--scopes`` (fmoe only) adds ``@<phase_levers>`` to every arm.

After the budget: a confirmation run (``--confirm-rounds``) of the finalists (the best ``--finalists`` configs,
the start, and the start plus every solo winner) with the win rule against both controls. Writes
``DIR/tune.json`` (ledger: solo gains, measured combinations, fitted interactions, finalists' verdicts; ratios
only) and keeps every batch's harness output under ``DIR/batch_<n>/`` (absolute times there: private).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def harness_cmd(family: str, arms: str, rounds: int, out: str, extra: list[str]) -> list[str]:
    kind, *rest = family.split(":")
    py = sys.executable
    if kind == "b3":
        variant, which = rest
        return [py, os.path.join(HERE, "rt_b3.py"), "--variant", variant, "--which", which, "--arms", arms,
                "--rounds", str(rounds), "--out", out, *extra]  # fmt: skip
    if kind == "fmoe":
        (rows,) = rest
        return [py, os.path.join(HERE, "rt_fmoe.py"), "--rows", rows, "--arms", arms, "--rounds", str(rounds),
                "--out", out, *extra]  # fmt: skip
    if kind == "head":
        epi, rows = rest
        return [py, os.path.join(HERE, "rt_head.py"), "--epi", epi, "--rows", rows, "--arms", arms,
                "--rounds", str(rounds), "--out", out, *extra]  # fmt: skip
    if kind == "q2":
        (variant,) = rest
        return [py, os.path.join(HERE, "rt_q2.py"), "--variant", variant, "--arms", arms, "--rounds", str(rounds),
                "--out", out, *extra]  # fmt: skip
    if kind == "probe":
        probe, pkind = rest
        bench = os.environ.get("RT_PROBE_BENCH") or os.path.join(os.environ.get("Q38_STAGES", ""),
                                                                  os.environ.get("RT_PROBES", ""), probe.upper())  # fmt: skip
        return [py, os.path.join(HERE, "rt_probe_ab.py"), "--probe", probe, "--kind", pkind, "--arms", arms,
                "--rounds", str(rounds), "--bench-dir", bench, "--out", out, *extra]  # fmt: skip
    raise SystemExit(f"unknown family {family!r}")


def run_batch(family: str, arms: dict[str, str], rounds: int, out: str, timeout: int) -> tuple[dict, dict]:
    """Runs the harness on ``arms`` ({name: levers}); returns ({name: per-round ratios vs C}, {name: status})."""
    os.makedirs(out, exist_ok=True)
    spec = ",".join(f"{n}={lv}" for n, lv in arms.items())
    env = dict(os.environ, RT_DROP_FAILING="1")
    cmd = harness_cmd(family, spec, rounds, out, [])
    t0 = time.time()
    with open(os.path.join(out, "harness.log"), "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    print(f"BATCH {os.path.basename(out)} arms={len(arms)} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
    status = {n: "run_error" for n in arms}
    ratios: dict[str, list[float]] = {}
    try:
        res = json.load(open(os.path.join(out, "ratios.json")))
        raw = _raw(family, out)
    except (OSError, ValueError, KeyError):
        return ratios, status
    alias = res.get("alias", {})
    dropped = set(res.get("dropped", []))
    for n in arms:
        src = alias.get(n, n)
        if n in dropped or src in dropped:
            status[n] = "numeric_fail"
            continue
        if family.startswith("head"):
            ms = sorted({k.split("|m")[1] for k in raw if k.startswith("C|m")}, key=int)
            if all(f"{src}|m{m}" in raw for m in ms):
                per_m = [[x / c for x, c in zip(raw[f"{src}|m{m}"], raw[f"C|m{m}"])] for m in ms]
                ratios[n] = [statistics.mean(v) for v in zip(*per_m)]
                status[n] = "ok"
        elif src in raw:
            ratios[n] = [x / c for x, c in zip(raw[src], raw["C"])]
            status[n] = "ok"
    return ratios, status


def _raw(family: str, out: str) -> dict:
    """Per-round times per arm of a harness run (fmoe: the MoE part, the run's ``pre`` prefix subtracted)."""
    doc = json.load(open(os.path.join(out, "raw.json")))
    if family.startswith("fmoe"):
        t = doc["us_per_launch"]
        return {n: [x - p for x, p in zip(v, t["pre"])] for n, v in t.items() if n not in ("pre", "full", "full2")}
    return doc["raw_us"]


def controls_of(family: str, out: str) -> list[list[float]]:
    raw = _raw(family, out)
    if family.startswith("head"):
        ms = sorted({k.split("|m")[1] for k in raw if k.startswith("C|m")}, key=int)
        return [[statistics.mean(v) for v in zip(*[[x / c for x, c in zip(raw[f"{k}|m{m}"], raw[f"C|m{m}"])] for m in ms])]
                for k in ("C2", "R")]  # fmt: skip
    return [[x / c for x, c in zip(raw[k], raw["C"])] for k in ("C2", "R")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--start", default="", help="+-joined levers of the start (the family's current best set)")
    ap.add_argument("--pool", default="", help="+-joined levers of the space (default: every lever)")
    ap.add_argument("--scopes", default="", help="fmoe: phase_levers added to every arm (@ part)")
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--confirm-rounds", type=int, default=24)
    ap.add_argument("--finalists", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per harness batch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    from rocke.dispatch.tuning.search.trial import Observation, Status

    from kernels.gfx950.qwen38_mono.tune import catalog as C
    from kernels.gfx950.qwen38_mono.tune.strategies import SignedLookback

    pool = tuple(x for x in a.pool.split("+") if x) or C.DIRECT_LEVERS
    ad = C.direct_levers(pool)
    space = ad.space()
    start_cfg = {n: True for n in a.start.split("+") if n}
    unknown = set(start_cfg) - set(pool)
    if unknown:
        raise SystemExit(f"start levers outside the pool: {sorted(unknown)}")
    g0 = space.encode({**ad.base_values(), **start_cfg})
    strat = SignedLookback(space, a.seed, start=g0, budget=a.budget, lookback_batch=a.batch)
    suffix = f"@{a.scopes}" if a.scopes else ""

    def levers_of(genome) -> str:
        return ad.build(space.decode(genome)) + suffix

    seen: dict[str, Observation] = {}  # key -> observation (the first measurement of a config)
    measured = 0
    batches = 0
    log = []
    while not strat.done and measured < a.budget:
        trials = strat.ask(a.batch)
        if not trials:
            break
        fresh, obs = {}, []
        for t in trials:
            key = space.key(t.genome)
            if key is None:
                obs.append(Observation(trial=t, key=None, status=Status.INVALID, detail="refused by the space"))
            elif key in seen:
                obs.append(seen[key].replace_trial(t, cached=True))
            else:
                fresh.setdefault(key, []).append(t)
        if fresh:
            names = {f"A{i}": key for i, key in enumerate(fresh)}
            arms = {n: levers_of(fresh[k][0].genome) for n, k in names.items()}
            out = os.path.join(a.out, f"batch_{batches}")
            batches += 1
            ratios, status = run_batch(a.family, arms, a.rounds, out, a.timeout)
            for n, key in names.items():
                for t in fresh[key]:
                    st = status[n]
                    if st == "ok" and ratios.get(n):
                        ob = Observation(trial=t, key=key, status=Status.OK, value=statistics.median(ratios[n]),
                                         samples=tuple(ratios[n]), batch=batches)  # fmt: skip
                    else:
                        ob = Observation(trial=t, key=key, status=Status.NUMERIC_FAIL if st == "numeric_fail" else Status.RUN_ERROR,
                                         detail=st, batch=batches)  # fmt: skip
                    seen.setdefault(key, ob)
                    obs.append(ob)
                v = seen[key].value
                log.append({"batch": batches, "levers": arms[n], "status": seen[key].status,
                            "ratio": round(v, 5) if math.isfinite(v) else None})  # fmt: skip
                print("EVAL", arms[n] or "(none)", seen[key].status, f"{v:.4f}" if math.isfinite(v) else "-", flush=True)
            measured += len(fresh)
        strat.tell(obs)
    report = strat.report()
    best = sorted(((ob.value, k) for k, ob in seen.items() if ob.ok), key=lambda x: x[0])
    finals = {}
    for i, (_v, k) in enumerate(best[: a.finalists]):
        finals[f"F{i}"] = levers_of(seen[k].trial.genome)
    finals["S"] = levers_of(g0)
    solo = getattr(strat, "solo", {}) or {}
    winners = [lab.split("=")[0] for lab, gains in solo.items() if gains and statistics.median(gains) > 0 and lab.endswith("=True")]
    if winners:
        finals["SW"] = "+".join(sorted(set(start_cfg) | set(winners))) + suffix
    finals = {n: lv for n, lv in finals.items() if list(finals.values()).index(lv) == list(finals).index(n)}
    out = os.path.join(a.out, "confirm")
    ratios, status = run_batch(a.family, finals, a.confirm_rounds, out, a.timeout)
    verdicts = {}
    if ratios:
        from rt_b3 import win_rule

        ctl = controls_of(a.family, out)
        for n, rs in ratios.items():
            verdicts[n] = {"levers": finals[n], "median": round(statistics.median(rs), 4), "min": round(min(rs), 4),
                           "max": round(max(rs), 4), "verdict": win_rule(rs, ctl)}  # fmt: skip
            print("CONFIRM", n, json.dumps(verdicts[n]), flush=True)
    doc = {
        "family": a.family, "pool": list(pool), "start": a.start, "scopes": a.scopes, "budget": a.budget,
        "measured": measured, "batches": batches, "evaluations": log, "report": report,
        "interactions": strat.interactions() if hasattr(strat, "interactions") else {},
        "combos": strat.combos() if hasattr(strat, "combos") else {}, "confirm": verdicts,
        "node": os.uname().nodename if hasattr(os, "uname") else "",
    }  # fmt: skip
    json.dump(doc, open(os.path.join(a.out, "tune.json"), "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
