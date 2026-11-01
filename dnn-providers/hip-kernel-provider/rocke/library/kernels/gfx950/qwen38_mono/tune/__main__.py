# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CLI of the monokernel tuner (``library/`` and ``platform/python`` on
``sys.path``)::

    python -m kernels.gfx950.qwen38_mono.tune catalog [--json]
    python -m kernels.gfx950.qwen38_mono.tune replay-grid  --files climb_*.json --out R.json
    python -m kernels.gfx950.qwen38_mono.tune replay-sched --files ab_*_abs.json --variants r1,r4s --out R.json
    python -m kernels.gfx950.qwen38_mono.tune live-toy --depth 8 --work 8 --budget 48 --out R.json --private-out A.json
    python -m kernels.gfx950.qwen38_mono.tune runner --batch TOKEN --out results.json      # job side
    python -m kernels.gfx950.qwen38_mono.tune export-defaults --db tune.db.json

Outputs named ``--out`` carry ratios only; ``--private-out`` carries the
absolute samples (keep it in the private store).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


def _w(path: str | None, obj) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps(obj, indent=1, sort_keys=True, default=str).encode())


def cmd_catalog(a) -> int:
    from . import catalog as C

    rows = [ad.describe() for ad in C.all_adapters_cpu()]
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    for d in rows:
        print(f"{d['name']:<10} {d['variant']:<26} knobs={len(d['knobs']):3d} bound={d['size_bound']:.3g} "
              f"excluded={sorted(d['excluded'])}")  # fmt: skip
    return 0


def _strategies(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def cmd_replay_grid(a) -> int:
    from . import catalog as C
    from . import experiments as EX
    from . import replay as RP

    grids = []
    for f in a.files:
        g = RP.load_lookback_grid(f)
        numa = any("numa" in c for c in g.configs.values())
        if not numa:
            g = RP.load_lookback_grid(f, start={})
        grids.append((g, numa, Path(f).stem))
    rows, transfer = [], []
    for g, numa, stem in grids:
        ad = C.b3(g.variant or "r1", numa=numa)
        defaults = {**ad.base_values(), **({"numa": 1} if numa else {})}
        order = {k.path: list(k.values) for k in ad.knobs}
        p = EX.grid_problem(g, defaults, order=order)
        p.name = f"b3:{stem}"
        p.meta["solo"] = EX.solo_analysis(p)
        print("SOLO", p.name, json.dumps({k: v for k, v in p.meta["solo"].items() if not k.startswith("_")}), flush=True)
        print(f"PROBLEM {p.name} start={p.meta['start']} cells={p.meta['cells']} noise={p.meta['noise']:.4f} "
              f"start/oracle={p.start_value() / p.oracle:.4f}", flush=True)  # fmt: skip
        for s in _strategies(a.strategies):
            for b in [int(x) for x in a.budgets.split(",")]:
                for seed in range(a.seeds):
                    rows.append(EX.run_strategy(p, s, b, seed, batch=a.batch))
        if a.transfer and len(grids) > 1:
            # warm start from the nearest OTHER variant's recorded climb best
            # (same base: NUMA grids transfer among NUMA grids)
            f0 = C.variant_features(g.variant)
            others = [o for o, n2, _s in grids if o.variant != g.variant and n2 == numa]
            if not others:
                continue
            near = min(others, key=lambda o: sum((math.log2(f0.get(k, 1)) - math.log2(C.variant_features(o.variant).get(k, 1))) ** 2
                                                for k in f0))  # fmt: skip
            for s in _strategies(a.transfer_strategies):
                for b in [int(x) for x in a.transfer_budgets.split(",")]:
                    for seed in range(a.seeds):
                        r = EX.run_strategy(p, s, b, seed, batch=a.batch, warm=[{**defaults, **near.climb_best}])
                        r["strategy"] = f"{s}+transfer:{near.variant}"
                        transfer.append(r)
    summ = EX.summarize(rows + transfer)
    for r in summ:
        print("SUMMARY", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    _w(a.out, {"runs": rows + transfer, "summary": summ})
    return 0


def cmd_replay_sched(a) -> int:
    from . import catalog as C
    from . import experiments as EX
    from . import gate as G
    from . import replay as RP

    out = {"runs": [], "decisions": []}
    for v in a.variants.split(","):
        table = RP.load_sched_rounds(a.files, v)
        if not table.rounds:
            print("SKIP", v, "no rounds", flush=True)
            continue
        ad = C.sched(a.kind, v)
        defaults = ad.base_values()
        p = EX.sched_problem(table, defaults, variant=v)
        p.name = f"sched:{a.kind}.{v}"
        p.meta["solo"] = EX.solo_analysis(p)
        print("SOLO", p.name, json.dumps({k: v for k, v in p.meta["solo"].items() if not k.startswith("_")}), flush=True)
        print(f"PROBLEM {p.name} cells={p.meta['cells']} controls={p.meta['controls']} "
              f"oracle={p.oracle:.4f}", flush=True)  # fmt: skip
        for s in _strategies(a.strategies):
            for b in [int(x) for x in a.budgets.split(",")]:
                for seed in range(a.seeds):
                    r = EX.run_strategy(p, s, b, seed, batch=a.batch)
                    out["runs"].append(r)
                    if seed == 0 and s == a.decide_strategy and b == int(a.budgets.split(",")[-1]):
                        # the DESIGN 9.1 decision on the recorded rounds of the pick
                        pick_cfg = {k: _lit(x) for k, x in (r["pick"] or {}).items()}
                        pick_cfg = {k: x for k, x in pick_cfg.items() if defaults.get(k) != x}
                        tk = json.dumps(dict(sorted(pick_cfg.items())), sort_keys=True)
                        ctl = [table.controls[c] for c in ("mono2", "mono_rc") if c in table.controls]
                        if tk in table.rounds and ctl:
                            ver = G.win_rule(table.rounds[tk], ctl).to_json()
                            out["decisions"].append({"variant": v, "pick": pick_cfg, "ratio_vs_mono_median": ver["median"],
                                                     "verdict": ver, "oracle": p.oracle})  # fmt: skip
                            print("DECISION", v, json.dumps(pick_cfg), json.dumps(ver), flush=True)
    summ = EX.summarize(out["runs"])
    for r in summ:
        print("SUMMARY", json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
    out["summary"] = summ
    _w(a.out, out)
    return 0


def _lit(s):
    if isinstance(s, str):
        t = s.strip("'\"")
        if t in ("True", "False"):
            return t == "True"
        try:
            return int(t)
        except ValueError:
            return t
    return s


def _lookback_options(a, ad) -> dict:
    """Budget hint, and the ledger's solo gains and pair terms as priors."""
    opts = {"budget": a.budget}
    if a.gains:
        from .tdb import GainLedger

        led = GainLedger(a.gains)
        opts["prior_gains"] = led.prior(ad.stem)
        opts["prior_pairs"] = led.prior_pairs(ad.stem)
    return opts


def cmd_live_toy(a) -> int:
    from rocke.dispatch.tuning.search.driver import run_search
    from rocke.dispatch.tuning.search.measure import Evaluator
    from rocke.dispatch.tuning.search.portfolio import WarmStart
    from rocke.dispatch.tuning.search.trial import Budget, Fidelity

    from .. import toy_chain as T
    from . import catalog as C
    from . import tdb as D
    from .live_toy import ToyChainBackend
    from .strategies import SignedLookback, mono_portfolio

    ad = C.toy_chain(a.depth, a.work, layers=a.layers)
    space = ad.space()
    ladder = (Fidelity("rounds", reps=a.rounds, warmup=0),)
    if a.via_runner:
        # every round's fresh candidates go to the batch runner in a separate
        # process (compile, gate with its mutant, interleaved timing), as a
        # cluster batch job would run them
        from .evaluate import JobEvaluator, LocalTransport

        target = {"kind": "toy_chain", "depth": a.depth, "work": a.work, "layers": a.layers,
                  "launches": a.launches, "replays": a.replays, "seed": a.seed}  # fmt: skip
        ev = JobEvaluator(space, target, LocalTransport(timeout=3600), ladder=ladder)
        be = None
    else:
        be = ToyChainBackend(ad, launches=a.launches, replays=a.replays, seed=a.seed)
        st = be.selftest()  # the gate's must-fail mutants BEFORE any timing
        print("GATE", json.dumps(st), flush=True)
        if not st["ok"]:
            be.close()
            return 3
        ev = Evaluator(space, be, ladder=ladder, compile_timeout=900, measure_timeout=600)
    recorded = ad.config_of(T.best_spec(a.depth, a.work, layers=a.layers))
    warm = [WarmStart({}, "shipped")]
    if a.warm == "neighbor":
        nd = a.neighbor_depth or (4 if a.depth != 4 else 8)
        nb = ad.config_of(T.best_spec(nd, a.work, layers=a.layers))
        warm.append(WarmStart(nb, f"neighbor:d{nd}w{a.work}l{a.layers}"))
    if a.strategy == "phased":
        from .strategies import phased

        start = {**ad.base_values(), **dict(warm[-1].config)}
        g0 = space.repair(space.encode(start), __import__("random").Random(a.seed))
        strat = phased(space, a.seed, start=g0, structural=[k.path for k in ad.knobs if k.structural], warm=warm,
                       lookback_options=_lookback_options(a, ad))
    elif a.strategy == "mono":
        strat = mono_portfolio(space, a.seed, warm=warm)
    elif a.strategy == "lookback":
        start = {**ad.base_values(), **dict(warm[-1].config)}  # the transfer start when given, else shipped
        g0 = space.repair(space.encode(start), __import__("random").Random(a.seed))
        strat = SignedLookback(space, a.seed, start=g0, structural=[k.path for k in ad.knobs if k.structural],
                               **_lookback_options(a, ad))
    else:
        from rocke.dispatch.tuning.search.api import make_strategy

        strat = make_strategy(a.strategy, space, a.seed, warm=warm)
    t0 = time.time()
    res = run_search(space, strat, ev, Budget(evaluations=a.budget), batch=a.batch, confirm=0, stall_rounds=24)
    search_s = time.time() - t0
    # ratios vs the baseline arm of the same batch (per round, median)
    ratio_of, curve = {}, []
    if be is None:
        for k, r in ev.prefetched.results.items():
            if r.get("ratio_vs_base") is not None:
                ratio_of[k] = r["ratio_vs_base"]
        be = ToyChainBackend(ad, launches=a.launches, replays=a.replays, seed=a.seed)
    for rec in be.records:
        if rec.get("confirm"):
            continue
        rows = rec["rounds"]
        for name in rows[0]:
            if name == "base":
                continue
            key = name.split(":", 1)[1]
            r = statistics.median(row[name] / row["base"] for row in rows)
            ratio_of.setdefault(key, r)
    best_so_far = math.inf
    for ob in res.observations:
        if ob.ok and not ob.cached and ob.key in ratio_of:
            best_so_far = min(best_so_far, ratio_of[ob.key])
            curve.append({"n": len(curve) + 1, "ratio": ratio_of[ob.key], "incumbent": best_so_far,
                          "origin": ob.trial.origin})  # fmt: skip
    fails = {}
    for ob in res.observations:
        if not ob.ok:
            fails[ob.status] = fails.get(ob.status, 0) + 1
    # finalists: the top distinct measured configs + the recorded optimum
    top = sorted(((v, k) for k, v in ratio_of.items()), key=lambda t: t[0])[: a.finalists]
    finalists = {}
    for i, (_v, k) in enumerate(top):
        g = next((ob.trial.genome for ob in res.observations if ob.key == k), None)
        finalists[f"search{i}"] = ad.canonical_knobs(space.decode(g))
    finalists["recorded_best"] = ad.canonical_knobs({**ad.base_values(), **recorded})
    # the solo-winners-only combination: the climb's start plus every level
    # that won ALONE (its best level per lever); the combination lookback's
    # pick is judged against it in the same rounds (PLAN decision 13)
    lb = strat if isinstance(strat, SignedLookback) else next(
        (m for m in getattr(strat, "members", ()) if isinstance(m, SignedLookback)), None)  # fmt: skip
    solo_info = None
    if lb is not None and lb.start_g is not None:
        g = list(lb.start_g)
        winners = []
        for i in range(space.dims):
            best = None
            for lv in range(space.params[i].size):
                gs = lb.solo.get(lb._label(i, lv))
                if gs and sum(gs) / len(gs) > 0 and (best is None or sum(gs) / len(gs) > best[0]):
                    best = (sum(gs) / len(gs), lv)
            if best is not None:
                g[i] = best[1]
                winners.append(lb._label(i, best[1]))
        r = space.repair(g, __import__("random").Random(0))
        if r is not None:
            sw = ad.canonical_knobs(space.decode(r))
            if all(sw != v for v in finalists.values()):
                finalists["solo_winners_only"] = sw
        solo_info = {"solo_winners": winners, "combos": lb.combos()}
    conf = be.confirm(finalists, rounds=a.confirm_rounds, seed=a.seed + 101)
    med = {n: statistics.median(v) for n, v in conf["ratios"].items()}
    # search pick vs the recorded optimum, per round (same rounds)
    s0 = min((n for n in finalists if n.startswith("search")), key=lambda n: med[n])
    head = [x / y for x, y in zip(conf["ratios"][s0], conf["ratios"]["recorded_best"])]
    out = {
        "target": {"depth": a.depth, "work": a.work, "layers": a.layers},
        "strategy": a.strategy, "warm": a.warm, "budget": a.budget, "seed": a.seed, "via_runner": a.via_runner,
        "batches": getattr(ev, "batches", None),
        "evaluations": res.evaluations, "stopped": res.stopped, "search_seconds": round(search_s, 1),
        "space_size_bound": ad.describe()["size_bound"], "failures": fails,
        "curve": curve,
        "finalists": {n: ad.label(c) for n, c in finalists.items()},
        "finalist_knobs": finalists,
        "confirm_median_vs_base": med,
        "verdicts": conf["verdicts"],
        "controls_median": {k: statistics.median(v) for k, v in conf["controls"].items()},
        "search_pick": s0,
        "search_vs_recorded_best": {"median": statistics.median(head), "frac_below_1": sum(x < 1 for x in head) / len(head),
                                    "rounds": len(head)},
        "post_check_ok": conf["post_check_ok"], "clocks": conf["clocks"], "gate_selftest": conf["gate_selftest"],
        "evals_to_beat_recorded": next((c["n"] for c in curve if c["incumbent"] <= med["recorded_best"]), None),
        "solo": solo_info,
    }  # fmt: skip
    if "solo_winners_only" in conf["ratios"]:
        sw = [x / y for x, y in zip(conf["ratios"][s0], conf["ratios"]["solo_winners_only"])]
        out["search_vs_solo_winners_only"] = {"median": statistics.median(sw),
                                              "frac_below_1": sum(x < 1 for x in sw) / len(sw), "rounds": len(sw)}  # fmt: skip
    def _ratios_only(c):
        """``combos`` with the measured values (absolute times in a live
        run) replaced by their ratio: best-with-solo-loser over
        best-solo-winners-only (the in-search measurements)."""
        if not c:
            return c
        c = json.loads(json.dumps(c))
        w, wo = c.get("best_with_solo_loser"), c.get("best_solo_winners_only")
        if w and wo:
            c["with_loser_vs_winners_only"] = w["value"] / wo["value"]
        for x in (w, wo):
            if x:
                x.pop("value", None)
        return c

    if solo_info is not None:
        solo_info["combos"] = _ratios_only(solo_info["combos"])
    if solo_info is not None:
        # the pick's levels that LOST alone (label "param=repr(value)")
        pick = finalists[s0]
        out["pick_solo_losers"] = [
            x for x in solo_info["combos"]["solo_losers"]
            if x.split("=", 1)[0] in pick and repr(pick[x.split("=", 1)[0]]) == x.split("=", 1)[1]
        ]  # fmt: skip
    if isinstance(strat, SignedLookback):
        out["lookback"] = strat.report()
    elif hasattr(strat, "members") and not hasattr(strat, "asks"):
        out["phased"] = {"members": [m.name for m in strat.members], "asked": strat.asked}
        for m in strat.members:
            if isinstance(m, SignedLookback):
                out["lookback"] = {k: v for k, v in m.report().items() if k != "ledger"}
    elif hasattr(strat, "members"):
        out["portfolio"] = {"asks": strat.asks, "wins": strat.wins, "members": [m.name for m in strat.members]}
        for m in strat.members:
            if isinstance(m, SignedLookback):
                out["lookback"] = {k: v for k, v in m.report().items() if k != "ledger"}
    print("LIVE", json.dumps({k: out[k] for k in ("evaluations", "stopped", "search_pick", "search_vs_recorded_best",
                                                  "evals_to_beat_recorded", "failures")}), flush=True)  # fmt: skip
    print("SOLO", json.dumps({"vs_solo_winners_only": out.get("search_vs_solo_winners_only"),
                              "pick_solo_losers": out.get("pick_solo_losers"),
                              "solo_winners": (solo_info or {}).get("solo_winners")}), flush=True)  # fmt: skip
    print("CONFIRM", json.dumps({n: round(v, 4) for n, v in med.items()}), json.dumps(conf["verdicts"]), flush=True)
    if "lookback" in out:
        out["lookback"]["combos"] = _ratios_only(out["lookback"].get("combos"))
    _w(a.out, out)
    _w(a.private_out, {"records": be.records, "observations": [o.to_json() for o in res.observations]})
    if a.db:
        db = D.TuningDB(a.db, min_speedup=1.0)
        ver = conf["verdicts"][s0]
        if ver["verdict"] == "WIN":
            from rocke.dispatch.tuning.search.stats import paired_t_greater

            # one-sided paired t on the per-round log ratios vs the baseline
            pv = paired_t_greater([-math.log(x) for x in conf["ratios"][s0]], 0.0)
            rec = D.make_record(ad, finalists[s0], ratio=med[s0], speedup=1.0 / med[s0], p_value=pv,
                                rounds=len(conf["ratios"][s0]), strategy=a.strategy, evaluations=res.evaluations,
                                seed=a.seed)  # fmt: skip
            db.put(rec)
            db.save()
        if a.gains and "lookback" in out:
            led = D.GainLedger(a.gains)
            led.add(ad.stem, out["lookback"].get("signed_gains", {}))
            led.add_pairs(ad.stem, out["lookback"].get("interactions", {}))
            led.add_verdicts(ad.stem, out["lookback"].get("verdicts", []))
            led.save()
    be.close()
    return 0


def cmd_runner(a) -> int:
    from .evaluate import decode_batch, run_batch

    res = run_batch(decode_batch(a.batch))
    _w(a.out, res)
    st = {}
    for r in res["items"]:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print("RUNNER", json.dumps(st), json.dumps(res["meta"].get("gate_selftest")), flush=True)
    return 0


def cmd_export(a) -> int:
    from rocke.dispatch.tuning.search.db import TuningDB

    from .tdb import export_defaults

    d = export_defaults(TuningDB(a.db))
    if a.out:
        _w(a.out, d)
    print(json.dumps(d, indent=1, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernels.gfx950.qwen38_mono.tune", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("catalog")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_catalog)

    p = sub.add_parser("replay-grid")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--strategies", default="phased,lookback,tpe,gp,portfolio,random")
    p.add_argument("--budgets", default="16,32,64")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--transfer", action="store_true")
    p.add_argument("--transfer-strategies", default="lookback,mono,hill")
    p.add_argument("--transfer-budgets", default="8,16")
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_replay_grid)

    p = sub.add_parser("replay-sched")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--variants", default="r1,r4s")
    p.add_argument("--kind", default="delta")
    p.add_argument("--strategies", default="phased,lookback,tpe,portfolio,random")
    p.add_argument("--budgets", default="8,16,24")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--decide-strategy", default="phased")
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_replay_sched)

    p = sub.add_parser("live-toy")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--work", type=int, default=8)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--strategy", default="phased", help="phased (default: signed lookback then TPE) | lookback | mono | any engine strategy")
    p.add_argument("--warm", choices=("shipped", "neighbor"), default="shipped")
    p.add_argument("--neighbor-depth", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--replays", type=int, default=3)
    p.add_argument("--launches", type=int, default=12)
    p.add_argument("--finalists", type=int, default=3)
    p.add_argument("--confirm-rounds", type=int, default=12)
    p.add_argument("--out", default="")
    p.add_argument("--private-out", default="")
    p.add_argument("--db", default="")
    p.add_argument("--gains", default="")
    p.add_argument("--via-runner", action="store_true", help="measure through the batch runner (subprocess)")
    p.set_defaults(fn=cmd_live_toy)

    p = sub.add_parser("runner")
    p.add_argument("--batch", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_runner)

    p = sub.add_parser("export-defaults")
    p.add_argument("--db", required=True)
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_export)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
