# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Replay experiments: does a search find the recorded optimum, and for how
many evaluations?

:func:`grid_problem` / :func:`sched_problem` turn recorded data into a
:class:`ReplayProblem` (space, backend factory, true values, start config);
:func:`run_strategy` runs one (strategy, budget, seed) and scores it against
the recorded optimum:

- ``frac_gain``: the share of the start -> optimum gain the search's pick
  achieves, by the pick's *recorded* value (the search sees only noisy
  measurements, so a pick can be a noisy near-tie);
- ``hit``: the pick is within ``tol`` (relative; default 0.002, the DESIGN
  9.1 margin) of the recorded optimum, i.e. inside the measurement noise;
- ``evals_to_hit`` / ``evals_to_95``: evaluations until the incumbent (by
  the search's own measurements) first is a hit / has 95 % of the gain.

Strategies by name: ``lookback`` (:class:`~.strategies.SignedLookback`),
``mono`` (:func:`~.strategies.mono_portfolio`), and every engine strategy
(``portfolio``, ``hill``, ``de``, ``tpe``, ``gp``, ``random``,
``exhaustive``). Every non-random strategy starts from the start config
(the shipped build): a warm start, as a real tuning run would.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from rocke.dispatch.tuning.search.api import make_strategy
from rocke.dispatch.tuning.search.driver import run_search
from rocke.dispatch.tuning.search.measure import Evaluator, TableBackend
from rocke.dispatch.tuning.search.params import SearchSpace, config_items_key
from rocke.dispatch.tuning.search.portfolio import WarmStart
from rocke.dispatch.tuning.search.trial import Budget, Fidelity

from .replay import RoundReplayBackend, RoundTable, listed_space
from .strategies import SignedLookback, mono_portfolio

__all__ = ["ReplayProblem", "grid_problem", "run_strategy", "sched_problem", "summarize"]


@dataclass
class ReplayProblem:
    name: str
    space: SearchSpace
    truth: dict  # engine key -> recorded value (median)
    make_backend: Callable[[int], object]
    start: dict
    reps: int = 3
    meta: dict | None = None

    @property
    def oracle(self) -> float:
        return min(self.truth.values())

    def start_value(self) -> float:
        k = self.space.key(self.space.encode(self.start))
        return self.truth[k]


def grid_problem(grid, defaults: Mapping[str, object], *, order=None, per_rep: float = 0.002) -> ReplayProblem:
    """A recorded lookback grid (:func:`.replay.load_lookback_grid`)."""
    space, key_map = listed_space(grid.configs, defaults, name=f"b3grid:{grid.variant}", order=order)
    truth = {ek: grid.value[tk] for ek, tk in key_map.items()}

    def backend(seed: int):
        return TableBackend(truth, noise=per_rep, call_noise=grid.noise, seed=seed)

    start = {**{k: defaults.get(k) for k in space.decode(tuple(0 for _ in space.params))}, **grid.start}
    start_note = "shipped"
    g0 = space.encode(start)
    if space.key(g0) is None:
        # the grid does not hold the shipped build (a lookback grid around a
        # climb's best): start from its nearest recorded member
        import random as _r

        g0 = space.repair(g0, _r.Random(0))
        start = space.decode(g0)
        start_note = "nearest member to the shipped build"
    return ReplayProblem(f"b3:{grid.variant}", space, truth, backend, start,
                         meta={"cells": len(truth), "noise": grid.noise, "structural": ["split"],
                               "start": start_note})  # fmt: skip


def sched_problem(table: RoundTable, defaults: Mapping[str, object], *, variant: str) -> ReplayProblem:
    """Recorded schedule A/B rounds (:func:`.replay.load_sched_rounds`)."""
    space, key_map = listed_space(table.configs, defaults, name=f"sched:{variant}")
    med = table.medians()
    truth = {ek: med[tk] for ek, tk in key_map.items()}

    def backend(seed: int):
        return RoundReplayBackend(table, key_map, seed=seed)

    start = {k: defaults.get(k) for k in space.decode(tuple(0 for _ in space.params))}
    return ReplayProblem(f"sched:{variant}", space, truth, backend, start, reps=1,
                         meta={"cells": len(truth), "controls": {k: len(v) for k, v in table.controls.items()},
                               "structural": ["numa"]})  # fmt: skip


def _changes(space: SearchSpace, g, ref) -> set:
    act = space.active(g)
    return {(i, int(g[i])) for i in range(space.dims) if act[i] and g[i] != ref[i]}


def solo_analysis(problem: "ReplayProblem") -> dict:
    """The recorded data's solo picture (PLAN decision 13): every recorded
    single change from the start, the levels that win alone, the best
    recorded config made ONLY of solo winners ("solo-winners-only optimum")
    and whether the recorded optimum needs a solo loser."""
    space = problem.space
    s0 = space.encode(problem.start)
    v0 = problem.truth[space.key(s0)]
    winners, losers = set(), set()
    for i in range(space.dims):
        for lv in range(space.params[i].size):
            if lv == s0[i]:
                continue
            g = list(s0)
            g[i] = lv
            k = space.key(g)
            if k is None or k not in problem.truth:
                continue
            (winners if problem.truth[k] < v0 else losers).add((i, lv))
    members = space.listed() or []
    swo = v0
    for g in members:
        if _changes(space, g, s0) <= winners:
            swo = min(swo, problem.truth[space.key(g)])
    ok = min(problem.truth, key=problem.truth.get)
    og = next(g for g in members if space.key(g) == ok)
    oc = _changes(space, og, s0)
    return {"start": v0, "solo_winners": len(winners), "solo_losers": len(losers),
            "solo_winners_only_opt": swo, "oracle": problem.oracle,
            "oracle_uses_non_winner": bool(oc - winners),
            "oracle_solo_losers": sorted(f"{space.params[i].name}={space.params[i].label(lv)}" for i, lv in oc & losers),
            "_winners": winners, "_s0": s0}  # fmt: skip


def _strategy(name: str, problem: ReplayProblem, seed: int, warm: Sequence[Mapping] = (), budget: int | None = None):
    space = problem.space
    start_g = space.encode(problem.start)
    ws = [WarmStart(problem.start, "shipped")] + [WarmStart(w, "transfer") for w in warm]
    structural = list((problem.meta or {}).get("structural", ()))
    if name == "lookback":
        if warm:
            # a transfer start: climb from the nearest variant's pick (repaired into this space)
            import random as _r

            g = space.repair(space.encode({**problem.start, **dict(warm[0])}), _r.Random(seed))
            start_g = g if g is not None else start_g
        return SignedLookback(space, seed, start=start_g, structural=structural, budget=budget)
    if name == "phased" or name.startswith("phased:"):
        from .strategies import phased

        members = tuple(name.split(":", 1)[1].split("+")) if ":" in name else ("lookback", "tpe")
        return phased(space, seed, members=members, start=start_g, structural=structural, warm=ws,
                      lookback_options={"budget": budget})
    if name == "mono" or name.startswith("mono:"):
        members = tuple(name.split(":", 1)[1].split("+")) if ":" in name else ("lookback", "hill", "de", "tpe")
        return mono_portfolio(space, seed, warm=ws, members=members, lookback_options={"structural": structural})
    if name in ("random", "exhaustive"):
        return make_strategy(name, space, seed)
    return make_strategy(name, space, seed, warm=ws)


def run_strategy(
    problem: ReplayProblem,
    name: str,
    budget: int,
    seed: int,
    *,
    batch: int = 4,
    tol: float = 0.002,
    warm: Sequence[Mapping] = (),
) -> dict:
    space = problem.space
    backend = problem.make_backend(seed)
    ev = Evaluator(space, backend, ladder=(Fidelity("full", reps=problem.reps, warmup=0),))
    strat = _strategy(name, problem, seed, warm, budget)
    res = run_search(space, strat, ev, Budget(evaluations=int(budget)), batch=batch, confirm=0, stall_rounds=16)
    v0, vopt = problem.start_value(), problem.oracle
    gain = v0 - vopt
    # anytime: the incumbent by the search's own measurements, scored by truth
    best_meas, inc_key = math.inf, None
    evals_to = {"95": None, "hit": None}
    for i, ob in enumerate([o for o in res.observations if o.ok and o.full and not o.cached], start=1):
        if ob.value < best_meas:
            best_meas, inc_key = ob.value, ob.key
        tv = problem.truth.get(inc_key, math.inf)
        frac = (v0 - tv) / gain if gain > 0 else 1.0
        if evals_to["95"] is None and frac >= 0.95:
            evals_to["95"] = i
        if evals_to["hit"] is None and tv <= vopt * (1.0 + tol):
            evals_to["hit"] = i
    picked = problem.truth.get(res.best_key, math.inf)
    sa = (problem.meta or {}).get("solo")
    solo = {}
    if sa is not None and res.best_genome is not None:
        ch = _changes(space, res.best_genome, sa["_s0"])
        solo = {"pick_uses_non_winner": bool(ch - sa["_winners"]),
                "beats_solo_winners_only": picked < sa["solo_winners_only_opt"] * (1.0 - tol)}
        solo["combo_with_non_winner_beats_swo"] = solo["pick_uses_non_winner"] and solo["beats_solo_winners_only"]
    out = {
        "problem": problem.name,
        "strategy": name,
        "budget": budget,
        "seed": seed,
        "evaluations": res.evaluations,
        "stopped": res.stopped,
        "pick_value": picked,
        "frac_gain": (v0 - picked) / gain if gain > 0 else 1.0,
        "hit": picked <= vopt * (1.0 + tol),
        "exact": picked == vopt,
        "evals_to_95": evals_to["95"],
        "evals_to_hit": evals_to["hit"],
        "pick": space.describe_genome(res.best_genome) if res.best_genome is not None else None,
        **solo,
    }
    if isinstance(strat, SignedLookback):
        out["lookback"] = {k: v for k, v in strat.report().items() if k not in ("ledger", "signed_gains")}
    return out


def summarize(rows: Sequence[Mapping]) -> list[dict]:
    """Per (problem, strategy, budget): mean / min fraction of the gain,
    hit rate, median evaluations to 98 % (over seeds; None counts as
    not reached)."""
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["problem"], r["strategy"], r["budget"]), []).append(r)
    out = []
    for (p, s, b), rs in sorted(groups.items()):
        reach = [r["evals_to_hit"] for r in rs if r["evals_to_hit"] is not None]
        r95 = [r["evals_to_95"] for r in rs if r["evals_to_95"] is not None]
        out.append(
            {
                "problem": p, "strategy": s, "budget": b, "seeds": len(rs),
                "mean_frac_gain": statistics.mean(r["frac_gain"] for r in rs),
                "min_frac_gain": min(r["frac_gain"] for r in rs),
                "hit_rate": sum(r["hit"] for r in rs) / len(rs),
                "exact_rate": sum(r["exact"] for r in rs) / len(rs),
                "reached_hit": len(reach) / len(rs),
                "median_evals_to_hit": statistics.median(reach) if reach else None,
                "reached_95": len(r95) / len(rs),
                "median_evals_to_95": statistics.median(r95) if r95 else None,
                "mean_evaluations": statistics.mean(r["evaluations"] for r in rs),
                "combo_with_non_winner_beats_swo": (statistics.mean(bool(r.get("combo_with_non_winner_beats_swo"))
                                                                    for r in rs) if "pick_uses_non_winner" in rs[0] else None),
            }
        )  # fmt: skip
    return out


def key_of(config: Mapping) -> str:
    return config_items_key(dict(config))


def dumps(rows) -> str:
    return json.dumps(rows, indent=1, sort_keys=True, default=str)
