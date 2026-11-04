#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
The adaptive search coordinator (tile_engine/ops/common/adaptive_search.py).

A fake OperationAdapter records every build, measurement and confirmation call;
scripted strategies propose fixed kernel names round by round, so each test can
state exactly which pairs are built and measured. The real ck_search strategies
run once end to end at the bottom.

No GPU or compiler needed. Run: python3 -m pytest -q tests/test_adaptive_search_coordinator.py
"""

import hashlib
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
TE_OPS = DISPATCHER_DIR.parent / "tile_engine" / "ops"
COMMON_DIR = TE_OPS / "common"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(COMMON_DIR))

import adaptive_search as AS  # noqa: E402
import ck_search  # noqa: E402
from _py38_check import py38_problems  # noqa: E402

KERNELS = "ABCDEFGHIJKL"
# The modules this change adds (also ASCII only) or edits, besides ck_search.
NEW_MODULES = (COMMON_DIR / "adaptive_search.py", TE_OPS / "gemm" / "gemm_search.py")
EDITED_MODULES = (
    TE_OPS / "gemm" / "gemm_full_benchmark.py",
    TE_OPS / "gemm" / "run_one_gemm_kernel.py",
    DISPATCHER_DIR / "python" / "ctypes_utils.py",
)


def rec(name):
    """One distinct scalar record per kernel, the same in every problem."""
    i = KERNELS.index(name)
    return {"tile_m": 16 * (i + 1), "tile_n": 64 if i % 2 else 128}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeAdapter:
    """``problems`` {problem: [kernel names]}; ``lat`` {(problem, name): ms};
    ``script`` {(problem, name): [status per measurement]} (default ok);
    ``confirm_script`` {(problem, name): [(status, ms) per fresh call]}
    (default ok at the discovery latency)."""

    def __init__(self, problems, lat=None, fail_build=(), script=None,
                 confirm_script=None, clock=None, advance=0.0):
        self.problems = problems
        self.lat = lat or {}
        self.fail_build = set(fail_build)
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.confirm_script = {k: list(v) for k, v in (confirm_script or {}).items()}
        self.clock, self.advance = clock, advance
        self.builds, self.measured, self.confirmed = [], [], []

    def context(self):
        return {"arch": "gfx950", "dtype": "fp16", "layout": "rcr", "variant": "fake"}

    def enumerate_eligible(self):
        return {
            p: [AS.Candidate(n, n, rec(n), {"cfg": n}) for n in names]
            for p, names in self.problems.items()
        }

    def build_missing(self, cands):
        self.builds.append([c.canonical_key for c in cands])
        return {
            c.canonical_key: AS.BuildResult(None, "fake compile error")
            if c.canonical_key in self.fail_build
            else AS.BuildResult(f"artifacts/{c.canonical_key}.so", digest="0" * 64)
            for c in cands
        }

    def _lat(self, r):
        return self.lat.get((r.problem_key, r.kernel_name), 1.0)

    def measure_pairs(self, requests):
        self.measured.extend(requests)
        if self.clock is not None:
            self.clock.t += self.advance
        out = []
        for r in requests:
            todo = self.script.get((r.problem_key, r.kernel_name))
            status = todo.pop(0) if todo else "ok"
            lat = self._lat(r) if status == "ok" else None
            out.append(AS.PairOutcome(r, status, lat, "" if lat else f"fake {status}"))
        return out

    def confirm(self, requests):
        self.confirmed.append(list(requests))
        out = []
        for r in requests:
            todo = self.confirm_script.get((r.problem_key, r.kernel_name))
            status, lat = todo.pop(0) if todo else ("ok", self._lat(r))
            out.append(AS.PairOutcome(r, status, lat))
        return out


class Scripted:
    """Proposes ``steps`` (lists of kernel names, or raw genome tuples), one
    step per ask."""

    def __init__(self, space, steps):
        self.space, self.steps = space, [list(s) for s in steps]
        self.asks, self.told, self.pending = [], [], {}
        self._uid = 0

    def ask(self, n):
        self.asks.append(n)
        if not self.steps:
            return []
        step = self.steps.pop(0)
        if len(step) > n:
            self.steps.insert(0, step[n:])
            step = step[:n]
        out = []
        for name in step:
            self._uid += 1
            g = name if isinstance(name, tuple) else self.space.encode(rec(name))
            t = ck_search.Trial(genome=g, uid=f"{self.space.name}:{self._uid}")
            self.pending[t.uid] = t
            out.append(t)
        return out

    def tell(self, observations):
        for ob in observations:
            self.pending.pop(ob.trial.uid, None)
            self.told.append(ob)

    @property
    def done(self):
        return not self.steps


def scripted(scripts, calls=None):
    made = {}

    def factory(name, space, seed, warm_records, prior):
        if calls is not None:
            calls.append({"name": name, "problem": space.name, "seed": seed,
                          "warm": list(warm_records), "prior": prior})
        made[space.name] = Scripted(space, scripts.get(space.name, []))
        return made[space.name]

    return factory, made


def settings(**kw):
    d = {"schema_version": 1, "strategy": "random", "trials_per_problem": 4,
         "new_builds_per_run": 20, "batch_per_problem": 2}
    d.update(kw)
    return AS.SearchSettings.from_json(d)


def run(tmp_path, adapter, scripts, clock=None, calls=None, pair_loader=None,
        prior_loader=None, factory=None, **kw):
    """One run with scripted strategies (or ``factory``); (summary, strategies
    by problem, coordinator)."""
    made = {}
    if factory is None:
        factory, made = scripted(scripts, calls)
    co = AS.SearchCoordinator(
        adapter, settings(**kw), tmp_path / "out", clock=clock or FakeClock(),
        strategy_factory=factory, pair_loader=pair_loader, prior_loader=prior_loader,
    )
    return co.run(), made, co


def json_pairs(path):
    """A pair_loader for tests that need no driver: the --pairs JSON as is."""
    return json.loads(Path(path).read_text())


def _no_constant(c):
    raise AssertionError(f"non-strict JSON constant {c}")


def strict(path):
    return json.loads(Path(path).read_text(), parse_constant=_no_constant)


def events(out_dir, kind=None):
    rows = [json.loads(line, parse_constant=_no_constant)
            for line in (Path(out_dir) / "trials.jsonl").read_text().splitlines()]
    return [r for r in rows if kind is None or r["event"] == kind]


def pairs(requests):
    return [(r.problem_key, r.kernel_name) for r in requests]


ABCD = list("ABCD")


# ----------------------------------------------------------------- import


@pytest.mark.parametrize("path", NEW_MODULES + EDITED_MODULES, ids=lambda p: p.name)
def test_module_is_python38_compatible(path):
    src = path.read_text(encoding="utf-8")
    assert py38_problems(src, path.name) == []
    assert src.isascii() or path not in NEW_MODULES


def test_import_is_lean():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import adaptive_search; "
        "bad = [m for m in ('ck_search', 'numpy', 'rocke', 'lightgbm', 'pandas') "
        "if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-B", "-c", code, str(COMMON_DIR)], check=True)


# ------------------------------------------------------- shared artifacts


def test_shared_artifacts_built_once_and_winners_confirmed(tmp_path):
    """P1 asks A,B then C; P2 asks B,C then A,D: 4 builds, 7 measurements."""
    lat = {("P1", "A"): 1.0, ("P1", "B"): 2.0, ("P1", "C"): 1.5,
           ("P2", "B"): 1.2, ("P2", "C"): 0.9, ("P2", "A"): 3.0, ("P2", "D"): 1.1}
    # P1's discovery best (A) was a lucky sample: confirmation picks C.
    confirm = {("P1", "A"): [("ok", 3.0), ("ok", 3.1), ("ok", 2.9)]}
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD}, lat=lat, confirm_script=confirm)
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C"]],
                                     "P2": [["B", "C"], ["A", "D"]]})
    assert ad.builds == [["A", "B", "C"], ["D"]]
    assert sorted(pairs(ad.measured)) == sorted(
        [("P1", "A"), ("P1", "B"), ("P2", "B"), ("P2", "C"),
         ("P1", "C"), ("P2", "A"), ("P2", "D")])
    # Confirmation: 3 fresh calls x 2 finalists x 2 problems, interleaved by call.
    assert [len(c) for c in ad.confirmed] == [4, 4, 4]
    ids = [r.call_id for c in ad.confirmed for r in c]
    assert len(set(ids)) == 12 and not set(ids) & {r.call_id for r in ad.measured}
    assert s.counts["measured"] == 7 and s.counts["confirm_calls"] == 12
    assert s.counts["built"] == 4 and co.ledger.builds == 4
    assert s.winners == {"P1": "C", "P2": "C"}
    assert (s.status, s.exit_code, s.stop_reason) == ("complete", 0, "searched")
    assert strict(tmp_path / "out" / "winners.json") == {"P1": ["C"], "P2": ["C"]}
    assert [len(made[p].told) for p in ("P1", "P2")] == [3, 4]
    assert len(events(tmp_path / "out", "build")) == 4
    assert len(events(tmp_path / "out", "result")) == 7
    assert len(events(tmp_path / "out", "confirm")) == 12


def test_failed_confirmation_is_no_winner(tmp_path):
    """Only a verified finite result can be a finalist; when it fails its fresh
    calls the problem has no winner (no fallback to the invalid rows)."""
    lat = {("P1", "B"): math.nan, ("P1", "C"): 2.0}
    ad = FakeAdapter({"P1": ABCD}, lat=lat,
                     script={("P1", "A"): ["numeric_fail"], ("P1", "D"): ["run_error"]},
                     confirm_script={("P1", "C"): [("ok", 2.0), ("numeric_fail", None)]})
    s, made, _ = run(tmp_path, ad, {"P1": [["A", "B"], ["C", "D"]]})
    assert [pairs(c) for c in ad.confirmed] == [[("P1", "C")], [("P1", "C")]]
    assert (s.status, s.exit_code, s.winners) == ("no_winner", 1, {})
    assert "no finalist passed confirmation" in s.unresolved["P1"]
    assert strict(tmp_path / "out" / "winners.json") == {}
    assert s.counts["failed_by_status"] == {
        "numeric_fail": 1, "invalid_measurement": 1, "run_error": 1}
    told = {ob.trial.uid: ob for ob in made["P1"].told}
    assert [ob.status for ob in told.values()] == [
        "numeric_fail", "run_error", "ok", "run_error"]
    assert told["P1:2"].detail.startswith("invalid_measurement")
    assert all(ob.value == math.inf for ob in told.values() if ob.status != "ok")


def test_partial_and_no_eligible_problem(tmp_path):
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": []}, fail_build={"C", "D"})
    s, _, _ = run(tmp_path, ad, {"P1": [["A"]], "P2": [["C", "D"]]})
    assert (s.status, s.exit_code) == ("partial", 1)
    assert s.winners == {"P1": "A"}
    assert s.unresolved["P2"].startswith("no verified discovery measurement")
    assert s.unresolved["P3"] == "no eligible candidates"


# ------------------------------------------------------------- failures


def test_compile_failure_shared_by_problems_is_one_build(tmp_path):
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": ABCD}, fail_build={"B"})
    s, made, co = run(tmp_path, ad, {"P1": [["B", "A"]], "P2": [["B", "C"]],
                                     "P3": [["D"], ["B"]]})
    assert ad.builds == [["B", "A", "C", "D"]]  # round 2 re-asks B: no rebuild
    assert (s.counts["built"], s.counts["build_failed"], co.ledger.builds) == (3, 1, 4)
    assert "B" not in [r.kernel_name for r in ad.measured]
    errs = {p: [ob.cached for ob in made[p].told if ob.status == "compile_error"]
            for p in made}
    # P1 and P2 asked for B in the round that built it, each with a trial; P3
    # asks once its build is known to have failed: told that result, no trial.
    assert errs == {"P1": [False], "P2": [False], "P3": [True]}
    assert s.counts["failed_by_status"] == {"compile_error": 2}
    assert s.counts["known_failed"] == 1 and co.ledger.trials["P3"] == 1
    assert made["P3"].told[-1].detail == "fake compile error"
    build = events(tmp_path / "out", "build")
    assert [b["waiting"] for b in build if b["kernel"] == "B"] == [["P1", "P2"]]
    (known,) = [e for e in events(tmp_path / "out", "trial")
                if e["admission"] == "known_failed"]
    assert (known["problem"], known["kernel"]) == ("P3", "B")


def test_known_build_failures_cost_no_trial(tmp_path):
    # P1's builds of A..D all fail. P2 then proposes A..D: each is told its
    # compile_error from the registry, so P2's trials go to G, E and F only.
    ad = FakeAdapter({"P1": ABCD, "P2": list("ABCDEFG")}, fail_build=set(ABCD))
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [["G"], ABCD, ["E", "F"]]},
                      trials_per_problem=6, batch_per_problem=4, max_stall_rounds=2)
    assert ad.builds == [list("ABCDG"), ["E", "F"]]
    assert sorted(pairs(ad.measured)) == [("P2", n) for n in "EFG"]
    assert co.ledger.trials == {"P1": 4, "P2": 3} and s.counts["known_failed"] == 4
    assert all(ob.cached for ob in made["P2"].told if ob.status == "compile_error")
    assert s.winners == {"P2": "E"} and s.stop_reason == "searched"


def test_failed_builds_are_no_work_after_the_cap(tmp_path):
    # Round 1 spends the cap on A..D, which all fail. P2's untried candidates
    # are then failed builds and E, which needs a build: nothing is left to
    # run, so the search stops at the cap instead of asking again.
    ad = FakeAdapter({"P1": ABCD, "P2": list("ABCDE")}, fail_build=set(ABCD))
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [["A"], ["E"]]},
                      new_builds_per_run=4, batch_per_problem=4)
    assert s.stop_reason == "build_cap" and s.counts["rounds"] == 1
    assert len(made["P2"].asks) == 1 and co.ledger.builds == 4


def test_unattempted_is_requeued_once(tmp_path):
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted", "ok"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C"]]})
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B"), ("P1", "B"), ("P1", "C")]
    assert ad.measured[1].trial_uid == ad.measured[2].trial_uid
    assert ad.measured[1].call_id != ad.measured[2].call_id
    assert made["P1"].asks == [2, 1]  # the re-queued trial takes a batch slot
    assert s.counts["requeued"] == 1 and co.ledger.trials["P1"] == 3
    assert [ob.status for ob in made["P1"].told] == ["ok", "ok", "ok"]
    assert [e["admission"] for e in events(tmp_path / "out", "trial")] == [
        "admitted", "admitted", "readmitted", "admitted"]


def test_unattempted_twice_is_run_error_and_cancel_is_dropped(tmp_path):
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted", "unattempted"],
                                           ("P1", "C"): ["cancelled"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B", "C"]]}, batch_per_problem=3)
    told = {ob.trial.uid: ob for ob in made["P1"].told}
    assert told["P1:2"].status == "run_error"
    assert "unattempted twice" in told["P1:2"].detail
    assert "P1:3" not in told and s.counts["cancelled"] == 1
    assert co.ledger.trials["P1"] == 2  # A and B charged, cancelled C refunded


def _warm_real(tmp_path, name, script, trials=10):
    """A real strategy on P1 (A..L, A fastest) warm-started with A then B."""
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["A", "B"]}))
    lat = {("P1", n): 1.0 + i for i, n in enumerate(KERNELS)}
    ad = FakeAdapter({"P1": list(KERNELS)}, lat=lat, script=script)
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=trials, new_builds_per_run=40,
                     batch_per_problem=4, warm_start={"pairs_file": str(picks)}),
        tmp_path / "out", clock=FakeClock(), pair_loader=json_pairs,
    )
    return co.run(), ad, co


@pytest.mark.parametrize("name", ["hill", "portfolio", "de"])
def test_requeued_trial_is_pending_not_done(tmp_path, name):
    # Hill and the portfolio's warm-start barrier wait for the re-admitted A:
    # their empty ask in that round means waiting, not done.
    s, ad, co = _warm_real(tmp_path, name, {("P1", "A"): ["unattempted"]})
    assert s.counts["requeued"] == 1 and co.ledger.trials["P1"] == 10
    summary = strict(tmp_path / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "trial budget"
    assert s.winners == {"P1": "A"}


def test_cancelled_trial_ends_a_strategy_that_waits_for_it(tmp_path):
    # The portfolio starts its members once every warm start is known; a
    # cancelled warm start is never told, so it would wait forever.
    s, ad, co = _warm_real(tmp_path, "portfolio", {("P1", "A"): ["cancelled"]})
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B")]
    assert s.counts["cancelled"] == 1 and len(events(tmp_path / "out", "cancel")) == 1
    finished = strict(tmp_path / "out" / "summary.json")["problems"]["P1"]["finished"]
    assert finished == "strategy waits for 1 cancelled trial(s)"
    assert s.stop_reason == "searched" and s.winners == {"P1": "B"}


def test_requeued_trial_cancelled_when_the_search_stops(tmp_path):
    clock = FakeClock()
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted"]},
                     clock=clock, advance=10.0)
    s, _, co = run(tmp_path, ad, {"P1": [["A", "B"]]}, clock=clock,
                   trials_per_problem=2, deadline_s=10)
    assert s.stop_reason == "deadline" and pairs(ad.measured) == [
        ("P1", "A"), ("P1", "B")]
    assert s.counts["cancelled"] == 1 and co.ledger.trials["P1"] == 1
    (cancel,) = events(tmp_path / "out", "cancel")
    assert cancel["trial_uid"] == "P1:2" and "before re-admission" in cancel["diag"]


def test_same_round_duplicate_waits_for_the_first(tmp_path):
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "A"): ["unattempted", "ok"]})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "A"]]})
    # One sample of A: the re-queued trial runs again, its twin is told the result.
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "A")]
    assert ad.measured[0].trial_uid == ad.measured[1].trial_uid == "P1:1"
    told = {ob.trial.uid: ob for ob in made["P1"].told}
    assert not told["P1:1"].cached and told["P1:2"].cached
    assert told["P1:2"].value == told["P1:1"].value == 1.0
    assert co.ledger.trials["P1"] == 1 and s.counts["cached"] == 1


def test_config_refused_by_the_space_is_not_charged(tmp_path):
    ad = FakeAdapter({"P1": ABCD})
    s, made, co = run(tmp_path, ad, {"P1": [[(0, 0), "B"]]})  # (16, 64): no kernel
    assert pairs(ad.measured) == [("P1", "B")]
    assert s.counts["invalid"] == 1 and co.ledger.trials["P1"] == 1
    assert made["P1"].told[0].status == "invalid" and made["P1"].told[0].key is None


def test_run_is_single_use(tmp_path):
    co = AS.SearchCoordinator(FakeAdapter({"P1": ABCD}), settings(), tmp_path / "o",
                              strategy_factory=scripted({"P1": [["A"]]})[0])
    assert co.run().status == "complete"
    with pytest.raises(RuntimeError, match="runs once"):
        co.run()


# -------------------------------------------------------------- budgets


def test_trial_budget_boundary(tmp_path):
    ad = FakeAdapter({"P1": list("ABCDEFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [[n] for n in "ABCDEFGH"]},
                      trials_per_problem=3, batch_per_problem=2)
    assert made["P1"].asks == [2, 2, 1]  # the last ask is cut to the trials left
    assert co.ledger.trials["P1"] == 3
    s2, made2, co2 = run(tmp_path / "b", FakeAdapter({"P1": list("ABCDEFGH")}),
                         {"P1": [list("ABCDEFGH")]}, trials_per_problem=3,
                         batch_per_problem=2)
    assert made2["P1"].asks == [2, 1]
    assert s2.counts["measured"] == 3 and co2.ledger.trials["P1"] == 3
    summary = strict(tmp_path / "b" / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "trial budget"


def test_build_cap_is_never_exceeded(tmp_path):
    ad = FakeAdapter({"P1": ABCD, "P2": list("EFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [list("EFGH")]},
                      new_builds_per_run=4, batch_per_problem=4)
    assert ad.builds == [ABCD] and co.ledger.builds == 4
    assert made["P2"].asks == []  # no capacity left for P2's first ask
    assert s.stop_reason == "build_cap" and s.status == "partial"
    assert "build_cap" in s.unresolved["P2"]


def test_build_cap_counts_new_builds_not_cached_trials(tmp_path):
    """Capacity is debited by the new builds a round needs: P2 re-using P1's
    kernels leaves room for P3 in the same round."""
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": list("EFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"]], "P2": [["A", "B"]],
                                     "P3": [["E", "F"]]},
                      new_builds_per_run=4, batch_per_problem=2)
    assert ad.builds == [["A", "B", "E", "F"]] and co.ledger.builds == 4
    assert [made[p].asks for p in ("P1", "P2", "P3")] == [[2], [2], [2]]
    assert s.counts["measured"] == 6 and s.status == "complete"


@pytest.mark.parametrize("factory", ["scripted", "random"])
def test_build_cap_admits_work_that_needs_no_build(tmp_path, factory):
    """Two identical four-kernel domains, trials = batch = cap = 4: P2 re-uses
    the four kernels P1 builds, in the same round, without another build."""
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD})
    s, made, co = run(tmp_path, ad, {"P1": [ABCD], "P2": [ABCD]},
                      factory=None if factory == "scripted" else AS._default_factory,
                      new_builds_per_run=4, batch_per_problem=4)
    assert [sorted(b) for b in ad.builds] == [ABCD] and co.ledger.builds == 4
    assert sorted(pairs(ad.measured)) == [(p, n) for p in ("P1", "P2") for n in ABCD]
    assert s.status == "complete" and s.counts["cancelled"] == 0


def test_requeued_trial_runs_again_after_the_build_cap(tmp_path):
    # The cap is spent in round 1, but B is built: re-admitting it needs none.
    ad = FakeAdapter({"P1": ABCD}, script={("P1", "B"): ["unattempted"]})
    s, _, co = run(tmp_path, ad, {"P1": [["A", "B"]]}, trials_per_problem=2,
                   new_builds_per_run=2)
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B"), ("P1", "B")]
    assert s.stop_reason == "searched" and s.counts["cancelled"] == 0
    assert co.ledger.trials["P1"] == 2 and co.ledger.builds == 2


def test_proposal_needing_a_build_past_the_cap_is_told_invalid(tmp_path):
    # Round 1: P1 plans the whole cap (A, B); P2 may still ask for kernels that
    # need no build: A runs, E would need a third build and is told invalid at
    # once, without a trial. Round 2: the cap is spent, but P2 re-uses B.
    ad = FakeAdapter({"P1": ABCD, "P2": list("ABCE")})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"]], "P2": [["A", "E"], ["B"]]},
                      trials_per_problem=2, new_builds_per_run=2)
    assert ad.builds == [["A", "B"]] and made["P2"].asks == [2, 1]
    assert sorted(pairs(ad.measured)) == [("P1", "A"), ("P1", "B"), ("P2", "A"),
                                          ("P2", "B")]
    assert not events(tmp_path / "out", "cancel") and s.counts["cancelled"] == 0
    (past,) = [e for e in events(tmp_path / "out", "trial")
               if e["admission"] == "past_build_cap"]
    assert (past["kernel"], past["trial_uid"]) == ("E", "P2:2")
    told = {ob.trial.uid: ob for ob in made["P2"].told}
    assert told["P2:2"].status == "invalid" and told["P2:2"].key is not None
    assert "new_builds_per_run" in told["P2:2"].detail
    assert s.counts["past_build_cap"] == 1 and co.ledger.trials["P2"] == 2
    assert not made["P2"].pending
    assert s.status == "complete"


def _cap_grid():
    """144 kernels, the same list for every problem."""
    shapes = itertools.product((32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128),
                               ("mem", "compv3", "compv4"))
    return {f"k{i:03d}": {"tile_m": m, "tile_n": n, "tile_k": k, "pipeline": pl}
            for i, (m, n, k, pl) in enumerate(shapes)}


class CapAdapter(FakeAdapter):
    """``_cap_grid`` for every problem; a fixed pseudo-random latency per pair."""

    def __init__(self, problems):
        super().__init__({p: [] for p in problems})
        self.grid = _cap_grid()

    def enumerate_eligible(self):
        return {p: [AS.Candidate(n, n, r, None) for n, r in self.grid.items()]
                for p in self.problems}

    def _lat(self, r):
        h = hashlib.sha256((r.problem_key + r.kernel_name).encode()).hexdigest()
        return 1.0 + int(h[:6], 16) % 997 / 100


@pytest.mark.parametrize("name", ["de", "hill", "gp", "portfolio"])
def test_binding_build_cap_strands_no_strategy(tmp_path, name):
    # Three problems on one 144-kernel list, 32 trials each, 40 builds. Past the
    # cap, proposals that need a build are told invalid at once: no strategy
    # waits for them (DE for its generation, hill for its stage, the portfolio
    # for its members), GP keeps no lie for them, and each problem goes on
    # with the kernels already built (before: DE 72 and hill 57 of the 96
    # trials, GP and the portfolio about 100 and 60 pending trials a problem).
    ad = CapAdapter(["P1", "P2", "P3"])
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=32, new_builds_per_run=40,
                     batch_per_problem=4),
        tmp_path / "out", clock=FakeClock())
    s = co.run()
    assert co.ledger.builds == 40 and s.counts["past_build_cap"] > 0
    assert s.counts["cancelled"] == 0
    for p in co.problems:
        assert not p.strategy.pending and "waits" not in p.finished, p.finished
    assert sum(co.ledger.trials.values()) >= 90, co.ledger.trials


def test_progress_lines_per_round_and_build(tmp_path):
    lines = []
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD}, fail_build={"D"})
    co = AS.SearchCoordinator(
        ad, settings(deadline_s=600), tmp_path / "out", clock=FakeClock(),
        strategy_factory=scripted({"P1": [["A", "B"]], "P2": [["B", "D"], ["C"]]})[0],
        log=lines.append)
    co.run()
    assert lines == [
        "  [search] round 1: building 3 kernel(s) (0/20 builds); 0/600 s",
        "  [search] round 1: 4 proposed, 4 admitted, 0 cached, 0 known failed, 0 past"
        " the build cap; 2 built, 1 failed (3/20 builds); 3 measured, 3 ok; 1/2"
        " problems active; 0/600 s",
        "  [search] round 2: building 1 kernel(s) (3/20 builds); 0/600 s",
        "  [search] round 2: 1 proposed, 1 admitted, 0 cached, 0 known failed, 0 past"
        " the build cap; 1 built, 0 failed (4/20 builds); 1 measured, 1 ok; 0/2"
        " problems active; 0/600 s",
        "  [search] confirming 4 finalist(s) of 2 problem(s), 3 fresh call(s) each",
    ]


def test_build_cap_rotates_the_first_problem(tmp_path):
    # Round 2 starts with P2, so the last two builds of the cap go to P2.
    ad = FakeAdapter({"P1": ABCD, "P2": list("EFGH")})
    s, made, co = run(tmp_path, ad, {"P1": [["A", "B"], ["C", "D"]],
                                     "P2": [["E", "F"], ["G", "H"]]},
                      new_builds_per_run=6, batch_per_problem=2)
    assert ad.builds == [["A", "B", "E", "F"], ["G", "H"]]
    assert s.stop_reason == "build_cap" and co.ledger.builds == 6


@pytest.mark.parametrize("deadline, measured, wall", [(25, "ABC", 30.0),
                                                      (20, "AB", 20.0)])
def test_deadline_stops_admission_then_confirms(tmp_path, deadline, measured, wall):
    # Each measurement takes 10 s; a clock exactly at the deadline admits nothing.
    clock = FakeClock()
    ad = FakeAdapter({"P1": list("ABCDEF")}, clock=clock, advance=10.0)
    s, made, _ = run(tmp_path, ad, {"P1": [[n] for n in "ABCDEF"]}, clock=clock,
                     trials_per_problem=6, batch_per_problem=1, deadline_s=deadline)
    assert pairs(ad.measured) == [("P1", n) for n in measured]
    assert s.stop_reason == "deadline"
    assert s.counts["confirm_calls"] == 6 and s.status == "complete"
    assert s.wall_s == wall


def test_no_deadline_unless_one_is_set(tmp_path):
    # Each measurement takes 1000 s: without deadline_s the run still spends
    # its whole trial budget.
    clock = FakeClock()
    ad = FakeAdapter({"P1": list("ABCDEF")}, clock=clock, advance=1000.0)
    s, _, co = run(tmp_path, ad, {"P1": [[n] for n in "ABCDEF"]}, clock=clock,
                   trials_per_problem=6, batch_per_problem=1)
    assert len(ad.measured) == 6 and s.stop_reason == "searched"
    assert co.ledger.deadline_s is None and s.wall_s == 6000.0
    assert strict(tmp_path / "out" / "summary.json")["budget"]["deadline_s"] is None


def test_deadline_is_checked_before_every_ask(tmp_path):
    clock = FakeClock()

    class Slow(Scripted):
        def ask(self, n):  # a strategy that takes 10 s to propose
            clock.t += 10
            return super().ask(n)

    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": ABCD})
    s, _, _ = run(tmp_path, ad, {}, clock=clock, deadline_s=15,
                  factory=lambda name, space, *a: Slow(space, [["A"]]))
    assert pairs(ad.measured) == [("P1", "A"), ("P2", "A")]
    assert s.stop_reason == "deadline" and "deadline" in s.unresolved["P3"]


class Undone(Scripted):
    """Never reports done: only its empty ask can finish its problem."""

    @property
    def done(self):
        return False


def test_empty_ask_with_nothing_pending_finishes_the_problem(tmp_path):
    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {}, max_stall_rounds=50,
                  factory=lambda name, space, *a: Undone(space, [["A"], ["B"]]))
    assert pairs(ad.measured) == [("P1", "A"), ("P1", "B")]
    summary = strict(tmp_path / "out" / "summary.json")
    assert summary["problems"]["P1"]["finished"] == "strategy done"
    assert s.stop_reason == "searched" and s.counts["rounds"] == 3


def test_stall_counter_resets_on_progress(tmp_path):
    # Two stalled rounds in a row stop the run, but never two in a row here.
    steps = [["A"], ["A"], ["B"], ["A"], ["C"], ["B"], ["D"]]
    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {}, max_stall_rounds=2,
                  factory=lambda name, space, *a: Undone(space, steps))
    assert [n for _, n in pairs(ad.measured)] == ["A", "B", "C", "D"]
    assert s.stop_reason == "searched" and s.counts["cached"] == 3


def test_stall_stops_duplicate_proposals(tmp_path):
    ad = FakeAdapter({"P1": ABCD})
    s, made, _ = run(tmp_path, ad, {"P1": [["A"]] * 20}, max_stall_rounds=3)
    assert pairs(ad.measured) == [("P1", "A")]
    assert s.stop_reason == "stalled" and s.counts["cached"] == 3
    cached = [ob for ob in made["P1"].told if ob.cached]
    assert len(cached) == 3 and all(ob.value == 1.0 for ob in cached)
    assert s.winners == {"P1": "A"}


# -------------------------------------------------------------- outputs


def test_outputs_are_strict_json_and_winners_load_as_pairs(tmp_path):
    sys.path.insert(0, str(TE_OPS / "gemm"))
    sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
    import gemm_full_benchmark as drv

    lat = {("512x512x512", "A"): math.inf, ("512x512x512", "B"): 0.5}
    ad = FakeAdapter({"512x512x512": ABCD}, lat=lat)
    s, _, _ = run(tmp_path, ad, {"512x512x512": [["A", "B"]]})
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted(AS.OUTPUT_FILES)
    for name in ("run.json", "summary.json", "winners.json"):
        strict(out / name)
    text = (out / "trials.jsonl").read_text()
    assert "Infinity" not in text and "NaN" not in text
    assert all(e["latency_ms"] is None for e in events(out, "result")
               if e["kernel"] == "A")
    assert drv.load_pair_list(out / "winners.json") == {"512x512x512": ["B"]}
    meta = strict(out / "run.json")
    assert (meta["sampling_policy"], meta["exhaustive"]) == ("adaptive", False)
    assert meta["ck_search"]["origin_sha"] == ck_search.ORIGIN_SHA
    assert meta["context"]["arch"] == "gfx950"
    assert meta["problems"]["512x512x512"]["eligible"] == 4
    summary = strict(out / "summary.json")
    assert summary["counts"]["failed_by_status"] == {"invalid_measurement": 1}
    assert summary["winners"]["512x512x512"]["calls_ms"] == [0.5, 0.5, 0.5]


def test_output_dir_holding_a_run_is_refused(tmp_path):
    run(tmp_path, FakeAdapter({"P1": ABCD}), {"P1": [["A"]]})
    with pytest.raises(AS.SettingsError, match="already holds a run"):
        run(tmp_path, FakeAdapter({"P1": ABCD}), {"P1": [["A"]]})


@pytest.mark.parametrize("out", ["out", "out/sub"])
def test_output_dir_that_is_a_file_is_refused(tmp_path, out):
    (tmp_path / "out").write_text("")
    with pytest.raises(AS.SettingsError, match="not a directory"):
        AS.RunWriter(tmp_path / out).check_fresh()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
@pytest.mark.parametrize("out", ["ro", "ro/run", "ro/run/sub"])
def test_output_dir_without_write_access_is_refused(tmp_path, out):
    (tmp_path / "ro").mkdir(mode=0o555)
    with pytest.raises(AS.SettingsError, match="ro is not writable"):
        AS.RunWriter(tmp_path / out).check_fresh()


class _FullDisk:
    """A trials.jsonl handle on a full disk."""

    def write(self, text):
        raise OSError(28, "No space left on device")

    def flush(self):
        pass

    def close(self):
        raise OSError(28, "No space left on device")


def test_output_dir_failing_mid_run_ends_with_a_summary_not_a_traceback(
        tmp_path, capsys):
    # The first failed write ends the search as an error; recording that error
    # and the confirmation are skipped instead of failing again, the final
    # files are still tried (here only trials.jsonl fails).
    ad = FakeAdapter({"P1": ABCD})
    co = AS.SearchCoordinator(ad, settings(), tmp_path / "out", clock=FakeClock(),
                              strategy_factory=scripted({"P1": [["A"], ["B"]]})[0])
    real = ad.measure_pairs

    def measure(requests):
        co.writer._fh = _FullDisk()
        return real(requests)

    ad.measure_pairs = measure
    s = co.run()
    assert (s.stop_reason, s.status, s.exit_code) == ("error", "no_winner", 1)
    assert "No space left on device" in s.error and "cannot write" in s.error
    assert ad.confirmed == [] and len(ad.measured) == 1
    summary = strict(tmp_path / "out" / "summary.json")
    assert (summary["status"], summary["exit_code"], summary["error"]) == (
        "no_winner", 1, s.error)
    assert "cannot write" in capsys.readouterr().err


@pytest.mark.parametrize("fail_on", ["winners.json", "stop", "summary.json"])
def test_failed_final_write_exits_1_with_what_can_be_written(
        tmp_path, capsys, monkeypatch, fail_on):
    # The first failed write is one of the last three: the run is not
    # complete, and each final file after it is still written and says why.
    # The stop event (trials.jsonl) is skipped after a failed write, like every
    # event, and cannot know of a later one.
    def failing(real):
        def write(self, name, data):
            if name == fail_on:
                raise OSError(28, "No space left on device")
            return real(self, name, data)
        return write

    for attr in ("_event", "_write_json"):
        monkeypatch.setattr(AS.RunWriter, attr, failing(getattr(AS.RunWriter, attr)))
    out = tmp_path / "out"
    rc = AS.run_search(FakeAdapter({"P1": ABCD}), settings(), out, ["0"])
    cap = capsys.readouterr()
    assert rc == 1 and "SEARCH PARTIAL: 1/1" in cap.out and "COMPLETE" not in cap.out
    assert "No space left on device" in cap.err
    written = {p.name for p in out.iterdir()}
    assert ("winners.json" in written) == (fail_on != "winners.json")
    assert ("summary.json" in written) == (fail_on != "summary.json")
    if fail_on != "summary.json":
        summary = strict(out / "summary.json")
        assert (summary["status"], summary["exit_code"]) == ("partial", 1)
        assert "No space left on device" in summary["error"]
    stops = [e["status"] for e in events(out, "stop")]
    assert stops == {"winners.json": [], "stop": [],
                     "summary.json": ["complete"]}[fail_on]


class Interrupting(FakeAdapter):
    """Ctrl-C during round 2's build, or during the first confirmation call."""

    def __init__(self, problems, when):
        super().__init__(problems)
        self.when = when

    def build_missing(self, cands):
        if self.when == "search" and self.builds:
            raise KeyboardInterrupt
        return super().build_missing(cands)

    def confirm(self, requests):
        if self.when == "confirmation":
            raise KeyboardInterrupt
        return super().confirm(requests)


@pytest.mark.parametrize("when", ["search", "confirmation"])
def test_interrupt_writes_the_summary_and_exits_130(tmp_path, when):
    ad = Interrupting({"P1": ABCD, "P2": ABCD}, when)
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]], "P2": [["C"], ["D"]]})
    assert (s.stop_reason, s.exit_code, s.winners) == ("interrupted", 130, {})
    assert f"interrupted during the {when}" in s.error
    out = tmp_path / "out"
    summary = strict(out / "summary.json")
    assert (summary["status"], summary["exit_code"]) == ("no_winner", 130)
    assert strict(out / "winners.json") == {}
    # The discovery results are kept, unconfirmed, for a rerun or a --pairs list.
    assert {k: p["discovery_best"]["kernel"] for k, p in summary["problems"].items()
            } == {"P1": "A", "P2": "C"}
    assert events(out, "stop")[-1]["stop_reason"] == "interrupted"
    assert ad.confirmed == [] or when == "confirmation"


@pytest.mark.parametrize("sigint", [signal.default_int_handler, signal.SIG_IGN],
                         ids=["sigint_default", "sigint_ignored"])
def test_run_search_turns_sigterm_into_an_interrupt(tmp_path, capsys, sigint):
    seen = []

    class Terminated(FakeAdapter):
        def measure_pairs(self, requests):
            seen.append([signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)])
            os.kill(os.getpid(), signal.SIGTERM)  # a batch system's time limit
            return super().measure_pairs(requests)

    old = signal.signal(signal.SIGINT, sigint)
    try:
        before = [signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)]
        rc = AS.run_search(Terminated({"P1": ABCD}), settings(), tmp_path / "out", ["0"])
        after = [signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)]
    finally:
        signal.signal(signal.SIGINT, old)
    assert rc == 130 and after == before
    assert strict(tmp_path / "out" / "summary.json")["stop_reason"] == "interrupted"
    assert "SEARCH NO_WINNER" in capsys.readouterr().out
    # Ctrl-C takes the same handler, unless SIGINT is ignored (a background job).
    (term, intr), = seen
    assert intr is (term if sigint is signal.default_int_handler else signal.SIG_IGN)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_interrupt_handler_ends_a_forked_child(sig):
    # A forked build worker inherits the handler: there the signal ends the
    # process (as by default) instead of raising in the job it runs.
    handler = AS._interrupt_handler(os.getpid())
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        try:
            handler(sig, None)
        finally:
            os._exit(3)  # reached only if the handler returned or raised
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == sig
    with pytest.raises(KeyboardInterrupt):
        handler(sig, None)


_POOL_RUN = """
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, sys.argv[1])
from test_adaptive_search_coordinator import ABCD, AS, FakeAdapter, settings

def build(marker):  # one build: say it started, then compile for a while
    open(marker, "w").close()
    time.sleep(60)

class Pool(FakeAdapter):  # builds like setup_multiple_gemm_dispatchers
    def build_missing(self, cands):
        with ProcessPoolExecutor(1) as ex:  # one worker: the second build waits
            futs = [ex.submit(build, sys.argv[2] + "/started-" + c.canonical_key)
                    for c in cands]
            for fut in as_completed(futs):
                fut.result()
        return super().build_missing(cands)

sys.exit(AS.run_search(Pool({"P1": ABCD}), settings(), sys.argv[2] + "/out", ["0"]))
"""


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups")
def test_sigterm_to_the_process_group_does_not_wait_for_queued_builds(tmp_path):
    # A batch system's time limit sends SIGTERM to every process of the job.
    # The build worker dies, the pool breaks and the summary is written at
    # once; a worker that turned it into one job's error would start the
    # queued build (a second marker) and the summary would wait for it.
    proc = subprocess.Popen([sys.executable, "-c", _POOL_RUN, str(Path(__file__).parent),
                             str(tmp_path)], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(600):
            if list(tmp_path.glob("started-*")) or proc.poll() is not None:
                break
            time.sleep(0.05)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        rc = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert rc == 130, proc.stderr.read().decode()
    assert len(list(tmp_path.glob("started-*"))) == 1
    assert strict(tmp_path / "out" / "summary.json")["stop_reason"] == "interrupted"


def _ended_by_error(tmp_path, s, message):
    """The run stopped on an error and still wrote every output file."""
    out = tmp_path / "out"
    assert (s.stop_reason, s.exit_code) == ("error", 1) and message in s.error
    assert sorted(p.name for p in out.iterdir()) == sorted(AS.OUTPUT_FILES)
    summary = strict(out / "summary.json")
    assert (summary["error"], summary["exit_code"]) == (s.error, 1)
    (err,) = events(out, "error")
    assert message in err["message"] and "Traceback" in err["traceback"]
    assert events(out, "stop")[-1]["stop_reason"] == "error"


class Broken(FakeAdapter):
    """Correct in round 1, then breaks its contract as ``how`` says."""

    def __init__(self, problems, how):
        super().__init__(problems)
        self.how, self.calls = how, 0

    def build_missing(self, cands):
        out = super().build_missing(cands)
        if self.how == "build_keys" and len(self.builds) > 1:
            out[cands[0].canonical_key + "x"] = out.pop(cands[0].canonical_key)
        return out

    def measure_pairs(self, requests):
        out = super().measure_pairs(requests)
        self.calls += 1
        if self.calls == 1:
            return out
        first = out[0]
        if self.how == "lost":
            return out[1:]
        if self.how == "repeated":
            return out + [first]
        if self.how == "unrequested":
            return [replace(first, request=replace(first.request, call_id="z9"))]
        if self.how == "status":
            return [replace(first, status="skipped")]
        return out


@pytest.mark.parametrize("how, message", [
    ("lost", "no outcome"),
    ("repeated", "unrequested or repeated"),
    ("unrequested", "unrequested or repeated"),
    ("status", "expected a PairOutcome"),
    ("build_keys", "one BuildResult per requested key"),
])
def test_adapter_contract_violation_ends_the_run_unconfirmed(tmp_path, how, message):
    # A in round 1 was fine, but an adapter that broke its contract is not
    # trusted with a confirmation either.
    ad = Broken({"P1": ABCD}, how)
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]]})
    _ended_by_error(tmp_path, s, message)
    assert pairs(ad.measured)[0] == ("P1", "A") and ad.confirmed == []
    assert (s.status, s.unresolved) == ("no_winner", {"P1": "search ended: error"})


class DiesInRound2(FakeAdapter):
    def build_missing(self, cands):
        if self.builds:
            raise RuntimeError("BrokenProcessPool: a build worker died")
        return super().build_missing(cands)


def test_adapter_failure_still_confirms_what_was_measured(tmp_path):
    ad = DiesInRound2({"P1": ABCD, "P2": ABCD, "P3": ABCD}, fail_build={"D"})
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]], "P2": [["C"], ["B"]],
                                 "P3": [["D"], ["B"]]})
    _ended_by_error(tmp_path, s, "build_missing failed: RuntimeError: BrokenProcessPool")
    assert s.counts["confirm_calls"] == 6 and s.winners == {"P1": "A", "P2": "C"}
    assert (s.status, s.exit_code) == ("partial", 1)  # never complete after an error
    assert strict(tmp_path / "out" / "winners.json") == {"P1": ["A"], "P2": ["C"]}
    assert s.unresolved["P3"].startswith("no verified discovery measurement")


def test_a_run_with_an_error_is_never_complete(tmp_path):
    ad = DiesInRound2({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {"P1": [["A"], ["B"]]})
    assert s.winners == {"P1": "A"} and (s.status, s.exit_code) == ("partial", 1)


def test_confirmation_failure_is_recorded(tmp_path):
    ad = FakeAdapter({"P1": ABCD})

    def boom(requests):
        raise OSError("device lost")

    ad.confirm = boom
    s, _, _ = run(tmp_path, ad, {"P1": [["A"]]})
    assert (s.stop_reason, s.status, s.exit_code) == ("searched", "no_winner", 1)
    assert s.error == "confirmation: confirm failed: OSError: device lost"
    assert s.unresolved == {"P1": s.error}
    assert strict(tmp_path / "out" / "summary.json")["error"] == s.error


def test_strategy_error_still_writes_the_summary(tmp_path):
    class Singular(Scripted):
        def ask(self, n):
            if self.told:
                raise FloatingPointError("singular kernel matrix")
            return super().ask(n)

    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {},
                  factory=lambda name, space, *a: Singular(space, [["A"], ["B"]]))
    _ended_by_error(tmp_path, s, "FloatingPointError: singular kernel matrix")
    assert s.winners == {"P1": "A"} and s.status == "partial"


def test_strategy_asking_more_than_n_ends_the_run(tmp_path):
    class Greedy(Scripted):
        def ask(self, n):
            return super().ask(n + 1)

    ad = FakeAdapter({"P1": ABCD})
    s, _, _ = run(tmp_path, ad, {},
                  factory=lambda name, space, *a: Greedy(space, [["A", "B", "C"]]))
    _ended_by_error(tmp_path, s, "RuntimeError: P1: strategy asked 3 > 2 trials")
    assert ad.measured == [] and s.status == "no_winner"


def test_adapter_failure_at_setup_still_writes_the_summary(tmp_path):
    ad = FakeAdapter({"P1": ABCD})

    def boom():
        raise OSError("disk full")

    ad.enumerate_eligible = boom
    s, _, _ = run(tmp_path, ad, {"P1": [["A"]]})
    _ended_by_error(tmp_path, s, "enumerate_eligible failed: OSError: disk full")
    assert s.status == "no_winner" and not s.unresolved
    assert strict(tmp_path / "out" / "run.json")["error"] == s.error


# ------------------------------------------------------------- settings


BASE = {"schema_version": 1, "strategy": "tpe", "trials_per_problem": 4,
        "new_builds_per_run": 8}


@pytest.mark.parametrize("change, match", [
    ({"schema_version": None}, "schema_version: required"),
    ({"schema_version": 2}, "expected 1"),
    ({"strategy": "annealing"}, "expected one of"),
    ({"trials_per_problem": 0}, ">= 1"),
    ({"trials_per_problem": True}, "expected type int"),
    ({"trials_per_problem": 4.0}, "expected type int"),
    ({"new_builds_per_run": 3}, ">= trials_per_problem"),
    ({"deadline_s": 0}, "> 0"),
    ({"deadline_s": math.inf}, "expected type float"),
    ({"deadline_s": math.nan}, "expected type float"),
    ({"deadline_s": "60"}, "expected type float"),
    ({"seed": 1.5}, "expected type int"),
    ({"seed": "7"}, "expected type int"),
    ({"seed_policy": "random"}, "one of"),
    ({"measurement_policy": "quick"}, "one of"),
    ({"batch_per_problem": 0}, ">= 1"),
    ({"max_stall_rounds": 0}, ">= 1"),
    ({"verbose": True}, "unknown key"),
    ({"confirm": {"finalists": 2, "calls": 3}}, "unknown key"),
    ({"confirm": {"fresh_calls": 0}}, ">= 1"),
    ({"confirm": {"finalists": 0}}, ">= 1"),
    ({"warm_start": {"top_k": 3}}, "pairs_file and/or ranked_csv"),
    ({"warm_start": {"pairs_file": "missing.json"}}, "no such file"),
    ({"warm_start": {"pairs_file": 7}}, "expected type str"),
    ({"warm_start": None}, "expected a JSON object"),
])
def test_settings_errors(change, match):
    d = {**BASE, **change}
    d = {k: v for k, v in d.items() if v is not None or k == "warm_start"}
    with pytest.raises(AS.SettingsError, match=match):
        AS.SearchSettings.from_json(d)


def test_settings_warm_rules_and_files(tmp_path):
    (tmp_path / "picks.json").write_text("{}")
    (tmp_path / "picks.csv").write_text("problem,rank,kernel,predicted_tflops\n")
    bad = [
        ({"strategy": "random", "warm_start": {"pairs_file": "picks.json"}}, "ignores"),
        ({"strategy": "de", "warm_start": {"ranked_csv": "picks.csv",
                                          "use_prior": True}}, "read a prior"),
        ({"warm_start": {"ranked_csv": "picks.csv"}}, "use_prior"),
        ({"warm_start": {"pairs_file": "picks.json", "use_prior": True}}, "use_prior"),
        ({"warm_start": {"pairs_file": "picks.json", "top_k": 0}}, ">= 1"),
        ({"warm_start": {"ranked_csv": "picks.csv", "use_prior": 1}}, "type bool"),
        ({"warm_start": {"ranked_csv": "picks.csv", "use_prior": "yes"}}, "type bool"),
    ]
    for change, match in bad:
        (tmp_path / "s.json").write_text(json.dumps({**BASE, **change}))
        with pytest.raises(AS.SettingsError, match=match):
            AS.SearchSettings.from_json(tmp_path / "s.json")
    for text, match in [("{", "invalid JSON"), ('{"seed": NaN}', "NaN"),
                        ('{"seed": 1, "seed": 2}', "duplicate")]:
        (tmp_path / "s.json").write_text(text)
        with pytest.raises(AS.SettingsError, match=match):
            AS.SearchSettings.from_json(str(tmp_path / "s.json"))


def test_settings_defaults_paths_and_round_trip(tmp_path):
    (tmp_path / "picks.json").write_text("{}")
    cfg = {**BASE, "warm_start": {"pairs_file": "picks.json"}}
    (tmp_path / "search.json").write_text(json.dumps(cfg))
    s = AS.SearchSettings.from_json(tmp_path / "search.json")
    assert s.warm_start.pairs_file == str(tmp_path / "picks.json")
    assert (s.seed, s.seed_policy, s.deadline_s, s.batch_per_problem) == (
        0, "shared", None, 4)
    assert "deadline_s" not in s.to_json()
    assert (s.max_stall_rounds, s.confirm.finalists, s.confirm.fresh_calls) == (8, 2, 3)
    assert s.warm_start.top_k == 5 and not s.warm_start.use_prior
    assert AS.SearchSettings.from_json(s.to_json()) == s
    assert AS.SearchSettings.from_json(BASE).to_json()["confirm"] == {
        "finalists": 2, "fresh_calls": 3}


# ------------------------------------------------- seeds, warm, prior


@pytest.mark.parametrize("policy, seeds", [("shared", [7, 7, 7]),
                                           ("per_problem", [7, 8, 9])])
def test_seed_policies(tmp_path, policy, seeds):
    calls = []
    ad = FakeAdapter({"P1": ABCD, "P2": ABCD, "P3": ABCD})
    run(tmp_path, ad, {}, calls=calls, seed=7, seed_policy=policy)
    assert [c["seed"] for c in calls] == seeds
    meta = strict(tmp_path / "out" / "run.json")
    assert [meta["problems"][p]["seed"] for p in ("P1", "P2", "P3")] == seeds


def test_warm_start_order_and_pair_loader(tmp_path):
    sys.path.insert(0, str(TE_OPS / "gemm"))
    sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
    import gemm_full_benchmark as drv

    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"P1": ["D", "Z", "B", "A"], "P2": "*", "P9": ["A"]}))
    warm = {"pairs_file": str(picks), "top_k": 2}
    ad = FakeAdapter({"P1": ABCD, "P2": ["C", "A", "B", "D"], "P3": ABCD})
    with pytest.raises(AS.SettingsError, match="pair_loader"):
        run(tmp_path, ad, {}, strategy="hill", warm_start=warm)
    calls = []
    run(tmp_path, ad, {}, calls=calls, strategy="hill", warm_start=warm,
        pair_loader=drv.load_pair_list)
    assert [c["warm"] for c in calls] == [
        [rec("D"), rec("B")], [rec("C"), rec("A")], []]
    meta = strict(tmp_path / "out" / "run.json")
    assert [meta["problems"][p]["warm_start"] for p in ("P1", "P2", "P3")] == [
        ["D", "B"], ["C", "A"], []]


def _ranked_settings(tmp_path, strategy="tpe"):
    """use_prior settings; the op's prior_loader reads the (placeholder) file."""
    path = tmp_path / "ranked.csv"
    path.write_text("read by the op's prior_loader\n")
    return {"strategy": strategy,
            "warm_start": {"ranked_csv": str(path), "use_prior": True}}


def test_prior_from_the_ops_prior_loader(tmp_path):
    # The op reads the file (GEMM: gemm_search.read_ranked_csv); the coordinator
    # keys the predicted latencies by config and makes a log-scale TablePrior.
    calls, loaded = [], []

    def prior_loader(path):
        loaded.append(path)
        return {"P1": {"B": 0.5, "A": 1.0, "Z": 0.1}}

    ad = FakeAdapter({"P1": ABCD, "P2": ["A", "B"]})
    with pytest.raises(AS.SettingsError, match="needs a prior_loader"):
        run(tmp_path, ad, {}, **_ranked_settings(tmp_path))
    run(tmp_path, ad, {}, calls=calls, prior_loader=prior_loader,
        **_ranked_settings(tmp_path))
    assert loaded == [str(tmp_path / "ranked.csv")]
    prior = calls[0]["prior"]
    got = prior.predict([rec("A"), rec("B"), rec("C"), rec("D")])
    assert got[:2] == pytest.approx([1.0, 0.5])
    assert math.isnan(got[2]) and math.isnan(got[3])
    assert prior.scale == "log" and prior.sign == 1.0  # lower is better
    assert calls[1]["name"] == "exhaustive" and calls[1]["prior"] is None
    meta = strict(tmp_path / "out" / "run.json")
    assert meta["problems"]["P1"]["prior_entries"] == 2


@pytest.mark.parametrize("error", [ValueError("line 3: bad number"),
                                   UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
                                   OSError("permission denied")])
def test_unreadable_warm_start_file_is_a_settings_error(tmp_path, error):
    def prior_loader(path):
        raise error

    with pytest.raises(AS.SettingsError, match="warm_start.ranked_csv: "):
        run(tmp_path, FakeAdapter({"P1": ABCD}), {}, prior_loader=prior_loader,
            **_ranked_settings(tmp_path))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["hill", "tpe", "gp", "de", "portfolio"])
def test_warm_starts_reach_the_real_strategies(tmp_path, name):
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"64x512x512": ["k17", "k03", "k05"]}))
    ad = GridAdapter(["64x512x512"])
    co = AS.SearchCoordinator(
        ad, settings(strategy=name, trials_per_problem=10, new_builds_per_run=24,
                     batch_per_problem=1, warm_start={"pairs_file": str(picks),
                                                      "top_k": 2}),
        tmp_path / "out", clock=FakeClock(), pair_loader=json_pairs,
    )
    co.run()
    assert [r.kernel_name for r in ad.measured[:2]] == ["k17", "k03"]
    assert co.ledger.trials["64x512x512"] == len(ad.measured) <= 10


def test_every_strategy_shares_the_runs_noise_model(tmp_path):
    ad = GridAdapter(["64x512x512", "256x512x512"])
    co = AS.SearchCoordinator(
        ad, settings(strategy="portfolio", trials_per_problem=6, new_builds_per_run=24),
        tmp_path / "out", clock=FakeClock())
    co.run()
    assert isinstance(co.noise, ck_search.NoiseModel)
    for p in co.problems:
        assert p.strategy.noise is co.noise
        assert all(m.noise is co.noise for m in p.strategy.members)
    meta = strict(tmp_path / "out" / "run.json")
    assert meta["ck_search"]["noise_model"]["floor"] == co.noise.floor


def test_prior_reaches_the_real_strategy(tmp_path, monkeypatch):
    made, real = [], ck_search.make_strategy

    def spy(name, space, seed=0, **kw):
        made.append((name, kw))
        return real(name, space, seed, **kw)

    monkeypatch.setattr(ck_search, "make_strategy", spy)
    ad = GridAdapter(["64x512x512"])
    co = AS.SearchCoordinator(
        ad, settings(trials_per_problem=10, new_builds_per_run=24,
                     **_ranked_settings(tmp_path)),
        tmp_path / "out", clock=FakeClock(),
        prior_loader=lambda path: {"64x512x512": {"k03": 1.0, "k17": 2.0}},
    )
    assert co.run().status == "complete"
    ((name, kw),) = made
    assert name == "tpe" and isinstance(kw["prior"], ck_search.TablePrior)
    got = kw["prior"].predict([ad.grid["k03"], ad.grid["k17"], ad.grid["k00"]])
    assert got[:2] == pytest.approx([1.0, 2.0]) and math.isnan(got[2])


def test_exhaustive_fallback_for_tiny_spaces(tmp_path):
    lat = {("P1", "A"): 2.0, ("P1", "B"): 1.0}
    ad = FakeAdapter({"P1": ["A", "B"], "P2": list("ABCDEF")}, lat=lat)
    co = AS.SearchCoordinator(ad, settings(trials_per_problem=3, batch_per_problem=3),
                              tmp_path / "out", clock=FakeClock())
    s = co.run()
    meta = strict(tmp_path / "out" / "run.json")
    assert meta["problems"]["P1"]["strategy"] == "exhaustive"
    assert "exhaustive" in meta["problems"]["P1"]["note"]
    assert meta["problems"]["P2"]["strategy"] == "random"
    assert sorted(r.kernel_name for r in ad.measured if r.problem_key == "P1") == [
        "A", "B"]
    assert s.winners["P1"] == "B" and s.status == "complete"


# -------------------------------------------------- real strategies


def _grid():
    out = {}
    for i, (tm, tn, k) in enumerate((tm, tn, k) for tm in (64, 128, 192, 256)
                                    for tn in (64, 128, 256) for k in (32, 64)):
        out[f"k{i:02d}"] = {"tile_m": tm, "tile_n": tn, "tile_k": k}
    return out


class GridAdapter(FakeAdapter):
    """24 kernels on a smooth landscape; each problem prefers another corner."""

    def __init__(self, problems):
        super().__init__(problems)
        self.grid = _grid()

    def enumerate_eligible(self):
        return {p: [AS.Candidate(n, n, r, None) for n, r in self.grid.items()]
                for p in self.problems}

    def _lat(self, r):
        g, m = self.grid[r.kernel_name], int(r.problem_key.split("x")[0])
        return 1.0 + abs(g["tile_m"] - m) / 64 + g["tile_n"] / 256 + g["tile_k"] / 64


@pytest.mark.parametrize("name", ck_search.STRATEGY_NAMES)
def test_real_strategies_end_to_end(tmp_path, name):
    ad = GridAdapter(["64x512x512", "256x512x512"])
    cfg = {"strategy": name, "trials_per_problem": 10, "new_builds_per_run": 24,
           "batch_per_problem": 3}
    co = AS.SearchCoordinator(ad, settings(**cfg), tmp_path / "out", clock=FakeClock())
    s = co.run()
    assert s.status == "complete" and s.exit_code == 0
    built = [k for b in ad.builds for k in b]
    assert len(built) == len(set(built)) == co.ledger.builds <= 24
    seen = set()
    for r in ad.measured:
        assert (r.problem_key, r.kernel_name) not in seen  # no duplicate samples
        seen.add((r.problem_key, r.kernel_name))
    for p in ad.problems:
        assert co.ledger.trials[p] == sum(r.problem_key == p for r in ad.measured)
        assert co.ledger.trials[p] <= 10
    assert len(events(tmp_path / "out", "result")) == len(ad.measured)
