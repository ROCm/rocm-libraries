# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Config-list spaces (member index) and offline replay of recorded sweeps."""

from __future__ import annotations

import itertools
import math
import random

import pytest
from rocke.dispatch.tuning.search import (
    FAMILY_DEFAULTS,
    SearchSpace,
    SyntheticBackend,
    TableBackend,
    recommend,
    search_configs,
    tune,
)
from rocke.dispatch.tuning.search.__main__ import main as cli_main
from rocke.dispatch.tuning.search.params import config_items_key
from rocke.dispatch.tuning.search.replay import (
    ReplayProblem,
    load_table,
    replay_problem,
)
from rocke.dispatch.tuning.search.surrogate import candidate_pool
from rocke.dispatch.tuning.search.trial import Budget, Fidelity


def _sparse_rows():
    """A list that covers 1/8 of its field product (as recorded grids do)."""
    rows = []
    for t, w, p, s in itertools.product(
        (16, 32, 64, 128), (1, 2, 4, 8), "abcd", (1, 2)
    ):
        if (3 * (t // 16) + 5 * w + 7 * ord(p) + 11 * s) % 8 == 0:
            rows.append({"tile": t, "warp": w, "pipe": p, "split": s, "fixed": 1})
    return rows


def _brute_nearest(sp, g):
    best = None
    for gg in sp.listed():
        d = sp.distance(g, gg)
        if best is None or d < best:
            best = d
    return best


def test_member_index_repair_is_exact_nearest_and_sampler_hits_members():
    rows = _sparse_rows()
    sp = SearchSpace.from_configs(rows)
    assert len(sp.listed()) == len(rows)
    assert {config_items_key(sp.decode(g)) for g in sp.listed()} == {
        config_items_key(r) for r in rows
    }
    rng = random.Random(0)
    for _ in range(200):
        g = sp.random_genome(rng)
        r = sp.repair(g, rng)
        assert r is not None and sp.valid(r)
        assert sp.distance(g, r) == pytest.approx(_brute_nearest(sp, g))
    for _ in range(50):
        g = sp.random_valid(rng)
        assert g is not None and sp.valid(g)


def test_member_index_ties_are_broken_by_rng():
    # 3 members in a product of 36: sparse, so the nearest member is tried first
    rows = [
        {"a": 1, "b": "x", "c": 0, "d": 0},
        {"a": 3, "b": "x", "c": 0, "d": 0},
        {"a": 2, "b": "y", "c": 1, "d": 2},
    ]
    sp = SearchSpace.from_configs(rows)
    g = sp.encode({"a": 2, "b": "x", "c": 0, "d": 0})  # a=1 and a=3 tie
    seen = {sp.key(sp.repair(g, random.Random(s))) for s in range(30)}
    assert len(seen) == 2


def test_dense_list_repairs_by_single_moves_first():
    # 3 members in a product of 6: dense, single moves (nearest first) win
    rows = [{"a": 1, "b": "x"}, {"a": 3, "b": "x"}, {"a": 2, "b": "y"}]
    sp = SearchSpace.from_configs(rows)
    g = sp.encode({"a": 2, "b": "x"})
    seen = {sp.key(sp.repair(g, random.Random(s))) for s in range(10)}
    # a=1 and a=3 are equally near (ties broken at random), the categorical
    # move b=y is farther
    assert seen == {sp.key(sp.encode(rows[0])), sp.key(sp.encode(rows[1]))}


def test_member_index_off_keeps_generic_repair():
    rows = _sparse_rows()
    sp = SearchSpace.from_configs(rows, member_index=False)
    assert sp.listed() is None
    rng = random.Random(1)
    g = sp.repair(sp.random_genome(rng), rng)
    assert g is None or sp.valid(g)


def test_candidate_pool_uses_listed_members():
    # 30 members in a product of 6^6: the pool is the member list itself
    rng = random.Random(5)
    rows, seen = [], set()
    while len(rows) < 30:
        r = {f"f{k}": rng.randrange(6) for k in range(6)}
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            rows.append(r)
    sp = SearchSpace.from_configs(rows)
    assert sp.size() > 10 * len(rows)
    pool = candidate_pool(sp, random.Random(0), len(rows))
    assert {sp.key(g) for g in pool} == {sp.key(g) for g in sp.listed()}
    assert len(candidate_pool(sp, random.Random(0), 8)) <= 8


def test_table_backend_call_noise_is_shared_by_reps_and_seeded():
    tb = TableBackend({"k": 100.0}, call_noise=0.1, seed=3)
    fid = Fidelity(reps=5)
    a = tb.measure({}, "k", "k", fid)
    assert len(set(a)) == 1 and a[0] != 100.0  # one factor per call, no per-rep noise
    b = tb.measure({}, "k", "k", fid)
    assert b[0] != a[0]
    tb2 = TableBackend({"k": 100.0}, call_noise=0.1, seed=3)
    assert tb2.measure({}, "k", "k", fid) == a
    tb3 = TableBackend({"k": 100.0}, noise=0.05, call_noise=0.1, seed=3)
    assert len(set(tb3.measure({}, "k", "k", fid))) == 5


def _problem(**kw):
    rows = _sparse_rows()
    vals, st = [], []
    for i, r in enumerate(rows):
        v = (
            10.0
            + abs(r["tile"] - 64) / 8
            + r["warp"]
            + (0.5 if r["pipe"] == "a" else 0.0)
            + i * 1e-3
        )
        vals.append(v)
        st.append("ok")
    # the fastest-looking row is only a lower bound, one row failed to compile
    vals[0], st[0] = 1.0, "early_stop"
    vals[1], st[1] = None, "compile_error"
    return rows, vals, st, ReplayProblem("toy", rows, vals, statuses=st, **kw)


def test_replay_problem_bounds_failures_and_oracle():
    rows, vals, st, rp = _problem()
    key0 = rp.space.key(rp.space.encode(rows[0]))
    key1 = rp.space.key(rp.space.encode(rows[1]))
    assert key0 in rp.bounds and rp.oracle_key != key0
    # a lower bound's true value is unknown: never credited, not even as
    # the oracle (it is below the oracle here: ambiguous)
    assert rp.ambiguous_bounds == 1 and math.isnan(rp.truth(key0))
    assert rp.failures == {key1: "compile_error"}
    exact = [v for v, s in zip(vals, st) if s == "ok"]
    assert rp.oracle == min(exact)
    assert len(rp) == len(rows)
    assert rp.truth(key1) == math.inf


def test_replay_failures_cost_an_evaluation():
    rows, _v, _s, rp = _problem()
    r = replay_problem(rp, "exhaustive", budget=len(rows), seed=0)
    assert r["failures_hit"] == 1
    assert r["evaluations"] == len(rows)
    assert r["best_seen"] == pytest.approx(1.0)
    assert (
        len(r["curve"])
        == len(r["incumbent_curve"])
        == len(r["cost"])
        == r["evaluations"]
    )
    assert r["curve"] == sorted(r["curve"])


def test_replay_incumbent_vs_ideal_under_noise_and_confirm():
    _rows, _v, _s, rp = _problem(noise=0.05, call_noise=0.02, seed=4)
    r = replay_problem(
        rp,
        "random",
        budget=24,
        seed=2,
        ladder=(Fidelity("r2", reps=2, cost=1.0),),
        confirm=3,
    )
    assert all(i <= c + 1e-12 for i, c in zip(r["incumbent_curve"], r["curve"]))
    # with heavy noise the strategy's own pick lags the best it measured
    _rows, _v, _s, loud = _problem(noise=0.0, call_noise=0.5, seed=4)
    lags = 0
    for seed in range(6):
        q = replay_problem(loud, "random", budget=16, seed=seed)
        lags += any(i < c - 1e-12 for i, c in zip(q["incumbent_curve"], q["curve"]))
    assert lags >= 1
    assert r["confirmed"]
    # the returned config's true value; unknown (None) when it is a bound
    assert r["final"] is None or 0.0 < r["final"] <= r["best_seen"] + 1e-12
    finals = [
        replay_problem(rp, "random", budget=24, seed=s, confirm=3)["final"]
        for s in range(6)
    ]
    assert all(f is None or 0.0 < f <= 1.0 + 1e-12 for f in finals)


def test_replay_never_credits_a_lower_bound_and_summary_reports_final():
    from rocke.dispatch.tuning.search.replay import replay_summary

    rows, _v, _s, rp = _problem()
    key0 = rp.space.key(rp.space.encode(rows[0]))  # the ambiguous bound
    r = replay_problem(rp, "exhaustive", budget=len(rows), seed=0)
    # the bound sits below the oracle; the ideal curve still tops at 1.0
    assert max(r["curve"]) == pytest.approx(1.0)
    runs = [
        dict(r, strategy="x", final=0.5, best_seen=1.0),
        dict(r, strategy="x", final=None, best_seen=1.0),
        dict(r, strategy="x", final=0.7, best_seen=0.9),
    ]
    row = replay_summary(runs)["x"]
    assert row["unknown"] == 1 and row["mean"] == pytest.approx(0.6)
    assert row["ideal_mean"] == pytest.approx(2.9 / 3)
    assert key0 in rp.bounds


def test_replay_cost_budget_and_members_option():
    _rows, _v, _s, rp = _problem()
    ladder = (Fidelity("r1", reps=1, cost=0.25), Fidelity("r4", reps=4, cost=1.0))
    r = replay_problem(rp, "hyperband", budget=Budget(cost=6.0), seed=1, ladder=ladder)
    assert r["cost"] and r["cost"][-1] <= 6.0 + 1.0
    r2 = replay_problem(rp, "portfolio", budget=16, seed=1, members=("hill", "de"))
    assert r2["evaluations"] <= 16 and r2["best_seen"] > 0


def test_replay_problem_rejects_bad_input():
    with pytest.raises(ValueError, match="differ in length"):
        ReplayProblem("x", [{"a": 1}], [1.0], statuses=["ok", "ok"])
    with pytest.raises(ValueError, match="no exact"):
        ReplayProblem(
            "x", [{"a": 1}, {"a": 2}], [1.0, 2.0], statuses=["early_stop"] * 2
        )


# ------------------------------------------------- per-family defaults
def test_recommend_enumerates_small_spaces_and_reads_family_table():
    assert recommend(None, 10, 32) == {"strategy": "exhaustive"}
    assert recommend("gemm", 10, 32) == {"strategy": "exhaustive"}
    big = recommend("gemm", 10_000, 64)
    assert big["strategy"] == "portfolio" and "dtype" in big["warm"]
    assert recommend("no-such-family", 10_000, 64) == dict(FAMILY_DEFAULTS["default"])
    assert recommend("permute", 140, 32)["strategy"] == "gp"
    big["strategy"] = "mutated"  # a copy, not the table
    assert FAMILY_DEFAULTS["gemm"]["strategy"] == "portfolio"


def test_tune_auto_resolves_by_space_size_and_family():
    rows = _sparse_rows()

    def obj(c):
        return 1.0 + c["tile"] / 64 + c["warp"] / 8

    sp = SearchSpace.from_configs(rows)
    small = tune(
        sp, SyntheticBackend(obj), strategy="auto", budget=Budget(evaluations=64)
    )
    assert small.strategy == "exhaustive"
    big = tune(sp, SyntheticBackend(obj), strategy="auto", budget=Budget(evaluations=8))
    assert big.strategy == "portfolio"
    fam = tune(
        sp, SyntheticBackend(obj), strategy="auto:permute", budget=Budget(evaluations=8)
    )
    assert fam.strategy == "gp"
    idx, res = search_configs(
        rows,
        measure=lambda c, fid: obj(c),
        check=lambda c: True,
        strategy="auto",
        budget=Budget(evaluations=64),
    )
    assert res.strategy == "exhaustive"
    assert rows[idx] == min(rows, key=obj)


def test_load_table_status_column_and_cli_replay(tmp_path, capsys):
    rows = _sparse_rows()
    path = tmp_path / "sweep.csv"
    lines = ["tile,warp,pipe,split,M,time_us,status"]
    for i, r in enumerate(rows):
        st = "ok"
        t = f"{10 + i}"
        if i == 0:
            st, t = "early_stop", "1"
        elif i == 1:
            st, t = "compile_fail", ""
        lines.append(f"{r['tile']},{r['warp']},{r['pipe']},{r['split']},64,{t},{st}")
    path.write_text("\n".join(lines) + "\n")
    cols = {
        "config_cols": ["tile", "warp", "pipe", "split"],
        "value_col": "time_us",
        "problem_cols": ["M"],
    }
    plain = load_table(path, **cols)
    configs, vals = plain["64"]
    assert len(configs) == len(rows) - 1  # the failed row has no time
    tagged = load_table(path, status_col="status", **cols)
    configs, vals, sts = tagged["64"]
    assert len(configs) == len(rows)
    assert sts[0] == "early_stop" and sts[1] == "compile_error" and vals[1] is None
    rp = ReplayProblem("64", configs, vals, statuses=sts)
    assert rp.oracle == 12.0 and len(rp.failures) == 1
    rc = cli_main(
        [
            "replay",
            "--csv",
            str(path),
            "--config-cols",
            "tile,warp,pipe,split",
            "--value-col",
            "time_us",
            "--problem-cols",
            "M",
            "--status-col",
            "status",
            "--strategies",
            "random,exhaustive",
            "--budgets",
            "16",
            "--seeds",
            "1",
            "--min-configs",
            "4",
            "--call-noise",
            "0.01",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"exhaustive"' in out and '"64"' in out
