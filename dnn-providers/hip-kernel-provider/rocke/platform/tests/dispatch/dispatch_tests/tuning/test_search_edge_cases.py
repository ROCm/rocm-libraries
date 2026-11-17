# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Edge-case tests of the search package: each test pins one exact boundary
or formula (comparison direction, off-by-one, constant, return value,
accumulation) of a strategy, the statistics, the evaluator, the tuning DB or
the driver, so that a change to it fails a test."""

from __future__ import annotations

import json
import math
import random

import pytest
from rocke.dispatch.tuning.axes import KnobAxis, values
from rocke.dispatch.tuning.search import (
    Budget,
    CompileOutcome,
    DifferentialEvolution,
    Evaluator,
    Exhaustive,
    Fidelity,
    HillClimb,
    Hyperband,
    Observation,
    Param,
    Portfolio,
    Race,
    SearchSpace,
    Status,
    SuccessiveHalving,
    SyntheticBackend,
    TableBackend,
    TunedRecord,
    TuningDB,
    WarmStart,
    default_ladder,
    make_strategy,
    problem_key,
    recommend,
    record_result,
    records_from_rows,
    run_search,
    tune,
)
from rocke.dispatch.tuning.search.params import config_items_key, params_from_axes
from rocke.dispatch.tuning.search.replay import load_table, replay_summary
from rocke.dispatch.tuning.search.stats import (
    NoiseModel,
    paired_t_greater,
    student_t_sf,
    welch_greater,
    wilcoxon_greater,
)
from rocke.dispatch.tuning.search.surrogate import GPEI, TPE, CallablePrior
from rocke.dispatch.tuning.search.synthetic import gemm_like, sphere
from rocke.dispatch.tuning.search.trial import BudgetMeter, Trial


# ------------------------------------------------------------------ stats
def _wilcoxon_exact(d):
    """Brute-force signed-rank p-value (all sign flips), for small n."""
    import itertools

    a = sorted(range(len(d)), key=lambda i: abs(d[i]))
    ranks = [0.0] * len(d)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and abs(d[a[j + 1]]) == abs(d[a[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[a[k]] = 0.5 * (i + j) + 1
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    hits = 0
    for signs in itertools.product((0, 1), repeat=len(d)):
        hits += sum(r for r, s in zip(ranks, signs) if s) >= w - 1e-9
    return hits / 2 ** len(d)


def test_wilcoxon_ties_match_brute_force():
    for d in ([1, 1, 2], [1, 2, 2, -2, 3], [0.5, -0.5, 1, 1, 1, 2]):
        assert wilcoxon_greater(d) == pytest.approx(_wilcoxon_exact(d))


def test_wilcoxon_normal_approximation_is_close_to_exact():
    rng = random.Random(3)
    d = [rng.gauss(0.15, 1.0) for _ in range(61)]  # above the exact limit
    approx = wilcoxon_greater(d)
    # exact distribution of the same ranks by DP (integer ranks, no ties)
    ranks = sorted(range(61), key=lambda i: abs(d[i]))
    r = [0] * 61
    for k, i in enumerate(ranks):
        r[i] = k + 1
    w = sum(rk for rk, x in zip(r, d) if x > 0)
    counts = [0] * (61 * 62 // 2 + 1)
    counts[0] = 1
    for x in range(1, 62):
        for s in range(len(counts) - 1, x - 1, -1):
            counts[s] += counts[s - x]
    exact = sum(counts[w:]) / 2**61
    assert approx == pytest.approx(exact, abs=0.004)
    # tied ranks in the approximation path (variance correction)
    tied = [1.0] * 30 + [-1.0] * 10 + [2.0] * 25
    assert 0.0 < wilcoxon_greater(tied) < 1e-4


def test_student_t_both_incomplete_beta_branches():
    # one-sided tail values from t tables
    assert student_t_sf(0.5, 3) == pytest.approx(0.3257, abs=2e-4)
    assert student_t_sf(10.0, 2) == pytest.approx(0.004926, abs=2e-5)
    assert student_t_sf(1.0, 30) == pytest.approx(0.1627, abs=2e-4)


def test_welch_needs_two_samples_on_each_side():
    assert welch_greater([1.0, 2.0, 3.0], [1.0]) == 1.0
    assert welch_greater([1.0], [1.0, 2.0, 3.0]) == 1.0


def test_paired_t_reference_values():
    # two diffs (df = 1): mean 1, sd sqrt(2), t = 1 -> p = 0.25
    assert paired_t_greater([0.0, 2.0]) == pytest.approx(0.25, abs=1e-6)
    # shift moves the null: mean 2 - shift 1, sd sqrt(2/3) over 4, df 3
    assert paired_t_greater([1.0, 2.0, 2.0, 3.0], shift=1.0) == pytest.approx(
        student_t_sf(1.0 / (math.sqrt(2 / 3) / 2), 3)
    )


def test_noise_model_edges():
    m = NoiseModel(floor=0.01)
    assert m.median_sigma(1) == pytest.approx(0.01 * math.sqrt(1 + math.pi / 2))
    assert m.median_sigma(0) == m.median_sigma(1)
    assert m.indistinct(2.0, 2.02) and not m.indistinct(2.0, 2.2)
    assert not m.indistinct(-1.0, 1.0) and not m.indistinct(1.0, 0.0)
    for _ in range(600):
        m.update([1.0, 1.01, 0.99])
    assert len(m.to_json()["spreads"]) == 512


def test_race_alpha_bounds():
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha"):
            Race(alpha=bad)


# ------------------------------------------------------------------ trial
def test_budget_bounds_and_meter_accounting():
    with pytest.raises(ValueError, match="evaluations"):
        Budget(evaluations=0)
    Budget(evaluations=1)
    with pytest.raises(ValueError, match="cost"):
        Budget(cost=0.0)
    meter = BudgetMeter(Budget(evaluations=10))
    t = Trial(genome=(0,), uid="u")
    fail = Observation(trial=t, key="k", status=Status.COMPILE_ERROR, compile_s=2.0)
    ok = Observation(trial=t, key="k2", status=Status.OK, value=1.0, compile_s=1.5)
    meter.charge(fail, 1.0)
    assert meter.cost == 0.0 and meter.evaluations == 1  # nothing was measured
    meter.charge(ok, 1.0)
    assert meter.cost == 1.0 and meter.compile_s == pytest.approx(3.5)
    again = BudgetMeter(Budget(evaluations=10))
    again.load(meter.to_json())
    assert again.compile_s == pytest.approx(3.5)


def test_fidelity_json_round_trip():
    f = Fidelity("p", reps=3, warmup=1, cold=True, proxy="small", cost=0.25)
    assert Fidelity.from_json(json.loads(json.dumps(f.to_json()))) == f


# --------------------------------------------------------------- strategy
def test_seen_and_uids():
    space, _f = sphere(levels=5, dims=2)
    s = make_strategy("random", space)
    refused = SearchSpace(
        (Param.ordinal("a", (0, 1, 2)),), is_valid=lambda c: c["a"] != 1
    )
    r = make_strategy("random", refused)
    assert r.seen((1,)) is True  # refused configs count as seen
    t = s.trial((1, 1))
    assert s.seen((1, 1)) is True and s.seen((2, 2)) is False
    uids = {t.uid} | {s.trial((x, 0)).uid for x in range(5)}
    assert len(uids) == 6


def test_exhaustive_resumes_where_it_stopped():
    space, _f = sphere(levels=4, dims=2)
    ex = Exhaustive(space)
    first = [t.genome for t in ex.ask(5)]
    again = Exhaustive(space)
    again.load_state_dict(json.loads(json.dumps(ex.state_dict())))
    more = [t.genome for t in again.ask(5)]
    full = Exhaustive(space)
    assert first + more == [t.genome for t in full.ask(10)]
    assert ex._pos == 5


# ----------------------------------------------------------------- driver
def test_run_search_refuses_mismatched_spaces():
    space, f = sphere(levels=4, dims=2)
    other, _g = sphere(levels=4, dims=2)
    ev = Evaluator(space, SyntheticBackend(f))
    with pytest.raises(ValueError, match="space"):
        run_search(space, make_strategy("random", other), ev, Budget(evaluations=3))
    with pytest.raises(ValueError, match="space"):
        run_search(
            space,
            make_strategy("random", space),
            Evaluator(other, SyntheticBackend(f)),
            Budget(evaluations=3),
        )


@pytest.mark.parametrize("batch", [1, 5])
def test_confirmation_reserve_is_exact(batch):
    space, f = sphere(levels=9, dims=3)
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=40),
        confirm=3,
        confirm_rounds=1,
        batch=batch,
        baseline=(0, 0, 0),
    )
    assert res.stopped == "evaluations"
    assert res.evaluations == 40  # search 36 + (3 finalists + baseline) x 1 round
    assert set(res.confirmed["rounds"].values()) == {1}
    fresh = sum(1 for o in res.observations if not o.cached and o.status != "invalid")
    assert fresh == 36


def test_origins_count_asks_and_fresh_evaluations():
    space, f = sphere(levels=5, dims=2)
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=7),
        confirm=0,
    )
    o = res.origins["random"]
    assert o["asks"] == 7 and o["fresh"] == 7 and o["best"] >= 1


def test_recommend_enumerates_a_space_exactly_the_budget():
    assert recommend("gemm", 64, 64) == {"strategy": "exhaustive"}
    assert recommend("gemm", 65, 64)["strategy"] == "portfolio"


def test_default_ladder_shape():
    ladder = default_ladder(20)
    assert [f.reps for f in ladder] == [3, 8, 20]
    assert [f.cost for f in ladder] == pytest.approx([0.15, 0.4, 1.0])
    assert default_ladder(200)[0].cost == pytest.approx(0.05)
    assert len({f.pool_id for f in ladder}) == 1


def test_hill_start_is_the_first_warm_start():
    space, _f = sphere(levels=5, dims=2)
    warm = [WarmStart({"x0": 1, "x1": 1}, "a"), WarmStart({"x0": 3, "x1": 3}, "b")]
    hc = make_strategy("hill", space, warm=warm)
    assert hc.start == (1, 1) and hc.seed_queue == [[3, 3]]


def test_record_result_without_confirmation_counts_the_winners_samples():
    space, f = sphere(levels=5, dims=2)
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="exhaustive",
        budget=Budget(evaluations=30),
        confirm=0,
    )
    rec = record_result(
        res,
        family="f",
        candidate="c",
        arch="a",
        problem="p",
        tuning_id=f"v@{res.best_key}",
        config_key=res.best_key,
        knobs={},
    )
    assert rec.samples == 20 and rec.speedup is None and rec.trace == ""
    assert not TuningDB(None).significant(rec)


# ---------------------------------------------------------------------- db
def _rec(**kw):
    d = {
        "family": "f",
        "candidate": "c",
        "arch": "gfx950",
        "problem": "p",
        "tuning_id": "v@k1",
        "config_key": "k1",
        "knobs": {"u": 1},
        "value": 1.0,
        "toolchain": "t",
    }
    d.update(kw)
    return TunedRecord(**d)


def test_db_equal_speedup_does_not_replace_and_created_is_kept(tmp_path):
    db = TuningDB(tmp_path / "a" / "b" / "db.json")
    assert db.put(_rec(speedup=1.1, p_value=0.01, created="2026-01-01T00:00:00"))
    assert db.get("f", "gfx950", "p", "c").created == "2026-01-01T00:00:00"
    assert not db.put(
        _rec(tuning_id="v@k2", config_key="k2", speedup=1.1, p_value=0.01)
    )
    assert db.put(_rec(tuning_id="v@k3", config_key="k3", speedup=1.11, p_value=0.01))
    assert db.get("f", "gfx950", "p", "c").created  # stamped when missing


def test_db_neighbors_filter_and_feature_distance():
    db = TuningDB(None)
    db.put(_rec(problem="p1", candidate="c", features={"m": 0.0, "n": -2.0}))
    db.put(_rec(problem="p2", candidate="d", features={"m": 4.0, "n": 2.0}))
    near = db.neighbors("f", "gfx950", {"m": 0.0, "n": -2.0}, k=5, candidate="c")
    assert [r.problem for _d, r in near] == ["p1"] and near[0][0] == 0.0
    both = db.neighbors("f", "gfx950", {"m": 0.0, "n": -2.0}, k=5)
    assert [r.problem for _d, r in both] == ["p1", "p2"]
    assert both[1][0] == pytest.approx(math.sqrt(16 + 16))


def test_db_reads_version_1_files_and_rows(tmp_path):
    path = tmp_path / "db.json"
    row = {
        k: v
        for k, v in _rec().to_json().items()
        if k
        not in ("speedup", "p_value", "baseline_key", "rounds", "toolchain", "machine")
    }
    path.write_text(json.dumps({"version": 1, "records": [row]}))
    db = TuningDB(path)
    assert len(db) == 1 and db.records()[0].speedup is None
    assert records_from_rows([row])[0].config_key == "k1"
    path.write_text(json.dumps({"version": 99, "records": []}))
    with pytest.raises(ValueError, match="version"):
        TuningDB(path)


def test_db_lock_timeout(tmp_path):
    path = tmp_path / "db.json"
    db = TuningDB(path, lock_timeout=0.2)
    (tmp_path / "db.json.lock").write_text("123")
    with pytest.raises(TimeoutError, match="lock"):
        db.put(_rec())


def test_problem_key_of_non_json_fields_uses_repr():
    class Obj:
        def __repr__(self):
            return "Obj()"

    a = problem_key({"m": 1, "o": Obj()})
    b = problem_key({"m": 1, "o": "Obj()"})
    assert a == b


# ---------------------------------------------------------------------- de
def test_de_parameter_bounds():
    space, _f = sphere()
    DifferentialEvolution(space, 0, CR=0.0)
    DifferentialEvolution(space, 0, p_best=1.0)
    with pytest.raises(ValueError, match="p_best"):
        DifferentialEvolution(space, 0, p_best=1.5)
    assert DifferentialEvolution(space, 0).done is False


def test_de_init_takes_the_best_history_then_stratified():
    space, f = sphere(levels=9, dims=3)
    de = DifferentialEvolution(space, 0, pop_size=8)
    ev = Evaluator(space, SyntheticBackend(f))
    other = make_strategy("random", space, 5)
    obs, _ = ev.evaluate(other.ask(12))
    de.tell(obs)
    de.ask(1)
    keys = [s["key"] for s in de._init_slots]
    assert len(keys) == 8 and len(set(keys)) == 8
    best4 = sorted(o.key for o in sorted(obs, key=lambda o: (o.value, o.key))[:4])
    assert sorted(keys[:4]) == best4  # half from the best known configs


def test_de_current_to_best_leads_with_the_best():
    space = SearchSpace((Param.ordinal("x", tuple(range(11))),))
    de = DifferentialEvolution(
        space, 0, pop_size=4, variant="current_to_best1bin", reset_prob=0.0
    )
    de.pop = [[0], [10], [5], [5]]
    de.fit = [9.0, 1.0, 5.0, 5.0]
    de.archive = []
    for seed in range(20):
        de.rng = random.Random(seed)
        v = de._mutant(0, F=1.0)
        # x + F (best - x) + F (r1 - r2) with best = 10: always far right of 0
        assert v[0] >= 5.0


def test_de_pbest_uses_the_top_fraction():
    space = SearchSpace((Param.ordinal("x", tuple(range(20))),))
    de = DifferentialEvolution(space, 0, pop_size=8, p_best=1.0)
    de.pop = [[i] for i in range(8)]
    de.fit = [float(i) for i in range(8)]
    de.archive = []
    leads = set()
    real_choice = de.rng.choice

    def spy(seq):
        leads.update(seq)
        return real_choice(seq)

    de.rng.choice = spy
    de._mutant(0, F=0.5)
    assert len(leads) == 8  # p_best = 1: every member is a candidate lead


def test_de_selection_with_noise_and_shade_memory_without_successes():
    space, _f = sphere(levels=9, dims=2)
    de = DifferentialEvolution(space, 0, pop_size=4, adapt="shade")
    noise = NoiseModel()
    for _ in range(10):
        noise.update([1.0, 1.05, 0.95, 1.02])  # 3-5% noise, above the floor
    de.attach_noise(noise)
    gain = de._noise_gain()
    assert gain > 0.03
    de.pop = [[0, 0], [1, 1], [2, 2], [3, 3]]
    de.fit = [1.0, 1.0, 1.0, 1.0]
    de.M_F, de.M_CR = [0.3] * de.H, [0.7] * de.H
    de._gen = [
        {
            "i": i,
            "g": [i + 4, i + 4],
            "key": f"k{i}",
            "F": 0.9,
            "CR": 0.1,
            "value": 1.0 * (1 + 0.5 * gain),
            "uid": None,
        }
        for i in range(4)
    ]
    de._select()
    # within noise: the trials replace their parents, but are no successes
    assert de.pop == [[4, 4], [5, 5], [6, 6], [7, 7]]
    assert de.M_F == [0.3] * de.H and de.M_CR == [0.7] * de.H


# ----------------------------------------------------------------- halving
def test_successive_halving_with_no_members_is_done():
    space, _f = sphere(levels=3, dims=1)
    keys = [space.key((x,)) for x in range(3)]
    sh = SuccessiveHalving(space, 0, n_configs=3, exclude=keys)
    assert sh.ask(3) == [] and sh.done
    assert sh.ask(3) == []


def test_sh_score_pools_calls_of_the_current_pool_only():
    space, _f = sphere(levels=5, dims=1)
    sh = SuccessiveHalving(space, 0, fidelities=(0, -1), n_configs=2, race=None)
    sh._pool = "|hot"
    sh.members = ["a", "b"]
    sh.rung_values = {"a": 2.0, "b": 1.5}
    sh.calls = {
        "a": [[0, 1.0, "|hot", 10], [1, 2.0, "|hot", 10]],  # gmean ~1.41
        "b": [[0, 0.1, "|cold", 10], [1, 1.5, "|hot", 10]],  # cold call ignored
    }
    sh.member_genomes = {"a": [0], "b": [1]}
    sh.rung = 0
    sh._advance()
    assert sh.members == ["a"]  # pooled hot calls rank a ahead of b


def test_hyperband_validation_ask_limit_and_termination():
    space, f = sphere(levels=3, dims=1)
    with pytest.raises(ValueError, match="eta"):
        Hyperband(space, 0, eta=1.0)
    hb = Hyperband(space, 0, fidelities=(0, 1, -1))
    ev = Evaluator(space, SyntheticBackend(f), ladder=default_ladder(8))
    for _ in range(200):
        ts = hb.ask(2)
        assert len(ts) <= 2
        if not ts:
            break
        obs, _ = ev.evaluate(ts)
        hb.tell(obs)
    assert hb.done is True


# -------------------------------------------------------------------- local
def test_hill_recombines_pairs_with_two_combo_knobs():
    params = tuple(Param.categorical(n, (0, 1)) for n in ("a", "b", "c"))
    space = SearchSpace(params)

    def f(c):
        if c["a"] == 1 and c["b"] == 1:
            return 0.5
        return 1.0 + 0.004 * (c["a"] + c["b"]) + 0.5 * c["c"]

    hc = HillClimb(space, 0, start=(0, 0, 0), combo_knobs=2, combo_tol=0.01)
    res = run_search(
        space, hc, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=6)
    )
    assert res.best_value == 0.5 and hc.restarts == 0


def test_hill_keeps_climbing_after_a_restart_and_respects_ask_size():
    space, f = sphere(levels=9, dims=2)
    hc = HillClimb(space, 0, start=(4, 4), init=[(0, 0), (8, 8), (0, 8)], lookback=0)
    ev = Evaluator(space, SyntheticBackend(f))
    for _ in range(40):
        ts = hc.ask(2)
        assert len(ts) <= 2
        if not ts:
            break
        obs, _ = ev.evaluate(ts)
        hc.tell(obs)
    assert hc.restarts >= 1 and len(hc.history) > 12


def test_hill_probe_waits_for_its_results():
    space, f = sphere(levels=9, dims=2)
    hc = HillClimb(space, 0, n_init=4)
    probe = hc.ask(8)
    assert len(probe) == 4 and hc.ask(8) == []  # waits for the probe
    obs, _ = Evaluator(space, SyntheticBackend(f)).evaluate(probe)
    hc.tell(obs)
    best = min(obs, key=lambda o: o.value)
    nxt = hc.ask(1)
    assert nxt and space.distance(best.trial.genome, nxt[0].genome) <= 1.0


# ------------------------------------------------------------------ measure
def test_spill_gate_rejects_any_positive_count():
    space, _f = sphere(levels=3, dims=1)
    backend = SyntheticBackend(lambda c: 1.0)
    backend.compile = lambda c, k: CompileOutcome(True, resources={"spill_vgpr": 1})
    ev = Evaluator(space, backend)
    (ob,), _ = ev.evaluate([make_strategy("random", space).trial((1,))])
    assert ob.status == Status.SPILL and ev.compiles == 1


def test_synthetic_objective_can_depend_on_fidelity():
    space, _f = sphere(levels=3, dims=1)
    ladder = (Fidelity("p", reps=2, cost=0.1), Fidelity("full", reps=2))
    backend = SyntheticBackend(lambda c, fid: 2.0 if fid.name == "p" else 1.0)
    ev = Evaluator(space, backend, ladder=ladder)
    s = make_strategy("random", space)
    (a,), _ = ev.evaluate([s.trial((1,), fidelity=0)])
    (b,), _ = ev.evaluate([s.trial((1,), fidelity=-1)])
    assert (a.value, b.value) == (2.0, 1.0)


def test_table_backend_failures_and_call_noise():
    space = SearchSpace.from_configs([{"a": i} for i in range(4)])
    keys = [space.key(space.encode({"a": i})) for i in range(4)]
    table = {keys[0]: 1.0, keys[1]: 2.0}
    tb = TableBackend(
        table, failures={keys[2]: Status.NUMERIC_FAIL, keys[3]: Status.COMPILE_ERROR}
    )
    ev = Evaluator(space, tb)
    s = make_strategy("random", space)
    obs, _ = ev.evaluate([s.trial(space.encode({"a": i})) for i in range(4)])
    assert [o.status for o in obs] == [
        "ok",
        "ok",
        Status.NUMERIC_FAIL,
        Status.COMPILE_ERROR,
    ]
    noisy = TableBackend(table, call_noise=0.05, seed=3)
    meds = sorted(
        noisy.measure({}, keys[0], None, Fidelity(reps=1))[0] for _ in range(301)
    )
    assert meds[150] == pytest.approx(1.0, abs=0.01)  # unbiased in the median


# ------------------------------------------------------------------- params
def test_param_label_and_describe():
    p = Param.ordinal("t", (64, 128))
    assert p.label(0) == "64"
    combo = Param("g", ((), (("g", True), ("d", 2))), "categorical")
    assert combo.label(0) == "base" and combo.label(1) == "g=True,d=2"
    d = p.describe()
    assert d == {
        "name": "t",
        "kind": "ordinal",
        "gate": None,
        "levels": [[["t", 64]], [["t", 128]]],
    }


def test_config_keys_of_non_json_and_unhashable_values():
    assert config_items_key({"a": (1, 2)}) == '[["a","(1, 2)"]]'
    space = SearchSpace.from_configs([{"a": [1, 2], "b": 1}, {"a": [3], "b": 2}])
    assert len(space.listed()) == 2


def test_params_from_axes_unique_names_and_dropped_axes():
    axes = (values("x", 1, (1, 2, 4)), values("x", 1, (1, 8, 16)))
    names = [p.name for p in params_from_axes(axes, {"x": 1})]
    assert names == ["x", "x#2"]
    restating = KnobAxis("y", ((), (("y", 3),)))
    assert params_from_axes((restating,), {"y": 3}) == ()
    assert len(params_from_axes((restating,), {"y": 4})) == 1


def test_member_index_density_threshold():
    # 4 members of a 3 x 4 product (density 0.33 >= 0.25): single moves first
    rows = [{"a": 0, "b": 0}, {"a": 1, "b": 1}, {"a": 2, "b": 2}, {"a": 0, "b": 3}]
    space = SearchSpace.from_configs(rows)
    assert space._nearest_first is False
    # 5 members of a 5 x 5 product (density 0.2 < 0.25): nearest member first
    sparse = SearchSpace.from_configs([{"a": i, "b": (2 * i) % 5} for i in range(5)])
    assert sparse._nearest_first is True


# ---------------------------------------------------------------- portfolio
def test_portfolio_bookkeeping():
    space, f = sphere(levels=3, dims=1)
    with pytest.raises(ValueError, match="discount"):
        Portfolio(space, 0, members=[make_strategy("random", space)], discount=0.0)
    warm = [WarmStart({"x0": 1}, "dispatcher")]
    p = make_strategy(
        "portfolio", space, 0, members=("random", "exhaustive"), warm=warm
    )
    alloc = p.allocation()
    assert all(v["asks"] == 0 and v["wins"] == 0 for v in alloc.values())
    ev = Evaluator(space, SyntheticBackend(f))
    res = run_search(space, p, ev, Budget(evaluations=50))
    assert p.warm_results == {"dispatcher": "ok"}
    assert res.stopped == "strategy_done" and p.done is True
    assert sum(v["wins"] for v in p.allocation().values()) == sum(p.wins)
    assert sum(p.wins) <= 2  # only two configs could beat the warm start


# ------------------------------------------------------------------- replay
def test_load_table_parses_booleans_and_inverts_rates(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("p,flag,tile,tflops\nx,True,64,100\nx,false,128,50\n")
    data = load_table(
        csv,
        config_cols=["flag", "tile"],
        value_col="tflops",
        problem_cols=["p"],
        higher_is_better=True,
    )
    configs, times = data["x"]
    assert configs[0]["flag"] is True and configs[1]["flag"] is False
    assert times == [pytest.approx(0.01), pytest.approx(0.02)]


def test_replay_summary_threshold_counting():
    runs = [
        {"strategy": "s", "final": 1.0, "best_seen": 1.0, "curve": [0.5, 0.99, 1.0]},
        {"strategy": "s", "final": 0.9, "best_seen": 0.95, "curve": [0.9, 0.95]},
    ]
    row = replay_summary(runs)["s"]
    assert row["reach95"] == 1.0 and row["evals95"] == 2
    assert row["reach99"] == 0.5 and row["evals99"] == 2
    assert row["reach98"] == 0.5


# ---------------------------------------------------------------- surrogate
def test_model_strategies_start_not_done_and_tpe_densities_normalize():
    space, _f = gemm_like(rugged=0.1, salt="m")
    tpe = TPE(space, 0, n_startup=1)
    assert tpe.done is False and GPEI(space, 0).done is False
    dens = tpe._density([space.encode({"tile_m": 128, "tile_n": 64})] * 3)
    for p, d in zip(space.params, dens):
        assert sum(d) == pytest.approx(1.0)
        assert all(x > 0 for x in d)
    i = space.index("tile_m")
    assert max(range(len(dens[i])), key=dens[i].__getitem__) == 2  # peaked at 128


def test_gp_with_an_uninformative_prior_and_failures():
    space, f = sphere(levels=9, dims=2)
    fail = lambda c: Status.NUMERIC_FAIL if c["x0"] >= 6 else None
    prior = CallablePrior(lambda c: math.nan)
    gp = GPEI(space, 0, n_startup=6, prior=prior)
    res = run_search(
        space,
        gp,
        Evaluator(space, SyntheticBackend(f, fail=fail)),
        Budget(evaluations=30),
    )
    assert res.best_value <= 2.0
    assert len(gp.failed) >= 1  # failures enter the model as bad observations


# ------------------------------------------------- second-round survivors
def test_welch_reference_value():
    # Welch's t-test example 1 (Wikipedia): t = -2.46, df = 24.9, two-sided
    # p = 0.021, so the one-sided p that b exceeds a is about 0.0106
    a = [
        27.5,
        21.0,
        19.0,
        23.6,
        17.0,
        17.9,
        16.9,
        20.1,
        21.9,
        22.6,
        23.1,
        19.6,
        19.0,
        21.7,
        21.4,
    ]
    b = [
        27.1,
        22.0,
        20.8,
        23.4,
        23.4,
        23.5,
        25.8,
        22.0,
        24.8,
        20.2,
        21.9,
        22.1,
        22.9,
        20.5,
        24.4,
    ]
    assert welch_greater(b, a) == pytest.approx(0.0106, abs=0.0015)
    assert welch_greater([1.0, 2.0], [0.0, 0.5]) < 0.5  # two samples are enough


def test_noise_model_ignores_invalid_call_pairs():
    m = NoiseModel()
    for a, b in ((0.0, 1.0), (1.0, -1.0), (math.inf, 1.0), (1.0, math.nan)):
        m.update_call(a, b)
    assert m.to_json()["pairs"] == []


def test_race_z_test_reference():
    race = Race(alpha=0.05, margin=0.01, min_samples=3)
    p = race.pvalue([(-1, 1.10)], [(-1, 1.00)], sigma=0.05)
    z = (math.log(1.10) - math.log1p(0.01)) / (0.05 * math.sqrt(2.0))
    assert p == pytest.approx(0.5 * math.erfc(z / math.sqrt(2.0)))
    p2 = race.pvalue([(-1, 1.10), (-1, 1.10)], [(-1, 1.00)], sigma=0.05)
    z2 = (math.log(1.10) - math.log1p(0.01)) / (0.05 * math.sqrt(1.5))
    assert p2 == pytest.approx(0.5 * math.erfc(z2 / math.sqrt(2.0)))


def test_sh_race_calls_pool_only_when_pooled():
    space, _f = sphere(levels=5, dims=1)
    for pooled, want in ((True, [(0, 1.0), (1, 2.0)]), (False, [(1, 2.0)])):
        sh = SuccessiveHalving(space, 0, fidelities=(0, -1), pooled=pooled)
        sh._pool = "|hot"
        sh.calls = {"a": [[0, 1.0, "|hot", 3], [1, 2.0, "|hot", 8]]}
        assert sh._race_calls("a") == want


def test_hyperband_bracket_shapes_and_seeds():
    space, _f = sphere(levels=9, dims=2)
    hb = Hyperband(space, 0, fidelities=(0, 1, -1), eta=3.0, init=[(1, 1)])
    assert hb._bracket_shape(0) == (9, (0, 1, -1))
    assert hb._bracket_shape(1) == (5, (1, -1))
    assert hb._bracket_shape(2) == (3, (-1,))
    assert hb._bracket_shape(3) == (9, (0, 1, -1))
    hb._new_bracket()
    assert [tuple(g) for g in hb.inner._seeds] == [(1, 1)]
    hb._new_bracket()
    assert hb.inner._seeds == []  # warm seeds only in the first bracket


def test_portfolio_warm_duplicates_are_measured_once():
    space, _f = sphere(levels=5, dims=1)
    warm = [
        WarmStart({"x0": 2}, "a"),
        WarmStart({"x0": 2}, "b"),
        WarmStart({"x0": 3}, "c"),
    ]
    p = make_strategy("portfolio", space, 0, warm=warm)
    first = p.ask(5)
    assert [t.genome for t in first] == [(2,), (3,)]


def test_replay_takes_the_median_of_repeated_rows():
    from rocke.dispatch.tuning.search.replay import ReplayProblem

    configs = [{"a": 1}, {"a": 1}, {"a": 1}, {"a": 2}]
    rp = ReplayProblem("x", configs, [3.0, 1.0, 2.0, 5.0])
    key = rp.space.key(rp.space.encode({"a": 1}))
    assert rp.table[key] == 2.0


def test_de_init_takes_exactly_half_from_history():
    space, f = sphere(levels=9, dims=3)
    de = DifferentialEvolution(space, 0, pop_size=8)
    ev = Evaluator(space, SyntheticBackend(f))
    obs, _ = ev.evaluate(make_strategy("random", space, 5).ask(12))
    de.tell(obs)
    de.ask(1)
    from_history = [s for s in de._init_slots if s["key"] in de.history]
    assert len(from_history) == 4


def test_model_startup_takes_half_from_the_prior():
    space, f = gemm_like(rugged=0.1, salt="s")
    prior = CallablePrior(f)
    tpe = TPE(space, 0, n_startup=10, prior=prior)
    tpe._build_startup()
    top = {space.key(g) for g in tpe._prior_top[:5]}
    picked = [space.key(g) for g in tpe._startup]
    assert len(picked) == 10 and set(picked[:5]) == top


def test_hill_stages_are_single_moves_combos_then_two_steps():
    params = (Param.ordinal("a", tuple(range(7))), Param.categorical("b", (0, 1, 2)))
    space = SearchSpace(params)
    hc = HillClimb(space, 0, start=(3, 0), combo_knobs=2, combo_tol=1.0)
    hc.history = {space.key(g): 1.0 for g in [(3, 0), (2, 0), (4, 0), (3, 1), (3, 2)]}
    s0 = hc._stage_genomes([3, 0], 0)
    s1 = hc._stage_genomes([3, 0], 1)
    s2 = hc._stage_genomes([3, 0], 2)
    assert s0[0] == (3, 0) and set(s0[1:]) == {(2, 0), (4, 0), (3, 1), (3, 2)}
    assert s1 and all(sum(x != y for x, y in zip(g, (3, 0))) == 2 for g in s1)
    assert set(s2) == {(1, 0), (5, 0)}


# ------------------------------------------------- third-round survivors
def test_validation_boundaries():
    space, _f = sphere(levels=3, dims=1)
    Portfolio(space, 0, members=[make_strategy("random", space)], discount=1.0)
    with pytest.raises(ValueError, match="cost"):
        Fidelity(cost=0.0)
    with pytest.raises(ValueError, match="seconds"):
        Budget(seconds=0.0)
    Race(min_samples=2)


def test_budget_meter_reasons():
    meter = BudgetMeter(Budget(evaluations=2, seconds=5.0, cost=3.0))
    assert meter.exhausted() == ""
    meter.seconds = 5.0
    assert meter.exhausted() == "seconds"
    meter.seconds, meter.cost = 0.0, 3.0
    assert meter.exhausted() == "cost"
    meter.evaluations = 2
    assert meter.exhausted() == "evaluations"


def test_race_pairs_with_exactly_min_samples_common_batches():
    race = Race(alpha=0.05, margin=0.0, min_samples=3)
    slow = [(0, 1.10), (1, 1.11), (2, 1.09)]
    fast = [(0, 1.00), (1, 1.00), (2, 1.00)]
    p = race.pvalue(slow, fast)
    logs = [math.log(a) - math.log(b) for (_x, a), (_y, b) in zip(slow, fast)]
    assert p == pytest.approx(paired_t_greater(logs, 0.0))


# ------------------------------------------------ fourth-round survivors
def test_check_timeout_is_a_failure_and_never_timed():
    import time as _time

    space, f = sphere(levels=3, dims=1)

    class SlowCheck(SyntheticBackend):
        def check(self, config, key, artifact):
            if config["x0"] == 1:
                _time.sleep(0.4)
            return super().check(config, key, artifact)

    backend = SlowCheck(f)
    ev = Evaluator(space, backend, measure_timeout=0.1)
    (ob,), _ = ev.evaluate([make_strategy("random", space).trial((1,))])
    assert ob.status == Status.TIMEOUT and ob.full and backend.measures == 0
    _time.sleep(0.5)
    ev.close()


def test_needs_compile_semantics():
    space, f = sphere(levels=3, dims=1)
    fail = lambda c: Status.COMPILE_ERROR if c["x0"] == 2 else None
    ev = Evaluator(space, SyntheticBackend(f, fail=fail))
    s = make_strategy("random", space)
    keys = [space.key((x,)) for x in range(3)]
    assert all(ev.needs_compile(k) for k in keys)
    ev.evaluate([s.trial((0,)), s.trial((2,))])
    assert not ev.needs_compile(keys[0])  # compiled
    assert not ev.needs_compile(keys[2])  # known failure: never rebuilt
    assert ev.needs_compile(keys[1])


def test_successive_halving_eta_and_unpooled_scores():
    space, _f = sphere(levels=5, dims=1)
    with pytest.raises(ValueError, match="eta"):
        SuccessiveHalving(space, 0, eta=1.0)
    sh = SuccessiveHalving(
        space, 0, fidelities=(0, -1), n_configs=2, race=None, pooled=False
    )
    sh._pool = "|hot"
    sh.members = ["a", "b"]
    sh.rung_values = {"a": 2.0, "b": 1.5}
    sh.calls = {
        "a": [[0, 1.0, "|hot", 10], [1, 2.0, "|hot", 10]],
        "b": [[1, 1.5, "|hot", 10]],
    }
    sh.member_genomes = {"a": [0], "b": [1]}
    sh._advance()
    assert sh.members == ["b"]  # unpooled: this rung's values only


def test_hyperband_brackets_exclude_known_configs_but_the_incumbent():
    space, f = sphere(levels=9, dims=2)
    hb = Hyperband(space, 0, fidelities=(0, 1, -1))
    ev = Evaluator(space, SyntheticBackend(f), ladder=default_ladder(8))
    obs, _ = ev.evaluate(
        [make_strategy("random", space, 3).trial((x, x)) for x in range(5)]
    )
    hb.tell(obs)
    best_key, _v = hb.best()
    hb._new_bracket()
    hb.inner._start()
    members = set(hb.inner.members)
    assert best_key in members
    assert not (members & (set(hb.history) - {best_key}))


def test_confirmation_with_one_measurable_config_is_empty():
    space = SearchSpace((Param.categorical("c", ("a", "b")),))
    fail = lambda c: Status.NUMERIC_FAIL if c["c"] == "b" else None
    res = tune(
        space,
        SyntheticBackend(lambda c: 1.0, fail=fail),
        strategy="exhaustive",
        budget=Budget(evaluations=10),
        confirm=2,
        confirm_rounds=1,
    )
    assert res.confirmed == {} and res.best_config == {"c": "a"}


def test_confirmation_tied_list_excludes_dominated_finalists():
    space = SearchSpace((Param.categorical("c", ("a", "b", "z")),))
    times = {"a": 1.0, "b": 1.0005, "z": 2.0}
    res = tune(
        space,
        SyntheticBackend(lambda c: times[c["c"]], noise=0.001, drift=0.0, seed=1),
        strategy="exhaustive",
        budget=Budget(evaluations=40),
        confirm=3,
        confirm_rounds=6,
    )
    c = res.confirmed
    zkey = space.key(space.encode({"c": "z"}))
    assert zkey in c["dominated"] and zkey not in c["tied_with_winner"]
    assert len(c["tied_with_winner"]) == 1


def test_db_get_any_candidate_prefers_the_largest_speedup():
    db = TuningDB(None)
    db.put(_rec(candidate="c1", speedup=1.05, p_value=0.01, value=0.5))
    db.put(_rec(candidate="c2", speedup=1.20, p_value=0.01, value=2.0))
    db.put(_rec(candidate="c3", value=0.1))  # no speedup at all
    assert db.get("f", "gfx950", "p").candidate == "c2"


def test_de_init_skips_failed_history_and_budget_seconds_fraction():
    space, _f = sphere(levels=9, dims=2)
    de = DifferentialEvolution(space, 0, pop_size=4)
    t = de.trial((4, 4))
    de.tell([Observation(trial=t, key=space.key((4, 4)), status=Status.NUMERIC_FAIL)])
    de.ask(1)
    assert space.key((4, 4)) not in [s["key"] for s in de._init_slots]
    Budget(seconds=0.5)
