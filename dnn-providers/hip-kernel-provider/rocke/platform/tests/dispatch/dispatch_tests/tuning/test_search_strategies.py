# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Strategies on synthetic objectives with known optima: correctness,
determinism, budget accounting, resume, and the strategy-specific
mechanisms (DE adaptation and gated crossover, hill-climb combination
re-sweep and lookback, prior seeding, successive-halving racing, portfolio
allocation and warm starts)."""

from __future__ import annotations

import json
import math
import random

import pytest
from rocke.dispatch.tuning.search import (
    Budget,
    DifferentialEvolution,
    Evaluator,
    Fidelity,
    HillClimb,
    Observation,
    Param,
    Portfolio,
    Race,
    SearchSpace,
    Status,
    Strategy,
    SuccessiveHalving,
    SyntheticBackend,
    TraceWriter,
    WarmStart,
    make_strategy,
    run_search,
    tune,
)
from rocke.dispatch.tuning.search.api import STRATEGY_NAMES
from rocke.dispatch.tuning.search.surrogate import CallablePrior
from rocke.dispatch.tuning.search.synthetic import (
    brute_force_optimum,
    gemm_like,
    sphere,
)

ALL = [s for s in STRATEGY_NAMES if s != "exhaustive"]


def _run(
    space, objective, strategy, budget, seed=0, *, noise=0.0, confirm=0, batch=1, **kw
):
    backend = SyntheticBackend(objective, noise=noise, seed=seed)
    return tune(
        space,
        backend,
        strategy=strategy,
        budget=Budget(evaluations=budget),
        seed=seed,
        confirm=confirm,
        batch=batch,
        **kw,
    )


@pytest.fixture(scope="module")
def gemm():
    space, f = gemm_like(rugged=0.15, salt="t")
    key, best, _count = brute_force_optimum(space, f)
    return space, f, key, best


# ----------------------------------------------------------- correctness
@pytest.mark.parametrize("strategy", ["de", "jde", "de_rand", "tpe", "gp", "portfolio"])
def test_converges_on_sphere(strategy):
    space, f = sphere(levels=9, dims=4)
    hits = 0
    for seed in range(3):
        res = _run(space, f, strategy, 256, seed)
        hits += f(res.best_config) == 1.0
    assert hits >= 2, f"{strategy} reached the sphere optimum in {hits}/3 runs"


def test_hill_climb_reaches_unimodal_optimum_exactly():
    space, f = sphere(levels=9, dims=4)
    for seed in range(3):
        res = _run(space, f, "hill", 200, seed)
        assert f(res.best_config) == 1.0


_RANDOM_AT_64 = {}


# Mean fraction of the optimum over 16 seeds at 64 evaluations; random
# reaches about 0.88 there. Floors sit 0.01-0.02 under the measured means,
# and each strategy must beat random by a clear margin, not by a hair.
@pytest.mark.parametrize(
    "strategy, floor, margin",
    [
        ("portfolio", 0.935, 0.05),
        ("gp", 0.94, 0.05),
        ("hill", 0.955, 0.06),
        ("tpe", 0.895, 0.02),
        ("de", 0.89, 0.015),
    ],
)
def test_beats_random_on_rugged_gemm_space(gemm, strategy, floor, margin):
    space, f, _key, best = gemm

    def score(s):
        return (
            sum(best / f(_run(space, f, s, 64, seed).best_config) for seed in range(16))
            / 16
        )

    if "r" not in _RANDOM_AT_64:
        _RANDOM_AT_64["r"] = score("random")
    mine, rand = score(strategy), _RANDOM_AT_64["r"]
    assert mine >= floor, f"{strategy} {mine:.4f} < {floor}"
    assert mine >= rand + margin, f"{strategy} {mine:.4f} vs random {rand:.4f}"


def test_exhaustive_finds_the_true_optimum_and_stops(gemm):
    space, f = sphere(levels=5, dims=3)
    key, best, count = brute_force_optimum(space, f)
    res = _run(space, f, "exhaustive", 10_000)
    assert res.best_key == key and res.best_value == best
    assert res.evaluations == count and res.stopped == "strategy_done"


# ----------------------------------------------------------- determinism
def _keys(res):
    return [o.key for o in res.observations]


@pytest.mark.parametrize("strategy", ALL)
def test_same_seed_same_run_and_seed_matters(gemm, strategy):
    space, f, _k, _b = gemm
    a = _run(space, f, strategy, 40, seed=7, noise=0.02)
    b = _run(space, f, strategy, 40, seed=7, noise=0.02)
    assert _keys(a) == _keys(b)
    assert [o.value for o in a.observations] == [o.value for o in b.observations]
    c = _run(space, f, strategy, 40, seed=8, noise=0.02)
    assert _keys(a) != _keys(c)


def test_batched_asks_are_deterministic_and_within_budget(gemm):
    space, f, _k, _b = gemm
    a = _run(space, f, "portfolio", 48, seed=1, batch=6)
    b = _run(space, f, "portfolio", 48, seed=1, batch=6)
    assert _keys(a) == _keys(b)
    assert a.evaluations <= 48


# ------------------------------------------------------------- budget
@pytest.mark.parametrize("strategy", ALL)
def test_budget_is_never_exceeded(gemm, strategy):
    space, f, _k, _b = gemm
    for batch in (1, 5):
        res = _run(space, f, strategy, 23, seed=3, batch=batch)
        assert res.evaluations <= 23
        fresh = [
            o for o in res.observations if not o.cached and o.status != Status.INVALID
        ]
        assert len(fresh) == res.evaluations


def test_cache_hits_and_refusals_are_free():
    space, f = sphere(levels=5, dims=2)
    ev = Evaluator(space, SyntheticBackend(f))

    class Repeat(Strategy):
        name = "repeat"

        def ask(self, n=1):
            return [self.trial((1, 1)) for _ in range(n)]

    res = run_search(
        space, Repeat(space), ev, Budget(evaluations=5), batch=3, stall_rounds=4
    )
    assert res.evaluations == 1
    assert res.meter.cache_hits >= 2 and res.stopped == "stalled"


def test_cost_budget_counts_fidelity_cost():
    space, f = sphere(levels=9, dims=3)
    ladder = (Fidelity("cheap", reps=2, cost=0.25), Fidelity("full", reps=10, cost=1.0))
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="sh",
        ladder=ladder,
        budget=Budget(cost=5.0),
        confirm=0,
        strategy_options={"n_configs": 9},
    )
    assert res.meter.cost <= 5.0 + 1.0
    cheap = sum(1 for o in res.observations if not o.cached and o.fidelity == 0)
    assert cheap >= 9


def test_seconds_budget_uses_the_clock():
    space, f = sphere(levels=9, dims=3)
    ticks = iter(range(10_000))
    ev = Evaluator(space, SyntheticBackend(f))
    res = run_search(
        space,
        make_strategy("random", space),
        ev,
        Budget(seconds=10),
        clock=lambda: float(next(ticks)),
    )
    assert res.stopped == "seconds"
    assert res.evaluations < 20


def test_confirm_reserves_budget_and_reports_finalists(gemm):
    space, f, _k, _b = gemm
    res = _run(space, f, "de", 30, seed=2, noise=0.03, confirm=3)
    assert res.evaluations <= 30
    c = res.confirmed
    assert len(c["finalists"]) == 3 and c["winner"] in c["finalists"]
    assert res.best_key == c["winner"]
    assert res.best_value == c["values"][c["winner"]]


# ---------------------------------------------------------------- resume
@pytest.mark.parametrize("strategy", ALL)
def test_resume_matches_an_uninterrupted_run(gemm, strategy, tmp_path):
    space, f, _k, _b = gemm
    full = _run(space, f, strategy, 36, seed=4, batch=3)
    ck = tmp_path / "ck.json"
    first = _run(space, f, strategy, 17, seed=4, batch=3, checkpoint=ck)
    assert first.evaluations == 17 and ck.exists()
    json.loads(ck.read_text())  # plain JSON
    second = _run(space, f, strategy, 36, seed=4, batch=3, checkpoint=ck, resume=True)
    assert _keys(second) == _keys(full)
    assert second.best_key == full.best_key
    assert second.evaluations == full.evaluations


def test_resume_refuses_another_space_or_strategy(tmp_path):
    space, f = sphere(levels=5, dims=2)
    ck = tmp_path / "ck.json"
    _run(space, f, "de", 8, checkpoint=ck)
    with pytest.raises(ValueError, match="strategy"):
        _run(space, f, "tpe", 16, checkpoint=ck, resume=True)
    other, g = sphere(levels=6, dims=2)
    with pytest.raises(ValueError, match="space"):
        _run(other, g, "de", 16, checkpoint=ck, resume=True)


@pytest.mark.parametrize("strategy", ALL)
def test_state_dict_is_json_and_round_trips(gemm, strategy):
    space, f, _k, _b = gemm
    s = make_strategy(strategy, space, 5)
    ev = Evaluator(space, SyntheticBackend(f))
    run_search(space, s, ev, Budget(evaluations=12), batch=2)
    blob = json.dumps(s.state_dict())
    t = make_strategy(strategy, space, 5)
    t.load_state_dict(json.loads(blob))
    assert json.loads(json.dumps(t.state_dict())) == json.loads(blob)
    assert [x.genome for x in s.ask(3)] == [x.genome for x in t.ask(3)]


# ---------------------------------------------------- measurement safety
def test_numeric_failures_are_never_the_answer(gemm):
    space, f, key, _best = gemm
    # Every config faster than the 98th percentile "computes garbage".
    vals = sorted(f(space.decode(g)) for g in space.enumerate() if space.valid(g))
    cut = vals[int(0.02 * len(vals))]
    fail = lambda c: Status.NUMERIC_FAIL if f(c) < cut else None
    res = tune(
        space,
        SyntheticBackend(f, fail=fail),
        strategy="exhaustive",
        budget=Budget(evaluations=20_000),
        confirm=0,
    )
    assert res.best_key != key
    assert f(res.best_config) >= cut
    fails = [o for o in res.observations if o.status == Status.NUMERIC_FAIL]
    assert fails and all(math.isinf(o.value) and not o.ok for o in fails)


def test_spills_and_compile_errors_are_rejected():
    space, _f = sphere(levels=5, dims=2)
    status = {(0, 0): Status.SPILL, (1, 1): Status.COMPILE_ERROR}
    fail = lambda c: status.get((c["x0"], c["x1"]))
    backend = SyntheticBackend(lambda c: 1.0, fail=fail)
    ev = Evaluator(space, backend)
    s = make_strategy("exhaustive", space)
    res = run_search(space, s, ev, Budget(evaluations=100))
    by = {(o.trial.genome[0], o.trial.genome[1]): o for o in res.observations}
    assert by[(0, 0)].status == Status.SPILL
    assert by[(1, 1)].status == Status.COMPILE_ERROR
    assert backend.checks == 25 - 2  # failed builds never reach the check
    ev2 = Evaluator(space, backend, reject_spill=False)
    obs, _ = ev2.evaluate([s.trial((0, 0))])
    assert obs[0].ok


def test_measure_timeout_and_errors():
    import time as _time

    space, f = sphere(levels=3, dims=1)

    class Slow(SyntheticBackend):
        def measure(self, config, key, artifact, fidelity):
            if config["x0"] == 2:
                _time.sleep(0.5)
            if config["x0"] == 1:
                raise RuntimeError("launch failed")
            return super().measure(config, key, artifact, fidelity)

    ev = Evaluator(space, Slow(f), measure_timeout=0.1)
    s = make_strategy("exhaustive", space)
    res = run_search(space, s, ev, Budget(evaluations=10))
    st = {o.trial.genome[0]: o.status for o in res.observations}
    assert st == {0: Status.OK, 1: Status.RUN_ERROR, 2: Status.TIMEOUT}


def test_backend_without_check_is_refused():
    space, _f = sphere(levels=3, dims=1)

    class NoCheck:
        def compile(self, config, key):
            return None

        def measure(self, config, key, artifact, fidelity):
            return [1.0]

    with pytest.raises(TypeError, match="numeric check is mandatory"):
        Evaluator(space, NoCheck())


def test_persistent_cache_spends_nothing_twice(tmp_path):
    space, f = sphere(levels=5, dims=2)
    path = tmp_path / "cache.json"
    b1 = SyntheticBackend(f)
    r1 = tune(
        space,
        b1,
        strategy="exhaustive",
        budget=Budget(evaluations=100),
        cache_path=path,
        confirm=0,
    )
    b2 = SyntheticBackend(f)
    r2 = tune(
        space,
        b2,
        strategy="exhaustive",
        budget=Budget(evaluations=100),
        cache_path=path,
        confirm=0,
    )
    assert b2.measures == 0 and b2.compiles == 0
    assert r2.evaluations == 0 and r2.best_key == r1.best_key


def test_interleaved_batches_are_paired():
    space, f = sphere(levels=5, dims=2)
    ev = Evaluator(space, SyntheticBackend(f, noise=0.01, drift=0.05))
    s = make_strategy("random", space, 1)
    obs, _ = ev.evaluate(s.ask(4))
    assert len({o.batch for o in obs}) == 1 and obs[0].batch > 0


def test_trace_rows(tmp_path):
    space, f = sphere(levels=5, dims=2)
    path = tmp_path / "t.jsonl"
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="de",
        budget=Budget(evaluations=10),
        trace=str(path),
        confirm=0,
    )
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[0]["type"] == "header" and rows[-1]["type"] == "end"
    obs = [r for r in rows if r["type"] == "obs"]
    assert len(obs) == len(res.observations)
    assert all("config" in r and "incumbent" in r for r in obs)


# ------------------------------------------------------------ DE details
def test_shade_memory_adapts_and_jde_keeps_successful_parameters():
    space, f = sphere(levels=9, dims=4)
    de = DifferentialEvolution(space, 1, adapt="shade")
    run_search(
        space, de, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=120)
    )
    assert de.generation >= 3
    assert de.M_F != [0.5] * de.H or de.M_CR != [0.5] * de.H
    jde = DifferentialEvolution(space, 1, adapt="jde", variant="rand1bin")
    run_search(
        space, jde, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=120)
    )
    assert any(x != jde.F0 for x in jde.Fi) or any(x != jde.CR0 for x in jde.CRi)


def test_de_gated_block_crossover():
    params = (
        Param.boolean("g"),
        Param.ordinal("d", (1, 2, 3), gate="g"),
        Param.ordinal("e", (1, 2, 3), gate="g"),
        Param.ordinal("x", (0, 1, 2)),
    )
    space = SearchSpace(params)
    de = DifferentialEvolution(space, 0, pop_size=4)
    x, v = [0, 2, 2, 0], [1, 0, 0, 2]  # gate off in x, on in v
    for seed in range(50):
        de.rng = random.Random(seed)
        u = de._crossover(x, v, CR=0.5)
        if u[0] == v[0]:
            assert u[1:3] == v[1:3]
        else:
            assert u[1:3] == x[1:3]


def test_de_rejects_bad_hyperparameters():
    space, _f = sphere()
    for kw in (
        {"variant": "best2"},
        {"adapt": "lshade"},
        {"F": 0.0},
        {"CR": 1.5},
        {"pop_size": 2},
        {"memory": 0},
        {"p_best": 0.0},
    ):
        with pytest.raises(ValueError):
            DifferentialEvolution(space, 0, **kw)


def test_de_immigrates_better_foreign_observations():
    space, f = sphere(levels=9, dims=4)
    de = DifferentialEvolution(space, 0, pop_size=6)
    ev = Evaluator(space, SyntheticBackend(f))
    run_search(space, de, ev, Budget(evaluations=12))
    opt = tuple((i * 3 + 2) % 9 for i in range(4))
    other = make_strategy("random", space, 9)
    t = other.trial(opt)
    (ob,), _ = ev.evaluate([t])
    de.tell([ob])
    for s in de._gen:
        if s["value"] is None:
            s["value"] = math.inf
    de._select()
    assert [list(opt)] == [g for g in de.pop if tuple(g) == opt]


# ---------------------------------------------------- hill-climb details
def test_hill_combination_resweep_finds_interacting_pair():
    """Each of a=1 and b=1 alone is slightly worse (within combo_tol);
    together they win. Single moves cannot see it; the re-sweep must."""
    params = tuple(Param.categorical(n, (0, 1)) for n in ("a", "b", "c", "d"))
    space = SearchSpace(params)

    def f(c):
        if c["a"] == 1 and c["b"] == 1:
            return 0.8
        return 1.0 + 0.005 * (c["a"] + c["b"]) + 0.002 * c["c"] + 0.05 * c["d"]

    def first_win(res):
        return next(i for i, o in enumerate(res.observations) if o.value == 0.8) + 1

    res = _run(space, f, "hill", 16, strategy_options={"start": (0, 0, 0, 0)})
    assert res.best_value == 0.8
    # center + 4 single moves, then the combinations of the near-neutral
    # moves c, a, b: (c,a), (c,b), (a,b)
    assert first_win(res) == 8
    # without the re-sweep the center's stages end without the pair: only a
    # restart (a random unseen config) can find it
    plain = HillClimb(space, 0, start=(0, 0, 0, 0), combo_knobs=0)
    ev = Evaluator(space, SyntheticBackend(f))
    res2 = run_search(space, plain, ev, Budget(evaluations=16))
    assert res2.best_value > 0.8 or plain.restarts >= 1
    assert plain.moves_made == 0 or res2.best_value == 0.8


def test_hill_probe_starts_from_best_of_stratified_sample():
    space, f = sphere(levels=9, dims=2)
    hc = HillClimb(space, 3, n_init=6)
    res = run_search(
        space, hc, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=7)
    )
    probe = [o for o in res.observations if not o.cached][:6]
    assert len({o.key for o in probe}) == 6
    best_probe = min(probe, key=lambda o: o.value)
    # the climb starts at the probe's best: its first own trial is a move of it
    seventh = [o for o in res.observations if not o.cached][6]
    assert space.distance(best_probe.trial.genome, seventh.trial.genome) <= 1.0
    assert all(o.key in hc.foreign for o in probe)
    # with a start the probe is skipped; inside a portfolio the climber has none
    hs = HillClimb(space, 3, start=(0, 0), n_init=6)
    assert hs._probed
    pf = make_strategy("portfolio", space, 0, members=("hill", "de"))
    climber = next(m for m in pf.members if isinstance(m, HillClimb))
    assert climber.n_init == 0 and climber._probed
    with pytest.raises(ValueError, match="n_init"):
        HillClimb(space, 0, n_init=-1)


def test_hill_restarts_from_foreign_finds_not_its_own_neighbors():
    """After a local optimum the climber restarts from a foreign config
    (warm start / other strategy's find) or a random unseen one, never from
    one of its own rejected neighbors (worse than an optimum it climbed)."""
    space, f = sphere(levels=9, dims=2)
    opt_key, _best, _n = brute_force_optimum(space, f)
    opt = next(g for g in space.enumerate() if space.key(g) == opt_key)
    far = tuple(0 if x > 4 else 8 for x in opt)

    class Recorder(HillClimb):
        restart_log: list

        def _set_center(self, key, genome, *, push):
            if not push and self.center is not None:
                self.restart_log.append((key, key in self.history, key in self.foreign))
            super()._set_center(key, genome, push=push)

    for init, expect_foreign in (([far], True), ((), False)):
        hc = Recorder(space, 0, start=opt, init=init, lookback=0)
        hc.restart_log = []
        run_search(
            space, hc, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=40)
        )
        assert hc.restarts >= 1 and hc.restart_log
        key, seen, foreign = hc.restart_log[0]
        if expect_foreign:
            assert key == space.key(far) and seen and foreign
        else:
            assert not seen  # a random unseen config, not an own neighbor


def test_hill_looks_back_and_restarts():
    space, f = sphere(levels=9, dims=2)
    hc = HillClimb(space, 0, start=(0, 0), lookback=2)
    run_search(space, hc, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=60))
    assert hc.moves_made >= 3 and hc.stack
    assert hc.best()[1] == 1.0


def test_hill_adopts_a_better_foreign_find():
    space, f = sphere(levels=9, dims=2)
    hc = HillClimb(space, 0, start=(0, 0))
    ev = Evaluator(space, SyntheticBackend(f))
    run_search(space, hc, ev, Budget(evaluations=3))
    other = make_strategy("random", space, 3)
    (ob,), _ = ev.evaluate([other.trial((2, 5))])
    hc.tell([ob])
    assert hc.center == space.key((2, 5))


# ----------------------------------------------------------- priors
def test_prior_seeds_tpe_and_gp(gemm):
    space, f, key, _best = gemm
    perfect = CallablePrior(f)
    for name in ("tpe", "gp"):
        res = _run(space, f, name, 6, prior=perfect)
        assert res.best_key == key, name


def test_wrong_prior_cannot_capture_the_search(gemm):
    space, f, _key, best = gemm
    inverted = CallablePrior(lambda c: 1.0 / f(c))
    res = _run(space, f, "tpe", 64, prior=inverted)
    assert best / f(res.best_config) >= 0.85


# --------------------------------------------- successive halving / racing
def test_successive_halving_promotes_the_true_best_under_noise():
    space, f = sphere(levels=9, dims=2)
    ladder = (
        Fidelity("r4", reps=4, cost=0.2),
        Fidelity("r16", reps=16, cost=0.5),
        Fidelity("full", reps=32),
    )
    sh = SuccessiveHalving(
        space,
        0,
        fidelities=(0, 1, -1),
        n_configs=27,
        eta=3,
        race=Race(alpha=0.05, margin=0.01),
    )
    backend = SyntheticBackend(
        lambda c: 1.0 + 0.02 * f(c), noise=0.02, drift=0.05, seed=3
    )
    res = run_search(
        space, sh, Evaluator(space, backend, ladder=ladder), Budget(evaluations=500)
    )
    assert sh.done and sh.winner is not None
    members = [k for k in sh.member_genomes]
    truth = {k: f(space.decode(sh.member_genomes[k])) for k in members}
    assert truth[sh.winner] == min(truth.values())
    assert any(v == "raced" for v in sh.dropped.values())
    assert res.evaluations < 27 * 3
    per_rung = [sum(1 for o in res.observations if o.fidelity == r) for r in range(3)]
    assert per_rung[0] == 27 and per_rung[1] <= 9 and per_rung[2] <= 3


def test_hyperband_runs_several_brackets():
    space, f = sphere(levels=9, dims=3)
    res = tune(
        space,
        SyntheticBackend(f, noise=0.01),
        strategy="hyperband",
        budget=Budget(evaluations=120),
        confirm=0,
    )
    assert res.evaluations <= 120
    fids = {o.fidelity for o in res.observations}
    assert fids == {0, 1, 2}


# --------------------------------------------------------- portfolio
class _Descender(Strategy):
    """Walks the sphere downhill one step per ask (a 'good' member)."""

    name = "good"

    def __init__(self, space, f):
        super().__init__(space, 0)
        self.f, self.g = f, (0,) * space.dims

    def ask(self, n=1):
        moves = [m for m in self.space.moves(self.g) if not self.seen(m)]
        if not moves:
            return []
        self.g = min(moves, key=lambda m: self.f(self.space.decode(m)))
        return [self.trial(self.g)]


class _Worst(Strategy):
    """Asks the worst unseen configs (a 'bad' member)."""

    name = "bad"

    def __init__(self, space, f):
        super().__init__(space, 1)
        self.order = sorted(space.enumerate(), key=lambda g: -f(space.decode(g)))

    def ask(self, n=1):
        while self.order:
            g = self.order.pop(0)
            if not self.seen(g):
                return [self.trial(g)]
        return []


def test_portfolio_allocates_to_the_productive_member():
    space, f = sphere(levels=40, dims=2, optimum=(39, 39))
    p = Portfolio(space, 0, members=[_Descender(space, f), _Worst(space, f)])
    run_search(space, p, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=60))
    alloc = p.allocation()
    assert alloc["good"]["asks"] > 2 * alloc["bad"]["asks"]
    assert alloc["good"]["wins"] > 0 and alloc["bad"]["asks"] >= 1


def test_portfolio_evaluates_warm_starts_first_and_repairs_them(gemm):
    space, f, key, _best = gemm
    opt_cfg = space.decode(next(g for g in space.enumerate() if space.key(g) == key))
    bad_cfg = {**opt_cfg, "tile_m": 256, "tile_n": 256}  # over the register budget
    warm = [WarmStart(opt_cfg, "selector"), WarmStart(bad_cfg, "neighbor:p1")]
    res = _run(space, f, "portfolio", 20, warm=warm)
    first = [o.trial.origin for o in res.observations[:2]]
    assert first == ["warm:selector", "warm:neighbor:p1+repaired"]
    assert res.best_key == key
    assert res.origins["warm:selector"]["best"] == 1


def test_portfolio_rejects_duplicate_members():
    space, _f = sphere()
    with pytest.raises(ValueError, match="repeat"):
        Portfolio(
            space,
            0,
            members=[make_strategy("de", space), make_strategy("de", space, 1)],
        )
    with pytest.raises(ValueError, match="portfolio"):
        make_strategy("portfolio", space, members=("portfolio",))


def test_unknown_strategy_name():
    space, _f = sphere()
    with pytest.raises(ValueError, match="expected one of"):
        make_strategy("annealing", space)


def test_trace_writer_in_memory():
    space, f = sphere(levels=5, dims=2)
    tw = TraceWriter()
    tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=5),
        trace=tw,
        confirm=0,
    )
    assert len(tw) == 1 + 5 + 1


def test_observation_json_round_trip():
    space, f = sphere(levels=5, dims=2)
    ev = Evaluator(space, SyntheticBackend(f, noise=0.01))
    s = make_strategy("random", space)
    (ob,), _ = ev.evaluate(s.ask(1))
    again = Observation.from_json(json.loads(json.dumps(ob.to_json())))
    assert again == ob


def test_cost_aware_portfolio_prefers_cheap_productive_members():
    """Two equally productive members; one's configs compile 8x slower. A
    cost-aware portfolio gives the cheap one most of the asks."""
    space = SearchSpace((Param.ordinal("x0", tuple(range(64))), Param.boolean("x1")))

    class Lane(Strategy):
        def __init__(self, space, lane):
            self.name = f"lane{lane}"
            super().__init__(space, lane)
            self.lane, self.k = lane, 0

        def ask(self, n=1):
            while self.k < 64:
                g = (self.k, self.lane)
                self.k += 1
                if not self.seen(g):
                    return [self.trial(g)]
            return []

    def f(c):
        return 100.0 - c["x0"]

    def cost(c):
        return 8.0 if c["x1"] else 1.0

    def run(aware):
        p = Portfolio(
            space, 0, members=[Lane(space, 0), Lane(space, 1)], cost_aware=aware
        )
        backend = SyntheticBackend(f, compile_seconds=cost)
        run_search(space, p, Evaluator(space, backend), Budget(evaluations=40))
        return p

    aware = run(True)
    assert aware.asks[0] > 2 * aware.asks[1]
    assert sum(aware.spent) < sum(run(False).spent)


def test_noise_raises_the_gain_threshold():
    space, f = sphere(levels=5, dims=2)
    hc = HillClimb(space, 0, min_gain=0.001)
    assert hc.gain_threshold(0.001, 10) == 0.001
    ev = Evaluator(space, SyntheticBackend(f, noise=0.05))
    run_search(space, hc, ev, Budget(evaluations=10))
    assert hc.noise is ev.noise
    assert hc.gain_threshold(0.001, 10) > 0.02
    assert hc.gain_threshold(0.001, 1000) < hc.gain_threshold(0.001, 10)


# ------------------------------------------------- exact-behaviour checks
def test_shade_memory_update_is_the_weighted_lehmer_mean():
    space, _f = sphere(levels=9, dims=2)
    de = DifferentialEvolution(space, 0, pop_size=4, adapt="shade")
    de.pop = [[0, 0], [1, 1], [2, 2], [3, 3]]
    de.fit = [4.0, 4.0, 4.0, 4.0]
    de.Fi, de.CRi = [0.5] * 4, [0.5] * 4
    de._gen = [
        {
            "i": 0,
            "g": [4, 4],
            "key": "a",
            "F": 0.2,
            "CR": 0.1,
            "value": 2.0,
            "uid": "x",
        },
        {
            "i": 1,
            "g": [5, 5],
            "key": "b",
            "F": 0.8,
            "CR": 0.9,
            "value": 1.0,
            "uid": "y",
        },
        {
            "i": 2,
            "g": [6, 6],
            "key": "c",
            "F": 0.6,
            "CR": 0.6,
            "value": 5.0,
            "uid": "z",
        },
        {
            "i": 3,
            "g": [7, 7],
            "key": "d",
            "F": 0.6,
            "CR": 0.6,
            "value": 4.0,
            "uid": "w",
        },
    ]
    de._select()
    w = [math.log(4.0 / 2.0), math.log(4.0 / 1.0)]
    w = [x / sum(w) for x in w]
    lehmer = (w[0] * 0.2**2 + w[1] * 0.8**2) / (w[0] * 0.2 + w[1] * 0.8)
    assert de.M_F[0] == pytest.approx(lehmer)
    assert de.M_CR[0] == pytest.approx(w[0] * 0.1 + w[1] * 0.9)
    assert de.k == 1
    # the equal trial replaced its parent (neutral drift) but is no success
    assert de.pop[3] == [7, 7] and de.pop[2] == [2, 2]


def test_failed_configs_are_never_rebuilt():
    space, f = sphere(levels=5, dims=2)
    backend = SyntheticBackend(
        f, fail=lambda c: Status.NUMERIC_FAIL if c["x0"] == 0 else None
    )
    ev = Evaluator(space, backend)
    s = make_strategy("random", space)
    (a,), _ = ev.evaluate([s.trial((0, 1))])
    (b,), _ = ev.evaluate([s.trial((0, 1))])
    assert a.status == b.status == Status.NUMERIC_FAIL
    assert b.cached and backend.compiles == 1 and backend.checks == 1


def test_confirmation_overrules_a_lucky_measurement():
    params = (Param.categorical("c", ("lucky", "steady", "slow")),)
    space = SearchSpace(params)
    calls = {}

    def measure(config, key, artifact, fidelity):
        n = calls[config["c"]] = calls.get(config["c"], 0) + 1
        if config["c"] == "lucky":
            return [0.5 if n == 1 else 2.0] * fidelity.reps
        return [1.0 if config["c"] == "steady" else 3.0] * fidelity.reps

    from rocke.dispatch.tuning.search import CallableBackend, CompileOutcome

    backend = CallableBackend(
        compile=lambda c, k: CompileOutcome(True),
        check=lambda c, k, a: (True, "ok"),
        measure=measure,
    )
    res = tune(
        space, backend, strategy="exhaustive", budget=Budget(evaluations=6), confirm=2
    )
    assert res.best_config == {"c": "steady"}
    assert res.best_value == 1.0
    assert res.confirmed["search_rank_winner"] != res.confirmed["winner"]


def test_gp_candidate_pool_uses_capped_repairs(gemm):
    # The GP's per-proposal pool repairs with a small check cap (a full repair
    # per pool point dominates searches on sparse-validity spaces).
    from rocke.dispatch.tuning.search.surrogate import POOL_REPAIR_CHECKS

    space, f, _k, _b = gemm
    seen = []
    orig = space.stratified

    def spy(rng, n, tries=8, *, repair_checks=None):
        seen.append((n, repair_checks))
        return orig(rng, n, tries, repair_checks=repair_checks)

    space.stratified = spy
    try:
        _run(space, f, "gp", 24, seed=1)
    finally:
        del space.stratified
    pools = [cap for n, cap in seen if n >= 64]
    assert pools and all(cap == POOL_REPAIR_CHECKS for cap in pools)


def test_confirmation_after_resume_remeasures_earlier_finalists(tmp_path):
    # Finalists measured before a resume have no artifact in the resumed
    # process; the confirmation round must rebuild them, not drop them (it
    # used to return nothing and keep the lucky first measurement).
    params = (Param.categorical("c", ("lucky", "steady", "slow")),)
    space = SearchSpace(params)
    calls = {}
    compiles = []

    def measure(config, key, artifact, fidelity):
        n = calls[config["c"]] = calls.get(config["c"], 0) + 1
        if config["c"] == "lucky":
            return [0.5 if n == 1 else 2.0] * fidelity.reps
        return [1.0 if config["c"] == "steady" else 3.0] * fidelity.reps

    from rocke.dispatch.tuning.search import CallableBackend, CompileOutcome

    def backend():
        return CallableBackend(
            compile=lambda c, k: compiles.append(c["c"]) or CompileOutcome(True),
            check=lambda c, k, a: (True, "ok"),
            measure=measure,
        )

    ck = tmp_path / "ck.json"
    first = tune(
        space,
        backend(),
        strategy="exhaustive",
        budget=Budget(evaluations=3),
        confirm=0,
        checkpoint=ck,
    )
    assert first.best_config == {"c": "lucky"} and first.evaluations == 3
    res = tune(
        space,
        backend(),
        strategy="exhaustive",
        budget=Budget(evaluations=6),
        confirm=2,
        checkpoint=ck,
        resume=True,
    )
    assert set(res.confirmed["finalists"]) == set(res.confirmed["values"])
    assert len(res.confirmed["values"]) == 2
    assert res.best_config == {"c": "steady"} and res.best_value == 1.0
    # the two finalists were rebuilt once each in the resumed process
    assert compiles.count("lucky") == 2 and compiles.count("steady") == 2


def test_hill_reopens_an_interrupted_center_before_restarting():
    params = (Param.ordinal("a", (0, 1, 2, 3, 4)), Param.categorical("b", (0, 1, 2)))
    space = SearchSpace(params)
    table = {
        (2, 0): 10.0,
        (1, 0): 9.0,
        (0, 0): 9.5,
        (1, 1): 9.6,
        (1, 2): 9.7,
        (3, 0): 9.8,
        (2, 1): 9.9,
        (2, 2): 5.0,
    }

    def f(c):
        return table.get((c["a"], c["b"]), 20.0)

    hc = HillClimb(space, 0, start=(2, 0), lookback=2, combo_knobs=0)
    run_search(space, hc, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=9))
    assert hc.best()[1] == 5.0
    assert hc.restarts == 0  # found by looking back at (2, 0), not by a restart


def test_portfolio_waits_for_warm_starts_before_members_ask():
    space, f = sphere(levels=9, dims=2)
    warm = [WarmStart({"x0": 1, "x1": 1}, "a"), WarmStart({"x0": 2, "x1": 2}, "b")]
    p = make_strategy("portfolio", space, 0, warm=warm)
    first = p.ask(4)
    assert [t.origin for t in first] == ["warm:a", "warm:b"]
    assert p.ask(4) == []  # still waiting on the warm results
    ev = Evaluator(space, SyntheticBackend(f))
    obs, _ = ev.evaluate(first)
    p.tell(obs)
    assert all(not t.origin.startswith("warm:") for t in p.ask(3))


def test_history_keeps_the_best_value_per_config():
    space, _f = sphere(levels=5, dims=2)
    s = make_strategy("random", space)
    t = s.trial((1, 1))
    key = space.key((1, 1))
    for v in (2.0, 1.0, 3.0):
        s.tell([Observation(trial=t, key=key, status=Status.OK, value=v, samples=(v,))])
    assert s.history[key] == 1.0
