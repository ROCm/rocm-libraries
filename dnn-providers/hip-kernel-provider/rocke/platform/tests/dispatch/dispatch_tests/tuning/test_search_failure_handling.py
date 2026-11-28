# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Property and regression tests of the search engine's failure handling:
stale persistent caches, mixed-fidelity deadlocks, repair symmetry,
``choices``-axis geometry, batch-timeout isolation, soft timeouts, proxy
fidelities, prior scales, the tuning DB's serving policy and locking, the
driver's budget edges, and the DE / portfolio / Hyperband details."""

from __future__ import annotations

import json
import math
import random
import threading
import time

import pytest
from rocke.dispatch.tuning.axes import choices, values
from rocke.dispatch.tuning.search import (
    Budget,
    CallableBackend,
    CompileOutcome,
    DifferentialEvolution,
    Evaluator,
    Fidelity,
    HillClimb,
    Hyperband,
    Observation,
    Param,
    Portfolio,
    PriorScale,
    SearchSpace,
    Status,
    Strategy,
    SuccessiveHalving,
    SyntheticBackend,
    TunedRecord,
    TuningDB,
    default_ladder,
    make_strategy,
    run_search,
    tune,
)
from rocke.dispatch.tuning.search.params import CATEGORICAL, ORDINAL, params_from_axes
from rocke.dispatch.tuning.search.surrogate import GPEI, CallablePrior, prior_ranked
from rocke.dispatch.tuning.search.synthetic import gemm_like, sphere


# ------------------------------------------------------------ stale caches
class _Counting(SyntheticBackend):
    def __init__(self, f, *, env, bad=None, **kw):
        super().__init__(f, **kw)
        self.env = env
        self.bad = bad or set()

    def environment(self):
        return {"toolchain": self.env, "device": "toy"}

    def check(self, config, key, artifact):
        self.checks += 1
        if (config["x0"], config["x1"]) in self.bad:
            return False, "miscompiled by this toolchain"
        return True, "ok"


def _exhaust(space, backend, cache):
    return tune(
        space,
        backend,
        strategy="exhaustive",
        budget=Budget(evaluations=100),
        cache_path=cache,
        confirm=0,
    )


def test_cache_from_another_toolchain_is_dropped_and_rechecked(tmp_path):
    space, f = sphere(levels=4, dims=2)
    cache = tmp_path / "c.json"
    b1 = _Counting(f, env="comgr-A")
    _exhaust(space, b1, cache)
    assert b1.measures == 16 and b1.checks == 16
    # same toolchain: nothing is spent twice
    b2 = _Counting(f, env="comgr-A")
    _exhaust(space, b2, cache)
    assert b2.measures == 0 and b2.compiles == 0
    # a new compiler miscompiles (1, 1): the old timings are not reused, the
    # config is checked again and never timed
    b3 = _Counting(f, env="comgr-B", bad={(1, 1)})
    r3 = _exhaust(space, b3, cache)
    assert b3.checks == 16 and b3.measures == 15
    bad = [o for o in r3.observations if o.trial.genome == (1, 1)]
    assert bad and bad[0].status == Status.NUMERIC_FAIL and not bad[0].ok


def test_every_artifact_is_checked_in_every_process(tmp_path):
    space, f = sphere(levels=3, dims=2)
    cache = tmp_path / "c.json"
    b1 = _Counting(f, env="A")
    ev = Evaluator(space, b1, cache_path=cache)
    s = make_strategy("random", space)
    ev.evaluate([s.trial((0, 0))])
    # a new process (same toolchain): the timing is served, but any rebuild
    # (a confirmation re-measure) is checked again before it is timed
    b2 = _Counting(f, env="A", bad={(0, 0)})
    ev2 = Evaluator(space, b2, cache_path=cache)
    (hit,), _ = ev2.evaluate([s.trial((0, 0))])
    assert hit.cached and b2.checks == 0
    out = ev2.remeasure([hit.key], {hit.key: (0, 0)}, rounds=2)
    assert out[hit.key] == [] and b2.checks == 1 and b2.measures == 0


def test_other_machine_keeps_structural_failures_drops_timings(tmp_path):
    space, f = sphere(levels=3, dims=2)
    cache = tmp_path / "c.json"
    fail = lambda c: Status.COMPILE_ERROR if c["x0"] == 0 else None
    b1 = _Counting(f, env="A", fail=fail)
    ev = Evaluator(space, b1, cache_path=cache, environment={"node": 1})
    s = make_strategy("random", space)
    ev.evaluate([s.trial((0, 0)), s.trial((1, 1))])
    data = json.loads(cache.read_text())
    data["fingerprint"]["machine"] = "another-node"
    cache.write_text(json.dumps(data))
    b2 = _Counting(f, env="A", fail=fail)
    ev2 = Evaluator(space, b2, cache_path=cache, environment={"node": 1})
    obs, _ = ev2.evaluate([s.trial((0, 0)), s.trial((1, 1))])
    assert obs[0].cached and obs[0].status == Status.COMPILE_ERROR
    assert not obs[1].cached and obs[1].ok and b2.measures == 1


def test_transient_failures_are_not_persisted(tmp_path):
    space, _f = sphere(levels=3, dims=1)
    cache = tmp_path / "c.json"
    flaky = {"on": True}

    def measure(config, key, artifact, fidelity):
        if flaky["on"]:
            raise RuntimeError("driver hiccup")
        return [1.0] * fidelity.reps

    backend = CallableBackend(
        compile=lambda c, k: CompileOutcome(True),
        check=lambda c, k, a: (True, "ok"),
        measure=measure,
    )
    s = make_strategy("random", space)
    ev = Evaluator(space, backend, cache_path=cache)
    (a,), _ = ev.evaluate([s.trial((1,))])
    assert a.status == Status.RUN_ERROR
    (again,), _ = ev.evaluate([s.trial((1,))])
    assert again.cached  # within one process it is not retried
    flaky["on"] = False
    ev2 = Evaluator(space, backend, cache_path=cache)
    (b,), _ = ev2.evaluate([s.trial((1,))])
    assert b.ok and not b.cached


def test_environment_fingerprint_covers_toolchain_and_device():
    from rocke.dispatch.tuning.search.measure import toolchain_environment

    env = toolchain_environment()
    assert set(env) == {"toolchain", "machine"}
    assert env["toolchain"]["rocke"] and "ROCKE_LLVM_FLAVOR" in env["toolchain"]["env"]
    space, f = sphere(levels=3, dims=1)
    a = Evaluator(space, _Counting(f, env="A")).fingerprint()
    b = Evaluator(space, _Counting(f, env="B")).fingerprint()
    c = Evaluator(space, _Counting(f, env="A"), environment={"lib": 2}).fingerprint()
    assert a["tool"] != b["tool"] and a["tool"] != c["tool"]
    assert a == Evaluator(space, _Counting(f, env="A")).fingerprint()


def test_library_builders_enter_the_fingerprint(tmp_path, monkeypatch):
    import sys
    import types

    from rocke.dispatch.tuning.search.measure import loaded_builders_id

    for name in [
        m for m in sys.modules if m.split(".", 1)[0] in ("kernels", "builders")
    ]:
        monkeypatch.delitem(sys.modules, name)
    assert loaded_builders_id() == ""
    src = tmp_path / "toy_builder.py"
    src.write_text("X = 1")
    mod = types.ModuleType("kernels.toy_builder")
    mod.__file__ = str(src)
    monkeypatch.setitem(sys.modules, "kernels.toy_builder", mod)
    a = loaded_builders_id()
    src.write_text("X = 2")
    assert a and loaded_builders_id() != a


# --------------------------------------------------------- mixed fidelity
def test_cached_failure_answers_a_full_fidelity_ask_as_full():
    space, f = sphere(levels=5, dims=2)
    fail = lambda c: Status.NUMERIC_FAIL if c["x0"] == 0 else None
    ladder = default_ladder(8)
    ev = Evaluator(space, SyntheticBackend(f, fail=fail), ladder=ladder)
    s = make_strategy("random", space)
    (cheap,), _ = ev.evaluate([s.trial((0, 1), fidelity=0)])
    assert cheap.status == Status.NUMERIC_FAIL and not cheap.full
    (full,), _ = ev.evaluate([s.trial((0, 1), fidelity=-1)])
    assert full.cached and full.full and full.fidelity == len(ladder) - 1
    assert full.status == Status.NUMERIC_FAIL


def test_transient_failure_at_a_cheap_rung_is_remeasured_at_full():
    space, _f = sphere(levels=3, dims=1)
    calls = []

    def measure(config, key, artifact, fidelity):
        calls.append(fidelity.name)
        if fidelity.name == "r3":
            raise RuntimeError("cheap rung launch failed")
        return [1.0] * fidelity.reps

    backend = CallableBackend(
        compile=lambda c, k: CompileOutcome(True),
        check=lambda c, k, a: (True, "ok"),
        measure=measure,
    )
    ev = Evaluator(space, backend, ladder=default_ladder(8))
    s = make_strategy("random", space)
    (a,), _ = ev.evaluate([s.trial((1,), fidelity=0)])
    (b,), _ = ev.evaluate([s.trial((1,), fidelity=-1)])
    assert a.status == Status.RUN_ERROR and b.ok and calls == ["r3", "full"]


def test_portfolio_of_hyperband_and_de_does_not_deadlock_on_failures():
    """A config that fails at a cheap Hyperband rung must still answer DE's
    full-fidelity ask; DE used to wait for it forever ('no_proposals')."""
    space, f = gemm_like(rugged=0.1, salt="mix")

    def fail(c):
        if c["tile_k"] == 16 or c["pipeline"] == "interwave":
            return Status.NUMERIC_FAIL
        if c["warps"] == "1x4":
            return Status.COMPILE_ERROR
        return None

    for seed in range(3):
        backend = SyntheticBackend(f, noise=0.02, seed=seed, fail=fail)
        res = tune(
            space,
            backend,
            strategy="portfolio",
            members=("hyperband", "de"),
            ladder=default_ladder(8),
            budget=Budget(evaluations=120),
            seed=seed,
            confirm=0,
        )
        assert res.stopped == "evaluations", res.stopped
        assert res.evaluations == 120
        assert any(o.status != Status.OK for o in res.observations)


def test_de_init_member_failed_at_a_cheap_rung_does_not_stall_de():
    """Regression: a DE init member timed out (or failed)
    at fidelity 0; DE's full-fidelity ask was answered from the cache with
    full=False, DE ignored it and stayed in 'init' forever."""
    space, f = sphere(levels=5, dims=2)
    for status in (Status.NUMERIC_FAIL, Status.COMPILE_ERROR):
        fail = lambda c, st=status: (st if (c["x0"], c["x1"]) == (1, 1) else None)
        ev = Evaluator(space, SyntheticBackend(f, fail=fail), ladder=default_ladder(8))
        other = make_strategy("random", space, 7)
        (pre,), _ = ev.evaluate([other.trial((1, 1), fidelity=0)])
        assert not pre.full and not pre.ok
        de = DifferentialEvolution(space, 0, pop_size=4, init=[(1, 1)])
        res = run_search(space, de, ev, Budget(evaluations=30))
        assert de.phase == "evolve" and res.evaluations > 10
        assert res.stopped in ("evaluations", "strategy_done"), res.stopped


def test_hill_stage_completes_with_a_failure_cached_at_another_rung():
    space, f = sphere(levels=5, dims=2)
    fail = lambda c: (Status.SPILL if (c["x0"], c["x1"]) == (2, 3) else None)
    ladder = default_ladder(8)
    ev = Evaluator(space, SyntheticBackend(f, fail=fail), ladder=ladder)
    other = make_strategy("random", space, 1)
    ev.evaluate([other.trial((2, 3), fidelity=0)])  # fails at the cheap rung
    hc = HillClimb(space, 0, start=(2, 2), combo_knobs=0)
    res = run_search(space, hc, ev, Budget(evaluations=40))
    assert res.stopped in ("strategy_done", "evaluations")
    assert hc.best()[1] == 1.0


# ---------------------------------------------------------- repair symmetry
def test_repair_is_symmetric_in_symmetric_knobs():
    """tile_m and tile_n play identical roles; repair must not prefer the
    lower parameter index (it changed tile_m 487 times vs tile_n 52)."""
    params = (
        Param.ordinal("tile_m", (32, 64, 128, 256, 512)),
        Param.ordinal("tile_n", (32, 64, 128, 256, 512)),
        Param.ordinal("tile_k", (16, 32, 64)),
    )

    def ok(c):
        tm, tn, tk = c["tile_m"], c["tile_n"], c["tile_k"]
        return tm * tn <= 128 * 256 and (tm + tn) * tk <= 16384

    space = SearchSpace(params, is_valid=ok)
    changed = {p.name: 0 for p in params}
    rng = random.Random(0)
    for _round in range(4):
        for g in space.enumerate():
            if space.valid(g):
                continue
            r = space.repair(g, rng)
            for p, a, b in zip(params, g, r):
                changed[p.name] += a != b
    m, n = changed["tile_m"], changed["tile_n"]
    assert m > 50 and n > 50
    assert 0.8 < m / n < 1.25, changed


def test_equal_distance_moves_are_ordered_by_rng_only_when_given():
    space = SearchSpace((Param.ordinal("a", (0, 1, 2)), Param.ordinal("b", (0, 1, 2))))
    fixed = space.moves((1, 1))
    assert fixed == space.moves((1, 1))  # stable without an rng
    firsts = {space.moves((1, 1), rng=random.Random(s))[0] for s in range(20)}
    assert len(firsts) == 4  # every nearest move can come first


def test_archive_repair_ties_are_random():
    space = SearchSpace(
        (Param.ordinal("a", tuple(range(5))),), is_valid=lambda c: c["a"] in (0, 4)
    )
    picks = {
        space.repair((2,), random.Random(s), archive=[(0,), (4,)], max_checks=0)
        for s in range(20)
    }
    assert picks == {(0,), (4,)}


# ------------------------------------------------- choices-axis geometry
def test_choices_axis_merges_the_base_level():
    (p,) = params_from_axes((choices("tile", (64, 128, 256, 512)),), {"tile": 128})
    assert p.kind == ORDINAL
    assert p.levels == ((("tile", 64),), (), (("tile", 256),), (("tile", 512),))
    space = SearchSpace((p,))
    base = space.encode({})
    up_down = {space.decode(m).get("tile") for m in space.moves(base)}
    assert up_down == {64, 256}  # both steps exist from the base
    # no two levels decode to the same config
    assert len({json.dumps(space.decode((i,))) for i in range(p.size)}) == p.size


def test_values_axis_inserts_the_base_and_unknown_base_is_categorical():
    (p,) = params_from_axes((values("tile", 128, (64, 128, 256, 512)),), {"tile": 128})
    assert p.kind == ORDINAL and p.levels[1] == ()
    (q,) = params_from_axes((choices("tile", (64, 128, 256)),), {})
    assert q.kind == CATEGORICAL and q.levels[0] == ()
    (r,) = params_from_axes((values("tile", 96, (64, 128, 256)),), {"tile": 96})
    assert r.kind == ORDINAL and r.levels == (
        (("tile", 64),),
        (),
        (("tile", 128),),
        (("tile", 256),),
    )


# --------------------------------------------------- batches and timeouts
class _BatchBackend(SyntheticBackend):
    def __init__(self, f, culprit, mode, delay=0.0):
        super().__init__(f)
        self.culprit, self.mode, self.delay = culprit, mode, delay

    def measure_batch(self, items, fidelity):
        culprit = any(c["x0"] == self.culprit for c, _k, _a in items)
        if culprit and self.mode == "raise":
            raise RuntimeError("batch launch failed")
        time.sleep(self.delay * len(items))
        return super().measure_batch(items, fidelity)

    def measure(self, config, key, artifact, fidelity):
        if config["x0"] == self.culprit:
            raise RuntimeError("this config's launch fails")
        time.sleep(self.delay)
        return super().measure(config, key, artifact, fidelity)


def test_one_failing_config_does_not_fail_its_batch():
    space, f = sphere(levels=5, dims=1)
    ev = Evaluator(space, _BatchBackend(f, culprit=2, mode="raise"))
    s = make_strategy("random", space)
    obs, _ = ev.evaluate([s.trial((x,)) for x in range(5)])
    st = {o.trial.genome[0]: o.status for o in obs}
    assert st == {0: "ok", 1: "ok", 2: Status.RUN_ERROR, 3: "ok", 4: "ok"}
    # isolated measurements are not paired
    assert all(o.batch == -1 for o in obs)


def test_batch_timeout_scales_with_the_batch():
    space, f = sphere(levels=5, dims=1)
    ev = Evaluator(
        space, _BatchBackend(f, culprit=-1, mode="", delay=0.06), measure_timeout=0.2
    )
    s = make_strategy("random", space)
    obs, _ = ev.evaluate([s.trial((x,)) for x in range(5)])  # 0.3 s < 5 * 0.2 s
    assert all(o.ok for o in obs) and len({o.batch for o in obs}) == 1


class _Hang(SyntheticBackend):
    def __init__(self, f, hang_s):
        super().__init__(f)
        self.hang_s = hang_s
        self.threads = set()
        self.recovered = 0

    def compile(self, config, key):
        self.threads.add(threading.get_ident())
        return super().compile(config, key)

    def measure(self, config, key, artifact, fidelity):
        self.threads.add(threading.get_ident())
        if config["x0"] == 0:
            time.sleep(self.hang_s)
        return super().measure(config, key, artifact, fidelity)

    def recover(self):
        self.recovered += 1


def test_measure_timeout_makes_the_evaluator_dirty_until_the_call_ends():
    space, f = sphere(levels=4, dims=1)
    backend = _Hang(f, hang_s=0.6)
    ev = Evaluator(space, backend, measure_timeout=0.1, compile_timeout=5.0)
    s = make_strategy("exhaustive", space)
    res = run_search(space, s, ev, Budget(evaluations=10))
    assert res.stopped == "evaluator_dirty"
    timed_out = [o for o in res.observations if o.status == Status.TIMEOUT]
    assert len(timed_out) == 1 and backend.recovered == 1
    # nothing was timed while the hung call could still be running
    assert not any(o.ok for o in res.observations if o.trial.genome[0] != 0)
    time.sleep(0.8)
    obs, left = ev.evaluate([s.trial((1,)), s.trial((2,))])
    assert not left and all(o.ok for o in obs) and not ev.dirty
    # calls with a limit run on one worker thread (thread-local GPU state)
    workers = backend.threads - {threading.get_ident()}
    assert len(workers) == 2  # the abandoned worker, then its replacement
    ev.close()


# ---------------------------------------------------------- proxy fidelity
class _ProxyBackend(SyntheticBackend):
    def __init__(self, f):
        super().__init__(f)
        self.built: list[tuple[str, str]] = []
        self.checked: list[tuple[str, str]] = []

    def compile(self, config, key, fidelity):
        self.built.append((key, fidelity.proxy))
        return CompileOutcome(True, artifact=(key, fidelity.proxy))

    def check(self, config, key, artifact, fidelity):
        assert artifact == (key, fidelity.proxy)
        self.checked.append((key, fidelity.proxy))
        if fidelity.proxy == "small" and config["x0"] == 1:
            return False, "the proxy specialization is wrong"
        return True, "ok"


def test_proxy_rungs_compile_and_check_their_own_artifact():
    space, f = sphere(levels=3, dims=1)
    ladder = (Fidelity("p", reps=2, proxy="small", cost=0.1), Fidelity("full", reps=4))
    backend = _ProxyBackend(f)
    ev = Evaluator(space, backend, ladder=ladder)
    s = make_strategy("random", space)
    obs, _ = ev.evaluate([s.trial((x,), fidelity=0) for x in range(3)])
    assert [o.status for o in obs] == ["ok", Status.NUMERIC_FAIL, "ok"]
    full, _ = ev.evaluate([s.trial((x,), fidelity=-1) for x in range(3)])
    # the proxy's numeric failure does not fail the real problem
    assert all(o.ok for o in full)
    keys = {space.key((x,)) for x in range(3)}
    assert set(backend.built) == {(k, p) for k in keys for p in ("small", "")}
    assert len(backend.checked) == 6
    ev.evaluate([s.trial((2,), fidelity=0)])  # cached: no second build/check
    assert len(backend.built) == 6 and len(backend.checked) == 6


def test_proxy_ladder_needs_a_fidelity_aware_backend():
    space, f = sphere(levels=3, dims=1)
    ladder = (Fidelity("p", reps=2, proxy="small"), Fidelity("full", reps=4))
    with pytest.raises(ValueError, match="proxy"):
        Evaluator(space, SyntheticBackend(f), ladder=ladder)


# ------------------------------------------------------------- prior scale
def test_prior_scale_is_fixed_once_and_kept():
    space, _f = sphere(levels=5, dims=1)
    pred = {0: 4.0, 1: 2.0, 2: 1.0, 3: -1.0, 4: 3.0}
    prior = CallablePrior(lambda c: pred[c["x0"]])
    scale = PriorScale(prior)
    first = scale.values(space, [(0,), (1,), (2,)])
    assert scale.mode == "log" and first == pytest.approx([math.log(4), math.log(2), 0])
    # a later set with a non-positive prediction stays on the log scale
    later = scale.values(space, [(2,), (3,)])
    assert later[0] == pytest.approx(0.0) and math.isnan(later[1])
    # the other way round: a mixed first set fixes the linear scale
    lin = PriorScale(prior)
    assert lin.values(space, [(2,), (3,)]) == [1.0, -1.0] and lin.mode == "linear"
    assert lin.values(space, [(0,)]) == [4.0]
    assert PriorScale(CallablePrior(lambda c: 1.0, scale="linear")).mode == "linear"
    with pytest.raises(ValueError, match="mode"):
        PriorScale(prior, mode="sqrt")


def test_prior_ranked_and_gp_share_one_scale():
    space, f = gemm_like(rugged=0.1, salt="p")
    rank = prior_ranked(PriorScale(CallablePrior(f)), space, random.Random(0), pool=64)
    assert len(rank) > 0
    gp = GPEI(space, 0, prior=CallablePrior(f), n_startup=4)
    run_search(space, gp, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=10))
    assert gp.prior_scale.mode == "log"
    st = json.loads(json.dumps(gp.state_dict()))
    assert st["own"]["prior_scale"] == "log"
    again = GPEI(space, 0, prior=CallablePrior(f), n_startup=4)
    again.load_state_dict(st)
    assert again.prior_scale.mode == "log"


# ------------------------------------------------------------- tuning DB
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
        "toolchain": "t1",
    }
    d.update(kw)
    return TunedRecord(**d)


def test_db_does_not_compare_raw_times_across_sessions(tmp_path):
    db = TuningDB(tmp_path / "db.json")
    assert db.put(_rec(speedup=1.10, p_value=0.001))
    # faster raw time, no significant speedup: kept out
    assert not db.put(_rec(tuning_id="v@k2", config_key="k2", value=0.5))
    # a significant but smaller speedup does not displace a larger one
    assert not db.put(
        _rec(tuning_id="v@k2", config_key="k2", value=0.9, speedup=1.05, p_value=0.01)
    )
    # a larger significant speedup does
    assert db.put(
        _rec(tuning_id="v@k3", config_key="k3", value=2.0, speedup=1.2, p_value=0.01)
    )
    assert db.get("f", "gfx950", "p", "c").config_key == "k3"


def test_db_replaces_stale_and_other_toolchain_records(tmp_path):
    db = TuningDB(tmp_path / "db.json")
    db.put(_rec(speedup=1.3, p_value=0.001))
    other = _rec(tuning_id="v@k2", config_key="k2", value=9.0)
    assert not db.put(other)
    assert db.put(other, stale=True)
    assert db.put(_rec(tuning_id="v@k4", config_key="k4", value=9.0, toolchain="t2"))
    assert db.get("f", "gfx950", "p", "c").config_key == "k4"


def test_db_serves_only_significant_picks():
    db = TuningDB(None, alpha=0.05, min_speedup=1.01)
    db.put(_rec(speedup=1.005, p_value=0.001))  # within the margin
    assert db.get("f", "gfx950", "p", "c", servable=True) is None
    db.put(_rec(speedup=1.2, p_value=0.2), force=True)  # not significant
    assert db.get("f", "gfx950", "p", servable=True) is None
    db.put(_rec(speedup=1.2, p_value=0.01), force=True)
    assert db.get("f", "gfx950", "p", "c", servable=True) is not None


def test_db_concurrent_writers_lose_nothing(tmp_path):
    path = tmp_path / "db.json"
    errors = []

    def writer(i):
        try:
            db = TuningDB(path)
            for j in range(5):
                db.put(_rec(problem=f"p{i}-{j}", speedup=1.1, p_value=0.01))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(TuningDB(path)) == 40
    assert not (tmp_path / "db.json.lock").exists()


def test_record_result_uses_confirmation_samples():
    from rocke.dispatch.tuning.search import record_result

    space, f = sphere(levels=5, dims=2)
    res = tune(
        space,
        SyntheticBackend(f, noise=0.01),
        strategy="random",
        budget=Budget(evaluations=30),
        confirm=2,
        confirm_rounds=3,
        baseline=(0, 0),
    )
    c = res.confirmed
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
    assert rec.value == c["values"][c["winner"]]
    assert rec.samples == c["samples"][c["winner"]] == 3 * 20
    assert rec.speedup == pytest.approx(c["speedup"]) and rec.speedup > 1.0
    assert rec.rounds == 3


# -------------------------------------------------------- driver budgets
def test_confirmation_is_skipped_when_the_budget_cannot_pay_for_it():
    space, f = sphere(levels=5, dims=2)
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=6),
        confirm=3,
        confirm_rounds=2,
    )
    assert res.confirmed.get("skipped") and res.evaluations == 6
    assert res.stopped == "evaluations"


def test_confirmation_reserves_evaluations_and_cost():
    space, f = sphere(levels=9, dims=2)
    res = tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(evaluations=30),
        confirm=3,
        confirm_rounds=2,
    )
    assert res.evaluations <= 30 and res.confirmed["winner"]
    assert sum(res.confirmed["rounds"].values()) == 6
    res2 = tune(
        space,
        SyntheticBackend(f),
        strategy="random",
        budget=Budget(cost=20.0),
        confirm=3,
        confirm_rounds=2,
    )
    assert res2.meter.cost <= 20.0 + 1e-9 and res2.confirmed["winner"]


def test_confirmation_reserves_time_under_a_seconds_budget():
    space, f = sphere(levels=9, dims=2)
    now = [0.0]

    def clock():
        return now[0]

    class Timed(SyntheticBackend):
        def measure_batch(self, items, fidelity):
            now[0] += 1.0 * len(items)
            return super().measure_batch(items, fidelity)

        def measure(self, config, key, artifact, fidelity):
            now[0] += 1.0
            return super().measure(config, key, artifact, fidelity)

    ev = Evaluator(space, Timed(f), clock=clock)
    res = run_search(
        space,
        make_strategy("random", space),
        ev,
        Budget(seconds=40.0),
        confirm=3,
        confirm_rounds=2,
        clock=clock,
    )
    assert res.confirmed["winner"]
    assert now[0] <= 40.0 + 1.0  # the confirmation fit in the budget


def test_tune_builds_a_ladder_for_a_multi_fidelity_instance():
    space, f = sphere(levels=9, dims=2)
    sh = SuccessiveHalving(space, 0, fidelities=(0, 1, -1), n_configs=9)
    res = tune(
        space,
        SyntheticBackend(f),
        strategy=sh,
        budget=Budget(evaluations=40),
        confirm=0,
    )
    assert {o.fidelity for o in res.observations} == {0, 1, 2}
    bad = SuccessiveHalving(space, 0, fidelities=(0, 5, -1))
    with pytest.raises(ValueError, match="fidelities"):
        tune(space, SyntheticBackend(f), strategy=bad, budget=Budget(evaluations=10))


# ------------------------------------------------------- strategy details
def test_de_categorical_terms_do_not_override_each_other():
    """current-to-pbest has two categorical terms. With the take
    probability capped at 0.5 and the first firing term winning, a gene
    keeps the target's level about a quarter of the time (it never did
    with F = 1, when the second term always overwrote the first)."""
    space = SearchSpace((Param.categorical("c", tuple("abcdef")),))
    de = DifferentialEvolution(space, 0, pop_size=4, reset_prob=0.0)
    de.pop = [[0], [1], [2], [3]]
    de.fit = [4.0, 1.0, 3.0, 2.0]
    de.archive = []
    stays = 0
    for seed in range(800):
        de.rng = random.Random(seed)
        stays += de._mutant(0, F=1.0)[0] == 0.0
    assert 0.17 < stays / 800 < 0.33


def test_de_pop_size_follows_valid_configs_and_reports_done():
    rows = [{"a": i, "b": j} for i in range(3) for j in range(2)][:5]
    space = SearchSpace.from_configs(rows)
    de = DifferentialEvolution(space, 0)
    assert de.pop_size == 4  # min(..., max(4, 5 // 2))
    f = lambda c: 1.0 + c["a"] + 0.5 * c["b"]
    res = run_search(
        space, de, Evaluator(space, SyntheticBackend(f)), Budget(evaluations=50)
    )
    assert de.done and res.stopped == "strategy_done"


def test_de_selection_tolerates_noise_only_when_noise_was_seen():
    space, f = sphere(levels=5, dims=2)
    de = DifferentialEvolution(space, 0, pop_size=4)
    assert de._noise_gain() == 0.0
    ev = Evaluator(space, SyntheticBackend(f, noise=0.05))
    run_search(space, de, ev, Budget(evaluations=12))
    assert de._noise_gain() > 0.05


def test_portfolio_decays_idle_arms():
    space, _f = sphere(levels=5, dims=2)
    p = Portfolio(
        space,
        0,
        members=[make_strategy("random", space), make_strategy("de", space)],
        discount=0.5,
    )
    p.alpha, p.beta = [5.0, 1.0], [1.0, 1.0]
    p._reward(1, False)
    assert p.alpha[0] == pytest.approx(3.0)  # the idle arm decayed too


def test_hyperband_and_sh_uids_never_collide_in_a_portfolio():
    space, f = sphere(levels=9, dims=2)
    fids = (0, 1, -1)
    pf = Portfolio(
        space,
        0,
        members=[
            SuccessiveHalving(space, 0, fidelities=fids, n_configs=9),
            Hyperband(space, 0, fidelities=fids),  # bracket 0 is SH seed 0 too
        ],
    )
    ev = Evaluator(space, SyntheticBackend(f), ladder=default_ladder(8))
    seen = set()
    for _ in range(30):
        ts = pf.ask(3)
        if not ts:
            break
        assert not ({t.uid for t in ts} & seen)
        seen |= {t.uid for t in ts}
        obs, _ = ev.evaluate(ts)
        pf.tell(obs)
    assert len(seen) > 10


def test_sh_pools_only_calls_of_the_same_fidelity_pool():
    space, f = sphere(levels=9, dims=2)
    ladder = (
        Fidelity("cold", reps=3, cold=True, cost=0.2),
        Fidelity("hot", reps=8, cost=1.0),
    )
    sh = SuccessiveHalving(space, 0, fidelities=(0, -1), n_configs=6, eta=2)
    run_search(
        space,
        sh,
        Evaluator(space, SyntheticBackend(f), ladder=ladder),
        Budget(evaluations=30),
    )
    assert sh.done
    for k in sh.members:
        pools = [c[2] for c in sh.calls[k]]
        assert pools == ["|cold", "|hot"]
        assert len(sh._race_calls(k)) == 1  # the cold call is not pooled


def test_make_strategy_names_are_class_names():
    space, _f = sphere()
    for name in ("de", "jde", "de_rand"):
        s = make_strategy(name, space)
        assert s.name == name and type(s).name == name
    assert type(make_strategy("jde", space)).__name__ == "JDE"


def test_search_configs_refuses_configs_that_flatten_alike():
    from rocke.dispatch.tuning.search import search_configs

    configs = [{"a": 1}, {"a": 1}, {"a": 2}]
    with pytest.raises(ValueError, match="flatten"):
        search_configs(
            configs,
            measure=lambda c, fid: 1.0,
            check=lambda c: True,
            flatten=lambda c: c,
            budget=Budget(evaluations=3),
        )


def test_hyperband_state_keeps_its_uid_prefix():
    space, f = sphere(levels=9, dims=2)
    hb = Hyperband(space, 3, fidelities=(0, 1, -1))
    ev = Evaluator(space, SyntheticBackend(f), ladder=default_ladder(8))
    run_search(space, hb, ev, Budget(evaluations=20))
    again = Hyperband(space, 3, fidelities=(0, 1, -1))
    again.load_state_dict(json.loads(json.dumps(hb.state_dict())))
    assert again.inner.uid_prefix == hb.inner.uid_prefix
    assert again.inner.uid_prefix.startswith("hyperband:3:b")


def test_observation_carries_its_pool():
    space, f = sphere(levels=3, dims=1)
    ladder = (Fidelity("c", reps=2, cold=True), Fidelity("h", reps=2))
    ev = Evaluator(space, SyntheticBackend(f), ladder=ladder)
    s = make_strategy("random", space)
    (a,), _ = ev.evaluate([s.trial((1,), fidelity=0)])
    assert a.pool == "|cold"
    assert Observation.from_json(json.loads(json.dumps(a.to_json()))) == a


def test_noise_model_learns_call_noise_from_repeated_calls():
    space, f = sphere(levels=9, dims=1)
    ev = Evaluator(space, SyntheticBackend(f, noise=0.001, call_noise=0.03, seed=1))
    s = make_strategy("random", space)
    keys = [space.key((x,)) for x in range(9)]
    ev.evaluate([s.trial((x,)) for x in range(9)])
    assert ev.noise.call_sigma is None
    ev.remeasure(keys, {k: (x,) for x, k in enumerate(keys)}, rounds=3)
    sigma = ev.noise.call_sigma
    assert sigma is not None and 0.015 < sigma < 0.06
    # the strategy's gain threshold now reflects the per-call noise
    st = Strategy(space, 0)
    st.attach_noise(ev.noise)
    assert st.gain_threshold(0.001, 1000) >= 2 * sigma - 1e-12
