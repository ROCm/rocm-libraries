# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The parallel-compile hook: ``backend.precompile`` and ``ProcessCompilePool``.

The hook may only change *when* a round's compiles happen, never what a search
decides, so these tests compare a search with the hook against one without it,
and check which compiles the evaluator offers.
"""

from __future__ import annotations

import operator
import warnings

import pytest
from rocke.dispatch.tuning.search import (
    Budget,
    CallableBackend,
    CompileOutcome,
    Fidelity,
    Param,
    ProcessCompilePool,
    SearchSpace,
    tune,
)

N = 12


def _space():
    return SearchSpace([Param.ordinal("x", tuple(range(N)))], name="line")


def _backend(*, precompile=None, compiled=None):
    """x = 7 is fastest; ``compiled`` records every compile the evaluator makes."""

    def compile_(config, key):
        if compiled is not None:
            compiled.append(config["x"])
        return CompileOutcome(True, artifact=config["x"])

    def check_(config, key, artifact):
        return True, "ok"

    def measure_(config, key, artifact, fid):
        return [1.0 + abs(config["x"] - 7)] * fid.reps

    return CallableBackend(compile_, check_, measure_, precompile=precompile)


def _run(backend, *, batch=4, strategy="exhaustive", budget=N):
    return tune(
        _space(),
        backend,
        strategy=strategy,
        budget=Budget(evaluations=budget),
        batch=batch,
        confirm=0,
        ladder=(Fidelity(reps=2, warmup=0),),
    )


def test_precompile_is_offered_each_fresh_compile_exactly_once():
    rounds = []
    _run(_backend(precompile=lambda items: rounds.append([c["x"] for c, _k in items])))
    flat = [x for r in rounds for x in r]
    assert sorted(flat) == list(range(N))  # every config, none twice
    assert all(1 <= len(r) <= 4 for r in rounds)  # at most one round's ask


def test_precompile_does_not_change_what_the_search_finds():
    plain = _run(_backend())
    hooked = _run(_backend(precompile=lambda items: None))
    assert hooked.best_key == plain.best_key
    assert [(o.key, o.status, o.value) for o in hooked.observations] == [
        (o.key, o.status, o.value) for o in plain.observations
    ]


def test_precompile_sees_only_budgeted_fresh_work():
    seen = []
    _run(_backend(precompile=lambda items: seen.extend(c["x"] for c, _k in items)), budget=5)
    compiled = []
    _run(_backend(compiled=compiled), budget=5)
    assert seen == compiled  # offered exactly what the evaluator then compiled


def test_a_failing_precompile_only_loses_the_speedup():
    def boom(items):
        raise RuntimeError("pool died")

    compiled = []
    with pytest.warns(RuntimeWarning, match="compiling serially"):
        res = _run(_backend(precompile=boom, compiled=compiled))
    assert res.best_key == _run(_backend()).best_key
    assert sorted(compiled) == list(range(N))  # the serial path compiled everything


def test_precompile_is_skipped_when_compile_depends_on_the_fidelity():
    calls = []

    def compile_(config, key, fidelity):
        return CompileOutcome(True, artifact=config["x"])

    be = CallableBackend(
        compile_,
        lambda config, key, artifact: (True, "ok"),
        lambda config, key, artifact, fid: [1.0 + abs(config["x"] - 7)] * fid.reps,
        precompile=lambda items: calls.append(items),
    )
    _run(be)
    assert calls == []


# ------------------------------------------------------- ProcessCompilePool
# The pool pickles its worker for a spawned child, which cannot import this test
# module by name; a stdlib callable keeps the test independent of pytest's rootdir.
_WORKER = operator.itemgetter("x")


def test_pool_builds_and_serves_each_result_once():
    with ProcessCompilePool(_WORKER, workers=2) as pool:
        items = [({"x": x}, f"k{x}") for x in range(4)] + [({"y": 0}, "bad")]
        pool.precompile(items)
        got = {k: pool.take(k) for _c, k in items}
        assert got["k3"].ok and got["k3"].payload == 3
        assert not got["bad"].ok and "KeyError" in got["bad"].detail
        assert pool.take("k3") is None  # served once
        assert pool.take("never-built") is None


def test_pool_rejects_a_nonpositive_worker_count():
    with pytest.raises(ValueError, match="workers"):
        ProcessCompilePool(_WORKER, workers=0)


def test_search_with_the_pool_matches_a_serial_search():
    pool = ProcessCompilePool(_WORKER, workers=2)
    served = []
    try:

        def compile_(config, key):
            r = pool.take(key)
            served.append(r is not None)
            if r is None or not r.ok:
                return CompileOutcome(False, detail="not prebuilt")
            return CompileOutcome(True, artifact=r.payload)

        be = CallableBackend(
            compile_,
            lambda config, key, artifact: (True, "ok"),
            lambda config, key, artifact, fid: [1.0 + abs(config["x"] - 7)] * fid.reps,
            precompile=pool.precompile,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pooled = _run(be)
    finally:
        pool.close()
    plain = _run(_backend())
    assert all(served) and len(served) == N  # every compile came from the pool
    assert pooled.best_key == plain.best_key
    assert [o.value for o in pooled.observations] == [o.value for o in plain.observations]
