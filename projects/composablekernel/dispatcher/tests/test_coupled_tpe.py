# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coupled multivariate TPE + local trust-region search tests."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from ck_search import (  # noqa: E402
    BuildPreference,
    Observation,
    Param,
    SearchSpace,
    Status,
    TablePrior,
    attach_build_preference,
    make_strategy,
)
from ck_search.surrogate import TPE  # noqa: E402


def _coupled_space():
    records = [
        {"pipeline": p, "tile_k": tk, "tile_m": tm, "pad": pad}
        for p, tk, tm, pad in itertools.product(
            ("mem", "compv3", "compv4"),
            (32, 64, 128),
            (64, 128, 256),
            (False, True),
        )
    ]
    return SearchSpace.from_configs(records, name="coupled_toy")


def _coupled_latency(rec):
    # Anti-diagonal coupling between pipeline and tile_k:
    # (compv4, 128) is global optimum (0.10 ms); (mem, 32) is second best (0.14 ms);
    # the cross-product chimera (compv4, 32) is pathological (2.50 ms).
    p, tk, tm = rec["pipeline"], rec["tile_k"], rec["tile_m"]
    if p == "compv4" and tk == 128:
        base = 0.10 + 0.01 * abs(tm - 128) / 64.0
    elif p == "mem" and tk == 32:
        base = 0.14 + 0.01 * abs(tm - 128) / 64.0
    elif p == "compv4" and tk == 32:
        base = 2.50
    elif p == "mem" and tk == 128:
        base = 2.20
    else:
        base = 0.45 + 0.02 * abs(tm - 128) / 64.0
    return base + (0.005 if rec["pad"] else 0.0)


def _settle(strategy, trials):
    for t in trials:
        rec = strategy.space.decode(t.genome)
        val = _coupled_latency(rec)
        strategy.tell([Observation(t, strategy.space.key(t.genome), Status.OK, value=val)])


def test_coupled_kde_avoids_univariate_chimera_trap():
    space = _coupled_space()
    seeded_obs = [
        ({"pipeline": "compv4", "tile_k": 128, "tile_m": 256, "pad": True}, 0.10),
        ({"pipeline": "compv4", "tile_k": 128, "tile_m": 64, "pad": False}, 0.11),
        ({"pipeline": "mem", "tile_k": 32, "tile_m": 64, "pad": False}, 0.12),
        ({"pipeline": "mem", "tile_k": 32, "tile_m": 256, "pad": True}, 0.125),
        ({"pipeline": "compv3", "tile_k": 64, "tile_m": 64, "pad": False}, 0.60),
        ({"pipeline": "compv3", "tile_k": 128, "tile_m": 64, "pad": True}, 0.70),
        ({"pipeline": "compv3", "tile_k": 128, "tile_m": 128, "pad": False}, 0.75),
        ({"pipeline": "mem", "tile_k": 64, "tile_m": 128, "pad": False}, 0.80),
        ({"pipeline": "mem", "tile_k": 128, "tile_m": 64, "pad": False}, 2.20),
        ({"pipeline": "mem", "tile_k": 128, "tile_m": 256, "pad": True}, 2.25),
    ]
    warm = [rec for rec, _ in seeded_obs]
    s_coupled = make_strategy("coupled_tpe", space, seed=1, warm_records=warm,
                              options={"n_startup": 1, "tr_budget": 0, "kappa": 0.5})
    s_listed = make_strategy("tpe", space, seed=1, warm_records=warm,
                             options={"candidate_pool": "listed", "n_startup": 1})
    for strat in (s_coupled, s_listed):
        asked = strat.ask(len(seeded_obs))
        assert len(asked) == len(seeded_obs)
        for t, (_rec, ms) in zip(asked, seeded_obs):
            strat.tell([Observation(t, space.key(t.genome), Status.OK, value=ms)])
    # Drain the single stratified anchor of coupled_tpe if present
    if s_coupled._startup:
        extra = s_coupled.ask(len(s_coupled._startup))
        _settle(s_coupled, extra)

    pick_listed = space.decode(s_listed.ask(1)[0].genome)
    pick_coupled = space.decode(s_coupled.ask(1)[0].genome)
    # Univariate listed TPE combines decoupled marginals (compv4 + tile_k=32) into the chimera
    assert (pick_listed["pipeline"], pick_listed["tile_k"]) == ("compv4", 32)
    assert (pick_coupled["pipeline"], pick_coupled["tile_k"]) in (
        ("compv4", 128),
        ("mem", 32),
    )


def test_pending_batch_repulsion_stays_local_without_pipeline_poisoning():
    space = _coupled_space()
    seeded_obs = [
        ({"pipeline": "compv4", "tile_k": 128, "tile_m": 64, "pad": False}, 0.10),
        ({"pipeline": "compv4", "tile_k": 128, "tile_m": 256, "pad": True}, 0.11),
        ({"pipeline": "mem", "tile_k": 64, "tile_m": 64, "pad": False}, 0.80),
        ({"pipeline": "mem", "tile_k": 32, "tile_m": 128, "pad": True}, 0.90),
        ({"pipeline": "compv3", "tile_k": 64, "tile_m": 128, "pad": False}, 1.10),
        ({"pipeline": "compv3", "tile_k": 32, "tile_m": 256, "pad": False}, 1.20),
    ]
    strat = make_strategy("coupled_tpe", space, seed=7,
                          warm_records=[r for r, _ in seeded_obs],
                          options={"n_startup": 1, "tr_budget": 0, "kappa": 0.5})
    asked = strat.ask(len(seeded_obs))
    for t, (_rec, ms) in zip(asked, seeded_obs):
        strat.tell([Observation(t, space.key(t.genome), Status.OK, value=ms)])
    if strat._startup:
        _settle(strat, strat.ask(len(strat._startup)))
    batch = strat.ask(3)
    assert len({space.key(t.genome) for t in batch}) == 3
    decoded = [space.decode(t.genome) for t in batch]
    assert sum(d["pipeline"] == "compv4" and d["tile_k"] == 128 for d in decoded) >= 2


def test_checkpoint_round_trip_matches_continuous_run_with_pending_and_tr_state():
    space = _coupled_space()
    a = make_strategy("coupled_tpe", space, seed=11, options={"n_startup": 4})
    _settle(a, a.ask(4))
    _settle(a, a.ask(4))
    pending = a.ask(3)

    b = make_strategy("coupled_tpe", space, seed=999, options={"n_startup": 4})
    payload = json.loads(json.dumps(a.state_dict(), allow_nan=False))
    b.load_state_dict(payload)

    _settle(a, pending)
    _settle(b, pending)
    for _ in range(5):
        ta, tb = a.ask(3), b.ask(3)
        assert ta == tb
        _settle(a, ta)
        _settle(b, tb)
    assert a.state_dict() == b.state_dict()
    with pytest.raises(ValueError, match="state for strategy"):
        TPE(space).load_state_dict(a.state_dict())


def test_warm_seeds_trim_redundant_blind_startup_and_use_trust_region():
    space = _coupled_space()
    recs = [space.decode(g) for g in space.listed()[:5]]
    strat = make_strategy("coupled_tpe", space, seed=2, warm_records=recs)
    first_six = strat.ask(6)
    assert [space.decode(t.genome) for t in first_six[:5]] == recs
    _settle(strat, first_six)
    # 7th trial is already model/trust-region guided rather than 4 more random probes
    assert strat._startup == []
    nxt = strat.ask(2)
    assert len(nxt) == 2


def test_prior_build_preference_gated_space_and_exhaustion():
    # Gated conditional space (unlisted) + prior + build preference
    space = SearchSpace([
        Param.boolean("use_tdm"),
        Param.ordinal("stages", [1, 2, 4], gate="use_tdm"),
        Param.categorical("pipe", ["a", "b", "c"]),
    ])
    prior = TablePrior({"True:4:c": 0.01},
                       lambda r: f"{r['use_tdm']}:{r.get('stages', 0)}:{r['pipe']}")
    strat = make_strategy("coupled_tpe", space, seed=5, prior=prior,
                          options={"n_startup": 3, "prior_weight": 50.0})
    pref = BuildPreference(rank_fraction=0.5, explore_probability=0.0)
    assert attach_build_preference(strat, pref) == 1

    seen = set()
    while True:
        asked = strat.ask(4)
        if not asked:
            break
        for t in asked:
            k = space.key(t.genome)
            assert k not in seen
            seen.add(k)
            cfg = space.decode(t.genome)
            fail = cfg["pipe"] == "a" and not cfg["use_tdm"]
            strat.tell([Observation(
                t, k, Status.COMPILE_ERROR if fail else Status.OK,
                value=float("inf") if fail else 1.0,
            )])
    assert strat.done and not strat.pending
    assert len(seen) == 12  # 3 pipes * (1 when False + 3 stages when True) = 12


@pytest.mark.parametrize("options, match", [
    ({"gamma": 0.0}, "gamma"),
    ({"kappa": -1.0}, "kappa"),
    ({"tr_budget": -1}, "tr_budget"),
    ({"combo_knobs": -1}, "combo_knobs"),
    ({"max_catalog": 0}, "max_catalog"),
])
def test_invalid_coupled_tpe_options_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        make_strategy("coupled_tpe", _coupled_space(), options=options)
