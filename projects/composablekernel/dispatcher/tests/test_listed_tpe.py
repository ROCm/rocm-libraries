# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Finite-pool acquisition: coverage, ownership, priors and checkpoint parity."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from ck_search import Observation, Param, SearchSpace, Status, TablePrior, make_strategy  # noqa: E402
from ck_search.surrogate import TPE  # noqa: E402


def space():
    return SearchSpace.from_configs([{"x": i, "pipeline": str(i % 3)} for i in range(96)])


def settle(strategy, trials):
    for t in trials:
        value = 1.0 + (strategy.space.decode(t.genome)["x"] - 47) ** 2
        strategy.tell([Observation(t, strategy.space.key(t.genome), Status.OK, value=value)])


def test_scores_every_unseen_member_and_registers_the_selected_trial():
    s = space()
    search = make_strategy("tpe", s, seed=3,
                           options={"candidate_pool": "listed", "n_startup": 2})
    settle(search, search.ask(2))
    selected = []

    class Capture:
        def choose(self, space, rng, genomes, scores, best):
            expected = {space.key(g) for g in space.listed()} - set(search.history) - search.pending_keys()
            assert {space.key(g) for g in genomes} == expected
            assert scores[best] == max(scores)
            selected.append(genomes[best])
            return genomes[best]

    search.build_preference = Capture()
    trials = search.ask(3)
    assert [t.genome for t in trials] == selected
    assert len(search.pending) == 3
    settle(search, trials)
    assert not search.pending
    assert len(search.history) == 5


def test_resume_matches_continuous_search_after_acquisition_and_pending_trials():
    s = space()
    options = {"candidate_pool": "listed", "n_startup": 2}
    search = make_strategy("tpe", s, seed=13, options=options)
    settle(search, search.ask(2))
    settle(search, search.ask(4))
    pending = search.ask(2)
    restored = make_strategy("tpe", s, seed=999, options=options)
    restored.load_state_dict(json.loads(json.dumps(search.state_dict(), allow_nan=False)))
    settle(search, pending)
    settle(restored, pending)
    for _ in range(6):
        a, b = search.ask(3), restored.ask(3)
        assert a == b
        settle(search, a)
        settle(restored, b)
    assert search.state_dict() == restored.state_dict()
    with pytest.raises(ValueError, match="state for strategy"):
        TPE(s).load_state_dict(search.state_dict())


def test_prior_is_applied_to_the_full_candidate_pool():
    s = space()
    records = [s.decode(g) for g in s.listed()]
    prior = TablePrior({str(i): 0.001 if i == 95 else 1.0 for i in range(96)},
                       lambda r: str(r["x"]))
    search = make_strategy("tpe", s, seed=5, warm_records=records[:2], prior=prior,
                           options={"candidate_pool": "listed", "n_startup": 2,
                                    "prior_weight": 100})
    settle(search, search.ask(2))
    assert s.decode(search.ask(1)[0].genome)["x"] == 95


def test_explicit_sampled_mode_preserves_default_trials_and_rng():
    s = space()
    a = make_strategy("tpe", s, seed=4)
    b = make_strategy("tpe", s, seed=4, options={"candidate_pool": "sampled"})
    for _ in range(8):
        ta, tb = a.ask(3), b.ask(3)
        assert ta == tb
        settle(a, ta)
        settle(b, tb)
    assert a.state_dict() == b.state_dict()


def test_listed_mode_has_the_same_startup_as_sampled_tpe():
    s = space()
    a = make_strategy("tpe", s, seed=7)
    b = make_strategy("tpe", s, seed=7, options={"candidate_pool": "listed"})
    ta, tb = a.ask(10), b.ask(10)
    assert [t.genome for t in ta] == [t.genome for t in tb]
    assert a.rng.getstate() == b.rng.getstate()


def test_exhaustion_and_failed_trials_do_not_repeat():
    s = SearchSpace.from_configs([{"x": i} for i in range(15)])
    search = make_strategy("tpe", s, options={"candidate_pool": "listed", "n_startup": 2})
    seen = set()
    while True:
        trials = search.ask(3)
        if not trials:
            break
        for t in trials:
            key = s.key(t.genome)
            assert key not in seen
            seen.add(key)
            x = s.decode(t.genome)["x"]
            search.tell([Observation(t, key, Status.COMPILE_ERROR if x % 4 == 0 else Status.OK,
                                     value=float("inf") if x % 4 == 0 else x + 1)])
    assert len(seen) == 15
    assert search.done and not search.pending


@pytest.mark.parametrize("options, message", [
    ({"candidate_pool": "invalid"}, "sampled or listed"),
    ({"candidate_pool": "listed", "n_candidates": 4}, "omit n_candidates"),
])
def test_invalid_pool_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        make_strategy("tpe", space(), options=options)


def test_unlisted_space_is_rejected():
    with pytest.raises(ValueError, match="nonempty listed space"):
        make_strategy("tpe", SearchSpace([Param.ordinal("x", [1, 2, 3])]),
                      options={"candidate_pool": "listed"})
