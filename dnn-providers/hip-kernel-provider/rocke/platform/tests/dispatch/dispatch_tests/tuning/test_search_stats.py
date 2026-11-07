# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Racing statistics: tests against hand-computed and textbook values, and
the false-elimination rate of :class:`Race` under the null -- with
per-call noise (the run-to-run part reps do not average away), a leader
picked from the same data, and repeated looks."""

from __future__ import annotations

import math
import random

import pytest
from rocke.dispatch.tuning.search.stats import (
    NoiseModel,
    Race,
    holm,
    median,
    normal_sf,
    paired_t_greater,
    rel_spread,
    student_t_sf,
    welch_greater,
    wilcoxon_greater,
)


def test_median_and_spread():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 2, 3]) == 2.5
    assert median([]) == math.inf
    xs = [math.exp(0.01 * z) for z in (-1, 0, 1, 0.5, -0.5)]
    assert 0.004 < rel_spread(xs) < 0.02
    assert rel_spread([1.0]) == 0.0


def test_wilcoxon_exact_small_cases():
    # all 5 differences positive: P(W+ = 15) = 1/32
    assert wilcoxon_greater([1, 2, 3, 4, 5]) == pytest.approx(1 / 32)
    # all negative: P(W+ >= 0) = 1
    assert wilcoxon_greater([-1, -2, -3]) == pytest.approx(1.0)
    # zeros dropped; shift applied before ranking
    assert wilcoxon_greater([0.0, 0.0]) == 1.0
    assert wilcoxon_greater([1.1, 1.2, 1.3], shift=1.0) == pytest.approx(1 / 8)
    # one negative of the smallest magnitude out of 4: W+ = 2+3+4 = 9 of 10;
    # sums >= 9 are {9 (drop 1), 10 (all)} -> 2/16
    assert wilcoxon_greater([-0.1, 0.2, 0.3, 0.4]) == pytest.approx(2 / 16)


def test_wilcoxon_ties_and_approximation_agree_at_the_boundary():
    rng = random.Random(5)
    d = [rng.gauss(0.2, 1.0) for _ in range(60)]
    exact = wilcoxon_greater(d)
    approx = wilcoxon_greater(d + [rng.gauss(0.2, 1.0)])
    assert abs(exact - approx) < 0.1
    tied = [1, 1, 1, 2, 2, -1]
    p = wilcoxon_greater(tied)
    assert 0.0 < p < 0.2


def test_student_t_sf_reference_values():
    assert student_t_sf(0.0, 7) == pytest.approx(0.5)
    # two-sided p for t=2.0, df=10 is 0.0734 (tables)
    assert 2 * student_t_sf(2.0, 10) == pytest.approx(0.07339, abs=2e-4)
    # t=2.228 is the 97.5% quantile at df=10
    assert student_t_sf(2.228, 10) == pytest.approx(0.025, abs=2e-4)
    assert student_t_sf(-2.0, 10) == pytest.approx(1 - student_t_sf(2.0, 10))
    # large df -> normal
    assert student_t_sf(1.96, 1e6) == pytest.approx(0.025, abs=1e-3)


def test_welch_greater():
    a = [10.0, 10.1, 9.9, 10.2, 10.0]
    b = [9.0, 9.1, 8.9, 9.0, 9.2]
    assert welch_greater(a, b) < 1e-4
    assert welch_greater(b, a) > 0.999
    assert welch_greater(a, b, shift=2.0) > 0.99
    assert welch_greater([1.0], [2.0]) == 1.0


def test_holm_reference():
    # classic example: p = (0.01, 0.04, 0.03, 0.005)
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def _calls(rng, truths, rounds, *, rep, call, drift, reps=20, paired=True):
    """One call per config and round: the median of ``reps`` reps with a
    per-call factor (``call``), a per-rep factor (``rep``) and, when
    ``paired``, a per-round drift factor shared by every config of the
    round (batch = round). Unpaired calls get batch -1 and their own drift."""
    out = {k: [] for k in truths}
    for r in range(rounds):
        shared = math.exp(rng.gauss(0, drift))
        for k, t in truths.items():
            f = shared if paired else math.exp(rng.gauss(0, drift))
            c = math.exp(rng.gauss(0, call))
            reps_ = sorted(t * f * c * math.exp(rng.gauss(0, rep)) for _ in range(reps))
            out[k].append((r if paired else -1, reps_[len(reps_) // 2]))
    return out


def test_race_paired_drops_slower_configs_under_drift():
    rng = random.Random(0)
    truths = {"best": 1.0, "slow": 1.05, "slower": 1.2}
    calls = _calls(rng, truths, 8, rep=0.005, call=0.003, drift=0.1)
    race = Race(alpha=0.05, margin=0.01)
    assert set(race.eliminate(calls)) == {"slow", "slower"}
    # the same calls without their batch tags: the drift hides the 5% gap
    unpaired = {k: [(-1, v) for _b, v in c] for k, c in calls.items()}
    assert "slow" not in race.eliminate(unpaired)


def test_race_never_drops_the_leader_or_thin_samples():
    race = Race(min_samples=3)
    calls = {"a": [(-1, 1.0), (-1, 1.0)], "b": [(-1, 2.0), (-1, 2.0)]}
    # two calls each, unpaired, no learned per-call noise: cannot tell
    assert race.eliminate(calls) == []
    # with a learned per-call sigma the z test can
    assert race.eliminate(calls, sigma=0.02) == ["b"]
    assert race.leader(calls) == "a"
    # identical values everywhere: nobody is dropped, the leader never
    same = {k: [(0, 1.0), (1, 1.0), (2, 1.0)] for k in "abc"}
    assert race.eliminate(same) == []


def test_race_looks_split_alpha():
    calls = {
        "a": [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        "b": [(0, 1.03), (1, 1.01), (2, 1.04), (3, 1.012)],
    }
    race = Race(alpha=0.05, margin=0.0)
    p = race.pvalue(calls["b"], calls["a"])
    assert p is not None and p < 0.05
    assert race.eliminate(calls) == ["b"]
    # the same comparison repeated at many rungs must clear alpha / looks
    looks = math.ceil(0.05 / p) + 1
    assert race.eliminate(calls, looks=looks) == []


def _false_rate(trials, make, race, **kw):
    rng = random.Random(11)
    wrong = 0
    for _ in range(trials):
        if race.eliminate(make(rng), **kw):
            wrong += 1
    return wrong / trials


def test_race_false_elimination_rate_under_per_call_noise():
    """Four identical configs, 2% per-call and 0.5% per-rep noise, 20 reps
    (a simulation). Treating the reps of one call as samples
    (Welch on reps) drops a config almost every time; racing on calls with
    the learned per-call sigma stays near alpha."""
    truths = {k: 1.0 for k in "abcd"}
    race = Race(alpha=0.05, margin=0.0)

    def reps_as_samples(rng):
        out = {}
        for k in truths:
            c = math.exp(rng.gauss(0, 0.02))
            out[k] = [c * math.exp(rng.gauss(0, 0.005)) for _ in range(20)]
        return out

    # the old, anti-conservative reading: reps treated as independent calls
    rng = random.Random(5)
    old_wrong = 0
    for _ in range(200):
        s = reps_as_samples(rng)
        calls = {k: [(-1, x) for x in v] for k, v in s.items()}
        old_wrong += bool(race.eliminate(calls))
    assert old_wrong / 200 > 0.5

    def one_call(rng):
        return _calls(rng, truths, 1, rep=0.005, call=0.02, drift=0.0, paired=False)

    sigma = math.sqrt(0.02**2 + (1.2533 * 0.005) ** 2 / 20)
    assert _false_rate(600, one_call, race, sigma=sigma) <= 0.07

    def three_calls(rng):
        return _calls(rng, truths, 3, rep=0.005, call=0.02, drift=0.0, paired=False)

    assert _false_rate(600, three_calls, race) <= 0.07

    def paired(rng):
        return _calls(rng, truths, 4, rep=0.005, call=0.02, drift=0.05)

    assert _false_rate(600, paired, race) <= 0.07


def test_race_leader_selection_does_not_inflate_errors():
    """Eight identical configs: the minimum of noisy calls is biased low,
    so testing everyone against it inflates false drops. Holm over every
    ordered pair keeps the family-wise rate near alpha."""
    truths = {f"k{i}": 1.0 for i in range(8)}
    race = Race(alpha=0.05, margin=0.0)

    def make(rng):
        return _calls(rng, truths, 1, rep=0.005, call=0.02, drift=0.0, paired=False)

    sigma = math.sqrt(0.02**2 + (1.2533 * 0.005) ** 2 / 20)
    assert _false_rate(400, make, race, sigma=sigma) <= 0.07


def test_race_power_on_a_real_gap():
    truths = {"a": 1.0, "b": 1.0, "slow": 1.15}
    race = Race(alpha=0.05, margin=0.01)

    def make(rng):
        return _calls(rng, truths, 1, rep=0.005, call=0.02, drift=0.0, paired=False)

    rng = random.Random(2)
    hits = sum(race.eliminate(make(rng), sigma=0.0201) == ["slow"] for _ in range(100))
    assert hits >= 90


def test_race_rejects_bad_parameters():
    with pytest.raises(ValueError, match="alpha"):
        Race(alpha=1.5)
    with pytest.raises(ValueError, match="margin"):
        Race(margin=-0.1)
    with pytest.raises(ValueError, match="min_samples"):
        Race(min_samples=1)


def test_paired_t_and_normal_sf_reference_values():
    assert normal_sf(1.96) == pytest.approx(0.025, abs=1e-4)
    assert normal_sf(0.0) == pytest.approx(0.5)
    # mean 1, sd sqrt(2/3) over 4 diffs: t = 2.449, df = 3 -> p = 0.0459
    # (t = 2.353 is the 95% quantile at df = 3)
    assert paired_t_greater([0.0, 1.0, 1.0, 2.0]) == pytest.approx(0.0459, abs=5e-4)
    assert paired_t_greater([1.0]) == 1.0
    assert paired_t_greater([0.5, 0.5, 0.5], shift=0.1) == 0.0


def test_noise_model():
    m = NoiseModel(floor=0.001)
    assert m.rel_sigma == 0.001 and m.call_sigma is None
    # without a learned call component: as large as the per-rep part
    assert m.median_sigma(4) == pytest.approx(0.001 * math.sqrt(1 + math.pi / 8))
    rng = random.Random(2)
    for _ in range(20):
        m.update([math.exp(rng.gauss(0, 0.02)) for _ in range(20)])
    assert 0.01 < m.rel_sigma < 0.03
    assert m.indistinct(1.0, 1.02)
    assert not m.indistinct(1.0, 1.2)
    assert not m.indistinct(1.0, math.inf)
    m.update_call(1.0, 1.01)
    m.update_call(1.0, 1.02)
    assert m.call_sigma is None  # fewer than min_pairs pairs
    for _ in range(200):
        m.update_call(math.exp(rng.gauss(0, 0.03)), math.exp(rng.gauss(0, 0.03)))
    assert 0.022 < m.call_sigma < 0.04
    assert m.median_sigma(1000) == m.call_sigma
    m.update_call(1.0, math.inf)  # ignored
    again = NoiseModel.from_json(m.to_json())
    assert again.rel_sigma == m.rel_sigma and again.call_sigma == m.call_sigma
