# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The knob report has to separate three things a build budget confuses.

A knob that never varies, a knob that varies without mattering, and a knob that decides
the answer all look alike in a KMD: three declared fields. They cost very differently --
the first costs nothing, the second multiplies the AOT build for noise, and the third
earns every kernel it asks for.

These plant each case with a known answer and assert the report recovers it. A corpus
whose structure is invented here rather than measured is the point: on real data every
number is plausible and none is checkable.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uhd_gen.knobs import analyse_knobs, knob_columns  # noqa: E402

_ENVELOPE = {
    "device": "devA",
    "pack": "pk",
    "dispatch": "dp",
    "minTimeMs": 0.0,
    "avgTimeMs": 0.0,
    "stddevMs": 0.001,
    "iters": 50,
    "is_valid": "true",
    "skip_reason": "",
    "collection_mode": "STANDARD",
    "problem_complete": "true",
    "shard_id": "0",
    "config_set_hash": "h",
    "applicability_id": "a",
}


def _corpus(problems: int = 60, seed: int = 5) -> pd.DataFrame:
    """A corpus with one knob of each kind, planted.

    - `kernel.block_m` DECIDES: which tile is fastest depends on the problem, so no
      single value can be pinned without roughly doubling the time on half the corpus.
    - `kernel.use_exp2_fast` is a uniform win: 1 is always at least as good, so pinning
      to it costs exactly nothing.
    - `kernel.waves_per_eu` is noise: no systematic effect.
    - `kernel.block_n` never varies.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(problems):
        seqlen = rng.choice([256, 512, 1024, 2048])
        best_bm = 256 if seqlen >= 1024 else 64
        for block_m in (64, 128, 256):
            for waves in (2, 4):
                for exp2 in (0, 1):
                    t = 0.05 if block_m == best_bm else 0.10
                    t *= 0.97 if exp2 else 1.0
                    t *= 1.0 + rng.gauss(0, 0.002)
                    rows.append(
                        {
                            **_ENVELOPE,
                            "benchmark": f"prob{i}",
                            "kernel": f"k{i}_{block_m}_{waves}_{exp2}",
                            "robustMeanMs": t,
                            "kernel.block_m": block_m,
                            "kernel.block_n": 64,
                            "kernel.waves_per_eu": waves,
                            "kernel.use_exp2_fast": exp2,
                        }
                    )
    return pd.DataFrame(rows)


def _knob(report: dict, name: str) -> dict:
    return next(k for k in report["knobs"] if k["name"] == name)


def test_knob_columns_are_the_kernel_axes_only():
    df = _corpus(4)
    df["q.seqlen_q"] = 512
    # `$q.*` describes the problem: it cannot be chosen away and is not an AOT axis.
    assert knob_columns(df) == [
        "kernel.block_m",
        "kernel.block_n",
        "kernel.waves_per_eu",
        "kernel.use_exp2_fast",
    ]


def test_a_field_that_never_varies_is_named_as_constant():
    # It costs no kernels, so it is not an AOT saving -- but it sits in the KMD claiming
    # to be a variant axis, and a model may be ranking on a column that cannot separate
    # anything. Silence about it would leave that undiscovered.
    report = analyse_knobs(_corpus())
    block_n = _knob(report, "kernel.block_n")
    assert block_n["constant"] is True
    assert block_n["distinct_values"] == 1
    assert block_n["per_value"] == []


def test_a_deciding_knob_is_expensive_to_pin():
    # The planted structure: the right tile depends on the problem, so every single
    # value is wrong on part of the corpus and pinning roughly doubles the time there.
    report = analyse_knobs(_corpus())
    block_m = _knob(report, "kernel.block_m")

    assert block_m["constant"] is False
    assert block_m["cost_of_pinning"] > 0.5, (
        "a knob that decides the winner cannot be pinned cheaply; "
        f"p95 regret was {block_m['cost_of_pinning']:.2%}"
    )


def test_a_noise_knob_is_nearly_free_to_pin():
    # Varies, costs kernels, changes nothing: the case the AOT budget wants to find.
    report = analyse_knobs(_corpus())
    waves = _knob(report, "kernel.waves_per_eu")

    assert waves["constant"] is False
    assert waves["cost_of_pinning"] < 0.02, (
        "a knob with no systematic effect must read as nearly free to pin; "
        f"p95 regret was {waves['cost_of_pinning']:.2%}"
    )


def test_a_uniformly_better_value_costs_exactly_nothing():
    # One value dominates everywhere, so only that value need ever be built.
    report = analyse_knobs(_corpus())
    exp2 = _knob(report, "kernel.use_exp2_fast")

    assert exp2["best_value"] == 1
    assert exp2["cost_of_pinning"] == pytest.approx(0.0, abs=1e-9)


def test_pinning_never_silently_drops_a_problem():
    """Coverage loss is not regret and must not be averaged into it.

    A value that cannot serve a problem leaves the engine with no kernel for it. That is
    categorically worse than a slower kernel, so it is counted separately and the best
    value is chosen on coverage first.
    """
    df = _corpus()
    # Make block_m=256 the only choice for one problem, then delete every other
    # candidate for it, so a pin to 64 or 128 cannot serve that problem at all.
    victim = df["benchmark"] == "prob0"
    df = df[~victim | (df["kernel.block_m"] == 256)]

    report = analyse_knobs(df)
    block_m = _knob(report, "kernel.block_m")
    by_value = {v["value"]: v for v in block_m["per_value"]}

    assert by_value[64]["uncovered"] == 1
    assert by_value[256]["uncovered"] == 0
    assert block_m["problems_lost_by_pinning"] == 0, (
        "the recommended value must be one that still serves every problem"
    )


def test_the_variant_curve_finds_the_cheap_covering_set():
    """The AOT question: how few kernels per geometry suffice.

    Twelve combinations are built here and only the tile actually matters, so a couple
    of variants should reach near-zero regret. The assertion is on the shape -- fewer
    variants than combinations, and flat by the end -- not on an exact count, which
    would pin the greedy tie-break rather than the finding.
    """
    report = analyse_knobs(_corpus())
    curve = report["variant_curve"]

    assert curve, "a corpus with varying knobs must produce a curve"
    assert curve[-1]["problems_uncovered"] == 0
    assert curve[-1]["mean_regret"] == pytest.approx(0.0, abs=1e-9)

    within_one_percent = next(
        (row["variants"] for row in curve if row["mean_regret"] < 0.01), None
    )
    assert within_one_percent is not None and within_one_percent <= 4, (
        "twelve built combinations should collapse to a handful; "
        f"needed {within_one_percent} variants to reach 1% mean regret"
    )


def test_an_empty_corpus_is_refused_rather_than_reported():
    df = _corpus(4)
    df["robustMeanMs"] = float("nan")
    with pytest.raises(ValueError, match="positive robustMeanMs"):
        analyse_knobs(df)
