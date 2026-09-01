#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RFC 0019.13 §11.2 regret, and the four ways it silently lies.

Every test here pins one arithmetic or bookkeeping property that, if it broke, would
produce a *plausible* number rather than an error:

- a model that always picks the oracle must score exactly 0, and one that always picks
  the worst must score the hand-computed shortfall -- the two ends that calibrate the
  scale;
- both objectives must yield non-negative regret, because a flipped direction turns a
  30% loss into a small positive-looking figure;
- the held-out split must move whole PROBLEMS: a row-wise split leaves the oracle row
  in training, so the evaluation side's "best" is the best of a subset and the regret
  collapses toward zero;
- a corpus with no device identity must say so, because the same graph on two GPUs
  conflated into one problem produces an oracle no candidate on the slower card could
  have reached;
- an `is_valid=False` row has no timing, so it must never become the oracle -- a
  blank-as-zero would make it the unbeatable minimum of every problem it appears in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uhd_gen.evaluate import (  # noqa: E402
    ObjectiveDirectionError,
    Split,
    evaluate_corpus,
    regret_of,
    resolve_grouping,
    split_problems,
)

DEVICE_A = "a1b2c3d4"
DEVICE_B = "ffffbbbb"


def make_corpus(rows: list[dict]) -> pd.DataFrame:
    """A §8.3-shaped corpus from terse row dicts, with the envelope filled in."""
    filled = []
    for row in rows:
        record = {
            "benchmark": row["benchmark"],
            "device": row.get("device", DEVICE_A),
            "kernel": row["kernel"],
            "pack": row.get("pack", "p"),
            "dispatch": row.get("dispatch", "d"),
            "minTimeMs": row.get("minTimeMs", ""),
            "avgTimeMs": row.get("avgTimeMs", ""),
            "stddevMs": row.get("stddevMs", ""),
            "iters": row.get("iters", 100),
            "is_valid": row.get("is_valid", "True"),
            "tflops": row.get("tflops", ""),
            "q.M": row.get("q.M", 128),
        }
        if "regime" in row:
            record["regime"] = row["regime"]
        filled.append(record)
    return pd.DataFrame(filled)


def oracle_scorer(target: str, objective: str):
    """A perfect model: its score IS the measured value, so its pick is the oracle."""

    def score(frame: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(frame[target]).to_numpy(dtype=float)

    return score


def worst_scorer(target: str, objective: str):
    """The exact inverse: ranks the worst measured candidate first."""

    def score(frame: pd.DataFrame) -> np.ndarray:
        return -pd.to_numeric(frame[target]).to_numpy(dtype=float)

    return score


def evaluate_all(df: pd.DataFrame, scorer, **kwargs):
    """Score every problem, so a test's arithmetic is over rows it wrote itself.

    The split is exercised on its own in the split tests; mixing it into every
    arithmetic test would mean each expected value depended on a hash.
    """
    kwargs.setdefault("eval_fraction", 1.0)
    return evaluate_corpus(df, scorer, **kwargs)


# ---------------------------------------------------------------------------------
# The two ends of the scale
# ---------------------------------------------------------------------------------


def test_oracle_picking_model_has_exactly_zero_regret():
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 2.0},
            {"benchmark": "g2", "kernel": "k1", "minTimeMs": 5.0},
            {"benchmark": "g2", "kernel": "k2", "minTimeMs": 4.0},
        ]
    )
    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )
    metrics = result.report["metrics"]

    assert metrics["problems_scored"] == 2
    assert metrics["top1_regret"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert metrics["regret_tail"]["fraction"] == 0.0
    assert metrics["topk_recall"]["strict"]["1"] == 1.0


def test_worst_picking_model_scores_the_hand_computed_regret():
    # g1: oracle 1.0, worst 2.0  -> 2.0/1.0 - 1 = 1.00
    # g2: oracle 4.0, worst 5.0  -> 5.0/4.0 - 1 = 0.25
    # mean = 0.625, p50 = 0.625 (two points, linear interpolation), max = 1.00
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 2.0},
            {"benchmark": "g2", "kernel": "k1", "minTimeMs": 5.0},
            {"benchmark": "g2", "kernel": "k2", "minTimeMs": 4.0},
        ]
    )
    result = evaluate_all(
        df, worst_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )
    regret = result.report["metrics"]["top1_regret"]

    assert regret["mean"] == pytest.approx(0.625)
    assert regret["p50"] == pytest.approx(0.625)
    assert regret["max"] == pytest.approx(1.0)
    # Both problems exceed 5%, so the tail is everything.
    assert result.report["metrics"]["regret_tail"]["fraction"] == 1.0
    # The oracle is dead last under this model, so strict recall@1 is 0 and even @3
    # cannot save it on a two-candidate problem... except k=3 >= |V(p)|, which is a
    # tautological hit and is reported as such.
    assert result.report["metrics"]["topk_recall"]["strict"]["1"] == 0.0
    assert result.report["metrics"]["topk_recall"]["trivial"]["3"] == 1.0


def test_regret_is_non_negative_under_both_objectives():
    """The same corpus read as throughput and as latency, worst pick both times."""
    minimise = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 4.0},
        ]
    )
    maximise = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "tflops": 100.0},
            {"benchmark": "g1", "kernel": "k2", "tflops": 25.0},
        ]
    )

    low = evaluate_all(
        minimise, worst_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )
    high = evaluate_all(
        maximise, worst_scorer("tflops", "max"), target="tflops", objective="max"
    )

    # min: 4/1 - 1 = 3.0.  max: 1 - 25/100 = 0.75. Different scales, same sign.
    assert low.report["metrics"]["top1_regret"]["mean"] == pytest.approx(3.0)
    assert high.report["metrics"]["top1_regret"]["mean"] == pytest.approx(0.75)
    assert low.report["metrics"]["top1_regret"]["mean"] >= 0.0
    assert high.report["metrics"]["top1_regret"]["mean"] >= 0.0


def test_backwards_objective_fails_loudly_instead_of_printing():
    """A negative regret means the direction is wrong; every figure would be inverted."""
    with pytest.raises(ObjectiveDirectionError, match="backwards"):
        # A pick of 1.0 against an "oracle" of 4.0 under `min`: 4.0 is not the minimum
        # of anything containing 1.0, so this is what a flipped direction looks like
        # from inside the metric -- it would otherwise print -0.75 as a regret.
        regret_of(picked=1.0, oracle=4.0, objective="min")
    with pytest.raises(ObjectiveDirectionError, match="backwards"):
        regret_of(picked=4.0, oracle=1.0, objective="max")


def test_ranking_direction_follows_the_objective():
    """One scorer, two directions, opposite ends of the same candidate set."""
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "slow", "minTimeMs": 10.0},
            {"benchmark": "g1", "kernel": "fast", "minTimeMs": 1.0},
        ]
    )

    def score(frame: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(frame["minTimeMs"]).to_numpy(dtype=float)

    lower_is_better = evaluate_all(df, score, target="minTimeMs", objective="min")
    higher_is_better = evaluate_all(df, score, target="minTimeMs", objective="max")

    # Nothing about the scorer changed, only the direction the report reads it in.
    assert lower_is_better.problems[0].picked_value == pytest.approx(1.0)
    assert higher_is_better.problems[0].picked_value == pytest.approx(10.0)
    assert lower_is_better.problems[0].regret == pytest.approx(0.0)
    assert higher_is_better.problems[0].regret == pytest.approx(0.0)


# ---------------------------------------------------------------------------------
# Rows that must never become the oracle
# ---------------------------------------------------------------------------------


def test_invalid_rows_are_never_the_oracle():
    """An is_valid=False row has no timing; blank-as-zero would win every argmin."""
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "ok_fast", "minTimeMs": 2.0},
            {"benchmark": "g1", "kernel": "ok_slow", "minTimeMs": 3.0},
            # Exactly what `export-benchmarks` writes for a candidate that would not
            # compile: identity and features kept, every timing column empty.
            {"benchmark": "g1", "kernel": "failed", "minTimeMs": "", "is_valid": "False"},
        ]
    )
    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )
    problem = result.problems[0]

    assert problem.oracle_value == pytest.approx(2.0)
    assert problem.candidates == 2, "the failed candidate must not be in V(p)"
    assert result.report["exclusions"]["invalid_rows"] == 1
    assert result.report["metrics"]["top1_regret"]["mean"] == pytest.approx(0.0)


def test_an_invalid_row_carrying_a_stale_timing_is_still_excluded():
    """is_valid is authoritative, not the presence of a number in the timing column."""
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "real", "minTimeMs": 2.0},
            {"benchmark": "g1", "kernel": "also_real", "minTimeMs": 3.0},
            {"benchmark": "g1", "kernel": "bogus", "minTimeMs": 0.01, "is_valid": "False"},
        ]
    )
    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )

    assert result.problems[0].oracle_value == pytest.approx(2.0)
    assert result.report["exclusions"]["invalid_rows"] == 1


def test_single_candidate_problems_are_excluded_and_counted():
    """With nothing to choose between, a correct pick is not evidence of anything."""
    df = make_corpus(
        [
            {"benchmark": "alone", "kernel": "k1", "minTimeMs": 7.0},
            {"benchmark": "pair", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "pair", "kernel": "k2", "minTimeMs": 2.0},
        ]
    )
    result = evaluate_all(
        df, worst_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )

    # If the lone problem were scored as regret 0 it would halve the reported mean.
    assert result.report["metrics"]["problems_scored"] == 1
    assert result.report["exclusions"]["problems_single_candidate"] == 1
    assert result.report["metrics"]["top1_regret"]["mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------
# Problem identity
# ---------------------------------------------------------------------------------


def test_two_devices_are_two_problems():
    """The same graph on a fast and a slow card has two oracles, not one."""
    df = make_corpus(
        [
            {"benchmark": "g1", "device": DEVICE_A, "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "device": DEVICE_A, "kernel": "k2", "minTimeMs": 1.1},
            {"benchmark": "g1", "device": DEVICE_B, "kernel": "k1", "minTimeMs": 10.0},
            {"benchmark": "g1", "device": DEVICE_B, "kernel": "k2", "minTimeMs": 11.0},
        ]
    )
    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )

    assert result.report["grouping"]["columns"] == ["benchmark", "device"]
    assert result.report["grouping"]["degraded"] is False
    assert result.report["metrics"]["problems_scored"] == 2
    # Conflated, the oracle would be 1.0 and the slow card's perfect pick would be
    # charged 10.0/1.0 - 1 = 9.0 of regret it had no way to avoid.
    assert result.report["metrics"]["top1_regret"]["max"] == pytest.approx(0.0)
    assert not any("GROUPING" in warning for warning in result.warnings)


def test_missing_device_column_degrades_loudly():
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 2.0},
        ]
    ).drop(columns=["device"])

    grouping = resolve_grouping(df)
    assert grouping.degraded is True
    assert grouping.columns == ("benchmark",)

    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )
    assert result.report["grouping"]["degraded"] is True
    assert result.report["warnings"], "a degraded grouping must never be silent"
    warning = result.report["warnings"][0]
    assert "DEGRADED PROBLEM GROUPING" in warning
    assert "two devices" in warning
    assert result.report["warnings"] == result.warnings


def test_empty_device_column_degrades_loudly_too():
    """The sibling exporter writes an empty string for pre-change logs, not a NaN."""
    df = make_corpus(
        [
            {"benchmark": "g1", "device": "", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "device": "", "kernel": "k2", "minTimeMs": 2.0},
        ]
    )
    df["device"] = df["device"].astype(str)

    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )
    assert result.report["grouping"]["columns"] == ["benchmark"]
    assert "DEGRADED PROBLEM GROUPING" in result.report["warnings"][0]
    assert "empty on all" in result.report["warnings"][0]


def test_partially_identified_device_column_warns_without_merging():
    df = make_corpus(
        [
            {"benchmark": "g1", "device": DEVICE_A, "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "device": DEVICE_A, "kernel": "k2", "minTimeMs": 2.0},
            {"benchmark": "g1", "device": "", "kernel": "k1", "minTimeMs": 9.0},
            {"benchmark": "g1", "device": "", "kernel": "k2", "minTimeMs": 8.0},
        ]
    )
    result = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    )

    assert result.report["grouping"]["degraded"] is False
    assert result.report["metrics"]["problems_scored"] == 2
    assert "PARTIAL DEVICE IDENTITY" in result.report["warnings"][0]


# ---------------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------------


def test_split_is_reproducible_and_seed_dependent():
    keys = [("g%02d" % i,) for i in range(40)]
    first = split_problems(keys, 0.25, seed=7)
    again = split_problems(list(reversed(keys)), 0.25, seed=7)
    other = split_problems(keys, 0.25, seed=8)

    assert first.eval_problems == again.eval_problems, "row order must not move the slice"
    assert len(first.eval_problems) == 10
    assert set(first.eval_problems) & set(first.train_problems) == set()
    assert first.eval_problems != other.eval_problems


def test_split_never_returns_an_empty_slice():
    single = split_problems([("only",)], 0.2, seed=0)
    assert single.eval_problems == (("only",),)


def test_split_moves_whole_problems_not_rows():
    """The property a row-wise split violates, stated directly.

    Every candidate of an evaluated problem has to be on the evaluation side. If the
    split ran over rows, some of a problem's candidates would sit in training, the
    evaluation-side oracle would be the best of what remained, and a model that picked
    a mediocre kernel would look correct because the better one was not there to
    compare against.
    """
    rows = []
    for problem in range(12):
        for candidate, time_ms in enumerate([1.0, 4.0, 4.0, 4.0]):
            rows.append(
                {
                    "benchmark": f"g{problem:02d}",
                    "kernel": f"k{candidate}",
                    "minTimeMs": time_ms,
                }
            )
    df = make_corpus(rows)

    # A model that always ranks the LAST candidate first: it never picks the 1.0 row,
    # so with the full candidate set every scored problem has regret 4/1 - 1 = 3.0.
    def last_first(frame: pd.DataFrame) -> np.ndarray:
        return -np.arange(len(frame), dtype=float)

    result = evaluate_corpus(
        df,
        last_first,
        target="minTimeMs",
        objective="min",
        eval_fraction=0.25,
        seed=3,
    )

    assert result.report["split"]["unit"] == "problem"
    assert result.report["split"]["eval_problems"] == 3
    assert result.report["metrics"]["problems_scored"] == 3
    for problem in result.problems:
        # The invariant: four candidates were measured for this problem, four were
        # scored. A row-wise split would show 1, 2 or 3 here.
        assert problem.candidates == 4
        assert problem.oracle_value == pytest.approx(1.0)
        assert problem.regret == pytest.approx(3.0)
    assert result.report["metrics"]["top1_regret"]["mean"] == pytest.approx(3.0)

    # And the split is recorded, so the figure is reproducible from the report alone.
    assert result.report["split"]["seed"] == 3
    assert len(result.report["split"]["eval_problem_keys"]) == 3


def test_a_row_wise_split_would_report_a_materially_better_number():
    """What the previous test is defending against, computed explicitly.

    Simulating the leak by hand -- dropping the oracle row of each problem, exactly
    what a row-wise split does to a fraction of them -- turns a regret of 3.0 into a
    regret of 0.0. The two numbers are not close, which is why the split unit is not a
    detail.
    """
    rows = []
    for problem in range(12):
        for candidate, time_ms in enumerate([1.0, 4.0, 4.0, 4.0]):
            rows.append(
                {
                    "benchmark": f"g{problem:02d}",
                    "kernel": f"k{candidate}",
                    "minTimeMs": time_ms,
                }
            )
    df = make_corpus(rows)

    def last_first(frame: pd.DataFrame) -> np.ndarray:
        return -np.arange(len(frame), dtype=float)

    group_aware = evaluate_corpus(
        df, last_first, target="minTimeMs", objective="min", eval_fraction=0.25, seed=3
    )
    leaked = evaluate_corpus(
        df[df["kernel"] != "k0"],
        last_first,
        target="minTimeMs",
        objective="min",
        eval_fraction=0.25,
        seed=3,
    )

    assert group_aware.report["metrics"]["top1_regret"]["mean"] == pytest.approx(3.0)
    assert leaked.report["metrics"]["top1_regret"]["mean"] == pytest.approx(0.0)


def test_full_corpus_evaluation_says_it_held_nothing_out():
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 2.0},
        ]
    )
    result = evaluate_corpus(
        df,
        oracle_scorer("minTimeMs", "min"),
        target="minTimeMs",
        objective="min",
        eval_fraction=1.0,
    )
    assert result.report["split"]["method"] == "full_corpus"
    assert any("NO HELD-OUT SLICE" in warning for warning in result.report["warnings"])


# ---------------------------------------------------------------------------------
# Recall, ties and regimes
# ---------------------------------------------------------------------------------


def test_top_k_recall_counts_the_oracles_rank():
    """Six candidates, the oracle ranked fourth: a miss at 1 and 3, a hit at 5."""
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": f"k{i}", "minTimeMs": t}
            for i, t in enumerate(times)
        ]
    )

    # Ranks k3, k4, k5 ahead of the oracle k0 by giving them lower scores under `min`.
    def score(frame: pd.DataFrame) -> np.ndarray:
        rank = {"k3": 0.0, "k4": 1.0, "k5": 2.0, "k0": 3.0, "k1": 4.0, "k2": 5.0}
        return np.array([rank[name] for name in frame["kernel"]], dtype=float)

    recall = evaluate_all(df, score, target="minTimeMs", objective="min").report[
        "metrics"
    ]["topk_recall"]

    assert recall["strict"]["1"] == 0.0
    assert recall["strict"]["3"] == 0.0
    assert recall["strict"]["5"] == 1.0


def test_tie_aware_recall_forgives_an_indistinguishable_second_choice():
    """§11.2's whole point: two kernels within noise, either choice costs nothing."""
    df = make_corpus(
        [
            # 1.000 vs 1.002 -- a 0.2% difference, inside both the 1% tolerance and the
            # +-2 standard errors implied by stddevMs=0.05 over 100 iterations.
            {"benchmark": "g1", "kernel": "a", "minTimeMs": 1.000, "stddevMs": 0.05},
            {"benchmark": "g1", "kernel": "b", "minTimeMs": 1.002, "stddevMs": 0.05},
            {"benchmark": "g1", "kernel": "c", "minTimeMs": 3.000, "stddevMs": 0.05},
        ]
    )

    def prefers_b(frame: pd.DataFrame) -> np.ndarray:
        rank = {"b": 0.0, "a": 1.0, "c": 2.0}
        return np.array([rank[name] for name in frame["kernel"]], dtype=float)

    report = evaluate_all(df, prefers_b, target="minTimeMs", objective="min").report
    recall = report["metrics"]["topk_recall"]

    # Strict says the model missed. Tie-aware says it did not, and the regret -- the
    # metric §11.2 actually trusts -- agrees with tie-aware: 1.002/1.000 - 1 = 0.002.
    assert recall["strict"]["1"] == 0.0
    assert recall["tie_aware"]["1"] == 1.0
    assert report["metrics"]["top1_regret"]["mean"] == pytest.approx(0.002)
    assert report["metrics"]["regret_tail"]["fraction"] == 0.0
    assert report["ties"]["noise_band_applied"] is True


def test_a_real_miss_is_not_forgiven_by_the_tie_rule():
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "a", "minTimeMs": 1.0, "stddevMs": 0.001},
            {"benchmark": "g1", "kernel": "b", "minTimeMs": 1.5, "stddevMs": 0.001},
        ]
    )

    def prefers_b(frame: pd.DataFrame) -> np.ndarray:
        return np.array([1.0 if name == "a" else 0.0 for name in frame["kernel"]])

    recall = evaluate_all(df, prefers_b, target="minTimeMs", objective="min").report[
        "metrics"
    ]["topk_recall"]
    assert recall["strict"]["1"] == 0.0
    assert recall["tie_aware"]["1"] == 0.0


def test_noise_band_is_not_applied_to_a_target_in_other_units():
    """stddevMs is milliseconds; a TFLOPS target must not be widened by it."""
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "a", "tflops": 100.0, "stddevMs": 50.0},
            {"benchmark": "g1", "kernel": "b", "tflops": 99.0, "stddevMs": 50.0},
        ]
    )
    report = evaluate_all(
        df, oracle_scorer("tflops", "max"), target="tflops", objective="max"
    ).report
    assert report["ties"]["noise_band_applied"] is False
    assert "not a millisecond timing column" in report["ties"]["policy"]


def test_per_regime_regret_when_the_corpus_carries_a_regime():
    df = make_corpus(
        [
            {"benchmark": "d1", "kernel": "k1", "minTimeMs": 1.0, "regime": "decode"},
            {"benchmark": "d1", "kernel": "k2", "minTimeMs": 3.0, "regime": "decode"},
            {"benchmark": "p1", "kernel": "k1", "minTimeMs": 1.0, "regime": "prefill"},
            {"benchmark": "p1", "kernel": "k2", "minTimeMs": 1.1, "regime": "prefill"},
        ]
    )
    per_regime = evaluate_all(
        df, worst_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    ).report["metrics"]["per_regime"]

    # decode: 3/1 - 1 = 2.0.  prefill: 1.1/1 - 1 = 0.1. The aggregate mean of 1.05
    # describes neither, which is exactly why §11.2 makes this table the primary form.
    assert per_regime["decode"]["mean_regret"] == pytest.approx(2.0)
    assert per_regime["prefill"]["mean_regret"] == pytest.approx(0.1)


def test_absent_regime_is_stated_not_omitted():
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 2.0},
        ]
    )
    metrics = evaluate_all(
        df, oracle_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    ).report["metrics"]

    assert "per_regime" in metrics, "the key must be present even when it is null"
    assert metrics["per_regime"] is None
    assert metrics["per_regime_status"].startswith("UNAVAILABLE")
    assert "regime" in metrics["per_regime_status"]


# ---------------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------------


def test_report_records_everything_needed_to_reproduce_it():
    df = make_corpus(
        [
            {"benchmark": "g1", "kernel": "k1", "minTimeMs": 1.0},
            {"benchmark": "g1", "kernel": "k2", "minTimeMs": 2.0},
            {"benchmark": "g2", "kernel": "k1", "minTimeMs": 3.0},
            {"benchmark": "g2", "kernel": "k2", "minTimeMs": 4.0},
        ]
    )
    report = evaluate_corpus(
        df,
        oracle_scorer("minTimeMs", "min"),
        target="minTimeMs",
        objective="min",
        eval_fraction=0.5,
        seed=11,
    ).report

    assert report["schema"] == "uhd_gen.eval_report/1"
    assert report["target"] == "minTimeMs"
    assert report["objective"] == "min"
    assert report["split"]["seed"] == 11
    assert report["split"]["eval_fraction"] == 0.5
    assert report["not_implemented"], "the §11.2 gaps must be stated, not implied"
    # Serialisable as written: the report is the artifact §10.4 names.
    json.dumps(report)


def test_regret_tail_threshold_is_the_five_percent_of_the_spec():
    df = make_corpus(
        [
            # 4% over the oracle: under the tail threshold.
            {"benchmark": "near", "kernel": "k1", "minTimeMs": 1.00},
            {"benchmark": "near", "kernel": "k2", "minTimeMs": 1.04},
            # 20% over: in the tail.
            {"benchmark": "far", "kernel": "k1", "minTimeMs": 1.00},
            {"benchmark": "far", "kernel": "k2", "minTimeMs": 1.20},
        ]
    )
    tail = evaluate_all(
        df, worst_scorer("minTimeMs", "min"), target="minTimeMs", objective="min"
    ).report["metrics"]["regret_tail"]

    assert tail["threshold"] == 0.05
    assert tail["problems"] == 1
    assert tail["fraction"] == pytest.approx(0.5)
