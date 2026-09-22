# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An engine that offers one kernel per problem must be told apart from a broken sweep.

Both produce a report with nothing scored in it, and they want opposite responses: one
is an engine whose kernel choice is a total function of the problem and wants L1, the
other is a collection failure and wants the sweep fixed. The corpora here plant each
case so the distinction can be asserted rather than eyeballed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uhd_gen.catalog import (  # noqa: E402
    DeterministicCatalogError,
    candidate_density,
    require_rankable,
)
from uhd_gen.evaluate import Exclusions, evaluate_corpus  # noqa: E402

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


def _corpus(candidates_per_problem, problems: int = 12) -> pd.DataFrame:
    """One row per candidate. `candidates_per_problem` may be an int or a per-problem list."""
    if isinstance(candidates_per_problem, int):
        counts = [candidates_per_problem] * problems
    else:
        counts = list(candidates_per_problem)
    rows = []
    for index, count in enumerate(counts):
        for candidate in range(count):
            rows.append({
                **_ENVELOPE,
                "benchmark": f"prob{index}",
                "kernel": f"k{index}_{candidate}",
                "robustMeanMs": 0.10 + 0.01 * candidate,
                "kernel.block_m": 64 * (candidate + 1),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- the census


def test_one_candidate_per_problem_is_a_deterministic_catalog():
    density = candidate_density(_corpus(1, problems=9))
    assert density.deterministic
    assert density.problems == 9
    assert density.max_candidates == 1
    assert density.single_candidate_problems == 9
    assert density.rankable_problems == 0
    assert density.histogram == {1: 9}


def test_a_contested_corpus_is_not_deterministic():
    density = candidate_density(_corpus([1, 1, 3, 4, 1]))
    assert not density.deterministic
    assert density.problems == 5
    assert density.single_candidate_problems == 3
    assert density.rankable_problems == 2
    assert density.max_candidates == 4
    assert density.histogram == {1: 3, 3: 1, 4: 1}


def test_an_empty_corpus_concludes_nothing():
    # Not deterministic: a corpus that failed to load must not be read as a finding
    # about the engine, or a ranking role is refused on the strength of a missing file.
    density = candidate_density(pd.DataFrame())
    assert not density.deterministic
    assert density.problems == 0
    assert density.near_deterministic_warning() is None


def test_the_same_graph_on_two_devices_is_two_problems():
    # Problem identity is `evaluate`'s. One kernel per (graph, device) is deterministic
    # even though `benchmark` alone would show two candidates per graph.
    df = _corpus(1, problems=4)
    other = df.copy()
    other["device"] = "devB"
    density = candidate_density(pd.concat([df, other], ignore_index=True))
    assert density.problems == 8
    assert density.deterministic


def test_the_grouping_used_is_reported():
    density = candidate_density(_corpus(2, problems=3))
    assert density.as_dict()["grouped_by"] == ["benchmark", "device"]


def test_as_dict_stringifies_histogram_keys_for_json():
    payload = candidate_density(_corpus([1, 2])).as_dict()
    assert payload["histogram"] == {"1": 1, "2": 1}
    assert payload["deterministic"] is False
    assert payload["rankable_problems"] == 1


# -------------------------------------------------------------------------- the guard


def test_require_rankable_refuses_a_deterministic_catalog_and_names_the_other_role():
    with pytest.raises(DeterministicCatalogError) as caught:
        require_rankable(_corpus(1), engine="hipkernel:Gfx950AttentionDense")
    message = str(caught.value)
    assert "predict_engine_tflops" in message
    assert "hipkernel:Gfx950AttentionDense" in message


def test_the_refusal_is_a_value_error_so_existing_handlers_still_catch_it():
    # `generate.py` preserves the collected stage on ValueError. The measurements are
    # exactly the labels an L1 run needs, so the sweep must not be discarded.
    assert issubclass(DeterministicCatalogError, ValueError)


def test_require_rankable_passes_a_contested_corpus_through():
    density = require_rankable(_corpus([2, 2, 3]))
    assert density.rankable_problems == 3


# ------------------------------------------------------------ the near-deterministic case


def test_a_corpus_that_ranks_almost_nothing_warns_but_does_not_refuse():
    density = candidate_density(_corpus([1] * 39 + [2]))
    assert not density.deterministic
    warning = density.near_deterministic_warning()
    assert warning is not None
    assert "39 of 40" in warning


def test_a_corpus_with_real_ranking_density_does_not_warn():
    assert candidate_density(_corpus([1, 2, 2, 3])).near_deterministic_warning() is None


def test_a_deterministic_corpus_raises_rather_than_warning():
    # Otherwise the operator gets a warning about the thing that already stopped the run.
    assert candidate_density(_corpus(1)).near_deterministic_warning() is None


def test_the_warning_threshold_is_adjustable():
    density = candidate_density(_corpus([1, 1, 1, 2]))
    assert density.near_deterministic_warning() is None
    assert density.near_deterministic_warning(threshold=0.7) is not None


# ---------------------------------------------------------- what the report says about it


def test_exclusions_name_the_cause_when_single_candidates_are_the_whole_reason():
    exclusions = Exclusions(problems_single_candidate=17)
    assert exclusions.deterministic_catalog(scored=0)


def test_exclusions_do_not_blame_the_catalog_when_the_sweep_also_failed():
    # Problems whose candidates never ran are a collection failure. Reporting them as a
    # deterministic catalog would send the operator to train L1 on a broken sweep.
    exclusions = Exclusions(problems_single_candidate=17, problems_no_measured_candidate=3)
    assert not exclusions.deterministic_catalog(scored=0)
    exclusions = Exclusions(problems_single_candidate=17, problems_non_positive_oracle=1)
    assert not exclusions.deterministic_catalog(scored=0)


def test_a_report_that_scored_something_is_never_a_deterministic_catalog():
    assert not Exclusions(problems_single_candidate=17).deterministic_catalog(scored=1)


def test_a_report_that_scored_nothing_for_no_stated_reason_is_not_blamed_on_the_catalog():
    assert not Exclusions().deterministic_catalog(scored=0)


def _oracle_scorer(frame: pd.DataFrame):
    """Rank by the measurement itself: the best a ranker could do, so nothing scores
    zero for want of a good model rather than for want of a choice."""
    return -pd.to_numeric(frame["robustMeanMs"]).to_numpy(dtype=float)


def test_the_evaluation_report_records_the_condition():
    result = evaluate_corpus(_corpus(1, problems=8), _oracle_scorer,
                             target="robustMeanMs", objective="min", eval_fraction=1.0)
    assert result.report["metrics"]["problems_scored"] == 0
    assert result.report["metrics"]["deterministic_catalog"] is True
    assert result.report["exclusions"]["problems_single_candidate"] == 8


def test_a_contested_report_records_the_condition_as_false():
    result = evaluate_corpus(_corpus(3, problems=8), _oracle_scorer,
                             target="robustMeanMs", objective="min", eval_fraction=1.0)
    assert result.report["metrics"]["problems_scored"] == 8
    assert result.report["metrics"]["deterministic_catalog"] is False


# ------------------------------------------------------------------- the knob report


def test_knob_analysis_reports_the_finding_instead_of_a_page_of_pinnable_fields(
        tmp_path, capsys):
    from uhd_gen.knobs import run_knobs

    corpus = tmp_path / "corpus.csv"
    _corpus(1, problems=6).to_csv(corpus, index=False)
    args = argparse.Namespace(input=str(corpus), target="robustMeanMs", objective="min",
                              device_column=None, manifest=None)
    assert run_knobs(args) == 0
    out = capsys.readouterr().out
    assert "deterministic catalog" in out
    assert "predict_engine_tflops" in out
    assert "PINNED" not in out
