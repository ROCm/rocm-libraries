# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collection must never associate a timing with an unaddressed candidate."""
import copy

import pytest

pytest.importorskip("pandas")
from uhd_gen.generate import collect_graph


def _page():
    return {"engine_id": 7, "graph_id": "graph", "device_id": "board", "device_arch": "gfx942",
            "problem_features": {"attention.query.dims[2]": 128, "graph.flops": 4e12},
            "device_features": {"device.cu_count": 120},
            "engine_descriptor_id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "engine_name": "hipkernel:test",
            "candidates": [{"id": "kernel-a", "knob_settings": {"tile_m": 32}, "kernel_features": {"kernel.tile_m": 32}}],
            "total_count": 1, "next_offset": None}


def _timing(page):
    candidate = page["candidates"][0]
    return {key: copy.deepcopy(page[key]) for key in
            ("engine_id", "graph_id", "device_id", "device_arch", "engine_descriptor_id", "engine_name", "problem_features", "device_features")} | {
        "results": [{"candidate_id": candidate["id"], "knob_settings": candidate["knob_settings"],
                     "kernel_features": candidate["kernel_features"], "succeeded": True,
                     "is_valid": True, "numerically_valid": True,
                     "validation": "agrees_with_catalog: 2 of 2 cross-checked candidates produced this output",
                     "robust_time_ms": 2.5, "min_time_ms": 2.0, "avg_time_ms": 2.6,
                     "stddev_ms": 0.01, "iterations": 40}]}


def _collect(monkeypatch, tmp_path, responses):
    iterator = iter(responses)
    monkeypatch.setattr("uhd_gen.generate._run_json", lambda *args: next(iterator))
    return collect_graph(["hipdnn_bench", "--graph", "graph.json", "--engine-id", "7"], {}, tmp_path, [],
                         engine_descriptor_id="13ab344f-4818-4772-bb8e-8e1441fec82c")


@pytest.mark.parametrize("mutation", ["candidate", "device", "arch", "knobs", "kernel_features", "provenance"])
def test_timing_must_resolve_exact_enrolled_identity(monkeypatch, tmp_path, mutation):
    page = _page()
    timed = _timing(page)
    if mutation == "provenance":
        page["engine_descriptor_id"] = "not-the-trained-engine"
    elif mutation == "candidate":
        timed["results"][0]["candidate_id"] = "kernel-b"
    elif mutation == "device":
        timed["device_id"] = "other-board"
    elif mutation == "arch":
        page["device_arch"] = "gfx942:sramecc+:xnack-"
        timed["device_arch"] = "gfx942:sramecc-:xnack-"
    elif mutation == "knobs":
        timed["results"][0]["knob_settings"] = {"tile_m": 64}
    else:
        timed["results"][0]["kernel_features"] = {"kernel.tile_m": 64}
    with pytest.raises(ValueError):
        _collect(monkeypatch, tmp_path, [page, timed])


def test_incomplete_enumeration_is_not_a_training_corpus(monkeypatch, tmp_path):
    page = _page()
    page["total_count"] = 2
    with pytest.raises(ValueError, match="truncation"):
        _collect(monkeypatch, tmp_path, [page])


def test_ambiguous_enrolled_tuple_is_rejected_before_timing(monkeypatch, tmp_path):
    page = _page()
    duplicate = copy.deepcopy(page["candidates"][0])
    duplicate["id"] = "kernel-b"
    page["candidates"].append(duplicate)
    page["total_count"] = 2
    with pytest.raises(ValueError, match="unique"):
        _collect(monkeypatch, tmp_path, [page])


def test_feature_suffixed_device_uses_bare_training_architecture(monkeypatch, tmp_path):
    page = _page()
    page["device_arch"] = "gfx942:sramecc+:xnack-"
    rows, _ = _collect(monkeypatch, tmp_path, [page, _timing(page)])
    assert {row["arch"] for row in rows} == {"gfx942"}


def test_collected_rows_carry_the_noise_columns_and_the_calibrated_rate(monkeypatch, tmp_path):
    """§8.3's envelope, and §11.1's cross-engine score off §11.2's statistic."""
    page = _page()
    row = _collect(monkeypatch, tmp_path, [page, _timing(page)])[0][0]
    assert (row["stddevMs"], row["iters"]) == (0.01, 40)
    # graph.flops / (avgTimeMs * 1e9): the mean, not the robust mean, because §11.2
    # (:2003) pins a calibrated score to `avgTimeMs`. Off robust_time_ms=2.5 this
    # would read 1600.0 instead.
    assert row["tflops"] == pytest.approx(4e12 / (2.6 * 1e9))


def test_an_engine_that_publishes_no_work_count_gets_no_fabricated_throughput(monkeypatch, tmp_path):
    """§8.3: `tflops` is derived from a declared FLOP count or it is absent."""
    page = _page()
    page["problem_features"].pop("graph.flops")
    row = _collect(monkeypatch, tmp_path, [page, _timing(page)])[0][0]
    assert "tflops" not in row


def test_a_collected_corpus_activates_the_evaluate_noise_band(monkeypatch, tmp_path):
    """§8.5 records the spread "so it can be used": the band keys on these exact names.

    `evaluate` looks for `stddevMs`/`iters`. Collection wrote neither, so the band was
    unreachable from a generated corpus however the run was configured -- and the report
    blamed the target's units for it.
    """
    import pandas as pd
    from uhd_gen.evaluate import evaluate_corpus

    page = _page()
    second = {"id": "kernel-b", "knob_settings": {"tile_m": 64},
              "kernel_features": {"kernel.tile_m": 64}}
    first_timing, second_timing = _timing(page), _timing(page)
    second_timing["results"][0].update(candidate_id="kernel-b", knob_settings={"tile_m": 64},
                                       kernel_features={"kernel.tile_m": 64},
                                       robust_time_ms=2.51, min_time_ms=2.01, avg_time_ms=2.61)
    page["candidates"].append(second)
    page["total_count"] = 2
    rows, _ = _collect(monkeypatch, tmp_path, [page, first_timing, second_timing])

    report = evaluate_corpus(
        pd.DataFrame(rows), lambda frame: frame["avgTimeMs"].to_numpy(dtype=float),
        target="avgTimeMs", objective="min", eval_fraction=1.0,
    ).report
    assert report["ties"]["noise_band_applied"] is True
    assert "standard errors" in report["ties"]["policy"]


def test_a_numerically_invalid_candidate_keeps_its_row_but_loses_its_timing(monkeypatch, tmp_path):
    """RFC 0019 §13.2: recorded with its measurement suppressed and an invalid marker.

    The two halves are one rule. Dropping the row loses the failure surface the section
    wants the model to learn; keeping the timing hands the ranker the group's best time,
    because a kernel that does not compute the answer is the fastest one in it.
    """
    page = _page()
    timed = _timing(page)
    timed["results"][0].update(numerically_valid=False,
                               validation="output_mismatch: tensor 'Y' element 2 is 9.9e+01")
    row = _collect(monkeypatch, tmp_path, [page, timed])[0][0]

    assert row["numerically_valid"] is False
    assert row["validation"].startswith("output_mismatch")
    # Suppressed where the row is built, so `corpus.json`, `corpus.csv` and the `evaluate`
    # regret pass that reads the corpus back all see the same absence.
    assert [row[column] for column in ("robustMeanMs", "minTimeMs", "avgTimeMs", "stddevMs")] == [None] * 4
    # No derived rate either: a throughput computed from a suppressed time would put the
    # measurement back under a different column name.
    assert "tflops" not in row
    # The candidate stays visible: which kernel was wrong, on which problem, and why.
    assert (row["kernel"], row["benchmark"], row["succeeded"], row["is_valid"]) == ("kernel-a", "graph", True, True)


def test_an_undecided_verdict_is_carried_rather_than_treated_as_correct(monkeypatch, tmp_path):
    """A null verdict keeps its measurement but never claims the candidate was checked.

    Open Question 19(a) has not settled what reference each op validates against, so a
    single-candidate problem is genuinely undecidable. Suppressing those timings would
    train on nothing; recording them as `True` would be the silent pass §13.2 forbids.
    """
    page = _page()
    timed = _timing(page)
    timed["results"][0].update(numerically_valid=None, validation="no_reference: one candidate ran")
    row = _collect(monkeypatch, tmp_path, [page, timed])[0][0]

    assert row["numerically_valid"] is None
    assert row["robustMeanMs"] == 2.5


@pytest.mark.parametrize("missing", ["numerically_valid", "validation"])
def test_a_benchmark_that_records_no_verdict_is_refused(monkeypatch, tmp_path, missing):
    """§13.2 records the verdict on the row, so an absent one is a tool that did not check.

    Refused rather than defaulted. Any default is wrong: `True` inverts the oracle, and
    `None` would let a benchmark silently regress out of validating while the corpus still
    looks well-formed.
    """
    page = _page()
    timed = _timing(page)
    timed["results"][0].pop(missing)
    with pytest.raises(ValueError, match="numerical-validation verdict"):
        _collect(monkeypatch, tmp_path, [page, timed])
