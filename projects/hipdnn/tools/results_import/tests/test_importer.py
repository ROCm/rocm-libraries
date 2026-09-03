# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The CSV-to-Parquet boundary, which is where RFC 0019.13 §8.3's checks are finally applied."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from results_import.importer import ValidationError, build_dataset, load_csvs, write_parquet

OPERATIONS = pathlib.Path(__file__).resolve().parents[2] / "corpus_gen" / "operations"


@pytest.fixture
def matmul() -> dict:
    with (OPERATIONS / "matmul.opmeta.json").open() as handle:
        return json.load(handle)


def rows(**overrides) -> pd.DataFrame:
    base = {
        "q.M": [1024, 1024], "q.N": [1024, 1024], "q.K": [1024, 1024],
        "q.dtype": ["float32", "float32"],
        "kernel.tile_m": [64, 128], "device.cu_count": [80, 80],
        "minTimeMs": [1.0, 2.0], "avgTimeMs": [1.1, 2.1],
        "stddevMs": [0.01, 0.01], "iters": [10, 10],
        "error": ["", ""],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_metrics_are_added_and_collection_bookkeeping_is_dropped(matmul):
    frame = rows(shard_id=[3, 3])
    out = build_dataset(frame, matmul)

    assert "tflops" in out.columns and "gbs" in out.columns
    assert out["tflops"].iloc[0] == pytest.approx(2 * 1024**3 / 1e-3 / 1e12)
    # shard_id distinguishes CSV parts during the gather and means nothing once merged.
    assert "shard_id" not in out.columns


def test_absent_optional_columns_take_their_defaults(matmul):
    """A foreign CSV carries q.*, kernel.*, device.* and a measurement, and nothing else."""
    out = build_dataset(rows(), matmul)
    assert out["problem_complete"].all()


def test_a_failed_row_keeps_null_metrics_and_downgrades_its_problem(matmul):
    """The failure is information about a candidate, so the row stays -- but the problem can no
    longer claim its space was fully measured, or regret over it reads as exact when it is not."""
    frame = rows(minTimeMs=[1.0, None], avgTimeMs=[1.1, None], stddevMs=[0.01, None],
                 iters=[10, None], error=["", "HIP error 700"], problem_complete=[True, True])
    out = build_dataset(frame, matmul)

    assert pd.isna(out["tflops"].iloc[1]) and pd.isna(out["gbs"].iloc[1])
    assert not out["problem_complete"].any(), "the errored candidate left the problem complete"


def test_a_row_claiming_both_a_measurement_and_an_error_is_rejected(matmul):
    frame = rows(error=["", "HIP error 700"])
    with pytest.raises(ValidationError, match="both"):
        build_dataset(frame, matmul)


def test_a_row_with_neither_is_rejected(matmul):
    """A pair that was never attempted does not belong in the results at all -- it is a filtered
    configuration, and those are recorded in the run's report."""
    frame = rows(minTimeMs=[1.0, None], avgTimeMs=[1.1, None])
    with pytest.raises(ValidationError, match="neither"):
        build_dataset(frame, matmul)


def test_a_complete_problem_spanning_two_candidate_sets_is_rejected(matmul):
    """The merge check. Two collections of one problem, each claiming completeness, that repeat a
    configuration cannot both be the whole candidate set."""
    frame = pd.concat([rows(), rows()], ignore_index=True)
    frame["problem_complete"] = True
    with pytest.raises(ValidationError, match="candidate sets"):
        build_dataset(frame, matmul)


def test_inconsistent_completeness_across_one_problem_is_rejected(matmul):
    frame = rows(problem_complete=[True, False])
    with pytest.raises(ValidationError, match="disagrees"):
        build_dataset(frame, matmul)


def test_min_above_avg_is_rejected(matmul):
    with pytest.raises(ValidationError, match="minTimeMs"):
        build_dataset(rows(minTimeMs=[9.0, 2.0]), matmul)


def test_a_corpus_that_identifies_no_problem_is_rejected(matmul):
    frame = rows().drop(columns=["q.M", "q.N", "q.K", "q.dtype"])
    with pytest.raises(ValidationError, match="q"):
        build_dataset(frame, matmul)


def test_shards_concatenate_and_round_trip_through_parquet(tmp_path, matmul):
    """Appending is why collection stays CSV; this is the merge, and the publish after it.

    Skips without pyarrow rather than failing: pandas carries no Parquet engine of its own, and
    an environment lacking one should report a skip, the way uhd_gen's suite treats a missing
    lightgbm. requirements.txt declares it for environments that do publish.
    """
    pytest.importorskip("pyarrow")
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    rows().to_csv(first, index=False)
    rows(**{"q.M": [512, 512]}).to_csv(second, index=False)

    out = build_dataset(load_csvs([first, second]), matmul)
    assert len(out) == 4

    destination = tmp_path / "out" / "results.parquet"
    write_parquet(out, destination)
    back = pd.read_parquet(destination)

    # Values, not dtype identity. pyarrow normalises pandas' object column to a real string
    # dtype on the way back, which is more correct rather than less -- asserting frame equality
    # would be testing pandas/pyarrow's type mapping instead of anything this module does.
    pd.testing.assert_frame_equal(back, out, check_dtype=False)

    # What the round trip actually has to preserve: a null stays null rather than becoming an
    # empty string or a zero, since null is the whole signal that a row has no measurement.
    nulled = build_dataset(
        rows(minTimeMs=[1.0, None], avgTimeMs=[1.1, None], stddevMs=[0.01, None],
             iters=[10, None], error=["", "HIP error 700"]),
        matmul,
    )
    write_parquet(nulled, destination)
    reread = pd.read_parquet(destination)
    assert reread["tflops"].isna().tolist() == [False, True]
    assert reread["minTimeMs"].isna().tolist() == [False, True]


def test_an_empty_csv_field_reads_back_as_a_null_metric(tmp_path, matmul):
    """The encoding §8.3 specifies: null is the empty field, and it survives to the dataset."""
    path = tmp_path / "partial.csv"
    path.write_text(
        "q.M,q.N,q.K,q.dtype,kernel.tile_m,device.cu_count,"
        "minTimeMs,avgTimeMs,stddevMs,iters,error\n"
        "1024,1024,1024,float32,64,80,,,,,HIP error 700\n"
    )
    out = build_dataset(load_csvs([path]), matmul)
    assert pd.isna(out["tflops"].iloc[0])
    assert out["error"].iloc[0] == "HIP error 700"
