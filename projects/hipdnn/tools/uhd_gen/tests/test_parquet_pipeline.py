# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The documented chain, end to end: sweep CSV -> results_import -> train.

RFC 0019.13 §8.3 collects as CSV and publishes as Parquet, and the README tells a reader
to run one into the other. The two ends spell "this candidate never ran" differently --
the collector writes `is_valid=False` and a `skip_reason`, the published dataset writes a
null measurement and an `error`, because a dataset that records the fact twice can
contradict itself -- and nothing else in the suite reads a corpus in both formats. So
these train the same corpus by both routes and require the same answer: the same rows
reach the fit, and an identity is the same name in both.

What breaks without them is quiet. A failed candidate that survives into the fit arrives
as a NaN target, and the error the trainer then raises names the corpus rather than the
missing filter; an identity inferred as an integer from one format and a string from the
other splits a problem that should be one.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from results_import.importer import build_dataset, load_csvs, write_parquet  # noqa: E402
from uhd_gen.corpus_io import read_corpus_frame  # noqa: E402
from uhd_gen.merge import merge_corpora  # noqa: E402

OPERATIONS = Path(__file__).resolve().parents[2] / "corpus_gen" / "operations"

#: Identities that look like numbers. `0123` read as an integer is 123, which is a
#: different name, and the round trip through two formats is where the two spellings meet.
DEVICE = "0007"

PROBLEMS = 8
TILES = (64, 128, 256)

#: Candidates that failed. A distinct tile, because a failed candidate is a candidate:
#: repeating a measured one would be a second collection of the same candidate set.
FAILED_TILE = 512


@pytest.fixture
def matmul() -> dict:
    with (OPERATIONS / "matmul.opmeta.json").open() as handle:
        return json.load(handle)


def _collected(path: Path, *, device: str = DEVICE, failures: int = 2) -> Path:
    """What `uhd_gen export-benchmarks` writes: the §8.3 envelope with a validity flag."""
    rows = []
    for index in range(PROBLEMS):
        for tile in TILES:
            elapsed = 1.0 + index / 10 + tile / 1024
            rows.append({
                "benchmark": f"{index:04d}", "device": device, "kernel": f"tile{tile}",
                "is_valid": "True", "skip_reason": "",
                "minTimeMs": elapsed, "avgTimeMs": elapsed * 1.05,
                "stddevMs": 0.01, "iters": 20, "problem_complete": "True",
                "q.M": 256 * (index + 1), "q.N": 512, "q.K": 128, "q.dtype": "fp32",
                "kernel.tile_m": tile, "device.cu_count": 304,
            })
    for index in range(failures):
        rows.append({
            "benchmark": f"{index:04d}", "device": device, "kernel": f"tile{FAILED_TILE}",
            "is_valid": "False", "skip_reason": "hip error 700",
            "minTimeMs": "", "avgTimeMs": "", "stddevMs": "", "iters": "",
            "problem_complete": "True",
            "q.M": 256 * (index + 1), "q.N": 512, "q.K": 128, "q.dtype": "fp32",
            "kernel.tile_m": FAILED_TILE, "device.cu_count": 304,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _publish(csv_path: Path, destination: Path, opmeta: dict) -> pd.DataFrame:
    dataset = build_dataset(load_csvs([csv_path]), opmeta)
    write_parquet(dataset, destination)
    return dataset


def _provenance(tmp_path: Path) -> Path:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({
        "ued": {"id": str(uuid.uuid4()), "revision": "1.0"},
        "kmd": {"id": str(uuid.uuid4()), "revision": "1.0"},
        "umd": [],
    }), encoding="utf-8")
    return path


def _train(corpus: Path, output_dir: Path, provenance: Path) -> dict:
    """Train on `corpus` and return the manifest, whose `num_samples` is what was fitted."""
    from uhd_gen.__main__ import main

    assert main([
        "train", "--input", str(corpus), "--provenance", str(provenance),
        "--features", "q.M", "kernel.tile_m", "device.cu_count",
        "--target", "avgTimeMs", "--objective", "min", "--timing-statistic", "avgTimeMs",
        "--num-boost-round", "4", "--early-stopping", "2",
        "--output-dir", str(output_dir),
    ]) == 0
    return json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))


def test_the_collected_csv_and_the_dataset_published_from_it_train_the_same_rows(tmp_path, matmul):
    """The chain the README documents, run end to end.

    Both routes must drop the two candidates that never ran and fit the rest. The collector
    spells that `is_valid=False`; the published dataset spells it as a null `avgTimeMs` and a
    non-empty `error`, and has no validity flag at all (§8.3). Reading only the flag, the
    dataset's failed rows reach the fit as NaN targets and `train_uhd` refuses the corpus with
    "target 'avgTimeMs' must contain finite nonnegative values" -- an error about the data that
    is really about the reader.
    """
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")

    collected = _collected(tmp_path / "bench.csv")
    dataset = _publish(collected, tmp_path / "dataset.parquet", matmul)
    provenance = _provenance(tmp_path)

    from_csv = _train(collected, tmp_path / "from_csv", provenance)
    from_parquet = _train(tmp_path / "dataset.parquet", tmp_path / "from_parquet", provenance)

    measured = PROBLEMS * len(TILES)
    assert from_csv["num_samples"] == measured
    assert from_parquet["num_samples"] == measured

    # The failed rows are still in the dataset -- a candidate that could not run is information
    # about feature space, and §8.3 keeps it. They are dropped at the fit, not at the import.
    assert len(dataset) == measured + 2
    assert "is_valid" not in dataset.columns


def test_an_identity_is_the_same_name_whichever_format_it_arrives_in(tmp_path, matmul):
    """`0007` is a device's name, not the number seven.

    The CSV reader infers unless told, and Parquet returns whatever type the writer froze, so
    without pinning at both reads the same board is `"0007"` on one route and `7` on the other.
    Every consumer of identity compares it -- problem grouping, the immediate corpus's
    canonical-string check -- and two spellings are two problems.
    """
    collected = _collected(tmp_path / "bench.csv")
    _publish(collected, tmp_path / "dataset.parquet", matmul)

    from_csv = read_corpus_frame(collected)
    from_parquet = read_corpus_frame(tmp_path / "dataset.parquet")

    assert from_csv["device"].tolist() == from_parquet["device"].tolist() == [DEVICE] * len(from_csv)
    assert from_csv["benchmark"].tolist() == from_parquet["benchmark"].tolist()
    assert from_parquet["benchmark"].iloc[0] == "0000"


def test_an_identity_frozen_as_an_integer_by_a_producer_still_reads_as_a_name(tmp_path):
    """The dataset need not come from our importer. A foreign publisher that wrote the column as
    int64 is read the same way here, because the pinning happens at the read rather than
    depending on what the writer chose.
    """
    path = tmp_path / "foreign.parquet"
    pd.DataFrame({"benchmark": [7, 8], "device": [7, 7], "avgTimeMs": [1.0, 2.0]}).to_parquet(
        path, index=False)

    frame = read_corpus_frame(path)

    assert frame["benchmark"].tolist() == ["7", "8"]
    assert frame["device"].tolist() == ["7", "7"]


def test_knobs_reads_the_published_dataset(tmp_path, matmul, capsys):
    """`knobs` analyses the corpus `train` fits, so `--input dataset.parquet` has to mean the
    same file in both. Read with `pd.read_csv` it died inside pandas on the Parquet magic bytes.
    """
    from uhd_gen.__main__ import main

    collected = _collected(tmp_path / "bench.csv")
    _publish(collected, tmp_path / "dataset.parquet", matmul)

    assert main(["knobs", "--input", str(tmp_path / "dataset.parquet"),
                 "--target", "avgTimeMs", "--objective", "min"]) == 0
    # The dataset was read, grouped by (benchmark, device) and stripped of the two candidates
    # that never ran -- which is the corpus `train` fits, arrived at through a different reader.
    assert "8 problem(s), 24 measurement(s)" in capsys.readouterr().out


def test_merge_joins_published_datasets_and_publishes_one(tmp_path, matmul):
    """Two boards' datasets merge into the corpus a single arch-keyed model is fitted on, and the
    merged file keeps the format its name claims -- a `.parquet` written as CSV is a file `train`
    hands straight to `read_parquet`.
    """
    from uhd_gen.__main__ import main

    paths = []
    for device in ("0007", "0008"):
        collected = _collected(tmp_path / f"bench-{device}.csv", device=device)
        published = tmp_path / f"dataset-{device}.parquet"
        _publish(collected, published, matmul)
        paths.append(published)

    merged, report = merge_corpora(paths)
    assert report["devices"] == ["0007", "0008"]
    # A problem is (benchmark, device), so two boards over one graph set is twice the problems.
    assert report["problems"] == 2 * PROBLEMS

    output = tmp_path / "merged.parquet"
    assert main(["merge", str(paths[0]), str(paths[1]), "--output", str(output)]) == 0
    assert len(read_corpus_frame(output)) == len(merged)
