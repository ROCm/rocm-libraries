# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The documented chain, end to end: sweep CSV -> uhd_gen.dataset -> train."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uhd_gen.dataset.publish import (
    build_dataset,
    load_csvs,
    write_parquet,
)  # noqa: E402
from uhd_gen.corpus_io import read_corpus_frame  # noqa: E402
from uhd_gen.merge import merge_corpora  # noqa: E402


#: A numeric-looking identity: read as an integer it would become a different name.
DEVICE = "0007"

PROBLEMS = 8
TILES = (64, 128, 256)

#: Tile of the failed candidates; distinct so they do not repeat a measured candidate.
FAILED_TILE = 512


def _collected(path: Path, *, device: str = DEVICE, failures: int = 2) -> Path:
    """What `uhd_gen export-benchmarks` writes: the §8.3 envelope with `is_valid`."""
    rows = []
    for index in range(PROBLEMS):
        for tile in TILES:
            elapsed = 1.0 + index / 10 + tile / 1024
            rows.append(
                {
                    "benchmark": f"{index:04d}",
                    "device": device,
                    "kernel": f"tile{tile}",
                    "is_valid": "True",
                    "skip_reason": "",
                    "minTimeMs": elapsed,
                    "avgTimeMs": elapsed * 1.05,
                    "stddevMs": 0.01,
                    "iters": 20,
                    "problem_complete": "True",
                    "q.M": 256 * (index + 1),
                    "q.N": 512,
                    "q.K": 128,
                    "q.dtype": "fp32",
                    "q.flops": 2 * 256 * (index + 1) * 512 * 128,
                    "q.bytes": 4
                    * (256 * (index + 1) * 512 + 512 * 128 + 256 * (index + 1) * 128),
                    "kernel.tile_m": tile,
                    "device.cu_count": 304,
                }
            )
    for index in range(failures):
        rows.append(
            {
                "benchmark": f"{index:04d}",
                "device": device,
                "kernel": f"tile{FAILED_TILE}",
                "is_valid": "False",
                "skip_reason": "hip error 700",
                "minTimeMs": "",
                "avgTimeMs": "",
                "stddevMs": "",
                "iters": "",
                "problem_complete": "True",
                "q.M": 256 * (index + 1),
                "q.N": 512,
                "q.K": 128,
                "q.dtype": "fp32",
                "kernel.tile_m": FAILED_TILE,
                "device.cu_count": 304,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _publish(csv_path: Path, destination: Path) -> pd.DataFrame:
    dataset = build_dataset(load_csvs([csv_path]))
    write_parquet(dataset, destination)
    return dataset


def _provenance(tmp_path: Path) -> Path:
    path = tmp_path / "provenance.json"
    path.write_text(
        json.dumps(
            {
                "ued": {"id": str(uuid.uuid4()), "revision": "1.0"},
                "kmd": {"id": str(uuid.uuid4()), "revision": "1.0"},
                "umd": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _train(corpus: Path, output_dir: Path, provenance: Path) -> dict:
    """Train on `corpus`; the manifest's `num_samples` is what was fitted."""
    from uhd_gen.__main__ import main

    assert (
        main(
            [
                "train",
                "--input",
                str(corpus),
                "--provenance",
                str(provenance),
                "--features",
                "q.M",
                "kernel.tile_m",
                "device.cu_count",
                "--target",
                "avgTimeMs",
                "--objective",
                "min",
                "--timing-statistic",
                "avgTimeMs",
                "--num-boost-round",
                "4",
                "--early-stopping",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    return json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))


def test_the_collected_csv_and_the_dataset_published_from_it_train_the_same_rows(
    tmp_path, evaluator
):
    """Failure is `is_valid=False` in CSV but a null `avgTimeMs` in the dataset."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")

    collected = _collected(tmp_path / "bench.csv")
    dataset = _publish(collected, tmp_path / "dataset.parquet")
    provenance = _provenance(tmp_path)

    from_csv = _train(collected, tmp_path / "from_csv", provenance)
    from_parquet = _train(
        tmp_path / "dataset.parquet", tmp_path / "from_parquet", provenance
    )

    measured = PROBLEMS * len(TILES)
    assert from_csv["num_samples"] == measured
    assert from_parquet["num_samples"] == measured

    # Failed rows stay in the dataset (§8.3); they are dropped at the fit.
    assert len(dataset) == measured + 2
    assert "is_valid" not in dataset.columns


def test_an_identity_is_the_same_name_whichever_format_it_arrives_in(tmp_path):
    """`0007` is a device's name, not the number seven."""
    collected = _collected(tmp_path / "bench.csv")
    _publish(collected, tmp_path / "dataset.parquet")

    from_csv = read_corpus_frame(collected)
    from_parquet = read_corpus_frame(tmp_path / "dataset.parquet")

    assert (
        from_csv["device"].tolist()
        == from_parquet["device"].tolist()
        == [DEVICE] * len(from_csv)
    )
    assert from_csv["benchmark"].tolist() == from_parquet["benchmark"].tolist()
    assert from_parquet["benchmark"].iloc[0] == "0000"


def test_an_identity_frozen_as_an_integer_by_a_producer_still_reads_as_a_name(tmp_path):
    """Identities are pinned to strings at the read, whatever type the writer chose."""
    path = tmp_path / "foreign.parquet"
    pd.DataFrame(
        {"benchmark": [7, 8], "device": [7, 7], "avgTimeMs": [1.0, 2.0]}
    ).to_parquet(path, index=False)

    frame = read_corpus_frame(path)

    assert frame["benchmark"].tolist() == ["7", "8"]
    assert frame["device"].tolist() == ["7", "7"]


def test_knobs_reads_the_published_dataset(tmp_path, capsys):
    """`knobs` reads `--input dataset.parquet` the same way `train` does."""
    # The CLI imports the trainer and the FlatBuffer converter.
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    from uhd_gen.__main__ import main

    collected = _collected(tmp_path / "bench.csv")
    _publish(collected, tmp_path / "dataset.parquet")

    assert (
        main(
            [
                "knobs",
                "--input",
                str(tmp_path / "dataset.parquet"),
                "--target",
                "avgTimeMs",
                "--objective",
                "min",
            ]
        )
        == 0
    )
    # Grouped by (benchmark, device), with the two failed candidates dropped.
    assert "8 problem(s), 24 measurement(s)" in capsys.readouterr().out


def test_merge_joins_published_datasets_and_publishes_one(tmp_path):
    """A merged `.parquet` output is written as Parquet, not CSV."""
    # The CLI imports the trainer and the FlatBuffer converter.
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    from uhd_gen.__main__ import main

    paths = []
    for device in ("0007", "0008"):
        collected = _collected(tmp_path / f"bench-{device}.csv", device=device)
        published = tmp_path / f"dataset-{device}.parquet"
        _publish(collected, published)
        paths.append(published)

    merged, report = merge_corpora(paths)
    assert report["devices"] == ["0007", "0008"]
    # A problem is (benchmark, device), so two boards double the problem count.
    assert report["problems"] == 2 * PROBLEMS

    output = tmp_path / "merged.parquet"
    assert main(["merge", str(paths[0]), str(paths[1]), "--output", str(output)]) == 0
    assert len(read_corpus_frame(output)) == len(merged)
