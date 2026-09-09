# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Joining sweeps from several boards of one architecture.

A UHD is arch-keyed, so one gfx942 model serves MI300X, MI325X, MI308X and MI300A, and
the corpus it trains on should hold all of them. The runtime already separates them --
a problem is `(benchmark, device)` and `device` is the DeviceKey hash -- so the join
itself is a concatenation.

What needs defending is the two ways it goes wrong without saying so: a corpus whose
columns do not match, which trains a NaN as a fact about one machine, and the same board
merged twice, which silently doubles its weight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uhd_gen.merge import MergeError, merge_corpora  # noqa: E402


def _corpus(path: Path, device: str, *, problems: int = 4, extra: dict | None = None) -> Path:
    rows = []
    for problem in range(problems):
        for block_m in (64, 256):
            row = {
                "benchmark": f"prob{problem}",
                "device": device,
                "kernel": f"k{block_m}",
                "robustMeanMs": 0.05 if block_m == 256 else 0.09,
                "is_valid": "true",
                "kernel.block_m": block_m,
                "device.cu_count": 304,
            }
            row.update(extra or {})
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_two_boards_join_into_one_corpus(tmp_path):
    """The whole point: one arch-keyed model, several boards' measurements."""
    a = _corpus(tmp_path / "a.csv", "aaaa")
    b = _corpus(tmp_path / "b.csv", "bbbb")

    merged, report = merge_corpora([a, b])

    assert len(merged) == 16
    assert report["devices"] == ["aaaa", "bbbb"]
    # Problem identity is the pair, so the same graph on two boards is two problems --
    # which is exactly why merging does not conflate their oracles.
    assert report["problems"] == 8
    assert not report["repeated_devices"]


def test_a_column_present_in_only_one_corpus_is_refused(tmp_path):
    """The silent failure. Concatenating leaves the column NaN precisely where one
    machine's rows are, so the model learns the absence as a property of that board."""
    a = _corpus(tmp_path / "a.csv", "aaaa")
    b = _corpus(tmp_path / "b.csv", "bbbb", extra={"device.total_global_mem": 192})

    with pytest.raises(MergeError, match="same feature space"):
        merge_corpora([a, b])


def test_the_refusal_names_the_column_and_the_fix(tmp_path):
    a = _corpus(tmp_path / "a.csv", "aaaa")
    b = _corpus(tmp_path / "b.csv", "bbbb", extra={"device.total_global_mem": 192})

    with pytest.raises(MergeError) as caught:
        merge_corpora([a, b])
    message = str(caught.value)
    assert "device.total_global_mem" in message
    assert "same build" in message, "a refusal without a remedy just blocks the user"


def test_the_same_board_twice_is_warned_about_not_refused(tmp_path, caplog):
    """Resampling one noisy card is legitimate, but those rows share problem identity,
    so that board ends up weighted more heavily than the others in every figure."""
    a = _corpus(tmp_path / "a.csv", "aaaa")
    b = _corpus(tmp_path / "b.csv", "aaaa")

    with caplog.at_level("WARNING"):
        merged, report = merge_corpora([a, b])

    assert len(merged) == 16
    assert list(report["repeated_devices"]) == ["aaaa"]
    assert "aaaa" in caplog.text
    assert "more weight" in caplog.text


def test_a_corpus_without_a_device_column_is_refused(tmp_path):
    """Without the device half of the key both boards collapse into one oracle, and
    every regret figure computed from the result is too small."""
    a = _corpus(tmp_path / "a.csv", "aaaa")
    frame = pd.read_csv(a).drop(columns=["device"])
    stripped = tmp_path / "stripped.csv"
    frame.to_csv(stripped, index=False)
    b = _corpus(tmp_path / "b.csv", "bbbb")

    with pytest.raises(MergeError, match="device"):
        merge_corpora([stripped, b])


def test_merging_one_corpus_is_refused(tmp_path):
    with pytest.raises(MergeError, match="at least two"):
        merge_corpora([_corpus(tmp_path / "a.csv", "aaaa")])


def test_an_empty_corpus_is_refused(tmp_path):
    a = _corpus(tmp_path / "a.csv", "aaaa")
    empty = tmp_path / "empty.csv"
    pd.read_csv(a).iloc[0:0].to_csv(empty, index=False)

    with pytest.raises(MergeError, match="no rows"):
        merge_corpora([a, empty])


def test_the_merged_corpus_still_groups_per_board(tmp_path):
    """The join has to survive the thing that consumes it: resolve_grouping keys on
    (benchmark, device), and a merge that lost the device column or collided its values
    would silently halve the problem count."""
    from uhd_gen.evaluate import resolve_grouping

    merged, _ = merge_corpora(
        [_corpus(tmp_path / "a.csv", "aaaa"), _corpus(tmp_path / "b.csv", "bbbb")]
    )
    grouping = resolve_grouping(merged)

    assert not grouping.degraded
    assert set(grouping.columns) == {"benchmark", "device"}
    assert merged.groupby(list(grouping.columns)).ngroups == 8
