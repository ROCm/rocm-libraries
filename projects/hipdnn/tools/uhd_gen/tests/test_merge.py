# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Joining sweeps from several boards of one architecture."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uhd_gen.merge import MergeError, merge_corpora  # noqa: E402


def _corpus(
    path: Path, device: str, *, problems: int = 4, extra: dict | None = None
) -> Path:
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
    a = _corpus(tmp_path / "a.csv", "aaaa")
    b = _corpus(tmp_path / "b.csv", "bbbb")

    merged, report = merge_corpora([a, b])

    assert len(merged) == 16
    assert report["devices"] == ["aaaa", "bbbb"]
    # A problem is (benchmark, device), so one graph on two boards is two problems.
    assert report["problems"] == 8
    assert not report["repeated_devices"]


def test_a_column_present_in_only_one_corpus_is_refused(tmp_path):
    """The column would be NaN on one board's rows and train as a fact about it."""
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
    """Resampling a board is legitimate but gives it extra weight."""
    a = _corpus(tmp_path / "a.csv", "aaaa")
    b = _corpus(tmp_path / "b.csv", "aaaa")

    with caplog.at_level("WARNING"):
        merged, report = merge_corpora([a, b])

    assert len(merged) == 16
    assert list(report["repeated_devices"]) == ["aaaa"]
    assert "aaaa" in caplog.text
    assert "more weight" in caplog.text


def test_a_corpus_without_a_device_column_is_refused(tmp_path):
    """Without `device`, boards collapse into one oracle and regret reads too small."""
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
    """`resolve_grouping` still keys the merged corpus on (benchmark, device)."""
    from uhd_gen.evaluate import resolve_grouping

    merged, _ = merge_corpora(
        [_corpus(tmp_path / "a.csv", "aaaa"), _corpus(tmp_path / "b.csv", "bbbb")]
    )
    grouping = resolve_grouping(merged)

    assert not grouping.degraded
    assert set(grouping.columns) == {"benchmark", "device"}
    assert merged.groupby(list(grouping.columns)).ngroups == 8
