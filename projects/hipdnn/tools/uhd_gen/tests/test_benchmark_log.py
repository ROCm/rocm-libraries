#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the ingestor benchmark log -> RFC 0019.13 §8.3 CSV exporter.

The records these parse are produced by `BenchmarkPlan::logCandidateTiming`; the
shapes below are copied from what it emits, and the C++ side asserts the same
field names (TestBenchmarkPlan.EveryTimedCandidateIsLoggedAsAParsableRecord).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from uhd_gen.benchmark_log import (
    CANDIDATE_EVENT,
    ENVELOPE_COLUMNS,
    SweepProvenance,
    convert,
    main,
)


def _ok(benchmark: str, kernel: str, min_ms: float, avg_ms: float = None) -> str:
    return json.dumps(
        {
            "event": CANDIDATE_EVENT,
            "benchmark": benchmark,
            "kernel": kernel,
            "pack": "pack-1",
            "dispatch": "dispatch-1",
            "status": "ok",
            "min_ms": min_ms,
            "avg_ms": avg_ms if avg_ms is not None else min_ms,
            "stddev_ms": 0.0004,
            "robust_mean_ms": min_ms,
            "iters": 7,
        }
    )


def _failed(benchmark: str, kernel: str, reason: str) -> str:
    return json.dumps(
        {
            "event": CANDIDATE_EVENT,
            "benchmark": benchmark,
            "kernel": kernel,
            "pack": "pack-1",
            "dispatch": "dispatch-1",
            "status": "failed",
            "reason": reason,
        }
    )


def _write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestExtraction:
    def test_every_candidate_becomes_a_row(self, tmp_path: Path):
        log = _write_log(
            tmp_path / "sweep.log",
            [_ok("graph-a", "k1", 0.0261), _ok("graph-a", "k2", 0.0247)],
        )

        convert([log], tmp_path / "out.csv")

        rows = _read_csv(tmp_path / "out.csv")
        assert [row["kernel"] for row in rows] == ["k1", "k2"]
        assert all(row["benchmark"] == "graph-a" for row in rows)
        assert all(row["is_valid"] == "True" for row in rows)

    def test_surrounding_log_noise_is_ignored(self, tmp_path: Path):
        """A real log interleaves these with everything else the process says."""
        log = _write_log(
            tmp_path / "sweep.log",
            [
                "[info] hipdnn: some unrelated prose",
                _ok("graph-a", "k1", 0.02),
                "[warn] ingestor: benchmarking candidate 'k9' threw",
                '{"event":"something.else","kernel":"k8"}',
                _ok("graph-a", "k2", 0.03),
            ],
        )

        convert([log], tmp_path / "out.csv")

        assert [row["kernel"] for row in _read_csv(tmp_path / "out.csv")] == ["k1", "k2"]

    def test_a_severity_prefix_does_not_hide_the_record(self, tmp_path: Path):
        """The sink prepends a timestamp and level; the object starts mid-line."""
        log = _write_log(
            tmp_path / "sweep.log",
            ["2026-08-27 10:00:00 [info] [hip_kernel_provider] " + _ok("g", "k1", 0.5)],
        )

        convert([log], tmp_path / "out.csv")

        rows = _read_csv(tmp_path / "out.csv")
        assert len(rows) == 1
        assert rows[0]["kernel"] == "k1"

    def test_several_logs_concatenate(self, tmp_path: Path):
        """A sharded campaign writes one log per worker."""
        first = _write_log(tmp_path / "a.log", [_ok("graph-a", "k1", 0.02)])
        second = _write_log(tmp_path / "b.log", [_ok("graph-b", "k1", 0.03)])

        convert([first, second], tmp_path / "out.csv")

        rows = _read_csv(tmp_path / "out.csv")
        assert [row["benchmark"] for row in rows] == ["graph-a", "graph-b"]

    def test_a_truncated_line_does_not_lose_the_corpus(self, tmp_path: Path):
        """Killed runs leave a half-written tail; the rest is still good data."""
        log = _write_log(
            tmp_path / "sweep.log",
            [_ok("graph-a", "k1", 0.02), '{"event":"ingestor.benchmark.cand'],
        )

        stats = convert([log], tmp_path / "out.csv")

        assert stats.malformed == 1
        assert len(_read_csv(tmp_path / "out.csv")) == 1


class TestSchema:
    def test_header_is_the_declared_envelope(self, tmp_path: Path):
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02)])

        convert([log], tmp_path / "out.csv")

        with open(tmp_path / "out.csv", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(ENVELOPE_COLUMNS)

    def test_a_failed_pair_is_recorded_with_no_timings(self, tmp_path: Path):
        """§8.3 rule 6: timing columns empty when is_valid is False.

        Empty rather than zero. A zero in minTimeMs is a measurement of zero, and
        a trainer filtering on `is_valid` alone would still fit to it if the
        column were populated.
        """
        log = _write_log(
            tmp_path / "sweep.log",
            [_ok("g", "k1", 0.02), _failed("g", "k2", "launch-not-timed")],
        )

        convert([log], tmp_path / "out.csv")

        rows = _read_csv(tmp_path / "out.csv")
        failed = next(row for row in rows if row["kernel"] == "k2")
        assert failed["is_valid"] == "False"
        assert failed["skip_reason"] == "launch-not-timed"
        assert failed["minTimeMs"] == ""
        assert failed["avgTimeMs"] == ""
        assert failed["stddevMs"] == ""
        assert failed["iters"] == ""

    def test_a_measured_row_has_no_skip_reason(self, tmp_path: Path):
        """§8.3 rule 7, the converse of the above."""
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02)])

        convert([log], tmp_path / "out.csv")

        assert _read_csv(tmp_path / "out.csv")[0]["skip_reason"] == ""

    def test_min_does_not_exceed_avg(self, tmp_path: Path):
        """§8.3 rule 4, carried through from the emitter rather than recomputed."""
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02, avg_ms=0.03)])

        convert([log], tmp_path / "out.csv")

        row = _read_csv(tmp_path / "out.csv")[0]
        assert float(row["minTimeMs"]) <= float(row["avgTimeMs"])


class TestProvenance:
    def test_sweep_fields_are_stamped_on_every_row(self, tmp_path: Path):
        """The five columns the runtime cannot know (§8.7, §8.8).

        They describe the campaign, not the dispatch, so they come from whoever
        drove it. Stamped uniformly because §8.3 rule 9 requires them identical
        across rows sharing a problem.
        """
        log = _write_log(
            tmp_path / "sweep.log", [_ok("g", "k1", 0.02), _ok("g", "k2", 0.03)]
        )

        convert(
            [log],
            tmp_path / "out.csv",
            SweepProvenance(
                collection_mode="targeted",
                problem_complete=True,
                shard_id=3,
                config_set_hash="sha256:abc",
                applicability_id="pointwise-v1",
            ),
        )

        for row in _read_csv(tmp_path / "out.csv"):
            assert row["collection_mode"] == "targeted"
            assert row["problem_complete"] == "True"
            assert row["shard_id"] == "3"
            assert row["config_set_hash"] == "sha256:abc"
            assert row["applicability_id"] == "pointwise-v1"

    def test_problem_complete_defaults_false(self, tmp_path: Path):
        """Claiming completeness by default would be a lie the runtime can't check."""
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02)])

        convert([log], tmp_path / "out.csv")

        assert _read_csv(tmp_path / "out.csv")[0]["problem_complete"] == "False"


class TestCli:
    def test_an_empty_log_is_an_error_that_names_the_cause(self, tmp_path: Path, caplog):
        """The overwhelmingly likely cause is forgetting HIPDNN_LOG_LEVEL.

        Silently writing a header-only CSV would send someone to debug their
        training data instead of their sweep command.
        """
        log = _write_log(tmp_path / "sweep.log", ["[info] nothing of interest here"])

        exit_code = main([str(log), "-o", str(tmp_path / "out.csv")])

        assert exit_code == 1
        assert "HIPDNN_LOG_LEVEL=info" in caplog.text

    def test_a_populated_log_exits_zero(self, tmp_path: Path):
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02)])

        assert main([str(log), "-o", str(tmp_path / "out.csv")]) == 0
