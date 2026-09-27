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


def _ok(
    benchmark: str,
    kernel: str,
    min_ms: float,
    avg_ms: float = None,
    features: dict | None = None,
    device: str | None = None,
) -> str:
    record = {
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
    # Omitted rather than emitted empty when absent, matching the runtime: a log
    # collected before BenchmarkPlan carried a device identity has no such key, and
    # the exporter has to keep working on it. Every default-argument caller below is
    # therefore also a regression test for that older shape.
    if device is not None:
        record["device"] = device
    record.update(features or {})
    return json.dumps(record)


def _failed(
    benchmark: str,
    kernel: str,
    reason: str,
    features: dict | None = None,
    device: str | None = None,
) -> str:
    record = {
        "event": CANDIDATE_EVENT,
        "benchmark": benchmark,
        "kernel": kernel,
        "pack": "pack-1",
        "dispatch": "dispatch-1",
        "status": "failed",
        "reason": reason,
    }
    if device is not None:
        record["device"] = device
    record.update(features or {})
    return json.dumps(record)


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
        """A log with no feature keys writes the envelope and nothing else."""
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


class TestFeatureColumns:
    """The columns a UHD is actually fitted against.

    Without them the CSV is timings and identity, and a human has to join every row
    back to the problem and kernel it measured before anything can be trained.
    """

    def test_feature_keys_become_columns_carrying_their_values(self, tmp_path: Path):
        log = _write_log(
            tmp_path / "sweep.log",
            [
                _ok(
                    "g",
                    "k1",
                    0.02,
                    features={"q.seqlen": 512, "kernel.tile_m": 128,
                              "kernel.dtype": "fp16"},
                )
            ],
        )

        convert([log], tmp_path / "out.csv")

        with open(tmp_path / "out.csv", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        assert header[: len(ENVELOPE_COLUMNS)] == list(ENVELOPE_COLUMNS)
        assert header[len(ENVELOPE_COLUMNS) :] == [
            "kernel.dtype",
            "kernel.tile_m",
            "q.seqlen",
        ]

        row = _read_csv(tmp_path / "out.csv")[0]
        assert row["q.seqlen"] == "512"
        assert row["kernel.tile_m"] == "128"
        assert row["kernel.dtype"] == "fp16"

    def test_a_value_lands_in_the_row_that_measured_it(self, tmp_path: Path):
        """Two kernels of one sweep differ only in their features."""
        log = _write_log(
            tmp_path / "sweep.log",
            [
                _ok("g", "k1", 0.02, features={"kernel.tile_m": 64}),
                _ok("g", "k2", 0.03, features={"kernel.tile_m": 256}),
            ],
        )

        convert([log], tmp_path / "out.csv")

        rows = {row["kernel"]: row for row in _read_csv(tmp_path / "out.csv")}
        assert rows["k1"]["kernel.tile_m"] == "64"
        assert rows["k2"]["kernel.tile_m"] == "256"

    def test_records_with_different_feature_keys_still_line_up(self, tmp_path: Path):
        """A key one row has and another lacks must leave a hole, not a shift.

        Two engines, or two kernels of one engine, declare different KMD fields.
        Writing each row against its own keys would slide every cell after the gap
        one column left, which reads as valid data and trains on nonsense.
        """
        log = _write_log(
            tmp_path / "sweep.log",
            [
                _ok("g", "k1", 0.02, features={"q.seqlen": 512, "kernel.tile_m": 64}),
                _ok("g", "k2", 0.03, features={"q.seqlen": 512, "kernel.split_k": 4}),
            ],
        )

        convert([log], tmp_path / "out.csv")

        rows = {row["kernel"]: row for row in _read_csv(tmp_path / "out.csv")}
        assert rows["k1"]["kernel.tile_m"] == "64"
        assert rows["k1"]["kernel.split_k"] == ""
        assert rows["k2"]["kernel.split_k"] == "4"
        assert rows["k2"]["kernel.tile_m"] == ""
        assert rows["k1"]["q.seqlen"] == rows["k2"]["q.seqlen"] == "512"
        assert rows["k1"]["minTimeMs"] == "0.02"

    def test_a_failed_pair_keeps_its_features(self, tmp_path: Path):
        """It is the only evidence the pair was tried; a row with no features
        cannot be placed in feature space and can only be discarded, which biases
        the corpus towards kernels that happened to work."""
        log = _write_log(
            tmp_path / "sweep.log",
            [
                _failed(
                    "g",
                    "k2",
                    "launch-not-timed",
                    features={"q.seqlen": 512, "kernel.tile_m": 64},
                )
            ],
        )

        convert([log], tmp_path / "out.csv")

        row = _read_csv(tmp_path / "out.csv")[0]
        assert row["is_valid"] == "False"
        assert row["minTimeMs"] == ""
        assert row["q.seqlen"] == "512"
        assert row["kernel.tile_m"] == "64"

    def test_the_envelope_columns_are_never_mistaken_for_features(
        self, tmp_path: Path
    ):
        """`kernel` is identity; `kernel.dtype` is a feature. One dot apart.

        `device` is the same pairing: the dotless envelope column naming which GPU
        the row was measured on, sitting beside the `device.*` property columns
        §8.3 lists as features. The dot rule has to keep them apart in both
        directions -- an identity discovered as a feature would be fitted against,
        and a property mistaken for the envelope would be dropped from the header.
        """
        log = _write_log(
            tmp_path / "sweep.log",
            [
                _ok(
                    "g",
                    "k1",
                    0.02,
                    features={"kernel.dtype": "fp16", "device.cu_count": 304},
                    device="dev-a",
                )
            ],
        )

        stats = convert([log], tmp_path / "out.csv")

        assert stats.feature_keys == {"kernel.dtype", "device.cu_count"}
        row = _read_csv(tmp_path / "out.csv")[0]
        assert row["kernel"] == "k1"
        assert row["device"] == "dev-a"
        assert row["device.cu_count"] == "304"


class TestDeviceIdentity:
    """The device half of the problem identity (RFC 0019.13 §11.2).

    The runtime keys its winner cache on (graph, device) because the same graph on
    two GPUs is two problems with two different best kernels. A corpus merged from
    several machines that carries only the graph collapses those rows, so the
    per-problem oracle `v*(p)` becomes a minimum taken across devices and every
    regret figure computed from it is understated -- silently, and flatteringly.
    """

    def test_the_device_column_is_in_the_envelope_beside_the_benchmark(
        self, tmp_path: Path
    ):
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02, device="dev-a")])

        convert([log], tmp_path / "out.csv")

        with open(tmp_path / "out.csv", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(ENVELOPE_COLUMNS)
        assert "device" in ENVELOPE_COLUMNS
        # Dotless, so the feature rule cannot claim it however the envelope is
        # reordered later.
        assert "." not in "device"

    def test_the_device_reaches_the_row_that_was_measured_on_it(self, tmp_path: Path):
        log = _write_log(
            tmp_path / "sweep.log",
            [_ok("g", "k1", 0.02, device="dev-a"), _ok("g", "k1", 0.05, device="dev-b")],
        )

        convert([log], tmp_path / "out.csv")

        rows = _read_csv(tmp_path / "out.csv")
        assert [row["device"] for row in rows] == ["dev-a", "dev-b"]
        # Same graph, same kernel, different device: two problems, not one measured
        # twice. Grouping on `benchmark` alone would make 0.05 look like a slow
        # repeat of 0.02 and hand the oracle a minimum across devices.
        assert [row["benchmark"] for row in rows] == ["g", "g"]

    def test_a_failed_pair_still_says_which_device_it_failed_on(self, tmp_path: Path):
        log = _write_log(
            tmp_path / "sweep.log",
            [_failed("g", "k2", "launch-not-timed", device="dev-a")],
        )

        convert([log], tmp_path / "out.csv")

        row = _read_csv(tmp_path / "out.csv")[0]
        assert row["is_valid"] == "False"
        assert row["device"] == "dev-a"
        assert row["benchmark"] == "g"

    def test_one_graph_on_two_devices_counts_as_two_problems(self, tmp_path: Path):
        log = _write_log(
            tmp_path / "sweep.log",
            [_ok("g", "k1", 0.02, device="dev-a"), _ok("g", "k1", 0.05, device="dev-b")],
        )

        stats = convert([log], tmp_path / "out.csv")

        assert len(stats.problems) == 2

    def test_a_log_from_before_the_field_existed_leaves_the_column_empty(
        self, tmp_path: Path
    ):
        """The column is present and empty, never absent and never invented.

        Logs already collected have no `device` key. Dropping the column would
        break every reader that selects it by name; filling it with a placeholder
        would look like a real identity and let a consumer group on it without
        noticing. Empty is the one value that reads as "unidentified".
        """
        log = _write_log(tmp_path / "sweep.log", [_ok("g", "k1", 0.02)])

        convert([log], tmp_path / "out.csv")

        row = _read_csv(tmp_path / "out.csv")[0]
        assert "device" in row
        assert row["device"] == ""


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
