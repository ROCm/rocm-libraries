# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the perf stat wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import perf as perf_mod
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once

# A minimal seven-column perf-stat -x, sample.
SAMPLE_CSV = """\
# started on Mon Jan  1 00:00:00 2026
1234567890,,cycles:u,123456789,100.00,,
987654321,,instructions:u,123456789,100.00,0.80,insn per cycle
123.45,msec,task-clock,123456789,100.00,,
12,,context-switches,123456789,100.00,,
3,,page-faults,123456789,100.00,,
"""


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


class TestParseCsv:
    def test_parses_user_events(self, tmp_path):
        csv = tmp_path / "perf.csv"
        csv.write_text(SAMPLE_CSV)
        parsed = perf_mod._parse_perf_csv(csv)
        assert parsed["cycles:u"] == 1234567890
        assert parsed["instructions:u"] == 987654321
        assert parsed["task-clock"] == 123.45
        assert parsed["context-switches"] == 12
        assert parsed["page-faults"] == 3

    def test_handles_not_counted_marker(self, tmp_path):
        csv = tmp_path / "perf.csv"
        csv.write_text("<not counted>,,cycles:u,0,0.00,,\n")
        parsed = perf_mod._parse_perf_csv(csv)
        assert parsed["cycles:u"] is None


class TestKernelEventGate:
    def test_paranoid_high_drops_kernel_events(self, monkeypatch, tmp_path):
        monkeypatch.setattr(perf_mod, "_read_perf_paranoid", lambda: 4)
        monkeypatch.setattr(perf_mod.shutil, "which", lambda _: "/usr/bin/perf")

        captured = {"argv": None}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            host_dir = Path(argv[argv.index("-o") + 1]).parent
            host_dir.mkdir(parents=True, exist_ok=True)
            (host_dir / "perf.csv").write_text(SAMPLE_CSV)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(perf_mod.subprocess, "run", side_effect=fake_run):
            extra = perf_mod.run(inner_argv=["python"], out_dir=tmp_path)

        # Argv must omit cycles:k / instructions:k.
        assert "cycles:k" not in ",".join(captured["argv"])
        assert extra["perf"]["kernel_perf_paranoid"] == 4
        assert extra["perf"]["kernel_events_skipped_reason"]
        assert extra["perf"]["cycles_kernel"] is None

    def test_paranoid_low_includes_kernel_events(self, monkeypatch, tmp_path):
        monkeypatch.setattr(perf_mod, "_read_perf_paranoid", lambda: 1)
        monkeypatch.setattr(perf_mod.shutil, "which", lambda _: "/usr/bin/perf")

        captured = {"argv": None}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            host_dir = Path(argv[argv.index("-o") + 1]).parent
            host_dir.mkdir(parents=True, exist_ok=True)
            (host_dir / "perf.csv").write_text(SAMPLE_CSV)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(perf_mod.subprocess, "run", side_effect=fake_run):
            perf_mod.run(inner_argv=["python"], out_dir=tmp_path)

        events_arg = captured["argv"][captured["argv"].index("-e") + 1]
        assert "cycles:k" in events_arg
        assert "instructions:k" in events_arg


class TestMissingBinary:
    def test_missing_perf_returns_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(perf_mod.shutil, "which", lambda _: None)
        extra = perf_mod.run(inner_argv=["python"], out_dir=tmp_path)
        assert extra["perf"]["skipped"].startswith("perf binary not found")


class TestIpcDerivation:
    def test_ipc_user_computed_clientside(self, monkeypatch, tmp_path):
        monkeypatch.setattr(perf_mod, "_read_perf_paranoid", lambda: 1)
        monkeypatch.setattr(perf_mod.shutil, "which", lambda _: "/usr/bin/perf")

        def fake_run(argv, **kwargs):
            host_dir = Path(argv[argv.index("-o") + 1]).parent
            host_dir.mkdir(parents=True, exist_ok=True)
            (host_dir / "perf.csv").write_text(SAMPLE_CSV)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(perf_mod.subprocess, "run", side_effect=fake_run):
            extra = perf_mod.run(inner_argv=["python"], out_dir=tmp_path)
        ipc = extra["perf"]["ipc_user"]
        assert ipc is not None
        assert abs(ipc - (987654321 / 1234567890)) < 1e-6
