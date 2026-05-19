# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the rocprof-compute roofline wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import roofline as roofline_mod
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


class TestArgvBuild:
    def test_passes_data_type_and_inner_argv(self, tmp_path):
        argv = roofline_mod._build_argv(
            data_type="FP32",
            workload_dir=tmp_path / "workload",
            inner_argv=["python", "-m", "dnn_benchmarking"],
            rocprof_compute_binary="/opt/rocm/bin/rocprof-compute",
        )
        # Absolute binary path preserved verbatim (see rocprof_pmc tests
        # for the rationale: PATH-based resolution from the activated
        # venv picks up a broken shim).
        assert argv[0] == "/opt/rocm/bin/rocprof-compute"
        assert "profile" in argv
        assert "--roof-only" in argv
        # Data type follows --roofline-data-type
        assert argv[argv.index("--roofline-data-type") + 1] == "FP32"
        # Inner argv follows '--'
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["python", "-m", "dnn_benchmarking"]


class TestRunHappyPath:
    def test_records_pdf_and_db_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            roofline_mod,
            "resolve_rocm_tool",
            lambda name: "/opt/rocm/bin/rocprof-compute",
        )

        def fake_run(argv, **kwargs):
            wl_dir = tmp_path / "workload"
            wl_dir.mkdir(parents=True, exist_ok=True)
            (wl_dir / "roofline.pdf").write_bytes(b"%PDF-1.4")
            (wl_dir / "workload.db").write_bytes(b"sqlite-bytes")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(roofline_mod.subprocess, "run", side_effect=fake_run):
            extra = roofline_mod.run(
                inner_argv=["python"], out_dir=tmp_path, data_type="FP32"
            )
        rl = extra["roofline"]
        assert rl["data_type"] == "FP32"
        assert rl["pdf_path"].endswith("roofline.pdf")
        assert rl["db_path"].endswith("workload.db")


class TestFailureModes:
    def test_missing_binary_returns_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(roofline_mod, "resolve_rocm_tool", lambda name: None)
        extra = roofline_mod.run(
            inner_argv=["python"], out_dir=tmp_path, data_type="FP32"
        )
        assert extra["roofline"]["skipped"] == "rocprof-compute binary not found"

    def test_nonzero_exit_records_error_tail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            roofline_mod,
            "resolve_rocm_tool",
            lambda name: "/opt/rocm/bin/rocprof-compute",
        )
        proc = MagicMock(
            returncode=1, stdout="", stderr="rocprof-compute: workload failed\n"
        )
        with patch.object(roofline_mod.subprocess, "run", return_value=proc):
            extra = roofline_mod.run(
                inner_argv=["python"], out_dir=tmp_path, data_type="FP32"
            )
        assert extra["roofline"]["returncode"] == 1
        assert "failed" in extra["roofline"]["error_tail"]

    def test_success_with_missing_artifacts_records_warning(
        self, monkeypatch, tmp_path
    ):
        """rocprof-compute exited 0 but produced no PDF and no DB — the
        run is technically successful but there's nothing to point the
        user at. We must record the gap rather than silently emit an
        empty `roofline` slice."""
        monkeypatch.setattr(
            roofline_mod,
            "resolve_rocm_tool",
            lambda name: "/opt/rocm/bin/rocprof-compute",
        )
        proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(roofline_mod.subprocess, "run", return_value=proc):
            extra = roofline_mod.run(
                inner_argv=["python"], out_dir=tmp_path, data_type="FP32"
            )
        rl = extra["roofline"]
        assert rl["data_type"] == "FP32"
        assert rl["warnings"] == ["no PDF or db produced"]
        assert "pdf_path" not in rl
        assert "db_path" not in rl
