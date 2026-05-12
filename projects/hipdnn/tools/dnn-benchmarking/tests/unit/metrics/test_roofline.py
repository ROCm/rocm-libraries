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
        )
        assert argv[0] == "rocprof-compute"
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
            roofline_mod.shutil, "which", lambda _: "/opt/rocm/bin/rocprof-compute"
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
        monkeypatch.setattr(roofline_mod.shutil, "which", lambda _: None)
        extra = roofline_mod.run(
            inner_argv=["python"], out_dir=tmp_path, data_type="FP32"
        )
        assert "skipped" in extra["roofline"]

    def test_nonzero_exit_records_error_tail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            roofline_mod.shutil, "which", lambda _: "/opt/rocm/bin/rocprof-compute"
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
