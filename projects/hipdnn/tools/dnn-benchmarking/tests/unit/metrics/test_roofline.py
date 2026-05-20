# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the rocprof-compute roofline wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import roofline as roofline_mod
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


class TestArgvBuild:
    def test_no_data_type_flag_in_profile_argv(self, tmp_path):
        """rocprof-compute's ``profile`` subcommand does not accept
        ``--roofline-data-type`` — that flag lives only under
        ``analyze``. Passing it to ``profile`` errors. Regression
        guard: ensure we never re-introduce it on the profile side."""
        argv = roofline_mod._build_argv(
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
        assert "--roofline-data-type" not in argv
        # Inner argv follows '--'
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["python", "-m", "dnn_benchmarking"]


class TestRunHappyPath:
    def test_records_csv_and_workload_paths(self, tmp_path, monkeypatch):
        """``profile --roof-only`` emits CSVs (roofline.csv +
        sysinfo.csv + per-IP results CSVs), no PDF and no SQLite. We
        record roofline.csv as the primary artifact and the workload
        directory so the user can point ``rocprof-compute analyze
        --path <dir>`` at it to render PDFs."""
        monkeypatch.setattr(
            roofline_mod,
            "resolve_rocm_tool",
            lambda name: "/opt/rocm/bin/rocprof-compute",
        )

        def fake_run(argv, **kwargs):
            # rocprof-compute nests its output one level deeper than
            # workload_dir (under <wl_dir>/<gpu>/). The _find_named
            # walker is recursive so the test mirrors that depth.
            inner = tmp_path / "workload" / "gfx90a"
            inner.mkdir(parents=True, exist_ok=True)
            (inner / "roofline.csv").write_text("Empirical_HBM,123\n")
            (inner / "sysinfo.csv").write_text("gpu,gfx90a\n")
            (inner / "results_pmc_perf_0.csv").write_text("k,v\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(roofline_mod.subprocess, "run", side_effect=fake_run):
            extra = roofline_mod.run(inner_argv=["python"], out_dir=tmp_path)
        rl = extra["roofline"]
        assert rl["roofline_csv"].endswith("roofline.csv")
        assert rl["sysinfo_csv"].endswith("sysinfo.csv")
        # workload_path is the parent dir of roofline.csv — what
        # `rocprof-compute analyze --path` expects.
        assert rl["workload_path"].endswith("workload/gfx90a")
        # No data_type, no pdf_path, no db_path under the new contract.
        assert "data_type" not in rl
        assert "pdf_path" not in rl
        assert "db_path" not in rl


class TestFailureModes:
    def test_missing_binary_returns_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(roofline_mod, "resolve_rocm_tool", lambda name: None)
        extra = roofline_mod.run(inner_argv=["python"], out_dir=tmp_path)
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
            extra = roofline_mod.run(inner_argv=["python"], out_dir=tmp_path)
        assert extra["roofline"]["returncode"] == 1
        assert "failed" in extra["roofline"]["error_tail"]

    def test_success_with_missing_artifacts_records_warning(
        self, monkeypatch, tmp_path
    ):
        """rocprof-compute exited 0, created a workload directory, but
        produced neither the ceilings CSV nor sysinfo. Distinct from the
        broken-tool case (no workload directory at all): here the tool
        ran far enough to set up its output tree but emitted no
        artefacts. We must record the gap rather than silently emit an
        empty `roofline` slice."""
        monkeypatch.setattr(
            roofline_mod,
            "resolve_rocm_tool",
            lambda name: "/opt/rocm/bin/rocprof-compute",
        )

        def fake_run(argv, **kwargs):
            # Mimic rocprof-compute creating the workload tree but no
            # CSVs (which is the post-tool-init / pre-PMC-write window).
            workload_dir = tmp_path / "workload"
            workload_dir.mkdir(parents=True, exist_ok=True)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(roofline_mod.subprocess, "run", side_effect=fake_run):
            extra = roofline_mod.run(inner_argv=["python"], out_dir=tmp_path)
        rl = extra["roofline"]
        assert rl["warnings"] == ["no roofline.csv or sysinfo.csv produced"]
        assert "roofline_csv" not in rl
        assert "sysinfo_csv" not in rl
        assert "workload_path" not in rl

    def test_success_with_no_workload_dir_records_specific_warning(
        self, monkeypatch, tmp_path
    ):
        """rocprof-compute exited 0 and wrote nothing — not even the
        workload directory. Almost always a tool/version mismatch
        (`profile --roof-only` is silently a no-op on some builds).
        The diagnostic must point at the tool rather than the missing
        CSVs, otherwise users chase a phantom output-format bug."""
        monkeypatch.setattr(
            roofline_mod,
            "resolve_rocm_tool",
            lambda name: "/opt/rocm/bin/rocprof-compute",
        )
        # Critically: no fake_run side effect that creates workload_dir.
        proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch.object(roofline_mod.subprocess, "run", return_value=proc):
            extra = roofline_mod.run(inner_argv=["python"], out_dir=tmp_path)
        rl = extra["roofline"]
        assert len(rl["warnings"]) == 1
        assert "no workload directory" in rl["warnings"][0]
