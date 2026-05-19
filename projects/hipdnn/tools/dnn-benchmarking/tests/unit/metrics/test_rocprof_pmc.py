# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for rocprofv3 PMC counter collection.

Avoids requiring a real rocprofv3 binary or rocpd db: the subprocess
is mocked, and the rocpd schema is reproduced just well enough that
the parser exercises its real SQL path against an in-test sqlite db.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.metrics import rocprof_pmc
from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once


@pytest.fixture(autouse=True)
def _reset():
    reset_warn_once()


class TestResolveCounterList:
    def test_known_arch_known_set(self):
        counters = rocprof_pmc._resolve_counter_list("gfx942", "basic")
        assert "GRBM_GUI_ACTIVE" in counters
        assert "SQ_WAVES" in counters

    def test_unknown_arch_falls_back(self):
        # Unknown arches resolve via the fallback table; the fallback
        # only defines 'basic', so other sets return [].
        counters = rocprof_pmc._resolve_counter_list("gfx-mystery", "basic")
        assert counters == ["GRBM_GUI_ACTIVE", "SQ_WAVES"]
        assert rocprof_pmc._resolve_counter_list("gfx-mystery", "memory") == []

    def test_all_unions_and_dedups(self):
        counters = rocprof_pmc._resolve_counter_list("gfx942", "all")
        # Union of basic+memory+flops, dedup-preserving order
        assert "GRBM_GUI_ACTIVE" in counters
        assert "TCC_HIT_sum" in counters
        assert "SQ_INSTS_VALU_MFMA_F16" in counters
        assert len(counters) == len(set(counters))


class TestArgvBuild:
    def test_passes_pmc_set_and_inner_argv(self, tmp_path):
        argv = rocprof_pmc._build_argv(
            counters=["GRBM_GUI_ACTIVE", "SQ_WAVES"],
            out_dir=tmp_path,
            inner_argv=["python", "-m", "dnn_benchmarking", "--internal-profiling-run"],
            rocprofv3_binary="/opt/rocm/bin/rocprofv3",
        )
        # The caller-supplied absolute binary is preserved verbatim — the
        # orchestrator must not silently rewrite it to the bare command name,
        # otherwise PATH-resolution in the spawned process picks up the
        # venv shim from torch's rocm_sdk_core wheel (broken on torch
        # workloads).
        assert argv[0] == "/opt/rocm/bin/rocprofv3"
        assert "--pmc" in argv
        # `-o results` strips the `<pid>_` prefix from rocprofv3's
        # default `<hostname>/<pid>_results.<ext>` filename so the
        # artifact path is stable and copy-pasteable.
        assert "-o" in argv
        assert argv[argv.index("-o") + 1] == "results"
        # Counter names follow --pmc and precede '--' separator
        sep = argv.index("--")
        assert "GRBM_GUI_ACTIVE" in argv[:sep]
        assert "python" in argv[sep + 1 :]


class TestRunHappyPath:
    def _make_synthetic_rocpd_db(self, db_path: Path) -> None:
        """Mirror the rocpd schema closely enough to exercise the parser.

        Uses uuid-suffixed table names so the parser's sqlite_master walk
        runs against shapes that match production output. Schema matches
        rocprofv3 1.2.2:
          * ``kernel_dispatch`` has ``dispatch_id`` + ``kernel_id`` but
            no ``kernel_name`` column.
          * Kernel names live in ``info_kernel_symbol.kernel_name`` and
            are joined via ``kernel_dispatch.kernel_id = info_kernel_symbol.id``.
          * ``pmc_event.event_id`` references ``kernel_dispatch.dispatch_id``.
        """
        suffix = "_abc123"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                f"""
                CREATE TABLE rocpd_pmc_event{suffix} (
                    event_id INTEGER, pmc_id INTEGER, value REAL
                );
                CREATE TABLE rocpd_kernel_dispatch{suffix} (
                    id INTEGER PRIMARY KEY, kernel_id INTEGER, dispatch_id INTEGER
                );
                CREATE TABLE rocpd_info_kernel_symbol{suffix} (
                    id INTEGER PRIMARY KEY, kernel_name TEXT
                );
                CREATE TABLE rocpd_info_pmc{suffix} (
                    id INTEGER PRIMARY KEY, name TEXT
                );
                INSERT INTO rocpd_info_pmc{suffix} VALUES (1, 'GRBM_GUI_ACTIVE');
                INSERT INTO rocpd_info_pmc{suffix} VALUES (2, 'SQ_WAVES');
                INSERT INTO rocpd_info_kernel_symbol{suffix} VALUES (100, 'conv2d_kernel');
                INSERT INTO rocpd_info_kernel_symbol{suffix} VALUES (200, 'gemm_kernel');
                -- (kd.id, kd.kernel_id, kd.dispatch_id)
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (1, 100, 10);
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (2, 200, 11);
                -- (pmc.event_id matches kd.dispatch_id, pmc.pmc_id, pmc.value)
                INSERT INTO rocpd_pmc_event{suffix} VALUES (10, 1, 1000);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (10, 1, 2000);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (10, 2, 50);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (11, 1, 500);
                INSERT INTO rocpd_pmc_event{suffix} VALUES (11, 2, 25);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_full_pipeline_with_synthetic_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(
            rocprof_pmc, "resolve_rocm_tool", lambda name: "/opt/rocm/bin/rocprofv3"
        )
        out_dir = tmp_path / "pmc_out"
        out_dir.mkdir()

        def fake_run(argv, **kwargs):
            # Drop the synthetic db where _find_rocpd_db will pick it up.
            host_dir = Path(argv[argv.index("-d") + 1])
            host_dir.mkdir(parents=True, exist_ok=True)
            db = host_dir / "results.db"
            self._make_synthetic_rocpd_db(db)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rocprof_pmc.subprocess, "run", side_effect=fake_run):
            extra = rocprof_pmc.run(
                inner_argv=["python", "-m", "dnn_benchmarking"],
                out_dir=out_dir,
                pmc_set="basic",
            )
        pmc = extra["pmc"]
        assert pmc["arch"] == "gfx942"
        assert pmc["set"] == "basic"
        assert "db_path" in pmc
        assert "counters" in pmc
        assert pmc["counters"]["GRBM_GUI_ACTIVE"]["sum"] == 3500
        assert "conv2d_kernel" in pmc["per_kernel"]
        assert pmc["per_kernel"]["conv2d_kernel"]["GRBM_GUI_ACTIVE"] == 1500.0
        assert pmc["per_kernel"]["gemm_kernel"]["SQ_WAVES"] == 25.0

    def test_missing_info_kernel_symbol_returns_warning(self, tmp_path, monkeypatch):
        """If the rocpd db omits info_kernel_symbol, the parser must not
        fall back to a broken SQL path — it should report the missing
        table and skip aggregation cleanly."""
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(
            rocprof_pmc, "resolve_rocm_tool", lambda name: "/opt/rocm/bin/rocprofv3"
        )
        out_dir = tmp_path / "pmc_out"
        out_dir.mkdir()

        def fake_run(argv, **kwargs):
            host_dir = Path(argv[argv.index("-d") + 1])
            host_dir.mkdir(parents=True, exist_ok=True)
            db = host_dir / "results.db"
            conn = sqlite3.connect(db)
            try:
                conn.executescript(
                    """
                    CREATE TABLE rocpd_pmc_event_x (event_id INTEGER, pmc_id INTEGER, value REAL);
                    CREATE TABLE rocpd_kernel_dispatch_x (id INTEGER, kernel_id INTEGER, dispatch_id INTEGER);
                    """
                )
                conn.commit()
            finally:
                conn.close()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(rocprof_pmc.subprocess, "run", side_effect=fake_run):
            extra = rocprof_pmc.run(
                inner_argv=["python"], out_dir=out_dir, pmc_set="basic"
            )
        assert "info_kernel_symbol" in extra["pmc"]["warnings"][0]


class TestRunFailureModes:
    def test_rocprofv3_nonzero_exit_records_error_tail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(
            rocprof_pmc, "resolve_rocm_tool", lambda name: "/opt/rocm/bin/rocprofv3"
        )
        proc = MagicMock(
            returncode=1,
            stdout="",
            stderr="rocprofv3: counter 'BOGUS' unsupported on this device\n",
        )
        with patch.object(rocprof_pmc.subprocess, "run", return_value=proc):
            extra = rocprof_pmc.run(
                inner_argv=["python"],
                out_dir=tmp_path,
                pmc_set="basic",
            )
        pmc = extra["pmc"]
        assert pmc["returncode"] == 1
        assert "BOGUS" in pmc["error_tail"]
        # Failure path must not raise — caller can still proceed.

    def test_invocation_raises_oserror_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(
            rocprof_pmc, "resolve_rocm_tool", lambda name: "/opt/rocm/bin/rocprofv3"
        )
        with patch.object(rocprof_pmc.subprocess, "run", side_effect=OSError("boom")):
            extra = rocprof_pmc.run(
                inner_argv=["python"],
                out_dir=tmp_path,
                pmc_set="basic",
            )
        assert "skipped" in extra["pmc"]

    def test_no_counters_for_arch_returns_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx-mystery")
        extra = rocprof_pmc.run(
            inner_argv=["python"],
            out_dir=tmp_path,
            pmc_set="memory",  # fallback table only defines 'basic'
        )
        assert extra["pmc"]["skipped"] == "no counters defined"

    def test_rocprofv3_binary_missing_returns_skipped(self, tmp_path, monkeypatch):
        """If neither /opt/rocm/bin/rocprofv3 nor a PATH-resolved one exists,
        the PMC pass skips cleanly instead of crashing or invoking
        whatever bare 'rocprofv3' shim happens to be on PATH."""
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
        monkeypatch.setattr(rocprof_pmc, "resolve_rocm_tool", lambda name: None)
        extra = rocprof_pmc.run(
            inner_argv=["python"], out_dir=tmp_path, pmc_set="basic"
        )
        assert extra["pmc"]["skipped"] == "rocprofv3 binary not found"
