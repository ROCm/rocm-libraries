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
        )
        assert argv[0] == "rocprofv3"
        assert "--pmc" in argv
        # Counter names follow --pmc and precede '--' separator
        sep = argv.index("--")
        assert "GRBM_GUI_ACTIVE" in argv[:sep]
        assert "python" in argv[sep + 1 :]


class TestRunHappyPath:
    def _make_synthetic_rocpd_db(self, db_path: Path) -> None:
        """Mirror the rocpd schema closely enough to exercise the parser.

        Uses uuid-suffixed table names so the parser's sqlite_master walk
        runs against shapes that match production output.
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
                    id INTEGER PRIMARY KEY, kernel_name TEXT
                );
                CREATE TABLE rocpd_info_pmc{suffix} (
                    id INTEGER PRIMARY KEY, name TEXT
                );
                INSERT INTO rocpd_info_pmc{suffix} VALUES (1, 'GRBM_GUI_ACTIVE');
                INSERT INTO rocpd_info_pmc{suffix} VALUES (2, 'SQ_WAVES');
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (10, 'conv2d_kernel');
                INSERT INTO rocpd_kernel_dispatch{suffix} VALUES (11, 'gemm_kernel');
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


class TestRunFailureModes:
    def test_rocprofv3_nonzero_exit_records_error_tail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rocprof_pmc, "detect_arch", lambda: "gfx942")
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
