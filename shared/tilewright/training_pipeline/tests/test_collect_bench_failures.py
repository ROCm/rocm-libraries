# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv

import pytest

import collect_bench_failures as cbf
from bench_fakes import BANNER, Gemm, measured_lines, preamble, skip, winner

G_NT = Gemm(64, 32, 4096, 4, "N", "T")
G_TN = Gemm(256, 128, 512)


def _log_lines():
    lines = list(BANNER)
    lines += preamble(3) + measured_lines(0, G_NT, 20, 3.0)
    lines += [
        "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: queue 0x1 aborting with error",
        "Callback: Queue aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
        "",
        "*** KILLED [fatal]: unrecoverable GPU queue abort in the log ***",
        "*** SKIPPED CRASHED GEMM at block-offset 1 (yaml line 2): M=64 N=32 K=4096 "
        "B=4 transA=N transB=T (rc=-6) ***",
    ]
    lines += BANNER + preamble(2) + measured_lines(0, G_TN, 30, 5.0)
    lines += measured_lines(1, G_TN, 31, 4.0) + winner(1, G_TN, 31, 4.0)
    lines += [
        "*** SKIPPED CRASHED GEMM at block-offset 2 (yaml line 3): M=1 N=2 K=3 "
        "(rc=134) ***",
        "*** ABORT: 3 consecutive sub-runs finished no GEMM; 1 GEMMs left "
        "unattempted ***",
    ]
    lines += BANNER + preamble(3) + measured_lines(0, G_TN, 40, 5.0)
    lines += [skip(1, 41, "1.2e+02"), "Segmentation fault (core dumped)"]
    return lines


@pytest.fixture
def logs_dir(tmp_path):
    d = tmp_path / "stage03" / "logs"
    d.mkdir(parents=True)
    (d / "block_0003_lines7-9_gpu1.log").write_text("\n".join(_log_lines()) + "\n")
    (d / "warmup_0003_gpu1.log").write_text("Segmentation fault\n")
    return d


def test_events_carry_shape_kernel_and_transposes(logs_dir):
    events, n_complete, n_cut = cbf.parse_log(
        str(logs_dir / "block_0003_lines7-9_gpu1.log")
    )
    assert (n_complete, n_cut) == (1, 2)
    fatal, crash_new, crash_old, abort, segv = events
    assert (fatal["block"], fatal["gpu"]) == ("block_0003", "1")
    assert (fatal["transA"], fatal["transB"], fatal["m"], fatal["batch"]) == (
        "N",
        "T",
        "64",
        "4",
    )
    assert fatal["sol_idx"] == "20" and fatal["sol_name"].startswith("Cijk_")
    assert fatal["error"] == "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION"
    assert " | " in fatal["detail"]
    assert (crash_new["transA"], crash_new["transB"], crash_new["batch"]) == (
        "N",
        "T",
        "4",
    )
    assert crash_new["error"] == "crash_skipped(rc=-6)"
    assert (crash_old["transA"], crash_old["transB"], crash_old["m"]) == ("", "", "1")
    assert abort["error"] == "block_abandoned" and "unattempted" in abort["detail"]
    assert (segv["sol_idx"], segv["sol_name"], segv["m"]) == ("41", "(skipped)", "256")
    assert segv["error"] == "Segmentation fault"


def test_csv_goes_next_to_the_logs_not_into_cwd(logs_dir, tmp_path, monkeypatch):
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    assert cbf.main([str(logs_dir)]) == 0
    assert list(cwd.iterdir()) == []
    out = logs_dir.parent / "bench_failures.csv"
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 and list(rows[0]) == cbf.COLUMNS
    custom = tmp_path / "custom.csv"
    assert cbf.main([str(logs_dir), "--out", str(custom)]) == 0
    assert custom.read_text() == out.read_text()
