# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/memwatch.py: phase parsing, per-process accounting, exit status, signals."""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(scripts_dir))

import memwatch  # noqa: E402

MEMWATCH = scripts_dir / "memwatch.py"


# ------------------------------------------------------------------ phases
@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "Generating assembly kernels: Launching 96 threads for 202493 tasks...",
            "Generating assembly kernels",
        ),
        (
            "Generating assembly kernels: Done. (1192.5 secs elapsed)",
            "after Generating assembly kernels",
        ),
        (
            "Writing master solution libraries: Launching 96 threads for 603 tasks...",
            "Writing master solution libraries",
        ),
        # ParallelMap2 prints Done. when it hands back a generator, before the
        # results are consumed, so this one is not a completion marker.
        (
            "Loading Logics...: Launching 96 threads for 843 tasks...",
            "Loading Logics...",
        ),
        ("Loading Logics...: Done. (0.2 secs elapsed)", None),
        # "Time to" lines are printed after the phase finished.
        ("Time to load yaml files (s): 210.95", "after load yaml files"),
        ("Time to generate kernels (s): 994.66", "after generate kernels"),
        (
            "Time to pass kernel info to library (s): 53.85",
            "after pass kernel info to library",
        ),
        ("# TensileLite Library Writer DONE", "after library writer"),
        ("# Tensile Library Writer DONE", "after library writer"),
        ("Total time (s): 1506.16", "done"),
        ("Number of unique solutions: 356190", None),
        ("", None),
    ],
)
def test_phase_from_line(line, expected):
    assert memwatch.phase_from_line(line) == expected


# ------------------------------------------------------------- classifying
@pytest.mark.parametrize(
    "cmdline, cls",
    [
        (
            "/venv/bin/python3\x00-m\x00tensilelite.create_library\x00--architecture=gfx942",
            "python-main",
        ),
        (
            "/venv/bin/python3\x00-m\x00joblib.externals.loky.backend.popen_loky_posix",
            "loky-worker",
        ),
        ("/opt/rocm/lib/llvm/bin/clang\x00-x\x00assembler", "clang"),
        ("/opt/rocm/lib/llvm/bin/ld.lld\x00-o\x00x.co", "lld"),
        ("/opt/rocm/lib/llvm/bin/clang-offload-bundler\x00--compress", "bundler"),
        ("/bin/sh\x00-c\x00true", "sh"),
        ("/usr/bin/sleep\x0030", "other"),
        # A Python path among a compiler's arguments does not make it the driver.
        (
            "/opt/rocm/lib/llvm/bin/clang\x00-I/venv/lib/python3.13/include\x00-c\x00k.s",
            "clang",
        ),
        ("/usr/bin/python3.13\x00/tmp/x.py", "python-main"),
    ],
)
def test_classify(cmdline, cls):
    assert memwatch.classify(cmdline) == cls


def test_every_class_has_csv_columns():
    assert set(memwatch.CLASS_NAMES) == {name for name, _ in memwatch.CLASSIFIERS} | {
        "other"
    }
    assert "sh" in memwatch.CLASS_NAMES


# ---------------------------------------------------------------- sampling
def test_largest_rss_and_largest_pss_are_tracked_separately(monkeypatch):
    # The root pid is always the "python-main" class. Two children in "other":
    # pid 11 has the larger RSS but mostly shared pages (small PSS); pid 12 has
    # the smaller RSS but it is all private, so it has the larger PSS.
    smaps = {
        10: {"Rss": 8 << 20, "Pss": 8 << 20, "Uss": 8 << 20},
        11: {"Rss": 100 << 20, "Pss": 20 << 20, "Uss": 10 << 20},
        12: {"Rss": 50 << 20, "Pss": 40 << 20, "Uss": 40 << 20},
    }
    monkeypatch.setattr(
        memwatch,
        "proc_table",
        lambda: {1: (0, 1), 10: (1, 50), 11: (10, 60), 12: (10, 70)},
    )
    monkeypatch.setattr(memwatch, "smaps_rollup", lambda pid: smaps.get(pid))
    monkeypatch.setattr(memwatch, "read_file", lambda path: b"/bin/sleep\x0030")
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    ms = sampler.max_single["other"]
    assert (ms["rss"], ms["rss_pid"]) == (100 << 20, 11)
    assert (ms["pss"], ms["pss_pid"]) == (40 << 20, 12)
    row = sampler.rows[0]
    assert row["rss_sum"] == 158 << 20 and row["pss_sum"] == 68 << 20
    assert sum(row[f"{c}.rss"] for c in memwatch.CLASS_NAMES) == row["rss_sum"]
    summary = sampler.summary(1.0, 0, ["sleep"])
    assert summary["per_class_peak"]["other"]["largest_single_rss_MiB"] == 100.0
    assert summary["per_class_peak"]["other"]["largest_single_pss_MiB"] == 40.0


def test_a_recycled_pid_is_classified_afresh(monkeypatch):
    tables = iter(
        [
            {1: (0, 1), 10: (1, 50), 11: (10, 60)},  # pid 11 is an assembler
            {1: (0, 1), 10: (1, 50), 11: (10, 900)},  # pid 11 reused by a worker
        ]
    )
    cmdlines = [
        b"/opt/rocm/lib/llvm/bin/clang\x00-c\x00k.s",
        b"/venv/bin/python3\x00-m\x00joblib.externals.loky.backend.popen_loky_posix",
    ]
    monkeypatch.setattr(memwatch, "proc_table", lambda: next(tables))
    monkeypatch.setattr(
        memwatch,
        "smaps_rollup",
        lambda pid: {"Rss": 1 << 20, "Pss": 1 << 20, "Uss": 1 << 20},
    )
    monkeypatch.setattr(
        memwatch,
        "read_file",
        lambda path: (
            cmdlines[0]
            if path.endswith("/11/cmdline")
            else b"/venv/bin/python3\x00-m\x00tensilelite.create_library"
        ),
    )
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    assert sampler.rows[0]["clang.n"] == 1 and sampler.rows[0]["loky-worker.n"] == 0
    cmdlines.reverse()
    sampler.sample()
    assert sampler.rows[1]["clang.n"] == 0 and sampler.rows[1]["loky-worker.n"] == 1
    assert set(sampler.cmdline_cache) == {
        (10, 50),
        (11, 900),
    }  # the dead (11, 60) entry is gone


def test_a_root_that_exited_before_the_sample_is_skipped(monkeypatch):
    # /proc listing taken after the child exited: the root is not in the table.
    monkeypatch.setattr(memwatch, "proc_table", lambda: {1: (0, 1)})
    monkeypatch.setattr(
        memwatch, "smaps_rollup", lambda pid: {"Rss": 1, "Pss": 1, "Uss": 1}
    )
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    assert sampler.rows[0]["nproc"] == 0 and sampler.rows[0]["rss_sum"] == 0


def test_unreadable_smaps_is_reported_rather_than_silent_zero(monkeypatch):
    monkeypatch.setattr(memwatch, "proc_table", lambda: {1: (0, 1), 10: (1, 50)})
    monkeypatch.setattr(memwatch, "smaps_rollup", lambda pid: None)
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    summary = sampler.summary(1.0, 0, ["x"])
    assert summary["peak"]["pss_sum"]["GiB"] == 0.0
    assert summary["warnings"] and "smaps_rollup" in summary["warnings"][0]


# ------------------------------------------------------------- exit status
@pytest.mark.parametrize("rc, status", [(0, 0), (3, 3), (-15, 143), (-9, 137)])
def test_exit_status_maps_signal_deaths_like_a_shell(rc, status):
    assert memwatch.exit_status(rc) == status


def _run_memwatch(tmp_path, *cmd, timeout=60):
    out = tmp_path / "summary.json"
    csv_path = tmp_path / "timeline.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(MEMWATCH),
            "--interval",
            "0.05",
            "--out",
            str(out),
            "--csv",
            str(csv_path),
            "--quiet",
            "--",
            *cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc, out, csv_path


def test_child_killed_by_signal_gives_shell_exit_status(tmp_path):
    proc, out, _ = _run_memwatch(tmp_path, "sh", "-c", "kill -TERM $$")
    assert proc.returncode == 143, proc.stdout + proc.stderr
    assert json.loads(out.read_text())["returncode"] == -15


def test_summary_and_csv_account_for_a_shell_in_the_tree(tmp_path):
    # A Python root (the "python-main" class) runs a shell, which runs sleep,
    # so the tree holds the "sh" and "other" classes as well.
    script = (
        'echo "Phase A: Launching 1 threads for 1 tasks..."; sleep 0.4; '
        'echo "Time to phase A (s): 0.4"; sleep 0.4'
    )
    proc, out, csv_path = _run_memwatch(
        tmp_path,
        sys.executable,
        "-c",
        f"import subprocess; subprocess.run(['sh', '-c', {script!r}])",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(out.read_text())
    assert summary["returncode"] == 0 and summary["samples"] >= 2
    assert summary["warnings"] == []
    assert (
        "Phase A" in summary["phase_peak"] and "after phase A" in summary["phase_peak"]
    )
    assert {"python-main", "sh", "other"} <= set(summary["per_class_peak"])
    rows = list(csv.DictReader(csv_path.open()))
    assert any(int(row["sh.n"]) > 0 for row in rows)
    for row in rows:
        assert sum(int(row[f"{c}.rss"]) for c in memwatch.CLASS_NAMES) == int(
            row["rss_sum"]
        )


def test_sigterm_to_memwatch_reaches_the_whole_process_group(tmp_path):
    marker = tmp_path / "started"
    # The child spawns a grandchild that would hold stdout open for 30 s and
    # records the grandchild's pid.
    child = (
        "import subprocess, time, pathlib, sys\n"
        "p = subprocess.Popen(['sleep', '30'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        "time.sleep(30)\n"
    )
    out = tmp_path / "summary.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(MEMWATCH),
            "--interval",
            "0.05",
            "--out",
            str(out),
            "--quiet",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 20
    while not marker.exists() or not marker.read_text():
        assert time.monotonic() < deadline, "child never started"
        time.sleep(0.05)
    grandchild = int(marker.read_text())
    t0 = time.monotonic()
    os.kill(proc.pid, signal.SIGTERM)  # memwatch's PID only
    output, _ = proc.communicate(timeout=20)
    assert time.monotonic() - t0 < 10, "memwatch waited for the orphaned grandchild"
    assert proc.returncode == 143, output
    assert json.loads(out.read_text())["returncode"] == -15
    # The grandchild was in the signalled group, so it is gone (or a zombie
    # about to be reaped by init) rather than sleeping on.
    for _ in range(100):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        state = Path(f"/proc/{grandchild}/stat").read_text().split(")")[-1].split()[0]
        if state == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {grandchild} still running")
