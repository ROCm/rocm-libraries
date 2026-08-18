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

scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
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
        lambda: {1: (0, 1, 1), 10: (1, 50, 10), 11: (10, 60, 10), 12: (10, 70, 10)},
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


def _fixed_smaps(pid):
    return {"Rss": 1 << 20, "Pss": 1 << 20, "Uss": 1 << 20}


def _cmdline_reader(by_pid):
    def read_file(path):
        for pid, cmdline in by_pid.items():
            if path.endswith(f"/{pid}/cmdline"):
                return cmdline
        return b"/venv/bin/python3\x00-m\x00tensilelite.create_library"

    return read_file


CLANG = b"/opt/rocm/lib/llvm/bin/clang\x00-c\x00k.s"
WORKER = b"/venv/bin/python3\x00-m\x00joblib.externals.loky.backend.popen_loky_posix"


def test_a_recycled_pid_is_classified_afresh(monkeypatch):
    tables = iter(
        [
            {1: (0, 1, 1), 10: (1, 50, 10), 11: (10, 60, 10)},  # pid 11 is an assembler
            {
                1: (0, 1, 1),
                10: (1, 50, 10),
                11: (10, 900, 10),
            },  # pid 11 reused by a worker
        ]
    )
    cmdlines = {11: CLANG}
    monkeypatch.setattr(memwatch, "proc_table", lambda: next(tables))
    monkeypatch.setattr(memwatch, "smaps_rollup", _fixed_smaps)
    monkeypatch.setattr(memwatch, "read_file", _cmdline_reader(cmdlines))
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    assert sampler.rows[0]["clang.n"] == 1 and sampler.rows[0]["loky-worker.n"] == 0
    cmdlines[11] = WORKER
    sampler.sample()
    assert sampler.rows[1]["clang.n"] == 0 and sampler.rows[1]["loky-worker.n"] == 1


def test_a_process_that_execs_is_reclassified(monkeypatch):
    # Same pid and start time; only the command line changes, as after exec.
    monkeypatch.setattr(
        memwatch,
        "proc_table",
        lambda: {1: (0, 1, 1), 10: (1, 50, 10), 11: (10, 60, 10)},
    )
    monkeypatch.setattr(memwatch, "smaps_rollup", _fixed_smaps)
    cmdlines = {11: b"/bin/sh\x00-c\x00exec clang -c k.s"}
    monkeypatch.setattr(memwatch, "read_file", _cmdline_reader(cmdlines))
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    assert sampler.rows[0]["sh.n"] == 1 and sampler.rows[0]["clang.n"] == 0
    cmdlines[11] = CLANG
    sampler.sample()
    assert sampler.rows[1]["sh.n"] == 0 and sampler.rows[1]["clang.n"] == 1


def test_a_worker_adopted_by_init_stays_in_the_tree():
    # Root 10 died; its worker 11 now has ppid 1 but still carries session 10.
    table = {1: (0, 1, 1), 11: (1, 60, 10), 20: (1, 5, 20)}
    assert memwatch.tree_members(10, table) == [10, 11]


def test_a_root_that_exited_before_the_sample_is_skipped(monkeypatch):
    # /proc listing taken after the child exited: the root is not in the table.
    monkeypatch.setattr(memwatch, "proc_table", lambda: {1: (0, 1, 1)})
    monkeypatch.setattr(
        memwatch, "smaps_rollup", lambda pid: {"Rss": 1, "Pss": 1, "Uss": 1}
    )
    monkeypatch.setattr(memwatch, "mem_available", lambda: None)
    sampler = memwatch.Sampler(root_pid=10, interval=1.0, csv_path=None)
    sampler.sample()
    assert sampler.rows[0]["nproc"] == 0 and sampler.rows[0]["rss_sum"] == 0


def test_unreadable_smaps_is_reported_rather_than_silent_zero(monkeypatch):
    monkeypatch.setattr(memwatch, "proc_table", lambda: {1: (0, 1, 1), 10: (1, 50, 10)})
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


def _is_gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def test_an_output_failure_terminates_the_command(tmp_path):
    marker = tmp_path / "pid"
    # A 70 KiB line overflows the log's buffer so the write to /dev/full fails
    # at once; the child would then sleep for 30 s.
    child = (
        "import os, time, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "print('x' * 70000, flush=True)\n"
        "time.sleep(30)\n"
    )
    out = tmp_path / "summary.json"
    t0 = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            str(MEMWATCH),
            "--interval",
            "0.05",
            "--out",
            str(out),
            "--log",
            "/dev/full",
            "--quiet",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        time.monotonic() - t0 < 20
    ), "memwatch kept running after the log write failed"
    assert proc.returncode != 0
    assert "could not handle the command's output" in proc.stderr, proc.stderr
    assert _is_gone(int(marker.read_text())), "the command survived the output failure"


def test_root_death_with_a_surviving_worker_ends_the_run(tmp_path):
    marker = tmp_path / "pid"
    # The root spawns a worker that keeps stdout open, then kills itself with
    # SIGKILL. The worker is reparented to init but stays in the session.
    child = (
        "import os, signal, subprocess, pathlib\n"
        "p = subprocess.Popen(['sleep', '30'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    out = tmp_path / "summary.json"
    t0 = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            str(MEMWATCH),
            "--interval",
            "0.05",
            "--grace",
            "1",
            "--out",
            str(out),
            "--quiet",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert time.monotonic() - t0 < 20, "memwatch waited for the worker's pipe"
    assert proc.returncode == 137, proc.stdout + proc.stderr
    assert "still running" in proc.stderr
    assert json.loads(out.read_text())["returncode"] == -9
    assert _is_gone(int(marker.read_text())), "the surviving worker was not terminated"


def _memwatch(tmp_path, *extra):
    out = tmp_path / "summary.json"
    return [
        sys.executable,
        str(MEMWATCH),
        "--interval",
        "0.05",
        "--out",
        str(out),
        "--quiet",
        *extra,
        "--",
    ], out


def test_a_term_resistant_child_is_killed_when_output_fails(tmp_path):
    marker = tmp_path / "pid"
    child = (
        "import os, signal, time, pathlib\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "print('x' * 70000, flush=True)\n"
        "time.sleep(60)\n"
    )
    argv, out = _memwatch(tmp_path, "--log", "/dev/full")
    t0 = time.monotonic()
    proc = subprocess.run(
        [*argv, sys.executable, "-c", child], capture_output=True, text=True, timeout=90
    )
    assert time.monotonic() - t0 < 30, "no bounded escalation to SIGKILL"
    assert (
        proc.returncode != 0 and "could not handle the command's output" in proc.stderr
    ), proc.stderr
    assert (
        json.loads(out.read_text())["returncode"] == -9
    )  # SIGTERM was ignored; SIGKILL ended it
    assert _is_gone(int(marker.read_text()))


def test_a_survivor_that_closed_its_output_is_still_terminated(tmp_path):
    marker = tmp_path / "pid"
    child = (
        "import subprocess, pathlib\n"
        "p = subprocess.Popen(['sleep', '30'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
    )
    argv, out = _memwatch(tmp_path, "--grace", "1")
    t0 = time.monotonic()
    proc = subprocess.run(
        [*argv, sys.executable, "-c", child], capture_output=True, text=True, timeout=60
    )
    assert time.monotonic() - t0 < 20
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "still running" in proc.stderr
    assert _is_gone(
        int(marker.read_text())
    ), "a worker without the pipe survived the grace period"


def test_an_unusable_log_path_starts_nothing(tmp_path):
    marker = tmp_path / "pid"
    child = f"import os, time, pathlib; pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    argv, out = _memwatch(
        tmp_path, "--log", str(tmp_path / "no" / "such" / "dir" / "x.log")
    )
    proc = subprocess.run(
        [*argv, sys.executable, "-c", child], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 1
    assert "memwatch: error:" in proc.stderr and "x.log" in proc.stderr, proc.stderr
    time.sleep(0.5)
    assert (
        not marker.exists()
    ), "the command was launched although the log could not be opened"


def _hup_default():
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


def _hup_ignored():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def _start_memwatch_with_sleeping_grandchild(tmp_path, preexec):
    marker = tmp_path / "started"
    child = (
        "import subprocess, time, pathlib\n"
        "p = subprocess.Popen(['sleep', '30'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        "time.sleep(30)\n"
    )
    argv, out = _memwatch(tmp_path)
    proc = subprocess.Popen(
        [*argv, sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=preexec,  # the runner's own SIGHUP disposition must not leak in
    )
    deadline = time.monotonic() + 20
    while not marker.exists() or not marker.read_text():
        assert time.monotonic() < deadline, "child never started"
        time.sleep(0.05)
    return proc, out, int(marker.read_text())


def test_sighup_to_memwatch_reaches_the_command(tmp_path):
    proc, out, grandchild = _start_memwatch_with_sleeping_grandchild(
        tmp_path, _hup_default
    )
    os.kill(proc.pid, signal.SIGHUP)
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 129, output  # the command died of the forwarded SIGHUP
    assert json.loads(out.read_text())["returncode"] == -1
    assert _is_gone(grandchild)


def test_an_inherited_sighup_ignore_is_respected(tmp_path):
    # Under nohup or a CI agent SIGHUP arrives ignored; memwatch and the
    # command both keep ignoring it, and SIGTERM still cleans up.
    proc, out, grandchild = _start_memwatch_with_sleeping_grandchild(
        tmp_path, _hup_ignored
    )
    os.kill(proc.pid, signal.SIGHUP)
    time.sleep(1.0)
    assert proc.poll() is None, "SIGHUP was not ignored"
    os.kill(proc.pid, signal.SIGTERM)
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143, output
    assert _is_gone(grandchild)


def test_a_log_that_fails_on_a_short_line_is_an_error(tmp_path):
    argv, out = _memwatch(tmp_path, "--log", "/dev/full")
    proc = subprocess.run(
        [*argv, "sh", "-c", "echo hello"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "could not handle the command's output" in proc.stderr
    assert out.exists()


def test_a_survivor_in_its_own_process_group_is_still_terminated(tmp_path):
    marker = tmp_path / "pid"
    # The worker leaves the root's process group (setpgrp makes it a group
    # leader) but stays in the session, and holds no pipe.
    child = (
        "import os, subprocess, pathlib\n"
        "p = subprocess.Popen(['sleep', '60'], preexec_fn=os.setpgrp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
    )
    argv, out = _memwatch(tmp_path, "--grace", "0.1")
    t0 = time.monotonic()
    proc = subprocess.run(
        [*argv, sys.executable, "-c", child], capture_output=True, text=True, timeout=60
    )
    assert time.monotonic() - t0 < 20
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "still running" in proc.stderr and "could not terminate" not in proc.stderr
    assert json.loads(out.read_text())["warnings"] == []
    assert _is_gone(
        int(marker.read_text())
    ), "a worker in its own process group survived"


def test_a_failing_stdout_destination_terminates_the_command(tmp_path):
    marker = tmp_path / "pid"
    child = (
        "import os, time, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "print('hello', flush=True)\n"
        "time.sleep(30)\n"
    )
    out = tmp_path / "summary.json"
    t0 = time.monotonic()
    with open("/dev/full", "w") as devfull:
        proc = subprocess.run(
            [
                sys.executable,
                str(MEMWATCH),
                "--interval",
                "0.05",
                "--out",
                str(out),
                "--",
                sys.executable,
                "-c",
                child,
            ],
            stdout=devfull,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    assert time.monotonic() - t0 < 20, "memwatch kept running with an unwritable stdout"
    assert (
        proc.returncode != 0 and "could not handle the command's output" in proc.stderr
    ), proc.stderr
    assert _is_gone(int(marker.read_text()))
