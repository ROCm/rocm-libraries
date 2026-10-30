# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""run_aotriton.py: command composition, collected.txt and result-directory rules (no pytest run)."""

import json
import subprocess
from pathlib import Path

import pytest

import aotriton_layout as layout
import conformance_contract as contract
import reconcile
import run_aotriton


class FakePytest:
    """Stands in for subprocess.run: answers collection with canned ids, records every command."""

    def __init__(self, ids_per_pass):
        self.ids_per_pass = list(ids_per_pass)
        self.commands = []

    def __call__(self, cmd, env=None, capture_output=False, text=False):
        self.commands.append((cmd, env))
        if "--collect-only" in cmd:
            ids = self.ids_per_pass.pop(0)
            out = (
                "".join(f"{i}\n" for i in ids)
                + f"\n{len(ids)} tests collected in 0.1s\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, 0)


class FakeWatchdog:
    """Stands in for ``start_watchdog``: records the command and the log file, stops on terminate."""

    started = []

    def __init__(self, cmd, env, log):
        self.cmd, self.env, self.log, self.stopped = cmd, env, log, False
        FakeWatchdog.started.append(self)

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def watchdog(monkeypatch):
    FakeWatchdog.started = []
    monkeypatch.setattr(run_aotriton, "start_watchdog", FakeWatchdog)
    return FakeWatchdog


@pytest.fixture(autouse=True)
def rocm_path(monkeypatch):
    """The setup_env.py activation sets ROCM_PATH; the runner's --hipdnn-prefix defaults to it."""
    monkeypatch.setenv("ROCM_PATH", "/opt/fake-rocm")


@pytest.fixture
def runner(monkeypatch):
    def go(ids_per_pass, *argv):
        fake = FakePytest(ids_per_pass)
        monkeypatch.setattr(run_aotriton.subprocess, "run", fake)
        status = run_aotriton.main(list(argv))
        return status, fake.commands

    return go


def test_forward_suite_names_test_forward_explicitly_and_exports_the_release(
    runner, tmp_path
):
    status, commands = runner(
        [["f::a", "f::b", "f::c"]],
        "--workers",
        "4",
        "--out",
        str(tmp_path),
        "--suite",
        "forward",
        "--release",
        "1",
        "--shard",
        "2/3",
        "--fake",
        "always-decline",
    )
    collect, run = commands
    assert status == 0
    assert str(layout.FLASH_TESTS / "test_forward.py") in collect[0]
    assert "--results-dir" not in " ".join(collect[0])
    for cmd, env in commands:
        assert {"-p", "hipdnn_aotriton_plugin"} <= set(cmd) and "--shard=2/3" in cmd
        assert "--fake=always-decline" in cmd and env["FOR_RELEASE"] == "1"
        assert env["PYTHONPATH"].split(":")[0] == str(layout.SHIM)
    assert f"--results-dir={tmp_path}" in run[0]
    assert run[0][run[0].index("-n") + 1] == "4" and "--max-worker-restart=3" in run[0]
    assert (tmp_path / "collected.txt").read_text() == "f::a\nf::b\nf::c\n"


def test_packed_gpu_adds_a_second_gpu_marked_pass_into_the_same_collected_list(
    runner, tmp_path
):
    status, commands = runner(
        [["v::a"], ["p::x", "p::y"]],
        "--workers",
        "1",
        "--out",
        str(tmp_path),
        "--packed-gpu",
    )
    assert len(commands) == 4  # collect + run, twice
    second = commands[2][0]
    assert (
        second[len(second) - 1 - second[::-1].index("-m") + 1] == "gpu"
        and str(layout.PACKED_TESTS) in second
    )
    assert "--max-worker-restart=2" in commands[3][0]
    assert (tmp_path / "collected.txt").read_text() == "v::a\np::x\np::y\n"


def test_default_out_is_arch_run_shard_directory(monkeypatch, runner):
    monkeypatch.setattr(run_aotriton, "DEFAULT_RESULTS", run_aotriton.Path("/nowhere"))
    args = run_aotriton.parse_args(
        ["--arch", "gfx950", "--run", "r1", "--shard", "2/4", "--workers", "1"]
    )
    assert (
        run_aotriton.resolve_out(args, None).as_posix()
        == "/nowhere/gfx950/r1/shard-2of4"
    )
    args = run_aotriton.parse_args(["--workers", "1"])
    with pytest.raises(SystemExit):
        run_aotriton.resolve_out(args, None)


def test_existing_results_are_never_appended_to(runner, tmp_path):
    (tmp_path / "results.csv").write_text(
        "nodeid,outcome,mode,fwd,bwd,reason\nx,pass,,,,\n"
    )
    with pytest.raises(SystemExit, match="already has rows"):
        runner([["a"]], "--workers", "1", "--out", str(tmp_path))


def test_empty_collection_runs_nothing_and_collection_errors_stop_the_run(
    runner, tmp_path, monkeypatch
):
    status, commands = runner([[]], "--workers", "1", "--out", str(tmp_path / "a"))
    assert status == 0 and len(commands) == 1

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 2, stdout="ERROR collecting x", stderr=""
        )

    monkeypatch.setattr(run_aotriton.subprocess, "run", failing)
    assert run_aotriton.main(["--workers", "1", "--out", str(tmp_path / "b")]) == 2


def test_every_pytest_command_checks_the_hipdnn_prefix(runner, tmp_path):
    status, commands = runner(
        [["t.py::a"]], "--workers", "1", "--out", str(tmp_path / "o")
    )
    assert status == 0
    assert commands and all(
        "--hipdnn-prefix=/opt/fake-rocm" in cmd for cmd, _ in commands
    )


def test_runner_refuses_to_start_without_a_hipdnn_prefix(monkeypatch, tmp_path):
    monkeypatch.delenv("ROCM_PATH")
    with pytest.raises(SystemExit, match="no hipDNN prefix"):
        run_aotriton.main(["--workers", "1", "--out", str(tmp_path / "o")])


def test_fwd_no_stats_is_refused_outside_test_forward(runner, tmp_path):
    for argv, ids in (
        (["--suite", "default"], []),
        (["--suite", "forward", "--packed-gpu"], []),
    ):
        with pytest.raises(SystemExit, match="--fwd-no-stats"):
            runner(
                ids,
                "--workers",
                "1",
                "--out",
                str(tmp_path / "o"),
                "--fwd-no-stats",
                *argv,
            )
    status, commands = runner(
        [["tests/test_forward.py::a", "tests/test_backward.py::b"]],
        "--workers", "1", "--out", str(tmp_path / "p"),
        "--suite", "forward", "--fwd-no-stats",
    )  # fmt: skip
    assert status == 2 and len(commands) == 1  # collection only; nothing ran


def test_fwd_no_stats_exports_the_env_var_only_when_asked(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setenv(
        run_aotriton.FWD_NO_STATS_ENV, "1"
    )  # stale value from the caller
    _, plain = runner([["t/test_forward.py::a"]], "--workers", "1", "--out", str(tmp_path / "a"), "--suite", "forward")  # fmt: skip
    _, flagged = runner([["t/test_forward.py::a"]], "--workers", "1", "--out", str(tmp_path / "b"), "--suite", "forward", "--fwd-no-stats")  # fmt: skip
    assert all(run_aotriton.FWD_NO_STATS_ENV not in env for _, env in plain)
    assert len(flagged) == 2
    assert all(env[run_aotriton.FWD_NO_STATS_ENV] == "1" for _, env in flagged)


FWD = "vendor/test_forward.py"
BWD = "vendor/test_backward.py"


def pass1_row(nodeid, fwd_state="", bwd_state="", fwd_engines="", bwd_engines=""):
    return {
        "nodeid": nodeid,
        "outcome": "pass",
        "fwd": fwd_state,
        "bwd": bwd_state,
        "fwd_engines": fwd_engines,
        "bwd_engines": bwd_engines,
    }


def test_all_engines_sweeps_each_non_winning_engine_over_its_cases_only(
    monkeypatch, tmp_path
):
    pass1 = [
        # forward cases: winner excluded per case; a case a lone engine served adds no work
        pass1_row(f"{FWD}::f1", "hipdnn:A", fwd_engines="A;B;C"),
        pass1_row(f"{FWD}::f2", "hipdnn:B", fwd_engines="A;B"),
        pass1_row(f"{FWD}::f3", "hipdnn:A", fwd_engines="A"),
        # backward cases read bwd; their forward list is not swept
        pass1_row(f"{BWD}::b1", "hipdnn:X", "hipdnn:A", "X;Y", "A;B"),
        pass1_row(f"{BWD}::b2", "ref:why", "hipdnn:B", "", "A;B"),
        pass1_row(f"{BWD}::b3", "hipdnn:X", "declined:no", "X", ""),
    ]
    fake = FakePytest([[r["nodeid"] for r in pass1]])
    fake_call = fake.__call__

    def call(cmd, env=None, **kw):
        if "--collect-only" not in cmd and f"--results-dir={tmp_path}" in cmd:
            (tmp_path / "results.csv").write_text(
                ",".join(run_aotriton.reconcile.FIELDS)
                + "\n"
                + "".join(run_aotriton.reconcile.format_row(r) for r in pass1)
            )
        return fake_call(cmd, env=env, **kw)

    monkeypatch.setattr(run_aotriton.subprocess, "run", call)
    monkeypatch.setenv("ROCM_PATH", "/opt/fake-rocm")
    status = run_aotriton.main(
        ["--workers", "2", "--out", str(tmp_path), "--all-engines", "--shard", "1/2"]
    )
    assert status == 0
    sweeps = {}
    for cmd, env in fake.commands[2:]:
        pins = [k for k in env if k in run_aotriton.ENGINE_ENV.values() and env[k]]
        if len(pins) != 1:
            continue  # pair passes: both pins, covered by their own test
        (pin,) = pins
        listing = next(a for a in cmd if a.startswith("--hipdnn-nodeids-file=")).split(
            "="
        )[1]
        sweeps[(pin, env[pin])] = [
            line.split("::")[-1] for line in open(listing).read().splitlines()
        ]
        assert not any(a.startswith("--shard") for a in cmd)  # the ids are the shard
        assert f"--results-dir={tmp_path}" not in cmd
    assert sweeps == {
        ("HIPDNN_FWD_ENGINE", "A"): ["f2"],
        ("HIPDNN_FWD_ENGINE", "B"): ["f1"],
        ("HIPDNN_FWD_ENGINE", "C"): ["f1"],
        ("HIPDNN_BWD_ENGINE", "A"): ["b2"],
        ("HIPDNN_BWD_ENGINE", "B"): ["b1"],
    }
    meta = json.loads(
        (tmp_path / "all-engines" / "fwd" / "C" / "pass.json").read_text()
    )
    assert (meta["kind"], meta["direction"], meta["pin_fwd"]) == ("sweep", "fwd", "C")


def test_all_engines_is_refused_with_fake_or_a_preset_pin(
    runner, tmp_path, monkeypatch
):
    with pytest.raises(SystemExit, match="--fake"):
        runner([], "--workers", "1", "--out", str(tmp_path), "--all-engines", "--fake", "always-decline")  # fmt: skip
    monkeypatch.setenv("HIPDNN_BWD_ENGINE", "X")
    with pytest.raises(SystemExit, match="HIPDNN_BWD_ENGINE"):
        runner([], "--workers", "1", "--out", str(tmp_path), "--all-engines")


# ------------------------------------------------------------------ passes, resume, watchdog, capture


def flag(cmd, name):
    return next((a.split("=", 1)[1] for a in cmd if a.startswith(f"{name}=")), None)


def rows_of(nodeids, **fields):
    return [
        {
            **dict.fromkeys(reconcile.FIELDS, ""),
            "nodeid": n,
            "outcome": "pass",
            **fields,
        }
        for n in nodeids
    ]


class Sim(FakePytest):
    """Stands in for subprocess.run and the plugin: every pytest run appends the rows
    ``behave(nodeid, pins, results_dir)`` returns (None: the case is not reached) to its
    results.csv, as the controller would."""

    def __init__(self, ids_per_pass, behave):
        super().__init__(ids_per_pass)
        self.behave, self.collected, self.runs = behave, [], []

    def __call__(self, cmd, env=None, capture_output=False, text=False):
        if "--collect-only" in cmd:
            status = super().__call__(cmd, env, capture_output, text)
            self.collected = [l for l in status.stdout.splitlines() if "::" in l]
            listed = flag(cmd, "--hipdnn-nodeids-file")
            if listed:
                wanted = Path(listed).read_text().split()
                self.collected = [i for i in self.collected if i in wanted]
            status.stdout = "".join(f"{i}\n" for i in self.collected)
            return status
        self.commands.append((cmd, env))
        listing = flag(cmd, "--hipdnn-nodeids-file")
        ids = Path(listing).read_text().split() if listing else self.collected
        results = Path(flag(cmd, "--results-dir"))
        pins = {
            "fwd": env.get("HIPDNN_FWD_ENGINE"),
            "bwd": env.get("HIPDNN_BWD_ENGINE"),
            "capture": env.get(contract.REPRO_CAPTURE_ENV),
            "slug": env.get(contract.PASS_SLUG_ENV),
        }
        self.runs.append((results, pins, ids))
        results.mkdir(parents=True, exist_ok=True)
        path = results / "results.csv"
        new = not path.exists() or not path.stat().st_size
        with open(path, "a") as f:
            if new:
                f.write(",".join(reconcile.FIELDS) + "\n")
            for nodeid in ids:
                row = self.behave(nodeid, pins, results)
                if row:
                    f.write(reconcile.format_row(row))
        return subprocess.CompletedProcess(cmd, 0)


def run_all_engines(monkeypatch, tmp_path, ids, behave, *extra):
    sim = Sim([ids], behave)
    monkeypatch.setattr(run_aotriton.subprocess, "run", sim)
    status = run_aotriton.main(
        ["--workers", "2", "--out", str(tmp_path), "--all-engines", *extra]
    )
    return status, sim


def listing(sim, results):
    """Node ids (test name only) the run into ``results`` was given, in run order."""
    return [[i.split("::")[-1] for i in ids] for r, _, ids in sim.runs if r == results]


def ok_behave(by_pass):
    """behave(): every case passes; ``by_pass[(fwd pin, bwd pin)]`` overrides the row fields."""

    def behave(nodeid, pins, results):
        name = nodeid.split("::")[-1]
        fields = by_pass.get((pins["fwd"], pins["bwd"], name)) or by_pass.get(
            (pins["fwd"], pins["bwd"]), {}
        )
        if callable(fields):
            fields = fields(name)
        return rows_of([nodeid], **fields)[0] if fields is not None else None

    return behave


def test_pair_passes_run_every_other_forward_with_every_backward_under_pair_dirs(
    monkeypatch, tmp_path
):
    ids = [f"{BWD}::b1", f"{BWD}::b2"]
    pass1 = dict(fwd="hipdnn:A", bwd="hipdnn:X", fwd_engines="A;B", bwd_engines="X;Y")
    status, sim = run_all_engines(
        monkeypatch, tmp_path, ids, ok_behave({(None, None): pass1})
    )
    pairs = tmp_path / "all-engines" / "pair"
    assert sorted(p.name for p in pairs.iterdir()) == ["B__X", "B__Y"]
    for name, (fwd, bwd) in {"B__X": ("B", "X"), "B__Y": ("B", "Y")}.items():
        ((results, pins, given),) = [r for r in sim.runs if r[0] == pairs / name]
        assert (pins["fwd"], pins["bwd"], pins["slug"]) == (fwd, bwd, f"pair-{name}")
        assert [i.split("::")[-1] for i in given] == ["b1", "b2"]
        meta = json.loads((pairs / name / "pass.json").read_text())
        assert (meta["kind"], meta["pin_fwd"], meta["pin_bwd"]) == ("pair", fwd, bwd)
        assert len(reconcile.read_rows(pairs / name / "results.csv")) == 2
    # the backward-only sweep still pins just the other backward on the heuristic forward
    assert listing(sim, tmp_path / "all-engines" / "bwd" / "Y") == [["b1", "b2"]]
    assert status == 0


def test_a_case_whose_forward_crashed_gets_refwd_discovery_refwd_and_pair_passes(
    monkeypatch, tmp_path
):
    ids = [f"{BWD}::b1", f"{BWD}::b2"]

    def behave(nodeid, pins, results):
        name = nodeid.split("::")[-1]
        crashed = dict(
            outcome="crash", fwd="hipdnn:A", fail_side="fwd", stage="execute",
            fwd_engines="A;B", reason="worker died (pid 5)",
        )  # fmt: skip
        healthy = dict(
            fwd="hipdnn:A", bwd="hipdnn:X", fwd_engines="A;B", bwd_engines="X;Y"
        )
        if (pins["fwd"], pins["bwd"]) == (None, None):
            return rows_of([nodeid], **(crashed if name == "b1" else healthy))[0]
        if pins["fwd"] == "reference":  # refwd discovery and per-backward passes
            return rows_of(
                [nodeid], fwd="ref:pinned reference", bwd=f"hipdnn:{pins['bwd'] or 'X'}",
                bwd_engines="" if pins["bwd"] else "X;Y",
            )[0]  # fmt: skip
        if pins["fwd"] == "A":  # never run: A is the winner everywhere
            raise AssertionError(f"forward A pinned for {name}")
        return rows_of(
            [nodeid], fwd=f"hipdnn:{pins['fwd']}", bwd=f"hipdnn:{pins['bwd']}"
        )[0]

    status, sim = run_all_engines(monkeypatch, tmp_path, ids, behave)
    root = tmp_path / "all-engines"
    discovery = [r for r in sim.runs if r[0] == root / "refwd" / "discovery"]
    assert [(p["fwd"], p["bwd"]) for _, p, _ in discovery] == [("reference", None)]
    assert listing(sim, root / "refwd" / "discovery") == [["b1"]]
    assert listing(sim, root / "refwd" / "Y") == [
        ["b1"]
    ]  # the other backward, forward = reference
    ((_, pins, _),) = [r for r in sim.runs if r[0] == root / "refwd" / "Y"]
    assert (pins["fwd"], pins["bwd"], pins["slug"]) == ("reference", "Y", "refwd-Y")
    # pairs: b2 from its pass-1 lists; b1 from the marker's forwards and the discovery's backwards
    for pair in (
        "B__X",
        "B__Y",
    ):  # b2 from its pass-1 lists first, then b1 once discovery named its backwards
        assert listing(sim, root / "pair" / pair) == [["b2"], ["b1"]]
        assert listing_of(root / "pair" / pair) == ["b1", "b2"]
    assert listing(sim, root / "bwd" / "Y") == [
        ["b2"]
    ]  # the crash is not swept on its backward
    assert (root / "refwd" / "X").exists() is False
    assert status == 0


def test_no_refwd_pass_exists_when_no_forward_failed(monkeypatch, tmp_path):
    ids = [f"{BWD}::b1"]
    pass1 = dict(fwd="hipdnn:A", bwd="hipdnn:X", fwd_engines="A;B", bwd_engines="X;Y")
    run_all_engines(monkeypatch, tmp_path, ids, ok_behave({(None, None): pass1}))
    assert not (tmp_path / "all-engines" / "refwd").exists()


def test_a_forward_crash_in_pass_one_schedules_the_other_engines_from_the_marker_lists(
    monkeypatch, tmp_path
):
    ids = [f"{FWD}::f1", f"{FWD}::f2"]

    def behave(nodeid, pins, results):
        if (pins["fwd"], pins["bwd"]) != (None, None):
            return rows_of([nodeid], fwd=f"hipdnn:{pins['fwd']}")[0]
        base = dict(
            outcome="crash", fail_side="fwd", stage="build", fwd_engines="A;B;C"
        )
        if nodeid.endswith("f1"):
            return rows_of([nodeid], fwd="hipdnn:A", **base)[0]  # engine known: A
        return rows_of([nodeid], **base)[0]  # killed while ranking: engine unknown

    run_all_engines(monkeypatch, tmp_path, ids, behave)
    swept = {
        p.name: listing_of(p) for p in (tmp_path / "all-engines" / "fwd").iterdir()
    }
    assert swept == {"A": ["f2"], "B": ["f1", "f2"], "C": ["f1", "f2"]}


def listing_of(pass_dir):
    return [i.split("::")[-1] for i in (pass_dir / "nodeids.txt").read_text().split()]


def test_resume_runs_every_missing_id_of_the_shard_without_shard_and_never_a_crash(
    monkeypatch, tmp_path
):
    ids = [f"{FWD}::{n}" for n in "abcd"]
    (tmp_path / "results.csv").write_text(
        ",".join(reconcile.FIELDS) + "\n"
        + reconcile.format_row(rows_of([ids[0]])[0])
        + reconcile.format_row(rows_of([ids[2]], outcome="crash")[0])
    )  # fmt: skip
    sim = Sim([ids], ok_behave({(None, None): {}}))
    monkeypatch.setattr(run_aotriton.subprocess, "run", sim)
    status = run_aotriton.main(
        ["--workers", "2", "--out", str(tmp_path), "--resume", "--shard", "1/2"]
    )
    collect, run, capture = (c for c, _ in sim.commands)
    assert "--shard=1/2" in collect
    assert not any(a.startswith("--shard") for a in run)
    assert [i.split("::")[-1] for i in sim.runs[0][2]] == ["b", "d"]
    assert "--max-worker-restart=2" in run
    # the earlier crash is not run as a case again; only its missing repro capture runs
    assert flag(capture, "--results-dir") == str(tmp_path / "capture")
    assert [i.split("::")[-1] for i in sim.runs[1][2]] == ["c"]
    rows = {
        r["nodeid"].split("::")[-1]: r["outcome"]
        for r in reconcile.read_rows(tmp_path / "results.csv")
    }
    assert rows == {"a": "pass", "b": "pass", "c": "crash", "d": "pass"}


def test_resume_of_a_pinned_pass_skips_recorded_ids_and_a_fresh_run_still_refuses_a_non_empty_out(
    monkeypatch, tmp_path
):
    ids = [f"{FWD}::{n}" for n in "ab"]
    sweep = tmp_path / "all-engines" / "fwd" / "B"
    sweep.mkdir(parents=True)
    (sweep / "results.csv").write_text(
        ",".join(reconcile.FIELDS)
        + "\n"
        + reconcile.format_row(rows_of([ids[0]], fwd="hipdnn:B")[0])
    )
    pass1 = dict(fwd="hipdnn:A", fwd_engines="A;B")
    sim = Sim(
        [ids], ok_behave({(None, None): pass1, ("B", None): dict(fwd="hipdnn:B")})
    )
    monkeypatch.setattr(run_aotriton.subprocess, "run", sim)
    (tmp_path / "results.csv").write_text("")
    run_aotriton.main(
        ["--workers", "1", "--out", str(tmp_path), "--all-engines", "--resume"]
    )
    assert listing(sim, sweep) == [["b"]]  # a was recorded
    assert listing_of(sweep) == [
        "a",
        "b",
    ]  # the pass still lists both for the accounting
    with pytest.raises(SystemExit, match="already has rows"):
        run_aotriton.main(["--workers", "1", "--out", str(tmp_path)])


def test_the_watchdog_runs_beside_each_pytest_run_and_a_kill_becomes_a_hang_row(
    monkeypatch, tmp_path, watchdog
):
    ids = [f"{FWD}::f1"]

    def behave(nodeid, pins, results):
        (results / "watchdog.log").parent.mkdir(parents=True, exist_ok=True)
        watchdog.started[-1].log.write(
            "pytest_gpu_lease.watchdog: sent SIGKILL to pid 99 (silent 31s)\n"
        )
        watchdog.started[-1].log.flush()
        return rows_of(
            [nodeid], outcome="crash", fail_side="fwd", reason="worker died (pid 99)"
        )[0]

    sim = Sim([ids], behave)
    monkeypatch.setattr(run_aotriton.subprocess, "run", sim)
    run_aotriton.main(
        [
            "--workers",
            "3",
            "--out",
            str(tmp_path),
            "--hang-timeout",
            "600",
            "--lock-dir",
            str(tmp_path / "locks"),
        ]
    )
    dog, capture_dog = watchdog.started  # the case run, then its repro capture
    assert Path(capture_dog.log.name) == tmp_path / "capture" / "watchdog.log"
    lock = str(tmp_path / "locks" / "gpu.lock")
    assert dog.cmd[1:3] == ["-m", "pytest_gpu_lease.watchdog"]
    assert dog.cmd[dog.cmd.index("--lockfile") + 1] == lock
    assert (
        dog.cmd[dog.cmd.index("--workers") + 1],
        dog.cmd[dog.cmd.index("--threshold") + 1],
    ) == ("3", "600.0")
    assert sim.commands[-1][1]["GPU_LEASE_LOCKFILE"] == lock
    assert (
        dog.stopped
        and capture_dog.stopped
        and Path(dog.log.name) == tmp_path / "watchdog.log"
    )
    (row,) = reconcile.read_rows(tmp_path / "results.csv")
    assert row["outcome"] == "crash" and row["reason"].startswith("hang")


def test_a_hang_timeout_of_zero_starts_no_watchdog(monkeypatch, tmp_path, watchdog):
    sim = Sim([[f"{FWD}::f1"]], ok_behave({(None, None): {}}))
    monkeypatch.setattr(run_aotriton.subprocess, "run", sim)
    run_aotriton.main(["--workers", "1", "--out", str(tmp_path), "--hang-timeout", "0"])
    assert watchdog.started == [] and sim.runs


def test_every_pass_gets_its_slug_and_a_clean_in_flight_directory(
    monkeypatch, tmp_path
):
    inflight = tmp_path / "inflight"
    stale = inflight / "main"
    stale.mkdir(parents=True)
    (stale / "gw0.json").write_text("{}")
    seen = []

    def behave(nodeid, pins, results):
        seen.append(
            (pins["slug"], sorted(p.name for p in (inflight / pins["slug"]).glob("*")))
        )
        return (
            rows_of([nodeid], fwd="hipdnn:A", fwd_engines="A;B")[0]
            if pins["slug"] == "main"
            else rows_of([nodeid], fwd="hipdnn:B")[0]
        )

    monkeypatch.setenv(contract.INFLIGHT_DIR_ENV, str(inflight))
    run_all_engines(monkeypatch, tmp_path, [f"{FWD}::f1"], behave)
    assert seen == [("main", []), ("fwd-B", [])]


def test_new_crashes_are_rerun_once_with_repro_capture_into_capture_dirs_within_the_cap(
    monkeypatch, tmp_path
):
    ids = [f"{FWD}::test_f[{i}]" for i in range(4)]
    state = {}

    def behave(nodeid, pins, results):
        if pins["capture"]:
            assert (
                results == tmp_path / "all-engines" / "fwd" / "B" / "capture"
                or results == tmp_path / "capture"
            )
            state.setdefault("capture", []).append((results, pins["fwd"], pins["slug"]))
            return rows_of(
                [nodeid],
                outcome="crash",
                fwd=f"hipdnn:{pins['fwd'] or 'A'}",
                fail_side="fwd",
            )[0]
        if pins["fwd"] == "B":
            return rows_of([nodeid], outcome="crash", fwd="hipdnn:B", fail_side="fwd")[
                0
            ]
        return rows_of(
            [nodeid],
            outcome="crash",
            fwd="hipdnn:A",
            fail_side="fwd",
            fwd_engines="A;B",
        )[0]

    run_all_engines(monkeypatch, tmp_path, ids, behave)
    captures = state["capture"]
    main_cap = [c for c in captures if c[0] == tmp_path / "capture"]
    sweep_cap = [c for c in captures if c[0] != tmp_path / "capture"]
    # per (engine, test function): 3 slots; pass 1 on engine A and the sweep on B each get their own
    assert len(main_cap) == contract.REPRO_MAX_PER_KEY == len(sweep_cap)
    assert {(c[1], c[2]) for c in main_cap} == {(None, "main")}
    assert {(c[1], c[2]) for c in sweep_cap} == {("B", "fwd-B")}
    capture_env = [
        e
        for _, e in run_aotriton.subprocess.run.commands
        if e.get(contract.REPRO_CAPTURE_ENV)
    ]
    assert {e[contract.REPRO_CAPTURE_ENV] for e in capture_env} == {
        str(tmp_path / contract.REPRO_DIR)
    }
    # capture results never enter the engine rows or the accounting
    assert not any("capture" in str(d) for d, _ in reconcile.pass_dirs(tmp_path))
    assert all(
        r["source"] in ("heuristic", "pinned") for r in reconcile.engine_rows(tmp_path)
    )
    assert "capture" not in reconcile.pass_accounting(tmp_path)


def test_nodeids_file_limits_pass_one_only(monkeypatch, tmp_path):
    ids = [f"{FWD}::f1", f"{FWD}::f2"]
    mine = tmp_path / "mine.txt"
    mine.write_text(f"{ids[0]}\n")
    pass1 = dict(fwd="hipdnn:A", fwd_engines="A;B")
    sim = Sim(
        [ids], ok_behave({(None, None): pass1, ("B", None): dict(fwd="hipdnn:B")})
    )
    monkeypatch.setattr(run_aotriton.subprocess, "run", sim)
    run_aotriton.main(
        [
            "--workers",
            "1",
            "--out",
            str(tmp_path / "o"),
            "--all-engines",
            "--nodeids-file",
            str(mine),
        ]
    )
    collect, run, sweep = (c for c, _ in sim.commands)
    assert flag(collect, "--hipdnn-nodeids-file") == str(mine)
    assert flag(run, "--hipdnn-nodeids-file") == str(mine)
    assert flag(sweep, "--hipdnn-nodeids-file") != str(mine)
    assert (tmp_path / "o" / "collected.txt").read_text() == f"{ids[0]}\n"


# ------------------------------------------------------------------ end to end: real pytest, real plugin

SUITE_FORWARD = """
import hipdnn_sdpa, torch

def test_f1():
    t = torch.zeros(1)
    hipdnn_sdpa.sdpa_fwd(t, t, t, o=t, stats=t, scale=1.0)
"""
SUITE_BACKWARD = """
import hipdnn_aotriton_plugin as plug
import hipdnn_sdpa, torch

def run_case():
    t = torch.zeros(1)
    try:
        hipdnn_sdpa.sdpa_fwd(t, t, t, o=t, stats=t, scale=1.0)
    except hipdnn_sdpa.EngineDeclined as declined:
        plug.record('fwd', 'ref', declined.reason)  # what the shim does
    hipdnn_sdpa.sdpa_bwd(t, t, t, t, t, t, dq=t, dk=t, dv=t, scale=1.0)

def test_b1():
    run_case()

def test_b2():
    run_case()
"""


@pytest.fixture
def real_run(monkeypatch, tmp_path):
    """The runner over a synthetic suite with real pytest subprocesses, the real plugin and the
    file-backed fake ``hipdnn_sdpa`` (``tests/fakes``); the lease watchdog is not started.
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_forward.py").write_text(SUITE_FORWARD)
    (suite / "test_backward.py").write_text(SUITE_BACKWARD)
    rocm = tmp_path / "rocm"
    (rocm / "lib").mkdir(parents=True)
    import ctypes

    ctypes.CDLL("libm.so.6")  # any shared object stands in for the backend library
    libm = next(l.split()[-1] for l in open("/proc/self/maps") if "libm.so" in l)
    (rocm / "lib" / "libhipdnn_backend.so").write_bytes(Path(libm).read_bytes())
    fakes = Path(__file__).parent / "fakes"
    monkeypatch.chdir(
        tmp_path
    )  # python -m pytest puts the cwd on sys.path: keep the real package out
    monkeypatch.setattr(layout, "SDPA_ROOT", fakes)
    monkeypatch.setattr(run_aotriton, "passes", lambda args: [("suite", [str(suite)])])
    monkeypatch.setenv("FAKE_ROCM", str(rocm))
    monkeypatch.setenv("FAKE_FWD_ENGINES", "A;B")
    monkeypatch.setenv("FAKE_BWD_ENGINES", "X;Y")
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setenv(contract.INFLIGHT_DIR_ENV, str(tmp_path / "inflight"))
    monkeypatch.setattr(run_aotriton, "start_watchdog", FakeWatchdog)
    out = tmp_path / "out"

    def go(*argv):
        status = run_aotriton.main(["--workers", "1", "--out", str(out), *argv])
        return status, out

    return go


def by_name(path):
    return {r["nodeid"].split("::")[-1]: r for r in reconcile.read_rows(path)}


def test_a_forward_that_dies_is_attributed_scheduled_around_and_captured_end_to_end(
    real_run, monkeypatch
):
    monkeypatch.setenv(contract.TEST_CRASH_ENV, "fwd:A")
    status, out = real_run("--all-engines")
    rows = by_name(out / "results.csv")
    assert {n: r["outcome"] for n, r in rows.items()} == {
        "test_f1": "crash", "test_b1": "crash", "test_b2": "crash",
    }  # fmt: skip
    f1 = rows["test_f1"]
    assert (f1["fwd"], f1["fail_side"], f1["stage"], f1["fwd_engines"], f1["fwd_kernel"]) == (
        "hipdnn:A", "fwd", "execute", "A;B", "kernel-A",
    )  # fmt: skip
    # the surviving forward engine ran the forward case
    sweep = by_name(out / "all-engines" / "fwd" / "B" / "results.csv")
    assert (sweep["test_f1"]["outcome"], sweep["test_f1"]["fwd"]) == (
        "pass",
        "hipdnn:B",
    )
    # the backward cases got the reference-forward passes and pairs on the other forward
    refwd = out / "all-engines" / "refwd"
    discovery = by_name(refwd / "discovery" / "results.csv")
    assert {r["bwd_engines"] for r in discovery.values()} == {"X;Y"}
    assert {r["outcome"] for r in discovery.values()} == {"pass"}
    assert {r["bwd"] for r in by_name(refwd / "Y" / "results.csv").values()} == {
        "hipdnn:Y"
    }
    for pair in ("B__X", "B__Y"):
        pass_rows = by_name(out / "all-engines" / "pair" / pair / "results.csv")
        assert set(pass_rows) == {"test_b1", "test_b2"}
        assert {r["outcome"] for r in pass_rows.values()} == {"pass"}
    # the crashes were re-run with capture; each left a repro directory the case name points at
    captured = by_name(out / "capture" / "results.csv")
    assert set(captured) == {"test_f1", "test_b1", "test_b2"}
    assert {r["outcome"] for r in captured.values()} == {"crash"}
    cases = sorted(
        p.name for p in (out / "repro").iterdir() if not p.name.startswith(".")
    )
    assert cases == sorted(
        contract.repro_case(r["nodeid"], "main") for r in rows.values()
    )
    # engines.csv: the crash names its engine; capture and every pass are accounted for
    reconcile.write_engines(out)
    table = list(reconcile.engine_rows(out))
    crash = [
        r for r in table if r["outcome"] == "crash" and r["nodeid"].endswith("test_f1")
    ]
    assert [(r["engine"], r["direction"], r["stage"]) for r in crash] == [
        ("A", "fwd", "execute")
    ]
    assert not [r for r in table if r["outcome"] == "not-run"]
    assert status != 0
