# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Launching, cancelling, reconciling -- everything the engine cannot know.

Hermetic: every process started here is this Python, either as the real worker
or as a stand-in for one. No flow, step, group or output name in this file
appears in any shipped flow, on purpose: the supervisor must not be able to tell.
"""
from __future__ import annotations

import json
import os
import textwrap
import threading
import time
from pathlib import Path

import pytest

from flowmcp import resources, schema
from flowmcp.supervisor import Supervisor, SupervisorError, merge_status

ROOT = Path(__file__).resolve().parents[1]

# -- flows ------------------------------------------------------------------


def single_step(agent_script: Path, sleep: float) -> str:
    return f"""
version: 1
name: single
description: One step.
inputs:
  target: {{type: string, default: unused}}
steps:
  - id: emit
    tool: agent
    args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
           "--sleep", "{sleep}", "--payload", '{{"bundle": "{agent_script.as_posix()}"}}']
    stdin: "produce something"
    result_file: "${{run.dir}}/emit.json"
    result_schema: {{required: [bundle]}}
    outputs:
      bundle: {{json_file: result, path: "$.bundle", type: path}}
"""


def bounded_loop(agent_script: Path, budget: int = 3) -> str:
    """A group whose exit condition is never satisfied, so it always spends its
    whole budget and then finishes rather than failing."""
    return f"""
version: 1
name: bounded
description: One group, a budget that is always spent.
inputs:
  target: {{type: string, default: unused}}
steps:
  - id: cycle
    loop:
      max_iterations: {budget}
      until: "${{steps.draft.outputs.opened}} == 0"
      on_exhausted: continue
    steps:
      - id: draft
        tool: agent
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--payload", '{{"opened": 4}}']
        stdin: "draft it"
        result_file: "${{loop.attempt_dir}}/draft.json"
        result_schema: {{required: [opened]}}
        outputs:
          opened: {{json_file: result, path: "$.opened", type: int}}
"""


def flat_flow(agent_script: Path) -> str:
    return single_step(agent_script, 0.0).replace("name: single", "name: flat")


# -- fake workers -----------------------------------------------------------

#: Rewrites the manifest on demand so the supervisor's change detection can be
#: driven step by step instead of raced against.
FAKE_TICKER = textwrap.dedent(
    """
    import json, os, pathlib, sys, time

    spec = json.loads(sys.stdin.read())
    run_dir = pathlib.Path(spec["runDir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    control = pathlib.Path(os.environ["FLOWMCP_TEST_CONTROL"])

    def emit(frame):
        sys.stdout.write(json.dumps(frame) + "\\n")
        sys.stdout.flush()

    def checkpoint(n):
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": spec["runId"], "flow": "x", "flow_path": spec["flow"],
            "status": "running", "steps": [], "loops": [], "mark": "m" * (n + 1),
        }))

    tick = 0
    checkpoint(tick)
    emit({"t": "log", "text": "checkpoint 0"})
    (control / "started").write_text("1")
    while not (control / "stop").exists():
        nxt = control / ("tick-%d" % (tick + 1))
        if nxt.exists():
            tick += 1
            checkpoint(tick)
            emit({"t": "log", "text": "checkpoint %d" % tick})
            (control / ("done-%d" % tick)).write_text("1")
        time.sleep(0.01)
    emit({"t": "done", "status": "ok", "error": None})
    """
)

#: Spawns a grandchild that keeps a heartbeat file advancing, relays its pid,
#: and then waits to be killed.
FAKE_TREE = textwrap.dedent(
    """
    import json, os, pathlib, subprocess, sys, time

    spec = json.loads(sys.stdin.read())
    control = pathlib.Path(os.environ["FLOWMCP_TEST_CONTROL"])
    heartbeat = control / "beat.txt"
    code = (
        "import pathlib, sys, time\\n"
        "p = pathlib.Path(sys.argv[1])\\n"
        "n = 0\\n"
        "while True:\\n"
        "    n += 1\\n"
        "    p.write_text(str(n))\\n"
        "    time.sleep(0.02)\\n"
    )
    extra = {} if os.name == "nt" else {"start_new_session": True}
    child = subprocess.Popen([sys.executable, "-c", code, str(heartbeat)], **extra)
    sys.stdout.write(json.dumps({"t": "pid", "step": "emit", "pid": child.pid}) + "\\n")
    sys.stdout.flush()
    (control / "ready").write_text(str(child.pid))
    while True:
        time.sleep(0.2)
    """
)

#: Dies before writing anything at all, the way a run whose preflight fails does.
FAKE_DOOMED = textwrap.dedent(
    """
    import sys
    sys.stdin.read()
    sys.stderr.write("the worker could not start the run\\n")
    sys.exit(3)
    """
)


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def flows_dir(tmp_path, agent_script):
    directory = tmp_path / "flows"
    directory.mkdir()
    (directory / "single.yaml").write_text(
        single_step(agent_script, 0.0), encoding="utf-8"
    )
    (directory / "slow.yaml").write_text(
        single_step(agent_script, 6.0).replace("name: single", "name: slow"),
        encoding="utf-8",
    )
    (directory / "bounded.yaml").write_text(
        bounded_loop(agent_script), encoding="utf-8"
    )
    (directory / "flat.yaml").write_text(flat_flow(agent_script), encoding="utf-8")
    return directory


@pytest.fixture
def run_root(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def events():
    collected: list[dict] = []
    lock = threading.Lock()

    def record(event):
        with lock:
            collected.append(event)

    record.collected = collected  # type: ignore[attr-defined]
    return record


@pytest.fixture
def supervisors(registry_file, flows_dir, run_root, events):
    """Builds supervisors and guarantees every child they start is gone after."""
    built: list[Supervisor] = []

    def build(**kwargs):
        supervisor = Supervisor(
            tools_path=registry_file,
            flows_dir=flows_dir,
            run_root=run_root,
            emit=kwargs.pop("emit", events),
            poll_interval_s=kwargs.pop("poll_interval_s", 0.1),
            **kwargs,
        )
        built.append(supervisor)
        return supervisor

    yield build
    for supervisor in built:
        for run_id in list(supervisor._runs):
            supervisor.cancel(run_id)
        supervisor.close()


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition never became true")


def wait_terminal(supervisor: Supervisor, run_id: str, timeout: float = 60.0) -> str:
    def settled():
        status = supervisor.status(run_id)["status"]
        return status if status in ("ok", "failed", "crashed", "cancelled") else None

    # Polled at a cadence a client would actually use. Every call opens the
    # manifest, and the engine replaces it; hammering it from a test only
    # manufactures a contention window no consumer has.
    return wait_for(settled, timeout=timeout, interval=0.1)


def fake_worker(tmp_path: Path, name: str, source: str) -> Path:
    directory = tmp_path / "fakeworkers"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.py").write_text(source, encoding="utf-8")
    return directory


# -- launching --------------------------------------------------------------


def test_launch_returns_while_the_run_is_still_going(supervisors):
    supervisor = supervisors()
    supervisor.start()

    result = supervisor.launch(flow="slow", inputs={"target": "x"})

    assert result["status"] == "running"
    assert result["runDir"].endswith(result["runId"])
    assert result["runJsonUri"] == schema.manifest_uri(result["runId"])
    assert isinstance(result["pid"], int)
    # The multi-hour call has already returned and the run is still alive.
    assert supervisor.status(result["runId"])["status"] == "running"


def test_launching_into_a_fresh_run_directory_succeeds(supervisors, run_root):
    # The regression guard for keeping the process record out of the run
    # directory: the engine refuses an explicit run directory that already has
    # anything in it, so a record written there before the worker starts would
    # fail every launch.
    supervisor = supervisors()
    supervisor.start()

    result = supervisor.launch(flow="single", inputs={"target": "x"})
    assert wait_terminal(supervisor, result["runId"]) == "ok"

    run_dir = Path(result["runDir"])
    record = run_root / ".supervisor" / f"{result['runId']}.json"
    assert record.is_file()
    assert not record.is_relative_to(run_dir)
    assert (run_dir / schema.MANIFEST_NAME).is_file()
    assert json.loads((run_dir / schema.MANIFEST_NAME).read_text())["status"] == "ok"


def test_the_concurrency_cap_refuses_a_second_run(supervisors):
    supervisor = supervisors(max_concurrent=1)
    supervisor.start()
    first = supervisor.launch(flow="slow", inputs={"target": "x"})

    with pytest.raises(SupervisorError, match="Concurrency cap"):
        supervisor.launch(flow="slow", inputs={"target": "y"})

    assert supervisor.status(first["runId"])["status"] == "running"


# -- the iteration clamp ----------------------------------------------------


def test_a_request_cannot_raise_the_budget_the_flow_declares(supervisors):
    supervisor = supervisors()
    supervisor.start()

    result = supervisor.launch(
        flow="bounded", inputs={"target": "x"}, max_iterations=20
    )

    assert result["maxIterations"] == 3
    assert result["warnings"] and "20" in result["warnings"][0]
    assert wait_terminal(supervisor, result["runId"]) == "ok"
    status = supervisor.status(result["runId"])
    assert status["loops"][0]["budget"] == 3
    assert status["loops"][0]["iterations"] == 3


def test_a_request_below_the_budget_is_applied_as_asked(supervisors):
    supervisor = supervisors()
    supervisor.start()

    result = supervisor.launch(
        flow="bounded", inputs={"target": "x"}, max_iterations=1
    )

    assert result["maxIterations"] == 1
    assert result["warnings"] == []
    assert wait_terminal(supervisor, result["runId"]) == "ok"
    assert supervisor.status(result["runId"])["loops"][0]["iterations"] == 1


def test_an_operator_ceiling_is_the_only_thing_that_lifts_the_bound(supervisors):
    supervisor = supervisors(max_iterations_ceiling=5, max_concurrent=4)
    supervisor.start()

    lifted = supervisor.launch(flow="bounded", inputs={"target": "x"}, max_iterations=5)
    clamped = supervisor.launch(
        flow="bounded", inputs={"target": "y"}, max_iterations=7
    )

    assert lifted["maxIterations"] == 5
    assert lifted["warnings"] == []
    assert clamped["maxIterations"] == 5
    assert clamped["warnings"] and "5" in clamped["warnings"][0]


def test_a_flow_without_a_loop_refuses_an_iteration_budget(supervisors):
    supervisor = supervisors()
    supervisor.start()

    with pytest.raises(SupervisorError, match="declares no loop"):
        supervisor.launch(flow="flat", inputs={"target": "x"}, max_iterations=2)


# -- cancellation -----------------------------------------------------------


def test_cancel_is_recorded_and_the_manifest_is_left_alone(supervisors, run_root):
    supervisor = supervisors()
    supervisor.start()
    result = supervisor.launch(flow="slow", inputs={"target": "x"})
    manifest = Path(result["runDir"]) / schema.MANIFEST_NAME
    # The engine checkpoints again when the step launches. Snapshot after that,
    # or the comparison is against a manifest the engine was always going to
    # replace on its own.
    wait_for(
        lambda: manifest.is_file()
        and any(
            record["status"] == "running"
            for record in json.loads(manifest.read_text())["steps"]
        )
    )
    frozen = manifest.read_bytes()

    cancelled = supervisor.cancel(result["runId"])

    assert cancelled["cancelled"] is True
    assert cancelled["state"] == "cancelled"
    assert cancelled["killedPid"] == result["pid"]
    record = json.loads(
        (run_root / ".supervisor" / f"{result['runId']}.json").read_text()
    )
    assert record["state"] == "cancelled"
    assert record["cancelledAt"]
    # The supervisor never writes the manifest. A tree-killed worker leaves its
    # last checkpoint exactly as it was, which is the evidence for the run.
    assert manifest.read_bytes() == frozen
    assert supervisor.status(result["runId"])["status"] == "cancelled"


def test_cancelling_a_run_we_never_launched_is_not_an_error(supervisors):
    supervisor = supervisors()
    supervisor.start()

    result = supervisor.cancel("20260915T171233Z-ffff")

    assert result["cancelled"] is False
    assert result["state"] == "unknown"
    assert result["killedPid"] is None


@pytest.mark.skipif(os.name != "nt", reason="taskkill /T is the Windows mechanism")
def test_cancel_kills_the_whole_tree(supervisors, tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setenv("FLOWMCP_TEST_CONTROL", str(control))
    supervisor = supervisors(
        cwd=fake_worker(tmp_path, "treeworker", FAKE_TREE), worker_module="treeworker"
    )
    supervisor.start()
    result = supervisor.launch(flow="single", inputs={"target": "x"})
    wait_for((control / "ready").is_file)
    heartbeat = control / "beat.txt"
    wait_for(lambda: heartbeat.is_file() and heartbeat.read_text())

    supervisor.cancel(result["runId"])

    # `taskkill /T` walks parent-pid links, so the agent's stand-in dies with
    # the worker. That is the only reason the engine runs in a child at all.
    settled = heartbeat.read_text()
    time.sleep(0.8)
    assert heartbeat.read_text() == settled


@pytest.mark.skipif(os.name == "nt", reason="the relayed-pid pass is the POSIX half")
def test_cancel_reaches_relayed_step_children(supervisors, tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setenv("FLOWMCP_TEST_CONTROL", str(control))
    supervisor = supervisors(
        cwd=fake_worker(tmp_path, "treeworker", FAKE_TREE), worker_module="treeworker"
    )
    supervisor.start()
    result = supervisor.launch(flow="single", inputs={"target": "x"})
    wait_for((control / "ready").is_file)
    agent_pid = int((control / "ready").read_text())

    supervisor.cancel(result["runId"])

    # Every step child gets its own session, so the worker's process group does
    # not contain it. Without the relayed-pid pass, "Stop" leaves it running.
    with pytest.raises(OSError):
        os.kill(agent_pid, 0)


# -- the merge table --------------------------------------------------------


def seed(run_root: Path, run_id: str, manifest: dict | None, record: dict | None):
    run_dir = run_root / "seeded" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (run_dir / schema.MANIFEST_NAME).write_text(
            json.dumps({"run_id": run_id, "flow": "seeded", **manifest}),
            encoding="utf-8",
        )
    if record is not None:
        records = run_root / ".supervisor"
        records.mkdir(parents=True, exist_ok=True)
        (records / f"{run_id}.json").write_text(
            json.dumps({"runId": run_id, "runDir": str(run_dir), **record}),
            encoding="utf-8",
        )
    return run_dir


@pytest.mark.parametrize(
    "process_state, engine_status, expected",
    [
        ("running", "running", "running"),
        ("running", None, "running"),
        ("running", "ok", "ok"),
        ("running", "failed", "failed"),
        ("exited", "ok", "ok"),
        ("exited", "failed", "failed"),
        ("exited", "running", "crashed"),
        ("exited", None, "failed"),
        ("failed-to-start", None, "failed"),
        ("cancelled", "running", "cancelled"),
        ("cancelled", "ok", "cancelled"),
        ("detached", "ok", "ok"),
        ("detached", "failed", "failed"),
        ("detached", "running", "unknown"),
        (None, "ok", "ok"),
        (None, "running", "unknown"),
        (None, None, "unknown"),
    ],
)
def test_the_merge_table(process_state, engine_status, expected):
    assert merge_status(process_state, engine_status) == expected


def test_a_worker_that_died_without_a_terminal_checkpoint_crashed(
    supervisors, run_root
):
    seed(
        run_root,
        "r-crashed",
        {"status": "running", "steps": [], "loops": []},
        {"state": "exited", "exitCode": 1},
    )
    supervisor = supervisors()

    status = supervisor.status("r-crashed")

    assert status["status"] == "crashed"
    assert status["engineStatus"] == "running"
    assert status["exitCode"] == 1


def test_a_terminal_manifest_outranks_a_stale_running_record(supervisors, run_root):
    seed(
        run_root,
        "r-left-running",
        {"status": "ok", "steps": [], "loops": []},
        {"state": "running", "exitCode": None},
    )
    supervisor = supervisors()

    status = supervisor.status("r-left-running")

    assert status["status"] == "ok"
    assert status["processState"] == "running"


def test_a_run_with_no_record_is_never_reported_as_running(supervisors, run_root):
    seed(run_root, "r-foreign", {"status": "running", "steps": [], "loops": []}, None)
    supervisor = supervisors()

    status = supervisor.status("r-foreign")

    # A live foreign run and an abandoned one are indistinguishable, and the
    # pid a record might carry proves nothing -- pids are reused.
    assert status["status"] == "unknown"
    assert status["processState"] is None


def test_an_unknown_run_is_reported_as_such(supervisors):
    supervisor = supervisors()

    with pytest.raises(SupervisorError, match="unknown run"):
        supervisor.status("20260915T171233Z-0000")


def test_startup_reconciliation_detaches_records_from_a_previous_session(
    supervisors, run_root
):
    seed(
        run_root,
        "r-abandoned",
        {"status": "running", "steps": [], "loops": []},
        {"state": "running", "pid": 999999, "exitCode": None},
    )
    seed(
        run_root,
        "r-finished",
        {"status": "ok", "steps": [], "loops": []},
        {"state": "running", "pid": 999998, "exitCode": None},
    )
    supervisor = supervisors()

    supervisor.start()

    assert supervisor.status("r-abandoned")["status"] == "unknown"
    assert supervisor.status("r-abandoned")["processState"] == "detached"
    assert supervisor.status("r-finished")["status"] == "ok"


def test_a_worker_that_dies_before_its_first_checkpoint_reports_failed(
    supervisors, tmp_path
):
    supervisor = supervisors(
        cwd=fake_worker(tmp_path, "doomed", FAKE_DOOMED), worker_module="doomed"
    )
    supervisor.start()

    result = supervisor.launch(flow="single", inputs={"target": "x"})
    assert wait_terminal(supervisor, result["runId"]) == "failed"

    status = supervisor.status(result["runId"])
    assert status["engineStatus"] is None
    assert status["exitCode"] == 3
    # With no manifest, the worker's own account of the failure is the only one.
    assert "could not start the run" in status["error"]


# -- update notifications ---------------------------------------------------


@pytest.fixture
def ticker(supervisors, tmp_path, monkeypatch, events):
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setenv("FLOWMCP_TEST_CONTROL", str(control))
    supervisor = supervisors(
        cwd=fake_worker(tmp_path, "ticker", FAKE_TICKER), worker_module="ticker"
    )
    supervisor.start()
    result = supervisor.launch(flow="single", inputs={"target": "x"})
    wait_for((control / "started").is_file)

    def tick(n: int) -> None:
        (control / f"tick-{n}").write_text("1")
        wait_for((control / f"done-{n}").is_file)
        # The supervisor reacts to the frame that follows the rewrite; give the
        # reader thread a moment to deliver it.
        time.sleep(0.3)

    return supervisor, result, control, tick


def updates(events, uri: str) -> int:
    return sum(
        1
        for event in events.collected
        if event.get("type") == "resource_updated" and event.get("uri") == uri
    )


def test_an_unsubscribed_client_is_told_nothing(ticker, events):
    supervisor, result, _, tick = ticker
    uri = schema.manifest_uri(result["runId"])

    tick(1)

    assert updates(events, uri) == 0
    # Supplementary prose still flows; it is not the state channel.
    assert any(event.get("type") == "message" for event in events.collected)


def test_a_checkpoint_produces_exactly_one_update_for_a_subscriber(ticker, events):
    supervisor, result, _, tick = ticker
    uri = schema.manifest_uri(result["runId"])
    supervisor.subscribe(uri)
    before = updates(events, uri)

    tick(1)

    assert updates(events, uri) == before + 1


def test_unsubscribing_stops_the_updates(ticker, events):
    supervisor, result, _, tick = ticker
    uri = schema.manifest_uri(result["runId"])
    supervisor.subscribe(uri)
    tick(1)
    supervisor.unsubscribe(uri)
    after_subscribed = updates(events, uri)

    tick(2)

    assert updates(events, uri) == after_subscribed


def test_only_a_manifest_can_be_subscribed_to(supervisors):
    supervisor = supervisors()

    with pytest.raises(SupervisorError):
        supervisor.subscribe("run://20260915T171233Z-a1b4/reports/junit/results.xml")


# -- resources --------------------------------------------------------------


def test_a_launched_runs_artifacts_are_addressable_and_confined(supervisors):
    supervisor = supervisors()
    supervisor.start()
    result = supervisor.launch(flow="single", inputs={"target": "x"})
    assert wait_terminal(supervisor, result["runId"]) == "ok"

    served = supervisor.read_resource(schema.manifest_uri(result["runId"]))

    assert served["mimeType"] == "application/json"
    assert json.loads(served["text"])["run_id"] == result["runId"]
    listed = {entry["uri"] for entry in supervisor.list_resources()}
    assert schema.manifest_uri(result["runId"]) in listed
    with pytest.raises(resources.PathRefused):
        supervisor.read_resource(
            f"run://{result['runId']}/../../../secrets.txt"
        )
