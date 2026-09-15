# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The child that runs the engine, driven the way the supervisor drives it.

Hermetic: the only executable launched is this Python, once as the worker and
once more as the fixture agent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def one_step_flow(agent_script: Path, sleep: float = 0.0) -> str:
    """A single agent step whose declared output is a file it really wrote."""
    return f"""
version: 1
name: single
description: One step, one declared path output.
inputs:
  target:
    type: string
    default: unused
steps:
  - id: emit
    tool: agent
    args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
           "--sleep", "{sleep}",
           "--payload", '{{"bundle": "{agent_script.as_posix()}"}}']
    stdin: "produce something"
    result_file: "${{run.dir}}/emit.json"
    result_schema: {{required: [bundle]}}
    outputs:
      bundle: {{json_file: result, path: "$.bundle", type: path}}
"""


def spec_for(tmp_path: Path, registry_file: Path, flow_text: str, run_id: str) -> dict:
    flow_path = tmp_path / "flow.yaml"
    flow_path.write_text(flow_text, encoding="utf-8")
    return {
        "flow": str(flow_path),
        "toolsPath": str(registry_file),
        "profile": None,
        "inputs": {"target": "unused"},
        "runId": run_id,
        "runDir": str(tmp_path / "runs" / "single" / run_id),
        "maxIterations": None,
    }


def start(spec: dict) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-m", "flowmcp.worker"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert process.stdin is not None
    process.stdin.write(json.dumps(spec))
    process.stdin.close()
    return process


def drain(process: subprocess.Popen) -> tuple[int, list[dict], str]:
    out, err = process.communicate(timeout=120)
    frames = [json.loads(line) for line in out.splitlines() if line.strip()]
    return process.returncode, frames, err


def test_the_worker_produces_a_real_run(tmp_path, registry_file, agent_script):
    spec = spec_for(
        tmp_path, registry_file, one_step_flow(agent_script), "20260915T171233Z-a1b4"
    )

    code, frames, _ = drain(start(spec))

    assert code == 0
    manifest = json.loads(Path(spec["runDir"], "run.json").read_text())
    assert manifest["status"] == "ok"
    assert [frame["t"] for frame in frames][-1] == "done"
    assert frames[-1]["status"] == "ok"
    assert any(frame["t"] == "log" for frame in frames)


def test_the_supplied_run_id_is_the_only_identity(
    tmp_path, registry_file, agent_script
):
    spec = spec_for(
        tmp_path, registry_file, one_step_flow(agent_script), "20260915T171233Z-beef"
    )

    code, _, _ = drain(start(spec))

    assert code == 0
    manifest = json.loads(Path(spec["runDir"], "run.json").read_text())
    assert manifest["run_id"] == "20260915T171233Z-beef"
    assert Path(spec["runDir"]).name == "20260915T171233Z-beef"


def test_a_step_child_is_announced_while_it_is_still_running(
    tmp_path, registry_file, agent_script
):
    # A pid observed after `launch()` returns is a dead pid, and cancellation
    # built on one is a no-op. The frame has to arrive while the child lives.
    spec = spec_for(
        tmp_path,
        registry_file,
        one_step_flow(agent_script, sleep=3.0),
        "20260915T171233Z-c0de",
    )
    process = start(spec)
    assert process.stdout is not None

    announced = None
    while True:
        line = process.stdout.readline()
        if not line:
            break
        frame = json.loads(line)
        if frame["t"] == "pid":
            announced = frame
            # The run is still in flight at the moment the pid reaches us.
            assert process.poll() is None
            break
        assert frame["t"] != "done", "the pid frame never arrived before the run ended"

    code, frames, _ = drain(process)

    assert announced is not None
    assert announced["step"] == "emit"
    assert isinstance(announced["pid"], int) and announced["pid"] > 0
    assert code == 0
    assert frames[-1]["t"] == "done"


def test_a_run_directory_that_is_already_occupied_fails_before_any_manifest(
    tmp_path, registry_file, agent_script
):
    # `run()` re-runs preflight and run-directory creation before its first
    # checkpoint, so this failure leaves no manifest at all -- the case the
    # supervisor reconciles from the exit code and this process's stderr.
    spec = spec_for(
        tmp_path, registry_file, one_step_flow(agent_script), "20260915T171233Z-dead"
    )
    occupied = Path(spec["runDir"])
    occupied.mkdir(parents=True)
    (occupied / "previous-evidence.txt").write_text("not ours", encoding="utf-8")

    code, frames, err = drain(start(spec))

    assert code != 0
    assert not (occupied / "run.json").exists()
    assert "already contains a run" in err
    assert frames[-1]["t"] == "done"
    assert frames[-1]["status"] == "failed"
    assert "already contains a run" in frames[-1]["error"]
