# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""End-to-end behaviour of the loop driver, against a fake agent CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from runner.engine import Engine
from runner.flow import Flow, bind_inputs, validate_refs
from runner.toolreg import ToolRegistry


def build(tmp_path: Path, registry_file: Path, flow_text: str, inputs=()) -> Engine:
    flow_path = tmp_path / "flow.yaml"
    flow_path.write_text(flow_text, encoding="utf-8")
    registry = ToolRegistry.load(registry_file)
    flow = Flow.load(flow_path)
    validate_refs(flow, registry.vars)
    return Engine(
        flow,
        registry,
        bind_inputs(flow, list(inputs)),
        run_dir=tmp_path / "run",
        log=lambda _message: None,
    )


def review_loop(agent_script: Path, sequence: str, max_iterations: int = 3) -> str:
    """A generate/review cycle whose reviewer returns `sequence` critical counts."""
    return f"""
version: 1
name: review-loop
steps:
  - id: cycle
    loop:
      max_iterations: {max_iterations}
      until: "${{steps.review.outputs.critical_count}} == 0"
      on_exhausted: fail
      feedback_from: "${{steps.review.outputs.feedback}}"
    steps:
      - id: generate
        tool: agent
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--payload", '{{"kernel_path": "{agent_script.as_posix()}"}}']
        result_file: "${{loop.attempt_dir}}/generate.json"
        result_schema: {{required: [kernel_path]}}
        outputs:
          kernel_path: {{json_file: result, path: "$.kernel_path", type: path}}
      - id: review
        tool: agent
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--counter", "${{run.dir}}/counter.txt", "--sequence", "{sequence}"]
        result_file: "${{loop.attempt_dir}}/review.json"
        result_schema: {{required: [verdict, critical_count, feedback]}}
        outputs:
          verdict: {{json_file: result, path: "$.verdict"}}
          critical_count: {{json_file: result, path: "$.critical_count", type: int}}
          feedback: {{json_file: result, path: "$.feedback"}}
"""


def test_loop_repeats_until_the_reviewer_reports_no_critical_issues(
    tmp_path, registry_file, agent_script
):
    engine = build(tmp_path, registry_file, review_loop(agent_script, "2,1,0"))
    report = engine.run()

    assert report.status == "ok"
    assert report.loops[0].iterations == 3
    assert report.loops[0].satisfied is True
    # Each failing round contributes one feedback section; the clean round does not.
    feedback = (engine.run_dir / "feedback.md").read_text()
    assert "## iteration 0" in feedback and "## iteration 1" in feedback
    assert "## iteration 2" not in feedback
    assert "round 0: 2 critical issue(s)" in feedback


def test_loop_that_never_converges_fails_the_run(tmp_path, registry_file, agent_script):
    engine = build(
        tmp_path, registry_file, review_loop(agent_script, "3", max_iterations=2)
    )
    report = engine.run()

    # The failure that matters: a repair loop must not report success just because it
    # ran out of iterations.
    assert report.status == "failed"
    assert "without satisfying" in (report.error or "")
    assert report.loops[0].satisfied is False
    assert report.loops[0].iterations == 2


def test_first_iteration_renders_previous_attempt_as_empty(
    tmp_path, registry_file, agent_script
):
    flow = f"""
version: 1
name: prev
steps:
  - id: cycle
    loop:
      max_iterations: 1
      until: "1 == 1"
    steps:
      - id: generate
        tool: agent
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--payload", '{{"kernel_path": "{agent_script.as_posix()}"}}']
        stdin: "previous kernel: [${{loop.previous.generate.outputs.kernel_path}}]"
        result_file: "${{loop.attempt_dir}}/generate.json"
        result_schema: {{required: [kernel_path]}}
        outputs:
          kernel_path: {{json_file: result, path: "$.kernel_path", type: path}}
"""
    engine = build(tmp_path, registry_file, flow)
    assert engine.run().status == "ok"
    rendered = (
        engine.run_dir / "cycle" / "iter-00" / "generate" / "stdin.txt"
    ).read_text()
    assert rendered.strip() == "previous kernel: []"


def test_missing_result_file_is_retried_then_fails(
    tmp_path, registry_file, agent_script
):
    flow = f"""
version: 1
name: contract
steps:
  - id: generate
    tool: agent
    args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}", "--skip-result"]
    result_file: "${{run.dir}}/generate.json"
    result_schema: {{required: [kernel_path]}}
    retries: 1
"""
    engine = build(tmp_path, registry_file, flow)
    report = engine.run()

    assert report.status == "failed"
    assert "result file" in (report.error or "")
    # One initial attempt plus one retry, each with its own evidence directory.
    attempts = [record for record in report.steps if record.id == "generate"]
    assert len(attempts) == 2
    assert (engine.run_dir / "generate.retry-1").is_dir()


def test_result_file_missing_a_required_key_fails_the_step(
    tmp_path, registry_file, agent_script
):
    flow = f"""
version: 1
name: contract
steps:
  - id: generate
    tool: agent
    args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
           "--payload", '{{"entry_point": "k"}}']
    result_file: "${{run.dir}}/generate.json"
    result_schema: {{required: [kernel_path, entry_point]}}
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "failed"
    assert "missing required key(s): kernel_path" in (report.error or "")


def test_assertion_failure_stops_the_run(tmp_path, registry_file, agent_script):
    flow = f"""
version: 1
name: asserts
steps:
  - id: review
    tool: agent
    args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
           "--payload", '{{"verdict": "pass", "critical_count": 4}}']
    result_file: "${{run.dir}}/review.json"
    result_schema: {{required: [verdict, critical_count]}}
    outputs:
      verdict: {{json_file: result, path: "$.verdict"}}
      critical_count: {{json_file: result, path: "$.critical_count", type: int}}
    assert:
      - that: "${{steps.review.outputs.critical_count}} == 0 or ${{steps.review.outputs.verdict}} == changes_required"
        message: "review listed critical issues but returned verdict 'pass'"
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "failed"
    assert "returned verdict 'pass'" in (report.error or "")


def test_when_false_skips_the_step(tmp_path, registry_file, agent_script):
    flow = f"""
version: 1
name: gated
steps:
  - id: probe
    tool: agent
    args: ["{agent_script.as_posix()}"]
  - id: diagnose
    tool: agent
    when: "${{steps.probe.outputs.exit_code}} != 0"
    args: ["{agent_script.as_posix()}"]
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "ok"
    assert [record.status for record in report.steps] == ["ok", "skipped"]


def test_unexpected_exit_code_fails_and_is_recorded(
    tmp_path, registry_file, agent_script
):
    flow = f"""
version: 1
name: exits
steps:
  - id: run
    tool: agent
    args: ["{agent_script.as_posix()}", "--exit-code", "3"]
"""
    engine = build(tmp_path, registry_file, flow)
    report = engine.run()
    assert report.status == "failed"
    assert "exited 3" in (report.error or "")
    manifest = json.loads((engine.run_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["steps"][0]["id"] == "run"


def test_run_manifest_records_inputs_and_outputs(tmp_path, registry_file, agent_script):
    flow = f"""
version: 1
name: manifest
inputs:
  arch: {{default: gfx1151}}
steps:
  - id: run
    tool: agent
    args: ["{agent_script.as_posix()}"]
    outputs:
      round: {{json: 0, path: "$.round", type: int}}
"""
    engine = build(tmp_path, registry_file, flow)
    report = engine.run()
    assert report.status == "ok"

    manifest = json.loads((engine.run_dir / "run.json").read_text())
    assert manifest["inputs"] == {"arch": "gfx1151"}
    assert manifest["steps"][0]["outputs"]["round"] == 0
    assert json.loads((engine.run_dir / "inputs.json").read_text()) == {
        "arch": "gfx1151"
    }


def test_dry_run_renders_prompts_without_launching(
    tmp_path, registry_file, agent_script
):
    prompt = tmp_path / "p.md"
    prompt.write_text(
        "arch=${inputs.arch} iteration=${loop.iteration}", encoding="utf-8"
    )
    flow = f"""
version: 1
name: plan
inputs:
  arch: {{default: gfx1151}}
steps:
  - id: cycle
    loop:
      max_iterations: 2
      until: "1 == 1"
    steps:
      - id: generate
        tool: agent
        args: ["{agent_script.as_posix()}"]
        prompt_file: p.md
"""
    engine = build(tmp_path, registry_file, flow)
    planned = engine.plan()
    assert planned[0]["stdin"] == "arch=gfx1151 iteration=0"
    assert not (tmp_path / "run").exists()
