# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""End-to-end behaviour of the loop driver, against a fake agent CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from runner.engine import Engine
from runner.errors import ConfigError
from runner.flow import Flow, bind_inputs, validate_refs
from runner.outputs import LazyText
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
    # The message has to say what was actually measured, not just restate the goal --
    # "3 == 0" is the difference between a diagnosis and an unexplained verdict.
    assert "(3 == 0)" in (report.error or "")
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


def test_step_failure_without_a_policy_stops_the_run(
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
"""
    engine = build(tmp_path, registry_file, flow)
    report = engine.run()

    assert report.status == "failed"
    assert "result file" in (report.error or "")
    # Exactly one attempt: a step is never silently re-run in place. Re-asking a
    # question whose inputs have not changed just repeats the failure at full cost.
    assert len([record for record in report.steps if record.id == "generate"]) == 1


def test_loop_retries_the_whole_iteration_when_a_step_fails(
    tmp_path, registry_file, agent_script
):
    """A contract violation abandons the iteration; the next one starts from the top."""
    flow = f"""
version: 1
name: iteration-retry
steps:
  - id: cycle
    loop:
      max_iterations: 3
      until: "${{steps.review.outputs.critical_count}} == 0"
      on_exhausted: fail
      on_step_failure: retry
    steps:
      - id: generate
        tool: agent
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--counter", "${{run.dir}}/gen_counter.txt",
               "--payload", '{{"kernel_path": "{agent_script.as_posix()}"}}']
        result_file: "${{loop.attempt_dir}}/generate.json"
        result_schema: {{required: [kernel_path]}}
        outputs:
          kernel_path: {{json_file: result, path: "$.kernel_path", type: path}}
      - id: review
        tool: agent
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--counter", "${{run.dir}}/counter.txt", "--sequence", "0",
               "--skip-result-on", "0"]
        result_file: "${{loop.attempt_dir}}/review.json"
        result_schema: {{required: [critical_count]}}
        outputs:
          critical_count: {{json_file: result, path: "$.critical_count", type: int}}
"""
    engine = build(tmp_path, registry_file, flow)
    report = engine.run()

    # Iteration 1: review writes nothing -> iteration abandoned, no `until` check.
    # Iteration 2: review answers 0 -> loop satisfied.
    assert report.status == "ok"
    assert report.loops[0].iterations == 2
    generate_runs = [r for r in report.steps if r.id == "generate" and r.status == "ok"]
    assert (
        len(generate_runs) == 2
    ), "the generator must re-run, not just the failed step"
    assert "did not complete" in (engine.run_dir / "feedback.md").read_text()


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


def test_json_stdout_gets_a_readable_sibling(tmp_path, registry_file, agent_script):
    """An agent envelope is one long line; evidence has to be readable to be used."""
    flow = f"""
version: 1
name: pretty
steps:
  - id: run
    tool: agent
    args: ["{agent_script.as_posix()}"]
"""
    engine = build(tmp_path, registry_file, flow)
    assert engine.run().status == "ok"

    raw = engine.run_dir / "run" / "stdout.log"
    pretty = engine.run_dir / "run" / "stdout.pretty.json"
    assert raw.read_text().count("\n") <= 1, "the raw log must stay byte-exact"
    assert json.loads(pretty.read_text())["session_id"] == "session-123"
    assert pretty.read_text().count("\n") > 1


def test_non_json_stdout_gets_no_sibling(tmp_path, registry_file):
    script = tmp_path / "plain.py"
    script.write_text("print('just prose, not json')", encoding="utf-8")
    flow = f"""
version: 1
name: plain
steps:
  - id: run
    tool: agent
    args: ["{script.as_posix()}"]
"""
    engine = build(tmp_path, registry_file, flow)
    assert engine.run().status == "ok"
    assert not (engine.run_dir / "run" / "stdout.pretty.json").exists()


# -- iteration isolation ----------------------------------------------------


def stale_output_loop(agent_script: Path, until: str) -> str:
    """`produce` runs in the first iteration only. The second iteration must not be able
    to read what it produced through `${steps...}`."""
    return f"""
version: 1
name: stale
steps:
  - id: cycle
    loop:
      max_iterations: 2
      until: "{until}"
      on_exhausted: continue
    steps:
      - id: probe
        tool: agent
        args: ["{agent_script.as_posix()}", "--counter", "${{run.dir}}/counter.txt"]
        outputs:
          round: {{json: 0, path: "$.round", type: int}}
      - id: produce
        tool: agent
        when: "${{steps.probe.outputs.round}} == 0"
        args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
               "--payload", '{{"value": 7}}']
        result_file: "${{loop.attempt_dir}}/produce.json"
        outputs:
          value: {{json_file: result, path: "$.value", type: int}}
"""


def test_skipped_step_does_not_expose_the_previous_iterations_outputs(
    tmp_path, registry_file, agent_script
):
    """The exit condition used to be satisfiable by a value the current iteration never
    produced: a skipped step inherited its own earlier outputs."""
    flow = stale_output_loop(
        agent_script,
        "${steps.probe.outputs.round} == 1 and ${steps.produce.outputs.value} == 7",
    )
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "failed"
    assert "declares no output 'value'" in (report.error or "")
    statuses = [(record.id, record.iteration, record.status) for record in report.steps]
    assert ("produce", 1, "skipped") in statuses


def test_previous_iterations_outputs_stay_available_explicitly(
    tmp_path, registry_file, agent_script
):
    """`${loop.previous...}` is the way to read an earlier iteration -- and it still is."""
    flow = stale_output_loop(
        agent_script,
        "${steps.probe.outputs.round} == 1 and "
        "${loop.previous.produce.outputs.value} == 7",
    )
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "ok"
    assert report.loops[0].satisfied is True
    assert report.loops[0].iterations == 2


# -- failure evidence -------------------------------------------------------


def test_failed_step_keeps_its_exit_code_duration_and_argv(
    tmp_path, registry_file, agent_script
):
    """The record an operator reads on failure used to be a fresh empty one: exit_code
    None, timed_out False, duration 0.0, argv []."""
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
    record = report.steps[0]
    assert record.status == "failed"
    assert record.exit_code == 3
    assert record.argv[-1] == "3"
    assert record.duration_s > 0
    assert record.error and "exited 3" in record.error
    manifest = json.loads((engine.run_dir / "run.json").read_text())
    assert manifest["steps"][0]["exit_code"] == 3
    assert manifest["steps"][0]["argv"]


def test_timed_out_step_is_recorded_as_a_timeout(tmp_path, registry_file, agent_script):
    flow = f"""
version: 1
name: slow
steps:
  - id: run
    tool: agent
    timeout: 1
    args: ["{agent_script.as_posix()}", "--sleep", "30"]
"""
    engine = build(tmp_path, registry_file, flow)
    report = engine.run()
    record = report.steps[0]
    assert record.status == "timed_out"
    assert record.timed_out is True
    assert record.duration_s > 0
    manifest = json.loads((engine.run_dir / "run.json").read_text())
    assert manifest["steps"][0]["status"] == "timed_out"


def test_manifest_is_readable_while_a_later_step_is_still_running(
    tmp_path, registry_file, agent_script
):
    """The second step reads run.json and fails unless the first is already recorded."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, sys\n"
        "manifest = json.loads(open(sys.argv[1], encoding='utf-8').read())\n"
        "done = [s['id'] for s in manifest['steps'] if s['status'] == 'ok']\n"
        "sys.exit(0 if 'first' in done else 1)\n",
        encoding="utf-8",
    )
    flow = f"""
version: 1
name: checkpoint
steps:
  - id: first
    tool: agent
    args: ["{agent_script.as_posix()}"]
  - id: second
    tool: agent
    args: ["{probe.as_posix()}", "${{run.dir}}/run.json"]
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "ok"


def test_tolerated_failure_still_exposes_its_process_metadata(
    tmp_path, registry_file, agent_script
):
    """`continue_on_error` exists so a later step can branch on what happened."""
    flow = f"""
version: 1
name: tolerated
steps:
  - id: flaky
    tool: agent
    continue_on_error: true
    args: ["{agent_script.as_posix()}", "--exit-code", "3"]
  - id: after
    tool: agent
    when: "${{steps.flaky.outputs.exit_code}} == 3"
    args: ["{agent_script.as_posix()}"]
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "ok"
    assert [record.status for record in report.steps] == ["failed", "ok"]
    assert report.steps[0].exit_code == 3


# -- run directories --------------------------------------------------------


def test_a_second_run_will_not_overwrite_an_occupied_run_directory(
    tmp_path, registry_file, agent_script
):
    flow = f"""
version: 1
name: once
steps:
  - id: run
    tool: agent
    args: ["{agent_script.as_posix()}"]
"""
    first = build(tmp_path, registry_file, flow)
    assert first.run().status == "ok"
    evidence = (first.run_dir / "run.json").read_bytes()

    with pytest.raises(ConfigError, match="already contains a run"):
        build(tmp_path, registry_file, flow).run()
    assert (first.run_dir / "run.json").read_bytes() == evidence


def test_generated_run_directories_do_not_collide_within_one_second(
    tmp_path, registry_file
):
    """Second-resolution run ids aimed two runs started in the same second at one
    directory, and `exist_ok=True` let the later one write over the earlier."""
    flow_path = tmp_path / "flow.yaml"
    flow_path.write_text(
        "version: 1\nname: t\nsteps:\n  - id: run\n    tool: agent\n", encoding="utf-8"
    )
    registry = ToolRegistry.load(registry_file)
    flow = Flow.load(flow_path)
    directories = {
        Engine(
            flow, registry, {}, run_root=tmp_path / "runs", log=lambda _m: None
        ).run_dir
        for _ in range(8)
    }
    assert len(directories) == 8


# -- log handling -----------------------------------------------------------


def test_large_logs_are_referenced_by_path_not_retained(tmp_path, registry_file):
    script = tmp_path / "loud.py"
    script.write_text(
        "import sys\nsys.stdout.write('x' * (2 * 1024 * 1024))\n", encoding="utf-8"
    )
    flow = f"""
version: 1
name: loud
steps:
  - id: run
    tool: agent
    args: ["{script.as_posix()}"]
"""
    engine = build(tmp_path, registry_file, flow)
    assert engine.run().status == "ok"

    held = engine.completed["run"]["stdout"]
    assert isinstance(held, LazyText)
    assert held.path == engine.run_dir / "run" / "stdout.log"
    # Still readable on demand -- the text is a view of the evidence, not a copy of it.
    assert len(str(held)) == 2 * 1024 * 1024
    recorded = json.loads((engine.run_dir / "run" / "result.json").read_text())
    assert "stdout" not in recorded
    assert recorded["stdout_path"].endswith("stdout.log")


def test_dry_run_reports_the_tool_wiring_it_would_launch_with(tmp_path, agent_script):
    """`--dry-run` resolves the same invocation the run does; it used to skip tool env."""
    registry_path = tmp_path / "tools-env.yaml"
    registry_path.write_text(
        "version: 1\n"
        "tools:\n"
        "  agent:\n"
        f"    exe: {Path(sys.executable).as_posix()}\n"
        "    env:\n"
        "      AGENT_MODE: batch\n",
        encoding="utf-8",
    )
    flow = f"""
version: 1
name: wired
steps:
  - id: run
    tool: agent
    args: ["{agent_script.as_posix()}"]
"""
    planned = build(tmp_path, registry_path, flow).plan()
    assert planned[0]["env"]["AGENT_MODE"] == "batch"
    assert planned[0]["argv"][0] == str(Path(sys.executable))


# -- review acceptance checks -----------------------------------------------


def test_review_that_passes_with_issues_listed_is_rejected(
    tmp_path, registry_file, agent_script
):
    """The accepted-anyway result from the review: verdict pass, critical_count 0, and a
    critical issue listed. The count is now measured from the list, so the two disagree.
    """
    payload = (
        '{"verdict": "pass", "critical_count": 0, '
        '"critical_issues": [{"title": "out of bounds write"}], '
        '"feedback": "fix the out of bounds write"}'
    )
    flow = f"""
version: 1
name: review-consistency
steps:
  - id: review
    tool: agent
    args: ["{agent_script.as_posix()}", "--result", "${{step.result_file}}",
           "--payload", '{payload}']
    result_file: "${{run.dir}}/review.json"
    result_schema: {{required: [verdict, critical_count, critical_issues, feedback]}}
    outputs:
      verdict: {{json_file: result, path: "$.verdict"}}
      critical_count: {{json_file: result, path: "$.critical_issues", type: count}}
      reported_critical_count: {{json_file: result, path: "$.critical_count", type: int}}
    assert:
      - that: "${{steps.review.outputs.reported_critical_count}} == ${{steps.review.outputs.critical_count}}"
        message: "critical_count disagrees with the number of entries in critical_issues"
      - that: "${{steps.review.outputs.critical_count}} == 0 or ${{steps.review.outputs.verdict}} == changes_required"
        message: "review listed critical issues but returned verdict 'pass'"
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "failed"
    assert "disagrees" in (report.error or "")


def test_kernel_reported_somewhere_else_fails_the_identity_check(
    tmp_path, registry_file
):
    """The reviewer reads the reported path while the hash watches the designated one.
    A kernel written anywhere else is reviewed but never integrity-checked."""
    writer = tmp_path / "writer.py"
    writer.write_text(
        "import json, pathlib, sys\n"
        "target = pathlib.Path(sys.argv[1])\n"
        "target.mkdir(parents=True, exist_ok=True)\n"
        "(target / 'kernel.hip').write_text('// placeholder')\n"
        "(target / 'actual.hip').write_text('// the real one')\n"
        "pathlib.Path(sys.argv[2]).write_text(\n"
        "    json.dumps({'kernel_path': str(target / 'actual.hip')}))\n",
        encoding="utf-8",
    )
    flow = f"""
version: 1
name: identity
vars:
  kernel_dir: "${{run.dir}}/kernel"
steps:
  - id: generate
    tool: agent
    args: ["{writer.as_posix()}", "${{vars.kernel_dir}}", "${{step.result_file}}"]
    result_file: "${{run.dir}}/generate.json"
    result_schema: {{required: [kernel_path]}}
    outputs:
      kernel_path: {{json_file: result, path: "$.kernel_path", type: path}}
      expected_kernel: {{file: "${{vars.kernel_dir}}/kernel.hip", type: path}}
    assert:
      - that: "${{steps.generate.outputs.kernel_path}} == ${{steps.generate.outputs.expected_kernel}}"
        message: "the kernel must be written to the designated path"
"""
    report = build(tmp_path, registry_file, flow).run()
    assert report.status == "failed"
    assert "designated path" in (report.error or "")
