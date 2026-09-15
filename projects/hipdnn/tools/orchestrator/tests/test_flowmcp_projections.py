# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The derivations that let every consumer stay ignorant of any particular flow.

Every fixture here is deliberately unlike the flows that ship today: several
loop groups of different sizes, steps outside every group, non-agent tool steps,
and outputs of every declared type. If a derivation only works for a two-step
review cycle, it fails here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowmcp import projections, schema
from runner.flow import Flow

#: An ungrouped preparation step, a two-step group with a budget of three, and a
#: four-step group with a budget of two. Denominator: 1 + 2*3 + 4*2 = 15.
MULTI_GROUP = """
version: 1
name: multi
description: Three shapes of node in one flow.
inputs:
  target:
    type: string
    default: here
steps:
  - id: prepare
    tool: shell
    args: ["--prepare"]
  - id: alpha
    loop:
      max_iterations: 3
      until: "${steps.draft.outputs.open_items} == 0"
      on_exhausted: fail
      on_step_failure: retry
    steps:
      - id: draft
        tool: writer
        stdin: "draft it"
        result_file: "${loop.attempt_dir}/draft.json"
        outputs:
          open_items: {json_file: result, path: "$.open_items", type: int}
      - id: inspect
        tool: writer
        stdin: "inspect it"
        result_file: "${loop.attempt_dir}/inspect.json"
        outputs:
          notes: {json_file: result, path: "$.notes"}
  - id: beta
    loop:
      max_iterations: 2
      until: "${steps.assemble.outputs.count} == 0"
      on_exhausted: continue
    steps:
      - id: assemble
        tool: shell
        args: ["--assemble"]
        outputs:
          count: {json: stdout, path: "$.items", type: count}
      - id: compile
        tool: shell
        args: ["--compile"]
      - id: check
        tool: shell
        args: ["--check"]
      - id: publish
        tool: shell
        args: ["--publish"]
"""


def load(tmp_path: Path, text: str) -> Flow:
    path = tmp_path / "flow.yaml"
    path.write_text(text, encoding="utf-8")
    return Flow.load(path)


@pytest.fixture
def multi(tmp_path):
    return load(tmp_path, MULTI_GROUP)


# -- step kind --------------------------------------------------------------


def test_step_kind_reads_the_declaration_not_the_tool_name(tmp_path):
    # Written so that a tool-name comparison cannot pass: the step carrying the
    # prompt/result contract is driven by a tool named after a build system, and
    # the argv-and-exit-code step is driven by a tool named after an agent CLI.
    flow = load(
        tmp_path,
        """
version: 1
name: names-mean-nothing
steps:
  - id: first
    tool: cmake
    stdin: "do the thing"
    result_file: "${run.dir}/first.json"
  - id: second
    tool: claude
    args: ["--build"]
""",
    )
    kinds = {
        step.id: projections.step_kind(step) for step, _ in projections.walk_steps(flow)
    }

    assert kinds == {"first": "agent", "second": "tool"}


def test_a_prompt_without_a_result_contract_is_not_an_agent_step(tmp_path):
    flow = load(
        tmp_path,
        """
version: 1
name: half
steps:
  - id: only
    tool: anything
    stdin: "printed, but nothing structured comes back"
""",
    )
    step = next(iter(flow.all_steps()))

    assert projections.step_kind(step) == "tool"


# -- flow declarations ------------------------------------------------------


def test_a_flow_summary_reports_every_node_in_order_grouped_or_not(multi):
    summary = projections.flow_summary_for(multi)

    assert [(step["id"], step["group"]) for step in summary["steps"]] == [
        ("prepare", None),
        ("draft", "alpha"),
        ("inspect", "alpha"),
        ("assemble", "beta"),
        ("compile", "beta"),
        ("check", "beta"),
        ("publish", "beta"),
    ]
    assert [(loop["id"], loop["maxIterations"]) for loop in summary["loops"]] == [
        ("alpha", 3),
        ("beta", 2),
    ]
    assert summary["stepCount"] == 7
    assert summary["loadError"] is None


def test_the_clampable_budget_is_the_largest_a_flow_declares(multi, tmp_path):
    assert projections.declared_budget(multi) == 3
    flat = tmp_path / "flat.yaml"
    flat.write_text(
        "version: 1\nname: flat\nsteps:\n  - id: only\n    tool: t\n", encoding="utf-8"
    )

    assert projections.declared_budget(Flow.load(flat)) is None


# -- progress ---------------------------------------------------------------


def test_the_denominator_sums_every_group_and_every_loose_step(multi):
    run_json = {
        "status": "running",
        "steps": [
            {"id": "prepare", "group": None, "iteration": None, "status": "ok"},
            {"id": "draft", "group": "alpha", "iteration": 0, "status": "ok"},
            {"id": "inspect", "group": "alpha", "iteration": 0, "status": "running"},
        ],
    }

    assert projections.progress_for(run_json, multi) == (2, 15)


def test_an_unreadable_flow_omits_the_denominator_rather_than_inventing_one():
    run_json = {
        "status": "running",
        "steps": [
            {"id": "a", "status": "ok"},
            {"id": "b", "status": "skipped"},
            {"id": "c", "status": "running"},
        ],
    }

    # A skipped step is settled; a running one is not.
    assert projections.progress_for(run_json, None) == (2, None)


def test_a_lowered_budget_lowers_the_denominator(multi):
    run_json = {"status": "running", "steps": []}

    assert projections.progress_for(run_json, multi, max_iterations=1) == (0, 7)


# -- the cursor -------------------------------------------------------------


def test_the_current_position_follows_the_flows_own_order(multi):
    run_json = {
        "status": "running",
        "loops": [],
        "steps": [
            {"id": "prepare", "group": None, "iteration": None, "status": "ok"},
            {"id": "draft", "group": "alpha", "iteration": 0, "status": "ok"},
            {"id": "inspect", "group": "alpha", "iteration": 0, "status": "ok"},
            {"id": "draft", "group": "alpha", "iteration": 1, "status": "running"},
        ],
    }

    assert projections.current_step_for(run_json, multi) == {
        "id": "inspect",
        "group": "alpha",
        "iteration": 1,
    }


def test_a_finished_group_is_stepped_past_even_with_budget_left(multi):
    run_json = {
        "status": "running",
        # The group's record says it will not iterate again, whatever its budget.
        "loops": [{"id": "alpha", "iterations": 1, "satisfied": True}],
        "steps": [
            {"id": "prepare", "group": None, "iteration": None, "status": "ok"},
            {"id": "draft", "group": "alpha", "iteration": 0, "status": "ok"},
            {"id": "inspect", "group": "alpha", "iteration": 0, "status": "ok"},
        ],
    }

    assert projections.current_step_for(run_json, multi) == {
        "id": "assemble",
        "group": "beta",
        "iteration": 0,
    }


def test_a_finished_run_has_no_current_position(multi):
    assert projections.current_step_for({"status": "ok", "steps": []}, multi) is None


# -- artifacts --------------------------------------------------------------

DECLARED_TYPES = """
version: 1
name: every-type
steps:
  - id: emit
    tool: writer
    stdin: "produce"
    result_file: "${run.dir}/emit.json"
    outputs:
      bundle:   {json_file: result, path: "$.bundle", type: path}
      label:    {json_file: result, path: "$.label", type: string}
      tally:    {json_file: result, path: "$.tally", type: int}
      ratio:    {json_file: result, path: "$.ratio", type: float}
      settled:  {json_file: result, path: "$.settled", type: bool}
      shape:    {json_file: result, path: "$.shape", type: json}
      opened:   {json_file: result, path: "$.opened", type: count}
"""


@pytest.fixture
def typed_run(tmp_path):
    flow = load(tmp_path, DECLARED_TYPES)
    run_dir = tmp_path / "runs" / "every-type" / "r-0001"
    step_dir = run_dir / "emit"
    step_dir.mkdir(parents=True)
    for name in ("cmd.txt", "stdout.log", "result.json"):
        (step_dir / name).write_text("evidence", encoding="utf-8")
    (run_dir / "bundle.kpack").write_text("payload", encoding="utf-8")
    # A string-typed output whose *value* is a real file inside the run. The
    # flow did not declare it a path, so it is not an output artifact.
    (run_dir / "looks-like-a-path.txt").write_text("decoy", encoding="utf-8")
    (run_dir / "notes.md").write_text("engine channel", encoding="utf-8")
    run_json = {
        "run_id": "r-0001",
        "status": "ok",
        "feedback_path": str(run_dir / "notes.md"),
        "loops": [],
        "steps": [
            {
                "id": "emit",
                "group": None,
                "iteration": None,
                "tool": "writer",
                "status": "ok",
                "dir": str(step_dir),
                "outputs": {
                    "bundle": str(run_dir / "bundle.kpack"),
                    "label": str(run_dir / "looks-like-a-path.txt"),
                    "tally": 3,
                    "ratio": 0.5,
                    "settled": True,
                    "shape": {"a": 1},
                    "opened": 2,
                },
            }
        ],
    }
    (run_dir / schema.MANIFEST_NAME).write_text(json.dumps(run_json), encoding="utf-8")
    return flow, run_dir, run_json


def test_only_outputs_the_flow_declared_as_paths_become_output_artifacts(typed_run):
    flow, run_dir, run_json = typed_run

    found = projections.artifacts_for(run_json, flow, run_id="r-0001", run_dir=run_dir)
    outputs = {item["outputName"] for item in found if item["source"] == "output"}

    assert outputs == {"bundle"}
    # The decoy is a real file and its value really is a path, but the flow
    # typed it `string`. It turns up only because the run directory was walked.
    decoy = next(item for item in found if item["label"] == "looks-like-a-path.txt")
    assert decoy["source"] == "run"
    assert decoy["outputName"] is None


def test_step_evidence_and_the_run_walk_are_both_discovered(typed_run):
    flow, run_dir, run_json = typed_run

    found = projections.artifacts_for(run_json, flow, run_id="r-0001", run_dir=run_dir)
    by_label = {item["label"]: item for item in found}

    assert by_label["stdout.log"]["source"] == "step"
    assert by_label["stdout.log"]["stepId"] == "emit"
    assert by_label[schema.MANIFEST_NAME]["source"] == "run"
    assert by_label["bundle"]["uri"] == "run://r-0001/bundle.kpack"
    # Every entry is addressable, and nothing was invented: each URI names a
    # file that exists.
    for item in found:
        _, relative = schema.parse_run_uri(item["uri"])
        assert (run_dir / relative).is_file()


def test_the_feedback_role_is_matched_by_path_not_by_name(typed_run):
    flow, run_dir, run_json = typed_run

    found = projections.artifacts_for(run_json, flow, run_id="r-0001", run_dir=run_dir)
    flagged = [item["label"] for item in found if item["role"] == "feedback"]

    # The run's recorded channel is what carries the role, whatever it is called.
    assert flagged == ["notes.md"]


def test_an_output_named_after_the_channel_does_not_inherit_its_role(tmp_path):
    flow = load(
        tmp_path,
        """
version: 1
name: collision
steps:
  - id: emit
    tool: writer
    stdin: "produce"
    result_file: "${run.dir}/emit.json"
    outputs:
      feedback: {json_file: result, path: "$.feedback", type: path}
""",
    )
    run_dir = tmp_path / "runs" / "collision" / "r-0002"
    run_dir.mkdir(parents=True)
    (run_dir / "feedback").write_text("a flow's own output", encoding="utf-8")
    (run_dir / "channel.md").write_text("the engine's channel", encoding="utf-8")
    run_json = {
        "status": "ok",
        "feedback_path": str(run_dir / "channel.md"),
        "steps": [
            {
                "id": "emit",
                "group": None,
                "iteration": None,
                "status": "ok",
                "dir": None,
                "outputs": {"feedback": str(run_dir / "feedback")},
            }
        ],
    }

    found = projections.artifacts_for(run_json, flow, run_id="r-0002", run_dir=run_dir)
    roles = {item["label"]: item["role"] for item in found}

    assert roles["feedback"] is None
    assert roles["channel.md"] == "feedback"


def test_a_path_output_written_outside_the_run_is_not_addressable(tmp_path):
    flow = load(tmp_path, DECLARED_TYPES)
    run_dir = tmp_path / "runs" / "every-type" / "r-0003"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "elsewhere.kpack"
    outside.write_text("payload", encoding="utf-8")
    run_json = {
        "status": "ok",
        "steps": [
            {
                "id": "emit",
                "group": None,
                "iteration": None,
                "status": "ok",
                "dir": None,
                "outputs": {"bundle": str(outside)},
            }
        ],
    }

    found = projections.artifacts_for(run_json, flow, run_id="r-0003", run_dir=run_dir)

    # `run://` addresses one run directory. An output the flow wrote elsewhere
    # is real, but there is no honest URI for it here.
    assert found == []


# -- step enrichment --------------------------------------------------------


def test_step_records_are_enriched_with_the_declaration_behind_them(typed_run):
    flow, run_dir, run_json = typed_run

    steps = projections.run_steps_for(run_json, flow, run_dir)

    assert steps[0]["kind"] == "agent"
    assert steps[0]["outputTypes"]["bundle"] == "path"
    assert steps[0]["outputTypes"]["tally"] == "int"
    # Recorded absolute, served relative: a consumer addresses it by `run://`.
    assert steps[0]["dir"] == "emit"


def test_a_run_whose_flow_is_gone_still_projects_its_steps(typed_run):
    _, run_dir, run_json = typed_run

    steps = projections.run_steps_for(run_json, None, run_dir)

    assert steps[0]["kind"] is None
    assert steps[0]["outputTypes"] == {}
    # The manifest is self-describing: the tool survives without the flow.
    assert steps[0]["tool"] == "writer"
