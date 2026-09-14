# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Registry and flow loading: the checks that must fire before anything launches."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from runner.errors import ConfigError
from runner.flow import Flow, bind_inputs, validate_refs
from runner.toolreg import ToolRegistry

FLOW_HEAD = "version: 1\nname: t\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -- registry ---------------------------------------------------------------


def test_registry_rejects_args_and_says_where_they_belong(tmp_path):
    path = write(
        tmp_path / "tools.yaml",
        "version: 1\ntools:\n  ping:\n    exe: ping\n    args: ['-n', '1']\n",
    )
    with pytest.raises(ConfigError, match="belongs on the step"):
        ToolRegistry.load(path)


def test_registry_vars_chain_and_reach_tool_paths(tmp_path):
    path = write(
        tmp_path / "tools.yaml",
        "version: 1\nvars:\n  root: /opt\n  bin: '${vars.root}/bin'\n"
        "tools:\n  thing:\n    exe: '${vars.bin}/thing'\n",
    )
    registry = ToolRegistry.load(path)
    assert registry.tools["thing"].exe == "/opt/bin/thing"


def test_registry_var_cycle_is_reported(tmp_path):
    path = write(
        tmp_path / "tools.yaml",
        "version: 1\nvars:\n  a: '${vars.b}'\n  b: '${vars.a}'\n"
        "tools:\n  thing:\n    exe: thing\n",
    )
    with pytest.raises(ConfigError, match="cycle or unknown reference"):
        ToolRegistry.load(path)


def test_missing_executable_fails_at_resolve_not_at_launch(tmp_path):
    path = write(
        tmp_path / "tools.yaml",
        "version: 1\ntools:\n  gone:\n    exe: /definitely/not/here/gone\n",
    )
    registry = ToolRegistry.load(path)
    with pytest.raises(ConfigError, match="does not exist"):
        registry.resolve_exe(registry.get("gone"))


def test_profile_overlays_exe(tmp_path):
    path = write(
        tmp_path / "tools.yaml",
        "version: 1\ntools:\n  agent:\n    exe: a\n"
        "profiles:\n  ci:\n    tools:\n      agent:\n        exe: b\n",
    )
    assert ToolRegistry.load(path).tools["agent"].exe == "a"
    assert ToolRegistry.load(path, "ci").tools["agent"].exe == "b"


# -- flow -------------------------------------------------------------------


def test_forward_reference_is_rejected(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + (
            "steps:\n"
            "  - id: first\n    tool: agent\n    args: ['${steps.second.outputs.exit_code}']\n"
            "  - id: second\n    tool: agent\n"
        ),
    )
    with pytest.raises(ConfigError, match="does not run before it"):
        validate_refs(Flow.load(flow), {})


def test_reference_to_undeclared_output_is_rejected(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + (
            "steps:\n"
            "  - id: first\n    tool: agent\n"
            "  - id: second\n    tool: agent\n    args: ['${steps.first.outputs.nope}']\n"
        ),
    )
    with pytest.raises(ConfigError, match="declares no output 'nope'"):
        validate_refs(Flow.load(flow), {})


def test_loop_without_until_is_rejected(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + (
            "steps:\n  - id: cycle\n    loop:\n      max_iterations: 2\n"
            "    steps:\n      - id: a\n        tool: agent\n"
        ),
    )
    with pytest.raises(ConfigError, match="has no 'until'"):
        Flow.load(flow)


def test_flow_vars_may_not_shadow_machine_vars(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + (
            "vars:\n  build_dir: /somewhere/else\n"
            "steps:\n  - id: a\n    tool: agent\n"
        ),
    )
    with pytest.raises(ConfigError, match="shadow machine vars"):
        validate_refs(Flow.load(flow), {"build_dir": "/real/build"})


def test_stdin_and_prompt_file_together_is_rejected(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + (
            "steps:\n  - id: a\n    tool: agent\n    stdin: hi\n    prompt_file: p.md\n"
        ),
    )
    with pytest.raises(ConfigError, match="same channel"):
        Flow.load(flow)


def test_prompt_file_contents_are_reference_checked(tmp_path):
    write(tmp_path / "p.md", "goal: ${inputs.nonexistent}\n")
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + (
            "inputs:\n  graph: {type: string}\n"
            "steps:\n  - id: a\n    tool: agent\n    prompt_file: p.md\n"
        ),
    )
    # A bad reference inside a prompt must fail at validate time; discovering it when
    # the step launches can mean an hour of agent time already spent.
    with pytest.raises(ConfigError, match="not a declared input"):
        validate_refs(Flow.load(flow), {})


def test_result_schema_without_result_file_is_rejected(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + ("steps:\n  - id: a\n    tool: agent\n    result_schema: {required: [x]}\n"),
    )
    with pytest.raises(ConfigError, match="without result_file"):
        Flow.load(flow)


def test_duplicate_step_ids_are_rejected(tmp_path):
    flow = write(
        tmp_path / "f.yaml",
        FLOW_HEAD
        + ("steps:\n  - id: a\n    tool: agent\n  - id: a\n    tool: agent\n"),
    )
    with pytest.raises(ConfigError, match="duplicate step id"):
        Flow.load(flow)


# -- inputs -----------------------------------------------------------------


def test_missing_required_input_lists_the_description(tmp_path):
    flow = Flow.load(
        write(
            tmp_path / "f.yaml",
            FLOW_HEAD
            + (
                "inputs:\n  graph:\n    required: true\n    description: the graph to satisfy\n"
                "steps:\n  - id: a\n    tool: agent\n"
            ),
        )
    )
    with pytest.raises(ConfigError, match="the graph to satisfy"):
        bind_inputs(flow, [])


def test_at_file_input_reads_the_file(tmp_path):
    brief = write(tmp_path / "brief.md", "long brief")
    flow = Flow.load(
        write(
            tmp_path / "f.yaml",
            FLOW_HEAD
            + ("inputs:\n  notes: {type: text}\nsteps:\n  - id: a\n    tool: agent\n"),
        )
    )
    assert bind_inputs(flow, [f"notes=@{brief}"]) == {"notes": "long brief"}


def test_path_input_must_exist(tmp_path):
    flow = Flow.load(
        write(
            tmp_path / "f.yaml",
            FLOW_HEAD
            + (
                "inputs:\n  graph: {type: path, required: true}\nsteps:\n  - id: a\n    tool: agent\n"
            ),
        )
    )
    with pytest.raises(ConfigError, match="does not exist"):
        bind_inputs(flow, [f"graph={tmp_path / 'nope.json'}"])


def test_unknown_input_is_rejected(tmp_path):
    flow = Flow.load(
        write(
            tmp_path / "f.yaml",
            FLOW_HEAD
            + (
                "inputs:\n  graph: {type: string}\nsteps:\n  - id: a\n    tool: agent\n"
            ),
        )
    )
    with pytest.raises(ConfigError, match="unknown input"):
        bind_inputs(flow, ["grahp=x"])


# -- the shipped configuration ---------------------------------------------


def test_shipped_flow_and_registry_validate_together():
    """The flow and prompts we ship must pass their own validation."""
    root = Path(__file__).resolve().parents[1]
    registry = ToolRegistry.load(root / "configs" / "tools.yaml")
    flow = Flow.load(root / "configs" / "flows" / "rtc-kernel-review.yaml")
    validate_refs(flow, registry.vars)
    assert flow.tools_used() <= set(registry.tools)
