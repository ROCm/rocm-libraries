# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An arm with `benchmarking: false` runs with HIPDNN_FORCE_BENCHMARKING=0 (the engine
serves its ranker's top-ranked candidate); every other arm keeps forcing it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.test_sweep_tools import Sweep, _needs_posix_exec

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import sweep as sweep_tool  # noqa: E402


@pytest.mark.parametrize(
    "arm, expected",
    [({}, "1"), ({"benchmarking": True}, "1"), ({"benchmarking": False}, "0")],
)
def test_phase_env_forces_benchmarking_unless_the_arm_turns_it_off(
    tmp_path, arm, expected
):
    arm = {"install_tree": str(tmp_path / "install"), **arm}
    env = sweep_tool.phase_env({"HIPDNN_FORCE_BENCHMARKING": "7"}, arm, tmp_path)
    assert env["HIPDNN_FORCE_BENCHMARKING"] == expected


def _edit_arm(staged, **keys):
    config = json.loads(staged.config_path.read_text())
    config["arms"][0].update(keys)
    staged.config_path.write_text(json.dumps(config))


@_needs_posix_exec
def test_a_ranker_arm_runs_and_its_flag_is_part_of_the_resume_fingerprint(tmp_path):
    staged = Sweep(tmp_path)
    assert staged.run().returncode == 0
    assert staged.resumed(staged.run())
    _edit_arm(staged, benchmarking=False)
    result = staged.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert not staged.resumed(result), "flipping the flag is a different experiment"


@_needs_posix_exec
@pytest.mark.parametrize("value", ["no", 0, None])
def test_a_non_boolean_benchmarking_value_is_invalid_configuration(tmp_path, value):
    staged = Sweep(tmp_path)
    _edit_arm(staged, benchmarking=value)
    result = staged.run()
    assert result.returncode == 2
    assert "benchmarking must be a boolean" in result.stderr + result.stdout
