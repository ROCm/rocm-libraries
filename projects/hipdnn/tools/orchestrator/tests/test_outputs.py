# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Output extraction and the process launcher."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from runner.errors import StepError
from runner.flow import OutputSpec
from runner.outputs import Artifacts, extract, load_result
from runner.process import build_env, launch


def artifacts(tmp_path: Path, stdout: str = "", stderr: str = "") -> Artifacts:
    (tmp_path / "stdout.log").write_text(stdout, encoding="utf-8")
    (tmp_path / "stderr.log").write_text(stderr, encoding="utf-8")
    return Artifacts(tmp_path / "stdout.log", tmp_path / "stderr.log", tmp_path)


def test_regex_takes_the_last_match_and_coerces_the_type(tmp_path):
    # Coverage summaries print at the end; an earlier partial line must not win.
    log = "Passed: 1\nrerunning\nPassed: 17\n"
    spec = OutputSpec(
        name="passed", kind="regex", argument=r"Passed:\s+(\d+)", type="int"
    )
    assert extract(spec, spec.argument, artifacts(tmp_path, log), None, 0) == 17


def test_regex_that_matches_nothing_is_an_error_not_an_empty_string(tmp_path):
    spec = OutputSpec(
        name="passed", kind="regex", argument=r"Passed:\s+(\d+)", type="int"
    )
    with pytest.raises(StepError, match="matched nothing"):
        extract(spec, spec.argument, artifacts(tmp_path, "no summary here"), None, 0)


def test_json_path_reads_nested_keys_and_indexes(tmp_path):
    payload = json.dumps({"result": {"issues": [{"title": "race"}, {"title": "oob"}]}})
    spec = OutputSpec(
        name="first", kind="json", argument=None, path="$.result.issues[1].title"
    )
    assert extract(spec, None, artifacts(tmp_path, payload), None, 0) == "oob"


def test_json_extractor_explains_non_json_stdout(tmp_path):
    spec = OutputSpec(name="verdict", kind="json", argument=None, path="$.verdict")
    with pytest.raises(StepError, match="write a result file instead"):
        extract(
            spec, None, artifacts(tmp_path, "I had a think about it and..."), None, 0
        )


def test_missing_json_key_names_what_was_available(tmp_path):
    spec = OutputSpec(
        name="verdict", kind="json_file", argument="result", path="$.verdict"
    )
    with pytest.raises(StepError, match="available: critical_count"):
        extract(spec, "result", artifacts(tmp_path), {"critical_count": 0}, 0)


def test_path_typed_output_must_exist(tmp_path):
    # An agent that reports a file it never wrote is the common failure; catching it
    # here keeps the next step from being handed a path to nothing.
    spec = OutputSpec(
        name="kernel_path",
        kind="json_file",
        argument="result",
        path="$.kernel_path",
        type="path",
    )
    with pytest.raises(StepError, match="does not exist"):
        extract(
            spec,
            "result",
            artifacts(tmp_path),
            {"kernel_path": str(tmp_path / "no.hip")},
            0,
        )


def test_result_file_errors_name_the_missing_keys(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"entry_point": "k"}), encoding="utf-8")
    with pytest.raises(StepError, match="missing required key\\(s\\): kernel_path"):
        load_result("generate", path, ["kernel_path", "entry_point"])


def test_result_file_that_is_not_json_is_reported_as_such(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("Sure! Here's the JSON:\n{...}", encoding="utf-8")
    with pytest.raises(StepError, match="not valid JSON"):
        load_result("generate", path, [])


def test_tail_returns_the_last_lines(tmp_path):
    spec = OutputSpec(name="tail", kind="tail", argument=2)
    text = "\n".join(f"line {n}" for n in range(10))
    assert extract(spec, 2, artifacts(tmp_path, text), None, 0) == "line 8\nline 9"


# -- process ----------------------------------------------------------------


def test_timeout_kills_the_process_and_is_reported(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(30)", encoding="utf-8")
    result = launch(
        [sys.executable, str(script)],
        cwd=None,
        env=build_env(),
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        timeout=1.0,
    )
    assert result.timed_out is True
    assert result.duration_s < 20


def test_stdin_reaches_the_process_and_output_is_captured(tmp_path):
    script = tmp_path / "echo.py"
    script.write_text(
        textwrap.dedent("import sys; sys.stdout.write(sys.stdin.read().upper())"),
        encoding="utf-8",
    )
    result = launch(
        [sys.executable, str(script)],
        cwd=None,
        env=build_env(),
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        stdin_text="prompt body",
    )
    assert result.exit_code == 0
    assert (tmp_path / "out.log").read_text() == "PROMPT BODY"


def test_path_prepend_lands_in_front_of_the_inherited_path():
    env = build_env(path_prepend=["/rocm/bin"])
    assert env["PATH"].startswith("/rocm/bin")
