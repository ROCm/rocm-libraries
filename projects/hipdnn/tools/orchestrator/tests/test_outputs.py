# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Output extraction and the process launcher."""
from __future__ import annotations

import json
import sys
import textwrap
import time
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


def test_sha256_detects_a_file_changing_between_steps(tmp_path):
    # The flows pair this with an assert so a reviewer that edits the kernel it is
    # reviewing is caught by evidence, not merely forbidden by a CLI flag.
    kernel = tmp_path / "kernel.hip"
    kernel.write_text("__global__ void k() {}", encoding="utf-8")
    spec = OutputSpec(name="kernel_sha", kind="sha256", argument=str(kernel))

    before = extract(spec, str(kernel), artifacts(tmp_path), None, 0)
    kernel.write_text("__global__ void k() { /* edited */ }", encoding="utf-8")
    after = extract(spec, str(kernel), artifacts(tmp_path), None, 0)

    assert len(before) == 64
    assert before != after


def test_sha256_of_a_missing_file_is_an_error(tmp_path):
    spec = OutputSpec(
        name="kernel_sha", kind="sha256", argument=str(tmp_path / "no.hip")
    )
    with pytest.raises(StepError, match="does not exist"):
        extract(spec, spec.argument, artifacts(tmp_path), None, 0)


def test_count_type_measures_the_list_the_agent_returned(tmp_path):
    # A review's own count and its issue list are two fields an agent writes; only one
    # of them can be checked against reality, so the count is derived from the list.
    spec = OutputSpec(
        name="critical_count",
        kind="json_file",
        argument="result",
        path="$.critical_issues",
        type="count",
    )
    result = {"critical_issues": [{"title": "oob write"}, {"title": "race"}]}
    assert extract(spec, "result", artifacts(tmp_path), result, 0) == 2


def test_count_type_rejects_a_value_that_is_not_a_list(tmp_path):
    spec = OutputSpec(
        name="critical_count",
        kind="json_file",
        argument="result",
        path="$.critical_issues",
        type="count",
    )
    with pytest.raises(StepError, match="needs a list or mapping"):
        extract(spec, "result", artifacts(tmp_path), {"critical_issues": "two"}, 0)


def test_scalar_type_rejects_a_list_instead_of_passing_it_through(tmp_path):
    # `type: int` holding [1, 2] makes every later comparison against it meaningless.
    spec = OutputSpec(
        name="failed", kind="json_file", argument="result", path="$.failed", type="int"
    )
    with pytest.raises(StepError, match="expects a single value"):
        extract(spec, "result", artifacts(tmp_path), {"failed": [1, 2]}, 0)


def test_invalid_regex_is_a_step_error_not_a_traceback(tmp_path):
    spec = OutputSpec(name="passed", kind="regex", argument="Passed: (")
    with pytest.raises(StepError, match="not a valid regular expression"):
        extract(spec, spec.argument, artifacts(tmp_path, "Passed: 1"), None, 0)


def test_result_file_that_is_not_utf8_is_a_step_error(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b'{"verdict": "\xff\xfe pass"}')
    with pytest.raises(StepError, match="not valid UTF-8"):
        load_result("review", path, [])


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


def test_timeout_covers_a_prompt_the_child_never_reads(tmp_path):
    """A prompt larger than the pipe buffer used to block the write *outside* the
    deadline: a 0.1s timeout took 1.3s and reported timed_out=False."""
    script = tmp_path / "deaf.py"
    script.write_text("import time; time.sleep(30)", encoding="utf-8")
    started = time.monotonic()
    result = launch(
        [sys.executable, str(script)],
        cwd=None,
        env=build_env(),
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        stdin_text="x" * (4 * 1024 * 1024),
        timeout=1.0,
    )
    assert result.timed_out is True
    assert time.monotonic() - started < 15
