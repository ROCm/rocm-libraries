################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Run-only test phase for YAML kernel configs.

This module is activated by the ``--use-cache`` pytest flag. Two mechanisms
keep it from running in other modes: the ``pytest_ignore_collect`` hook in
``conftest.py`` excludes it at collection time, and the test function itself
calls ``pytest.skip`` if the flag is absent (fallback for pytest versions or
invocation styles where the hook is not called).

Role in the split-CI workflow::

    build (test_config_build.py)  -->  artifact (.tar.gz)  -->  run (this file)

This phase extracts a pre-built artifact from ``--artifact-dir`` and benchmarks
the cached kernels on the target GPU. It requires a GPU and an artifact
previously produced by ``test_config_build.py``. The artifact is extracted into
``tmpdir`` rather than back into ``--artifact-dir``, so the shared artifact
directory is never modified by the run phase.

For local single-machine testing (build + run in one pytest session) see
``test_config.py``, which calls ``_build`` and ``_run`` via isolated
subprocesses to exercise the same artifact round-trip.
"""

import os
import re
import subprocess
import sys

import py
import pytest
import yaml

from tensilelite import run

from artifact_helpers import artifact_name_for_config, artifact_timing, extract_artifact


def _require_verification(log, verify_reuse: bool, verify_upload: bool) -> None:
    log.seek(0)
    output = log.read()
    print(output, end="")
    if verify_reuse:
        counts = re.findall(r"^REFERENCE_REUSE_VERIFIED:(\d+)$", output, re.MULTILINE)
        assert sum(map(int, counts)) > 0, "No CPU reference reuse verification executed"
    if verify_upload:
        for kind in ("input", "batched-pointer"):
            counts = re.findall(
                rf"^Verified (\d+) skipped GPU {kind} uploads(?: against the host data)?\.$",
                output, re.MULTILINE)
            assert sum(map(int, counts)) > 0, f"No GPU {kind} upload-skip verification executed"


def _launch_summaries(output: str) -> list[dict[str, str]]:
    """The "key = value" fields of every StreamK launch summary in output, plus its kernel."""
    summaries = []
    for block in output.split("TensileLite::"):
        if not block.startswith("StreamK LAUNCH SUMMARY"):
            continue
        fields = dict(re.findall(r"^ +(\w+) += (\S+)", block, re.MULTILINE))
        fields["kernel"] = re.search(r"kernel='([^']+)'", block).group(1)
        summaries.append(fields)
    return summaries


def _field_matches(actual, wanted) -> bool:
    """wanted is an int to equal or a string ">N" to exceed."""
    if actual is None:
        return False
    if isinstance(wanted, str) and wanted.startswith(">"):
        return int(actual) > int(wanted[1:])
    return int(actual) == int(wanted)


def _require_streamk_launches(log, expected: list[dict]) -> None:
    """Each entry names a tile count and summary fields: every StreamK launch of that
    many tiles must report them, and at least one launch must have that many tiles."""
    log.seek(0)
    summaries = _launch_summaries(log.read())
    for expectation in expected:
        tiles = expectation["tiles"]
        launches = [row for row in summaries if row.get("tiles") == str(tiles)]
        assert launches, f"No StreamK launch of {tiles} tiles: {log.name}"
        for row in launches:
            for field, wanted in expectation.items():
                assert _field_matches(row.get(field), wanted), (
                    f"{row['kernel']}: {tiles} tiles launched with {field} = {row.get(field)},"
                    f" expected {wanted}: {log.name}")


def _run_client_process(args: list[str], env: dict[str, str], log) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from tensilelite.run import main; main(sys.argv[1:])",
         *args], env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, f"Numerical run failed: {log.name}"


def _run(config: str, output_dir: str, artifact_dir: str, tensile_args: list[str]) -> None:
    """Extract a pre-built artifact and run benchmarks against it.

    Callable from both the pytest wrapper below and from test_config.py via
    subprocess (where it runs in a clean process to avoid global-state bleed).
    """
    artifact_name = artifact_name_for_config(config)
    tarball = os.path.join(artifact_dir, artifact_name + ".tar.gz")
    assert os.path.isfile(tarball), f"Artifact tarball not found: {tarball}"
    with artifact_timing("python_artifact_extract", tensile_args):
        extract_artifact(tarball, output_dir)
    args = [config, output_dir, "--use-cache", *tensile_args]
    with open(config) as source:
        parameters = yaml.safe_load(source).get("TestParameters", {})
    schedules = parameters.get("streamk_schedules", [])
    verify_reuse = parameters.get("verify_reference_reuse", False)
    verify_upload = parameters.get("verify_upload_skip", False)
    # Expected StreamK launch summary fields per tile count; unscheduled configs only.
    launches = parameters.get("streamk_launches", [])
    assert not (launches and schedules), "streamk_launches applies to unscheduled configs"
    # Environment for the client alone, e.g. debug overrides only an env var reaches.
    client_env = {key: str(value) for key, value in parameters.get("client_env", {}).items()}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), **client_env}
    if verify_reuse:
        env["TENSILELITE_VERIFY_REFERENCE_REUSE"] = "1"
    if verify_upload:
        env["TENSILELITE_VERIFY_UPLOAD_SKIP"] = "1"
    if launches:
        env["TENSILE_DB"] = "0x200000"
    if not schedules:
        if verify_reuse or verify_upload or client_env or launches:
            # A separate process scopes verification, client_env and the launch summaries
            # to this config, and its log holds the client's output even when the client
            # inherits its stdout file descriptor.
            log_name = "reuse-verification.log" if verify_reuse or verify_upload else "client.log"
            with open(os.path.join(output_dir, log_name), "w+") as log:
                _run_client_process(args, env, log)
                _require_verification(log, verify_reuse, verify_upload)
                if launches:
                    _require_streamk_launches(log, launches)
        else:
            run.main(args)
        return
    for index, schedule in enumerate(schedules):
        mode = schedule["reduction"]
        assert mode in ("tree", "parallel", "whole"), mode
        schedule_env = {**env, "TENSILE_DB": "0x200000",
                        "TENSILE_PERSISTENT_FIXED_GRID": str(schedule["grid"]),
                        "TENSILE_PERSISTENT_DYNAMIC_GRID": "6" if mode == "parallel" else "0"}
        parameters = {"ParallelGpuExecution": 1, "ForceRedoBenchmarkProblems": True,
                      "ExitOnFails": 1, **schedule.get("parameters", {})}
        overrides = [f"{key}={value!r}" for key, value in parameters.items()]
        run_args = [*args, *(["--global-parameters", *overrides] if overrides else [])]
        log_path = os.path.join(output_dir, f"streamk-schedule-{index}.log")
        # Each client run uses a fresh process and the same compiled artifact.
        with open(log_path, "w+") as log:
            _run_client_process(run_args, schedule_env, log)
            if verify_reuse or verify_upload:
                _require_verification(log, verify_reuse, verify_upload)
            log.seek(0)
            records = {}
            for fields in _launch_summaries(log.read()):
                if "physicalPartialSlots" in fields:
                    records.setdefault(fields["kernel"], []).append(fields)
        assert records, f"No clustered StreamK launches observed: {log_path}"
        for kernel, launches in records.items():
            if mode == "whole":
                assert all(int(row["physicalPartialSlots"]) == 0 for row in launches), kernel
                matched = any(row["fallback"] == "insufficient-workspace" for row in launches)
            else:
                matched = any(
                    row["reduction"].startswith(mode) and int(row["physicalPartialSlots"]) > 0
                    and (int(row["flagEntries"]) > 0 if mode == "tree"
                         else int(row["parallelSplits"]) > 1)
                    for row in launches)
            assert matched, f"{kernel} never exercised {mode}: {log_path}"


def test_config_run(tensile_args: list[str], config: str, tmpdir: py.path.local, pytestconfig: pytest.Config) -> None:
    """Pytest wrapper: extract a pre-built artifact and benchmark on the GPU.

    Activated only when ``--use-cache`` is passed; collection is skipped in all
    other modes by ``conftest.pytest_ignore_collect``. Requires a GPU.

    ``--artifact-dir`` must point to the directory where ``test_config_build``
    wrote the artifact. The artifact is extracted into ``tmpdir`` so the shared
    artifact directory is not modified by this phase.
    """
    if not pytestconfig.getoption("--use-cache"):
        pytest.skip("requires --use-cache")
    artifact_name = artifact_name_for_config(config)
    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    _run(config, output_dir, pytestconfig.getoption("--artifact-dir"), tensile_args)
