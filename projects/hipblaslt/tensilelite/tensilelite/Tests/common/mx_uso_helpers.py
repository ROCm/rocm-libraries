# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Replay the focused MX USO regression with one output dump per launch."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np


def assert_row_uniform(path, sizes):
    """Check a packed column-major FP32 dump, including its non-vacuity controls."""
    m, n, batch, _ = sizes
    bits = np.fromfile(path, dtype="<u4")
    assert bits.size == m * n * batch, f"Wrong tensor size: {path}"
    values = bits.view("<f4")
    assert np.isfinite(values).all(), f"Nonfinite output: {path}"
    rows = bits.reshape(batch, n, m)
    assert np.all(rows == rows[:, :, :1]), f"Rows differ bitwise: {path}"
    assert np.all(np.any(rows[:, :, 0] != rows[:, :1, 0], axis=1)), f"Constant output: {path}"


def assert_static_partials(log, kernel):
    """A successful numerical result must have used the intended split path."""
    launches = []
    for block in log.split("TensileLite::StreamK LAUNCH SUMMARY")[1:]:
        name = re.search(r"kernel='([^']+)'", block)
        assert name and name.group(1) == kernel, "Unexpected kernel in launch summary"
        launches.append(dict(re.findall(r"^ +(\w+) += (\S+)", block, re.MULTILINE)))
    assert launches, "No StreamK launch summary"
    for fields in launches:
        assert fields.get("partials") == "yes", "No partial reduction"
        assert fields.get("allocated") == "yes", "No partial workspace"
        assert fields.get("reduction") == "tree", "Expected tree reduction"
        assert fields.get("isDynamic") == "no", "Expected static assignment"
        assert fields.get("dpOnly") == "no", "Fell back to whole tiles"
        assert int(fields["final"]) > int(fields["tiles"]), "Expected multiple workers per tile"


def run_mx_uso_checks(output_dir, client, settings):
    """Reuse built kernels and retain each numerical/bitwise run in its own directory.

    The fixture supplies dense TN GEMMs with FP32 output. A=One and MXSA=One
    repeat the effective A row exactly; B and its scales vary with K/column.
    The CPU reference must also have identical rows and finite, nonconstant
    values, so an invalid input pattern cannot earn a passing row check.
    """
    from tensilelite.Common import ClientExecutionLock
    from tensilelite.Common.GlobalParameters import globalParameters
    from tensilelite.Utilities.BinaryLibraryIO import iterSolutionRecords

    output_dir = Path(output_dir)
    configs = sorted(
        output_dir.glob("1_BenchmarkProblems/*/*/caches/*/source/ClientParameters.ini")
    )
    assert configs, "No generated client configurations"
    # Grid selection mode 0 fixes tree reduction; the default mode 6 may
    # select parallel reduction even for a statically assigned kernel.
    env = {
        **os.environ,
        "TENSILE_DB": "0x200000",
        "TENSILE_PERSISTENT_FIXED_GRID": str(settings["grid"]),
        "TENSILE_PERSISTENT_DYNAMIC_GRID": "0",
    }
    solutions_seen = 0
    checks = 0
    for config_index, config in enumerate(configs):
        entries = [
            line.split("=", 1)
            for line in config.read_text().splitlines()
            if "=" in line and not line.startswith("#")
        ]
        values = dict(entries)
        assert values["d-type"] == "Float" and values["c-equal-d"] == "False"
        assert values["problem-identifier"] == "Contraction_l_Alik_Bljk_Cijk_Dijk"
        library = Path(values["library-file"])
        if not library.exists():
            library = Path(str(library) + ".zlib")
        solutions = [record for _, record in iterSolutionRecords(library)]
        sizes = [
            list(map(int, value.split(","))) for key, value in entries if key == "problem-size"
        ]
        strides = [list(map(int, value.split(","))) for key, value in entries if key == "d-strides"]
        assert len(strides) == len(sizes)
        solutions_seen += len(solutions)
        for solution in solutions:
            for problem_index, shape in enumerate(sizes):
                m, n, batch, _ = shape
                assert strides[problem_index][0] in (-1, 1)
                assert strides[problem_index][1] == m
                assert strides[problem_index][2] in (-1, m * n)
                run_dir = (
                    output_dir
                    / "mx-uso"
                    / f"config{config_index}-solution{solution['index']}-problem{problem_index}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                # Remove stale dumps before every replay, even when --use-cache
                # repeats an earlier run in the same output directory.
                for name in ("D", "Ref"):
                    (run_dir / f"tensor_{name}.bin").unlink(missing_ok=True)
                overrides = {
                    "solution-start-idx": solution["index"],
                    "num-solutions": 1,
                    "problem-start-idx": problem_index,
                    "num-problems": 1,
                    "uniform-summation-order": 1,
                    "num-elements-to-validate": -1,
                    "init-a": "One",
                    "init-mx-a": "One",
                    "init-b": "TrigSin",
                    "init-mx-b": "Random",
                    "init-alpha": "One",
                    "init-beta": "Zero",
                    "print-tensor-d": "True",
                    "print-tensor-ref": "True",
                    "dump-tensors": "True",
                    "log-level": "Debug",
                    "exit-on-error": "True",
                    "num-warmups": 1,
                    "num-benchmarks": 1,
                    "num-enqueues-per-sync": 1,
                    "num-syncs-per-benchmark": 0,
                    "results-file": str(run_dir / "results.csv"),
                }
                ini = run_dir / "client.ini"
                ini.write_text(
                    "".join(f"{key}={value}\n" for key, value in entries if key not in overrides)
                    + "".join(f"{key}={value}\n" for key, value in overrides.items())
                )
                log_path = run_dir / "client.log"
                with ClientExecutionLock(
                    globalParameters["ClientExecutionLockPath"]
                ), log_path.open("w") as log:
                    result = subprocess.run(
                        [str(client), "--config-file", str(ini)],
                        cwd=run_dir,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
                assert result.returncode == 0, f"Numerical validation failed: {log_path}"
                assert_static_partials(log_path.read_text(), solution["kernelName"])
                for name in ("Ref", "D"):
                    assert_row_uniform(run_dir / f"tensor_{name}.bin", shape)
                checks += 1
    assert (
        solutions_seen == settings["expected_solutions"]
    ), f"Expected {settings['expected_solutions']} solutions, generated {solutions_seen}"
    print(f"MX USO: {solutions_seen} solutions, {checks} numerical and bitwise split checks passed")
