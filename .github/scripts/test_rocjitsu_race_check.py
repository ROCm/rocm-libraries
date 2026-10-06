#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise CI orchestration independently of the installed sweep tooling."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


def workflow_steps():
    workflow = (
        Path(__file__).parents[1] / "workflows/therock-rocjitsu-race-check-linux.yml"
    )
    return yaml.safe_load(workflow.read_text())["jobs"]["rocjitsu-race-check-linux"][
        "steps"
    ]


class RaceCheckTests(unittest.TestCase):
    def test_container_paths_and_installed_requirements(self):
        for output in ("./build", "artifact tree", "absolute"):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                artifact = root / output
                requirements = (
                    artifact / "share/hipblaslt/tensilelite/rocjitsu/requirements.txt"
                )
                requirements.parent.mkdir(parents=True)
                requirements.write_text("msgpack\nPyYAML\n")
                installer = root / "build_tools/install_additional_requirements.py"
                installer.parent.mkdir()
                # TheRock passes the path straight to uv from its checkout root.
                # Read that file here without downloading or installing packages.
                installer.write_text(
                    "from pathlib import Path\nimport sys\n"
                    "assert sys.argv[1] == '--requirements-files'\n"
                    "print(Path(sys.argv[2]).read_text(), end='')\n"
                )
                env = {
                    **os.environ,
                    "PATH": str(Path(sys.executable).parent)
                    + os.pathsep
                    + os.environ["PATH"],
                    "OUTPUT_ARTIFACTS_DIR": (
                        str(artifact) if output == "absolute" else output
                    ),
                    "GITHUB_ENV": str(root / "github-env"),
                    # GitHub's workspace context can still name the host mount.
                    **{
                        name: "/host/workspace/" + name.lower()
                        for name in (
                            "VENV_DIR",
                            "ROCM_PATH",
                            "ROCJITSU_SOURCE_DIR",
                            "ROCJITSU_BUILD_DIR",
                            "RACE_REPORT_DIR",
                        )
                    },
                }
                installed = None
                for step in workflow_steps():
                    if step["name"] not in {
                        "Resolve race-check paths",
                        "Install GEMM sweep requirements",
                    }:
                        continue
                    result = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", step["run"]],
                        cwd=root,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    exports = Path(env["GITHUB_ENV"])
                    if exports.exists():
                        env.update(
                            line.split("=", 1)
                            for line in exports.read_text().splitlines()
                        )
                    if step["name"] == "Install GEMM sweep requirements":
                        installed = result.stdout
                self.assertEqual(installed, requirements.read_text())
                self.assertEqual(Path(env["ROCM_PATH"]), artifact)
                for name, relative in (
                    ("VENV_DIR", ".venv"),
                    ("ROCJITSU_SOURCE_DIR", "rocm-systems/emulation/rocjitsu"),
                    ("ROCJITSU_BUILD_DIR", "rocjitsu-build"),
                    ("RACE_REPORT_DIR", "race-reports"),
                ):
                    self.assertEqual(Path(env[name]), root / relative)

    def test_driver_runs_both_sweeps_and_propagates_each_failure(self):
        driver = Path(__file__).with_name("run_rocjitsu_hipblaslt_race_check.sh")
        # Exercise the real post-setup stage sequence with lightweight workloads.
        stages = driver.read_text().split("\ncheck_status=0\n", 1)[1]
        setup = """
set -euo pipefail
check_status=0
ROCJITSU_BIN=emulator
TENSILELITE_CLIENT=client
HIPBLASLT_BENCH=bench
ROCJITSU_SWEEP_SEED=pr-revision
ROCJITSU_CONFIG=config
ROCJITSU_GPU_TARGET=gfx942
ROCM_PATH=artifact
RACE_REPORT_DIR=reports
SWEEP_DRIVER=artifact/sweep.py
run_timed() { echo "STAGE: $1"; shift; "$@"; }
python3() { printf 'ARG: %s\n' "$@"; if [[ "$3" == tensile ]]; then return "$TENSILE_SWEEP_STATUS"; else return "$BENCH_SWEEP_STATUS"; fi; }
"""
        for statuses in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            with self.subTest(statuses=statuses):
                result = subprocess.run(
                    ["bash", "-c", setup + stages],
                    env={
                        **os.environ,
                        **dict(
                            zip(
                                [
                                    "TENSILE_SWEEP_STATUS",
                                    "BENCH_SWEEP_STATUS",
                                ],
                                map(str, statuses),
                            )
                        ),
                    },
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, int(any(statuses)), result.stderr)
                self.assertEqual(
                    [
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("STAGE:")
                    ],
                    [
                        "STAGE: tensile sampled race sweep",
                        "STAGE: bench sampled race sweep",
                    ],
                )
                for flag, value in [
                    ("--workers", "4"),
                    ("--kernels", "100"),
                    ("--target", "gfx942"),
                    ("--seed", "pr-revision"),
                    ("--suite-timeout", "1500"),
                ]:
                    self.assertEqual(
                        result.stdout.count(f"ARG: {flag}\nARG: {value}\n"), 2
                    )
                self.assertIn("ARG: reports/sweep-tensile", result.stdout)
                self.assertIn("ARG: reports/sweep-bench", result.stdout)

    def test_ci_publishes_partial_and_missing_reports(self):
        publish = next(
            s
            for s in workflow_steps()
            if s["name"] == "Publish sampled solution results"
        )
        self.assertEqual(publish["if"], "${{ always() }}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sweep-tensile").mkdir()
            report = "| 7 | FAIL | 4/4 | 4/4 |\nRun incomplete.\n"
            (root / "sweep-tensile/summary.md").write_text(report)
            subprocess.run(
                ["bash", "-euc", publish["run"]],
                env={
                    **os.environ,
                    "RACE_REPORT_DIR": str(root),
                    "GITHUB_STEP_SUMMARY": str(root / "job.md"),
                },
                check=True,
                capture_output=True,
                timeout=10,
            )
            text = (root / "job.md").read_text()
            self.assertIn(report, text)
            self.assertIn("bench sampled race sweep", text)
            self.assertIn("No report", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
