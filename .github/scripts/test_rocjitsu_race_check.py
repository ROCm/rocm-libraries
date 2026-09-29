#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise CI orchestration independently of the installed sweep tooling."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


class RaceCheckTests(unittest.TestCase):
    def test_driver_runs_all_stages_and_propagates_each_failure(self):
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
run_timed() { echo "STAGE: $1"; shift; "$@"; }
run_hipblaslt_bench_check() { return "$BENCH_STATUS"; }
run_tensilelite_client_check() { return "$CLIENT_STATUS"; }
python3() { printf 'ARG: %s\n' "$@"; if [[ "$3" == tensile ]]; then return "$TENSILE_SWEEP_STATUS"; else return "$BENCH_SWEEP_STATUS"; fi; }
"""
        for statuses in [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]:
            with self.subTest(statuses=statuses):
                result = subprocess.run(
                    ["bash", "-c", setup + stages],
                    env={
                        **os.environ,
                        **dict(
                            zip(
                                [
                                    "BENCH_STATUS",
                                    "CLIENT_STATUS",
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
                        "STAGE: hipblaslt-bench race check",
                        "STAGE: tensilelite-client race check",
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
        workflow = (
            Path(__file__).parents[1]
            / "workflows/therock-rocjitsu-race-check-linux.yml"
        )
        steps = yaml.safe_load(workflow.read_text())["jobs"][
            "rocjitsu-race-check-linux"
        ]["steps"]
        publish = next(
            s for s in steps if s["name"] == "Publish sampled solution results"
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
