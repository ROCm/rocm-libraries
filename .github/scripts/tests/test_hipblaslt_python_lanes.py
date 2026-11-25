# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check which hipBLASLt Python bindings lanes run, and on which runners."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from hipblaslt_python_lanes import GFX950_LABEL, gpu_runner, plan_lanes

CONFIG = {
    "gpu_runner_labels": {
        "gfx94x": {"linux": {"test-runs-on": "linux-gfx942-test"}},
        "gfx950": {"linux": {"test-runs-on": "linux-gfx950-test"}},
        "gfx1151": {"windows": {"test-runs-on": "windows-gfx1151-test"}},
    }
}


def names(matrix):
    return [lane["name"] for lane in matrix]


class HipblasltPythonLanesTest(unittest.TestCase):
    def plan(self, labels=(), event="pull_request", all_lanes=False):
        return plan_lanes(labels, event, all_lanes, "windows-cpu", CONFIG)

    def test_a_bindings_pr_runs_gfx942_and_windows_cpu(self):
        lanes = self.plan()
        self.assertEqual(names(lanes["linux_gpu"]), ["gfx942"])
        self.assertEqual(names(lanes["windows"]), ["CPU (non-blocking)"])

    def test_the_label_adds_gfx950(self):
        lanes = self.plan(labels=["organization: ROCm", GFX950_LABEL])
        self.assertEqual(lanes["linux_gpu"][1]["runner"], "linux-gfx950-test")
        self.assertEqual(lanes["linux_gpu"][1]["device"], "gfx950")

    def test_a_similar_label_does_not(self):
        lanes = self.plan(labels=["ci:run-gfx950-please"])
        self.assertEqual(names(lanes["linux_gpu"]), ["gfx942"])

    def test_a_lane_change_adds_windows_gpu_but_not_gfx950(self):
        # gfx950 machines are scarce; a change to the lanes runs every other lane.
        lanes = self.plan(all_lanes=True)
        self.assertEqual(names(lanes["linux_gpu"]), ["gfx942"])
        self.assertEqual(lanes["windows"][1]["runner"], "windows-gfx1151-test")
        self.assertEqual(lanes["windows"][1]["require_gpu"], "1")

    def test_a_manual_run_adds_every_lane(self):
        lanes = self.plan(event="workflow_dispatch", all_lanes=True)
        self.assertEqual(len(lanes["linux_gpu"]), 2)
        self.assertEqual(len(lanes["windows"]), 2)

    def test_runner_labels_come_from_the_config(self):
        self.assertEqual(self.plan()["linux_gpu"][0]["runner"], "linux-gfx942-test")
        self.assertEqual(self.plan()["windows"][0]["runner"], "windows-cpu")

    def test_a_missing_runner_is_an_error_not_an_empty_label(self):
        with self.assertRaises(SystemExit):
            gpu_runner({"gpu_runner_labels": {}}, "gfx94x", "linux")
        with self.assertRaises(SystemExit):
            gpu_runner(
                {"gpu_runner_labels": {"gfx94x": {"linux": {"test-runs-on": ""}}}},
                "gfx94x",
                "linux",
            )

    def test_the_script_writes_both_matrices(self):
        # The whole script, as the workflow runs it, outside a pull request.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "runner-config-v2.json"
            config.write_text(json.dumps(CONFIG))
            output = Path(tmp) / "output"
            env = dict(
                os.environ,
                GITHUB_OUTPUT=str(output),
                PR="",
                EVENT="workflow_dispatch",
                ALL_LANES="true",
                WINDOWS_RUNNER="windows-cpu",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "hipblaslt_python_lanes.py"),
                    "--config",
                    str(config),
                ],
                env=env,
                check=True,
                capture_output=True,
            )
            written = dict(
                line.split("=", 1) for line in output.read_text().splitlines() if line
            )
        self.assertEqual(len(json.loads(written["linux_gpu"])), 2)
        self.assertEqual(len(json.loads(written["windows"])), 2)


if __name__ == "__main__":
    unittest.main()
