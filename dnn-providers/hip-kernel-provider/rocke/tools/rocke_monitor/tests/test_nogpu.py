# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/nogpu.py: every check keeps its runs off the GPU the same way (no HIP_VISIBLE_DEVICES, an empty
CUDA_VISIBLE_DEVICES): HIP_VISIBLE_DEVICES=-1 aborts this host's HIP runtime as soon as a run imports torch.
"""

import os
import pathlib
import re
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rkmon_lib import nogpu, prcode  # noqa: E402

# An assignment in code: a keyword argument, a dict entry, or an export in a script line.
SET_MINUS_ONE = re.compile(
    r"""HIP_VISIBLE_DEVICES=["']-1|["']HIP_VISIBLE_DEVICES["']\s*:\s*["']-1"""
    r"|export .*HIP_VISIBLE_DEVICES=-1"
)


class TestNoGpu(unittest.TestCase):
    def test_env_drops_hip_and_empties_cuda(self):
        env = nogpu.env(
            {"HIP_VISIBLE_DEVICES": "-1", "PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
        )
        self.assertEqual(env, {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": ""})

    def test_the_shell_line_does_the_same(self):
        show = 'echo "HIP=${HIP_VISIBLE_DEVICES-unset} CUDA=[${CUDA_VISIBLE_DEVICES-unset}]"'
        child = "bash -c 'echo \"child CUDA=[${CUDA_VISIBLE_DEVICES-unset}]\"'"
        out = subprocess.run(
            ["bash", "-c", f"{nogpu.SHELL}; {show}; {child}"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "-1"},
        ).stdout
        self.assertEqual(out.split("\n")[:2], ["HIP=unset CUDA=[]", "child CUDA=[]"])

    def test_no_check_sets_hip_visible_devices_to_minus_one(self):
        hits = []
        for p in sorted(
            [
                *ROOT.glob("rkmon_lib/**/*.py"),
                *ROOT.glob("tools/**/*.py"),
                *ROOT.glob("tools/**/*.sh"),
                *ROOT.glob("worker/**/*.py"),
                *ROOT.glob("worker/**/*.sh"),
            ]
        ):
            for i, line in enumerate(
                p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if SET_MINUS_ONE.search(line) and not line.lstrip().startswith("#"):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
        self.assertEqual(hits, [])

    def test_the_check_scripts_hide_the_gpu(self):
        """The review's checks, the drift run and the sanity run each set the line up before they run anything."""
        from rkmon_lib import drift, guard, review

        cfg = {
            "rocke_path": "rk",
            "repo": "ROCm/r",
            "paths": {"working_dir": "/nonexistent", "repo": "/nonexistent"},
        }
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        scripts = {
            "checks": review.checks_script(cfg, 1, "a" * 40, "first-round", plan, None),
            "drift": drift.script(
                cfg, "/j", "d" * 40, [{"n": 1, "head": "1" * 40, "tests": ["t.py"]}]
            ),
            "sanity": guard.sanity_script(
                cfg, "a" * 40, "/x/wtsanity", "/x/out", ["gemm"], True
            ),
        }
        for name, s in scripts.items():
            lines = s.replace(
                prcode.SHELL, ""
            ).splitlines()  # the slot commands' definitions run nothing
            self.assertIn(nogpu.SHELL, lines, name)
            first_run = next(
                i for i, x in enumerate(lines) if re.search(r"\bpython3?\b|pytest", x)
            )
            self.assertLess(lines.index(nogpu.SHELL), first_run, name)


if __name__ == "__main__":
    unittest.main()
