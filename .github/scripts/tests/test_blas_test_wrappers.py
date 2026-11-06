# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check CPU budgets and smoke coverage at the BLAS test launch boundary."""

import fnmatch
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]


class BlasTestWrappersTest(unittest.TestCase):
    def launch(self, component, artifact_group, budget=None):
        env = {
            "ARTIFACT_GROUP": artifact_group,
            "AMDGPU_FAMILIES": "gfx950",
            "RUNNER_OS": "Linux",
            "THEROCK_BIN_DIR": "/unused/bin",
            "THEROCK_DIR": str(REPO_ROOT),
            "TEST_TYPE": "quick",
            # The first shard also performs unrelated layout and tuning-store checks.
            "SHARD_INDEX": "2",
            "TOTAL_SHARDS": "2",
            "HSA_XNACK": "0",
        }
        if budget is not None:
            env.update(OMP_NUM_THREADS=budget, OPENBLAS_NUM_THREADS=budget)
        with patch.dict(os.environ, env, clear=True), patch("subprocess.run") as run:
            runpy.run_path(str(REPO_ROOT / "test/therock" / f"test_{component}.py"))
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], f"/unused/bin/{component}-test")
        self.assertTrue(kwargs["check"])
        return args[0], kwargs["env"]

    def test_host_asan_preserves_cpu_budget(self):
        for component in ("hipblaslt", "hipsparselt"):
            for budget in (None, "1", "25"):
                with self.subTest(component=component, budget=budget):
                    _, env = self.launch(component, "gfx950-host-asan", budget)
                    self.assertEqual(env.get("OMP_NUM_THREADS"), budget)
                    self.assertEqual(env.get("OPENBLAS_NUM_THREADS"), budget)
                    self.assertEqual(env["HSA_XNACK"], "0")

    def test_device_asan_keeps_hipblaslt_workarounds(self):
        _, env = self.launch("hipblaslt", "gfx950-asan", "25")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "25")
        self.assertEqual(env["HSA_XNACK"], "1")

    def test_sparse_smoke_includes_128_by_128_under_asan(self):
        for artifact in ("gfx950", "gfx950-host-asan", "gfx950-asan"):
            with self.subTest(artifact=artifact):
                cmd, _ = self.launch("hipsparselt", artifact, "25")
                filter_arg = next(
                    arg for arg in cmd if arg.startswith("--gtest_filter=")
                )
                positive, _, negative = filter_arg.split("=", 1)[1].partition("-")

                def selected(name):
                    return any(
                        fnmatch.fnmatchcase(name, p) for p in positive.split(":")
                    ) and not any(
                        fnmatch.fnmatchcase(name, p) for p in negative.split(":")
                    )

                for transpose in ("NN", "NT", "TN", "TT"):
                    name = f"spmm/smoke_spmm_strided_batched_smoke_f16_r_{transpose}_128_128_128"
                    self.assertTrue(selected(name), name)
                    self.assertFalse(selected(name + "_known_bug"))


if __name__ == "__main__":
    unittest.main()
