# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


DRIVER = (
    Path(__file__).resolve().parents[5]
    / ".github/scripts/run_rocjitsu_hipblaslt_gemm_race_checks.sh"
)


class GemmDriverTests(unittest.TestCase):
    def test_fetched_toolchain_and_runtime_reach_build_and_workload(self):
        for layout in ("", os.uname().machine + "-unknown-linux-gnu"):
            with self.subTest(
                runtime_layout=layout
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = root / "payload"
                tools = root / "tools"
                tools.mkdir()
                for name in ("mkdir", "dirname", "date", "tee", "uname"):
                    (tools / name).symlink_to(shutil.which(name))
                (tools / "python3").symlink_to(sys.executable)
                for name in ("amdclang", "amdclang++"):
                    path = payload / "lib/llvm/bin" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("#!/bin/sh\nexit 0\n")
                    path.chmod(0o755)
                for name in (
                    "hipblaslt-bench",
                    "hipblaslt_gentest.py",
                    "hipblaslt_template.yaml",
                    "hipblaslt_common.yaml",
                ):
                    path = payload / "bin" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                runtime = payload / "lib/llvm/lib" / layout
                runtime.mkdir(parents=True)
                (runtime / "libomp.so").touch()
                config = root / "config.json"
                config.write_text("{}")
                build = root / "build"
                emulator = build / "tools/rocjitsu/rocjitsu"
                emulator.parent.mkdir(parents=True)
                emulator.write_text("#!/bin/sh\necho fixture-version\n")
                emulator.chmod(0o755)
                recorder = (
                    f"#!{sys.executable}\n"
                    "import json, os, sys\n"
                    "with open(os.environ['CAPTURE'], 'a') as output:\n"
                    " json.dump({'argv': sys.argv, 'env': dict(os.environ)}, output)\n"
                    " output.write('\\n')\n"
                )
                cmake = tools / "cmake"
                cmake.write_text(recorder)
                cmake.chmod(0o755)
                runner = root / "runner"
                runner.mkdir()
                (runner / "rocjitsu_random_gemms.py").write_text(recorder)
                capture = root / "calls.jsonl"
                result = subprocess.run(
                    ["/bin/bash", str(DRIVER)],
                    cwd=root,
                    env={
                        "PATH": str(tools),
                        "LD_LIBRARY_PATH": str(root / "inherited-libs"),
                        "ROCM_PATH": str(payload),
                        "AMDGPU_TARGETS": "gfx950",
                        "ROCJITSU_CONFIG": str(config),
                        "ROCJITSU_BUILD_DIR": str(build),
                        "RUNNER_DIR": str(runner),
                        "CAPTURE": str(capture),
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                self.assertEqual(len(calls), 3)
                self.assertIn(
                    f"-DCMAKE_C_COMPILER={payload}/lib/llvm/bin/amdclang",
                    calls[0]["argv"],
                )
                self.assertIn(
                    f"-DCMAKE_CXX_COMPILER={payload}/lib/llvm/bin/amdclang++",
                    calls[0]["argv"],
                )
                self.assertIn("--build", calls[1]["argv"])
                self.assertEqual(
                    calls[2]["argv"][0], str(runner / "rocjitsu_random_gemms.py")
                )
                for call in calls:
                    self.assertEqual(call["env"]["ROCM_PATH"], str(payload))
                    libraries = call["env"]["LD_LIBRARY_PATH"].split(os.pathsep)
                    for directory in (
                        payload / "lib",
                        payload / "lib/rocm_sysdeps/lib",
                        runtime,
                        root / "inherited-libs",
                    ):
                        self.assertIn(str(directory), libraries)


if __name__ == "__main__":
    unittest.main()
