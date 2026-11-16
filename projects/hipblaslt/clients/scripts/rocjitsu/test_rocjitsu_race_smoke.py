# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rocjitsu_race_smoke as smoke

# Lines in the format the rocjitsu race and logging plugins print.
DISPATCH = '[rocjitsu] Kernel dispatch: "Cijk_example" symbol="Cijk_example"\n'
MFMA = "[rocjitsu] mfma detected in dispatch 3\n"
RACE = (
    "RACE kernel=Cijk_example symbol=Cijk_example dispatch=3 type=RAW "
    "access=read reg=v12 wave=0\n"
)


class WorkloadFailuresTest(unittest.TestCase):
    def test_clean_run_passes(self):
        self.assertEqual(smoke.workload_failures(0, DISPATCH + MFMA), [])

    def test_reported_races_fail(self):
        # Console output interleaves workload and plugin output, so a race line
        # need not start at the beginning of a line.
        output = DISPATCH + RACE + "bench: " + RACE + MFMA
        self.assertEqual(
            smoke.workload_failures(0, output), ["race detector reported 2 race(s)"]
        )

    def test_run_without_detector_output_fails(self):
        # Without the race plugin a clean-looking run must not pass.
        self.assertEqual(
            smoke.workload_failures(0, MFMA),
            ["race detector recorded no kernel dispatch"],
        )

    def test_run_without_mfma_is_inconclusive(self):
        self.assertEqual(
            smoke.workload_failures(0, DISPATCH),
            ["no MFMA dispatch reached; the race check is inconclusive"],
        )

    def test_failed_or_timed_out_workload_fails(self):
        output = DISPATCH + MFMA
        self.assertEqual(smoke.workload_failures(3, output), ["exited with status 3"])
        self.assertEqual(smoke.workload_failures(None, output), ["timed out"])


class RunnerTest(unittest.TestCase):
    def test_benchmark_verification_failure_fails_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write(relative, contents):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
                path.chmod(0o755)

            write("share/rocjitsu/configs/gfx942_cdna3_kmd.json", "{}")
            for path in (
                "bin/amdclang++",
                "lib/llvm/lib/libomp.so",
                "libexec/hipblaslt/tensilelite/tensilelite-client",
            ):
                write(path, "")
            write(
                "share/hipblaslt/tensilelite/tensilelite/bin/Tensile",
                "print(',PASSED,')\n",
            )
            write(
                "bin/rocjitsu",
                f"#!{sys.executable}\n"
                "import os, sys\n"
                "if '--version' in sys.argv:\n"
                "    sys.exit(0)\n"
                f"print({DISPATCH + MFMA!r}, flush=True)\n"
                "cmd = sys.argv[sys.argv.index('--') + 1:]\n"
                "os.execv(cmd[0], cmd)\n",
            )
            for mismatch in (False, True):
                with self.subTest(mismatch=mismatch):
                    # Model verification_exit_code in hipblaslt_bench_options.cpp:
                    # a failed comparison changes the exit status only in strict mode.
                    write(
                        "bin/hipblaslt-bench",
                        f"#!{sys.executable}\n"
                        "import sys\n"
                        f"print('verification failures: {int(mismatch)}')\n"
                        f"sys.exit(2 if {mismatch} and '--verify_strict' in sys.argv else 0)\n",
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(smoke.__file__)),
                            "--rocm-path",
                            str(root),
                            "--amdgpu-families",
                            "gfx94X-dcgpu",
                            "--report-dir",
                            str(root / "reports"),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, int(mismatch), result.stderr)
                    self.assertIn(
                        f"verification failures: {int(mismatch)}", result.stdout
                    )
                    self.assertIn(",PASSED,", result.stdout)
                    if mismatch:
                        self.assertEqual(
                            result.stderr,
                            "error: hipblaslt-bench: exited with status 2\n",
                        )
                        self.assertNotIn("rocjitsu race checks passed", result.stdout)
                    else:
                        self.assertIn("rocjitsu race checks passed", result.stdout)


class RunWithTimeoutTest(unittest.TestCase):
    def test_timeout_kills_processes_started_by_the_workload(self):
        # Killing only the direct child would leave the sleep running, holding
        # the output pipe open, so run_with_timeout would not return in time.
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            cmd = ["sh", "-c", f"sleep 30 & echo $! > {pid_file}; wait"]
            start = time.monotonic()
            returncode, _ = smoke.run_with_timeout(cmd, tmp, None, timeout=1)
            self.assertLess(time.monotonic() - start, 10)
            self.assertIsNone(returncode)
            child = Path("/proc") / pid_file.read_text().strip()
            deadline = time.monotonic() + 5
            while child.exists() and "State:\tZ" not in _status(child):
                self.assertLess(time.monotonic(), deadline, "child still running")
                time.sleep(0.05)


def _status(proc_dir):
    # The process can exit between the existence check and this read.
    try:
        return (proc_dir / "status").read_text()
    except OSError:
        return "State:\tZ"


class SelectTargetTest(unittest.TestCase):
    def test_families_map_to_targets(self):
        # TheRock passes family names such as gfx94X-dcgpu, not gfx942.
        self.assertEqual(smoke.select_target("gfx94X-dcgpu")[0], "gfx942")
        self.assertEqual(smoke.select_target("gfx950-dcgpu")[0], "gfx950")
        with self.assertRaises(SystemExit):
            smoke.select_target("gfx120X-all")


if __name__ == "__main__":
    unittest.main()
