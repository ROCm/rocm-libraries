# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "hipblaslt_asan_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("asan_diagnostics", SCRIPT)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_for_file(path, process):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise AssertionError(f"fixture exited with {process.returncode}")
        time.sleep(0.02)
    raise AssertionError("fixture did not become ready")


def wait_for_exit(pid):
    # kill(2) returning does not mean the target has processed the signal yet.
    deadline = time.monotonic() + 5
    while running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if running(pid):
        raise AssertionError(f"PID {pid} survived process-group cleanup")


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.console = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.console)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def executable(self, name, script):
        path = self.root / name
        path.write_text("#!/bin/sh\n" + script)
        path.chmod(0o755)
        return path

    def test_failed_probe_keeps_stderr(self):
        result = diagnostics.probe(
            [
                sys.executable,
                "-c",
                "import sys; print('load-order error', file=sys.stderr); sys.exit(7)",
            ],
            self.root / "probe.log",
            5,
        )
        self.assertEqual(result["returncode"], 7)
        self.assertIn("load-order error", self.console.getvalue())
        self.assertFalse(result["timed_out"])

    def test_probe_timeout_keeps_partial_output(self):
        result = diagnostics.probe(
            [
                sys.executable,
                "-c",
                "import time; print('before hang', flush=True); time.sleep(60)",
            ],
            self.root / "probe.log",
            0.2,
        )
        self.assertTrue(result["timed_out"])
        self.assertIn("before hang", (self.root / "probe.log").read_text())

    def test_sanity_continues_after_failure_and_missing_tool(self):
        self.executable("rocminfo", "echo actual-gpu-uuid\n")
        for smi_present in (False, True):
            with self.subTest(smi_present=smi_present):
                if smi_present:
                    self.executable("amd-smi", "echo smi-error >&2; exit 1\n")
                with patch.object(
                    diagnostics, "host_snapshot", return_value={"kernel": "example"}
                ):
                    diagnostics.sanity(self.root, self.root, 5)
                results = json.loads((self.root / "sanity-results.json").read_text())
                self.assertEqual(results["rocminfo"]["returncode"], 0)
                self.assertIn(
                    "actual-gpu-uuid", (self.root / "rocminfo.log").read_text()
                )
                self.assertTrue((self.root / "host-before.json").exists())
                if smi_present:
                    self.assertEqual(results["amd-smi"]["returncode"], 1)
                else:
                    self.assertIn("error", results["amd-smi"])

    def test_asan_preload_is_scoped_to_python_diagnostic(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        runtime = (
            self.root
            / "lib/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so"
        )
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        with patch.dict(
            os.environ, {"ASAN_OPTIONS": "quarantine_size_mb=600"}
        ), patch.object(diagnostics, "host_snapshot", return_value={}), patch.object(
            diagnostics, "probe", return_value={}
        ) as probe:
            diagnostics.sanity(self.root, bin_dir, 5)
            smi_env = probe.call_args_list[0].args[3]
            self.assertIn(str(runtime), smi_env["LD_PRELOAD"])
            self.assertEqual(
                smi_env["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                str(self.root / "lib/rocm_sysdeps/lib"),
            )
            self.assertIn("detect_leaks=0", smi_env["ASAN_OPTIONS"])
            self.assertIsNone(probe.call_args_list[1].args[3])
            self.assertEqual(os.environ["ASAN_OPTIONS"], "quarantine_size_mb=600")

    def test_command_exit_codes_are_not_called_timeouts(self):
        for code in (0, 7, 124, 137):
            with self.subTest(code=code), patch.object(
                diagnostics, "capture_timeout"
            ) as capture:
                actual = diagnostics.run_test(
                    [sys.executable, "-c", f"raise SystemExit({code})"], self.root, 5
                )
                self.assertEqual(actual, code)
                capture.assert_not_called()
                self.assertFalse(
                    json.loads((self.root / "test-result.json").read_text())[
                        "timed_out"
                    ]
                )
        self.assertNotIn("::error::", self.console.getvalue())

    def test_signal_exit_is_preserved(self):
        actual = diagnostics.run_test(
            [
                sys.executable,
                "-c",
                "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
            ],
            self.root,
            5,
        )
        self.assertEqual(actual, 143)
        self.assertFalse(
            json.loads((self.root / "test-result.json").read_text())["timed_out"]
        )

    def test_timeout_captures_live_children_then_cleans_only_its_group(self):
        child_file = self.root / "child"
        driver = self.root / "driver.py"
        driver.write_text(
            "import os, pathlib, subprocess, sys\n"
            "os.environ['GITHUB_TOKEN'] = 'must-not-be-recorded'\n"
            "os.environ['HSA_XNACK'] = '0'\n"
            "p = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
            f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid))\n"
            "p.wait()\n"
        )
        sentinel = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        self.addCleanup(sentinel.wait)
        self.addCleanup(sentinel.kill)
        real_capture = diagnostics.capture_timeout
        captured = []

        def capture(output, group):
            child = int(child_file.read_text())
            self.assertTrue(running(child), "child must still be alive during capture")
            captured.append(child)
            with patch.object(
                diagnostics.shutil, "which", return_value=None
            ), patch.object(
                diagnostics, "probe", return_value={"error": "permission denied"}
            ):
                real_capture(output, group)

        with patch.object(diagnostics, "capture_timeout", side_effect=capture):
            result = diagnostics.run_test([sys.executable, str(driver)], self.root, 0.5)
        self.assertEqual(result, 124)
        self.assertTrue(captured)
        wait_for_exit(captured[0])
        self.assertIsNone(sentinel.poll())
        text = (self.root / "processes.json").read_text()
        self.assertNotIn("must-not-be-recorded", text)
        self.assertNotIn("GITHUB_TOKEN", text)
        children = {p["pid"]: p for p in json.loads(text)}
        self.assertEqual(children[captured[0]]["environment"]["HSA_XNACK"], "0")
        self.assertIn("maps", children[captured[0]])
        self.assertIn("gdb unavailable", self.console.getvalue())

    def test_diagnostic_error_does_not_hide_timeout(self):
        with patch.object(
            diagnostics, "capture_timeout", side_effect=OSError("cannot capture")
        ):
            code = diagnostics.run_test(
                [sys.executable, "-c", "import time; time.sleep(60)"], self.root, 0.1
            )
        self.assertEqual(code, 124)
        self.assertIn("cannot capture", self.console.getvalue())

    def test_cancellation_terminates_children(self):
        for mode in ("run", "sanity"):
            with self.subTest(mode=mode):
                ready = self.root / f"ready-{mode}"
                script = (
                    "import os,pathlib,time; "
                    f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
                )
                command = [
                    sys.executable,
                    str(SCRIPT),
                    mode,
                    "--output-dir",
                    str(self.root),
                ]
                fixture = [sys.executable, "-c", script]
                if mode == "run":
                    command += ["--"] + fixture
                else:
                    self.executable("amd-smi", f"exec {shlex.join(fixture)}\n")
                    command += ["--bin-dir", str(self.root)]
                wrapper = subprocess.Popen(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self.addCleanup(wrapper.wait)
                self.addCleanup(wrapper.kill)
                wait_for_file(ready, wrapper)
                pid = int(ready.read_text())
                self.addCleanup(diagnostics.signal_group, pid, signal.SIGKILL)
                wrapper.terminate()
                self.assertEqual(wrapper.wait(timeout=5), 143)
                wait_for_exit(pid)

    def driver_fixture(self):
        # Match the GPU job's sparse checkout and use the actual driver. Native
        # cat/true fixtures need neither a compiler nor a GPU in the setup job.
        driver = self.root / "rocm-libraries/test/therock/test_hipblaslt.py"
        driver.parent.mkdir(parents=True)
        shutil.copy2(SCRIPT.parents[2] / "test/therock/test_hipblaslt.py", driver)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("hipblaslt-test", "hipblaslt-test-tuning-store"):
            shutil.copy2(shutil.which("true"), bin_dir / name)
        env = os.environ.copy()
        env.update(
            RUNNER_OS="Linux",
            ARTIFACT_GROUP="gfx90a-host-asan",
            THEROCK_DIR=str(self.root),
            THEROCK_BIN_DIR=str(bin_dir),
            TEST_TYPE="full",
            SHARD_INDEX="1",
            TOTAL_SHARDS="1",
        )
        return driver, bin_dir, env

    def test_driver_ptrace_requires_asan_and_positive_pid(self):
        driver, _, env = self.driver_fixture()
        env["HIPBLASLT_DIAGNOSTIC_PTRACER_PID"] = "-1"
        for artifact, expected in (("gfx90a", 0), ("gfx90a-host-asan", 1)):
            with self.subTest(artifact=artifact):
                env["ARTIFACT_GROUP"] = artifact
                result = subprocess.run(
                    [sys.executable, str(driver)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected:
                    self.assertIn("must be a positive PID", result.stderr)

    def require_debugger(self):
        if not shutil.which("gdb"):
            if os.getenv("HIPBLASLT_DIAGNOSTICS_CHECK_PTRACE") == "1":
                self.fail("gdb is required for the GPU job's timeout diagnostics")
            self.skipTest("gdb is not installed")

    def test_real_debugger_captures_native_children_through_driver(self):
        self.require_debugger()
        driver, bin_dir, env = self.driver_fixture()
        for name in ("hipblaslt-test", "hipblaslt-test-tuning-store"):
            with self.subTest(binary=name):
                binary = bin_dir / name
                shutil.copy2(shutil.which("cat"), binary)
                output = self.root / name
                output.mkdir()
                # With an open stdin pipe and no input, cat blocks in read().
                # The collector must pass its PID through the production driver
                # and capture this live native child before group cleanup.
                with (output / "collector.log").open("w+") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "run",
                            "--output-dir",
                            str(output),
                            "--timeout",
                            "2",
                            "--",
                            sys.executable,
                            str(driver),
                        ],
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    self.addCleanup(process.stdin.close)
                    self.addCleanup(process.wait)
                    self.addCleanup(process.kill)
                    process.wait(timeout=60)
                    log.seek(0)
                    console = log.read()
                self.assertEqual(process.returncode, 124, console)
                processes = json.loads((output / "processes.json").read_text())
                targets = [p for p in processes if p.get("exe") == str(binary)]
                self.assertEqual(len(targets), 1, console)
                pid = targets[0]["pid"]
                capture = json.loads((output / "capture-results.json").read_text())
                self.assertEqual(len(capture["samples"]), 2, console)
                for sample in capture["samples"]:
                    result = sample["backtraces"][str(pid)]
                    self.assertEqual(result["returncode"], 0, console)
                    self.assertFalse(result["timed_out"], console)
                    trace = (
                        output / sample["directory"] / f"backtrace-{pid}.log"
                    ).read_text()
                    self.assertIn(f"=== Main thread LWP {pid} ===", trace)
                    self.assertIn("#0", trace)
                    self.assertIn("read", trace)
                    self.assertIn("detached", trace.lower())
                    self.assertNotIn("Diagnostic command failed", trace)
                wait_for_exit(pid)
                shutil.copy2(shutil.which("true"), binary)

    def test_deep_main_stack_and_live_progress_between_samples(self):
        self.require_debugger()
        binary = self.root / "hipblaslt-test"
        shutil.copy2(sys.executable, binary)
        pulse = self.root / "pulse"
        # Nested ctypes callbacks create real native frames without requiring
        # a compiler in the GPU image. The deepest frame keeps making progress.
        code = f"""
import ctypes, pathlib, time
libc = ctypes.CDLL(None)
libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
assert libc.prctl(0x59616d61, {os.getpid()}, 0, 0, 0) == 0
pulse = pathlib.Path({str(pulse)!r})
temporary = pulse.with_suffix('.tmp')
def descend(depth):
    if depth:
        callback(depth - 1)
    else:
        count = 0
        while True:
            temporary.write_text(str(count))
            temporary.replace(pulse)
            count += 1
            time.sleep(0.01)
callback = ctypes.CFUNCTYPE(None, ctypes.c_int)(descend)
callback(24)
"""
        process = subprocess.Popen([str(binary), "-c", code], start_new_session=True)
        self.addCleanup(diagnostics.stop_process, process)
        wait_for_file(pulse, process)
        counts = []
        real_probe = diagnostics.probe

        def observe_probe(command, output, timeout, env=None):
            result = real_probe(command, output, timeout, env)
            if output.name.startswith("backtrace-"):
                counts.append(int(pulse.read_text()))
            return result

        with patch.object(diagnostics, "probe", side_effect=observe_probe):
            diagnostics.capture_timeout(self.root, process.pid, interval=0.2)
        capture = json.loads((self.root / "capture-results.json").read_text())
        self.assertEqual(len(capture["samples"]), 2, self.console.getvalue())
        self.assertEqual(len(counts), 2)
        self.assertGreater(counts[1], counts[0], "target must run between samples")
        self.assertIsNone(process.poll(), "capture must leave the target running")
        for sample in capture["samples"]:
            trace = (
                self.root / sample["directory"] / f"backtrace-{process.pid}.log"
            ).read_text()
            self.assertEqual(
                sample["backtraces"][str(process.pid)]["returncode"], 0, trace
            )
            main = trace.split(f"=== Main thread LWP {process.pid} ===", 1)[1]
            self.assertRegex(main, r"(?m)^#79\s")
            self.assertRegex(main, r"(?m)^(rip|pc)\s+0x")
            self.assertRegex(main, r"=>\s+0x")
            self.assertNotIn("Diagnostic command failed", main)

    def test_stalled_debugger_consumes_budget_without_second_attachment(self):
        binary = self.root / "hipblaslt-test"
        shutil.copy2(sys.executable, binary)
        ready = self.root / "ready"
        process = subprocess.Popen(
            [
                str(binary),
                "-c",
                f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(60)",
            ],
            start_new_session=True,
        )
        self.addCleanup(diagnostics.stop_process, process)
        wait_for_file(ready, process)
        invocations = self.root / "gdb-invocations"
        debugger = self.executable(
            "gdb", f"echo invoked >> {shlex.quote(str(invocations))}\nsleep 60\n"
        )
        started = time.monotonic()
        with patch.object(diagnostics.shutil, "which", return_value=str(debugger)):
            diagnostics.capture_timeout(self.root, process.pid, budget=0.2)
        self.assertLess(time.monotonic() - started, 5)
        capture = json.loads((self.root / "capture-results.json").read_text())
        self.assertEqual(len(capture["samples"]), 1)
        self.assertTrue(
            capture["samples"][0]["backtraces"][str(process.pid)]["timed_out"]
        )
        self.assertIn("second_sample_skipped", capture)
        self.assertEqual(invocations.read_text().splitlines(), ["invoked"])
        self.assertIsNone(process.poll())

    def test_second_sample_failure_preserves_first_and_kernel_log_attempt(self):
        first = {"timestamp_utc": "example", "backtraces": {"123": {"returncode": 0}}}
        with patch.object(
            diagnostics, "capture_sample", side_effect=[first, OSError("second failed")]
        ), patch.object(diagnostics.time, "sleep"), patch.object(
            diagnostics, "probe", return_value={"returncode": 1}
        ) as probe:
            diagnostics.capture_timeout(self.root, 123)
        capture = json.loads((self.root / "capture-results.json").read_text())
        self.assertEqual(capture["samples"], [first])
        self.assertIn("second failed", capture["second_sample_error"])
        self.assertEqual(probe.call_args.args[0], ["dmesg", "--level=err,warn"])


if __name__ == "__main__":
    unittest.main()
