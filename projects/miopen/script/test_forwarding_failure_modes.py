#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How the MIOpen wrapper routes, traces and reports a hipDNN that cannot be used.

Every switch involved is read once per process: MIOPEN_HIPDNN_FORWARDING,
MIOPEN_DISABLE_HIPDNN_FOR, MIOPEN_LOG_LEVEL, MIOPEN_ENABLE_LOGGING and whether hipDNN
loaded. So each case runs forwarding_failure_probe in a new process, with every MIOPEN_*
and HIPDNN_* variable removed and only the case's own set.

hipDNN is made unusable with a fake backend that has the real backend's file name and no
version, so hipDNN loads it and then refuses it. Every case points hipDNN at it, so none
can reach a real backend; a hipDNN frontend that predates HIPDNN_BACKEND_LIBRARY_PATH
ignores it and loads the real one, so the cases fail if the fake never loads. A missing backend cannot be simulated: when the override
directory has none, hipDNN falls back to the real one next to the MIOpen library, then to
the loader's search path. MIOpen handles the fake the same way as a missing backend.

ctest sets MIOPEN_FAILURE_PROBE (the probe) and MIOPEN_FAKE_HIPDNN_BACKEND (the fake's
directory). Without them every case is skipped.

Three cases use a second backend instead, from MIOPEN_REENTRANT_HIPDNN_BACKEND: hipDNN
accepts it, and its hipdnnDestroy calls miopenDestroy. One checks that miopenDestroy does
not hang when destroying its hipDNN handle re-enters it. One makes the backend fail to
set a stream, and checks that the forwarded call that destroys the hipDNN handle it just
created does not hang either. One exits with a forwarded handle still alive, and checks
that MIOpen destroys nothing at exit, where the re-entry would find MIOpen's handle map
torn down. All need a GPU, because the backend creates an MIOpen handle, and are skipped
without one.

Uses unittest rather than pytest, like the other wrapper script tests, because nothing
installs pytest on a machine that builds MIOpen.

    python3 -m unittest test_forwarding_failure_modes
"""

import os
import subprocess
import unittest
from pathlib import Path

PROBE_ENV = "MIOPEN_FAILURE_PROBE"
FAKE_BACKEND_ENV = "MIOPEN_FAKE_HIPDNN_BACKEND"
REENTRANT_BACKEND_ENV = "MIOPEN_REENTRANT_HIPDNN_BACKEND"

# miopenStatus_t values the probe prints.
SUCCESS = 0
BAD_PARM = 3
INTERNAL_ERROR = 5

LOADED_MARKER = "FAKE_HIPDNN_BACKEND_LOADED"
DESTROYED_MARKER = "REENTRANT_HIPDNN_BACKEND_DESTROYED_MIOPEN_HANDLE="
UNAVAILABLE = "[MIOpen] hipDNN forwarding is unavailable"
FORWARDED_FAILURE = "[MIOpen] [hipDNN-forwarded] "
FORWARDED_CALL = "[MIOpen] hipDNN forwarded: miopenConvolutionForward"
CONV_TRACE = "[MIOpen] hipDNN routing: miopenConvolutionForward "
FORWARDED_PREFIX = "[hipDNN-forwarded] "

FORWARDING_ON = {"MIOPEN_HIPDNN_FORWARDING": "1"}


class ProbeRun:
    def __init__(self, completed):
        self.stderr = completed.stderr
        self.stderr_lines = completed.stderr.splitlines()
        self.results = [line.split("=", 1) for line in completed.stdout.splitlines()]

    def values(self, step):
        return [value for name, value in self.results if name == step]

    def statuses(self, step):
        return [int(value) for value in self.values(step)]

    def lines_starting(self, prefix):
        return [line for line in self.stderr_lines if line.startswith(prefix)]

    def loaded_fake_backend(self):
        return LOADED_MARKER in self.stderr_lines


class ForwardingFailureModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = os.environ.get(PROBE_ENV)
        fake_backend = os.environ.get(FAKE_BACKEND_ENV)
        if not probe or not fake_backend:
            raise unittest.SkipTest(
                f"set {PROBE_ENV} and {FAKE_BACKEND_ENV}; ctest does this in a build "
                "with MIOPEN_ENABLE_HIPDNN_WRAPPER=ON"
            )
        # hipDNN ignores a relative HIPDNN_BACKEND_LIBRARY_PATH, and the installed test
        # list passes paths relative to its own directory.
        cls.probe = Path(probe).resolve()
        cls.fake_backend = Path(fake_backend).resolve()
        reentrant_backend = os.environ.get(REENTRANT_BACKEND_ENV)
        cls.reentrant_backend = (
            Path(reentrant_backend).resolve() if reentrant_backend else None
        )

    # Set once a run shows hipDNN loading the fake, so the check runs once per process.
    fake_backend_loads = False

    def require_fake_backend(self):
        if ForwardingFailureModesTest.fake_backend_loads:
            return
        run = self.run_probe(["conv_forward"], FORWARDING_ON, backend=self.fake_backend)
        if not run.loaded_fake_backend():
            # Otherwise the cases reach the real backend and fail one by one, each
            # for a reason that hides this one.
            self.fail(
                f"hipDNN never loaded the fake backend in {self.fake_backend}: MIOpen was "
                "built against a hipDNN frontend that ignores HIPDNN_BACKEND_LIBRARY_PATH"
                f"\nstderr:\n{run.stderr}"
            )
        ForwardingFailureModesTest.fake_backend_loads = True

    def run_probe(self, steps, env=None, backend=None, timeout=120):
        if backend is None:
            self.require_fake_backend()
        clean = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("MIOPEN_", "HIPDNN_"))
        }
        clean["HIPDNN_BACKEND_LIBRARY_PATH"] = str(backend or self.fake_backend)
        if os.name == "nt":
            # Windows finds a program's DLLs beside it or on PATH, and the installed
            # probe is one folder below MIOpen.dll.
            clean["PATH"] = os.pathsep.join(
                [
                    str(self.probe.parent),
                    str(self.probe.parent.parent),
                    clean.get("PATH", ""),
                ]
            )
        clean.update(env or {})
        try:
            completed = subprocess.run(
                [str(self.probe), *steps],
                env=clean,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as hung:
            self.fail(
                f"probe hung for {timeout} s\nstdout:\n{hung.stdout}\nstderr:\n{hung.stderr}"
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"probe failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return ProbeRun(completed)

    def test_refused_backend_fails_every_forwarded_call(self):
        run = self.run_probe(["conv_forward", "repeat", "2"], FORWARDING_ON)
        self.assertTrue(run.loaded_fake_backend(), run.stderr)
        self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR] * 3)
        # hipDNN's availability is checked once per process, so the reason prints once.
        self.assertEqual(len(run.lines_starting(UNAVAILABLE)), 1, run.stderr)
        self.assertEqual(len(run.lines_starting(FORWARDED_FAILURE)), 3, run.stderr)

    def test_refused_backend_fails_every_forwarded_batchnorm_call(self):
        run = self.run_probe(["batchnorm"], FORWARDING_ON)
        self.assertTrue(run.loaded_fake_backend(), run.stderr)
        for step in ("bn_train", "bn_infer", "bn_backward"):
            self.assertEqual(run.statuses(step), [INTERNAL_ERROR], run.stderr)
        self.assertEqual(len(run.lines_starting(UNAVAILABLE)), 1, run.stderr)
        self.assertEqual(len(run.lines_starting(FORWARDED_FAILURE)), 3, run.stderr)

    def test_trace_names_the_unavailable_backend_as_the_reason(self):
        # Level 7 (Trace) includes everything level 6 prints.
        for level in ("6", "7"):
            with self.subTest(MIOPEN_LOG_LEVEL=level):
                run = self.run_probe(
                    ["conv_forward"], {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": level}
                )
                self.assertIn(
                    CONV_TRACE + "mode=enabled route=hipdnn reason=hipdnn-unavailable",
                    run.stderr_lines,
                )

    def test_trace_prints_once_per_entry_point(self):
        run = self.run_probe(
            ["conv_forward", "repeat", "2"], {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": "6"}
        )
        self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR] * 3)
        self.assertEqual(
            run.lines_starting(CONV_TRACE),
            [CONV_TRACE + "mode=enabled route=hipdnn reason=hipdnn-unavailable"],
            run.stderr,
        )

    def test_forwarded_failure_is_told_apart_until_a_later_call_succeeds(self):
        run = self.run_probe(
            [
                "conv_forward",
                "error_string",
                "last_forwarded",
                "get_version",
                "last_forwarded",
                "error_string",
            ],
            FORWARDING_ON,
        )
        self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR])
        self.assertEqual(run.statuses("get_version"), [SUCCESS])
        self.assertEqual(run.statuses("last_forwarded"), [INTERNAL_ERROR, SUCCESS])
        first, second = run.values("error_string")
        self.assertTrue(first.startswith(FORWARDED_PREFIX), first)
        self.assertEqual(second, first[len(FORWARDED_PREFIX) :])

    def test_null_is_rejected_and_reading_does_not_clear_the_failure(self):
        run = self.run_probe(
            ["conv_forward", "last_forwarded_null", "last_forwarded", "last_forwarded"],
            FORWARDING_ON,
        )
        self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR])
        self.assertEqual(run.statuses("last_forwarded_null"), [BAD_PARM])
        self.assertEqual(
            run.statuses("last_forwarded"), [INTERNAL_ERROR, INTERNAL_ERROR]
        )

    def test_forwarding_off_never_loads_hipdnn(self):
        run = self.run_probe(
            ["conv_forward", "last_forwarded", "get_version", "last_forwarded"],
        )
        self.assertFalse(run.loaded_fake_backend(), run.stderr)
        self.assertEqual(
            [line for line in run.stderr_lines if "hipDNN" in line], [], run.stderr
        )
        # MIOpen rejects the null handle itself.
        self.assertEqual(run.statuses("conv_forward"), [BAD_PARM])
        self.assertEqual(run.statuses("last_forwarded"), [SUCCESS, SUCCESS])

    def test_disabling_the_entry_point_keeps_it_on_miopen(self):
        run = self.run_probe(
            ["conv_forward"],
            {
                **FORWARDING_ON,
                "MIOPEN_DISABLE_HIPDNN_FOR": "miopenConvolutionForward",
                "MIOPEN_LOG_LEVEL": "6",
            },
        )
        self.assertIn(
            CONV_TRACE + "mode=enabled route=miopen reason=disabled-by-env",
            run.stderr_lines,
        )
        self.assertFalse(run.loaded_fake_backend(), run.stderr)
        self.assertEqual(run.statuses("conv_forward"), [BAD_PARM])

    def test_disabling_everything_keeps_it_on_miopen(self):
        run = self.run_probe(
            ["conv_forward"],
            {**FORWARDING_ON, "MIOPEN_DISABLE_HIPDNN_FOR": "*"},
        )
        self.assertFalse(run.loaded_fake_backend(), run.stderr)
        self.assertEqual(run.statuses("conv_forward"), [BAD_PARM])

    def test_levels_that_hide_miopens_errors_hide_the_failure_lines(self):
        # MIOpen narrows the level to int, so -1 reads as negative and hides
        # errors too.
        for level in ("1", "2", "-1"):
            with self.subTest(MIOPEN_LOG_LEVEL=level):
                run = self.run_probe(
                    ["conv_forward"],
                    {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": level},
                )
                self.assertTrue(run.loaded_fake_backend(), run.stderr)
                self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR])
                self.assertEqual(run.lines_starting(UNAVAILABLE), [], run.stderr)
                self.assertEqual(run.lines_starting(FORWARDED_FAILURE), [], run.stderr)

    def test_trace_with_forwarding_off(self):
        run = self.run_probe(["get_version"], env={"MIOPEN_LOG_LEVEL": "6"})
        self.assertIn(
            "[MIOpen] hipDNN routing: miopenGetVersion mode=disabled route=miopen "
            "reason=forwarding-off",
            run.stderr_lines,
        )

    def test_trace_for_an_entry_point_this_build_does_not_forward(self):
        run = self.run_probe(
            ["get_version"], {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": "6"}
        )
        self.assertIn(
            "[MIOpen] hipDNN routing: miopenGetVersion mode=enabled route=miopen "
            "reason=not-in-forwarding-set",
            run.stderr_lines,
        )

    def test_api_logging_lists_every_forwarded_call(self):
        run = self.run_probe(
            ["conv_forward", "repeat", "2"],
            {**FORWARDING_ON, "MIOPEN_ENABLE_LOGGING": "1"},
        )
        self.assertEqual(run.lines_starting(FORWARDED_CALL), [FORWARDED_CALL] * 3)

    def test_forwarded_calls_are_not_listed_without_api_logging(self):
        run = self.run_probe(["conv_forward"], FORWARDING_ON)
        self.assertEqual(run.lines_starting(FORWARDED_CALL), [], run.stderr)

    def test_destroy_returns_when_the_backend_calls_back_into_miopen(self):
        if self.reentrant_backend is None:
            self.skipTest(f"set {REENTRANT_BACKEND_ENV}")
        run = self.run_probe(
            ["destroy_after_forward"],
            FORWARDING_ON,
            backend=self.reentrant_backend,
            timeout=60,
        )
        if run.statuses("create") != [SUCCESS]:
            self.skipTest(f"miopenCreate failed, so no GPU\n{run.stderr}")
        self.assertEqual(run.statuses("destroy"), [SUCCESS], run.stderr)
        # Two hipDNN handles are destroyed: the one hipDNN's availability check makes,
        # and the one the forwarded call gave the probe's handle. Fewer means the
        # forwarded call never got that far, and miopenDestroy had nothing to re-enter.
        self.assertEqual(
            run.lines_starting(DESTROYED_MARKER),
            [f"{DESTROYED_MARKER}{SUCCESS}"] * 2,
            run.stderr,
        )

    def test_exit_destroys_no_hipdnn_handle_a_live_miopen_handle_holds(self):
        if self.reentrant_backend is None:
            self.skipTest(f"set {REENTRANT_BACKEND_ENV}")
        # run_probe fails a run that crashes at exit, as it fails any nonzero exit.
        run = self.run_probe(
            ["exit_after_forward"],
            FORWARDING_ON,
            backend=self.reentrant_backend,
            timeout=60,
        )
        if run.statuses("create") != [SUCCESS]:
            self.skipTest(f"miopenCreate failed, so no GPU\n{run.stderr}")
        self.assertEqual(len(run.statuses("forward_with_handle")), 1, run.stderr)
        # Only the availability check's hipDNN handle is destroyed. The one the
        # forwarded call gave the probe's handle is left to the operating system: a
        # second marker means exit destroyed it, re-entering miopenDestroy while
        # MIOpen's handle map was being torn down. The destroy case above shows the
        # same call does give the handle one, so one marker is not a vacuous pass.
        self.assertEqual(
            run.lines_starting(DESTROYED_MARKER),
            [f"{DESTROYED_MARKER}{SUCCESS}"],
            run.stderr,
        )

    def test_a_failed_stream_set_returns_when_the_backend_calls_back_into_miopen(self):
        if self.reentrant_backend is None:
            self.skipTest(f"set {REENTRANT_BACKEND_ENV}")
        run = self.run_probe(
            ["destroy_after_forward_on_stream"],
            {**FORWARDING_ON, "REENTRANT_HIPDNN_BACKEND_FAIL_SET_STREAM": "1"},
            backend=self.reentrant_backend,
            timeout=60,
        )
        if run.statuses("create") != [SUCCESS]:
            self.skipTest(
                f"no stream or miopenCreateWithStream failed, so no GPU\n{run.stderr}"
            )
        self.assertEqual(
            run.statuses("forward_with_handle"), [INTERNAL_ERROR], run.stderr
        )
        self.assertTrue(
            any(
                line.endswith("could not create a hipDNN handle")
                for line in run.lines_starting(FORWARDED_FAILURE)
            ),
            run.stderr,
        )
        self.assertEqual(run.statuses("destroy"), [SUCCESS], run.stderr)
        # Two hipDNN handles are destroyed: the availability check's, and the one the
        # forwarded call created and could not set the stream on. The probe's handle
        # never kept one, so miopenDestroy destroys none.
        self.assertEqual(
            run.lines_starting(DESTROYED_MARKER),
            [f"{DESTROYED_MARKER}{SUCCESS}"] * 2,
            run.stderr,
        )


if __name__ == "__main__":
    unittest.main()
