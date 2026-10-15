#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How the MIOpen wrapper routes, traces and reports a hipDNN that cannot be used.

Every switch involved is read once per process: MIOPEN_HIPDNN_FORWARDING,
MIOPEN_DISABLE_HIPDNN_FOR, MIOPEN_LOG_LEVEL, MIOPEN_ENABLE_LOGGING and whether hipDNN
loaded. So each case runs forwarding_failure_probe in a new process, with every MIOPEN_*
and HIPDNN_* variable removed and only the case's own set.

hipDNN is made unusable with fake backends that it loads and then refuses: wrong_major
reports a major version no frontend has, and no_version has no version at all. A missing
backend cannot be simulated: when the override directory has none, hipDNN falls back to
the real one next to libMIOpen.so, then to the loader's search path. Both fakes take the
same MIOpen path a missing backend would.

ctest sets MIOPEN_FAILURE_PROBE (the probe) and MIOPEN_FAKE_HIPDNN_BACKENDS (a directory
with one subdirectory per fake backend). Without them every case is skipped.

Uses unittest rather than pytest, like the other wrapper script tests, because nothing
installs pytest on a machine that builds MIOpen.

    python3 -m unittest test_forwarding_failure_modes
"""

import os
import subprocess
import unittest
from pathlib import Path

PROBE_ENV = "MIOPEN_FAILURE_PROBE"
BACKENDS_ENV = "MIOPEN_FAKE_HIPDNN_BACKENDS"

# miopenStatus_t values the probe prints.
SUCCESS = 0
BAD_PARM = 3
INTERNAL_ERROR = 5

LOADED_MARKER = "FAKE_HIPDNN_BACKEND_LOADED:"
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

    def loaded_backends(self):
        return [
            line[len(LOADED_MARKER) :]
            for line in self.stderr_lines
            if line.startswith(LOADED_MARKER)
        ]


class ForwardingFailureModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = os.environ.get(PROBE_ENV)
        backends = os.environ.get(BACKENDS_ENV)
        if not probe or not backends:
            raise unittest.SkipTest(
                f"set {PROBE_ENV} and {BACKENDS_ENV}; ctest does this in a build "
                "with MIOPEN_ENABLE_HIPDNN_WRAPPER=ON"
            )
        # hipDNN ignores a relative HIPDNN_BACKEND_LIBRARY_PATH, and the installed test
        # list passes paths relative to its own directory.
        cls.probe = Path(probe).resolve()
        cls.backends = Path(backends).resolve()

    def run_probe(self, steps, backend=None, env=None):
        clean = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("MIOPEN_", "HIPDNN_"))
        }
        if backend is not None:
            clean["HIPDNN_BACKEND_LIBRARY_PATH"] = str(self.backends / backend)
        clean.update(env or {})
        completed = subprocess.run(
            [str(self.probe), *steps],
            env=clean,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"probe failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return ProbeRun(completed)

    def assert_refused_backend_fails_each_call(self, backend):
        run = self.run_probe(["conv_forward", "repeat", "2"], backend, FORWARDING_ON)
        self.assertEqual(run.loaded_backends(), [backend], run.stderr)
        self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR] * 3)
        # hipDNN's availability is checked once per process, so the reason prints once.
        self.assertEqual(len(run.lines_starting(UNAVAILABLE)), 1, run.stderr)
        self.assertEqual(len(run.lines_starting(FORWARDED_FAILURE)), 3, run.stderr)

    def test_wrong_major_fails_every_forwarded_call(self):
        self.assert_refused_backend_fails_each_call("wrong_major")

    def test_backend_without_a_version_fails_every_forwarded_call(self):
        self.assert_refused_backend_fails_each_call("no_version")

    def test_trace_names_the_unavailable_backend_as_the_reason(self):
        run = self.run_probe(
            ["conv_forward"], "wrong_major", {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": "6"}
        )
        self.assertIn(
            CONV_TRACE + "mode=enabled route=hipdnn reason=hipdnn-unavailable",
            run.stderr_lines,
        )

    def test_trace_is_on_above_level_6_too(self):
        """Level 7 (Trace) includes everything level 6 prints."""
        run = self.run_probe(
            ["conv_forward"], "wrong_major", {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": "7"}
        )
        self.assertIn(
            CONV_TRACE + "mode=enabled route=hipdnn reason=hipdnn-unavailable",
            run.stderr_lines,
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
            "wrong_major",
            FORWARDING_ON,
        )
        self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR])
        self.assertEqual(run.statuses("get_version"), [SUCCESS])
        self.assertEqual(run.statuses("last_forwarded"), [INTERNAL_ERROR, SUCCESS])
        first, second = run.values("error_string")
        self.assertTrue(first.startswith(FORWARDED_PREFIX), first)
        self.assertEqual(second, first[len(FORWARDED_PREFIX) :])

    def test_forwarding_off_never_loads_hipdnn(self):
        run = self.run_probe(
            ["conv_forward", "last_forwarded", "get_version", "last_forwarded"],
            "wrong_major",
        )
        self.assertEqual(run.loaded_backends(), [], run.stderr)
        self.assertEqual(
            [line for line in run.stderr_lines if "hipDNN" in line], [], run.stderr
        )
        # MIOpen rejects the null handle itself.
        self.assertEqual(run.statuses("conv_forward"), [BAD_PARM])
        self.assertEqual(run.statuses("last_forwarded"), [SUCCESS, SUCCESS])

    def test_disabling_the_entry_point_keeps_it_on_miopen(self):
        run = self.run_probe(
            ["conv_forward"],
            "wrong_major",
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
        self.assertEqual(run.loaded_backends(), [], run.stderr)
        self.assertEqual(run.statuses("conv_forward"), [BAD_PARM])

    def test_disabling_everything_keeps_it_on_miopen(self):
        run = self.run_probe(
            ["conv_forward"],
            "wrong_major",
            {**FORWARDING_ON, "MIOPEN_DISABLE_HIPDNN_FOR": "*"},
        )
        self.assertEqual(run.loaded_backends(), [], run.stderr)
        self.assertEqual(run.statuses("conv_forward"), [BAD_PARM])

    def test_quiet_and_fatal_levels_hide_the_failure_line(self):
        for level in ("1", "2"):
            with self.subTest(MIOPEN_LOG_LEVEL=level):
                run = self.run_probe(
                    ["conv_forward"],
                    "wrong_major",
                    {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": level},
                )
                self.assertEqual(run.statuses("conv_forward"), [INTERNAL_ERROR])
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
            ["get_version"], "wrong_major", {**FORWARDING_ON, "MIOPEN_LOG_LEVEL": "6"}
        )
        self.assertIn(
            "[MIOpen] hipDNN routing: miopenGetVersion mode=enabled route=miopen "
            "reason=not-in-forwarding-set",
            run.stderr_lines,
        )

    def test_api_logging_lists_every_forwarded_call(self):
        run = self.run_probe(
            ["conv_forward", "repeat", "2"],
            "wrong_major",
            {**FORWARDING_ON, "MIOPEN_ENABLE_LOGGING": "1"},
        )
        self.assertEqual(run.lines_starting(FORWARDED_CALL), [FORWARDED_CALL] * 3)

    def test_forwarded_calls_are_not_listed_without_api_logging(self):
        run = self.run_probe(["conv_forward"], "wrong_major", FORWARDING_ON)
        self.assertEqual(run.lines_starting(FORWARDED_CALL), [], run.stderr)


if __name__ == "__main__":
    unittest.main()
