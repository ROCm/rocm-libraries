#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run one forwarded convolution through a hipDNN plugin built against the public MIOpen.

The wrapper must fail that call and say why, instead of recursing, and the handle must still
be destroyable afterwards. This runs the re-entry test (test/gtest/hipdnn_reentry.cpp) with
forwarding on and hipDNN pointed at the stand-in plugin, which calls the public
miopenConvolutionForward from inside hipDNN. With the stand-in, the test passes only when the
convolution failed with miopenStatusInternalError, miopenGetLastForwardedError agrees and
miopenDestroy returned; this script also requires the wrapper's message in the output.

    check_plugin_reentry.py --gtest <test binary> --stand-in-dir <dir with the plugin>

Exit status: 0 passed, 1 failed, 77 skipped (ctest's SKIP_RETURN_CODE).

The check needs a built test binary, a built stand-in, a GPU the test supports and a usable
hipDNN. Rather than guess which of these is missing, a run that did not pass is repeated
with hipDNN's default plugins and no expectation of a rejection. If that control run does not
pass either, the machine cannot run the test and the check is skipped; only a machine that
passes the control and still fails with the stand-in has a real failure (the guard is gone,
the call looped or hung, or the handle could not be destroyed). A binary that has no such test
is a failure, not a skip: the control run would fail too and hide a renamed test.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PASS, FAIL, SKIP = 0, 1, 77

MESSAGE = "a hipDNN plugin called the public MIOpen entry point"
DEFAULT_FILTER = "GPU_HipdnnReentry_FP32.RejectedCallThenDestroyReturns"
EXPECT_REJECTION = "MIOPEN_TEST_EXPECT_REENTRY_REJECTION"
PASSED_LINE = "[  PASSED  ] 1 test"
NO_TEST_LINE = "Running 0 tests"

# Settings of the caller that would change what a run tests. A shard or filter variable can
# select no test, a deny list keeps the call on MIOpen, a plugin directory replaces the one
# under test, and a log level of 1 or 2 hides the wrapper's message.
CALLER_SETTINGS = (
    "GTEST_FILTER",
    "GTEST_SHARD_INDEX",
    "GTEST_TOTAL_SHARDS",
    "GTEST_ALSO_RUN_DISABLED_TESTS",
    "HIPDNN_PLUGIN_DIR",
    "MIOPEN_DISABLE_HIPDNN_FOR",
    "MIOPEN_LOG_LEVEL",
    EXPECT_REJECTION,
)

TAIL_LINES = 25


def run_test(gtest, test_filter, env, timeout):
    """Return (finished, status, output). A run that exceeds `timeout` has not finished and
    has no status (None)."""
    try:
        done = subprocess.run(
            [str(gtest), f"--gtest_filter={test_filter}"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        partial = expired.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return False, None, partial
    return True, done.returncode, done.stdout


def passed(finished, status, output):
    """The pass line alone is not enough: the process can still abort or hang afterwards."""
    return finished and status == 0 and PASSED_LINE in output


def tail(output):
    return "\n".join(output.splitlines()[-TAIL_LINES:])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gtest", required=True, help="the gtest binary with the test")
    parser.add_argument(
        "--stand-in-dir",
        required=True,
        help="directory holding only the stand-in plugin; hipDNN loads plugins from it",
    )
    parser.add_argument("--filter", default=DEFAULT_FILTER, help="the test to run")
    parser.add_argument(
        "--timeout", type=float, default=120, help="seconds allowed for each run"
    )
    args = parser.parse_args(argv)

    gtest = Path(args.gtest)
    stand_in_dir = Path(args.stand_in_dir)
    if not gtest.is_file():
        print(f"SKIP: {gtest} is not built")
        return SKIP
    if not stand_in_dir.is_dir() or not any(stand_in_dir.glob("*.so")):
        print(
            f"SKIP: no stand-in plugin in {stand_in_dir}; build miopen_plugin_public_stand_in"
        )
        return SKIP

    control_env = {k: v for k, v in os.environ.items() if k not in CALLER_SETTINGS}
    control_env["MIOPEN_HIPDNN_FORWARDING"] = "1"
    env = dict(control_env)
    # hipDNN resolves a relative plugin directory against its own library's directory, not
    # against this process's working directory, so only an absolute path finds the stand-in.
    env["HIPDNN_PLUGIN_DIR"] = str(stand_in_dir.resolve())
    env[EXPECT_REJECTION] = "1"

    finished, status, output = run_test(gtest, args.filter, env, args.timeout)
    if NO_TEST_LINE in output:
        print(
            f"FAIL: {args.filter} selects no test in {gtest}. Is the test renamed or removed?"
        )
        return FAIL
    if passed(finished, status, output) and MESSAGE in output:
        print(
            "PASS: the wrapper failed the plugin's call, printed its reason and the handle "
            "was destroyed"
        )
        return PASS

    finished_control, status_control, control = run_test(
        gtest, args.filter, control_env, args.timeout
    )
    if not passed(finished_control, status_control, control):
        print(
            "SKIP: this machine cannot run the re-entry test with hipDNN's default plugins, "
            "so the check cannot tell a regression from a missing environment (no GPU, an "
            "unsupported GPU or no hipDNN). Control run:"
        )
        print(tail(control))
        return SKIP

    if not finished:
        how = "did not finish (the call looped or hung)"
    elif passed(finished, status, output):
        how = "passed without the wrapper's message"
    elif PASSED_LINE in output:
        how = f"printed its pass line but then exited with status {status}"
    else:
        how = "failed"
    print(
        f"FAIL: the run with the stand-in plugin {how}, although the same test passes with "
        "hipDNN's default plugins. Output of the stand-in run:"
    )
    print(tail(output))
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
