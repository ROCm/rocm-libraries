#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run one forwarded convolution through a hipDNN plugin built against the public MIOpen.

The wrapper must fail that call and say why, instead of recursing. This runs the shim
convolution test with forwarding on and hipDNN pointed at the stand-in plugin, which calls
the public miopenConvolutionForward from inside hipDNN, and looks for the wrapper's message.
The test itself is expected to fail.

    check_plugin_reentry.py --gtest <miopen_gtest> --stand-in-dir <dir with the plugin>

Exit status: 0 the message was printed, 1 it was not although this machine runs the test,
77 skipped (ctest's SKIP_RETURN_CODE).

The check needs a built test binary, a built stand-in, a GPU the shim test supports and a
usable hipDNN. Rather than guess which of these is missing, a run without the message is
repeated with hipDNN's default plugins. If that control run does not pass either, the machine
cannot run the test and the check is skipped; only a machine that passes the control and
still shows no message has a real failure (the guard is gone, or the call looped).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PASS, FAIL, SKIP = 0, 1, 77

MESSAGE = "a hipDNN plugin called the public MIOpen entry point"
DEFAULT_FILTER = "GPU_HipdnnShimConvFwdApi_FP32.FindAndForwardMatchCpuReference"
PASSED_LINE = "[  PASSED  ] 1 test"

# Lines of output shown when a run is reported.
TAIL_LINES = 25


def run_test(gtest, test_filter, env, timeout):
    """Return (finished, output). A run that exceeds `timeout` has not finished."""
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
        return False, partial
    return True, done.stdout


def tail(output):
    return "\n".join(output.splitlines()[-TAIL_LINES:])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--gtest", required=True, help="the gtest binary with the shim test"
    )
    parser.add_argument(
        "--stand-in-dir",
        required=True,
        help="directory holding only the stand-in plugin; hipDNN loads plugins from it",
    )
    parser.add_argument("--filter", default=DEFAULT_FILTER, help="the shim test to run")
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

    env = dict(os.environ)
    env["MIOPEN_HIPDNN_FORWARDING"] = "1"
    # A deny list or plugin directory from the caller would change what both runs test.
    env.pop("MIOPEN_DISABLE_HIPDNN_FOR", None)
    env.pop("HIPDNN_PLUGIN_DIR", None)
    control_env = dict(env)
    env["HIPDNN_PLUGIN_DIR"] = str(stand_in_dir)

    finished, output = run_test(gtest, args.filter, env, args.timeout)
    if MESSAGE in output:
        print("PASS: the wrapper failed the plugin's call and printed its reason")
        return PASS

    finished_control, control = run_test(gtest, args.filter, control_env, args.timeout)
    if not (finished_control and PASSED_LINE in control):
        print(
            "SKIP: this machine cannot run the shim convolution test with hipDNN's "
            "default plugins, so the check cannot tell a regression from a missing "
            "environment (no GPU, an unsupported GPU or no hipDNN). Control run:"
        )
        print(tail(control))
        return SKIP

    how = "did not finish" if not finished else "finished"
    print(
        f"FAIL: the run with the stand-in plugin {how} without the wrapper's message "
        "(the re-entry guard is missing, or the call looped), although the same test "
        "passes with hipDNN's default plugins. Output of the stand-in run:"
    )
    print(tail(output))
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
