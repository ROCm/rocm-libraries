# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CTest: both engines refuse the same configs of a parity family, with the
same message.

The byte-identity gate reads a config both engines reject as BOTH_REJECTED
without comparing why, so a refusal reworded, or raised for another reason, in
one engine passes it. This runner takes a prebuilt native emitter and its
Python twin, runs every config in the ``ir`` mode until both report the end of
the list, and classifies each pair with the gate's own rules (``run_diff``).
Any verdict other than IDENTICAL or BOTH_REJECTED fails, and a BOTH_REJECTED
config must carry the same message: the native ``builder error: <message>``
line against the Python ``<ExceptionType>: <message>`` traceback line. The
number of refused configs is pinned, so a refusal case that starts to build in
both engines cannot leave the check comparing nothing.

CMake supplies the paths; this runner never compiles.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import run_diff

# The native emitters report a builder refusal on stderr behind this prefix.
NATIVE_REFUSAL_PREFIX = "builder error: "


def _emission(command: list[str]) -> run_diff.Emission:
    result = subprocess.run(
        command, capture_output=True, timeout=run_diff.TIMEOUT, check=False
    )
    return run_diff.Emission(
        result.returncode,
        bool(result.stdout),
        run_diff.sh(result.stdout) if result.stdout else None,
        result.stderr.decode("utf-8", "replace"),
    )


def native_message(emission: run_diff.Emission) -> str:
    """The message of a native refusal."""
    line = run_diff.last_error_line(emission)
    if not line.startswith(NATIVE_REFUSAL_PREFIX):
        raise AssertionError(f"native emitter failed, not refused: {line!r}")
    return line.removeprefix(NATIVE_REFUSAL_PREFIX)


def python_message(emission: run_diff.Emission) -> str:
    """The message of a Python refusal (its exception line)."""
    line = run_diff.last_error_line(emission)
    exception_type, _, message = line.partition(": ")
    if not message or not exception_type.endswith("Error"):
        raise AssertionError(f"Python emitter failed, not refused: {line!r}")
    return message


def compare_refusals(
    native: Path, launcher: Path, emitter: Path, expected_refusals: int
) -> int:
    """Compare every config's outcome; return the number of configs compared."""
    for path in (native, launcher, emitter):
        if not path.is_file():
            raise FileNotFoundError(f"refusal comparison artifact is missing: {path}")
    python_prefix = [sys.executable, "-S", str(launcher), str(emitter)]
    refused = 0
    for index in range(run_diff.MAX_CFG):
        native_emission = _emission([str(native), str(index), "ir"])
        python_emission = _emission([*python_prefix, str(index), "ir"])
        verdict, _ = run_diff.classify_emissions(native_emission, python_emission)
        if verdict == "END":
            break
        if verdict == "BOTH_REJECTED":
            native_text = native_message(native_emission)
            python_text = python_message(python_emission)
            if native_text != python_text:
                raise AssertionError(
                    f"config {index}: native refusal {native_text!r} != "
                    f"Python refusal {python_text!r}"
                )
            refused += 1
        elif verdict != "IDENTICAL":
            raise AssertionError(f"config {index}: the engines disagree ({verdict})")
    else:
        raise AssertionError(f"no end of the config list within {run_diff.MAX_CFG}")
    if refused != expected_refusals:
        raise AssertionError(
            f"{refused} configs refused by both engines, expected {expected_refusals}"
        )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--emitter", type=Path, required=True)
    parser.add_argument("--expected-refusals", type=int, required=True)
    args = parser.parse_args(argv)
    count = compare_refusals(
        args.native, args.launcher, args.emitter, args.expected_refusals
    )
    print(
        f"PASS: {count} configs, {args.expected_refusals} refused by both engines "
        "with the same message"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
