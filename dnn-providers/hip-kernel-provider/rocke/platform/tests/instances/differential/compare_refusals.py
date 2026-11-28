# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CTest: both engines refuse the same configs of a parity family, with the
same message.

The byte-identity gate reads a config both engines reject as BOTH_REJECTED
without comparing why, so a refusal reworded, or raised for another reason, in
one engine passes it. This runner takes a prebuilt native emitter and its
Python twin, runs every config in the ``ir`` mode until both report the end of
the list, and classifies each pair with the gate's own rules (``run_diff``).
Any verdict other than IDENTICAL or BOTH_REJECTED fails, and refusals are
compared (``run_diff.classify_emissions`` with ``compare_refusals``): a config
both engines reject must carry the same message, or it is REFUSAL_MISMATCH. The
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
        verdict, _ = run_diff.classify_emissions(
            native_emission, python_emission, compare_refusals=True
        )
        if verdict == "END":
            break
        if verdict == "REFUSAL_MISMATCH":
            raise AssertionError(
                f"config {index}: native refusal "
                f"{run_diff.last_error_line(native_emission)!r} != Python refusal "
                f"{run_diff.last_error_line(python_emission)!r}"
            )
        if verdict == "BOTH_REJECTED":
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
