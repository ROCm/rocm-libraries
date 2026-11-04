# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CTest compares the real C builder with a selected Python HIP emitter.

The serialized-IR binding checks lowerer parity, not the C emitter's builder-call
ordering. This runner consumes a prebuilt C binary; it never compiles during a
test or infers artifacts from a source checkout. CMake supplies relocatable paths.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import subprocess
import sys
import unittest


def _invoke(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


def compare_emitters(native: Path, launcher: Path, emitter: Path, count: int) -> None:
    """Compare every advertised HIP configuration and verify the range boundary."""
    if count <= 0:
        raise ValueError("emitter comparison count must be positive")
    for path in (native, launcher, emitter):
        if not path.is_file():
            raise FileNotFoundError(
                f"required emitter comparison artifact is missing: {path}"
            )
    python_prefix = [sys.executable, "-S", str(launcher), str(emitter)]
    for index in range(count):
        python = _invoke([*python_prefix, str(index), "hip"])
        cpp = _invoke([str(native), str(index), "hip"])
        if python.returncode or cpp.returncode:
            raise RuntimeError(
                f"HIP emitter config {index} failed: Python={python.returncode}, "
                f"C={cpp.returncode}\n{python.stderr}{cpp.stderr}"
            )
        if python.stdout != cpp.stdout:
            diff = "".join(
                difflib.unified_diff(
                    python.stdout.splitlines(keepends=True),
                    cpp.stdout.splitlines(keepends=True),
                    fromfile="Python HIP",
                    tofile="C HIP",
                )
            )
            raise AssertionError(f"HIP emitter config {index} differs:\n{diff}")
    # An omitted/new configuration must not silently leave this gate green.
    expected = f"unknown config index {count}"
    for command in (
        [*python_prefix, str(count), "hip"],
        [str(native), str(count), "hip"],
    ):
        result = _invoke(command)
        if result.returncode <= 0 or result.stdout or result.stderr.strip() != expected:
            raise AssertionError(f"HIP emitter range boundary differs: {result}")
    print(f"PASS: {count} C/Python HIP emitter configurations and range boundary")


class TestEmitterComparisonAdmission(unittest.TestCase):
    def test_empty_comparison_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count must be positive"):
            compare_emitters(Path("missing"), Path("missing"), Path("missing"), 0)

    def test_missing_prebuilt_artifact_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "artifact is missing"):
            compare_emitters(
                Path(__file__) / "missing", Path(__file__), Path(__file__), 1
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--emitter", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(argv)
    compare_emitters(args.native, args.launcher, args.emitter, args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
