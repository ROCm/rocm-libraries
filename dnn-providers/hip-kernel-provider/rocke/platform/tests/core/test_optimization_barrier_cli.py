# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CTest compares the real C builder with a selected Python HIP emitter.

The serialized-IR binding checks lowerer parity, not the C emitter's builder-call
ordering. This runner consumes a prebuilt C binary; it never compiles during a
test or infers artifacts from a source checkout. CMake supplies relocatable paths.

With ``--rocke-shard-from-gtest-env`` it compares only its CI shard's share
(``rocke.tests.ci_shard``, imported from the package on ``PYTHONPATH``): each configuration and the range-boundary check fall
in exactly one shard, so the shards of a run compare everything once, and a run
without the shard variables compares everything.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import os
import subprocess
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from rocke.tests.ci_shard import (
    Shard,
    add_shard_argument,
    header_line,
    shard_from_arguments,
)

# The shard key of the check that both emitters refuse the first index past the
# advertised range.
RANGE_BOUNDARY_KEY = "range boundary"


def _invoke(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


def config_key(index: int) -> str:
    """The shard key of HIP configuration ``index``."""
    return f"hip config {index}"


def comparison_keys(count: int) -> list[str]:
    """Every shard key of a comparison of ``count`` configurations."""
    return [*(config_key(index) for index in range(count)), RANGE_BOUNDARY_KEY]


def owned_checks(
    count: int, owns: Callable[[str], bool] | None
) -> tuple[list[int], bool]:
    """The configuration indices and whether the range-boundary check that a run
    owning the keys ``owns`` accepts performs; ``None`` owns every key."""
    indices = [
        index for index in range(count) if owns is None or owns(config_key(index))
    ]
    return indices, owns is None or owns(RANGE_BOUNDARY_KEY)


def compare_emitters(
    native: Path,
    launcher: Path,
    emitter: Path,
    count: int,
    owns: Callable[[str], bool] | None = None,
) -> None:
    """Compare the advertised HIP configurations and verify the range boundary.

    ``owns`` selects the shard keys (:func:`comparison_keys`) this run checks;
    ``None`` checks them all.
    """
    if count <= 0:
        raise ValueError("emitter comparison count must be positive")
    for path in (native, launcher, emitter):
        if not path.is_file():
            raise FileNotFoundError(
                f"required emitter comparison artifact is missing: {path}"
            )
    python_prefix = [sys.executable, "-S", str(launcher), str(emitter)]

    indices, boundary = owned_checks(count, owns)
    if not indices and not boundary:
        print("PASS: no HIP emitter check falls in this shard")
        return
    for index in indices:
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
    if boundary:
        # An omitted/new configuration must not silently leave this gate green.
        expected = f"unknown config index {count}"
        for command in (
            [*python_prefix, str(count), "hip"],
            [str(native), str(count), "hip"],
        ):
            result = _invoke(command)
            if (
                result.returncode <= 0
                or result.stdout
                or result.stderr.strip() != expected
            ):
                raise AssertionError(f"HIP emitter range boundary differs: {result}")
    print(
        f"PASS: {len(indices)} of {count} C/Python HIP emitter configurations"
        + (" and the range boundary" if boundary else "")
    )


class TestEmitterComparisonAdmission(unittest.TestCase):
    def test_empty_comparison_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count must be positive"):
            compare_emitters(Path("missing"), Path("missing"), Path("missing"), 0)

    def test_missing_prebuilt_artifact_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "artifact is missing"):
            compare_emitters(
                Path(__file__) / "missing", Path(__file__), Path(__file__), 1
            )


class TestEmitterComparisonSharding(unittest.TestCase):
    def test_shards_of_a_run_check_every_key_exactly_once(self):
        count = 17
        checked = []
        for index in range(4):
            indices, boundary = owned_checks(count, Shard(f"{index}/4").owns)
            checked += [config_key(config) for config in indices]
            checked += [RANGE_BOUNDARY_KEY] if boundary else []
        self.assertEqual(sorted(checked), sorted(comparison_keys(count)))

    def test_a_run_without_a_shard_checks_everything(self):
        self.assertEqual(owned_checks(17, None), (list(range(17)), True))

    def test_the_flag_takes_the_shard_and_reports_its_split(self):
        output = io.StringIO()
        missing = Path(__file__).with_name("missing")
        argv = ["--native", str(missing), "--launcher", str(missing)]
        argv += ["--emitter", str(missing), "--count", "17"]
        argv += ["--rocke-shard-from-gtest-env"]
        with (
            mock.patch.dict(os.environ, {"ROCKE_TEST_SHARD": "0/1"}),
            contextlib.redirect_stdout(output),
            self.assertRaisesRegex(FileNotFoundError, "artifact is missing"),
        ):
            main(argv)
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "rocke-ctest-shard: 0/1 (ROCKE_TEST_SHARD=0/1)")
        self.assertEqual(lines[1], "rocke-ctest-shard 0/1: split 18 of 18; kept 18")

    def test_a_malformed_shard_is_a_usage_error(self):
        missing = Path(__file__).with_name("missing")
        argv = ["--native", str(missing), "--launcher", str(missing)]
        argv += ["--emitter", str(missing), "--count", "17"]
        argv += ["--rocke-shard-from-gtest-env"]
        with (
            mock.patch.dict(os.environ, {"ROCKE_TEST_SHARD": "4/4"}),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exit_info,
        ):
            main(argv)
        self.assertEqual(exit_info.exception.code, 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--emitter", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    add_shard_argument(parser, help="compare only this CI shard's configurations")
    args = parser.parse_args(argv)
    shard = shard_from_arguments(parser, args)
    if args.shard_from_gtest_env:
        print(header_line(shard))
    owns = None
    if shard is not None:
        print(shard.split_line(comparison_keys(args.count)))
        owns = shard.owns
    compare_emitters(args.native, args.launcher, args.emitter, args.count, owns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
