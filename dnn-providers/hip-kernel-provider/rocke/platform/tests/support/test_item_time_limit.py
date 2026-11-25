# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``rocke.tests.item_time_limit`` pytest plugin.

Each case runs pytest with the plugin and pytest-timeout's --timeout in a
child process over a small generated test module, as an installed GPU entry
loads them, and reads the outcome. The cases that let a test run past its limit
use pytest-timeout's signal method, so the run goes on to the next test and
reports each outcome (the GPU entries' thread method ends the process at the
first stall); that method exists on POSIX only.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from rocke.tests.item_time_limit import (
    LIMIT_HEADROOM,
    MEASURED_SECONDS_MARKER,
    PLUGIN_NAME,
)
from rocke.tests.run_records import fixture_session_environment
from rocke.tests.subprocess_launcher import rocke_subprocess_env

pytest.importorskip("pytest_timeout", reason="the per-test limit needs pytest-timeout")

_CHILD_TIMEOUT_SECONDS = 120
_ENTRY_LIMIT_SECONDS = 1
_SIGNAL_METHOD = "--timeout-method=signal"
needs_signal_method = pytest.mark.skipif(
    sys.platform == "win32", reason="pytest-timeout's signal method is POSIX only"
)


def _run(tmp_path: Path, source: str, *arguments: str) -> subprocess.CompletedProcess:
    """pytest with the plugin over a module holding ``source``, run without
    ``ROCKE_RUN_RECORDS_DIR`` (``fixture_session_environment``): the child
    session's outcomes belong to this test, not to the run's records."""
    (tmp_path / "test_mod.py").write_text(
        "import time\nimport pytest\n" + textwrap.dedent(source)
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            PLUGIN_NAME,
            "-v",
            *arguments,
        ],
        cwd=tmp_path,
        env=rocke_subprocess_env(fixture_session_environment()),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )


@needs_signal_method
def test_a_measured_time_raises_only_its_own_tests_limit(tmp_path):
    """Two tests take 1.5 times the entry's limit: the one that declares its
    measured time gets twice that as its limit and passes; the other stops at
    the entry's limit."""
    sleep_seconds = 1.5 * _ENTRY_LIMIT_SECONDS
    assert LIMIT_HEADROOM * _ENTRY_LIMIT_SECONDS > sleep_seconds
    result = _run(
        tmp_path,
        f"""
        @pytest.mark.{MEASURED_SECONDS_MARKER}({_ENTRY_LIMIT_SECONDS})
        def test_declared():
            time.sleep({sleep_seconds})

        def test_undeclared():
            time.sleep({sleep_seconds})
        """,
        f"--timeout={_ENTRY_LIMIT_SECONDS}",
        _SIGNAL_METHOD,
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout[-3000:]
    assert "test_mod.py::test_declared PASSED" in result.stdout
    assert "test_mod.py::test_undeclared FAILED" in result.stdout
    assert "Timeout" in result.stdout


@pytest.mark.parametrize("mark_arguments", ["0.01", "seconds=0.01"])
def test_a_measured_time_never_lowers_the_entry_limit(tmp_path, mark_arguments):
    result = _run(
        tmp_path,
        f"""
        @pytest.mark.{MEASURED_SECONDS_MARKER}({mark_arguments})
        def test_declared_fast():
            time.sleep(0.3)
        """,
        f"--timeout={_ENTRY_LIMIT_SECONDS}",
    )
    assert result.returncode == pytest.ExitCode.OK, result.stdout[-3000:]


@pytest.mark.parametrize(
    "marks, message",
    [
        (["pytest.mark.timeout(5)"], "timeout mark - seconds=5, expected a positive"),
        (["pytest.mark.timeout(timeout=5)"], "timeout mark - seconds=5, expected"),
        (["pytest.mark.timeout(0)"], "timeout mark - seconds=0, expected"),
        (["pytest.mark.timeout(-1)"], "timeout mark - seconds=-1, expected"),
        (["pytest.mark.timeout(None)"], "timeout mark - seconds=None, expected"),
        (["pytest.mark.timeout('x')"], "timeout mark - seconds='x', expected"),
        (
            [f"pytest.mark.{MEASURED_SECONDS_MARKER}(3)", "pytest.mark.timeout(1)"],
            f"{MEASURED_SECONDS_MARKER} and timeout marks - seconds=3, timeout=1",
        ),
        ([f"pytest.mark.{MEASURED_SECONDS_MARKER}(0)"], "expected one positive"),
        ([f"pytest.mark.{MEASURED_SECONDS_MARKER}('3')"], "expected one positive"),
        (
            [f"pytest.mark.{MEASURED_SECONDS_MARKER}(minutes=3)"],
            "expected one positive",
        ),
        (
            [f"pytest.mark.{MEASURED_SECONDS_MARKER}(float('inf'))"],
            "expected one positive",
        ),
    ],
    ids=[
        "timeout-above-limit",
        "timeout-keyword-above-limit",
        "timeout-zero-turns-the-limit-off",
        "timeout-negative",
        "timeout-none",
        "timeout-text",
        "both-marks",
        "zero",
        "text",
        "unknown-keyword",
        "infinite",
    ],
)
def test_a_raise_other_than_a_measured_time_is_a_usage_error(tmp_path, marks, message):
    decorators = "".join(f"@{mark}\n" for mark in marks)
    result = _run(
        tmp_path,
        f"{decorators}def test_marked():\n    pass\n",
        f"--timeout={_ENTRY_LIMIT_SECONDS}",
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout[-3000:]
    assert "test_mod.py::test_marked" in result.stderr
    assert message in result.stderr


@needs_signal_method
def test_a_timeout_mark_may_tighten_the_entry_limit(tmp_path):
    result = _run(
        tmp_path,
        """
        @pytest.mark.timeout(0.5)
        def test_tight():
            time.sleep(0.8)
        """,
        f"--timeout={_ENTRY_LIMIT_SECONDS}",
        _SIGNAL_METHOD,
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout[-3000:]
    assert "Timeout" in result.stdout


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="calls libc sleep")
def test_the_thread_method_waits_for_a_native_call_that_holds_the_gil(tmp_path):
    """The limit the GPU entries' comment states: pytest-timeout's thread
    method needs the GIL to fire, so a native call that holds it runs past the
    limit, and the method ends the process only once the call returns."""
    native_seconds = 3
    started = time.monotonic()
    result = _run(
        tmp_path,
        f"""
        import ctypes

        def test_holds_the_gil():
            ctypes.PyDLL(None).sleep({native_seconds})
        """,
        f"--timeout={_ENTRY_LIMIT_SECONDS}",
        "--timeout-method=thread",
    )
    assert time.monotonic() - started >= native_seconds
    assert result.returncode != pytest.ExitCode.OK, result.stdout[-3000:]
    assert "Timeout" in result.stdout + result.stderr


def test_the_plugin_needs_an_entry_limit(tmp_path):
    result = _run(tmp_path, "def test_any():\n    pass\n")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout[-3000:]
    assert f"{PLUGIN_NAME}: --timeout - value=None, expected" in result.stderr
