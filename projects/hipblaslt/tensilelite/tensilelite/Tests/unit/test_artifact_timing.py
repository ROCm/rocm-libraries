# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for the artifact packing/extraction timers in Tests/common/artifact_helpers.py
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from artifact_helpers import artifact_timing, timing_requested  # noqa: E402

pytestmark = pytest.mark.unit


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def timing_lines():
    """The TIMING lines written to the tensile.timing logger during the test."""
    # Import first: TimingInstrumentation sets the logger up (INFO level) only if it has no
    # handlers yet, so attaching ours before its first import would leave the logger at WARNING.
    import tensilelite.Common.TimingInstrumentation  # noqa: F401
    handler = _Records()
    logger = logging.getLogger("tensile.timing")
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


@pytest.mark.parametrize("args, expected", [
    ([], False),
    (["--global-parameters", "ClientLogLevel=0"], False),
    (["--global-parameters", "ClientLogLevel=0", "--global-parameters", "TimingInstrumentation=True"], True),
    (["--cxx-compiler", "/usr/bin/amdclang++", "--global-parameters", "TimingInstrumentation=True"], True),
    (["--global-parameters", "ClientLogLevel=0", "TimingInstrumentation=True"], True),
    (["--global-parameters", "TimingInstrumentation=True", "--global-parameters", "TimingInstrumentation=False"], False),
    (["--global-parameters", "TimingInstrumentation=1"], True),
    # Tensile reads the value with ast.literal_eval and uses it as a flag.
    (["--global-parameters", "TimingInstrumentation=2"], True),
    (["--global-parameters", "TimingInstrumentation=0"], False),
    (["--global-parameters", "TimingInstrumentation=False"], False),
    (["--global-parameters", "TimingInstrumentation=None"], False),
    # Not a literal: Tensile rejects it.
    (["--global-parameters", "TimingInstrumentation=yes"], False),
    # Not a --global-parameters value.
    (["--prebuilt-client", "TimingInstrumentation=True"], False),
])
def test_timing_requested_follows_global_parameters(args, expected):
    assert timing_requested(args) is expected


def test_artifact_timing_prints_a_timing_record_when_requested(timing_lines):
    with artifact_timing("python_artifact_compress", ["--global-parameters", "TimingInstrumentation=True"]):
        pass
    lines = [l for l in timing_lines if l.startswith("TIMING:python_artifact_compress:")]
    assert len(lines) == 1
    assert float(lines[0].rsplit(":", 1)[1]) >= 0.0


def test_artifact_timing_prints_nothing_when_not_requested(timing_lines):
    with artifact_timing("python_artifact_compress", ["--global-parameters", "ClientLogLevel=0"]):
        pass
    assert timing_lines == []


def test_artifact_timing_records_even_when_the_block_raises(timing_lines):
    with pytest.raises(RuntimeError):
        with artifact_timing("python_artifact_extract", ["--global-parameters", "TimingInstrumentation=True"]):
            raise RuntimeError("boom")
    assert any(l.startswith("TIMING:python_artifact_extract:") for l in timing_lines)
