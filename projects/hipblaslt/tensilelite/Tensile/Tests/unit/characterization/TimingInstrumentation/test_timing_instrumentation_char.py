################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.TimingInstrumentation.timing_context``
— the timing on/off branches gated by the process-global
``globalParameters["TimingInstrumentation"]``."""

import contextlib

import pytest

from Tensile.Common.GlobalParameters import globalParameters
from Tensile.Common.TimingInstrumentation import flush_timing_buffer, timing_context

pytestmark = pytest.mark.unit


@contextlib.contextmanager
def _timing_flag(value):
    saved = globalParameters.get("TimingInstrumentation", None)
    globalParameters["TimingInstrumentation"] = value
    try:
        yield
    finally:
        if saved is None:
            globalParameters.pop("TimingInstrumentation", None)
        else:
            globalParameters["TimingInstrumentation"] = saved


def test_timing_context_disabled_yields():
    # Flag off -> the plain `else: yield` path; body runs, nothing logged.
    with _timing_flag(False):
        ran = []
        with timing_context("cat"):
            ran.append(1)
        assert ran == [1]


def test_timing_context_enabled_times_and_logs():
    # Flag on -> the timing path; body runs and a TIMING line is buffered until
    # flush_timing_buffer() emits it. The "tensile.timing" logger has
    # propagate=False, so attach a capture handler directly to it (caplog/capsys
    # can't see it).
    import logging

    messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = logging.getLogger("tensile.timing")
    handler = _Capture()
    flush_timing_buffer()
    logger.addHandler(handler)
    try:
        with _timing_flag(True):
            ran = []
            with timing_context("mycat"):
                ran.append(1)
            assert ran == [1]
            assert not any("TIMING:mycat:" in m for m in messages)
            flush_timing_buffer()
    finally:
        logger.removeHandler(handler)
        flush_timing_buffer()
    assert any("TIMING:mycat:" in m for m in messages)
