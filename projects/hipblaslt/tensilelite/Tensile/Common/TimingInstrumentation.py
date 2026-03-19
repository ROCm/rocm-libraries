################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import logging
import sys
import time
from contextlib import contextmanager

from Tensile.Common.GlobalParameters import globalParameters

_timing_logger = logging.getLogger("tensile.timing")
if not _timing_logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _timing_logger.addHandler(_h)
    _timing_logger.setLevel(logging.INFO)
    _timing_logger.propagate = False

# Deferred I/O buffer: list of (category, duration_ms) tuples.
# All formatting and I/O happens in flush_timing_buffer().
_timing_buffer = []

# Calibrated per-invocation overhead of timing_context (nanoseconds).
# Covers context-manager protocol, clock calls, buffer append.
_per_call_overhead_ns = 0

# Total number of timing_context completions. Used to count child invocations
# within a parent scope for overhead subtraction.
_invocation_count = 0


def calibrate_timing(iterations=10000):
    """Measure the full per-invocation cost of timing_context.

    Must be called after globalParameters["TimingInstrumentation"] is set to True.
    """
    global _per_call_overhead_ns, _invocation_count
    if not globalParameters.get("TimingInstrumentation", False):
        return
    # Warmup — let CPython's adaptive specialization settle
    for _ in range(100):
        with timing_context("calibrate_python_timing_overhead"):
            pass
    buf_before = len(_timing_buffer)
    count_before = _invocation_count
    start = time.time_ns()
    for _ in range(iterations):
        with timing_context("calibrate_python_timing_overhead"):
            pass
    total = time.time_ns() - start
    # Remove calibration entries from buffer and counter
    del _timing_buffer[buf_before:]
    _invocation_count = count_before
    _per_call_overhead_ns = total // iterations


@contextmanager
def timing_context(category_name):
    """Context manager for timing instrumentation."""
    if globalParameters.get("TimingInstrumentation", False):
        global _invocation_count
        count_snapshot = _invocation_count
        start = time.time_ns()
        try:
            yield
        finally:
            elapsed_ns = time.time_ns() - start
            child_invocations = _invocation_count - count_snapshot
            adjusted_ns = elapsed_ns - child_invocations * _per_call_overhead_ns
            _timing_buffer.append((category_name, adjusted_ns / 1_000_000))
            _invocation_count += 1
    else:
        yield


def flush_timing_buffer():
    """Write all buffered timing records and reset."""
    global _invocation_count
    for category, duration_ms in _timing_buffer:
        _timing_logger.info(f"TIMING:{category}:{duration_ms:.3f}")
    _timing_buffer.clear()
    _invocation_count = 0
