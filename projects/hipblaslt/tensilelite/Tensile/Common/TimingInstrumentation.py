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
# Raw timings are stored without overhead adjustment; the analysis script
# (analyze_timing.py) handles overhead subtraction using the hierarchy and
# the timing_overhead record emitted by the C++ client.
_timing_buffer = []


@contextmanager
def timing_context(category_name):
    """Context manager for timing instrumentation.

    Records raw wall-clock time with no overhead subtraction.  Python-side
    overhead (context-manager protocol, time.time_ns, dict lookup) is not
    tracked because there are only ~a dozen calls per run — the overhead
    is negligible relative to the seconds-scale measurements.  C++ overhead
    is tracked separately via a calibrated timing_overhead record.
    """
    if globalParameters.get("TimingInstrumentation", False):
        start = time.time_ns()
        try:
            yield
        finally:
            elapsed_ns = time.time_ns() - start
            _timing_buffer.append((category_name, elapsed_ns / 1_000_000))
    else:
        yield


def flush_timing_buffer():
    """Write all buffered timing records and reset."""
    for category, duration_ms in _timing_buffer:
        _timing_logger.info(f"TIMING:{category}:{duration_ms:.3f}")
    _timing_buffer.clear()
