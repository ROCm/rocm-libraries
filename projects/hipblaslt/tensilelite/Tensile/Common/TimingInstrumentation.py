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

# Accumulated overhead of the timing instrumentation itself (in nanoseconds).
# Tracks f-string formatting and logger.info calls — excludes time.time_ns().
_timing_overhead_ns = 0


@contextmanager
def timing_context(category_name):
    """Context manager for timing instrumentation."""
    if globalParameters.get("TimingInstrumentation", False):
        global _timing_overhead_ns
        overhead_snapshot = _timing_overhead_ns
        start = time.time_ns()
        try:
            yield
        finally:
            elapsed_ns = time.time_ns() - start
            # Subtract overhead accumulated by child timers during our span
            child_overhead_ns = _timing_overhead_ns - overhead_snapshot
            adjusted_ms = (elapsed_ns - child_overhead_ns) / 1_000_000
            t0 = time.time_ns()
            _timing_logger.info(f"TIMING:{category_name}:{adjusted_ms:.3f}")
            t1 = time.time_ns()
            _timing_overhead_ns += t1 - t0
    else:
        yield


def emit_timing_overhead():
    """Emit accumulated Python timing overhead as a TIMING line and reset."""
    global _timing_overhead_ns
    if _timing_overhead_ns > 0:
        ms = _timing_overhead_ns / 1_000_000
        _timing_logger.info(f"TIMING:python_timing_overhead:{ms:.3f}")
        _timing_overhead_ns = 0
