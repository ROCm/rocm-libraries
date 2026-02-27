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


@contextmanager
def timing_context(category_name):
    """Context manager for timing instrumentation."""
    if globalParameters.get("TimingInstrumentation", False):
        start = time.time_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.time_ns() - start) / 1_000_000
            _timing_logger.info(f"TIMING:{category_name}:{elapsed_ms:.3f}")
    else:
        yield
