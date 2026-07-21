################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from timeit import default_timer as timer
from typing import Callable

from .Shared import envVariableIsSet

TIMING_ENV_VAR: str = "TENSILE_PRINT_TIMING"


def timing(func: Callable) -> Callable:
    f"""Timing decorator to measure execution time of a function.

  Add ``@timing`` to mark a function for timing; set the environment variable
  {TIMING_ENV_VAR}=ON to enable timing decorated functions.
  """
    if not envVariableIsSet(TIMING_ENV_VAR):
        return func

    def wrapper(*args, **kwargs):
        start = timer()
        res = func(*args, **kwargs)
        end = timer()
        print(f"{func.__name__} took {end - start} seconds")
        return res

    return wrapper
