################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import itertools
import os
from typing import Any, Callable

from .Utilities.ConditionalImports import joblib


def CPUThreadCount(enable=True):
    from .Common import globalParameters

    if not enable:
        return 1
    else:
        if os.name == "nt":
            # Windows supports at most 61 workers because the scheduler uses
            # WaitForMultipleObjects directly, which has the limit (the limit
            # is actually 64, but some handles are needed for accounting).
            cpu_count = min(os.cpu_count(), 61)
        else:
            cpu_count = len(os.sched_getaffinity(0))
        cpuThreads = globalParameters["CpuThreads"]
        if cpuThreads < 1:
            return min(cpu_count, 64)  # Max build threads to avoid out-of-memory
        return min(cpu_count, cpuThreads)


def OverwriteGlobalParameters(newGlobalParameters):
    from . import Common

    Common.globalParameters.clear()
    Common.globalParameters.update(newGlobalParameters)


def pcallWithGlobalParamsMultiArg(f, args, newGlobalParameters):
    OverwriteGlobalParameters(newGlobalParameters)
    return f(*args)


def pcallWithGlobalParamsSingleArg(f, arg, newGlobalParameters):
    OverwriteGlobalParameters(newGlobalParameters)
    return f(arg)


def ParallelMap(
    function: Callable,
    objects: Any,
    message: str = "",
    enable: bool = True,
    multiArg: bool = True,
) -> list:
    """Executes a function over a list of objects in parallel or sequentially.

    This function is generally equivalent to ``list(map(function, objects))``. However, it provides
    additional functionality to run in parallel, depending on the 'enable' flag and available CPU
    threads.

    Args:
        function: The function to apply to each item in 'objects'. If 'multiArg' is True, 'function'
                  should accept multiple arguments.
        objects: An iterable of objects to be processed by 'function'. If 'multiArg' is True, each
                 item in 'objects' should be an iterable of arguments for 'function'.
        message: Optional; a message describing the operation. Default is an empty string.
        enable: Optional; if False, disables parallel execution and runs sequentially. Default is True.
        multiArg: Optional; if True, treats each item in 'objects' as multiple arguments for
                  'function'. Default is True.
        verbose: Optional; verbosity level for parallel execution. Default is 0.

    Returns:
        A list containing the results of applying **function** to each item in **objects**.
    """

    from . import Utils
    from .Common import globalParameters

    threadCount = CPUThreadCount(enable)

    message += (
        f": {threadCount} thread(s)" + f", {len(objects)} tasks"
        if hasattr(objects, "__len__")
        else ""
    )

    if threadCount <= 1 or joblib is None:
        f = lambda x: function(*x) if multiArg else function(x)
        return [f(x) for x in Utils.tqdm(objects, desc=message)]

    inputs = list(zip(objects, itertools.repeat(globalParameters)))
    pargs = Utils.tqdm(inputs, desc=message)
    pcall = pcallWithGlobalParamsMultiArg if multiArg else pcallWithGlobalParamsSingleArg

    return joblib.Parallel(n_jobs=threadCount)(
        joblib.delayed(pcall)(function, a, params) for a, params in pargs
    )
