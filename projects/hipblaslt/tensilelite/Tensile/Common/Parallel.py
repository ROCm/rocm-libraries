################################################################################
#
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import multiprocessing
import os
import re
import sys
import time
from functools import partial
from typing import Any, Callable

from .Utilities import tqdm


def get_inherited_job_limit() -> int:
    # 1. Check CMAKE_BUILD_PARALLEL_LEVEL (CMake 3.12+)
    if 'CMAKE_BUILD_PARALLEL_LEVEL' in os.environ:
        try:
            return int(os.environ['CMAKE_BUILD_PARALLEL_LEVEL'])
        except ValueError:
            pass

    # 2. Parse MAKEFLAGS for -jN
    makeflags = os.environ.get('MAKEFLAGS', '')
    match = re.search(r'-j\s*(\d+)', makeflags)
    if match:
        return int(match.group(1))

    return -1


def CPUThreadCount(enable=True):
    if not enable:
        return 1
    from .GlobalParameters import globalParameters

    # Priority order:
    # 1. Inherited from build system (CMAKE_BUILD_PARALLEL_LEVEL or MAKEFLAGS)
    # 2. Explicit --jobs flag
    # 3. Auto-detect
    inherited_limit = get_inherited_job_limit()
    cpuThreads = inherited_limit if inherited_limit > 0 else globalParameters["CpuThreads"]

    if cpuThreads < 1:
        if os.name == "nt":
            cpuThreads = os.cpu_count()
        else:
            cpuThreads = len(os.sched_getaffinity(0))

    if os.name == "nt":
        # Windows supports at most 61 workers because the scheduler uses
        # WaitForMultipleObjects directly, which has the limit (the limit
        # is actually 64, but some handles are needed for accounting).
        cpuThreads = min(cpuThreads, 61)
    return max(1, cpuThreads)


def pcallWithGlobalParamsMultiArg(f, args, newGlobalParameters):
    OverwriteGlobalParameters(newGlobalParameters)
    return f(*args)


def pcallWithGlobalParamsSingleArg(f, arg, newGlobalParameters):
    OverwriteGlobalParameters(newGlobalParameters)
    return f(arg)


def worker_function(args, function, multiArg):
    """Worker function that executes in the pool process."""
    try:
        if multiArg:
            return function(*args)
        else:
            return function(args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def OverwriteGlobalParameters(newGlobalParameters):
    from . import GlobalParameters

    GlobalParameters.globalParameters.clear()
    GlobalParameters.globalParameters.update(newGlobalParameters)


def progress_logger(iterable, total, message, min_log_interval=5.0):
    """
    Generator that wraps an iterable and logs progress with time-based throttling.

    Only logs progress if at least min_log_interval seconds have passed since last log.
    Only prints completion message if task took >= min_log_interval seconds.

    Yields (index, item) tuples.
    """
    start_time = time.time()
    last_log_time = start_time
    log_interval = 1 + (total // 100)

    for idx, item in enumerate(iterable):
        if idx % log_interval == 0:
            current_time = time.time()
            if (current_time - last_log_time) >= min_log_interval:
                print(f"{message}\t{idx+1: 5d}/{total: 5d}")
                last_log_time = current_time
        yield idx, item

    elapsed = time.time() - start_time
    final_idx = idx + 1 if 'idx' in locals() else 0

    if elapsed >= min_log_interval or last_log_time > start_time:
        print(f"{message} done in {elapsed:.1f}s!\t{final_idx: 5d}/{total: 5d}")


def imap_with_progress(pool, func, iterable, total, message, chunksize):
    results = []
    for _, result in progress_logger(pool.imap(func, iterable, chunksize=chunksize), total, message):
        results.append(result)
    return results


def _ParallelMap_generator(worker, objects, objLen, message, chunksize, threadCount, globalParameters, maxtasksperchild):
    # separate fn because yield makes the entire fn a generator even if unreachable
    ctx = multiprocessing.get_context('forkserver' if os.name != 'nt' else 'spawn')

    with ctx.Pool(processes=threadCount, maxtasksperchild=maxtasksperchild,
                  initializer=OverwriteGlobalParameters, initargs=(globalParameters,)) as pool:
        for _, result in progress_logger(pool.imap_unordered(worker, objects, chunksize=chunksize), objLen, message):
            yield result


def ParallelMap2(
    function: Callable,
    objects: Any,
    message: str = "",
    enable: bool = True,
    multiArg: bool = True,
    minChunkSize: int = 1,
    maxWorkers: int = -1,
    maxtasksperchild: int = 1024,
    return_as: str = "list"
):
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
        return_as: Optional; "list" (default) or "generator_unordered" for streaming results

    Returns:
        A list or generator containing the results of applying **function** to each item in **objects**.
    """
    from .GlobalParameters import globalParameters

    threadCount = CPUThreadCount(enable)

    if not hasattr(objects, "__len__"):
        objects = list(objects)

    objLen = len(objects)
    if objLen == 0:
        return [] if return_as == "list" else iter([])

    f = (lambda x: function(*x)) if multiArg else function
    if objLen == 1:
        print(f"{message}: (1 task)")
        result = [f(x) for x in objects]
        return result if return_as == "list" else iter(result)

    extra_message = (
        f": {threadCount} thread(s)" + f", {objLen} tasks"
        if objLen
        else ""
    )

    print(f"ParallelMap {message}{extra_message}")

    if threadCount <= 1:
        result = [f(x) for x in objects]
        return result if return_as == "list" else iter(result)

    if maxWorkers > 0:
        threadCount = min(maxWorkers, threadCount)

    chunksize = max(minChunkSize, objLen // 2000)
    worker = partial(worker_function, function=function, multiArg=multiArg)
    if return_as == "generator_unordered":
        # yield results as they complete without buffering
        return _ParallelMap_generator(worker, objects, objLen, message, chunksize, threadCount, globalParameters, maxtasksperchild)
    else:
        ctx = multiprocessing.get_context('forkserver' if os.name != 'nt' else 'spawn')
        with ctx.Pool(processes=threadCount, maxtasksperchild=maxtasksperchild,
                      initializer=OverwriteGlobalParameters, initargs=(globalParameters,)) as pool:
            return list(imap_with_progress(pool, worker, objects, objLen, message, chunksize))
