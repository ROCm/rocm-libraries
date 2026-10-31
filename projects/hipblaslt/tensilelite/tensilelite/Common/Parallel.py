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

import concurrent.futures
import itertools
import math
import os
import sys
import time

from joblib import Parallel, delayed
from joblib.externals.loky import cpu_count as lokyCpuCount

from .Utilities import tqdm

# All pools started concurrently by one machine's pytest-xdist workers together use at most this
# many processes per available CPU; a single process never uses more than one per CPU.
OVERSUBSCRIPTION_FACTOR = 2


def joblibParallelSupportsGenerator():
    import joblib
    from packaging.version import Version

    joblibVer = joblib.__version__
    return Version(joblibVer) >= Version("1.4.0")


def _kubeCpuRequest():
    """CPUs requested for the Kubernetes pod (``KUBE_CPU_REQUEST``, e.g. "25" or "8.0"), rounded
    down and at least 1; None when unset or not a positive number. A request is not a limit, but
    CI (TheRock) already treats it as the pod's CPU budget."""
    value = os.environ.get("KUBE_CPU_REQUEST")
    if not value:
        return None
    try:
        request = float(value)
    except ValueError:
        return None
    if not math.isfinite(request) or request <= 0:
        return None
    return max(1, int(request))


def availableCPUCount():
    """CPUs this process may use: the affinity mask, limited by the cgroup CPU quota (loky's
    cpu_count) and by KUBE_CPU_REQUEST when set."""
    if os.name == "nt":
        # Windows supports at most 61 workers because the scheduler uses
        # WaitForMultipleObjects directly, which has the limit (the limit
        # is actually 64, but some handles are needed for accounting).
        count = min(os.cpu_count() or 1, 61)
    else:
        count = min(len(os.sched_getaffinity(0)), lokyCpuCount())
    request = _kubeCpuRequest()
    if request is not None:
        count = min(count, request)
    return max(1, count)


def concurrentWorkerCount():
    """Processes sharing the CPUs: PYTEST_XDIST_WORKER_COUNT (set by pytest-xdist in every
    worker), else 1. Other ways of running several Tensile processes at once are not detected."""
    try:
        return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    except ValueError:
        return 1


def poolSizeBudget():
    """Pool size for one process: OVERSUBSCRIPTION_FACTOR x the available CPUs shared among the
    concurrent workers, at most the available CPUs, at least 1."""
    cpus = availableCPUCount()
    return max(1, min(cpus, (OVERSUBSCRIPTION_FACTOR * cpus) // concurrentWorkerCount()))


def CPUThreadCount(enable=True):
    from .GlobalParameters import globalParameters

    if not enable:
        return 1
    budget = poolSizeBudget()
    cpuThreads = globalParameters["CpuThreads"]
    if cpuThreads == -1:
        return budget

    return min(budget, cpuThreads)


def pcallWithGlobalParamsMultiArg(f, args, newGlobalParameters):
    OverwriteGlobalParameters(newGlobalParameters)
    return f(*args)


def pcallWithGlobalParamsSingleArg(f, arg, newGlobalParameters):
    OverwriteGlobalParameters(newGlobalParameters)
    return f(arg)


def apply_print_exception(item, *args):
    # print(item, args)
    try:
        if len(args) > 0:
            func = item
            args = args[0]
            return func(*args)
        else:
            func, item = item
            return func(item)
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


def ProcessingPool(enable=True, maxTasksPerChild=None):
    import multiprocessing
    import multiprocessing.dummy

    threadCount = CPUThreadCount()

    if (not enable) or threadCount <= 1:
        return multiprocessing.dummy.Pool(1)

    if multiprocessing.get_start_method() == "spawn":
        from . import GlobalParameters

        return multiprocessing.Pool(
            threadCount,
            initializer=OverwriteGlobalParameters,
            maxtasksperchild=maxTasksPerChild,
            initargs=(GlobalParameters.globalParameters,),
        )
    else:
        return multiprocessing.Pool(threadCount, maxtasksperchild=maxTasksPerChild)


def ParallelMap(function, objects, message="", enable=True, method=None, maxTasksPerChild=None):
    """
    Generally equivalent to list(map(function, objects)), possibly executing in parallel.

      message: A message describing the operation to be performed.
      enable: May be set to false to disable parallelism.
      method: A function which can fetch the mapping function from a processing pool object.
          Leave blank to use .map(), other possiblities:
             - `lambda x: x.starmap` - useful if `function` takes multiple parameters.
             - `lambda x: x.imap` - lazy evaluation
             - `lambda x: x.imap_unordered` - lazy evaluation, does not preserve order of return value.
    """
    from .GlobalParameters import globalParameters

    threadCount = CPUThreadCount(enable)
    pool = ProcessingPool(enable, maxTasksPerChild)

    if threadCount <= 1 and globalParameters["ShowProgressBar"]:
        # Provide a progress bar for single-threaded operation.
        # This works for method=None, and for starmap.
        mapFunc = map
        if method is not None:
            # itertools provides starmap which can fill in for pool.starmap.  It provides imap on Python 2.7.
            # If this works, we will use it, otherwise we will fallback to the "dummy" pool for single threaded
            # operation.
            try:
                mapFunc = method(itertools)
            except NameError:
                mapFunc = None

        if mapFunc is not None:
            return list(mapFunc(function, tqdm(objects, message)))

    mapFunc = pool.map
    if method:
        mapFunc = method(pool)

    objects = zip(itertools.repeat(function), objects)
    function = apply_print_exception

    countMessage = ""
    try:
        countMessage = " for {} tasks".format(len(objects))
    except TypeError:
        pass

    if message != "":
        message += ": "

    print("{0}Launching {1} threads{2}...".format(message, threadCount, countMessage))
    sys.stdout.flush()
    currentTime = time.time()
    rv = mapFunc(function, objects)
    totalTime = time.time() - currentTime
    print("{0}Done. ({1:.1f} secs elapsed)".format(message, totalTime))
    sys.stdout.flush()
    pool.close()
    return rv


def ParallelMapReturnAsGenerator(function, objects, message="", enable=True, multiArg=True):
    from .GlobalParameters import globalParameters

    threadCount = CPUThreadCount(enable)
    print("{0}Launching {1} threads...".format(message, threadCount))

    if threadCount <= 1 and globalParameters["ShowProgressBar"]:
        # Provide a progress bar for single-threaded operation.
        callFunc = lambda args: function(*args) if multiArg else lambda args: function(args)
        return [callFunc(args) for args in tqdm(objects, message)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=threadCount) as executor:
        resultFutures = (executor.submit(function, *arg if multiArg else arg) for arg in objects)
        for result in concurrent.futures.as_completed(resultFutures):
            yield result.result()


def ParallelMap2(
    function, objects, message="", enable=True, multiArg=True, return_as="list", procs=None
):
    """
    Generally equivalent to list(map(function, objects)), possibly executing in parallel.

      message: A message describing the operation to be performed.
      enable: May be set to false to disable parallelism.
      multiArg: True if objects represent multiple arguments
                  (differentiates multi args vs single collection arg)
      procs: At most this many processes; the CPUThreadCount() budget stays the upper bound.
             None or a value below 1 means the budget.
    """
    if return_as in ("generator", "generator_unordered") and not joblibParallelSupportsGenerator():
        return ParallelMapReturnAsGenerator(function, objects, message, enable, multiArg)

    from .GlobalParameters import globalParameters

    threadCount = CPUThreadCount(enable)
    if procs is not None and procs > 0:
        threadCount = min(threadCount, procs)

    if threadCount <= 1 and globalParameters["ShowProgressBar"]:
        # Provide a progress bar for single-threaded operation.
        return [function(*args) if multiArg else function(args) for args in tqdm(objects, message)]

    countMessage = ""
    try:
        countMessage = " for {} tasks".format(len(objects))
    except TypeError:
        pass

    if message != "":
        message += ": "
    print("{0}Launching {1} threads{2}...".format(message, threadCount, countMessage))
    sys.stdout.flush()
    currentTime = time.time()

    pcall = pcallWithGlobalParamsMultiArg if multiArg else pcallWithGlobalParamsSingleArg
    pargs = zip(objects, itertools.repeat(globalParameters))

    if joblibParallelSupportsGenerator():
        rv = Parallel(n_jobs=threadCount, timeout=99999, return_as=return_as)(
            delayed(pcall)(function, a, params) for a, params in pargs
        )
    else:
        rv = Parallel(n_jobs=threadCount, timeout=99999)(
            delayed(pcall)(function, a, params) for a, params in pargs
        )

    totalTime = time.time() - currentTime
    print("{0}Done. ({1:.1f} secs elapsed)".format(message, totalTime))
    sys.stdout.flush()
    return rv
