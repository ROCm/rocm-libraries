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
from joblib import cpu_count as lokyCpuCount

from .Utilities import printWarning, tqdm

# All pools started concurrently by one machine's pytest-xdist workers together use at most this
# many processes per available CPU; a single process never uses more than one per CPU.
OVERSUBSCRIPTION_FACTOR = 2

# Peak memory per kernel-generation pool process: ~270 MB in a 4-config --build-only run (13.1 GB
# over 49 processes), ~0.3 GiB RSS on the gfx1250-strict CI host (RSS double-counts shared pages).
# Override: TENSILE_POOL_PROCESS_PEAK_MIB. POOL_MEMORY_FRACTION: share of the cgroup memory limit
# that all concurrent workers' pools may use.
POOL_PROCESS_PEAK_BYTES = 300 * 2**20
POOL_MEMORY_FRACTION = 0.8


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


def _poolProcessPeakBytes():
    """POOL_PROCESS_PEAK_BYTES, or TENSILE_POOL_PROCESS_PEAK_MIB when it is a positive integer."""
    try:
        mib = int(os.environ.get("TENSILE_POOL_PROCESS_PEAK_MIB", ""))
    except ValueError:
        return POOL_PROCESS_PEAK_BYTES
    return mib * 2**20 if mib > 0 else POOL_PROCESS_PEAK_BYTES


def _cgroupMemoryLimit(root="/sys/fs/cgroup", selfCgroup="/proc/self/cgroup"):
    """Smallest memory limit (bytes) from this process's cgroup up to ``root``: v2 ``memory.max``,
    v1 ``memory.limit_in_bytes`` and ``hierarchical_memory_limit``. A parent's limit binds its
    children (Slurm, systemd scopes), even when a child sets a larger one. None if no level sets a
    limit, nothing is readable, or on Windows. ``root``, a container's own cgroup, is checked last.
    Inside a cgroup namespace, v2 ancestors above the container are invisible."""
    if os.name == "nt":
        return None
    files = []  # leaf first
    try:
        with open(selfCgroup) as f:
            for line in f:
                hierarchy, _, rest = line.rstrip("\n").partition(":")
                controllers, _, path = rest.partition(":")
                if hierarchy == "0" and not controllers:
                    base, name = root, "memory.max"
                elif "memory" in controllers.split(","):
                    base, name = os.path.join(root, "memory"), "memory.limit_in_bytes"
                else:
                    continue
                parts = [p for p in path.split("/") if p]
                files += [os.path.join(base, *parts[:i], name) for i in range(len(parts), -1, -1)]
    except OSError:
        pass
    files += [
        os.path.join(root, "memory.max"),
        os.path.join(root, "memory", "memory.limit_in_bytes"),
    ]
    limits = []
    for path in dict.fromkeys(files):
        try:
            with open(path) as f:
                limits.append(int(f.read().strip()))
        except (OSError, ValueError):
            pass  # absent, unreadable, or "max": no limit at this level
        if path.endswith("memory.limit_in_bytes"):
            # v1 hides a parent's limit outside the container's view; memory.stat's
            # hierarchical_memory_limit still reports it.
            try:
                with open(os.path.join(os.path.dirname(path), "memory.stat")) as f:
                    for line in f:
                        key, _, value = line.partition(" ")
                        if key == "hierarchical_memory_limit":
                            limits.append(int(value))
            except (OSError, ValueError):
                pass
    # cgroup v1 reports "no limit" as a value near 2**63.
    limits = [limit for limit in limits if 0 < limit < 2**62]
    return min(limits) if limits else None


def memoryPoolSizeCap(workers=None):
    """Pool size per process that keeps the pools of all concurrent workers within
    POOL_MEMORY_FRACTION of the cgroup memory limit, at POOL_PROCESS_PEAK_BYTES per pool process;
    None without a limit. The CPU budget cannot see other containers sharing the machine's CPUs,
    while the memory limit is per container."""
    limit = _cgroupMemoryLimit()
    if limit is None:
        return None
    workers = concurrentWorkerCount() if workers is None else max(1, workers)
    perWorker = int(limit * POOL_MEMORY_FRACTION) // workers
    return max(1, perWorker // _poolProcessPeakBytes())


def poolSizeBudget(workers=None):
    """Pool size for one process: OVERSUBSCRIPTION_FACTOR x the available CPUs shared among the
    concurrent workers, at most the available CPUs and the memory cap, at least 1. ``workers``
    overrides PYTEST_XDIST_WORKER_COUNT, which the pytest controller lacks."""
    workers = concurrentWorkerCount() if workers is None else max(1, workers)
    cpus = availableCPUCount()
    budget = max(1, min(cpus, (OVERSUBSCRIPTION_FACTOR * cpus) // workers))
    memoryCap = memoryPoolSizeCap(workers)
    return budget if memoryCap is None else min(budget, memoryCap)


def describePoolSizeBudget(workers=None):
    """The pool budget and its inputs, as one log line."""
    workers = concurrentWorkerCount() if workers is None else max(1, workers)
    limit = _cgroupMemoryLimit()
    memory = "none" if limit is None else "%.1f GiB" % (limit / 2**30)
    return (
        "TensileLite pool budget: %d processes per worker "
        "(CPUs %d, workers %d, memory limit %s, %d MiB per process)"
        % (poolSizeBudget(workers), availableCPUCount(), workers, memory,
           _poolProcessPeakBytes() // 2**20)
    )


def shutdownWorkerPool():
    """Terminate the idle joblib (loky) worker pool, if one exists.

    loky keeps its workers alive between Parallel calls, including any heap
    pages their allocators retain after temporary objects are freed. Call this
    after a phase's results have been consumed to release the workers' private
    memory. The next ParallelMap2 call creates fresh workers. No-op when no
    reusable executor exists.
    """
    try:
        from joblib.externals.loky import reusable_executor
    except ImportError:  # pragma: no cover - joblib without loky
        return
    # loky has no public call for "end the reusable pool", so this reaches its
    # module-level executor. Say so if a joblib upgrade moves it, rather than
    # silently keeping every worker alive.
    if not hasattr(reusable_executor, "_executor"):
        printWarning(
            "loky's reusable_executor has no _executor attribute; the worker "
            "pool is not recycled between phases"
        )
        return
    executor = reusable_executor._executor
    if executor is None:
        return
    executor.shutdown(wait=True, kill_workers=False)


_libc = None


def releaseFreedMemory():
    """Hand freed heap pages back to the OS (glibc malloc_trim).

    Allocators can retain pages after Python and rocisa objects are freed.
    malloc_trim can return whole free pages throughout glibc's arenas; it
    cannot release live allocations or free blocks held by other allocators.
    A no-op where libc has no malloc_trim.
    """
    global _libc
    if _libc is False:
        return
    if _libc is None:
        try:
            import ctypes

            _libc = ctypes.CDLL("libc.so.6")
            _libc.malloc_trim.argtypes = [ctypes.c_size_t]
            _libc.malloc_trim.restype = ctypes.c_int
        except (OSError, AttributeError):
            _libc = False
            return
    _libc.malloc_trim(0)


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

    if newGlobalParameters is GlobalParameters.globalParameters:
        return
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


def _lengthOrMaterialise(objects):
    """The task count when the input can state one without being consumed, else None."""
    try:
        return len(objects)
    except TypeError:
        return None


def ParallelMap2(
    function, objects, message="", enable=True, multiArg=True, return_as="list", procs=None,
    spawn_worker=False, count=None,
):
    """
    Generally equivalent to list(map(function, objects)), possibly executing in parallel.

      message: A message describing the operation to be performed.
      enable: May be set to false to disable parallelism.
      multiArg: True if objects represent multiple arguments
                  (differentiates multi args vs single collection arg)
      procs: At most this many processes; the CPUThreadCount() budget stays the upper bound.
             None or a value below 1 means the budget.
      spawn_worker: Keep ``n_jobs`` at least 2. joblib runs ``n_jobs=1`` in this
                  process, which cannot switch the already-imported rocisa backend.
    """
    if return_as in ("generator", "generator_unordered") and not joblibParallelSupportsGenerator():
        return ParallelMapReturnAsGenerator(function, objects, message, enable, multiArg)

    from .GlobalParameters import globalParameters

    threadCount = CPUThreadCount(enable)
    # Only the COUNT is wanted here; materialising costs 949 bytes per task before one has run.
    objectCount = count if count is not None else _lengthOrMaterialise(objects)
    if objectCount is None:
        objects = list(objects)
        objectCount = len(objects)
    threadCount = min(threadCount, max(objectCount, 1))
    if procs is not None and procs > 0:
        threadCount = min(threadCount, procs)
    # One task would stay in this process. A worker handoff needs a new
    # interpreter so it re-imports rocisa cleanly.
    if spawn_worker:
        threadCount = max(threadCount, 2)

    if threadCount <= 1 and globalParameters["ShowProgressBar"]:
        # Provide a progress bar for single-threaded operation.
        return [function(*args) if multiArg else function(args)
                for args in tqdm(objects, message, total=objectCount)]

    if message != "":
        message += ": "
    print("{0}Launching {1} threads for {2} tasks...".format(message, threadCount, objectCount))
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
