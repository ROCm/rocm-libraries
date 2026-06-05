################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest
import os
import sys

try:
    import xdist  # noqa
except ImportError:
    @pytest.fixture(scope="session")
    def worker_id():
        return None

testdir = os.path.dirname(__file__)
moddir = os.path.dirname(testdir)
rootdir = os.path.dirname(moddir)
sys.path.append(rootdir)

def pytest_addoption(parser):
    parser.addoption("--tensile-options")
    parser.addoption("--global-parameters")
    parser.addoption("--prebuilt-client")
    parser.addoption("--no-common-build", action="store_true")
    parser.addoption("--builddir", "--client-dir")
    parser.addoption("--timing-file", default=None)

@pytest.fixture(scope="session")
def timing_path(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--timing-file")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("results") / "timing.csv")

@pytest.fixture(autouse=True)
def incremental_timer(request, worker_lock_instance, timing_path):
    #import pdb
    #pdb.set_trace()
    import datetime
    start = datetime.datetime.now()
    yield
    stop = datetime.datetime.now()

    testtime = stop - start
    with worker_lock_instance:
        with open(timing_path, "a") as f:
            f.write("{},{}\n".format(request.node.name, testtime.total_seconds()))

@pytest.fixture(scope="session")
def builddir(pytestconfig, tmpdir_factory):
    userDir = pytestconfig.getoption("--builddir")
    if userDir is not None:
        return userDir
    return str(tmpdir_factory.mktemp("0_Build"))

@pytest.fixture(scope="session")
def worker_lock_path(tmp_path_factory, worker_id):
    if not worker_id:
        return None

    # Under FFM each worker gets its own emulator instance, so there is
    # no GPU contention — give each worker a private lock so client
    # invocations can run in parallel across workers.
    if os.environ.get("HSA_MODEL_MEMFILE"):
      return tmp_path_factory.getbasetemp().parent / f"client_execution_{worker_id}.lock"

    return tmp_path_factory.getbasetemp().parent / "client_execution.lock"

@pytest.fixture(scope="session", autouse=True)
def assign_gpu_to_worker(worker_id):
    """
    Assigns a GPU to each pytest-xdist worker using modulo arithmetic.
    Worker IDs are in format 'gw0', 'gw1', 'gw2', etc.
    Sets HIP_VISIBLE_DEVICES to isolate GPU access per worker.

    If you have N GPUs and M workers:
    - Worker 0 -> GPU 0, Worker 1 -> GPU 1, ..., Worker N-1 -> GPU N-1
    - Worker N -> GPU 0, Worker N+1 -> GPU 1, etc. (wraps around)
    """
    if not worker_id or worker_id == "master":
        # Single worker or master process - use all GPUs
        return

    import re
    from Tensile.ParallelExecution import detectAvailableGpus

    base_memfile = os.environ.get("HSA_MODEL_MEMFILE", "")
    if base_memfile:
        os.environ["HSA_MODEL_MEMFILE"] = f"{base_memfile}_{worker_id}"
        print(f"Worker {worker_id}: HSA_MODEL_MEMFILE={os.environ['HSA_MODEL_MEMFILE']}")
        return

    num_gpus = detectAvailableGpus()
    # Extract numeric ID from worker_id (e.g., 'gw0' -> 0, 'gw1' -> 1)
    match = re.search(r'\d+', worker_id)
    if match:
        worker_num = int(match.group())
        gpu_id = worker_num % num_gpus  # Use modulo to wrap around available GPUs
        os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Worker {worker_id} assigned to GPU {gpu_id} (total GPUs: {num_gpus})")
    else:
        print(f"Warning: Could not parse worker_id '{worker_id}' for GPU assignment")

@pytest.fixture(autouse=True)
def clear_ffm_memfile_per_test():
   """
   Unlinks the worker's HSA_MODEL_MEMFILE after each test so the next
   tensilelite-client subprocess starts with a fresh FFM emulator backing.
   Helps isolate flakiness caused by residual state in the shared memfile
   between consecutive tests on the same worker.
   """
   yield
   memfile = os.environ.get("HSA_MODEL_MEMFILE")
   if not memfile:
     return
   try:
     os.unlink(memfile)
   except FileNotFoundError:
     pass
   except OSError as e:
     print(f"warn: could not unlink {memfile}: {e}", file=sys.stderr)

@pytest.fixture
def tensile_script_path():
    return os.path.join(moddir, 'bin', 'Tensile')

@pytest.fixture
def worker_lock_instance(worker_lock_path):
    if not worker_lock_path:
        return open(os.devnull)

    import filelock
    return filelock.FileLock(str(worker_lock_path))

@pytest.fixture
def tensile_args(pytestconfig, builddir, worker_lock_path):
    rv = []
    if worker_lock_path:
        rv += ["--client-lock", str(worker_lock_path)]

    extraOptions = pytestconfig.getoption("--tensile-options")
    if extraOptions is not None:
        rv += extraOptions.split(",")
    if pytestconfig.getoption("--global-parameters"):
        rv += ["--global-parameters", pytestconfig.getoption("--global-parameters")]
    if not pytestconfig.getoption("--no-common-build"):
        if pytestconfig.getoption("--prebuilt-client"):
            rv += ["--prebuilt-client", pytestconfig.getoption("--prebuilt-client")]

    return rv

def pytest_collection_modifyitems(items):
    """
    Mainly for tests that aren't simple YAML files (including unit tests).
    Adds a mark for the root directory name to each test.
    """
    for item in items:
        
        relpath = item.fspath.relto(testdir)
        components = relpath.split(os.path.sep)
        # print(f"Items: {item}, Testdir: {testdir}, Components: {components}")
        if len(components) > 0 and len(components[0]) > 0:
            item.add_marker(getattr(pytest.mark, components[0]))

@pytest.fixture
def useGlobalParameters(tensile_args):
    from Tensile import Common
    from Tensile import Tensile
    import argparse

    class gpUpdater:
        def __init__(self, **params):
            self.additionalParams = params

        def __enter__(self):
            argParser = argparse.ArgumentParser()
            Tensile.addCommonArguments(argParser)
            args = argParser.parse_args(tensile_args)

            Common.restoreDefaultGlobalParameters()
            if args.CxxCompiler:
                Common.globalParameters["CxxCompiler"] = args.CxxCompiler
            isa = Common.detectGlobalCurrentISA(args.device)
            Common.assignGlobalParameters({}, isa)

            overrideParameters = Tensile.argUpdatedGlobalParameters(args)
            for key, value in overrideParameters.items():
                Common.globalParameters[key] = value

            for key, value in self.additionalParams.items():
                Common.globalParameters[key] = value

            return Common.globalParameters

        def __exit__(self, exc_type, exc_value, traceback):
            Common.restoreDefaultGlobalParameters()

    return gpUpdater

