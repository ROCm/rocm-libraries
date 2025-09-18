################################################################################
#
# Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
from Tensile.Utilities.Toolchain import validateToolchain, ToolchainDefaults

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
    parser.addoption("--builddir", "--client-dir")
    parser.addoption("--timing-file", default=None)
    parser.addoption("--asm-cache")

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

    return tmp_path_factory.getbasetemp().parent / "client_execution.lock"

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
    if pytestconfig.getoption("--prebuilt-client"):
        rv += ["--prebuilt-client", pytestconfig.getoption("--prebuilt-client")]
    if pytestconfig.getoption("--asm-cache"):
        rv += ["--asm-cache", pytestconfig.getoption("--asm-cache")]

    return rv

def pytest_collection_modifyitems(items):
    """
    Mainly for tests that aren't simple YAML files (including unit tests).
    Adds a mark for the root directory name to each test.
    """
    for item in items:
        relpath = item.fspath.relto(testdir)
        components = relpath.split(os.path.sep)
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
            (
                cxxCompiler,
                cCompiler,
                assembler,
                offloadBundler,
                hipconfig,
                deviceEnumerator
            ) = validateToolchain(
                args.CxxCompiler,
                args.CCompiler,
                args.Assembler,
                args.OffloadBundler,
                ToolchainDefaults.HIP_CONFIG,
                ToolchainDefaults.DEVICE_ENUMERATOR
            )
            params = {
                "CxxCompiler": cxxCompiler,
                "CCompiler": cCompiler,
                "Assembler": assembler,
                "OffloadBundler": offloadBundler,
                "HipConfig": hipconfig,
                "ROCmAgentEnumeratorPath": deviceEnumerator,
            }
            Common.assignGlobalParameters(params)

            overrideParameters = Tensile.argUpdatedGlobalParameters(args)
            for key, value in overrideParameters.items():
                Common.globalParameters[key] = value

            for key, value in self.additionalParams.items():
                Common.globalParameters[key] = value

            return Common.globalParameters

        def __exit__(self, exc_type, exc_value, traceback):
            Common.restoreDefaultGlobalParameters()

    return gpUpdater

@pytest.fixture
def initGlobalParametersForTCL():
    from Tensile import Common
    from Tensile.TensileCreateLib.ParseArguments import parseArguments

    class gpUpdater:
        def __init__(self, tcl_args):
            self.args = tcl_args

        def __enter__(self):
            args = parseArguments(self.args)
            Common.assignGlobalParameters(args)
            return Common.globalParameters

        def __exit__(self, exc_type, exc_value, traceback):
            Common.restoreDefaultGlobalParameters()

    return gpUpdater
