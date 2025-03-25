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

import itertools
import os
import subprocess
from typing import Optional
from pathlib import Path

from . import SOURCE_PATH
from Tensile.Common import print2, ClientExecutionLock, ensurePath, CLIENT_BUILD_DIR
from Tensile.Common.GlobalParameters import globalParameters

class CMakeEnvironment:
    def __init__(self, sourceDir, buildDir, **options):
        self.sourceDir = sourceDir
        self.buildDir  = buildDir
        self.options = options

    def generate(self):

        args = ['cmake']
        args += itertools.chain.from_iterable([ ['-D', '{}={}'.format(key, value)] for key,value in self.options.items()])
        args += [self.sourceDir]

        print2(' '.join(args))
        with ClientExecutionLock(globalParameters["ClientExecutionLockPath"]):
            subprocess.check_call(args, cwd=ensurePath(self.buildDir))

    def build(self):
        args = ['make', '-j']
        print2(' '.join(args))
        with ClientExecutionLock(globalParameters["ClientExecutionLockPath"]):
            subprocess.check_call(args, cwd=self.buildDir)

    def builtPath(self, path, *paths):
        return os.path.join(self.buildDir, path, *paths)

def clientExecutableEnvironment(builddir: Optional[str], cxxCompiler: str, cCompiler: str):
    sourcedir = SOURCE_PATH

    builddir = ensurePath(builddir)

    options = {'CMAKE_BUILD_TYPE': globalParameters["CMakeBuildType"],
               'TENSILE_USE_MSGPACK': 'ON',
               'TENSILE_USE_LLVM': 'OFF' if (os.name == "nt") else 'ON',
               'Tensile_LIBRARY_FORMAT': globalParameters["LibraryFormat"],
               'Tensile_ENABLE_MARKER' : globalParameters["EnableMarker"],
               'CMAKE_CXX_COMPILER': os.path.join(globalParameters["ROCmBinPath"], cxxCompiler),
               'CMAKE_C_COMPILER': os.path.join(globalParameters["ROCmBinPath"], cCompiler)}

    if "CCACHE_BASEDIR" in os.environ:
        options.update({'CMAKE_C_COMPILER_LAUNCHER': 'ccache', 'CMAKE_CXX_COMPILER_LAUNCHER': 'ccache'})
        print('Is Using CCACHE')

    return CMakeEnvironment(sourcedir, builddir, **options)


buildEnv = None

def getClientExecutable(cxxCompiler: str, cCompiler: str, builddir: Path):
    if "PrebuiltClient" in globalParameters:
        return globalParameters["PrebuiltClient"]

    global buildEnv

    if buildEnv is None:
        buildEnv = clientExecutableEnvironment(builddir / CLIENT_BUILD_DIR, cxxCompiler, cCompiler)
        buildEnv.generate()
        buildEnv.build()

    return buildEnv.builtPath("client/tensile_client")
