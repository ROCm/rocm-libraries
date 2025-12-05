################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import subprocess
import shutil
import sys
from pathlib import Path
from Tensile import TensileLibLogicToYaml
from Tensile import LibraryIO
from Tensile import Tensile

# generate config yaml from library
def generateConfigLib(configYaml, libName, solutionIndex):
    sys.argv = [
        "",
        "--input",
        f"{libName}",
        "--indices",
        f"{solutionIndex}",
        "--output",
        f"{configYaml}",
    ]

    TensileLibLogicToYaml.main()


def buildTensile(tensilelitePath, clientPath, buildDir):
    if not os.path.isfile(clientPath):
        print("Building tensilelite client...")
        subprocess.call(
            ["invoke", "build-client", "--build-dir", buildDir], cwd=tensilelitePath
        )


def generateLiblogic(clientPath, workDir, configYaml):
    shutil.rmtree(workDir, ignore_errors=True)

    sys.argv = [
        "",
        "--prebuilt-client",
        f"{clientPath}",
        f"{configYaml}",
        f"{workDir}",
    ]

    Tensile.main()

def main(tensileClient, workspace, libName, solutionIndex):

    # generate config yaml from library
    configYaml = os.path.join(workspace, "config.yaml")
    generateConfigLib(configYaml, libName, solutionIndex)

    # update output the output filename with the index
    configYaml = os.path.join(workspace, f"config_{solutionIndex}.yaml")

    # call Tensile to generate liblogic
    workDir = os.path.join(workspace, "WDirDevice")
    generateLiblogic(tensileClient, workDir, configYaml)

    # create a library from the new library logic
    newLiblogicPath = Path(os.path.join(workDir, "3_LibraryLogic"))
    newLiblogicName = next(newLiblogicPath.glob("*.yaml"))
    
    libYaml = LibraryIO.readYAML(libName)

    newLibYaml = LibraryIO.readYAML(newLiblogicName)

    if libYaml != newLibYaml:
        ValueError("The generated and existing libraries do not match!")

    return True
