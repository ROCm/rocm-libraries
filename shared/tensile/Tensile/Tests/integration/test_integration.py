################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import os, subprocess, shutil, random, pytest
from filelock import FileLock
from Tensile import ClientWriter, LibraryIO, Common
from Tensile.SolutionStructs import ProblemType, ProblemSizesMock

# 1. Call TensileCreateLibrary
# 2. Get client instance from ClientExecutable()
# 3. Write some problem types/sizes to ClientParameters with writeClientConfig()
# 4. Run client with small subset of GEMM problems
# $ pytest Tensile/Tests -m integration --capture=tee-sys --builddir /tmp/pytest

# TODO fix TensileCreateLibrary build configs:
#   --library-format=msgpack is currently known to fail
#   --short-file-names is currently known to fail

def downloadLogicFiles(logicDir):
  prefix = "library/src/blas3/Tensile/Logic/asm_full"
  testData = {
    "pre_checkin" : [
      "arcturus/arcturus_Cijk_Ailk_Bjlk_SB.yaml",
      "vega20/vega20_Cijk_Ailk_Bjlk_SB.yaml",
      "vega10/vega10_Cijk_Ailk_Bjlk_SB.yaml",
      "arcturus/arcturus_Cijk_Ailk_Bjlk_BBH.yaml",
      "arcturus/arcturus_Cijk_Ailk_Bjlk_HBH.yaml",
      "vega20/vega20_Cijk_Ailk_Bjlk_HBH.yaml",
      "vega10/vega10_Cijk_Ailk_Bjlk_HBH.yaml",
      "hip/hip_Cijk_Ailk_Bjlk_CB.yaml"
    ],
    "quick" : [
      "arcturus/arcturus_Cijk_Ailk_Bjlk_SB.yaml",
      "vega20/vega20_Cijk_Ailk_Bjlk_SB.yaml",
      "vega10/vega10_Cijk_Ailk_Bjlk_SB.yaml"
    ]
  }

  parentDir = os.path.dirname(logicDir)
  shutil.rmtree(logicDir, ignore_errors=True)

  # basically to query the latest zip release weblink, download it and unzip
  # selected files to destination folder
  cmd = """#!/bin/bash
  set -x
  wget -nc https://api.github.com/repos/ROCmSoftwarePlatform/rocBLAS/releases/latest
  weblink=$(grep -oP '(?<="zipball_url": ")[a-zA-Z:/\.\-0-9]*' latest)
  wget -nc $weblink
  archive=$(basename $weblink)
  rootDir=$(zipinfo -1 $archive | head -n 1)
  """
  for schedule in list(testData.keys()):
    for f in testData[schedule]:
      dir = os.path.join(logicDir, schedule)
      Common.ensurePath(dir)
      cmd += "unzip -j -d %s -x $archive ${rootDir}%s\n"%(dir, os.path.join(prefix,f))

  scriptFile = os.path.join(parentDir,"get_logic.sh")
  with open(scriptFile, "w") as file:
    file.write(cmd)
  os.chmod(scriptFile, 0o777)

  subprocess.run(scriptFile, cwd=parentDir, check=True)

@pytest.fixture(scope="session")
def getLogicFileDir(tmp_path_factory, worker_id):

  if worker_id == "master":
    rootTmpDir = tmp_path_factory.getbasetemp()
    destDir = os.path.join(rootTmpDir, "logic_yaml")
    downloadLogicFiles(destDir)
  else:
    rootTmpDir = tmp_path_factory.getbasetemp().parent
    destDir = os.path.join(rootTmpDir, "logic_yaml")
    lockPath = os.path.join(rootTmpDir, "get_logic.lock")

    with FileLock(lockPath):
      if not os.path.isdir(destDir):
        downloadLogicFiles(destDir)

  return destDir

def isSkippedTest(testYamls, mergeFiles, libraryFormat, shortNames, legacyComponents):
  if testYamls == "pre_checkin":
    # for more extensive tests,
    # we run only on single combination of build params
    if (mergeFiles           == True
        and shortNames       == False
        and libraryFormat    == "yaml"
        and legacyComponents == False):
      return False
    else:
      return True
  elif testYamls == "quick":
    return False
  else:
    assert(False) # should've caught all params already

def str2bool(mergeFiles, shortNames, legacyComponents):
  return (True if mergeFiles=="mergeFiles" else False,
          True if shortNames=="shortNames" else False,
          True if legacyComponents=="legacyComponents" else False)

@pytest.mark.parametrize("testYamls",         ["quick", "pre_checkin"])
@pytest.mark.parametrize("mergeFiles",        ["mergeFiles", "noMergeFiles"])
@pytest.mark.parametrize("libraryFormat",     ["yaml", "msgpack"])
@pytest.mark.parametrize("shortNames",        ["shortNames", "noShortName"])
@pytest.mark.parametrize("legacyComponents",  ["legacyComponents", "noLegacyComponents"])
def test_integration(useGlobalParameters, builddir, getLogicFileDir,
                     testYamls, mergeFiles, libraryFormat, shortNames, legacyComponents):
  mergeFiles, shortNames, legacyComponents = str2bool(mergeFiles, shortNames, legacyComponents)

  if isSkippedTest(testYamls, mergeFiles, libraryFormat, shortNames, legacyComponents):
    pytest.skip("only run pre_checkin test in certain combo")

  logicFileDir = os.path.join(getLogicFileDir, testYamls)
  outputDir    = os.path.join(builddir, "lib")

  with useGlobalParameters(OutputPath=outputDir,
                           WorkingPath=outputDir,
                           MergeFiles=mergeFiles,
                           LibraryFormat=libraryFormat,
                           LegacyComponents=legacyComponents,
                           ShortNames=shortNames,
                           GenerateManifestAndExit=False
                           ):
    Common.ensurePath(outputDir)

    createLibraryScript = ClientWriter.getBuildClientLibraryScript(outputDir, logicFileDir)
    subprocess.run(createLibraryScript, cwd=outputDir, check=True)

    coList = []
    libList = []
    coExt = "co"
    libExt = "yaml" if libraryFormat == "yaml" else "dat"
    with open(os.path.join(outputDir,"library","TensileManifest.txt"), "r") as f:
      lines = f.read().split("\n")
      coList = [line for line in lines if coExt in line]
      libList = [line for line in lines if libExt in line]

    logicFiles = [os.path.join(logicFileDir, f) for f in os.listdir(logicFileDir) \
      if (os.path.isfile(os.path.join(logicFileDir, f)) and os.path.splitext(f)[1]==".yaml")]

    clientParametersPaths = []
    isaStr = "".join([str(e) for e in Common.globalParameters["CurrentISA"]])
    for logicFileName in logicFiles:
      (scheduleName, archName, problemType, _, exactLogic, newLibrary) = \
          LibraryIO.parseLibraryLogicFile(logicFileName)
      problemSizes = ProblemSizesMock(random.sample(exactLogic, min(len(exactLogic), 16))) # sample at most 16 problems
      if isaStr in archName:
        clientParametersPaths.append(ClientWriter.writeClientConfig(
                                      forBenchmark=False,
                                      solutions=None,
                                      problemSizes=problemSizes,
                                      stepName=str(ProblemType(problemType)),
                                      stepBaseDir=outputDir,
                                      newLibrary=newLibrary,
                                      configBase="ClientParameters_%s_%s"%(scheduleName, str(ProblemType(problemType))),
                                      codeObjectFiles=coList,
                                      tileAwareSelection=False,
                                      libraryFile=libList[0]))

    forBenchmark = False
    enableTileSelection = False
    returncode = ClientWriter.runClient(logicFileDir, forBenchmark, enableTileSelection, clientParametersPaths)

    assert(returncode == 0)
