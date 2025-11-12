################################################################################
#
# Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import contextlib
import functools
import glob
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, call, mock_open, patch

import pytest
import yaml

import Tensile.ClientWriter as ClientWriter
import Tensile.Common as Common
import Tensile.LibraryIO as LibraryIO
import Tensile.SolutionLibrary as SolutionLibrary
import Tensile.TensileCreateLibrary as tcl
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriterBase import KernelWriterBase
from Tensile.KernelWriterSource import KernelWriterSource
from Tensile.SolutionStructs import ProblemSizes, Solution
from Tensile.Utilities.ConditionalImports import yamlLoader
from Tensile.Utilities.Toolchain import ToolchainDefaults, validateToolchain

mylogger = logging.getLogger()


@pytest.fixture
def mock_openFile():
    with patch("builtins.open", mock_open()) as mock:
        yield mock


@pytest.fixture
def mock_toFile():
    with patch("Tensile.TensileCreateLibrary.toFile") as mock:
        yield mock


@pytest.fixture
def mock_getKernelSourceAndHeaderCode():
    with patch("Tensile.TensileCreateLibrary.getKernelSourceAndHeaderCode") as mock:
        yield mock


@pytest.fixture
def mock_printWarning():
    with patch("Tensile.TensileCreateLibrary.printWarning") as mock:
        yield mock


@pytest.fixture
def mock_kernelSourceAndHeaderFiles():
    return MagicMock(name="source_file_mock"), MagicMock(name="header_file_mock")


@pytest.fixture
def mock_KernelWriterBase():
    mock = MagicMock(spec=KernelWriterBase)
    mock.getKernelName.return_value = "TestKernelName"
    return mock


def test_loadSolutions(caplog, useGlobalParameters):
    with useGlobalParameters():
        mylogger.debug("this is a test of debug log")
        mylogger.info("this is some info")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
        solutionsFilePath = os.path.join(dataDir, "solutions", "solutions_nn_3.yaml")

        fileSolutions = LibraryIO.parseSolutionsFile(solutionsFilePath)
        solutions = fileSolutions[1]
        kernels, _, _ = tcl.generateKernelObjectsFromSolutions(solutions)
        assert len(solutions) == 3
        assert len(kernels) == 3

        _, kernelWriterAssembly, _, _ = tcl.getKernelWriters(
            solutions, kernels, removeTemporaries=False
        )

        expectedKernelName0 = "Cijk_Ailk_Bljk_SB_MT128x128x2_SE_K1_TT8_8_WG16_16_1"
        expectedKernelName1 = "Cijk_Ailk_Bljk_SB_MT64x64x2_SE_K1_TT4_4_WG16_16_1"
        expectedKernelName2 = "Cijk_Ailk_Bljk_SB_MT64x64x2_SE_K1_TT4_8_WG16_8_1"

        actualKernelName0 = kernelWriterAssembly.getKernelName(kernels[0])
        actualKernelName1 = kernelWriterAssembly.getKernelName(kernels[1])
        actualKernelName2 = kernelWriterAssembly.getKernelName(kernels[2])

        assert expectedKernelName0 == actualKernelName0
        assert expectedKernelName1 == actualKernelName1
        assert expectedKernelName2 == actualKernelName2


@pytest.mark.skip(reason="System issue with find assempler called when assigning defaults")
def test_WriteClientLibraryFromSolutions(tmpdir):
    Common.globalParameters["MergeFiles"] = True
    Common.globalParameters["CodeObjectVersion"] = "default"
    Common.globalParameters["YAML"] = True
    Common.globalParameters["CxxCompiler"] = "amdclang++"
    Common.assignGlobalParameters({})

    libraryWorkingPath = tmpdir.mkdir("lib")
    buildWorkingPath = tmpdir.mkdir("build")

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    solutionsFilePath = os.path.join(dataDir, "solutions", "solutions_nn_3.yaml")

    fileSolutions = LibraryIO.parseSolutionsFile(solutionsFilePath)
    solutions = fileSolutions[1]

    Common.setWorkingPath(buildWorkingPath)
    tcl.WriteClientLibraryFromSolutions(solutions, libraryWorkingPath)
    Common.popWorkingPath()

    tensileLibraryPath = os.path.join(libraryWorkingPath, "library")

    hsacoFiles = glob.glob(tensileLibraryPath + "/*hsaco")
    assert len(hsacoFiles) > 0

    coFiles = glob.glob(tensileLibraryPath + "/*TensileLibrary*co")
    assert len(coFiles) > 0

    tensileYamlFilePath = os.path.join(tensileLibraryPath, "TensileLibrary.yaml")
    assert os.path.exists(tensileYamlFilePath) == 1

    config = None
    try:
        stream = open(tensileYamlFilePath, "r")
    except IOError:
        mylogger.error("Cannot open file: %s" % tensileYamlFilePath)
    config = yaml.load(stream, yamlLoader)
    stream.close()
    actualSolutions = config["solutions"]

    assert len(actualSolutions) == 3

    metadataYamlFilePath = os.path.join(tensileLibraryPath, "metadata.yaml")
    assert os.path.exists(metadataYamlFilePath) == 1

    metadata = None
    try:
        stream = open(metadataYamlFilePath, "r")
    except IOError:
        mylogger.error("Cannot open file: %s" % metadataYamlFilePath)
    metadata = yaml.load(stream, yamlLoader)
    stream.close()
    actualProblemType = metadata["ProblemType"]

    assert len(actualProblemType) > 0


def test_CreateBenchmarkClientParametersForSizes(tmpdir):

    Common.globalParameters["CurrentISA"] = (9, 0, 6)
    dataWorkingPath = tmpdir.mkdir("Data")
    configWorkingPath = tmpdir.mkdir("run_configs")
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    testDataPath = os.path.join(dataDir, "library_data")
    libraryPath = os.path.join(testDataPath, "library")
    metadataFilepath = os.path.join(libraryPath, "metadata.yaml")

    metadataFile = LibraryIO.readYAML(metadataFilepath)
    problemTypeDict = metadataFile["ProblemType"]
    sizes = [{"Exact": [196, 256, 64, 1024]}]
    problemSizes = ProblemSizes(problemTypeDict, sizes)

    dataFilePath = os.path.join(dataWorkingPath, "results.csv")
    configFile = os.path.join(configWorkingPath, "ClientParameters.ini")
    ClientWriter.CreateBenchmarkClientParametersForSizes(
        testDataPath, problemSizes, dataFilePath, configFile
    )

    assert os.path.exists(configFile) == 1


def test_verifyManifest():

    manifestFile = Path("test_manifest.txt")
    testFoo = Path("foo.asm")
    testBar = Path("bar.asm")

    # ensure clean state before running test
    with contextlib.suppress(FileNotFoundError):
        os.remove(manifestFile)
        os.remove(testFoo)
        os.remove(testBar)

    # Verification should fail if the manifest can't be found
    with pytest.raises(
        FileNotFoundError, match=r"(.*) No such file or directory: 'test_manifest.txt'"
    ):
        tcl.verifyManifest(manifestFile)

    # Create an empty manifest
    with open(manifestFile, mode="x") as manifest:

        assert tcl.verifyManifest(manifestFile), "an empty manifest should always succeed"

        # add to file manifest that is not on disk
        manifest.write("foo.asm\n")
        manifest.flush()
        assert not tcl.verifyManifest(
            manifestFile
        ), "file in manifest are on disk, but shouldn't be"

        with open(testFoo, mode="x"):
            assert tcl.verifyManifest(manifestFile), "file in manifest isn't on disk, but should be"

        manifest.write("bar.asm\n")
        manifest.flush()
        assert not tcl.verifyManifest(manifestFile), "bar.asm in manifest should not be on disk"

    with open(testBar, mode="x"):
        assert tcl.verifyManifest(manifestFile), "files in manifest isn't on disk, but should be"

    with open(manifestFile, "a") as generatedFile:
        for filePath in range(5):
            generatedFile.write("%s\n" % (filePath))

    assert not tcl.verifyManifest(manifestFile), "files in manifest are on disk, but shouldn't be"


def test_findLogicFiles():

    def setup():
        baseDir = Path("no-commit-test-logic-files")

        # Start with clean state
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(baseDir)
        baseDir.mkdir()

        lazyLoading = False
        experimentalDir = "/experimental/"
        return baseDir, lazyLoading, experimentalDir

    def outputMatchesOldLogic1():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")  # Remove the value `_` associated with key `all`

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, "foo", "yaml")

        result = tcl.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return set(result) == set(expected)

    def outputMatchesOldLogic2():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")  # Remove the value `_` associated with key `all`

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, d, "yaml")

        result = tcl.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return set(result) == set(expected)

    def outputMatchesOldLogic3():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap["all"])

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, d, "yaml")

        result = tcl.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return set(result) == set(expected)

    def verifyYamlAndYml():
        baseDir, lazyLoading, experimentalDir = setup()
        # Create directory structure with files that *don't* have arch in filename
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / "yaml" / d, d, "yaml")
            createDirectoryWithYamls(baseDir / "yml" / d, d, "yml")

        result = tcl.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return len(result) == len(expected) * 2

    assert outputMatchesOldLogic1(), "Output differs from old logic, not backwards compatible."
    assert outputMatchesOldLogic2(), "Output differs from old logic, not backwards compatible."
    assert outputMatchesOldLogic3(), "Output differs from old logic, not backwards compatible."
    assert (
        verifyYamlAndYml()
    ), "Output should have twice as many files as old logic (which only parses .yaml)"


def test_sanityCheck():
    # setup some dummy lists with files
    srcPaths = ["foo.hsaco", "bar.hsaco"]
    asmPaths = ["baz.co", "gru.co"]
    coPathsMatch = ["foo.hsaco", "bar.hsaco", "baz.co", "gru.co"]
    coPathsExtra = ["foo.hsaco", "bar.hsaco", "baz.co", "gru.co", "tux.hsaco"]
    coPathsMissing = ["foo.hsaco", "bar.hsaco", "baz.co"]

    tcl.sanityCheck(srcPaths, asmPaths, coPathsMatch, False)
    # Ensure that old logic also succeeds
    sanityCheck_oldLogic(srcPaths, asmPaths, coPathsMatch, False)

    with pytest.raises(ValueError, match=r"(.*) unexpected code object files: \['tux.hsaco'\]"):
        tcl.sanityCheck(srcPaths, asmPaths, coPathsExtra, False)
    # Ensure that old logic also fails
    with pytest.raises(Exception):
        try:
            sanityCheck_oldLogic(srcPaths, asmPaths, coPathsExtra, False)
        except:
            raise Exception

    with pytest.raises(ValueError, match=r"(.*) missing expected code object files: \['gru.co'\]"):
        tcl.sanityCheck(srcPaths, asmPaths, coPathsMissing, False)
    # Ensure that old logic also fails
    with pytest.raises(Exception):
        try:
            sanityCheck_oldLogic(srcPaths, asmPaths, coPathsMissing, False)
        except:
            raise Exception


def test_generateClientConfig():

    outputPath: Path = Path.cwd() / "my-output"
    masterLibrary: Path = outputPath / "masterlib.data"
    codeObjectFiles = ["library/foo.hsaco", "library/bar.co"]
    configFile = outputPath / "best-solution.ini"

    cleanClientConfigTest(outputPath, masterLibrary, codeObjectFiles, configFile)

    with pytest.raises(
        FileNotFoundError,
        match=r"(.*) No such file or directory: '(.*)/my-output/best-solution.ini'",
    ):
        tcl.generateClientConfig(outputPath, masterLibrary, codeObjectFiles)

    setupClientConfigTest(outputPath, masterLibrary, codeObjectFiles)

    tcl.generateClientConfig(outputPath, masterLibrary, codeObjectFiles)

    assert configFile.is_file(), "{configFile} was not generated"

    with open(configFile, "r") as f:
        result = f.readlines()
        assert "library-file" in result[0], "missing library-file entry"
        assert "code-object" in result[1], "missing code-object entry"
        assert "code-object" in result[2], "missing code-object entry"
        assert "best-solution" in result[3], "missing best-solution entry"

    cleanClientConfigTest(outputPath, masterLibrary, codeObjectFiles, configFile)


class MasterLibraryMock:
    def __init__(self, libraries, data):
        self.lazyLibraries = libraries
        self.data = data


def test_generateMasterFileList():

    archs = ["arch0", "arch1"]
    libraries = {
        "arch0": MasterLibraryMock({"mylib0": MasterLibraryMock(None, 2)}, 0),
        "arch1": MasterLibraryMock({"mylib1": MasterLibraryMock(None, 3)}, 1),
    }

    result = tcl.generateMasterFileList(libraries, archs, lazy=False)
    for idx, t in enumerate(result):
        assert t[0] == "TensileLibrary_arch" + str(idx), "Incorrect naming for key."
        assert isinstance(t[1], MasterLibraryMock), "Incorrect type for value."
        assert t[1].data == idx, "Incorrect data."

    result = tcl.generateMasterFileList(libraries, archs, lazy=True)

    for idx, t in enumerate(result[0:2]):
        assert t[0] == "TensileLibrary_lazy_arch" + str(idx), "Incorrect naming for key."
        assert isinstance(t[1], MasterLibraryMock), "Incorrect type for value."
        assert t[1].data == idx, "Incorrect data."

    for idx, t in enumerate(result[2:4]):
        assert t[0] == "mylib" + str(idx), "Incorrect naming for key."
        assert isinstance(t[1], MasterLibraryMock), "Incorrect type for value."
        assert t[1].data == (idx + 2), "Incorrect data."


def test_logicDataAndSolutionsConstruction(initGlobalParametersForTCL):

    def testCase1(logicFiles: List[LibraryIO.LibraryLogic], separateArch: bool):
        # clear the set to prevent testing errors caused
        # by the fact that ArchitectureSet is shared across
        # instances of MasterSolutionLibrary.
        SolutionLibrary.MasterSolutionLibrary.ArchitectureSet.clear()

        masterLibraries = tcl.makeMasterLibraries(logicFiles, separate=separateArch)
        arch = "gfx900" if separateArch else "full"

        if separateArch:
            assert (
                len(masterLibraries[arch].solutions.values()) == 17
            ), f"There should be 17 solutions prior to adding the fallback for {arch}."

            tcl.addFallback(masterLibraries)

            assert (
                len(masterLibraries[arch].solutions.values()) == 19
            ), f"There should be 19 solutions after adding the fallback for {arch}."
        else:
            assert (
                len(masterLibraries[arch].solutions.values()) == 19
            ), f"There should be 19 solutions for {arch}."

        solutions = tcl.generateSolutions(masterLibraries, separate=separateArch)
        assert isinstance(solutions, list), "generateSolutions should return a list."
        assert len(solutions) == 19, "There should be 19 solutions after adding the fallback."

    def testCase2(yamlFiles: List[str], separateArch: bool):
        # clear the set to prevent testing errors caused
        # by the fact that ArchitectureSet is shared across
        # instances of MasterSolutionLibrary.
        SolutionLibrary.MasterSolutionLibrary.ArchitectureSet.clear()

        masterLibraries = tcl.generateLogicData(
            yamlFiles, version="foo", printLevel=0, separate=separateArch
        )
        arch = "gfx900" if separateArch else "full"

        assert (
            len(masterLibraries[arch].solutions.values()) == 19
        ), f"There should be 19 solutions for {arch}."

        solutions = tcl.generateSolutions(masterLibraries, separate=separateArch)
        assert isinstance(solutions, list), "generateSolutions should return a list."
        assert len(solutions) == 19, "There should be 19 solutions after adding the fallback."

    def testCase3(logicFiles: List[LibraryIO.LibraryLogic]):
        # clear the set to prevent testing errors caused
        # by the fact that ArchitectureSet is shared across
        # instances of MasterSolutionLibrary.
        SolutionLibrary.MasterSolutionLibrary.ArchitectureSet.clear()

        masterLibraries = tcl.makeMasterLibraries(logicFiles, separate=True)
        arch = "gfx900"

        assert (
            len(masterLibraries[arch].lazyLibraries.keys()) == 1
        ), f"There should be 1 key prior to adding the fallback for {arch}."
        assert (
            len(next(iter(masterLibraries[arch].lazyLibraries.values())).solutions.values()) == 17
        ), f"There should be 17 solutions prior to adding the fallback for {arch}."

        tcl.addFallback(masterLibraries)

        assert (
            len(masterLibraries[arch].lazyLibraries.keys()) == 2
        ), f"There should be 2 keys after adding the fallback for {arch}."

        for name, lib in masterLibraries[arch].lazyLibraries.items():
            if "fallback" in name:
                assert len(lib.solutions.values()) == 2, "There should be 2 fallback solutions."
            else:
                assert len(lib.solutions.values()) == 17, "There should be 17 gfx900 solutions."

        solutions = tcl.generateSolutions(masterLibraries, separate=True)
        assert isinstance(solutions, list), "generateSolutions should return a list."
        assert len(solutions) == 19, "There should be 19 solutions after adding the fallback."

    requiredArgs = ["--jobs=2", "/unused/logic/path", "/unused/output/path", "HIP"]
    rootPath = Path(__file__).parent.parent / "test_data" / "unit" / "solutions"
    yamlFiles = [
        rootPath / f for f in ["vega10_Cijk_Ailk_Bjlk_CB_GB.yaml", "hip_Cijk_Ailk_Bjlk_CB_GB.yaml"]
    ]

    with initGlobalParametersForTCL(["--architecture=gfx900"] + requiredArgs):
        logicFiles = tcl.parseLibraryLogicFiles(yamlFiles)
        assert len(logicFiles) == 2, "The length of the logic files list is incorrect."

        testCase1(logicFiles, separateArch=True)
        testCase2(yamlFiles, separateArch=True)

    with initGlobalParametersForTCL(["--architecture=gfx900"] + requiredArgs):
        logicFiles = tcl.parseLibraryLogicFiles(yamlFiles)
        assert len(logicFiles) == 2, "The length of the logic files list is incorrect."

        testCase1(logicFiles, separateArch=False)
        testCase2(yamlFiles, separateArch=False)

    with initGlobalParametersForTCL(
        ["--architecture=gfx900", "--lazy-library-loading"] + requiredArgs
    ):
        logicFiles = tcl.parseLibraryLogicFiles(yamlFiles)
        assert len(logicFiles) == 2, "The length of the logic files list is incorrect."
        testCase3(logicFiles)


@pytest.fixture
def unittestPath(request):
    """Returns the path to the directory containing the current test file"""
    return request.path.parent


@pytest.fixture
def setupSolutionsAndKernels(
    unittestPath,
) -> Tuple[List[Solution], List[Solution], KernelWriterAssembly, KernelWriterSource]:
    """Reusable logic for setting up testable solutions and kernels"""

    (cxxCompiler, cCompiler, assembler, offloadBundler, hipconfig, deviceEnumerator) = (
        validateToolchain(
            ToolchainDefaults.CXX_COMPILER,
            ToolchainDefaults.C_COMPILER,
            ToolchainDefaults.ASSEMBLER,
            ToolchainDefaults.OFFLOAD_BUNDLER,
            ToolchainDefaults.HIP_CONFIG,
            ToolchainDefaults.DEVICE_ENUMERATOR,
        )
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
    _, _, _, _, _, lib = LibraryIO.parseLibraryLogicFile(
        unittestPath.parent / "test_data" / "unit" / "aldebaran_Cijk_AlikC_Bljk_ZB_GB.yaml"
    )
    solutions = [sol.originalSolution for sol in lib.solutions.values()]
    kernels, _, _ = tcl.generateKernelObjectsFromSolutions(solutions)
    kernelWriterSource, kernelWriterAssembly, _, _ = tcl.getKernelWriters(
        solutions, kernels, removeTemporaries=False
    )
    return solutions, kernels, kernelWriterAssembly, kernelWriterSource


def test_prepAsm(setupSolutionsAndKernels):
    solutions, kernels, kernelWriterAssembly, kernelWriterSource = setupSolutionsAndKernels
    buildPath = Path("no-commit-prep-asm")
    buildPath.mkdir(exist_ok=True)

    def testLinux():
        tcl.prepAsm(kernelWriterAssembly, True, Path("no-commit-prep-asm"), (9, 0, 10), 1)

        expected = """#!/bin/sh
# usage: asm-new.sh kernelName(no extension) [--wave32]
f=$1
shift
if [ ! -z "$1" ] && [ "$1" = "--wave32" ]; then
    wave=32
    shift
else
    wave=64
fi
h=gfx90a
if [ $wave -eq 32 ]; then
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mno-wavefrontsize64 -c -o $f.o $f.s
else
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mwavefrontsize64 -c -o $f.o $f.s
fi
/opt/rocm/bin/amdclang++ -target amdgcn-amd-amdhsa -Xlinker --build-id -o $f.co $f.o
cp $f.co ../../../library/${f}_$h.co
mkdir -p ../../../asm_backup && cp $f.s ../../../asm_backup/$f.s
"""

        with open(buildPath / "assembly" / "asm-new.sh", "r") as f:
            contents = f.read()
            assert contents == expected, "Assembler script doesn't match expectation"

    def testWindows():
        tcl.prepAsm(kernelWriterAssembly, False, Path("no-commit-prep-asm"), (9, 0, 10), 1)

        expected = """@echo off
set f=%1

set arg2=--wave64
if [%2] NEQ [] set arg2=%2

set /A wave=64
if %arg2% EQU --wave32 set /A wave=32

set h=gfx90a
if %wave% == 32 (/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mno-wavefrontsize64 -c -o %f%.o %f%.s) else (/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mwavefrontsize64 -c -o %f%.o %f%.s)
/opt/rocm/bin/amdclang++ -target amdgcn-amd-amdhsa -Xlinker --build-id -o %f%.co %f%.o
copy %f%.co ..\\..\\..\library\%f%_%h%.co
"""

        with open(buildPath / "assembly" / "asm-new.bat", "r") as f:
            contents = f.read()
            assert contents == expected, "Assembler script doesn't match expectation"

    testLinux()
    testWindows()


class MockSolution:
    def __init__(self, name: str, lang: str):
        self.name = name
        self.lang = lang

    def __getitem__(self, key):
        if key == "KernelLanguage":
            return self.lang
        raise KeyError(f"Key {key} not found")


class MockKernelWriter:
    def getKernelFileBase(self, kernel: MockSolution):
        return kernel.name

    def getKernelName(self, kernel: MockSolution):
        return f"{kernel.name}_name"


def test_markDuplicateKernels():

    kernelsAsm = [MockSolution(name, "Assembly") for name in ["A", "B", "C"]]
    kernelWriterAssembly = MockKernelWriter()
    kernelsOut = tcl.markDuplicateKernels(kernelsAsm, kernelWriterAssembly)

    assert len(kernelsOut) == len(kernelsAsm), "Lengths of input and output should match"
    assert all([not k.duplicate for k in kernelsOut]), "All kernels should be unique"

    kernelsAsm = [MockSolution(name, "Assembly") for name in ["A", "B", "B", "C"]]
    kernelsOut = tcl.markDuplicateKernels(kernelsAsm, kernelWriterAssembly)

    assert len(kernelsOut) == len(kernelsAsm), "Lengths of input and output should match"
    for i in range(len(kernelsOut)):
        isDup = kernelsOut[i].duplicate
        assert isDup if i == 2 else not isDup, "Duplicate status is incorrect"

    kernelsSrc = [MockSolution(name, "Source") for name in ["D", "E", "E", "F"]]
    kernelsOut = tcl.markDuplicateKernels(kernelsSrc, kernelWriterAssembly)

    assert len(kernelsOut) == len(kernelsSrc), "Lengths of input and output should match"
    for i in range(len(kernelsOut)):
        with pytest.raises(
            AttributeError, match="'MockSolution' object has no attribute 'duplicate'"
        ):
            kernelsOut[i].duplicate

    kernelsAll = kernelsSrc + kernelsAsm
    kernelsOut = tcl.markDuplicateKernels(kernelsAll, kernelWriterAssembly)

    assert len(kernelsOut) == len(kernelsAll), "Lengths of input and output should match"
    for i in range(len(kernelsAll)):
        if i < len(kernelsSrc):
            with pytest.raises(
                AttributeError, match="'MockSolution' object has no attribute 'duplicate'"
            ):
                kernelsOut[i].duplicate
        else:
            isDup = kernelsOut[i].duplicate
            assert isDup if i == 6 else not isDup, "Duplicate status is incorrect"


@pytest.mark.skip(reason="Debugging function")
def test_filterProcessingErrors(setupSolutionsAndKernels):
    solutions, kernels, kernelWriterAssembly, kernelWriterSource = setupSolutionsAndKernels
    kernels = tcl.markDuplicateKernels(kernels, kernelWriterAssembly)

    results = [(-2, 0, 0, 0, 0)] * len(kernels)

    kernelsOut, solutionsOut, resultsOut = tcl.filterProcessingErrors(
        kernels, solutions, results, printLevel=1, ignoreErr=True
    )
    assert len(kernelsOut) == 0, "Kernels should be of length zero"
    assert len(solutionsOut) == 0, "Solutions should be of length zero"
    assert len(resultsOut) == 0, "Results should be of length zero"

    results = [(-2, 0, 0, 0, 0)] + [(0, 0, 0, 0, 0)] * (len(kernels) - 1)
    with pytest.raises(ValueError, match=r"Found 1 error\(s\) (.*)"):
        tcl.filterProcessingErrors(kernels, solutions, results, printLevel=1, ignoreErr=False)


def test_processKernelSource(setupSolutionsAndKernels):
    _, kernels, kernelWriterAssembly, kernelWriterSource = setupSolutionsAndKernels

    kernels = tcl.markDuplicateKernels(kernels, kernelWriterAssembly)

    fn = functools.partial(
        tcl.processKernelSource,
        kernelWriterSource=kernelWriterSource,
        kernelWriterAssembly=kernelWriterAssembly,
    )

    results = list(map(fn, kernels))
    expected = [
        (
            0,
            "",
            "",
            "Cijk_AlikC_Bljk_ZB_GB_MT16x16x16_MI16x16x4x1_SN_w7sFP__lQopabvLxzkL9V_rkJczrhhWwRy4dGREBaJY=",
            None,
        ),
        (
            0,
            "",
            "",
            "Cijk_AlikC_Bljk_ZB_GB_MT64x96x8_MI16x16x4x1_SN_BT-2Hl1ATHdIekMN_ygeV-CN0NYIERWlaWZvB-9FqxI0=",
            None,
        ),
        (
            0,
            "",
            "",
            "Cijk_AlikC_Bljk_ZB_GB_MT128x64x8_MI16x16x4x1_SN_jLNBWWn_GYkesnvb5x6jwDhR80VmCUqMaBE6X8-iojw=",
            None,
        ),
    ]

    assert results == expected, "Assembly files shouldn't have any header or source content"


def test_generateKernelSourceAndHeaderFiles_generic():
    outputPath = Path("no-commit-kernel-build-files")
    outputPath.mkdir(exist_ok=True)

    results: List[tcl.ProcessedKernelResult] = [
        (-2, "", "", "asm1", None),
        (0, "", "", "asm2", None),
        (0, "", "", "asm3", None),
        (-2, '#include "Kernels1.h"', "#pragma once", "src1", None),
        (0, '#include "Kernels2.h"', "#pragma twice", "src2", None),
        (0, '#include "Kernels3.h"', "#pragma thrice", "src3", None),
    ]

    filesToWrite = tcl.collectFilesToWrite(results, Path(outputPath), True, True, 1)

    kernelFiles = tcl.generateKernelSourceAndHeaderFiles(filesToWrite)

    # Undocumented internal logic of generateKernelSourceAndHeaderFiles
    assert len(kernelFiles) == 1, "Only one file should be created when mergeFiles == True"

    assert (
        kernelFiles[0] == "no-commit-kernel-build-files/Kernels.cpp"
    ), "Cpp source file doesn't match"


def test_generateKernelSourceAndHeaderFiles_noLazyMergeFallbackNames():
    outputPath = Path("no-commit-kernel-build-files")
    outputPath.mkdir(exist_ok=True)

    results = [
        (-2, "", "", "asm1", None),
        (0, "", "", "asm2", None),
        (0, "", "", "asm3", None),
        (-2, '#include "Kernels1.h"', "#pragma once", "src1", None),
        (0, '#include "Kernels2.h"', "#pragma twice", "src2", "kfile2"),
        (0, '#include "Kernels3.h"', "#pragma thrice", "src3", None),
    ]

    filesToWrite = tcl.collectFilesToWrite(results, Path(outputPath), False, False, 1)

    kernelFiles = tcl.generateKernelSourceAndHeaderFiles(filesToWrite)

    assert (
        len(kernelFiles) == 3
    ), "3 files should be created. Since some filenames are present and some not, we use the kernel name directly as a fallback, e.g., src1"
    assert kernelFiles == [
        "no-commit-kernel-build-files/src1.cpp",
        "no-commit-kernel-build-files/kfile2.cpp",
        "no-commit-kernel-build-files/src3.cpp",
    ]


def test_generateKernelSourceAndHeaderFiles_mergeWithNonEmptyAsm():
    outputPath = Path("no-commit-kernel-build-files")
    outputPath.mkdir(exist_ok=True)

    results = [
        (-2, "A1", "#pragma 1", "asm1", None),
        (0, "A2", "#pragma 2", "asm2", "afile2"),
        (0, "A3", "#pragma 3", "asm3", None),
    ]

    filesToWrite = tcl.collectFilesToWrite(results, Path(outputPath), False, True, 1)

    kernelFiles = tcl.generateKernelSourceAndHeaderFiles(filesToWrite)

    assert (
        len(kernelFiles) == 2
    ), "2 files should be created. Since one filename is present and merge files is True => Kernels.cpp will be created."
    assert kernelFiles == [
        "no-commit-kernel-build-files/Kernels.cpp",
        "no-commit-kernel-build-files/afile2.cpp",
    ]
    with open(outputPath / "Kernels.cpp", "r") as f:
        contents = f.readlines()
        assert contents[-1] == "A1A3"
        print(contents[-2])
        assert contents[-2] == '#include "no-commit-kernel-build-files/Kernels.h"\n'
    with open(outputPath / "Kernels.h", "r") as f:
        contents = f.readlines()
        assert contents[-1] == "#pragma 1#pragma 3"


def test_generateKernelSourceAndHeaderFiles_noMerge_WithNonEmptyAsm():
    outputPath = Path("no-commit-kernel-build-files")
    outputPath.mkdir(exist_ok=True)

    results = [
        (-2, "A1", "#pragma 1", "asm1", None),
        (0, "A2", "#pragma 2", "asm2", "afile2"),
        (0, "A3", "#pragma 3", "asm3", None),
    ]

    filesToWrite = tcl.collectFilesToWrite(results, Path(outputPath), False, False, 1)

    kernelFiles = tcl.generateKernelSourceAndHeaderFiles(filesToWrite)

    assert (
        len(kernelFiles) == 3
    ), "3 files should be created. Since some filenames are present and some not (src kernels), we use the kernel name directly as a fallback, e.g., src1"
    assert kernelFiles == [
        "no-commit-kernel-build-files/asm1.cpp",
        "no-commit-kernel-build-files/afile2.cpp",
        "no-commit-kernel-build-files/asm3.cpp",
    ]


def test_generateKernelSourceAndHeaderFiles_lazyMerge3Src():

    outputPath = Path("no-commit-kernel-build-files")
    outputPath.mkdir(exist_ok=True)

    results = [
        (0, "", "", "asm1", None),
        (0, "", "", "asm2", None),
        (0, "", "", "asm3", "blah.asm"),
        (0, '#include "Kernels1.h"', "#pragma once", "src1", "kfile1"),
        (0, '#include "Kernels2.h"', "#pragma twice", "src2", "kfile2"),
        (0, '#include "Kernels3.h"', "#pragma thrice", "src3", "kfile3"),
    ]

    filesToWrite = tcl.collectFilesToWrite(results, Path(outputPath), True, True, 1)

    kernelFiles = tcl.generateKernelSourceAndHeaderFiles(filesToWrite)

    assert (
        len(kernelFiles) == 4
    ), "4 files should be created because they have file names AND some source code AND lazy loading/merge files is True"
    assert kernelFiles == [
        "no-commit-kernel-build-files/kfile1.cpp",
        "no-commit-kernel-build-files/kfile2.cpp",
        "no-commit-kernel-build-files/kfile3.cpp",
        "no-commit-kernel-build-files/Kernels.cpp",
    ]


def test_generateKernelSourceAndHeaderFiles_noLazyMerge3Src():

    outputPath = Path("no-commit-kernel-build-files")
    outputPath.mkdir(exist_ok=True)

    results = [
        (0, "", "", "asm1", None),
        (0, "", "", "asm2", None),
        (0, "", "", "asm3", "blah.asm"),
        (0, '#include "Kernels1.h"', "#pragma once", "src1", "kfile1"),
        (0, '#include "Kernels2.h"', "#pragma twice", "src2", "kfile2"),
        (0, '#include "Kernels3.h"', "#pragma thrice", "src3", "kfile3"),
    ]

    filesToWrite = tcl.collectFilesToWrite(results, Path(outputPath), False, False, 1)

    kernelFiles = tcl.generateKernelSourceAndHeaderFiles(filesToWrite)

    assert (
        len(kernelFiles) == 3
    ), "3 files should be created because they have file names AND some source code, but lazy loading/merge files is False"
    assert kernelFiles == [
        "no-commit-kernel-build-files/kfile1.cpp",
        "no-commit-kernel-build-files/kfile2.cpp",
        "no-commit-kernel-build-files/kfile3.cpp",
    ]


def test_filterBuildErrors():

    kernels = [MockSolution(name, "Assembly") for name in ["A", "B", "C"]]
    kernels += [MockSolution(name, "Source") for name in ["D", "E", "F"]]
    kernelWriter = MockKernelWriter()
    kernelsWithBuildErrors = {
        "A_name": -2,
        "D_name": -2,
    }
    writerSelector = lambda lang: kernelWriter

    def noBuildFailures():
        kernelsToBuild = tcl.filterBuildErrors(kernels, {}, writerSelector, ignoreErr=False)
        assert kernelsToBuild == kernels, "All kernels should be built without failure"

    def buildFailuresIgnoreErr():

        expectedKernelsToBuild = [MockSolution(name, "Assembly") for name in ["B", "C"]]
        expectedKernelsToBuild += [MockSolution(name, "Source") for name in ["E", "F"]]

        kernelsToBuild = tcl.filterBuildErrors(
            kernels, kernelsWithBuildErrors, writerSelector, ignoreErr=True
        )

        print("Kernels wtih build errors:", [k for k in kernelsWithBuildErrors])
        print("Kernels to build:", [k.name for k in kernelsToBuild])
        assert len(kernelsToBuild) == len(
            expectedKernelsToBuild
        ), "Length of built kernels is incorrect"

        assert all(
            [
                k.name == e.name and k.lang == e.lang
                for k, e in zip(kernelsToBuild, expectedKernelsToBuild)
            ]
        ), "Kernels should be filtered to only those without build errors"

    def buildFailuresNoIgnoreErr():
        with pytest.raises(RuntimeError, match=r"Kernel compilation failed (.*)"):
            tcl.filterBuildErrors(kernels, kernelsWithBuildErrors, writerSelector, ignoreErr=False)

    noBuildFailures()
    buildFailuresIgnoreErr()
    buildFailuresNoIgnoreErr()


@pytest.fixture
def setup_writeKernelHelpersTests():
    kernelFiles = []
    kernWriter = MockKernelWriter()
    basepath = Path("/fake/path")
    return kernelFiles, kernWriter, basepath


def test_writeKernelHelpers_createFiles(
    setup_writeKernelHelpersTests, mock_toFile, mock_openFile, mock_getKernelSourceAndHeaderCode
):
    kernelFiles, kernWriter, basepath = setup_writeKernelHelpersTests
    mock_getKernelSourceAndHeaderCode.return_value = (
        0,
        ["source_code", "abc"],
        ["header_code", "def"],
        "kernelName",
    )

    tcl.writeKernelHelpers(kernWriter, None, None, basepath, kernelFiles)

    assert mock_toFile.call_args_list == [
        call(basepath / "Kernels" / "kernelName.cpp", ["source_code", "abc"]),
        call(basepath / "Kernels" / "kernelName.h", ["header_code", "def"]),
    ]
    assert kernelFiles == [
        "/fake/path/Kernels/kernelName.cpp"
    ], "kernelFiles should be updated with the path to the new kernel"


def test_writeKernelHelpers_withOpenFiles(
    setup_writeKernelHelpersTests,
    mock_toFile,
    mock_getKernelSourceAndHeaderCode,
    mock_kernelSourceAndHeaderFiles,
):
    kernelSourceFile, kernelHeaderFile = mock_kernelSourceAndHeaderFiles
    kernelFiles, kernWriter, basepath = setup_writeKernelHelpersTests
    mock_getKernelSourceAndHeaderCode.return_value = (
        0,
        ["source_code", "abc"],
        ["header_code", "def"],
        "kernelName",
    )

    tcl.writeKernelHelpers(kernWriter, kernelSourceFile, kernelHeaderFile, basepath, kernelFiles)

    expected_calls = [
        call(kernelSourceFile, ["source_code", "abc"]),
        call(kernelHeaderFile, ["header_code", "def"]),
    ]
    assert mock_toFile.call_args_list == expected_calls
    assert kernelFiles == [], "kernelFiles should remain unchanged when opened files are provided"


def test_writeKernelHelpers_failure(
    setup_writeKernelHelpersTests, mock_toFile, mock_printWarning, mock_getKernelSourceAndHeaderCode
):
    kernelFiles, kernWriter, basepath = setup_writeKernelHelpersTests
    mock_getKernelSourceAndHeaderCode.return_value = (
        -2,
        ["// src comment", ""],
        ["// hdr comment", ""],
        "kernelName",
    )

    tcl.writeKernelHelpers(kernWriter, None, None, basepath, kernelFiles)

    mock_printWarning.assert_called_once_with("Invalid kernel: kernelName may be corrupt")
    expected_calls = [
        call(basepath / "Kernels" / "kernelName.cpp", ["// src comment", ""]),
        call(basepath / "Kernels" / "kernelName.h", ["// hdr comment", ""]),
    ]
    assert mock_toFile.call_args_list == expected_calls
    assert kernelFiles == [
        "/fake/path/Kernels/kernelName.cpp"
    ], "kernelFiles should be updated with the new kernel name"


def test_getKernelSourceAndHeaderCode_success(mock_KernelWriterBase):
    mock_KernelWriterBase.getSourceFileString.return_value = (0, "source_code")
    mock_KernelWriterBase.getHeaderFileString.return_value = "header_code"

    err, src, hdr, name = tcl.getKernelSourceAndHeaderCode(mock_KernelWriterBase)

    mock_KernelWriterBase.getKernelName.assert_called_once_with()
    mock_KernelWriterBase.getSourceFileString.assert_called_once_with()
    mock_KernelWriterBase.getHeaderFileString.assert_called_once_with()

    assert err == 0
    assert src == [Common.CHeader, "source_code"]
    assert hdr == [Common.CHeader, "header_code"]
    assert name == "TestKernelName"


def test_getKernelSourceAndHeaderCode_sourceFailure(mock_KernelWriterBase):
    mock_KernelWriterBase.getSourceFileString.return_value = (-1, "")
    mock_KernelWriterBase.getHeaderFileString.return_value = "header_code"

    err, src, hdr, name = tcl.getKernelSourceAndHeaderCode(mock_KernelWriterBase)

    mock_KernelWriterBase.getKernelName.assert_called_once_with()
    mock_KernelWriterBase.getSourceFileString.assert_called_once_with()
    mock_KernelWriterBase.getHeaderFileString.assert_called_once_with()

    assert err == -1
    assert src == [Common.CHeader, ""]
    assert hdr == [Common.CHeader, "header_code"]
    assert name == "TestKernelName"


# ----------------
# Helper functions
# ----------------
def setupClientConfigTest(outputPath, masterLibrary, codeObjectFiles):
    outputPath.mkdir()
    with open(masterLibrary, "w") as testFile:
        testFile.write("foo")
    (outputPath / "library").mkdir()
    for f in codeObjectFiles:
        with open(outputPath / f, "w") as testFile:
            testFile.write("foo")


def cleanClientConfigTest(outputPath: Path, masterLibrary, codeObjectFiles, configFile):
    with contextlib.suppress(FileNotFoundError):
        os.remove(configFile)
        os.remove(masterLibrary)
        for f in codeObjectFiles:
            os.remove(outputPath / f)
        next(outputPath.iterdir()).rmdir()
        outputPath.rmdir()


def createDirectoryWithYamls(currentDir, prefix, ext, depth=3, nChildren=3):
    def recurse(currentDir, depth, nChildren):
        if depth == 0:
            return

        currentDir.mkdir(parents=True, exist_ok=True)
        file = f"{prefix}_{str(uuid.uuid4().hex)}.{ext}"
        with open(currentDir / file, mode="w"):
            pass

        for n in range(nChildren):
            recurse(currentDir / str(n), depth - 1, nChildren)

    recurse(currentDir, depth, nChildren)


def findLogicFiles_oldLogic(logicPath, logicArchs, lazyLoading, experimentalDir):
    # Recursive directory search
    logicFiles = []
    for root, dirs, files in os.walk(str(logicPath)):
        logicFiles += [
            os.path.join(root, f)
            for f in files
            if os.path.splitext(f)[1] == ".yaml"
            and (
                any(logicArch in os.path.splitext(f)[0] for logicArch in logicArchs)
                or "hip" in os.path.splitext(f)[0]
            )
        ]

    if not lazyLoading:
        logicFiles = [f for f in logicFiles if not experimentalDir in f]
    return logicFiles


def sanityCheck_oldLogic(sourceLibPaths, asmLibPaths, codeObjectFiles, genSourcesAndExit):
    bothLibSet = set(sourceLibPaths + asmLibPaths)
    setA = set(map(os.path.normcase, set(codeObjectFiles)))
    setB = set(map(os.path.normcase, bothLibSet))

    sanityCheck0 = setA - setB
    sanityCheck1 = setB - setA

    assert len(sanityCheck0) == 0, "Unexpected code object files: {}".format(sanityCheck0)
    if not genSourcesAndExit:
        assert len(sanityCheck1) == 0, "Missing expected code object files: {}".format(sanityCheck1)
