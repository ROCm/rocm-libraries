################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

from ..Common import (
    IsaVersion,
    ParallelMap,
    ensurePath,
    globalParameters,
    splitArchs,
    tPrint,
)


def _computeSourceCodeObjectPath(
    target: str, base: str, buildPath: Union[Path, str], arch
) -> Union[Path, None]:
    """Generates a code object file path using the target, base, and build path.

    Args:
        target: The target triple.
        base: The base name for the output file (name without extension).
        buildPath: The build directory path.

    Returns:
        Path to the code object file if a name can be computed, otherwise None, which
        indicates that the code object file should not be generated.
    """
    coPath = None
    buildPath = Path(buildPath)
    if "TensileLibrary" in base and "fallback" in base:
        coPath = buildPath / "{0}_{1}.hsaco.raw".format(base, arch)
    elif "TensileLibrary" in base:
        variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
        baseVariant = base + "-" + variant if variant else base
        if arch in baseVariant:
            coPath = buildPath / (baseVariant + ".hsaco.raw")
    else:
        coPath = buildPath / "{0}.so-000-{1}.hsaco.raw".format(base, arch)
    return coPath


def _compileSourceObjectFile(
    cmdlineArchs: List[str],
    cxxCompiler: str,
    cxxSrcPath: str,
    objDestPath: str,
    outputPath: str,
    compilerVer: IsaVersion,
):
    """Compiles a HIP source file (.cpp) into an object file (.o).

    Args:
        cmdlineArchs: List of architectures for offloading.
        cxxCompiler: The C++ compiler to use.
        cxxSrcPath: The path to the kernel source file.
        objDestPath: The build directory path.
        outputPath: The output directory path.
        compilerVer: The compiler version.

    Raises:
        subprocess.CalledProcessError: If the compilation command fails.
    """
    archFlags = ["--offload-arch=" + arch for arch in cmdlineArchs]

    hipFlags = ["-D__HIP_HCC_COMPAT_MODE__=1"]
    # TODO(@tensile-infra): hipcc is deprecated and should be removed for ROCm 6.5
    hipFlags += (
        ["--genco"]
        if os.path.basename(cxxCompiler) == "hipcc"
        else ["--cuda-device-only", "-x", "hip", "-O3"]
    )
    hipFlags += ["-I", outputPath]

    if len(compilerVer) >= 2 and (
        compilerVer[0] > 5 or (compilerVer[0] == 5 and compilerVer[1] > 2)
    ):
        hipFlags += ["-Xoffload-linker", "--build-id"]

    launcher = shlex.split(os.environ.get("Tensile_CXX_COMPILER_LAUNCHER", ""))

    if os.name == "nt":
        hipFlags += [
            "-std=c++14",
            "-fms-extensions",
            "-fms-compatibility",
            "-fPIC",
            "-Wno-deprecated-declarations",
        ]

    args = (
        launcher + [cxxCompiler] + hipFlags + archFlags + [str(cxxSrcPath), "-c", "-o", objDestPath]
    )

    tPrint(2, f"Compile source object file command: {args}")
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}" if out else "")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error compiling source object file: {err.output}\nFailed command: {' '.join(args)}"
        )


def _listTargetTriples(bundler: str, objFile: str):
    """Invokes the bundler to list target triples in an object file.

    Args:
        bundler: The path to the bundler executable, typically ``clang-offload-bundler``.
        objFile: String-like path to the object file.
    """
    args = [bundler, "--type=o", f"--input={objFile}", "-list"]
    try:
        listing = subprocess.check_output(args, stderr=subprocess.STDOUT).decode().split("\n")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error listing target triples in object files: {err.output}\nFailed command: {' '.join(args)}"
        )
    return listing


def _unbundleSourceCodeObjects(bundler: str, target: str, infile: str, outfileRaw: str):
    """Unbundles code object files from the provided target triple.

    Args:
        bundler (str): The path to the Clang Offload Bundler executable.
        target (str): The target architecture string.
        inflag (str): The input flag for the bundler.
        infile (str): The input file path.
        outflag (str): The output flag for the bundler.
        outfileRaw (str): The output raw file path.

    Raises:
        RuntimeError: If unbundling the source code object file fails.
    """
    args = [
        bundler,
        "--type=o",
        f"--targets={target}",
        f"--input={infile}",
        f"--output={outfileRaw}",
        "--unbundle",
    ]

    tPrint(2, "Unbundling source code object file: " + " ".join(args))
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}" if out else "")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error unbundling source code object file: {err.output}\nFailed command: {' '.join(args)}"
        )


def _buildSourceCodeObjectFile(
    cxxCompiler: str,
    outputPath: Union[Path, str],
    kernelPath: Union[Path, str],
    removeTemporaries: bool,
) -> List[str]:

    buildPath = Path(ensurePath(os.path.join(globalParameters["WorkingPath"], "code_object_tmp")))
    destPath = Path(ensurePath(os.path.join(outputPath, "library")))
    kernelPath = Path(kernelPath)

    objectFilename = kernelPath.stem + ".o"
    coPathsRaw = []
    coPaths = []

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
        os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    _, cmdlineArchs = splitArchs()

    # Add build-id for builds with rocm 5.3+
    compilerVer = globalParameters["HipClangVersion"].split(".")[:2]
    compilerVer = [int(c) for c in compilerVer]

    objPath = str(buildPath / objectFilename)
    _compileSourceObjectFile(
        cmdlineArchs, cxxCompiler, str(kernelPath), objPath, str(outputPath), compilerVer
    )

    bundler = globalParameters["ClangOffloadBundlerPath"]
    if not bundler:
        raise RuntimeError(
            "No bundler found; set TENSILE_ROCM_OFFLOAD_BUNDLER_PATH to point to clang-offload-bundler"
        )

    for target in _listTargetTriples(bundler, objPath):
        match = re.search("gfx.*$", target)
        if match:
            arch = re.sub(":", "-", match.group())
            coPathRaw = _computeSourceCodeObjectPath(target, kernelPath.stem, buildPath, arch)
            if not coPathRaw:
                continue
            _unbundleSourceCodeObjects(bundler, target, objPath, str(coPathRaw))

            coPath = str(destPath / coPathRaw.stem)
            coPaths.append(coPath)
            coPathsRaw.append(coPathRaw)

    for src, dst in zip(coPathsRaw, coPaths):
        shutil.copy(src, dst)
    if removeTemporaries:
        for file in coPathsRaw:
            os.remove(file)

    return coPaths


def buildSourceCodeObjectFiles(
    cxxCompiler: str, kernelFiles: List[Path], outputPath: Path, removeTemporaries: bool
) -> List[str]:
    """Compiles HIP source code files into code object files.

    Args:
        cxxCompiler: The C++ compiler to use.
        kernelFiles: List of paths to the kernel source files.
        outputPath: The output directory path where code objects will be placed.
        removeTemporaries: Whether to clean up temporary files.

    Returns:
        List of paths to the created code objects.
    """

    args = zip(
        itertools.repeat(cxxCompiler),
        itertools.repeat(outputPath),
        kernelFiles,
        itertools.repeat(removeTemporaries),
    )
    coFiles = ParallelMap(_buildSourceCodeObjectFile, args, "Compiling source kernels")

    return itertools.chain.from_iterable(coFiles)
