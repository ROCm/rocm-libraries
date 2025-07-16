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


import functools

from pathlib import Path
from typing import List, Dict, NamedTuple

from Tensile.Common import ParallelMap2, print1, IsaVersion, IsaInfo, setVerbosity
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters
from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain

from .ParseArguments import parseArguments
from .ValidMatrixInstruction import _validateMatrixInstruction
from .ValidWorkGroup import _validateWorkGroup
from .HandleCustomKernel import handleCustomKernel, hasCustomKernel


class Check(NamedTuple):
    OnlyCustomKernels: bool
    All: bool


def _runChecks(
    logicPath: Path, isaInfoMap: Dict[IsaVersion, IsaInfo], check: Check, files: List[Path]
):
    """
    Run checks on the given logic files.

    Args:
        logicPath: Path to a directory containing logic files or to an individual logic file.
        isaInfoMap: Map of IsaVersion to IsaInfo.
        check: Object containing flags for checking.
        files: List of logic files to check.

    Returns:
        Tuple of (keep, total) where keep is the number of unrejected solutions and
        total is the total number of solutions parsed.
    """
    keep, total = 0, 0
    for file in files:
        if "Experimental" in file.parts:
            return keep, total

        solutions = []
        data = readYAML(file)
        problemType = data[4]
        if check.OnlyCustomKernels and hasCustomKernel(file):
            print1(f">> {file.relative_to(logicPath)}")
            solutions = data[5]  # Solutions are the 5th index
        elif check.All:
            print1(f">> {file.relative_to(logicPath)}")
            solutions = data[5]  # Solutions are the 5th index

        for s in solutions:
            s, isCustom = handleCustomKernel(s, isaInfoMap)
            if check.OnlyCustomKernels and not isCustom:
                continue

            s["ProblemType"] = problemType
            if all(
                [
                    _validateMatrixInstruction(s, isaInfoMap, file.relative_to(logicPath)),
                    _validateWorkGroup(s, file.relative_to(logicPath)),
                ]
            ):
                keep += 1
            total += 1

    return keep, total


def _setup():
    args = parseArguments()

    setVerbosity(args.Verbose)
    jobs = int(args.Jobs)
    cxxCompiler = validateToolchain(args.CxxCompiler)
    logicPath = Path(args.LogicPath)

    if not any([args.CheckAll, args.CheckOnlyCustomKernels]):
        print1("No checks specified. Exiting.")
        exit(0)
    check = Check(
        OnlyCustomKernels=args.CheckOnlyCustomKernels,
        All=args.CheckAll,
    )

    if logicPath.is_file() and logicPath.suffix == ".yaml":
        files = [logicPath]
    else:
        pattern = "**/*.yaml"
        files = list(logicPath.glob(pattern))
    if len(files) == 0:
        print1(f"No files found in {logicPath}")
        exit(1)
    print1(f"Found {len(files)} files")

    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, str(cxxCompiler))
    assignGlobalParameters({"PrintSolutionRejectionReason": True}, isaInfoMap)

    return jobs, isaInfoMap, logicPath, files, check


def main():
    jobs, isaInfoMap, logicPath, files, check = _setup()

    batchSize = len(files) // min(len(files), jobs)
    batches = (files[i : i + batchSize] for i in range(0, len(files), batchSize))

    fn = functools.partial(_runChecks, logicPath, isaInfoMap, check)
    keep, total = 0, 0

    results = ParallelMap2(fn, batches, multiArg=False, procs=jobs, return_as="list")

    for _keep, _total in results:
        keep += _keep
        total += _total

    rejects = total - keep
    print(f"Total  {total} solutions")
    print(f"Keep   {keep} solutions")
    print(f"Reject {rejects} solutions")

    if rejects > 0:
        exit(1)
