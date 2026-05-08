################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
import sys
import threading
import time
import warnings

from pathlib import Path
from typing import FrozenSet, List, Dict, NamedTuple, Tuple

from Tensile.Common import ParallelMap2, print1, print2, IsaVersion, IsaInfo, setVerbosity
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters
from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain

from .ParseArguments import parseArguments
from .KnownBugs import KnownBugKey, is_known_bug, load_known_bugs
from .ValidMatrixInstruction import _validateMatrixInstruction
from .ValidWorkGroup import _validateWorkGroup
from .ValidWorkGroupMappingXCC import _validateWorkGroupMappingXCC, reset_reported_failures
from .HandleCustomKernel import handleCustomKernel, hasCustomKernel


class Check(NamedTuple):
    OnlyCustomKernels: bool
    All: bool


def _runChecks(
    logicPath: Path,
    isaInfoMap: Dict[IsaVersion, IsaInfo],
    check: Check,
    known_bugs: FrozenSet[KnownBugKey],
    files: List[Path],
) -> Tuple[int, int, int]:
    """
    Run checks on the given logic files.

    Args:
        logicPath: Path to a directory containing logic files or to an individual logic file.
        isaInfoMap: Map of IsaVersion to IsaInfo.
        check: Object containing flags for checking.
        files: List of logic files to check.

    Returns:
        Tuple of (keep, total, known_bug_skips) where keep is the number of
        unrejected solutions, total is the total number of solutions parsed, and
        known_bug_skips counts solutions accepted via the known-bugs list only.
    """
    keep, total, known_bug_skips = 0, 0, 0
    for file in files:
        if "Experimental" in file.parts:
            continue

        solutions = []
        data = readYAML(file)
        problemType = data[4]
        if check.OnlyCustomKernels and hasCustomKernel(file):
            print2(f">> {file.relative_to(logicPath)}")
            solutions = data[5]  # Solutions are the 5th index
        elif check.All:
            print2(f">> {file.relative_to(logicPath)}")
            solutions = data[5]  # Solutions are the 5th index

        for list_idx, s in enumerate(solutions):
            s, isCustom = handleCustomKernel(s, isaInfoMap)
            if check.OnlyCustomKernels and not isCustom:
                continue

            s["ProblemType"] = problemType
            rel = file.relative_to(logicPath)
            sol_index = int(s.get("SolutionIndex", list_idx))

            if known_bugs and is_known_bug(known_bugs, rel, sol_index):
                keep += 1
                total += 1
                known_bug_skips += 1
                continue

            if all(
                [
                    _validateMatrixInstruction(s, isaInfoMap, rel),
                    _validateWorkGroup(s, rel),
                    _validateWorkGroupMappingXCC(s, rel),
                ]
            ):
                keep += 1
            total += 1

    return keep, total, known_bug_skips


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
    print2(f"Found {len(files)} files")

    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, str(cxxCompiler))
    # Suppress capability table unless -v 2 or higher (keep default output short)
    if args.Verbose < 2:
        setVerbosity(0)
    # Only set PrintSolutionRejectionReason when verbose to avoid "unrecognised" warning in quiet mode
    gp_config = {"PrintSolutionRejectionReason": True} if args.Verbose >= 2 else {}
    assignGlobalParameters(gp_config, isaInfoMap)
    if args.Verbose < 2:
        setVerbosity(args.Verbose)

    return jobs, isaInfoMap, logicPath, files, check, args


def _progress_loop(stop_event: threading.Event, interval: float = 5.0) -> None:
    """Print a periodic 'Validating... Ns elapsed' line so the user knows the run is active."""
    start = time.time()
    while not stop_event.wait(timeout=interval):
        elapsed = int(time.time() - start)
        sys.stdout.write(f"\rValidating library logic... {elapsed}s elapsed    ")
        sys.stdout.flush()
    # Clear the progress line so the following Total/Keep/Reject output is clean
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()


def main():
    # Suppress noisy joblib warnings (serial fallback, timeout) before any imports that pull in joblib
    warnings.filterwarnings("ignore", message=".*will operate in serial mode.*")
    warnings.filterwarnings("ignore", message=".*timeout.*will not be used.*")

    reset_reported_failures()
    jobs, isaInfoMap, logicPath, files, check, args = _setup()

    try:
        known_bugs = load_known_bugs(args.KnownBugs)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)

    # Use more, smaller batches for better load balancing (workers stay busy as tasks complete)
    num_batches_target = min(len(files), jobs * 8)
    batchSize = max(1, len(files) // num_batches_target)
    batches = list(
        files[i : i + batchSize] for i in range(0, len(files), batchSize)
    )

    fn = functools.partial(_runChecks, logicPath, isaInfoMap, check, known_bugs)
    keep, total = 0, 0
    known_bug_skips = 0

    # Show periodic progress when in quiet mode (no per-file output)
    progress_stop = threading.Event()
    progress_thread = None
    if args.Verbose < 2:
        progress_thread = threading.Thread(
            target=_progress_loop, args=(progress_stop,), daemon=True
        )
        progress_thread.start()

    try:
        results = ParallelMap2(fn, batches, multiArg=False, procs=jobs, return_as="list")

        for _keep, _total, _kb in results:
            keep += _keep
            total += _total
            known_bug_skips += _kb
    finally:
        progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1.5)

    rejects = total - keep
    print(f"Total  {total} solutions")
    print(f"Keep   {keep} solutions")
    print(f"Reject {rejects} solutions")
    if known_bug_skips > 0:
        print(f"Known-bugs skip  {known_bug_skips} solutions (see --known-bugs YAML)")

    if rejects > 0:
        exit(1)
