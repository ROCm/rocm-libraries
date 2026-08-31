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


import contextlib
import functools
import io
import sys
import threading
import time
import warnings

import rocisa

from pathlib import Path
from typing import FrozenSet, List, Dict, NamedTuple, Tuple

from Tensile.Common import ParallelMap2, print1, print2, IsaVersion, IsaInfo, setVerbosity
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, defaultSolution
from Tensile.CustomYamlLoader import load_logic_gfx_arch, archMatch
from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain

from .ParseArguments import parseArguments
from .KnownBugs import (
    KnownBugKey,
    is_known_bug,
    load_known_bugs,
    normalize_logic_relative_path,
)
from .ValidChipId import _validateChipId
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
    rocIsaData=None,
) -> Tuple[int, int, int, int, int]:
    """
    Run checks on the given logic files.

    Args:
        logicPath: Path to a directory containing logic files or to an individual logic file.
        isaInfoMap: Map of IsaVersion to IsaInfo.
        check: Object containing flags for checking.
        files: List of logic files to check.
        rocIsaData: Capabilities snapshot from the parent's rocIsa singleton, or
            None to leave this process's singleton as it is.

    Returns:
        Tuple of (keep, total, known_bug_skips, chip_id_failures, stale_known_bugs)
        where keep is the number of unrejected solutions, total is the total
        number of solutions parsed, known_bug_skips counts solutions accepted via
        the known-bugs list that still fail validation, chip_id_failures is the
        number of files that failed chip-ID validation (independent of
        per-solution accounting), and stale_known_bugs counts known-bugs entries
        whose solution now passes validation (a landed fix; the entry can be
        removed).
    """
    # ParallelMap2 hands a worker globalParameters and nothing else, so the rocIsa
    # singleton in this process has never been init'd and every capability lookup
    # would read a default. Validators that ask the assembly backend what a solution
    # emits need the parent's capabilities to answer the same way the emitter will.
    if rocIsaData is not None:
        rocisa.rocIsa.getInstance().setData(rocIsaData)

    keep, total, known_bug_skips, chip_id_failures = 0, 0, 0, 0
    stale_known_bugs = 0
    for file in files:
        if "Experimental" in file.parts:
            continue

        rel = file.relative_to(logicPath)

        # --- File level validation ---
        # Chip-ID validation is a property of the logic file's location and
        # YAML header, not of any individual solution.
        chip_id_path = file if rel == Path(".") else rel
        chip_id_valid = _validateChipId(
            file,
            logic_relative_path=chip_id_path,
            report_path=chip_id_path,
        )
        if not chip_id_valid:
            chip_id_failures += 1
            # The whole file is invalid; skip per-solution validators and let
            # the file-level failure stand.
            continue

        # --- Solution level validation ---
        solutions = []
        data = readYAML(file)
        if isinstance(data, dict):
            problemType = data.get("ProblemType")
            all_solutions = data.get("Solutions") or []

            # Per-file defaults: fill missing solution keys from this first, then defaultSolution.            
            for solution in all_solutions:
                for key, val in data.get("DefaultSolution", {}).items():
                    if key not in solution:
                        solution[key] = val
                for key, val in defaultSolution.items():
                    if key not in solution:
                        solution[key] = val
            
        else:
            problemType = data[4]
            all_solutions = data[5]
        if check.OnlyCustomKernels and hasCustomKernel(file):
            print2(f">> {rel}")
            solutions = all_solutions
        elif check.All:
            print2(f">> {rel}")
            solutions = all_solutions

        for list_idx, s in enumerate(solutions):
            s, isCustom = handleCustomKernel(s, isaInfoMap)
            if check.OnlyCustomKernels and not isCustom:
                continue

            s["ProblemType"] = problemType
            sol_name = s.get("SolutionNameMin")

            if known_bugs and is_known_bug(known_bugs, rel, sol_name):
                # Re-validate documented known bugs so a landed fix is detected
                # rather than silently skipped forever. A still-failing known bug
                # is expected, so suppress the validators' own error output; only
                # a now-passing solution is worth surfacing. XCC gets report=False
                # so this expected re-failure does not bump its per-file dedup
                # counter and swallow a later *real* XCC failure's message.
                with contextlib.redirect_stdout(io.StringIO()):
                    now_valid = all(
                        [
                            _validateMatrixInstruction(s, isaInfoMap, rel),
                            _validateWorkGroup(s, rel),
                            _validateWorkGroupMappingXCC(s, rel, report=False),
                        ]
                    )
                keep += 1
                total += 1
                if now_valid:
                    stale_known_bugs += 1
                    print(
                        "WARNING: known-bugs entry no longer fails validation: "
                        f"{normalize_logic_relative_path(rel)} :: {sol_name} "
                        "-- fix confirmed; remove this entry from the known-bugs YAML."
                    )
                else:
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

    return keep, total, known_bug_skips, chip_id_failures, stale_known_bugs


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

    archs = args.Architecture.split(";")
    if "all" not in archs:
        files = [f for f in files if archMatch(load_logic_gfx_arch(f), archs)]
        if len(files) == 0:
            print1(f"No files found in {logicPath} for architectures: {', '.join(archs)}")
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

    fn = functools.partial(
        _runChecks, logicPath, isaInfoMap, check, known_bugs,
        rocIsaData=rocisa.rocIsa.getInstance().getData(),
    )
    keep, total = 0, 0
    known_bug_skips = 0
    chip_id_failures = 0
    stale_known_bugs = 0

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

        for _keep, _total, _kb, _cid, _stale in results:
            keep += _keep
            total += _total
            known_bug_skips += _kb
            chip_id_failures += _cid
            stale_known_bugs += _stale
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
    if chip_id_failures > 0:
        print(f"Chip-ID failures  {chip_id_failures} files")
    if stale_known_bugs > 0:
        print(
            f"Stale known-bugs  {stale_known_bugs} entries now pass validation "
            "(remove them from the known-bugs YAML)"
        )

    strict_stale = getattr(args, "StrictKnownBugs", False) and stale_known_bugs > 0
    if rejects > 0 or chip_id_failures > 0 or strict_stale:
        exit(1)
