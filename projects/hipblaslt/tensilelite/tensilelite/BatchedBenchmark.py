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

"""Batched benchmark execution with per-solution-batch subprocess isolation.

Wraps the tensilelite-client invocation so that solutions are benchmarked in
batches of configurable size.  Each batch runs in a separate subprocess with a
hard timeout; a batch that hangs or crashes is killed and its solutions are
either retried individually or recorded as failures.

Failed kernels are captured in a Failed_Kernels/ directory alongside the
benchmark data for reproduction and debugging of GPU hangs or
illegal-instruction faults.

Usage from ClientWriter.runClient:
    if globalParameters["SolutionsPerProcess"] != -1:
        return runClientBatched(configPath, resultsFile, ...)
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .Common import print1, printWarning
from .Common.GlobalParameters import globalParameters

import math

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Outcome of a single subprocess batch."""
    startIdx: int
    count: int
    returncode: Optional[int] = None
    timedOut: bool = False
    resultsFile: Optional[str] = None
    elapsed_s: float = 0.0


@dataclass
class FailedKernel:
    """Record of a kernel that caused a hang or crash."""
    solutionIdx: int
    batchStartIdx: int
    batchCount: int
    errorType: str  # "timeout", "crash", "nonzero_exit"
    returncode: Optional[int] = None
    problemSize: str = ""
    configFile: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _get_total_solutions(configPath: str | Path) -> int:
    """Parse the ClientParameters.ini to estimate the number of solutions.

    The C++ client discovers solutions from the library file; we cannot
    cheaply determine the exact count from the INI alone.  Instead, callers
    should pass totalSolutions explicitly when it is known (e.g. from
    len(solutions) in BenchmarkProblems).  This function returns -1 to
    signal "unknown" so the caller can fall back to single-batch mode.
    """
    return -1


def _estimate_batch_timeout(
    numSolutions: int,
    *,
    perSolutionOverhead_s: float = 0.2,
    safetyMultiplier: float = 3.0,
) -> float:
    """Estimate a reasonable timeout for a batch of *numSolutions* solutions.

    The heuristic is::

        timeout = safetyMultiplier * numSolutions * perSolutionTime

    where perSolutionTime accounts for warmups, enqueues, compile overhead,
    and a generous safety margin.  The per-solution time is derived from the
    global parameters::

        kernel_launches = NumWarmups + (EnqueuesPerSync * SyncsPerBenchmark * NumBenchmarks)
        perSolutionTime = perSolutionOverhead_s * kernel_launches

    The safety multiplier (default 3×) covers adaptive timing retries, GPU
    frequency throttling, slow solution compilation, and other variance.

    Args:
        numSolutions: Number of solutions in this batch.
        perSolutionOverhead_s: Estimated wall-clock time per kernel launch
            (seconds).  0.2 s is conservative for large GEMMs.
        safetyMultiplier: Multiplier on top of the estimated time.

    Returns:
        Estimated timeout in seconds (always ≥ 30).
    """
    numWarmups = max(globalParameters.get("NumWarmups", 0), 1)
    enqPerSync = max(globalParameters.get("EnqueuesPerSync", 1), 1)
    syncsPerBm = max(globalParameters.get("SyncsPerBenchmark", 1), 1)
    numBm = max(globalParameters.get("NumBenchmarks", 1), 1)

    kernelLaunches = numWarmups + (enqPerSync * syncsPerBm * numBm)
    perSolution_s = perSolutionOverhead_s * kernelLaunches
    raw = numSolutions * perSolution_s * safetyMultiplier
    timeout = max(math.ceil(raw), 30)  # minimum 30 s
    return float(timeout)


def _build_client_cmd(
    clientExe: str,
    configPath: str,
    *,
    solutionStartIdx: int = -1,
    numSolutions: int = -1,
    resultsFile: Optional[str] = None,
) -> List[str]:
    """Assemble the tensilelite-client command line for one batch."""
    cmd = [clientExe, "--config-file", str(configPath)]

    if solutionStartIdx >= 0:
        cmd.extend(["--solution-start-idx", str(solutionStartIdx)])
    if numSolutions >= 0:
        cmd.extend(["--num-solutions", str(numSolutions)])
    if resultsFile:
        cmd.extend(["--results-file", str(resultsFile)])

    if globalParameters.get("TimingInstrumentation", False):
        cmd.append("--timing-instrumentation")

    mxFmt = globalParameters.get("MXScaleFormat")
    if mxFmt:
        cmd.extend(["--mx-scale-format", str(mxFmt)])

    return cmd


def _run_batch(
    clientExe: str,
    configPath: str,
    startIdx: int,
    count: int,
    resultsFile: str,
    timeout_s: float,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> BatchResult:
    """Launch one batch of solutions in a subprocess with a hard timeout."""
    cmd = _build_client_cmd(
        clientExe,
        configPath,
        solutionStartIdx=startIdx,
        numSolutions=count,
        resultsFile=resultsFile,
    )

    result = BatchResult(startIdx=startIdx, count=count, resultsFile=resultsFile)
    t0 = time.monotonic()

    timeoutStr = f"{timeout_s:.0f}s" if timeout_s < 86000 else "disabled"
    print1(f"# BatchedBenchmark: launching solutions [{startIdx}..{startIdx + count - 1}] "
           f"(timeout={timeoutStr})")

    try:
        proc = subprocess.Popen(
            cmd,
            env=env or os.environ.copy(),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so we can kill the process group
        )
        try:
            proc.wait(timeout=timeout_s)
            result.returncode = proc.returncode
        except subprocess.TimeoutExpired:
            result.timedOut = True
            # Kill the entire process group to clean up any child processes
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except OSError:
                proc.kill()
            proc.wait()
            result.returncode = proc.returncode
            printWarning(f"BatchedBenchmark: batch [{startIdx}..{startIdx + count - 1}] "
                         f"timed out after {timeout_s}s — killed")
    except OSError as e:
        printWarning(f"BatchedBenchmark: failed to launch client: {e}")
        result.returncode = -1

    result.elapsed_s = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Failed kernel tracking
# ---------------------------------------------------------------------------

def _extract_problem_sizes(solutionsYamlPath: str | Path) -> str:
    """Extract the ProblemSizes line from the 00_Final.yaml header.

    Returns a string like 'M=2880,N=14745,B=1,K=65536' or '' if not found.
    The Exact format is [M, N, B, K, lda, ldb, ldc, ldd].
    """
    try:
        with open(solutionsYamlPath, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("- Exact:"):
                    # Parse [M, N, B, K, ...]
                    bracket = stripped.split("[", 1)
                    if len(bracket) == 2:
                        nums = bracket[1].rstrip("]").split(",")
                        if len(nums) >= 4:
                            m, n, b, k = [s.strip() for s in nums[:4]]
                            return f"M={m},N={n},B={b},K={k}"
                # Stop after the header (first solution block starts)
                if stripped.startswith("1LDSBuffer:"):
                    break
    except OSError:
        pass
    return ""


def _extract_solution_yaml(solutionsYamlPath: str | Path, solutionIdx: int) -> Optional[str]:
    """Extract one solution's YAML block from the 00_Final.yaml file.

    Solutions are separated by lines starting with '- 1LDSBuffer:' (after the
    header entries).  Each solution contains 'SolutionIndex: N'.  Returns the
    raw YAML text for the matching solution, or None if not found.
    """
    try:
        with open(solutionsYamlPath, "r") as f:
            lines = f.readlines()
    except OSError:
        return None

    # Find the start of each solution block
    solStarts = [i for i, line in enumerate(lines) if line.startswith("- 1LDSBuffer:")]

    for idx, start in enumerate(solStarts):
        end = solStarts[idx + 1] if idx + 1 < len(solStarts) else len(lines)
        block = "".join(lines[start:end])
        if f"SolutionIndex: {solutionIdx}" in block:
            return block

    return None


def _find_solutions_yaml(
    benchmarkProblemsDir: Optional[str | Path] = None,
    configPath: Optional[str | Path] = None,
) -> Optional[Path]:
    """Locate the 00_Final.yaml (solutions YAML with full kernel configs).

    Searches for Data/00_Final.yaml under benchmarkProblemsDir first, then
    walks up from configPath as a fallback.
    """
    # Primary: look under benchmarkProblemsDir/Data/
    if benchmarkProblemsDir:
        yamlPath = Path(benchmarkProblemsDir) / "Data" / "00_Final.yaml"
        if yamlPath.is_file():
            return yamlPath

    # Fallback: walk up from configPath
    # Config INI is at .../caches/<hash>/source/ClientParameters.ini
    # Data dir is at .../<stepName>/Data/
    if configPath:
        configPath = Path(configPath)
        for depth in range(2, 6):
            parent = configPath
            for _ in range(depth):
                parent = parent.parent
            yamlPath = parent / "Data" / "00_Final.yaml"
            if yamlPath.is_file():
                return yamlPath

    return None


def _record_failed_kernels(
    failures: List[FailedKernel],
    failedKernelsDir: Path,
    benchmarkProblemsDir: Optional[str | Path] = None,
    configPath: Optional[str | Path] = None,
) -> None:
    """Write failed kernel configs to the Failed_Kernels directory.

    For each failed solution, extracts the full kernel YAML config from
    00_Final.yaml and saves it as a standalone file.  Also writes a
    summary.txt for quick reference.  Accumulates across tuning rounds.
    """
    failedKernelsDir.mkdir(parents=True, exist_ok=True)

    # Find the solutions YAML to extract kernel configs from
    solutionsYaml = _find_solutions_yaml(
        benchmarkProblemsDir=benchmarkProblemsDir, configPath=configPath
    )

    # Extract problem size from the YAML header
    problemSize = ""
    if solutionsYaml:
        problemSize = _extract_problem_sizes(solutionsYaml)

    summary_path = failedKernelsDir / "summary.txt"

    for fk in failures:
        # Populate the problem size on the failure record
        if not fk.problemSize and problemSize:
            fk.problemSize = problemSize

        # Extract and save the kernel YAML config
        kernelYaml = None
        if solutionsYaml:
            kernelYaml = _extract_solution_yaml(solutionsYaml, fk.solutionIdx)

        if kernelYaml:
            kernelFile = failedKernelsDir / f"solution_{fk.solutionIdx}.yaml"
            with open(kernelFile, "w") as f:
                f.write(f"# Failed kernel — solution {fk.solutionIdx}\n")
                f.write(f"# Problem size: {fk.problemSize}\n")
                f.write(f"# Error: {fk.errorType} (rc={fk.returncode})\n")
                f.write(f"# Timestamp: {fk.timestamp}\n")
                f.write(f"# Config: {fk.configFile}\n")
                f.write(f"# Source: {solutionsYaml}\n\n")
                f.write(kernelYaml)

        # Append to summary
        with open(summary_path, "a") as f:
            saved = f"saved=solution_{fk.solutionIdx}.yaml" if kernelYaml else "yaml=not_found"
            f.write(f"[{fk.timestamp}] solution={fk.solutionIdx} "
                    f"size=({fk.problemSize}) "
                    f"error={fk.errorType} rc={fk.returncode} "
                    f"batch=[{fk.batchStartIdx}..{fk.batchStartIdx + fk.batchCount - 1}] "
                    f"{saved}\n")

    print1(f"# BatchedBenchmark: recorded {len(failures)} failed kernel(s) "
           f"in {failedKernelsDir}")


# ---------------------------------------------------------------------------
# Results merging
# ---------------------------------------------------------------------------

def _merge_results_csvs(partialFiles: Sequence[str], finalFile: str) -> None:
    """Merge partial per-batch CSVs into the final results file.

    Each partial CSV has the same header.  Rows are concatenated in order.
    If a partial file is missing or empty (timed-out batch), it is skipped.
    """
    header_written = False
    with open(finalFile, "w", newline="") as out:
        writer = None
        for pf in partialFiles:
            if not os.path.isfile(pf) or os.path.getsize(pf) == 0:
                continue
            with open(pf, "r", newline="") as inp:
                reader = csv.reader(inp)
                for i, row in enumerate(reader):
                    if i == 0:
                        if not header_written:
                            if writer is None:
                                writer = csv.writer(out)
                            writer.writerow(row)
                            header_written = True
                        continue
                    if writer is not None:
                        writer.writerow(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def runClientBatched(
    configPath: str | Path,
    resultsFile: str | Path,
    *,
    totalSolutions: int,
    batchSize: int = -1,
    timeout_s: Optional[float] = None,
    retryTimeout_s: Optional[float] = None,
    benchmarkProblemsDir: Optional[str | Path] = None,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> int:
    """Run the tensilelite-client with subprocess-level batch isolation.

    Args:
        configPath: Path to the ClientParameters.ini file.
        resultsFile: Path where the merged final results CSV should be written.
        totalSolutions: Total number of solutions in the benchmark.  Required.
        batchSize: Number of solutions per subprocess.
            -1 = all in one process (legacy behavior with timeout).
             1 = one solution per process (maximum isolation).
             N = N solutions per process.
        timeout_s: Seconds to wait for each batch before killing.
            None or 0 = auto-estimate from kernel launch counts and batch size.
            -1 = no timeout (wait forever, same as legacy).
            >0 = use this explicit value.
        retryTimeout_s: Timeout for individual-solution retries of failed batches.
            None or 0 = auto-estimate for a single solution.
            -1 = no timeout.
            >0 = use this explicit value.
        benchmarkProblemsDir: Path to 1_BenchmarkProblems/<ProblemType>/
            for writing Failed_Kernels/.  If None, failed kernels are not
            tracked on disk.
        env: Environment dict for subprocesses.
        cwd: Working directory for subprocesses.

    Returns:
        0 if at least some solutions succeeded, nonzero if all failed.
    """
    configPath = str(configPath)
    resultsFile = str(resultsFile)
    clientExe = globalParameters.get("PrebuiltClient", "")

    if not clientExe or not os.path.isfile(clientExe):
        from .ClientWriter import getClientExecutablePath
        clientExe = getClientExecutablePath()

    # Determine batches
    if batchSize == -1 or totalSolutions <= 0:
        batches = [(0, totalSolutions if totalSolutions > 0 else -1)]
    else:
        batches = []
        for start in range(0, totalSolutions, batchSize):
            count = min(batchSize, totalSolutions - start)
            batches.append((start, count))

    # Resolve timeouts
    # SolutionBatchTimeout from GlobalParameters: 0=auto, -1=no timeout, >0=explicit
    configTimeout = globalParameters.get("SolutionBatchTimeout", 0)

    if timeout_s is None:
        timeout_s = configTimeout
    if retryTimeout_s is None:
        retryTimeout_s = configTimeout

    # Compute effective timeouts per batch
    noTimeout = False
    if timeout_s == -1:
        noTimeout = True
        # Use a very large value so subprocess.wait() effectively never times out
        timeout_s = 86400.0  # 24 hours
    elif timeout_s == 0 or timeout_s is None:
        # Auto-estimate based on batch size and kernel launch counts
        maxBatchCount = max(count for _, count in batches) if batches else 1
        timeout_s = _estimate_batch_timeout(maxBatchCount)

    if retryTimeout_s == -1:
        retryTimeout_s = 86400.0
    elif retryTimeout_s == 0 or retryTimeout_s is None:
        retryTimeout_s = _estimate_batch_timeout(1)

    if noTimeout:
        print1(f"# BatchedBenchmark: {totalSolutions} solutions in {len(batches)} batch(es) "
               f"(batchSize={batchSize}, timeout=disabled)")
    else:
        print1(f"# BatchedBenchmark: {totalSolutions} solutions in {len(batches)} batch(es) "
               f"(batchSize={batchSize}, timeout={timeout_s:.0f}s per batch, "
               f"retry timeout={retryTimeout_s:.0f}s per solution)")

    # Run batches
    partialFiles: List[str] = []
    failedBatches: List[BatchResult] = []
    allFailedKernels: List[FailedKernel] = []
    resultsDir = Path(resultsFile).parent
    resultsDir.mkdir(parents=True, exist_ok=True)

    for batchIdx, (startIdx, count) in enumerate(batches):
        partialFile = str(resultsDir / f"_batch_{batchIdx:04d}_results.csv")
        partialFiles.append(partialFile)

        result = _run_batch(
            clientExe, configPath, startIdx, count,
            partialFile, timeout_s, env=env, cwd=cwd,
        )

        if result.timedOut or (result.returncode is not None and result.returncode != 0):
            failedBatches.append(result)
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            if result.timedOut:
                errorType = "timeout"
            elif result.returncode and (result.returncode & 0x80):
                errorType = f"signal_{result.returncode & 0x7F}"
            else:
                errorType = "nonzero_exit"

            # If batch has more than 1 solution, retry individually
            if count > 1 and batchSize != -1:
                print1(f"# BatchedBenchmark: retrying batch {batchIdx} "
                       f"[{startIdx}..{startIdx + count - 1}] individually")
                for solIdx in range(startIdx, startIdx + count):
                    retryFile = str(resultsDir / f"_retry_{solIdx:06d}_results.csv")
                    retryResult = _run_batch(
                        clientExe, configPath, solIdx, 1,
                        retryFile, retryTimeout_s, env=env, cwd=cwd,
                    )
                    if retryResult.timedOut or (retryResult.returncode and retryResult.returncode != 0):
                        retryError = "timeout" if retryResult.timedOut else errorType
                        allFailedKernels.append(FailedKernel(
                            solutionIdx=solIdx,
                            batchStartIdx=solIdx,
                            batchCount=1,
                            errorType=retryError,
                            returncode=retryResult.returncode,
                            configFile=configPath,
                            timestamp=now,
                        ))
                        printWarning(f"BatchedBenchmark: solution {solIdx} failed "
                                     f"({retryError}, rc={retryResult.returncode})")
                    else:
                        # This solution succeeded on retry — add its results
                        partialFiles.append(retryFile)
                        print1(f"# BatchedBenchmark: solution {solIdx} succeeded on retry "
                               f"({retryResult.elapsed_s:.1f}s)")
            else:
                # Single-solution batch or batchSize=-1: record all solutions as failed
                for solIdx in range(startIdx, startIdx + max(count, 0)):
                    allFailedKernels.append(FailedKernel(
                        solutionIdx=solIdx,
                        batchStartIdx=startIdx,
                        batchCount=count,
                        errorType=errorType,
                        returncode=result.returncode,
                        configFile=configPath,
                        timestamp=now,
                    ))
        else:
            print1(f"# BatchedBenchmark: batch {batchIdx} "
                   f"[{startIdx}..{startIdx + count - 1}] completed "
                   f"({result.elapsed_s:.1f}s)")

    # Merge results
    _merge_results_csvs(partialFiles, resultsFile)

    # Record failed kernels with their full YAML configs
    if allFailedKernels and benchmarkProblemsDir:
        failedKernelsDir = Path(benchmarkProblemsDir) / "Failed_Kernels"
        _record_failed_kernels(
            allFailedKernels, failedKernelsDir,
            benchmarkProblemsDir=benchmarkProblemsDir, configPath=configPath,
        )

    # Cleanup partial files
    for pf in partialFiles:
        try:
            if os.path.isfile(pf):
                os.remove(pf)
        except OSError:
            pass

    # Return 0 if we got any results, nonzero if all failed
    if os.path.isfile(resultsFile) and os.path.getsize(resultsFile) > 0:
        nFailed = len(allFailedKernels)
        nTotal = totalSolutions if totalSolutions > 0 else len(batches)
        print1(f"# BatchedBenchmark: completed — {nTotal - nFailed}/{nTotal} solutions "
               f"succeeded, {nFailed} failed")
        return 0
    else:
        printWarning("BatchedBenchmark: all batches failed — no results produced")
        return 1
