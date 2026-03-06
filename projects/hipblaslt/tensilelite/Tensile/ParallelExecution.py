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

"""
Parallel GPU execution utilities for Tensile client.

This module provides functions to distribute benchmark workloads across
multiple GPUs for faster execution.

Default Execution (Single GPU)
==============================
Without parallel execution, the Tensile client processes all benchmark problems
sequentially on a single GPU (device 0). For a config file with N problems,
each problem is executed one after another:

    Config File (N problems)
            |
            v
    +-------------------+
    |  GPU 0 (device 0) |
    |  Problem 1        |
    |  Problem 2        |
    |  ...              |
    |  Problem N        |
    +-------------------+
            |
            v
    results.csv (all results)

This can be slow when benchmarking many problems, as only one GPU is utilized.

Parallel Execution (Multiple GPUs)
==================================
With parallel execution enabled, problems are distributed across available GPUs.
Each GPU processes a subset of the problems concurrently:

    Config File (N problems)
            |
            v
    +------------------+
    | Split workload   |
    | N/G problems/GPU |
    +------------------+
            |
    +-------+-------+-------+
    |       |       |       |
    v       v       v       v
  +-----+ +-----+ +-----+ +-----+
  |GPU 0| |GPU 1| |GPU 2| |GPU 3|   (G GPUs)
  |P1-Pk| |Pk+1 | |...  | |...PN|
  +-----+ +-----+ +-----+ +-----+
    |       |       |       |
    v       v       v       v
  gpu0_   gpu1_   gpu2_   gpu3_
  results results results results
    |       |       |       |
    +-------+-------+-------+
            |
            v
    +------------------+
    | Merge CSV files  |
    +------------------+
            |
            v
    results.csv (all results, same format as single-GPU)

Process Flow:
1. Detect available GPUs via rocm-smi or hipInfo
2. Count problems in the config file
3. Create per-GPU config files with:
   - Assigned device-idx (GPU index)
   - Separate results-file path
   - problem-start-idx and num-problems to define the subset
4. Launch all GPU clients in parallel using subprocess
5. Wait for all processes to complete
6. Merge per-GPU CSV results into the original output file

The merged results file has the same format as single-GPU execution, making
the parallelization transparent to downstream processing.

Key Functions:
- detectAvailableGpus(): Find number of GPUs in the system
- countProblemsInConfig(): Count problem-size entries in config
- createPerGpuConfig(): Generate per-GPU config with problem subset
- mergeResultsCsv(): Combine CSV files preserving header
- runClientParallel(): Main orchestration function
"""

import os
import re
import shutil
import subprocess
import time

from Tensile.Common import print1, printWarning, ClientExecutionLock
from Tensile.Common.GlobalParameters import globalParameters

# Wall-clock deadline in seconds from launch; all GPU processes must finish within this window
GPU_WALL_TIMEOUT_SECS = 60


def detectAvailableGpus():
    """Detect the number of available GPUs using rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Count unique GPU indices (GPU[N] appears multiple times per device)
            gpu_indices = set(re.findall(r'GPU\[(\d+)\]', result.stdout))
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass

    # Fallback: try to parse from HIP
    try:
        result = subprocess.run(
            ["hipInfo"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            match = re.search(r'Number of devices:\s*(\d+)', result.stdout)
            if match:
                return int(match.group(1))
    except Exception:
        pass

    return 1  # Default to single GPU


def countProblemsInConfig(configPath):
    """Count the number of problem-size entries in a config file."""
    count = 0
    with open(configPath, 'r') as f:
        for line in f:
            if line.strip().startswith('problem-size='):
                count += 1
    return count


def createPerGpuConfig(originalConfigPath, gpuIdx, problemStartIdx, numProblems, tempDir):
    """Create a modified config file for a specific GPU."""
    baseName = os.path.basename(originalConfigPath)
    newConfigPath = os.path.join(tempDir, f"gpu{gpuIdx}_{baseName}")
    newResultsPath = os.path.join(tempDir, f"gpu{gpuIdx}_results.csv")

    with open(originalConfigPath, 'r') as inFile:
        lines = inFile.readlines()

    with open(newConfigPath, 'w') as outFile:
        for line in lines:
            # Replace device-idx
            if line.strip().startswith('device-idx='):
                outFile.write(f"device-idx={gpuIdx}\n")
            # Replace results-file
            elif line.strip().startswith('results-file='):
                outFile.write(f"results-file={newResultsPath}\n")
            else:
                outFile.write(line)

        # Add problem range parameters at the end
        outFile.write(f"problem-start-idx={problemStartIdx}\n")
        outFile.write(f"num-problems={numProblems}\n")

    return newConfigPath, newResultsPath


def mergeResultsCsv(resultsPaths, outputPath):
    """Merge multiple CSV result files into one."""
    headerWritten = False

    with open(outputPath, 'w') as outFile:
        for resultsPath in resultsPaths:
            if not os.path.exists(resultsPath):
                continue
            with open(resultsPath, 'r') as inFile:
                lines = inFile.readlines()
                if not lines:
                    continue

                if not headerWritten:
                    outFile.writelines(lines)
                    headerWritten = True
                else:
                    outFile.writelines(lines[1:])


def runClientParallel(buildPath, configPaths, numGpus, timingEnabled, getClientExecutablePath):
    """Run the client in parallel across multiple GPUs.

    Args:
        buildPath: Path to the build directory
        configPaths: List of config file paths to process
        numGpus: Number of GPUs to use
        timingEnabled: Whether timing instrumentation is enabled
        getClientExecutablePath: Function to get the client executable path

    Returns:
        Overall return code (max of all GPU return codes)
    """
    clientExe = getClientExecutablePath()

    # Process each config file
    overallReturnCode = 0

    for configPath in configPaths:
        # Count problems in this config
        totalProblems = countProblemsInConfig(configPath)

        if totalProblems == 0:
            printWarning(f"No problems found in {configPath}, skipping")
            continue

        # Determine how many GPUs to actually use (don't use more than problems)
        effectiveGpus = min(numGpus, totalProblems)
        problemsPerGpu = (totalProblems + effectiveGpus - 1) // effectiveGpus  # Ceiling division

        print1(f"# Parallel execution: {totalProblems} problems across {effectiveGpus} GPUs ({problemsPerGpu} problems/GPU)")

        # Create a persistent directory for per-GPU configs and results.
        # Only cleaned up after a successful merge; preserved on failure for debugging.
        parallelDir = os.path.join(str(buildPath), "parallel_gpu")
        if os.path.exists(parallelDir):
            shutil.rmtree(parallelDir)
        os.makedirs(parallelDir)

        # Create per-GPU config files
        gpuConfigs = []  # List of (configPath, resultsPath, gpuIdx)
        for gpuIdx in range(effectiveGpus):
            problemStartIdx = gpuIdx * problemsPerGpu
            # Last GPU gets remaining problems
            if gpuIdx == effectiveGpus - 1:
                gpuNumProblems = totalProblems - problemStartIdx
            else:
                gpuNumProblems = min(problemsPerGpu, totalProblems - problemStartIdx)

            if gpuNumProblems <= 0:
                continue

            perGpuConfig, perGpuResults = createPerGpuConfig(
                configPath, gpuIdx, problemStartIdx, gpuNumProblems, parallelDir
            )
            gpuConfigs.append((perGpuConfig, perGpuResults, gpuIdx))

        # TODO: Support PinClocks for parallel execution (per-GPU clock pinning/reset).
        #       The single-GPU path pins clocks via writeRunScript; this path skips it.

        # Launch all GPU clients in parallel
        processes = []
        with ClientExecutionLock(globalParameters["ClientExecutionLockPath"]):
            try:
                for perGpuConfig, perGpuResults, gpuIdx in gpuConfigs:
                    args = [clientExe, "--config-file", perGpuConfig]
                    if timingEnabled:
                        args.append("--timing-instrumentation")

                    print1(f"# Launching client on GPU {gpuIdx}")
                    proc = subprocess.Popen(args, cwd=str(buildPath))
                    processes.append((proc, gpuIdx, perGpuResults))

                # Wait for all processes with a shared wall-clock deadline
                launchTime = time.time()
                for proc, gpuIdx, _ in processes:
                    remaining = max(0, GPU_WALL_TIMEOUT_SECS - (time.time() - launchTime))
                    try:
                        proc.wait(timeout=remaining if remaining > 0 else 0.001)
                    except subprocess.TimeoutExpired:
                        print1(f"# Client on GPU {gpuIdx} exceeded {GPU_WALL_TIMEOUT_SECS}s wall-clock deadline, killing...")
                        proc.kill()
                        proc.wait()
                    if proc.returncode and proc.returncode > 0:
                        printWarning(f"Client on GPU {gpuIdx} exited with code {proc.returncode}")
                        overallReturnCode = max(overallReturnCode, proc.returncode)
            finally:
                # Final cleanup - kill anything still running
                for proc, gpuIdx, _ in processes:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()

        # Merge results from all GPUs
        originalResultsPath = None
        with open(configPath, 'r') as f:
            for line in f:
                if line.strip().startswith('results-file='):
                    originalResultsPath = line.strip().split('=', 1)[1]
                    break

        if originalResultsPath:
            resultsPaths = [r for _, r, _ in gpuConfigs]
            mergeResultsCsv(resultsPaths, originalResultsPath)
            print1(f"# Merged results from {len(resultsPaths)} GPUs to {originalResultsPath}")
            # Clean up per-GPU files only after successful merge
            shutil.rmtree(parallelDir)
        else:
            printWarning(f"No results-file found in {configPath}; per-GPU results preserved at {parallelDir}")

    return overallReturnCode
