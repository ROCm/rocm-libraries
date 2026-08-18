# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
TensileBackend: Exhaustive fork parameter enumeration strategy.

This backend generates all possible fork parameter permutations upfront and
benchmarks all of them. It's the original Tensile strategy for solution generation.
"""

from typing import List, Dict, Any, Callable, Tuple
from Tensile.BenchmarkStructs import constructForkPermutations

from Tensile.Common.TimingInstrumentation import timing_context
from Tensile.Common import print1

from .base import OptimizationBackend


class TensileBackend(OptimizationBackend):
    """Backend for exhaustive fork parameter enumeration.
    
    Generates all possible permutations of fork parameters upfront, then
    benchmarks all of them before returning results.
    """

    def __init__(self):
        """Initialize the TensileBackend."""
        self._fork_permutations: List[Dict[str, Any]] = []

    def run(self, 
            backend_config: Dict[str, Any],
            benchmark_config: Dict[str, Any],
            benchmark_runner: Callable[[List[Any]], Tuple[str, int]],
            cacheValid: bool = False,
            buildOnly: bool = False) -> None:
        """Execute exhaustive enumeration loop with solution generation.
        
        Generates fork parameter permutations and custom kernels, converts to solutions,
        and benchmarks all of them.
        
        Args:
            backend_config: Not used by this backend, but included for interface consistency.
            benchmark_config: Backend step configuration containing:
                - forkParams, constantParams, paramGroups
                - customKernels, internalSupportParams, customKernelWildcard  
                - ForkParameters flag
                - problemType, assembler, debugConfig, isaInfoMap
            benchmark_runner: Function returning (resultsFileName, returncode)
            cacheValid: If True, use cached solutions if available
            buildOnly: If True, skip benchmarking
            
        Returns:
            None
        """
        # Validate required config
        required_keys = [
            "forkParametersEnabled", 
            "problemType", 
            "assembler", 
            "debugConfig", 
            "isaInfoMap", 
            "benchmarkStep", 
            "solutionPoolIndex"
        ]
        for key in required_keys:
            if key not in benchmark_config:
                raise ValueError(f"BenchmarkProblems: Missing required backend config key: {key}")

        # Extract configuration
        forkParametersEnabled = benchmark_config["forkParametersEnabled"]
        problemType = benchmark_config["problemType"]
        assembler = benchmark_config["assembler"]
        debugConfig = benchmark_config["debugConfig"]
        isaInfoMap = benchmark_config["isaInfoMap"]
        benchmarkStep = benchmark_config["benchmarkStep"]
        solutionPoolIndex = benchmark_config["solutionPoolIndex"] or {}

        # Import locals to avoid circular dependency
        from Tensile.BenchmarkProblems import (_generateForkedSolutions, _generateCustomKernelSolutions,
                                               _constructAllPoolSolutions)
    
        configPTStr = str(problemType)
        useSolutionPool = configPTStr in solutionPoolIndex

        solutions = None
        if not cacheValid or useSolutionPool:
            if useSolutionPool:
                poolEntries = solutionPoolIndex[configPTStr]
                with timing_context("python_solution_pool_construction"):
                    solutions = _constructAllPoolSolutions(poolEntries, assembler, debugConfig, isaInfoMap)
                print1("# Total {} solutions from {} pool file(s)".format(len(solutions), len(poolEntries)))
                maxPossibleSolutions = len(solutions)
            else:
                # enumerate benchmark permutations and create resulting solution objects
                with timing_context("python_solution_generation"):
                    with timing_context("python_solgen_fork_permutations"):
                        forkPermutations = constructForkPermutations(benchmarkStep.forkParams, \
                                benchmarkStep.paramGroups) if forkParametersEnabled else []
                        maxPossibleSolutions = len(forkPermutations)

                    with timing_context("python_solgen_forked_solutions"):
                        regSolutions = _generateForkedSolutions(problemType, \
                                benchmarkStep.constantParams, forkPermutations, assembler, \
                                    debugConfig, isaInfoMap)

                    with timing_context("python_solgen_custom_kernels"):
                        kcSolutions = _generateCustomKernelSolutions(problemType, \
                                benchmarkStep.customKernels, benchmarkStep.internalSupportParams, \
                                not benchmarkStep.customKernelWildcard, assembler, debugConfig, \
                                    isaInfoMap)

                    maxPossibleSolutions += len(kcSolutions)
                    solutions = regSolutions + kcSolutions

            print1("# Actual Solutions: {} / {} after SolutionStructs\n" \
                .format(len(solutions), maxPossibleSolutions))

        # Benchmark all solutions
        benchmark_runner(solutions, isCached=cacheValid and not useSolutionPool, buildOnly=buildOnly)

    def supports_solution_pool(self) -> bool:
        """TensileBackend supports loading solutions from pool files.
        
        Returns:
            True - TensileBackend can use solution pool files
        """
        return True
