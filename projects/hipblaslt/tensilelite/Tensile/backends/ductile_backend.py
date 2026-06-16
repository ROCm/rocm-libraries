################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
DuctileBackend: Genetic algorithm-based parameter search strategy.

This backend uses a genetic algorithm to search the fork parameter space,
iteratively evaluating promising candidates and culling poor performers.
It requires the Ductile GA modules.
"""

from typing import List, Dict, Any, Callable, Tuple
import numpy as np
import pandas as pd
import functools
import os
import shutil
from copy import deepcopy

from Tensile.Common import Path, print1, printExit, printWarning
from Tensile.Common.GlobalParameters import globalParameters
from Tensile.SolutionStructs.Naming import getKernelFileBase
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from .base import OptimizationBackend


from Tensile.ductile import config as ductile_config
from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from Tensile.ductile.algorithm import GeneticAlgorithm



def _generate_single_solution_with_groups(perm, problemType, constantParams, assembler, debugConfig, isaInfoMap, silent=False):
    """Generate a single solution from a permutation, handling group_ parameter expansion."""
    from Tensile.BenchmarkProblems import _build_and_validate_solution

    solution = {
        "ProblemType": deepcopy(problemType.state),
        "ISA": next(iter(isaInfoMap.keys()))
    }
    solution.update(constantParams)
    solution.update(perm)

    # Expand group_ parameters
    for p in list(perm.keys()):
        if p.startswith("group_"):
            solution.update(perm[p])
            if p in solution:
                del solution[p]

    return _build_and_validate_solution(solution, assembler, debugConfig, isaInfoMap, silent=silent)


def _validate_solution(problemType, constantParams, assembler, debugConfig, isaInfoMap, perm, get_kernel_src=False):
    """Validate a solution candidate for GA SearchSpace constraint checking."""
    solution_object = _generate_single_solution_with_groups(
        perm, problemType, constantParams, assembler, debugConfig, isaInfoMap, silent=True
    )
    if solution_object is None:
        return False

    kernelWriterAssembly = KernelWriterAssembly(assembler, debugConfig)
    try:
        kernelWriterAssembly._initKernel(solution_object, {}, {})
        if get_kernel_src:
            kernelWriterAssembly._getKernelSource(solution_object)
    except RuntimeError:
        return False

    return True


def _generate_ga_solutions(problemType, constantParams, individuals, assembler, debugConfig, isaInfoMap):
    """Match Ductile/BenchmarkProblems.py generateGASolutions behavior.

    Important: returns a list aligned with individuals, inserting None for duplicates
    or invalid candidates so the GA index mapping remains stable.
    """
    solutions = []
    solutionSet = set()
    baseSet = set()

    for perm in individuals:
        solutionObject = _generate_single_solution_with_groups(
            perm, problemType, constantParams, assembler, debugConfig, isaInfoMap
        )

        if solutionObject is not None:
            base = getKernelFileBase(debugConfig.splitGSU, solutionObject)
            if solutionObject not in solutionSet and base not in baseSet:
                solutionSet.add(solutionObject)
                baseSet.add(base)
                solutions.append(solutionObject)
            else:
                solutions.append(None)
        else:
            solutions.append(None)

    return solutions


class DuctileBackend(OptimizationBackend):
    """Backend for genetic algorithm-based parameter optimization.
    
    Uses Ductile's GA implementation to iteratively search the fork parameter space,
    focusing effort on promising regions and avoiding exhaustive enumeration.
    """

    def run(self, 
            backend_config: Dict[str, Any],
            benchmark_config: Dict[str, Any],
            benchmark_runner: Callable[[List[Any]], Tuple[str, int]],
            cacheValid: bool = False,
            buildOnly: bool = False) -> None:
        """Execute GA optimization loop.
        
        Creates GA instance with _evaluate callback, runs optimization,
        and performs post-optimization verification.
        
        Args:
            backend_config: Configuration dict containing backend-specific settings.
            benchmark_config: Backend step configuration containing:
                - forkParams, constantParams, paramGroups
                - problemType, assembler, debugConfig, isaInfoMap
                - sourcePath
            benchmark_runner: Function returning (resultsFileName, returncode)
            cacheValid: DuctileBackend does not support caching; warns and forces False
            buildOnly: DuctileBackend does not support build-only mode; warns and forces False
            
        Returns:
            None
        """
        source_path = benchmark_config.get("sourcePath", None)

        if cacheValid:
            printWarning("DuctileBackend: cacheValid is not supported; running with cacheValid=False")
        if buildOnly:
            printWarning("DuctileBackend: buildOnly is not supported; running full benchmark")

        if "benchmarkStep" not in benchmark_config or benchmark_config["benchmarkStep"] is None:
            raise ValueError("BenchmarkProblems: Missing required backend config key: benchmarkStep")
        
        benchmark_step = benchmark_config["benchmarkStep"]
        fork_params = benchmark_step.forkParams.copy()
        param_groups = benchmark_step.paramGroups
        constant_params = benchmark_step.constantParams
        problem_type = benchmark_config.get("problemType")
        assembler = benchmark_config.get("assembler")
        debug_config = benchmark_config.get("debugConfig")
        isa_info_map = benchmark_config.get("isaInfoMap")

        if problem_type is None or assembler is None or debug_config is None or isa_info_map is None:
            raise ValueError("DuctileBackend: Missing required config keys: problemType, assembler, debugConfig, isaInfoMap")

        validate_fn = functools.partial(_validate_solution, problem_type, constant_params, assembler, debug_config, isa_info_map)

        # Merge once per run from defaults.yaml + overrides.
        merged_config = ductile_config.update(backend_config)

        # Apply group_x logic from Ductile/BenchmarkProblems.py.
        for i, group in enumerate(param_groups):
            if len(group) == 1:
                for p in group:
                    constant_params.update(p)
            else:
                fork_params[f"group_{i}"] = group

        space = SearchSpace(
            fork_params,
            valid=validate_fn,
            max_iters=merged_config["max_iters"]
        )

        selection = Selection.get(**ductile_config.populate(merged_config, "selection"))
        crossover = Crossover.get(**ductile_config.populate(merged_config, "crossover"))
        mutation = Mutation(space, **merged_config["mutation"])
        mating = Mating(
            space=space,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            max_iters=merged_config["max_iters"]
        )
        survival = Survival.get(**ductile_config.populate(merged_config, "survival"))
        
        print1(f"# DuctileBackend: Starting GA optimization")
        
        def _evaluate(individuals):
            """Fitness callback for GA - returns multi-dimensional fitness array.
            
            Takes GA individuals (parameter dicts), benchmarks them, reads CSV results,
            and returns fitness scores array matching multi-dimensional benchmark results.
            Uses Ductile/BenchmarkProblems.py error handling patterns.
            
            Args:
                individuals: List of parameter dicts from GA population
                
            Returns:
                2D numpy array of fitness scores with shape (n_sizes, len(individuals))
                where n_sizes is the number of benchmark problem sizes
            """
            # Ductile path uses Ductile-native generation to preserve per-individual alignment and duplicate handling.
            converted = _generate_ga_solutions(
                problem_type,
                constant_params,
                individuals,
                assembler,
                debug_config,
                isa_info_map,
            )
            solutions = []
            for idx, solution in enumerate(converted):
                if solution is not None:
                    solution.solIdx = idx
                    solutions.append(solution)
            
            print1(f"# DuctileBackend: Generated {len(solutions)} valid solutions from {len(individuals)} candidates")
            
            # Ductile always clears previous run artifacts before re-benchmarking each population.
            if source_path and os.path.isdir(source_path):
                shutil.rmtree(source_path)

            # Benchmark solutions - Ductile forces no cache and no build-only.
            results_filename, returncode = benchmark_runner(solutions, isCached=False, buildOnly=False)
            
            if not results_filename or not os.path.isfile(results_filename):
                printExit(f"BenchmarkProblems: Expected results file does not exist: {results_filename}")
            
            df = pd.read_csv(results_filename)
            
            # Extract benchmark columns (Cijk_*)
            cols = [c for c in df.columns.tolist() if c.lstrip().startswith("Cijk_")]
            
            if len(cols) != len(solutions):
                printExit(f"BenchmarkProblems: Mismatch between result columns and valid solutions "
                        f"(cols={len(cols)}, solutions={len(solutions)}) in {results_filename}")
            
            # Extract scores from CSV
            n_sizes = df.shape[0]
            scores = df[cols].values.astype(np.float32)
            if n_sizes == 1:
                scores = scores[None, ...]
            
            # Build multi-dimensional fitness array: (n_sizes, len(individuals))
            # nGFlops[i, j] = performance of solution j at problem size i
            nGFlops = np.zeros((n_sizes, len(individuals)), dtype=np.float32)
            idxs = [si.solIdx for si in solutions]
            nGFlops[:, idxs] = scores
            
            return nGFlops
        

        root = Path(benchmark_config.get("rootPath", "."))
        cfg_name = benchmark_config["configName"]
        if benchmark_config.get("totalBenchmarkSteps", 1) == 1:
            log_file = root / f"{cfg_name}-optimization.log"
        else:
            log_file = root / f"cfg-{cfg_name}__step-{benchmark_config.get('benchmarkStepIdx', 0):02d}__optimization.log"

        ckpt_path = root / f"step-{benchmark_config.get('benchmarkStepIdx', 0):02d}__ductile.checkpoint"
        
        # Create GA instance with evaluate callback
        ga = GeneticAlgorithm(
            space,
            mating,
            evaluate=_evaluate,
            survival=survival,
            pop_size=merged_config["pop_size"],
            n_gen=merged_config["n_gen"],
            soo=merged_config["soo"],
            period=merged_config["period"],
            tol=merged_config["tol"],
            div_thr=merged_config["div_thr"],
            seed=merged_config["seed"],
            verbose=merged_config["verbose"],
            log_file=log_file,
            checkpoint_path=ckpt_path,
            weights=merged_config["weights"],
            weight_beta=merged_config["weight_beta"],
        )

        if ckpt_path.is_file():
            printWarning(f"DuctileBackend: Found existing checkpoint at {ckpt_path}, resuming optimization from checkpoint")
            try:
                ga.load(ckpt_path)
            except Exception as e:
                printWarning(f"DuctileBackend: Failed to load checkpoint: {e}. Starting fresh optimization.")
                
        # Run GA optimization - GA internally calls _evaluate and orchestrates the loop
        best, _ = ga.optimize()
        
        netv_original = globalParameters.get("NumElementsToValidate", 128)
        try:
            # Temporarily override validation
            globalParameters["NumElementsToValidate"] = merged_config.get("n_elements_to_validate", 0)
            
            # Re-evaluate best solutions
            res = ga.evaluate(best)
            
            # Check for validation failures (-1 indicates failed validation)
            if (res == -1).any():
                if res.ndim == 1:
                    res = res[None, ...]
                
                # Find solutions that passed validation
                keep = (res > 0).all(axis=0)
                
                if keep.sum() == 0:
                    printExit(f"DuctileBackend:No solutions passed the verification stage")
                elif keep.sum() < len(best):
                    printWarning(f"# DuctileBackend: {keep.sum()}/{len(best)} passed verification")
                    best = [best[i] for i in np.where(keep)[0]]
                    
                    # Re-evaluate filtered set
                    res = ga.evaluate(best)
            
        finally:
            # Always restore original validation setting
            globalParameters["NumElementsToValidate"] = netv_original

    def supports_solution_pool(self) -> bool:
        """DuctileBackend does not support loading solutions from pool files.
        
        The GA-driven backend requires generating solutions through its own
        optimization loop rather than using pre-computed solution pools.
        
        Returns:
            False - DuctileBackend does not use solution pool files
        """
        return False
        