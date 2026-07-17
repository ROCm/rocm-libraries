# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Utility functions for 'operations.minimum_solution_bank'.

Provides functions to cluster solutions, solve the assignment problem,
and to execute the main workflow for a single library.

Used to prune the library. Refer to the description to the prune_library function.

Functions:
    cluster_solutions: Cluster solutions and sizes by the
        MacroTile x DepthU of the given solutions.
    solve_set_cover: Solve the set cover problem with its greedy
        algorithm implementation.
    min_assigment: Find the minimum set of solutions that maintain
        performance.
"""

import copy
import logging
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Set, List, Tuple, Sequence

from geko.library import Library
from geko.constants import GEMM_FIELDS
from geko import bench


logger = logging.getLogger("GEKO")


def cluster_solutions(lib: Library, other_keys: Sequence[str] = None) -> dict:
    """Cluster solutions and sizes by the MacroTile x DepthU of the given solutions.

       Used in library prunning. Refer to the description to the prune_library function.

    Args:
        lib (Library): Input Library data structure with solutions and sizes.

    Returns:
        dict: Dictionary containing clusters (MacroTile x DepthU) with their corresponding
              solutions and sizes.
        other_keys (Sequence[str], optional): Additional parameters to cluster on.
            Defaults to None.
    """
    other_keys = other_keys or []
    key_names = tuple(["MacroTile0", "MacroTile1", "DepthU"] + list(other_keys))
    MTDU = {}
    for i, sol in enumerate(lib.solutions):
        key = tuple(int(sol[kn]) if isinstance(sol[kn], bool) else sol[kn] for kn in key_names)
        sizes_ = copy.deepcopy([sz for sz in lib.sizes if sz[1][0] == i])

        if len(sizes_) == 0:
            continue

        if key not in MTDU:
            MTDU[key] = {"solutions": [], "sizes": []}

        sol_ = copy.deepcopy(sol)
        kidx = len(MTDU[key]["solutions"])
        sol_["SolutionIndex"] = kidx
        for sz in sizes_:
            sz[1][0] = kidx

        MTDU[key]["solutions"].append(sol_)
        MTDU[key]["sizes"].extend(sizes_)

    return MTDU


def solve_set_cover(
    universe: Set[int],
    subsets: List[Tuple[int, Set[int]]],
) -> Set[int]:
    """Solve the set cover problem with its greedy algorithm implementation.

    Repeatedly select the set that covers the most new, uncovered elements.

    Args:
        universe (Set[int]): Set of problems to solve, in this case GEMM IDs.
        subsets (List[Tuple[int, Set[int]]]): List of solutions and the probleam
            each solves.

    Returns:
        Set[int]: Solution requireds to cover the whole universe (solve GEMMs).
    """
    elements = set(e for s in subsets for e in s[1])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s[1] - covered))
        cover.append(subset[0])
        covered |= subset[1]
    return cover

def _scale_tol(flops, tol, c=9.0, cap=2.5, max_delta=0.03):
    """Scale tolerance with a smooth inverse-saturating function.

    Keeps tolerance unchanged when the normalized performance ratio is at least 1,
    and increases it for smaller ratios following an inverse trend with saturation.

    Args:
        flops (float): Mean FLOPs used to derive a normalized ratio.
        tol (float): Base tolerance.
        c (float, optional): Saturation control parameter. Higher values make the
            curve rise more slowly. Defaults to 9.0.
        cap (float, optional): Maximum multiplicative factor on ``tol``.
            Defaults to 2.5.
        max_delta (float, optional): Maximum absolute increase allowed over
            base ``tol``. Defaults to 0.03.

    Returns:
        float: Scaled tolerance in ``[tol, min(cap * tol, tol + max_delta)]``.
    """
    r = flops / 2e9
    if r >= 1.0:
        return tol
    # Inverse term (bigger when r is smaller), then saturate smoothly.
    x = (1.0 / r) - 1.0
    factor = 1.0 + 1.5 * (x / (x + c))  # in [1, 2.5)
    scaled_tol = tol * min(cap, factor)

    # Prevent overly permissive pruning when base tolerance is already high.
    return min(scaled_tol, tol + max_delta)

def min_assigment(
    hipblaslt_path: Path,
    lib: Library,
    cluster_dir: Path,
    custom_lib_dir: Path,
    devices: Sequence[int] | None = None,
    tol: float = 0.02,
    scale_tol: bool = True,
) -> Tuple[List[dict], List]:
    """Find the minimum set of solutions that maintain performance.

    Prune solution with similar performance for each size.
    For a given cluster, solves the assignment problem of finding the minimum
    set of solutions that will maintain optimal performance for the sizes
    they solve.

    Workflow:

    1 - Benchmarks all sizes against all solutions in the cluster.
    2 - Filters out non-feasible pairs based of the given tolerance (e.g.
        size_x-kernel_y with relative performance < (1.0 - tol)).
    3 - Reformulates the problem as a set cover problem (see 'solve_set_cover')
        and computes the necessary solutions.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation
        lib (Library): Input Library data structure with solutions and sizes.
        cluster_dir (str | Path): Working directory path for the given cluster.
        custom_lib_dir (str | Path): Custom library directory.
        devices (Sequence[int], optional): GPU device IDs to use for benchmarking.
            Defaults to None, which is interpreted as [0] if not specified.
        tol (float, optional): Performance error threshold to consider a solution
            'optimal' for a size. Defaults to 0.02 (2%).
        scale_tol (bool, optional): Whether to scale the tolerance based on the FLOPs 
            of the GEMM. This allows for more aggressive pruning on smaller GEMMs where 
            performance differences are more significant. Defaults to True.

    Returns:
        Tuple[List[dict], List]: Tuple containing the new subset of solutions
            and their sizes.
    """
    if devices is None:
        devices = [0]
    bench_file = lib.create_bench_input(cluster_dir)[0]
    bench_data = bench.log.parse(bench_file)
    bench_data = bench.log.update(bench_data, beta=False, flush=False)[0]
    all_flops = []
    for config in bench_data:
        flops = 2 * config["M"] * config["N"] * config["K"] * config["batch_count"]
        iters = min(max(int(7e13 / flops), 10), 1000)
        config["iters"] = iters
        config["cold_iters"] = int(iters * 0.2)
        config["algo_method"] = 0
        config["requested_solution_num"] = -1
        all_flops.append(flops)
    
    if scale_tol:
        tol = _scale_tol(np.mean(all_flops), tol)

    bench_data = pd.DataFrame(bench_data).drop_duplicates().to_dict(orient="records")

    bench.log.dump(bench_data, bench_file)

    log_file = cluster_dir / "latency.out"
    try:
        df = bench.utils.parse_benchmark_output(log_file)
        if len(df.groupby(list(GEMM_FIELDS))) != len(bench_data):
            raise ValueError()
    except (FileNotFoundError, ValueError):
        df = bench.run(
            hipblaslt_path,
            bench_file,
            log_file,
            custom_lib_dir=custom_lib_dir,
            devices=devices,
            cache=False,
            silent=True if len(devices) == 1 else False,  # Only mute for single-device runs
        )

    df = df.drop_duplicates().reset_index(drop=True)
    df["solutionIdx"] = df["solutionIdx"].astype(int)

    df["GEMMID"] = df.groupby(list(GEMM_FIELDS)).ngroup()
    df["EFF"] = df.groupby(list(GEMM_FIELDS))["hipblaslt-Gflops"].transform(lambda x: x / x.max())
    df = df[df["EFF"] > (1.0 - tol)]

    valid_sols = []
    for gsi, row in df.groupby("solutionIdx"):
        valid_sols.append((gsi, set(row["GEMMID"].values.tolist())))

    keep_indices = solve_set_cover(set(df["GEMMID"]), valid_sols)
    if keep_indices is None:
        logger.warning(
            "Skipping pruning for cluster '%s': infeasible set cover after EFF filtering (tol=%s)",
            cluster_dir,
            tol,
        )
        return copy.deepcopy(lib.solutions), copy.deepcopy(lib.sizes)

    new_sizes = []
    new_sols = copy.deepcopy([lib.solutions[i] for i in keep_indices])
    for i, (_, gemm) in enumerate(df.groupby("GEMMID")):
        gemm = gemm[gemm["solutionIdx"].isin(keep_indices)]  # Filter non-picked solutions
        best_sol_idx = gemm.iloc[gemm["EFF"].argmax()]["solutionIdx"]  # Max perf from picked solutions
        new_sol_idx = keep_indices.index(best_sol_idx)  # New solution index
        size = [[int(dim) for dim in gemm.iloc[0][["m", "n", "batch_count", "k"]].values], [new_sol_idx, 0.0]]
        new_sizes.append(size)

    for i, sol in enumerate(new_sols):
        sol["SolutionIndex"] = i

    return new_sols, new_sizes
