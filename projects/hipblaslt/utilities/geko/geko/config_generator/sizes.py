# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import bisect
import logging
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

from geko.config_generator.constants import *

logger = logging.getLogger("GEKO")


# =====================================================================
# Public API
# =====================================================================

def get_sizes(config: Dict) -> List[List[int]]:
    """Resolve size categories based on the configured SIZE_OPTION.

    Returns the deduplicated list of [M, N, B, K] sizes for the two
    explicit-size modes:

        * SIZE_OPTION == 0 -- explicit list from YAML (config["Sizes"]).
        * SIZE_OPTION == 1 -- internally generated grid (M,N from the CU
          boundary x an internal K ladder); honours GRID_DENSITY.

    SIZE_OPTION == 2 (workload / log file via GEMM_LOG_PATH) does not go
    through this helper; sizes are populated on config["GemmProblems"] by
    gemm_configs_from_gemm_log_path, so calling get_sizes in that mode raises.

    Args:
        config: Configuration dictionary with size-related settings.

    Returns:
        Deduplicated list of sizes, each [M, N, B, K].

    Raises:
        ValueError: If Sizes is missing for SIZE_OPTION 0, the
            SIZE_OPTION is 2, or any other unsupported value.
    """
    if config["SIZE_OPTION"] == 0:  # Explicit [M,N,B,K] list from YAML (default).
        if "Sizes" not in config.keys() or not hasattr(config["Sizes"], "__len__"):
            raise ValueError(f"'Sizes' field must exist and be a non-empty list")
        return _deduplicate(config["Sizes"])

    elif config["SIZE_OPTION"] == 1:  # Grid-based (M,N) from CU boundary + internal K ladder.
        if "GRID_DENSITY" not in config.keys():
            logger.info("Grid density not specified, using default value of 4")
            config["GRID_DENSITY"] = 4

        if config['MACROTILE_OPT'] and config['GA']:
            raise NotImplementedError(f'GA MacroTile Optimization is not implemented in Grid mode.')
        logger.info(" Generating Grid-based library sizes ...")
        all_sizes = _generate_grid_sizes(config)
        return _deduplicate([s for sizez in all_sizes for s in sizez])
    elif config["SIZE_OPTION"] == 2:
        raise ValueError(
            "SIZE_OPTION 2 uses GEMM_LOG_PATH; sizes are in config['GemmProblems']. "
            "Do not call get_sizes for this mode."
        )
    else:  # SIZE_OPTION not in {0, 1, 2}.
        raise ValueError(
            "Incorrect SIZE_OPTION; valid options: "
            "0 = size list from YAML (default), 1 = grid (internal M,N × K ladder), "
            "2 = workload / log file via GEMM_LOG_PATH."
        )


# =====================================================================
# Deduplication
# =====================================================================

def _deduplicate(sizes: List[List[int]]) -> List[List[int]]:
    """Remove duplicate sizes while preserving order."""
    seen = dict.fromkeys(tuple(s) for s in sizes)
    deduped = [list(s) for s in seen]
    n_removed = len(sizes) - len(deduped)
    if n_removed:
        logger.info(" Removed %s duplicate size(s).", n_removed)
    return deduped


# =====================================================================
# Grid-based size generation (SIZE_OPTION=1)
# =====================================================================

def _factorize(X: int) -> List[Tuple[int, float]]:
    """Generate ordered factor pairs for a given integer.

    Args:
        X: Integer to factorize.

    Returns:
        List of (a, b) pairs where a * b == X and a >= 2.
    """
    pairs = []
    min_f = 2
    for a in range(min_f, X+1):
        if X % a:
            continue
        if int(X/a) < min_f:
            continue
        pairs.append((a, X/a))
    return pairs


def _generate_boundary(num_CUs: int, round_x: int, round_y: int, MT0: int, MT1: int) -> List[Tuple[float, float]]:
    """Generate boundary curve points for the grid.

    Args:
        num_CUs: Number of compute units.
        round_x: Rounding multiplier for X dimension.
        round_y: Rounding multiplier for Y dimension.
        MT0: Macro-tile dimension for M.
        MT1: Macro-tile dimension for N.

    Returns:
        Sorted list of boundary points (m, n).
    """
    MN_factor = _factorize(num_CUs)
    boundary_points = []

    m_min, m_max = float('inf'), 0
    n_min, n_max = float('inf'), 0

    for (m_factor, n_factor) in MN_factor:
        m = m_factor * MT0 * round_x
        n = n_factor * MT1 * round_y
        boundary_points.append((m, n))
        m_min = min(m_min, m)
        m_max = max(m_max, m)
        n_min = min(n_min, n)
        n_max = max(n_max, n)

    boundary_points.append((m_max, n_min))
    boundary_points.append((m_min, n_max))
    boundary_points = sorted(boundary_points, key=lambda x: (x[1], x[0]))
    return boundary_points


def _generate_points(boundary: Sequence[Tuple[int, int]], density: int = 1) -> List[Tuple[int, int]]:
    """Generate a square grid of points inside boundary constraints.

    Args:
        boundary: Sorted boundary points.
        density: Grid density scaling factor.

    Returns:
        List of (m, n) points.
    """
    m_max = boundary[0][0]
    n_max = boundary[-1][1]
    m_max = n_max = int(max(m_max, n_max))

    m_dim = [16, 32, 64, 96] + list(range(128, 256, 16*density)) + list(range(256, 512, 32*density)) + list(range(512, 1024, 64*density)) \
        + list(range(1024, 2048, 256*density)) + list(range(2048, 4096, 512*density)) + \
        list(range(4096, 16384, 1024*density)) + list(range(16384, m_max+1, 2048*density))

    n_dim = m_dim = set(m_dim)

    points = []
    for m in m_dim:
        for n in n_dim:
            points.append((m, n))

    return points


@lru_cache(maxsize=None)
def _find_first_point_above(m: int, M: Tuple[int, ...]) -> int:
    """Find index of the first point above a threshold."""
    index = bisect.bisect_right(M, m)
    if index < len(M):
        return index
    else:
        return -1


def _is_left(m: int, n: int, boundary: Sequence[Tuple[int, int]]) -> bool:
    """Determine if a point lies to the left of the boundary."""
    idx = _find_first_point_above(n, tuple([b[1] for b in boundary]))
    if idx == -1:
        return False
    m_top = boundary[idx][0]
    m_bottom = boundary[idx-1][0]
    n_top = boundary[idx][1]
    n_bottom = boundary[idx-1][1]
    if m_top != m_bottom:
        M = (n_top-n_bottom)/(m_top-m_bottom)
        C = n_top - M*m_top
        return (m*M+C >= n)
    else:
        return m < m_top


def _remove_points_outside_boundary(
    points: Sequence[Tuple[int, int]],
    boundary: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Filter out points that lie outside the boundary."""
    valid_points = []
    for idx in range(len(points)):
        m, n = points[idx]
        if not _is_left(m, n, boundary):
            continue
        valid_points.append((m, n))
    return valid_points


def _generate_sizes_from_points(
    points: Sequence[Tuple[int, int]],
    K_dim_list: Sequence[int],
) -> Tuple[int, List[List[List[int]]]]:
    """Generate size lists for each K dimension.

    Args:
        points: Grid points for (m, n).
        K_dim_list: List of K dimensions.

    Returns:
        Tuple of total grid size and list of size lists.
    """
    totalgrid = 0
    sizes = []
    for k in K_dim_list:
        size_k = []
        for point in points:
            if k > 4096 and point[0] > 1024 and point[1] > 1024:
                continue
            size_k.append([point[0], point[1], 1, k])
            totalgrid += 1
        sizes.append(size_k)
    return totalgrid, sizes


def _generate_grid_sizes(config: Dict[str, Any]) -> List[List[List[List[int]]]]:
    """Generate grid-based sizes from config constraints.

    Filters points in a square grid based on a boundary determined
    by sizes corresponding to MT=256x256 and rounds=2x2 for CU combinations.

    Args:
        config: Configuration dictionary with grid settings.

    Returns:
        Nested list of size groups suitable for the config generator.
    """
    CUs = config["CUs"]
    MT0, MT1 = GRID_BOUNDARY_MT
    round_x, round_y = 2, 2
    density = config["GRID_DENSITY"]
    assert density >= 1 and density <= 5, "Grid density range 1-5"

    boundary = _generate_boundary(CUs, round_x, round_y, MT0, MT1)
    points = _generate_points(boundary, density)
    points = _remove_points_outside_boundary(points, boundary)
    points = sorted(points)
    logger.info(" Total number of sizes in the grid: %s", len(points))
    totalgrid, sizes = _generate_sizes_from_points(points, GRID_BOUNDARY_K_LEVELS)
    logger.info(" Total number of sizes in the grid: %s", totalgrid)

    return sizes
