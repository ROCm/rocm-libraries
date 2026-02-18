# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Ranking Regression Tests for Origami

These tests verify that config rankings remain stable across code changes.
Rankings are compared against golden baseline files to detect unintended
changes from PRs.

Usage:
    # Run tests (compares against baseline)
    pytest test_ranking_regression.py -v

    # Generate new baseline files (run from develop branch)
    pytest test_ranking_regression.py -v --generate-baseline

    # Update baseline for specific architecture
    pytest test_ranking_regression.py -v --generate-baseline -k gfx942
"""

import csv
from pathlib import Path

import pytest

import origami
from test_utils import SUPPORTED_ARCHITECTURES, create_hardware


BASELINE_DIR = Path(__file__).parent / "baselines" / "rankings"
PROBLEM_DATA_FILE = Path(__file__).parent / "data" / "problem_data.csv"

SUPPORTED_DTYPES = ["f16", "bf16", "f32"]


def get_matrix_instructions(hardware: origami.hardware_t, dtype: str) -> list[tuple[int, int, int]]:
    """Get valid matrix instructions from hardware for the given dtype.
    
    Filters out very small instructions (like 1x1x64 or 4x4x4) that are
    not typically used for GEMM tiling.
    """
    dtype_enum = origami.string_to_datatype(dtype)
    instructions = hardware.get_valid_matrix_instructions(dtype_enum)
    
    # Filter to reasonable GEMM tile sizes (skip dot product and very small instructions)
    result = []
    for mi in instructions:
        if mi.m >= 16 and mi.n >= 16:
            result.append((mi.m, mi.n, mi.k))
    return result


def is_dtype_supported(arch_name: str, dtype: str) -> bool:
    """Check if a dtype is supported for the given architecture."""
    hardware = create_hardware(arch_name)
    return len(get_matrix_instructions(hardware, dtype)) > 0

def load_problem_sizes() -> list[tuple[int, int, int, int]]:
    """Load problem sizes from CSV file.
    
    Returns:
        List of (m, n, k, batch) tuples.
    """
    problems = []
    with open(PROBLEM_DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["m"])
            n = int(row["n"])
            k = int(row["k"])
            batch = int(row["batch_count"])
            problems.append((m, n, k, batch))
    return problems


TEST_PROBLEM_SIZES = load_problem_sizes()


def create_configs(hardware: origami.hardware_t, dtype: str) -> list[origami.config_t]:
    """Generate a representative set of configs for testing."""
    mi_list = get_matrix_instructions(hardware, dtype)
    if not mi_list:
        return []

    configs = []

    mt_sizes = [16, 32, 48, 96, 128, 192, 224, 256, 336, 448, 512]
    depth_unroll_values = [16, 32, 64, 128, 512, 1024]
    occupancy_values = [1, 2]
    wgm_values = [1, 4, 8]

    for mi in mi_list:
        mi_m, mi_n, mi_k = mi
        for mt_m in mt_sizes:
            if mt_m < mi_m or mt_m % mi_m != 0:
                continue
            for mt_n in mt_sizes:
                if mt_n < mi_n or mt_n % mi_n != 0:
                    continue
                if mt_m * mt_n > 256 * 256:
                    continue

                for mt_k in depth_unroll_values:
                    if mt_k < mi_k or mt_k % mi_k != 0:
                        continue

                    for occ in occupancy_values:
                        for wgm in wgm_values:
                            config = origami.config_t()
                            config.mt = origami.dim3_t(mt_m, mt_n, mt_k)
                            config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
                            config.occupancy = occ
                            config.workgroup_mapping = wgm
                            configs.append(config)

    return configs


def create_problem(m: int, n: int, k: int, dtype: str, batch: int = 1) -> origami.problem_t:
    """Create a problem specification."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = batch
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype(dtype)
    problem.b_dtype = origami.string_to_datatype(dtype)
    problem.d_dtype = origami.string_to_datatype(dtype)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def config_to_tuple(config: origami.config_t) -> tuple:
    """Convert config_t to a compact tuple."""
    return (
        config.mt.m, config.mt.n, config.mt.k,
        config.mi.m, config.mi.n, config.mi.k,
        config.occupancy, config.workgroup_mapping,
    )


def result_to_row(problem_key: str, rank: int, result: origami.prediction_result_t) -> list:
    """Convert prediction result to a CSV row."""
    cfg = result.config
    return [
        problem_key, rank,
        f"{result.latency:.6g}",
        cfg.mt.m, cfg.mt.n, cfg.mt.k,
        cfg.mi.m, cfg.mi.n, cfg.mi.k,
        cfg.occupancy, cfg.workgroup_mapping,
    ]


BASELINE_HEADER = ["problem", "rank", "latency", "mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k", "occ", "wgm"]
TOP_K = 10


def generate_rankings(arch_name: str, dtype: str) -> list[list]:
    """Generate rankings for all test problem sizes.
    
    Returns a list of CSV rows, each containing:
    [problem_key, rank, latency, mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]
    """
    hardware = create_hardware(arch_name)
    configs = create_configs(hardware, dtype)

    if not configs:
        return []

    rows = []
    for m, n, k, batch in TEST_PROBLEM_SIZES:
        problem = create_problem(m, n, k, dtype, batch)
        try:
            ranked = origami.select_topk_configs(problem, hardware, configs, TOP_K)
            if ranked:
                key = f"{m}x{n}x{k}x{batch}"
                for rank, result in enumerate(ranked):
                    rows.append(result_to_row(key, rank, result))
        except Exception:
            pass

    return rows


def get_baseline_path(arch_name: str, dtype: str) -> Path:
    """Get the path to the baseline file."""
    return BASELINE_DIR / f"{arch_name}_{dtype}.csv"


def load_baseline(arch_name: str, dtype: str) -> dict[str, list[list]] | None:
    """Load baseline rankings from CSV file.
    
    Returns a dict mapping problem_key -> list of [rank, latency, mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]
    """
    path = get_baseline_path(arch_name, dtype)
    if not path.exists():
        return None
    
    baseline = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            problem_key = row[0]
            rank_data = [
                int(row[1]),    # rank
                float(row[2]),  # latency
                int(row[3]), int(row[4]), int(row[5]),  # mt
                int(row[6]), int(row[7]), int(row[8]),  # mi
                int(row[9]), int(row[10]),  # occ, wgm
            ]
            if problem_key not in baseline:
                baseline[problem_key] = []
            baseline[problem_key].append(rank_data)
    return baseline


def save_baseline(arch_name: str, dtype: str, rows: list[list]) -> None:
    """Save rankings to CSV baseline file."""
    path = get_baseline_path(arch_name, dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BASELINE_HEADER)
        writer.writerows(rows)


def compare_rankings(
    current_rows: list[list], baseline: dict[str, list[list]], tolerance: float = 1e-4
) -> list[str]:
    """
    Compare current rankings against baseline.

    Returns a list of differences found.
    """
    differences = []
    
    # Group current rows by problem key
    current_dict: dict[str, list[list]] = {}
    for row in current_rows:
        problem_key = row[0]
        rank_data = [int(row[1]), float(row[2])] + [int(x) for x in row[3:]]
        if problem_key not in current_dict:
            current_dict[problem_key] = []
        current_dict[problem_key].append(rank_data)

    for problem_key, base_ranks in baseline.items():
        if problem_key not in current_dict:
            differences.append(f"Missing problem: {problem_key}")
            continue

        curr_ranks = current_dict[problem_key]
        
        if len(curr_ranks) != len(base_ranks):
            differences.append(
                f"{problem_key}: Different rank count (curr={len(curr_ranks)}, base={len(base_ranks)})"
            )

        for curr_rank, base_rank in zip(curr_ranks, base_ranks):
            rank_idx = curr_rank[0]
            curr_latency = curr_rank[1]
            base_latency = base_rank[1]
            
            # Compare latency with relative tolerance
            if base_latency > 0:
                rel_diff = abs(curr_latency - base_latency) / base_latency
                if rel_diff > tolerance:
                    differences.append(
                        f"{problem_key} rank {rank_idx}: Latency diff {rel_diff*100:.2f}% "
                        f"(curr={curr_latency:.6g}, base={base_latency:.6g})"
                    )

            # Compare config (mt, mi, occ, wgm) - indices 2-9
            curr_cfg = tuple(curr_rank[2:])
            base_cfg = tuple(base_rank[2:])
            if curr_cfg != base_cfg:
                differences.append(
                    f"{problem_key} rank {rank_idx}: Config mismatch\n"
                    f"  Current:  MT={curr_cfg[0:3]}, MI={curr_cfg[3:6]}, occ={curr_cfg[6]}, wgm={curr_cfg[7]}\n"
                    f"  Baseline: MT={base_cfg[0:3]}, MI={base_cfg[3:6]}, occ={base_cfg[6]}, wgm={base_cfg[7]}"
                )

    for problem_key in current_dict:
        if problem_key not in baseline:
            differences.append(f"New problem not in baseline: {problem_key}")

    return differences


@pytest.mark.regression
class TestRankingRegression:
    """Test suite for ranking regression tests."""

    @pytest.mark.parametrize("arch_name", list(SUPPORTED_ARCHITECTURES.keys()))
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_ranking_stability(self, arch_name: str, dtype: str, generate_baseline: bool):
        """Test that rankings remain stable compared to baseline."""
        if not is_dtype_supported(arch_name, dtype):
            pytest.skip(f"No {dtype} support for {arch_name}")

        current_rows = generate_rankings(arch_name, dtype)

        if not current_rows:
            pytest.skip(f"No valid configs generated for {arch_name}/{dtype}")

        if generate_baseline:
            save_baseline(arch_name, dtype, current_rows)
            pytest.skip(f"Generated baseline for {arch_name}/{dtype}")

        baseline = load_baseline(arch_name, dtype)
        if baseline is None:
            pytest.fail(
                f"No baseline file found for {arch_name}/{dtype}. "
                f"Run with --generate-baseline to create it."
            )

        differences = compare_rankings(current_rows, baseline)

        if differences:
            diff_summary = "\n".join(differences[:10])
            if len(differences) > 10:
                diff_summary += f"\n... and {len(differences) - 10} more differences"
            pytest.fail(
                f"Ranking regression detected for {arch_name}/{dtype}:\n{diff_summary}"
            )
