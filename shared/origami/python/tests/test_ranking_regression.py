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
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

import origami
from helpers import HARDWARE, create_config_list, get_matrix_instructions


BASELINE_DIR = Path(__file__).parent / "baselines" / "rankings"
PROBLEM_DATA_FILE = Path(__file__).parent / "data" / "problem_data.csv"

SUPPORTED_DTYPES = ["f16", "bf16", "f32", "xf32", "f8"]
TRANSPOSE_VALUES = [origami.transpose_t.T, origami.transpose_t.N]

TOP_K = 5


def is_dtype_supported(arch_name: str, dtype: str) -> bool:
    """Check if a dtype is supported for the given architecture."""
    hardware = HARDWARE[arch_name]
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


def create_problem(
    m: int,
    n: int,
    k: int,
    dtype: str,
    batch: int = 1,
    transA: origami.transpose_t = origami.transpose_t.T,
    transB: origami.transpose_t = origami.transpose_t.N,
) -> origami.problem_t:
    """Create a problem specification."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = batch
    problem.a_transpose = transA
    problem.b_transpose = transB
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


def result_to_config_tuple(result: origami.prediction_result_t) -> list[int]:
    """Convert prediction result to a config tuple [mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]."""
    cfg = result.config
    return [
        cfg.mt.m, cfg.mt.n, cfg.mt.k,
        cfg.mi.m, cfg.mi.n, cfg.mi.k,
        cfg.occupancy, cfg.workgroup_mapping,
    ]


def generate_rankings(
    arch_name: str,
    dtype: str,
    transA: origami.transpose_t = origami.transpose_t.T,
    transB: origami.transpose_t = origami.transpose_t.N,
) -> dict[str, list[list[int]]]:
    """Generate rankings for all test problem sizes.
    
    Returns a dict mapping problem_key -> list of config tuples.
    Each config tuple is [mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm].
    """
    hardware = HARDWARE[arch_name]
    configs = create_config_list(
        hardware, dtype,
        mt_sizes=[16, 32, 48, 96, 128, 192, 224, 256, 336, 448, 512],
        depth_unroll=[16, 32, 64, 128, 512, 1024],
        occupancy_values=[1, 2],
        wgm_values=[1, 4, 8],
    )

    if not configs:
        return {}

    rankings = {}
    for m, n, k, batch in TEST_PROBLEM_SIZES:
        problem = create_problem(m, n, k, dtype, batch, transA, transB)
        try:
            ranked = origami.select_topk_configs(problem, hardware, configs, TOP_K)
            if ranked:
                key = f"{m}x{n}x{k}x{batch}"
                rankings[key] = [result_to_config_tuple(r) for r in ranked]
        except Exception:
            pass

    return rankings


try:
    from yaml import CSafeDumper as SafeDumper, CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeDumper, SafeLoader


def get_baseline_path(arch_name: str) -> Path:
    """Get the path to the baseline file for an architecture."""
    return BASELINE_DIR / f"{arch_name}.yaml"


@lru_cache(maxsize=None)
def load_arch_baseline(arch_name: str) -> dict | None:
    """Load and cache the full baseline for an architecture.
    
    Returns the parsed YAML dict, or None if file doesn't exist.
    """
    path = get_baseline_path(arch_name)
    if not path.exists():
        return None
    
    with open(path, "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def transpose_key(transA: origami.transpose_t, transB: origami.transpose_t) -> str:
    """Convert transpose values to a string key for baseline storage."""
    a = "T" if transA == origami.transpose_t.T else "N"
    b = "T" if transB == origami.transpose_t.T else "N"
    return f"{a}{b}"


def load_baseline(
    arch_name: str, dtype: str, transA: origami.transpose_t, transB: origami.transpose_t
) -> dict[str, list[list[int]]] | None:
    """Load baseline rankings for a specific arch/dtype/transpose combination."""
    baseline = load_arch_baseline(arch_name)
    if baseline is None:
        return None
    
    try:
        return baseline[dtype][transpose_key(transA, transB)]
    except KeyError:
        return None


def save_baseline(
    arch_name: str,
    dtype: str,
    transA: origami.transpose_t,
    transB: origami.transpose_t,
    rankings: dict[str, list[list[int]]],
) -> None:
    """Save rankings to the architecture's YAML baseline file."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = get_baseline_path(arch_name)
    
    if path.exists():
        with open(path, "r") as f:
            baseline = yaml.load(f, Loader=SafeLoader) or {}
    else:
        baseline = {}
    
    if dtype not in baseline:
        baseline[dtype] = {}
    baseline[dtype][transpose_key(transA, transB)] = rankings
    
    # Write with compact format: flow style for problem entries, sorted keys
    # Custom representer to use flow style for the innermost lists (config lists per problem)
    class CompactDumper(SafeDumper):
        pass
    
    def represent_problem_configs(dumper, data):
        # Use flow style for list of config lists (the value under each problem key)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    
    # Apply flow style to lists that contain lists (problem -> [[config], [config], ...])
    CompactDumper.add_representer(list, represent_problem_configs)
    
    with open(path, "w") as f:
        yaml.dump(baseline, f, Dumper=CompactDumper, default_flow_style=False, sort_keys=True, width=1000)
    
    # Clear cache so subsequent loads see the updated file
    load_arch_baseline.cache_clear()


def compare_rankings(
    current: dict[str, list[list[int]]], baseline: dict[str, list[list[int]]]
) -> list[str]:
    """
    Compare current rankings against baseline.

    Returns a list of differences found.
    """
    differences = []

    for problem_key, base_configs in baseline.items():
        if problem_key not in current:
            differences.append(f"Missing problem: {problem_key}")
            continue

        curr_configs = current[problem_key]
        
        if len(curr_configs) != len(base_configs):
            differences.append(
                f"{problem_key}: Different rank count (curr={len(curr_configs)}, base={len(base_configs)})"
            )

        for rank_idx, (curr_cfg, base_cfg) in enumerate(zip(curr_configs, base_configs)):
            if curr_cfg != base_cfg:
                differences.append(
                    f"{problem_key} rank {rank_idx}: Config mismatch\n"
                    f"  Current:  MT={tuple(curr_cfg[0:3])}, MI={tuple(curr_cfg[3:6])}, occ={curr_cfg[6]}, wgm={curr_cfg[7]}\n"
                    f"  Baseline: MT={tuple(base_cfg[0:3])}, MI={tuple(base_cfg[3:6])}, occ={base_cfg[6]}, wgm={base_cfg[7]}"
                )

    for problem_key in current:
        if problem_key not in baseline:
            differences.append(f"New problem not in baseline: {problem_key}")

    return differences


@pytest.mark.regression
class TestRankingRegression:
    """Test suite for ranking regression tests."""

    @pytest.mark.parametrize("arch_name", list(HARDWARE.keys()))
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("transA", TRANSPOSE_VALUES)
    @pytest.mark.parametrize("transB", TRANSPOSE_VALUES)
    def test_ranking_stability(
        self,
        arch_name: str,
        dtype: str,
        transA: origami.transpose_t,
        transB: origami.transpose_t,
        generate_baseline: bool,
    ):
        """Test that rankings remain stable compared to baseline."""
        trans_str = transpose_key(transA, transB)

        if not is_dtype_supported(arch_name, dtype):
            pytest.skip(f"No {dtype} support for {arch_name}")

        current = generate_rankings(arch_name, dtype, transA, transB)

        if not current:
            pytest.skip(f"No valid configs generated for {arch_name}/{dtype}/{trans_str}")

        if generate_baseline:
            save_baseline(arch_name, dtype, transA, transB, current)
            pytest.skip(f"Generated baseline for {arch_name}/{dtype}/{trans_str}")

        baseline = load_baseline(arch_name, dtype, transA, transB)
        if baseline is None:
            pytest.fail(
                f"No baseline file found for {arch_name}/{dtype}/{trans_str}. "
                f"Run with --generate-baseline to create it."
            )

        differences = compare_rankings(current, baseline)

        if differences:
            diff_summary = "\n".join(differences[:10])
            if len(differences) > 10:
                diff_summary += f"\n... and {len(differences) - 10} more differences"
            pytest.fail(
                f"Ranking regression detected for {arch_name}/{dtype}/{trans_str}:\n{diff_summary}"
            )
