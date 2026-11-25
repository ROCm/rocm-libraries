# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Ranking Regression Tests for W4A16 (int4 A, bf16/fp16 activations) on gfx1150 and gfx1151

Same baselines and comparison as test_ranking_regression.py, with two differences that the
W4A16 model needs:
- The configs are the kernels hipBLASLt ships (data/w4a16_configs.csv), with the wave
  layout, Stream-K and occupancy that the W4A16 latency model and its grid rule read, not a
  synthetic grid. A decode kernel serves one token count and is only ranked for it.
- The problems are W4A16 shapes (data/w4a16_problem_data.csv): prefill and 1 to 5 tokens.

Usage:
    pytest test_w4a16_ranking_regression.py -v
    pytest test_w4a16_ranking_regression.py -v --generate-baseline
"""

import csv
from pathlib import Path

import pytest

import origami
from helpers import W4A16_HARDWARE
from test_ranking_regression import (
    TOP_K,
    compare_rankings,
    load_baseline,
    result_to_config_tuple,
    save_baseline,
)


DATA_DIR = Path(__file__).parent / "data"
CONFIG_DATA_FILE = DATA_DIR / "w4a16_configs.csv"
PROBLEM_DATA_FILE = DATA_DIR / "w4a16_problem_data.csv"

# The kernels are TN: A is the int4 weight matrix, transposed.
TRANS_A = origami.transpose_t.T
TRANS_B = origami.transpose_t.N

UNLIMITED = 2**64 - 1  # size_t max

# Activation type of each kernel set.
SET_DTYPE = {
    "i4_bf16_sym": "bf16",
    "i4_bf16_zp": "bf16",
    "i4_f16_sym": "f16",
    "i4_f16_zp": "f16",
}


def create_config(row: dict) -> origami.config_t:
    """An Origami config as hipBLASLt builds it for a W4A16 kernel."""
    value = lambda name: int(row[name])
    config = origami.config_t()
    config.mt = origami.dim3_t(value("mt_m"), value("mt_n"), value("mt_k"))
    config.mi = origami.dim3_t(value("mi_m"), value("mi_n"), value("mi_k"))
    config.occupancy = value("occupancy")
    config.workgroup_mapping = value("workgroup_mapping")
    config.stream_k = value("stream_k")
    # Ranked with unlimited workspace, as the Prediction library does.
    config.workspace_size = UNLIMITED
    config.workspace_size_per_elem_c = UNLIMITED
    config.cache_hints_a = value("cache_hints_a")
    config.cache_hints_b = value("cache_hints_b")
    config.cache_hints_d = value("cache_hints_d")
    config.grvw_a = value("grvw_a")
    config.grvw_b = value("grvw_b")
    config.gwvw_d = value("gwvw_d")
    config.vector_width_a = value("vector_width_a")
    config.vector_width_b = value("vector_width_b")
    tensile = origami.tensile_params_t()
    tensile.local_split_u = value("local_split_u")
    tensile.wave_group_m = value("wave_group_m")
    tensile.wave_group_n = value("wave_group_n")
    tensile.wavefront_size = value("wavefront_size")
    tensile.direct_to_vgpr_a = bool(value("direct_to_vgpr_a"))
    tensile.direct_to_vgpr_b = bool(value("direct_to_vgpr_b"))
    tensile.prefetch_global_read = value("prefetch_global_read")
    tensile.prefetch_local_read = value("prefetch_local_read")
    tensile.one_lds_buffer = bool(value("one_lds_buffer"))
    tensile.source_swap = bool(value("source_swap"))
    tensile.unroll_major_lds_a = bool(value("unroll_major_lds_a"))
    tensile.unroll_major_lds_b = bool(value("unroll_major_lds_b"))
    tensile.half_plr = value("half_plr")
    config.set_tensile_params(tensile)
    return config


def load_configs() -> dict[tuple[str, str], list[tuple[origami.config_t, int]]]:
    """(arch, kernel set) -> [(config, tokens)]; tokens is 0 for a kernel of any token count."""
    configs: dict[tuple[str, str], list[tuple[origami.config_t, int]]] = {}
    with open(CONFIG_DATA_FILE, "r") as f:
        for row in csv.DictReader(f):
            configs.setdefault((row["arch"], row["set"]), []).append(
                (create_config(row), int(row["tokens"]))
            )
    return configs


def load_problem_sizes() -> list[tuple[int, int, int]]:
    """(m, n, k): m weight rows, n tokens, k."""
    with open(PROBLEM_DATA_FILE, "r") as f:
        return [(int(r["m"]), int(r["n"]), int(r["k"])) for r in csv.DictReader(f)]


W4A16_CONFIGS = load_configs()
W4A16_PROBLEM_SIZES = load_problem_sizes()


def create_problem(m: int, n: int, k: int, dtype: str) -> origami.problem_t:
    """An int4-A problem with activations, output and MI of type dtype."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = 1
    problem.a_transpose = TRANS_A
    problem.b_transpose = TRANS_B
    problem.a_dtype = origami.string_to_datatype("i4")
    problem.b_dtype = origami.string_to_datatype(dtype)
    problem.d_dtype = problem.b_dtype
    problem.c_dtype = problem.b_dtype
    problem.mi_dtype = problem.b_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def generate_rankings(arch_name: str, kernel_set: str) -> dict[str, list[list[int]]]:
    """Top configs of the set for every W4A16 problem size, keyed as MxNxKxBatch."""
    hardware = W4A16_HARDWARE[arch_name]
    kernels = W4A16_CONFIGS[(arch_name, kernel_set)]
    rankings = {}
    for m, n, k in W4A16_PROBLEM_SIZES:
        # As hipBLASLt's size assertions do: a decode kernel only for its token count.
        configs = [config for config, tokens in kernels if tokens in (0, n)]
        problem = create_problem(m, n, k, SET_DTYPE[kernel_set])
        ranked = origami.select_topk_configs(problem, hardware, configs, TOP_K, origami.model_t.gemm)
        if ranked:
            rankings[f"{m}x{n}x{k}x1"] = [result_to_config_tuple(r) for r in ranked]
    return rankings


@pytest.mark.regression
class TestW4A16RankingRegression:
    """Ranking regression of the W4A16 kernel sets."""

    @pytest.mark.parametrize("arch_name,kernel_set", sorted(W4A16_CONFIGS))
    def test_ranking_stability(self, arch_name: str, kernel_set: str, generate_baseline: bool):
        """Test that the W4A16 rankings remain stable compared to baseline."""
        current = generate_rankings(arch_name, kernel_set)
        assert current, f"No ranking for {arch_name}/{kernel_set}"

        if generate_baseline:
            save_baseline(arch_name, kernel_set, TRANS_A, TRANS_B, current)
            pytest.skip(f"Generated baseline for {arch_name}/{kernel_set}")

        baseline = load_baseline(arch_name, kernel_set, TRANS_A, TRANS_B)
        if baseline is None:
            pytest.fail(
                f"No baseline found for {arch_name}/{kernel_set}. "
                f"Run with --generate-baseline to create it."
            )

        differences = compare_rankings(current, baseline)
        if differences:
            diff_summary = "\n".join(differences[:10])
            if len(differences) > 10:
                diff_summary += f"\n... and {len(differences) - 10} more differences"
            pytest.fail(f"Ranking regression detected for {arch_name}/{kernel_set}:\n{diff_summary}")
