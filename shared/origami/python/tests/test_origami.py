# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for origami configuration selection."""

import pytest
import origami
import csv
from .conftest import create_config_list


@pytest.mark.integration
def test_select_config_basic(hardware, mat_inst):
    """Test basic configuration selection."""
    configs = create_config_list("gfx950", "f16", mat_inst)
    assert len(configs) > 0

    problem = origami.problem_t()
    problem.size = origami.dim3_t(2048, 2048, 2048)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0

    result = origami.select_config(problem, hardware, configs)
    assert result.latency > 0
    assert result.config is not None
    assert result.config.mt.m > 0
    assert result.config.mt.n > 0
    assert result.config.mt.k > 0


@pytest.mark.integration
def test_rank_configs(hardware, mat_inst):
    """Test ranking multiple configurations."""
    configs = create_config_list("gfx950", "f16", mat_inst)
    assert len(configs) > 0

    problem = origami.problem_t()
    problem.size = origami.dim3_t(8192, 8192, 8192)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype

    ranked_configs = origami.rank_configs(problem, hardware, configs)
    assert len(ranked_configs) > 0
    assert len(ranked_configs) <= len(configs)

    # Check that results are sorted by latency (best first)
    for i in range(len(ranked_configs) - 1):
        assert ranked_configs[i].latency <= ranked_configs[i + 1].latency


@pytest.mark.integration
def test_select_config_mnk(hardware, mat_inst):
    """Test select_config_mnk function."""
    configs = create_config_list("gfx950", "f16", mat_inst)
    assert len(configs) > 0

    result = origami.select_config_mnk(2048, 2048, 2048, hardware, configs)
    assert result.latency > 0
    assert result.config is not None


@pytest.mark.integration
def test_select_topk_configs(hardware, mat_inst):
    """Test select_topk_configs function."""
    configs = create_config_list("gfx950", "f16", mat_inst)
    assert len(configs) > 0

    problem = origami.problem_t()
    problem.size = origami.dim3_t(8192, 8192, 8192)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype

    topk = 5
    top_configs = origami.select_topk_configs(problem, hardware, configs, topk)
    assert len(top_configs) <= topk
    assert len(top_configs) > 0

    # Check that results are sorted by latency (best first)
    for i in range(len(top_configs) - 1):
        assert top_configs[i].latency <= top_configs[i + 1].latency


@pytest.mark.integration
@pytest.mark.slow
def test_select_config_with_csv(tmp_path, hardware, mat_inst):
    """Test configuration selection with CSV input file."""
    # Create a test CSV file
    csv_file = tmp_path / "test_sizes.csv"
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow([2048, 2048, 1, 2048])
        writer.writerow([4096, 4096, 1, 4096])

    configs = create_config_list("gfx950", "f16", mat_inst)

    with open(csv_file, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            M = int(row[0])
            N = int(row[1])
            B = int(row[2])
            K = int(row[3])

            problem = origami.problem_t()
            problem.size = origami.dim3_t(M, N, K)
            problem.batch = B
            problem.a_transpose = origami.transpose_t.T
            problem.b_transpose = origami.transpose_t.N
            problem.a_dtype = origami.string_to_datatype("f16")
            problem.b_dtype = origami.string_to_datatype("f16")
            problem.d_dtype = origami.string_to_datatype("f16")
            problem.c_dtype = problem.d_dtype
            problem.mi_dtype = problem.a_dtype

            best_config = origami.select_config(problem, hardware, configs)
            assert best_config.latency > 0
            assert best_config.config is not None


@pytest.mark.integration
def test_hardware_print(hardware):
    """Test hardware print functionality."""
    # This should not raise an exception
    hardware.print()


@pytest.mark.integration
def test_compute_perf_gflops(hardware, mat_inst):
    """Test compute_perf_gflops function."""
    configs = create_config_list("gfx950", "f16", mat_inst)
    problem = origami.problem_t()
    problem.size = origami.dim3_t(2048, 2048, 2048)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype

    result = origami.select_config(problem, hardware, configs)
    gflops = origami.compute_perf_gflops(hardware, problem, result.latency)
    assert gflops > 0


# StreamK Grid Selection Tests


@pytest.mark.integration
def test_select_grid_size(hardware):
    """Test grid size selection."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(8192, 8192, 8192)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = origami.string_to_datatype("f16")
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0

    config = origami.config_t()
    config.mt = origami.dim3_t(32, 32, 256)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 1
    config.workgroup_mapping = 6

    grid_size = origami.select_grid_size(
        problem, hardware, config, origami.grid_selection_t.analytical, hardware.N_CU
    )
    assert grid_size > 0


@pytest.mark.integration
def test_select_reduction(hardware):
    """Test reduction strategy selection."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(8192, 8192, 8192)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = origami.string_to_datatype("f16")

    config = origami.config_t()
    config.mt = origami.dim3_t(32, 32, 256)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 1
    config.workgroup_mapping = 6

    reduction = origami.select_reduction(
        problem, hardware, config, origami.grid_selection_t.analytical
    )
    assert reduction is not None


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm",
    [
        origami.grid_selection_t.number_of_cus,
        origami.grid_selection_t.min_resources,
        origami.grid_selection_t.energy_aware,
        origami.grid_selection_t.reduction_cost_aware,
        origami.grid_selection_t.data_parallel,
        origami.grid_selection_t.analytical,
        origami.grid_selection_t.k_split_aware,
    ],
)
def test_select_grid_size_algorithms(hardware, algorithm):
    """Test different grid selection algorithms."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(8192, 8192, 8192)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = origami.string_to_datatype("f16")

    config = origami.config_t()
    config.mt = origami.dim3_t(32, 32, 256)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 1
    config.workgroup_mapping = 6

    grid_size = origami.select_grid_size(problem, hardware, config, algorithm, hardware.N_CU)
    assert grid_size > 0


@pytest.mark.integration
def test_compute_number_of_output_tiles():
    """Test compute_number_of_output_tiles function."""
    num_tiles = origami.compute_number_of_output_tiles(32, 32, 8192, 8192, 1)
    assert num_tiles > 0
    assert num_tiles == (8192 // 32) * (8192 // 32) * 1


@pytest.mark.integration
def test_select_workgroup_mapping(hardware):
    """Test workgroup mapping selection."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(8192, 8192, 8192)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.d_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = origami.string_to_datatype("f16")

    config = origami.config_t()
    config.mt = origami.dim3_t(32, 32, 256)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 1
    config.workgroup_mapping = 6

    sk_grid = 100
    result = origami.select_workgroup_mapping(problem, hardware, config, sk_grid)
    assert isinstance(result, tuple)
    assert len(result) == 3


@pytest.mark.integration
def test_gfx950_bfloat16_recommended_matrix_instruction():
    """Test that gfx950 recommends 16x16x32 matrix instruction for bfloat16."""
    # Create hardware object for gfx950
    hardware = origami.hardware_t(
        origami.architecture_t.gfx950,
        304,    # N_CU
        65536,  # lds_capacity
        12,     # NUM_XCD
        1.0,    # mem1_perf_ratio
        1.0,    # mem2_perf_ratio
        1.0,    # mem3_perf_ratio
        25165824,  # L2_capacity
        2.1,    # compute_clock_ghz
        4,      # parallel_mi_cu
        (1.0, 1.0, 1.0)  # mem_bw_per_wg_coefficients
    )
    
    # Get recommended matrix instruction for bfloat16
    bfloat16_dtype = origami.data_type_t.BFloat16
    recommended_mi = hardware.get_recommended_matrix_instruction(bfloat16_dtype)
    
    # Verify it's 16x16x32
    assert recommended_mi.m == 16, f"Expected m=16, got {recommended_mi.m}"
    assert recommended_mi.n == 16, f"Expected n=16, got {recommended_mi.n}"
    assert recommended_mi.k == 32, f"Expected k=32, got {recommended_mi.k}"
