# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for origami configuration selection."""

import pytest
import origami
import csv
from helpers import create_config_list


@pytest.mark.integration
def test_select_config_basic(hardware):
    """Test basic configuration selection."""
    configs = create_config_list(hardware, "f16")
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
def test_rank_configs(hardware):
    """Test ranking multiple configurations."""
    configs = create_config_list(hardware, "f16")
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
def test_select_config_mnk(hardware):
    """Test select_config_mnk function."""
    configs = create_config_list(hardware, "f16")
    assert len(configs) > 0

    result = origami.select_config_mnk(2048, 2048, 2048, hardware, configs)
    assert result.latency > 0
    assert result.config is not None


@pytest.mark.integration
def test_select_topk_configs(hardware):
    """Test select_topk_configs function."""
    configs = create_config_list(hardware, "f16")
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
def test_select_config_with_csv(tmp_path, hardware):
    """Test configuration selection with CSV input file."""
    # Create a test CSV file
    csv_file = tmp_path / "test_sizes.csv"
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow([2048, 2048, 1, 2048])
        writer.writerow([4096, 4096, 1, 4096])

    configs = create_config_list(hardware, "f16")

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
def test_compute_perf_gflops(hardware):
    """Test compute_perf_gflops function."""
    configs = create_config_list(hardware, "f16")
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
    assert isinstance(result, origami.workgroup_mapping_t)


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


# Formocast Simulation Mode Tests

@pytest.mark.integration
def test_prediction_modes_enum_exists():
    """Test that prediction_modes_t enum is available."""
    assert hasattr(origami, 'prediction_modes_t')
    assert hasattr(origami.prediction_modes_t, 'estimation')
    assert hasattr(origami.prediction_modes_t, 'simulation')


@pytest.mark.integration
def test_config_has_formocast_fields():
    """Test that config_t has Formocast-specific fields via variant backend."""
    config = origami.config_t()
    
    # Check prediction_mode field
    assert hasattr(config, 'prediction_mode')
    
    # Check tensile accessor method exists
    assert hasattr(config, 'tensile')
    assert callable(config.tensile)
    assert hasattr(config, 'has_tensile_params')
    assert hasattr(config, 'set_tensile_params')
    
    # Get tensile params (initializes if not set)
    tensile = config.tensile()
    
    # Check vectorization fields on config (generic, not Tensile-specific)
    assert hasattr(config, 'grvw_a')
    assert hasattr(config, 'grvw_b')
    assert hasattr(config, 'gwvw_d')
    assert hasattr(config, 'vector_width_a')
    assert hasattr(config, 'vector_width_b')
    
    # Check Tensile-specific fields
    assert hasattr(tensile, 'depth_u')
    assert hasattr(tensile, 'global_split_u')
    assert hasattr(tensile, 'global_accumulation')
    assert hasattr(tensile, 'local_split_u')
    assert hasattr(tensile, 'direct_to_vgpr_a')
    assert hasattr(tensile, 'direct_to_vgpr_b')
    assert hasattr(tensile, 'wave_num')
    assert hasattr(tensile, 'wave_group_m')
    assert hasattr(tensile, 'wave_group_n')
    assert hasattr(tensile, 'prefetch_global_read')
    
    # Verify has_tensile_params returns True after accessing
    assert config.has_tensile_params()


@pytest.mark.integration
def test_simulation_mode_returns_valid_latency():
    """Test that simulation mode returns a valid positive latency."""
    # Create hardware for gfx942
    hardware = origami.hardware_t(
        origami.architecture_t.gfx942,
        304,    # N_CU
        65536,  # lds_capacity
        8,      # NUM_XCD
        1.0,    # mem1_perf_ratio
        1.0,    # mem2_perf_ratio
        1.0,    # mem3_perf_ratio
        4000000,  # L2_capacity
        1.5,    # compute_clock_ghz
        1,      # parallel_mi_cu
        (0.0, 0.015, 0.0)  # mem_bw_per_wg_coefficients
    )
    
    # Create problem
    problem = origami.problem_t()
    problem.size = origami.dim3_t(2048, 2048, 2048)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.data_type_t.Half
    problem.b_dtype = origami.data_type_t.Half
    problem.c_dtype = origami.data_type_t.Half
    problem.d_dtype = origami.data_type_t.Half
    problem.mi_dtype = origami.data_type_t.Half
    
    # Create config with simulation mode
    config = origami.config_t()
    config.mt = origami.dim3_t(128, 128, 32)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 2
    config.workgroup_mapping = 8
    config.prediction_mode = origami.prediction_modes_t.simulation
    
    # Set Formocast-specific parameters (via tensile nested struct)
    config.tensile().depth_u = 32
    config.tensile().global_split_u = 1
    config.grvw_a = 4
    config.grvw_b = 4
    config.gwvw_d = 4
    config.tensile().wave_num = 4
    config.tensile().wave_group_m = 2
    config.tensile().wave_group_n = 2
    config.tensile().prefetch_global_read = 2
    
    # Call compute_total_latency with simulation mode
    latency = origami.compute_total_latency(problem, hardware, config, hardware.N_CU)
    
    assert latency > 0, f"Expected positive latency, got {latency}"


@pytest.mark.integration
def test_simulation_mode_via_compute_total_latency():
    """Test that compute_total_latency uses Formocast when prediction_mode is simulation."""
    # Create hardware for gfx942
    hardware = origami.hardware_t(
        origami.architecture_t.gfx942,
        304,    # N_CU
        65536,  # lds_capacity
        8,      # NUM_XCD
        1.0,    # mem1_perf_ratio
        1.0,    # mem2_perf_ratio
        1.0,    # mem3_perf_ratio
        4000000,  # L2_capacity
        1.5,    # compute_clock_ghz
        1,      # parallel_mi_cu
        (0.0, 0.015, 0.0)  # mem_bw_per_wg_coefficients
    )
    
    # Create problem
    problem = origami.problem_t()
    problem.size = origami.dim3_t(2048, 2048, 2048)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.data_type_t.Half
    problem.b_dtype = origami.data_type_t.Half
    problem.c_dtype = origami.data_type_t.Half
    problem.d_dtype = origami.data_type_t.Half
    problem.mi_dtype = origami.data_type_t.Half
    
    # Create config with estimation mode
    config_estimation = origami.config_t()
    config_estimation.mt = origami.dim3_t(128, 128, 32)
    config_estimation.mi = origami.dim3_t(16, 16, 16)
    config_estimation.occupancy = 2
    config_estimation.workgroup_mapping = 8
    config_estimation.prediction_mode = origami.prediction_modes_t.estimation
    
    # Create config with simulation mode
    config_simulation = origami.config_t()
    config_simulation.mt = origami.dim3_t(128, 128, 32)
    config_simulation.mi = origami.dim3_t(16, 16, 16)
    config_simulation.occupancy = 2
    config_simulation.workgroup_mapping = 8
    config_simulation.prediction_mode = origami.prediction_modes_t.simulation
    config_simulation.tensile().depth_u = 32
    config_simulation.tensile().global_split_u = 1
    config_simulation.grvw_a = 4
    config_simulation.grvw_b = 4
    config_simulation.gwvw_d = 4
    config_simulation.tensile().wave_num = 4
    config_simulation.tensile().wave_group_m = 2
    config_simulation.tensile().wave_group_n = 2
    config_simulation.tensile().prefetch_global_read = 2
    
    # Get latencies from both modes
    latency_estimation = origami.compute_total_latency(
        problem, hardware, config_estimation, hardware.N_CU)
    latency_simulation = origami.compute_total_latency(
        problem, hardware, config_simulation, hardware.N_CU)
    
    # Both should return positive values
    assert latency_estimation > 0, f"Estimation latency should be positive: {latency_estimation}"
    assert latency_simulation > 0, f"Simulation latency should be positive: {latency_simulation}"
    
    # The values should be different (different models)
    # Note: They use different units (cycles vs microseconds), so they will definitely differ
    assert latency_estimation != latency_simulation, \
        "Estimation and simulation should produce different results"


@pytest.mark.integration
@pytest.mark.parametrize("m,n,k", [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 512),
    (512, 4096, 4096),
])
def test_simulation_mode_various_problem_sizes(m, n, k):
    """Test simulation mode with various problem sizes."""
    # Create hardware for gfx942
    hardware = origami.hardware_t(
        origami.architecture_t.gfx942,
        304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.5, 1, (0.0, 0.015, 0.0)
    )
    
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.data_type_t.Half
    problem.b_dtype = origami.data_type_t.Half
    problem.c_dtype = origami.data_type_t.Half
    problem.d_dtype = origami.data_type_t.Half
    problem.mi_dtype = origami.data_type_t.Half
    
    config = origami.config_t()
    config.mt = origami.dim3_t(128, 128, 32)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 2
    config.workgroup_mapping = 8
    config.prediction_mode = origami.prediction_modes_t.simulation
    config.tensile().depth_u = 32
    config.tensile().global_split_u = 1
    config.grvw_a = 4
    config.grvw_b = 4
    config.gwvw_d = 4
    config.tensile().wave_num = 4
    config.tensile().wave_group_m = 2
    config.tensile().wave_group_n = 2
    
    latency = origami.compute_total_latency(problem, hardware, config, hardware.N_CU)
    assert latency > 0, f"Expected positive latency for {m}x{n}x{k}, got {latency}"
