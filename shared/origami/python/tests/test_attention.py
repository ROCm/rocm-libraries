# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for Attention origami functions."""

import pytest
import origami


def create_attention_problem(q_seq_len, kv_seq_len, head_dim, q_heads=32, batch=1, dtype="f16",
                             causal=False):
    """Create an attention problem specification."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(q_seq_len, kv_seq_len, head_dim)
    problem.batch = batch
    problem.q_heads = q_heads
    problem.causal = causal
    problem.a_transpose = origami.transpose_t.N
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype(dtype)
    problem.b_dtype = origami.string_to_datatype(dtype)
    problem.d_dtype = origami.string_to_datatype(dtype)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def create_attention_config(mt_m, mt_n, mt_k, mi_m=16, mi_n=16, mi_k=16, occupancy=1, wgm=1):
    """Create an attention configuration."""
    config = origami.config_t()
    config.mt = origami.dim3_t(mt_m, mt_n, mt_k)
    config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
    config.occupancy = occupancy
    config.workgroup_mapping = wgm
    return config


@pytest.fixture
def hardware():
    """Fixture for gfx942 hardware."""
    return origami.get_hardware_for_arch(
        arch=origami.architecture_t.gfx942,
        N_CU=304,
        lds_capacity=64 * 1024,
        rf_capacity=512 * 1024,
        L2_capacity=4000000,
        compute_clock_khz=2100000
    )


@pytest.mark.integration
def test_att_calculate_work_utilization(hardware):
    """Test attention work utilization calculation."""
    # Perfect tile alignment
    problem = create_attention_problem(4096, 4096, 128)
    config = create_attention_config(128, 128, 64)

    result = origami.att_calculate_work_utilization(problem, config)
    assert result == 1.0

    # Non-aligned dimensions
    problem = create_attention_problem(4100, 4096, 128)
    result = origami.att_calculate_work_utilization(problem, config)
    assert result < 1.0
    assert result > 0.97


@pytest.mark.integration
def test_att_calculate_output_utilization(hardware):
    """Test attention output utilization calculation."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    result = origami.att_calculate_output_utilization(problem, config, 1)
    assert result > 0.0
    assert result <= 1.1  # Can be slightly > 1.0 with vector_elems


@pytest.mark.integration
def test_att_compute_cu_occupancy(hardware):
    """Test attention CU occupancy calculation."""
    problem = create_attention_problem(2048, 2048, 128, q_heads=1, batch=1)
    config = create_attention_config(128, 128, 64)

    num_wgs, num_active_cus, num_timesteps, split_factor = origami.att_compute_cu_occupancy(
        problem, hardware, config, origami.grid_selection_t.k_split_aware, hardware.N_CU, 1
    )

    # With q_heads=1, batch=1: num_tiles = (2048/128) * (2048/128) * 1 * 1 = 256
    assert num_wgs == 256
    assert num_active_cus <= hardware.N_CU
    assert num_timesteps >= 1
    assert split_factor == 1


@pytest.mark.integration
def test_att_compute_cu_occupancy_with_qheads(hardware):
    """Test attention CU occupancy with multiple query heads."""
    problem = create_attention_problem(2048, 2048, 128, q_heads=32, batch=1)
    config = create_attention_config(128, 128, 64)

    num_wgs, num_active_cus, num_timesteps, split_factor = origami.att_compute_cu_occupancy(
        problem, hardware, config, origami.grid_selection_t.k_split_aware, hardware.N_CU, 1
    )

    # With q_heads=32, batch=1: num_tiles = 16 * 16 * 1 * 32 = 8192
    assert num_wgs == 8192
    assert num_active_cus <= hardware.N_CU


@pytest.mark.integration
def test_att_compute_cu_occupancy_with_split(hardware):
    """Test attention CU occupancy with split > 1."""
    problem = create_attention_problem(2048, 2048, 128, q_heads=1, batch=1)
    config = create_attention_config(128, 128, 64)

    num_wgs, num_active_cus, num_timesteps, split_factor = origami.att_compute_cu_occupancy(
        problem, hardware, config, origami.grid_selection_t.k_split_aware, hardware.N_CU, 4
    )

    # num_wgs = 256 * 4 = 1024
    assert num_wgs == 1024
    assert split_factor == 4


@pytest.mark.integration
def test_att_arithmetic_intensity(hardware):
    """Test attention arithmetic intensity calculation."""
    # arithmetic_intensity(m, n, k, bytes_per_element)
    ai = origami.att_arithmetic_intensity(2048.0, 2048.0, 128.0, 2.0)  # FP16 = 2 bytes
    assert ai > 0.0


@pytest.mark.integration
def test_att_emulated_tf32_arithmetic_intensity(hardware):
    """Test emulated TF32 arithmetic intensity for attention."""
    # emulated_tf32_arithmetic_intensity(m, n, k, bytes_per_element)
    ai = origami.att_emulated_tf32_arithmetic_intensity(2048.0, 2048.0, 128.0, 4.0)  # FP32 = 4 bytes
    assert ai > 0.0


@pytest.mark.integration
def test_att_round_elements_to_128B(hardware):
    """Test rounding elements to 128B boundary for attention."""
    # FP16 (2 bytes per element)
    rounded = origami.att_round_elements_to_128B(100, 2)
    assert rounded >= 100
    assert (rounded * 2) % 128 == 0


@pytest.mark.integration
def test_att_compute_number_matrix_instructions(hardware):
    """Test computing number of matrix instructions for attention."""
    mt = origami.dim3_t(128, 128, 64)
    mi = origami.dim3_t(16, 16, 16)

    num_instrs = origami.att_compute_number_matrix_instructions(mt, mi)
    # (128/16) * (128/16) * (64/16) = 8 * 8 * 4 = 256
    assert num_instrs == 256


@pytest.mark.integration
def test_att_compute_mt_compute_latency(hardware):
    """Test macro-tile compute latency for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64, mi_m=16, mi_n=16, mi_k=16)

    latency = origami.att_compute_mt_compute_latency(problem, hardware, config)
    assert latency > 0.0


@pytest.mark.integration
def test_att_check_lds_capacity(hardware):
    """Test LDS capacity check for attention."""
    problem = create_attention_problem(2048, 2048, 128)

    # Signature: check_lds_capacity(hardware, mt, dtype)
    mt_small = origami.dim3_t(64, 64, 32)
    assert origami.att_check_lds_capacity(hardware, mt_small, problem.a_dtype) == True

    # Very large config that might not fit
    mt_large = origami.dim3_t(512, 512, 128)
    result = origami.att_check_lds_capacity(hardware, mt_large, problem.a_dtype)
    assert isinstance(result, bool)


@pytest.mark.integration
def test_att_estimate_l2_hit(hardware):
    """Test L2 hit rate estimation for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    l2_hit = origami.att_estimate_l2_hit(problem, hardware, config, 1)
    assert 0.0 <= l2_hit <= 1.0


@pytest.mark.integration
def test_att_estimate_mall_hit(hardware):
    """Test MALL hit rate estimation for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    # Signature: estimate_mall_hit(problem, hardware, config, num_active_cus, splitting_factor)
    mall_hit = origami.att_estimate_mall_hit(problem, hardware, config, 256, 1)
    assert 0.0 <= mall_hit <= 1.0


@pytest.mark.integration
def test_att_compute_l2_hit_rate_global(hardware):
    """Test global L2 hit rate for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    # Signature: compute_l2_hit_rate_global(problem, hardware, config, num_active_cus)
    l2_hit_rate = origami.att_compute_l2_hit_rate_global(problem, hardware, config, 256)
    assert 0.0 <= l2_hit_rate <= 1.0


@pytest.mark.integration
def test_att_compute_mem_bw_from_occupancy(hardware):
    """Test memory bandwidth from occupancy for attention."""
    # Signature: compute_mem_bw_from_occupancy(hardware, num_active_cus)
    mem_bw = origami.att_compute_mem_bw_from_occupancy(hardware, 200)
    assert mem_bw > 0.0


@pytest.mark.integration
def test_att_compute_memory_latency(hardware):
    """Test memory latency calculation for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    # Signature: compute_memory_latency(problem, hardware, config, num_active_cus, splitting_factor)
    latency = origami.att_compute_memory_latency(problem, hardware, config, 256, 1)
    assert latency >= 0.0


@pytest.mark.integration
def test_att_compute_tile_latency(hardware):
    """Test tile latency calculation for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    # Signature: compute_tile_latency(problem, hardware, config, num_active_cus, splitting_factor)
    latency = origami.att_compute_tile_latency(problem, hardware, config, 256, 1)
    assert latency > 0.0


@pytest.mark.integration
def test_att_compute_timestep_latency(hardware):
    """Test timestep latency calculation for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    # Signature: compute_timestep_latency(problem, hardware, config, num_active_cus, splitting_factor)
    latency = origami.att_compute_timestep_latency(problem, hardware, config, 256, 1)
    assert latency > 0.0


@pytest.mark.integration
def test_att_compute_total_latency(hardware):
    """Test total latency calculation for attention."""
    problem = create_attention_problem(2048, 2048, 128)
    config = create_attention_config(128, 128, 64)

    latency = origami.att_compute_total_latency(problem, hardware, config)
    assert latency > 0.0


@pytest.mark.integration
def test_att_gqa_scenario(hardware):
    """Test attention functions with Grouped Query Attention."""
    # GQA: 32 query heads, 8 kv heads
    problem = create_attention_problem(2048, 2048, 128, q_heads=32, batch=1)
    config = create_attention_config(128, 128, 64)

    # Work utilization should still work
    work_util = origami.att_calculate_work_utilization(problem, config)
    assert work_util > 0.0

    # Total latency should be positive
    latency = origami.att_compute_total_latency(problem, hardware, config)
    assert latency > 0.0


@pytest.mark.integration
def test_att_mqa_scenario(hardware):
    """Test attention functions with Multi-Query Attention."""
    # MQA: 16 query heads, 1 kv head
    problem = create_attention_problem(1024, 1024, 64, q_heads=16, batch=1)
    config = create_attention_config(64, 64, 32)

    # Work utilization should still work
    work_util = origami.att_calculate_work_utilization(problem, config)
    assert work_util > 0.0

    # Total latency should be positive
    latency = origami.att_compute_total_latency(problem, hardware, config)
    assert latency > 0.0


@pytest.mark.integration
def test_att_batched_scenario(hardware):
    """Test attention functions with batch size > 1."""
    problem = create_attention_problem(1024, 1024, 128, q_heads=8, batch=4)
    config = create_attention_config(128, 128, 64)

    # CU occupancy should account for batch
    num_wgs, _, _, _ = origami.att_compute_cu_occupancy(
        problem, hardware, config, origami.grid_selection_t.k_split_aware, hardware.N_CU, 1
    )

    # With batch=4, q_heads=8: num_tiles = (1024/128) * (1024/128) * 4 * 8 = 2048
    assert num_wgs == 2048


@pytest.mark.integration
def test_att_various_sequence_lengths(hardware):
    """Test attention with various sequence length combinations."""
    test_cases = [
        (512, 512, 64),      # Small
        (2048, 2048, 128),   # Medium
        (4096, 4096, 128),   # Large
        (1024, 2048, 64),    # Rectangular (cross-attention)
    ]

    for q_len, kv_len, head_dim in test_cases:
        problem = create_attention_problem(q_len, kv_len, head_dim, q_heads=8)
        config = create_attention_config(128, 128, 64)

        latency = origami.att_compute_total_latency(problem, hardware, config)
        assert latency > 0.0, f"Failed for dims ({q_len}, {kv_len}, {head_dim})"


@pytest.mark.integration
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_att_various_dtypes(hardware, dtype):
    """Test attention functions with various data types."""
    problem = create_attention_problem(1024, 1024, 128, dtype=dtype)
    config = create_attention_config(128, 128, 64)

    latency = origami.att_compute_total_latency(problem, hardware, config)
    assert latency > 0.0


@pytest.mark.integration
def test_att_different_block_sizes(hardware):
    """Test attention with different macro-tile sizes."""
    problem = create_attention_problem(2048, 2048, 128)

    block_sizes = [
        (64, 64, 32),
        (128, 128, 64),
        (256, 256, 64),
    ]

    for mt_m, mt_n, mt_k in block_sizes:
        config = create_attention_config(mt_m, mt_n, mt_k)
        latency = origami.att_compute_total_latency(problem, hardware, config)
        assert latency > 0.0, f"Failed for block size ({mt_m}, {mt_n}, {mt_k})"


# Attention Config Selection Tests (using model_t.attention)


def create_attention_config_list(hardware: origami.hardware_t, dtype: str = "f16") -> list[origami.config_t]:
    """Create a list of configurations for attention testing."""
    configs = []

    # Get valid matrix instructions for the dtype
    dtype_enum = origami.string_to_datatype(dtype)
    mi_list = hardware.get_valid_matrix_instructions(dtype_enum)

    if not mi_list:
        return []

    # Use common attention block sizes
    block_sizes = [(64, 64, 32), (128, 128, 64), (256, 128, 64)]
    occupancy_values = [1, 2]
    wgm_values = [1, 4]

    for mi in mi_list:
        for mt_m, mt_n, mt_k in block_sizes:
            for occ in occupancy_values:
                for wgm in wgm_values:
                    config = origami.config_t()
                    config.mt = origami.dim3_t(mt_m, mt_n, mt_k)
                    config.mi = origami.dim3_t(mi.m, mi.n, mi.k)
                    config.occupancy = occ
                    config.workgroup_mapping = wgm
                    configs.append(config)

    return configs


@pytest.mark.integration
def test_att_select_config_basic(hardware):
    """Test basic attention configuration selection using model_t.attention."""
    configs = create_attention_config_list(hardware, "f16")
    assert len(configs) > 0

    problem = create_attention_problem(2048, 2048, 128, q_heads=32, batch=1, dtype="f16")

    result = origami.select_config(problem, hardware, configs, origami.model_t.attention)
    assert result.latency > 0
    assert result.config is not None
    assert result.config.mt.m > 0
    assert result.config.mt.n > 0
    assert result.config.mt.k > 0


@pytest.mark.integration
def test_att_rank_configs(hardware):
    """Test ranking multiple attention configurations."""
    configs = create_attention_config_list(hardware, "f16")
    assert len(configs) > 0

    problem = create_attention_problem(4096, 4096, 128, q_heads=32, batch=1, dtype="f16")

    ranked_configs = origami.rank_configs(problem, hardware, configs, origami.model_t.attention)
    assert len(ranked_configs) > 0
    assert len(ranked_configs) <= len(configs)

    # Check that results are sorted by latency (best first)
    for i in range(len(ranked_configs) - 1):
        assert ranked_configs[i].latency <= ranked_configs[i + 1].latency


@pytest.mark.integration
def test_att_select_topk_configs(hardware):
    """Test select_topk_configs for attention problems."""
    configs = create_attention_config_list(hardware, "f16")
    assert len(configs) > 0

    problem = create_attention_problem(2048, 2048, 128, q_heads=16, batch=1, dtype="f16")

    topk = 5
    top_configs = origami.select_topk_configs(problem, hardware, configs, topk, origami.model_t.attention)
    assert len(top_configs) <= topk
    assert len(top_configs) > 0

    # Check that results are sorted by latency (best first)
    for i in range(len(top_configs) - 1):
        assert top_configs[i].latency <= top_configs[i + 1].latency


@pytest.mark.integration
def test_att_various_problem_sizes(hardware):
    """Test attention config selection with various problem sizes."""
    configs = create_attention_config_list(hardware, "f16")
    assert len(configs) > 0

    test_cases = [
        (512, 512, 64, 8),       # Small
        (2048, 2048, 128, 32),   # Medium
        (4096, 4096, 128, 32),   # Large
        (1024, 2048, 64, 16),    # Rectangular (cross-attention)
    ]

    for q_len, kv_len, head_dim, q_heads in test_cases:
        problem = create_attention_problem(q_len, kv_len, head_dim, q_heads=q_heads, batch=1)
        result = origami.select_config(problem, hardware, configs, origami.model_t.attention)
        assert result.latency > 0, f"Failed for size ({q_len}, {kv_len}, {head_dim}, q_heads={q_heads})"


@pytest.mark.integration
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_att_config_selection_dtypes(hardware, dtype):
    """Test attention config selection with various data types."""
    configs = create_attention_config_list(hardware, dtype)

    # Skip if no configs available for this dtype
    if len(configs) == 0:
        pytest.skip(f"No valid configs for dtype {dtype}")

    problem = create_attention_problem(1024, 1024, 128, q_heads=16, batch=1, dtype=dtype)
    result = origami.select_config(problem, hardware, configs, origami.model_t.attention)
    assert result.latency > 0


@pytest.mark.integration
def test_att_batched_config_selection(hardware):
    """Test attention config selection with batched inputs."""
    configs = create_attention_config_list(hardware, "f16")
    assert len(configs) > 0

    # Test with batch size > 1
    problem = create_attention_problem(1024, 1024, 128, q_heads=16, batch=8, dtype="f16")
    result = origami.select_config(problem, hardware, configs, origami.model_t.attention)
    assert result.latency > 0
    assert result.config is not None


# Causal attention tests


@pytest.mark.integration
def _brute_causal_pairs(M, N, mt_m, mt_n):
    """Independent brute-force reference for causal active (Q-tile, KV-tile) pairs."""
    import math
    grid_m = math.ceil(M / mt_m)
    grid_n = math.ceil(N / mt_n)
    offset = max(N - M, 0)
    total = 0
    for q in range(grid_m):
        last_q_row = min((q + 1) * mt_m, M) - 1
        kv_tiles = (last_q_row + offset) // mt_n + 1
        total += min(kv_tiles, grid_n)
    return total


def test_att_causal_active_tile_pairs(hardware):
    """Test triangular total active (Q-tile, KV-tile) pairs (square tiles)."""
    # grid=32 -> 32*33/2 = 528 (~half of 32*32=1024).
    assert origami.att_causal_active_tile_pairs(32, 32, 128, 128, 4096, 4096) == 528
    # grid=16 -> 16*17/2 = 136.
    assert origami.att_causal_active_tile_pairs(16, 16, 128, 128, 2048, 2048) == 136
    # Decode scenario: single Q-tile (M=128), large KV context (N=1024)
    assert origami.att_causal_active_tile_pairs(1, 8, 128, 128, 128, 1024) == 8


def test_att_causal_active_tile_pairs_non_square(hardware):
    """Regression: mt_m != mt_n. Diagonal advances mt_m/mt_n KV-tiles per Q-tile.

    The old grid_m*(grid_m+1)/2 form ignored the tile aspect ratio and
    undercounted by mt_m/mt_n. Expected values verified against brute force.
    """
    cap = origami.att_causal_active_tile_pairs
    # (grid_m, grid_n, mt_m, mt_n, M, N, expected)
    cases = [
        (16, 64, 128, 32, 2048, 2048, 544),   # r=4, square problem (pure triangular)
        (8, 64, 128, 16, 1024, 1024, 288),    # r=8
        (16, 64, 128, 64, 2048, 4096, 784),   # r=2, with offset (M<N)
        (64, 16, 32, 128, 2048, 2048, 544),   # r=1/4
        (16, 8, 64, 128, 1024, 1024, 72),     # r=1/2
        (1, 128, 128, 32, 128, 4096, 128),    # decode, non-square
        (2, 256, 128, 32, 256, 8192, 508),    # saturation (large offset)
        (3, 128, 128, 32, 270, 4096, 380),    # partial final M tile, non-square
        (12, 79, 256, 64, 3000, 5000, 695),   # partial M and N, non-square
    ]
    for grid_m, grid_n, mt_m, mt_n, M, N, expected in cases:
        got = cap(grid_m, grid_n, mt_m, mt_n, M, N)
        assert got == expected, f"{(grid_m, grid_n, mt_m, mt_n, M, N)}: {got} != {expected}"
        assert got == _brute_causal_pairs(M, N, mt_m, mt_n)
        assert grid_m <= got <= grid_m * grid_n


def test_att_causal_active_tile_pairs_matches_brute_force(hardware):
    """Sweep many shapes/tiles (incl. non-square) against the brute-force reference."""
    import math
    tiles = [16, 32, 64, 128, 256]
    seqs = [128, 256, 270, 512, 1024, 2048, 4096]
    for M in seqs:
        for N in seqs:
            if N < M:
                continue  # model requires M <= N for causal
            for mt_m in tiles:
                for mt_n in tiles:
                    grid_m = math.ceil(M / mt_m)
                    grid_n = math.ceil(N / mt_n)
                    got = origami.att_causal_active_tile_pairs(grid_m, grid_n, mt_m, mt_n, M, N)
                    assert got == _brute_causal_pairs(M, N, mt_m, mt_n), \
                        f"mismatch at {(M, N, mt_m, mt_n)}: {got}"


@pytest.mark.integration
def test_att_causal_total_latency_lower(hardware):
    """Causal total latency should be lower than non-causal for a square problem."""
    config = create_attention_config(128, 128, 64)

    noncausal = create_attention_problem(4096, 4096, 128, causal=False)
    causal = create_attention_problem(4096, 4096, 128, causal=True)

    lat_noncausal = origami.att_compute_total_latency(noncausal, hardware, config)
    lat_causal = origami.att_compute_total_latency(causal, hardware, config)

    assert lat_causal > 0.0
    assert lat_causal < lat_noncausal
    # Large square problem: loop work is ~triangular. Generous tolerance.
    assert lat_causal < 0.75 * lat_noncausal


@pytest.mark.integration
def test_att_causal_hit_rates_bounded(hardware):
    """Causal hit rates stay in [0,1] and do not exceed the non-causal values."""
    config = create_attention_config(128, 128, 64)

    noncausal = create_attention_problem(2048, 2048, 128, causal=False)
    causal = create_attention_problem(2048, 2048, 128, causal=True)

    l2_c = origami.att_estimate_l2_hit(causal, hardware, config, 1)
    l2_nc = origami.att_estimate_l2_hit(noncausal, hardware, config, 1)
    assert 0.0 <= l2_c <= 1.0
    assert l2_c <= l2_nc + 1e-9

    mall_c = origami.att_estimate_mall_hit(causal, hardware, config, 256, 1)
    mall_nc = origami.att_estimate_mall_hit(noncausal, hardware, config, 256, 1)
    assert 0.0 <= mall_c <= 1.0
    assert mall_c <= mall_nc + 1e-9


@pytest.mark.integration
def test_att_causal_config_selection(hardware):
    """Config selection still works for causal problems."""
    configs = create_attention_config_list(hardware, "f16")
    assert len(configs) > 0

    problem = create_attention_problem(4096, 4096, 128, q_heads=32, batch=1, dtype="f16",
                                       causal=True)
    result = origami.select_config(problem, hardware, configs, origami.model_t.attention)
    assert result.latency > 0
    assert result.config is not None
    assert result.config.mt.m > 0
