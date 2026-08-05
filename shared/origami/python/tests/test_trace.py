# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Calibration trace parity tests."""

import origami


def test_trace_reuses_production_latency_chain():
    problem = origami.problem_t()
    problem.size = origami.dim3_t(4096, 4096, 4096)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = problem.b_dtype = origami.string_to_datatype("f16")
    problem.c_dtype = problem.d_dtype = origami.string_to_datatype("f16")
    problem.mi_dtype = origami.string_to_datatype("f32")
    hardware = origami.get_hardware_for_arch(
        origami.architecture_t.gfx1151, 40, 64 * 1024, 512 * 1024, 2 * 1024 * 1024, 2900000
    )
    config = origami.config_t()
    config.mt = origami.dim3_t(128, 128, 32)
    config.mi = origami.dim3_t(16, 16, 16)
    config.occupancy = 1
    config.workgroup_mapping = 8

    trace = origami.trace_total_latency(problem, hardware, config)
    assert trace.accepted
    assert trace.total_latency == origami.compute_total_latency(problem, hardware, config)
    assert trace.timestep_latency == trace.tile_latency
    assert trace.total_latency == (
        trace.timestep_latency * trace.context.num_timesteps + trace.parallel_reduction_latency
    )
    assert trace.context.k_per_split > 0
    assert trace.context.k_iters > 0
    assert len(trace.cache_hit_rates) == 6
