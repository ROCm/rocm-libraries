# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the gfx1151 tiny-N StaggerU policy."""

import origami


def make_case(n=16, k=4096):
    hardware = origami.get_hardware_for_arch(
        origami.architecture_t.gfx1151,
        40,
        64 * 1024,
        512 * 1024,
        2 * 1024 * 1024,
        2900000,
    )
    problem = origami.problem_t()
    problem.size = origami.dim3_t(4096, n, k)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype("f16")
    problem.b_dtype = origami.string_to_datatype("f16")
    problem.mi_dtype = origami.string_to_datatype("f32")

    config = origami.config_t()
    config.mt = origami.dim3_t(64, 16, 32)
    tensile = config.tensile()
    tensile.depth_u = 32
    tensile.prefetch_global_read = 2
    tensile.prefetch_local_read = 0
    tensile.stream_k = 3
    grid = (problem.size.m + config.mt.m - 1) // config.mt.m
    grid *= (problem.size.n + config.mt.n - 1) // config.mt.n
    return problem, hardware, config, grid


def test_gfx1151_tiny_n_stagger_policy():
    problem, hardware, config, grid = make_case()
    mapping = origami.select_workgroup_mapping(problem, hardware, config, grid)
    assert (mapping.wgm, mapping.wgmxcc, mapping.wgmxccchunk) == (8, 1, 0)
    result = origami.select_staggerU(problem, hardware, config, grid, mapping.wgm)
    assert (result.staggerUMapping, result.staggerU, result.staggerUStrideShift) == (0, 16, 2)


def test_gfx1151_tiny_n_stagger_policy_boundaries_fall_back():
    problem, hardware, config, grid = make_case(n=129)
    result = origami.select_staggerU(problem, hardware, config, grid, 8)
    assert (result.staggerUMapping, result.staggerU, result.staggerUStrideShift) != (0, 16, 2)

    problem, hardware, config, grid = make_case()
    problem.a_transpose = origami.transpose_t.N
    result = origami.select_staggerU(problem, hardware, config, grid, 8)
    assert (result.staggerUMapping, result.staggerU, result.staggerUStrideShift) != (0, 16, 2)

    problem, hardware, config, grid = make_case(k=1024)
    result = origami.select_staggerU(problem, hardware, config, grid, 8)
    assert (result.staggerUMapping, result.staggerU, result.staggerUStrideShift) != (0, 16, 2)
