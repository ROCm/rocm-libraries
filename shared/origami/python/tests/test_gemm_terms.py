# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the GEMM admission rules and the tile-term breakdown."""

import pytest

import origami
from helpers import HARDWARE


def make_problem(m, n, k):
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = 1
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    dtype = origami.data_type_t.BFloat16
    problem.a_dtype = problem.b_dtype = problem.c_dtype = problem.d_dtype = dtype
    problem.mi_dtype = dtype
    return problem


def make_config(hardware):
    mi = hardware.get_recommended_matrix_instruction(origami.data_type_t.BFloat16)
    config = origami.config_t()
    config.mt = origami.dim3_t(128, 128, 64)
    config.mi = origami.dim3_t(mi.m, mi.n, mi.k)
    config.occupancy = 1
    return config


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_admitted_config_has_finite_latency(arch):
    hardware = HARDWARE[arch]
    problem = make_problem(4096, 4096, 4096)
    config = make_config(hardware)

    assert origami.admit(problem, hardware, config) == origami.rejection_t.none
    assert origami.compute_total_latency(problem, hardware, config) < 1e300


def test_rejection_names_the_rule():
    hardware = HARDWARE["gfx950"]
    problem = make_problem(4096, 4096, 4096)

    invalid = make_config(hardware)
    invalid.occupancy = 0
    assert origami.admit(problem, hardware, invalid) == origami.rejection_t.invalid_config

    # K = 1 is not a whole number of cache lines, so a non-temporal config is turned away.
    non_temporal = make_config(hardware)
    non_temporal.cache_hints_a = 4
    k1 = make_problem(4096, 4096, 1)
    assert origami.admit(k1, hardware, non_temporal) == origami.rejection_t.non_temporal_misaligned
    assert origami.compute_total_latency(k1, hardware, non_temporal) > 1e300


def test_non_temporal_rules_are_off_on_cdna5():
    hardware = HARDWARE["gfx1250"]
    config = make_config(hardware)
    config.cache_hints_a = 4

    assert origami.admit(make_problem(4096, 4096, 1), hardware, config) == origami.rejection_t.none


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1201", "gfx1250"])
def test_tile_terms_add_up(arch):
    hardware = HARDWARE[arch]
    problem = make_problem(4096, 4096, 4096)
    config = make_config(hardware)
    context = origami.context_t(problem, hardware, config)

    terms = origami.compute_tile_terms(problem, hardware, config, context)

    assert terms.L_tile_total == origami.compute_tile_latency(problem, hardware, config, context)
    mainloop = (
        terms.L_main
        + terms.L_ngll
        + terms.L_nll
        + terms.L_tail
        + terms.L_pgr_stall
        + terms.L_loop_overhead
        + terms.L_single_lds_buffer
        + terms.L_du_waste
        + terms.L_cluster_sync
    )
    assert terms.L_mainloop == mainloop
    assert terms.memory.L_mem > 0.0
