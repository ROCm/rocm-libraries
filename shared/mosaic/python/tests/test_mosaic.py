# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Smoke tests for the mosaic Python bindings."""

import math

import mosaic


def _make_hardware():
    hw = mosaic.Hardware()
    hw.N_CU = 256
    hw.lds_capacity = 65536
    hw.L2_capacity = 4194304
    hw.parallel_mi_cu = 1
    hw.mem_bw_per_wg_coefficients = (0.0, 0.008, 0.0)
    return hw


def _make_problem():
    p = mosaic.Problem()
    p.size = mosaic.Dim3()
    p.size.m = 8192
    p.size.n = 8192
    p.size.k = 8192
    p.batch = 1
    p.a_transpose = mosaic.Transpose.T
    p.b_transpose = mosaic.Transpose.N
    p.a_dtype = mosaic.DataType.BFloat16
    p.b_dtype = mosaic.DataType.BFloat16
    p.c_dtype = mosaic.DataType.BFloat16
    p.d_dtype = mosaic.DataType.BFloat16
    p.mi_dtype = mosaic.DataType.BFloat16
    return p


def _make_configs():
    configs = []
    for i in range(4):
        c = mosaic.Config()
        c.mt = mosaic.Dim3()
        c.mt.m = 128 + i * 32
        c.mt.n = 128 + i * 16
        c.mt.k = 64
        c.mi = mosaic.Dim3()
        c.mi.m = 16
        c.mi.n = 16
        c.mi.k = 32
        c.occupancy = 1 + i
        c.index = 1000 + i
        configs.append(c)
    return configs


def test_enums_exist():
    assert int(mosaic.DataType.BFloat16) >= 0
    assert mosaic.Transpose.T != mosaic.Transpose.N


def test_dim3_helpers():
    d = mosaic.Dim3()
    d.m, d.n, d.k = 4, 8, 16
    assert d.mn() == 32
    assert d.mk() == 64
    assert d.nk() == 128


def test_hardware_tuple_roundtrip():
    hw = _make_hardware()
    assert tuple(hw.mem_bw_per_wg_coefficients) == (0.0, 0.008, 0.0)


def test_route_returns_int():
    p = _make_problem()
    assert mosaic.route(p) >= -1


def test_rank_configs_contract():
    p = _make_problem()
    hw = _make_hardware()
    configs = _make_configs()

    results = mosaic.rank_configs(p, hw, configs)

    for res in (results,):
        assert len(res) == len(configs)
        # Every input config index is covered exactly once.
        assert sorted(r.config_index for r in res) == list(range(len(configs)))

        survivors = [r for r in res if r.scored]
        filtered = [r for r in res if not r.scored]
        # Survivors precede filtered-out entries.
        assert res[: len(survivors)] == survivors or all(
            r.scored for r in res[: len(survivors)]
        )
        # Survivor scores are finite and non-increasing.
        for a, b in zip(survivors, survivors[1:]):
            assert math.isfinite(a.score)
            assert a.score >= b.score
        for r in filtered:
            assert not r.scored
