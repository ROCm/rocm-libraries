# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Smoke tests for the tilewright Python bindings."""

import math

import tilewright


def _make_hardware():
    hw = tilewright.Hardware()
    hw.N_CU = 256
    hw.lds_capacity = 65536
    hw.L2_capacity = 4194304
    hw.parallel_mi_cu = 1
    hw.mem_bw_per_wg_coefficients = (0.0, 0.008, 0.0)
    return hw


def _make_problem():
    p = tilewright.Problem()
    p.size = tilewright.Dim3()
    p.size.m = 8192
    p.size.n = 8192
    p.size.k = 8192
    p.batch = 1
    p.a_transpose = tilewright.Transpose.T
    p.b_transpose = tilewright.Transpose.N
    p.a_dtype = tilewright.DataType.BFloat16
    p.b_dtype = tilewright.DataType.BFloat16
    p.c_dtype = tilewright.DataType.BFloat16
    p.d_dtype = tilewright.DataType.BFloat16
    p.mi_dtype = tilewright.DataType.BFloat16
    return p


def _make_configs():
    configs = []
    for i in range(4):
        c = tilewright.Config()
        c.mt = tilewright.Dim3()
        c.mt.m = 128 + i * 32
        c.mt.n = 128 + i * 16
        c.mt.k = 64
        c.mi = tilewright.Dim3()
        c.mi.m = 16
        c.mi.n = 16
        c.mi.k = 32
        c.occupancy = 1 + i
        c.index = 1000 + i
        configs.append(c)
    return configs


def test_enums_exist():
    assert int(tilewright.DataType.BFloat16) >= 0
    assert tilewright.Transpose.T != tilewright.Transpose.N


def test_dim3_helpers():
    d = tilewright.Dim3()
    d.m, d.n, d.k = 4, 8, 16
    assert d.mn() == 32
    assert d.mk() == 64
    assert d.nk() == 128


def test_hardware_tuple_roundtrip():
    hw = _make_hardware()
    assert tuple(hw.mem_bw_per_wg_coefficients) == (0.0, 0.008, 0.0)


def test_route_returns_int():
    p = _make_problem()
    assert tilewright.route(p) >= -1


def test_rank_configs_contract():
    p = _make_problem()
    hw = _make_hardware()
    configs = _make_configs()

    results = tilewright.rank_configs(p, hw, configs)

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
