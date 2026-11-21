# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmark_implicit_gemm_conv._worker_pool: a CU count the parent resolved is
# seeded into the build workers, so a worker resolves auto split-K degrees for
# the same device as the parent without querying a device itself. Host-only.

from __future__ import annotations

import pytest

from benchmarks.common import benchmark_implicit_gemm_conv as B


@pytest.fixture
def cus(monkeypatch):
    saved = list(B._DEVICE_NUM_CUS)
    yield
    B._DEVICE_NUM_CUS[:] = saved


@pytest.mark.parametrize("value", [123, None])
def test_workers_take_the_parent_value(cus, value):
    B._DEVICE_NUM_CUS[:] = [value]
    with B._worker_pool(1) as pool:
        assert pool.submit(B._device_num_cus).result(timeout=120) == value


def test_seed_sets_the_cache(cus):
    B._DEVICE_NUM_CUS[:] = []
    B._seed_device_num_cus(77)
    assert B._device_num_cus() == 77
