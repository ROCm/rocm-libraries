# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Differential table of the wgrad split-K degree the implicit-GEMM conv
# benchmark launches (benchmarks/common/benchmark_implicit_gemm_conv.py: the
# JIT atomic and two-stage legs and the --run-from-cache degree list), before
# and after it started resolving for the CU count of the timed device. Host-only.
#
# Three columns per row:
#   develop  -- the rule before this change: no CU count (the arch default) and
#               the CK formula for gfx942 wgrad (no conv-local cost model).
#               Computed here by clearing _WGRAD_COST_MODELS; over the full
#               domain below widened to tile_m / tile_n 16..256, tile_k 16..64,
#               block sizes 64..1024, bf16 / fp16 / fp32 and 54 problems
#               (284,310 rows), this matches the resolver at develop
#               (bc1cac006f) row for row.
#   default  -- the resolver now, for the arch's default part (num_cus=None):
#               what builders and dispatch resolve.
#   device   -- the resolver now, for the timed device's CU count: what the
#               benchmark resolves.
#
# A degree > 1 is not baked into the wgrad binary (its name carries only the
# split-K marker and the degree is a kernarg), so a diverging row changes the
# launch -- gridDim.z = groups * degree, and for the two-stage leg the
# workspace slabs -- and the degree the sweep times, not the binary or its
# cache key.

from __future__ import annotations

import collections
import itertools

import pytest
from kernels.common import conv_split_k as csk
from kernels.common.conv_implicit_gemm import ConvProblem

# (N, Di, Hi, Wi, C, K, Z, Y, X, pD, pH, pW, groups, stride): the wgrad cases of
# the gfx942 3-D UNet pin table (halo and padded forms) and 2-D backward cases
# of benchmarks/common/grouped_conv/bench_cases_conv_bwd.json (dense, grouped,
# depthwise).
_UNET3D = [
    (64, 64, 258, 130),
    (128, 64, 258, 130),
    (256, 128, 130, 66),
    (128, 128, 130, 66),
    (512, 256, 66, 34),
    (256, 256, 66, 34),
    (512, 512, 34, 18),
    (1024, 512, 34, 18),
    (64, 128, 130, 66),
    (1024, 1024, 18, 10),
]
PROBLEMS = (
    [(1, d, h, h, c, k, 3, 3, 3, 0, 0, 0, 1, 1) for c, k, d, h in _UNET3D]
    + [(1, d - 2, h - 2, h - 2, c, k, 3, 3, 3, 1, 1, 1, 1, 1) for c, k, d, h in _UNET3D]
    + [
        (256, None, 56, 56, 64, 64, None, 3, 3, None, 1, 1, 1, 1),
        (64, None, 56, 56, 256, 256, None, 3, 3, None, 1, 1, 32, 1),
        (128, None, 56, 56, 144, 144, None, 3, 3, None, 1, 1, 144, 1),
        (1, None, 256, 256, 512, 512, None, 3, 3, None, 1, 1, 1, 1),
    ]
)
# (arch, CU count of the timed device): the two default parts and a gfx942
# part with fewer CUs.
DEVICES = (("gfx942", 304), ("gfx942", 228), ("gfx950", 256))
TILES = tuple(itertools.product((64, 128, 256), (64, 128, 256), (32,)))
BLOCK_SIZE = 256  # 2 x 2 waves of 64
DTYPES = ("bf16", "fp32")
# The benchmark's three consumers: the JIT atomic leg (may stay unsplit), a
# cached split-K atomic binary and the two-stage leg (both need a split).
LEGS = (("atomic", True), ("atomic", False), ("workspace", False))


def _problem(t) -> ConvProblem:
    N, Di, Hi, Wi, C, K, Z, Y, X, pD, pH, pW, g, s = t
    depth = {} if Di is None else {"Di": Di, "Z": Z, "sD": 1, "pD": pD, "dD": 1}
    return ConvProblem(
        N=N,
        Hi=Hi,
        Wi=Wi,
        C=C,
        K=K,
        Y=Y,
        X=X,
        sH=s,
        sW=s,
        pH=pH,
        pW=pW,
        groups=g,
        **depth,
    )


def _degree(p, arch, dtype, tile, leg, num_cus) -> int:
    reduction, allow_unsplit = leg
    try:
        return csk.resolve_conv_split_k(
            "wgrad",
            p,
            arch=arch,
            dtype=dtype,
            tile_m=tile[0],
            tile_n=tile[1],
            tile_k=tile[2],
            block_size=BLOCK_SIZE,
            split_k=-1,
            reduction=reduction,
            allow_unsplit=allow_unsplit,
            num_cus=num_cus,
        ).split_k
    except ValueError:
        return 0  # the leg cannot serve the problem: the benchmark skips it


def _keys():
    for t, (arch, cus), dtype, tile, leg in itertools.product(
        PROBLEMS, DEVICES, DTYPES, TILES, LEGS
    ):
        yield (arch, cus, dtype, leg, tile, t)


@pytest.fixture(scope="module")
def table():
    probs = {t: _problem(t) for t in PROBLEMS}
    rows = {}
    csk._resolve_cached.cache_clear()
    for key in _keys():
        arch, cus, dtype, leg, tile, t = key
        p = probs[t]
        rows[key] = [
            None,
            _degree(p, arch, dtype, tile, leg, None),
            _degree(p, arch, dtype, tile, leg, cus),
        ]
    saved = dict(csk._WGRAD_COST_MODELS)
    csk._WGRAD_COST_MODELS.clear()
    csk._resolve_cached.cache_clear()
    try:
        for (arch, _cus, dtype, leg, tile, t), row in rows.items():
            row[0] = _degree(probs[t], arch, dtype, tile, leg, None)
    finally:
        csk._WGRAD_COST_MODELS.update(saved)
        csk._resolve_cached.cache_clear()
    return rows


def _diverging(table, a, b):
    c = collections.Counter()
    for (arch, cus, _dtype, leg, _tile, _t), row in table.items():
        if row[a] != row[b]:
            c[(arch, cus, leg[0], leg[1])] += 1
    return dict(c)


DEVELOP, DEFAULT, DEVICE = 0, 1, 2
# Rows per (device, leg): 24 problems x 2 dtypes x 9 tiles = 432.
ROWS_PER_DEVICE_LEG = len(PROBLEMS) * len(DTYPES) * len(TILES)

# Diverging rows per (arch, device CUs, reduction, allow_unsplit), out of
# ROWS_PER_DEVICE_LEG each. Any (device, leg) not listed has none.
DEVICE_DIVERGING = {  # default vs device: the benchmark's new CU-count input
    ("gfx942", 228, "atomic", True): 146,
    ("gfx942", 228, "atomic", False): 160,
    ("gfx942", 228, "workspace", False): 162,
}
MODEL_DIVERGING = {  # develop vs default: the gfx942 wgrad cost model
    ("gfx942", 304, "atomic", True): 323,
    ("gfx942", 304, "atomic", False): 415,
    ("gfx942", 304, "workspace", False): 430,
    ("gfx942", 228, "atomic", True): 323,
    ("gfx942", 228, "atomic", False): 415,
    ("gfx942", 228, "workspace", False): 430,
}

_S2 = PROBLEMS[1]  # 3-D UNet shape 2, halo form
_S3 = PROBLEMS[2]  # 3-D UNet shape 3, halo form
_R50 = PROBLEMS[20]  # 2-D dense 3x3, N=256, 56x56, 64 -> 64
_T = (128, 128, 32)
# (key, develop, default, device) of representative rows.
PINNED = [
    (("gfx942", 228, "bf16", ("atomic", True), _T, _S2), 22, 90, 64),
    (("gfx942", 228, "bf16", ("workspace", False), _T, _S2), 22, 256, 512),
    (("gfx942", 228, "bf16", ("atomic", True), _T, _R50), 121, 121, 128),
    (("gfx942", 304, "bf16", ("workspace", False), _T, _S3), 11, 128, 128),
    (("gfx950", 256, "bf16", ("atomic", True), _T, _S3), 32, 32, 32),
]


def test_device_cu_count_changes_nothing_on_the_default_parts(table):
    # The benchmark's new input is the timed device's CU count: on a part with
    # the arch's default count it resolves what builders and dispatch do.
    for (arch, cus, *_), row in table.items():
        if cus == csk.arch_num_cus(arch):
            assert row[DEVICE] == row[DEFAULT]


def test_device_cu_count_diverging_rows(table):
    # Only the gfx942 part with fewer CUs diverges from the arch default.
    assert _diverging(table, DEFAULT, DEVICE) == DEVICE_DIVERGING


def test_develop_rule_diverging_rows(table):
    # gfx950 resolves as before. gfx942 wgrad moved from the CK formula to the
    # provisional conv-local cost model, on every part.
    assert _diverging(table, DEVELOP, DEFAULT) == MODEL_DIVERGING


@pytest.mark.parametrize("key, develop, default, device", PINNED)
def test_pinned_rows(table, key, develop, default, device):
    assert table[key] == [develop, default, device]
