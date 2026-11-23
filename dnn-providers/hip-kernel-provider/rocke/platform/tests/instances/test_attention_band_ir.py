# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for the runtime attention band helper (emission and validation).

The numeric behaviour of the band (keep rule, row validity, K tile range) is
exercised on device by ``test_attention_fwd_ext_numeric.py``.
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import F32, I32, IRBuilder
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.attention_band import AttnRuntimeBounds, BandEmitter


@pytest.mark.parametrize("field", ["left_bound", "right_bound"])
@pytest.mark.parametrize("value", [-2, -100])
def test_static_bound_below_minus_one_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        AttnRuntimeBounds(8, 8, **{field: value})


@pytest.mark.parametrize("value", [-1, 0, 3])
def test_static_bounds_accepted(value):
    AttnRuntimeBounds(8, 8, left_bound=value, right_bound=value)


def _builder(name="band"):
    b = IRBuilder(name)
    sq = b.param("sq", I32)
    sk = b.param("sk", I32)
    q = b.param("q", I32)
    k = b.param("k", I32)
    out = b.param("out", F32)
    return b, sq, sk, q, k, out


def _ops(bounds_kwargs):
    b2, sq, sk, q, k, _ = _builder()
    band = BandEmitter(b2, AttnRuntimeBounds(sq, sk, **bounds_kwargs))
    rb = band.row_band(q)
    keep = band.cell_keep(rb, k, band.col_in(k))
    return b2, rb, keep


def test_unbounded_static_band_has_no_diagonal_arithmetic():
    _, rb, _ = _ops({})
    assert rb.lo is None and rb.hi is None


def test_static_one_sided_band_sets_only_that_side():
    _, rb, _ = _ops(dict(right_bound=0))
    assert rb.hi is not None and rb.lo is None
    _, rb, _ = _ops(dict(left_bound=4))
    assert rb.lo is not None and rb.hi is None


def test_runtime_bounds_emit_a_negative_means_unbounded_select():
    b = IRBuilder("band_rt")
    sq = b.param("sq", I32)
    sk = b.param("sk", I32)
    left = b.param("left", I32)
    right = b.param("right", I32)
    q = b.param("q", I32)
    band = BandEmitter(b, AttnRuntimeBounds(sq, sk, 0, left, right))
    rb = band.row_band(q)
    assert rb.lo is not None and rb.hi is not None
    b.ret()
    ir = lower(b.kernel, arch="gfx942", llvm_flavor="llvm22")
    assert len(re.findall(r"\bselect\b", ir)) >= 2


@pytest.mark.parametrize("static_bounds", [True, False])
def test_k_tile_range_is_a_pair_of_i32_values(static_bounds):
    b = IRBuilder("band_range")
    sq = b.param("sq", I32)
    sk = b.param("sk", I32)
    q = b.param("q", I32)
    kwargs = dict(left_bound=5, right_bound=2)
    if not static_bounds:
        kwargs = dict(left_bound=b.param("l", I32), right_bound=b.param("r", I32))
    band = BandEmitter(b, AttnRuntimeBounds(sq, sk, 0, **kwargs))
    start, stop = band.k_tile_range(q, 16, 16)
    assert start.type.name == "i32" and stop.type.name == "i32"
    count = band.k_tile_count_ceil(16)
    assert count.type.name == "i32"
    b.ret()
    lower(b.kernel, arch="gfx942", llvm_flavor="llvm22")


def test_diag_offset_value_is_added_to_the_row():
    b = IRBuilder("band_diag")
    sq = b.param("sq", I32)
    sk = b.param("sk", I32)
    off = b.param("off", I32)
    q = b.param("q", I32)
    k = b.param("k", I32)
    band = BandEmitter(b, AttnRuntimeBounds(sq, sk, off, right_bound=0))
    rb = band.row_band(q)
    keep = band.cell_keep(rb, k, band.col_in(k))
    assert keep.type.name == "i1"
