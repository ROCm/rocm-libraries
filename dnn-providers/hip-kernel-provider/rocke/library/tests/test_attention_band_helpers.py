# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for the shared attention band helpers (no GPU needed).

``kernels/common/_attention_band.py`` names the band in kernel names, validates
``right_bound`` for the tiled and dense specs, and emits the right-bound compares
of the bodies. Its contract: the causal band (``right_bound == 0``) emits nothing
beyond the causal IR, so a default spec builds the IR it built before the band.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from kernels.common import _attention_band as band
from rocke.core.ir import I32, IRBuilder


def _builder(name):
    b = IRBuilder(name)
    return b, [b.param(n, I32) for n in ("x", "y", "z")]


def _new_ops(b, fn):
    """``fn()``'s result and the names of the ops it emitted."""
    before = len(b.kernel.body.ops)
    out = fn()
    return out, [op.name for op in b.kernel.body.ops[before:]]


# ---------------------------------------------------------------------------
# Kernel-name parts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("right_bound,tag", [(0, ""), (-1, "rbu"), (16, "rb16")])
def test_right_bound_tag(right_bound, tag):
    assert band.right_bound_tag(right_bound) == tag


@pytest.mark.parametrize(
    "top_left,right_bound,parts",
    [
        (False, 0, ("", "")),
        (True, 0, ("tl", "")),
        (False, -1, ("", "rbu")),
        (True, 16, ("tl", "rb16")),
    ],
)
def test_tiled_band_name_parts(top_left, right_bound, parts):
    assert band.tiled_band_name_parts(top_left, right_bound) == parts


@pytest.mark.parametrize("right_bound,parts", [(0, ()), (7, ("rb7",))])
def test_dense_band_name_parts(right_bound, parts):
    """Dense names only a lookahead; ``causal=False`` names the unbounded side."""
    assert band.dense_band_name_parts(right_bound) == parts


# ---------------------------------------------------------------------------
# right_bound validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "right_bound,kw", [(0, {}), (-1, {}), (16, {}), (0, {"use_alibi": True})]
)
def test_tiled_right_bound_accepted(right_bound, kw):
    band.check_right_bound(right_bound, **kw)


@pytest.mark.parametrize(
    "right_bound,kw,message",
    [
        (-2, {}, "right_bound must be >= -1 (-1 = unbounded), got -2"),
        (16, {"use_alibi": True}, "incompatible with use_alibi / use_qq_bias"),
        (-1, {"use_qq_bias": True}, "incompatible with use_alibi / use_qq_bias"),
    ],
)
def test_tiled_right_bound_rejected(right_bound, kw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        band.check_right_bound(right_bound, **kw)


def _dense(right_bound, **kw):
    fields = dict(
        right_bound=right_bound,
        causal=True,
        causal_bottom_right=False,
        paged=False,
        varlen=False,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "spec",
    [
        _dense(0),
        _dense(0, causal=False),
        _dense(0, causal_bottom_right=True),
        _dense(8),
    ],
)
def test_dense_right_bound_accepted(spec):
    band.check_dense_right_bound(spec)


@pytest.mark.parametrize(
    "spec,message",
    [
        (_dense(-1), "right_bound must be >= 0"),
        (_dense(8, causal=False), "right_bound > 0 requires causal=True"),
        (
            _dense(8, causal_bottom_right=True),
            "right_bound > 0 is not supported with causal_bottom_right",
        ),
        (_dense(8, paged=True), "right_bound > 0 is not supported with paged / varlen"),
        (
            _dense(8, varlen=True),
            "right_bound > 0 is not supported with paged / varlen",
        ),
    ],
)
def test_dense_right_bound_rejected(spec, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        band.check_dense_right_bound(spec)


# ---------------------------------------------------------------------------
# IR emission: the causal band adds no op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("right_bound", [0, -1])
def test_upper_lim_without_lookahead_emits_nothing(right_bound):
    b, (x, _, _) = _builder("band_upper_lim")
    out, ops = _new_ops(b, lambda: band.upper_lim(b, x, right_bound))
    assert out is x and ops == []


def test_upper_lim_lookahead_adds_one_constant_and_one_add():
    b, (x, _, _) = _builder("band_upper_lim_rb")
    _, ops = _new_ops(b, lambda: band.upper_lim(b, x, 5))
    assert len(ops) == 2


def test_diagonal_offset():
    b, (kv_len, q_len, _) = _builder("band_diagonal")
    _, top_left = _new_ops(b, lambda: band.diagonal_offset(b, True, kv_len, q_len))
    _, bottom_right = _new_ops(b, lambda: band.diagonal_offset(b, False, kv_len, q_len))
    assert len(top_left) == 1 and len(bottom_right) == 1
    assert top_left != bottom_right


def test_causal_band_mask_emits_only_the_causal_ops():
    """Each ``BandMask`` helper at ``right_bound == 0`` emits exactly what the
    causal body did; a lookahead adds one constant and one add per limit."""
    for rb, extra in ((0, 0), (5, 2)):
        b, (x, y, z) = _builder(f"band_mask_rb{rb}")
        m = band.BandMask(b, rb)
        _, ops = _new_ops(b, lambda: m.upper_lim(x))
        assert len(ops) == extra
        upper, ops = _new_ops(b, lambda: m.upper_ok(x, y))
        assert len(ops) == 1 + extra
        _, ops = _new_ops(b, lambda: m.row_mask(x, upper, z))
        assert len(ops) == 2
        _, ops = _new_ops(b, lambda: m.last_key(x, y, z))
        assert len(ops) == 1 + extra
        _, ops = _new_ops(b, lambda: m.valid_tail(x, y))
        assert len(ops) == 2 + extra


def test_unbounded_band_mask_skips_the_right_compare():
    b, (x, y, z) = _builder("band_mask_rbu")
    m = band.BandMask(b, -1)
    upper, ops = _new_ops(b, lambda: m.upper_ok(x, y))
    assert upper is None and ops == []
    _, ops = _new_ops(b, lambda: m.row_mask(x, upper, z))
    assert len(ops) == 1
    out, ops = _new_ops(b, lambda: m.max_prefix_len(x, y, 3, z))
    assert out is z and ops == []
    out, ops = _new_ops(b, lambda: m.valid_tail(x, y))
    assert out is y and ops == []
