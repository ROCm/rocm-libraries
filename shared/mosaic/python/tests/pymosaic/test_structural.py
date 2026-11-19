# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the structural accessors on ``pymosaic.Layout``.

These exercise the pure-Python tuple-algebra layer (``shape_at``,
``stride_at``, ``size_at``, ``get``, ``flatten``, ``take``, ``select``,
``group``, ``append``, ``prepend``) and their CuTe-style free-function
aliases. They only inspect the canonical ``(shape, stride)`` tuples — no
layout evaluation — so no g++/ninja toolchain is required.
"""

from __future__ import annotations

import pytest

import pymosaic as pm
from pymosaic import Layout


# ---------------------------------------------------------------------------
# Per-mode shape / stride / size accessors
# ---------------------------------------------------------------------------


class TestModeAccessors:
    def test_shape_stride_at_flat(self):
        L = Layout.row_major(4, 8)  # (4,8):(8,1)
        assert L.shape_at(0) == 4
        assert L.shape_at(1) == 8
        assert L.stride_at(0) == 8
        assert L.stride_at(1) == 1

    def test_shape_stride_at_nested(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        assert L.shape_at(0) == (2, 2)
        assert L.stride_at(0) == (1, 2)
        assert L.shape_at(1) == 3
        assert L.stride_at(1) == 3

    def test_size_at(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        assert L.size_at(0) == 4  # 2*2
        assert L.size_at(1) == 3

    def test_negative_index(self):
        L = Layout.row_major(4, 8)
        assert L.shape_at(-1) == 8
        assert L.stride_at(-2) == 8

    def test_out_of_range_raises(self):
        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError):
            L.shape_at(2)
        with pytest.raises(IndexError):
            L.shape_at(-3)

    def test_bool_index_rejected(self):
        L = Layout.row_major(4, 8)
        with pytest.raises(TypeError):
            L.shape_at(True)


# ---------------------------------------------------------------------------
# get — mode as its own Layout
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_flat_mode_is_rank1(self):
        L = Layout.row_major(4, 8)  # (4,8):(8,1)
        m0 = L.get(0)
        assert m0.shape == (4,)
        assert m0.stride == (8,)
        m1 = L.get(1)
        assert m1.shape == (8,)
        assert m1.stride == (1,)

    def test_get_nested_mode_keeps_substructure(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        m0 = L.get(0)
        assert m0.shape == (2, 2)
        assert m0.stride == (1, 2)

    def test_get_returns_layout(self):
        L = Layout.row_major(4, 8)
        assert isinstance(L.get(0), Layout)


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flatten_nested(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        F = L.flatten()
        assert F.shape == (2, 2, 3)
        assert F.stride == (1, 2, 3)
        assert F.depth == 1

    def test_flatten_already_flat_is_noop(self):
        L = Layout.row_major(4, 8)
        F = L.flatten()
        assert F.shape == (4, 8)
        assert F.stride == (8, 1)


# ---------------------------------------------------------------------------
# take — sub-layout over [i, j)
# ---------------------------------------------------------------------------


class TestTake:
    def test_take_subrange(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        T = L.take(0, 2)
        assert T.shape == (2, 3)
        assert T.stride == (1, 2)

    def test_take_single_mode_keeps_tuple(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        T = L.take(0, 1)
        assert T.shape == ((2, 2),)
        assert T.stride == ((1, 2),)

    def test_take_full_range(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        T = L.take(0, 3)
        assert T.shape == (2, 3, 4)

    def test_take_empty_range_raises(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        with pytest.raises(IndexError):
            L.take(1, 1)


# ---------------------------------------------------------------------------
# select — pick / reorder modes
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_reorder(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        S = L.select(2, 0)
        assert S.shape == (4, 2)
        assert S.stride == (6, 1)

    def test_select_repeat(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        S = L.select(1, 1)
        assert S.shape == (3, 3)
        assert S.stride == (2, 2)

    def test_select_empty_raises(self):
        L = Layout(shape=(2, 3), stride=(1, 2))
        with pytest.raises(ValueError):
            L.select()


# ---------------------------------------------------------------------------
# group — nest [i, j) into one mode
# ---------------------------------------------------------------------------


class TestGroup:
    def test_group_prefix(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        G = L.group(0, 2)
        assert G.shape == ((2, 3), 4)
        assert G.stride == ((1, 2), 6)
        assert G.rank == 2

    def test_group_suffix(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        G = L.group(1, 3)
        assert G.shape == (2, (3, 4))
        assert G.stride == (1, (2, 6))

    def test_group_empty_range_raises(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        with pytest.raises(IndexError):
            L.group(2, 2)


# ---------------------------------------------------------------------------
# append / prepend
# ---------------------------------------------------------------------------


class TestAppendPrepend:
    def test_append_rank1_mode(self):
        L = Layout.row_major(4, 8)  # (4,8):(8,1)
        m = Layout(shape=(2,), stride=(1,))
        A = L.append(m)
        assert A.shape == (4, 8, (2,))
        assert A.stride == (8, 1, (1,))

    def test_append_scalar_mode(self):
        L = Layout.row_major(4, 8)
        m = Layout(shape=2, stride=1)
        A = L.append(m)
        assert A.shape == (4, 8, 2)
        assert A.stride == (8, 1, 1)

    def test_prepend(self):
        L = Layout.row_major(4, 8)
        m = Layout(shape=2, stride=1)
        P = L.prepend(m)
        assert P.shape == (2, 4, 8)
        assert P.stride == (1, 8, 1)

    def test_append_non_layout_raises(self):
        L = Layout.row_major(4, 8)
        with pytest.raises(TypeError):
            L.append((2,))


# ---------------------------------------------------------------------------
# Free-function aliases mirror the methods
# ---------------------------------------------------------------------------


class TestFreeFunctions:
    def test_shape_stride_rank_depth(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        assert pm.shape(L) == ((2, 2), 3)
        assert pm.stride(L) == ((1, 2), 3)
        assert pm.rank(L) == 2
        assert pm.depth(L) == 2

    def test_size_total_and_per_mode(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        assert pm.size(L) == 12  # 2*2*3
        assert pm.size(L, 0) == 4
        assert pm.size(L, 1) == 3

    def test_accessor_aliases_match_methods(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        assert pm.shape_at(L, 1) == L.shape_at(1)
        assert pm.stride_at(L, 1) == L.stride_at(1)
        assert pm.size_at(L, 1) == L.size_at(1)
        assert pm.get(L, 0) == L.get(0)
        assert pm.flatten(L) == L.flatten()
        assert pm.take(L, 0, 2) == L.take(0, 2)
        assert pm.select(L, 2, 0) == L.select(2, 0)
        assert pm.group(L, 0, 2) == L.group(0, 2)

    def test_free_fn_rejects_non_layout(self):
        with pytest.raises(TypeError):
            pm.flatten((2, 3))


# ---------------------------------------------------------------------------
# Round-trip / consistency invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_group_then_flatten_roundtrip(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        assert L.group(0, 2).flatten() == L

    def test_take_matches_select_range(self):
        L = Layout(shape=(2, 3, 4), stride=(1, 2, 6))
        assert L.take(0, 2) == L.select(0, 1)

    def test_get_matches_shape_at(self):
        L = Layout(shape=((2, 2), 3), stride=((1, 2), 3))
        assert L.get(0).shape == L.shape_at(0)
