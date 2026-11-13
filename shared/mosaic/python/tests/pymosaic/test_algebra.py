# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the pure-Python layout algebra: ``blocked_product``,
``complement``, and ``right_inverse``.

Rather than golden-pinning against the (deliberately simplified) compile-time
helpers in ``algebra.hpp``, these tests assert each operation's *defining
invariant* against the JIT-evaluated layout map — a strictly stronger check:

- ``blocked_product``: the tiled layout is a bijection onto ``[0, size)`` and
  each block is an exact, non-overlapping copy of the inner layout.
- ``complement``: ``L`` and ``complement(L, N)`` produce disjoint offsets that
  together tile ``[0, N)``.
- ``right_inverse``: ``L[R[i]] == i`` for every ``i`` in the domain.

Evaluating the map requires the C++ JIT toolchain (g++/ninja).
"""

from __future__ import annotations

import pytest

import pymosaic as pm
from pymosaic import Layout


def _image(L: Layout) -> list[int]:
    """All offsets the layout produces, in flat (column-major) index order."""
    return L.enumerate()


# ---------------------------------------------------------------------------
# blocked_product
# ---------------------------------------------------------------------------


class TestBlockedProduct:
    def test_shape_and_stride_formula(self):
        inner = Layout.row_major(2, 2)  # (2,2):(2,1), cosize 4
        outer = Layout.column_major(2, 2)  # (2,2):(1,2)
        B = inner.blocked_product(outer)
        # ((inner.shape[i], outer.shape[i]), ...) with outer scaled by cosize(inner)=4
        assert B.shape == ((2, 2), (2, 2))
        assert B.stride == ((2, 4), (1, 8))

    def test_is_bijection(self):
        inner = Layout.row_major(2, 2)
        outer = Layout.row_major(3, 3)
        B = inner.blocked_product(outer)
        img = _image(B)
        assert sorted(img) == list(range(len(B)))  # every offset hit exactly once

    def test_blocks_are_copies_of_inner(self):
        # A 2x2 row-major tile, tiled over a 2x2 grid. Each grid cell should hold
        # an exact copy of the inner tile, offset by cosize(inner) per step.
        inner = Layout.row_major(2, 2)
        outer = Layout.row_major(2, 2)
        B = inner.blocked_product(outer)
        inner_img = _image(inner)
        cos = inner.cosize
        # Outer coordinate k contributes outer_offset(k) * cosize(inner).
        # Outer coordinate contributes outer_offset(coord) * cosize(inner).
        for k in range(len(outer)):
            o0, o1 = outer.idx2crd(k)
            base = outer[k] * cos
            got = [
                B[((i0, o0), (i1, o1))]
                for (i0, i1) in (inner.idx2crd(i) for i in range(len(inner)))
            ]
            want = [base + v for v in inner_img]
            assert got == want

    def test_rank_mismatch_raises(self):
        inner = Layout.row_major(2, 2)
        outer = Layout(shape=(3,), stride=(1,))
        with pytest.raises(ValueError):
            inner.blocked_product(outer)

    def test_free_function_matches_method(self):
        inner = Layout.row_major(2, 3)
        outer = Layout.column_major(2, 2)
        assert (
            pm.blocked_product(inner, outer).shape == inner.blocked_product(outer).shape
        )
        assert (
            pm.blocked_product(inner, outer).stride
            == inner.blocked_product(outer).stride
        )


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------


def _tiles(L: Layout, C: Layout, n: int) -> bool:
    """True iff the Minkowski sum of the two images is exactly [0, n)."""
    sums = [a + b for a in _image(L) for b in _image(C)]
    return sorted(sums) == list(range(n)) and len(sums) == len(set(sums))


class TestComplement:
    @pytest.mark.parametrize(
        "layout, n",
        [
            (Layout(shape=(2,), stride=(1,)), 8),
            (Layout(shape=(4,), stride=(2,)), 8),
            (Layout(shape=(2,), stride=(3,)), 6),
            (Layout(shape=(2, 2), stride=(1, 4)), 16),
        ],
    )
    def test_tiles_codomain(self, layout, n):
        C = layout.complement(n)
        assert _tiles(layout, C, n)

    def test_known_results(self):
        assert Layout(shape=(2,), stride=(1,)).complement(8).shape == (4,)
        assert Layout(shape=(2,), stride=(1,)).complement(8).stride == (2,)
        assert Layout(shape=(4,), stride=(2,)).complement(8).shape == (2,)
        assert Layout(shape=(4,), stride=(2,)).complement(8).stride == (1,)

    def test_compact_layout_has_trivial_complement(self):
        # A compact layout already fills [0, cosize); its complement is a point.
        L = Layout.row_major(4, 8)  # bijection onto [0, 32)
        C = L.complement()  # default cosize == L.cosize
        assert C.shape == (1,)
        assert _image(C) == [0]

    def test_default_cosize_compact(self):
        # The no-arg default uses ``L.cosize`` as the codomain. For a compact
        # layout that span equals the cosize, so the complement is trivial.
        L = Layout.row_major(2, 3)
        assert L.complement().shape == L.complement(L.cosize).shape == (1,)

    def test_default_cosize_non_compact_raises(self):
        # A non-compact layout's natural span exceeds its cosize, so no exact
        # tiling complement fits in [0, cosize); the caller must pass a codomain
        # that is a multiple of the span.
        with pytest.raises(ValueError):
            Layout(shape=(4,), stride=(2,)).complement()  # cosize 7, span 8

    def test_non_disjoint_raises(self):
        # Overlapping modes are not an ordered/disjoint layout.
        with pytest.raises(ValueError):
            Layout(shape=(2, 2), stride=(1, 1)).complement(4)

    def test_free_function_matches_method(self):
        L = Layout(shape=(4,), stride=(2,))
        assert pm.complement(L, 8).shape == L.complement(8).shape
        assert pm.complement(L, 8).stride == L.complement(8).stride


# ---------------------------------------------------------------------------
# right_inverse
# ---------------------------------------------------------------------------


class TestRightInverse:
    @pytest.mark.parametrize(
        "layout",
        [
            Layout.row_major(2, 3),
            Layout.column_major(2, 3),
            Layout.row_major(4, 8),
            Layout.column_major(4, 8),
            Layout(shape=(2, 2), stride=(2, 1)),  # row-major 2x2
            Layout(shape=((2, 2), 2), stride=((1, 2), 4)),  # nested compact
        ],
    )
    def test_left_compose_is_identity(self, layout):
        R = layout.right_inverse()
        assert all(layout[R[i]] == i for i in range(len(layout)))

    def test_column_major_is_self_inverse(self):
        # A compact column-major layout is the identity map, so is its inverse.
        L = Layout.column_major(4, 8)
        R = L.right_inverse()
        assert all(R[i] == i for i in range(len(L)))

    def test_non_compact_raises(self):
        # Strided (non-bijective) layout has no simple right inverse.
        with pytest.raises(ValueError):
            Layout(shape=(4,), stride=(2,)).right_inverse()

    def test_free_function_matches_method(self):
        L = Layout.row_major(2, 3)
        assert pm.right_inverse(L).shape == L.right_inverse().shape
        assert pm.right_inverse(L).stride == L.right_inverse().stride


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_exports_present():
    for name in (
        "blocked_product",
        "complement",
        "right_inverse",
        "filter",
        "composition",
        "logical_divide",
        "zipped_divide",
        "tiled_divide",
        "flat_divide",
        "logical_product",
        "zipped_product",
        "tiled_product",
        "flat_product",
    ):
        assert name in pm.__all__
        assert hasattr(pm, name)
        assert hasattr(Layout, name)
