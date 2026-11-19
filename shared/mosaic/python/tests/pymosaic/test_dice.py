# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for hierarchical slicing (``L[:, (1,1)]``) and ``dice`` — the two
slicing features that previously raised ``NotImplementedError``.

Both are pure structural (tuple) transforms mirroring mosaic's
``slice`` / ``dice`` (``include/mosaic/slice.hpp``): ``:`` (full slice)
is the underscore placeholder, an ``int`` marks a fixed mode.

  - slicing keeps the ``:`` modes, drops the fixed modes;
  - ``dice`` keeps the fixed modes, drops the ``:`` modes.

A nested-tuple coord descends into the matching nested mode; leaves kept
from inside a partially-sliced mode are flattened up into the result.
Per mosaic semantics the base offset of the dropped modes is not carried.
"""

from __future__ import annotations

import pytest

from pymosaic import Layout, dice

_FULL = slice(None, None, None)


# ---------------------------------------------------------------------------
# Hierarchical slicing via L[...]
# ---------------------------------------------------------------------------


class TestHierarchicalSlice:
    def test_fix_whole_nested_mode(self):
        # mode 1 fully fixed by nested ints -> dropped; mode 0 kept intact.
        L = Layout(shape=((2, 2), (2, 2)), stride=((1, 2), (4, 8)))
        sub = L[:, (1, 1)]
        assert sub.shape == ((2, 2),)
        assert sub.stride == ((1, 2),)

    def test_partial_inner_slice_flattens_up(self):
        # mode 1 = (1, :) keeps only its second leaf, flattened to top level.
        L = Layout(shape=((2, 2), (2, 2)), stride=((1, 2), (4, 8)))
        sub = L[:, (1, _FULL)]
        assert sub.shape == ((2, 2), 2)
        assert sub.stride == ((1, 2), 8)

    def test_inner_slice_in_first_mode(self):
        L = Layout(shape=((2, 2), (2, 2)), stride=((1, 2), (4, 8)))
        sub = L[(0, _FULL), 2]
        assert sub.shape == (2,)
        assert sub.stride == (2,)
        assert sub.enumerate() == [0, 2]

    def test_inner_slice_both_modes(self):
        L = Layout(shape=((2, 2), (2, 2)), stride=((1, 2), (4, 8)))
        sub = L[(0, _FULL), (1, _FULL)]
        assert sub.shape == (2, 2)
        assert sub.stride == (2, 8)

    def test_keeps_map_minus_base_offset(self):
        # The defining property: sliced offsets + base == full-layout
        # offsets at the fixed coords. base = offset from the dropped modes.
        L = Layout(shape=((2, 2), (2, 2)), stride=((1, 2), (4, 8)))
        sub = L[:, (1, 1)]  # keep mode 0, fix mode 1 at (1,1)
        base = L[(0, 0), (1, 1)]  # = 1*4 + 1*8 = 12
        # sub is rank-1 with a single nested (2,2) mode; index it nested.
        for i0 in range(2):
            for i1 in range(2):
                assert sub[((i0, i1),)] + base == L[(i0, i1), (1, 1)]


# ---------------------------------------------------------------------------
# dice — complement of slice
# ---------------------------------------------------------------------------


class TestDice:
    def test_flat_keep_second(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        d = F.dice((_FULL, 3))  # ':' dropped, int kept -> keep mode 1
        assert d.shape == (8,)
        assert d.stride == (1,)

    def test_flat_keep_first(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        d = F.dice((2, _FULL))
        assert d.shape == (4,)
        assert d.stride == (8,)

    def test_all_fixed_is_identity(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        d = F.dice((2, 3))  # every mode fixed -> keep all
        assert d.shape == (4, 8)
        assert d.stride == (8, 1)

    def test_scalar_int_keeps_whole_nested_mode(self):
        # A bare int paired with a nested mode keeps that mode intact.
        L = Layout(shape=((2, 2), 8), stride=((1, 2), 6))
        d = L.dice((1, _FULL))  # keep mode 0 (the nested one)
        assert d.shape == ((2, 2),)
        assert d.stride == ((1, 2),)

    def test_nested_dice_flattens(self):
        L = Layout(shape=((2, 2), (2, 2)), stride=((1, 2), (4, 8)))
        d = L.dice((_FULL, (1, 1)))  # drop mode 0, keep mode 1's leaves
        assert d.shape == (2, 2)
        assert d.stride == (4, 8)

    def test_rank1_bare_coord(self):
        L = Layout(shape=(6,), stride=(2,))
        assert L.dice(0).shape == (6,)  # the single fixed mode is kept

    def test_complement_of_slice_flat(self):
        # slice keeps ':' modes, dice keeps int modes -> together they
        # account for every top-level mode exactly once.
        F = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        coord = (1, _FULL, 3)
        sl = F[coord]  # keep mode 1
        di = F.dice(coord)  # keep modes 0 and 2
        assert sl.shape == (5,)
        assert di.shape == (2, 7)
        # union of kept leaves == all leaves
        assert sorted(sl.shape + di.shape) == sorted(F.shape)


# ---------------------------------------------------------------------------
# Free-function form and errors
# ---------------------------------------------------------------------------


class TestDiceApi:
    def test_free_function_matches_method(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        assert dice((_FULL, 3), F).shape == F.dice((_FULL, 3)).shape

    def test_free_function_arg_order(self):
        # mosaic order: dice(coord, layout)
        F = Layout(shape=(4, 8), stride=(8, 1))
        d = dice((2, _FULL), F)
        assert (d.shape, d.stride) == ((4,), (8,))

    def test_dice_all_slice_raises(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        with pytest.raises(ValueError, match="keeps no modes"):
            F.dice((_FULL, _FULL))

    def test_dice_rejects_range_slice(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        with pytest.raises(NotImplementedError, match="range slicing"):
            F.dice((slice(0, 2), 3))

    def test_dice_rejects_bool(self):
        F = Layout(shape=(4, 8), stride=(8, 1))
        with pytest.raises(TypeError, match="bool"):
            F.dice((True, _FULL))

    def test_dice_rejects_non_layout(self):
        with pytest.raises(TypeError):
            dice((_FULL, 3), (4, 8))  # type: ignore[arg-type]


class TestHierarchicalSliceErrors:
    def test_nested_coord_on_scalar_mode_raises(self):
        # A nested coord paired with a scalar mode is a nesting mismatch.
        L = Layout(shape=(4, 8), stride=(8, 1))
        with pytest.raises(IndexError, match="nesting must match"):
            L[(0, _FULL), 3]

    def test_nested_coord_wrong_length_raises(self):
        L = Layout(shape=((2, 2), 8), stride=((1, 2), 6))
        with pytest.raises(IndexError, match="entries but the nested mode"):
            L[(0, 1, _FULL), 3]
