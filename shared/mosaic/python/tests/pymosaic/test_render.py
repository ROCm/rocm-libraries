# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for ``pymosaic._render``.

These tests exercise the pure-Python tuple/render layer used by the
per-layout JIT. No compiled extensions, no g++, no ninja — runnable
on any host with ``pytest`` and Python 3.10+.
"""

from __future__ import annotations

import json

import pytest

from pymosaic._render import (
    cache_key_payload,
    canonicalize_int_tuple,
    canonicalize_shape,
    canonicalize_stride,
    compact_col_major,
    compact_row_major,
    compatible,
    congruent,
    depth,
    expand_ellipsis,
    flatten,
    format_tuple,
    is_int_tuple,
    rank,
    render_int_tuple,
    render_layout_type,
    render_python_repr,
    render_shape,
    render_stride,
    size,
    slice_by_mask,
    structure_signature,
)


# ---------------------------------------------------------------------------
# canonicalize_*
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_flat_tuple_of_ints(self):
        assert canonicalize_int_tuple((4, 8)) == (4, 8)

    def test_nested_tuple(self):
        assert canonicalize_int_tuple(((2, 2), 3)) == ((2, 2), 3)

    def test_list_coerced_to_tuple(self):
        assert canonicalize_int_tuple([4, 8]) == (4, 8)
        assert canonicalize_int_tuple([[2, 2], 3]) == ((2, 2), 3)

    def test_index_protocol(self):
        class Idx:
            def __init__(self, v):
                self.v = v

            def __index__(self):
                return self.v

        assert canonicalize_int_tuple((Idx(4), Idx(8))) == (4, 8)

    def test_numpy_int_accepted(self):
        np = pytest.importorskip("numpy")
        assert canonicalize_int_tuple((np.int64(4), np.int32(8))) == (4, 8)

    def test_float_rejected(self):
        with pytest.raises(TypeError, match="expected int"):
            canonicalize_int_tuple((4.0, 8))

    def test_bool_rejected(self):
        # bool is technically an int subclass; we explicitly reject it
        # because shape=(True, False) is almost always a bug.
        with pytest.raises(TypeError, match="bool"):
            canonicalize_int_tuple((True, 8))

    def test_string_rejected(self):
        with pytest.raises(TypeError, match="expected int"):
            canonicalize_int_tuple(("4", 8))

    def test_canonicalize_shape_rejects_negative(self):
        with pytest.raises(ValueError, match="negative leaf"):
            canonicalize_shape((4, -8))

    def test_canonicalize_shape_allows_zero(self):
        # Zero-extent shapes are valid (len(L) == 0)
        assert canonicalize_shape((0, 8)) == (0, 8)

    def test_canonicalize_stride_allows_negative(self):
        # Negative strides are unusual but well-formed
        assert canonicalize_stride((-1, 8)) == (-1, 8)


# ---------------------------------------------------------------------------
# is_int_tuple
# ---------------------------------------------------------------------------


class TestIsIntTuple:
    def test_int(self):
        assert is_int_tuple(7) is True

    def test_flat_tuple(self):
        assert is_int_tuple((4, 8)) is True

    def test_nested_tuple(self):
        assert is_int_tuple(((2, 2), 3)) is True

    def test_list_is_not_tuple(self):
        # We're strict: tuples only at the recursive level.
        # Canonicalization coerces lists; introspection doesn't.
        assert is_int_tuple([4, 8]) is False

    def test_float_in_tuple(self):
        assert is_int_tuple((4, 8.0)) is False

    def test_bool_rejected(self):
        # bool subclasses int but is NOT a valid leaf — consistent with
        # _to_int rejecting bools for shape/stride values.
        assert is_int_tuple(True) is False
        assert is_int_tuple(False) is False
        assert is_int_tuple((True, 2)) is False
        assert is_int_tuple(((1, 2), False)) is False


# ---------------------------------------------------------------------------
# flatten / rank / depth / size
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flat(self):
        assert flatten((4, 8)) == (4, 8)

    def test_nested(self):
        assert flatten(((2, 2), 3)) == (2, 2, 3)

    def test_deeply_nested(self):
        assert flatten((((1, 2), 3), (4, (5, 6)))) == (1, 2, 3, 4, 5, 6)

    def test_scalar(self):
        assert flatten(7) == (7,)


class TestRank:
    def test_scalar(self):
        assert rank(7) == 0

    def test_flat(self):
        assert rank((4, 8)) == 2

    def test_nested_still_returns_top_level(self):
        assert rank(((2, 2), 3)) == 2

    def test_singleton(self):
        assert rank((4,)) == 1


class TestDepth:
    def test_scalar(self):
        assert depth(7) == 0

    def test_flat(self):
        assert depth((4, 8)) == 1

    def test_nested(self):
        assert depth(((2, 2), 3)) == 2

    def test_mixed_depth(self):
        # max of children depths + 1
        assert depth(((1, (2, 3)), 4)) == 3


class TestSize:
    def test_flat(self):
        assert size((4, 8)) == 32

    def test_nested(self):
        assert size(((2, 2), 3)) == 12

    def test_scalar(self):
        assert size(7) == 7

    def test_empty(self):
        assert size(()) == 1  # empty product

    def test_zero_extent(self):
        assert size((0, 8)) == 0


# ---------------------------------------------------------------------------
# congruent / compatible
# ---------------------------------------------------------------------------


class TestCongruent:
    def test_same_flat(self):
        assert congruent((4, 8), (8, 1)) is True

    def test_same_nested(self):
        assert congruent(((2, 2), 3), ((1, 2), 4)) is True

    def test_rank_mismatch(self):
        assert congruent((4, 8), (4,)) is False

    def test_nesting_mismatch(self):
        assert congruent(((2, 2), 3), (4, 3)) is False

    def test_scalar_vs_tuple(self):
        assert congruent(7, (7,)) is False

    def test_both_scalars(self):
        assert congruent(7, 42) is True


class TestCompatible:
    def test_same_product_flat(self):
        assert compatible((4, 8), (8, 4)) is True

    def test_same_product_different_nesting(self):
        # ((2, 2), 3) has product 12; (12,) has product 12
        assert compatible(((2, 2), 3), (12,)) is True

    def test_different_product(self):
        assert compatible((4, 8), (4, 9)) is False


# ---------------------------------------------------------------------------
# compact strides
# ---------------------------------------------------------------------------


class TestCompactColMajor:
    def test_flat_rank2(self):
        assert compact_col_major((4, 8)) == (1, 4)

    def test_flat_rank3(self):
        assert compact_col_major((2, 3, 4)) == (1, 2, 6)

    def test_nested(self):
        # ((2, 2), 3): flat leaves [2, 2, 3], strides [1, 2, 4]
        assert compact_col_major(((2, 2), 3)) == ((1, 2), 4)

    def test_deeply_nested(self):
        # ((2, 3), (4, 5)): leaves [2, 3, 4, 5], strides [1, 2, 6, 24]
        assert compact_col_major(((2, 3), (4, 5))) == ((1, 2), (6, 24))

    def test_gemm_workgroup_shape(self):
        # Regression for the C++ flat-stride bug: shape ((cu, wf), xcd) must
        # yield a congruent nested stride. pycute: ((1, 32), 128).
        assert compact_col_major(((32, 4), 8)) == ((1, 32), 128)

    def test_total_product_matches(self):
        for shape in [(4, 8), (2, 3, 5), ((2, 2), 3), ((2, 3), (4, 5))]:
            d = compact_col_major(shape)
            # cosize of compact col-major == product of shape leaves
            assert flatten(d)[-1] * flatten(shape)[-1] == size(shape)


class TestCompactRowMajor:
    def test_flat_rank2(self):
        assert compact_row_major((4, 8)) == (8, 1)

    def test_flat_rank3(self):
        # leaves [2, 3, 4], strides from right: [1, 4, 12]
        assert compact_row_major((2, 3, 4)) == (12, 4, 1)

    def test_nested(self):
        # ((2, 2), 3): leaves [2, 2, 3]; row-major strides [6, 3, 1]
        assert compact_row_major(((2, 2), 3)) == ((6, 3), 1)

    def test_deeply_nested(self):
        # ((2, 3), (4, 5)): leaves [2, 3, 4, 5]; row-major [60, 20, 5, 1]
        assert compact_row_major(((2, 3), (4, 5))) == ((60, 20), (5, 1))

    def test_gemm_workgroup_shape(self):
        # Regression for the nested-stride bug, row-major variant.
        assert compact_row_major(((32, 4), 8)) == ((32, 8), 1)

    def test_consistency_with_col_major(self):
        # For rank-1, row and col major are the same.
        assert compact_row_major((7,)) == compact_col_major((7,))


class TestStrideCongruence:
    @pytest.mark.parametrize(
        "shape",
        [(4, 8), (2, 3, 5), ((2, 2), 3), ((2, 3), (4, 5)), (((1, 2), 3), (4, 5))],
    )
    def test_col_major_congruent(self, shape):
        assert congruent(shape, compact_col_major(shape))

    @pytest.mark.parametrize(
        "shape",
        [(4, 8), (2, 3, 5), ((2, 2), 3), ((2, 3), (4, 5)), (((1, 2), 3), (4, 5))],
    )
    def test_row_major_congruent(self, shape):
        assert congruent(shape, compact_row_major(shape))


# ---------------------------------------------------------------------------
# C++ rendering
# ---------------------------------------------------------------------------


# Dynamic route: every leaf renders as a runtime ::std::size_t — the leaf
# *value* is never embedded, only the tuple structure.
class TestRenderIntTuple:
    def test_flat_shape(self):
        assert render_int_tuple((4, 8), kind="Shape") == (
            "::mosaic::Shape<::std::size_t, ::std::size_t>"
        )

    def test_flat_stride(self):
        assert render_int_tuple((8, 1), kind="Stride") == (
            "::mosaic::Stride<::std::size_t, ::std::size_t>"
        )

    def test_nested(self):
        assert render_int_tuple(((2, 2), 3), kind="Shape") == (
            "::mosaic::Shape<::mosaic::Shape<::std::size_t, ::std::size_t>, ::std::size_t>"
        )

    def test_scalar_leaf(self):
        assert render_int_tuple(7, kind="Shape") == "::std::size_t"

    def test_values_do_not_affect_output(self):
        # The whole point of the dynamic route: leaf values are erased, so
        # any two same-shaped tuples render identically.
        assert render_int_tuple((4, 8), kind="Shape") == render_int_tuple(
            (999, 1), kind="Shape"
        )
        assert render_int_tuple((-1, 4), kind="Stride") == render_int_tuple(
            (8, 1), kind="Stride"
        )

    def test_zero_extent(self):
        assert render_int_tuple((0, 8), kind="Shape") == (
            "::mosaic::Shape<::std::size_t, ::std::size_t>"
        )

    def test_bad_kind(self):
        with pytest.raises(ValueError, match="kind must be"):
            render_int_tuple((4, 8), kind="Bogus")


class TestRenderShapeStride:
    def test_render_shape(self):
        assert render_shape((4, 8)) == ("::mosaic::Shape<::std::size_t, ::std::size_t>")

    def test_render_stride(self):
        assert render_stride((8, 1)) == (
            "::mosaic::Stride<::std::size_t, ::std::size_t>"
        )


class TestRenderLayoutType:
    def test_flat(self):
        got = render_layout_type((4, 8), (8, 1))
        assert got == (
            "::mosaic::Layout<"
            "::mosaic::Shape<::std::size_t, ::std::size_t>, "
            "::mosaic::Stride<::std::size_t, ::std::size_t>"
            ">"
        )

    def test_hierarchical(self):
        got = render_layout_type(((2, 2), 3), ((1, 2), 4))
        assert got == (
            "::mosaic::Layout<"
            "::mosaic::Shape<"
            "::mosaic::Shape<::std::size_t, ::std::size_t>, "
            "::std::size_t"
            ">, "
            "::mosaic::Stride<"
            "::mosaic::Stride<::std::size_t, ::std::size_t>, "
            "::std::size_t"
            ">"
            ">"
        )

    def test_mismatched_nesting_raises(self):
        with pytest.raises(ValueError, match="different nesting profiles"):
            render_layout_type((4, 8), ((1, 2), 4))


class TestStructureSignature:
    def test_signature_ignores_values(self):
        # Same nesting profile → identical signature regardless of values.
        assert structure_signature((4, 8), (8, 1)) == structure_signature(
            (16, 32), (32, 1)
        )

    def test_signature_distinguishes_structure(self):
        # Different rank / nesting → different signature.
        assert structure_signature((4, 8), (8, 1)) != structure_signature((4,), (1,))
        assert structure_signature((4, 8), (8, 1)) != structure_signature(
            ((2, 2), 3), ((1, 2), 4)
        )


# ---------------------------------------------------------------------------
# format_tuple / render_python_repr
# ---------------------------------------------------------------------------


class TestFormatTuple:
    def test_flat(self):
        assert format_tuple((4, 8)) == "(4, 8)"

    def test_nested(self):
        assert format_tuple(((2, 2), 3)) == "((2, 2), 3)"

    def test_singleton_has_trailing_comma(self):
        assert format_tuple((4,)) == "(4,)"

    def test_scalar(self):
        assert format_tuple(7) == "7"

    def test_round_trip_via_eval(self):
        # The output should parse as a Python expression giving back
        # the same tuple. This is the property python_repr.txt relies on.
        cases = [(4, 8), ((2, 2), 3), (((1, 2), 3), 4), (4,)]
        for t in cases:
            assert eval(format_tuple(t)) == t


class TestRenderPythonRepr:
    def test_plain_layout(self):
        got = render_python_repr((4, 8), (8, 1))
        assert got == "Layout(shape=(4, 8), stride=(8, 1))"

    def test_hierarchical_layout(self):
        got = render_python_repr(((2, 2), 3), ((1, 2), 4))
        assert got == "Layout(shape=((2, 2), 3), stride=((1, 2), 4))"


# ---------------------------------------------------------------------------
# cache_key_payload — structure-only (dynamic route)
# ---------------------------------------------------------------------------


class TestCacheKeyPayload:
    def test_structure_only(self):
        payload = cache_key_payload((4, 8), (8, 1))
        d = json.loads(payload)
        assert set(d) == {"structure", "version"}
        assert d["structure"] == structure_signature((4, 8), (8, 1))

    def test_values_do_not_change_key(self):
        # Same structure, different values → identical payload. This is the
        # property that makes one .so serve every value combination.
        assert cache_key_payload((4, 8), (8, 1)) == cache_key_payload((16, 2), (2, 1))

    def test_structure_changes_key(self):
        assert cache_key_payload((4, 8), (8, 1)) != cache_key_payload((4,), (1,))
        assert cache_key_payload((4, 8), (8, 1)) != cache_key_payload(
            ((2, 2), 3), ((1, 2), 4)
        )

    def test_version_changes_key(self):
        a = cache_key_payload((4, 8), (8, 1), version="1")
        b = cache_key_payload((4, 8), (8, 1), version="2")
        assert a != b

    def test_deterministic(self):
        # Two calls with the same inputs produce byte-identical output.
        # (Key cache-stability property.)
        a = cache_key_payload(((2, 2), 3), ((1, 2), 4))
        b = cache_key_payload(((2, 2), 3), ((1, 2), 4))
        assert a == b


# ---------------------------------------------------------------------------
# expand_ellipsis — NumPy-style Ellipsis expansion
# ---------------------------------------------------------------------------


# Convenience: Python's "[:]" is slice(None, None, None). Spelling it out
# in tests keeps the intent explicit (no surprise about start/stop/step).
_FULL = slice(None, None, None)


class TestExpandEllipsis:
    def test_no_ellipsis_unchanged(self):
        assert expand_ellipsis((1, 2, 3), 3) == (1, 2, 3)
        assert expand_ellipsis((), 0) == ()
        assert expand_ellipsis((_FULL,), 1) == (_FULL,)

    def test_leading_ellipsis(self):
        assert expand_ellipsis((Ellipsis, 3), 3) == (_FULL, _FULL, 3)

    def test_trailing_ellipsis(self):
        assert expand_ellipsis((1, Ellipsis), 3) == (1, _FULL, _FULL)

    def test_middle_ellipsis(self):
        assert expand_ellipsis((1, Ellipsis, 3), 4) == (1, _FULL, _FULL, 3)

    def test_lone_ellipsis_fills_all(self):
        assert expand_ellipsis((Ellipsis,), 4) == (_FULL,) * 4

    def test_ellipsis_with_exact_remaining_count(self):
        # When the non-Ellipsis count already equals rank, Ellipsis
        # expands to zero entries (still a valid index).
        assert expand_ellipsis((1, Ellipsis, 2, 3), 3) == (1, 2, 3)

    def test_rejects_multiple_ellipsis(self):
        with pytest.raises(IndexError, match="single Ellipsis"):
            expand_ellipsis((Ellipsis, 1, Ellipsis), 3)

    def test_too_many_indices_does_not_raise_here(self):
        # The expander is permissive (no negative fill — drops the
        # Ellipsis with zero replacement). The CALLER is expected to
        # rank-check the result and raise IndexError. Pinning this
        # behaviour so the layout-side error path is exercised.
        got = expand_ellipsis((1, 2, 3, 4, Ellipsis), 3)
        assert got == (1, 2, 3, 4)  # too long, caller's job to reject


# ---------------------------------------------------------------------------
# slice_by_mask — drop/keep top-level modes per the int/slice pattern
# ---------------------------------------------------------------------------


class TestSliceByMask:
    def test_keep_first_drop_second(self):
        # L[:, 3] on (4, 8) : (8, 1) — keep mode 0, drop mode 1.
        assert slice_by_mask((_FULL, 3), (4, 8), (8, 1)) == ((4,), (8,))

    def test_drop_first_keep_second(self):
        # L[1, :] on (4, 8) : (8, 1) — drop mode 0, keep mode 1.
        assert slice_by_mask((1, _FULL), (4, 8), (8, 1)) == ((8,), (1,))

    def test_all_slice_identity(self):
        assert slice_by_mask((_FULL, _FULL), (4, 8), (8, 1)) == ((4, 8), (8, 1))

    def test_rank3_middle_kept(self):
        # L[1, :, 3] on (2, 5, 7) : (35, 7, 1) — keep mode 1 only.
        assert slice_by_mask((1, _FULL, 3), (2, 5, 7), (35, 7, 1)) == ((5,), (7,))

    def test_rank3_two_kept(self):
        assert slice_by_mask((_FULL, 2, _FULL), (2, 5, 7), (35, 7, 1)) == (
            (2, 7),
            (35, 1),
        )

    def test_hierarchical_top_level_int_drops_whole_mode(self):
        # L = ((2, 3), 8) : ((1, 2), 6). L[1, :] fixes mode 0 — drops
        # the entire nested (2, 3) mode regardless of the int value.
        # The kept mode is the flat (8,) : (6,) — base offset is lost,
        # caller recovers it.
        assert slice_by_mask((1, _FULL), ((2, 3), 8), ((1, 2), 6)) == (
            (8,),
            (6,),
        )

    def test_hierarchical_keep_nested_mode(self):
        # L[:, 3] keeps mode 0 (the nested (2,3) tuple), drops mode 1.
        # Result is a rank-1 layout whose only mode is itself a tuple.
        assert slice_by_mask((_FULL, 3), ((2, 3), 8), ((1, 2), 6)) == (
            ((2, 3),),
            ((1, 2),),
        )

    def test_rejects_range_slice(self):
        with pytest.raises(NotImplementedError, match="range slicing"):
            slice_by_mask((slice(0, 2), 3), (4, 8), (8, 1))

    def test_rejects_bool_index(self):
        # bool subclasses int; reject explicitly so L[True, :] isn't
        # silently L[1, :].
        with pytest.raises(TypeError, match="bool"):
            slice_by_mask((True, _FULL), (4, 8), (8, 1))

    def test_rejects_out_of_range_int(self):
        with pytest.raises(IndexError, match="out of range"):
            slice_by_mask((4, _FULL), (4, 8), (8, 1))  # 4 >= shape 4
        with pytest.raises(IndexError, match="out of range"):
            slice_by_mask((-1, _FULL), (4, 8), (8, 1))  # negatives forbidden

    def test_rejects_rank_mismatch(self):
        with pytest.raises(IndexError, match="3 entries"):
            slice_by_mask((1, _FULL, 2), (4, 8), (8, 1))

    def test_rejects_non_top_tuple_layout(self):
        # Defensive guard: scalar-shaped layouts have no top-level
        # positions to slice. The caller should never invoke this on
        # one, but we make the error explicit if it ever does.
        with pytest.raises(TypeError, match="top-level tuples"):
            slice_by_mask((1,), 4, 1)

    # --- hierarchical (nested-coord) slicing -------------------------------

    def test_nested_fix_whole_mode(self):
        # L = ((2,2),(2,2)) : ((1,2),(4,8)). L[:, (1,1)] keeps mode 0
        # intact, fully fixes mode 1.
        assert slice_by_mask((_FULL, (1, 1)), ((2, 2), (2, 2)), ((1, 2), (4, 8))) == (
            ((2, 2),),
            ((1, 2),),
        )

    def test_nested_partial_slice_flattens(self):
        # mode 1 = (1, :) keeps only its second leaf, flattened up.
        assert slice_by_mask(
            (_FULL, (1, _FULL)), ((2, 2), (2, 2)), ((1, 2), (4, 8))
        ) == (((2, 2), 2), ((1, 2), 8))

    def test_nested_coord_on_scalar_mode_raises(self):
        with pytest.raises(IndexError, match="nesting must match"):
            slice_by_mask(((0, _FULL), 3), (4, 8), (8, 1))

    def test_nested_coord_length_mismatch_raises(self):
        with pytest.raises(IndexError, match="entries but the nested mode"):
            slice_by_mask(((0, 1, _FULL), 3), ((2, 2), 8), ((1, 2), 6))

    # --- dice (keep=False) -------------------------------------------------

    def test_dice_keeps_fixed_modes(self):
        # Opposite of slice: int modes survive, ':' modes drop.
        assert slice_by_mask((_FULL, 3), (4, 8), (8, 1), keep=False) == ((8,), (1,))
        assert slice_by_mask((1, _FULL), (4, 8), (8, 1), keep=False) == ((4,), (8,))

    def test_dice_nested_flattens(self):
        assert slice_by_mask(
            (_FULL, (1, 1)), ((2, 2), (2, 2)), ((1, 2), (4, 8)), keep=False
        ) == ((2, 2), (4, 8))

    def test_dice_all_underscore_empty(self):
        # Every mode dropped -> empty result; the Layout-level dice()
        # turns this into a ValueError, but the raw mask just empties.
        assert slice_by_mask((_FULL, _FULL), (4, 8), (8, 1), keep=False) == ((), ())
