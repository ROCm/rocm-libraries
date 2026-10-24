# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end JIT integration tests for ``pymosaic.layout.Layout``.

Unlike ``test_render.py`` (pure Python, no compile), these tests
exercise the full JIT pipeline: render -> ninja -> g++ -> dlopen ->
eval. They take ~1-2 s per cold-cache build, sub-millisecond on
cache hit.

Each test directs the JIT cache at a per-test temp directory so tests
never read or modify the user's real ``~/.cache/mosaic/jit/``.

Skipped automatically if ``g++`` or ``ninja`` is missing from PATH.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import pytest

# Skip the whole module if the toolchain isn't available — keeps CI
# from failing on machines without a working C++ toolchain.
_HAS_GXX = shutil.which("g++") is not None
_HAS_NINJA = shutil.which("ninja") is not None
pytestmark = pytest.mark.skipif(
    not (_HAS_GXX and _HAS_NINJA),
    reason="JIT tests require g++ and ninja on PATH",
)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point the JIT cache at a per-test temp directory and clear any
    module-level cache state.
    """
    monkeypatch.setenv("MOSAIC_JIT_CACHE", str(tmp_path / "jit"))

    # Force a fresh import so the env var takes effect. (mosaic._jit
    # reads MOSAIC_JIT_CACHE in cache_root(), so we don't actually need
    # to re-import — but we DO need to clear sys.modules entries for
    # any already-loaded JIT extensions to avoid hitting stale modules.)
    import sys

    for mod_name in list(sys.modules):
        if mod_name.startswith("mosaic_jit_"):
            del sys.modules[mod_name]

    yield tmp_path / "jit"


# ---------------------------------------------------------------------------
# Basic construction + flat-index evaluation
# ---------------------------------------------------------------------------


class TestRowMajor:
    def test_simple_row_major_4x8(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        assert L.shape == (4, 8)
        assert L.stride == (8, 1)
        assert len(L) == 32
        assert L.cosize == 32
        assert L.rank == 2
        assert L.depth == 1

    def test_default_constructor_is_column_major(self, tmp_cache):
        # Bare Layout(shape=) defaults to column-major (CuTe canonical),
        # matching mosaic::make_layout on the C++ side.
        from pymosaic.layout import Layout

        L = Layout(shape=(4, 8))
        assert L.stride == (1, 4)

    def test_coord_eval_row_major(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        # row-major: offset = row * 8 + col * 1
        assert L[0, 0] == 0
        assert L[2, 3] == 19
        assert L[1, 5] == 13
        assert L[3, 7] == 31


class TestColumnMajor:
    def test_simple_column_major(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.column_major(4, 8)
        assert L.shape == (4, 8)
        assert L.stride == (1, 4)

    def test_flat_eval_sequential(self, tmp_cache):
        # For a compact column-major layout, mosaic's flat-eval
        # (idx2crd column-major then inner product) yields the identity:
        # L[i] == i for all i in [0, size).
        from pymosaic.layout import Layout

        L = Layout.column_major(4, 8)
        for i in range(len(L)):
            assert L[i] == i

    def test_coord_eval_column_major(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.column_major(4, 8)
        # column-major: offset = row * 1 + col * 4
        assert L[0, 0] == 0
        assert L[1, 0] == 1
        assert L[0, 1] == 4
        assert L[2, 3] == 14


# ---------------------------------------------------------------------------
# Round-trip identity: L[i] must equal inner_product(idx2crd(i), stride)
# ---------------------------------------------------------------------------


def _idx2crd_col_major(i: int, shape):
    """Reference Python idx2crd matching mosaic's C++ convention.
    Column-major: leftmost varies fastest."""
    if isinstance(shape, tuple):
        out = []
        for s in shape:
            sz = _shape_size(s)
            out.append(_idx2crd_col_major(i % sz, s))
            i //= sz
        return tuple(out)
    return i % shape


def _shape_size(s):
    if isinstance(s, tuple):
        out = 1
        for x in s:
            out *= _shape_size(x)
        return out
    return s


def _inner_product(crd, stride):
    if isinstance(crd, tuple):
        return sum(_inner_product(c, d) for c, d in zip(crd, stride))
    return crd * stride


class TestRoundTripIdentity:
    @pytest.mark.parametrize(
        "shape, stride",
        [
            ((4, 8), (8, 1)),  # row-major
            ((4, 8), (1, 4)),  # column-major
            ((3, 5, 2), (10, 2, 1)),  # rank-3 row-major
            ((3, 5, 2), (1, 3, 15)),  # rank-3 column-major
            ((2, 3, 4), (12, 4, 1)),
        ],
    )
    def test_flat_eval_matches_reference(self, tmp_cache, shape, stride):
        from pymosaic.layout import Layout

        L = Layout(shape=shape, stride=stride)
        for i in range(len(L)):
            expected = _inner_product(_idx2crd_col_major(i, shape), stride)
            assert L[i] == expected, f"shape={shape} stride={stride} i={i}"


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_iter_matches_enumerate(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        assert list(L) == L.enumerate()

    def test_enumerate_length(self, tmp_cache):
        from pymosaic.layout import Layout

        for shape in [(4, 8), (3, 5, 2), ((2, 2), 3)]:
            L = Layout(shape=shape)
            assert len(L.enumerate()) == len(L)


# ---------------------------------------------------------------------------
# Hierarchical layouts
# ---------------------------------------------------------------------------


class TestHierarchical:
    def test_from_modes(self, tmp_cache):
        from pymosaic.layout import Layout

        inner = Layout.column_major(2, 2)
        outer = Layout.row_major(3, 3)
        H = Layout.from_modes(inner, outer)

        assert H.shape == ((2, 2), (3, 3))
        assert H.stride == ((1, 2), (3, 1))
        assert H.rank == 2
        assert H.depth == 2
        assert len(H) == 36

    def test_from_modes_requires_two_or_more(self, tmp_cache):
        from pymosaic.layout import Layout

        with pytest.raises(ValueError, match="at least 2"):
            Layout.from_modes(Layout.row_major(4))

    def test_hierarchical_explicit_construction(self, tmp_cache):
        from pymosaic.layout import Layout

        H = Layout(shape=((2, 2), 3), stride=((1, 2), 4))
        assert H.rank == 2
        assert H.depth == 2
        assert len(H) == 12

    def test_hierarchical_stride_must_match_shape_nesting(self, tmp_cache):
        from pymosaic.layout import Layout

        with pytest.raises(ValueError, match="nesting profiles"):
            Layout(shape=((2, 2), 3), stride=(1, 2, 4))


# ---------------------------------------------------------------------------
# Indexing semantics — int vs tuple vs unsupported
# ---------------------------------------------------------------------------


class TestIndexingSemantics:
    def test_out_of_range_raises(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError, match="out of range"):
            L[100]

    def test_negative_flat_index_rejected(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError, match="out of range"):
            L[-1]

    def test_wrong_coord_arity(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError, match="rank mismatch"):
            L[1, 2, 3]

    def test_bad_coord_type(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(TypeError, match="must be int"):
            L["foo"]

    def test_nested_coord_inner_product(self, tmp_cache):
        # Hierarchical layout with a nested coord — should use the
        # Python inner-product fallback path.
        from pymosaic.layout import Layout

        H = Layout(shape=((2, 2), 3), stride=((1, 2), 4))
        # coord ((1, 1), 2) -> 1*1 + 1*2 + 2*4 = 11
        assert H[(1, 1), 2] == 11
        # coord ((0, 1), 2) -> 0*1 + 1*2 + 2*4 = 10
        assert H[(0, 1), 2] == 10

    def test_scalar_coord_on_hierarchical_mode(self, tmp_cache):
        # H[1, 2]: mode 0 is hierarchical, so 1 gets idx2crd'd against
        # (2, 2) -> (1, 0), then inner-product. 1*1 + 0*2 + 2*4 = 9.
        # This goes through the Python fallback path because not all
        # coords are flat scalars (mode 0 expects a tuple but got an int).
        # Actually it goes through eval_coord because all elements are
        # ints, and the C++ side does the idx2crd internally.
        from pymosaic.layout import Layout

        H = Layout(shape=((2, 2), 3), stride=((1, 2), 4))
        assert H[1, 2] == 9


# ---------------------------------------------------------------------------
# repr / str / equality / hashing
# ---------------------------------------------------------------------------


class TestReprEqHash:
    def test_repr_round_trip(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        ns = {"Layout": Layout}
        L_round = eval(repr(L), ns)
        assert L == L_round

    def test_str_matches_cpp_format(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        # mosaic's operator<< format is "(shape):(stride)" without spaces
        assert str(L) == "(4,8):(8,1)"

    def test_equal_when_canonical_same(self, tmp_cache):
        from pymosaic.layout import Layout

        assert Layout.row_major(4, 8) == Layout.row_major(4, 8)
        assert Layout.row_major(4, 8) != Layout.column_major(4, 8)

    def test_hashable(self, tmp_cache):
        from pymosaic.layout import Layout

        L1 = Layout.row_major(4, 8)
        L2 = Layout.row_major(4, 8)
        L3 = Layout.column_major(4, 8)
        assert hash(L1) == hash(L2)
        # Used as dict keys — equal layouts collapse to a single key.
        d = {L1: "a", L2: "b", L3: "c"}
        assert len(d) == 2
        assert d[L1] == "b"

    def test_eq_other_type_returns_notimplemented(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        assert (L == "not a layout") is False
        assert (L == 42) is False


# ---------------------------------------------------------------------------
# Cache discipline
# ---------------------------------------------------------------------------


class TestCacheDiscipline:
    def test_cache_dir_populated_after_first_call(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        _ = L[0]  # trigger JIT

        cache = pathlib.Path(os.environ["MOSAIC_JIT_CACHE"])
        entries = [p for p in cache.iterdir() if p.is_dir()]
        assert len(entries) == 1

        entry = entries[0]
        # Every promised artifact is present.
        for name in [
            "module.cpp",
            "build.ninja",
            "compile_commands.json",
            "compile.log",
            "manifest.json",
            "python_repr.txt",
        ]:
            assert (entry / name).is_file(), f"missing {name}"

        # Exactly one .so.
        sos = list(entry.glob("*.so"))
        assert len(sos) == 1

    def test_same_layout_hits_cache(self, tmp_cache):
        from pymosaic.layout import Layout

        L1 = Layout.row_major(4, 8)
        _ = L1[0]
        # Second Layout with identical canonical (shape, stride) reuses
        # the same .so and the same module entry in the cache.
        L2 = Layout.row_major(4, 8)
        _ = L2[0]
        cache = pathlib.Path(os.environ["MOSAIC_JIT_CACHE"])
        entries = [p for p in cache.iterdir() if p.is_dir()]
        assert len(entries) == 1

    def test_same_structure_shares_one_entry(self, tmp_cache):
        # Dynamic route: the module is keyed on STRUCTURE, not values.
        # row-major (4,8), column-major (4,8), and row-major (8,4) are all
        # flat rank-2 layouts → one shared .so regardless of their values.
        from pymosaic.layout import Layout

        L1 = Layout.row_major(4, 8)
        L2 = Layout.column_major(4, 8)
        L3 = Layout.row_major(8, 4)
        for L in (L1, L2, L3):
            _ = L[0]
        cache = pathlib.Path(os.environ["MOSAIC_JIT_CACHE"])
        entries = [p for p in cache.iterdir() if p.is_dir()]
        assert len(entries) == 1

    def test_different_structures_different_entries(self, tmp_cache):
        # Distinct nesting profiles compile distinct modules: flat rank-2,
        # flat rank-1, and a hierarchical rank-2 are three structures.
        from pymosaic.layout import Layout

        flat2 = Layout.row_major(4, 8)
        flat1 = Layout(shape=(8,), stride=(1,))
        nested = Layout(shape=((2, 2), 3), stride=((1, 2), 4))
        for L in (flat2, flat1, nested):
            _ = L[0]
        cache = pathlib.Path(os.environ["MOSAIC_JIT_CACHE"])
        entries = [p for p in cache.iterdir() if p.is_dir()]
        assert len(entries) == 3

    def test_python_repr_matches_constructor(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        _ = L[0]

        cache = pathlib.Path(os.environ["MOSAIC_JIT_CACHE"])
        entry = next(p for p in cache.iterdir() if p.is_dir())
        repr_txt = (entry / "python_repr.txt").read_text().strip()
        assert repr_txt == "Layout(shape=(4, 8), stride=(8, 1))"

    def test_manifest_contains_host_info(self, tmp_cache):
        import json

        from pymosaic import _jit
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        _ = L[0]

        cache = pathlib.Path(os.environ["MOSAIC_JIT_CACHE"])
        entry = next(p for p in cache.iterdir() if p.is_dir())
        manifest = json.loads((entry / "manifest.json").read_text())
        # Dynamic route: the module is keyed on structure, so the manifest
        # records the structure signature plus the representative layout
        # (the first one that triggered the compile).
        assert "structure" in manifest
        assert manifest["example_shape"] == [4, 8]
        assert manifest["example_stride"] == [8, 1]
        assert "host" in manifest
        assert "timestamp" in manifest
        # Pin to whatever the package currently advertises rather than a
        # hard-coded string — the version bumps with each schema change.
        assert manifest["version"] == _jit.VERSION


# ---------------------------------------------------------------------------
# Slicing — pure-Python tuple transform, returns a fresh Layout
# ---------------------------------------------------------------------------


# Python's ":" canonicalised. Spelling it out at module level keeps test
# bodies readable.
_FULL = slice(None, None, None)


class TestSliceShapeAndStride:
    """The most basic property: a sliced sub-layout has exactly the
    shape/stride mosaic's free-function ``slice(coord, L)`` would
    produce — the kept positions' shape/stride, in order, with the
    fixed positions dropped.
    """

    def test_keep_first_drop_second_rank2(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        col = L[:, 3]
        assert isinstance(col, Layout)
        assert col.shape == (4,)
        assert col.stride == (8,)

    def test_drop_first_keep_second_rank2(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        row = L[1, :]
        assert isinstance(row, Layout)
        assert row.shape == (8,)
        assert row.stride == (1,)

    def test_rank3_keep_middle_only(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        sub = L[1, :, 3]
        assert sub.shape == (5,)
        assert sub.stride == (7,)

    def test_rank3_keep_two(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        sub = L[:, 2, :]
        assert sub.shape == (2, 7)
        assert sub.stride == (35, 1)


class TestSliceOffsetSemantics:
    """Mosaic's slicing semantics: the base offset contributed by the
    fixed modes is NOT carried into the sub-layout. The kept-mode
    offsets are produced fresh from the kept strides; the caller
    recovers the base offset.

    Each test pins both halves: the sub-layout matches the kept
    strides AND the sum (sub_offset + base_offset) reproduces the
    full-layout offset.
    """

    def test_column_slice_row_major(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)  # (4,8):(8,1)
        col = L[:, 3]  # (4,):(8,)
        # col[i] = i * 8; the base offset is 3 * 1 (col-3 contribution).
        base = 3 * 1
        for i in range(len(col)):
            assert col[i] + base == L[i, 3]

    def test_row_slice_row_major(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)  # (4,8):(8,1)
        row = L[1, :]  # (8,):(1,)
        # row[j] = j * 1; the base offset is 1 * 8 (row-1 contribution).
        base = 1 * 8
        for j in range(len(row)):
            assert row[j] + base == L[1, j]

    def test_rank3_slice_offset(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        sub = L[1, :, 3]  # (5,):(7,)
        base = 1 * 35 + 3 * 1
        for i in range(len(sub)):
            assert sub[i] + base == L[1, i, 3]


class TestSliceIdentity:
    """All-slice and Ellipsis-only forms must return ``self`` (object
    identity). Catches a real footgun: if slicing always built a new
    Layout we'd silently waste a JIT cache entry on the no-op case
    and break ``L[:] is L`` invariants users may rely on.
    """

    def test_all_slice_returns_self_rank2(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        assert L[:, :] is L

    def test_ellipsis_returns_self(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        assert L[...] is L

    def test_single_slice_on_rank1_returns_self(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(8,), stride=(1,))
        assert L[:] is L


class TestSliceEllipsis:
    """Ellipsis expands to fill remaining axes with ``:``. Sliced
    sub-layouts come out the same as if the user had spelled the
    full slice pattern explicitly.
    """

    def test_leading_ellipsis(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        a = L[..., 3]
        b = L[:, :, 3]
        assert a.shape == b.shape == (2, 5)
        assert a.stride == b.stride == (35, 7)

    def test_trailing_ellipsis(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        a = L[1, ...]
        b = L[1, :, :]
        assert a.shape == b.shape == (5, 7)
        assert a.stride == b.stride == (7, 1)

    def test_middle_ellipsis_rank4(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 3, 5, 7), stride=(105, 35, 7, 1))
        a = L[1, ..., 3]
        b = L[1, :, :, 3]
        assert a.shape == b.shape == (3, 5)
        assert a.stride == b.stride == (35, 7)


class TestSliceHierarchical:
    """Top-level slicing of a hierarchical layout. A fixed top-level
    mode drops the entire (possibly nested) sub-shape. A kept
    top-level mode whose shape is itself a tuple becomes a nested
    mode in the result.
    """

    def test_drop_hierarchical_mode(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=((2, 3), 8), stride=((1, 2), 6))
        sub = L[1, :]  # drops the (2,3) mode
        assert sub.shape == (8,)
        assert sub.stride == (6,)
        assert sub.rank == 1

    def test_keep_hierarchical_mode(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=((2, 3), 8), stride=((1, 2), 6))
        sub = L[:, 3]
        # The kept mode 0 is itself a tuple — the result is rank-1
        # with a nested mode (depth-2 layout).
        assert sub.shape == ((2, 3),)
        assert sub.stride == ((1, 2),)
        assert sub.rank == 1
        assert sub.depth == 2


class TestSliceComposition:
    """Slicing a sliced sub-layout. The sub-layout is itself a Layout,
    so further indexing works without any special-casing.
    """

    def test_double_slice_to_scalar_via_repeated_int(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        row_then_col = L[1, :, 3]  # rank-1, shape (5,)
        # row_then_col[2] is the kept-mode-only offset, equal to 2 * 7.
        assert row_then_col[2] == 2 * 7

    def test_progressive_slicing(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        # Two slices of rank: L[1, :, :] -> rank-2, then [:, 3] -> rank-1.
        slab = L[1, :, :]
        col = slab[:, 3]
        assert col.shape == (5,)
        assert col.stride == (7,)


class TestSliceErrors:
    def test_range_slice_rejected(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(NotImplementedError, match="range slicing"):
            L[0:2, :]
        with pytest.raises(NotImplementedError, match="range slicing"):
            L[1:3]
        with pytest.raises(NotImplementedError, match="range slicing"):
            L[::2, :]

    def test_too_many_ellipsis(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(2, 5, 7), stride=(35, 7, 1))
        with pytest.raises(IndexError, match="single Ellipsis"):
            L[..., 1, ...]

    def test_rank_mismatch_in_slice(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError):
            L[1, :, 2]  # one too many entries

    def test_out_of_range_int_in_slice(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError, match="out of range"):
            L[4, :]  # row 4 doesn't exist
        with pytest.raises(IndexError, match="out of range"):
            L[:, 8]  # column 8 doesn't exist

    def test_bool_in_index_rejected(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(TypeError, match="bool"):
            L[True, :]
        with pytest.raises(TypeError, match="bool"):
            L[True]

    def test_hierarchical_slice_keeps_inner_mode(self, tmp_cache):
        # Hierarchical slicing (formerly NotImplementedError): a nested
        # coord descends into the nested mode. L[(0, :), 3] keeps the
        # inner ':' leaf of mode 0 (shape 3, stride 2) and drops mode 1.
        from pymosaic.layout import Layout

        L = Layout(shape=((2, 3), 8), stride=((1, 2), 6))
        sub = L[(0, _FULL), 3]
        assert sub.shape == (3,)
        assert sub.stride == (2,)
        assert sub.enumerate() == [0, 2, 4]


# ---------------------------------------------------------------------------
# idx2crd / crd2idx — the JIT-bound coord <-> flat-index pair
# ---------------------------------------------------------------------------
#
# These bind the mosaic C++ ``idx2crd(i, L)`` (layout.hpp:266-269) and the
# ``L(tuple<...>)`` overload (layout.hpp:122-128). The composition identity
# ``L.crd2idx(L.idx2crd(i)) == L[i]`` is the canonical correctness check.


class TestIdx2CrdShape:
    """``idx2crd(i)`` must return a tuple whose nesting matches L.shape."""

    def test_flat_rank1(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout(shape=(8,), stride=(1,))
        coord = L.idx2crd(5)
        assert coord == (5,)

    def test_flat_rank2_col_major(self, tmp_cache):
        # Column-major (4, 8): the leftmost axis varies fastest.
        # idx=5 = 1*1 + 1*4  → coord (1, 1)
        from pymosaic.layout import Layout

        L = Layout.column_major(4, 8)
        assert L.idx2crd(0) == (0, 0)
        assert L.idx2crd(1) == (1, 0)
        assert L.idx2crd(4) == (0, 1)
        assert L.idx2crd(5) == (1, 1)
        assert L.idx2crd(31) == (3, 7)

    def test_hierarchical(self, tmp_cache):
        # ((2, 2), 8): inner tile of 4, outer of 8. coord nesting must
        # match — top level is rank-2, the first element is itself a
        # rank-2 tuple, the second is a scalar int.
        from pymosaic.layout import Layout

        L = Layout(shape=((2, 2), 8), stride=((1, 2), 4))
        assert L.idx2crd(0) == ((0, 0), 0)
        assert L.idx2crd(5) == ((1, 0), 1)
        assert L.idx2crd(7) == ((1, 1), 1)
        assert L.idx2crd(31) == ((1, 1), 7)

    def test_singleton_mode_nesting_preserved(self, tmp_cache):
        # Rank-1 with a single mode (3,) — result must keep the singleton
        # tuple shape; emitter trailing-comma matters.
        from pymosaic.layout import Layout

        L = Layout(shape=(3,), stride=(1,))
        assert L.idx2crd(2) == (2,)


class TestIdx2CrdComposition:
    """``L.crd2idx(L.idx2crd(i)) == L[i]`` for every i in the domain.

    Sweep across flat, col/row-major, and hierarchical layouts to lock
    in the bidirectional identity.
    """

    @pytest.mark.parametrize(
        "shape,stride",
        [
            ((8,), (1,)),
            ((4, 8), (8, 1)),  # row-major
            ((4, 8), (1, 4)),  # col-major
            ((2, 3, 5), (15, 5, 1)),  # row-major rank-3
            (((2, 2), 8), ((1, 2), 4)),  # hierarchical
            (((2, 3), 4), ((1, 8), 24)),  # hierarchical, non-compact
            ((1, 32), (0, 1)),  # singleton mode
        ],
    )
    def test_roundtrip(self, tmp_cache, shape, stride):
        from pymosaic.layout import Layout

        L = Layout(shape=shape, stride=stride)
        for i in range(len(L)):
            assert (
                L.crd2idx(L.idx2crd(i)) == L[i]
            ), f"roundtrip failed at i={i} for {L!r}"

    def test_crd2idx_matches_eval_for_top_level_ints(self, tmp_cache):
        # When L has no nested modes, crd2idx on a flat tuple must agree
        # with eval_coord (the dedicated top-level-ints fast path).
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        for r in range(4):
            for c in range(8):
                assert L.crd2idx((r, c)) == L[r, c]


class TestIdx2CrdErrors:
    def test_idx2crd_out_of_range(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(IndexError, match="out of range"):
            L.idx2crd(32)
        with pytest.raises(IndexError, match="out of range"):
            L.idx2crd(-1)

    def test_idx2crd_rejects_bool(self, tmp_cache):
        from pymosaic.layout import Layout

        L = Layout.row_major(4, 8)
        with pytest.raises(TypeError, match="non-bool int"):
            L.idx2crd(True)

    def test_crd2idx_rank_mismatch(self, tmp_cache):
        # Hierarchical layout demands matching nesting profile — pass a
        # flat tuple and the C++ side complains.
        from pymosaic.layout import Layout

        L = Layout(shape=((2, 2), 8), stride=((1, 2), 4))
        with pytest.raises(IndexError, match="rank|tuple"):
            L.crd2idx((1, 0, 1))  # flat where nested expected

    def test_crd2idx_scalar_on_tuple_mode_routed_via_eval(self, tmp_cache):
        # crd2idx is strict — passing a scalar where a tuple is expected
        # is an error. Users who want the scalar-on-tuple-mode convenience
        # are pointed at eval (L[i]) / eval_coord (L[i, j]) instead.
        from pymosaic.layout import Layout

        L = Layout(shape=((2, 2), 8), stride=((1, 2), 4))
        # The scalar-on-tuple-mode form goes through eval_coord, which
        # mosaic's C++ handles correctly via internal idx2crd.
        assert L[1, 1] == 5
        # But crd2idx with a scalar at the tuple position is rejected.
        with pytest.raises(IndexError, match="tuple"):
            L.crd2idx((1, 1))  # first element should be a tuple


class TestIdx2CrdFreeFunctions:
    def test_free_functions_match_methods(self, tmp_cache):
        from pymosaic.layout import Layout, crd2idx, idx2crd

        L = Layout(shape=((2, 2), 8), stride=((1, 2), 4))
        for i in range(len(L)):
            assert idx2crd(i, L) == L.idx2crd(i)
            assert crd2idx(L.idx2crd(i), L) == L.crd2idx(L.idx2crd(i))

    def test_free_functions_reject_non_layout(self):
        from pymosaic.layout import crd2idx, idx2crd

        with pytest.raises(TypeError, match="expected a Layout"):
            idx2crd(0, ((4, 8), (8, 1)))
        with pytest.raises(TypeError, match="expected a Layout"):
            crd2idx((0, 0), ((4, 8), (8, 1)))
