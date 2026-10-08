#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for GemmUniversalVecFeatureEngine.

The base class keeps `extract` and `extract_batch` as two independent
implementations, and the subclass makes that worse because the base
`extract_batch` sizes its output from the OVERRIDDEN get_feature_names. So the
parity tests here are the point of the file, and they are parametrized over the
input classes that actually differ -- NaN, empty string, None -- not just clean
integers. A parity test over clean integers proves only that two clean-integer
paths agree, which is not where either path was wrong.

`extract(problem, kernel)` is always called with two DISTINCT dicts, because
that is how `predict._predict_single` calls it. Passing one merged dict hides
which argument each value is actually read from.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HEURISTICS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HEURISTICS))

from feature_engine import GemmUniversalFeatureEngine  # noqa: E402
from feature_engine_vec import (  # noqa: E402
    VEC_FEATURES,
    GemmUniversalVecFeatureEngine,
)

#: The four layouts the engine accepts -- the same four the base engine's
#: LAYOUT_MAP and data_pipeline.parse_kernel_name know. The column-major-C forms
#: are rejected (see TestLayoutValidation); gemm_tile_divides_problem covers
#: them on the codegen side, where they ARE resolvable.
LAYOUTS = ("rcr", "rrr", "crr", "ccr")


def _problem(m=512, n=512, k=512, layout="rcr", dtype="bf16", upper=False):
    """Problem-side keys only -- no vector widths."""
    keys = ("M", "N", "K") if upper else ("m", "n", "k")
    return {
        "op_type": "gemm_universal",
        "dtype": dtype,
        "layout": layout,
        "arch": "gfx950",
        keys[0]: m,
        keys[1]: n,
        keys[2]: k,
        "split_k": 1,
    }


#: A geometry whose native widths DISCRIMINATE the things that can go wrong.
#:
#: The default fixture geometry saturates: for bf16/gfx950 it returns (8, 8, *)
#: for every layout, and returns the SAME triple if `waves` and `warp_tile` are
#: transposed, or if the arch changes -- so it cannot detect the module's own
#: documented hazards. This one resolves to (1, 1, 1) on gfx950/default, and
#: gives (8, 8, 1) under a waves<->warp_tile swap and (4, 4, 1) on a wave-32
#: arch, so each error is visible by value.
DISCRIMINATING = {
    "tile_m": 32,
    "tile_n": 32,
    "tile_k": 16,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 32,
    "warp_tile_n": 32,
    "warp_tile_k": 16,
}


def _kernel(vec=(0, 0, 0), **over):
    """Kernel-side keys only -- the vector widths live here."""
    return {
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 128,
        "warp_m": 2,
        "warp_n": 2,
        "warp_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "compv3",
        "epilogue": "default",
        "scheduler": "intrawave",
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": True,
        "vec_a": vec[0],
        "vec_b": vec[1],
        "vec_c": vec[2],
        **over,
    }


def _row(**kw):
    """One merged problem+kernel row. Kernel-side keys may be overridden."""
    pk = ("m", "n", "k", "layout", "dtype")
    kernel_over = {k: v for k, v in kw.items() if k not in (*pk, "vec", "arch")}
    row = {
        **_problem(**{k: v for k, v in kw.items() if k in pk}),
        **_kernel(kw.get("vec", (0, 0, 0)), **kernel_over),
        "measured_tflops": 1.0,
        "latency_ms": 1.0,
        "is_valid": True,
    }
    if "arch" in kw:
        row["arch"] = kw["arch"]
    return row


def _frame(**kw):
    """One-row DataFrame equivalent of _problem + _kernel merged."""
    return pd.DataFrame([_row(**kw)])


def _legal(m, n, k, layout="rcr", dtype="bf16"):
    """The problem's widest legal A/B/C widths -- the fraction's denominator."""
    from codegen_common import gemm_problem_vector_sizes
    from feature_engine_vec import _out_dtype

    return gemm_problem_vector_sizes(m, n, k, layout, dtype, dtype, _out_dtype(dtype))


class TestSchema:
    def test_adds_exactly_six_features(self):
        base = len(GemmUniversalFeatureEngine().get_feature_names())
        ext = GemmUniversalVecFeatureEngine().get_feature_names()
        assert len(ext) == base + 6
        assert ext[-6:] == VEC_FEATURES

    def test_base_engine_is_untouched(self):
        """train.py's warm-start requires an exact schema match, so a base that
        grew would silently invalidate every model already trained on it."""
        names = GemmUniversalFeatureEngine().get_feature_names()
        assert not any(f in names for f in VEC_FEATURES)

    def test_extract_width_matches_the_name_list(self):
        fe = GemmUniversalVecFeatureEngine()
        assert fe.extract(_problem(), _kernel()).shape == (len(fe.get_feature_names()),)

    def test_extract_batch_width_matches_the_name_list(self):
        """The hazard: the base extract_batch sizes from the overridden
        get_feature_names, so a naive append double-counts the new columns."""
        fe = GemmUniversalVecFeatureEngine()
        assert fe.extract_batch(_frame()).shape == (1, len(fe.get_feature_names()))


class TestParity:
    """extract and extract_batch are independent implementations. Compare them
    on the input classes where they previously disagreed, not only clean ints.

    The extents are chosen so the three operands have DIFFERENT legal widths --
    gcd(1792, 8) = 8, gcd(300, 8) = 4, gcd(20226, 8) = 2. With the earlier
    all-multiples-of-8 extents every dim gave 8, so reading an operand off the
    wrong dimension produced an identical feature vector and parity proved
    nothing about the layout mapping."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize(
        "vec", [(0, 0, 0), (8, 8, 8), (4, 4, 8), (1, 1, 1), (2, 8, 8)]
    )
    def test_single_matches_batch(self, layout, vec):
        fe = GemmUniversalVecFeatureEngine()
        p, kn = _problem(m=1792, n=300, k=20226, layout=layout), _kernel(vec)
        np.testing.assert_allclose(
            fe.extract(p, kn),
            fe.extract_batch(_frame(m=1792, n=300, k=20226, layout=layout, vec=vec))[0],
            rtol=0,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "stored",
        [4, "4", 4.0, 8, "8"],
        ids=["int", "str", "float", "int8", "str8"],
    )
    def test_width_coercion_agrees(self, stored):
        """String and float spellings of a real width must agree across paths."""
        fe = GemmUniversalVecFeatureEngine()
        kn = {**_kernel(), "vec_a": stored}
        df = _frame()
        df["vec_a"] = pd.Series([stored], dtype=object)
        np.testing.assert_allclose(
            fe.extract(_problem(), kn), fe.extract_batch(df)[0], rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize(
        "stored", [float("nan"), "", None], ids=["nan", "empty-string", "none"]
    )
    def test_both_paths_reject_a_null_width(self, stored):
        """Null is NOT native. data_pipeline writes 0 for a native kernel and
        backfills unpopulated canonical columns with None, so a null width means
        the kernel name was never parsed. Both paths must say so."""
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="native"):
            fe.extract(_problem(), {**_kernel(), "vec_a": stored})
        df = _frame()
        df["vec_a"] = pd.Series([stored], dtype=object)
        with pytest.raises(ValueError, match="native"):
            fe.extract_batch(df)

    def test_parity_across_a_mixed_batch(self):
        fe = GemmUniversalVecFeatureEngine()
        specs = [
            dict(m=1, n=32, k=3072, layout="rrr", vec=(8, 1, 1)),
            dict(m=2048, n=512, k=192, layout="rcr", vec=(0, 0, 0)),
            dict(m=256, n=256, k=133120, layout="crr", vec=(2, 8, 8)),
            dict(m=1024, n=1024, k=1562, layout="ccr", vec=(8, 2, 8)),
        ]
        batch = fe.extract_batch(
            pd.concat([_frame(**s) for s in specs], ignore_index=True)
        )
        for i, s in enumerate(specs):
            p = _problem(**{k: v for k, v in s.items() if k != "vec"})
            np.testing.assert_allclose(
                fe.extract(p, _kernel(s["vec"])), batch[i], rtol=0, atol=1e-12
            )


class TestArgumentSplit:
    """predict._predict_single passes disjoint problem/kernel dicts. Merging
    them in a test hides which argument a value is read from."""

    def test_widths_come_from_the_kernel_dict(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(), _kernel((4, 4, 8)))[-6:]
        assert list(v[:3]) == [4.0, 4.0, 8.0]

    def test_widths_in_the_problem_dict_are_ignored(self):
        """Reading widths from `problem` instead of `kernel` must be visible."""
        fe = GemmUniversalVecFeatureEngine()
        p = {**_problem(), "vec_a": 1, "vec_b": 1, "vec_c": 1}
        v = fe.extract(p, _kernel((4, 4, 8)))[-6:]
        assert list(v[:3]) == [4.0, 4.0, 8.0], (
            "widths must be read from the kernel dict"
        )

    def test_extents_come_from_the_problem_dict(self):
        fe = GemmUniversalVecFeatureEngine()
        # k=4 caps the rcr A/B legal width at 4, so vec 4 is a perfect fit
        v = fe.extract(_problem(k=4), _kernel((4, 4, 8)))[-6:]
        assert v[3] == pytest.approx(1.0)

    def test_uppercase_extents_are_accepted(self):
        """The base engine reads problem.get("m", problem.get("M", 0)); not
        matching it gave correct base features and vec features from m=n=k=1."""
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=4, upper=True), _kernel((4, 4, 8)))[-6:]
        assert v[3] == pytest.approx(1.0), "uppercase M/N/K must resolve like lowercase"

    def test_missing_extent_raises(self):
        fe = GemmUniversalVecFeatureEngine()
        p = {k: v for k, v in _problem().items() if k != "k"}
        with pytest.raises(KeyError):
            fe.extract(p, _kernel())


class TestVectorFeatures:
    def test_native_resolves_to_the_kernels_own_width(self):
        """vec == 0 is the KERNEL's native width -- not a literal 0, and not the
        problem's widest legal width. The fixture uses the default epilogue, so
        A and B reach the problem's legal width but C stores one element."""
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(), _kernel((0, 0, 0)))[-6:]
        assert list(v[:3]) == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(v[3:5], [1.0, 1.0])
        assert v[5] < 1.0, "default-epilogue C is narrower than the legal width"

    def test_fraction_is_problem_relative(self):
        """The same kernel is well matched to one problem and wasteful on
        another -- the reason the raw width alone is insufficient."""
        fe = GemmUniversalVecFeatureEngine()
        kn = _kernel((4, 4, 8))
        tight = fe.extract(_problem(k=4), kn)[-6:]  # rcr A/B contiguous in k
        loose = fe.extract(_problem(k=512), kn)[-6:]
        assert list(tight[:3]) == list(loose[:3])
        assert tight[3] == pytest.approx(1.0)  # 4 of 4 legal
        assert loose[3] == pytest.approx(0.5)  # 4 of 8 legal

    def test_fraction_exceeds_one_when_width_is_illegal(self):
        """A width wider than the problem allows is rejected by the kernel, so
        it should not reach training data -- left unclamped so that if it does,
        it is visible rather than indistinguishable from an exact fit."""
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=4), _kernel((8, 8, 8)))[-6:]
        assert v[3] == pytest.approx(2.0)
        np.testing.assert_allclose(
            fe.extract_batch(_frame(k=4, vec=(8, 8, 8)))[0][-6:], v
        )


class TestBatchValidation:
    """A NaN extent casts to INT64_MIN and gcd(INT64_MIN, cap) == cap, which
    reaches the model as a confident wrong number rather than as missing data.
    Fail loudly instead."""

    @pytest.mark.parametrize("col", ["m", "n", "k", "layout", "dtype"])
    def test_nan_in_a_required_column_raises(self, col):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df[col] = np.nan if col in ("m", "n", "k") else None
        with pytest.raises(ValueError, match="contains null"):
            fe.extract_batch(df)

    @pytest.mark.parametrize("col", ["m", "n", "k", "layout", "dtype"])
    def test_missing_required_column_raises(self, col):
        """Match the guard's own message: pandas raises a bare KeyError from the
        BASE extract_batch before this guard is reached, so an unmatched
        pytest.raises(KeyError) passes with the guard deleted."""
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(KeyError, match="required to compute vector-width"):
            fe.extract_batch(_frame().drop(columns=[col]))

    def test_unknown_layout_raises(self):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df["layout"] = "zzz"
        with pytest.raises(ValueError, match="layout"):
            fe.extract_batch(df)

    def test_unknown_dtype_raises(self):
        """codegen raises KeyError on an unknown dtype; defaulting to a
        plausible width would silently fabricate legal widths."""
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df["dtype"] = "float128"
        with pytest.raises(KeyError):
            fe.extract_batch(df)

    def test_empty_frame_is_well_formed(self):
        fe = GemmUniversalVecFeatureEngine()
        assert fe.extract_batch(_frame().iloc[0:0]).shape == (
            0,
            len(fe.get_feature_names()),
        )


class TestWidthEncoding:
    """Parity alone cannot catch a self-consistent wrong encoding: mapping a
    width to the wrong number agrees across both paths while inverting the
    feature. Assert values, not just agreement."""

    def test_zero_resolves_to_the_kernels_native_width(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(), _kernel((0, 0, 0)))[-6:]
        assert list(v[:3]) == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(v[3:5], [1.0, 1.0])

    def test_default_epilogue_native_c_is_one_element_not_the_legal_width(self):
        """THE case that motivated resolving native widths from the kernel.
        For row-major C the default epilogue stores one element at a time
        (DefaultGemm2DEpilogue::GetVectorSizeC) no matter how well aligned N is.
        Substituting the problem's legal width recorded vec_frac_c = 1.0 and
        told the model a single-element store was a full-width one."""
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)  # gcd(1536, 8) == 8
        cs = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "cshuffle"})[-1]
        de = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "default"})[-1]
        assert cs == pytest.approx(1.0), "cshuffle reaches the legal width here"
        assert de == pytest.approx(0.125), "default epilogue stores 1 of 8"

    def test_the_epilogue_changes_only_the_c_fraction(self):
        """A and B are loads; the epilogue governs the C store alone."""
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)
        cs = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "cshuffle"})[-6:]
        de = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "default"})[-6:]
        np.testing.assert_allclose(cs[3:5], de[3:5])
        assert cs[5] != de[5]

    def test_a_fixed_width_ignores_the_epilogue(self):
        """Only the native case consults the kernel geometry."""
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)
        cs = fe.extract(p, {**_kernel((4, 4, 4)), "epilogue": "cshuffle"})[-6:]
        de = fe.extract(p, {**_kernel((4, 4, 4)), "epilogue": "default"})[-6:]
        np.testing.assert_allclose(cs, de)

    def test_explicit_widths_are_not_collapsed(self):
        """Guards the other direction: a real width must not read as native."""
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=512), _kernel((1, 1, 1)))[-6:]
        assert list(v[:3]) == [1.0, 1.0, 1.0]
        assert v[3] == pytest.approx(0.125)

    def test_negative_width_raises(self):
        from feature_engine_vec import _width

        with pytest.raises(ValueError, match=">= 0"):
            _width(-4)

    def test_absent_key_raises_on_the_scalar_path(self):
        """extract_batch raises on an absent column; the scalar path must agree.
        It is production inference (predict._predict_single), and search.py
        generates candidates from a parameter space with no width axes -- so a
        native default would score every candidate as native and return a
        confident ranking instead of an error."""
        fe = GemmUniversalVecFeatureEngine()
        kn = {k: v for k, v in _kernel().items() if k != "vec_a"}
        with pytest.raises(KeyError, match="absent from both"):
            fe.extract(_problem(), kn)

    def test_c_width_uses_the_output_dtype_not_the_input(self):
        """fp8 accumulates out to fp16, so C holds 8 elements per 16 bytes where
        A and B hold 16. Using the input dtype for all three reports a kernel at
        the widest legal C width as being at half of it."""
        fe = GemmUniversalVecFeatureEngine()
        # rcr contiguous dims are (k, k, n). K=16 makes the A/B cap visible:
        # gcd(16, 16) = 16 under the fp8 cap but gcd(16, 8) = 8 under the fp16
        # (output) cap, so an operand wired to the wrong caps array changes the
        # answer. N=1536 gives C gcd(1536, 8) = 8 against gcd(1536, 16) = 16.
        p = _problem(m=1792, n=1536, k=16, dtype="fp8")
        scalar = fe.extract(p, _kernel((16, 16, 8)))[-6:]
        batch = fe.extract_batch(
            _frame(m=1792, n=1536, k=16, dtype="fp8", vec=(16, 16, 8))
        )[0][-6:]
        np.testing.assert_allclose(scalar, batch)
        # A and B take the INPUT dtype's cap, so 16 is the widest legal.
        assert scalar[3] == pytest.approx(1.0), "vec_a=16 is the widest legal fp8 A"
        assert scalar[4] == pytest.approx(1.0), "vec_b=16 is the widest legal fp8 B"
        # C takes the OUTPUT dtype's cap: 8, not 16.
        assert scalar[5] == pytest.approx(1.0), "vec_c=8 is the widest legal fp8 C"

    def test_b_operand_does_not_use_the_c_cap(self):
        """Pins the A/B side of the split: at K=16 the A/B cap (16) and the C
        cap (8) give different legal widths, so an operand reading the wrong
        caps array is visible by value."""
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(m=1792, n=1536, k=16, dtype="fp8"), _kernel((8, 8, 8)))
        assert v[-3] == pytest.approx(0.5), "vec_a=8 is half the legal 16"
        assert v[-2] == pytest.approx(0.5), "vec_b=8 is half the legal 16"
        assert v[-1] == pytest.approx(1.0), "vec_c=8 is all of the legal 8"

    def test_bf16_c_width_is_unchanged(self):
        """bf16 accumulates out to bf16, so the output-dtype fix must be a no-op
        there -- otherwise it would silently move every existing bf16 feature."""
        from feature_engine_vec import _out_dtype

        assert _out_dtype("bf16") == "bf16"


class TestRequiredVectorColumns:
    """A dataset with no vec columns is not one this engine can be used on.
    Defaulting them to native made the engine a silent no-op on any parquet
    whose converter does not parse the _vec suffix."""

    @pytest.mark.parametrize("col", ["vec_a", "vec_b", "vec_c"])
    def test_missing_vec_column_raises(self, col):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(KeyError, match="needs the fixed vector widths"):
            fe.extract_batch(_frame().drop(columns=[col]))


class TestLayoutValidation:
    """extract and extract_batch must agree on which layouts exist. The scalar
    path is production inference (predict._predict_single), and
    gemm_contiguous_dims indexes the string positionally without validating, so
    an unknown layout silently yields fabricated contiguous dims."""

    def test_scalar_rejects_unknown_layout(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="layout"):
            fe.extract(_problem(layout="zzz"), _kernel((4, 4, 8)))

    def test_batch_rejects_unknown_layout(self):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df["layout"] = "zzz"
        with pytest.raises(ValueError, match="layout"):
            fe.extract_batch(df)

    def test_uppercase_layout_is_rejected_not_mis_resolved(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="layout"):
            fe.extract(_problem(layout="RCR"), _kernel((4, 4, 8)))


class TestExtentValidation:
    def test_zero_extent_raises_scalar(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match=">= 1"):
            fe.extract(_problem(k=0), _kernel((4, 4, 8)))

    def test_zero_extent_raises_batch(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match=">= 1"):
            fe.extract_batch(_frame(k=0))


class TestScalarRequiresLayoutAndDtype:
    """_required() replaced defaults of "rcr" and "fp8". Nothing exercised the
    absent case, so restoring either default left the whole suite green -- and
    the scalar path is production inference, where a fabricated layout silently
    computes all three fractions for a problem that never said it was rcr."""

    @pytest.mark.parametrize("key", ["layout", "dtype"])
    def test_absent_from_both_dicts_raises(self, key):
        fe = GemmUniversalVecFeatureEngine()
        problem = {k: v for k, v in _problem().items() if k != key}
        kernel = {k: v for k, v in _kernel().items() if k != key}
        with pytest.raises(KeyError, match="required to compute vector-width"):
            fe.extract(problem, kernel)

    @pytest.mark.parametrize("key", ["layout", "dtype"])
    def test_supplied_on_the_kernel_side_is_accepted(self, key):
        """The scalar path is handed two dicts; either may carry the value."""
        fe = GemmUniversalVecFeatureEngine()
        problem = {k: v for k, v in _problem().items() if k != key}
        kernel = {**_kernel(), key: _problem()[key]}
        assert fe.extract(problem, kernel).shape[0] == len(fe.get_feature_names())


class TestLayoutSetTracksTheBaseEngine:
    """_LAYOUTS is narrowed to the four the rest of the package supports, and
    the reason is a cross-module invariant: widening it alone pairs a correct C
    width taken from M with a base `layout` feature LAYOUT_MAP encodes as rcr."""

    def test_matches_the_base_engines_layout_map(self):
        from feature_engine import LAYOUT_MAP
        from feature_engine_vec import _LAYOUTS

        assert _LAYOUTS == frozenset(LAYOUT_MAP), (
            "_LAYOUTS, LAYOUT_MAP and parse_kernel_name must widen together"
        )

    def test_the_parser_accepts_exactly_those_layouts(self):
        from data_pipeline import parse_kernel_name
        from feature_engine_vec import _LAYOUTS

        tail = (
            "compv3_cshuffle_intrawave_False_False_False_True"
            "_128x128x128_2x2x1_32x32x16"
        )
        for layout in _LAYOUTS:
            assert parse_kernel_name(f"gemm_universal_bf16_{layout}_{tail}"), layout
        assert parse_kernel_name(f"gemm_universal_bf16_rcc_{tail}") == {}


class TestBaseBlockIsUnperturbed:
    """The _N_BASE slice assumes the base fills its columns by non-negative
    literal index. A base refactored to negative indexing (result[:, -12:] = hw)
    would write hardware features into the vec columns, and parity -- which
    compares subclass to subclass -- would not notice."""

    def test_the_first_n_base_columns_equal_the_base_engines_output(self):
        df = _frame(m=1792, n=300, k=20226)
        base = GemmUniversalFeatureEngine().extract_batch(df)
        vec = GemmUniversalVecFeatureEngine().extract_batch(df)
        n = GemmUniversalVecFeatureEngine._N_BASE
        assert base.shape[1] == n
        np.testing.assert_allclose(vec[:, :n], base, rtol=0, atol=0)

    def test_the_scalar_path_agrees_too(self):
        p, kn = _problem(), _kernel((4, 4, 8))
        n = GemmUniversalVecFeatureEngine._N_BASE
        np.testing.assert_allclose(
            GemmUniversalVecFeatureEngine().extract(p, kn)[:n],
            GemmUniversalFeatureEngine().extract(p, kn),
            rtol=0,
            atol=0,
        )


class TestNativeWidthsAreKernelDerived:
    """The four properties the native path depends on, each pinned on a geometry
    that can actually tell right from wrong. The default fixture saturates at
    (8, 8, *) and is identical under a waves<->warp_tile swap or an arch change,
    so every one of these survived mutation against it."""

    P = dict(m=1792, n=1536, k=7168, layout="rcr", dtype="bf16")

    def _native(self, **over):
        """Resolve through codegen directly, as the oracle."""
        from codegen_common import gemm_native_vector_sizes

        g = {**DISCRIMINATING, **over}
        return gemm_native_vector_sizes(
            dtype_a=over.get("dtype", "bf16"),
            dtype_b=over.get("dtype", "bf16"),
            dtype_c=over.get("dtype_c", over.get("dtype", "bf16")),
            layout="rcr",
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(g["warp_m"], g["warp_n"], g["warp_k"]),
            warp_tile=(g["warp_tile_m"], g["warp_tile_n"], g["warp_tile_k"]),
            gfx_arch=over.get("arch", "gfx950"),
            epilogue=over.get("epilogue", "default"),
        )

    def test_the_geometry_actually_discriminates(self):
        """Guard the guard: if this ever stops holding, the tests below go
        vacuous without failing."""
        from codegen_common import gemm_native_vector_sizes as nat

        g = DISCRIMINATING
        kw = dict(
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_c="bf16",
            layout="rcr",
            gfx_arch="gfx950",
            epilogue="default",
        )
        right = nat(
            **kw,
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(2, 2, 1),
            warp_tile=(32, 32, 16),
        )
        swapped = nat(
            **kw,
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(32, 32, 16),
            warp_tile=(2, 2, 1),
        )
        assert right != swapped, "fixture no longer detects a waves/warp_tile swap"
        wave32 = nat(
            **{**kw, "gfx_arch": "gfx1250"},
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(2, 2, 1),
            warp_tile=(32, 32, 16),
        )
        assert right != wave32, "fixture no longer detects an arch change"

    def test_waves_and_warp_tile_are_not_transposed(self):
        """The module's own comment calls this the hazard: the engine's warp_m
        is warps-per-block (codegen's `waves`) and warp_tile_m is codegen's
        `warp_tile`. A swap yields a native width for a different kernel."""
        fe = GemmUniversalVecFeatureEngine()
        legal = _legal(**self.P)
        got = fe.extract(_problem(**self.P), _kernel((0, 0, 0), **DISCRIMINATING))[-3:]
        exp = [nat / lim for nat, lim in zip(self._native(), legal)]
        np.testing.assert_allclose(got, exp)

    @pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8", "int8"])
    def test_native_c_uses_the_output_dtype(self, dtype):
        """The native path must apply the same A/B-vs-C dtype split as the fixed
        path. Every other native test is bf16, where _out_dtype is the identity,
        so passing the input dtype through survived there."""
        from feature_engine_vec import _out_dtype

        fe = GemmUniversalVecFeatureEngine()
        P = {**self.P, "dtype": dtype}
        got = fe.extract(
            _problem(**P), _kernel((0, 0, 0), **DISCRIMINATING, epilogue="cshuffle")
        )[-3:]
        exp_nat = self._native(
            dtype=dtype, dtype_c=_out_dtype(dtype), epilogue="cshuffle"
        )
        exp = [nat / lim for nat, lim in zip(exp_nat, _legal(**P))]
        np.testing.assert_allclose(got, exp)

    @pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
    def test_the_rows_arch_selects_the_wavefront_size(self, arch):
        """Native widths go through block_size, which depends on the wavefront
        size, which the gfx family selects. Nothing varied arch before."""
        fe = GemmUniversalVecFeatureEngine()
        got = fe.extract(
            _problem(**self.P) | {"arch": arch},
            _kernel((0, 0, 0), **DISCRIMINATING),
        )[-3:]
        exp = [nat / lim for nat, lim in zip(self._native(arch=arch), _legal(**self.P))]
        np.testing.assert_allclose(got, exp)

    def test_a_wave32_arch_differs_from_a_wave64_one(self):
        """Pins that the parametrize above is not vacuous."""
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        a = fe.extract(_problem(**self.P) | {"arch": "gfx950"}, k)[-3:]
        b = fe.extract(_problem(**self.P) | {"arch": "gfx1250"}, k)[-3:]
        assert list(a) != list(b)


class TestHeterogeneousBatch:
    """extract_batch resolves native widths once per distinct geometry and
    scatters them back by factorize codes. Every existing batch test used a
    frame of identical kernels, so broadcasting row 0 to every row -- or
    reversing the scatter -- produced the same answer and survived."""

    def _mixed(self):
        rows = [
            _row(m=1792, n=1536, k=7168, epilogue="default"),
            _row(m=1792, n=1536, k=7168, epilogue="cshuffle"),
            _row(m=1792, n=1536, k=7168, **DISCRIMINATING),
            _row(m=1792, n=1536, k=7168, **DISCRIMINATING, epilogue="cshuffle"),
            _row(m=1792, n=1536, k=7168, arch="gfx1250", **DISCRIMINATING),
            _row(m=512, n=300, k=20226, layout="ccr", vec=(4, 4, 8)),
        ]
        return pd.DataFrame(rows), rows

    def test_each_row_matches_its_own_scalar_extraction(self):
        fe = GemmUniversalVecFeatureEngine()
        df, rows = self._mixed()
        batch = fe.extract_batch(df)
        for i, row in enumerate(rows):
            np.testing.assert_allclose(
                batch[i],
                fe.extract(row, row),
                rtol=0,
                atol=1e-12,
                err_msg=f"row {i} disagrees with the scalar path",
            )

    def test_the_rows_are_not_all_the_same(self):
        """Guard the guard: a frame of identical rows would make the parity
        assertion above pass under a broadcast bug."""
        fe = GemmUniversalVecFeatureEngine()
        df, _ = self._mixed()
        fracs = {tuple(r) for r in fe.extract_batch(df)[:, -3:]}
        assert len(fracs) >= 3, (
            f"frame is too homogeneous to detect a regroup bug: {fracs}"
        )


class TestRequiredColumns:
    """Each load-bearing column, individually.

    The list is written out rather than taken from ``_REQUIRED``: parametrizing
    over the constant under test means deleting a name from it silently deletes
    its test case instead of failing one. Verified by mutation -- dropping
    "epilogue" from _REQUIRED left the self-referential version green."""

    EXPECTED = (
        "m",
        "n",
        "k",
        "layout",
        "dtype",
        "arch",
        "epilogue",
        "vec_a",
        "vec_b",
        "vec_c",
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_k",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
    )

    def test_the_constant_lists_exactly_these(self):
        assert set(GemmUniversalVecFeatureEngine._REQUIRED) == set(self.EXPECTED)

    @pytest.mark.parametrize("col", EXPECTED)
    def test_dropping_any_required_column_raises_naming_it(self, col):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(KeyError, match=col):
            fe.extract_batch(_frame().drop(columns=[col]))


class TestInputValidation:
    """arch and epilogue became load-bearing; neither is validated by codegen."""

    #: Real targets, including the two-digits-plus-letter shape (gfx90a is
    #: MI210) that a three-digit pattern would wrongly reject.
    REAL_ARCHES = (
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1201",
        "gfx1250",
        "gfx950:xnack-",
    )

    @pytest.mark.parametrize("arch", REAL_ARCHES)
    def test_every_supported_arch_is_accepted(self, arch):
        """Validation must not reject a part this repo builds for."""
        fe = GemmUniversalVecFeatureEngine()
        assert (
            fe.extract(_problem() | {"arch": arch}, _kernel((0, 0, 0))).shape[0] == 78
        )

    @pytest.mark.parametrize(
        "arch", ["unknown", "nan", "", "GFX950", "gfx", "gfx8", "gfx9zzz", "gfx12345"]
    )
    def test_an_unrecognised_arch_raises(self, arch):
        """data_pipeline defaults arch to "unknown" and backfills unpopulated
        columns with None. normalize_gfx_arch only strips suffixes, and the
        wavefront size is chosen by startswith("gfx9"), so an unknown value
        silently yields wave-32 widths on a wave-64 part."""
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="architecture"):
            fe.extract(_problem() | {"arch": arch}, _kernel((0, 0, 0)))

    @pytest.mark.parametrize("epi", ["cshufle", "comp_tdm", "", "gemm"])
    def test_an_unmodelled_epilogue_raises(self, epi):
        """Anything _vector_geometry does not name falls into its cshuffle
        branch and reports a wider C store than the kernel performs."""
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="epilogue"):
            fe.extract(_problem(), _kernel((0, 0, 0), epilogue=epi))

    def test_the_batch_path_validates_too(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="architecture"):
            fe.extract_batch(_frame(arch="unknown"))


class TestGeometryPrecedence:
    """Geometry describes the kernel, so the kernel dict wins."""

    def test_the_kernels_geometry_beats_the_problems(self):
        fe = GemmUniversalVecFeatureEngine()
        p = {**_problem(m=1792, n=1536, k=7168), "tile_m": 999, "warp_tile_n": 999}
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        np.testing.assert_allclose(
            fe.extract(p, k), fe.extract(_problem(m=1792, n=1536, k=7168), k)
        )

    def test_a_missing_geometry_knob_raises_rather_than_defaulting(self):
        """A guessed tile yields a native width for a kernel that was never
        built, and the caller would never know."""
        fe = GemmUniversalVecFeatureEngine()
        k = {kk: v for kk, v in _kernel((0, 0, 0)).items() if kk != "warp_tile_n"}
        with pytest.raises(KeyError, match="warp_tile_n"):
            fe.extract(_problem(), k)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
