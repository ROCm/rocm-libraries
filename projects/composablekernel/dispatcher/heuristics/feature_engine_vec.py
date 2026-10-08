#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Universal-GEMM feature engine extended with fixed vector widths.

The base ``GemmUniversalFeatureEngine`` has no feature for the fixed A/B/C
global vector widths that the GEMM bridge can emit (the ``_vecA_B_C`` kernel
name suffix). Two kernels differing only in those widths therefore produce an
identical feature vector, and a model trained on data that varies them sees the
resulting performance difference as unexplainable noise. On data where the
widths are swept this is not a marginal effect: grouping candidates within a
shape by every knob except the widths leaves groups whose best and worst member
differ by several times.

Six features are added::

    vec_a/b/c         the fixed width per operand, in ELEMENTS, 0 meaning native
    vec_frac_a/b/c    the width actually used over the widest the problem allows

The raw triple alone is not enough because it is **problem-relative**: a kernel
at widths (4, 4, 8) is well matched to a problem whose legal widths are
(4, 4, 8) and strictly wasteful on one allowing (8, 8, 8). The ratio carries
that, and generalises to problems whose absolute widths were never trained on.

Note the raw column mixes a sentinel with an ordinal scale: 0 means *native*,
so it sorts below 1 while usually meaning a wider access. The fraction column is
the one with a clean ordering.

**Native is a property of the KERNEL, not of the problem.** ``0`` resolves to
the width that kernel uses when it fixes none -- a function of tile shape, warps
per block, warp tile, epilogue and architecture, obtained from
``codegen_common.gemm_native_vector_sizes``. It is *not* the widest width the
problem allows: for row-major C the default epilogue stores one element at a
time however well aligned N is. An earlier revision assumed the two were the
same, which recorded ``vec_frac_c = 1.0`` for every native kernel and told the
model a single-element store was a full-width one.

.. warning::
   That makes ``vec_frac_*`` incompatible between models trained before and
   after this correction. A model whose ``feature_spec.json`` predates it was
   fitted on the old encoding; re-train rather than mixing the two. The feature
   NAMES and count are unchanged, so nothing detects the difference
   automatically.

``vec_frac`` can exceed 1.0 when a kernel's fixed width does not divide the
problem's legal width. Such a pairing is rejected by the kernel, so it should
not appear in training data; the value is left unclamped so that if it does
appear, it is visible rather than silently indistinguishable from a legal
exact-fit.

Selecting this engine
---------------------
Pass ``--operation gemm_universal_vec`` to ``train.py``; the trained model
records its engine in ``feature_spec.json`` and :class:`predict.Predictor`
reconstructs it, so ``evaluate.py``/``predict.py`` need no flag.

WHY A SUBCLASS AND NOT AN EDIT
------------------------------
The base engine's feature count and encoding maps are pinned by tests, and
``train.py``'s warm-start path (``check_feature_compatibility``) requires an
exact feature-schema match -- so widening the base invalidates every
already-trained model that uses it. Opting in by class keeps those working.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine import GemmUniversalFeatureEngine

# codegen_common owns the authoritative layout and vector-width rules and
# imports stdlib only, so this costs no third-party dependency. The sys.path
# hop matches ml_heuristic_sweep.py, which reaches into dispatcher/python the
# same way. Importing rather than restating is deliberate: an earlier revision
# mirrored these two helpers plus a dtype table, and the dtype table had already
# drifted from its source before the copy was a day old.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codegen"))

from codegen_common import (
    _VEC_ELEMENT_BYTES,
    CommonTypeMappings,
    gemm_contiguous_dims,
    gemm_native_vector_sizes,
    gemm_problem_vector_sizes,
)


def _dtype_max_width(dtype: str) -> int:
    """Widest vector width for a dtype, from codegen's own table.

    Deliberately unguarded: an unknown dtype raises ``KeyError`` here exactly as
    it does in codegen, rather than defaulting to a plausible-but-wrong width.
    """
    return 16 // _VEC_ELEMENT_BYTES[dtype]


def _out_dtype(dtype: str) -> str:
    """The C operand's dtype, which is not always the A/B dtype.

    A 16-byte vector holds a number of ELEMENTS, so C's widest legal width is set
    by C's element size. fp8 and bf8 accumulate out to fp16 and int8 to int32
    (``CommonTypeMappings.get_output_dtype``), which halves or quarters the C
    width relative to A/B. Passing the input dtype for all three -- as an earlier
    revision did -- leaves ``vec_frac_c`` a factor of 2 (fp8/bf8) or 4 (int8) too
    small on a kernel that is in fact at the widest legal C width, with no error.
    ``gemm_vector_fallback`` passes the output dtype here for the same reason.
    """
    return CommonTypeMappings.get_output_dtype(dtype)


VEC_FEATURES = [
    "vec_a",
    "vec_b",
    "vec_c",
    "vec_frac_a",
    "vec_frac_b",
    "vec_frac_c",
]

#: Layouts this engine understands. gemm_contiguous_dims indexes the string
#: positionally and validates nothing, so "zzz" would resolve to ("m","k","m")
#: and yield fabricated legal widths rather than an error.
#:
#: Deliberately the four the rest of the package supports, not the eight
#: gemm_contiguous_dims can resolve. The BASE engine's LAYOUT_MAP holds these
#: four and encodes anything else as 0 (= rcr) via .fillna, and
#: data_pipeline.parse_kernel_name matches only these four. Admitting the
#: column-major-C forms here would pair a correct C width taken from M with a
#: base `layout` feature that says rcr -- two halves of one vector describing
#: different layouts. Widening this set means widening LAYOUT_MAP and the
#: parser in the same change.
_LAYOUTS = frozenset({"rcr", "rrr", "crr", "ccr"})


def _width(value) -> int:
    """Coerce one stored vector width to an int. 0 means native.

    Shared by the scalar and batch paths so they cannot disagree.

    A MISSING value (``None``, ``""``, ``NaN``, ``pd.NA``) is an error, not a
    synonym for native. Native is written as ``0`` -- ``data_pipeline`` emits
    ``(0, 0, 0)`` for a kernel name with no ``_vec`` suffix -- so the only way a
    null reaches here is that nothing populated the column. ``data_pipeline``
    backfills every canonical column it did not fill with ``None``, so treating
    null as native would turn "the parser never ran" into six constant features
    and a training run that reports normal-looking metrics on data this engine
    cannot model. Distinguishing them per row, rather than rejecting a frame
    whose widths are all null, keeps genuinely all-native data (all zeros)
    usable.
    """
    if value is None or (isinstance(value, str) and value == "") or pd.isna(value):
        raise ValueError(
            "vector width is missing; native must be encoded as 0, not as "
            f"null (got {value!r}). Only data_pipeline.parse_kernel_name "
            "populates vec_a/b/c."
        )
    width = int(value)
    if width < 0:
        raise ValueError(f"vector width must be >= 0 (0 means native), got {value!r}")
    return width


def _vec_widths(problem: dict, kernel: dict) -> list[int]:
    """The three fixed widths, searching kernel then problem.

    Canonical rows are flat, so the batch path sees one namespace; the scalar
    path is handed two dicts by ``predict._predict_single``. Searching both
    keeps the two paths agreeing on where a width may live.

    An ABSENT key raises, matching ``extract_batch``'s required-column guard,
    and so does a present-but-null one -- native is written as ``0``. The
    distinction matters because this is the path production inference takes:
    ``search.py`` generates candidates from the base engine's parameter space,
    which carries no width axes, so defaulting an absent key to native would
    score every candidate in a surrogate search as native-width and report a
    confident ranking rather than an error.
    """
    out = []
    for operand in ("a", "b", "c"):
        key = "vec_" + operand
        for src in (kernel, problem):
            if key in src:
                out.append(_width(src[key]))
                break
        else:
            raise KeyError(
                f"{key!r} absent from both the problem and the kernel config; "
                f"{', '.join(VEC_FEATURES[:3])} are required by this engine. "
                "Pass 0 to mean native (the widest legal width)."
            )
    return out


def _extent(problem: dict, kernel: dict, lower: str) -> int:
    """One problem extent, accepting the same key spellings as the base engine.

    ``feature_engine.extract`` reads ``problem.get("m", problem.get("M", 0))``,
    so the two must agree on spelling or one row yields base features from one
    problem and vec features from another. Unlike the base, a missing or
    non-positive extent raises: ``gcd(0, cap)`` is ``cap``, so a zero extent
    would make every width read as a perfect fit instead of as nonsense.
    """
    for src in (problem, kernel):
        for key in (lower, lower.upper()):
            if key in src and src[key] is not None:
                extent = int(src[key])
                if extent < 1:
                    raise ValueError(
                        f"extent {lower!r} must be >= 1, got {extent}; "
                        "a zero extent makes every vector width look legal"
                    )
                return extent
    raise KeyError(
        f"missing extent {lower!r}/{lower.upper()!r} in problem or kernel; "
        "cannot compute legal vector widths without it"
    )


def _required(problem: dict, kernel: dict, key: str) -> str:
    """A key that must be supplied, searched problem-first then kernel."""
    for src in (problem, kernel):
        if src.get(key) is not None:
            return str(src[key])
    raise KeyError(
        f"{key!r} is required to compute vector-width features and is absent "
        "from both the problem and the kernel config"
    )


#: Geometry keys a native width depends on, in the order
#: gemm_native_vector_sizes takes them. NOTE the warp/wave naming inversion the
#: dispatcher bridge also has to deal with: the feature engine's ``warp_m/n/k``
#: is warps-per-block (codegen's ``waves``), while its ``warp_tile_m/n/k`` is
#: codegen's ``warp_tile``. Reading them the other way round silently produces a
#: native width for a different kernel.
_GEOMETRY_KEYS = (
    ("tile_m", "tile_n", "tile_k"),
    ("warp_m", "warp_n", "warp_k"),
    ("warp_tile_m", "warp_tile_n", "warp_tile_k"),
)


def _native_widths(dtype: str, layout: str, arch: str, epilogue: str, geom) -> tuple:
    """The A/B/C widths this KERNEL uses when it fixes none of them.

    ``0`` in a ``vec_*`` column means "this kernel's native width", which is a
    property of the KERNEL -- tile shape, warps per block, warp tile, epilogue
    and architecture -- not of the problem. It is emphatically NOT "the widest
    width the problem allows": for row-major C the default epilogue stores one
    element at a time (``DefaultGemm2DEpilogue::GetVectorSizeC``) however well
    aligned N is, and even cshuffle caps C at ``warp_tile_n * warp_n``.

    An earlier revision substituted the problem's legal width here, which
    recorded ``vec_frac_c = 1.0`` for every native kernel and so told the model
    a one-element store was a full-width one.
    """
    tile, waves, warp_tile = geom
    return gemm_native_vector_sizes(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=_out_dtype(dtype),
        layout=layout,
        tile=tile,
        waves=waves,
        warp_tile=warp_tile,
        gfx_arch=arch,
        epilogue=epilogue,
    )


def _effective_widths(raw, native, legal) -> list:
    """Width each operand actually uses, and its share of what the problem allows.

    A fixed width is used as given; a native one resolves to the kernel's own.
    The denominator stays the problem's legal width, so the fraction keeps its
    meaning: how much of the available alignment this kernel exploits.
    """
    out = [float(v) for v in raw]
    for v, nat, lg in zip(raw, native, legal):
        out.append(float(nat if v == 0 else v) / float(lg))
    return out


def _geom_value(problem: dict, kernel: dict, key: str) -> int:
    """One geometry knob, kernel-side first.

    These describe the KERNEL, so the kernel dict wins; the batch path sees a
    single flat row and cannot distinguish them anyway. Missing raises rather
    than defaulting: a guessed tile yields a native width for a kernel that was
    never built, and the caller would never know.
    """
    for src in (kernel, problem):
        if src.get(key) is not None:
            return int(src[key])
    raise KeyError(
        f"{key!r} is required to resolve this kernel's native vector widths "
        "and is absent from both the problem and the kernel config"
    )


def _check_layout(layout: str) -> None:
    """Reject a layout neither path can resolve. Shared so they cannot differ."""
    if layout not in _LAYOUTS:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of {sorted(_LAYOUTS)}. "
            "Column-major-C layouts need LAYOUT_MAP and parse_kernel_name "
            "widened first; see the _LAYOUTS comment."
        )


def _legal_widths(
    m: int, n: int, k: int, layout: str, dtype: str
) -> tuple[int, int, int]:
    """Widest legal A/B/C width for this problem, with the layout validated.

    C takes the OUTPUT dtype, not the input dtype -- see :func:`_out_dtype`.
    """
    _check_layout(layout)
    return gemm_problem_vector_sizes(m, n, k, layout, dtype, dtype, _out_dtype(dtype))


class GemmUniversalVecFeatureEngine(GemmUniversalFeatureEngine):
    """``GemmUniversalFeatureEngine`` plus the six vector-width features.

    SUBCLASSING HAZARD, and the reason both methods slice: the base
    ``extract_batch`` sizes its output from ``self.get_feature_names()`` -- the
    OVERRIDDEN method. Widening the name list therefore makes ``super()`` return
    an array that *already* contains this subclass's columns, zero-filled, and
    appending to it yields a vector wider than ``get_feature_names()`` reports.
    The base ``extract`` has no such coupling (it builds a literal list), so the
    two paths disagree. Both are sliced back to the base width to be safe
    against that asymmetry moving.
    """

    #: Fallback architecture when a row does not carry one. A native width
    #: depends on the arch (gfx11/gfx12 WMMA store different C widths), but the
    #: inference paths build candidate dicts from a KernelSpec and a problem,
    #: neither of which has an ``arch`` field -- ml_heuristic_sweep and
    #: create_ml_heuristic both know the arch, they just do not repeat it per
    #: row. Taking it at construction keeps those working without forcing every
    #: caller to widen its dicts; an ``arch`` column still wins where present,
    #: so training data stays self-describing.
    DEFAULT_ARCH = "gfx950"

    def __init__(self, *args, arch: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._arch = arch or self.DEFAULT_ARCH

    #: Base feature count, read from the base class's own implementation so a
    #: subclass override cannot perturb it. Instantiated rather than called
    #: unbound: a future base that derives names from ``self`` would otherwise
    #: raise at import time of this module.
    _N_BASE = len(GemmUniversalFeatureEngine().get_feature_names())

    def get_feature_names(self):
        return GemmUniversalFeatureEngine.get_feature_names(self) + VEC_FEATURES

    # get_parameter_space is deliberately NOT overridden.
    #
    # Adding vec_a/b/c axes to it would be inert and unsafe. Inert because
    # SurrogateSearch is the only consumer that generates candidates from the
    # space, and search.py builds it with no engine (`SurrogateSearch(predictor,
    # strategy=...)`), so it always uses the base engine's space. Unsafe because
    # a legal width is a function of the PROBLEM, and the contract that would
    # have to filter it -- validate_config(config) -- receives no problem, so
    # nothing could reject vec_a=8 on a problem whose legal width is 4 and the
    # search would recommend kernels the kernel itself refuses to launch.
    #
    # This engine is therefore for training and ranking a given candidate pool,
    # not for surrogate search. Making search work would mean teaching
    # SurrogateSearch to sample problem-aware configs, which is a change to
    # search.py, not to this class.

    def _vec_row(self, problem: dict, kernel: dict) -> list[float]:
        # No defaults: extract_batch raises on a missing layout/dtype column, so
        # defaulting here would let the scalar path compute all three fractions
        # for row-col-row on a problem that never said it was row-col-row -- the
        # fabricated-layout result _LAYOUTS exists to prevent.
        layout = _required(problem, kernel, "layout")
        dtype = _required(problem, kernel, "dtype")
        legal = _legal_widths(
            _extent(problem, kernel, "m"),
            _extent(problem, kernel, "n"),
            _extent(problem, kernel, "k"),
            layout,
            dtype,
        )
        raw = _vec_widths(problem, kernel)
        native = _native_widths(
            dtype,
            layout,
            self._arch_for(problem, kernel),
            _required(problem, kernel, "epilogue"),
            tuple(
                tuple(_geom_value(problem, kernel, k) for k in group)
                for group in _GEOMETRY_KEYS
            ),
        )
        return _effective_widths(raw, native, legal)

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        base = super().extract(problem, kernel)[
            : self._N_BASE
        ]  # no-op; see class docstring
        return np.concatenate(
            [base, np.array(self._vec_row(problem, kernel), dtype=np.float64)]
        )

    #: Columns extract_batch needs beyond the base engine's. arch, epilogue and
    #: the nine geometry knobs are required because a native width is a property
    #: of the kernel, not of the problem.
    _REQUIRED = (
        "m",
        "n",
        "k",
        "layout",
        "dtype",
        "epilogue",
        "vec_a",
        "vec_b",
        "vec_c",
        *(k for group in _GEOMETRY_KEYS for k in group),
    )

    def _arch_for(self, problem: dict, kernel: dict) -> str:
        """The architecture for this row: an explicit one wins, else the
        engine's. Never guessed silently from nothing."""
        for src in (problem, kernel):
            if src.get("arch") is not None:
                return str(src["arch"])
        return self._arch

    def _native_width_columns(self, df, layout, dtypes) -> np.ndarray:
        """Per-row native A/B/C widths, resolved once per distinct geometry.

        gemm_native_vector_sizes is pure Python, and a sweep frame is tens of
        thousands of rows over a few thousand distinct kernels, so it is called
        per unique (layout, dtype, arch, epilogue, geometry) tuple rather than
        per row.
        """
        keys = [
            "layout",
            "dtype",
            "arch",
            "epilogue",
            *(k for g in _GEOMETRY_KEYS for k in g),
        ]
        if "arch" not in df.columns:
            df = df.assign(arch=self._arch)
        combos = df[keys].astype({k: np.int64 for g in _GEOMETRY_KEYS for k in g})
        codes, uniques = pd.factorize(
            pd.MultiIndex.from_frame(combos), use_na_sentinel=False
        )
        resolved = np.empty((len(uniques), 3), dtype=np.int64)
        for j, row in enumerate(uniques):
            lay, dt, arch, epi = row[0], row[1], row[2], row[3]
            geom = tuple(
                tuple(int(row[4 + 3 * g + o]) for o in range(3)) for g in range(3)
            )
            resolved[j] = _native_widths(str(dt), str(lay), str(arch), str(epi), geom)
        return resolved[codes]

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        # vec_a/b/c are required, not optional. Defaulting a missing column to
        # native is indistinguishable from a dataset that genuinely has no
        # fixed-width kernels, so the engine would contribute six constant
        # columns and training would report normal-looking metrics on data it
        # cannot model.
        missing = [c for c in ("vec_a", "vec_b", "vec_c") if c not in df.columns]
        if missing:
            raise KeyError(
                f"{missing} absent; {type(self).__name__} needs the fixed vector "
                "widths. Only data_pipeline.parse_kernel_name populates them, "
                "from the kernel name's _vec suffix. Use "
                "GemmUniversalFeatureEngine on data with no width information."
            )
        for col in self._REQUIRED:
            if col not in df.columns:
                raise KeyError(f"{col!r} is required to compute vector-width features")
            if df[col].isna().any():
                # A NaN extent casts to INT64_MIN, and gcd(INT64_MIN, cap) is
                # cap -- a fabricated legal width that reaches the model as a
                # confident number rather than as missing data.
                raise ValueError(
                    f"column {col!r} contains null; cannot derive legal widths. "
                    "data_pipeline backfills columns it did not populate with "
                    "None, so a null width means the kernel name was never "
                    "parsed -- native is written as 0."
                )

        # Validate BEFORE delegating: the base extract_batch indexes df["m"]
        # directly, so on a frame missing a required column pandas raises a bare
        # KeyError('m') from inside the base and these messages never reach the
        # caller.
        layout = df["layout"].astype(str)
        unknown = sorted(set(layout) - _LAYOUTS)
        if unknown:
            raise ValueError(
                f"unknown layout(s) {unknown}; expected {sorted(_LAYOUTS)}"
            )
        if (df[["m", "n", "k"]].to_numpy(dtype=np.int64) < 1).any():
            raise ValueError(
                "extents must be >= 1; a zero extent makes every width look legal"
            )

        # A/B take the input dtype, C the output dtype -- see _out_dtype. Using
        # one caps array for all three understates vec_frac_c by 2x on fp8/bf8
        # and 4x on int8.
        dtypes = df["dtype"].astype(str)
        caps_ab = np.array([_dtype_max_width(d) for d in dtypes], dtype=np.int64)
        caps_c = np.array(
            [_dtype_max_width(_out_dtype(d)) for d in dtypes], dtype=np.int64
        )
        operand_caps = (caps_ab, caps_ab, caps_c)
        dims = {d: df[d].to_numpy(dtype=np.int64) for d in ("m", "n", "k")}

        native = self._native_width_columns(df, layout, dtypes)

        extra = np.empty((len(df), len(VEC_FEATURES)), dtype=np.float64)
        for i, operand in enumerate(("a", "b", "c")):
            which = layout.map(lambda s, i=i: gemm_contiguous_dims(s)[i]).to_numpy()
            extent = np.select(
                [which == "m", which == "n", which == "k"],
                [dims["m"], dims["n"], dims["k"]],
                default=-1,
            )
            if (extent < 0).any():
                raise AssertionError(
                    "gemm_contiguous_dims returned a dim outside m/n/k"
                )
            legal = np.gcd(extent, operand_caps[i])

            raw = np.array([_width(v) for v in df["vec_" + operand]], dtype=np.int64)

            # 0 means native, which is the KERNEL's width, not the problem's --
            # see _native_widths. Substituting `legal` here recorded a full
            # fraction for every native kernel, including default-epilogue ones
            # that store a single element at a time.
            eff = np.where(raw == 0, native[:, i], raw)
            extra[:, i] = raw
            extra[:, 3 + i] = eff / legal

        base = super().extract_batch(df)[
            :, : self._N_BASE
        ]  # load-bearing; see class docstring
        return np.hstack([base, extra])
