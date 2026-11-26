# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knob-class tests of the gfx950 flattened-pixel direct forward
(``conv_direct_fwd_flex_mfma``, variants ``direct_fwd_flex_mfma`` and
``direct_dgrad_flex_mfma``) and of the tiled dgrad filter transpose.

``test_direct_fwd_flex_mfma.py`` covers the family's default tile; this file has
one row per knob class of the kept AOT grid: the tile shapes (bk 64 / 192, bx
128), the in-block reduction splits (cs 4 / cs 2), the XCD tile walks (xcd 1 /
2 / 3, kmaj), the channel tails (c_tail, flat, k_tail), the positions-inner
sweep (sw), the 128-pixel wave shape (xt8), grouped problems (grp), the
channel-chunk split (split_c up to 16), the dgrad's own filter (wt) and the
strided dgrad in one launch (ds with h1, zf and zl).

CPU: every row builds a kernel in both dtypes, the validator refuses what the
knob classes forbid (with its reason), the AOT grid offers every class and the
expected label counts, grid jobs map back from their identities, and the C++
engine agrees with Python on every row (validator, name, grid, workspace, IR).
GPU (gfx950): each class against a float64 numpy reference of the stored
operands, poisoned outputs, both dtypes: forward rows, the dgrad through the
forward kernel (own filter, or the tiled transpose first), the strided dgrad,
and the tiled filter transpose itself.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_fwd_flex_knobs.py -v
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.launcher import KernelLauncher, LaunchConfig
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.lanes import BINDING_LANE, import_or_skip_until

from kernels.common.conv_abi import conv_direct_transform_signature
from kernels.common.conv_direct_fwd_flex_mfma import (
    FwdFlexMfmaSpec,
    build_fwd_flex_mfma,
    fwd_flex_mfma_reason,
)
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectTransposeWeightsDgradSpec,
    build_direct_transpose_weights_dgrad,
    dgrad_4c_transposed_problem,
    direct_launch_geometry,
)

from ._direct_conv_numeric import (
    direct_dgrad_reference,
    direct_fwd_reference,
    launch_direct_kernel,
    poisoned,
    stored_uniform,
)

_FWD = "direct_fwd_flex_mfma"
_DGRAD = "direct_dgrad_flex_mfma"
_TRANSPOSE = "direct_dgrad_transpose_rtc"
_SPLIT_REDUCE = "direct_split_reduce"
_DTYPES = ("bf16", "fp16")


def _problem(
    N=2,
    H=19,
    W=21,
    C=128,
    K=136,
    KH=3,
    KW=3,
    PAD=1,
    pad_w=-1,
    stride=1,
    groups=1,
    dil=(1, 1),
    dtype="bf16",
):
    """A forward problem (``C`` / ``K`` per group): X is N x H x W x C."""
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=stride,
        dil_h=dil[0],
        dil_w=dil[1],
        dtype=dtype,
    )


# Forward rows: (id, problem kwargs, spec knobs). One per knob class of the
# kept grid; each is valid as listed. Sizes are small (image edges, pixel and
# channel tails included); the IR does not depend on them.
_FWD_ROWS = [
    # tile shapes
    ("bk64", dict(), dict(bk=64)),
    ("bk192", dict(), dict(bk=192)),
    ("bx128_bk128", dict(), dict(bx=128, bk=128)),
    ("bx128_bk256", dict(), dict(bx=128)),
    # in-block reduction split: 4-way (bx 128, bk 128) and 2-way (bx 128, bk 256)
    ("cs4", dict(C=256), dict(bx=128, bk=128, cs=4)),
    (
        "cs4_xcd_kmaj_s2",
        dict(C=256, stride=2),
        dict(bx=128, bk=128, cs=4, xcd=True, kmaj=True),
    ),
    (
        "cs4_xcd_split2",
        dict(C=256),
        dict(bx=128, bk=128, cs=4, xcd=True, split_c=2),
    ),
    ("cs2", dict(), dict(bx=128, cs=2)),
    (
        "cs2_xcd_kmaj_split2",
        dict(C=256),
        dict(bx=128, cs=2, xcd=True, kmaj=True, split_c=2),
    ),
    # XCD tile walks
    ("xcd", dict(), dict(xcd=True)),
    ("kmaj", dict(), dict(xcd=True, kmaj=True)),
    ("xcd2_sw_kmaj", dict(), dict(bk=128, xcd=2, sw=True, kmaj=True)),
    ("xcd3_sw", dict(), dict(xcd=3, sw=True)),
    # channel tails and the flattened reduction
    ("c_tail", dict(C=200), dict(c_tail=True)),
    ("c_tail_sw_xcd", dict(C=200), dict(bk=128, c_tail=True, sw=True, xcd=True)),
    ("flat", dict(C=200), dict(bk=128, flat=True)),
    ("flat_split2", dict(C=264), dict(bk=192, flat=True, split_c=2)),
    ("k_tail", dict(K=27), dict(bk=64, k_tail=True)),
    ("k_tail_split2", dict(K=36, C=256), dict(bk=64, k_tail=True, split_c=2)),
    # positions-inner sweep, 128-pixel waves
    ("sw", dict(), dict(bk=128, sw=True)),
    ("sw_split2", dict(C=256), dict(sw=True, split_c=2)),
    ("xt8", dict(), dict(xt8=True)),
    # grouped
    ("grp", dict(C=64, K=64, groups=4), dict(bk=64, grp=True)),
    ("grp_c_tail", dict(C=56, K=64, groups=2), dict(bk=64, grp=True, c_tail=True)),
    ("grp_sw_bx128", dict(C=128, K=256, groups=2), dict(bx=128, grp=True, sw=True)),
    # channel-chunk split (fp32 slabs and the fold)
    ("split3", dict(C=192), dict(bk=192, split_c=3)),
    ("split6_s2", dict(C=384, stride=2), dict(bk=128, split_c=6)),
    ("split16", dict(C=1024, H=9, W=11), dict(split_c=16)),
]

# Own-filter rows (wt): the kernel reads a dgrad's filter as it is, so these
# serve a dgrad as a forward over dY. The problem is that forward (over dY);
# the dgrad's channel counts are swapped.
_WT_ROWS = [
    ("wt", dict(), dict(wt=True)),
    ("wt_split3", dict(C=192), dict(bk=192, wt=True, split_c=3)),
    ("wt_bx128", dict(), dict(bx=128, bk=128, wt=True)),
    ("wt_xt8_c_tail", dict(C=200), dict(wt=True, xt8=True, c_tail=True)),
    ("wt_xt8_sw_xcd", dict(), dict(wt=True, xt8=True, sw=True, xcd=True)),
    (
        "cs4_wt_xcd_split2",
        dict(C=256),
        dict(bx=128, bk=128, cs=4, wt=True, xcd=True, split_c=2),
    ),
    (
        "cs2_wt_xcd_kmaj_split2",
        dict(C=256),
        dict(bx=128, cs=2, wt=True, xcd=True, kmaj=True, split_c=2),
    ),
    (
        "grp_sw_wt_bx128",
        dict(C=128, K=256, groups=2),
        dict(bx=128, grp=True, sw=True, wt=True),
    ),
]

# Stride-1 dgrad rows through the forward kernel after the tiled filter
# transpose (no wt). Same problem convention as _WT_ROWS (C, K % 8 == 0).
_TRANSPOSED_ROWS = [
    ("plain", dict(), dict()),
    ("bx128", dict(), dict(bx=128, bk=128)),
    ("c_tail_sw_xcd", dict(C=200), dict(bk=128, c_tail=True, sw=True, xcd=True)),
    ("flat", dict(C=200), dict(bk=128, flat=True)),
    ("split3", dict(C=192), dict(bk=192, split_c=3)),
]

# Strided dgrad in one launch (ds): (id, DGRAD problem kwargs, knobs). The
# problem is the dgrad's: H / W are dX's, cpg is dX's channels, kpg dY's.
_DS_ROWS = [
    ("s2_3x3_wt", dict(stride=2, C=256, K=128), dict(bx=128, bk=128, wt=True)),
    (
        "s2_3x3_p0_wt",
        dict(stride=2, PAD=0, C=256, K=128),
        dict(bx=128, bk=128, wt=True),
    ),
    (
        "s2_2x2_wt",
        dict(stride=2, KH=2, KW=2, PAD=0, C=256, K=128),
        dict(bk=128, wt=True),
    ),
    ("s2_3x3_tiled_transpose", dict(stride=2, C=256, K=128), dict()),
    ("s2_3x3_tiled_transpose_split2", dict(stride=2, C=256, K=128), dict(split_c=2)),
    (
        "s8_1x16_h1",
        dict(KH=1, KW=16, PAD=0, pad_w=4, stride=8, H=1, W=64, C=128, K=128),
        dict(bk=128, wt=True, h1=True),
    ),
    (
        "s2_1x1_zf",
        dict(KH=1, KW=1, PAD=0, stride=2, C=256, K=128),
        dict(bk=192, wt=True, zf=True),
    ),
    (
        "s2_1x1_zf_zl8",
        dict(KH=1, KW=1, PAD=0, stride=2, C=256, K=128),
        dict(bx=128, wt=True, zf=True, zl=8),
    ),
    (
        "s2_1x1_zf_zl8_c_tail",
        dict(KH=1, KW=1, PAD=0, stride=2, C=256, K=264),
        dict(bx=128, wt=True, zf=True, zl=8, c_tail=True),
    ),
]


def _fwd_spec(row, dtype):
    _, pkw, knobs = row
    return FwdFlexMfmaSpec(problem=_problem(dtype=dtype, **pkw), **knobs)


def _ds_spec(row, dtype):
    _, pkw, knobs = row
    return FwdFlexMfmaSpec(problem=_problem(dtype=dtype, **pkw), ds=True, **knobs)


def _all_rows():
    """Every row as ``(id, spec builder, row)``."""
    return (
        [(r[0], _fwd_spec, r) for r in _FWD_ROWS]
        + [(f"wt_{r[0]}", _fwd_spec, r) for r in _WT_ROWS]
        + [(f"nowt_{r[0]}", _fwd_spec, r) for r in _TRANSPOSED_ROWS]
        + [(f"ds_{r[0]}", _ds_spec, r) for r in _DS_ROWS]
    )


def _knob_specs(dtype):
    return [(rid, build(row, dtype)) for rid, build, row in _all_rows()]


# --------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("rid,build,row", _all_rows(), ids=[r[0] for r in _all_rows()])
def test_rows_are_valid_and_build(rid, build, row, dtype):
    spec = build(row, dtype)
    assert fwd_flex_mfma_reason(spec) is None
    kernel = build_fwd_flex_mfma(spec, arch="gfx950")
    assert kernel.name == spec.kernel_name()
    assert kernel.attrs["max_workgroup_size"] == spec.threads_per_block
    grid, block = spec.launch_geometry()
    assert 1 <= grid[0] <= 256 and block == (spec.threads_per_block, 1, 1)


def test_row_names_are_distinct_per_knob_set():
    """The kernel name carries every knob: rows of different knobs never share
    one (the AOT cache and the run records key on it)."""
    names: dict[str, set] = {}
    for rid, spec in _knob_specs("bf16"):
        knobs = tuple(
            (k, v)
            for k, v in sorted(vars(spec).items())
            if k not in ("problem", "name")
        )
        names.setdefault(spec.kernel_name(), set()).add(knobs)
    assert all(len(knobs) == 1 for knobs in names.values()), names
    tokens = {
        "xcd": "xcd",
        "kmaj": "km",
        "c_tail": "ct",
        "flat": "fr",
        "k_tail": "kx",
        "sw": "sw",
        "xt8": "x8",
        "grp": "gr",
        "wt": "wt",
        "ds": "ds",
        "h1": "h1",
        "zf": "zf",
    }
    for rid, spec in _knob_specs("bf16"):
        parts = spec.kernel_name().split("_")
        for flag, token in tokens.items():
            assert (token in parts) == bool(getattr(spec, flag)), (rid, flag)
        if spec.cs > 1:
            assert f"cs{spec.cs}" in parts, rid
        if spec.zl:
            assert f"zl{spec.zl}" in parts, rid


@pytest.mark.parametrize(
    "rid", ["cs4", "c_tail", "k_tail", "sw", "grp", "xt8", "xcd3_sw", "flat"]
)
def test_ir_is_shape_independent(rid):
    """One binary serves every batch, extent and channel count of a class."""
    from rocke.core.ir_serialize import serialize

    row = next(r for r in _FWD_ROWS if r[0] == rid)
    spec = _fwd_spec(row, "bf16")
    other = replace(
        spec,
        problem=replace(spec.problem, N=3, H=23, W=17),
    )
    assert fwd_flex_mfma_reason(other) is None

    def ir(s):
        return serialize(build_fwd_flex_mfma(s, arch="gfx950")).replace(
            s.kernel_name(), "k"
        )

    assert ir(spec) == ir(other)


def test_knob_class_reasons():
    """The validator refuses what the knob classes forbid, and says why."""
    base = _problem()

    def why(problem=base, arch="gfx950", **knobs):
        return fwd_flex_mfma_reason(FwdFlexMfmaSpec(problem=problem, **knobs), arch)

    assert "bk must be 64, 128, 192 or 256" in why(bk=96)
    assert "bx must be 256, or 128 with bk 128 / 256" in why(bx=64)
    assert "cs must be 1, 2 or 4" in why(cs=3)
    assert "tap count must be a multiple of 3" in why(
        problem=_problem(KH=2, KW=2, PAD=0), bx=128, cs=2
    )
    assert "cs 4: bx 128 and bk 128 only" in why(problem=_problem(C=256), cs=4)
    assert "cs 4: C must be a multiple of 64 and >= 256" in why(
        bx=128, bk=128, cs=4, split_c=2
    )
    assert "xt8: bx 256, bk 128 / 256, cs 1 only" in why(bx=128, xt8=True)
    assert "grp: grouped problems only" in why(grp=True)
    assert "grp: cs 1 and split_c 1" in why(
        problem=_problem(C=64, K=64, groups=4), bk=64, grp=True, split_c=2
    )
    assert "kmaj: needs xcd" in why(kmaj=True)
    assert "xcd: off, on, 2" in why(xcd=4)
    assert "c_tail: C 128 must be a multiple of 8 and not of 64" in why(c_tail=True)
    assert "C 200 must be a multiple of 64" in why(problem=_problem(C=200))
    assert "k_tail: K 136 must not be a multiple of 8" in why(k_tail=True)
    assert "K 27 must be a multiple of 8" in why(problem=_problem(K=27))
    assert "flat: C 128 must be a multiple of 8 and not of 64" in why(flat=True)
    assert "flat: no c_tail, wt, ds or cs" in why(
        problem=_problem(C=200), c_tail=True, flat=True
    )
    assert "sw: cs 1 only" in why(
        problem=_problem(C=256), sw=True, bx=128, bk=128, cs=4
    )
    assert "must be in [1, C / 64]" in why(split_c=0)
    assert "must be in [1, C / 64]" in why(split_c=3)
    strided_1x1 = _problem(KH=1, KW=1, PAD=0, stride=2, C=256, K=128)
    assert "zf: split_c 1 and wt only" in why(problem=strided_1x1, ds=True, zf=True)
    assert "ds: a dense strided dgrad without dilation" in why(
        problem=_problem(C=256, K=128), ds=True, wt=True
    )
    assert "LDS" in why(problem=_problem(KH=7, KW=7, PAD=3))
    assert why(arch="gfx942") == (
        "fwd_flex_mfma needs the 16-byte LDS DMA (absent on gfx942)"
    )
    assert why(arch="gfx1250") == "fwd_flex_mfma is a wave64 MFMA kernel"
    with pytest.raises(ValueError, match="invalid FwdFlexMfmaSpec"):
        build_fwd_flex_mfma(FwdFlexMfmaSpec(problem=base, bk=96), arch="gfx950")


def test_geometry_of_the_tile_classes():
    """Items per class: pixel tiles x K tiles x groups x split_c; ds adds
    phases; zl adds store-only units to the grid."""
    p = _problem(N=2, H=19, W=21, K=136)
    # 2 x 19 x 21 = 798 pixels: 4 tiles of 256, 7 of 128; K 136: 3/2/1 tiles
    assert FwdFlexMfmaSpec(problem=p).items() == 4 * 1
    assert FwdFlexMfmaSpec(problem=p, bk=64).items() == 4 * 3
    assert FwdFlexMfmaSpec(problem=p, bk=128, bx=128).items() == 7 * 2
    assert FwdFlexMfmaSpec(problem=p, split_c=2).items() == 4 * 2
    g = _problem(C=64, K=64, groups=4)
    assert FwdFlexMfmaSpec(problem=g, bk=64, grp=True).items() == 4 * 4
    assert FwdFlexMfmaSpec(problem=p, split_c=2).workspace_bytes == 4 * 2 * 798 * 136
    assert FwdFlexMfmaSpec(problem=p).workspace_bytes == 0
    ds = replace(
        FwdFlexMfmaSpec(
            problem=_problem(KH=1, KW=1, PAD=0, stride=2, C=256, K=128),
            bx=128,
            wt=True,
            ds=True,
            zf=True,
        ),
    )
    zl = replace(ds, zl=8)
    assert zl.items() == ds.items()
    assert zl.grid()[0] >= ds.grid()[0]
    # a ds spec with split_c keeps its slabs in dX's extents
    ds2 = FwdFlexMfmaSpec(problem=_problem(stride=2, C=256, K=128), ds=True, split_c=2)
    p2 = ds2.problem
    assert ds2.workspace_bytes == 4 * 2 * p2.N * p2.H * p2.W * p2.total_c


def test_dgrad_through_forward_problem():
    """Stride-1 dgrad as a forward over dY: the wt rows' forward problem is the
    transposed dgrad problem, and transposing it again gives the dgrad."""
    for rid, pkw, knobs in _WT_ROWS:
        fwd = _problem(**pkw)
        dgrad = dgrad_4c_transposed_problem(fwd)
        assert (dgrad.cpg, dgrad.kpg) == (fwd.kpg, fwd.cpg), rid
        assert dgrad_4c_transposed_problem(dgrad) == fwd, rid
        assert (dgrad.Ho, dgrad.Wo) == (fwd.H, fwd.W), rid


def test_tiled_transpose_builds_and_launches_in_tiles():
    for dtype in _DTYPES:
        for KH, KW in ((3, 3), (2, 2)):
            dp = _problem(
                KH=KH,
                KW=KW,
                PAD=0 if KH == 2 else 1,
                stride=2,
                C=200,
                K=72,
                dtype=dtype,
            )
            spec = DirectTransposeWeightsDgradSpec(
                problem=dp, runtime_channels=True, tiled=True
            )
            kernel = build_direct_transpose_weights_dgrad(spec, arch="gfx950")
            plain = build_direct_transpose_weights_dgrad(
                replace(spec, tiled=False), arch="gfx950"
            )
            assert kernel.name != plain.name
            grid, block = direct_launch_geometry(spec)
            # one 64 x 64 (k, c) tile per block, 16-byte accesses by 256 threads
            assert grid == (KH * KW, 2, 4) and block == (256, 1, 1)


# ------------------------------------------------------------------------ AOT grid


def _flex_labels():
    """Labels of the gfx950 flex family and its helpers over forward and
    dgrad, both dtypes, by algorithm."""
    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    seen: dict[str, set[str]] = {}
    jobs = []
    for dtype in ("fp16", "bf16"):
        for job in dks.enumerate_direct_jobs(
            arch="gfx950", target=target, directions=("fwd", "dgrad"), dtype=dtype
        ):
            ident = job.identity
            if ident.algorithm in (_FWD, _DGRAD, _TRANSPOSE, _SPLIT_REDUCE):
                seen.setdefault(ident.algorithm, set()).add(ident.label())
                jobs.append(job)
    return seen, jobs


@pytest.fixture(scope="module")
def flex_grid():
    return _flex_labels()


def test_aot_grid_counts(flex_grid):
    """3372 forward and 56 dgrad flex identities over both dtypes, and the
    helpers they need: 6 filter transposes (2 per-element + 4 tiled) and 14
    split-C folds (8 + the new degrees 3 / 6 / 8 / 12 / 16)."""
    seen, _ = flex_grid
    assert len(seen[_FWD]) == 3372
    assert len(seen[_DGRAD]) == 56
    assert len(seen[_TRANSPOSE]) == 6
    assert len(seen[_SPLIT_REDUCE]) == 14


def test_aot_grid_covers_every_knob_class(flex_grid):
    """Each knob class of the rows appears in the kept grid (forward and
    dgrad), every kept knob set is valid on its probe, and no other
    architecture gets the family."""
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        DGRAD_FLEX_KEPT,
        FWD_FLEX_KEPT,
        _flex_knobs,
        _knob_grid,
        grid_caches_variant,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

    fwd = [_flex_knobs(k) for k in FWD_FLEX_KEPT]
    dgrad = [_flex_knobs(k) for k in DGRAD_FLEX_KEPT]
    for flag in (
        "xcd",
        "kmaj",
        "c_tail",
        "wt",
        "k_tail",
        "flat",
        "xt8",
        "grp",
        "sw",
    ):
        assert any(k.get(flag) for k in fwd), flag
    assert {k["bk"] for k in fwd} == {64, 128, 192, 256}
    assert {k.get("bx", 256) for k in fwd} == {128, 256}
    assert {k.get("cs", 1) for k in fwd} == {1, 2, 4}
    assert {k.get("xcd") for k in fwd} >= {None, True, 2}
    assert {k["split_c"] for k in fwd} >= {1, 2, 3, 4, 6, 8, 12, 16}
    for flag in ("h1", "zf", "c_tail", "wt"):
        assert any(k.get(flag) for k in dgrad), flag
    assert any(k.get("zl") for k in dgrad)
    assert any(not k.get("wt") for k in dgrad)  # the tiled-transpose form
    # Every kept dgrad knob set is valid on its probe (the forward ones are
    # checked by test_direct_fwd_flex_mfma.py::test_aot_grid).
    direction, caps_list = DIRECT_CAPABILITIES[_DGRAD]
    assert direction == "dgrad"
    for caps in caps_list:
        for dtype in _DTYPES:
            for knobs in _knob_grid(_DGRAD, "gfx950", dtype, caps):
                probe = probe_problem(caps, dtype, _DGRAD, knobs)
                spec = make_direct_spec(_DGRAD, probe, knobs)
                ok, why = validate_direct_spec(_DGRAD, spec, "gfx950")
                assert ok, (caps, knobs, why)
    for arch in ("gfx942", "gfx1250"):
        assert not grid_caches_variant(_FWD, arch)
        assert not grid_caches_variant(_DGRAD, arch)


def test_aot_grid_helpers_follow_the_kernels(flex_grid):
    """Every split_c > 1 flex kernel has its fold in the grid, every non-wt
    dgrad flex kernel its tiled transpose."""
    _, jobs = flex_grid
    degrees = set()
    folds = set()
    tiled = set()
    for job in jobs:
        ident = job.identity
        knobs = json.loads(ident.knobs) if ident.knobs else {}
        if ident.algorithm in (_FWD, _DGRAD):
            degrees.add((ident.dtype_a, knobs.get("split_c", 1)))
        elif ident.algorithm == _SPLIT_REDUCE:
            folds.add((ident.dtype_a, knobs["split_c"]))
        elif ident.algorithm == _TRANSPOSE and knobs.get("tiled"):
            tiled.add(ident.dtype_a)
    assert {d for d in degrees if d[1] > 1} <= folds
    assert tiled == {"fp16", "bf16"}


def test_aot_grid_jobs_round_trip(flex_grid):
    """Jobs of every knob class build and map back from their identity."""
    from kernels.conv.direct_aot_jobs import direct_job_for_identity

    _, jobs = flex_grid
    by_class: dict[tuple, object] = {}
    for job in jobs:
        ident = job.identity
        if ident.algorithm not in (_FWD, _DGRAD):
            continue
        knobs = json.loads(ident.knobs)
        key = (
            ident.algorithm,
            tuple(sorted(k for k, v in knobs.items() if v is True)),
            knobs.get("cs", 1),
            knobs.get("bx", 256),
            knobs.get("bk"),
        )
        by_class.setdefault(key, job)
    assert len(by_class) >= 40
    for job in by_class.values():
        assert direct_job_for_identity(job.identity).identity == job.identity
    # the helpers too
    for job in jobs:
        if job.identity.algorithm in (_TRANSPOSE, _SPLIT_REDUCE):
            assert direct_job_for_identity(job.identity).identity == job.identity


# ------------------------------------------------------------------ C++ engine


def _describe_specs():
    specs = []
    for dtype in _DTYPES:
        specs += [spec for _, spec in _knob_specs(dtype)]
    base = _problem()
    refused = [
        FwdFlexMfmaSpec(problem=base, bk=96),
        FwdFlexMfmaSpec(problem=base, bx=64),
        FwdFlexMfmaSpec(problem=base, cs=3),
        FwdFlexMfmaSpec(problem=_problem(KH=2, KW=2, PAD=0), bx=128, cs=2),
        FwdFlexMfmaSpec(problem=_problem(C=256), cs=4),
        FwdFlexMfmaSpec(problem=base, bx=128, xt8=True),
        FwdFlexMfmaSpec(problem=base, grp=True),
        FwdFlexMfmaSpec(problem=base, kmaj=True),
        FwdFlexMfmaSpec(problem=base, xcd=4),
        FwdFlexMfmaSpec(problem=base, c_tail=True),
        FwdFlexMfmaSpec(problem=_problem(C=200), c_tail=True, flat=True),
        FwdFlexMfmaSpec(problem=base, k_tail=True),
        FwdFlexMfmaSpec(problem=base, flat=True),
        FwdFlexMfmaSpec(problem=_problem(C=256), sw=True, bx=128, bk=128, cs=4),
        FwdFlexMfmaSpec(problem=base, split_c=0),
        FwdFlexMfmaSpec(problem=_problem(KH=7, KW=7, PAD=3)),
        FwdFlexMfmaSpec(
            problem=_problem(KH=1, KW=1, PAD=0, stride=2, C=256, K=128),
            ds=True,
            zf=True,
        ),
        FwdFlexMfmaSpec(problem=_problem(C=256, K=128), ds=True, wt=True),
        FwdFlexMfmaSpec(
            problem=_problem(stride=2, C=256, K=128), ds=True, wt=True, zl=8
        ),
        FwdFlexMfmaSpec(problem=base, wave_size=32),
    ]
    return specs + refused


def test_knob_describe_matches_python():
    """The C++ validator, kernel name, grid and workspace agree with Python's
    on every knob-class row (forward, own-filter, ds), accepted and refused,
    on three architectures."""
    eng = import_or_skip_until("rocke_library_engine", BINDING_LANE)
    from kernels.conv.engine_bridge import conv_direct_fwd_flex_mfma_spec_to_dict

    for arch in ("gfx950", "gfx942", "gfx1250"):
        for spec in _describe_specs():
            ok, why_c, name, grid, ws = eng.conv_direct_fwd_flex_mfma_describe(
                conv_direct_fwd_flex_mfma_spec_to_dict(spec), arch
            )
            why = fwd_flex_mfma_reason(spec, arch)
            assert ok == (why is None), (arch, spec, why, why_c)
            if why is not None:
                assert why_c == why, (arch, spec, why, why_c)
            else:
                assert tuple(grid) == spec.grid(), (arch, spec)
                assert ws == spec.workspace_bytes, (arch, spec)
            assert name == spec.kernel_name()


def test_knob_classes_lower_identically_on_both_engines():
    """One row per knob class lowers to the same .ll through the C++ engine
    and through Python."""
    import_or_skip_until("rocke_library_engine", BINDING_LANE)
    from kernels.conv.engine_bridge import lower_conv_direct_fwd_flex_mfma

    for rid, spec in _knob_specs("bf16"):
        py = lower_conv_direct_fwd_flex_mfma(spec, arch="gfx950", backend="python")
        cpp = lower_conv_direct_fwd_flex_mfma(spec, arch="gfx950", backend="cpp")
        assert py.llvm_text == cpp.llvm_text, rid


# --------------------------------------------------------------------------- GPU

_TOL = {"bf16": 2e-2, "fp16": 5e-3}


def _compile(spec):
    from rocke import compile_kernel

    return compile_kernel(build_fwd_flex_mfma(spec, arch="gfx950"), arch="gfx950")


def _fold(split_c, dtype, p, ws, out):
    """The split-C fold over the fp32 slabs ``ws`` into ``out``; ``p`` is the
    problem whose output ``out`` is (the fold's grid follows its extents)."""
    from rocke import compile_kernel

    from kernels.common.conv_direct_nongrouped import (
        DirectSplitReduceSpec,
        build_direct_conv_split_reduce,
    )

    rspec = DirectSplitReduceSpec(dtype=dtype, split_c=split_c)
    art = compile_kernel(
        build_direct_conv_split_reduce(rspec, arch="gfx950"), arch="gfx950"
    )
    launch_direct_kernel(
        art, p, (ws, ws, out), grid=rspec.grid(p), block=(rspec.threads, 1, 1)
    )


def _tiled_transpose(dp, w_dev, dtype):
    """``w`` [K][R][S][C] of the dgrad problem ``dp`` through the tiled
    runtime-channel transpose; returns the poisoned-then-written device array
    [C][R][S][K] (flipped taps)."""
    from rocke import compile_kernel

    tspec = DirectTransposeWeightsDgradSpec(
        problem=dp, runtime_channels=True, tiled=True
    )
    art = compile_kernel(
        build_direct_transpose_weights_dgrad(tspec, arch="gfx950"), arch="gfx950"
    )
    out = DeviceArray(poisoned((dp.total_c, dp.KH, dp.KW, dp.kpg), dtype, 0xFF), dtype)
    grid, block = direct_launch_geometry(tspec)
    KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_transform_signature(dtype, True),
    )(
        {
            "A": w_dev.data_ptr(),
            "D": out.data_ptr(),
            "A_bytes": w_dev.nbytes,
            "D_bytes": out.nbytes,
            "p_cpg": dp.cpg,
            "p_kpg": dp.kpg,
        },
        config=LaunchConfig(grid=grid, block=block, fence=True),
    )
    return out


def _check(out_dev, shape, dtype, ref):
    out = storage.decode(out_dev.read(), dtype).reshape(shape)
    assert not np.isnan(out).any()
    rel = peak_relative_error(out, ref, floor=1.0)
    assert rel < _TOL[dtype], rel


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _FWD_ROWS, ids=lambda r: r[0])
def test_forward_knob_class_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the flex MFMA forward")
    spec = _fwd_spec(row, dtype)
    p = spec.problem
    rng = np.random.default_rng(0)
    x = stored_uniform(rng, (p.N, p.H, p.W, p.total_c), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    y_shape = (p.N, p.Ho, p.Wo, p.total_k)
    y = DeviceArray(poisoned(y_shape, dtype, 0xFF), dtype)
    x_dev, w_dev = DeviceArray(x, dtype), DeviceArray(w, dtype)
    art, (grid, block) = _compile(spec), spec.launch_geometry()
    if spec.split_c == 1:
        launch_direct_kernel(art, p, (x_dev, w_dev, y), grid=grid, block=block)
    else:
        ws = DeviceArray(poisoned((spec.workspace_bytes // 4,), "fp32", 0xFF))
        launch_direct_kernel(art, p, (x_dev, w_dev, ws), grid=grid, block=block)
        _fold(spec.split_c, dtype, p, ws, y)
    ref = direct_fwd_reference(p, storage.decode(x, dtype), storage.decode(w, dtype))
    _check(y, y_shape, dtype, ref)


def _run_dgrad_through_forward(row, dtype, own_filter):
    current_device_arch_in(("gfx950",), subject="the flex MFMA dgrad")
    _, pkw, knobs = row
    spec = FwdFlexMfmaSpec(problem=_problem(dtype=dtype, **pkw), **knobs)
    assert spec.wt == own_filter
    fp = spec.problem  # the forward over dY
    dp = dgrad_4c_transposed_problem(fp)  # the dgrad
    rng = np.random.default_rng(1)
    dy = stored_uniform(rng, (dp.N, dp.Ho, dp.Wo, dp.total_k), dtype)
    w = stored_uniform(rng, (dp.total_k, dp.KH, dp.KW, dp.cpg), dtype)
    dx = DeviceArray(poisoned((dp.N, dp.H, dp.W, dp.total_c), dtype, 0xFF), dtype)
    dy_dev, w_dev = DeviceArray(dy, dtype), DeviceArray(w, dtype)
    b_dev = w_dev if own_filter else _tiled_transpose(dp, w_dev, dtype)
    art, (grid, block) = _compile(spec), spec.launch_geometry()
    if spec.split_c == 1:
        launch_direct_kernel(art, fp, (dy_dev, b_dev, dx), grid=grid, block=block)
    else:
        ws = DeviceArray(poisoned((spec.workspace_bytes // 4,), "fp32", 0xFF))
        launch_direct_kernel(art, fp, (dy_dev, b_dev, ws), grid=grid, block=block)
        _fold(spec.split_c, dtype, fp, ws, dx)
    ref = direct_dgrad_reference(
        dp, storage.decode(dy, dtype), storage.decode(w, dtype)
    )
    _check(dx, (dp.N, dp.H, dp.W, dp.total_c), dtype, ref)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _WT_ROWS, ids=lambda r: r[0])
def test_dgrad_through_forward_own_filter(row, dtype):
    _run_dgrad_through_forward(row, dtype, own_filter=True)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _TRANSPOSED_ROWS, ids=lambda r: r[0])
def test_dgrad_through_forward_tiled_transpose(row, dtype):
    _run_dgrad_through_forward(row, dtype, own_filter=False)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _DS_ROWS, ids=lambda r: r[0])
def test_strided_dgrad_in_one_launch(row, dtype):
    current_device_arch_in(("gfx950",), subject="the flex MFMA strided dgrad")
    spec = _ds_spec(row, dtype)
    p = spec.problem  # the dgrad: H / W are dX's
    rng = np.random.default_rng(2)
    dy = stored_uniform(rng, (p.N, p.Ho, p.Wo, p.total_k), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    dx_shape = (p.N, p.H, p.W, p.total_c)
    dx = DeviceArray(poisoned(dx_shape, dtype, 0xFF), dtype)
    dy_dev, w_dev = DeviceArray(dy, dtype), DeviceArray(w, dtype)
    b_dev = w_dev if spec.wt else _tiled_transpose(p, w_dev, dtype)
    art, (grid, block) = _compile(spec), spec.launch_geometry()
    if spec.split_c == 1:
        launch_direct_kernel(
            art, p, (dy_dev, b_dev, dx), grid=grid, block=block, direction="dgrad"
        )
    else:
        ws = DeviceArray(poisoned((spec.workspace_bytes // 4,), "fp32", 0xFF))
        launch_direct_kernel(
            art, p, (dy_dev, b_dev, ws), grid=grid, block=block, direction="dgrad"
        )
        # the fold sums the slabs over a 1 x 1 problem of dX's extents
        fp = replace(p, kpg=p.cpg, KH=1, KW=1, PAD=0, stride=1, pad_w=-1)
        _fold(spec.split_c, dtype, fp, ws, dx)
    ref = direct_dgrad_reference(p, storage.decode(dy, dtype), storage.decode(w, dtype))
    _check(dx, dx_shape, dtype, ref)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "KH,KW,pad,C,K",
    [(3, 3, 1, 200, 72), (2, 2, 0, 72, 200), (3, 3, 1, 64, 64), (1, 1, 0, 136, 40)],
)
def test_tiled_transpose_matches_the_flipped_transpose(KH, KW, pad, C, K, dtype):
    """``W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]``, bit exact, with
    channel counts that are not multiples of the 64 x 64 tile."""
    current_device_arch_in(("gfx950",), subject="the tiled dgrad filter transpose")
    dp = _problem(KH=KH, KW=KW, PAD=pad, stride=2, C=C, K=K, dtype=dtype)
    rng = np.random.default_rng(3)
    w = stored_uniform(rng, (K, KH, KW, C), dtype)
    out = _tiled_transpose(dp, DeviceArray(w, dtype), dtype).read()
    want = np.ascontiguousarray(w[:, ::-1, ::-1, :].transpose(3, 1, 2, 0))
    assert np.array_equal(out.view(np.uint16), want.view(np.uint16))
