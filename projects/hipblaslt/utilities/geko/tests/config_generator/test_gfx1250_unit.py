# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 optimization-parameter and post-processor profiles."""

from __future__ import annotations

import pytest

from geko.config_generator.fork_params import optimization_param as opt_param
from geko.config_generator.fork_params import post_processor as base_pp
from geko.config_generator.fork_params.hw_profiles.gfx1250 import optimization_param as g1250_opt
from geko.config_generator.fork_params.hw_profiles.gfx1250 import post_processor as g1250_pp
from geko.config_generator.fork_params.hw_profiles.gfx1250.cluster_dim import (
    select_cluster_dims,
    tile_grid,
)
from geko.config_generator.mi_designer import MFMAParameters
from geko.config_generator.shared_utils import ForkParameter, SizeContext
from geko.schemas import GemmConfig, GemmType


def _cfg(
    arch: str = "gfx1250",
    dt: str = "H",
    dd: str | None = None,
    cd: str | None = None,
    trans_a: str = "T",
    trans_b: str = "N",
    streamk: bool = False,
    library_type: str = "OOB",
    mx: bool = False,
) -> dict:
    dd = dd or dt
    cd = cd or ("S" if dt not in ("D", "Z") else dt)
    gt = GemmType.from_tensile(trans_a, trans_b, dt, dd, cd)
    gp = GemmConfig(gt, [[16, 16, 1, 16]], mx=mx)
    return {
        "ARCH": arch,
        "StreamK": streamk,
        "LIBRARY_TYPE": library_type,
        "GemmProblem": gp,
    }


@pytest.fixture(autouse=True)
def _no_tensile_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opt_param, "load_tensile_metadata", lambda: {})


# ---------------------------------------------------------------------------
# GFX1250Params (heuristic)
# ---------------------------------------------------------------------------

_SIZES = [(16, 16, 1, 16), (4096, 4096, 1, 8192)]

_HEURISTIC_COMBOS = [
    dict(dt="H", trans_a="T", trans_b="N"),
    dict(dt="H", trans_a="N", trans_b="N"),
    dict(dt="H", trans_a="N", trans_b="T"),
    dict(dt="H", trans_a="T", trans_b="T"),
    dict(dt="H", trans_a="N", trans_b="T", dd="S"),
    dict(dt="F8", trans_a="T", trans_b="N"),
    dict(dt="F4", trans_a="T", trans_b="N"),
    dict(dt="S", trans_a="T", trans_b="N"),
    dict(dt="D", trans_a="T", trans_b="N"),
    dict(dt="X", trans_a="T", trans_b="N"),
]


@pytest.mark.parametrize("combo", _HEURISTIC_COMBOS)
@pytest.mark.parametrize("streamk", [False, True])
def test_gfx1250_heuristic_profile_branches(combo: dict, streamk: bool) -> None:
    profile = g1250_opt.GFX1250Params(_cfg(streamk=streamk, **combo))
    for size in _SIZES:
        params, groups = profile.generate_for_size(size)
        assert isinstance(params, dict)
        assert isinstance(groups, list)


# ---------------------------------------------------------------------------
# GFX1250GAParams (generic) + GFX1250GAOrigamiPolicy
# ---------------------------------------------------------------------------

_GA_COMBOS = [
    dict(arch="gfx1250", dt="H", trans_a="T", trans_b="N", streamk=False, library_type="OOB"),
    dict(arch="gfx1250", dt="H", trans_a="N", trans_b="N", streamk=True, library_type="OOB"),
    dict(arch="gfx1250", dt="H", trans_a="T", trans_b="N", streamk=True, library_type="Equality"),
    dict(arch="gfx1250-strict_96cu", dt="F4", trans_a="T", trans_b="N", streamk=False, library_type="OOB"),
    dict(arch="gfx1250_96cu", dt="F8", trans_a="T", trans_b="N", streamk=True, library_type="Equality", mx=True),
    dict(arch="gfx1250", dt="X", trans_a="T", trans_b="N", streamk=False, library_type="OOB"),
]


@pytest.mark.parametrize("combo", _GA_COMBOS)
def test_gfx1250_ga_profile_branches(combo: dict) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(**combo))
    for size in _SIZES:
        params, groups = profile.generate_for_size(size)
        assert isinstance(params, dict)
        assert isinstance(groups, list)


def test_gfx1250_ga_origami_policy_equality_keeps_non_persistent_entry() -> None:
    profile = g1250_opt.GFX1250GAParams(
        _cfg(dt="H", trans_a="T", trans_b="N", streamk=True, library_type="Equality")
    )
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    entries = profile.execution_policy(ctx)
    assert any(e["TileProcessingStrategy"].values == ["None"] for e in entries)


def test_gfx1250_ga_origami_policy_oob_drops_non_persistent_entry() -> None:
    profile = g1250_opt.GFX1250GAParams(
        _cfg(dt="H", trans_a="T", trans_b="N", streamk=True, library_type="OOB")
    )
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    entries = profile.execution_policy(ctx)
    assert all(e["TileProcessingStrategy"].values != ["None"] for e in entries)
    assert all(e["PersistentXCCMapping"].values == [0] for e in entries)


def test_gfx1250_ga_execution_policy_none_without_streamk() -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(dt="H", streamk=False))
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    assert profile.execution_policy(ctx) is None


# ---------------------------------------------------------------------------
# post_processor.py: free helper functions
# ---------------------------------------------------------------------------


def _mi_entry(mi: list) -> dict:
    return {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=mi)}


def test_wave_tile_ok_short_mi_is_kept() -> None:
    assert g1250_pp._wave_tile_ok(_mi_entry([1, 2, 3])) is True


def test_wave_tile_ok_checks_both_components() -> None:
    good = [16, 16, 32, 1, 1, 8, 8, 2, 2]
    bad = [16, 16, 32, 1, 1, 1, 8, 2, 2]
    assert g1250_pp._wave_tile_ok(_mi_entry(good)) is True
    assert g1250_pp._wave_tile_ok(_mi_entry(bad)) is False


def test_mi_wave_tile_and_wave_group() -> None:
    mi = [16, 16, 32, 1, 2, 8, 4, 2, 2]
    assert g1250_pp._mi_wave_tile(mi) == (8, 4)
    assert g1250_pp._mi_wave_group(mi) == (2, 2)


def test_mi_wave_group_block_bm_defaults_to_one_when_mi4_is_zero() -> None:
    mi = [16, 16, 32, 1, 0, 8, 4, 2, 2]
    # block_bm falls back to 1 when mi[4] is falsy.
    assert g1250_pp._mi_wave_group(mi) == (1, 4)


def test_mis_keeps_only_nine_element_matrix_instructions() -> None:
    good = [16, 16, 32, 1, 1, 8, 8, 2, 2]
    groups = [_mi_entry(good), _mi_entry([1, 2, 3])]
    assert g1250_pp._mis(groups) == [good]


# ---------------------------------------------------------------------------
# _GFX1250WaveTileFilter
# ---------------------------------------------------------------------------


def test_filter_small_wave_tiles_drops_only_bad_entries() -> None:
    good = _mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])
    bad = _mi_entry([16, 16, 32, 1, 1, 1, 8, 2, 2])
    groups = [good, bad]
    f = g1250_pp._GFX1250WaveTileFilter()
    fork, kept = f.filter_small_wave_tiles({}, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert kept == [good]
    assert fork == {}


def test_filter_small_wave_tiles_keeps_all_when_every_entry_is_bad() -> None:
    bad = _mi_entry([16, 16, 32, 1, 1, 1, 8, 2, 2])
    groups = [bad, bad]
    f = g1250_pp._GFX1250WaveTileFilter()
    _, kept = f.filter_small_wave_tiles({}, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert kept is groups


def test_filter_small_wave_tiles_keeps_all_when_every_entry_is_good() -> None:
    good = _mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])
    groups = [good, good]
    f = g1250_pp._GFX1250WaveTileFilter()
    _, kept = f.filter_small_wave_tiles({}, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert kept is groups


# ---------------------------------------------------------------------------
# _GFX1250GeometryNarrowing
# ---------------------------------------------------------------------------


def test_narrow_to_mi_geometry_noop_without_matrix_instructions() -> None:
    n = g1250_pp._GFX1250GeometryNarrowing()
    fork = {"HalfPLR": ForkParameter(name="HalfPLR", values=[0, 1, 2, 3])}
    groups = [_mi_entry([1, 2, 3])]
    f2, g2 = n.narrow_to_mi_geometry(fork, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert f2 is fork and g2 is groups


def test_narrow_to_mi_geometry_restricts_half_plr_and_lds_interleave() -> None:
    # MIWaveTile = (mi[5], mi[6]). Odd/odd so HalfPLR bits 1 and 2 are unusable.
    # MIWaveGroup computed from mi[4], mi[0], mi[7], mi[8].
    mi = [16, 16, 32, 1, 1, 3, 3, 2, 2]
    groups = [_mi_entry(mi)]
    fork = {
        "HalfPLR": ForkParameter(name="HalfPLR", values=[0, 1, 2, 3]),
        "LDSSegmentInterleave": ForkParameter(name="LDSSegmentInterleave", values=[0, 1]),
    }
    n = g1250_pp._GFX1250GeometryNarrowing()
    f2, g2 = n.narrow_to_mi_geometry(fork, groups, SizeContext(M=1, N=1, B=1, K=1))
    # h=0 is always usable; h=1/2/3 need an even component, both are odd here.
    assert f2["HalfPLR"].values == [0]
    assert g2 == groups


def test_narrow_to_mi_geometry_falls_back_when_nothing_usable() -> None:
    mi = [16, 16, 32, 1, 1, 3, 3, 2, 2]
    groups = [_mi_entry(mi)]
    # Only non-zero HalfPLR values on offer; none usable -> falls back to [0].
    fork = {"HalfPLR": ForkParameter(name="HalfPLR", values=[1, 2, 3])}
    n = g1250_pp._GFX1250GeometryNarrowing()
    f2, _ = n.narrow_to_mi_geometry(fork, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert f2["HalfPLR"].values == [0]


def test_narrow_to_mi_geometry_ignores_non_list_or_missing_params() -> None:
    mi = [16, 16, 32, 1, 1, 8, 8, 2, 2]
    groups = [_mi_entry(mi)]
    fork = {"HalfPLR": ForkParameter(name="HalfPLR", values="not-a-list")}
    n = g1250_pp._GFX1250GeometryNarrowing()
    f2, _ = n.narrow_to_mi_geometry(fork, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert f2["HalfPLR"].values == "not-a-list"


# ---------------------------------------------------------------------------
# _GFX1250DropMIGroupGSU
# ---------------------------------------------------------------------------


def test_drop_mi_group_gsu_noop_without_global_split_u() -> None:
    d = g1250_pp._GFX1250DropMIGroupGSU()
    groups = [{"X": ForkParameter(name="X", values=[1])}]
    fork, g2 = d.drop_mi_group_gsu({}, groups, SizeContext(M=1, N=1, B=1, K=1))
    assert g2 is groups


def test_drop_mi_group_gsu_strips_and_dedupes() -> None:
    d = g1250_pp._GFX1250DropMIGroupGSU()
    entry_a = {
        "MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[1]),
        "GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[1]),
    }
    entry_b = {
        "MatrixInstruction": ForkParameter(name="MatrixInstruction", values=[1]),
        "GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[2]),
    }
    fork = {"GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[-1])}
    _, deduped = d.drop_mi_group_gsu(fork, [entry_a, entry_b], SizeContext(M=1, N=1, B=1, K=1))
    assert len(deduped) == 1
    assert "GlobalSplitU" not in deduped[0]


def test_drop_mi_group_gsu_dedupes_despite_differing_comment_metadata() -> None:
    """Groups MIDesign emits for the same geometry carry a per-candidate comment
    and metadata (GSU, LSU, totalGranularity, ...) on the ``MatrixInstruction``
    ForkParameter even when the geometry itself -- the only thing left once
    GlobalSplitU is stripped -- is identical. Real mi_groups from
    MIDesign._to_group_dimension always have distinct, non-empty comment/metadata,
    so a dedupe that only collapses bit-for-bit identical ForkParameters (comment
    and metadata included) never fires in practice and every GSU/DepthU variant
    MIDesign tried for a geometry leaks into the generic search space as a
    separate, duplicate group.
    """
    d = g1250_pp._GFX1250DropMIGroupGSU()
    entry_a = {
        "MatrixInstruction": ForkParameter(
            name="MatrixInstruction", values=[1],
            comment="GSU 48 - totalGranularity 0.75000", metadata={"GSU": 48},
        ),
        "GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[48]),
    }
    entry_b = {
        "MatrixInstruction": ForkParameter(
            name="MatrixInstruction", values=[1],
            comment="GSU 47 - totalGranularity 0.73438", metadata={"GSU": 47},
        ),
        "GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[47]),
    }
    fork = {"GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[-1])}
    _, deduped = d.drop_mi_group_gsu(fork, [entry_a, entry_b], SizeContext(M=1, N=1, B=1, K=1))
    assert len(deduped) == 1


# ---------------------------------------------------------------------------
# _GFX1250ClusterDimCoupling (via the full GA post-processor; needs _make_param)
# ---------------------------------------------------------------------------


def _ga_pp(arch: str = "gfx1250") -> g1250_pp.GFX1250GAPostProcessor:
    return g1250_pp.GFX1250GAPostProcessor(_cfg(arch=arch, dt="H"))


def test_couple_cluster_dim_noop_without_active_cluster_dim() -> None:
    pp = _ga_pp()
    groups = [_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])]
    ctx = SizeContext(M=128, N=128, B=1, K=128)

    # Missing entirely.
    f1, g1 = pp.couple_cluster_dim({}, groups, ctx)
    assert g1 is groups

    # Present but inactive.
    fork = {"ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1], [2, 2]], active=False)}
    f2, g2 = pp.couple_cluster_dim(fork, groups, ctx)
    assert g2 is groups

    # Present, active, but fewer than 2 shapes.
    fork = {"ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1]], active=True)}
    f3, g3 = pp.couple_cluster_dim(fork, groups, ctx)
    assert g3 is groups


def test_couple_cluster_dim_expands_mi_groups_per_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    pp = _ga_pp()
    monkeypatch.setattr(
        base_pp.MIDesign,
        "calculate_mfma_parameters",
        lambda _mi: MFMAParameters(MT0=256, MT1=352, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1),
    )
    groups = [_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])]
    fork = {"ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1], [2, 1], [2, 2], [4, 2], [4, 4]])}
    ctx = SizeContext(M=4096, N=4096, B=1, K=16)

    new_fork, coupled = pp.couple_cluster_dim(fork, groups, ctx)
    assert "ClusterDim" not in new_fork
    assert len(coupled) >= 1
    assert all("ClusterDim" in entry for entry in coupled)


# The published gfx1250 MXFP4/NVFP4 MAF kernel: 512x128 unswapped, 256x256 with
# SourceSwap=True.
_MI_32X16 = [32, 16, 128, 1, 1, 8, 4, 2, 2]
_MI_SQUARE = [16, 16, 128, 1, 1, 8, 8, 2, 2]
_SIZE_4096X3072 = (4096, 3072, 1, 65536)


def _mx_f4_cfg(streamk: bool = False) -> dict:
    return _cfg(arch="gfx1250_96cu", dt="F4", streamk=streamk, mx=True)


def _profile_cluster_shapes() -> list:
    """The ClusterDim shapes the gfx1250 GA profile offers before coupling."""
    params, _ = g1250_opt.GFX1250GAParams(_mx_f4_cfg()).generate_for_size(_SIZE_4096X3072)
    return [list(s) for s in params["ClusterDim"].values]


def _ss(values, active: bool = True) -> ForkParameter:
    return ForkParameter(name="SourceSwap", values=values, active=active)


def _coupled_shapes(mi: list, source_swap=None, entry_source_swap=None) -> list:
    """ClusterDim shapes the GA post-processor couples to one MI entry at 4096x3072."""
    fork = {"ClusterDim": ForkParameter(name="ClusterDim", values=_profile_cluster_shapes())}
    if source_swap is not None:
        fork["SourceSwap"] = source_swap
    entry = _mi_entry(mi)
    if entry_source_swap is not None:
        entry["SourceSwap"] = entry_source_swap
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    _, coupled = _ga_pp().couple_cluster_dim(fork, [entry], ctx)
    return [e["ClusterDim"].values[0] for e in coupled]


def _selected(macro_tile: tuple) -> list:
    grid = tile_grid(4096, 3072, macro_tile)
    return [list(s) for s in select_cluster_dims(_profile_cluster_shapes(), grid, macro_tile)]


def test_source_swapped_macro_tile_matches_tensile_effective_mi() -> None:
    swapped = g1250_pp._source_swapped_macro_tile
    assert base_pp.mi_macro_tile(_mi_entry(_MI_32X16)) == (512, 128)
    assert swapped(_MI_32X16, (512, 128)) == (256, 256)
    assert swapped([16, 32, 128, 1, 1, 4, 8, 2, 2], (128, 512)) == (256, 256)
    assert swapped(_MI_SQUARE, (256, 256)) == (256, 256)


def test_source_swap_values_defaults_and_precedence() -> None:
    values = g1250_pp._source_swap_values
    entry = _mi_entry(_MI_32X16)
    assert values({}, entry) == [False]
    assert values({"SourceSwap": _ss([True], active=False)}, entry) == [False]
    assert values({"SourceSwap": _ss([False, True])}, entry) == [False, True]
    own = {**entry, "SourceSwap": _ss([True])}
    assert values({"SourceSwap": _ss([False, True])}, own) == [True]
    assert values({"SourceSwap": _ss([])}, entry) == [False]
    assert values({"SourceSwap": _ss(True)}, entry) == [True]


def test_couple_cluster_dim_adds_the_source_swap_tile_shapes() -> None:
    unswapped = _selected((512, 128))
    swapped = _selected((256, 256))
    # Among equal-size shapes the least-traffic one differs: [2, 8] at 512x128, [4, 4] at 256x256.
    assert [2, 8] in unswapped and [4, 4] not in unswapped
    assert [4, 4] in swapped and [2, 8] not in swapped

    shapes = _coupled_shapes(_MI_32X16, _ss([False, True]))
    assert [4, 4] in shapes and [2, 8] in shapes
    assert sorted(map(tuple, shapes)) == sorted({tuple(s) for s in unswapped + swapped})
    assert shapes == sorted(shapes, key=lambda s: (s[0] * s[1], s[0]))

    assert _coupled_shapes(_MI_32X16, _ss([True])) == swapped
    # An MI group's own SourceSwap wins over the flat parameter.
    assert _coupled_shapes(_MI_32X16, _ss([False]), _ss([True])) == swapped


def test_couple_cluster_dim_unchanged_without_source_swap_true() -> None:
    unswapped = _selected((512, 128))
    assert _coupled_shapes(_MI_32X16) == unswapped
    assert _coupled_shapes(_MI_32X16, _ss([False])) == unswapped
    assert _coupled_shapes(_MI_32X16, _ss([False, True], active=False)) == unswapped


def test_couple_cluster_dim_square_mi_ignores_source_swap() -> None:
    expected = _selected((256, 256))
    assert _coupled_shapes(_MI_SQUARE, _ss([False, True])) == expected
    assert _coupled_shapes(_MI_SQUARE) == expected


def _shapes_by_mi_and_swap(groups: list) -> dict:
    """ClusterDim shapes per (MI, SourceSwap).

    Any other group field splits an entry into variants; every variant of one
    (MI, SourceSwap) must get the same shapes.
    """
    variants: dict = {}
    for e in groups:
        key = (tuple(e["MatrixInstruction"].values), e["SourceSwap"].values[0])
        rest = tuple(
            (k, repr(fp.values))
            for k, fp in e.items()
            if k not in ("MatrixInstruction", "SourceSwap", "ClusterDim")
        )
        variants.setdefault(key, {}).setdefault(rest, []).append(e["ClusterDim"].values[0])
    out: dict = {}
    for key, per_variant in variants.items():
        shapes = list(per_variant.values())
        assert all(s == shapes[0] for s in shapes)
        out[key] = shapes[0]
    return out


@pytest.mark.parametrize("streamk", [False, True])
def test_ga_postprocessor_ranks_each_source_swap_entry_on_its_own_tile(streamk: bool) -> None:
    """Profile output through the full GA post-processor, as the generator runs it.

    _GFX1250SourceSwapSplit leaves one SourceSwap value per MI entry, so the
    ClusterDim coupling ranks every entry on the tile it builds, not a union.
    """
    cfg = _mx_f4_cfg(streamk)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_SIZE_4096X3072)
    assert True in params["SourceSwap"].values
    groups = [_mi_entry(_MI_SQUARE), _mi_entry(_MI_32X16)]
    new_fork, coupled = g1250_pp.GFX1250GAPostProcessor(cfg).apply(params, groups, _SIZE_4096X3072)
    assert "SourceSwap" not in new_fork and "ClusterDim" not in new_fork
    # Tensile reads group values literally: every one stays single-valued.
    assert all(
        len(fp.values) == 1 for e in coupled for k, fp in e.items() if k != "MatrixInstruction"
    )
    by_key = _shapes_by_mi_and_swap(coupled)
    square = _selected((256, 256))
    assert by_key[(tuple(_MI_SQUARE), False)] == square
    assert by_key[(tuple(_MI_SQUARE), True)] == square
    assert by_key[(tuple(_MI_32X16), False)] == _selected((512, 128))
    assert by_key[(tuple(_MI_32X16), True)] == _selected((256, 256))


@pytest.mark.parametrize("dt", ["F4", "F8"])
@pytest.mark.parametrize("streamk", [False, True])
def test_ga_profile_and_postprocessor_offer_the_published_mxfp4_cluster(
    streamk: bool, dt: str
) -> None:
    """The shipped MXFP4/NVFP4 MAF MI, SourceSwap and cluster are reachable.

    Runs for MX F4 and F8 data. F4 also stands for NVFP4, which differs from
    MXFP4 only in the MX block size and scale type; the GA profile and these
    post-process steps only check whether the problem is MX.

    [32, 16, 128, 1, 1, 8, 4, 2, 2] with SourceSwap=True builds 256x256 and
    ships with ClusterDim [4, 4]; with SourceSwap=False it builds 512x128, where
    [2, 8] is the least-traffic shape of that cluster size at 4096x3072.
    """
    cfg = _cfg(arch="gfx1250_96cu", dt=dt, streamk=streamk, mx=True)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_SIZE_4096X3072)
    pp = g1250_pp.GFX1250GAPostProcessor(cfg)
    new_fork, groups = pp.apply(params, [_mi_entry(_MI_32X16)], _SIZE_4096X3072)
    assert "SourceSwap" not in new_fork
    by_key = _shapes_by_mi_and_swap(groups)
    swapped = by_key[(tuple(_MI_32X16), True)]
    unswapped = by_key[(tuple(_MI_32X16), False)]
    assert swapped == _selected((256, 256))
    assert [4, 4] in swapped and [2, 8] not in swapped
    assert unswapped == _selected((512, 128))
    assert [2, 8] in unswapped and [4, 4] not in unswapped


def _nvfp4_cfg(streamk: bool, library_type: str = "OOB") -> dict:
    gt = GemmType.from_tensile("T", "N", "F4", "F4", "S")
    gp = GemmConfig(gt, [[16, 16, 1, 16]], mx=True, mx_block=16, mx_scale_type="F8")
    return {"ARCH": "gfx1250_96cu", "StreamK": streamk, "LIBRARY_TYPE": library_type, "GemmProblem": gp}


_MX_CFGS = {
    "mxfp4": lambda sk, lib: _cfg(arch="gfx1250_96cu", dt="F4", streamk=sk, library_type=lib, mx=True),
    "nvfp4": _nvfp4_cfg,
    "mxfp8": lambda sk, lib: _cfg(arch="gfx1250_96cu", dt="F8", streamk=sk, library_type=lib, mx=True),
}


def _execution_policy(groups: list) -> list:
    found = [g for g in groups if g and "TileProcessingStrategy" in g[0]]
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("library_type", ["OOB", "Equality"])
@pytest.mark.parametrize("kind", sorted(_MX_CFGS))
def test_ga_profile_searches_shadow_init_on_non_persistent_mx(kind: str, library_type: str) -> None:
    """Without StreamK, MX problems search ForceDisableShadowInit [False, True] as a flat parameter."""
    cfg = _MX_CFGS[kind](False, library_type)
    params, groups = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_SIZE_4096X3072)
    fp = params["ForceDisableShadowInit"]
    assert fp.values == [False, True] and fp.active
    assert not any("TileProcessingStrategy" in g[0] for g in groups if g)
    new_fork, _ = g1250_pp.GFX1250GAPostProcessor(cfg).apply(params, [_mi_entry(_MI_32X16)], _SIZE_4096X3072)
    assert new_fork["ForceDisableShadowInit"].values == [False, True]


@pytest.mark.parametrize("library_type", ["OOB", "Equality"])
@pytest.mark.parametrize("kind", sorted(_MX_CFGS))
def test_ga_profile_disables_shadow_init_on_persistent_mx(kind: str, library_type: str) -> None:
    """With StreamK, persistent MX entries pin ForceDisableShadowInit True; a None entry searches both.

    With shadow init on, some persistent MX kernels fail to compile in KernelWriter
    and keep the unpaired LDSSegmentInterleave layout.
    """
    cfg = _MX_CFGS[kind](True, library_type)
    params, groups = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_SIZE_4096X3072)
    assert "ForceDisableShadowInit" not in params
    policy = _execution_policy(groups)
    assert all(len(fp.values) == 1 for e in policy for fp in e.values())
    persistent = [e for e in policy if e["TileProcessingStrategy"].values != ["None"]]
    plain = [e for e in policy if e["TileProcessingStrategy"].values == ["None"]]
    assert persistent and all(e["ForceDisableShadowInit"].values == [True] for e in persistent)
    if library_type == "OOB":
        assert not plain
    else:
        assert sorted(e["ForceDisableShadowInit"].values[0] for e in plain) == [False, True]
        assert all(len(e) == 2 for e in plain)


@pytest.mark.parametrize("streamk", [False, True])
@pytest.mark.parametrize("dt", ["B", "F8", "H"])
def test_ga_profile_leaves_shadow_init_default_for_non_mx(dt: str, streamk: bool) -> None:
    """Non-MX problems emit no ForceDisableShadowInit, so Tensile's default applies."""
    cfg = _cfg(arch="gfx1250_96cu", dt=dt, streamk=streamk, library_type="Equality")
    params, groups = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_SIZE_4096X3072)
    assert "ForceDisableShadowInit" not in params
    assert not any("ForceDisableShadowInit" in e for g in groups for e in g)


# ---------------------------------------------------------------------------
# _GFX1250SourceSwapSplit
# ---------------------------------------------------------------------------


def test_mi_is_square() -> None:
    assert g1250_pp._mi_is_square(_mi_entry(_MI_SQUARE)) is True
    assert g1250_pp._mi_is_square(_mi_entry(_MI_32X16)) is False
    assert g1250_pp._mi_is_square(_mi_entry([16, 32, 128, 1, 1, 4, 8, 2, 2])) is False
    # Unreadable entries count as square, so the split never acts on them.
    assert g1250_pp._mi_is_square(_mi_entry([16])) is True
    assert g1250_pp._mi_is_square({}) is True


def _cd(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="ClusterDim", values=values, active=active)


def test_source_swap_split_noop_cases() -> None:
    pp = _ga_pp()
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    mixed = [_mi_entry(_MI_SQUARE), _mi_entry(_MI_32X16)]
    cd = _cd([[1, 1], [2, 2]])

    # Missing entirely.
    _, g1 = pp.split_source_swap({"ClusterDim": cd}, mixed, ctx)
    assert g1 is mixed

    # Only False offered: nothing to split.
    fork = {"SourceSwap": _ss([False]), "ClusterDim": cd}
    f2, g2 = pp.split_source_swap(fork, mixed, ctx)
    assert g2 is mixed and f2["SourceSwap"].values == [False]

    # Inactive.
    fork = {"SourceSwap": _ss([False, True], active=False), "ClusterDim": cd}
    _, g3 = pp.split_source_swap(fork, mixed, ctx)
    assert g3 is mixed

    # All MIs square: both values build the same tile, so the flat parameter stays.
    square = [_mi_entry(_MI_SQUARE), _mi_entry([16, 16, 128, 1, 1, 4, 16, 2, 2])]
    fork = {"SourceSwap": _ss([False, True]), "ClusterDim": cd}
    f4, g4 = pp.split_source_swap(fork, square, ctx)
    assert g4 is square and f4["SourceSwap"].values == [False, True]

    # No ClusterDim axis to rank (absent, single shape as on gfx1250-strict, or inactive).
    for fork in (
        {"SourceSwap": _ss([False, True])},
        {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1]])},
        {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1], [2, 2]], active=False)},
    ):
        f5, g5 = pp.split_source_swap(fork, mixed, ctx)
        assert g5 is mixed and f5["SourceSwap"].values == [False, True]


def test_source_swap_split_gives_every_mi_both_values() -> None:
    pp = _ga_pp()
    fork = {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1], [2, 2]])}
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)

    new_fork, split = pp.split_source_swap(fork, [_mi_entry(_MI_SQUARE), _mi_entry(_MI_32X16)], ctx)
    assert "SourceSwap" not in new_fork and "ClusterDim" in new_fork
    pairs = [(e["MatrixInstruction"].values[:2], e["SourceSwap"].values) for e in split]
    assert pairs == [
        ([16, 16], [False]), ([16, 16], [True]), ([32, 16], [False]), ([32, 16], [True]),
    ]


def test_ga_post_processor_runs_source_swap_split_before_cluster_dim() -> None:
    pp = _ga_pp()
    order = pp._post_process_methods
    assert order.index("narrow_to_mi_geometry") < order.index("narrow_tdm_fuse")
    assert order.index("drop_mi_group_gsu") < order.index("narrow_tdm_fuse")
    assert order.index("narrow_tdm_fuse") < order.index("split_source_swap")
    assert order.index("split_source_swap") < order.index("couple_cluster_dim")
    # TDMFuse, LDSSegmentInterleave and TemporalHintA/B are never coupled into the MI groups.
    assert not {"fit_tdm_fuse_to_wave_count", "couple_lds_segment_interleave", "couple_temporal_hint"} & set(order)
    fork = {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1], [2, 2]])}
    groups = [_mi_entry(_MI_SQUARE), _mi_entry([32, 16, 128, 1, 1, 4, 8, 2, 2])]
    new_fork, coupled = pp.apply(fork, groups, (4096, 3072, 1, 65536))
    assert "SourceSwap" not in new_fork and "ClusterDim" not in new_fork
    for mi in ([16, 16], [32, 16]):
        entries = [e for e in coupled if e["MatrixInstruction"].values[:2] == mi]
        assert {e["SourceSwap"].values[0] for e in entries} == {False, True}
    assert all("ClusterDim" in e for e in coupled)


# ---------------------------------------------------------------------------
# TDMFuse: profile values + _GFX1250TDMFuseNarrowing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _GA_COMBOS)
def test_gfx1250_ga_profile_offers_flat_tdm_fuse(combo: dict) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(**combo))
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    assert profile.tdm_fuse(ctx).values == ([0, 1, 3] if profile._mx_block() else [0, 1])


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (dict(arch="gfx1250_96cu", dt="B"), [0, 1]),
        (dict(arch="gfx1250_96cu", dt="H"), [0, 1]),
        (dict(arch="gfx1250_96cu", dt="F8"), [0, 1]),
        (dict(arch="gfx1250_96cu", dt="F8", mx=True), [0, 1, 3]),
        (dict(arch="gfx1250_96cu", dt="F4", mx=True), [0, 1, 3]),
    ],
    ids=["bf16", "fp16", "fp8", "mxfp8", "mxfp4"],
)
def test_gfx1250_ga_profile_offers_tdm_fuse_3_only_for_mx(cfg: dict, expected: list) -> None:
    """Without MX scales TDMFuse 1-3 issue the same loads, so 3 would only rename 1."""
    profile = g1250_opt.GFX1250GAParams(_cfg(**cfg))
    assert profile.tdm_fuse(SizeContext(M=128, N=128, B=1, K=128)).values == expected


def test_tdm_fuse_builds_follows_tensile_wave_rules() -> None:
    builds = g1250_pp._tdm_fuse_builds
    for waves in (0, 1, 2, 4, 8):
        assert builds(0, waves, True) and builds(0, waves, False)
    assert not builds(1, 1, False) and not builds(3, 1, False) and not builds(3, 1, True)
    assert builds(1, 8, True) and builds(3, 8, False)
    assert builds(3, 2, True) and builds(3, 4, True) and not builds(3, 8, True)
    assert builds(2, 4, True) and not builds(2, 8, True)
    # An MI whose waves cannot be read keeps every value.
    assert builds(3, 0, True) and builds(1, 0, False)


def test_gfx1250_heuristic_profile_leaves_tdm_fuse_unset() -> None:
    profile = g1250_opt.GFX1250Params(_cfg(dt="H"))
    params, _ = profile.generate_for_size((4096, 4096, 1, 8192))
    assert "TDMFuse" not in params


def test_mi_num_waves() -> None:
    assert g1250_pp._mi_num_waves(_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])) == 4
    assert g1250_pp._mi_num_waves(_mi_entry([16, 16, 32, 1, 1, 8, 8, 1, 1])) == 1
    lsu = {
        **_mi_entry([16, 16, 32, 1, 1, 8, 8, 1, 1]),
        "WorkGroup": ForkParameter(name="WorkGroup", values=[16, 16, 2]),
    }
    assert g1250_pp._mi_num_waves(lsu) == 2
    assert g1250_pp._mi_num_waves(_mi_entry([1, 2, 3])) == 0
    assert g1250_pp._mi_num_waves({}) == 0


_LSI_SIZE = (4096, 3072, 1, 65536)
_LSI_CTX = SizeContext(M=4096, N=3072, B=1, K=65536)
# MIWaveGroup [2, 2]: can interleave.
_MI_32X16 = [32, 16, 128, 1, 1, 8, 4, 2, 2]
# MIWaveGroup [2, 4]: neither [2, 2] nor asymmetric, so it cannot interleave.
_MI_NO_INTERLEAVE = [16, 16, 128, 1, 1, 4, 4, 2, 4]
# One wave: TDMFuse 0 only, and MIWaveGroup [1, 1] cannot interleave.
_MI_ONE_WAVE = [16, 16, 128, 1, 1, 8, 8, 1, 1]
_MI_TWO_WAVES = [16, 16, 128, 1, 1, 8, 8, 2, 1]
# Flat parameters that stay out of the MI groups.
_FLAT_ONLY = ("TDMFuse", "LDSSegmentInterleave", "TemporalHintA", "TemporalHintB")


def _lsi(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="LDSSegmentInterleave", values=values, active=active)


def _tdm(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="TDMFuse", values=values, active=active)


def _mx_pp() -> g1250_pp.GFX1250GAPostProcessor:
    return g1250_pp.GFX1250GAPostProcessor(_mx_f4_cfg())


def _narrowed(pp, groups: list, tdm: list, lsi=None) -> list:
    fork = {"TDMFuse": _tdm(tdm)}
    if lsi is not None:
        fork["LDSSegmentInterleave"] = _lsi(lsi)
    new_fork, new_groups = pp.narrow_tdm_fuse(fork, groups, _LSI_CTX)
    assert new_groups is groups
    return new_fork["TDMFuse"].values


def test_tdm_fuse_narrowing_keeps_tdm_fuse_flat() -> None:
    """The step only narrows the flat list; it never moves TDMFuse into the MI groups."""
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_ONE_WAVE), _mi_entry(_MI_NO_INTERLEAVE)]
    assert _narrowed(_mx_pp(), groups, [0, 1, 3], lsi=[0, 1]) == [0, 1, 3]
    assert all("TDMFuse" not in e for e in groups)


def test_tdm_fuse_narrowing_drops_3_no_mi_builds_for_mx() -> None:
    """With both MX scales Tensile builds TDMFuse 3 on 2 or 4 waves only."""
    eight = [_mi_entry(_MI_NO_INTERLEAVE)]
    assert _narrowed(_mx_pp(), eight, [0, 1, 3], lsi=[0, 1]) == [0, 1]
    for mi in (_MI_32X16, _MI_TWO_WAVES):
        assert _narrowed(_mx_pp(), eight + [_mi_entry(mi)], [0, 1, 3], lsi=[0, 1]) == [0, 1, 3]
    four_lsu2 = {**_mi_entry(_MI_32X16), "WorkGroup": ForkParameter(name="WorkGroup", values=[32, 4, 2])}
    assert _narrowed(_mx_pp(), [four_lsu2], [0, 1, 3], lsi=[0, 1]) == [0, 1]


def test_tdm_fuse_narrowing_keeps_3_on_eight_waves_without_mx() -> None:
    """Without MX scales the 2-or-4-wave limit does not apply."""
    assert _narrowed(_ga_pp(), [_mi_entry(_MI_NO_INTERLEAVE)], [0, 1, 3], lsi=[0, 1]) == [0, 1, 3]


def test_tdm_fuse_narrowing_keeps_0_only_where_it_is_needed() -> None:
    """0 serves LDSSegmentInterleave=1 and the one-wave MIs; without either it is dropped."""
    multi = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE)]
    assert _narrowed(_mx_pp(), multi, [0, 1, 3], lsi=[0, 1]) == [0, 1, 3]
    assert _narrowed(_mx_pp(), multi, [0, 1, 3], lsi=[0]) == [1, 3]
    assert _narrowed(_mx_pp(), multi, [0, 1, 3]) == [1, 3]
    one = multi + [_mi_entry(_MI_ONE_WAVE)]
    assert _narrowed(_mx_pp(), one, [0, 1, 3], lsi=[0]) == [0, 1, 3]


def test_tdm_fuse_narrowing_on_one_wave_mis_keeps_0() -> None:
    one = [_mi_entry(_MI_ONE_WAVE), {
        **_mi_entry([16, 16, 32, 1, 1, 1, 1, 1, 1]),
        "WorkGroup": ForkParameter(name="WorkGroup", values=[16, 16, 1]),
    }]
    assert _narrowed(_mx_pp(), one, [0, 1, 3], lsi=[0, 1]) == [0]
    # Never empties the list: without 0 to fall back to, the profile's values stay.
    assert _narrowed(_mx_pp(), one, [1, 3]) == [1, 3]


@pytest.mark.parametrize(
    "fork",
    [{}, {"TDMFuse": _tdm([0, 1, 3], active=False)}],
    ids=["missing", "inactive"],
)
def test_tdm_fuse_narrowing_noop(fork: dict) -> None:
    groups = [_mi_entry(_MI_NO_INTERLEAVE)]
    f2, g2 = _mx_pp().narrow_tdm_fuse(dict(fork), groups, _LSI_CTX)
    assert g2 is groups and set(f2) == set(fork)
    if fork:
        assert f2["TDMFuse"].values == [0, 1, 3]


def test_tdm_fuse_stays_flat_through_cluster_dim_coupling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base_pp.MIDesign,
        "calculate_mfma_parameters",
        lambda _mi: MFMAParameters(MT0=256, MT1=352, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1),
    )
    pp = _ga_pp()
    fork = {
        "TDMFuse": _tdm([0, 1]),
        "ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1], [2, 1], [2, 2]]),
    }
    groups = [_mi_entry([16, 16, 32, 1, 1, 8, 8, 1, 1]), _mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])]
    f2, g2 = pp.apply(fork, groups, (4096, 4096, 1, 16))
    assert f2["TDMFuse"].values == [0, 1]
    assert g2 and all("TDMFuse" not in e and "ClusterDim" in e for e in g2)


# ---------------------------------------------------------------------------
# LDSSegmentInterleave and TemporalHintA/B: profile values, flat through the GA
# post-processor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _GA_COMBOS)
def test_gfx1250_ga_profile_offers_lds_segment_interleave_with_b_rotate_0(combo: dict) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(**combo))
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    assert profile.lds_segment_interleave(ctx).values == [0, 1]
    assert profile.lds_segment_b_rotate(ctx).values == [0]


@pytest.mark.parametrize("streamk", [False, True])
def test_gfx1250_heuristic_profile_keeps_lds_segment_interleave_0(streamk: bool) -> None:
    profile = g1250_opt.GFX1250Params(_cfg(dt="F4", streamk=streamk, mx=True))
    params, _ = profile.generate_for_size(_LSI_SIZE)
    assert params["LDSSegmentInterleave"].values == [0]


def test_interleavable_matches_tensile_mi_wave_groups() -> None:
    ok = g1250_pp._interleavable
    assert ok((2, 2)) and ok((1, 4)) and ok((4, 1)) and ok((1, 2))
    assert not ok((1, 1)) and not ok((2, 4)) and not ok((4, 4))
    assert g1250_pp._mi_wave_group(_MI_32X16) == (2, 2)
    assert g1250_pp._mi_wave_group(_MI_NO_INTERLEAVE) == (2, 4)
    assert g1250_pp._mi_wave_group(_MI_ONE_WAVE) == (1, 1)


def test_no_interleavable_mi_narrows_lds_segment_interleave_and_tdm_fuse_0() -> None:
    """Rule C narrows LDSSegmentInterleave to [0]; with no one-wave MI, TDMFuse 0 goes too."""
    pp = _ga_pp()
    fork = {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([0, 1])}
    groups = [_mi_entry(_MI_NO_INTERLEAVE)]
    f2, g2 = pp.apply(fork, groups, _LSI_SIZE)
    assert f2["LDSSegmentInterleave"].values == [0] and f2["TDMFuse"].values == [1]
    assert all("LDSSegmentInterleave" not in e and "TDMFuse" not in e for e in g2)


@pytest.mark.parametrize("streamk", [False, True])
def test_ga_pipeline_keeps_tdm_fuse_interleave_and_hints_flat(streamk: bool) -> None:
    """Profile output through the full GA post-processor, as the generator runs it.

    MatrixInstruction, SourceSwap and ClusterDim stay coupled in the MI groups;
    TDMFuse, LDSSegmentInterleave and TemporalHintA/B stay flat, one list each,
    so the GA can change them without redrawing the MI. TensileLite rejects
    the combinations that cannot build.
    """
    cfg = _cfg(arch="gfx1250_96cu", dt="F4", streamk=streamk, mx=True)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_LSI_SIZE)
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE), _mi_entry(_MI_ONE_WAVE)]
    f2, coupled = g1250_pp.GFX1250GAPostProcessor(cfg).apply(params, groups, _LSI_SIZE)
    assert f2["TDMFuse"].values == [0, 1, 3]
    assert f2["LDSSegmentInterleave"].values == [0, 1]
    for name in ("TemporalHintA", "TemporalHintB", "TemporalHintD"):
        assert f2[name].values == [0, 1] and f2[name].active
    assert not {"SourceSwap", "ClusterDim"} & set(f2)
    assert coupled
    for e in coupled:
        assert not set(_FLAT_ONLY) & set(e)
        assert set(e) == {"MatrixInstruction", "SourceSwap", "ClusterDim"}
        assert len(e["SourceSwap"].values) == 1 and len(e["ClusterDim"].values) == 1
    # Every MI keeps one entry per (SourceSwap, ClusterDim) pair, nothing more.
    keys = [
        (tuple(e["MatrixInstruction"].values), e["SourceSwap"].values[0], tuple(e["ClusterDim"].values[0]))
        for e in coupled
    ]
    assert len(keys) == len(set(keys))
    assert {k[0] for k in keys} == {tuple(_MI_32X16), tuple(_MI_NO_INTERLEAVE), tuple(_MI_ONE_WAVE)}


# ---------------------------------------------------------------------------
# End-to-end: GFX1250PostProcessor / GFX1250GAPostProcessor.apply()
# ---------------------------------------------------------------------------


def test_gfx1250_postprocessor_apply_runs_mt_du_filter() -> None:
    cfg = _cfg(dt="H")
    cfg["MT_DU"] = [64, 32, 16]
    pp = g1250_pp.GFX1250PostProcessor(cfg)
    fork = {"DepthU": ForkParameter(name="DepthU", values=[8, 16, 32])}
    groups = [_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])]
    f2, g2 = pp.apply(fork, groups, (16, 16, 1, 16))
    assert isinstance(f2, dict)
    assert isinstance(g2, list)


def test_gfx1250_ga_postprocessor_apply_runs_every_mixin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base_pp.MIDesign,
        "calculate_mfma_parameters",
        lambda _mi: MFMAParameters(MT0=256, MT1=352, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1),
    )
    pp = _ga_pp()
    fork = {
        "HalfPLR": ForkParameter(name="HalfPLR", values=[0, 1, 2, 3]),
        "LDSSegmentInterleave": ForkParameter(name="LDSSegmentInterleave", values=[0, 1]),
        "ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1], [2, 1], [2, 2]]),
        "GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[-1]),
    }
    groups = [
        {
            **_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2]),
            "GlobalSplitU": ForkParameter(name="GlobalSplitU", values=[1]),
        },
    ]
    f2, g2 = pp.apply(fork, groups, (4096, 4096, 1, 16))
    assert isinstance(f2, dict)
    assert isinstance(g2, list)


# ---------------------------------------------------------------------------
# SourceSwap: profile values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _GA_COMBOS)
def test_gfx1250_ga_profile_offers_both_source_swap_values(combo: dict) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(**combo))
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    assert profile.source_swap(ctx).values == [False, True]
