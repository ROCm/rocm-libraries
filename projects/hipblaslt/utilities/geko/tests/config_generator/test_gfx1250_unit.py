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
    hardware_cluster_dims,
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
_SHAPES_16 = [list(s) for s in hardware_cluster_dims(16)]


def _ss(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="SourceSwap", values=values, active=active)


def _coupled_shapes(mi: list, source_swap=None, entry_source_swap=None) -> list:
    """ClusterDim shapes the GA post-processor couples to one MI entry at 4096x3072."""
    fork = {"ClusterDim": ForkParameter(name="ClusterDim", values=list(_SHAPES_16))}
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
    return [list(s) for s in select_cluster_dims(_SHAPES_16, grid, macro_tile)]


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
    # The size-16 winner differs: [2, 8] at 512x128, [4, 4] at 256x256.
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


def _shapes_by_mi_and_swap(groups: list, lsi: int = 0) -> dict:
    """ClusterDim shapes per (MI, SourceSwap), for one LDSSegmentInterleave value."""
    out: dict = {}
    for e in groups:
        if "LDSSegmentInterleave" in e and e["LDSSegmentInterleave"].values != [lsi]:
            continue
        key = (tuple(e["MatrixInstruction"].values), e["SourceSwap"].values[0])
        out.setdefault(key, []).append(e["ClusterDim"].values[0])
    return out


@pytest.mark.parametrize("streamk", [False, True])
def test_ga_postprocessor_ranks_each_source_swap_entry_on_its_own_tile(streamk: bool) -> None:
    """Profile output through the full GA post-processor, as the generator runs it.

    _GFX1250SourceSwapSplit leaves one SourceSwap value per MI entry, so the
    ClusterDim coupling ranks every entry on the tile it builds, not a union.
    """
    cfg = _cfg(arch="gfx1250_96cu", dt="F4", streamk=streamk, mx=True)
    size = (4096, 3072, 1, 65536)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(size)
    assert True in params["SourceSwap"].values
    groups = [_mi_entry(_MI_SQUARE), _mi_entry(_MI_32X16)]
    _, coupled = g1250_pp.GFX1250GAPostProcessor(cfg).apply(params, groups, size)
    square = _selected((256, 256))
    for lsi in (0, 1):
        by_key = _shapes_by_mi_and_swap(coupled, lsi)
        assert by_key[(tuple(_MI_SQUARE), False)] == square
        assert by_key[(tuple(_MI_SQUARE), True)] == square
        assert by_key[(tuple(_MI_32X16), False)] == _selected((512, 128))
        assert by_key[(tuple(_MI_32X16), True)] == _selected((256, 256))


# ---------------------------------------------------------------------------
# TDMFuse: profile pin + _GFX1250TDMFuseOneWave
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _GA_COMBOS)
def test_gfx1250_ga_profile_pins_tdm_fuse_1(combo: dict) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(**combo))
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    assert profile.tdm_fuse(ctx).values == [1]


def test_gfx1250_heuristic_profile_leaves_tdm_fuse_unset() -> None:
    profile = g1250_opt.GFX1250Params(_cfg(dt="H"))
    params, _ = profile.generate_for_size((4096, 4096, 1, 8192))
    assert "TDMFuse" not in params


# ---------------------------------------------------------------------------
# LDSSegmentInterleave / LDSSegmentBRotate
# ---------------------------------------------------------------------------


_LSI_SIZE = (4096, 3072, 1, 65536)
# MIWaveGroup [2, 4]: neither [2, 2] nor asymmetric, so it cannot interleave.
_MI_NO_INTERLEAVE = [16, 16, 32, 1, 1, 4, 4, 2, 4]


def _lsi(values: list) -> ForkParameter:
    return ForkParameter(name="LDSSegmentInterleave", values=values)


def _tdm(values: list) -> ForkParameter:
    return ForkParameter(name="TDMFuse", values=values)


def _vals(entry: dict) -> dict:
    return {k: v.values for k, v in entry.items() if k != "MatrixInstruction"}


@pytest.mark.parametrize("streamk", [False, True])
def test_gfx1250_ga_profile_offers_lds_segment_interleave_with_b_rotate_0(streamk: bool) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(dt="F4", streamk=streamk, mx=True))
    params, _ = profile.generate_for_size(_LSI_SIZE)
    assert params["LDSSegmentInterleave"].values == [0, 1]
    assert params["LDSSegmentBRotate"].values == [0]


def test_gfx1250_heuristic_profile_keeps_lds_segment_interleave_0() -> None:
    params, _ = g1250_opt.GFX1250Params(_cfg(dt="F4", mx=True)).generate_for_size(_LSI_SIZE)
    assert params["LDSSegmentInterleave"].values == [0]


def test_interleavable_matches_tensile_mi_wave_groups() -> None:
    ok = g1250_pp._interleavable
    assert ok((2, 2)) and ok((1, 4)) and ok((4, 1)) and ok((1, 2))
    assert not ok((1, 1)) and not ok((2, 4)) and not ok((4, 4))
    assert g1250_pp._mi_wave_group(_MI_32X16) == (2, 2)
    assert g1250_pp._mi_wave_group(_MI_NO_INTERLEAVE) == (2, 4)


def test_gfx1250_ga_postprocessor_keeps_lds_segment_interleave_for_interleavable_mi() -> None:
    # The published MXFP4 MAF MI has MIWaveGroup [2, 2], which can interleave.
    fork = {"LDSSegmentInterleave": _lsi([0, 1])}
    n = g1250_pp._GFX1250GeometryNarrowing()
    f2, _ = n.narrow_to_mi_geometry(fork, [_mi_entry(_MI_32X16)], SizeContext(M=1, N=1, B=1, K=1))
    assert f2["LDSSegmentInterleave"].values == [0, 1]
    f3, _ = n.narrow_to_mi_geometry(
        {"LDSSegmentInterleave": _lsi([0, 1])},
        [_mi_entry(_MI_NO_INTERLEAVE)],
        SizeContext(M=1, N=1, B=1, K=1),
    )
    assert f3["LDSSegmentInterleave"].values == [0]


@pytest.mark.parametrize("mx", [False, True])
def test_lds_segment_interleave_1_gets_the_paired_layout(mx: bool) -> None:
    pp = g1250_pp.GFX1250GAPostProcessor(_cfg(dt="F4" if mx else "H", mx=mx))
    fork = {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([1])}
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE)]
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    f2, coupled = pp.couple_lds_segment_interleave(fork, groups, ctx)
    assert "LDSSegmentInterleave" not in f2 and "TDMFuse" not in f2
    fdsi = {"ForceDisableShadowInit": [True]} if mx else {}
    paired = {"LDSSegmentInterleave": [1], "TDMFuse": [0], **fdsi}
    unpaired = {"LDSSegmentInterleave": [0], "TDMFuse": [1], **fdsi}
    assert [(e["MatrixInstruction"].values, _vals(e)) for e in coupled] == [
        (_MI_32X16, unpaired),
        (_MI_32X16, paired),
        (_MI_NO_INTERLEAVE, unpaired),
    ]


def test_lds_segment_interleave_layout_keeps_group_values_single() -> None:
    """A multi-valued flat TDMFuse or LDSSegmentInterleave becomes one entry per value."""
    pp = _ga_pp()
    fork = {"LDSSegmentInterleave": _lsi([-1, 0, 1]), "TDMFuse": _tdm([0, 1])}
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE)]
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    f2, coupled = pp.couple_lds_segment_interleave(fork, groups, ctx)
    assert f2 == {}
    assert all(len(fp.values) == 1 for e in coupled for name, fp in e.items()
               if name != "MatrixInstruction")
    unpaired = [
        {"LDSSegmentInterleave": [lsi], "TDMFuse": [tdm]} for lsi in (-1, 0) for tdm in (0, 1)
    ]
    assert [(e["MatrixInstruction"].values, _vals(e)) for e in coupled] == [
        *[(_MI_32X16, v) for v in unpaired],
        (_MI_32X16, {"LDSSegmentInterleave": [1], "TDMFuse": [0]}),
        *[(_MI_NO_INTERLEAVE, v) for v in unpaired],
    ]


def test_can_interleave_needs_local_split_u_1() -> None:
    lsu = lambda n: {  # noqa: E731
        **_mi_entry(_MI_32X16),
        "WorkGroup": ForkParameter(name="WorkGroup", values=[32, 4, n]),
    }
    assert g1250_pp._can_interleave(_mi_entry(_MI_32X16))
    assert g1250_pp._can_interleave(lsu(1))
    assert not g1250_pp._can_interleave(lsu(2))
    assert not g1250_pp._can_interleave(_mi_entry(_MI_NO_INTERLEAVE))
    assert g1250_pp._can_interleave(_mi_entry([1, 2, 3]))
    _, coupled = _ga_pp().couple_lds_segment_interleave(
        {"LDSSegmentInterleave": _lsi([0, 1])}, [lsu(2)], SizeContext(M=16, N=16, B=1, K=16)
    )
    assert [e["LDSSegmentInterleave"].values for e in coupled] == [[0]]


def test_lds_segment_interleave_layout_keeps_group_tdm_fuse() -> None:
    pp = _ga_pp()
    one_wave = {**_mi_entry([16, 16, 32, 1, 1, 1, 1, 1, 1]), "TDMFuse": _tdm([0])}
    four = {**_mi_entry(_MI_32X16), "TDMFuse": _tdm([1])}
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    f2, coupled = pp.couple_lds_segment_interleave(
        {"LDSSegmentInterleave": _lsi([0, 1])}, [one_wave, four], ctx
    )
    assert f2 == {}
    assert [_vals(e) for e in coupled] == [
        {"TDMFuse": [0], "LDSSegmentInterleave": [0]},
        {"TDMFuse": [1], "LDSSegmentInterleave": [0]},
        {"TDMFuse": [0], "LDSSegmentInterleave": [1]},
    ]


@pytest.mark.parametrize("values", [None, [0], [1]])
def test_lds_segment_interleave_layout_noop(values) -> None:
    pp = _ga_pp()
    fork = {"TDMFuse": _tdm([1])}
    if values is not None:
        fork["LDSSegmentInterleave"] = _lsi(values)
    groups = [_mi_entry(_MI_32X16)]
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    f2, g2 = pp.couple_lds_segment_interleave(fork, groups, ctx)
    assert g2 is groups
    assert f2["TDMFuse"].values == [1]
    inactive = {"LDSSegmentInterleave": _lsi([0, 1])}
    inactive["LDSSegmentInterleave"].active = False
    assert pp.couple_lds_segment_interleave(inactive, groups, ctx)[1] is groups


@pytest.mark.parametrize("streamk", [False, True])
def test_ga_pipeline_builds_lds_segment_interleave_1_paired(streamk: bool) -> None:
    """Profile output through the full GA post-processor: every LDSSegmentInterleave=1
    entry has TDMFuse 0, every MX entry ForceDisableShadowInit True, for every cluster shape."""
    cfg = _cfg(arch="gfx1250_96cu", dt="F4", streamk=streamk, mx=True)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_LSI_SIZE)
    pp = g1250_pp.GFX1250GAPostProcessor(cfg)
    f2, groups = pp.apply(params, [_mi_entry(_MI_32X16)], _LSI_SIZE)
    assert "LDSSegmentInterleave" not in f2 and "TDMFuse" not in f2
    one = [e for e in groups if e["LDSSegmentInterleave"].values == [1]]
    zero = [e for e in groups if e["LDSSegmentInterleave"].values == [0]]
    assert len(one) == len(zero) > 1
    assert all(e["TDMFuse"].values == [0] for e in one)
    assert all(e["TDMFuse"].values == [1] for e in zero)
    assert all(e["ForceDisableShadowInit"].values == [True] for e in groups)
    assert [4, 4] in [e["ClusterDim"].values[0] for e in one]


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


def test_tdm_fuse_one_wave_noop_without_one_wave_mi() -> None:
    pp = _ga_pp()
    groups = [_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2]), _mi_entry([16, 16, 32, 1, 1, 4, 4, 2, 1])]
    ctx = SizeContext(M=128, N=128, B=1, K=128)

    f1, g1 = pp.keep_one_wave_tdm_fuse_zero({}, groups, ctx)
    assert g1 is groups

    fork = {"TDMFuse": ForkParameter(name="TDMFuse", values=[1])}
    f2, g2 = pp.keep_one_wave_tdm_fuse_zero(fork, groups, ctx)
    assert g2 is groups
    assert f2["TDMFuse"].values == [1]


def test_tdm_fuse_one_wave_moves_into_mi_groups() -> None:
    pp = _ga_pp()
    four = _mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])
    one = _mi_entry([16, 16, 32, 1, 1, 1, 1, 1, 1])
    one_lsu2 = {
        **_mi_entry([16, 16, 32, 1, 1, 1, 1, 1, 1]),
        "WorkGroup": ForkParameter(name="WorkGroup", values=[16, 16, 2]),
    }
    fork = {"TDMFuse": ForkParameter(name="TDMFuse", values=[1])}
    ctx = SizeContext(M=16, N=16, B=1, K=128)

    new_fork, coupled = pp.keep_one_wave_tdm_fuse_zero(fork, [four, one, one_lsu2], ctx)
    assert "TDMFuse" not in new_fork
    assert [e["TDMFuse"].values for e in coupled] == [[1], [0], [1]]


def test_tdm_fuse_one_wave_keeps_zero_when_profile_offers_it() -> None:
    pp = _ga_pp()
    fork = {"TDMFuse": ForkParameter(name="TDMFuse", values=[0, 1])}
    groups = [_mi_entry([16, 16, 32, 1, 1, 1, 1, 1, 1]), _mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])]
    ctx = SizeContext(M=16, N=16, B=1, K=128)
    _, coupled = pp.keep_one_wave_tdm_fuse_zero(fork, groups, ctx)
    assert [e["TDMFuse"].values for e in coupled] == [[0], [0, 1]]


def test_tdm_fuse_one_wave_survives_cluster_dim_coupling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base_pp.MIDesign,
        "calculate_mfma_parameters",
        lambda _mi: MFMAParameters(MT0=256, MT1=352, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1),
    )
    pp = _ga_pp()
    fork = {
        "TDMFuse": ForkParameter(name="TDMFuse", values=[1]),
        "ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1], [2, 1], [2, 2]]),
    }
    groups = [_mi_entry([16, 16, 32, 1, 1, 8, 8, 1, 1]), _mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])]
    f2, g2 = pp.apply(fork, groups, (4096, 4096, 1, 16))
    assert "TDMFuse" not in f2
    assert {tuple(e["TDMFuse"].values) for e in g2 if e["MatrixInstruction"].values[7:] == [1, 1]} == {(0,)}
    assert {tuple(e["TDMFuse"].values) for e in g2 if e["MatrixInstruction"].values[7:] == [2, 2]} == {(1,)}


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
# SourceSwap: profile values + _GFX1250SourceSwapSplit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _GA_COMBOS)
def test_gfx1250_ga_profile_offers_both_source_swap_values(combo: dict) -> None:
    profile = g1250_opt.GFX1250GAParams(_cfg(**combo))
    ctx = SizeContext(M=128, N=128, B=1, K=128)
    assert profile.source_swap(ctx).values == [False, True]


def test_mi_is_square() -> None:
    assert g1250_pp._mi_is_square(_mi_entry([16, 16, 32, 1, 1, 8, 8, 2, 2])) is True
    assert g1250_pp._mi_is_square(_mi_entry([32, 16, 128, 1, 1, 8, 4, 2, 2])) is False
    assert g1250_pp._mi_is_square(_mi_entry([1])) is True
    assert g1250_pp._mi_is_square({}) is True


def _cd(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="ClusterDim", values=values, active=active)


def test_source_swap_split_noop_cases() -> None:
    pp = _ga_pp()
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    mixed = [_mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2]), _mi_entry([32, 16, 128, 1, 1, 8, 4, 2, 2])]
    cd = _cd([[1, 1], [2, 2]])

    # Missing entirely.
    f1, g1 = pp.split_source_swap({"ClusterDim": cd}, mixed, ctx)
    assert g1 is mixed

    # Only False offered: nothing to split.
    fork = {"SourceSwap": _ss([False]), "ClusterDim": cd}
    f2, g2 = pp.split_source_swap(fork, mixed, ctx)
    assert g2 is mixed and f2["SourceSwap"].values == [False]

    # Inactive.
    fork = {"SourceSwap": _ss([False, True], active=False), "ClusterDim": cd}
    f3, g3 = pp.split_source_swap(fork, mixed, ctx)
    assert g3 is mixed

    # All MIs square: both values build the same tile, so the flat parameter stays.
    square = [_mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2]), _mi_entry([16, 16, 128, 1, 1, 4, 16, 2, 2])]
    fork = {"SourceSwap": _ss([False, True]), "ClusterDim": cd}
    f4, g4 = pp.split_source_swap(fork, square, ctx)
    assert g4 is square and f4["SourceSwap"].values == [False, True]

    # No ClusterDim axis to rank (absent, single shape as on A0, or inactive).
    for fork in (
        {"SourceSwap": _ss([False, True])},
        {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1]])},
        {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1], [2, 2]], active=False)},
    ):
        f5, g5 = pp.split_source_swap(fork, mixed, ctx)
        assert g5 is mixed and f5["SourceSwap"].values == [False, True]


def test_source_swap_split_gives_every_mi_both_values() -> None:
    pp = _ga_pp()
    square = _mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2])
    rect = _mi_entry([32, 16, 128, 1, 1, 8, 4, 2, 2])
    fork = {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1], [2, 2]])}
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)

    new_fork, coupled = pp.split_source_swap(fork, [square, rect], ctx)
    assert "SourceSwap" not in new_fork and "ClusterDim" in new_fork
    pairs = [(e["MatrixInstruction"].values[:2], e["SourceSwap"].values) for e in coupled]
    assert pairs == [
        ([16, 16], [False]), ([16, 16], [True]), ([32, 16], [False]), ([32, 16], [True]),
    ]


def test_ga_post_processor_runs_source_swap_before_cluster_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base_pp.MIDesign,
        "calculate_mfma_parameters",
        lambda _mi: MFMAParameters(MT0=256, MT1=256, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1),
    )
    pp = _ga_pp()
    fork = {
        "SourceSwap": _ss([False, True]),
        "ClusterDim": _cd([[1, 1], [2, 2]]),
    }
    groups = [_mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2]), _mi_entry([32, 16, 128, 1, 1, 4, 8, 2, 2])]
    new_fork, coupled = pp.apply(fork, groups, (4096, 3072, 1, 65536))
    assert "SourceSwap" not in new_fork and "ClusterDim" not in new_fork
    for mi in ([16, 16], [32, 16]):
        entries = [e for e in coupled if e["MatrixInstruction"].values[:2] == mi]
        assert {e["SourceSwap"].values[0] for e in entries} == {False, True}
    assert all("ClusterDim" in e for e in coupled)


@pytest.mark.parametrize("dt", ["F4", "F8"])
@pytest.mark.parametrize("streamk", [False, True])
def test_ga_profile_and_postprocessor_offer_the_published_mxfp4_cluster(
    streamk: bool, dt: str
) -> None:
    """The shipped MXFP4/NVFP4 MAF MI and cluster are reachable again.

    Runs for MX F4 and F8 data. F4 also stands for NVFP4, which differs from
    MXFP4 only in the MX block size and scale type; the GA profile and these
    post-process steps only check whether the problem is MX.

    [32, 16, 128, 1, 1, 8, 4, 2, 2] with SourceSwap=True builds 256x256 and
    ships with ClusterDim [4, 4]; with SourceSwap=False it builds 512x128, whose
    size-16 shape at 4096x3072 is [2, 8].
    """
    cfg = _cfg(arch="gfx1250_96cu", dt=dt, streamk=streamk, mx=True)
    size = (4096, 3072, 1, 65536)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(size)
    pp = g1250_pp.GFX1250GAPostProcessor(cfg)
    new_fork, groups = pp.apply(params, [_mi_entry(_MI_32X16)], size)
    assert "SourceSwap" not in new_fork and "LDSSegmentInterleave" not in new_fork
    for lsi in (0, 1):
        by_key = _shapes_by_mi_and_swap(groups, lsi)
        swapped = by_key[(tuple(_MI_32X16), True)]
        unswapped = by_key[(tuple(_MI_32X16), False)]
        assert swapped == _selected((256, 256))
        assert [4, 4] in swapped and [2, 8] not in swapped
        assert unswapped == _selected((512, 128))
        assert [2, 8] in unswapped and [4, 4] not in unswapped
    # The shipped kernel's own layout: SourceSwap True, [4, 4], LDSSegmentInterleave 1,
    # TDMFuse 0, ForceDisableShadowInit True.
    published = [
        e for e in groups
        if (e["SourceSwap"].values, e["ClusterDim"].values) == ([True], [[4, 4]])
        if e["LDSSegmentInterleave"].values == [1]
    ]
    assert len(published) == 1
    assert published[0]["TDMFuse"].values == [0]
    assert published[0]["ForceDisableShadowInit"].values == [True]
