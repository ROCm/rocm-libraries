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

    The other group fields (LDSSegmentInterleave / TDMFuse) split an entry into
    variants; every variant of one (MI, SourceSwap) must get the same shapes.
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
    assert order.index("drop_mi_group_gsu") < order.index("keep_one_wave_tdm_fuse_zero")
    assert order.index("keep_one_wave_tdm_fuse_zero") < order.index("couple_lds_segment_interleave")
    assert order.index("couple_lds_segment_interleave") < order.index("couple_temporal_hint")
    assert order.index("couple_temporal_hint") < order.index("split_source_swap")
    assert order.index("split_source_swap") < order.index("couple_cluster_dim")
    fork = {"SourceSwap": _ss([False, True]), "ClusterDim": _cd([[1, 1], [2, 2]])}
    groups = [_mi_entry(_MI_SQUARE), _mi_entry([32, 16, 128, 1, 1, 4, 8, 2, 2])]
    new_fork, coupled = pp.apply(fork, groups, (4096, 3072, 1, 65536))
    assert "SourceSwap" not in new_fork and "ClusterDim" not in new_fork
    for mi in ([16, 16], [32, 16]):
        entries = [e for e in coupled if e["MatrixInstruction"].values[:2] == mi]
        assert {e["SourceSwap"].values[0] for e in entries} == {False, True}
    assert all("ClusterDim" in e for e in coupled)


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
# LDSSegmentInterleave: profile values + _GFX1250SegmentInterleaveLayout
# ---------------------------------------------------------------------------


_LSI_SIZE = (4096, 3072, 1, 65536)
_LSI_CTX = SizeContext(M=4096, N=3072, B=1, K=65536)
# MIWaveGroup [2, 2]: can interleave.
_MI_32X16 = [32, 16, 128, 1, 1, 8, 4, 2, 2]
# MIWaveGroup [2, 4]: neither [2, 2] nor asymmetric, so it cannot interleave.
_MI_NO_INTERLEAVE = [16, 16, 128, 1, 1, 4, 4, 2, 4]
# One wave: TDMFuse 0 only, and MIWaveGroup [1, 1] cannot interleave.
_MI_ONE_WAVE = [16, 16, 128, 1, 1, 8, 8, 1, 1]


def _lsi(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="LDSSegmentInterleave", values=values, active=active)


def _tdm(values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name="TDMFuse", values=values, active=active)


def _lsi_tdm(entry: dict) -> tuple:
    tdm = entry["TDMFuse"].values[0] if "TDMFuse" in entry else None
    return entry["LDSSegmentInterleave"].values[0], tdm


def _lsi_tdm_by_mi(groups: list) -> dict:
    """The (LDSSegmentInterleave, TDMFuse) pairs of every MI, in entry order."""
    out: dict = {}
    for e in groups:
        pair = _lsi_tdm(e)
        pairs = out.setdefault(tuple(e["MatrixInstruction"].values), [])
        if pair not in pairs:
            pairs.append(pair)
    return out


def _single_valued(groups: list) -> bool:
    return all(
        isinstance(fp.values, list) and len(fp.values) == 1
        for e in groups for name, fp in e.items() if name != "MatrixInstruction"
    )


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


def test_can_interleave_needs_wave_group_and_local_split_u_1() -> None:
    def lsu(n: int) -> dict:
        return {**_mi_entry(_MI_32X16), "WorkGroup": ForkParameter(name="WorkGroup", values=[32, 4, n])}

    assert g1250_pp._can_interleave(_mi_entry(_MI_32X16))
    assert g1250_pp._can_interleave(lsu(1))
    assert not g1250_pp._can_interleave(lsu(2))
    assert not g1250_pp._can_interleave(_mi_entry(_MI_NO_INTERLEAVE))
    assert not g1250_pp._can_interleave(_mi_entry(_MI_ONE_WAVE))
    # Unreadable MIs are not narrowed.
    assert g1250_pp._can_interleave(_mi_entry([1, 2, 3]))


def test_segment_interleave_layout_couples_tdm_fuse_per_mi() -> None:
    pp = _ga_pp()
    fork = {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([1])}
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE)]
    f2, coupled = pp.couple_lds_segment_interleave(fork, groups, _LSI_CTX)
    assert "LDSSegmentInterleave" not in f2 and "TDMFuse" not in f2
    assert [(e["MatrixInstruction"].values, _lsi_tdm(e)) for e in coupled] == [
        (_MI_32X16, (0, 0)),
        (_MI_32X16, (0, 1)),
        (_MI_32X16, (1, 0)),
        (_MI_NO_INTERLEAVE, (0, 0)),
        (_MI_NO_INTERLEAVE, (0, 1)),
    ]
    assert _single_valued(coupled)


def test_segment_interleave_layout_skips_local_split_u_above_1() -> None:
    entry = {**_mi_entry(_MI_32X16), "WorkGroup": ForkParameter(name="WorkGroup", values=[32, 4, 2])}
    _, coupled = _ga_pp().couple_lds_segment_interleave(
        {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([1])}, [entry], _LSI_CTX
    )
    assert [_lsi_tdm(e) for e in coupled] == [(0, 0), (0, 1)]


def test_segment_interleave_layout_keeps_one_wave_mi_at_tdm_fuse_0() -> None:
    """After _GFX1250TDMFuseOneWave the groups carry their own TDMFuse; one-wave MIs
    keep only 0, every other MI gets 0 next to the value it had."""
    pp = _ga_pp()
    fork = {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([1])}
    groups = [_mi_entry(_MI_ONE_WAVE), _mi_entry(_MI_32X16)]
    f1, groups = pp.keep_one_wave_tdm_fuse_zero(fork, groups, _LSI_CTX)
    assert "TDMFuse" not in f1
    f2, coupled = pp.couple_lds_segment_interleave(f1, groups, _LSI_CTX)
    assert f2 == {}
    assert _lsi_tdm_by_mi(coupled) == {
        tuple(_MI_ONE_WAVE): [(0, 0)],
        tuple(_MI_32X16): [(0, 0), (0, 1), (1, 0)],
    }
    assert _single_valued(coupled)


def test_segment_interleave_layout_splits_multi_valued_tdm_fuse() -> None:
    pp = _ga_pp()
    fork = {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([0, 1, 2])}
    _, coupled = pp.couple_lds_segment_interleave(fork, [_mi_entry(_MI_32X16)], _LSI_CTX)
    assert [_lsi_tdm(e) for e in coupled] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert _single_valued(coupled)


def test_segment_interleave_layout_without_tdm_fuse() -> None:
    """No TDMFuse to couple (absent or inactive): the 0 entries carry none, the 1
    entries still get TDMFuse 0, and an inactive flat TDMFuse is left in place."""
    pp = _ga_pp()
    for fork in ({}, {"TDMFuse": _tdm([1], active=False)}):
        fork["LDSSegmentInterleave"] = _lsi([0, 1])
        f2, coupled = pp.couple_lds_segment_interleave(fork, [_mi_entry(_MI_32X16)], _LSI_CTX)
        assert "LDSSegmentInterleave" not in f2
        assert ("TDMFuse" in f2) == ("TDMFuse" in fork)
        assert [_lsi_tdm(e) for e in coupled] == [(0, None), (1, 0)]


@pytest.mark.parametrize(
    "lsi",
    [None, _lsi([0]), _lsi([1]), _lsi([0, 1], active=False)],
    ids=["missing", "only-0", "only-1", "inactive"],
)
def test_segment_interleave_layout_noop(lsi) -> None:
    pp = _ga_pp()
    fork = {"TDMFuse": _tdm([1])}
    if lsi is not None:
        fork["LDSSegmentInterleave"] = lsi
    groups = [_mi_entry(_MI_32X16)]
    f2, g2 = pp.couple_lds_segment_interleave(fork, groups, _LSI_CTX)
    assert g2 is groups
    assert f2["TDMFuse"].values == [1]


def test_segment_interleave_layout_noop_when_no_mi_can_interleave() -> None:
    """Rule C narrows the flat parameter to [0] first, so nothing is coupled."""
    pp = _ga_pp()
    fork = {"LDSSegmentInterleave": _lsi([0, 1]), "TDMFuse": _tdm([1])}
    groups = [_mi_entry(_MI_NO_INTERLEAVE)]
    f2, g2 = pp.apply(fork, groups, _LSI_SIZE)
    assert f2["LDSSegmentInterleave"].values == [0] and f2["TDMFuse"].values == [1]
    assert all("LDSSegmentInterleave" not in e and "TDMFuse" not in e for e in g2)


@pytest.mark.parametrize("streamk", [False, True])
def test_ga_pipeline_couples_lds_segment_interleave_and_tdm_fuse(streamk: bool) -> None:
    """Profile output through the full GA post-processor, as the generator runs it."""
    cfg = _cfg(arch="gfx1250_96cu", dt="F4", streamk=streamk, mx=True)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_LSI_SIZE)
    assert params["LDSSegmentInterleave"].values == [0, 1]
    assert params["TDMFuse"].values == [1]
    assert len(params["ClusterDim"].values) > 1
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE), _mi_entry(_MI_ONE_WAVE)]
    f2, coupled = g1250_pp.GFX1250GAPostProcessor(cfg).apply(params, groups, _LSI_SIZE)
    assert not {"LDSSegmentInterleave", "TDMFuse", "ClusterDim"} & set(f2)
    assert _lsi_tdm_by_mi(coupled) == {
        tuple(_MI_32X16): [(0, 0), (0, 1), (1, 0)],
        tuple(_MI_NO_INTERLEAVE): [(0, 0), (0, 1)],
        tuple(_MI_ONE_WAVE): [(0, 0)],
    }
    assert all("ClusterDim" in e for e in coupled)
    assert _single_valued(coupled)
    # Every (LDSSegmentInterleave, TDMFuse) entry of an MI gets the same ClusterDim shapes.
    for mi in (_MI_32X16, _MI_NO_INTERLEAVE):
        shapes: dict = {}
        for e in coupled:
            if e["MatrixInstruction"].values == mi:
                key = (_lsi_tdm(e), _th(e), e["SourceSwap"].values[0])
                shapes.setdefault(key, []).append(e["ClusterDim"].values[0])
        per_swap: dict = {}
        for (pair, th, swap), s in shapes.items():
            per_swap.setdefault(swap, []).append(s)
        for lists in per_swap.values():
            assert len(lists) >= 2 and all(s == lists[0] and s for s in lists)


# ---------------------------------------------------------------------------
# _GFX1250TemporalHintCoupling
# ---------------------------------------------------------------------------


def _th_param(name: str, values: list, active: bool = True) -> ForkParameter:
    return ForkParameter(name=name, values=values, active=active)


def _th_fork(a=(0, 1), b=(0, 1), active: bool = True) -> dict:
    return {
        "TemporalHintA": _th_param("TemporalHintA", list(a), active),
        "TemporalHintB": _th_param("TemporalHintB", list(b), active),
        "TemporalHintD": _th_param("TemporalHintD", [0, 1]),
    }


def _th(entry: dict):
    if "TemporalHintA" not in entry:
        return None
    return entry["TemporalHintA"].values[0], entry["TemporalHintB"].values[0]


def _with_tdm(entry: dict, tdm: int) -> dict:
    return {**entry, "TDMFuse": _tdm([tdm])}


def test_temporal_hint_coupling_pairs_hints_with_each_entry_tdm_fuse() -> None:
    pp = _ga_pp()
    groups = [_with_tdm(_mi_entry(_MI_32X16), 0), _with_tdm(_mi_entry(_MI_32X16), 1)]
    f2, coupled = pp.couple_temporal_hint(_th_fork(), groups, _LSI_CTX)
    assert "TemporalHintA" not in f2 and "TemporalHintB" not in f2
    assert f2["TemporalHintD"].values == [0, 1]
    assert [(e["TDMFuse"].values[0], _th(e)) for e in coupled] == [
        (0, (0, 0)), (0, (1, 1)),
        (1, (0, 0)), (1, (0, 1)), (1, (1, 0)), (1, (1, 1)),
    ]
    assert _single_valued(coupled)


def test_temporal_hint_coupling_keeps_only_agreeing_pairs_on_tdm_fuse_0() -> None:
    """One flat hint multi-valued, the other single: TDMFuse 0 keeps the pair that agrees."""
    pp = _ga_pp()
    groups = [_with_tdm(_mi_entry(_MI_ONE_WAVE), 0), _with_tdm(_mi_entry(_MI_32X16), 1)]
    _, coupled = pp.couple_temporal_hint(_th_fork(a=(0, 1), b=(1,)), groups, _LSI_CTX)
    assert [(e["TDMFuse"].values[0], _th(e)) for e in coupled] == [
        (0, (1, 1)), (1, (0, 1)), (1, (1, 1)),
    ]


def test_temporal_hint_coupling_with_flat_tdm_fuse() -> None:
    """TDMFuse still flat: a flat 0 couples the hints, a flat 1 leaves them flat."""
    pp = _ga_pp()
    groups = [_mi_entry(_MI_32X16)]
    f2, coupled = pp.couple_temporal_hint({**_th_fork(), "TDMFuse": _tdm([0])}, groups, _LSI_CTX)
    assert "TemporalHintA" not in f2 and f2["TDMFuse"].values == [0]
    assert [_th(e) for e in coupled] == [(0, 0), (1, 1)]
    fork = {**_th_fork(), "TDMFuse": _tdm([1])}
    f3, g3 = pp.couple_temporal_hint(fork, groups, _LSI_CTX)
    assert g3 is groups and f3["TemporalHintA"].values == [0, 1]


@pytest.mark.parametrize(
    "fork",
    [
        {},
        {"TemporalHintA": _th_param("TemporalHintA", [0, 1])},
        _th_fork(active=False),
        _th_fork(a=(0,), b=(1,)),
    ],
    ids=["missing", "only-a", "inactive", "single-valued"],
)
def test_temporal_hint_coupling_noop(fork: dict) -> None:
    pp = _ga_pp()
    groups = [_with_tdm(_mi_entry(_MI_32X16), 0)]
    f2, g2 = pp.couple_temporal_hint(dict(fork), groups, _LSI_CTX)
    assert g2 is groups
    assert set(f2) == set(fork)


def test_temporal_hint_coupling_noop_when_an_entry_has_no_single_tdm_fuse() -> None:
    pp = _ga_pp()
    groups = [_with_tdm(_mi_entry(_MI_32X16), 0), _mi_entry(_MI_32X16)]
    f2, g2 = pp.couple_temporal_hint(_th_fork(), groups, _LSI_CTX)
    assert g2 is groups and "TemporalHintA" in f2


@pytest.mark.parametrize("streamk", [False, True])
def test_ga_pipeline_couples_temporal_hint_with_tdm_fuse(streamk: bool) -> None:
    """Profile output through the full GA post-processor: no TDMFuse 0 entry carries
    disagreeing hints, and TDMFuse 1 entries keep every pair."""
    cfg = _cfg(arch="gfx1250_96cu", dt="F4", streamk=streamk, mx=True)
    params, _ = g1250_opt.GFX1250GAParams(cfg).generate_for_size(_LSI_SIZE)
    assert params["TemporalHintA"].values == [0, 1] and params["TemporalHintB"].values == [0, 1]
    groups = [_mi_entry(_MI_32X16), _mi_entry(_MI_NO_INTERLEAVE), _mi_entry(_MI_ONE_WAVE)]
    f2, coupled = g1250_pp.GFX1250GAPostProcessor(cfg).apply(params, groups, _LSI_SIZE)
    assert not {"TemporalHintA", "TemporalHintB"} & set(f2)
    assert "TemporalHintD" in f2
    assert _single_valued(coupled)
    variants: dict = {}
    for e in coupled:
        variants.setdefault(tuple(e["MatrixInstruction"].values), set()).add((_lsi_tdm(e), _th(e)))
    agree = [(0, 0), (1, 1)]
    every = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert variants[tuple(_MI_32X16)] == (
        {((0, 0), p) for p in agree} | {((0, 1), p) for p in every} | {((1, 0), p) for p in agree}
    )
    assert variants[tuple(_MI_NO_INTERLEAVE)] == (
        {((0, 0), p) for p in agree} | {((0, 1), p) for p in every}
    )
    assert variants[tuple(_MI_ONE_WAVE)] == {((0, 0), p) for p in agree}


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
