# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 optimization-parameter and post-processor profiles."""

from __future__ import annotations

import pytest

from geko.config_generator.fork_params import optimization_param as opt_param
from geko.config_generator.fork_params import post_processor as base_pp
from geko.config_generator.fork_params.hw_profiles.gfx1250 import optimization_param as g1250_opt
from geko.config_generator.fork_params.hw_profiles.gfx1250 import post_processor as g1250_pp
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
# SourceSwap: profile values + _GFX1250SourceSwapCoupling
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


def test_source_swap_coupling_noop_cases() -> None:
    pp = _ga_pp()
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)
    mixed = [_mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2]), _mi_entry([32, 16, 128, 1, 1, 8, 4, 2, 2])]

    # Missing entirely.
    f1, g1 = pp.pin_non_square_source_swap({}, mixed, ctx)
    assert g1 is mixed

    # Only False offered: nothing to pin.
    fork = {"SourceSwap": ForkParameter(name="SourceSwap", values=[False])}
    f2, g2 = pp.pin_non_square_source_swap(fork, mixed, ctx)
    assert g2 is mixed and f2["SourceSwap"].values == [False]

    # Inactive.
    fork = {"SourceSwap": ForkParameter(name="SourceSwap", values=[False, True], active=False)}
    f3, g3 = pp.pin_non_square_source_swap(fork, mixed, ctx)
    assert g3 is mixed

    # All MIs square: the flat parameter stays.
    square = [_mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2]), _mi_entry([16, 16, 128, 1, 1, 4, 16, 2, 2])]
    fork = {"SourceSwap": ForkParameter(name="SourceSwap", values=[False, True])}
    f4, g4 = pp.pin_non_square_source_swap(fork, square, ctx)
    assert g4 is square and f4["SourceSwap"].values == [False, True]


def test_source_swap_coupling_pins_non_square_mi_to_false() -> None:
    pp = _ga_pp()
    square = _mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2])
    rect = _mi_entry([32, 16, 128, 1, 1, 8, 4, 2, 2])
    fork = {"SourceSwap": ForkParameter(name="SourceSwap", values=[False, True])}
    ctx = SizeContext(M=4096, N=3072, B=1, K=65536)

    new_fork, coupled = pp.pin_non_square_source_swap(fork, [square, rect], ctx)
    assert "SourceSwap" not in new_fork
    pairs = [(e["MatrixInstruction"].values[:2], e["SourceSwap"].values) for e in coupled]
    assert pairs == [([16, 16], [False]), ([16, 16], [True]), ([32, 16], [False])]


def test_ga_post_processor_runs_source_swap_before_cluster_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base_pp.MIDesign,
        "calculate_mfma_parameters",
        lambda _mi: MFMAParameters(MT0=256, MT1=256, TT0=1, TT1=1, WG0=1, WG1=1, MIBlockM=1),
    )
    pp = _ga_pp()
    fork = {
        "SourceSwap": ForkParameter(name="SourceSwap", values=[False, True]),
        "ClusterDim": ForkParameter(name="ClusterDim", values=[[1, 1], [2, 2]]),
    }
    groups = [_mi_entry([16, 16, 128, 1, 1, 8, 8, 2, 2]), _mi_entry([32, 16, 128, 1, 1, 4, 8, 2, 2])]
    new_fork, coupled = pp.apply(fork, groups, (4096, 3072, 1, 65536))
    assert "SourceSwap" not in new_fork and "ClusterDim" not in new_fork
    rect = [e for e in coupled if e["MatrixInstruction"].values[:2] == [32, 16]]
    square = [e for e in coupled if e["MatrixInstruction"].values[:2] == [16, 16]]
    assert rect and all(e["SourceSwap"].values == [False] for e in rect)
    assert {e["SourceSwap"].values[0] for e in square} == {False, True}
    assert all("ClusterDim" in e for e in coupled)
