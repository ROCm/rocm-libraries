# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from geko.config_generator import mi_candidates
from geko.config_generator import mi_designer as mid
from geko.schemas import GemmConfig, GemmType


def _config(streamk: bool = False) -> dict:
    gt = GemmType.from_tensile("N", "N", "H", "H", "S")
    return {
        "ARCH": "gfx950",
        "CUs": 256,
        "MI_FILTER": 1,
        "StreamK": streamk,
        "GemmProblem": GemmConfig(gt, [[16, 16, 1, 16]]),
    }


def test_calculate_granularities_and_mfma_parameters() -> None:
    out = mid.MIDesign.calculate_granularities(
        MT0=64,
        MT1=64,
        M=128,
        N=128,
        batch_count=1,
        CUs=64,
        LSU=1,
        GSU=1,
        wave=[2, 2],
    )
    assert out.NumTile0 == 2.0
    assert out.NumTile1 == 2.0

    mi = mid.MFMA(M=16, N=16, K=16, B=1, MIBlockM=1, waveTileM=2, waveTileN=2, waveM=2, waveN=2)
    params = mid.MIDesign.calculate_mfma_parameters(mi)
    assert params.MT0 > 0 and params.MT1 > 0


def _write_list(cfg: dict, mis: list, depthu: dict = None, searched: list = ()) -> Path:
    """A buildable-MI list for cfg's ARCH and GEMM type (under the patched MI_LISTS_DIR); ``depthu`` maps an MI
    to the DepthUs it builds at, ``searched`` lists the DepthUs searched."""
    depthu = depthu or {}
    path = mi_candidates.buildable_list_path(cfg["ARCH"], cfg["GemmProblem"].name)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = "".join(
        f"  - {{MatrixInstruction: {list(m)}" + (f", DepthU: {depthu[m]}" if m in depthu else "") + "}\n" for m in mis
    )
    path.write_text(f"DepthU: {list(searched)}\nMatrixInstructions:\n{entries}")
    return path


def _mi9(m: mid.MFMA) -> tuple:
    return (m.M, m.N, m.K, m.B, m.MIBlockM, m.waveTileM, m.waveTileN, m.waveM, m.waveN)


def test_load_candidate_mfmas_uses_override_outside_allowlist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    cfg = _config()
    cfg["BUILDABLE_MIS"] = True
    cfg["MFMA"] = [16, 16, 16, 1]
    _write_list(cfg, [(16, 16, 32, 1, 1, 1, 1, 4, 1)])
    d = mid.MIDesign(tmp_path, cfg)
    valid, sm, sn = d.load_candidate_mfmas()
    assert sm == 16 and sn == 16
    # [16, 16, 16, 1] is not in gfx950's H allowlist, so no list can cover it: every candidate of that MFMA.
    assert [_mi9(m) for m in valid] == mi_candidates.enumerate_candidate_mis("gfx950", "H", None, [16, 16, 16, 1])


@pytest.mark.parametrize("search_space", [None, "heuristic", "generic"])
def test_load_candidate_mfmas_reads_buildable_list(monkeypatch, tmp_path: Path, search_space) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    cfg = _config()
    cfg["BUILDABLE_MIS"] = True
    cfg["search_space"] = search_space                         # one list for every non-subtile search space
    listed = [(32, 32, 16, 1, 1, 1, 1, 4, 1), (16, 16, 32, 1, 1, 2, 3, 2, 2)]
    _write_list(cfg, listed)
    valid, sm, sn = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == listed                  # the list's MIs, in its order
    assert (sm, sn) == (16, 16)                                # still from the MFMA allowlist

    cfg["MFMA"] = [16, 16, 32, 1]                              # an override inside the allowlist filters the list
    valid, _, _ = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == [(16, 16, 32, 1, 1, 2, 3, 2, 2)]


def test_load_candidate_mfmas_ignores_list_for_subtile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    cfg = _config()
    cfg["BUILDABLE_MIS"] = True
    cfg["search_space"] = "subtile"
    _write_list(cfg, [(16, 16, 32, 1, 1, 1, 1, 4, 1)])
    valid, _, _ = mid.MIDesign(tmp_path, cfg, subtile_enabled=True).load_candidate_mfmas()
    # The lists hold non-subtile verdicts, so subtile kernels keep every candidate of the subtile space.
    assert [_mi9(m) for m in valid] == mi_candidates.enumerate_candidate_mis("gfx950", "H", "subtile")


def test_load_candidate_mfmas_enumerates_by_default_without_list_or_when_disabled(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    cfg = _config()
    every = mi_candidates.enumerate_candidate_mis("gfx950", "H", None)
    _write_list(cfg, [(16, 16, 32, 1, 1, 1, 1, 4, 1)])
    valid, _, _ = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == every                   # lists are opt-in: unset means every MI

    cfg["BUILDABLE_MIS"] = False
    valid, _, _ = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == every                   # list ignored when disabled

    cfg["BUILDABLE_MIS"] = True
    mi_candidates.buildable_list_path(cfg["ARCH"], cfg["GemmProblem"].name).unlink()
    with caplog.at_level("WARNING", logger="GEKO"):
        valid, _, _ = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == every                   # enabled but no list for this GEMM type
    assert "generate_buildable_mis.py --arch gfx950 --types H:H:S --layouts NN" in caplog.text


def test_load_candidate_mfmas_falls_back_when_the_list_is_empty(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    cfg = _config()
    cfg["BUILDABLE_MIS"] = True
    _write_list(cfg, []).write_text("MatrixInstructions: []\n")
    with caplog.at_level("WARNING", logger="GEKO"):
        valid, _, _ = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == mi_candidates.enumerate_candidate_mis("gfx950", "H", None)
    assert "lists no MI" in caplog.text


def test_load_candidate_mfmas_does_not_suggest_generating_mx_lists(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    gt = GemmType.from_tensile("T", "N", "F8", "B", "S")
    cfg = dict(_config(), ARCH="gfx1250", BUILDABLE_MIS=True, GemmProblem=GemmConfig(gt, [[16, 16, 1, 16]], mx_block=32))
    with caplog.at_level("WARNING", logger="GEKO"):
        valid, _, _ = mid.MIDesign(tmp_path, cfg).load_candidate_mfmas()
    assert [_mi9(m) for m in valid] == mi_candidates.enumerate_candidate_mis("gfx1250", "F8", None)
    assert "do not cover MX formats (F8BS_MXE8B32_TN" in caplog.text and "generate_buildable_mis" not in caplog.text


def _pairs(groups) -> set:
    return {(tuple(g["MatrixInstruction"].values), g["DepthU"].values[0] if "DepthU" in g else None) for g in groups}


def test_buildable_list_restricts_each_mi_to_its_depthus(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mi_candidates, "MI_LISTS_DIR", tmp_path / "lists")
    cfg = dict(_config(), BUILDABLE_MIS=True, MI_FILTER=0)
    a, b, c = (16, 16, 32, 1, 1, 1, 1, 4, 1), (32, 32, 16, 1, 1, 1, 1, 4, 1), (16, 16, 32, 1, 1, 2, 1, 4, 1)
    _write_list(cfg, [a, b, c], depthu={a: [64, 128], b: [512]}, searched=[32, 64, 128, 256, 512])
    d = mid.MIDesign(tmp_path, cfg)
    pairs = _pairs(d.generate_for_size((4096, 4096, 1, 4096), depthu_values=[16, 64, 256]))
    # a: its DepthU 64, plus 16, which the search never tried; b: no recorded DepthU offered, so only 16;
    # c: no DepthUs recorded, so every offered one.
    assert pairs == {(a, 16), (a, 64), (b, 16), (c, 16), (c, 64), (c, 256)}
    groups = d.generate_for_size((4096, 4096, 1, 4096), depthu_values=None)  # no DepthU axis: nothing to restrict
    assert {du for _, du in _pairs(groups)} == {None}


def test_without_a_list_mi_groups_keep_the_depthu_axis(tmp_path: Path) -> None:
    groups = mid.MIDesign(tmp_path, dict(_config(), MI_FILTER=0)).generate_for_size((4096, 4096, 1, 4096), [16, 64])
    assert {du for _, du in _pairs(groups)} == {None}



def test_buildable_list_path_is_per_compiler_target(tmp_path: Path) -> None:
    a = mi_candidates.buildable_list_path("gfx950_128cu", "BBS_TN")
    b = mi_candidates.buildable_list_path("gfx950", "BBS_TN")
    assert a == b and a.parts[-2:] == ("gfx950", "BBS_TN.yaml")


def test_load_buildable_list_rejects_malformed_and_old_format_files(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    for text in ("Architecture: gfx950\n", "MatrixInstructions:\n  - [16, 16, 32, 1, 1, 1, 1, 4, 1]\n"):
        p.write_text(text)
        with pytest.raises(ValueError, match="not a buildable-MI list"):
            mi_candidates.load_buildable_list(p)


def test_allowed_depthu_keeps_depthus_the_search_did_not_try() -> None:
    mi = (16, 16, 32, 1, 1, 1, 1, 4, 1)
    bl = mi_candidates.BuildableList([mi], {mi: [64, 128]}, [32, 64, 128, 256])
    assert bl.allowed_depthu(mi, [16, 32, 64, 128, 256, 512]) == [16, 64, 128, 512]
    assert bl.allowed_depthu((32, 32, 16, 1, 1, 1, 1, 4, 1), [32, 64]) == [32, 64]  # nothing recorded for it


def test_sort_and_remove_gsu_duplicates_and_log_path(tmp_path: Path) -> None:
    d = mid.MIDesign(tmp_path, _config())
    entries = [
        mid.MFMACandidate(totalGranularity=0.8, TilesPerCU=2.2, mfma=mid.MFMA(16, 16, 16, 1, 1, 1, 1, 2, 2), NumTile0=0, NumTile1=0, Tile0Granularity=0, Tile1Granularity=0, TotalTiles=0, CUGranularity=0, waveGranularity=0, GSU=2, LSU=1, MT0=64, MT1=64, TT0=1, TT1=1, WG0=16, WG1=16, MIBlockM=1),
        mid.MFMACandidate(totalGranularity=0.9, TilesPerCU=2.2, mfma=mid.MFMA(16, 16, 16, 1, 1, 1, 1, 2, 2), NumTile0=0, NumTile1=0, Tile0Granularity=0, Tile1Granularity=0, TotalTiles=0, CUGranularity=0, waveGranularity=0, GSU=3, LSU=1, MT0=64, MT1=64, TT0=1, TT1=1, WG0=16, WG1=16, MIBlockM=1),
    ]
    sorted_entries = d._sort_mfmas(entries[:])
    assert len(sorted_entries) == 2
    dedup = d._remove_GSU_duplicates(sorted_entries)
    assert len(dedup) == 1
    p = d.get_mi_finder_log_name((16, 32, 1, 64))
    assert p.name.endswith("_M16_N32_B1_K64.log")


def test_to_group_dimension_and_generate_for_size_streamk(monkeypatch, tmp_path: Path) -> None:
    cfg = _config(streamk=True)
    d = mid.MIDesign(tmp_path, cfg)

    fake_mfma = [
        mid.MFMACandidate(totalGranularity=1.0, TilesPerCU=2.0, mfma=mid.MFMA(16, 16, 16, 1, 1, 1, 1, 2, 2), NumTile0=1, NumTile1=1, Tile0Granularity=1, Tile1Granularity=1, TotalTiles=1, CUGranularity=1, waveGranularity=1, GSU=1, LSU=1, MT0=64, MT1=64, TT0=1, TT1=1, WG0=16, WG1=16, MIBlockM=1),
        mid.MFMACandidate(totalGranularity=1.0, TilesPerCU=2.0, mfma=mid.MFMA(16, 16, 16, 1, 1, 1, 1, 2, 2), NumTile0=1, NumTile1=1, Tile0Granularity=1, Tile1Granularity=1, TotalTiles=1, CUGranularity=1, waveGranularity=1, GSU=4, LSU=1, MT0=64, MT1=64, TT0=1, TT1=1, WG0=16, WG1=16, MIBlockM=1),
    ]

    monkeypatch.setattr(d, "_find_mi_for_size", lambda *_a, **_k: (fake_mfma, 2.0))

    out = d.generate_for_size((16, 16, 1, 16))
    assert len(out) == 1
    assert "MatrixInstruction" in out[0]
    fp = out[0]["MatrixInstruction"]
    assert fp.name == "MatrixInstruction"
    assert fp.metadata["MT"] == (64, 64)


def test_find_mi_for_size_refine_filter_level2(monkeypatch, tmp_path: Path) -> None:
    cfg = _config(streamk=False)
    cfg["MI_FILTER"] = 2
    d = mid.MIDesign(tmp_path, cfg)

    # Force deterministic MT/WG properties for two candidate MIs so refine filtering runs.
    def _calc_mfma(MI, waveFrontSize=64):
        mi = MI
        if isinstance(mi, mid.MFMA):
          m_val = mi.M
        else:
          m_val = mi[0]
        if m_val == 16:
            return mid.MFMAParameters(MT0=64, MT1=64, TT0=2, TT1=2, WG0=8, WG1=8, MIBlockM=1)
        return mid.MFMAParameters(MT0=128, MT1=128, TT0=1, TT1=1, WG0=4, WG1=4, MIBlockM=1)

    monkeypatch.setattr(mid.MIDesign, "calculate_mfma_parameters", staticmethod(_calc_mfma))

    # Keep granularity formulas simple and drive a spread in rounds/granularity.
    def _calc_gran(
        MT0, MT1, M, N, batch_count, CUs, LSU, GSU, wave
    ):
        total_tiles = max(1, (M // max(MT0, 1)) * (N // max(MT1, 1)) * batch_count * GSU * LSU)
        tiles_per_cu = total_tiles / CUs
        gran = 1.0 if (MT0, MT1) == (64, 64) else 0.35
        return mid.GranularityMetrics(
            NumTile0=M / MT0,
            NumTile1=N / MT1,
            Tile0Granularity=1.0,
            Tile1Granularity=1.0,
            TotalTiles=total_tiles,
            TilesPerCU=tiles_per_cu,
            CUGranularity=min(1.0, tiles_per_cu),
            waveGranularity=1.0,
            totalGranularity=gran,
        )

    monkeypatch.setattr(mid.MIDesign, "calculate_granularities", staticmethod(_calc_gran))

    valid_mfmas = [
        mid.MFMA(16, 16, 16, 1, 1, 1, 1, 2, 2),
        mid.MFMA(32, 32, 16, 1, 1, 1, 1, 2, 2),
    ]

    mfma_list, max_tiles = d._find_mi_for_size(valid_mfmas, 16, 16, (512, 512, 1, 2048))
    assert max_tiles >= 0
    # Level-2 filter should still leave at least one candidate.
    assert len(mfma_list) >= 1
