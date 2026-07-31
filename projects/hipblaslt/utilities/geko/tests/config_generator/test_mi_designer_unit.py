# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from geko.config_generator import mi_designer as mid
from geko.schemas import GemmType


def _config(streamk: bool = False) -> dict:
    gt = GemmType.from_tensile("N", "N", "H", "H", "S")
    return {
        "ARCH": "gfx950",
        "CUs": 256,
        "MI_FILTER": 1,
        "StreamK": streamk,
        "GemmProblem": type("GP", (), {"gemm_type": gt})(),
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
    assert len(out) == 9
    assert out[0] == 2.0
    assert out[1] == 2.0

    mi = [16, 16, 16, 1, 1, 2, 2, 2, 2]
    params = mid.MIDesign.calculate_mfma_parameters(mi)
    assert len(params) == 7
    assert params[0] > 0 and params[1] > 0


def test_generate_all_mfmas_uses_override(tmp_path: Path) -> None:
    cfg = _config()
    cfg["MFMA"] = [16, 16, 16, 1]
    d = mid.MIDesign(tmp_path, cfg)
    valid, sm, sn = d.generate_all_mfmas()
    assert sm == 16 and sn == 16
    assert len(valid) > 0


def test_sort_and_remove_gsu_duplicates_and_log_path(tmp_path: Path) -> None:
    d = mid.MIDesign(tmp_path, _config())
    entries = [
        (0.8, 2.2, (16, 16, 16, 1, 1, 1, 1, 2, 2), 0, 0, 0, 0, 0, 0, 0, 2, 1, 64, 64, 1, 1, 16, 16, 1),
        (0.9, 2.2, (16, 16, 16, 1, 1, 1, 1, 2, 2), 0, 0, 0, 0, 0, 0, 0, 3, 1, 64, 64, 1, 1, 16, 16, 1),
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
        (1.0, 2.0, (16, 16, 16, 1, 1, 1, 1, 2, 2), 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 64, 1, 1, 16, 16, 1),
        (1.0, 2.0, (16, 16, 16, 1, 1, 1, 1, 2, 2), 1, 1, 1, 1, 1, 1, 1, 4, 1, 64, 64, 1, 1, 16, 16, 1),
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
        if mi[0] == 16:
            return (64, 64, 2, 2, 8, 8, 1)
        return (128, 128, 1, 1, 4, 4, 1)

    monkeypatch.setattr(mid.MIDesign, "calculate_mfma_parameters", staticmethod(_calc_mfma))

    # Keep granularity formulas simple and drive a spread in rounds/granularity.
    def _calc_gran(
        MT0, MT1, M, N, batch_count, CUs, LSU, GSU, wave
    ):
        total_tiles = max(1, (M // max(MT0, 1)) * (N // max(MT1, 1)) * batch_count * GSU * LSU)
        tiles_per_cu = total_tiles / CUs
        gran = 1.0 if (MT0, MT1) == (64, 64) else 0.35
        return (
            M / MT0,
            N / MT1,
            1.0,
            1.0,
            total_tiles,
            tiles_per_cu,
            min(1.0, tiles_per_cu),
            1.0,
            gran,
        )

    monkeypatch.setattr(mid.MIDesign, "calculate_granularities", staticmethod(_calc_gran))

    valid_mfmas = [
        [16, 16, 16, 1, 1, 1, 1, 2, 2],
        [32, 32, 16, 1, 1, 1, 1, 2, 2],
    ]

    mfma_list, max_tiles = d._find_mi_for_size(valid_mfmas, 16, 16, (512, 512, 1, 2048))
    assert max_tiles >= 0
    # Level-2 filter should still leave at least one candidate.
    assert len(mfma_list) >= 1
