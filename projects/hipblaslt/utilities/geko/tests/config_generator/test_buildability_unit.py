# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geko.config_generator.buildability import verdict
from geko.config_generator.buildability.measurement import Measurement
from geko.config_generator.buildability.search import SettingSearch


def test_buildable_depthus_searches_every_depthu_below_the_witness_and_stops_upward(monkeypatch) -> None:
    builds = {32, 128, 256, 1024}  # 64 fails between buildable DepthUs; 512 ends the upward run
    monkeypatch.setattr(verdict, "classify",
                        lambda checker, search, mi: SimpleNamespace(buildable=search.depthu in builds))
    space = SimpleNamespace(fork={"DepthU": [32, 64, 128, 256, 512, 1024]})
    search = SimpleNamespace(space=space, pinned=lambda d: SimpleNamespace(depthu=d))
    found = verdict.buildable_depthus(None, search, (16, 16, 32, 1, 1, 1, 1, 4, 1), {"DepthU": 128})
    assert found == [32, 128, 256]


# gfx942 BBS TT DepthU 128 built only through a neighbor repair had already measured; the fix for that lost 44
# gfx950 pairs that build only through a neighbor the walk measures itself.
@pytest.mark.parametrize("winning_widths", [(4, 2), (2, 4)], ids=["via-repaired-neighbor", "via-new-neighbor"])
def test_classify_follows_both_kinds_of_improving_neighbor(monkeypatch, winning_widths) -> None:
    mi = (4, 4, 4, 16, 1, 12, 1, 1, 1)
    group = "group_MatrixInstruction"
    space = SimpleNamespace(
        wavefront=64, mi_group=group, groups={},
        fork={"GlobalReadVectorWidthA": [1, 2, 4], "GlobalReadVectorWidthB": [2, 4], "1LDSBuffer": [0, 1],
              "DepthU": [128]},
    )
    cfg = {"ARCH": "gfx942", "GemmProblems": [SimpleNamespace(gemm_type=SimpleNamespace(data_type="B"))]}
    search = SettingSearch(space, cfg)
    seed = {group: {"MatrixInstruction": list(mi)}, "DepthU": 128, "1LDSBuffer": 0,
            "GlobalReadVectorWidthA": 2, "GlobalReadVectorWidthB": 2}
    refused = {**seed, "GlobalReadVectorWidthA": 1}  # repair measures {A: 4} before the walk
    witness = {**seed, "GlobalReadVectorWidthA": winning_widths[0], "GlobalReadVectorWidthB": winning_widths[1],
               "1LDSBuffer": 1}
    monkeypatch.setattr(search, "presets", lambda _mi, entry: [refused] if entry == seed[group] else [])
    monkeypatch.setattr(search, "core", lambda _mi, entry: [seed] if entry == seed[group] else [])
    monkeypatch.setattr(verdict, "MULTI_STARTS", 1)  # walk from the seed only
    lds = {(2, 2): 120, (4, 2): 110, (2, 4): 115}  # the repaired neighbor improves most

    def measure(setting):
        if setting == witness:
            return Measurement(err=0)
        widths = (setting["GlobalReadVectorWidthA"], setting["GlobalReadVectorWidthB"])
        if setting["1LDSBuffer"] == 0 and widths in lds:
            return Measurement(err=8, lds=lds[widths], lds_cap=100)
        return Measurement(err=-1, reason="invalid GlobalReadVectorWidthA")

    checker = SimpleNamespace(measure=measure, accepts=lambda setting: setting["GlobalReadVectorWidthA"] != 1)
    assert verdict.classify(checker, search, mi).witness == witness
