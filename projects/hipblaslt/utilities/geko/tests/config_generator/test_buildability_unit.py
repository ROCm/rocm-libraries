# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

from geko.config_generator.buildability import verdict


def test_buildable_depthus_searches_every_depthu_below_the_witness_and_stops_upward(monkeypatch) -> None:
    builds = {32, 128, 256, 1024}  # 64 fails between buildable DepthUs; 512 ends the upward run
    monkeypatch.setattr(verdict, "classify",
                        lambda checker, search, mi: SimpleNamespace(buildable=search.depthu in builds))
    space = SimpleNamespace(fork={"DepthU": [32, 64, 128, 256, 512, 1024]})
    search = SimpleNamespace(space=space, pinned=lambda d: SimpleNamespace(depthu=d))
    found = verdict.buildable_depthus(None, search, (16, 16, 32, 1, 1, 1, 1, 4, 1), {"DepthU": 128})
    assert found == [32, 128, 256]
