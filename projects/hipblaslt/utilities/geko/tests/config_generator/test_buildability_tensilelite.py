# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The buildable-MI search against real TensileLite (needs --hipblaslt-path, e.g. tox -e integration): the
TensileLite rules and calculations buildability.py restates still match TensileLite, on gfx950 SSS NN."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

# From the committed gfx950 SSS_NN list: buildable, and rejected for VGPRs (code 1) and for occupancy (code 4).
BUILDABLE = (16, 16, 4, 1, 1, 1, 1, 4, 1)
REJECTED = [(16, 16, 4, 1, 1, 1, 49, 4, 1), (16, 16, 4, 1, 1, 1, 55, 4, 1)]
CODE_TERM = {1: "vgpr", 2: "sgpr", 4: "occupancy"}


@pytest.fixture(scope="module")
def tl(request, tmp_path_factory):
    hip = request.config.getoption("--hipblaslt-path")
    if hip is None:
        pytest.skip("Requires --hipblaslt-path")
    from geko.utils import ensure_tensile_importable

    ensure_tensile_importable(Path(hip))
    from geko.config_generator import buildability as B

    cfg = dict(TRANSA="N", TRANSB="N", DataType="S", DestDataType="S", ComputeDataType="S", ARCH="gfx950")
    path, cfg = B.write_probe_config(cfg, Path(hip), tmp_path_factory.mktemp("probe"))
    checker = B.CodegenChecker(path, "gfx950")
    return B, checker, B.SettingSearch(checker.space, cfg)


def test_a_listed_mi_builds_within_every_modelled_cap(tl) -> None:
    B, checker, search = tl
    v = B.classify(checker, search, BUILDABLE)
    assert v.buildable and v.over_cap == []


@pytest.mark.parametrize("mi", [BUILDABLE, (16, 16, 4, 1, 1, 1, 1, 1, 1)])  # LSU 1; LSU 1, 2, 4
def test_depthus_below_the_modelled_minimum_are_ones_tensilelite_refuses(tl, mi) -> None:
    B, checker, search = tl
    for entry in search.mi_entries(mi):
        m = checker.measure(dict(search._base(entry), DepthU=search._min_depthu(entry) // 2))
        assert m.err == -1 and ("DepthU" in m.reason or "LoopIters" in m.reason), (entry, m.reason)


@pytest.mark.parametrize("mi", REJECTED)
def test_overflow_codes_are_explained_by_the_matching_term(tl, mi) -> None:
    B, checker, search = tl
    v = B.classify(checker, search, mi)
    assert not v.buildable
    m = checker.measure(v.witness)  # the closest kernel
    assert m.err in CODE_TERM, m.reason
    assert m._usage()[CODE_TERM[m.err]] > 1, (m.err, m._usage())


def test_lds_refusals_carry_a_distance(tl) -> None:
    B, checker, search = tl
    witness = B.classify(checker, search, BUILDABLE).witness
    m = checker.measure(dict(witness, DepthU=max(search.space.fork["DepthU"]) * 8))
    assert m.err != 0 and m.resource() == "lds" and m.overflow() > 0
