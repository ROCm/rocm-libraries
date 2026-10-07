################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R8 — SubtileGREmit TDM iterate-mode characterization (gfx1250, CPU-only).

Targets the TDM-subtile iterate-mode global-read emit added by the
"subtile tdm iterate mode for large depthu" feature (#9410,
24975741a1e). No prior char config paired UseSubtileImpl with TDMInst at a
DepthU large enough to trip the iterate predicate, so
Tensile/Components/Subtile/SubtileGREmit.py's iterate arms (around lines
1155-1210) were never emitted by the characterization lane.

isSubtileIterateMode(state, tc) (Tensile/SolutionStructs/Utilities.py) is:

    UseSubtileImpl AND enableTDM<tc> AND DepthU * DataType<tc>.numBytes() > 1024

For BF16 (numBytes == 2):
  DepthU=1024 -> 2048 > 1024 -> iterate TRUE  (large-DepthU iterate arms)
  DepthU=256  ->  512 < 1024 -> iterate FALSE (non-iterate baseline arms)

The config forks 2 MatrixInstruction x 2 DepthU = 4 valid gfx1250 kernels,
so a single emit exercises both the iterate and non-iterate descriptor paths.

pytestmark = pytest.mark.unit. CPU-only; no GPU, no compile, no hardware.
"""

import os

import pytest

from config_harness import emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "subtile_tdm_iterate.yaml",
)


def test_r8_iterate_predicate_tracks_depthu():
    """isSubtileIterateMode is True only for the large-DepthU (1024) fork.

    Pins the #9410 trigger: DepthU=1024 BF16 crosses the 1024-byte iterate
    threshold on both A and B; DepthU=256 stays under it. If the predicate
    stops tracking DepthU the iterate arms would go dark again.
    """
    from Tensile.SolutionStructs.Utilities import isSubtileIterateMode

    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=8)
    assert len(sols) == 4, f"Expected 4 forked solutions, got {len(sols)}"

    by_depthu = {}
    for s in sols:
        assert s.get("Valid", False), f"solution not valid: DepthU={s.get('DepthU')}"
        assert s.get("UseSubtileImpl") is True, "UseSubtileImpl must be set"
        assert s.get("enableTDMA") and s.get("enableTDMB"), "TDM must be enabled A+B"
        du = s["DepthU"]
        by_depthu.setdefault(du, []).append(
            (isSubtileIterateMode(s, "A"), isSubtileIterateMode(s, "B"))
        )

    assert set(by_depthu) == {256, 1024}, f"Expected DepthU {{256,1024}}, got {set(by_depthu)}"
    for flags in by_depthu[1024]:
        assert flags == (True, True), f"DepthU=1024 must be iterate on A+B, got {flags}"
    for flags in by_depthu[256]:
        assert flags == (False, False), f"DepthU=256 must be non-iterate, got {flags}"


def test_r8_iterate_config_emits_assembly():
    """All 4 iterate/non-iterate forks emit real gfx1250 assembly (err == 0).

    Drives the full emit pipeline through the iterate-mode subtile global-read
    descriptor path for the DepthU=1024 forks and the non-iterate path for the
    DepthU=256 forks.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) == 4, f"Expected 4 kernels, got {len(results)} (config: {_CONFIG})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )

    depthus = set()
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"kernel {base!r}: suspiciously short assembly"
        )
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx1250" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"
        if "MT32x32x1024" in base:
            depthus.add(1024)
        elif "MT32x32x256" in base:
            depthus.add(256)

    assert depthus == {256, 1024}, (
        f"Expected both DepthU=256 and DepthU=1024 kernels, got {depthus}"
    )
