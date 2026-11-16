################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 WGM-algo SpaceFillingCurveWalk characterization.

Drives the R2 designed WGM config
(``data/test_data/_designed/gfx942/wgm.yaml``) through the config-driven emit
harness and asserts every emitted kernel is real gfx942 AMDGCN assembly with
err==0.

Target coverage: tensilelite/Components/WorkGroupMappingAlgos.py lines 369-410,
430-628, 655-763, 781-841, 850-1037 — the entire SpaceFillingCurveWalk code
path that is gated by ``len(kernel["SpaceFillingAlgo"]) > 0``.

Fork sweep (5 values, all within limit=8):
  [0]   col-major single-level  -> chiplet_transform (369-410) +
        SpaceFillingCurveWalk body + ColRowMajor (816-841)
  [1]   row-major single-level  -> same paths (isCol=False branch)
  [2]   Hilbert single-level    -> SpaceFillCurveSimpleImpl orderID=2 (850-1037)
  [3]   Morton-Z single-level   -> SpaceFillCurveSimpleImpl orderID=3 (850-1037)
  [0,1] two-level col+row       -> multi-level TransformNLevels paths (495-608,
        660-732) and SpaceFillingCurveWalk multi-level block (519-608)
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "wgm.yaml",
)


def test_r2_wgm_gfx942_emits_assembly():
    """SpaceFillingAlgo sweep emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        "Expected >=1 kernel from WGM SpaceFillingAlgo sweep, got 0"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got errors: {[(b, e) for b, _, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 500, (
            f"Kernel {base!r} has suspiciously short assembly ({len(src.splitlines())} lines)"
        )
        assert ".amdgcn_target" in src, f"Missing .amdgcn_target in {base!r}"
        assert "gfx942" in src, f"Missing gfx942 in {base!r}"
        assert base.startswith("Cijk_"), f"Unexpected basename prefix: {base!r}"


def test_r2_wgm_gfx942_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the WGM sweep emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot


_SPACE_FILLING_GFX1250 = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "common", "gemm", "gfx12", "space_filling.yaml"
)
_TILE_COUNT_MOVE = re.compile(r"^\s*s_mov_b32\s+s\[?(\d+)\]?,\s*s\[sgprNumWorkGroups([01])\]", re.M)
_PAIR_READ = re.compile(r"^\s*s_mov_b64\s+s\[\d+:\d+\],\s*s\[(\d+):\d+\]", re.M)


def test_r2_wgm_tile_counts_are_an_aligned_sgpr_pair(monkeypatch):
    """The walk keeps NumTilesM/N in a pair that s_mov_b64 and s_cselect_b64 read as one 64-bit
    operand, so M must sit on an even SGPR with N right after it, whatever the register pool holds.
    The walk is entered here with an odd first free SGPR."""
    import rocisa
    import tensilelite.KernelWriterAssembly as kernelWriterAssembly

    # The harness emits in this process only for the ISA the backend serves, so the patch below can reach it.
    if getattr(rocisa, "_BACKEND", "") == "stinkytofu":
        config, arch = _SPACE_FILLING_GFX1250, "gfx1250"
    else:
        config, arch = _CONFIG, _ARCH
    walk = kernelWriterAssembly.SpaceFillingCurveWalk
    entered = []

    def walkFromAnOddFreeSgpr(writer, kernel, sgprWGM):
        entered.append(kernel)
        held = []
        while True:
            sgprIdx = writer.sgprPool.checkOut(1, tag="test_oddFirstFree")
            if sgprIdx % 2:
                writer.sgprPool.checkIn(sgprIdx)
                break
            held.append(sgprIdx)
        module = walk(writer, kernel, sgprWGM)
        for sgprIdx in held:
            writer.sgprPool.checkIn(sgprIdx)
        return module

    monkeypatch.setattr(kernelWriterAssembly, "SpaceFillingCurveWalk", walkFromAnOddFreeSgpr)
    results = emit_kernels_from_config(config, limit=8, arch=arch, canonical=False)
    assert results
    assert len(entered) >= len(results), (len(entered), len(results))
    pairReads = 0
    for base, src, err in results:
        assert err == 0, base
        start = src.find("Start of Generic WGM algo")
        assert start >= 0, f"no SpaceFillingCurveWalk in {base!r}"
        tileCounts = {}
        for m in _TILE_COUNT_MOVE.finditer(src, start):
            tileCounts.setdefault(int(m.group(2)), int(m.group(1)))
        assert tileCounts[0] % 2 == 0 and tileCounts[1] == tileCounts[0] + 1, (base, tileCounts)
        pairReads += any(int(m.group(1)) == tileCounts[0] for m in _PAIR_READ.finditer(src, start))
    if config == _CONFIG:
        # The Hilbert and Morton forks read the pair with s_mov_b64 in SpaceFillCurveSimpleImpl.
        assert pairReads >= 2, pairReads
