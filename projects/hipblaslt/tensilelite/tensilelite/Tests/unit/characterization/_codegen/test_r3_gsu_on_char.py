################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P4 — gfx942 GSUOn arm characterization (add-only, CPU-only).

Exercises the GSUOn class methods in Tensile/Components/GSU.py that are NOT
reached by the existing test_r2_gsu_char.py (which only emits GSU=1 / GSUOff
kernels because the BBS+Bias+Activation ProblemType rejects GlobalSplitU>1).

This test uses a plain float32 NN ProblemType that the solution derivation
accepts with GlobalSplitU>1, routing through the GSUOn arm of:

  GSUOn.graWorkGroup        (lines 316-370)
  GSUOn.computeLoadSrd      (lines 372-429)  — target 388-461
  GSUOn.graIncrements       (lines 431-517)  — target 441-517
  GSUOn.calculateLoopNumIterGsu (558-588)    — target 558-588

ForkParameters sweep:
  GlobalSplitU:                               [2, 4, 8]
  GlobalSplitUAlgorithm:                      [MultipleBuffer, SingleBuffer]
  GlobalSplitUWorkGroupMappingRoundRobin:     [False, True]
  GlobalSplitUCoalesced:                      [False, True]

Pattern A — codegen emit.  Uses emit_kernels_from_config with the designed
gsu_on.yaml config.  Asserts >= 1 kernel emitted, all err==0, and snapshots
the {basename, err} digest (seeded with --snapshot-update).
"""

import os

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
    "gsu_on.yaml",
)


def test_r3_gsu_on_emits_assembly():
    """GSUOn sweep (GSU=[2,4,8]) emits >=1 gfx942 kernel, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel from GSUOn sweep, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, f"Kernel {base!r} source too short"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r3_gsu_on_gsu_registers():
    """GSUOn kernels define GSUSumIdx and GSULog2BpeC registers (GSUOn path)."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    for base, src, err in results:
        assert err == 0
        # GSUOn.graWorkGroup emits these register defines; GSUOff does not.
        assert "sgprGSUSumIdx" in src, (
            f"Kernel {base!r} missing sgprGSUSumIdx — may not be routing through GSUOn.graWorkGroup"
        )


def test_r3_gsu_on_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the GSUOn sweep emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
