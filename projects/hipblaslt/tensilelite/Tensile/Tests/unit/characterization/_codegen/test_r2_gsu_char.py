################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P4 — gfx942 GlobalSplitU (GSU) family characterization (add-only, CPU-only).

Drives the GSU-sweep designed config through the config-driven emit harness and
exercises the uncovered GSU.py branches in methodology-A missing ranges:
  170-233, 246-301 (GSUOff paths)
  388-461, 480-588 (GSUOn computeLoadSrd / graIncrements)
  646-698 (GSUOn noLoadLoop / tailLoopNumIter)

ForkParameters swept:
  GlobalSplitU:                        [1, 3, 8]
  GlobalSplitUAlgorithm:               [SingleBuffer, MultipleBuffer]
  GlobalSplitUWorkGroupMappingRoundRobin: [False, True]

Routing:
  GSU=1   -> GSUOff (lines 197-301)
  GSU>1   -> GSUOn  (lines 303+)
  GSUWGMRR=True -> round-robin branch in graWorkGroup (lines 354-357)
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
    "gsu.yaml",
)


def test_r2_gsu_emits_assembly():
    """GSU-family sweep emits >=1 gfx942 kernel and all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100
        assert base.startswith("Cijk_")


def test_r2_gsu_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the GSU sweep emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
