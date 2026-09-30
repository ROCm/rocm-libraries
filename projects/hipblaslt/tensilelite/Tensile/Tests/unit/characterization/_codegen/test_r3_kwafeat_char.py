################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 — KWA broad feature-breadth characterization (gfx942, no GPU required).

Sweeps ScheduleIterAlg:[1,2] x StaggerU:[0,32] x ExpandPointerSwap:[0,1] to
exercise previously uncovered emit arms in Tensile/KernelWriterAssembly.py:

Target missing ranges in KernelWriterAssembly.py (miss=3987 at HEAD):
  - 5893-5966 : declareStaggerParms — activated by StaggerU != 0
  - 6000-6099 : staggerU address adjustment in globalAddressIncrement
  - 7380, 7627-7674 : ExpandPointerSwap loop-exit and odd/even select
  - 12865, 12895 : EPS local-read swap-offsets

Target missing ranges in Components/SIA.py (miss=128 at HEAD):
  - 102-119 : SIA2.schedIntoIteration (only runs when _ScheduleIterAlg==2)
  - 130-149 : SIA1.schedIntoIteration (only runs when _ScheduleIterAlg==1)

Target missing ranges in KernelWriter.py (miss=1879 at HEAD):
  - 951-997  : SIA1 scheduling code block in makeSchedule()
  - 998-1125 : SIA2 scheduling code block in makeSchedule()

The sweep uses SGEMM NT (f32), Batched, MatrixInstruction=[16,16,4,...] which
is the minimal shape that satisfies SIA2's MatrixInstruction requirement on
gfx942 without DirectToVgpr (rejected when SIA<3).

CPU-only. No GPU, no compile, no hardware access.
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
    "kwafeat.yaml",
)


def test_r3_kwafeat_gfx942_emits_assembly():
    """KWA feature-breadth sweep emits >=1 valid gfx942 kernel, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 kernel from kwafeat sweep, got {len(results)}"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r}: suspiciously short assembly"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r3_kwafeat_gfx942_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for the KWA feature sweep."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
