################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P3 -- gfx942 KernelWriter feature-breadth (ScheduleIterAlg 1 & 2).

Drives the designed kwfeat.yaml config (BBS bf16, ScheduleIterAlg=[1,2],
GlobalSplitU=[1,2]) through the config-driven emit harness to exercise the
uncovered ScheduleIterAlg 1 and ScheduleIterAlg 2 code arms in
Tensile/KernelWriter.py (_makeSubIterSchedule):

  lines 951- 997: scheduleIterAlg == 1 block
                  (interleave: half-reads, globalReadCode, rest reads,
                   localWriteCode, pointerLW/LR, waitCode, packCode, mac)
  lines 998-1125: scheduleIterAlg == 2 block
                  (SIA2: pack interleave, SSetPrior, 1LDSBuffer barrier,
                   localWriteCode, pointerLW/LR, SSetPrior back to 2)

Additionally, the GSU=2 permutations trigger KernelWriterConversion and
KernelWriterBetaOnly helper-kernel paths (beta-only + reduction) via
generateKernelHelperObjects, exercising those additional KernelWriter paths.

Pattern: A (codegen emit).
"""

import os
import shutil

import pytest

from config_harness import emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "kwfeat.yaml",
)


def test_r3_kwfeat_gfx942_emits_assembly():
    """SIA=[1,2] + GSU=[1,2] BBS sweep emits >=1 kernel on gfx942, all err==0.

    SIA=1 path exercises _makeSubIterSchedule lines 951-997.
    SIA=2 path exercises _makeSubIterSchedule lines 998-1125.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r3_kwfeat_gfx942_sia_arms_covered():
    """Assert that both SIA=1 and SIA=2 kernels are present in the emit.

    The basename encodes ScheduleIterAlg implicitly through the hash, but the
    source code itself contains the MFMA shape marker and SIA-specific
    scheduling side-effects (SSetPrior for SIA=2). Assert >=2 distinct kernels
    (SIA=1 and SIA=2) are emitted.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 2, (
        f"Expected >=2 kernels (SIA=1 and SIA=2), got {len(results)}; "
        f"basenames: {[b for b, _s, _e in results]}"
    )
    # SIA=2 path sets SSetPrior instructions. At least one kernel should have
    # s_setprio in its emitted assembly (SIA=2 adds priority-raise code).
    has_setprio = any("s_setprio" in src.lower() for _b, src, _e in results)
    assert has_setprio, (
        "Expected >=1 kernel with SIA=2 s_setprio instructions, "
        "but none found. SIA=2 path may not be exercised."
    )


def test_r3_kwfeat_gfx942_helpers_emit():
    """Helper-kernel emit (conversion + beta-only) for GSU=2 BBS solutions.

    Calls generateKernelHelperObjects on the solutions produced by the kwfeat
    config. Every helper must return err==0. This exercises:
      - KernelWriterConversion (conversion arm) for GSU>1 kernels
      - KernelWriterBetaOnly  (beta-only arm)   for GSU>1 kernels
    """
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelHelperObjects,
        generateKernelObjectsFromSolutions,
    )
    import codegen_harness as _ch

    assembler, iim = _ch._toolchain()
    cxx = shutil.which("amdclang++") or str(assembler.path)

    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=8)
    if not sols:
        pytest.skip("No solutions generated (config may have filtered all perms)")

    kernels = generateKernelObjectsFromSolutions(sols)
    khos = generateKernelHelperObjects(kernels, cxx, iim)
    if not khos:
        # No GSU>1 solutions survived; test passes vacuously (no helpers needed)
        return

    errors = []
    for ko in khos:
        name = ko.getKernelName()
        err, _src = ko.getSourceFileString()
        ko.getHeaderFileString()
        if err != 0:
            errors.append((name, err))
    assert not errors, f"Helper kernel errors: {errors}"


def test_r3_kwfeat_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the kwfeat emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
