################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 KernelWriterConversion / BetaOnly / Reduction characterization.

Drives the designed kwconv.yaml config (BBS bf16 + GSU=[2,4]) through the
config-driven emit harness and additionally exercises the helper-kernel emit
paths (KernelWriterConversion, KernelWriterBetaOnly) that are otherwise
under-covered by the methodology-A baseline.

Target source: Tensile/KernelWriterConversion.py (115 uncovered lines in the
beta-only + reduction + conversion arms per the methodology-A gap report).

Phase:
  - test_kwconv_gfx942_emits_assembly: asserts >=1 GEMM kernel, all err==0.
  - test_kwconv_gfx942_helpers_emit:   exercises conversion + beta-only helper
    paths via generateKernelHelperObjects on the same config solutions.
  - test_kwconv_gfx942_golden:         order-invariant {basename, err} snapshot.
"""

import os

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
    "kwconv.yaml",
)


def test_kwconv_gfx942_emits_assembly():
    """BBS + GSU=[2,4] config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100
        assert base.startswith("Cijk_")


def test_kwconv_gfx942_helpers_emit():
    """Helper-kernel emit (conversion + beta-only) for GSU>1 BBS solutions.

    Calls generateKernelHelperObjects on the solutions produced by the kwconv
    config. Every helper must return err==0. This exercises:
      - KernelWriterConversion.__init__ + kernelBody (conversion arm)
      - KernelWriterBetaOnly.__init__ + kernelBody (beta-only arm)
    """
    import shutil

    from Tensile.TensileCreateLibrary.Run import (
        generateKernelHelperObjects,
        generateKernelObjectsFromSolutions,
    )
    import codegen_harness as _ch

    assembler, iim = _ch._toolchain()
    cxx = shutil.which("amdclang++") or str(assembler.path)

    sols = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=8)
    # generateKernelHelperObjects takes the kernel objects (dict-like solutions),
    # not the Solution objects directly — use the same path Run.py uses.
    kernels = generateKernelObjectsFromSolutions(sols)
    khos = generateKernelHelperObjects(kernels, cxx, iim)
    assert khos, "Expected >=1 helper kernel object from GSU>1 BBS config"

    errors = []
    for ko in khos:
        name = ko.getKernelName()
        err, _src = ko.getSourceFileString()
        ko.getHeaderFileString()
        if err != 0:
            errors.append((name, err))
    assert not errors, f"Helper kernel errors: {errors}"


def test_kwconv_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the kwconv emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
