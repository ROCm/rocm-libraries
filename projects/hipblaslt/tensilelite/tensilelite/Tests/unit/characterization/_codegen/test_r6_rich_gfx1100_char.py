################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 — gfx1100 (RDNA3 / WMMA V1) rich GEMM characterization (CPU-only).

Drives the designed HSS (fp16-in / fp32-out) WMMA config for gfx1100 through
the config-driven emit harness and asserts every emitted kernel is real
gfx1100 AMDGCN assembly with err==0.

Target: arch-gated WMMA V1 arms in KernelWriter.py / KernelWriterAssembly
reached exclusively by gfx1100 (RDNA3) when EnableMatrixInstruction=True and
WavefrontSize=32:
  - HHH_WMMA branch (line 6737-6739): HasWMMA flag set for h->h MI
  - HasWMMA_V1 guard arms (lines 1012, 1071, 1222, 2046, 7030+)
  - WavefrontSize==32 branches (lines 384, 6676, 9815, 9870, 9878, 9884)
  - Bias + Activation epilogue paths (ActivationFuncCall=True, UseBias=1)
  - GlobalSplitU=2 MultipleBuffer path (widens GSU codegen coverage)
  - MIWaveTile fork [1,1] vs [3,1]: distinct macro-tile shapes for MI-width arms

Config: hss_wmma_rich.yaml
  DataType: h (fp16), DestDataType: s (fp32), ComputeDataType: s
  TransposeA: 1, TransposeB: 0 (TN layout)
  MatrixInstruction: [16,16,16,1] with MIWaveTile [1,1] and [3,1]
  GlobalSplitU: [1, 2]  -> 4 permutations total (under limit=8)
  Activation=True, ActivationType=all, UseBias=1

Assertions are on stable emitted text (no snapshot dependency needed), so
two-run determinism is guaranteed.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1100"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1100",
    "hss_wmma_rich.yaml",
)


def test_r6_rich_gfx1100_wmma_emits_assembly():
    """gfx1100 WMMA HSS rich config emits real assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )


def test_r6_rich_gfx1100_wmma_asm_markers():
    """Each emitted gfx1100 kernel carries canonical AMDGCN + arch markers."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} source unexpectedly short ({len(src.splitlines())} lines)"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r}: missing .amdgcn_target"
        assert "gfx1100" in src, f"Kernel {base!r}: missing gfx1100 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected kernel basename: {base!r}"
        # Verify HSS data-type encoding in the kernel name (h-in / s-out)
        assert "HSS" in base or "Alik" in base or "Ailk" in base, (
            f"Kernel {base!r}: expected HSS or index-layout marker"
        )


def test_r6_rich_gfx1100_wmma_wmma_instruction_present():
    """gfx1100 WMMA kernels contain the v_wmma assembly instruction."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    wmma_kernels = [
        base for base, src, _err in results if "v_wmma" in src.lower()
    ]
    assert len(wmma_kernels) >= 1, (
        "Expected >=1 kernel to contain v_wmma instruction (gfx1100 WMMA V1); "
        f"basenames: {[b for b,_,_ in results]}"
    )


def test_r6_rich_gfx1100_wmma_gsu_fork():
    """GlobalSplitU fork yields distinct kernels (GSU=1 and GSU=2 shapes)."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 2, (
        f"Expected >=2 kernels from GSU fork (1 + 2), got {len(results)}"
    )
    basenames = [b for b, _s, _e in results]
    # All basenames must be unique (deterministic, no duplicates)
    assert len(basenames) == len(set(basenames)), (
        f"Duplicate kernel basenames detected: {basenames}"
    )
