################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 -- gfx1250 rich GEMM (SB / XF32-emulation) characterization (CPU-only).

Exercises gfx1250-specific codegen arms in ``KernelWriterAssembly.py`` that
are only reachable on WMMA_V3 architectures (gfx1250).  The key gate is
``is_wmma_v3 = asmCaps["HasWMMA_V3"]`` combined with
``kernel["UseF32XEmulation"]``, which is set by Solution.py when
``F32XdlMathOp=X`` (XFloat32, enum=10) is requested on a WMMA_V3 arch.

Target missing ranges in Tensile/KernelWriterAssembly.py:
  8646-8651  is_wmma_v3+UseF32XEmulation branch (vgprPerSet0Group=1)
  8669-8674  is_wmma_v3+UseF32XEmulation multiplyBy for group==0
  8680-8687  is_wmma_v3+UseF32XEmulation kOffsetA for group>0
  9403-9439  UseF32XEmulation 3-MFMA chain (src0_h*src1_h, cross terms)
  7964-7966  UseScaleAlphaVec sgpr->vgpr address (loadScaleAlphaVec arm)
  8080-8083  UseScaleAlphaVec post-loop address setup
 13340-13348 UseScaleAlphaVec vgprPool checkOut (allocTmpSgpr block)
 13358-13365 shiftSrd gfx125x block (version[:2]==(12,5))
 14760-14810 UseScaleAlphaVec bias-D epilogue
 14871-14960 BiasDataTypeList multi-type loop in writeBiasToGlobal

Config: DataType=S (float32), F32XdlMathOp=X (XFloat32 emulation), NT layout,
WavefrontSize=32, TDMInst=3, UseScaleAB=Scalar, UseScaleAlphaVec=1,
BiasDataTypeList=[s,b], Activation=True (hipblaslt_all), BiasSrc=D.
Fork on StaggerU x ScheduleIterAlg x WorkGroupMapping (4 permutations;
dedup yields 2 unique kernels, both with err==0).

Pure-assert test; no syrupy snapshot required.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "rich_gemm_sb.yaml",
)


def test_r6_rich_gfx1250_emits_assembly():
    """F32->XF32 rich GEMM emits real gfx1250 AMDGCN assembly with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 kernel from rich_gemm_sb.yaml, got {len(results)}. "
        "Check that DataType=S + F32XdlMathOp=X + gfx1250 WMMA_V3 is valid."
    )
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} emitted with err={err}, expected 0"
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} source suspiciously short ({len(src.splitlines())} lines)"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r6_rich_gfx1250_xf32_emulation_present():
    """F32->XF32 kernels contain the UseF32XEmulation 3-MFMA chain marker.

    KernelWriterAssembly.py lines 9403-9418 emit ``src0_h`` / ``src1_h``
    labels when ``UseF32XEmulation=True``, splitting the f32 input into
    high-half (``src0_h``) and low-half operands for the 3-MFMA expansion.
    This string is canonical across gfx1250 XF32 emulation kernels.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Need at least one kernel to check XF32 markers"
    found_xf32 = False
    for base, src, err in results:
        if err != 0 or not src:
            continue
        # KWA 9408-9418: UseF32XEmulation splits input into src0_h/src1_h halves
        if "src0_h" in src or "src1_h" in src:
            found_xf32 = True
        # Verify the gfx1250 WMMA_V3 target is present
        assert "gfx1250" in src, f"Kernel {base!r}: missing gfx1250 target string"
    assert found_xf32, (
        "Expected at least one kernel containing 'src0_h' (UseF32XEmulation "
        "3-MFMA chain, KWA 9408-9418) — check F32XdlMathOp=X on gfx1250"
    )


def test_r6_rich_gfx1250_bias_and_scale_present():
    """F32->XF32 kernels contain bias and scale alpha-vec address setup.

    KernelWriterAssembly.py lines 13340-13348 and 14760-14810 emit moves
    and checks for UseScaleAlphaVec. The comment ``addrScaleAlphaVec``
    appears in the allocation block (13341) and the post-loop setup (8080).
    The bias write path (14871+) uses ``BiasDataTypeList`` iteration.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Need at least one kernel to check bias/scale"
    found_scale = False
    found_bias = False
    for base, src, err in results:
        if err != 0 or not src:
            continue
        src_lower = src.lower()
        if "scalealphavec" in src_lower or "addressscale" in src_lower:
            found_scale = True
        if "bias" in src_lower:
            found_bias = True
    assert found_scale, (
        "Expected at least one kernel to reference ScaleAlphaVec "
        "(KWA 13340-13348, 7964, 8080) — check UseScaleAlphaVec=1"
    )
    assert found_bias, (
        "Expected at least one kernel to reference bias "
        "(KWA 14760-14960) — check UseBias=1, BiasSrc=D"
    )
