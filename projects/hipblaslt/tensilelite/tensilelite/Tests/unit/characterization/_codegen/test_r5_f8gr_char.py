################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P5 — FP8 global-read conversion characterization (CPU-only, gfx942).

Drives two designed FP8 mixed-type BenchmarkProblems configs through the
config-driven emit harness to exercise the FP8 global-read conversion arms in
``KernelWriterAssembly.py``.

Target ranges
-------------
12115-12225  toF8 arm  (config: fp8_gr_conv.yaml):
    ``DataType{tc}.isHalf() and MacDataType{tc}.is8bitFloat()``
    Emits F16->F8 local-write conversion instructions (VCvtPkF32toFP8 /
    VCvtSRF32toFP8).  ``MacDataTypeA.isAnyFloat8()`` sets ``toF8=True``
    selecting the FP8 (not BF8) instruction path.
    Activated by ``DataType: F8N`` with ``DataTypeA: h, DataTypeB: h``.
    GLVW fork [1, 4] hits:
      - GLVW=1 -> newBlockWidth==0.5 single-element arm (lines ~12121-12159).
      - GLVW=4 -> wide conversion loop arm (lines ~12160-12225).

12231-12293  F8->H arm  (config: fp8_gr_conv_f8h.yaml):
    ``DataType{tc}.isAnyFloat8() and MacDataType{tc}.isHalf()``
    Emits FP8->F16 local-write conversion (VCvtFP8toF32 + ECvtF32toF16).
    Activated by ``DataType: h`` (MacDataType=h) with ``DataTypeA: F8N``.
    GLVW fork [1, 4] hits:
      - GLVW=1 -> ``if tP["glvw"] == 1`` single-element branch (line 12232).
      - GLVW=4 -> the ``else`` wide-conversion arm.

Architecture: gfx942 (HasMFMA_f8=1, HasCvtFP8toF16=0, NoSDWA=0).
CPU-only.  No GPU, no compile, no hardware access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"
_LIMIT = 8

_DIR = os.path.join(
    os.path.dirname(__file__), "data", "test_data", "_designed", "gfx942"
)
_CONFIG_TOF8 = os.path.join(_DIR, "fp8_gr_conv.yaml")
_CONFIG_F8H = os.path.join(_DIR, "fp8_gr_conv_f8h.yaml")


# ---------------------------------------------------------------------------
# toF8 arm: lines 12115-12225
# DataType: F8N, DataTypeA: h, DataTypeB: h
# ---------------------------------------------------------------------------


def test_r5_f8gr_toF8_emits_assembly():
    """toF8 arm (12115-12225): H+H->F8N mixed config emits gfx942 assembly, all err==0.

    DataType: F8N with DataTypeA: h, DataTypeB: h triggers
    ``DataType{tc}.isHalf() AND MacDataType{tc}.is8bitFloat()`` branch.
    GlobalReadVectorWidthA fork [1, 4] exercises:
      - GLVW=1 -> single-element (newBlockWidth==0.5) VCvtPkF32toFP8 arm.
      - GLVW=4 -> wide conversion loop (newBlockWidth>0.5) VCvtPkF32toFP8 arm.
    """
    results = emit_kernels_from_config(_CONFIG_TOF8, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG_TOF8})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"kernel {base!r}: assembly suspiciously short"
        )
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"
        # Kernels use FP8N MAC type — expect the F8N marker in the base name.
        assert "F8N" in base, (
            f"kernel {base!r}: expected F8N marker in basename (MacDataTypeA should be F8N)"
        )


def test_r5_f8gr_toF8_both_glvw_arms():
    """Both GLVW sub-arms of the toF8 path produce distinct, non-trivial kernels.

    GLVW=1 and GLVW=4 permutations both emit valid assembly.  The two kernels
    should have different basenames (different hash), confirming the fork
    produced distinct solution objects.
    """
    results = emit_kernels_from_config(_CONFIG_TOF8, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 2, (
        f"expected >=2 kernels (one per GLVW fork), got {len(results)}"
    )
    basenames = [b for b, _s, _e in results]
    assert len(set(basenames)) == len(basenames), (
        f"duplicate basenames: {basenames} — GLVW fork may have collapsed"
    )
    for base, src, err in results:
        assert err == 0, f"kernel {base!r}: err={err}"
        assert len(src.splitlines()) > 100, f"kernel {base!r}: assembly too short"


# ---------------------------------------------------------------------------
# F8->H arm: lines 12231-12293
# DataType: h (Mac=h), DataTypeA: F8N (GR=fp8)
# ---------------------------------------------------------------------------


def test_r5_f8gr_f8toH_emits_assembly():
    """F8->H arm (12231-12293): F8N->H mixed config emits gfx942 assembly, all err==0.

    DataType: h with DataTypeA: F8N triggers
    ``DataType{tc}.isAnyFloat8() AND MacDataType{tc}.isHalf()`` branch.
    GlobalReadVectorWidthA fork [1, 4] exercises:
      - GLVW=1 -> ``if tP["glvw"] == 1`` single-element arm (line 12232).
      - GLVW=4 -> wide conversion arm (``else`` block).
    gfx942 has NoSDWA=0 so the SDWA VCvtFP8toF32 path fires.
    """
    results = emit_kernels_from_config(_CONFIG_F8H, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG_F8H})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"kernel {base!r}: assembly suspiciously short"
        )
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"
        # F8->H kernels: DataTypeA is F8N, MAC is half (H).
        assert "F8N" in base, (
            f"kernel {base!r}: expected F8N marker in basename (DataTypeA should be F8N)"
        )


def test_r5_f8gr_f8toH_both_glvw_arms():
    """Both GLVW sub-arms of the F8->H path produce distinct, non-trivial kernels."""
    results = emit_kernels_from_config(_CONFIG_F8H, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 2, (
        f"expected >=2 kernels (one per GLVW fork), got {len(results)}"
    )
    basenames = [b for b, _s, _e in results]
    assert len(set(basenames)) == len(basenames), (
        f"duplicate basenames: {basenames} — GLVW fork may have collapsed"
    )
    for base, src, err in results:
        assert err == 0, f"kernel {base!r}: err={err}"
        assert len(src.splitlines()) > 100, f"kernel {base!r}: assembly too short"
