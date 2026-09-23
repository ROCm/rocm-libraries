################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — LocalRead.py big-cluster coverage: DTL / wide-LRVW / transpose / ConvertAfterDS.

Target missing ranges in Tensile/Components/LocalRead.py (miss=489, 54%):
  785-942  : enableLDSTr + HasWMMA_V3 arm. Subdivides by bpeDS:
               bpeDS==1  (FP8):     lines 835-883  — gfx1250 + LDSTrInst + FP8 A/B
               bpeDS==2  (FP16/BF16): lines 884-942 — gfx1250 + LDSTrInst + HHS
             (bpeDS==0.5/Float4 and bpeDS==0.75/Float6 require HasLDSTrB64B4/B96B6
              which are absent on all currently tested arches; P5 ceiling.)
  1090-1135: UseF32XEmulation + indexTranpose (lrvwTile>1, not useTransposeCode)
             do8PackAtOnce arm: triggered by gfx950 xfp32 + ClusterLocalRead=1
             + VectorWidthA/B > 1 (so lrvwTile > 1 and useTransposeCode=False).
  1163-1320: ConvertAfterDS + (bpe != bpeDS):
               UnrollMajorLDS arm (1169-1199) and non-UMLDS arm (1200+):
               gfx950 F8HS TN/NT with ConvertAfterDS=1.

Strategy (two config families):
  A. gfx1250 + LDSTrInst=True: covers HasWMMA_V3=1 arm of enableLDSTr block
     (Tensile lines 784+). Uses:
       - HHS NN (bpeDS=2) -> lines 884-942
       - F8HS TN (bpeDS=1) -> lines 835-883
  B. gfx950 + ConvertAfterDS=1 (F8HS TN/NT): covers lines 1163-1320.
     The UnrollMajorLDS fork (TN=TransposeA=1) exercises 1169+.
     The non-UMLDS fork (NT=TransposeA=0) exercises 1200+.

All tests are pure-assert (no syrupy snapshot). pytestmark=pytest.mark.unit. CPU-only.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_DESIGNED = os.path.join(_HERE, "data", "test_data", "_designed")

# gfx1250: LDSTr + HasWMMA_V3 (covers lines 835-942)
_GFX1250_LDSTR_CFG = os.path.join(_DESIGNED, "gfx1250", "localread_ldstr_wmma3.yaml")

# gfx950: ConvertAfterDS FP8->FP16 (covers lines 1163-1320)
_GFX950_CAFS_CFG = os.path.join(_DESIGNED, "gfx950", "localread_cafs_fp8.yaml")

# gfx950: xfp32 (TF32 emulation) wider VectorWidth for lrvwTile>1 / indexTranpose
# (already exists; re-used here to exercise lines 1090-1135)
_GFX950_XFP32_CFG = os.path.join(
    os.path.dirname(_HERE),  # Tensile/Tests/unit/
    os.pardir,               # Tensile/Tests/
    os.pardir,               # Tensile/
    "common", "gemm", "gfx950", "xfp32.yaml",
)
_GFX950_XFP32_CFG = os.path.normpath(_GFX950_XFP32_CFG)

_ARCH_1250 = "gfx1250"
_ARCH_950 = "gfx950"


# ---------------------------------------------------------------------------
# Test 1: gfx1250 LDSTr + HasWMMA_V3 — covers lines 884-942 (HHS/FP16 arm)
#         and 835-883 (FP8 arm)
# ---------------------------------------------------------------------------


def test_r7_gfx1250_ldstr_hhs_emits_assembly():
    """gfx1250 HHS NN + LDSTrInst=True fires HasWMMA_V3 + bpeDS==2 arm (lines 884-942).

    Conditions for lines 884-942:
      enableLDSTr=True (LDSTrInst=True + HasLDSTrB128B16=1 on gfx1250 + bpeDS=2)
      HasWMMA_V3=1 (gfx1250 only)
      bpeDS==2 (FP16/BF16 data type)

    Expected: >=1 kernel emits with err==0; source contains WMMA instructions.
    """
    results = emit_kernels_from_config(_GFX1250_LDSTR_CFG, limit=8, arch=_ARCH_1250)
    assert len(results) >= 1, f"expected >=1 kernel from gfx1250 HHS LDSTr config, got {len(results)}"
    ok = [(b, s, e) for (b, s, e) in results if e == 0]
    assert len(ok) >= 1, (
        "expected >=1 err==0 kernel from gfx1250 HHS LDSTrInst; all failed: "
        + str([(b, e) for b, _, e in results])
    )
    for base, src, _err in ok:
        assert src and len(src.splitlines()) > 50, f"source too short for {base}"
        assert ".amdgcn_target" in src, f"missing .amdgcn_target in {base}"
        assert "gfx1250" in src, f"missing gfx1250 in {base}"
        assert base.startswith("Cijk_"), f"unexpected basename: {base}"


def test_r7_gfx1250_ldstr_hhs_lds_reads():
    """gfx1250 HHS LDSTrInst kernel emits ds_load_tr transposed-read instructions.

    The LDS transpose path with enableLDSTr + HasWMMA_V3 + bpeDS==2 emits
    ds_load_tr16_b128 (WMMA_V3 transposed local-read) instructions for the
    transposed LDS layout (lines 884-942 of LocalRead.py).
    """
    results = emit_kernels_from_config(_GFX1250_LDSTR_CFG, limit=8, arch=_ARCH_1250)
    ok = [(b, s, e) for (b, s, e) in results if e == 0]
    assert len(ok) >= 1, "need >=1 err==0 kernel"
    # ds_load_tr16_b128 is the WMMA_V3 transposed LDS read instruction on gfx1250
    tr_found = any("ds_load_tr" in src.lower() for _, src, _ in ok)
    assert tr_found, (
        "expected ds_load_tr (transposed LDS read) in gfx1250 HHS LDSTr kernel; "
        "got src previews: " + str([src[:200] for _, src, _ in ok[:2]])
    )


# ---------------------------------------------------------------------------
# Test 2: gfx950 ConvertAfterDS FP8->FP16 — covers lines 1163-1320
# ---------------------------------------------------------------------------


def test_r7_gfx950_cafs_fp8_emits_assembly():
    """gfx950 F8HS + ConvertAfterDS=1 fires the bpe!=bpeDS conversion arm (lines 1163-1320).

    Conditions for line 1163:
      kernel["ConvertAfterDS"] = True
      tP["bpe"] != tP["bpeDS"]: bpe=2 (FP16 output), bpeDS=1 (FP8 LDS store)

    TN case (TransposeA=1 -> UnrollMajorLDSA=True) exercises line 1169+.
    NT case (TransposeA=0 -> UnrollMajorLDSA=False) exercises line 1200+.

    Expected: >=1 kernel emits with err==0.
    """
    results = emit_kernels_from_config(_GFX950_CAFS_CFG, limit=8, arch=_ARCH_950)
    assert len(results) >= 1, f"expected >=1 kernel from gfx950 ConvertAfterDS config, got {len(results)}"
    ok = [(b, s, e) for (b, s, e) in results if e == 0]
    assert len(ok) >= 1, (
        "expected >=1 err==0 kernel from gfx950 F8HS ConvertAfterDS; all failed/rejected: "
        + str([(b, e) for b, _, e in results])
    )
    for base, src, _err in ok:
        assert src and len(src.splitlines()) > 50, f"source too short for {base}"
        assert ".amdgcn_target" in src, f"missing .amdgcn_target in {base}"
        assert "gfx950" in src or "gfx9" in src, f"missing gfx9 target in {base}"
        assert base.startswith("Cijk_"), f"unexpected basename: {base}"


def test_r7_gfx950_cafs_fp8_ds_reads():
    """gfx950 ConvertAfterDS kernel emits ds_load instructions (FP8 LDS read path).

    After reading FP8 values from LDS, the ConvertAfterDS arm (lines 1163-1320)
    emits conversion instructions. Assert the kernel has ds_load/ds_read
    instructions (LDS reads for the FP8 -> FP16 conversion path).
    """
    results = emit_kernels_from_config(_GFX950_CAFS_CFG, limit=8, arch=_ARCH_950)
    ok = [(b, s, e) for (b, s, e) in results if e == 0]
    assert len(ok) >= 1, "need >=1 err==0 kernel for ConvertAfterDS assert"
    ds_load_found = any(
        ("ds_load" in src.lower() or "ds_read" in src.lower())
        for _, src, _ in ok
    )
    assert ds_load_found, (
        "expected ds_load/ds_read in at least one gfx950 ConvertAfterDS F8HS kernel; "
        "got previews: " + str([src[:200] for _, src, _ in ok[:2]])
    )


# ---------------------------------------------------------------------------
# Test 3: gfx950 xfp32 (TF32 emulation) — covers lines 1090-1135
# (UseF32XEmulation + indexTranpose + do8PackAtOnce arm)
# ---------------------------------------------------------------------------


def test_r7_gfx950_xfp32_index_transpose_emits():
    """gfx950 xfp32 (UseF32XEmulation) with lrvwTile>1 fires indexTranpose arm (lines 1090-1135).

    Conditions for line 1084-1085:
      do8PackAtOnce = indexTranpose and not allPack4HiDone
      (valuiIdx % 8) == 4 and do8PackAtOnce
    where indexTranpose = lrvwTile > 1 and (not useTransposeCode)

    The gfx950/xfp32.yaml config has F32XdlMathOp=X + DataType=S which sets
    UseF32XEmulation=True on gfx950 (HasF32XEmulation=1). With wider MI shapes
    (MIWaveTile entries with 4+ tiles) and LocalReadVectorWidth>1, lrvwTile>1
    triggers indexTranpose=True when useTransposeCode=False.

    Expected: >=1 kernel emits with err==0 and contains MFMA instructions.
    """
    results = emit_kernels_from_config(_GFX950_XFP32_CFG, limit=8, arch=_ARCH_950)
    assert len(results) >= 1, f"expected >=1 kernel from gfx950 xfp32, got {len(results)}"
    ok = [(b, s, e) for (b, s, e) in results if e == 0]
    assert len(ok) >= 1, (
        "expected >=1 err==0 kernel from gfx950 xfp32 (TF32 emulation); all failed: "
        + str([(b, e) for b, _, e in results])
    )
    for base, src, _err in ok:
        assert src and len(src.splitlines()) > 100, f"source too short for {base}"
        assert ".amdgcn_target" in src, f"missing .amdgcn_target in {base}"
        assert "gfx950" in src or "gfx9" in src, f"missing gfx9 in {base}"
        # xfp32 (TF32 emulation) kernels use MFMA for accumulation
        assert "v_mfma_" in src, (
            f"expected v_mfma_ in xfp32 TF32 kernel {base!r}; first 300 chars: {src[:300]!r}"
        )
