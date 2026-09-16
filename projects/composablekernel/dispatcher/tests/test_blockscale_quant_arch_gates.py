#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for the gfx12 arch gates in the five non-grouped block-scale
quant bridges (aquant, abquant, bquant, rowcolquant, tensor_quant).

These bridges contain two DIFFERENT kinds of gfx12 predicate, and the whole point
of this suite is to pin the difference so a future "cleanup" cannot collapse one
into the other:

  * 8-bit ``warp_tile_k`` selectors -- must match gfx1250 EXACTLY.  gfx1250 is
    the only gfx12 part with a 16x16x128 8-bit WMMA fragment; gfx1200/gfx1201
    expose 16x16x16.  A K=128 tile on gfx1200/gfx1201 still compiles and then
    silently returns garbage, the same failure mode as K=128 on gfx942, so the
    negative cases below are correctness assertions, not style checks.

  * OCP fp8 encoding predicates -- must stay FAMILY-WIDE.  Every gfx12xx part
    uses OCP e4m3/e5m2, so narrowing these to gfx1250 would break fp8 on
    gfx1200/gfx1201.

Real devices report feature suffixes (``gfx1250:xnack-``), so every exact match
is also tested in suffixed form.  No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_blockscale_quant_arch_gates.py -v
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import gemm_abquant_utils
import gemm_aquant_utils
import gemm_bquant_utils
import gemm_rowcolquant_utils
import gemm_tensor_quant_utils


# gfx1250 with and without the feature suffix a real agent reports.
GFX1250_FORMS = ["gfx1250", "gfx1250:xnack-"]

# Other gfx12 parts: OCP fp8, but only a 16x16x16 8-bit WMMA fragment.
GFX12_NON_1250 = ["gfx1200", "gfx1201"]

# Legacy MFMA archs whose behaviour must be byte-identical to before this change.
LEGACY_ARCHS = ["gfx90a", "gfx942"]


# =============================================================================
# warp_tile_k selectors -- EXACT gfx1250, not the gfx12 family
#
# Each entry is (id, callable taking gfx_arch -> int).
# =============================================================================

WARP_TILE_K_SELECTORS = [
    ("aquant_decode",
     lambda arch: gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=False)),
    ("aquant_preshufflequant",
     lambda arch: gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=True)),
    ("abquant_fp8",
     lambda arch: gemm_abquant_utils._warp_tile_k_for("fp8", arch)),
    ("abquant_bf8_flatmm",
     lambda arch: gemm_abquant_utils._warp_tile_k_for("bf8", arch, is_flat_mm=True)),
    ("bquant_decode",
     lambda arch: gemm_bquant_utils._warp_tile_k_for(arch, is_flatmm=False)),
    ("bquant_preshuffleb",
     lambda arch: gemm_bquant_utils._warp_tile_k_for(arch, is_flatmm=True)),
    ("rowcolquant_fp8",
     lambda arch: gemm_rowcolquant_utils._warp_tile_k_for("fp8", arch)),
    ("rowcolquant_bf8",
     lambda arch: gemm_rowcolquant_utils._warp_tile_k_for("bf8", arch)),
    ("tensor_quant_fp8",
     gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch),
]

WARP_TILE_K_IDS = [name for name, _ in WARP_TILE_K_SELECTORS]
WARP_TILE_K_FNS = [fn for _, fn in WARP_TILE_K_SELECTORS]


@pytest.mark.parametrize("selector", WARP_TILE_K_FNS, ids=WARP_TILE_K_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_gfx1250_selects_128(selector, arch):
    """gfx1250 has the 16x16x128 8-bit WMMA fragment -> K warp tile 128."""
    assert selector(arch) == 128


@pytest.mark.parametrize("selector", WARP_TILE_K_FNS, ids=WARP_TILE_K_IDS)
@pytest.mark.parametrize("arch", GFX12_NON_1250)
def test_other_gfx12_parts_never_select_128(selector, arch):
    """gfx1200/gfx1201 must NOT inherit gfx1250's K=128 tile.

    Their 8-bit WMMA fragment is 16x16x16; a K=128 kernel compiles for them and
    then silently returns wrong results, so a family-wide "gfx12" test here is a
    silent-correctness bug rather than a cosmetic one.
    """
    assert selector(arch) != 128


@pytest.mark.parametrize("selector", WARP_TILE_K_FNS, ids=WARP_TILE_K_IDS)
def test_feature_suffix_does_not_change_result(selector):
    """``gfx1250:xnack-`` must normalize to the same answer as ``gfx1250``."""
    assert selector("gfx1250:xnack-") == selector("gfx1250")


# -----------------------------------------------------------------------------
# Legacy arch outputs: unchanged by gfx1250 enablement.
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("arch,expected", [
    ("gfx90a", 32),
    ("gfx942", 32),
    ("gfx950", 128),
])
def test_aquant_decode_legacy_unchanged(arch, expected):
    assert gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=False) == expected


@pytest.mark.parametrize("arch,expected", [
    ("gfx90a", 64),
    ("gfx942", 64),
    ("gfx950", 128),   # gfx950 ignores IsFlatMM
])
def test_aquant_preshufflequant_legacy_unchanged(arch, expected):
    assert gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=True) == expected


@pytest.mark.parametrize("variant,arch,is_flat_mm,expected", [
    ("fp8", "gfx942", False, 32),
    ("fp8", "gfx942", True, 64),
    ("fp8", "gfx950", False, 128),
    ("fp8", "gfx950", True, 128),    # IsFlatMM ignored on gfx950
    ("fp4", "gfx950", True, 32),     # non-8bit-float stays 32 on gfx950
    ("fp4", "gfx1250", True, 32),    # ... and likewise on gfx1250
])
def test_abquant_legacy_and_fp4_unchanged(variant, arch, is_flat_mm, expected):
    assert gemm_abquant_utils._warp_tile_k_for(variant, arch, is_flat_mm=is_flat_mm) == expected


@pytest.mark.parametrize("arch,is_flatmm,expected", [
    ("gfx90a", False, 32),
    ("gfx942", False, 32),
    ("gfx942", True, 64),
    ("gfx950", False, 128),
    ("gfx950", True, 128),
])
def test_bquant_legacy_unchanged(arch, is_flatmm, expected):
    assert gemm_bquant_utils._warp_tile_k_for(arch, is_flatmm=is_flatmm) == expected


@pytest.mark.parametrize("variant,arch,expected", [
    ("fp8", "gfx942", 32),
    ("bf8", "gfx942", 32),
    ("fp8", "gfx950", 128),
    ("bf8", "gfx950", 128),
])
def test_rowcolquant_legacy_unchanged(variant, arch, expected):
    assert gemm_rowcolquant_utils._warp_tile_k_for(variant, arch) == expected


@pytest.mark.parametrize("arch,expected", [
    ("gfx942", 32),
    ("gfx950", 128),
])
def test_tensor_quant_legacy_unchanged(arch, expected):
    assert gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch(arch) == expected


# =============================================================================
# OCP fp8 predicates -- FAMILY-WIDE gfx12 is correct here
# =============================================================================

OCP_PREDICATES = [
    ("bquant", gemm_bquant_utils._uses_ocp_fp8),
    ("rowcolquant", gemm_rowcolquant_utils._uses_ocp_fp8),
]
OCP_IDS = [name for name, _ in OCP_PREDICATES]
OCP_FNS = [fn for _, fn in OCP_PREDICATES]


@pytest.mark.parametrize("predicate", OCP_FNS, ids=OCP_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS + GFX12_NON_1250 + ["gfx950"])
def test_ocp_fp8_is_family_wide(predicate, arch):
    """Every gfx12xx part uses OCP fp8/bf8, so this predicate stays family-wide.

    Narrowing it to an exact gfx1250 match would silently switch gfx1200/gfx1201
    host-side encoding to FNUZ and produce mismatched/NaN comparisons.
    """
    assert predicate(arch) is True


@pytest.mark.parametrize("predicate", OCP_FNS, ids=OCP_IDS)
@pytest.mark.parametrize("arch", LEGACY_ARCHS)
def test_ocp_fp8_false_on_legacy_fnuz_archs(predicate, arch):
    assert predicate(arch) is False


@pytest.mark.parametrize("predicate", OCP_FNS, ids=OCP_IDS)
def test_ocp_fp8_defaults_to_ocp_when_arch_unknown(predicate):
    assert predicate(None) is True


# =============================================================================
# Supported-arch lists accept gfx1250, including with a feature suffix
# =============================================================================

VALIDATORS = [
    ("aquant", gemm_aquant_utils._validate_arch),
    ("abquant", gemm_abquant_utils._validate_arch),
    ("tensor_quant", gemm_tensor_quant_utils._validate_arch),
]
VALIDATOR_IDS = [name for name, _ in VALIDATORS]
VALIDATOR_FNS = [fn for _, fn in VALIDATORS]


@pytest.mark.parametrize("validate", VALIDATOR_FNS, ids=VALIDATOR_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS + ["gfx942", "gfx950"])
def test_validate_arch_accepts_supported(validate, arch):
    assert validate(arch) == arch


@pytest.mark.parametrize("validate", VALIDATOR_FNS, ids=VALIDATOR_IDS)
@pytest.mark.parametrize("arch", ["gfx1030", "gfx803", "sm_90"])
def test_validate_arch_rejects_unsupported(validate, arch):
    with pytest.raises(ValueError):
        validate(arch)


@pytest.mark.parametrize("module", [
    gemm_aquant_utils,
    gemm_abquant_utils,
    gemm_tensor_quant_utils,
], ids=["aquant", "abquant", "tensor_quant"])
def test_gfx1250_listed_in_supported_archs(module):
    assert "gfx1250" in module._SUPPORTED_ARCHS


# =============================================================================
# The private normalizer itself
# =============================================================================

NORMALIZERS = [
    ("aquant", gemm_aquant_utils._is_gfx1250),
    ("abquant", gemm_abquant_utils._is_gfx1250),
    ("bquant", gemm_bquant_utils._is_gfx1250),
    ("rowcolquant", gemm_rowcolquant_utils._is_gfx1250),
    ("tensor_quant", gemm_tensor_quant_utils._is_gfx1250),
]
NORMALIZER_IDS = [name for name, _ in NORMALIZERS]
NORMALIZER_FNS = [fn for _, fn in NORMALIZERS]


@pytest.mark.parametrize("is_gfx1250", NORMALIZER_FNS, ids=NORMALIZER_IDS)
@pytest.mark.parametrize("arch,expected", [
    ("gfx1250", True),
    ("gfx1250:xnack-", True),
    ("gfx1250:xnack+", True),
    ("gfx1200", False),
    ("gfx1201", False),
    ("gfx12", False),
    ("gfx125", False),
    ("gfx12500", False),   # substring matching would wrongly accept this
    ("gfx942", False),
    ("gfx950", False),
    ("", False),
    (None, False),
])
def test_is_gfx1250_exact_match(is_gfx1250, arch, expected):
    assert is_gfx1250(arch) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
