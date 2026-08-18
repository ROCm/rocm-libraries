################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 -- gfx942 AsmAddressCalculation advanced-addressing arms characterization.

Two configs exercise the 64-bit / edge / multi-batch addressing arms of
Tensile/AsmAddressCalculation.py that are NOT covered by the existing
test_r3_addrstore_char.py:

Config A: fp32 NT batched, MIWaveTile=[2,8] MIWaveGroup=[2,2], SourceSwap=1.
  Target ranges:
    - 101-102  : addScaled with scale != 1 (batch-boundary rowInc=113 != 1)
    - 135-137  : emitAddressCoordIncrement coordOffset0 > 64
    - 149-151  : emitAddressCoordIncrement rowInc > 64

Config B: bf16 BBS, StoreRemapVectorWidth=4 (SRVW), Bias enabled,
          VectorWidthA=VectorWidthB=1, SourceSwap=0.
  Target ranges:
    - 135-137  : emitAddressCoordIncrement coordOffset0 > 64
    - 168, 178 : getRowPtr/getAddrVgpr Bias branches
    - 289-309  : emitScaleToBpe optSingleColVgpr Bias arm
    - 439-451  : emitScaleToBpe else-branch Bias arm
    - 841      : incrementToNextRow numRows > 1 (SRVW enables optSrdIncForRow)

Golden: order-invariant {basename, err} digest snapshot (P4 R4).
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
)

_CFG_FP32 = os.path.join(_DIR, "asmaddr2_fp32.yaml")
_CFG_BF16_SRVW = os.path.join(_DIR, "asmaddr2_bf16_srvw.yaml")


def _run_config(cfg_path, limit=4):
    """Emit kernels from a config and return (basename, src, err) triples."""
    return emit_kernels_from_config(cfg_path, limit=limit, arch=_ARCH)


def test_r4_asmaddr2_fp32_emits_assembly():
    """fp32 large-tile config emits real gfx942 assembly with err==0."""
    results = _run_config(_CFG_FP32, limit=4)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"


def test_r4_asmaddr2_bf16_srvw_emits_assembly():
    """bf16 SRVW+Bias config emits real gfx942 assembly with err==0."""
    results = _run_config(_CFG_BF16_SRVW, limit=4)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"


def test_r4_asmaddr2_fp32_golden(snapshot):
    """P4 golden: order-invariant {basename, err} digest of the fp32 emit."""
    results = _run_config(_CFG_FP32, limit=4)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot


def test_r4_asmaddr2_bf16_srvw_golden(snapshot):
    """P4 golden: order-invariant {basename, err} digest of the bf16 SRVW emit."""
    results = _run_config(_CFG_BF16_SRVW, limit=4)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
