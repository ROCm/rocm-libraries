################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 -- gfx942 INT8 GEMM + HighPrecisionAccumulate characterization.

Drives the designed int8_hpa.yaml config (I8/I8/s, HighPrecisionAccumulate=True,
Bias, GSU fork) through the config-driven emit harness to exercise the distinct
INT8 MAC/accumulate and conversion arms in KernelWriter.py that are absent from
all existing gfx942 seeds (BBS, F8N, GG, HSS, DB, etc. use non-I8 data types).

INT8 with HPA exercises:
  KernelWriter.py:
    - line 6409/6415 : MacDataTypeA/B.isInt8() -> numVgprBufferPackA/B branch
    - line 6745      : DataType.isInt8() in HPA validity check
    - line 7855      : DataType.isInt8() -> startVgprValuPackTemp allocation
    - line 8895      : DataType.isInt8() && HPA -> canCheckValueC
  Components/MAC_I8_HPA.py:
    - lines 50-82    : FMA_I8_HPA sign-extend (v_lshlrev_b32/v_ashrrev_i32) +
                       v_mad_i32_i24 accumulate loop

ForkParameters: MatrixInstruction x GlobalSplitU -> 4 kernels (<=8 budget).
GSU=2 also triggers the GlobalSplitU reduction helper kernel path.

Pattern: A (codegen emit). Pure-assert (no syrupy snapshot).
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
    "int8_hpa.yaml",
)


def test_r4_int8_hpa_gfx942_emits_assembly():
    """INT8 + HPA seed config emits real gfx942 assembly, all err==0.

    Exercises the I8I8 MFMA path, HPA validity guard, MacDataType pack branches,
    and FMA_I8_HPA accumulate loop in MAC_I8_HPA.py.
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


def test_r4_int8_hpa_gfx942_int8_mac_in_asm():
    """INT8 HPA kernels use integer MAC instructions (v_mad_i32_i24 or mfma_i32).

    The FMA_I8_HPA path emits v_mad_i32_i24 (non-MFMA) accumulate; MFMA shapes
    emit v_mfma_i32_* instructions. Either confirms the INT8 accumulate arm is
    active.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    int8_mac_found = any(
        ("v_mad_i32_i24" in src or "mfma_i32" in src.lower())
        for _b, src, _e in results
    )
    assert int8_mac_found, (
        "Expected >=1 kernel with INT8 MAC instruction (v_mad_i32_i24 or mfma_i32), "
        "but none found. The INT8 HPA accumulate arm may not be exercised."
    )


def test_r4_int8_hpa_gfx942_multi_kernels():
    """MatrixInstruction x GSU fork produces multiple distinct INT8 kernels.

    The ForkParameters cartesian product (2 MI shapes x 2 GSU values) should
    yield >=2 emitted kernels, confirming the fork breadth and distinct basenames.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    basenames = [b for b, _s, _e in results]
    assert len(basenames) >= 2, (
        f"Expected >=2 kernels from MI x GSU fork, got {len(basenames)}: {basenames}"
    )
    # All basenames must be unique (no duplicate emit).
    assert len(set(basenames)) == len(basenames), (
        f"Duplicate kernel basenames detected: {basenames}"
    )
