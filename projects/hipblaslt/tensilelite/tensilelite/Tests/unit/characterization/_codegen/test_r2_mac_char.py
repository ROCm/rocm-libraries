################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — non-MFMA fp16 HPA (DOT2) MAC characterization.

Drives the designed non-MFMA HHS config
(``data/test_data/_designed/gfx90a/mac.yaml``) through the config-driven emit
harness on gfx942 to exercise the ``FMA_F16_HPA_DOT2.__call__`` path in
``tensilelite/Components/MAC_F16_HPA.py`` (lines 44-88; 78 lines miss in baseline).

Architecture selection:
  * The target component ``FMA_F16_HPA_DOT2`` requires ``UseDotInstruction=True``
    which is set in ``Solution.py`` only for gfx942 (ISA 9.4.2) or gfx950
    (ISA 9.5.0) with DataType=Half and HighPrecisionAccumulate=True.
  * On gfx90a, ``UseDotInstruction`` is forced False and ``FMA_F16_HPA_MAD_MIX``
    would be selected; however that class has a signature mismatch with the call
    site in KernelWriterAssembly (latent bug) and raises TypeError.  The task
    spec allows "gfx90a or gfx942"; this test targets gfx942.
  * gfx942 has ``v_dot2_f32_f16=1``, so FMA_F16_HPA_DOT2 selects VDot2F32F16
    (line 52/77).

ForkParameters sweep:
  * ThreadTile x WorkGroup grid -> 6 fork permutations (<=8 limit).
  * LocalReadVectorWidth=2 satisfies the dot-kernel constraint
    (NumDotElements=2 * InnerUnroll=1 = 2).
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
    "gfx90a",
    "mac.yaml",
)

# The same problem and fork sweep with LocalReadVectorWidth 1, which is what
# selects the non-dot path: on gfx90a UseDotInstruction is forced False, so
# FMA_F16_HPA_MAD_MIX is the component that runs.
_CONFIG_MAD_MIX = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx90a",
    "mac_mad_mix.yaml",
)


def test_r2_mac_hhs_dot2_emits_assembly():
    """Non-MFMA HHS DOT2 config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"expected all err==0, got {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 500, (
            f"suspiciously short source for {base}"
        )
        assert ".amdgcn_target" in src, f"missing .amdgcn_target in {base}"
        assert "gfx942" in src, f"missing gfx942 target marker in {base}"
        assert base.startswith("Cijk_"), f"unexpected basename pattern: {base}"
        # Confirm DOT2 MAC instructions are present in the emitted assembly.
        assert "v_dot2_f32_f16" in src, (
            f"expected v_dot2_f32_f16 MAC instructions in {base}"
        )


def test_r2_mac_hhs_dot2_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the mac config emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot


def test_r2_mac_hhs_mad_mix_emits_assembly():
    """The non-dot fp16 HPA MAC path emits real gfx90a assembly, all err==0.

    On gfx90a ``UseDotInstruction`` is forced False, so
    ``FMA_F16_HPA_MAD_MIX`` is selected instead of ``FMA_F16_HPA_DOT2``.  No
    shipped logic tree reaches that class -- its ``__call__`` signature
    mismatched the call site, so it raised before it could emit anything -- and
    nothing exercised it.  It has to work now, and the accumulator index has to
    be a plain integer: rocisa parses a register name into a base name and a
    list of integer offsets, so an expression left in the name is read as its
    first integer and the rest is dropped.
    """
    results = emit_kernels_from_config(_CONFIG_MAD_MIX, limit=8, arch="gfx90a")
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"expected all err==0, got {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 500, (
            f"suspiciously short source for {base}"
        )
        assert ".amdgcn_target" in src, f"missing .amdgcn_target in {base}"
        assert "gfx90a" in src, f"missing gfx90a target marker in {base}"
        # The mix MAC instruction is what distinguishes this path from the dot2 one.
        assert ("v_fma_mix_f32" in src) or ("v_mad_mix_f32" in src), (
            f"expected a mix MAC instruction in {base}"
        )
