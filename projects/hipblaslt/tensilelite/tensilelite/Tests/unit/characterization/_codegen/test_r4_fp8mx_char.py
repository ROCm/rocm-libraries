################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P4 — gfx950 MXFP8 scale/swizzle characterization (no GPU required).

Drives the designed MX FP8 scale-swizzle config
(``data/_designed/gfx950/mx_fp8_scale_swizzle.yaml``) through the
config-driven emit harness and asserts every emitted kernel is real gfx950
AMDGCN assembly with err==0.

Target: KernelWriterAssembly.py — gfx950-specific MX scale / swizzle arms
  ~4207-4298  computeLoadSrd isMxSwizzledScaleLayout (HostPreSwizzle) path:
              MX scale tensor SRD-limit with swizzleSize0=32 / swizzleSize1=256,
              fixedSrd2 tile-boundary limit math.
  ~4527-4533  MX scale stride/padding, isMxSwizzledScale branch (bytesPerKElement).
  ~8725-8830  mfmaIter / shiftK: separate gfx950 MX tail-loop VCndMaskB32
              zeroing for MXBlockA and MXBlockB (``if ... and isgfx950``).
  ~12659-12740 MX scale LDS stride calculation for HostPreSwizzle in local-read emit.

The config uses:
  DataType: F8, MXBlockA: 32, MXBlockB: 32 (MXFP8 block-scaling A and B)
  UseSubtileImpl: 1, StreamK: 3  (required combination for gfx950 subtile MX)
  MXScaleFormat resolves automatically to "HostPreSwizzle" on gfx950.

CPU-only. No GPU, no compile, no hardware access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_LIMIT = 8

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "mx_fp8_scale_swizzle.yaml",
)


def test_r4_fp8mx_gfx950_emits_assembly():
    """MXFP8 MXBlockA/B config emits real gfx950 assembly, all err==0.

    Exercises the HostPreSwizzle MX scale SRD-limit (computeLoadSrd), the
    isgfx950 MX shiftK tail-loop zeroing (mfmaIter), and the MX scale LDS
    stride / swizzled padding arms in KernelWriterAssembly.
    """
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"kernel {base!r}: assembly unexpectedly short"
        )
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"
        # Confirm MX scale block-scaling kernels carry the MXAE8B32 marker in name.
        assert "MX" in base, (
            f"kernel {base!r}: expected MX marker in basename (MXBlockA/B should appear)"
        )


def test_r4_fp8mx_gfx950_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest for all MXFP8 scale kernels."""
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
