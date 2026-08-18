################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 — gfx1250 XCC cluster-remap workgroup-init characterization (CPU-only).

Exercises the "Init workgroup id from ttmp with cluster remap" arm in
``Tensile/KernelWriterAssembly.py`` lines 2398-2453.

Gate conditions:
  - ``self.states.archCaps["WorkGroupIdFromTTM"]`` is True on gfx12xx
    (isaVersion[0] == 12; set in rocisa hardware_caps.hpp line ~517)
  - ``kernel["ClusterDim"][0] * kernel["ClusterDim"][1] != 1``

Only gfx12xx architectures have WorkGroupIdFromTTM=True.  gfx1250
(ISA [12,5,0]) is used because it also has the validated TDMInst=3 path and
existing gfx1250 YAML templates to build on.

Design (xccremap.yaml):
  - gfx1250 BF16 TN, TDMInst=3 (enables TDMA+TDMB)
  - ClusterDim=[2,2]  → product=4, so enableCluster=True → cluster remap arm
  - Two MatrixInstruction shapes:
      [16,16,32,1,1,1,1,1,1]  MIWaveGroup=[1,1] prod==1 → MulticastMaskA/B path
      [16,16,32,1,1,1,1,2,2]  MIWaveGroup=[2,2] prod==4 → MulticastMask (combined)
  → 2 emitted kernels (within the <=8 budget)

CPU-only: no GPU required.  The emit harness runs Python+rocisa codegen
without compiling or launching any GPU kernels.
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
    "xccremap.yaml",
)


def test_r4_xccremap_gfx1250_emits_assembly():
    """XCC cluster-remap config emits real gfx1250 assembly with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"
        # The cluster-remap arm emits a RemapWorkGroupDone label when
        # enableCluster=True (ClusterDim product != 1).
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: expected 'RemapWorkGroupDone' label from the "
            "cluster-remap WG-init arm (KernelWriterAssembly.py:2402), "
            "but it is absent -- ClusterDim may not be activating the remap path"
        )
