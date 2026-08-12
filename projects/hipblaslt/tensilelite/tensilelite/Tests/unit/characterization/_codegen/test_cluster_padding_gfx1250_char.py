################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 padded-WG edge-size cluster codegen characterization.

A non-StreamK ClusterDim=[2,2] GEMM whose tile count (23x23 with MT32x32) is not
a multiple of ClusterDim. The launch grid is padded up to a multiple, so the
kernel must:
  1. early-exit the padded work-groups before any load/barrier
     (label 'ClusterPad_EarlyStop'), and
  2. reduce the TDM multicast mask to the WGs actually present in the boundary
     cluster ('reduce multicast mask to real WGs in cluster') so the broadcast
     does not wait for the multicast timeout on the missing padded WGs.
"""

import os

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "cluster_padding.yaml",
)


def test_cluster_padding_gfx1250_emits_early_exit_and_reduced_mask():
    """Non-SK ClusterDim=[2,2] padding size: padded early-exit + reduced mask."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert_assembles(src, base)
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        # Cluster WG-id decode arm must be present.
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: missing cluster WG-id decode ('RemapWorkGroupDone')"
        )
        # Padded work-groups must early-exit before any load/barrier.
        assert "ClusterPad_EarlyStop" in src, (
            f"Kernel {base!r}: missing padded-WG early-exit ('ClusterPad_EarlyStop')"
        )
        assert "s_endpgm" in src, f"Kernel {base!r}: early-exit missing s_endpgm"
        # The multicast mask must be reduced to the real WGs of the cluster.
        assert "reduce multicast mask to real WGs in cluster" in src, (
            f"Kernel {base!r}: multicast mask not reduced for boundary cluster"
        )
