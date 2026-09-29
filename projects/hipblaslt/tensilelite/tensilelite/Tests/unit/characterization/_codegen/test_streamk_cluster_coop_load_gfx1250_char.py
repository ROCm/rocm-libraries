# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 StreamK=3 + ClusterDim cooperative-load codegen characterization.

``StreamKMulticast`` has no YAML opt-in: a StreamK=3 kernel given a ClusterDim
other than ``[1, 1]`` derives it on. This test drives a config that sets only
ClusterDim and verifies both halves of what that turns on:

  * the cluster WG-id decode (RemapWorkGroupDone) and the skipped ttmp9/ttmp7
    preLoop reread ("workaround" absent) under clustering -- defineAndResources
    leaves the cluster-decoded rank in WorkGroup0/1/2, so the reread guard skips
    it; and
  * the auto-derived split multicast masks, one per operand descriptor.

Uses MX-FP4 with ClusterDim=[2, 1] and StreamKForceDPOnly=1, the smallest cluster
that still has B-sharing peers.
"""

import os

import pytest

from config_harness import (
    assert_assembles,
    assert_cluster_barrier_balanced,
    assert_real_gfx1250_kernels,
    assert_split_multicast_masks,
    emit_kernels_from_config,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_cluster_coop_load.yaml",
)


def test_streamk_cluster_coop_load_gfx1250_emits_assembly():
    """StreamK=3 + ClusterDim=[2,1]: cluster decode present, preLoop reread
    skipped, and the auto-derived cooperative multicast path emitted."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        # Cluster WG-id decode arm.
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: missing cluster WG-id decode ('RemapWorkGroupDone')"
        )
        # preLoop ttmp reread must be skipped under clustering ("workaround" is
        # unique to that reread block).
        assert "workaround" not in src, (
            f"Kernel {base!r}: ttmp reread emitted under ClusterDim != [1, 1]"
        )
        # Auto-derived cooperative multicast: one mask per operand descriptor.
        assert_split_multicast_masks(src, base)
        # The multicast loads carry the cluster-scope barrier handshake
        # (s_barrier_signal/wait -3) that keeps the C cluster peers in lockstep.
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier signal (-3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier wait (-3)"
        )
        # Cluster-scope split-barrier balance (shared check).
        assert_cluster_barrier_balanced(src, base)
