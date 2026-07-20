################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 StreamK=3 + ClusterDim cooperative-load codegen characterization.

A StreamK=3 kernel with ClusterDim != [1, 1] now
AUTO-ENABLES the DP cooperative B-multicast fast path (the "bare StreamK cluster"
state was collapsed; StreamKMulticast is a derived-only internal state with no
YAML opt-in). This test therefore verifies BOTH halves of that path from a config
that only sets ClusterDim:

  * the cluster WG-id decode (RemapWorkGroupDone) and the skipped ttmp9/ttmp7
    preLoop reread ("workaround" absent) under clustering -- defineAndResources
    leaves the cluster-decoded rank in WorkGroup0/1/2, so the reread guard skips
    it; and
  * the auto-derived multicast masks: the split B-broadcast / A-self masks, the
    runtime clusterMulticastValid predicate, and the DP->SK boundary clear.

Uses MX-FP4 with ClusterDim=[2, 1] (C=2), complementing the MX-FP8 C=4 coverage
in the sibling test_streamk_cluster_multicast_gfx1250_char.py.
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
    "streamk_cluster_coop_load.yaml",
)


def test_streamk_cluster_coop_load_gfx1250_emits_assembly():
    """StreamK=3 + ClusterDim=[2,1]: cluster decode present, preLoop reread
    skipped, and the auto-derived DP cooperative B-multicast path emitted."""
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
        # Cluster WG-id decode arm.
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: missing cluster WG-id decode ('RemapWorkGroupDone')"
        )
        # preLoop ttmp reread must be skipped under clustering ("workaround" is
        # unique to that reread block).
        assert "workaround" not in src, (
            f"Kernel {base!r}: ttmp reread emitted under ClusterDim != [1, 1]"
        )
        # Auto-derived DP cooperative B-multicast: B descriptor carries the
        # broadcast mask, A the self-only mask (the split topology).
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
            f"Kernel {base!r}: missing auto-derived B-broadcast mask (StreamKMulticast)"
        )
        assert "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in src, (
            f"Kernel {base!r}: missing self-only mask on the A descriptor"
        )
        # Runtime clusterMulticastValid predicate gates the broadcast.
        assert "nWG0 aligned to C?" in src, (
            f"Kernel {base!r}: missing multicast M-alignment predicate"
        )
        assert "cluster fully populated?" in src, (
            f"Kernel {base!r}: missing multicast cluster-population predicate"
        )
        # DP->SK boundary clear drops the B broadcast for the SK region.
        assert "DP->SK: drop B broadcast -> self-only" in src, (
            f"Kernel {base!r}: missing DP->SK boundary mask clear"
        )
        # The multicast loads carry the cluster-scope barrier handshake
        # (s_barrier_signal/wait -3) that keeps the C cluster peers in lockstep.
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier signal (-3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r}: missing cluster-scope barrier wait (-3)"
        )
        # The cluster-scope split barrier must be balanced: every arrive
        # (s_barrier_signal -3) is matched by a completion (s_barrier_wait -3).
        # The prologue wave-0 arrive pairs the pass's first-load wait; an
        # imbalance (unpaired wait) deadlocks all cluster waves on HW.
        n_signal = src.count("s_barrier_signal -3")
        n_wait = src.count("s_barrier_wait -3")
        assert n_signal == n_wait, (
            f"Kernel {base!r}: imbalanced cluster barrier: "
            f"{n_signal} signal(-3) vs {n_wait} wait(-3)"
        )
