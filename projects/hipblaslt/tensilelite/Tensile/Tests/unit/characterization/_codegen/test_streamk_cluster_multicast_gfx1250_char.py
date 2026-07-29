# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StreamK DP cooperative B-multicast -- gfx1250 characterization (CPU-only).

Exercises the StreamK DP cooperative-load fast path added to
``Tensile/Components/StreamK.py`` + ``Tensile/Components/ClusterLoad.py``
(the derived StreamKMulticast on-state: ClusterDim=[C,1]
on SK3, which auto-derives StreamKMulticast=1). It drives the same config ->
Solutions -> emit path as the sibling
``test_streamk_cluster_coop_load_gfx1250_char.py``, but through the (explicitly
named) multicast designed config ``_designed/gfx1250/streamk_cluster_multicast.yaml``.

Asserts:
  * every kernel emits real gfx1250 assembly with ``err == 0``;
  * the B TDM descriptor carries the split B-broadcast mask
    (``MulticastMaskB`` OR'd into ``tdmBGroup1``) while the A descriptor carries
    the self-only ``MulticastMaskA`` -- i.e. B is multicast, A is per-WG;
  * the runtime ``clusterMulticastValid`` predicate gates the broadcast (M
    alignment + fully-populated cluster, else self-only B); and
  * the DP->SK boundary clear drops the B broadcast for the SK region; and
  * an order-invariant ``{basename, err}`` syrupy snapshot is pinned.

CPU-only: no GPU required. The emit harness instantiates rocisa and runs
Python+rocisa codegen without compiling or launching any GPU kernels.
"""

import os

import pytest

from config_harness import assert_cluster_barrier_balanced, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_cluster_multicast.yaml",
)


def test_streamk_cluster_multicast_gfx1250_emits_assembly():
    """gfx1250 SK3 cluster config (StreamKMulticast auto-derived on) emits real
    assembly, err==0, with the split B-broadcast mask, runtime predicate, and
    DP->SK boundary clear."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"
        # Split A/B multicast: B broadcast, A self-only.
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
            f"Kernel {base!r} missing B-broadcast mask on the B descriptor"
        )
        assert "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in src, (
            f"Kernel {base!r} missing self-only mask on the A descriptor"
        )
        # Runtime clusterMulticastValid predicate.
        assert "nWG0 aligned to C?" in src, (
            f"Kernel {base!r} missing multicast M-alignment predicate"
        )
        assert "cluster fully populated?" in src, (
            f"Kernel {base!r} missing multicast cluster-population predicate"
        )
        # DP->SK boundary clear (inline comment survives canonicalization; the
        # addComment0 banner does not).
        assert "DP->SK: drop B broadcast -> self-only" in src, (
            f"Kernel {base!r} missing DP->SK boundary mask clear"
        )
        # The multicast tensor_load_to_lds is wrapped by the cluster-scope barrier
        # handshake (s_barrier_signal/wait -3) that keeps the C cluster peers in
        # lockstep on the multicast loads.
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r} missing cluster-scope barrier signal (-3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r} missing cluster-scope barrier wait (-3)"
        )
        # Cluster-scope split-barrier balance (shared check).
        assert_cluster_barrier_balanced(src, base)


def test_streamk_cluster_multicast_gfx1250_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the multicast emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
