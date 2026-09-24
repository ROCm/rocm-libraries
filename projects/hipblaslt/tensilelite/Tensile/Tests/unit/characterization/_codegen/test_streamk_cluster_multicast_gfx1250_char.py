# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StreamK cluster multicast -- gfx1250 characterization (CPU-only).

Exercises the DataParallel cluster cooperative-load path in
``Tensile/Components/WorkAssignment.py`` + ``Tensile/Components/ClusterLoad.py``.
It uses StaticGrid assignment and a ``ClusterDim`` other than ``[1, 1]``.

Each arm is a (PrefetchGlobalRead, ClusterDim) pair and is pinned separately.

The cluster shapes:

  * ``[4, 1]`` -- Cs = 4 X-peers on M-adjacent tiles reuse B; Ck = 1, so A has no
    peers and its mask collapses to the self bit; and
  * ``[2, 2]`` -- Cs = 2 X-peers reuse B and Ck = 2 Y-peers on N-adjacent tiles
    reuse A, so BOTH operands are multicast.

Both shapes take the same code path: each hardware cluster is folded into a
persistent cluster rank that walks whole Cs x Ck tile blocks, a peer past the
tile edge aliases the edge tile and skips its store, and the broadcast masks are
bound onto the TDM descriptors for every tile. Each tile's first-load cluster wait
pairs the prologue arrive (first tile) or the loop-close arrive (later tiles).
Ck is a spatial N-tiling axis, never a K-split, so a K-slice decode must be absent.

The PrefetchGlobalRead variants:

  * ``PrefetchGlobalRead=1`` -- the single-buffered prologue.
  * ``PrefetchGlobalRead=2`` with K > DepthU -- the prologue emits a second,
    double-buffered ("LDS1") cooperative multicast prefetch load. The loop-close
    arrive follows every LDS read of the previous tile, so the first-load wait
    covers both prefetch buffers and the LDS1 load needs no handshake of its own.

CPU-only: no GPU required. The emit harness instantiates rocisa and runs
Python+rocisa codegen without compiling or launching any GPU kernels.
"""

import os

import pytest

from config_harness import (
    assert_assembles,
    assert_cluster_barrier_balanced,
    assert_real_gfx1250_kernels,
    assert_split_multicast_masks,
    emit_kernels_from_config,
    golden_digest,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data",
                         "_designed", "gfx1250")

# PrefetchGlobalRead variant -> the designed config that pins it. The PGR=2
# config also raises K above DepthU so the double-buffered prologue prefetch
# actually materializes.
_CONFIGS = {
    1: os.path.join(_DESIGNED, "streamk_cluster_multicast.yaml"),
    2: os.path.join(_DESIGNED, "streamk_cluster_multicast_pgr2.yaml"),
}

# Cluster shapes the configs sweep, with the A-side mask constant each implies:
# maskA has one bit per Ck row (1 | 1<<Cs | ...), so Ck == 1 collapses it to 0x1.
_MASK_A = {(4, 1): "0x1", (2, 2): "0x5"}

_ARMS = [(pgr, shape) for pgr in sorted(_CONFIGS) for shape in _MASK_A]
_ARM_IDS = ["pgr%d_cs%d_ck%d" % (pgr, shape[0], shape[1]) for pgr, shape in _ARMS]


def _emit(pgr, cluster_dim):
    # The limit truncates the fork permutations BEFORE the ClusterDim filter, so
    # it has to cover the whole sweep: 2 cluster shapes x 2 ScheduleIterAlg arms
    # x 4 MatrixInstruction = 16. At 8 the [2, 2] shape falls outside the window
    # and the filter finds nothing.
    return emit_kernels_from_config(_CONFIGS[pgr], limit=16, arch=_ARCH,
                                    cluster_dim=cluster_dim)


def _next_tile_arrive_follows_lds_drain(src):
    """Return True iff the loop-close arrive runs after the tile's LDS work.

    Between the ``PersistentLoopClose`` label and the back edge, the arrive skips
    the last tile, drains LDS, joins the workgroup, and then wave 0 signals ``-3``.
    """
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if not ln.startswith("label_PersistentLoopClose:"):
            continue
        window = lines[i : i + 32]
        text = "\n".join(window)
        order = ["label_PersistentMC_SkipNextTileArrive", "s_wait_dscnt 0",
                 "s_barrier_wait -1", "s_barrier_signal -3",
                 "label_PersistentMC_SkipNextTileArrive:"]
        pos = [text.find(token) for token in order]
        return all(p >= 0 for p in pos) and pos == sorted(pos)
    return False


@pytest.mark.parametrize("pgr, cluster_dim", _ARMS, ids=_ARM_IDS)
def test_streamk_cluster_multicast_gfx1250_emits_assembly(pgr, cluster_dim):
    """Each (PGR, cluster shape) arm emits real assembly (err==0) with the
    cluster-block fold and walk, phantom-tile handling, the per-tile cluster
    arrive, and both multicast masks bound to their descriptors."""
    results = _emit(pgr, cluster_dim)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "DP fold: rank = cluster*Cs*Ck + peerY*Cs + peerX" in src, (
            f"Kernel {base!r} missing the cluster-rank fold"
        )
        assert "totalTiles = blocks * Cs*Ck" in src, (
            f"Kernel {base!r} missing the whole-block tile bound"
        )
        assert "N tile = blockN*Ck + peerY" in src, (
            f"Kernel {base!r} missing the cluster-block tile decode"
        )
        # Every peer stays in the cluster for every block, so there is no pad exit
        # and a peer past the tile edge only skips its store.
        assert "padded work-group: exit before any cluster barrier/load" not in src, (
            f"Kernel {base!r} still emits the padded boundary-peer early exit"
        )
        assert "phantom tiles skip the store" in src, (
            f"Kernel {base!r} missing the phantom-tile store skip"
        )
        assert _next_tile_arrive_follows_lds_drain(src), (
            f"Kernel {base!r} loop-close arrive does not follow the tile's LDS drain"
        )
        # Multicast loads keep their prefetch pipelining: no per-iteration drain.
        assert "retire cooperative tensor_load_to_lds" not in src, (
            f"Kernel {base!r} drains every cooperative load in the main loop"
        )
        # B broadcasts along Cs and is bound for every tile. A is bound too unless
        # Ck == 1, where its self-only mask is freed after the prologue.
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
            f"Kernel {base!r} missing B-broadcast mask on the B descriptor"
        )
        if cluster_dim[1] > 1:
            assert_split_multicast_masks(src, base)
        mask_a = _MASK_A[cluster_dim]
        assert f"s[sgprMulticastMaskA], {mask_a}" in src, (
            f"Kernel {base!r} A mask is not {mask_a} for ClusterDim={list(cluster_dim)}"
        )
        # The multicast tensor_load_to_lds is wrapped by the cluster-scope barrier
        # handshake that keeps the peers in lockstep on the multicast loads.
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r} missing cluster-scope barrier signal (-3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r} missing cluster-scope barrier wait (-3)"
        )
        assert_cluster_barrier_balanced(src, base)
        # Edge peers are handled structurally (phantom tiles), so the runtime
        # "is this cluster usable" selection guard must not be emitted.
        assert "nWG0 aligned to C?" not in src, (
            f"Kernel {base!r} emitted the runtime multicast selection guard"
        )
        # Ck is a spatial N-tiling axis: no K-split decode or maskB shift.
        assert "k = PersistentWorkGroupIndex & (Ck-1)" not in src, (
            f"Kernel {base!r} wrongly emitted a K-slice reduction decode"
        )
        if pgr >= 2:
            # The PGR>=2 prologue double-buffer prefetch region exists for K>DepthU.
            assert "skipPGR2" in src, (
                f"Kernel {base!r} missing the PGR2 prologue double-buffer region"
            )
            # The first-load wait covers both prefetch buffers, so the LDS1 load
            # issues right behind the LDS0 load.
            assert "PersistentMC_SkipPrefetchSignal" not in src, (
                f"Kernel {base!r} serializes the PGR2 prefetch behind a cluster handshake"
            )


@pytest.mark.parametrize("pgr, cluster_dim", _ARMS, ids=_ARM_IDS)
def test_streamk_cluster_multicast_gfx1250_golden(snapshot, pgr, cluster_dim):
    """Golden: order-invariant {basename, err} digest, one per (PGR, shape) arm."""
    assert golden_digest(_emit(pgr, cluster_dim)) == snapshot
