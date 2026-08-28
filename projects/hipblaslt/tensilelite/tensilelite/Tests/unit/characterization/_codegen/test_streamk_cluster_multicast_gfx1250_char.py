# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StreamK cluster multicast -- gfx1250 characterization (CPU-only).

Exercises the StreamK ForceDPOnly cluster cooperative-load path added to
``Tensile/Components/StreamK.py`` + ``Tensile/Components/ClusterLoad.py``, which
a ``ClusterDim`` other than ``[1, 1]`` on StreamK=3 turns on.

Each arm is a (PrefetchGlobalRead, ClusterDim) pair and is pinned separately.

The cluster shapes:

  * ``[4, 1]`` -- Cs = 4 X-peers on M-adjacent tiles reuse B; Ck = 1, so A has no
    peers and its mask collapses to the self bit; and
  * ``[2, 2]`` -- Cs = 2 X-peers reuse B and Ck = 2 Y-peers on N-adjacent tiles
    reuse A, so BOTH operands are multicast.

Both shapes take the same code path: one work-group per output tile, the HW
work-group coords folded into the linear tile index, padded boundary peers
exiting before the cluster barrier, and the broadcast masks bound onto the TDM
descriptors. Ck is a spatial N-tiling axis, never a K-split, so a K-slice decode
must be absent.

The PrefetchGlobalRead variants:

  * ``PrefetchGlobalRead=1`` -- the single-buffered prologue.
  * ``PrefetchGlobalRead=2`` with K > DepthU -- the prologue emits a second,
    double-buffered ("LDS1") cooperative multicast prefetch load. That load sits
    inside the single-iteration guard branch, past the generic per-load
    cluster-barrier bracketing boundary, so
    ``StreamK.streamKMulticastProloguePrefetchHandshake`` has to bracket it with
    a dedicated cluster-scope split-barrier handshake of its own -- otherwise a
    peer can issue the prefetch while another peer is still behind the guard.

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


def _skip_prefetch_handshake_brackets_load(src):
    """Return True iff the prologue prefetch (LDS1) multicast load is bracketed.

    The double-buffered prologue prefetch load sits in the ``skipPGR2`` guard
    segment. The dedicated handshake elects wave 0 (branch to
    ``SKMC_SkipPrefetchSignal``), signals ``-3``, then all waves wait ``-3``
    immediately before the LDS1 ``tensor_load_to_lds`` group.
    """
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "label_SKMC_SkipPrefetchSignal:" not in ln:
            continue
        # A cluster-scope wait must follow the skip label, before the LDS1 load.
        window = lines[i : i + 6]
        has_wait = any("s_barrier_wait -3" in w for w in window)
        has_load = any("tensor_load_to_lds" in w for w in window)
        # A wave-0 signal must precede the skip label.
        pre = lines[max(0, i - 4) : i]
        has_signal = any("s_barrier_signal -3" in p for p in pre)
        if has_wait and has_load and has_signal:
            return True
    return False


@pytest.mark.parametrize("pgr, cluster_dim", _ARMS, ids=_ARM_IDS)
def test_streamk_cluster_multicast_gfx1250_emits_assembly(pgr, cluster_dim):
    """Each (PGR, cluster shape) arm emits real assembly (err==0) with the
    tile-index fold, the padded-peer exit, and both multicast masks bound to
    their descriptors."""
    results = _emit(pgr, cluster_dim)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        # One work-group per tile: the HW coords are folded into the linear index.
        assert "DP fold: WorkGroup1 * nWG0 (N-tile row)" in src, (
            f"Kernel {base!r} missing the N-tile-row fold (WorkGroup1*nWG0)"
        )
        assert "DP fold: StreamKIdx = batch*(nWG0*nWG1) + N*nWG0 + M" in src, (
            f"Kernel {base!r} missing the linear tile-index fold"
        )
        # The grid is rounded up to the cluster, so padded peers must exit before
        # the first cluster barrier or their peers wait on an arrive that never
        # comes.
        assert "padded work-group: exit before any cluster barrier/load" in src, (
            f"Kernel {base!r} missing the padded boundary-peer early exit"
        )
        # Both masks are bound: B broadcasts along Cs, A along Ck (self-only when
        # Ck == 1).
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
        # Absent peers are handled structurally (pad-exit + reduced masks), so the
        # runtime "is this cluster usable" selection guard must not be emitted.
        assert "nWG0 aligned to C?" not in src, (
            f"Kernel {base!r} emitted the runtime multicast selection guard"
        )
        # Ck is a spatial N-tiling axis: no K-split decode or maskB shift.
        assert "k = StreamKIdx & (Ck-1)" not in src, (
            f"Kernel {base!r} wrongly emitted a K-slice reduction decode"
        )
        if pgr >= 2:
            # The PGR>=2 prologue double-buffer prefetch region exists for K>DepthU.
            assert "skipPGR2" in src, (
                f"Kernel {base!r} missing the PGR2 prologue double-buffer region"
            )
            # That load sits past the generic bracketing boundary, so it needs its
            # own cluster-scope handshake.
            assert _skip_prefetch_handshake_brackets_load(src), (
                f"Kernel {base!r} PGR2 prologue prefetch load is NOT bracketed by "
                f"a cluster-scope -3 handshake"
            )


@pytest.mark.parametrize("pgr, cluster_dim", _ARMS, ids=_ARM_IDS)
def test_streamk_cluster_multicast_gfx1250_golden(snapshot, pgr, cluster_dim):
    """Golden: order-invariant {basename, err} digest, one per (PGR, shape) arm."""
    assert golden_digest(_emit(pgr, cluster_dim)) == snapshot
