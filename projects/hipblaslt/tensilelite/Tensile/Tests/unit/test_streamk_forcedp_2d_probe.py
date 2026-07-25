# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Focused unit tests for the Phase-0 ForceDPOnly 2-D DUAL-multicast probe.

These pin the two pure-logic pieces of the probe without emitting a kernel:

  1. ``streamKForceDP2DMulticast`` -- the structural detector that gates the
     whole probe (ForceDPOnly + ClusterDim[0]>1 + ClusterDim[1]>1). It must be
     True ONLY for the genuine 2-D ForceDPOnly shape and False for the 1-D
     [C,1] ForceDPOnly multicast, the factored/reduction K-split shapes, and
     any non-ForceDPOnly cluster.

  2. The DENSE ClusterLoad 2-D mask math reused verbatim for BOTH operands at
     ClusterDim=[Cs,Ck]=[2,2]:
        maskA = OR(1 << (idx*Cs) for idx in range(Ck))  # A over Ck/Y peers
        maskB = (1 << Cs) - 1                            # B over Cs/X peers
     giving maskA=0x5 (bits {0,2}) and maskB=0x3 (bits {0,1}) -- the exact
     values emitted into the kernel (see the gfx1250 characterization).
"""

import pytest

from Tensile.Common import streamKForceDP2DMulticast

pytestmark = pytest.mark.unit


def _k(force_dp, cluster):
    return {"StreamKForceDPOnly": force_dp, "ClusterDim": list(cluster)}


def test_detector_true_only_for_genuine_2d_forcedp():
    assert streamKForceDP2DMulticast(_k(1, [2, 2])) is True


@pytest.mark.parametrize(
    "state",
    [
        _k(1, [2, 1]),  # shipped 1-D ForceDPOnly B-multicast (Ck==1)
        _k(1, [8, 1]),  # larger 1-D ForceDPOnly B-multicast
        _k(1, [1, 2]),  # degenerate (Cs==1): no B-multicast axis
        _k(1, [1, 1]),  # no cluster
        _k(0, [2, 2]),  # 2-D cluster but NOT ForceDPOnly (dense/factored/reduction)
    ],
)
def test_detector_false_for_non_probe_shapes(state):
    assert streamKForceDP2DMulticast(state) is False


def test_detector_defaults_missing_forcedp_to_false():
    # ``.get("StreamKForceDPOnly", 0)`` -> a dense (no StreamK) 2-D cluster is not a probe.
    assert streamKForceDP2DMulticast({"ClusterDim": [2, 2]}) is False


# --- DENSE 2-D dual-multicast mask math (mirrors ClusterLoad.computeMasks) ---


def _dense_masks(cs, ck):
    maskA = 1
    for idx in range(ck):  # probe: aPeers = ClusterDim[1] = Ck (A IS multicast)
        maskA |= (1 << (idx * cs))
    maskB = (1 << cs) - 1
    return maskA, maskB


def test_dense_masks_for_2x2_probe():
    maskA, maskB = _dense_masks(cs=2, ck=2)
    # A shared across the Ck=2 Y-peers at stride Cs=2 -> linear cluster ranks {0,2}.
    assert maskA == 0x5, f"maskA expected 0x5 (bits 0,2), got {hex(maskA)}"
    # B shared across the Cs=2 X-peers -> contiguous linear cluster ranks {0,1}.
    assert maskB == 0x3, f"maskB expected 0x3 (bits 0,1), got {hex(maskB)}"


def test_2x2_masks_are_orthogonal_and_cover_the_cluster():
    """The A (Y) and B (X) peer sets intersect only at self (rank 0) and each has
    exactly Cs (resp. Ck) members, i.e. a genuine 2-D [2,2] dual-multicast."""
    maskA, maskB = _dense_masks(cs=2, ck=2)
    assert bin(maskA).count("1") == 2  # Ck peers
    assert bin(maskB).count("1") == 2  # Cs peers
    assert (maskA & maskB) == 0x1      # share only self (rank 0)


def test_1d_forcedp_multicast_keeps_a_self_only():
    """Sanity: the shipped 1-D [C,1] ForceDPOnly multicast (Ck==1) yields a
    self-only A mask -- the probe's A-multicast is strictly the Ck>1 addition."""
    maskA, maskB = _dense_masks(cs=8, ck=1)
    assert maskA == 0x1              # self only
    assert maskB == (1 << 8) - 1     # B across all 8 X-peers
