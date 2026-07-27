# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Target A: STANDARD StreamK 2-D DUAL-multicast -- gfx1250 characterization.

Phase-1 [2,2] probe on the STANDARD two-tile StreamK path
(``StreamKForceDPOnly=0``): a genuine 2-D cluster ClusterDim=[Cs,Ck]=[2,2] where

  * Cs = ClusterDim[0] = 2 : X-peers on M-ADJACENT tiles reuse B; and
  * Ck = ClusterDim[1] = 2 : Y-peers on N-ADJACENT tiles reuse A

on the DP (full-tile) round, while the SK (partial-tile) round reduces 1-D via
the workspace exactly as today. It is opted in via ``StreamKDualMulticast=1``
(mutually exclusive with the factored [Cs,Ck] K-reduction path).

Unlike the ForceDPOnly-2D probe (``streamk_forcedp_2d``) this kernel HAS a real
SK round, so it additionally asserts:
  * the DP->SK boundary drops BOTH masks to self-only (not just B); and
  * the SK partial-tile workspace/reduction machinery is intact.

Asserts (see ``_designed/gfx1250/streamk_dual_2d_multicast.yaml``):
  * err == 0, real gfx1250 assembly;
  * DP round binds BOTH MulticastMaskA (0x5) and MulticastMaskB (0x3) -- A is
    genuinely multicast -- via the dense 2-D masks (keep-dense short-circuit);
  * the 2-D DP tile fold StreamKIdx = batch*(nWG0*nWG1)+N*nWG0+M is present;
  * the DP->SK boundary clears BOTH masks (self = maskA & maskB);
  * the SK partial round + cluster split-barrier arrive/wait are present; and
  * the factored K-split decode/shift are ABSENT (Ck is a spatial N-axis).

CPU-only. The DEFINITIVE correctness check is the user's HW run of
``Tests/common/streamk/gfx1250/core/sk_mxf4_2d_multicast_probe.yaml``.
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
    "streamk_dual_2d_multicast.yaml",
)


def test_streamk_dual_2d_multicast_gfx1250_emits_assembly():
    """gfx1250 standard-StreamK [2,2] dual-2D probe emits real assembly, err==0,
    with dual DP masks (A on Ck/Y, B on Cs/X), the 2-D DP fold, a DP->SK BOTH-mask
    clear, an intact SK partial round, and NO factored K-split decode."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src and "gfx1250" in src, (
            f"Kernel {base!r} missing gfx1250 target"
        )
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"

        # --- (a) DP round binds BOTH operands' masks with the dense 2-D values ---
        # maskA=0x5 (bits {0,2}) multicasts A across the Ck=2 Y-peers; maskB=0x3
        # (bits {0,1}) multicasts B across the Cs=2 X-peers.
        assert "s[sgprMulticastMaskA], 0x5" in src, (
            f"Kernel {base!r} DP round does not set maskA=0x5 (A NOT multicast!)"
        )
        assert "s[sgprMulticastMaskB], 0x3" in src, (
            f"Kernel {base!r} DP round does not set maskB=0x3 (B NOT multicast!)"
        )
        assert "keep dense 2-D masks" in src, (
            f"Kernel {base!r} missing the dense-mask short-circuit note"
        )
        assert "sgprMulticastMaskA" in src and "sgprMulticastMaskB" in src, (
            f"Kernel {base!r} never binds both multicast masks to TDM descriptors"
        )

        # --- (b) 2-D DP tile fold: StreamKIdx = batch*(nWG0*nWG1) + N*nWG0 + M ---
        assert "2-D DP: WorkGroup1 * nWG0 (N-tile row)" in src, (
            f"Kernel {base!r} missing the N-tile-row fold (WorkGroup1*nWG0)"
        )
        assert "2-D DP: StreamKIdx = batch*(nWG0*nWG1) + N*nWG0 + M" in src, (
            f"Kernel {base!r} missing the linear 2-D DP StreamKIdx fold"
        )

        # --- (c) DP->SK boundary clears BOTH masks (dual mode), self=maskA&maskB ---
        assert "clear BOTH A & B broadcast masks at DP->SK boundary" in src, (
            f"Kernel {base!r} missing the dual DP->SK BOTH-mask clear"
        )
        assert "self = maskA & maskB" in src, (
            f"Kernel {base!r} missing the self = maskA & maskB reduction bit"
        )
        assert "drop A & B broadcast -> self-only" in src, (
            f"Kernel {base!r} missing the A&B self-only drop"
        )

        # --- (d) a real SK partial round exists (this is NOT ForceDPOnly) ---
        # DP-first grid-stride shift + the SK-section offset are only emitted on
        # the two-tile (StreamKForceDPOnly=0) schedule.
        assert "DP iterations shift" in src, (
            f"Kernel {base!r} missing the DP grid-stride shift (no DP round?)"
        )
        assert "Offset to start of SK section" in src, (
            f"Kernel {base!r} missing the SK-section offset (no SK partial round?)"
        )

        # --- (e) cluster split-barrier present (prologue arrive + wait phases) ---
        assert src.count("s_barrier_signal -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        )
        assert src.count("s_barrier_wait -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"
        )
        assert "cluster_barrier signal (arrive)" in src, (
            f"Kernel {base!r} missing the prologue cluster arrive"
        )

        # --- (f) SK-round deadlock fix: the cluster prologue arrive is emitted
        # PER PERSISTENT PASS (INSIDE the persistent loop), not once before it.
        # The multicast split barrier is 1 arrive + 1 first-load wait PER PASS; the
        # SK partial round re-enters the persistent loop, so a single pre-loop arrive
        # would leave the SK pass's wait unpaired -> cluster -3 barrier deadlock
        # (HW-observed hang). The arrive must sit after label_PersistentLoopStart so
        # every DP and SK pass balances its own first-load/zero-iter wait.
        loop_top = src.find("label_PersistentLoopStart:")
        assert loop_top != -1, f"Kernel {base!r} has no persistent loop"
        arrive_pos = src.find("cluster_barrier signal (arrive)")
        assert arrive_pos > loop_top, (
            f"Kernel {base!r} emits the cluster prologue arrive BEFORE the persistent "
            f"loop (once) -- the SK partial round's first-load wait would be unpaired "
            f"(the [2,2] SK-round deadlock). It must be per-pass (inside the loop)."
        )

        # --- factored K-split decode/shift MUST be ABSENT (Ck is a spatial axis) ---
        assert "k = StreamKIdx & (Ck-1)" not in src, (
            f"Kernel {base!r} wrongly emitted the factored K-slice decode"
        )
        assert "MulticastMaskB = maskB_base << k" not in src, (
            f"Kernel {base!r} wrongly emitted the factored K-split maskB shift"
        )
