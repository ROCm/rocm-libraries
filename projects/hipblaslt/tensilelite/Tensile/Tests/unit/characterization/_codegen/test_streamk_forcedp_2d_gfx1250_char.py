# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""ForceDPOnly 2-D DUAL-multicast probe -- gfx1250 characterization (CPU-only).

Phase-0 probe: a StreamK==3 ``StreamKForceDPOnly`` (dense data-parallel, NO
K-split reduction) kernel given a GENUINE 2-D cluster ClusterDim=[Cs,Ck]=[2,2]
in which BOTH operands are TDM-multicast:

  * Cs = ClusterDim[0] = 2 : X-peers on M-ADJACENT output tiles reuse B
    (exactly as the shipped 1-D [C,1] ForceDPOnly multicast); and
  * Ck = ClusterDim[1] = 2 : Y-peers on N-ADJACENT output tiles reuse A.

Unlike the factored cluster (``streamk_factored_cluster``) Ck here is an
N-tiling / A-multicast axis, NOT a K-split reduction axis, so the factored
"k = StreamKIdx & (Ck-1)" decode and the "MulticastMaskB = maskB_base << k"
shift MUST be ABSENT.

Asserts (see ``_designed/gfx1250/streamk_forcedp_2d.yaml``):
  * the kernel emits real gfx1250 assembly with ``err == 0``;
  * the 2-D DP tile decode folds the genuine 2-D (+batch) HW workgroup coords
    into the linear ForceDPOnly index
      StreamKIdx = batch*(nWG0*nWG1) + WorkGroup1*nWG0 + WorkGroup0
    (M-fastest) so X-peers land M-adjacent (reuse B) and Y-peers N-adjacent
    (reuse A);
  * the DENSE ClusterLoad 2-D masks are reused verbatim for BOTH operands and
    the runtime preLoop overwrite is short-circuited (keep-dense-masks note);
  * both MulticastMaskA and MulticastMaskB are bound onto their TDM descriptors;
  * the cluster split-barrier (``s_barrier_signal/-wait -3``) is present and the
    prologue/epilogue arrive+wait phases co-exist; and
  * the factored K-split decode/shift are ABSENT (this is a spatial N-axis).

CPU-only: no GPU required. The DEFINITIVE correctness check is the user's HW run
of ``Tests/common/streamk/gfx1250/core/sk_mxf4_force_dp_only_cluster_2d_multicast.yaml`` (functional-sim
is a regression check only).
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
    "streamk_forcedp_2d.yaml",
)


def test_streamk_forcedp_2d_gfx1250_emits_assembly():
    """gfx1250 ForceDPOnly [2,2] probe emits real assembly, err==0, with the
    2-D DP tile decode + DENSE dual-multicast masks (A on Ck/Y, B on Cs/X) and
    NO factored K-split decode."""
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

        # --- 2-D DP tile decode: StreamKIdx = batch*(nWG0*nWG1) + N*nWG0 + M ---
        assert "2-D DP: WorkGroup1 * nWG0 (N-tile row)" in src, (
            f"Kernel {base!r} missing the N-tile-row fold (WorkGroup1*nWG0)"
        )
        assert "2-D DP: StreamKIdx = batch*(nWG0*nWG1) + N*nWG0 + M" in src, (
            f"Kernel {base!r} missing the linear 2-D DP StreamKIdx fold"
        )

        # --- dense 2-D masks reused for BOTH operands; no preLoop overwrite ---
        assert "keep dense 2-D masks" in src, (
            f"Kernel {base!r} missing the dense-mask short-circuit note"
        )
        assert "sgprMulticastMaskA" in src, (
            f"Kernel {base!r} never binds MulticastMaskA (A is NOT multicast!)"
        )
        assert "sgprMulticastMaskB" in src, (
            f"Kernel {base!r} never binds MulticastMaskB to a B descriptor"
        )

        # --- cluster split-barrier present + prologue/epilogue phases co-exist ---
        assert src.count("s_barrier_signal -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        )
        assert src.count("s_barrier_wait -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"
        )
        assert "cluster_barrier signal (arrive)" in src, (
            f"Kernel {base!r} missing the prologue cluster arrive"
        )
        assert "cluster_barrier wait" in src, (
            f"Kernel {base!r} missing a cluster barrier wait phase"
        )

        # --- factored K-split decode/shift MUST be ABSENT (Ck is a spatial axis) ---
        assert "k = StreamKIdx & (Ck-1)" not in src, (
            f"Kernel {base!r} wrongly emitted the factored K-slice decode"
        )
        assert "MulticastMaskB = maskB_base << k" not in src, (
            f"Kernel {base!r} wrongly emitted the factored K-split maskB shift"
        )
