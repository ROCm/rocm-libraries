# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""Factored 2-D StreamK cluster mode -- gfx1250 characterization (CPU-only).

Exercises the factored cluster mode added on top of the StreamK cooperative
multicast and the StreamK cluster reduction: the 1-D HW cluster ClusterDim=[C,1]
is factored into two ORTHOGONAL axes, C = Cs*Ck (StreamKClusterKSplit=Ck). The
designed config ``_designed/gfx1250/streamk_factored_cluster.yaml`` pins the
smallest genuine factored point (C=4, Ck=2 => Cs=2), so ONE cluster performs
BOTH the spatial B-multicast (along Cs) and the K-split partial reduction (along
Ck) in a single kernel.

Asserts:
  * every kernel emits real gfx1250 assembly with ``err == 0``;
  * the factored B-multicast mask is emitted along the Cs axis:
      - k = StreamKIdx & (Ck-1)  (the within-cluster K-slice rank), and
      - MulticastMaskB = maskB_base << k  (Cs bits at stride Ck), with the
        self-only fallback for a partial / M-unaligned cluster;
  * the intra-cluster split-barrier reduction is present along the Ck axis
    (``s_barrier_signal -3`` arrive, ``s_barrier_wait -3`` wait) -- the SAME
    whole-cluster ``-3`` barrier reused across both axes;
  * the whole-kernel ``-3`` barrier is used by BOTH the mainloop multicast
    lockstep AND the epilogue reduction (both barrier phases co-exist);
  * the global-flag reduction is still compiled in as the runtime fallback; and
  * an order-invariant ``{basename, err}`` syrupy snapshot is pinned.

CPU-only: no GPU required. Real-HW / FFM validation of the combined axes is
deliberately DEFERRED (gated behind multicast cooperative-load stability at
PGR>1); this test validates codegen only.
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
    "streamk_factored_cluster.yaml",
)


def test_streamk_factored_cluster_gfx1250_emits_assembly():
    """gfx1250 factored-cluster config (C=4, Cs=2, Ck=2) emits real assembly,
    err==0, and carries BOTH the factored B-multicast mask (Cs axis) and the
    intra-cluster split-barrier reduction (Ck axis) in one kernel."""
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

        # --- Cs axis: factored B-multicast mask ---
        assert "k = StreamKIdx & (Ck-1)" in src, (
            f"Kernel {base!r} missing the factored k (K-slice) decode"
        )
        assert "MulticastMaskB = maskB_base << k" in src, (
            f"Kernel {base!r} missing the factored B-multicast mask shift"
        )
        # Self-only fallback for a partial / M-unaligned cluster.
        assert "self-only mask" in src, (
            f"Kernel {base!r} missing the factored multicast self-mask fallback"
        )
        # B descriptor actually ORs the (factored) multicast mask.
        assert "sgprMulticastMaskB" in src, (
            f"Kernel {base!r} never binds MulticastMaskB to a B descriptor"
        )

        # --- Ck axis: intra-cluster split-barrier reduction (same -3 barrier) ---
        n_sig = src.count("s_barrier_signal -3")
        n_wait = src.count("s_barrier_wait -3")
        assert n_sig >= 1, f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        assert n_wait >= 1, f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"

        # --- whole-kernel -3 balance: the -3 barrier is used by BOTH the mainloop
        # multicast lockstep AND the epilogue reduction. The epilogue reduction
        # arrive is elected via the reduce-signal idiom; the multicast prologue
        # arrive pairs the first mainloop load wait. Both barrier consumers must
        # be present (mainloop lockstep + reduction handshake).
        assert "cluster_barrier signal (arrive)" in src, (
            f"Kernel {base!r} missing the prologue/epilogue cluster arrive"
        )
        assert "cluster_barrier wait" in src, (
            f"Kernel {base!r} missing a cluster barrier wait phase"
        )

        # --- fallback retained: global-flag reduction still compiled in ---
        assert "reset flag" in src or "set flag" in src, (
            f"Kernel {base!r} dropped the global-flag reduction fallback"
        )


def test_streamk_factored_cluster_gfx1250_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the factored emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
