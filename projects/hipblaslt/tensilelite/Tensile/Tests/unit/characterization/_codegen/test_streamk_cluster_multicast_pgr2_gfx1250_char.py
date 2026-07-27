# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StreamK DP cooperative B-multicast, double-buffered prologue prefetch
(PrefetchGlobalRead=2) -- gfx1250 characterization (CPU-only).

Companion to ``test_streamk_cluster_multicast_gfx1250_char.py`` (which pins
PrefetchGlobalRead=1). This exercises the ``PrefetchGlobalRead=2`` path with
K > DepthU, so the prologue emits the second, double-buffered ("LDS1")
cooperative multicast prefetch load. That prefetch load is emitted inside the
single-iteration guard branch, past the generic per-load cluster-barrier
bracketing boundary, so ``StreamK.streamKMulticastProloguePrefetchHandshake``
must bracket it with a dedicated cluster-scope split-barrier handshake.

Asserts, in addition to the sibling config's split-mask / predicate checks:
  * the double-buffered prologue prefetch load is bracketed by a wave-0-elected
    cluster-scope ``s_barrier_signal -3`` + ``s_barrier_wait -3`` handshake
    (label ``SKMC_SkipPrefetchSignal``); and
  * the cluster-scope split barrier stays balanced: every ``s_barrier_signal -3``
    is matched by an ``s_barrier_wait -3``.

CPU-only: no GPU required.
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
    "streamk_cluster_multicast_pgr2.yaml",
)


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


def test_streamk_cluster_multicast_pgr2_gfx1250_emits_assembly():
    """gfx1250 SK3 cluster multicast config at PrefetchGlobalRead=2 (K>DepthU)
    emits real assembly with err==0, and the double-buffered prologue prefetch
    multicast load is bracketed by a balanced cluster-scope -3 handshake."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 target"
        # Split A/B multicast: B broadcast, A self-only.
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
            f"Kernel {base!r} missing B-broadcast mask on the B descriptor"
        )
        # The PGR>=2 prologue double-buffer prefetch region must exist for K>DepthU.
        assert "skipPGR2" in src, (
            f"Kernel {base!r} missing the PGR2 prologue double-buffer region"
        )
        # The double-buffered prologue prefetch multicast load must be bracketed
        # by a dedicated cluster-scope handshake (the fix under test).
        assert _skip_prefetch_handshake_brackets_load(src), (
            f"Kernel {base!r} PGR2 prologue prefetch load is NOT bracketed by a "
            f"cluster-scope -3 handshake"
        )
        # Cluster-scope split-barrier balance (shared check).
        assert_cluster_barrier_balanced(src, base)


def test_streamk_cluster_multicast_pgr2_gfx1250_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the PGR2 multicast emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
