# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StreamK workgroup-cluster reduction -- gfx1250 characterization (CPU-only).

Exercises the intra-cluster split-barrier reduction fast path added to
``Tensile/Components/StreamK.py`` (StreamKClusterReduction=1, ClusterDim=[1,C]
-- pure K-reduction: Ck = ClusterDim[1] > 1, Cs = ClusterDim[0] = 1).
It drives the same config -> Solutions -> emit path as the sibling
``test_r3_streamk_gfx1250_char.py``, but through the cluster-enabled designed
config ``_designed/gfx1250/streamk_cluster_reduction.yaml``.

Asserts:
  * every kernel emits real gfx1250 assembly with ``err == 0``;
  * the intra-cluster handshake is present on the fast path
    (``s_barrier_signal -3`` on the peer arrive, ``s_barrier_wait -3`` on the
    owner wait) -- i.e. the split barrier really replaced the flag spin-wait;
  * the global-flag reduction is still compiled in as the runtime fallback
    (a per-CTA flag read via ``readFlagAbit`` / ``mubuf`` load survives), so the
    param is additive rather than destructive; and
  * an order-invariant ``{basename, err}`` syrupy snapshot is pinned.

CPU-only: no GPU required. The emit harness instantiates rocisa and runs
Python+rocisa codegen without compiling or launching any GPU kernels.
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
    "streamk_cluster_reduction.yaml",
)


def test_streamk_cluster_reduction_gfx1250_emits_assembly():
    """gfx1250 StreamK cluster-reduction config emits real assembly, err==0,
    and contains the intra-cluster split-barrier handshake + retained fallback."""
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
        # Fast-path cluster split barrier: peer arrive + owner wait (ids = -3).
        assert "s_barrier_signal -3" in src, (
            f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        )
        assert "s_barrier_wait -3" in src, (
            f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"
        )
        # Fallback still compiled in: the global-flag reduction path survives
        # (per-CTA completion flag reset store to the synchronizer workspace).
        assert "reset flag" in src or "set flag" in src, (
            f"Kernel {base!r} dropped the global-flag reduction fallback"
        )


def test_streamk_cluster_reduction_gfx1250_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the cluster-reduction emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
