# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ForceDPOnly=0 PGR=2: skip prefetch -3 after DP->SK self-only mask clear."""

import os

import pytest

from config_harness import (
    assert_assembles,
    assert_cluster_barrier_balanced,
    assert_real_gfx1250_kernels,
    assert_pgr2_persist_prefetch_close_wait,
    assert_skip_pgr2_leftover_tdm_drain,
    assert_skip_pgr2_skip_path_handshake,
    assert_zero_iter_prefetch_handshake_preserves_scc,
    emit_kernels_from_config,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_cluster_multicast_pgr2_cd22.yaml",
)

_SELF_ONLY = (
    "SK-tail self-only (maskA==maskB after DP->SK clear): skip prefetch -3"
)


def test_streamk_cluster_multicast_pgr2_skips_self_only_prefetch_hs():
    results = emit_kernels_from_config(_CONFIG, limit=2, arch=_ARCH)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "continuing SK: skip per-pass multicast -3 (self-only)" in src, (
            f"Kernel {base!r}: continue-SK must skip graWorkGroup per-pass -3"
        )
        assert _SELF_ONLY in src, (
            f"Kernel {base!r}: PGR>=2 SK-tail must skip prefetch -3 when "
            f"masks are self-only (maskA==maskB)"
        )
        assert "self-only SK: skip multicast prefetch -3" in src, (
            f"Kernel {base!r}: missing self-only skipPGR2 / ZeroIter branch"
        )
        assert_skip_pgr2_skip_path_handshake(src, base)
        assert_skip_pgr2_leftover_tdm_drain(src, base)
        assert_cluster_barrier_balanced(src, base)
        assert_pgr2_persist_prefetch_close_wait(src, base)
        assert_zero_iter_prefetch_handshake_preserves_scc(src, base)
        assert "drain persist leftover TDM before persist re-entry" not in src, (
            f"Kernel {base!r}: persist close must not tdmWait(0) leftover TDM"
        )
