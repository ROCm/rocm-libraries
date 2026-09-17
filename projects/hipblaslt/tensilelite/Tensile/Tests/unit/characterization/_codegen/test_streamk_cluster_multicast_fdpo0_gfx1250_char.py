# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ForceDPOnly=0 cluster multicast: DP multicast, then workspace/flag SK."""

import os

import pytest

from config_harness import (
    assert_assembles,
    assert_real_gfx1250_kernels,
    assert_pgr1_persist_dp_close_wait,
    emit_kernels_from_config,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_DESIGNED = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
)

_CONFIG = os.path.join(_DESIGNED, "streamk_cluster_multicast_fdpo0.yaml")


def _assert_loop_omits_cluster_barrier(src, base):
    begin = src.find("label_LoopBeginL:")
    end = src.find("label_LoopEndL:")
    loop = src[begin:end] if begin >= 0 and end > begin else ""
    assert loop, f"Kernel {base!r}: missing LoopBeginL/LoopEndL span"
    assert "s_barrier_signal -3" not in loop, (
        f"Kernel {base!r}: ForceDPOnly=0 multicast LoopBeginL must not emit "
        f"cluster -3 (mixed LoopCounter cannot pair Rule 3)"
    )
    assert "s_barrier_wait -3" not in loop, (
        f"Kernel {base!r}: ForceDPOnly=0 multicast LoopBeginL must not wait "
        f"cluster -3"
    )


def test_streamk_cluster_multicast_fdpo0_emits_assembly():
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "cluster_last" not in src, (
            f"Kernel {base!r}: SK partials must stay on the workspace flag path"
        )
        assert "skip global-flag store" not in src, (
            f"Kernel {base!r}: SK partials must not skip the workspace flag"
        )
        assert "set flag" in src, (
            f"Kernel {base!r}: SK partials must publish the workspace flag"
        )
        assert "continuing SK: skip per-pass multicast -3 (self-only)" in src, (
            f"Kernel {base!r}: continue-SK must skip graWorkGroup per-pass -3 "
            f"(owner still in fixup; SK loads are self-only)"
        )
        assert "last DP tile: still need per-pass multicast -3" in src, (
            f"Kernel {base!r}: DP->SK switch must not KernelEnd before last DP -3"
        )
        assert "clear BOTH A & B broadcast masks at DP->SK boundary" in src, (
            f"Kernel {base!r}: missing DP->SK dual-mask clear"
        )
        assert "padded if WorkGroup1 >= gridY (skGrid/nWG0)" in src, (
            f"Kernel {base!r}: missing persistent-gridY pad-exit (Ck-split Y-peers)"
        )
        assert "padded if WorkGroup1 (N-tile) >= tilesN" not in src, (
            f"Kernel {base!r}: ForceDPOnly=0 [Cs,Ck] must not pad-exit Ck-split "
            f"Y-peers against tilesN"
        )
        _assert_loop_omits_cluster_barrier(src, base)
        assert "retire cooperative tensor_load_to_lds before back-edge" in src, (
            f"Kernel {base!r}: multicast must retire cooperative tensor_load "
            f"(no loop Rule 3 to drain the broadcast)"
        )
        assert "SK-tail self-only (maskA==maskB after DP->SK clear): skip prefetch -3" not in src, (
            f"Kernel {base!r}: PGR=1 has no prefetch -3 skip (no skipPGR2 / ZeroIter)"
        )
        assert_pgr1_persist_dp_close_wait(src, base)
        assert "drain persist leftover TDM before persist re-entry" not in src, (
            f"Kernel {base!r}: persist close must not tdmWait(0) leftover TDM "
            f"(vlcnt=0 deadlocks persist overlap; KernelEnd drains TDM)"
        )
