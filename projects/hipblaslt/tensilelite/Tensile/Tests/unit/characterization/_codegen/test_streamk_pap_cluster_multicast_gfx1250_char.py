# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StreamK cluster multicast + PrefetchAcrossPersistent -- gfx1250
characterization (CPU-only).

PAP companion to ``test_streamk_cluster_multicast_gfx1250_char.py``. PAP re-emits
the TDM descriptor setup on every persistent-loop iteration, so a multicast mask
SGPR that the prologue frees would be referenced after it is undeclared. The
masks therefore have to stay live across the refresh -- with one exception, which
is what the two cluster shapes here pin:

  * ``[4, 1]`` -- Ck = 1, so A has no peers and its mask is just the self bit.
    Re-applying it is a no-op, so ``ClusterLoad.papDropsSelfOnlyMaskA`` frees the
    SGPR and skips the A attach, buying back a register against the 106-SGPR
    budget. B, a real broadcast, stays live.
  * ``[2, 2]`` -- Ck = 2, so A IS a real multicast: BOTH masks must stay live and
    both must still be attached. This guards the refinement above against
    dropping a genuine A-multicast.

Both shapes must fit the SGPR budget: at 107 SGPRs the kernel is replaced by an
``s_endpgm`` stub and the output tensor is left unwritten.

CPU-only: no GPU required.
"""

import os
import re

import pytest

from config_harness import (
    assert_assembles,
    assert_real_gfx1250_kernels,
    emit_kernels_from_config,
    golden_digest,
)

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_pap_cluster_multicast.yaml",
)

_SGPR_BUDGET = 106
_NEXT_FREE_SGPR = re.compile(r"\.amdhsa_next_free_sgpr\s+(\d+)")

_A_ATTACH = "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]"
_B_ATTACH = "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]"

# (ClusterDim, is A a real multicast that must survive the PAP refresh?)
_SHAPES = [((4, 1), False), ((2, 2), True)]
_SHAPE_IDS = ["cs4_ck1", "cs2_ck2"]


def _max_sgpr(src):
    return max((int(m.group(1)) for m in _NEXT_FREE_SGPR.finditer(src)), default=0)


def _emit(cluster_dim):
    # The limit truncates the fork permutations BEFORE the ClusterDim filter, so
    # it is a coverage cliff, not a performance knob: once the sweep outgrows it
    # the later cluster shapes fall outside the window and the filter finds
    # nothing to check. Kept at 16, matching the sibling multicast driver, so the
    # whole sweep stays inside it.
    return emit_kernels_from_config(_CONFIG, limit=16, arch=_ARCH,
                                    cluster_dim=cluster_dim)


@pytest.mark.parametrize("cluster_dim, a_has_peers", _SHAPES, ids=_SHAPE_IDS)
def test_streamk_pap_cluster_multicast_gfx1250_mask_liveness(cluster_dim, a_has_peers):
    """PAP keeps every mask that carries real peers live across the persistent
    refresh, and only frees the A mask when Ck == 1 makes it self-only."""
    results = _emit(cluster_dim)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        # B broadcasts along Cs in both shapes, so it is always re-applied.
        assert _B_ATTACH in src, (
            f"Kernel {base!r} dropped the B-multicast mask (MulticastMaskB)"
        )
        if a_has_peers:
            assert _A_ATTACH in src, (
                f"Kernel {base!r} dropped a real A-multicast mask under PAP "
                f"(ClusterDim={list(cluster_dim)} gives A peers along Ck)"
            )
        else:
            assert _A_ATTACH not in src, (
                f"Kernel {base!r} still attaches the self-only A mask under PAP; "
                "its SGPR is freed, so the reference would not assemble"
            )

        assert src.count("s_barrier_signal -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier arrive (s_barrier_signal -3)"
        )
        assert src.count("s_barrier_wait -3") >= 1, (
            f"Kernel {base!r} missing cluster barrier wait (s_barrier_wait -3)"
        )

        sgprs = _max_sgpr(src)
        assert 0 < sgprs <= _SGPR_BUDGET, (
            f"Kernel {base!r} uses {sgprs} SGPRs, exceeds the {_SGPR_BUDGET} budget"
        )
        # ClusterBarrier keeps WaveIdx live for the handshake; do not pack ArgType bit 8.
        assert re.search(r"s_bitcmp1_b32 s\[sgprArgType\],\s*(?:8|0x8)\b", src) is None, (
            f"Kernel {base!r} packed TDM parity into ArgType; ClusterBarrier must skip pack"
        )
        assert "s[sgprWaveIdx]" in src, (
            f"Kernel {base!r} lost the ClusterBarrier WaveIdx handshake"
        )
        assert re.search(r"^\s*\.set\s+sgprWaveIdx\s*,\s*UNDEF\s*$", src, re.MULTILINE) is None, (
            f"Kernel {base!r} undefined WaveIdx; ClusterBarrier must keep it live"
        )


@pytest.mark.parametrize("cluster_dim, _a_has_peers", _SHAPES, ids=_SHAPE_IDS)
def test_streamk_pap_cluster_multicast_gfx1250_golden(snapshot, cluster_dim, _a_has_peers):
    """Golden: order-invariant {basename, err} digest, one per cluster shape."""
    assert golden_digest(_emit(cluster_dim)) == snapshot
