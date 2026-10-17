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
    Re-applying it is a no-op, so ``ClusterLoad.persistentDropsSelfOnlyMaskA`` frees the
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
    solutions_from_config,
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


def _assert_pap_handoff(src, *, late=False):
    # A redundant waitcnt may merge with an earlier drain; the complete
    # local/cluster rendezvous must remain between eligibility and first issue.
    start = src.index("Do not prefetch a reservation twice")
    checkpoint = src.index("checkpoint WorkGroup0 for PAP restore", start)
    issue = src.index("tensor_load_to_lds", start)
    assert checkpoint < issue
    # The issue helper can add its own automatic cluster phases, and SIA4
    # can remove the setup module comment. Bound the explicit PAP phase by
    # its elected-wave join and the following all-peer cluster wait instead.
    phase = re.search(
        r"^label_PersistentMC_SkipPapSignal\w*:\s*\n\s*s_barrier_wait\s+-3\b",
        src[start:issue], re.MULTILINE)
    assert phase is not None
    phase_end = start + phase.end() if late else checkpoint
    assert phase_end < issue
    region = src[start:phase_end]
    barriers = list(re.finditer(r"s_barrier_(signal|wait)\s+(-[13])", region))
    assert [match.groups() for match in barriers] == [
        ("signal", "-1"), ("wait", "-1"), ("signal", "-3"), ("wait", "-3")]
    handoff_start = start + barriers[0].start()
    handoff_end = start + barriers[-1].end()
    if late:
        # Both parity alternatives, including the scale-address pair, and
        # the next-loop alpha check must join before every wave rendezvous.
        joins = list(re.finditer(
            r"^label_TDM(?:Init|GlobalOffset)\w*End\w*:\s*$", region, re.MULTILINE))
        assert len(joins) >= 4
        alpha_join = re.search(r"^label_PersistentAlphaCheck\w*:\s*$", region, re.MULTILINE)
        assert alpha_join is not None
        assert checkpoint < start + joins[-1].end() < handoff_start
        assert start + alpha_join.end() < handoff_start
    else:
        # SIA0 and unsupported layouts retain their original early handoff.
        assert handoff_end < checkpoint
    assert handoff_end < issue
    assert "restore current PersistentPhantomTile after PAP" in src
    assert "PAP already completed the next-tile cluster handoff?" in src


def _max_sgpr(src):
    return max((int(m.group(1)) for m in _NEXT_FREE_SGPR.finditer(src)), default=0)


def _expected_handoff_timing(cluster_dim):
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions

    # The legacy-named fixture resolves to DataParallel/StaticGrid. Its SIA0
    # kernels retain the early handoff; eligible SIA4 kernels defer it. Derive
    # that expectation from configuration, independently of assembly placement.
    solutions = solutions_from_config(_CONFIG, arch=_ARCH, limit_solutions=16)
    kernels = [kernel for kernel in generateKernelObjectsFromSolutions(solutions)
               if tuple(kernel["ClusterDim"]) == tuple(cluster_dim)]
    assert all(kernel["TileProcessingStrategy"] == "DataParallel"
               and kernel["WorkAssignment"] == "StaticGrid" for kernel in kernels)
    schedules = {getKernelFileBase(False, kernel): kernel["ScheduleIterAlg"]
                 for kernel in kernels}
    assert len(schedules) == len(kernels)
    assert set(schedules.values()) == {0, 4}
    return {base: schedule == 4 for base, schedule in schedules.items()}


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
    late_by_kernel = _expected_handoff_timing(cluster_dim)
    results = _emit(cluster_dim)
    assert_real_gfx1250_kernels(results)
    assert {base for base, _src, _err in results} == set(late_by_kernel)
    for base, src, _err in results:
        assert_assembles(src, base)
        _assert_pap_handoff(src, late=late_by_kernel[base])
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


def test_dataparallel_halfplr_pap_preserves_handoff_after_barrier_rebuild():
    config = os.path.join(os.path.dirname(_CONFIG), "dataparallel_halfplr_pap_multicast.yaml")
    results = emit_kernels_from_config(config, limit=1, arch=_ARCH)
    assert_real_gfx1250_kernels(results)
    for base, src, _err in results:
        assert_assembles(src, base)
        _assert_pap_handoff(src, late=True)
        assert 0 < _max_sgpr(src) <= _SGPR_BUDGET
