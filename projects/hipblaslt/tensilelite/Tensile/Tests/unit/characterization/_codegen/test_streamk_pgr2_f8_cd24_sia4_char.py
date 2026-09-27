# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""F8 ForceDPOnly=1 ClusterDim=[2,4] PGR=2 SIA=4 skipPGR2 handshake."""

import os
import re

import pytest

from config_harness import (
    assert_assembles,
    assert_cluster_barrier_balanced,
    assert_real_gfx1250_kernels,
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
    "streamk_cluster_multicast_pgr2_cd24_sia4.yaml",
)


def _emit():
    # Single-permutation yaml (one MI, one CD, one SIA). Do not pass
    # cluster_dim= into emit_kernels_from_config: that filter runs after
    # generateKernelObjectsFromSolutions and the Solution object's ClusterDim
    # is not always a plain list at that point.
    return emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)


def _skip_window(src):
    lines = src.splitlines()
    skip_idx = join_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("label_skipPGR2_1:"):
            skip_idx = i
        elif ln.startswith("label_skipPGR2_2:") and skip_idx is not None:
            join_idx = i
            break
    assert skip_idx is not None and join_idx is not None
    return lines[skip_idx:join_idx]


def _first_no_work_kernelend_line(src):
    """Line index of the prologue no-work branch to KernelEnd (before skipPGR2)."""
    lines = src.splitlines()
    skip_idx = next(i for i, ln in enumerate(lines) if ln.startswith("label_skipPGR2_1:"))
    hits = [
        i
        for i, ln in enumerate(lines)
        if i < skip_idx and "label_KernelEnd" in ln and "s_add_i32" in ln
    ]
    assert hits, "expected a prologue no-work branch to KernelEnd before skipPGR2"
    return hits[0]


def _lds0_to_pingpong_xor_window(src):
    """ISA from the first prologue LDS0 tensor_load through the TDM LDS XOR."""
    lines = src.splitlines()
    skip_idx = next(i for i, ln in enumerate(lines) if ln.startswith("label_skipPGR2_1:"))
    first_tl = xor_idx = None
    for i, ln in enumerate(lines):
        if i >= skip_idx:
            break
        if "tensor_load_to_lds" in ln and "sync LDS0" in ln and first_tl is None:
            first_tl = i
        if first_tl is not None and "s_xor_b32 s[sgprtdmAGroup0+1]" in ln:
            xor_idx = i
            break
    assert first_tl is not None, "expected a prologue LDS0 tensor_load_to_lds before skipPGR2"
    assert xor_idx is not None, "expected TDM LDS ping-pong XOR of tdmAGroup0+1 before skipPGR2"
    return lines[first_tl : xor_idx + 1]


def test_streamk_pgr2_f8_cd24_sia4_skip_handshake():
    """skipPGR2 LC==1 is paired; SIA=4 keeps tensorcnt plus cluster -3."""
    results = _emit()
    assert_real_gfx1250_kernels(results)
    assert len(results) == 1, f"expected the pinned MT64 CD[2,4] PGR2 SIA4 kernel, got {results!r}"
    base, src, _err = results[0]
    assert_assembles(src, base)
    # File basename is a hash; the kernel symbol carries the Tensile min name.
    assert "PGR2" in src and "SIA4" in src and "CD2_4" in src, base
    assert_cluster_barrier_balanced(src, base)
    assert_skip_pgr2_skip_path_handshake(src, base)
    assert_skip_pgr2_leftover_tdm_drain(src, base)
    assert_zero_iter_prefetch_handshake_preserves_scc(src, base)
    assert "PGR1 persist-DP: wait last -3 at persist close" not in src, (
        f"Kernel {base!r}: PGR2 / ForceDPOnly=1 must not wait persist-DP -3 at persist close"
    )
    assert "PGR2 skipPGR2: wait last -3 at persist close" not in src, (
        f"Kernel {base!r}: ForceDPOnly=1 must keep self-contained skipPGR2 wait "
        f"(persist-close skipPGR2 wait is ForceDPOnly=0 leftover close)"
    )
    assert "SK-tail self-only (maskA==maskB after DP->SK clear): skip prefetch -3" not in src, (
        f"Kernel {base!r}: ForceDPOnly=1 must not skip skipPGR2 -3 on maskA==maskB"
    )

    window = _skip_window(src)
    sigs = [i for i, w in enumerate(window) if "s_barrier_signal -3" in w]
    waits = [i for i, w in enumerate(window) if "s_barrier_wait -3" in w]
    assert sigs and waits and sigs[0] < waits[0], (
        f"Kernel {base!r} skipPGR2 -3 signal must precede wait"
    )
    assert not any("s_wait_tensorcnt" in w for w in window[: waits[0] + 1]), (
        f"Kernel {base!r}: s_wait_tensorcnt must not sit on skipPGR2 -3"
    )
    assert any("s_cbranch_scc1 label_toPGR1" in ln for ln in src.splitlines()), (
        f"Kernel {base!r}: LC==1 must take toPGR1 (K==DepthU skip of loop Rule 3)"
    )

    kend = _first_no_work_kernelend_line(src)
    # Pads already handshake-exited. Non-pad no-work is now flagged as a pad
    # (StreamKIdx >= totalTiles) so those WGs s_endpgm after -3 and never
    # ld_bcst. The KernelEnd below is a safety net, not the live no-work path.
    assert "Make sure there's work to do" in "\n".join(src.splitlines()[kend - 12 : kend + 1]), (
        f"Kernel {base!r}: prologue no-work KernelEnd is not the DP StreamKIter check"
    )

    lines = src.splitlines()
    pad_nw = next((i for i, ln in enumerate(lines)
                   if "no-work if StreamKIdx >= totalTiles" in ln), None)
    mask_nw = next((i for i, ln in enumerate(lines)
                    if "no-work cluster if WorkGroup2 >= batch" in ln), None)
    hs = next((i for i, ln in enumerate(lines)
               if "elect wave 0 to signal -3; all waves wait" in ln), None)
    skip_idx = next(i for i, ln in enumerate(lines) if ln.startswith("label_skipPGR2_1:"))
    assert pad_nw is not None, (
        f"Kernel {base!r}: missing ForceDPOnly no-work pad predicate "
        f"(StreamKIdx >= totalTiles must be treated like a pad)"
    )
    assert mask_nw is not None, (
        f"Kernel {base!r}: missing ForceDPOnly no-work mask clamp "
        f"(WorkGroup2 >= batch must drop KernelEnd-before-TDM WGs like pads)"
    )
    assert "no-work: handshake then s_endpgm; never ld_bcst" in src, (
        f"Kernel {base!r}: no-work WGs must handshake then s_endpgm (never ld_bcst)"
    )
    assert "empty maskRow: KernelEnd-before-TDM WGs excluded like pads" in src, (
        f"Kernel {base!r}: maskRow must drop KernelEnd-before-TDM WGs"
    )
    assert hs is not None and mask_nw < pad_nw < hs < kend < skip_idx, (
        f"Kernel {base!r}: expected mask clamp, then no-work pad flag, then "
        f"prologue -3, then safety-net KernelEnd, then skipPGR2 "
        f"(mask={mask_nw}, pad={pad_nw}, hs={hs}, kend={kend}, skip={skip_idx})"
    )

    # CD[2,4] cluster-local masks (A along Ck stride Cs=2, B along Cs).
    # Batch/Z is cluster_z with clusterDim.z=1 and does not enter the mask.
    assert "s_mov_b32 s[sgprMulticastMaskA], 0x55" in src, (
        f"Kernel {base!r}: expected maskA=0x55 for ClusterDim[2,4]"
    )
    assert "s_mov_b32 s[sgprMulticastMaskB], 0x3" in src, (
        f"Kernel {base!r}: expected maskB=0x3 for ClusterDim[2,4]"
    )

    tensorcnt_imms = re.findall(r"s_wait_tensorcnt\s+(\d+)", src)
    assert tensorcnt_imms, f"Kernel {base!r}: expected s_wait_tensorcnt"
    # Skip-path / ping-pong drains must be a full wait (0). Other sites may
    # use a non-zero remaining count (e.g. waitcnt insertion at skipPGR2_2).

    pingpong = _lds0_to_pingpong_xor_window(src)
    pingpong_waits = [ln for ln in pingpong if "s_wait_tensorcnt" in ln]
    assert any(re.search(r"s_wait_tensorcnt\s+0\b", ln) for ln in pingpong_waits), (
        f"Kernel {base!r}: prologue LDS0 TDM (A/MXSA aliased with B/MXSB) must "
        f"s_wait_tensorcnt 0 before the LDS ping-pong XOR; skipPGR2 after the XOR "
        f"is too late for F8 MX. window waits={pingpong_waits!r}"
    )
    xor_i = next(i for i, ln in enumerate(pingpong) if "s_xor_b32 s[sgprtdmAGroup0+1]" in ln)
    wait_is = [i for i, ln in enumerate(pingpong) if re.search(r"s_wait_tensorcnt\s+0\b", ln)]
    assert wait_is and wait_is[0] < xor_i, (
        f"Kernel {base!r}: s_wait_tensorcnt 0 must precede tdmAGroup0+1 XOR "
        f"(WaitDataflow or Tensile may drain; do not require the InsertClusterBarrierPass "
        f"comment). wait_is={wait_is!r} xor_i={xor_i}"
    )
