# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) codegen characterization.

Covers the gfx1250 arms of tensilelite/Components/SdmaRingEmitter.py,
SdmaPacketEmitter.py and the FusedA2A: GemmA2A store path in
GlobalWriteBatch.py: the ring goes through VMEM stores and atomics, ring polls
carry a scope, the PUSH D stores are device scope, and the handshake keeps its
barrier.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # tensilelite/Tests/
        "common", "comm", "gfx1250", "fused_a2a.yaml",
    )
)

_A2A_MARKERS = (
    "global_atomic_inc_u32",
    "global_atomic_max_u64",
    "global_atomic_cmpswap_b64",
    "global_store_b128",
    "s_bfm_b32",
)

# ATOMIC DW0 and COPY_LINEAR_SUBWIN DW12 with their scope fields at SCOPE_SYS.
_GFX12_PACKET_CONSTANTS = ("0x1e30000a", "0xc0c0000")

# Ring polls, matched on their comments, and the scope each must carry.
_POLL_SCOPES = {
    "cachedHwReadIndex = hardware rptr": "SCOPE_SYS",
    "hwWptr = *wptr": "SCOPE_SYS",
    "seed cur = cachedWptr": "SCOPE_DEV",
    "poll committedWptr": "SCOPE_DEV",
}

_HANDSHAKE_WAIT = "fused-A2A: my A2A stores"


def _instructions(src):
    return [ln.strip() for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith(("/", ";", "."))
            and not ln.strip().split("//")[0].rstrip().endswith(":")]


def _packet_runs(insts):
    """Ring packet store runs in program order, each as (kind, store EXEC values,
    EXEC values of the VALU that built it), EXEC being the last
    `s_mov_b32 exec_lo, X` before the instruction."""
    exe, runs, cur, builds = None, [], None, []
    for i, ln in enumerate(insts):
        m = re.match(r"s_mov_b32 exec_lo, (\S+)", ln)
        if m:
            exe = m.group(1)
            continue
        if ln.startswith("v_"):
            for kind, tag in (("COPY", "SUBWIN DW"), ("ATOMIC", "ATOMIC DW")):
                if tag in ln:
                    builds.append((i, kind, exe))
        m = re.search(r"ring\[addr \+ (\d+)\] = packet dword", ln)
        if not m:
            continue
        if m.group(1) == "0":
            cur = {"start": i, "dwords": set(), "exec": set()}
            runs.append(cur)
        cur["dwords"].add(int(m.group(1)))
        cur["exec"].add(exe)
    out, since = [], {"COPY": -1, "ATOMIC": -1}
    for run in runs:
        kind = "COPY" if 12 in run["dwords"] else "ATOMIC"
        built = {e for (i, k, e) in builds if k == kind and since[kind] < i < run["start"]}
        since[kind] = run["start"]
        out.append((kind, run["exec"], built))
    return out


def _next_non_wait(insts, i):
    """The instruction after insts[i], past waits and the backend's MSB/ALU hints."""
    for ln in insts[i + 1:]:
        if not ln.startswith(("s_wait_", "s_set_vgpr_msb", "s_delay_alu", "s_nop")):
            return ln
    return ""


@pytest.fixture(scope="module")
def results():
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    return results


def test_fused_a2a_gfx1250_emits_assembly(results):
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert_assembles(src, base)
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        missing = [m for m in _A2A_MARKERS + _GFX12_PACKET_CONSTANTS if m not in src]
        assert not missing, f"Kernel {base!r}: missing {missing}"


def test_fused_a2a_gfx1250_ring_polls_carry_scope(results):
    for base, src, _err in results:
        for what, scope in _POLL_SCOPES.items():
            polls = [ln for ln in _instructions(src)
                     if ln.startswith("s_load") and what in ln]
            assert polls, f"Kernel {base!r}: no s_load for {what!r}"
            wrong = [ln for ln in polls if "scope:" + scope not in ln]
            assert not wrong, f"Kernel {base!r}: {what!r} without {scope}: {wrong[:2]}"


def test_fused_a2a_gfx1250_push_d_stores_are_device_scope(results):
    for base, src, _err in results:
        stores = [ln for ln in _instructions(src)
                  if re.match(r"buffer_store_\S+ .*sgprSrdD", ln)]
        dev = sum("scope:SCOPE_DEV" in ln for ln in stores)
        cu = sum("scope:SCOPE_CU" in ln for ln in stores)
        assert dev > 0 and dev == cu and dev + cu == len(stores), (
            f"Kernel {base!r}: {dev} PUSH (DEV) vs {cu} local (CU) D stores "
            f"of {len(stores)}")


def test_fused_a2a_gfx1250_handshake_keeps_its_barrier(results):
    for base, src, _err in results:
        insts = _instructions(src)
        waits = [i for i, ln in enumerate(insts) if _HANDSHAKE_WAIT in ln]
        assert waits, f"Kernel {base!r}: no handshake store wait"
        missing = [insts[i] for i in waits
                   if not _next_non_wait(insts, i).startswith("s_barrier_signal")]
        assert not missing, (
            f"Kernel {base!r}: {len(missing)}/{len(waits)} handshake waits "
            f"without a barrier: {missing[:2]}")


def test_fused_a2a_gfx1250_packets_one_per_lane(results):
    for base, src, _err in results:
        runs = _packet_runs(_instructions(src))
        assert any(kind == "COPY" for kind, _s, _b in runs), f"Kernel {base!r}: no COPY packet"
        assert any(kind == "ATOMIC" for kind, _s, _b in runs), f"Kernel {base!r}: no ATOMIC packet"
        wrong = []
        second = False
        for i, (kind, stores, built) in enumerate(runs):
            second = kind == "ATOMIC" and not second and i > 0 and runs[i - 1][0] == "ATOMIC"
            want = {"2"} if second else {"1"}
            if stores != want or not built or built != want:
                wrong.append((i, kind, sorted(stores), sorted(built)))
        assert not wrong, (
            f"Kernel {base!r}: {len(wrong)}/{len(runs)} packets off their lane "
            f"(index, kind, store EXEC, build EXEC): {wrong[:3]}")


def test_fused_a2a_gfx1250_golden(results, snapshot):
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
