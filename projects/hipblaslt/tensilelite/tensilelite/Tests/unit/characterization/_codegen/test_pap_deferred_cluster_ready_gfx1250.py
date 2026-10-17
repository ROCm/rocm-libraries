# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native ordering for the complete deferred multicast publication unit."""

import re

import pytest
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from test_pap_pipeline_overlap_gfx1250 import _config
from Tensile.Components.PersistentLoop import PersistentKernelState

pytestmark = pytest.mark.unit


def _resources(src):
    return tuple(max(map(int, re.findall(r"\.amdhsa_next_free_" + kind + r"\s+(\d+)", src)))
                 for kind in ("sgpr", "vgpr"))


def _entry_trace(lines, start, stop, state, q, wave):
    """Trace the real entry branches; this does not model GPU completion."""
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(":")}
    state_reg = "s[sgprPersistentPrefetchState]"
    values = {state_reg: state, "s[sgprLoopCounterL]": q}
    trace, pc, scc = [], start, None
    for _ in range(stop - start + 10):
        if pc == stop:
            return trace, values[state_reg]
        assert start <= pc < stop
        trace.append(pc)
        code = lines[pc].split("//", 1)[0].strip()
        pc += 1
        if not code or code.endswith(":"):
            continue
        op, _, operands = code.partition(" ")
        args = [arg.strip() for arg in operands.split(",")]
        if op == "s_branch":
            pc = labels[args[0]]
        elif op in ("s_cbranch_scc0", "s_cbranch_scc1"):
            assert scc is not None
            if scc == (op == "s_cbranch_scc1"):
                pc = labels[args[0]]
        elif op == "s_mov_b32":
            assert args[0] in values, f"unmodeled destination: {code}"
            values[args[0]] = values[args[1]] if args[1] in values else int(args[1], 0)
        elif op == "s_bitcmp1_b32":
            assert args[0] == state_reg
            scc = bool(values[state_reg] & (1 << int(args[1], 0)))
        elif op in ("s_and_b32", "s_or_b32"):
            assert args[0] == args[1] == state_reg
            immediate = int(args[2], 0)
            state = values[state_reg]
            values[state_reg] = state & immediate if op == "s_and_b32" else state | immediate
            scc = values[state_reg] != 0
        elif op == "v_readfirstlane_b32":
            assert re.fullmatch(r"v\[vgprSerial(?:-\d+)?\]", args[1])
            values[args[0]] = wave * 32
        elif op in ("s_cmp_eq_u32", "s_cmp_ge_u32"):
            assert args[0] in values, f"unmodeled comparison: {code}"
            value = values[args[0]]
            immediate = int(args[1], 0)
            scc = value == immediate if op == "s_cmp_eq_u32" else value >= immediate
        elif op == "s_xor_b32":
            # These LDS-bank updates change SCC, but not any traced value.
            assert args[0] == args[1] and re.fullmatch(r"s\[sgprtdm\w+Group0\+1\]", args[0]), code
            scc = None
        elif op == "ds_load_b128":
            assert args[0].startswith("v[vgprValu"), code  # cannot overwrite election input
        else:
            # Completion and VGPR encoding are checked outside this control-flow model.
            assert op in {"s_nop", "s_set_vgpr_msb", "s_wait_alu", "s_wait_dscnt",
                          "s_wait_tensorcnt", "s_barrier_signal", "s_barrier_wait",
                          "tensor_load_to_lds"}, f"unmodeled entry instruction: {code}"
    raise AssertionError("entry did not reach openLoop")


def test_entry_trace_observes_pending_state_clear():
    lines = ["s_mov_b32 s[sgprPersistentPrefetchState], 0",
             "s_bitcmp1_b32 s[sgprPersistentPrefetchState], 7",
             "s_cbranch_scc0 done", "s_barrier_wait -3", "done:"]
    trace, state = _entry_trace(lines, 0, 4, state=128, q=4, wave=0)
    assert trace == [0, 1, 2] and state == 0


@pytest.mark.parametrize("instruction", [
    "s_add_u32 s[sgprPersistentPrefetchState], s[sgprPersistentPrefetchState], 1",
    "s_cbranch_execz done",
    "v_mov_b32 v[vgprSerial], 0",
    "s_xor_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1",
])
def test_entry_trace_rejects_unmodeled_control_effects(instruction):
    with pytest.raises(AssertionError):
        _entry_trace([instruction, "done:"], 0, 1, state=128, q=4, wave=0)


@pytest.mark.parametrize("cluster", [[2, 4], [2, 1]])
def test_complete_ready1_moves_after_carried_plr_before_next_phase(tmp_path, monkeypatch, cluster):
    config = _config(tmp_path, AssertSummationElementMultiple=256, ClusterDim=cluster)
    candidate = emit_kernels_from_config(config, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(candidate)
    with monkeypatch.context() as baseline:
        baseline.setattr(PersistentKernelState, "papPgr2DeferredClusterReadyEnabled", lambda *args: False)
        controls = emit_kernels_from_config(config, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(controls)
    for (name, src, _), (_, control, _) in zip(candidate, controls):
        assert_assembles(src, name)
        (tmp_path / "deferred_ready1.s").write_text(src)
        (tmp_path / "immediate_ready1.s").write_text(control)
        assert _resources(src) == _resources(control)
        lines = src.splitlines()
        primed = next(i for i, line in enumerate(lines) if line.startswith("label_PapGroup1ReadyOrPending:"))
        carried = next(i for i, line in enumerate(lines) if "PAP complete all carried first PLR" in line)
        pending = next(i for i, line in enumerate(lines) if "PAP group1 publication pending for this entry?" in line)
        consumed = next(i for i, line in enumerate(lines) if "PAP consume entry-local group1 publication" in line)
        assert primed < carried < pending < consumed
        # Cold PLR has a full wait in linear text. Trace the branches to
        # prove carried entries skip it and publish exactly once on every wave.
        start = next(i for i, line in enumerate(lines) if "PAP second group already issued?" in line)
        stop = next(i for i, line in enumerate(lines) if line.startswith("label_openLoopL:"))
        for state in (0, 1, 3, 9, 11, 25, 27):
            for q in (1, 2, 3, 4, 6):
                for wave in range(4):
                    trace, remaining = _entry_trace(lines, start, stop, state, q, wave)
                    active = bool(state & 8)
                    deferred = active and bool(state & 16) and q > 2
                    assert remaining == 0
                    assert (consumed in trace) == deferred
                    cluster_waits = [i for i in trace if "s_barrier_wait -3" in lines[i]]
                    cluster_arrives = [i for i in trace if "s_barrier_signal -3" in lines[i]]
                    assert len(cluster_waits) == int(active)
                    assert len(cluster_arrives) == int(active and wave == 0)
                    assert all((i > pending) == deferred for i in cluster_waits)
                    if deferred:
                        publication = [i for i in trace if pending < i < consumed]
                        full_waits = [i for i in publication if "s_wait_tensorcnt 0" in lines[i]]
                        local_arrives = [i for i in publication if "s_barrier_signal -1" in lines[i]]
                        local_waits = [i for i in publication if "s_barrier_wait -1" in lines[i]]
                        assert len(full_waits) == len(local_arrives) == len(local_waits) == 1
                        assert full_waits[0] < local_arrives[0] < local_waits[0] < cluster_waits[0]
                        if wave == 0:
                            assert local_waits[0] < cluster_arrives[0] < cluster_waits[0]
                        before = [lines[i] for i in trace if i < pending]
                        assert not any("s_wait_tensorcnt 0" in line for line in before)
                        assert not any("tensor_load_to_lds" in line or "ds_load_" in line for line in before)
        moved = lines[pending:consumed]
        full = next(i for i, line in enumerate(moved) if "s_wait_tensorcnt 0" in line)
        arrive = [i for i, line in enumerate(moved) if "s_barrier_signal -3" in line]
        waits = [i for i, line in enumerate(moved) if "s_barrier_wait -3" in line]
        assert len(arrive) == len(waits) == 1
        assert full < arrive[0] < waits[0]
        assert any("s_barrier_signal -1" in line for line in moved[full:arrive[0]])
        assert any("s_barrier_wait -1" in line for line in moved[full:arrive[0]])
        following_arrive = next(i for i in range(consumed + 1, len(lines)) if "s_barrier_signal -3" in lines[i])
        first_wmma = next(i for i in range(consumed + 1, len(lines)) if "v_wmma_" in lines[i])
        assert consumed < following_arrive and consumed < first_wmma
        merge = next(line for line in lines if "PAP second PGR common merge" in line)
        assert re.search(r", (?:0x90|144)\s+//", merge), merge
        assert src.count("PAP defer entry-local group1 publication") == 1
        assert src.count("PAP consume entry-local group1 publication") == 1
        assert "PAP complete older TDM group on every main-loop trip" in src
        assert "PAP defer group1 only for more than two full K chunks" in src
