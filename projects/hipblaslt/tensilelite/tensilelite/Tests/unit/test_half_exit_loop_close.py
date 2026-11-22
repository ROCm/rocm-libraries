# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Loop closing must use per-workgroup trip proofs, not global K alone.

Each test fails again if its listed loop shape re-enters the ASEM-only gate.
The retained recipes use registry-valid ASEM values and GSU=1 for plain GEMM.
"""

from pathlib import Path

import pytest

_DATA = Path(__file__).resolve().parent / "test_data"
# gfx950 MT32x64x128 solution, StreamK=3, UnrollLoopSwapGlobalReadOrder=1 (two
# main loop copies with ExpandPointerSwap=0).
_LOGIC = _DATA / "half_exit_gfx950_mt32x64.yaml"
# gfx950 MT256x256x64 CMS solution, DataParallel (StreamK=3 + StreamKForceDPOnly=1).
_LOGIC_CMS = _DATA / "half_exit_gfx950_cms_mt256x256.yaml"

from loop_exit_test_helpers import (emit as _emit, dec_blocks as _dec_blocks,
                                    assemble as _assemble, pristine_default_solution,
                                    instructions)

pytestmark = pytest.mark.unit


def test_cms_asem_multiple_of_2du_loop_close(tmp_path):
    """CMS kernels build their own single-copy loop; they must keep the plain
    dec+cmp loop-counter code (it used to fail CMS validation with "LCC has 2
    instructions, but 4 instructions are required"). This checks the always-1
    decrement, not the CMS clause of the guard: every CMS schedule uses PGR2,
    so the no-exit case is never reached with or without that clause."""
    src = _emit(tmp_path, "cms128", _LOGIC_CMS, {"AssertSummationElementMultiple": 128})
    decs = _dec_blocks(src)
    assert decs and all(d == "1" for d, _ in decs)
    assert all(n.startswith("s_cmp_eq_i32 s[sgprLoopCounterL], 0x2") for _, n in decs)
    _assemble(tmp_path, "cms128", src)


def test_custom_schedule_clause_checks_every_two_copy_exit(tmp_path, monkeypatch):
    """Feature/defensive guard: removing only the CMS conjunct drops an exit.

    All existing CMS recipes are persistent, so their false even-trip proof
    masks this independent safeguard. Derive a valid, plain PGR1 EPS recipe;
    present the custom-schedule marker only at the real closer's boundary.
    No invalid CMS/PGR1 YAML is accepted or used as a numerical recipe.
    """
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    close = KernelWriterAssembly.closeLoop

    def custom_close(writer, kernel, *args, **kwargs):
        close_kernel = dict(kernel)
        close_kernel["UseCustomMainLoopSchedule"] = 1
        return close(writer, close_kernel, *args, **kwargs)

    monkeypatch.setattr(KernelWriterAssembly, "closeLoop", custom_close)
    source = _emit(tmp_path, "custom_close_guard", _LOGIC,
                   {"StreamK": 0, "GlobalSplitU": 1, "PrefetchGlobalRead": 1,
                    "AssertSummationElementMultiple": 256, "ExpandPointerSwap": 1,
                    "UnrollLoopSwapGlobalReadOrder": 0})
    decs = _dec_blocks(source)
    assert len(decs) == 2 and all(operand == "1" for operand, _ in decs)
    assert all(next_instruction.startswith("s_cmp_") for _, next_instruction in decs)
    _assemble(tmp_path, "custom_close_guard", source)


@pytest.mark.parametrize("policy", [{"StreamK": 0, "GlobalSplitU": 1}], ids=["fixed-gsu1"])
@pytest.mark.parametrize("pgr", [1, 2])
def test_two_copy_asem_multiple_of_2du_half_exit(tmp_path, pgr, policy):
    """Two valid alternating copies decrement exactly one DepthU each.
    Restore the undefined decValue or PGR2 decrement-by-two to fail again.
    """
    shape = {"ExpandPointerSwap": pgr == 1, "UnrollLoopSwapGlobalReadOrder": int(pgr == 2)}
    src = _emit(tmp_path, f"two{pgr}", _LOGIC,
                {"PrefetchGlobalRead": pgr, "AssertSummationElementMultiple": 256, **shape, **policy})
    decs = _dec_blocks(src)
    assert len(decs) == 2 and all(d == "1" for d, _ in decs)
    if pgr == 1:
        assert decs[0][1].startswith("s_cmp_eq_i32 s[sgprLoopCounterL], 0x1")
        assert decs[1][1].startswith("s_branch label_LoopBeginL")
    else:
        assert decs[0][1].startswith("s_cmp_le_u32 s[sgprLoopCounterL], 0x2")
        assert decs[1][1].startswith("s_cmp_eq_i32 s[sgprLoopCounterL], 0x2")
    _assemble(tmp_path, f"two{pgr}", src)


@pytest.mark.parametrize("pgr", [1, 2])
def test_single_copy_asem_multiple_of_2du_keeps_every_exit(tmp_path, pgr):
    """One copy (ExpandPointerSwap=0, no forced second loop): every iteration
    decrements by 1 and runs the exit test. PGR1 used to raise NameError
    'decValue'; PGR2 decremented by 2 when StaggerU == 0 (half of K ran)."""
    src = _emit(tmp_path, f"one{pgr}", _LOGIC,
                {"PrefetchGlobalRead": pgr, "AssertSummationElementMultiple": 256,
                 "StreamK": 0, "GlobalSplitU": 1, "UnrollLoopSwapGlobalReadOrder": 0})
    decs = _dec_blocks(src)
    assert len(decs) == 1 and decs[0][0] == "1"
    assert decs[0][1].startswith("s_cmp_eq_i32 s[sgprLoopCounterL], 0x%x" % pgr)
    _assemble(tmp_path, f"one{pgr}", src)


@pytest.mark.parametrize("pgr", [1, 2])
def test_streamk_two_copy_asem_multiple_of_2du_keeps_every_exit(tmp_path, pgr):
    """Stream-K gives each workgroup an arbitrary range of DepthU iterations,
    so ASEM % (2*DepthU) == 0 does not make its trip count even: both copies
    keep the exit test and decrement by 1. PGR1 used to raise NameError
    'decValue'; PGR2 decremented copy 1 by a StaggerU-selected 2/1.

    Only PGR1 depends on the Stream-K clause of the guard. At PGR2 copy 0
    exits through the forced-second-loop "<= PGR" test and copy 1 keeps its
    exit anyway, so PGR2 only checks that the 2/1 decrement is gone."""
    shape = {"ExpandPointerSwap": pgr == 1, "UnrollLoopSwapGlobalReadOrder": int(pgr == 2)}
    src = _emit(tmp_path, f"sk{pgr}", _LOGIC,
                {"PrefetchGlobalRead": pgr, "AssertSummationElementMultiple": 256, **shape})
    assert "s[sgprStreamKLocalEnd]" in src
    decs = _dec_blocks(src)
    assert len(decs) == 2 and all(d == "1" for d, _ in decs)
    assert all(n.startswith("s_cmp_") for _, n in decs)
    _assemble(tmp_path, f"sk{pgr}", src)


@pytest.mark.parametrize("pgr", [1])
def test_eps_two_copy_asem_multiple_of_2du_half_exit(tmp_path, pgr):
    """PGR1 pointer-swap copies retain one exit and decrement once per copy.
    Restoring the undefined decValue fails this fixed-GSU1 build.
    """
    src = _emit(tmp_path, f"eps{pgr}", _LOGIC,
                {"PrefetchGlobalRead": pgr, "AssertSummationElementMultiple": 256,
                 "StreamK": 0, "GlobalSplitU": 1, "ExpandPointerSwap": 1, "UnrollLoopSwapGlobalReadOrder": 0})
    decs = _dec_blocks(src)
    assert len(decs) == 2 and all(d == "1" for d, _ in decs)
    assert decs[0][1].startswith("s_cmp_eq_i32 s[sgprLoopCounterL], 0x1")
    assert any(i.startswith("s_cbranch_scc1 label_LoopEndL_oddexit") for i in instructions(src))
    assert decs[1][1].startswith("s_branch label_LoopBeginL")
    _assemble(tmp_path, f"eps{pgr}", src)


@pytest.mark.parametrize("gsu,runtime", [(1, True), (2, False), (-1, False)], ids=["runtime-gsu1", "gsu2", "auto-gsu"])
@pytest.mark.parametrize("pgr,shape", [
    (1, {"ExpandPointerSwap": 1, "UnrollLoopSwapGlobalReadOrder": 0}),
    (2, {"ExpandPointerSwap": 0, "UnrollLoopSwapGlobalReadOrder": 1}),
], ids=["eps-1", "swap-2"])
def test_gsu_two_copy_asem_multiple_of_2du_keeps_every_exit(tmp_path, pgr, shape, gsu, runtime):
    """Runtime/auto GSU and fixed GSU>1 must check both loop copies.
    Reverting only closeLoop to an ASEM-only half-exit breaks this test.
    """
    src = _emit(tmp_path, f"gsu{pgr}", _LOGIC,
                {"PrefetchGlobalRead": pgr, "AssertSummationElementMultiple": 256,
                 "StreamK": 0, "GlobalSplitU": gsu,
                 "InternalSupportParams": {"SupportUserGSU": runtime, "SupportCustomWGM": True,
                                           "SupportCustomStaggerU": False, "UseUniversalArgs": True,
                                           "UseSFC": False}, **shape})
    assert "GSUSumIdx+1" in src
    decs = _dec_blocks(src)
    assert len(decs) == 2 and all(d == "1" for d, _ in decs)
    assert all(n.startswith("s_cmp_") for _, n in decs)
    _assemble(tmp_path, f"gsu{pgr}", src)


@pytest.mark.parametrize("streamk", [3, 4, 5], ids=["sk3", "sk4", "sk5"])
@pytest.mark.parametrize("pgr", [1, 2], ids=["pgr1", "pgr2"])
def test_close_loop_streamk_truth_table(tmp_path, streamk, pgr):
    """A fix at entrance alone cannot protect an arbitrary SK split at close."""
    source = _emit(tmp_path, "sk_close", _LOGIC,
                   {"StreamK": streamk, "PrefetchGlobalRead": pgr,
                    "AssertSummationElementMultiple": 256,
                    "ExpandPointerSwap": pgr == 1,
                    "UnrollLoopSwapGlobalReadOrder": int(pgr == 2)})
    decs = _dec_blocks(source)
    assert len(decs) == 2 and all(operand == "1" for operand, _ in decs)
    assert all(next_instruction.startswith("s_cmp_") for _, next_instruction in decs)


@pytest.mark.parametrize("pgr", [1, 2], ids=["pgr1", "pgr2"])
@pytest.mark.parametrize("copies", [1, 2], ids=["one-copy", "two-copies"])
def test_close_loop_without_even_proof_checks_every_copy(tmp_path, pgr, copies):
    """Preservation: removing closeLoop's ordinary comparison loses the exit."""
    source = _emit(tmp_path, "ordinary_close", _LOGIC,
                   {"StreamK": 0, "GlobalSplitU": 1, "PrefetchGlobalRead": pgr,
                    "AssertSummationElementMultiple": 128,
                    "ExpandPointerSwap": copies == 2 and pgr == 1,
                    "UnrollLoopSwapGlobalReadOrder": int(copies == 2 and pgr == 2)})
    decs = _dec_blocks(source)
    assert len(decs) == copies and all(operand == "1" for operand, _ in decs)
    assert all(next_instruction.startswith("s_cmp_") for _, next_instruction in decs)
