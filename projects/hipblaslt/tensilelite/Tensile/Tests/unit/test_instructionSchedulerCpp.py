# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the optional C++ (nanobind) subtile InstructionScheduler.

These tests compare the pure-Python slot-placement algorithm in
``Tensile.Components.Subtile.InstructionScheduler`` against the compiled
``tensile_writer.subtile.instruction_scheduler`` port. They run only when both
``rocisa`` and the ``tensile_writer`` extension are importable; otherwise they
skip, so the default (Python-only) TensileLite build is unaffected.

Synthetic emitted-module chains are built from *real* rocisa instructions
covering MFMA, local read (ds_read), global read (buffer_load), waitcnt, and
common / m0-update instructions. Each instruction carries a unique ``comment``
so the emitted ordering can be compared exactly between the two code paths,
along with the waitcnt vmcnt post-pass result.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import pytest

# Both the ISA layer (rocisa) and the scheduler extension must be present.
pytest.importorskip("rocisa")
cppsched = pytest.importorskip("tensile_writer.subtile.instruction_scheduler")

from rocisa.code import Label, Module
from rocisa.container import RegisterContainer, sgpr, vgpr
from rocisa.enum import InstType
from rocisa.instruction import (
    CommonInstruction,
    GlobalReadInstruction,
    LocalReadInstruction,
    MFMAInstruction,
    SWaitCnt,
)

from Tensile.Components.Subtile import InstructionScheduler as isched
from Tensile.Components.Subtile.InstructionScheduler import (
    _instructionSchedulePython,
    instructionSchedule,
)


# ---------------------------------------------------------------------------
# Synthetic emitted-module model
# ---------------------------------------------------------------------------
class FakeEmittedModule:
    """Minimal stand-in for LogicalScheduler.EmittedModule.

    ``instructionSchedule`` and the C++ shim only read ``moduleId``,
    ``opType``, ``before`` and ``instructions``.
    """

    def __init__(self, moduleId, opType, instructions, before=None):
        self.moduleId = moduleId
        self.opType = opType
        self.instructions = instructions
        self.before = before


# Instruction factories. Each takes a unique ``tag`` used as the rocisa comment
# so the emitted order is identifiable across the two independent builds.
def mfma(tag):
    return MFMAInstruction(InstType.INST_BF16, InstType.INST_F32, [16, 16, 16, 1],
                           False, vgpr("acc", 4), vgpr("a", 2), vgpr("b", 2),
                           comment=tag)


def ds_read(tag):
    return LocalReadInstruction(InstType.INST_B32, vgpr("d", 1), vgpr("s", 1),
                                comment=tag)


def buffer_load(tag):
    return GlobalReadInstruction(InstType.INST_B32, vgpr("d", 1), comment=tag)


def waitcnt(tag, vlcnt=-1):
    return SWaitCnt(vlcnt=vlcnt, comment=tag)


def common(tag):
    return CommonInstruction(InstType.INST_B32, vgpr("d", 1), [vgpr("s", 1)],
                             comment=tag)


def m0_update(tag):
    m0reg = RegisterContainer("m", None, 0, 1.0)
    return CommonInstruction(InstType.INST_B32, m0reg, [sgpr("s", 1)], comment=tag)


def _signature(module):
    """(comment, vlcnt-or-None) for each instruction in emission order."""
    out = []
    for inst in module.flatitems():
        vl = getattr(inst, "vlcnt", None)
        out.append((inst.comment, vl))
    return out


# ---------------------------------------------------------------------------
# Scenario builders. Each returns a fresh list of FakeEmittedModule with fresh
# instruction objects (no shared state), so the two code paths never alias.
#
# moduleId is deliberately offset from the list index (10 * index) so the
# C++ idToIdx / before-link resolution is exercised, not just positional ids.
# ---------------------------------------------------------------------------
def _mid(i):
    return 10 * i


def scenario_rich_multi():
    """4 MFMAs (3 intervals), preMFMA wait, LR path, GR-spread path, a
    wait_gr path packed in reverse, and an m0 path."""
    return [
        FakeEmittedModule(_mid(0), "mfma",
                          [mfma("mfma0"), mfma("mfma1"), mfma("mfma2"), mfma("mfma3")],
                          before=_mid(1)),
        FakeEmittedModule(_mid(1), "wait_lr", [waitcnt("prewait")]),
        FakeEmittedModule(_mid(2), "lr", [ds_read("lrA"), ds_read("lrB")]),
        FakeEmittedModule(_mid(3), "gr",
                          [buffer_load("gr0"), buffer_load("gr1"), buffer_load("gr2")]),
        FakeEmittedModule(_mid(4), "wait_gr", [waitcnt("wgr", vlcnt=2)], before=_mid(5)),
        FakeEmittedModule(_mid(5), "lr_inc", [common("lrinc")]),
        FakeEmittedModule(_mid(6), "m0", [m0_update("m0a")]),
    ]


def scenario_single_mfma():
    """One MFMA: the no-interleave branch (preMFMA, MFMA, then paths)."""
    return [
        FakeEmittedModule(_mid(0), "mfma", [mfma("mfma0")], before=_mid(1)),
        FakeEmittedModule(_mid(1), "wait_lr", [waitcnt("prewait", vlcnt=1)]),
        FakeEmittedModule(_mid(2), "lr", [ds_read("lrA"), ds_read("lrB")]),
        FakeEmittedModule(_mid(3), "gr", [buffer_load("gr0"), buffer_load("gr1")]),
    ]


def scenario_ds_read_wait_gap():
    """ds_read + waitcnt mix over many intervals to exercise the min-gap rules
    and one ds_read per interval."""
    return [
        FakeEmittedModule(_mid(0), "mfma",
                          [mfma(f"mfma{i}") for i in range(6)]),
        FakeEmittedModule(_mid(1), "lr",
                          [ds_read("lr0"), ds_read("lr1"), ds_read("lr2"), ds_read("lr3")]),
        FakeEmittedModule(_mid(2), "wait", [waitcnt("w0", vlcnt=0)]),
        FakeEmittedModule(_mid(3), "gr",
                          [buffer_load("g0"), buffer_load("g1"), buffer_load("g2"),
                           buffer_load("g3")]),
    ]


def scenario_chained_path():
    """A multi-module before-link chain plus an m0/buffer-load interplay."""
    return [
        FakeEmittedModule(_mid(0), "mfma",
                          [mfma("mfma0"), mfma("mfma1"), mfma("mfma2")]),
        # chain: 1 -> 2 -> 3 (pred links via `before`)
        FakeEmittedModule(_mid(1), "lr", [ds_read("lrA")]),
        FakeEmittedModule(_mid(2), "gr", [buffer_load("gr0"), buffer_load("gr1")],
                          before=_mid(1)),
        FakeEmittedModule(_mid(3), "lr_inc", [common("inc")], before=_mid(2)),
        FakeEmittedModule(_mid(4), "m0", [m0_update("m0a")]),
    ]


def scenario_two_mfma_minimal():
    """Smallest interleaving case: 2 MFMAs, 1 interval (2 slots)."""
    return [
        FakeEmittedModule(_mid(0), "mfma", [mfma("mfma0"), mfma("mfma1")]),
        FakeEmittedModule(_mid(1), "lr", [ds_read("lrA")]),
        FakeEmittedModule(_mid(2), "gr", [buffer_load("gr0")]),
    ]


ALL_SCENARIOS = {
    "rich_multi": scenario_rich_multi,
    "single_mfma": scenario_single_mfma,
    "ds_read_wait_gap": scenario_ds_read_wait_gap,
    "chained_path": scenario_chained_path,
    "two_mfma_minimal": scenario_two_mfma_minimal,
}


# ---------------------------------------------------------------------------
# Parity: C++ shim ordering + vmcnt post-pass match the pure-Python algorithm.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_SCENARIOS))
def test_cpp_shim_matches_python(name):
    build = ALL_SCENARIOS[name]
    py = _instructionSchedulePython(build())
    cpp = cppsched.instructionSchedule(build())
    assert _signature(cpp) == _signature(py), (
        f"{name}: C++ scheduler order/vmcnt differs from Python.\n"
        f"  python: {_signature(py)}\n"
        f"  cpp:    {_signature(cpp)}"
    )


@pytest.mark.parametrize("name", list(ALL_SCENARIOS))
def test_cpp_order_is_a_permutation_of_input(name):
    """Sanity: every input instruction is emitted exactly once."""
    mods = ALL_SCENARIOS[name]()
    cpp = cppsched.instructionSchedule(mods)
    emitted = sorted(i.comment for i in cpp.flatitems())
    # MFMA module contributes only its MFMA instructions; everything else is
    # contributed by the non-MFMA modules.
    expected = sorted(i.comment for m in mods for i in m.instructions)
    assert emitted == expected


# ---------------------------------------------------------------------------
# Data-only model: the C++ algorithm runs without any rocisa objects.
# ---------------------------------------------------------------------------
class TestDataOnlyModel:
    def test_schedule_returns_all_instructions(self):
        K = cppsched.InstKind
        I = cppsched.Instruction
        M = cppsched.ModuleRef
        modules = [
            M(0, "mfma", None, [I(K.Mfma), I(K.Mfma)]),
            M(1, "wait_lr", None, [I(K.WaitCnt, 3, True)]),
            M(2, "lr", None, [I(K.LocalRead)]),
            M(3, "gr", None, [I(K.GlobalRead)]),
        ]
        res = cppsched.schedule(modules)
        # 2 MFMA + 1 wait + 1 lr + 1 gr = 5 instructions emitted.
        assert len(res.order) == 5
        assert len(res.kinds) == 5
        # With no preMFMA dependency, the MFMAs bracket the single interval.
        assert res.kinds[0] == K.Mfma
        assert res.kinds[-1] == K.Mfma

    def test_vmcnt_post_pass_bumps_waitcnt(self):
        """A waitcnt emitted after N buffer_loads gets vlcnt += N."""
        K = cppsched.InstKind
        I = cppsched.Instruction
        M = cppsched.ModuleRef
        # Single MFMA so the order is deterministic: preMFMA, MFMA, then paths.
        # GR path (2 buffer loads) precedes a trailing waitcnt path.
        modules = [
            M(0, "mfma", None, [I(K.Mfma)]),
            M(1, "gr", None, [I(K.GlobalRead), I(K.GlobalRead)]),
            M(2, "wait", None, [I(K.WaitCnt, 5, True)]),
        ]
        res = cppsched.schedule(modules)
        # Find the waitcnt entry; it must have been bumped by the 2 buffer loads.
        wait_idx = [i for i, k in enumerate(res.kinds) if k == K.WaitCnt]
        assert len(wait_idx) == 1
        assert res.vlcnt[wait_idx[0]] == 5 + 2
        assert res.vmcntAdjustments == [(wait_idx[0], 2)]

    def test_no_adjust_when_flag_false(self):
        K = cppsched.InstKind
        I = cppsched.Instruction
        M = cppsched.ModuleRef
        modules = [
            M(0, "mfma", None, [I(K.Mfma)]),
            M(1, "gr", None, [I(K.GlobalRead)]),
            M(2, "wait", None, [I(K.WaitCnt, 5, False)]),  # adjustVmcnt=False
        ]
        res = cppsched.schedule(modules)
        assert res.vmcntAdjustments == []

    def test_empty_chain(self):
        assert cppsched.schedule([]).order == []

    def test_multiple_mfma_modules_raises(self):
        K = cppsched.InstKind
        I = cppsched.Instruction
        M = cppsched.ModuleRef
        modules = [
            M(0, "mfma", None, [I(K.Mfma)]),
            M(1, "mfma", None, [I(K.Mfma)]),
        ]
        with pytest.raises((ValueError, RuntimeError)):
            cppsched.schedule(modules)


# ---------------------------------------------------------------------------
# Delegation wiring + clean fallback.
# ---------------------------------------------------------------------------
def test_default_path_is_python_only():
    """With the env flag unset, delegation must be disabled by default."""
    import os
    if os.environ.get("TENSILE_WRITER_CPP", "").strip().lower() not in (
            "", "0", "false", "no", "off"):
        pytest.skip("TENSILE_WRITER_CPP is set; default-off behavior not under test")
    assert isched._USE_CPP is False
    assert isched._CPP is None


def test_unclassifiable_instruction_falls_back(monkeypatch):
    """The public instructionSchedule must fall back to Python when the C++
    path cannot classify a live instruction (here: a rocisa Label)."""
    monkeypatch.setattr(isched, "_CPP", cppsched)
    monkeypatch.setattr(isched, "_USE_CPP", True)

    def build():
        return [
            FakeEmittedModule(_mid(0), "mfma", [mfma("mfma0"), mfma("mfma1")]),
            FakeEmittedModule(_mid(1), "lr", [ds_read("lrA"), Label("lbl", "")]),
            FakeEmittedModule(_mid(2), "gr", [buffer_load("gr0")]),
        ]

    # Public wrapper (delegation enabled) must still produce the Python result.
    got = instructionSchedule(build())
    want = _instructionSchedulePython(build())
    got_tags = [getattr(i, "comment", None) for i in got.flatitems()]
    want_tags = [getattr(i, "comment", None) for i in want.flatitems()]
    assert got_tags == want_tags


@pytest.mark.parametrize("name", list(ALL_SCENARIOS))
def test_delegation_enabled_matches_python(monkeypatch, name):
    """With delegation explicitly enabled, the public ``instructionSchedule``
    routes through C++ and still matches the pure-Python result."""
    monkeypatch.setattr(isched, "_CPP", cppsched)
    monkeypatch.setattr(isched, "_USE_CPP", True)
    build = ALL_SCENARIOS[name]
    got = instructionSchedule(build())
    want = _instructionSchedulePython(build())
    assert _signature(got) == _signature(want)
