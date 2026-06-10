# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the C++ (nanobind) subtile InstructionScheduler.

The subtile instruction-scheduling slot-placement algorithm is C++-only: the
public ``Tensile.Components.Subtile.InstructionScheduler.instructionSchedule``
is a thin adapter that classifies the live rocisa emitted-module objects into
the data-only C++ model, runs the compiled ``tensile_writer.subtile.\
instruction_scheduler`` algorithm, and rebuilds the rocisa ``Module`` in the
returned order (with the waitcnt vmcnt post-pass applied). There is no
pure-Python twin and no opt-in flag.

These tests pin the final emission *order* and *vmcnt* of the C++ path against
golden signatures built from real rocisa instructions (MFMA, local read
(ds_read), global read (buffer_load), waitcnt, common / m0-update, and a
generic label that classifies as ``Other``). They run only when both ``rocisa``
and the ``tensile_writer`` extension are importable; otherwise they skip.
"""

import pytest

# Both the ISA layer (rocisa) and the scheduler extension must be present.
pytest.importorskip("rocisa")
cppsched = pytest.importorskip("tensile_writer.subtile.instruction_scheduler")

from rocisa.code import Label
from rocisa.container import RegisterContainer, sgpr, vgpr
from rocisa.enum import InstType
from rocisa.instruction import (
    CommonInstruction,
    GlobalReadInstruction,
    LocalReadInstruction,
    MFMAInstruction,
    SWaitCnt,
)

from Tensile.Components.Subtile.InstructionScheduler import instructionSchedule


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
# so the emitted order is identifiable.
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
# instruction objects (no shared state).
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


def scenario_generic_other():
    """A non-MFMA path carrying a generic rocisa Label, which the classifier
    treats as ``Other`` and the scheduler places generically (mirroring how the
    original Python algorithm handled instructions matching none of its
    isinstance predicates)."""
    return [
        FakeEmittedModule(_mid(0), "mfma", [mfma("mfma0"), mfma("mfma1")]),
        FakeEmittedModule(_mid(1), "lr", [ds_read("lrA"), Label("lbl", "")]),
        FakeEmittedModule(_mid(2), "gr", [buffer_load("gr0")]),
    ]


ALL_SCENARIOS = {
    "rich_multi": scenario_rich_multi,
    "single_mfma": scenario_single_mfma,
    "ds_read_wait_gap": scenario_ds_read_wait_gap,
    "chained_path": scenario_chained_path,
    "two_mfma_minimal": scenario_two_mfma_minimal,
    "generic_other": scenario_generic_other,
}


# Golden final emission signatures (comment, vlcnt) through the C++ scheduler.
# These pin the slot-placement order plus the waitcnt vmcnt post-pass for the
# data model the scheduler is responsible for. Regenerate deliberately (and
# review the diff) only when the slot-placement algorithm intentionally changes.
GOLDEN_SIGNATURES = {
    "rich_multi": [
        ("prewait", -1),
        ("mfma0", None),
        ("lrA", None),
        ("gr0", None),
        ("lrB", None),
        ("gr1", None),
        ("mfma1", None),
        ("gr2", None),
        ("mfma2", None),
        ("lrinc", None),
        ("m0a", None),
        ("wgr", 5),
        ("mfma3", None),
    ],
    "single_mfma": [
        ("prewait", 1),
        ("mfma0", None),
        ("lrA", None),
        ("lrB", None),
        ("gr0", None),
        ("gr1", None),
    ],
    "ds_read_wait_gap": [
        ("mfma0", None),
        ("lr0", None),
        ("w0", 0),
        ("g0", None),
        ("mfma1", None),
        ("lr1", None),
        ("lr2", None),
        ("g1", None),
        ("mfma2", None),
        ("lr3", None),
        ("g2", None),
        ("mfma3", None),
        ("g3", None),
        ("mfma4", None),
        ("mfma5", None),
    ],
    "chained_path": [
        ("mfma0", None),
        ("lrA", None),
        ("m0a", None),
        ("gr0", None),
        ("mfma1", None),
        ("gr1", None),
        ("inc", None),
        ("mfma2", None),
    ],
    "two_mfma_minimal": [
        ("mfma0", None),
        ("lrA", None),
        ("gr0", None),
        ("mfma1", None),
    ],
    # The rocisa Label carries an empty comment, so it surfaces as ('', None);
    # it is placed generically (Other) like any non-rule instruction.
    "generic_other": [
        ("mfma0", None),
        ("lrA", None),
        ("gr0", None),
        ("", None),
        ("mfma1", None),
    ],
}


# ---------------------------------------------------------------------------
# Order + vmcnt: the C++ scheduler matches the pinned golden signatures.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_SCENARIOS))
def test_cpp_schedule_matches_golden(name):
    got = _signature(instructionSchedule(ALL_SCENARIOS[name]()))
    assert got == GOLDEN_SIGNATURES[name], (
        f"{name}: C++ schedule order/vmcnt drifted from golden.\n"
        f"  golden: {GOLDEN_SIGNATURES[name]}\n"
        f"  got:    {got}"
    )


@pytest.mark.parametrize("name", list(ALL_SCENARIOS))
def test_cpp_order_is_a_permutation_of_input(name):
    """Sanity: every input instruction is emitted exactly once."""
    mods = ALL_SCENARIOS[name]()
    out = instructionSchedule(mods)
    emitted = sorted(i.comment for i in out.flatitems())
    expected = sorted(i.comment for m in mods for i in m.instructions)
    assert emitted == expected


@pytest.mark.parametrize("name", list(ALL_SCENARIOS))
def test_mfma_order_preserved(name):
    """MFMA instructions appear in their original relative order."""
    mods = ALL_SCENARIOS[name]()
    mfma_in = [i.comment for m in mods for i in m.instructions
               if isinstance(i, MFMAInstruction)]
    out = instructionSchedule(mods)
    mfma_out = [i.comment for i in out.flatitems()
                if isinstance(i, MFMAInstruction)]
    assert mfma_out == mfma_in


def test_empty_chain_returns_empty_module():
    assert _signature(instructionSchedule([])) == []


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
# Classification: rocisa objects map to the kinds the slot rules key on, and
# anything else is placed generically (Other).
# ---------------------------------------------------------------------------
class TestClassification:
    def test_known_kinds(self):
        K = cppsched.InstKind
        assert cppsched.classifyInstruction(mfma("m")).kind == K.Mfma
        assert cppsched.classifyInstruction(ds_read("l")).kind == K.LocalRead
        assert cppsched.classifyInstruction(buffer_load("g")).kind == K.GlobalRead
        assert cppsched.classifyInstruction(m0_update("m0")).kind == K.M0Update

    def test_waitcnt_carries_vmcnt_fields(self):
        K = cppsched.InstKind
        inst = cppsched.classifyInstruction(waitcnt("w", vlcnt=7))
        assert inst.kind == K.WaitCnt
        assert inst.vlcnt == 7
        assert inst.adjustVmcnt is True

    def test_unknown_is_other(self):
        """A generic instruction / label classifies as Other (no raise),
        matching the original Python algorithm's generic handling."""
        K = cppsched.InstKind
        assert cppsched.classifyInstruction(common("c")).kind == K.Other
        assert cppsched.classifyInstruction(Label("lbl", "")).kind == K.Other
