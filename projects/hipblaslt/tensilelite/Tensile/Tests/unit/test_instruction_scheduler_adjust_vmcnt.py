# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for InstructionScheduler WaitGR adjustVmcnt post-pass.

Plain SWaitCnt(vlcnt=0) (NoSwizzle scale gather drains) must not be rewritten
when adjustVmcnt is absent / False. Only SWaitCntEx with adjustVmcnt=True
(WaitGR) may have vlcnt bumped by prior buffer_loads in the subIterK.

Defaulting getattr(..., True) previously weakened NoSwizzle drains to
vmcnt(N), which returns before the 4 ubyte loads complete.
"""

from types import SimpleNamespace

import pytest
from rocisa.container import MUBUFModifiers, sgpr, vgpr
from rocisa.enum import InstType
from rocisa.instruction import BufferLoadU8, MFMAInstruction, SWaitCnt

from Tensile.Components.Subtile.InstructionEmitter import SWaitCntEx
from Tensile.Components.Subtile.InstructionScheduler import instructionSchedule
from Tensile.Components.Subtile.LogicalScheduler import EmittedModule

pytestmark = pytest.mark.unit


def _mfma(comment):
    return MFMAInstruction(
        InstType.INST_F32, InstType.INST_F32, [16, 16, 4],
        False, vgpr(0, 4), vgpr(4), vgpr(5), 0, comment=comment,
    )


def _buffer_load(dst_idx, vaddr_idx, comment):
    return BufferLoadU8(
        dst=vgpr(dst_idx), vaddr=vgpr(vaddr_idx),
        saddr=sgpr(0, 4), soffset=0,
        mubuf=MUBUFModifiers(offen=True),
        comment=comment,
    )


def _schedule_with_wait(wait_inst):
    """Two MFMAs so instructionSchedule runs the WaitGR post-pass."""
    em_mfma = EmittedModule(
        moduleId=0,
        instructions=[_mfma("mfma0"), _mfma("mfma1")],
        source=SimpleNamespace(kind="mfma"),
    )
    em_gr = EmittedModule(
        moduleId=1,
        instructions=[
            _buffer_load(10, 11, "load0"),
            _buffer_load(12, 13, "load1"),
            wait_inst,
        ],
        before=0,
        source=SimpleNamespace(kind="gr"),
    )
    return instructionSchedule([em_mfma, em_gr])


def _waitcnts(scheduled):
    return [i for i in scheduled.flatitems()
            if isinstance(i, SWaitCnt) and i.vlcnt >= 0]


def test_plain_swaitcnt_vlcnt0_not_rewritten_by_post_pass():
    """Plain SWaitCnt has no adjustVmcnt; post-pass must leave vlcnt=0 alone."""
    wait = SWaitCnt(vlcnt=0, comment="scale gather drain")
    assert getattr(wait, "adjustVmcnt", False) is False

    waits = _waitcnts(_schedule_with_wait(wait))
    assert len(waits) == 1
    assert waits[0].vlcnt == 0


def test_swaitcntex_adjust_true_bumps_vlcnt_by_prior_loads():
    """WaitGR (SWaitCntEx adjustVmcnt=True) accounts for prior buffer_loads."""
    wait = SWaitCntEx(vlcnt=0, adjustVmcnt=True, comment="wait_gr")
    waits = _waitcnts(_schedule_with_wait(wait))
    assert len(waits) == 1
    assert waits[0].vlcnt == 2  # two BufferLoadU8 before the wait


def test_swaitcntex_adjust_false_left_alone():
    """Explicit adjustVmcnt=False must not rewrite vlcnt (default for non-WaitGR)."""
    wait = SWaitCntEx(
        vlcnt=0, adjustVmcnt=False, isWaitGr=False, comment="no adjust",
    )
    waits = _waitcnts(_schedule_with_wait(wait))
    assert len(waits) == 1
    assert waits[0].vlcnt == 0
