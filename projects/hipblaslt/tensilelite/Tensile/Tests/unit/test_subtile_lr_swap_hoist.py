# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from rocisa.code import Module
from rocisa.container import DSModifiers, vgpr
from rocisa.instruction import (
    DSLoadB32, SBarrier, SWaitCnt, SWaitTensorcnt, VXorB32,
)

from Tensile.Components.Subtile.WaitAluInsertion import hoistLRSwapAboveWaitTDM

TDM = {"enableTDMA": True, "enableTDMB": True}


def _loop(drained):
  m = Module("loop")
  m.add(DSLoadB32(dst=vgpr(40), src=vgpr(8), ds=DSModifiers(offset=0)))
  if drained:
    m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1))
  m.add(SWaitTensorcnt(tensorcnt=4))
  m.add(SBarrier())
  m.addComment0("Emit code to swap A LR vgpr offsets")
  m.add(VXorB32(dst=vgpr(8), src0=vgpr(8), src1=vgpr(9)))
  m.add(VXorB32(dst=vgpr(10), src0=vgpr(10), src1=vgpr(11)))
  m.add(DSLoadB32(dst=vgpr(41), src=vgpr(8), ds=DSModifiers(offset=0)))
  return m


def _kinds(module):
  return [type(i).__name__ for i in module.flatitems() if type(i).__name__ != "TextBlock"]


def test_swaps_move_before_tdm_wait_when_reads_drained():
  out = hoistLRSwapAboveWaitTDM(_loop(drained=True), None, TDM)
  assert _kinds(out) == ["DSLoadB32", "SWaitCnt", "VXorB32", "VXorB32",
                         "SWaitTensorcnt", "SBarrier", "DSLoadB32"]


def test_swaps_stay_after_barrier_with_read_in_flight():
  loop = _loop(drained=False)
  out = hoistLRSwapAboveWaitTDM(loop, None, TDM)
  assert _kinds(out) == _kinds(loop)


def test_non_tdm_kernel_unchanged():
  loop = _loop(drained=True)
  assert hoistLRSwapAboveWaitTDM(loop, None, {}) is loop
