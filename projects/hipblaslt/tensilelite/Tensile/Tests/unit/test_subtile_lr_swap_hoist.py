# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from rocisa.code import Module
from rocisa.container import DSModifiers, vgpr
from rocisa.instruction import (
    DSLoadB32, DSLoadB128, MFMAInstruction, SBarrier, SSetVgprMsb, SWaitAlu,
    SWaitCnt, SWaitTensorcnt, VAddU32, VCmpGEU32, VCndMaskB32, VSubU32, VXorB32,
)  # SWaitCnt used by tighten_wait_lr tests
from rocisa.enum import InstType

from Tensile.Components.Subtile.WaitAluInsertion import (
    hoistLRSwapAboveWaitTDM, spreadLRSwapAcrossPriorMfma, spreadPostReadyDsLoads,
    splitWaitLRPerTile,
)

TDM = {"enableTDMA": True, "enableTDMB": True}


def _mfma(dst=vgpr(96, 16), a=vgpr(0, 16), b=vgpr(16, 8)):
  # Minimal stand-in; only the class matters for the spread pass.
  return MFMAInstruction(InstType.INST_F32, InstType.INST_F32, [16, 16, 4],
                         False, dst, a, b)


def _loop(drained, with_scale=False):
  m = Module("loop")
  m.add(DSLoadB32(dst=vgpr(40), src=vgpr(8), ds=DSModifiers(offset=0)))
  if with_scale:
    m.add(DSLoadB32(dst=vgpr(41), src=vgpr(33), ds=DSModifiers(offset=0)))
  if drained:
    m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1))
  m.add(SWaitTensorcnt(tensorcnt=4))
  m.add(SBarrier())
  m.addComment0("Emit code to swap A LR vgpr offsets")
  m.add(VXorB32(dst=vgpr(8), src0=vgpr(8), src1=vgpr(9)))
  m.add(VXorB32(dst=vgpr(10), src0=vgpr(10), src1=vgpr(11)))
  if with_scale:
    m.addComment0("Emit code to swap MXSA LR vgpr offsets")
    m.add(VAddU32(dst=vgpr(33), src0=vgpr(33), src1=hex(8192)))
    m.add(VSubU32(dst=vgpr(62), src0=vgpr(33), src1=hex(24576)))
    m.add(VCmpGEU32(vgpr(0), vgpr(33), vgpr(34)))  # dst unused in kind check
    m.add(VCndMaskB32(dst=vgpr(33), src0=vgpr(33), src1=vgpr(62), src2=vgpr(0)))
  m.add(DSLoadB32(dst=vgpr(42), src=vgpr(8), ds=DSModifiers(offset=0)))
  return m


def _kinds(module):
  return [type(i).__name__ for i in module.flatitems() if type(i).__name__ != "TextBlock"]


def test_swaps_move_before_tdm_wait_when_reads_drained():
  out = hoistLRSwapAboveWaitTDM(_loop(drained=True), None, TDM)
  assert _kinds(out) == ["DSLoadB32", "SWaitCnt", "VXorB32", "VXorB32",
                         "SWaitTensorcnt", "SBarrier", "DSLoadB32"]


def test_scale_ring_moves_before_tdm_wait_with_ab_swaps():
  out = hoistLRSwapAboveWaitTDM(_loop(drained=True, with_scale=True), None, TDM)
  assert _kinds(out) == [
      "DSLoadB32", "DSLoadB32", "SWaitCnt",
      "VXorB32", "VXorB32",
      "VAddU32", "VSubU32", "VCmpGEU32", "VCndMaskB32",
      "SWaitTensorcnt", "SBarrier", "DSLoadB32",
  ]


def test_swaps_spread_across_prior_mfma():
  m = Module("loop")
  m.add(DSLoadB32(dst=vgpr(40), src=vgpr(8), ds=DSModifiers(offset=0)))
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1))
  m.add(_mfma())
  m.add(_mfma())
  # Hoisted swap blob sitting right before READY wait.
  m.add(VXorB32(dst=vgpr(8), src0=vgpr(8), src1=vgpr(9)))
  m.add(VXorB32(dst=vgpr(10), src0=vgpr(10), src1=vgpr(11)))
  m.add(VXorB32(dst=vgpr(12), src0=vgpr(12), src1=vgpr(13)))
  m.add(VXorB32(dst=vgpr(14), src0=vgpr(14), src1=vgpr(15)))
  m.add(SWaitTensorcnt(tensorcnt=1))
  m.add(SBarrier())
  m.add(DSLoadB32(dst=vgpr(42), src=vgpr(8), ds=DSModifiers(offset=0)))
  out = spreadLRSwapAcrossPriorMfma(m, None, TDM)
  kinds = _kinds(out)
  assert kinds.count("VXorB32") == 4
  # Swaps no longer form one contiguous blob immediately before the wait.
  wait = kinds.index("SWaitTensorcnt")
  assert kinds[wait - 1] != "VXorB32" or kinds[wait - 4:wait].count("VXorB32") < 4
  # At least one xor landed before the second MFMA.
  mfmas = [i for i, k in enumerate(kinds) if k == "MFMAInstruction"]
  assert any(kinds[i] == "VXorB32" for i in range(mfmas[0], mfmas[1]))


def test_swaps_stay_after_barrier_with_read_in_flight():
  loop = _loop(drained=False)
  out = hoistLRSwapAboveWaitTDM(loop, None, TDM)
  assert _kinds(out) == _kinds(loop)


def test_non_tdm_kernel_unchanged():
  loop = _loop(drained=True)
  assert hoistLRSwapAboveWaitTDM(loop, None, {}) is loop
  assert spreadLRSwapAcrossPriorMfma(loop, None, {}) is loop
  assert spreadPostReadyDsLoads(loop, None, {}) is loop


def test_post_ready_b128_burst_interleaves_with_wmma():
  m = Module("MAINLOOP")
  m.add(SWaitTensorcnt(tensorcnt=1))
  m.add(SBarrier())
  m.add(SWaitAlu(va_vdst=0, comment="wait for LR offset swap to complete"))
  m.add(SSetVgprMsb(128))
  for i in range(6):
    m.add(DSLoadB128(dst=vgpr(100 + 4 * i, 4), src=vgpr(8), ds=DSModifiers(offset=i)))
  m.add(SSetVgprMsb(14))
  for _ in range(6):
    m.add(_mfma())
    m.add(DSLoadB32(dst=vgpr(40), src=vgpr(33), ds=DSModifiers(offset=0)))  # scale stays
  out = spreadPostReadyDsLoads(m, None, TDM)
  kinds = _kinds(out)
  # First post-READY compute is a WMMA, not a solid b128 burst.
  post = kinds[kinds.index("SWaitAlu") + 1:]
  first_b128 = post.index("DSLoadB128")
  first_wmma = post.index("MFMAInstruction")
  assert first_wmma < first_b128
  # All 6 burst loads still present; scale b32s unchanged in count.
  assert kinds.count("DSLoadB128") == 6
  assert kinds.count("DSLoadB32") == 6
  assert kinds.count("MFMAInstruction") == 6


def test_split_wait_lr_per_tile_keeps_late_loads_across_early_wmma():
  # B0 (early consumer) then B1 (later): first wait may keep B1 outstanding.
  m = Module("MAINLOOP")
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0)))   # A
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0)))  # B0
  m.add(DSLoadB128(dst=vgpr(204, 4), src=vgpr(26), ds=DSModifiers(offset=1)))  # B1
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))  # needs A+B0
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(204, 4)))  # needs A+B1
  out = splitWaitLRPerTile(m, None, TDM)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert len(waits) >= 1
  # First wait leaves B1 (newest) outstanding.
  assert waits[0].dscnt == 1
  # Second frontier drains B1.
  assert any(w.dscnt == 0 for w in waits[1:])


def test_split_wait_lr_empty_outstanding_keeps_hard_drain():
  # MAINLOOP entry: PRELOOP loads not visible → dscnt=0.
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 8)))
  out = splitWaitLRPerTile(m, None, TDM)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert len(waits) == 1 and waits[0].dscnt == 0
