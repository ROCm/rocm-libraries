# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from rocisa.code import Label, Module
from rocisa.container import DSModifiers, sgpr, vgpr
from rocisa.instruction import (
    DSLoadB32, DSLoadB128, MFMAInstruction, SAddCU32, SAddU32,
    SBarrier, SCBranchSCC0, SCmpEQU32, SCmpLeU32, SCMovB32, SMovB32, SSetVgprMsb,
    SWaitAlu, SWaitCnt, SWaitTensorcnt, SXorB32, TensorLoadToLds, VAddCOU32,
    VAddCCOU32, VAddU32, VCmpGEU32, VCndMaskB32, VSubU32, VXorB32,
)  # SWaitCnt used by tighten_wait_lr tests
from rocisa.enum import InstType
from rocisa.container import VCC

from tensilelite.Components.Subtile.WaitAluInsertion import (
    hoistLRSwapAboveWaitTDM, spreadLRSwapAcrossPriorMfma, spreadPostReadyDsLoads,
    advanceDsLoadsForWarSlack, splitWaitLRPerTile, mergeFreeReadyBarriers,
    hideWgpBarrierBehindWmma, placeFreeBarrierAfterWmma,
    relocateFreeToLastConsumer, spreadTDMSingleIssuerAdvance,
    spreadGL2PrefetchIntoWmma, tensorcntExactN, dropOrphanPreFreeDscnt,
)

TDM = {"enableTDMA": True, "enableTDMB": True}
TDMSI = {"enableTDMA": True, "enableTDMB": True, "TDMSingleIssuer": True}


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


def test_post_ready_cover_crosses_tdm_advance_salu():
  # Issuer advance SALU between WMMAs must not truncate the cover window —
  # otherwise the READY b128 burst stays bunched after the first WMMA.
  from rocisa.instruction import SCmpEQU32, SCBranchSCC0, SXorB32
  from rocisa.code import Label
  m = Module("MAINLOOP")
  m.add(SWaitTensorcnt(tensorcnt=1))
  m.add(SBarrier())
  m.add(SSetVgprMsb(128))
  for i in range(8):
    m.add(DSLoadB128(dst=vgpr(100 + 4 * i, 4), src=vgpr(8), ds=DSModifiers(offset=i)))
  m.add(SSetVgprMsb(14))
  m.add(_mfma())
  skip = Label("tdm_si_adv_B_test", "")
  m.add(SCmpEQU32(sgpr(0), 1, "TDMSingleIssuer: B advance"))
  m.add(SCBranchSCC0(skip.getLabelName(), "skip"))
  m.add(SXorB32(dst=sgpr(1), src0=sgpr(1), src1=sgpr(2), comment="desc LDS ^= xorMask"))
  m.add(skip)
  m.add(_mfma())
  m.add(_mfma())
  m.add(_mfma())
  out = spreadPostReadyDsLoads(m, None, TDM)
  kinds = _kinds(out)
  # Loads should appear after more than one WMMA (spread across the cover).
  post = kinds[kinds.index("SBarrier") + 1:]
  wmma_idxs = [i for i, k in enumerate(post) if k == "MFMAInstruction"]
  b128_idxs = [i for i, k in enumerate(post) if k == "DSLoadB128"]
  assert wmma_idxs and b128_idxs
  assert wmma_idxs[0] < b128_idxs[0]
  assert max(b128_idxs) > wmma_idxs[1], "burst must reach past the advance island"
  assert kinds.count("DSLoadB128") == 8


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


def test_fine_dscnt_loop_head_hard_drain_before_in_module_loads():
  # First WMMA uses preloop tiles; in-module loads follow — still need head 0.
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))  # preloop A/B
  m.add(DSLoadB128(dst=vgpr(204, 4), src=vgpr(26), ds=DSModifiers(offset=0)))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(204, 4)))  # needs new B
  out = splitWaitLRPerTile(m, None, TDM)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert len(waits) >= 2
  assert waits[0].dscnt == 0
  assert waits[0].comment and "Wait for LR to complete" in waits[0].comment
  kinds = _kinds(out)
  assert kinds.index("SWaitCnt") < kinds.index("MFMAInstruction")


def test_fine_dscnt_interleaved_loads_decreasing_n():
  # Mid-stream loads between WMMAs create later frontiers (proto fine_waits).
  # Load burst then early use, with more loads issued before later WMMAs →
  # multiple waits with decreasing N (not a single frontier at the head).
  m = Module("MAINLOOP")
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0)))   # A  ord=1
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0)))  # B0 ord=2
  m.add(DSLoadB128(dst=vgpr(204, 4), src=vgpr(26), ds=DSModifiers(offset=1)))  # B1 ord=3
  m.add(DSLoadB128(dst=vgpr(208, 4), src=vgpr(28), ds=DSModifiers(offset=2)))  # B2 ord=4
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))  # need max(A,B0)=2 → N=4-2=2
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(204, 4)))  # need 3 → N=1
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(208, 4)))  # need 4 → N=0
  out = splitWaitLRPerTile(m, None, TDM)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert [w.dscnt for w in waits] == [2, 1, 0]
  kinds = _kinds(out)
  wait_idxs = [i for i, k in enumerate(kinds) if k == "SWaitCnt"]
  wmma_idxs = [i for i, k in enumerate(kinds) if k == "MFMAInstruction"]
  assert len(wait_idxs) == len(wmma_idxs) == 3
  for wi, mi in zip(wait_idxs, wmma_idxs):
    assert wi == mi - 1
  assert "tiles of the next WMMA landed" in (waits[0].comment or "")


def test_fine_dscnt_keeps_newer_loads_outstanding():
  # Two loads then early WMMA that only needs the older → dscnt 1, then 0.
  m = Module("MAINLOOP")
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0)))   # A
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0)))  # B0
  m.add(DSLoadB128(dst=vgpr(204, 4), src=vgpr(26), ds=DSModifiers(offset=1)))  # B1
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))  # need B0 → N=1
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(204, 4)))  # need B1 → N=0
  out = splitWaitLRPerTile(m, None, TDM)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert [w.dscnt for w in waits] == [1, 0]
  assert "tiles of the next WMMA landed" in (waits[0].comment or "")


def test_fine_dscnt_free_no_glued_drain():
  # Last consumer's fine wait is FREE proof — no extra dscnt=0 on the barrier.
  m = Module("MAINLOOP")
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0)))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0)))
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))
  # Scheduler-style FREE wait_lr (to be stripped) glued before barrier.
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  out = splitWaitLRPerTile(m, None, TDM)
  kinds = _kinds(out)
  assert kinds.count("SWaitCnt") == 1
  assert kinds.index("SWaitCnt") < kinds.index("MFMAInstruction")
  # No wait immediately before the barrier.
  bar_i = kinds.index("SBarrier")
  assert kinds[bar_i - 1] == "MFMAInstruction"
  hole = placeFreeBarrierAfterWmma(out, None, TDM, nWmma=1)
  hk = _kinds(hole)
  # Fine wait remains the FREE drain; hole WMMAs sit between it and the barrier.
  assert hk.count("SWaitCnt") == 1
  wi, bi, ti = (hk.index(k) for k in ("SWaitCnt", "SBarrier", "TensorLoadToLds"))
  assert wi < bi < ti
  assert hk[wi + 1:bi]  # at least one WMMA in the FREE-hole
  assert all(k == "MFMAInstruction" for k in hk[wi + 1:bi])
  assert hk[bi - 1] == "MFMAInstruction"  # no glued dscnt=0 on barrier


def test_advance_ds_loads_stops_at_war_fence():
  # Last WMMA that reads B dsts must stay before the refill.
  m = Module("MAINLOOP")
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))  # last reader of B tile
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(300, 4)))  # unrelated B
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(300, 4)))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=1 read=0"))
  out = advanceDsLoadsForWarSlack(m, None, TDM, maxWmmaCross=64)
  kinds = _kinds(out)
  assert kinds.index("DSLoadB128") == 1  # after first WMMA only


def test_advance_ds_loads_respects_wmma_cap():
  m = Module("MAINLOOP")
  for _ in range(6):
    m.add(_mfma(a=vgpr(400, 16), b=vgpr(500, 4)))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=1 read=0"))
  out = advanceDsLoadsForWarSlack(m, None, TDM, maxWmmaCross=2)
  kinds = _kinds(out)
  # Crossed only 2 WMMAs → load sits after the first 4.
  assert kinds.index("DSLoadB128") == 4


def test_advance_ds_loads_prefers_b_before_a():
  m = Module("MAINLOOP")
  for _ in range(4):
    m.add(_mfma(a=vgpr(400, 16), b=vgpr(500, 4)))  # no overlap with A/B dsts
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0),
                   comment="SubtileA[0, 0] subIterK=1 read=0"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=1 read=0"))
  out = advanceDsLoadsForWarSlack(m, None, TDM, maxWmmaCross=64)
  loads = [i for i in out.flatitems() if type(i).__name__ == "DSLoadB128"]
  assert "SubtileB[" in loads[0].comment
  assert "SubtileA[" in loads[1].comment
  # Both pulled before the WMMA stream.
  kinds = _kinds(out)
  assert kinds.index("DSLoadB128") < kinds.index("MFMAInstruction")


def test_advance_ds_loads_keeps_wmma_after_wait_lr():
  # Opening wait_dscnt then WMMAs then a late B load: must not dump the load
  # into the pre-WMMA gap (loop-head burst).
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  for _ in range(4):
    m.add(_mfma(a=vgpr(400, 16), b=vgpr(500, 4)))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=1 read=0"))
  out = advanceDsLoadsForWarSlack(m, None, TDM, maxWmmaCross=64)
  kinds = _kinds(out)
  wait_i = kinds.index("SWaitCnt")
  wmma_i = kinds.index("MFMAInstruction")
  load_i = kinds.index("DSLoadB128")
  assert wait_i < wmma_i < load_i


def _tdm():
  return TensorLoadToLds(group0=sgpr(0, 4), group1=sgpr(0, 8),
                         group2=None, group3=None, comment="tdm")


def test_merge_free_ready_barriers_hoists_tcnt_and_drops_free():
  # FREE bar → TDM → WMMA → tensorcnt → READY bar  ⇒  tcnt → bar → TDM → WMMA
  m = Module("MAINLOOP")
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=0 read=0"))
  out = mergeFreeReadyBarriers(m, None, TDM, forceDrain=False)
  kinds = _kinds(out)
  assert kinds.count("SBarrier") == 1
  assert kinds.index("SWaitTensorcnt") < kinds.index("SBarrier")
  assert kinds.index("SBarrier") < kinds.index("TensorLoadToLds")
  assert kinds.index("TensorLoadToLds") < kinds.index("MFMAInstruction")
  assert kinds.index("MFMAInstruction") < kinds.index("DSLoadB128")
  # No TDM before FREE → exact N=0 (await prologue / prior fill).
  tcnt = next(x for x in out.flatitems() if type(x).__name__ == "SWaitTensorcnt")
  assert "s_wait_tensorcnt 0" in str(tcnt)
  out_drain = mergeFreeReadyBarriers(m, None, TDM, forceDrain=True)
  tcnt = next(x for x in out_drain.flatitems()
              if type(x).__name__ == "SWaitTensorcnt")
  assert "s_wait_tensorcnt 0" in str(tcnt)


def test_merge_free_ready_skips_tensorcnt_drain():
  m = Module("MAINLOOP")
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  m.add(SWaitTensorcnt(tensorcnt=0, comment="drain"))
  m.add(SBarrier(comment="Barrier"))
  out = mergeFreeReadyBarriers(m, None, TDM)
  assert _kinds(out).count("SBarrier") == 2


def test_merge_free_ready_noop_without_tdm():
  m = Module("MAINLOOP")
  m.add(SBarrier(comment="Barrier"))
  m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1))
  m.add(SBarrier(comment="Barrier"))
  assert mergeFreeReadyBarriers(m, None, TDM) is m
  assert mergeFreeReadyBarriers(m, None, {}) is m


def test_sig_early_free_pulls_wmma_before_drain():
  # Adjacent: WMMAs ; wait_dscnt ; barrier ; TDM  →  dscnt ; sig ; WMMAs ; wait ; TDM
  m = Module("MAINLOOP")
  for _ in range(3):
    m.add(_mfma())
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  out = hideWgpBarrierBehindWmma(m, None, TDM, nWmma=2)
  kinds = _kinds(out)
  # Two barriers (split signal/wait) with exactly 2 WMMAs between them.
  bar = [i for i, k in enumerate(kinds) if k == "SBarrier"]
  assert len(bar) == 2
  between = kinds[bar[0] + 1:bar[1]]
  assert between == ["MFMAInstruction", "MFMAInstruction"]
  assert kinds.index("SWaitCnt") < bar[0]
  assert bar[1] < kinds.index("TensorLoadToLds")
  # One WMMA remains before the drain.
  assert kinds[:kinds.index("SWaitCnt")].count("MFMAInstruction") == 1


def test_sig_early_free_hole_opens_cover_between_signal_wait():
  # After placeFreeBarrier: wait ; WMMA ; barrier ; TDM
  # → wait ; signal ; WMMA ; wait ; TDM  (cover hides the WGP wait).
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=44, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(_mfma())
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  out = hideWgpBarrierBehindWmma(m, None, TDM, nWmma=2)
  kinds = _kinds(out)
  bar = [i for i, k in enumerate(kinds) if k == "SBarrier"]
  assert len(bar) == 2
  assert kinds[kinds.index("SWaitCnt") + 1] == "SBarrier"  # signal
  assert kinds[bar[0] + 1:bar[1]] == ["MFMAInstruction"]
  assert bar[1] < kinds.index("TensorLoadToLds")


def _init_rocisa_gfx1250():
  import shutil
  from rocisa import rocIsa
  from tensilelite.Common.Architectures import gfxToIsa
  ri = rocIsa.getInstance()
  isa = gfxToIsa("gfx1250")
  asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
  ri.init(isa, asmpath)
  ri.setKernel(isa, 32)


def test_drop_orphan_pre_free_dscnt_before_cw():
  # Proto: FREE has no barrier-local drain.  After SIG_EARLY+CW:
  #   dscnt ; CW ; WS ; WMMA ; WW ; TDM  →  CW ; WS ; WMMA ; WW ; TDM
  _init_rocisa_gfx1250()
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=44, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(SBarrier(True, True, True, "cluster_barrier wait"))
  m.add(SBarrier(True, False, False, comment="SIG_EARLY signal"))
  m.add(_mfma())
  m.add(SBarrier(True, True, False, comment="Barrier"))
  m.add(_tdm())
  kern = {**TDM, "ClusterBarrier": True}
  out = dropOrphanPreFreeDscnt(m, None, kern)
  kinds = _kinds(out)
  assert "SWaitCnt" not in kinds
  assert kinds[0] == "SBarrier"  # CW
  bars = [i for i, k in enumerate(kinds) if k == "SBarrier"]
  assert kinds[bars[1] + 1] == "MFMAInstruction"  # cover after WS
  assert bars[2] < kinds.index("TensorLoadToLds")


def test_drop_orphan_pre_free_dscnt_keeps_proof_without_cover():
  # No SIG_EARLY cover → drain is still the FREE proof; leave it.
  _init_rocisa_gfx1250()
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=44, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(SBarrier(True, True, True, "cluster_barrier wait"))
  m.add(SBarrier(True, True, False, comment="Barrier"))
  m.add(_tdm())
  out = dropOrphanPreFreeDscnt(m, None, {**TDM, "ClusterBarrier": True})
  assert out is m


def test_sig_early_ready_forward_opt_in():
  # READY forward-pull is opt-in (default off — delayed ds_loads on DU512).
  m = Module("MAINLOOP")
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=0 read=0"))
  for _ in range(3):
    m.add(_mfma())
  assert hideWgpBarrierBehindWmma(m, None, TDM, nWmma=2) is m
  out = hideWgpBarrierBehindWmma(m, None, TDM, nWmma=2, allowForward=True)
  kinds = _kinds(out)
  bar = [i for i, k in enumerate(kinds) if k == "SBarrier"]
  assert len(bar) == 2
  assert kinds[bar[0] + 1:bar[1]] == ["MFMAInstruction", "MFMAInstruction"]
  assert kinds.index("SWaitTensorcnt") < bar[0]
  assert bar[1] < kinds.index("DSLoadB128")
  assert kinds.count("MFMAInstruction") == 3


def test_sig_early_noop_without_tdm_or_mmas():
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1))
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  assert hideWgpBarrierBehindWmma(m, None, TDM) is m
  m2 = Module("MAINLOOP")
  m2.add(_mfma())
  m2.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1))
  m2.add(SBarrier(comment="Barrier"))
  assert hideWgpBarrierBehindWmma(m2, None, {}) is m2


def test_free_hole_barrier_after_wmma():
  # FREE: WMMAs ; dscnt ; barrier ; TDM  →  dscnt ; WMMAs ; barrier ; TDM
  m = Module("MAINLOOP")
  for _ in range(3):
    m.add(_mfma())
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  out = placeFreeBarrierAfterWmma(m, None, TDM, nWmma=2)
  kinds = _kinds(out)
  assert kinds.count("SBarrier") == 1
  wait_i = kinds.index("SWaitCnt")
  bar_i = kinds.index("SBarrier")
  tdm_i = kinds.index("TensorLoadToLds")
  assert wait_i < bar_i < tdm_i
  assert kinds[wait_i + 1:bar_i] == ["MFMAInstruction", "MFMAInstruction"]
  assert kinds[:wait_i] == ["MFMAInstruction"]


def test_free_hole_skips_ready_tensorcnt_site():
  m = Module("MAINLOOP")
  for _ in range(2):
    m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=0 read=0"))
  assert placeFreeBarrierAfterWmma(m, None, TDM, nWmma=2) is m


def test_free_hole_uses_nearest_fine_wait_as_drain():
  # Fine proof already above FREE: wait ; WMMA ; barrier — grow hole from above.
  # Last-consumer placement owns the proof — do not harden dscnt to 0.
  m = Module("MAINLOOP")
  for _ in range(3):
    m.add(_mfma())
  m.add(SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(_mfma())  # already between wait and barrier
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  out = placeFreeBarrierAfterWmma(m, None, TDM, nWmma=2)
  kinds = _kinds(out)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert len(waits) == 1 and waits[0].dscnt == 2
  wi = kinds.index("SWaitCnt")
  bi = kinds.index("SBarrier")
  assert kinds[wi + 1:bi] == ["MFMAInstruction", "MFMAInstruction"]
  assert kinds[:wi].count("MFMAInstruction") == 2


def test_relocate_free_to_last_consumer():
  # Loads → wait → consumer WMMA → extra WMMAs → FREE → TDM → READY
  # → wait ; consumer ; barrier ; TDM  (no hard dscnt=0 on the barrier).
  m = Module("MAINLOOP")
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0)))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0)))
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))  # consumer of last load
  for _ in range(3):
    m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  out = relocateFreeToLastConsumer(m, None, TDM)
  kinds = _kinds(out)
  waits = [i for i in out.flatitems() if type(i).__name__ == "SWaitCnt"]
  assert waits and waits[0].dscnt == 0  # existing fine wait, not a new harden
  wi = kinds.index("SWaitCnt")
  # First barrier is FREE (followed by TDM); second is READY.
  bars = [i for i, k in enumerate(kinds) if k == "SBarrier"]
  tdm = kinds.index("TensorLoadToLds")
  assert len(bars) == 2
  assert wi < bars[0] < tdm < bars[1]
  # FREE glued on the consumer: wait ; WMMA ; barrier ; TDM
  assert kinds[wi + 1] == "MFMAInstruction"
  assert kinds[wi + 2] == "SBarrier"
  assert kinds[wi + 3] == "TensorLoadToLds"
  # Trailing WMMAs moved after TDM (proto: FREE right after first consumer).
  assert kinds[tdm + 1:bars[1]].count("MFMAInstruction") == 3


def _tdm_si_issue_gate(wave, tag, label_name):
  """Issuer-only gate: cmp ; branch ; load ; skip label."""
  skip = Label(label_name, "")
  return [
      SCmpEQU32(src0=sgpr(0), src1=wave,
                comment="TDMSingleIssuer: %s issue = wave %d" % (tag, wave)),
      SCBranchSCC0(labelName=skip.getLabelName(),
                   comment="other waves skip %s TDM issue" % tag),
      TensorLoadToLds(sgpr(40, 4), sgpr(44, 8), sgpr(52, 4), sgpr(56, 4),
                      comment="TDMSingleIssuer: %s tensor_load (wave %d)" % (tag, wave)),
      skip,
  ]


def test_relocate_free_keeps_tdmsi_issue_gates_forward():
  # FREE relocate must move issue cmp/branch with the load body. Orphaning the
  # gate leaves a backward branch into the relocated skip label → hang.
  m = Module("MAINLOOP")
  m.add(DSLoadB128(dst=vgpr(100, 4), src=vgpr(8), ds=DSModifiers(offset=0)))
  m.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))
  for _ in range(3):
    m.add(_mfma(a=vgpr(100, 16), b=vgpr(200, 4)))
  m.add(SBarrier(comment="Barrier"))
  for inst in _tdm_si_issue_gate(0, "A+MXSA", "tdm_si_ld_A_ut"):
    m.add(inst)
  for inst in _tdm_si_issue_gate(2, "B+MXSB", "tdm_si_ld_B_ut"):
    m.add(inst)
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))

  out = relocateFreeToLastConsumer(m, None, TDMSI)
  items = [i for i in out.flatitems() if type(i).__name__ != "TextBlock"]
  # Every issue-gate branch must target a Label that appears *after* it.
  label_pos = {}
  for i, it in enumerate(items):
    if isinstance(it, Label):
      label_pos[it.getLabelName()] = i
  for i, it in enumerate(items):
    if not isinstance(it, SCBranchSCC0):
      continue
    if "TDM issue" not in (it.comment or ""):
      continue
    tgt = it.labelName if hasattr(it, "labelName") else it.getParams()[0]
    # rocisa may store label as string name
    name = tgt if isinstance(tgt, str) else str(tgt)
    assert name in label_pos, "missing skip label %s" % name
    assert label_pos[name] > i, "backward issue-gate branch to %s" % name
  # FREE barrier immediately followed by issue cmp (gate moved with loads).
  bars = [i for i, it in enumerate(items) if isinstance(it, SBarrier)]
  assert isinstance(items[bars[0] + 1], SCmpEQU32)
  assert "issue = wave" in (items[bars[0] + 1].comment or "")


def test_merge_uses_tcnt_exact_not_scheduler_n():
  # Prior TDM then FREE site: exact at FREE awaits that TDM → after=0.
  # Scheduler READY N=1 would race once the wait sits before the site TDM.
  m = Module("MAINLOOP")
  m.add(_tdm())  # previous fill
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=0 read=0"))
  items = list(m.flatitems())
  free_i = next(i for i, x in enumerate(items) if type(x).__name__ == "SBarrier")
  assert tensorcntExactN(items, free_i) == 0
  # Classic READY after site TDM: await site TDM → after=0; await prior → after=1.
  tcnt_i = next(i for i, x in enumerate(items)
                if type(x).__name__ == "SWaitTensorcnt")
  site_tdm = next(i for i, x in enumerate(items)
                  if type(x).__name__ == "TensorLoadToLds" and i > free_i)
  prior_tdm = next(i for i, x in enumerate(items)
                   if type(x).__name__ == "TensorLoadToLds")
  assert tensorcntExactN(items, tcnt_i, await_tdm_i=site_tdm) == 0
  assert tensorcntExactN(items, tcnt_i, await_tdm_i=prior_tdm) == 1
  out = mergeFreeReadyBarriers(m, None, TDM, forceDrain=False)
  kinds = _kinds(out)
  assert kinds.count("SBarrier") == 1
  assert kinds[0] == "TensorLoadToLds"  # prior fill kept
  tcnt = next(x for x in out.flatitems() if type(x).__name__ == "SWaitTensorcnt")
  assert "s_wait_tensorcnt 0" in str(tcnt)
  assert "tcnt_exact" in (getattr(tcnt, "comment", "") or "")


def test_merge_with_tdm_si_advances_uses_exact():
  # tcnt_exact unblocks MERGE on wave-gated covers (no longer skip).
  m = Module("MAINLOOP")
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  m.add(_mfma())
  m.add(SCmpEQU32(src0=sgpr(0), src1=0, comment="TDMSingleIssuer: A advance = wave 0"))
  m.add(SCBranchSCC0(labelName="label_tdm_si_adv_A_skip", comment="other waves skip"))
  m.add(Label("label_tdm_si_adv_A_skip", ""))
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=0 read=0"))
  out = mergeFreeReadyBarriers(m, None, TDM, forceDrain=False)
  assert _kinds(out).count("SBarrier") == 1
  tcnt = next(x for x in out.flatitems() if type(x).__name__ == "SWaitTensorcnt")
  assert "s_wait_tensorcnt 0" in str(tcnt)


def test_merge_does_not_harden_free_fine_wait():
  # Fine wait above FREE stays at N>0 — last-consumer owns the proof.
  m = Module("MAINLOOP")
  m.add(SWaitCnt(dscnt=48, vlcnt=-1, vscnt=-1,
                 comment="tiles of the next WMMA landed"))
  m.add(_mfma())
  m.add(SBarrier(comment="Barrier"))
  m.add(_tdm())
  m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(comment="Barrier"))
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB[0, 0] subIterK=0 read=0"))
  out = mergeFreeReadyBarriers(m, None, TDM, forceDrain=False)
  waits = [x for x in out.flatitems() if type(x).__name__ == "SWaitCnt"]
  assert len(waits) == 1 and waits[0].dscnt == 48
  assert "MERGE_BAR FREE proof" not in (getattr(waits[0], "comment", "") or "")


def _tdm_si_load():
  return TensorLoadToLds(
      sgpr(40, 4), sgpr(44, 8), sgpr(52, 4), sgpr(56, 4),
      comment="TDMSingleIssuer: ungated tensor_load (per-wave desc)")


def _tdm_si_advance_chunk(tc, wave, label_name):
  """One wave-gated A/B-style advance chunk (cmp + branch + xor/add + label)."""
  items = []
  skip = Label(label_name, "")
  items.append(SCmpEQU32(src0=sgpr(0), src1=wave,
                         comment="TDMSingleIssuer: %s advance = wave %d" % (tc, wave)))
  items.append(SCBranchSCC0(labelName=skip.getLabelName(),
                            comment="other waves skip %s advance" % tc))
  items.append(SXorB32(dst=sgpr(41), src0=sgpr(41), src1=sgpr(60),
                       comment="desc LDS ^= xorMask (buf0^buf1)"))
  items.append(SAddU32(dst=sgpr(42), src0=sgpr(42), src1=sgpr(61),
                       comment="desc global += inc"))
  items.append(SAddCU32(dst=sgpr(43), src0=sgpr(43), src1=0,
                        comment="desc global carry"))
  items.append(skip)
  return items


def _tdm_si_advance_wall():
  m = Module("wall")
  for tc, wave in (("A", 0), ("B", 1), ("MXSA", 2), ("MXSB", 3)):
    for inst in _tdm_si_advance_chunk(tc, wave, "tdm_si_adv_%s_ut" % tc):
      m.add(inst)
  return list(m.flatitems())


def test_spread_tdmsi_advance_into_wmma_holes():
  # tensor_load ; 4 gated advances ; 4 WMMAs ; next load
  # → each advance chunk lands after a distinct cover WMMA.
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _tdm_si_advance_wall():
    m.add(inst)
  for _ in range(4):
    m.add(_mfma())
  m.add(_tdm_si_load())

  out = spreadTDMSingleIssuerAdvance(m, None, TDMSI)
  items = [i for i in out.flatitems() if type(i).__name__ != "TextBlock"]
  kinds = [type(i).__name__ for i in items]

  assert kinds[0] == "TensorLoadToLds"
  assert kinds[-1] == "TensorLoadToLds"
  # Wall no longer sits immediately after the first load.
  assert kinds[1] == "MFMAInstruction"
  # Four advance cmps, each immediately after an MFMA.
  cmp_idx = [i for i, it in enumerate(items)
             if isinstance(it, SCmpEQU32) and "advance" in (it.comment or "")]
  assert len(cmp_idx) == 4
  for ci in cmp_idx:
    assert kinds[ci - 1] == "MFMAInstruction"
  # All advances finish before the next tensor_load.
  next_load = kinds.index("TensorLoadToLds", 1)
  assert max(cmp_idx) < next_load


def test_spread_tdmsi_advance_defers_past_ready():
  # Align arrival: gated advances must not sit between TDM and READY wait.
  # tensor_load ; wall ; cover WMMAs ; tcnt ; READY ; post WMMAs ; next load
  # → READY wait precedes every advance cmp; chunks hide under post-READY WMMAs.
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _tdm_si_advance_wall():
    m.add(inst)
  for _ in range(2):
    m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(True, True, False, comment="READY wait"))
  for _ in range(4):
    m.add(_mfma())
  m.add(_tdm_si_load())

  out = spreadTDMSingleIssuerAdvance(m, None, TDMSI)
  items = [i for i in out.flatitems() if type(i).__name__ != "TextBlock"]
  ready_i = next(i for i, it in enumerate(items)
                 if isinstance(it, SBarrier) and "READY" in (it.comment or ""))
  cmp_idx = [i for i, it in enumerate(items)
             if isinstance(it, SCmpEQU32) and "advance" in (it.comment or "")]
  assert len(cmp_idx) == 4
  assert all(ci > ready_i for ci in cmp_idx)
  for ci in cmp_idx:
    assert type(items[ci - 1]).__name__ == "MFMAInstruction"
  # Pre-READY cover WMMAs stay immediately after the load (no elect between).
  assert type(items[1]).__name__ == "MFMAInstruction"
  assert not any(
      isinstance(it, SCmpEQU32) and "advance" in (it.comment or "")
      for it in items[1:ready_i])


def test_spread_tdmsi_advance_wall_after_ready_without_post_cover():
  # READY then loads/next TDM with no post-READY WMMAs → wall right after wait.
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _tdm_si_advance_wall():
    m.add(inst)
  m.add(_mfma())
  m.add(SWaitTensorcnt(tensorcnt=1, comment="READY"))
  m.add(SBarrier(True, True, False, comment="READY wait"))
  m.add(_tdm_si_load())

  out = spreadTDMSingleIssuerAdvance(m, None, TDMSI)
  items = [i for i in out.flatitems() if type(i).__name__ != "TextBlock"]
  ready_i = next(i for i, it in enumerate(items)
                 if isinstance(it, SBarrier) and "READY" in (it.comment or ""))
  assert isinstance(items[ready_i + 1], SCmpEQU32)
  assert "advance" in (items[ready_i + 1].comment or "")
  assert all(
      not (isinstance(it, SCmpEQU32) and "advance" in (it.comment or ""))
      for it in items[:ready_i])


def test_spread_tdmsi_advance_fallback_too_few_wmmas():
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _tdm_si_advance_wall():
    m.add(inst)
  m.add(_mfma())  # only 1 cover WMMA for 4 chunks
  m.add(_tdm_si_load())
  assert spreadTDMSingleIssuerAdvance(m, None, TDMSI) is m


def test_spread_tdmsi_advance_noop_without_flag():
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _tdm_si_advance_wall():
    m.add(inst)
  for _ in range(4):
    m.add(_mfma())
  m.add(_tdm_si_load())
  assert spreadTDMSingleIssuerAdvance(m, None, TDM) is m


def _gl2_pf_block():
  """Minimal Inc+Issue wall matching emit_gl2_prefetch_inc/issue shape.

  Uses ``SMovB32`` stand-ins for ``global_prefetch_b8`` (asm-cap gated in rocisa).
  """
  items = []
  items.append(SCmpLeU32(src0=sgpr(10), src1=4,
                         comment="counterL <= PGR(2)+PGL(2)?"))
  items.append(SCMovB32(dst=sgpr(20), src=0, comment="zero GL2PrefetchIncA"))
  items.append(SCMovB32(dst=sgpr(21), src=0, comment="zero GL2PrefetchIncB"))
  for tag, wave, addr, inc in (("IncA", 1, 100, 20), ("A", 1, 100, 20),
                               ("IncB", 3, 110, 21), ("B", 3, 110, 21)):
    skip = Label("gl2_pf_%s_ut" % tag, "")
    items.append(SCmpEQU32(src0=sgpr(0), src1=wave,
                           comment="TDMSingleIssuer: GL2 PF %s = wave %d" % (tag, wave)))
    items.append(SCBranchSCC0(labelName=skip.getLabelName(),
                              comment="other waves skip GL2 PF %s" % tag))
    if tag.startswith("Inc"):
      items.append(VAddCOU32(vgpr(addr), VCC(), vgpr(addr), sgpr(inc)))
      items.append(VAddCCOU32(vgpr(addr + 1), VCC(), vgpr(addr + 1), 0, VCC()))
    else:
      items.append(SMovB32(dst=sgpr(30), src=0,
                           comment="global_prefetch_b8 stand-in %s" % tag))
    items.append(skip)
  return items


def test_spread_gl2_prefetch_into_wmma_after_lr():
  # TDM ; PF wall ; gap WMMAs ; early ds ; WMMA ; late ds ; WMMA
  # → PF moves after the *last* post-gap LR (proto: after the block's LRs).
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _gl2_pf_block():
    m.add(inst)
  for _ in range(10):  # clear tdmGapInsts=8
    m.add(_mfma())
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0),
                   comment="SubtileB early"))
  m.add(_mfma())
  m.add(DSLoadB128(dst=vgpr(204, 4), src=vgpr(24), ds=DSModifiers(offset=1),
                   comment="SubtileB late"))
  m.add(_mfma())

  kern = {**TDMSI, "PrefetchGL2": 2}
  out = spreadGL2PrefetchIntoWmma(m, None, kern, tdmGapInsts=8)
  items = list(out.flatitems())
  pf_idx = [i for i, it in enumerate(items)
            if "global_prefetch_b8" in (getattr(it, "comment", "") or "")]
  assert len(pf_idx) == 2
  tdm_i = next(i for i, it in enumerate(items)
               if type(it).__name__ == "TensorLoadToLds")
  lr_idxs = [i for i, it in enumerate(items)
             if type(it).__name__ == "DSLoadB128"]
  assert len(lr_idxs) == 2
  assert lr_idxs[0] > tdm_i
  # Prefetch sits after the late LR, before its following WMMA.
  assert min(pf_idx) > lr_idxs[1]
  assert any(type(it).__name__ == "MFMAInstruction"
             for it in items[max(pf_idx):])


def test_spread_gl2_prefetch_noop_without_pgl():
  m = Module("MAINLOOP")
  m.add(_tdm_si_load())
  for inst in _gl2_pf_block():
    m.add(inst)
  for _ in range(10):
    m.add(_mfma())
  m.add(DSLoadB128(dst=vgpr(200, 4), src=vgpr(24), ds=DSModifiers(offset=0)))
  m.add(_mfma())
  assert spreadGL2PrefetchIntoWmma(m, None, TDMSI) is m
