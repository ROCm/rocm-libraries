# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM descriptor primitives with the region walk taken out, for GIR to place.

`globalReadDo` reconstructs which region it is on from the loop it is running; GIR replaces
that reconstruction, so these expose the same work one region at a time.  `tdmGirIsIter`,
`tdmSplitDim1ForRegion` and `tdmMaskShortMembers` are per-region facts the scaffold's own split
walk needs too, so they have two callers and one derivation.
"""

from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SAddCU32, SAddU32, SAddU64, SCSelectB32, SCmpEQU32, SCmpLtU32, \
    SLShiftLeftB32, SLShiftRightB32, SMovB32, SOrB32, SSubBU32, SSubU32

from Tensile.Components import TDMSplit as _tdm_split
from ..TensorDataMover import TensorDataMoverLoad
from ..TDMFuse import tdmSetIncsSgpr


def tdmIncrementGir(writer, kernel, tP, wrapLead=None, tPFused=None) -> Module:
  """Advance the TDM descriptor by one chunk: delta, then a 64-bit `addr += delta`.

  The delta is the per-iteration stride, or `WrapU` on the wrap iteration.  `wrapLead` is the
  offset between the chunk this advance produces and the frame the loop counter denotes; `None`
  means no StaggerU wrap on this path.  GIR supplies it directly, where `tdmIncrementAB`
  reconstructs the same thing from `PrefetchGlobalRead` and the pipeline phase.

  `tPFused` is the peer of a Φ-fused movement, where one descriptor register set serves both
  operands.  One movement is still one advance, so only the stride (`tdm{tc}{tcPeer}Incs`, the
  per-operand `GlobalReadIncs*` having been returned to the pool) and the parity-selected
  `WrapU` differ.  Sparse is rejected under UseLoopModel.
  """
  fused: bool = tPFused is not None
  if not (kernel.get("UseSubtileImpl") or fused):
    # The invariant is register independence, not wave count, and it holds per-operand: under
    # `A_MX` the group is {A, MXSA, MXSB} while B keeps its own descriptor.
    _tcHere = tP["tensorChar"]
    _owner = writer.tdmDescriptorOwners(kernel).get(_tcHere, _tcHere)
    assert _owner == _tcHere, \
        (f"unfused advance requested for {_tcHere} while it aliases {_owner}'s descriptor "
         f"(TDMFuse={kernel['TDMFuse']}) — two independent advances against one aliased set")
  comp: TensorDataMoverLoad = TensorDataMoverLoad.find(writer)
  tc: str = tP['tensorChar']
  mod = Module("TDM increment (GIR, fused)" if fused else "TDM increment (GIR)")
  tdmGroup0 = f"tdm{tc}Group0"
  if fused:
    tcPeer: str = tPFused['tensorChar']
    incSgprName = tdmSetIncsSgpr(kernel, tc)
  else:
    incSgprName = f"GlobalReadIncs{tc}"

  if writer.states.staggerUCode and wrapLead is not None:
    # A fused advance needs two more temps: the parity compare and the wrap-lead compare both
    # land in SCC, so the wrap is materialized before the second compare overwrites it.
    with writer.allocTmpSgpr(4 if fused else 2, tag="tdmIncrementGir_tmpSgprInfo") as tmpSgprInfo:
      incTmpLo = tmpSgprInfo.idx
      incTmpHi = tmpSgprInfo.idx + 1
      if fused:
        # Each operand walks back to its own base, so the wrap is per-operand even though the
        # descriptor is shared, and it is selected before the wrap-lead compare takes SCC.
        wrapTmpLo = tmpSgprInfo.idx + 2
        wrapTmpHi = tmpSgprInfo.idx + 3
        _members, _ranges = writer.tdmFuseGroupOf(kernel, tc)
        if _ranges is None:
          # Two members: a 2-instruction s_cselect.  `tdmWaveSelect` would emit a branch chain.
          if writer.isTdmWaveIdxLive(kernel):
            writer._emitTdmWaveParitySCC(mod, kernel, comment="check wave parity")
          else:
            with writer.allocTmpSgpr(1, tag="tdmIncrementGir_waveIdTmp") as waveIdTmp:
              writer._emitTdmWaveParitySCC(mod, kernel, waveIdTmp.idx, "check wave parity")
          mod.add(SCSelectB32(dst=sgpr(wrapTmpLo), src0=sgpr(f"WrapU{tcPeer}+0"),
                              src1=sgpr(f"WrapU{tc}+0"),
                              comment="select WrapU by wave parity (lo)"))
          mod.add(SCSelectB32(dst=sgpr(wrapTmpHi), src0=sgpr(f"WrapU{tcPeer}+1"),
                              src1=sgpr(f"WrapU{tc}+1"),
                              comment="select WrapU by wave parity (hi)"))
        else:
          # The ranges are ascending and contiguous, so a cselect chain settles on the right member,
          # each wrap overwriting the running value for every wave at or above its first.  It compares
          # the wave index rather than a parity bit.
          if not writer._tdmWaveIdxNeedsTmp(kernel):
            writer._emitTdmWrapUSelectChain(mod, _members, _ranges,
                                          sgpr(writer._emitTdmWaveIdx(mod, kernel)),
                                          wrapTmpLo, wrapTmpHi)
          else:
            with writer.allocTmpSgpr(1, tag="tdmIncrementGir_waveIdTmp") as waveIdTmp:
              _wid = writer._emitTdmWaveIdx(mod, kernel, waveIdTmp.idx)
              writer._emitTdmWrapUSelectChain(mod, _members, _ranges,
                                            sgpr(_wid), wrapTmpLo, wrapTmpHi)
        wrapLo, wrapHi = sgpr(wrapTmpLo), sgpr(wrapTmpHi)
      else:
        wrapLo, wrapHi = sgpr(f"WrapU{tc}+0"), sgpr(f"WrapU{tc}+1")
      # (1) DELTA: WrapU on the wrap iteration, else the plain stride.  `wrapLead` shifts the
      # compare to the frame the advance runs in: issued n chunks ahead, compare `counter + n`.
      if wrapLead:
        # A negative lead: a steady advance runs one chunk behind the counter's frame.  `SAddU32`
        # takes no negative literal, so it subtracts instead.
        if wrapLead > 0:
          mod.add(SAddU32(dst=sgpr(incTmpLo),
                          src0=writer.loopCounter(kernel, writer.states.unrollIdx),
                          src1=wrapLead, comment="GIR wrap lead(+%u)" % wrapLead))
        else:
          mod.add(SSubU32(dst=sgpr(incTmpLo),
                          src0=writer.loopCounter(kernel, writer.states.unrollIdx),
                          src1=-wrapLead, comment="GIR wrap lead(-%u)" % -wrapLead))
        mod.add(SCmpEQU32(src0=sgpr("StaggerUIter"), src1=sgpr(incTmpLo),
                          comment="Is this wrapIter? (GIR lead)"))
      else:
        mod.add(SCmpEQU32(src0=writer.loopCounter(kernel, writer.states.unrollIdx),
                          src1=sgpr("StaggerUIter"), comment="Is this the wrapIter? (GIR)"))
      mod.add(SCSelectB32(dst=sgpr(incTmpLo), src0=wrapLo, src1=sgpr(incSgprName),
                          comment="select WrapU or normal inc (lo)"))
      mod.add(SCSelectB32(dst=sgpr(incTmpHi), src0=wrapHi, src1=0,
                          comment="select WrapU or normal inc (hi)"))
      # (2) APPLY
      mod.add(SAddU64(dst=sgpr(f"{tdmGroup0}+2", 2), src0=sgpr(f"{tdmGroup0}+2", 2),
                      src1=sgpr(incTmpLo, 2), comment="TDM addr += inc (with wrap, 64-bit)"))
  else:
    # `incSgprName` is the fused stride when fused; `GlobalReadIncs{tc}` is back in the pool.
    mod.add(comp.incrementGlobalAddr(writer, tdmGroup0, incSgprName))

  # (3) No TDMSplit correction: GIR walks the regions itself and closes the walk with
  # `back=True`, so the descriptor sits at the chunk base and the stride above is the whole
  # advance.  `(#fwd - #back)*splitInc + duInc == stride` is checked as G-WALK in gir/verify.py.
  return mod

# The TDMSplit primitives follow.  `globalReadDo` leaves the descriptor one region ahead for
# `tdmIncrementGir` to subtract back; GIR walks the regions itself instead.


def tdmGirIsIter(writer, kernel, tc) -> bool:
  """Whether this operand's descriptor uses the iterate-enabled (group2/group3) form.

  True for the operand's own `_TDMIterateMode` or any peer's: an aliased descriptor is one
  register set, so either member asking for iterate mode puts the pair in it.  The peers are the
  operand's Φ group, not a fixed A/B pair -- under `A_MX` the group is (A, MXSA, MXSB) and B is
  unfused with its own descriptor.
  """
  members, _ = writer.tdmFuseGroupOf(kernel, tc)
  isIter = kernel.get("_TDMIterateMode%s" % tc, False)
  for _m in (members or ()):
    if _m != tc:
      isIter = isIter or kernel.get("_TDMIterateMode%s" % _m, False)
  return bool(isIter)


def tdmSplitDim1ForRegion(writer, mod, kernel, tP, hrIdx, h0Idx, h1Idx):
  """Recompute (H0, H1), the full and second-half dim1 extents, from the live descriptor.

  Multi-wave only.  A split tile's two storage regions differ along dim1: region 1 covers
  `H0 - halfRows`, clamped at 0.  `halfRows` is per-wave and parity-selected, since even waves
  move A and odd move B off one aliased descriptor.  Reading the live `Group1+2/+3` rather than a
  cached value makes this safe to call at any point in the walk.
  """
  group1 = f"tdm{tP['tensorChar']}Group1"
  # Parity means A-vs-B only while the two share this descriptor; a Φ grouping that separates
  # them leaves one tensor here, and every wave shortens by its own span.
  _members, _ = writer.tdmFuseGroupOf(kernel, tP["tensorChar"])
  if not (_members and "A" in _members and "B" in _members):
    halfRowsA = halfRowsB = writer.tdmSplitGeometry(kernel, tP).dim1SpanPerRegion
  else:
    halfRowsA = writer.tdmSplitGeometry(kernel, writer.tPA).dim1SpanPerRegion
    halfRowsB = writer.tdmSplitGeometry(kernel, writer.tPB).dim1SpanPerRegion
  if halfRowsA == halfRowsB:
    mod.add(SMovB32(sgpr(hrIdx), halfRowsA, "halfRows"))
  else:
    if writer.isTdmWaveIdxLive(kernel):
      writer._emitTdmWaveParitySCC(mod, kernel)
    else:
      with writer.allocTmpSgpr(1, tag="tdmSplitDim1ParityGir") as waveIdTmp:
        writer._emitTdmWaveParitySCC(mod, kernel, waveIdTmp.idx)
    mod.add(SMovB32(sgpr(hrIdx), halfRowsA, "halfRows = A"))
    mod.add(SCSelectB32(sgpr(hrIdx), halfRowsB, sgpr(hrIdx), "halfRows = parity ? B : A"))
  mod.add(SLShiftRightB32(sgpr(h0Idx), hex(16), sgpr(f"{group1}+2"), "H0 = dim1 lo"))
  mod.add(SLShiftLeftB32(sgpr(h1Idx), hex(16), sgpr(f"{group1}+3"), "H0 hi << 16"))
  mod.add(SOrB32(sgpr(h0Idx), sgpr(h0Idx), sgpr(h1Idx), "H0 = full dim1"))
  mod.add(SSubU32(sgpr(h1Idx), sgpr(h0Idx), sgpr(hrIdx), "H1 = H0 - halfRows"))
  mod.add(SCSelectB32(sgpr(h1Idx), 0, sgpr(h1Idx), "clamp H1 to 0"))
  return group1


def tdmLoadRegionGir(writer, kernel, tP, memToken, region: int = 0) -> Module:
  """One `tensor_load_to_lds` for one region, carrying the token GIR assigned to it.

  No region loop and no descriptor mutation: the caller has already placed the descriptor via
  `tdmRegionIncrementGir`, and `globalReadDo`'s internal load-advance-load would double-walk.
  `memToken` is this region's buffer, so each region gets its own token.

  The StreamK tail LDS bank select and the `enableTDM{A,B}` guard raise rather than being
  dropped: both lie outside what GIR models, and a load against the wrong bank is silent.
  """
  tc = tP["tensorChar"]
  if not kernel.get(f"enableTDM{tc}", False):
    raise NotImplementedError(
        f"tdmLoadRegionGir({tc}): enableTDM{tc} is off, so the scaffold's arm emits nothing "
        f"while this primitive would emit a load (#236).")
  if writer.states.inTailLoop and not kernel["1LDSBuffer"] and kernel["StreamK"]:
    raise NotImplementedError(
        f"tdmLoadRegionGir({tc}): StreamK selects the tail's LDS bank per tile (PAP bank "
        f"select), which GIR does not model, and StreamK is rejected under UseLoopModel "
        f"(#236).  The ordinary tail rebinds both pointers itself and needs no refusal.")
  comp: TensorDataMoverLoad = TensorDataMoverLoad.find(writer)
  comp.setMemToken([int(t) for t in memToken] if memToken is not None
                   else [writer.states.ldsTensorTokenIdx])
  isIter = tdmGirIsIter(writer, kernel, tc)
  g2 = f"tdm{tc}Group2" if isIter else None
  g3 = f"tdm{tc}Group3" if isIter else None
  if kernel["NumWaves"] > 1 and region:
    # A non-zero region has its own dim1 under multi-wave, so the load is bracketed: narrow to
    # H1, issue, restore H0.  The descriptor is loop-carried, so leaving it narrowed would
    # shrink every later load, region 0 included.
    mod = Module(f"GIR region {region} load {tc}")
    with writer.allocTmpSgpr(3, tag="tdmSplitDim1Gir") as t:
      h0, h1, hr = t.idx, t.idx + 1, t.idx + 2
      group1 = tdmSplitDim1ForRegion(writer, mod, kernel, tP, hr, h0, h1)
      mod.add(comp.setTensorDim1(group1, h1, writer))
      mod.add(comp.issueLoad(f"tdm{tc}Group0", f"tdm{tc}Group1", g2, g3))
      mod.add(comp.setTensorDim1(group1, h0, writer))
    return mod
  return comp.issueLoad(f"tdm{tc}Group0", f"tdm{tc}Group1", g2, g3)


def tdmDescriptorEnableGir(writer, kernel, tP, enabled: bool) -> Module:
  """Set the descriptor `count` for the waves carrying `tP`: 0 NULLs the load, 1 leaves it valid.

  A fused group whose members own different region counts NULLs the short member on the steps it
  does not move.  The SGPRs are per-wave, so the write is scoped to that member's waves.  Only
  `tc`'s arm writes anything, which lowers to the ownership test plus one `s_cselect` -- no
  branch and no label, unlike `tdmWaveSelect`'s two-sided form.
  """
  tc = tP["tensorChar"]
  mod = Module("tdmDescriptorEnableGir%s" % tc)
  if not writer.isTdmWaveSeparated(kernel):
    return mod
  members, ranges = writer.tdmFuseGroupOf(kernel, tc)
  if not members:
    return mod

  # Null the descriptor the set's LOAD reads.  `tdmDescriptorOwners` names the register owner,
  # which is not always the issuer, so ask who issues first.  Nulling anything else is a no-op.
  owner = next((m for m in members if writer.tdmIssuesOwnLoad(kernel, m)),
               writer.tdmDescriptorOwners(kernel).get(tc, tc))
  dst = sgpr(f"tdm{owner}Group0+0")
  value = 1 if enabled else 0
  comment = "%s descriptor %s" % (tc, "valid" if enabled else "NULL")

  def _select(owns):
    """`dst = owns ? value : dst`, reading the SCC just set."""
    src0, src1 = (value, dst) if owns else (dst, value)
    return SCSelectB32(dst=dst, src0=src0, src1=src1, comment=comment)

  if ranges is None:
    # Parity: member 0 is the even waves, member 1 the odd, and SCC carries "odd".
    writer._emitTdmWaveParitySCCAuto(mod, kernel, comment="wave parity",
                                   tmpTag="tdmDescriptorEnableGir_waveIdTmp")
    mod.add(_select(members.index(tc) == 1))
    return mod

  # Contiguous wave ranges: `first <= wId < first+count` is one UNSIGNED compare of `wId - first`,
  # because a wave below `first` underflows past `count`.
  first, count = next((f, c) for mi, f, c in ranges if members[mi] == tc)
  owns = f"{tc} owns waves {first}..{first + count - 1}?"
  if writer.isTdmWaveIdxLive(kernel) and not first:
    mod.add(SCmpLtU32(src0=sgpr("WaveIdx"), src1=count, comment=owns))
    mod.add(_select(True))
    return mod
  with writer.allocTmpSgpr(1, tag="tdmDescriptorEnableGir_waveIdTmp") as tmp:
    wid = writer._emitTdmWaveIdx(mod, kernel, tmp.idx)
    if first or wid != tmp.idx:
      mod.add(SSubU32(dst=sgpr(tmp.idx), src0=sgpr(wid), src1=first, comment=f"wId - {first}"))
    mod.add(SCmpLtU32(src0=sgpr(tmp.idx), src1=count, comment=owns))
    mod.add(_select(True))
  return mod


def tdmMaskShortMembers(writer, mod, kernel, tc: str, region: int, enabled: bool):
  """NULL (or restore) the Φ group members of `tc` that do not reach storage region `region`.

  A group is ONE descriptor, so a member with fewer regions rides a region step it does not
  carry and loads at a displaced address.  GIR brackets its copies with this
  (`gir_to_rocisa._emit_copy_masked`); `globalReadDo`'s own region loop owes the same, and it
  is that path the tail runs.

  `tc` ITSELF can be short: the walk length is the set's, so the arm issuing it need not be the
  member that reaches the last region.
  """
  members, _ = writer.tdmFuseGroupOf(kernel, tc)
  for m in (members or ()):
    if _tdm_split.split_of(kernel, m)[0] <= region:
      mod.add(tdmDescriptorEnableGir(writer, kernel, writer.tdmTpByChar(m), enabled))
  return mod


def tdmRegionIncrementGir(writer, kernel, tP, back: bool = False) -> Module:
  """Move the TDM descriptor by one storage region; `back=True` for the inverse.

  The per-region step `globalReadDo` performs inline between its two loads, lifted out for GIR to
  place.  GIR walks the regions forward and closes the walk with `back=True`, so the descriptor
  ends each chunk where it started and `tdmIncrementGir` applies the plain stride.

  Multi-wave MX-scale and sparse raise: that path recomputes the split increments transiently and
  brackets the second load with `setTensorDim1`, so the advance is not separable from the load.
  """
  tc = tP["tensorChar"]
  if "MXS" in tc or kernel["ProblemType"]["Sparse"]:
    raise NotImplementedError(
        f"tdmRegionIncrementGir({tc}): MX-scale / sparse region splits are out of scope (#106).")
  g0 = f"tdm{tc}Group0"
  mod = Module(f"GIR region {'back' if back else 'fwd'} {tc}")
  if kernel["NumWaves"] > 1:
    # Multi-wave increments are not persistent SGPRs: each parity moves its own member off one
    # aliased descriptor, so both steps are per-tensor and recomputed by wave parity.  `back`
    # recomputes and subtracts, since the temps do not survive the loads in between.
    with writer.allocTmpSgpr(2, tag="tdmSplitIncGir") as incTmp:
      gIncIdx, scratchIdx = incTmp.idx, incTmp.idx + 1
      ldsIncOp = writer._tdmSplitGroupInc(mod, kernel, gIncIdx, scratchIdx, tc)
      if back:
        mod.add(SSubU32(sgpr(f"{g0}+1"), sgpr(f"{g0}+1"), ldsIncOp, f"tdm{tc} region -= lds"))
        mod.add(SSubU32(sgpr(f"{g0}+2"), sgpr(f"{g0}+2"), sgpr(gIncIdx), f"tdm{tc} region -= glb"))
        mod.add(SSubBU32(sgpr(f"{g0}+3"), sgpr(f"{g0}+3"), 0, f"tdm{tc} region borrow"))
      else:
        mod.add(SAddU32(sgpr(f"{g0}+1"), sgpr(f"{g0}+1"), ldsIncOp, f"tdm{tc} region += lds"))
        mod.add(SAddU32(sgpr(f"{g0}+2"), sgpr(f"{g0}+2"), sgpr(gIncIdx), f"tdm{tc} region += glb"))
        mod.add(SAddCU32(sgpr(f"{g0}+3"), sgpr(f"{g0}+3"), 0, f"tdm{tc} region carry"))
    return mod
  lds = sgpr(f"tdm{tc}LdsSplitIncs")
  glb = sgpr(f"tdm{tc}GlobalSplitIncs")
  if back:
    mod.add(SSubU32(sgpr(f"{g0}+1"), sgpr(f"{g0}+1"), lds, f"tdm{tc} region -= lds split"))
    mod.add(SSubU32(sgpr(f"{g0}+2"), sgpr(f"{g0}+2"), glb, f"tdm{tc} region -= global split"))
    mod.add(SSubBU32(sgpr(f"{g0}+3"), sgpr(f"{g0}+3"), 0, f"tdm{tc} region borrow"))
  else:
    mod.add(SAddU32(sgpr(f"{g0}+1"), sgpr(f"{g0}+1"), lds, f"tdm{tc} region += lds split"))
    mod.add(SAddU32(sgpr(f"{g0}+2"), sgpr(f"{g0}+2"), glb, f"tdm{tc} region += global split"))
    mod.add(SAddCU32(sgpr(f"{g0}+3"), sgpr(f"{g0}+3"), 0, f"tdm{tc} region carry"))
  return mod
