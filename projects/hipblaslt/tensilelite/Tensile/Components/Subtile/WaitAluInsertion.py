# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# Post-schedule s_wait_alu insertion for the LR offset-swap RAW hazard.
#
# In SCHED_MODE 2 the LR swap's v_xor -> ds_read RAW on the offset VGPR is no
# longer guarded by hardware.  A wait is only needed when the read is too close
# to the swap: if MIN_MMA_BEFORE_LR_READ WMMAs separate them the v_xor has
# already retired, otherwise we insert one SWaitAlu(va_vdst=0).
#
#
################################################################################

from rocisa.code import Module, TextBlock
from rocisa.container import RegisterContainer
from rocisa.instruction import (
    DSLoadB128, LocalReadInstruction, MFMAInstruction, MXMFMAInstruction,
    SBarrier, SMovB32, SSetVgprMsb, SWaitAlu, SWaitCnt, SWaitTensorcnt,
    VAddU32, VCmpGEU32, VCndMaskB32, VSubU32, VXorB32,
)

# Number of WMMAs that must issue between a swap XOR and its dependent ds_read
# for the v_xor latency to be fully hidden (no s_wait_alu required).
# Conversative value based on s_wait_alu latency.
MIN_MMA_BEFORE_LR_READ = 4

_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


def _mmaSrcVgprs(inst):
  """VGPR indices read by an MFMA/MXMFMA (A/B matrices + MX scales)."""
  idxs = set()
  for attr in ("a", "b", "mxsa", "mxsb"):
    idxs.update(_vgprIndices(getattr(inst, attr, None)))
  return idxs


def _isWaitLR(inst):
  return (isinstance(inst, SWaitCnt) and inst.dscnt is not None
          and "Wait for LR to complete" in (getattr(inst, "comment", "") or ""))


def _vgprIndices(container):
  """Concrete VGPR indices a register operand covers, or empty for non-VGPRs."""
  if not isinstance(container, RegisterContainer) or container.regType != 'v':
    return ()
  return range(container.regIdx, container.regIdx + container.regNum)


def setMatrixReuse(module, writer, kernel, matrix):
  """Enable the gfx1250 WMMA matrix reuse hint for *matrix* ('a' or 'b').

  No-op unless gfx1250 (HasWmmaArbStallBit).  Mutates instructions in place.
  """
  assert matrix in ('a', 'b'), f"matrix must be 'a' or 'b', got {matrix!r}"
  if not writer.states.archCaps.get("HasWmmaArbStallBit", False):
    return module

  operand = matrix.lower()          # 'a' or 'b'
  flag    = f"reuse{matrix.upper()}" # 'reuseA' or 'reuseB'

  mmas = [inst for inst in module.flatitems() if _isMMA(inst)]
  for prev, cur in zip(mmas, mmas[1:]):
    cur_op  = getattr(cur, operand)
    prev_op = getattr(prev, operand)
    if tuple(_vgprIndices(cur_op)) and _vgprIndices(cur_op) == _vgprIndices(prev_op):
      setattr(prev, flag, True)
  return module


def insertLRSwapRawWaitAlu(module, writer, kernel):
  """Guard the LR offset-swap -> ds_read RAW hazard.

  For each swap window, if fewer than MIN_MMA_BEFORE_LR_READ WMMAs separate the
  swap XORs from the first dependent ds_read, insert one SWaitAlu(va_vdst=0)
  before that read (which validates all pending swaps).  No-op unless SCHED_MODE
  2 (HasWmmaArbStallBit) is active.  Returns a rebuilt Module; the input is left
  untouched.
  """
  # gfx1250 only
  if not writer.states.archCaps.get("HasWmmaArbStallBit", False):
    return module

  dirty = set()       # offset VGPR indices written by a swap-XOR, not yet drained
  mmaSinceSwap = 0    # WMMAs issued since the first un-drained swap XOR
  result = Module(module.name)

  for inst in module.flatitems():
    # Producer: a swap-XOR writes an LR offset VGPR.
    if isinstance(inst, VXorB32):
      if not dirty:
        mmaSinceSwap = 0
      dirty.update(_vgprIndices(inst.dst))
      result.add(inst)
      continue

    if dirty and _isMMA(inst):
      mmaSinceSwap += 1

    # Consumer: a ds_read whose address source is a swapped offset VGPR.
    if dirty and isinstance(inst, LocalReadInstruction):
      # DSLoad.getParams() -> [dst, addrSrc]; the address source is index 1.
      params = inst.getParams()
      addr = params[1] if len(params) > 1 else None
      if any(idx in dirty for idx in _vgprIndices(addr)):
        if mmaSinceSwap < MIN_MMA_BEFORE_LR_READ:
          result.add(SWaitAlu(va_vdst=0, comment="wait for LR offset swap to complete"))
        dirty.clear()

    result.add(inst)

  return result


def _isSwapXor(inst):
  if not isinstance(inst, VXorB32) or not inst.srcs:
    return False
  dst = tuple(_vgprIndices(inst.dst))
  return bool(dst) and dst == tuple(_vgprIndices(inst.srcs[0]))


# Scale-ring rotate (see emitLRBufferSwapOffsets): optional legacy SMov stride/
# span, then VAdd, VSub, VCmp, VCndMask. Keep SMov so older kernels still hoist.
_SCALE_RING_SWAP_TYPES = (SMovB32, VAddU32, VSubU32, VCmpGEU32, VCndMaskB32)


def _isLRSwapInst(inst):
  """True for A/B xor toggles or scale-ring rotate ops that follow READY."""
  return _isSwapXor(inst) or isinstance(inst, _SCALE_RING_SWAP_TYPES)


def _swapDstVgprs(inst):
  """VGPR indices an LR swap overwrites (xor / add / cndmask destinations)."""
  if _isSwapXor(inst) or isinstance(inst, (VAddU32, VCndMaskB32, VSubU32)):
    return set(_vgprIndices(inst.dst))
  return set()


def _lrSwapChunks(swaps):
  """Split a hoistable swap block into atomic chunks (one xor or one rotate)."""
  chunks, i = [], 0
  while i < len(swaps):
    # Leading comments attach to the next real op.
    start = i
    while i < len(swaps) and isinstance(swaps[i], TextBlock):
      i += 1
    if i >= len(swaps):
      break
    if _isSwapXor(swaps[i]):
      chunks.append(swaps[start:i + 1])
      i += 1
      continue
    # Scale rotate: [SMov*] VAdd VSub VCmp VCndMask
    j = i
    while j < len(swaps) and isinstance(swaps[j], SMovB32):
      j += 1
    if (j + 3 < len(swaps)
        and isinstance(swaps[j], VAddU32)
        and isinstance(swaps[j + 1], VSubU32)
        and isinstance(swaps[j + 2], VCmpGEU32)
        and isinstance(swaps[j + 3], VCndMaskB32)):
      chunks.append(swaps[start:j + 4])
      i = j + 4
      continue
    # Fallback: single instruction.
    chunks.append(swaps[start:i + 1])
    i += 1
  return chunks


def hoistLRSwapAboveWaitTDM(module, writer, kernel):
  """Move LR offset swaps that follow a TDM wait + barrier to before the wait.

  Covers A/B xor toggles and scale-ring rotates (depth >= 3).  The swaps are
  per-wave and only redirect the next ds_reads, so they need neither the TDM
  data nor the barrier; after the barrier they sit on the loop's critical path,
  before the wait they hide under the TDM latency — matching the ldsv7-w1
  prototype READY order.  A swap block only moves if every ds_read of the
  offsets it overwrites is drained by an s_wait_dscnt 0 before the wait (no WAR
  on SCHED_MODE 2).  TDM kernels only.  Returns a rebuilt Module.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())

  def _nextInst(i):
    while i < len(items) and isinstance(items[i], TextBlock):
      i += 1
    return i

  def _drained(emitted, dsts):
    for inst in reversed(emitted):
      if isinstance(inst, SWaitCnt) and inst.dscnt == 0:
        return True
      if isinstance(inst, LocalReadInstruction):
        params = inst.getParams()
        addr = params[1] if len(params) > 1 else None
        if any(idx in dsts for idx in _vgprIndices(addr)):
          return False
    return False

  result = []
  i = 0
  while i < len(items):
    inst = items[i]
    if isinstance(inst, SWaitTensorcnt):
      b = _nextInst(i + 1)
      if b < len(items) and isinstance(items[b], SBarrier):
        end = b + 1
        while end < len(items) and (
            isinstance(items[end], TextBlock) or _isLRSwapInst(items[end])):
          end += 1
        while end > b + 1 and isinstance(items[end - 1], TextBlock):
          end -= 1
        swaps = items[b + 1:end]
        dsts = {idx for s in swaps for idx in _swapDstVgprs(s)}
        if dsts and _drained(result, dsts):
          result.extend(swaps)
          result.extend(items[i:b + 1])
          i = end
          continue
    result.append(inst)
    i += 1

  hoisted = Module(module.name)
  for inst in result:
    hoisted.add(inst)
  return hoisted


def spreadLRSwapAcrossPriorMfma(module, writer, kernel):
  """Distribute a hoisted READY LR-swap block across prior MFMA gaps.

  After hoistLRSwapAboveWaitTDM the swaps sit as one blob immediately before
  s_wait_tensorcnt.  That leaves the VALU pipe busy and the MFMA pipe idle
  between the last few WMMAs and the next ds_loads.  Spread one atomic chunk
  (xor toggle or scale-ring rotate) after each preceding MFMA, stopping at the
  most recent s_wait_dscnt 0 so WAR on the offset VGPRs stays closed.  TDM
  kernels only.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  out = []
  i = 0
  while i < len(items):
    # Detect [swaps...][SWaitTensorcnt][SBarrier] that hoist produced.
    if isinstance(items[i], SWaitTensorcnt):
      out.append(items[i])
      i += 1
      continue

    if not _isLRSwapInst(items[i]) and not isinstance(items[i], TextBlock):
      out.append(items[i])
      i += 1
      continue

    # Peek: is this the start of a swap block ending at wait+barrier?
    j = i
    while j < len(items) and (
        isinstance(items[j], TextBlock) or _isLRSwapInst(items[j])):
      j += 1
    if (j + 1 >= len(items)
        or not isinstance(items[j], SWaitTensorcnt)
        or not isinstance(items[j + 1], SBarrier)):
      out.append(items[i])
      i += 1
      continue

    swaps = items[i:j]
    chunks = _lrSwapChunks(swaps)
    if len(chunks) < 2:
      out.extend(swaps)
      i = j
      continue

    # Window of MFMAs in `out` after the last dscnt=0 (or start).
    drainAt = -1
    for k in range(len(out) - 1, -1, -1):
      if isinstance(out[k], SWaitCnt) and out[k].dscnt == 0:
        drainAt = k
        break
    mfmaIdx = [k for k in range(drainAt + 1, len(out)) if _isMMA(out[k])]
    if not mfmaIdx:
      out.extend(swaps)
      i = j
      continue

    # nM+1 slots: before the first MFMA (hides under whatever preceded the
    # drain — e.g. FREE TDM) and after each MFMA in the drained window.
    nM, nC = len(mfmaIdx), len(chunks)
    nSlots = nM + 1
    slots = [[] for _ in range(nSlots)]
    for ci, ch in enumerate(chunks):
      slots[min(ci * nSlots // nC, nSlots - 1)].extend(ch)

    rebuilt = out[:drainAt + 1]
    rebuilt.extend(slots[0])
    mi = 0
    for k in range(drainAt + 1, len(out)):
      rebuilt.append(out[k])
      if mi < nM and k == mfmaIdx[mi]:
        rebuilt.extend(slots[mi + 1])
        mi += 1
    out = rebuilt
    i = j

  spread = Module(module.name)
  for inst in out:
    spread.add(inst)
  return spread


def _instText(inst):
  if isinstance(inst, TextBlock):
    return str(inst)
  return getattr(inst, "comment", "") or ""


def _preFreeTDMSALUChunks(blob):
  """Split pre-FREE A/B/MXS addr+rebuild into atomic SGPR-scratch chunks.

  Group1 packing and scale-ring wrap stage through shared temps, so each
  tensor's update must stay contiguous (same cuts as patch_salu_slim).
  """
  def find(pred, start=0):
    for i in range(start, len(blob)):
      if pred(blob[i]):
        return i
    return None

  a_addr = find(lambda x: "TDM addr update: A" in _instText(x))
  a_swap = find(lambda x: "swap A LDS" in _instText(x), a_addr or 0)
  b_addr = find(lambda x: "TDM addr update: B" in _instText(x), a_swap or 0)
  b_swap = find(lambda x: "swap B LDS" in _instText(x), b_addr or 0)
  sa = find(lambda x: "addr update: MXSA" in _instText(x), b_swap or 0)
  sb = find(lambda x: "addr update: MXSB" in _instText(x), sa or 0)
  cuts = [i for i in (a_addr, a_swap, b_addr, b_swap, sa, sb) if i is not None]
  if not cuts:
    return [blob] if blob else []
  if cuts[0] != 0:
    cuts = [0] + cuts
  cuts = sorted(set(cuts))
  chunks = []
  for i, c in enumerate(cuts):
    end = cuts[i + 1] if i + 1 < len(cuts) else len(blob)
    chunk = [x for x in blob[c:end] if not isinstance(x, TextBlock)]
    if chunk:
      chunks.append(chunk)
  return chunks


def spreadPreFreeTDMSALU(module, writer, kernel):
  """Interleave pre-FREE TDM addr/descriptor SALU into the prior WMMA stream.

  hoistTDMSetup already lifts A/B rebuild above the FREE barrier, but the
  instruction scheduler still emits that SALU as one contiguous blob after the
  trailing K-group WMMAs — leaving the MFMA pipe idle.  Split into per-tensor
  chunks and issue them after preceding WMMAs (descriptor SGPRs are not WMMA
  operands).  TDM kernels only.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  out = []
  i = 0
  while i < len(items):
    if not (isinstance(items[i], TextBlock)
            and "TDM addr update: A" in _instText(items[i])):
      out.append(items[i])
      i += 1
      continue

    # Blob runs through the wait_dscnt that precedes the FREE barrier.
    wait = None
    for t in range(i + 1, len(items) - 1):
      if (isinstance(items[t], SWaitCnt) and items[t].dscnt == 0
          and isinstance(items[t + 1], SBarrier)):
        wait = t
        break
    if wait is None:
      out.append(items[i])
      i += 1
      continue

    blob = items[i:wait]
    chunks = _preFreeTDMSALUChunks(blob)
    if len(chunks) < 4:
      out.extend(blob)
      out.append(items[wait])
      i = wait + 1
      continue

    # Trailing WMMA/ds_read window only — stop at wait/barrier/other, and keep
    # at most one cover MFMA per chunk so we don't push SALU into early K-group
    # WMMAs (that regresses vs leaving the blob after the last few WMMAs).
    win_start = len(out)
    while win_start > 0:
      prev = out[win_start - 1]
      if isinstance(prev, SWaitCnt) or isinstance(prev, SBarrier):
        break
      if _isMMA(prev) or isinstance(prev, (LocalReadInstruction, TextBlock)):
        win_start -= 1
        continue
      break
    window = out[win_start:]
    mfmaIdx = [k for k, x in enumerate(window) if _isMMA(x)]
    # Use only the last N MFMAs as cover (N ~= number of SALU chunks).
    cover = min(len(mfmaIdx), max(len(chunks), 4))
    if cover < 4:
      out.extend(blob)
      out.append(items[wait])
      i = wait + 1
      continue
    mfmaIdx = mfmaIdx[-cover:]
    # Rebase window to start at the first cover MFMA (keep earlier WMMAs as-is).
    first_cover = mfmaIdx[0]
    kept = window[:first_cover]
    window = window[first_cover:]
    mfmaIdx = [k - first_cover for k in mfmaIdx]

    out = out[:win_start]
    out.extend(kept)
    nM, nC = len(mfmaIdx), len(chunks)
    slots = [[] for _ in range(nM)]
    for ci, ch in enumerate(chunks):
      slots[min(ci * nM // nC, nM - 1)].extend(ch)

    mi = 0
    for k, x in enumerate(window):
      out.append(x)
      if mi < nM and k == mfmaIdx[mi]:
        out.extend(slots[mi])
        mi += 1
    out.append(items[wait])
    i = wait + 1

  spread = Module(module.name)
  for inst in out:
    spread.add(inst)
  return spread


def _emitWithMsb(items, last_msb):
  """Emit (msb, inst) pairs, inserting SSetVgprMsb only when the bank changes."""
  out = []
  for msb, inst in items:
    if msb is not None and msb != last_msb:
      out.append(SSetVgprMsb(msb, comment="ds-spread msb"))
      last_msb = msb
    out.append(inst)
  return out, last_msb


def spreadPostReadyDsLoads(module, writer, kernel):
  """Interleave the post-READY ds_load_b128 burst with following WMMAs.

  After READY the scheduler emits a solid A/B b128 prefetch before K-group
  WMMAs resume, so ~half the loads sit with nothing to hide behind.  Move that
  leading b128 burst into the following WMMA stream (1:1 style), matching
  patch_ds_spread: scale ds_load_b32 stays put (WAR on live WMMA scale VGPRs);
  s_set_vgpr_msb prefixes travel with their consumer.  Call only on MAINLOOP
  modules.  TDM kernels only.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  out = []
  i = 0
  current_msb = None
  while i < len(items):
    inst = items[i]
    if isinstance(inst, SSetVgprMsb):
      current_msb = int(inst.vgprMsb)

    # Anchor on READY: wait_tensorcnt + barrier (+ optional LR-swap wait_alu).
    if (isinstance(inst, SWaitTensorcnt)
        and i + 1 < len(items) and isinstance(items[i + 1], SBarrier)):
      out.append(inst)
      out.append(items[i + 1])
      i += 2
      if (i < len(items) and isinstance(items[i], SWaitAlu)
          and "LR offset swap" in (getattr(items[i], "comment", "") or "")):
        out.append(items[i])
        i += 1

      j = i
      pending = None
      msb = current_msb
      burst = []  # (msb, inst)
      while j < len(items):
        if isinstance(items[j], TextBlock):
          j += 1
          continue
        if isinstance(items[j], SSetVgprMsb):
          pending = int(items[j].vgprMsb)
          j += 1
          continue
        if isinstance(items[j], DSLoadB128):
          if pending is not None:
            msb = pending
            pending = None
          burst.append((msb, items[j]))
          j += 1
          continue
        break
      if len(burst) < 4:
        # Not a real burst — fall through so wait_alu / msb / loads emit normally.
        continue

      k = j
      rest = []  # (msb, inst, kind) kind in {W, L128, Lother}
      while k < len(items):
        if isinstance(items[k], TextBlock):
          k += 1
          continue
        if isinstance(items[k], SSetVgprMsb):
          pending = int(items[k].vgprMsb)
          k += 1
          continue
        if _isMMA(items[k]):
          if pending is not None:
            msb = pending
            pending = None
          rest.append((msb, items[k], "W"))
          k += 1
          continue
        if isinstance(items[k], DSLoadB128):
          if pending is not None:
            msb = pending
            pending = None
          rest.append((msb, items[k], "L128"))
          k += 1
          continue
        if isinstance(items[k], LocalReadInstruction):
          if pending is not None:
            msb = pending
            pending = None
          rest.append((msb, items[k], "Lother"))  # scale b32 — do not move
          k += 1
          continue
        break
      # Trailing pending MSB before a foreign instr means the rest stream
      # ended uncleanly — leave the region alone.
      if pending is not None or not any(kind == "W" for _, _, kind in rest):
        continue

      idx = 0
      preamble = []
      while idx < len(rest) and rest[idx][2] != "W":
        preamble.append((rest[idx][0], rest[idx][1]))
        idx += 1
      groups = []
      while idx < len(rest):
        assert rest[idx][2] == "W"
        w = (rest[idx][0], rest[idx][1])
        idx += 1
        trail = []
        while idx < len(rest) and rest[idx][2] != "W":
          trail.append((rest[idx][0], rest[idx][1]))
          idx += 1
        groups.append((w, trail))

      n_w = len(groups)
      slots = [[] for _ in range(n_w)]
      for bi, ld in enumerate(burst):
        slots[min(bi * n_w // len(burst), n_w - 1)].append(ld)

      emitted = list(preamble)
      for gi, (w, trail) in enumerate(groups):
        emitted.append(w)
        emitted.extend(slots[gi])
        emitted.extend(trail)

      text_insts, current_msb = _emitWithMsb(emitted, current_msb)
      end_msb = rest[-1][0]
      if current_msb != end_msb:
        text_insts.append(SSetVgprMsb(end_msb, comment="ds-spread restore msb"))
        current_msb = end_msb

      out.extend(text_insts)
      i = k
      continue

    out.append(inst)
    i += 1

  spread = Module(module.name)
  for inst in out:
    spread.add(inst)
  return spread


def splitWaitLRPerTile(module, writer, kernel):
  """Split each subIterK ``wait_lr`` into per-tile waits with precise dscnt.

  The scheduler emits one ``s_wait_dscnt 0`` before a whole K-group of WMMAs,
  which drains every outstanding LR — including tiles that will not be used
  for many MFMAs.  Replace that with a wait before each tile frontier: when a
  WMMA first needs a load that is still outstanding, emit
  ``s_wait_dscnt N`` where N is the number of *newer* loads not required by
  any WMMA up to and including this one (FIFO: newest may remain).

  Empty outstanding (MAINLOOP entry; PRELOOP LRs invisible) keeps a hard
  ``dscnt=0``.  Call after final LR/WMMA interleave (B-first + dspread).
  """
  items = list(module.flatitems())
  if not any(_isWaitLR(x) for x in items):
    return module

  outstanding = []  # {"dsts": set[int]}
  out = []
  i = 0
  while i < len(items):
    inst = items[i]

    if _isWaitLR(inst):
      # Find the WMMA stream this wait guards (until next wait_lr / foreign).
      j = i + 1
      while j < len(items) and isinstance(items[j], (TextBlock, SSetVgprMsb)):
        j += 1
      # Skip leading non-MMA content (already emitted after we handle the wait).
      body_start = i + 1
      k = body_start
      wmma_idxs = []
      while k < len(items) and not _isWaitLR(items[k]):
        if _isMMA(items[k]):
          wmma_idxs.append(k)
        elif (not isinstance(items[k], (TextBlock, SSetVgprMsb, LocalReadInstruction,
                                        DSLoadB128))
              and wmma_idxs
              and not isinstance(items[k], SWaitAlu)):
          # Stop at foreign ops after we've seen WMMAs (e.g. TDM/swap).
          # Keep scanning loads/WMMAs interleaved; only break on true foreign
          # after the WMMA region would normally end — use next wait_lr only.
          pass
        k += 1
      # Body runs until next wait_lr (or end).
      body_end = k

      if not outstanding or not wmma_idxs:
        out.append(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                            comment="Wait for LR to complete"))
        outstanding = []
        # Emit body as-is (without the original wait).
        for t in range(body_start, body_end):
          if isinstance(items[t], LocalReadInstruction):
            params = items[t].getParams()
            dst = set(_vgprIndices(params[0])) if params else set()
            if dst:
              outstanding.append({"dsts": dst})
          if _isMMA(items[t]):
            srcs = _mmaSrcVgprs(items[t])
            if srcs:
              outstanding = [r for r in outstanding if not (r["dsts"] & srcs)]
          out.append(items[t])
        i = body_end
        continue

      # Walk WMMAs; emit a wait when newly required loads force a stricter drain.
      live = list(outstanding)
      emitted_body_upto = body_start
      did_wait = False
      out_at_wait = len(out)

      def _flush_to(limit):
        nonlocal emitted_body_upto, live
        while emitted_body_upto < limit:
          t = emitted_body_upto
          if isinstance(items[t], LocalReadInstruction):
            params = items[t].getParams()
            dst = set(_vgprIndices(params[0])) if params else set()
            if dst:
              live.append({"dsts": dst})
          if _isMMA(items[t]):
            srcs = _mmaSrcVgprs(items[t])
            if srcs:
              live = [r for r in live if not (r["dsts"] & srcs)]
          out.append(items[t])
          emitted_body_upto += 1

      for wi in wmma_idxs:
        srcs = _mmaSrcVgprs(items[wi])
        needed = [idx for idx, rec in enumerate(live) if rec["dsts"] & srcs]
        if not needed:
          continue
        # Newest M may remain iff none of them are needed by this WMMA.
        needed_set = set(needed)
        M = 0
        for m in range(0, len(live) + 1):
          newest = range(len(live) - m, len(live)) if m else []
          if all(idx not in needed_set for idx in newest):
            M = m
          else:
            break
        _flush_to(wi)
        out.append(SWaitCnt(
            dscnt=M, vlcnt=-1, vscnt=-1,
            comment=("Wait for LR to complete" if M == 0
                     else f"Wait for LR to complete (tile keep {M})")))
        live = live[len(live) - M:] if M else []
        did_wait = True

      if not did_wait:
        # No tracked load overlapped a WMMA — keep the original hard drain.
        out.insert(out_at_wait, SWaitCnt(
            dscnt=0, vlcnt=-1, vscnt=-1,
            comment="Wait for LR to complete"))
        live = []

      _flush_to(body_end)
      outstanding = live
      i = body_end
      continue

    if isinstance(inst, LocalReadInstruction):
      params = inst.getParams()
      dst = set(_vgprIndices(params[0])) if params else set()
      if dst:
        outstanding.append({"dsts": dst})

    if _isMMA(inst):
      srcs = _mmaSrcVgprs(inst)
      if srcs:
        outstanding = [r for r in outstanding if not (r["dsts"] & srcs)]

    out.append(inst)
    i += 1

  result = Module(module.name)
  for inst in out:
    result.add(inst)
  return result


def insertLRSwapWarWaitAlu(module, writer, kernel):
  """Guard the ds_read -> LR offset-swap WAR hazard (gfx1250 SCHED_MODE 2).

  A ds_read reads an LR offset VGPR (address, tracked by VM_VSRC); the swap
  v_xor then overwrites it.  Without the hardware interlock the xor can race
  ahead of that read, so wait before it.  We emit one SWaitAlu(vm_vsrc=0) at
  the first xor whose offset still has an undrained read; the full drain
  covers the rest of the block.  No-op off SCHED_MODE 2.
  """
  if not writer.states.archCaps.get("HasWmmaArbStallBit", False):
    return module

  # VGPR indices with an in-flight ds_read address use, not yet drained.
  pending = set()
  result = Module(module.name)
  for inst in module.flatitems():
    # Producer: a ds_read puts its address VGPR's read in flight on VM_VSRC.
    if isinstance(inst, LocalReadInstruction):
      params = inst.getParams()
      addr = params[1] if len(params) > 1 else None
      pending.update(_vgprIndices(addr))
      result.add(inst)
      continue

    # Drain: s_wait_dscnt 0 retires all in-flight DS (LDS) source reads, so the
    # offset reads are guaranteed complete and can no longer be clobbered.
    if isinstance(inst, SWaitCnt) and inst.dscnt == 0:
      pending.clear()
      result.add(inst)
      continue

    # Consumer: a swap v_xor that overwrites an offset VGPR with a pending read
    # is the WAR.  Drain once (covers every pending offset), then clear.
    if isinstance(inst, VXorB32) and any(idx in pending for idx in _vgprIndices(inst.dst)):
      result.add(SWaitAlu(vm_vsrc=0,
                          comment="wait for LR offset read before swap (WAR)"))
      pending.clear()

    result.add(inst)
  return result
