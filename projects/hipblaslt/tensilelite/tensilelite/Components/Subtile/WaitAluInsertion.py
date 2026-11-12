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

from rocisa.code import Label, Module, TextBlock
from rocisa.container import RegisterContainer
from rocisa.instruction import (
    BranchInstruction, DSLoadB128, GlobalPrefetchB8, LocalReadInstruction,
    MFMAInstruction, MXMFMAInstruction, SBarrier, SCmpLeU32, SMovB32,
    SSetVgprMsb, SWaitAlu, SWaitCnt, SWaitTensorcnt, TensorLoadToLds, VAddU32,
    VCmpGEU32, VCndMaskB32, VSubU32, VXorB32,
)

# gemm_wt SIG_EARLY: WMMAs between WGP signal and wait so the ~14-cyc wait
# overlaps an XDL queue refilled after wait_dscnt/tensorcnt drained it.
# Default 2: n=4 with READY forward-hoist was 0.92× (logs/20261029-202659) —
# delaying post-READY ds_loads hurt more than the wait hide gained. Prefer
# FREE backward-pull; keep READY forward off unless explicitly requested.
SIG_EARLY_WMMA_DEFAULT = 2
SIG_EARLY_ALLOW_FORWARD_DEFAULT = False

# Proto FREE-hole: barrier immediately after last-consumer WMMAs.
# ``wait_dscnt ; WMMAs ; barrier`` (dscnt slid earlier; wait not between signal/WMMAs).
# n=1 best so far: logs/20261029-203847 ~1.015× vs esmBit2 (noisy, 4/8).
FREE_HOLE_WMMA_DEFAULT = 1

# Number of WMMAs that must issue between a swap XOR and its dependent ds_read
# for the v_xor latency to be fully hidden (no s_wait_alu required).
# Conversative value based on s_wait_alu latency.
MIN_MMA_BEFORE_LR_READ = 4

# gemm_wt pf_tdm_gap proxy: keep GL2 prefetch clear of the TDM 256 B/clk stream.
# 8 (was 24): exclusion after tensor_load. Placement is still after the cover's
# last post-gap LR — earliest-hole (gap8) was 0.979× vs slimBias
# (logs/20261102-183606); last-LR keeps PF out of the live TDM stream.
GL2_PF_TDM_GAP_INSTS = 8

_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


def _isGL2PrefetchBlockStart(inst):
  """True for the end-of-K ``s_cmp_le`` that opens a GL2 Prefetch Inc block."""
  if not isinstance(inst, SCmpLeU32):
    return False
  cmt = getattr(inst, "comment", "") or ""
  return "PGL" in cmt or "PGR" in cmt


def _isGL2PrefetchIssueInst(inst):
  """True for a real or unit-test stand-in ``global_prefetch_b8``."""
  if isinstance(inst, GlobalPrefetchB8):
    return True
  cmt = getattr(inst, "comment", "") or ""
  return "global_prefetch_b8" in cmt


def _isGL2PrefetchBodyInst(inst):
  """Instructions that belong to the wave-gated Inc / Issue bodies."""
  if _isGL2PrefetchIssueInst(inst):
    return True
  if isinstance(inst, (TextBlock, Label, BranchInstruction)):
    return True
  if isinstance(inst, SSetVgprMsb):
    # Prefetch addr VGPRs may need an msb primer immediately before the issue.
    return True
  name = type(inst).__name__
  # SCMovB32 Inc zero, VAddCO addr bump, wave-gate SCmp/SBranch, SNop pads.
  if name in ("SCMovB32", "SCmpEQU32", "SCBranchSCC0", "SNop",
              "VAddCOU32", "VAddCCOU32", "SMovB32"):
    return True
  cmt = getattr(inst, "comment", "") or ""
  return "GL2" in cmt or "gl2_pf" in cmt


def _collectGL2PrefetchBlock(items, start):
  """Return exclusive end index of the Inc+Issue wall starting at ``start``."""
  j = start + 1
  seen_pf = False
  while j < len(items):
    inst = items[j]
    if _isGL2PrefetchIssueInst(inst):
      seen_pf = True
      j += 1
      continue
    if _isGL2PrefetchBodyInst(inst):
      j += 1
      continue
    # Trailing skip label after last gated body.
    if isinstance(inst, Label) and "gl2_pf" in (inst.getLabelName()
                                                if hasattr(inst, "getLabelName")
                                                else str(inst)):
      j += 1
      continue
    break
  # A wall cut short by an interleaved WMMA (no issue reached) must not move:
  # lifting only its head would reorder the end-of-K zero past later Incs.
  return j if seen_pf else start


def spreadGL2PrefetchIntoWmma(module, writer, kernel,
                              tdmGapInsts=GL2_PF_TDM_GAP_INSTS):
  """Move the post-GR GL2 Inc+Issue wall into post-LR WMMA holes.

  gemm_wt keeps ``global_prefetch_b8`` clear of the TDM 256 B/clk stream
  (``pf_tdm_gap``).  Tensile bolts Inc+Issue onto the last GR (``tensor_load``
  under TDMSingleIssuer).  Lift the PrefetchGL2 block and reinsert after the
  **last** post-gap ds_load that still has a following WMMA (proto: after the
  block's LRs).  Earliest-hole placement with gap=8 collided with the live TDM
  stream (logs/20261102-183606, 0.979×).  No-op if PrefetchGL2 is off, the
  block is missing, or no safe cover window exists.
  """
  if not kernel.get("PrefetchGL2", 0):
    return module
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  changed = False
  # Re-scan after each move so indices stay valid (even/odd bodies are
  # separate modules; a single MAINLOOP copy usually has one PF wall).
  while True:
    start = end = None
    for i, inst in enumerate(items):
      if _isGL2PrefetchBlockStart(inst):
        e = _collectGL2PrefetchBlock(items, i)
        if e > i + 1:
          start, end = i, e
          break
    if start is None:
      break

    tdm = None
    for j in range(start - 1, -1, -1):
      if isinstance(items[j], TensorLoadToLds):
        tdm = j
        break
    if tdm is None:
      break

    next_tdm = end
    while next_tdm < len(items) and not isinstance(items[next_tdm], TensorLoadToLds):
      next_tdm += 1

    gap_at = tdm + 1
    counted = 0
    while gap_at < next_tdm and counted < tdmGapInsts:
      if gap_at == start:
        gap_at = end
        continue
      inst = items[gap_at]
      if not isinstance(inst, (TextBlock, Label, SSetVgprMsb)):
        counted += 1
      gap_at += 1

    last_lr = None
    k = gap_at
    while k < next_tdm:
      if start <= k < end:
        k = end
        continue
      if isinstance(items[k], (DSLoadB128, LocalReadInstruction)):
        last_lr = k
      k += 1
    if last_lr is None:
      break

    place = last_lr + 1
    while place < next_tdm:
      if start <= place < end:
        place = end
        continue
      if _isMMA(items[place]):
        break
      place += 1
    else:
      break

    if start >= gap_at and start <= last_lr + 1 and end <= place:
      break
    if start == place:
      break

    block = items[start:end]
    del items[start:end]
    if place > start:
      place -= (end - start)
    items[place:place] = block
    changed = True
    break

  if not changed:
    return module
  out = Module(module.name)
  for inst in items:
    out.add(inst)
  return out


def _mmaSrcVgprs(inst):
  """VGPR indices read by an MFMA/MXMFMA (A/B matrices + MX scales)."""
  idxs = set()
  for attr in ("a", "b", "mxsa", "mxsb"):
    idxs.update(_vgprIndices(getattr(inst, attr, None)))
  return idxs


def _isWaitLR(inst):
  """True for LR / fine-dscnt waits this pass owns (not READY tensorcnt)."""
  if not (isinstance(inst, SWaitCnt) and inst.dscnt is not None):
    return False
  cmt = getattr(inst, "comment", "") or ""
  return ("Wait for LR to complete" in cmt
          or "tiles of the next WMMA landed" in cmt
          or "tile keep" in cmt)


# Hardware dscnt saturates here; N >= this is a no-op (gemm_wt DSCNT_MAX).
_DSCNT_MAX = 63


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


def _isTDMSingleIssuerLoad(inst):
  return (isinstance(inst, TensorLoadToLds)
          and "TDMSingleIssuer" in _instText(inst))


def _isTDMSingleIssuerAdvanceCmp(inst):
  """Wave-gate cmp that opens one issuer's post-load advance chunk."""
  t = _instText(inst)
  return "TDMSingleIssuer:" in t and "advance" in t


def _isTDMSingleIssuerIssueCmp(inst):
  """Wave-gate cmp that opens one issuer's load block."""
  t = _instText(inst)
  return "TDMSingleIssuer:" in t and "issue = wave" in t


def _expandTdmIssueGateLo(items, tdm_lo, floor):
  """Include leading ``cmp ; cbranch`` so relocate moves the whole issue gate.

  Issuer bodies start with ``SCmpEQU32`` / ``SCBranchSCC0`` before the
  first ``TensorLoadToLds``.  Moving only the loads leaves an orphaned
  branch whose skip label relocated with the body — a backward jump that
  infinite-loops non-issuer waves (mainloop hang after the NLL-only sizes).
  """
  lo = tdm_lo
  while lo > floor + 1 and isinstance(items[lo - 1], TextBlock):
    lo -= 1
  if (lo >= floor + 2
      and _isTDMSingleIssuerIssueCmp(items[lo - 2])
      and isinstance(items[lo - 1], BranchInstruction)):
    lo -= 2
    while lo > floor + 1 and isinstance(items[lo - 1], TextBlock):
      lo -= 1
  return lo


def _collectTDMSingleIssuerAdvanceChunks(items, start):
  """Collect consecutive wave-gated advance chunks starting at ``start``.

  Each chunk is ``s_cmp`` (advance comment) … through its skip ``Label``.
  Returns ``(chunks, end_idx)`` or ``([], start)`` if the wall is absent /
  malformed. Chunks stay atomic (MX rotate + gate both use SCC).
  """
  chunks = []
  i = start
  while i < len(items) and _isTDMSingleIssuerAdvanceCmp(items[i]):
    c0 = i
    i += 1
    while i < len(items) and not isinstance(items[i], Label):
      i += 1
    if i >= len(items) or not isinstance(items[i], Label):
      return [], start
    i += 1  # include skip label
    chunks.append(items[c0:i])
  return chunks, i


def _findReadyWaitInWindow(window):
  """Index of READY's WGP wait inside ``window``, or None.

  Classic / SIG_EARLY: ``wait_tensorcnt`` then barrier pair; use the wait
  half so SIG_EARLY cover WMMAs stay pre-sync. Bare combined/wait-only
  barriers also count (no tensorcnt in the window).
  """
  k = 0
  while k < len(window):
    if isinstance(window[k], SWaitTensorcnt) and k + 1 < len(window):
      parsed = _parseWgBarrierPair(window, k + 1)
      if parsed is not None:
        return parsed[1]
    if _isCombinedWgBarrier(window[k]) or _isWgWaitOnly(window[k]):
      return k
    k += 1
  return None


def _emitAdvanceChunksIntoCover(out, cover, chunks):
  """Append ``cover``, splicing one advance chunk after each of the first N MMAs."""
  nC = len(chunks)
  mfmaIdx = [k for k, x in enumerate(cover) if _isMMA(x)]
  if len(mfmaIdx) < nC:
    for ch in chunks:
      out.extend(ch)
    out.extend(cover)
    return
  coverIdx = mfmaIdx[:nC]
  mi = 0
  for k, x in enumerate(cover):
    out.append(x)
    if mi < nC and k == coverIdx[mi]:
      out.extend(chunks[mi])
      mi += 1


def spreadTDMSingleIssuerAdvance(module, writer, kernel):
  """Interleave TDMSingleIssuer post-load advances into following WMMA holes.

  Emit still places the four wave-gated advance chunks as a contiguous wall
  after ``tensor_load``. That idles the matrix pipe. Split into per-issuer
  chunks and issue one after each of the next N WMMAs (deadline = next
  ``tensor_load`` that will consume the updated descriptor).

  When a READY WGP wait sits in the cover window, chunks are placed **only
  after** that wait so FREE→READY stays free of wave-gated elect branches
  (aligned barrier arrival). Pre-READY cover is emitted as-is; post-READY
  WMMAs hide the chunks when enough exist, otherwise the wall sits right
  after the wait. With no READY in the window, barriers /
  ``s_wait_tensorcnt`` may still be crossed (legacy hole fill). Fewer than
  N cover WMMAs and no READY → leave the wall in place.
  """
  if not kernel.get("TDMSingleIssuer"):
    return module

  items = list(module.flatitems())
  out = []
  i = 0
  changed = False
  while i < len(items):
    if not _isTDMSingleIssuerLoad(items[i]):
      out.append(items[i])
      i += 1
      continue

    out.append(items[i])
    i += 1
    chunks, end = _collectTDMSingleIssuerAdvanceChunks(items, i)
    if not chunks:
      continue

    # Cover window: up to (not including) the next descriptor-consuming load.
    next_load = end
    while next_load < len(items) and not isinstance(items[next_load], TensorLoadToLds):
      next_load += 1
    window = items[end:next_load]
    nC = len(chunks)
    ready_wait = _findReadyWaitInWindow(window)

    if ready_wait is not None:
      # Align READY: no elect branches between FREE sync and this wait.
      out.extend(window[:ready_wait + 1])
      _emitAdvanceChunksIntoCover(out, window[ready_wait + 1:], chunks)
      changed = True
      i = next_load
      continue

    mfmaIdx = [k for k, x in enumerate(window) if _isMMA(x)]
    if len(mfmaIdx) < nC:
      out.extend(items[i:end])
      i = end
      continue

    coverIdx = mfmaIdx[:nC]
    mi = 0
    for k, x in enumerate(window):
      out.append(x)
      if mi < nC and k == coverIdx[mi]:
        out.extend(chunks[mi])
        mi += 1
    changed = True
    i = next_load

  if not changed:
    return module
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


def _isPostReadyCoverGlue(inst):
  """Instrs that may sit between post-READY WMMAs without ending the cover window.

  After ``spreadTDMSingleIssuerAdvance``, issuer ``s_cmp`` / xor / add / branch
  islands live between K=3 WMMAs.  Treating them as opaque used to truncate the
  cover after the first WMMA and leave the READY ``ds_load_b128`` burst intact.
  """
  if isinstance(inst, (Label, BranchInstruction, SWaitAlu, TextBlock)):
    return True
  if isinstance(inst, (SBarrier, SWaitCnt, SWaitTensorcnt, SSetVgprMsb,
                       TensorLoadToLds, DSLoadB128, LocalReadInstruction)):
    return False
  if _isMMA(inst):
    return False
  # Remaining SALU / VALU / SNop used by TDM advances and scheduling pads.
  name = type(inst).__name__
  return name.startswith("S") or name.startswith("V")


def spreadPostReadyDsLoads(module, writer, kernel):
  """Interleave the post-READY ds_load_b128 burst with following WMMAs.

  After READY the scheduler emits a solid A/B b128 prefetch before K-group
  WMMAs resume, so ~half the loads sit with nothing to hide behind.  Move that
  leading b128 burst into the following WMMA stream (1:1 style), matching
  patch_ds_spread: scale ds_load_b32 stays put (WAR on live WMMA scale VGPRs);
  s_set_vgpr_msb prefixes travel with their consumer.  TDM-advance SALU between
  WMMAs is treated as cover glue (does not end the window).  Call only on
  MAINLOOP modules.  TDM kernels only.
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
      # (msb, inst, kind, glue_before) — glue = advance SALU kept before this item.
      rest = []
      glue = []
      while k < len(items):
        if isinstance(items[k], TextBlock):
          glue.append(items[k])
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
          rest.append((msb, items[k], "W", glue))
          glue = []
          k += 1
          continue
        if isinstance(items[k], DSLoadB128):
          if pending is not None:
            msb = pending
            pending = None
          rest.append((msb, items[k], "L128", glue))
          glue = []
          k += 1
          continue
        if isinstance(items[k], LocalReadInstruction):
          if pending is not None:
            msb = pending
            pending = None
          rest.append((msb, items[k], "Lother", glue))  # scale b32 — do not move
          glue = []
          k += 1
          continue
        if _isPostReadyCoverGlue(items[k]):
          glue.append(items[k])
          k += 1
          continue
        break
      # Trailing glue / pending MSB before a foreign instr: leave region alone
      # unless we already have WMMAs to cover (glue re-emitted by not advancing k
      # past it — drop trailing glue by rewinding k).
      if glue:
        k -= len(glue)
        glue = []
      if pending is not None or not any(kind == "W" for _, _, kind, _ in rest):
        continue

      idx = 0
      preamble = []  # glued loads before first WMMA (keep glued, no burst mix-in)
      while idx < len(rest) and rest[idx][2] != "W":
        m, ins, _, g = rest[idx]
        preamble.append((g, m, ins))
        idx += 1
      groups = []
      while idx < len(rest):
        assert rest[idx][2] == "W"
        m, ins, _, g = rest[idx]
        w = (g, m, ins)
        idx += 1
        trail = []
        while idx < len(rest) and rest[idx][2] != "W":
          tm, tins, _, tg = rest[idx]
          trail.append((tg, tm, tins))
          idx += 1
        groups.append((w, trail))

      n_w = len(groups)
      slots = [[] for _ in range(n_w)]
      for bi, ld in enumerate(burst):
        slots[min(bi * n_w // len(burst), n_w - 1)].append(ld)

      # Mix of raw glue Instructions and (msb, inst) pairs for _emitWithMsb.
      emitted = []
      for g, m, ins in preamble:
        emitted.extend(g)
        emitted.append((m, ins))
      for gi, (w, trail) in enumerate(groups):
        g, m, ins = w
        emitted.extend(g)
        emitted.append((m, ins))
        for sm, sinst in slots[gi]:
          emitted.append((sm, sinst))
        for tg, tm, tins in trail:
          emitted.extend(tg)
          emitted.append((tm, tins))

      text_insts = []
      tagged = []

      def flush_tagged():
        nonlocal current_msb, tagged, text_insts
        if not tagged:
          return
        chunk, current_msb = _emitWithMsb(tagged, current_msb)
        text_insts.extend(chunk)
        tagged = []

      for e in emitted:
        if isinstance(e, tuple):
          tagged.append(e)
        else:
          flush_tagged()
          text_insts.append(e)
      flush_tagged()

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
  """Proto-style fine ``s_wait_dscnt N`` before WMMAs (ordinal done-watermark).

  gemm_wt ``fine_waits``: loads get issue ordinals; before each WMMA, if it
  needs a load not yet proven (``need > done``), emit
  ``s_wait_dscnt (total - need)`` and set ``done = need``.  A wait already
  implied by an earlier one is skipped.  FREE needs no barrier-local drain —
  the last consumer's wait is the proof (``placeFreeBarrierAfterWmma`` slides
  that wait for the FREE-hole).

  Strips prior LR / fine waits and rebuilds.  MAINLOOP entry with unseen
  preloop tiles keeps a hard ``dscnt=0``.  Call after final LR/WMMA interleave.
  """
  items = list(module.flatitems())
  if not any(_isMMA(x) or isinstance(x, LocalReadInstruction) for x in items):
    return module
  # Nothing to refine if the scheduler never emitted LR waits and there are
  # no loads/WMMAs to analyze — still run when waits exist OR loads+WMMAs do.
  if not any(_isWaitLR(x) for x in items) and not any(
      isinstance(x, LocalReadInstruction) for x in items):
    return module

  total = 0          # loads issued
  done = 0           # highest ordinal proven complete
  dst_last = {}      # vgpr idx -> ordinal of latest load into it
  out = []
  emitted_any = False

  def _emit_wait(n, hard=False):
    nonlocal emitted_any
    if hard or n == 0:
      cmt = "Wait for LR to complete"
    else:
      cmt = "tiles of the next WMMA landed"
    out.append(SWaitCnt(dscnt=n, vlcnt=-1, vscnt=-1, comment=cmt))
    emitted_any = True

  for inst in items:
    if _isWaitLR(inst):
      # Drop scheduler / prior-pass LR waits; this pass re-emits.
      continue

    if isinstance(inst, LocalReadInstruction):
      params = inst.getParams()
      dst = set(_vgprIndices(params[0])) if params else set()
      if dst:
        total += 1
        for d in dst:
          dst_last[d] = total
      out.append(inst)
      continue

    if _isMMA(inst):
      srcs = _mmaSrcVgprs(inst)
      if srcs:
        # Overlap-based need (MFMA operands may be wider than a single ds_load
        # dst).  Preloop tiles are invisible here — first such WMMA keeps a
        # hard dscnt=0 (dsGlue loop-head wait); do not bump ``done`` to
        # ``total`` (that would falsely prove later in-module loads).
        seen = [dst_last[s] for s in srcs if s in dst_last]
        if seen:
          need = max(seen)
          if need > done:
            n = total - need
            if n < _DSCNT_MAX:
              _emit_wait(n, hard=(n == 0))
            done = need
        elif not emitted_any:
          _emit_wait(0, hard=True)
      out.append(inst)
      continue

    out.append(inst)

  if not emitted_any and any(_isMMA(x) for x in items):
    # No LR dependency resolved — keep a single hard drain at first WMMA.
    rebuilt = []
    placed = False
    for inst in out:
      if not placed and _isMMA(inst):
        rebuilt.append(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                                comment="Wait for LR to complete"))
        placed = True
      rebuilt.append(inst)
    out = rebuilt

  result = Module(module.name)
  for inst in out:
    result.add(inst)
  return result


def _isAbDataB128(inst):
  """True for A/B tile ds_load_b128 (excludes scale and unknown b128)."""
  if not isinstance(inst, DSLoadB128):
    return False
  c = getattr(inst, "comment", "") or ""
  return ("SubtileA[" in c) or ("SubtileB[" in c)


def _isHardScheduleBoundary(inst):
  """Ops A/B data loads must not cross when sliding for WAR slack."""
  if isinstance(inst, (SBarrier, SWaitCnt, SWaitTensorcnt, SWaitAlu)):
    return True
  if isinstance(inst, (VXorB32, VAddU32, VSubU32, VCmpGEU32, VCndMaskB32, SMovB32)):
    return True
  return False


# Max WMMAs a single A/B data load may cross when sliding left.  Uncapped
# advance dumps every refill at the WAR fence (~31 WMMA lead) and measured
# ~0.93× vs B-then-A+splitWait; a cap keeps interleave while lifting the
# tight wraparound leads (A0 wrap was ~2 WMMA).  ~12 WMMAs ≈ 96 clk toward
# gemm_wt SCHED=3's ≥100 clk RAW slack target at 8 clk/WMMA.
_WAR_SLACK_MAX_WMMA_CROSS = 12
# Tried wrap-only deeper slide (subIterK=0 cap 28 → ~31 WMMA lead): 10/10 but
# ~0.97× vs cap12 — same bunching/regression as fully uncapped. Leave uniform.


def advanceDsLoadsForWarSlack(module, writer, kernel,
                              maxWmmaCross=_WAR_SLACK_MAX_WMMA_CROSS,
                              wrapWmmaCross=None):
  """Slide A/B ``ds_load_b128`` left toward the WAR fence (last reader + 1).

  gemm_wt SCHED=3 / mlplan program-order: a refill must not issue before the
  last WMMA that still reads its destination VGPRs.  Pulling late A/B data
  loads up (capped) grows ds_load→first-use RAW slack without a CP-SAT
  solver.  Scale ``ds_load_b32``, barriers, waits, and LR swaps are hard
  boundaries.  Prefer B then A.  TDM MAINLOOP only.
  """
  if wrapWmmaCross is None:
    wrapWmmaCross = maxWmmaCross
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  if not any(_isAbDataB128(x) for x in items):
    return module

  def _ok_after_wait(insert):
    """Reject landing in the loop-head post-``wait_dscnt`` gap before any WMMA.

    Cap12 advance piled next-K B loads right after the opening
    ``s_wait_dscnt 0`` (12-wide burst before first WMMA).  Only fence that
    *first* wait_lr — mid-loop waits still allow early refill into their
    holes (blocking every wait measured ~2% slower).
    """
    first_wait = next((t for t, x in enumerate(items) if _isWaitLR(x)), None)
    if first_wait is None or insert <= first_wait:
      return True
    # Is the nearest hard boundary on the left that first wait (no WMMA yet)?
    k = insert - 1
    wmmas = 0
    while k >= 0:
      x = items[k]
      if isinstance(x, (TextBlock, SSetVgprMsb)):
        k -= 1
        continue
      if _isMMA(x):
        wmmas += 1
        k -= 1
        continue
      if _isHardScheduleBoundary(x):
        if k == first_wait:
          return wmmas >= 1
        return True
      k -= 1
    return True

  def _earliest(i, dst, moving_is_b, cap):
    """Index to insert the load before; never before a WAR-producing WMMA."""
    insert = i
    j = i - 1
    crossed = 0
    while j >= 0:
      x = items[j]
      if isinstance(x, (TextBlock, SSetVgprMsb)):
        j -= 1
        continue
      if _isMMA(x):
        if _mmaSrcVgprs(x) & dst:
          break
        if crossed >= cap:
          break
        if _ok_after_wait(j):
          insert = j
        else:
          break
        crossed += 1
        j -= 1
        continue
      if _isAbDataB128(x):
        # B may pass A (claim earlier holes); nothing passes a peer B/A.
        other_is_b = "SubtileB[" in (getattr(x, "comment", "") or "")
        if moving_is_b and not other_is_b:
          if _ok_after_wait(j):
            insert = j
          else:
            break
          j -= 1
          continue
        break
      if isinstance(x, LocalReadInstruction):
        # Scale b32 and other LRs: independent VGPRs.
        if _ok_after_wait(j):
          insert = j
        else:
          break
        j -= 1
        continue
      if _isHardScheduleBoundary(x):
        break
      # Unknown op: treat as hard boundary.
      break
    return insert

  def _move_one(i, moving_is_b):
    load = items[i]
    params = load.getParams()
    dst = set(_vgprIndices(params[0])) if params else set()
    if not dst:
      return False
    start = i
    chunk = [load]
    if i > 0 and isinstance(items[i - 1], SSetVgprMsb):
      start = i - 1
      chunk = [items[i - 1], load]
    insert = _earliest(start, dst, moving_is_b, maxWmmaCross)
    if insert >= start:
      return False
    del items[start:start + len(chunk)]
    items[insert:insert] = chunk
    return True

  # B first (read-ahead priority), then A; each load moves at most once so the
  # WMMA-cross cap is meaningful (an iterative re-slide would ignore it).
  for want_b in (True, False):
    i = 0
    while i < len(items):
      inst = items[i]
      if _isAbDataB128(inst):
        c = getattr(inst, "comment", "") or ""
        is_b = "SubtileB[" in c
        if is_b == want_b:
          if _move_one(i, is_b):
            # Landed earlier; continue after the moved chunk.
            i += 1
            continue
      i += 1

  result = Module(module.name)
  for inst in items:
    result.add(inst)
  return result


def _isWgBarrier(inst):
  """Workgroup barrier (not cluster -3). Requires a wait, not signal-only."""
  if not isinstance(inst, SBarrier):
    return False
  c = (getattr(inst, "comment", "") or "").lower()
  if "cluster" in c:
    return False
  s = str(inst)
  if "-3" in s or "cluster" in s.lower():
    return False
  # Full WG barrier / wait-only, or plain ``s_barrier`` before gfx1250 init.
  return ("s_barrier_wait" in s) or ("s_barrier_signal" not in s)


def _isClusterBarrier(inst):
  if not isinstance(inst, SBarrier):
    return False
  c = (getattr(inst, "comment", "") or "").lower()
  if "cluster" in c or getattr(inst, "cluster_barrier", False):
    return True
  s = str(inst)
  return "-3" in s or "cluster" in s.lower()


def _isCombinedWgBarrier(inst):
  """Single SBarrier that expands to signal+wait (or classic ``s_barrier``)."""
  if not isinstance(inst, SBarrier) or _isClusterBarrier(inst):
    return False
  s = str(inst)
  if "s_barrier_signal" in s and "s_barrier_wait" in s:
    return True
  # Classic / caps-off form — still a glued sync point.
  return ("s_barrier_signal" not in s) and ("s_barrier_wait" not in s)


def _isWgSignalOnly(inst):
  if not isinstance(inst, SBarrier) or _isClusterBarrier(inst):
    return False
  s = str(inst)
  return ("s_barrier_signal" in s) and ("s_barrier_wait" not in s)


def _isWgWaitOnly(inst):
  if not isinstance(inst, SBarrier) or _isClusterBarrier(inst):
    return False
  s = str(inst)
  return ("s_barrier_wait" in s) and ("s_barrier_signal" not in s)


def _isBarrierDrainWait(inst):
  """wait_dscnt / wait_tensorcnt that immediately precedes a FREE/READY barrier."""
  if isinstance(inst, SWaitTensorcnt):
    return True
  return isinstance(inst, SWaitCnt) and inst.dscnt is not None


def _mmaChunkBounds(items, mma_idx):
  """Half-open [lo, hi) covering optional ``s_set_vgpr_msb`` + MMA at *mma_idx*."""
  lo = mma_idx
  if lo > 0 and isinstance(items[lo - 1], SSetVgprMsb):
    lo -= 1
  return lo, mma_idx + 1


def _trailingMmaChunksBefore(items, end, n):
  """Up to *n* contiguous trailing MMA chunks immediately before *end*."""
  chunks = []
  i = end - 1
  while i >= 0 and len(chunks) < n:
    if isinstance(items[i], (TextBlock,)):
      i -= 1
      continue
    if _isMMA(items[i]):
      lo, hi = _mmaChunkBounds(items, i)
      chunks.append((lo, hi))
      i = lo - 1
      continue
    break
  chunks.reverse()
  return chunks


def _forwardMmaChunksAfter(items, start, n):
  """Up to *n* MMA chunks at/after *start*, skipping filler; stop at barriers/branches/TDM.

  Used at READY (post-barrier ds_loads then K-step WMMAs).  Refuses FREE-like
  sites where ``tensor_load_to_lds`` appears before the first collectible MMA —
  hoisting those would delay TDM and steal the cluster-pinned post-FREE WMMA.
  """
  chunks = []
  i = start
  seen_mma = False
  while i < len(items) and len(chunks) < n:
    inst = items[i]
    if isinstance(inst, TextBlock):
      i += 1
      continue
    if _isCombinedWgBarrier(inst) or _isWgSignalOnly(inst) or _isWgWaitOnly(inst):
      break
    if _isClusterBarrier(inst) or isinstance(inst, BranchInstruction):
      break
    if isinstance(inst, TensorLoadToLds):
      break
    if _isMMA(inst):
      lo, hi = _mmaChunkBounds(items, i)
      # Chunk lo may include an msb before *i*; only legal if we own that range.
      if lo < start and not seen_mma:
        lo = i
      chunks.append((lo, hi))
      seen_mma = True
      i = hi
      continue
    # Skip ds_loads / swaps / SALU between READY wait and the K-step WMMAs.
    if seen_mma and isinstance(inst, (LocalReadInstruction, DSLoadB128,
                                      SWaitCnt, SWaitTensorcnt)):
      # Keep scanning for more interleaved WMMAs after a load gap.
      i += 1
      continue
    if not seen_mma:
      i += 1
      continue
    # After the first MMA, stop on unfamiliar non-MMA (don't drag a long tail).
    if isinstance(inst, SSetVgprMsb):
      i += 1
      continue
    break
  return chunks


def _splitCombinedWgBarrier(inst):
  """Replace a glued WGP barrier with separate signal + wait halves."""
  comment = getattr(inst, "comment", "") or ""
  return (
      SBarrier(True, False, False, comment="SIG_EARLY signal"),
      SBarrier(True, True, False, comment=comment or "SIG_EARLY wait"),
  )


def _parseWgBarrierPair(items, bar_i):
  """Return ``(sig_i, wait_i, combined)`` for a WGP barrier at *bar_i*, or None."""
  if _isCombinedWgBarrier(items[bar_i]):
    return bar_i, bar_i, True
  if (bar_i + 1 < len(items)
      and _isWgSignalOnly(items[bar_i])
      and _isWgWaitOnly(items[bar_i + 1])):
    return bar_i, bar_i + 1, False
  return None


def _freeHoleCoverOk(items, lo, hi):
  """True if ``items[lo:hi]`` is only MMAs / primers (FREE-hole cover)."""
  if lo >= hi:
    return False
  saw_mma = False
  for j in range(lo, hi):
    inst = items[j]
    if isinstance(inst, (TextBlock, SSetVgprMsb)):
      continue
    if _isMMA(inst):
      saw_mma = True
      continue
    return False
  return saw_mma


def hideWgpBarrierBehindWmma(module, writer, kernel, nWmma=SIG_EARLY_WMMA_DEFAULT,
                              allowForward=SIG_EARLY_ALLOW_FORWARD_DEFAULT):
  """gemm_wt ``SIG_EARLY``: put ≥1 WMMA between WGP ``signal`` and ``wait``.

  ``wait_dscnt`` / ``wait_tensorcnt`` drain the XDL queue; a glued
  ``signal; wait`` then stalls with nothing in flight.  Refill the queue::

      drain ; signal ; [N WMMAs] ; wait ; ...

  Two FREE/READY shapes:

  * **Adjacent drain** (READY): ``tensorcnt ; barrier``.  Pull trailing
    WMMAs from *before* the drain (or forward post-barrier if
    *allowForward*).
  * **FREE-hole** (after ``placeFreeBarrierAfterWmma``):
    ``wait_dscnt ; WMMAs ; barrier ; TDM``.  Keep those hole WMMAs, but
    open the barrier so they sit between signal and wait (and optionally
    pull more from above the drain up to *nWmma*).

  READY forward-pull is opt-in — on DU512 it delayed ds_loads (~8%).
  Cluster barriers (-3) are left alone.  Call on MAINLOOP after
  ``placeFreeBarrierAfterWmma``, before ``insertClusterBarrier``.
  """
  if nWmma <= 0:
    return module
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  # Each plan: (drain_i, sig_i, wait_i, combined, skip_idx, mid_insts)
  # Emit at drain_i: drain ; signal ; mid ; wait.  skip_idx covers mid + bars.
  plans = []
  claimed = set()
  i = 0
  while i < len(items):
    parsed = _parseWgBarrierPair(items, i)
    if parsed is None:
      i += 1
      continue
    sig_i, wait_i, combined = parsed

    drain_i = None
    mid = []
    skip_idx = set()

    # Path A: drain glued to barrier (READY / classic).
    if sig_i > 0 and _isBarrierDrainWait(items[sig_i - 1]):
      drain_i = sig_i - 1
      chunks = _trailingMmaChunksBefore(items, drain_i, nWmma)
      if not chunks and allowForward:
        chunks = _forwardMmaChunksAfter(items, wait_i + 1, nWmma)
      if not chunks:
        i = wait_i + 1
        continue
      for lo, hi in chunks:
        skip_idx.update(range(lo, hi))
        mid.extend(items[lo:hi])

    # Path B: FREE-hole — drain ; cover WMMAs ; barrier ; TDM.
    elif _tdmFollowsBarrier(items, wait_i):
      # Nearest free dscnt above with only MMA cover between it and signal.
      d = sig_i - 1
      while d >= 0 and isinstance(items[d], (TextBlock, SSetVgprMsb)):
        d -= 1
      # Walk back over MMA cover to the drain.
      j = d
      while j >= 0 and (_isMMA(items[j]) or isinstance(items[j], (TextBlock, SSetVgprMsb))):
        j -= 1
      if j < 0 or not _isFreeDscntDrain(items[j]):
        i = wait_i + 1
        continue
      drain_i = j
      hole_lo, hole_hi = drain_i + 1, sig_i
      if not _freeHoleCoverOk(items, hole_lo, hole_hi):
        i = wait_i + 1
        continue
      # Optional: pad cover to nWmma from above the drain.
      have = sum(1 for k in range(hole_lo, hole_hi) if _isMMA(items[k]))
      extra = []
      if have < nWmma:
        chunks = _trailingMmaChunksBefore(items, drain_i, nWmma - have)
        for lo, hi in chunks:
          skip_idx.update(range(lo, hi))
          extra.extend(items[lo:hi])
      skip_idx.update(range(hole_lo, hole_hi))
      mid = extra + items[hole_lo:hole_hi]
    else:
      i = wait_i + 1
      continue

    if not mid:
      i = wait_i + 1
      continue
    if skip_idx & claimed:
      i = wait_i + 1
      continue
    if drain_i in skip_idx or sig_i in skip_idx or wait_i in skip_idx:
      i = wait_i + 1
      continue

    skip_idx.add(sig_i)
    if wait_i != sig_i:
      skip_idx.add(wait_i)
    plans.append((drain_i, sig_i, wait_i, combined, skip_idx, mid))
    claimed |= skip_idx
    claimed.add(drain_i)
    i = wait_i + 1

  if not plans:
    return module

  by_drain = {p[0]: p for p in plans}
  skip = set()
  for _, _, _, _, skip_idx, _ in plans:
    skip |= skip_idx

  out = Module(module.name)
  i = 0
  while i < len(items):
    if i in skip:
      i += 1
      continue
    plan = by_drain.get(i)
    if plan is None:
      out.add(items[i])
      i += 1
      continue

    drain_i, sig_i, wait_i, combined, skip_idx, mid = plan
    if combined:
      sig, wait = _splitCombinedWgBarrier(items[sig_i])
    else:
      sig, wait = items[sig_i], items[wait_i]

    out.add(items[drain_i])
    out.add(sig)
    for inst in mid:
      out.add(inst)
    out.add(wait)
    i = wait_i + 1

  return out


def _isFreeDscntDrain(inst):
  """FREE / fine LR drain: ``s_wait_dscnt`` (not READY ``wait_tensorcnt``)."""
  return _isWaitLR(inst) or (
      isinstance(inst, SWaitCnt) and inst.dscnt is not None
      and "tensorcnt" not in (getattr(inst, "comment", "") or "").lower())


def _tdmFollowsBarrier(items, wait_i):
  """True if a ``tensor_load_to_lds`` appears soon after the WGP wait (FREE shape).

  Window covers a leading TDMSingleIssuer issue gate (cmp+branch) before the
  first load; SALU between gate and load is still inside the FREE TDM wall.
  """
  for j in range(wait_i + 1, min(wait_i + 24, len(items))):
    if isinstance(items[j], TensorLoadToLds):
      return True
    if _isMMA(items[j]) or _isCombinedWgBarrier(items[j]) or _isWgSignalOnly(items[j]):
      return False
    if isinstance(items[j], (SWaitTensorcnt, SWaitCnt)):
      return False
  return False


def _clusterSignalFollows(items, wait_i):
  """True if the wave-0 cluster signal block opens right after the WGP wait."""
  for j in range(wait_i + 1, min(wait_i + 4, len(items))):
    if _isClusterBarrier(items[j]):
      return "s_barrier_signal -3" in str(items[j])
    if _isMMA(items[j]) or isinstance(items[j], SBarrier):
      return False
  return False


def _parseWgBarrierAt(items, bar_i):
  """Return (sig_i, wait_i) for a WGP barrier at ``bar_i``, or None."""
  if bar_i >= len(items):
    return None
  if _isCombinedWgBarrier(items[bar_i]):
    return bar_i, bar_i
  if (bar_i + 1 < len(items)
      and _isWgSignalOnly(items[bar_i])
      and _isWgWaitOnly(items[bar_i + 1])):
    return bar_i, bar_i + 1
  return None


def _nearestDrainBefore(items, bar_i):
  """Index of the nearest fine/LR ``s_wait_dscnt`` before FREE, or None.

  Skips primers / text and VGPR-only WMMAs (FREE-hole candidates).  Stops at
  foreign ops (loads, TDM, another barrier, tensorcnt).
  """
  j = bar_i - 1
  while j >= 0:
    inst = items[j]
    if _isFreeDscntDrain(inst):
      return j
    if isinstance(inst, (TextBlock, SSetVgprMsb)):
      j -= 1
      continue
    if _isMMA(inst):
      j -= 1
      continue
    break
  return None


def _freeLastConsumerIndex(items, bar_i):
  """Index of the FREE-proof WMMA (proto last-consumer stand-in).

  Prefer the earliest WMMA whose fine ``need`` covers every pre-FREE load that
  some pre-barrier WMMA consumes (``need_free``).  If ordinal overlap cannot
  resolve a consumer (common when preloop tiles dominate early WMMAs), fall
  back to the first WMMA after the nearest fine wait — that wait is the FREE
  proof and FREE belongs immediately behind its consumer.
  """
  total = 0
  dst_last = {}
  loads = []  # (ordinal, dst_set)
  wmmas = []  # (idx, need)
  for i in range(bar_i):
    inst = items[i]
    if isinstance(inst, LocalReadInstruction):
      params = inst.getParams()
      dst = set(_vgprIndices(params[0])) if params else set()
      if dst:
        total += 1
        for d in dst:
          dst_last[d] = total
        loads.append((total, dst))
      continue
    if _isMMA(inst):
      srcs = _mmaSrcVgprs(inst)
      seen = [dst_last[s] for s in srcs if s in dst_last]
      need = max(seen) if seen else -1
      wmmas.append((i, need))

  need_free = -1
  if loads and wmmas:
    for ord_i, dst in loads:
      for wi, _need in wmmas:
        if _mmaSrcVgprs(items[wi]) & dst:
          need_free = max(need_free, ord_i)
          break
    if need_free >= 0:
      for wi, need in wmmas:
        if need >= need_free:
          return wi

  # Fallback: first WMMA after the nearest fine/LR wait before FREE.
  drain = _nearestDrainBefore(items, bar_i)
  if drain is None:
    return None
  for j in range(drain + 1, bar_i):
    if _isMMA(items[j]):
      return j
  return None


def _tdmBlockBoundsAfter(items, wait_i):
  """Return (tdm_lo, tdm_hi_exclusive) for the FREE TDM wall after ``wait_i``.

  Includes any leading TDMSingleIssuer issue gate so ``relocateFreeToLastConsumer``
  keeps ``cmp/branch`` glued to the issuer load body.
  """
  tdm = None
  for j in range(wait_i + 1, min(wait_i + 24, len(items))):
    if isinstance(items[j], TensorLoadToLds):
      tdm = j
      break
    if _isMMA(items[j]) or _isCombinedWgBarrier(items[j]) or _isWgSignalOnly(items[j]):
      return None
    if isinstance(items[j], (SWaitTensorcnt, SWaitCnt)):
      return None
  if tdm is None:
    return None
  lo = _expandTdmIssueGateLo(items, tdm, wait_i)
  end = tdm + 1
  while end < len(items) and not _tdmBlockEnds(items[end]):
    end += 1
  return lo, end


def sinkPreFreeDscntBelowSignal(module, writer, kernel):
  """Move the fine ``s_wait_dscnt`` glued before FREE's CW/WS below the WS.

  Proto: FREE has no wait of its own — the consumer WMMA's ``s_wait_dscnt`` is
  the proof.  After SIG_EARLY + ClusterBarrier the common shape is::

      s_wait_dscnt N ; CW ; WS ; [cover WMMAs] ; WW ; TDM
   →  CW ; WS ; s_wait_dscnt N ; [cover WMMAs] ; WW ; TDM

  with no MMA between the wait and CW.  The wait is the cover WMMAs' own
  (the first cover WMMA can read a tile whose loads no earlier wait covers),
  so it cannot be dropped; below WS it no longer holds back CW or the signal.

  Call after ``insertClusterBarrier``.  SIG_EARLY puts cover WMMAs between
  WS and WW, so the FREE pair is not adjacent — locate FREE by WW→TDM, then
  walk back to WS.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module
  if not kernel.get("ClusterBarrier"):
    return module

  items = list(module.flatitems())
  sinkTo = {}  # drain index -> WS index it goes right after

  i = 0
  while i < len(items):
    # FREE wait: wait-only or combined WGP barrier with TDM after.
    is_free_wait = (
        (_isWgWaitOnly(items[i]) or _isCombinedWgBarrier(items[i]))
        and (_tdmFollowsBarrier(items, i)
             or (kernel.get("TDMClusterSyncAfterFree")
                 and _clusterSignalFollows(items, i))))
    if not is_free_wait:
      i += 1
      continue
    wait_i = i

    # SIG_EARLY: nearest prior WGP signal with only MMA/primers between.
    sig_i = None
    j = wait_i - 1
    while j >= 0:
      inst = items[j]
      if isinstance(inst, (TextBlock, SSetVgprMsb, Label)) or _isMMA(inst):
        j -= 1
        continue
      if _isWgSignalOnly(inst):
        sig_i = j
      break
    if sig_i is None:
      i = wait_i + 1
      continue
    if not any(_isMMA(items[k]) for k in range(sig_i + 1, wait_i)):
      i = wait_i + 1
      continue

    # Walk back over primers / CW to the candidate orphan drain.
    j = sig_i - 1
    while j >= 0 and isinstance(items[j], (TextBlock, SSetVgprMsb, Label)):
      j -= 1
    if j >= 0 and _isClusterBarrier(items[j]):
      j -= 1
      while j >= 0 and isinstance(items[j], (TextBlock, SSetVgprMsb, Label)):
        j -= 1
    if j < 0 or not _isFreeDscntDrain(items[j]):
      i = wait_i + 1
      continue

    drain_i = j
    between = items[drain_i + 1:sig_i]
    if any(_isMMA(x) or isinstance(x, (LocalReadInstruction, DSLoadB128,
                                         TensorLoadToLds, SWaitCnt,
                                         SWaitTensorcnt))
           for x in between):
      i = wait_i + 1
      continue

    sinkTo[drain_i] = sig_i
    i = wait_i + 1

  if not sinkTo:
    return module

  after = {sig_i: items[drain_i] for drain_i, sig_i in sinkTo.items()}
  out = Module(module.name)
  for idx, inst in enumerate(items):
    if idx in sinkTo:
      continue
    out.add(inst)
    if idx in after:
      out.add(after[idx])
  return out


def relocateFreeToLastConsumer(module, writer, kernel):
  """Park FREE ``barrier ; TDM`` right after the last-consumer WMMA (proto).

  gemm_wt places FREE behind the WMMA that first consumes the freed slice's
  last-loaded tile so that WMMA's fine ``s_wait_dscnt`` is the proof — no
  barrier-local hard drain.  Call after ``splitWaitLRPerTile``, before
  ``placeFreeBarrierAfterWmma``.  READY left alone for ``mergeFreeReadyBarriers``.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  moves = []  # (consumer_i, sig_i, wait_i, tdm_lo, tdm_hi)
  i = 0
  while i < len(items):
    parsed = _parseWgBarrierAt(items, i)
    if parsed is None:
      i += 1
      continue
    sig_i, wait_i = parsed
    if not _tdmFollowsBarrier(items, wait_i):
      i = wait_i + 1
      continue
    bounds = _tdmBlockBoundsAfter(items, wait_i)
    if bounds is None:
      i = wait_i + 1
      continue
    tdm_lo, tdm_hi = bounds
    consumer = _freeLastConsumerIndex(items, sig_i)
    if consumer is None:
      i = wait_i + 1
      continue
    # Never park FREE above a ds_load the scheduler put before it: with a
    # per-operand FREE (LRLookaheadB > 1) the freed slice's last reads are not
    # consumed before the barrier, so the ordinal proof can point above them.
    lastLoad = max((k for k in range(consumer + 1, sig_i)
                    if isinstance(items[k], LocalReadInstruction)), default=None)
    if lastLoad is not None:
      consumer = next((k for k in range(lastLoad + 1, sig_i) if _isMMA(items[k])), None)
      if consumer is None:
        i = wait_i + 1
        continue
    # Already immediately before FREE (only primers between)?
    j = consumer + 1
    while j < sig_i and isinstance(items[j], (TextBlock, SSetVgprMsb)):
      j += 1
    if j == sig_i:
      i = wait_i + 1
      continue
    moves.append((consumer, sig_i, wait_i, tdm_lo, tdm_hi))
    i = wait_i + 1

  if not moves:
    return module

  skip = set()
  insert_after = {}
  for consumer, sig_i, wait_i, tdm_lo, tdm_hi in moves:
    if consumer in insert_after:
      # Two FREEs claiming the same consumer — leave later site alone.
      continue
    skip.update(range(sig_i, wait_i + 1))
    skip.update(range(tdm_lo, tdm_hi))
    insert_after[consumer] = (items[sig_i:wait_i + 1] + items[tdm_lo:tdm_hi])

  out = Module(module.name)
  for idx, inst in enumerate(items):
    if idx in skip:
      continue
    out.add(inst)
    extra = insert_after.get(idx)
    if extra is not None:
      for x in extra:
        out.add(x)
  return out


def placeFreeBarrierAfterWmma(module, writer, kernel, nWmma=FREE_HOLE_WMMA_DEFAULT):
  """gemm_wt FREE-hole: ``wait_dscnt ; WMMAs ; barrier`` (wait right after WMMA).

  Unlike ``hideWgpBarrierBehindWmma`` (SIG_EARLY: ``drain ; signal ; WMMAs ; wait``),
  this keeps signal+wait glued and parks the barrier *immediately after* the
  last-consumer WMMAs::

      [prefix WMMAs] ; wait_dscnt ; [N trailing WMMAs] ; signal ; wait ; TDM

  With fine dscnt + ``relocateFreeToLastConsumer`` the drain already sits at
  the proving WMMA.  Find that nearest preceding wait; if fewer than ``nWmma``
  WMMAs lie between it and the barrier, slide the wait earlier by hoisting
  trailing VGPR-only WMMAs.  Does **not** harden the wait to ``dscnt=0`` —
  the fine wait is the FREE proof.  FREE sites only; READY left alone.
  Call after ``relocateFreeToLastConsumer``, before ``insertClusterBarrier``.
  """
  if nWmma <= 0:
    return module
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())
  # Plans: (drain_i, sig_i, wait_i, hoist_idx, hoist_insts, between_idx)
  plans = []
  claimed = set()

  i = 0
  while i < len(items):
    parsed = _parseWgBarrierAt(items, i)
    if parsed is None:
      i += 1
      continue
    sig_i, wait_i = parsed
    if not _tdmFollowsBarrier(items, wait_i):
      i = wait_i + 1
      continue

    drain_i = _nearestDrainBefore(items, sig_i)
    if drain_i is None:
      i = wait_i + 1
      continue

    # WMMAs already between drain and barrier count toward the hole.
    between = sum(1 for k in range(drain_i + 1, sig_i) if _isMMA(items[k]))
    need_hoist = max(0, nWmma - between)
    hoist_idx = set()
    hoist = []
    if need_hoist > 0:
      chunks = _trailingMmaChunksBefore(items, drain_i, need_hoist)
      if not chunks and between == 0:
        i = wait_i + 1
        continue
      for lo, hi in chunks:
        hoist_idx.update(range(lo, hi))
        hoist.extend(items[lo:hi])

    if not hoist:
      i = wait_i + 1
      continue
    if hoist_idx & claimed:
      i = wait_i + 1
      continue
    if drain_i in hoist_idx or sig_i in hoist_idx or wait_i in hoist_idx:
      i = wait_i + 1
      continue

    between_idx = set(range(drain_i + 1, sig_i))
    plans.append((drain_i, sig_i, wait_i, hoist_idx, hoist, between_idx))
    claimed |= hoist_idx
    claimed |= between_idx
    claimed.add(drain_i)
    claimed.add(sig_i)
    claimed.add(wait_i)
    i = wait_i + 1

  if not plans:
    return module

  by_drain = {p[0]: p for p in plans}
  skip = set()
  for _, _, _, hoist_idx, _, between_idx in plans:
    skip |= hoist_idx
    skip |= between_idx

  out = Module(module.name)
  i = 0
  while i < len(items):
    if i in skip:
      i += 1
      continue
    plan = by_drain.get(i)
    if plan is None:
      out.add(items[i])
      i += 1
      continue

    drain_i, sig_i, wait_i, hoist_idx, hoist, between_idx = plan
    # Target: wait ; [hoisted] ; [already-between] ; barrier ; TDM
    out.add(items[drain_i])
    for inst in hoist:
      out.add(inst)
    for k in sorted(between_idx):
      out.add(items[k])
    if sig_i == wait_i:
      out.add(items[sig_i])
    else:
      out.add(items[sig_i])
      out.add(items[wait_i])
    i = wait_i + 1

  return out


def _tdmBlockEnds(inst):
  """True once the post-FREE TDM issue/advance window is over."""
  if _isMMA(inst) or _isWaitLR(inst) or _isWgBarrier(inst):
    return True
  if isinstance(inst, (SWaitTensorcnt, LocalReadInstruction, DSLoadB128)):
    return True
  return False


def tensorcntExactN(items, wait_i, await_tdm_i=None):
  """gemm_wt ``tcnt_exact`` on a flat stream: TDMs issued after the awaited one.

  Hardware ``s_wait_tensorcnt N`` retires when outstanding <= N.  If N is larger
  than the number of TDMs issued *after* the slice being waited on, the wait is
  a no-op while that slice is still in flight (READY then races).  Proto's
  ``tcnt_exact`` sets N to that after-count (min over replayed loop passes).

  ``await_tdm_i``: index of the ``TensorLoadToLds`` being waited on.  Default:
  the latest TDM before ``wait_i`` (READY for the most recent fill).  Empty
  prefix → 0 (prologue fill; nothing after it in this module).
  """
  issued = [j for j in range(wait_i) if isinstance(items[j], TensorLoadToLds)]
  if not issued:
    return 0
  target = issued[-1] if await_tdm_i is None else await_tdm_i
  if target not in issued:
    return 0
  return len(issued) - 1 - issued.index(target)


def mergeFreeReadyBarriers(module, writer, kernel, forceDrain=False):
  """gemm_wt ``MERGE_BAR``: FREE+READY → ``tensorcnt`` + one WGP barrier + TDMs.

  Prototype emit order when FREE and READY share a hole::

      s_wait_tensorcnt ; s_barrier ; TDMs

  instead of ``FREE barrier → TDMs → tensorcnt → READY barrier``.  Drop the
  FREE barrier, hoist READY's wait+barrier above the TDM issue, leave mid-hole
  WMMAs after the TDM block.

  **Not wired in the scheduler.**  Hoisting the wait before this iteration's
  TDM makes ``tensorcntExactN`` 0 (await latest fill, nothing after it yet),
  which drains every iteration and loses ~10–15% vs classic.  Proto's
  mainloop avoids that stall with ``wait_tensorcnt 1`` *after* TDM+cover
  (latency-hidden pipeline), or MERGE + SIG_EARLY / multi-issuer ``after>0``.
  Keep this helper for experiments; prefer classic READY N=1 for perf.

  Default (``forceDrain=False``): ``tensorcntExactN`` at FREE (typically 0).
  ``forceDrain=True`` forces 0.  Sites with no post-TDM MMA cover stay classic.
  """
  if not (kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)):
    return module

  items = list(module.flatitems())

  def _probe_merge(bar_i):
    """Return (tdm, end_tdm, tcnt, after, wait_alu) or None if site cannot merge."""
    tdm = None
    for j in range(bar_i + 1, len(items)):
      if isinstance(items[j], TensorLoadToLds):
        tdm = j
        break
      if _isWgBarrier(items[j]) or isinstance(items[j], SWaitTensorcnt):
        break
    if tdm is None:
      return None
    tcnt = None
    for j in range(tdm + 1, len(items)):
      if _isWgBarrier(items[j]):
        break
      if (isinstance(items[j], SWaitTensorcnt)
          and j + 1 < len(items) and _isWgBarrier(items[j + 1])):
        if "s_wait_tensorcnt 0" in str(items[j]):
          break
        tcnt = j
        break
    if tcnt is None:
      return None
    end_tdm = tdm
    k = tdm + 1
    while k < tcnt and not _tdmBlockEnds(items[k]):
      end_tdm = k
      k += 1
    middle = items[end_tdm + 1:tcnt]
    if not any(_isMMA(x) for x in middle) and not forceDrain:
      return None
    after = tcnt + 2
    wait_alu = []
    if (after < len(items) and isinstance(items[after], SWaitAlu)
        and "LR offset swap" in (getattr(items[after], "comment", "") or "")):
      wait_alu = [items[after]]
      after += 1
    return tdm, end_tdm, tcnt, after, wait_alu

  # Pass 1: identify merges.  Do NOT harden the FREE fine wait to dscnt=0 —
  # last-consumer placement already owns the proof; a full drain here waits
  # for unrelated in-flight LRs (MERGE_BAR FREE proof was ~0.86× with tcnt_exact).
  merge_at = {}
  for i in range(len(items)):
    if not _isWgBarrier(items[i]):
      continue
    probed = _probe_merge(i)
    if probed is None:
      continue
    merge_at[i] = probed

  if not merge_at:
    return module

  out = Module(module.name)
  i = 0
  merged = 0
  while i < len(items):
    if i not in merge_at:
      out.add(items[i])
      i += 1
      continue

    tdm, end_tdm, tcnt, after, wait_alu = merge_at[i]
    between = items[i + 1:tdm]
    tdm_block = items[tdm:end_tdm + 1]
    middle = items[end_tdm + 1:tcnt]
    old_cmt = getattr(items[tcnt], "comment", "") or ""
    # Exact N at FREE: await latest TDM before this site's issue (not the
    # site TDM — it is not out yet).  forceDrain keeps the explicit 0 path.
    exact_n = 0 if forceDrain else tensorcntExactN(items, i)
    tag = "MERGE_BAR drain" if forceDrain else "MERGE_BAR tcnt_exact"
    tcnt_inst = SWaitTensorcnt(
        tensorcnt=exact_n,
        comment=("Wait TDM (%s N=%d): " % (tag, exact_n) + old_cmt).strip())
    ready_bar = items[tcnt + 1]

    for inst in between:
      out.add(inst)
    out.add(tcnt_inst)
    out.add(ready_bar)
    for inst in wait_alu:
      out.add(inst)
    for inst in tdm_block:
      out.add(inst)
    for inst in middle:
      out.add(inst)
    merged += 1
    i = after

  return out if merged else module


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
