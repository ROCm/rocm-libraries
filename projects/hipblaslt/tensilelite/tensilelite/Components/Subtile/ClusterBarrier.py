# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cluster-scope barrier handshake for the subtile mainloop.

The handshake is split into a *signal* half and a *wait* half so the wait can be
moved away from the signal, hiding the cluster barrier's cross-CU latency behind
the WMMAs that issue in between instead of exposing it as a stall.

Slide-safe FREE / multicast entry order (no WG↔cluster interleave):

  wave0:  CS-3 → CW-3 → WS-1 → WW-1 → tensor_load_to_lds
  others:         CW-3 → WS-1 → WW-1 → tensor_load_to_lds

CW is placed immediately before the WGP signal that pairs with FREE (SIG_EARLY
``s_barrier_signal -1`` when present, otherwise the FREE wait itself).  CS is
emitted ``CBSIG_WMMA`` WMMAs earlier.  Non-FREE sections are a no-op.
"""

from __future__ import annotations

from rocisa.code import Label, Module
from rocisa.container import sgpr
from rocisa.instruction import (SBarrier, BranchInstruction, SCBranchSCC0,
                                SCmpEQU32, TensorLoadToLds,
                                MFMAInstruction, MXMFMAInstruction)

# gemm_wt GEMM_V7_CBSIG_WMMA: WMMAs between cluster signal and its wait.
CBSIG_WMMA_DEFAULT = 16
# gemm_wt GEMM_V7_CBW_SIGONLY: in-loop wait on signalling wave only.
# Default off: every wave takes the cluster wait so peers arrive aligned at WS.
CBW_SIGONLY_DEFAULT = False

_isWgBarrier = lambda x: isinstance(x, SBarrier) and "s_barrier_wait -1" in str(x)
_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


def _isWgSignalOnly(inst) -> bool:
  """True for a standalone WGP signal (SIG_EARLY), not a glued signal+wait."""
  if not isinstance(inst, SBarrier):
    return False
  text = str(inst)
  return "s_barrier_signal -1" in text and "s_barrier_wait -1" not in text


def _findNextMFMA(items, start):
  """Index of the first MFMA at/after ``start``, or ``None`` if none follows."""
  for j in range(start, len(items)):
    if _isMMA(items[j]):
      return j
  return None


def _findFreeBarriers(items):
  """Indices of all FREE WGP barriers, in order.

  FREE is a ``s_barrier_wait -1`` that is followed by a ``TensorLoadToLds``
  before the next WGP barrier (READY has ``s_wait_tensorcnt`` then barrier,
  with no TDM between).  Per-operand FREE (LRLookaheadB > 1) gives several.
  """
  frees = []
  for i, inst in enumerate(items):
    if not _isWgBarrier(inst):
      continue
    j = i + 1
    while j < len(items) and not _isWgBarrier(items[j]):
      if isinstance(items[j], TensorLoadToLds):
        frees.append(i)
        break
      j += 1
  return frees


def _findClusterWaitAnchor(items, freeIdx):
  """Index at which to insert CW-3.

  Prefer immediately before SIG_EARLY ``s_barrier_signal -1`` so the barrier
  order is CS→CW→WS→WW.  If there is no standalone WGP signal before FREE
  (glued FREE signal+wait), insert immediately before FREE itself.
  """
  for i in range(freeIdx - 1, -1, -1):
    if _isWgSignalOnly(items[i]):
      return i
    if _isWgBarrier(items[i]):
      break
  return freeIdx


def _splitSignalAtBranch(signalItems):
  """Split the signal block at its single wave-0 election branch."""
  brIdxs = [k for k, s in enumerate(signalItems)
            if isinstance(s, SCBranchSCC0)]
  assert len(brIdxs) == 1, \
      "signal block must contain exactly one wave-0 election branch"
  brIdx = brIdxs[0]
  return signalItems[:brIdx], signalItems[brIdx:]


def subtileClusterBarrierSignal(writer, kernel) -> Module:
  """Wave-0-only cluster_barrier signal.

  Wave 0 alone issues the cluster_barrier signal; all other waves branch over
  it. Ends at the ``skipPreSignal`` label so all waves fall through to whatever
  work follows; the matching wait is emitted later by ``subtileClusterBarrierWait``.
  """
  mod = Module("subtile_cluster_barrier_signal")
  skipPreSignal = Label(writer.labels.getUniqueNamePrefix("skipCBPreSignal"), "", 16)
  # Elect wave 0 to issue the single cluster_barrier signal.
  mod.add(SCmpEQU32(sgpr("WaveIdx"), 0, "wave 0?"))
  mod.add(SCBranchSCC0(skipPreSignal.getLabelName(), "only wave 0 signals the cluster"))
  mod.add(SBarrier(True, False, True, "cluster_barrier signal"))
  mod.add(skipPreSignal)
  return mod


def subtileClusterBarrierWait(writer, kernel, sigOnly=False) -> Module:
  """Cluster_barrier wait that closes the handshake.

  ``sigOnly`` (gemm_wt ``CBW_SIGONLY``): wave 0 alone executes
  ``s_barrier_wait -3``; other waves branch over it.  Peers then meet at the
  following WGP signal/wait.  Default ``sigOnly=False`` is a full rendezvous.
  """
  mod = Module("subtile_cluster_barrier_wait")
  if not sigOnly:
    mod.add(SBarrier(True, True, True, "cluster_barrier wait"))
    return mod
  skip = Label(writer.labels.getUniqueNamePrefix("skipCBWait"), "", 16)
  mod.add(SCmpEQU32(sgpr("WaveIdx"), 0, "CBW_SIGONLY: wave 0 waits the cluster"))
  mod.add(SCBranchSCC0(skip.getLabelName(),
                       "other waves skip; WGP barrier orders them"))
  mod.add(SBarrier(True, True, True, "cluster_barrier wait (sigonly)"))
  mod.add(skip)
  return mod


def _emitSignalHiddenBehindMFMA(result, items, mfmaIdx, signalItems, skip):
  """Emit wave-0 elect around ``items[mfmaIdx]``: s_cmp; MFMA; branch+signal.

  Marks consumed indices in ``skip``. Caller must not have already emitted
  ``items[mfmaIdx]``.
  """
  pre, post = _splitSignalAtBranch(signalItems)
  for s in pre:
    result.add(s)
  result.add(items[mfmaIdx])
  skip.add(mfmaIdx)
  for s in post:
    result.add(s)


def _insertLegacyAfterWgBarrier(module, writer, kernel, signalItems, waitItems):
  """Preloop / non-FREE: signal after first WGP barrier, wait glued to signal."""
  items = module.flatitems()
  result = Module(module.name)
  done = False
  skip = set()
  for i, inst in enumerate(items):
    if i in skip:
      continue
    result.add(inst)
    if not done and _isWgBarrier(inst):
      done = True
      mfmaIdx = _findNextMFMA(items, i + 1)
      if mfmaIdx is None:
        for s in signalItems:
          result.add(s)
      else:
        for k in range(i + 1, mfmaIdx):
          result.add(items[k])
          skip.add(k)
        _emitSignalHiddenBehindMFMA(result, items, mfmaIdx, signalItems, skip)
  if not done:
    head = Module(module.name)
    head.add(SBarrier(True, False, False))
    head.add(SBarrier(True, True, False, "workgroup barrier wait"))
    for s in signalItems:
      head.add(s)
    for inst in result.flatitems():
      head.add(inst)
    result = head

  skipLabel = next(s for s in signalItems if isinstance(s, Label))
  items = result.flatitems()
  patched = Module(result.name)
  waitPlaced = False
  for inst in items:
    patched.add(inst)
    if not waitPlaced and inst is skipLabel:
      for w in waitItems:
        patched.add(w)
      waitPlaced = True
  if not waitPlaced:
    for w in waitItems:
      patched.add(w)
  return patched


def insertClusterBarrier(module, writer, kernel,
                         cbsigWmma=CBSIG_WMMA_DEFAULT,
                         cbwSigOnly=CBW_SIGONLY_DEFAULT):
  """Splice the cluster-scope barrier handshake into the post-schedule order.

  No-op unless ``ClusterBarrier`` is enabled.

  On a FREE site (WGP barrier followed by ``TensorLoadToLds``): emit
  ``CS-3`` / ``CW-3`` so the barrier order is ``CS→CW→WS→WW`` (wave 0) or
  ``CW→WS→WW`` (peers).  ``CW`` sits immediately before SIG_EARLY ``WS-1``
  when present, otherwise immediately before FREE ``WW-1``.  ``CS`` is
  ``cbsigWmma`` WMMAs earlier.  Slide 25: next ``CS`` only after this FREE
  ``WW``.

  On non-FREE sections (no WGP→TDM site): no-op.  Call
  ``_insertLegacyAfterWgBarrier`` directly only if that older behaviour is
  required.

  Returns a rebuilt Module; the input is left untouched.
  """
  if not kernel.get("ClusterBarrier"):
    return module

  signalItems = subtileClusterBarrierSignal(writer, kernel).flatitems()

  # ClusterBarrier is only supported on gfx1250.
  assert writer.states.asmCaps.get("HasClusterBarrier", False), \
      "ClusterBarrier requires the HasClusterBarrier asm capability"

  items = list(module.flatitems())
  freeIdxs = _findFreeBarriers(items)
  if not freeIdxs:
    # No FREE site (WGP barrier followed by tensor_load_to_lds): this section
    # is not a multicast entry.  Skipping avoids slide-21 "long stall" legacy
    # CS+CW glued onto READY/tail handshakes that do not guard a new TDM.
    return module

  # One CS/CW pair per FREE site.  A site's CS is picked among the WMMAs after
  # the previous site's FREE (slide 25), so the handshakes never interleave.
  waitAt = {}    # item index -> wait items emitted right before it
  signalAt = {}  # MFMA index -> signal items hidden behind it
  for n, freeIdx in enumerate(freeIdxs):
    sig = signalItems if n == 0 else subtileClusterBarrierSignal(writer, kernel).flatitems()
    wait = subtileClusterBarrierWait(writer, kernel, sigOnly=cbwSigOnly).flatitems()
    # CW before SIG_EARLY WS (or before FREE if WS is glued into FREE).
    waitAnchor = _findClusterWaitAnchor(items, freeIdx)
    lo = freeIdxs[n - 1] + 1 if n > 0 else 0
    mfmaBefore = [i for i in range(lo, waitAnchor) if _isMMA(items[i])]
    if len(mfmaBefore) >= cbsigWmma:
      signalMfma = mfmaBefore[-cbsigWmma]
    elif mfmaBefore:
      signalMfma = mfmaBefore[0]
    else:
      signalMfma = None
    if signalMfma is not None:
      signalAt[signalMfma] = sig
      waitAt[waitAnchor] = wait
    else:
      # No MFMA before wait: emit signal intact just before the wait.
      waitAt[waitAnchor] = list(sig) + list(wait)

  result = Module(module.name)
  skip = set()
  for i, inst in enumerate(items):
    if i in skip:
      continue

    # Cluster wait immediately before WS (SIG_EARLY) or FREE WW.
    for w in waitAt.pop(i, ()):
      result.add(w)

    # Signal pinned to an MFMA cbsigWmma before wait (hide elect behind it).
    if i in signalAt:
      _emitSignalHiddenBehindMFMA(result, items, i, signalAt.pop(i), skip)
      continue

    result.add(inst)

  return result
