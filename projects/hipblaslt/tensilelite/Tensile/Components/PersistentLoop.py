# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent lifecycle, prefetch/reuse orchestration, and emitter state."""

from contextlib import contextmanager
from copy import deepcopy
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr
from rocisa.instruction import SMovB32, VMovB32, VReadfirstlaneB32, SCmpEQU32, SCBranchSCC1, SBranch, SWaitCnt, SBitcmp1B32, SBarrier, SLShiftRightB32, SLongBranchNegative
from Tensile.ExecutionPolicy import isPersistent
from math import ceil, log2
from ..Component import Component
import abc

class PersistentKernelState:
    """Emitter state and borrowed registers for the persistent loop lifecycle."""

    def papTileIdentityNames(self, kernel):
        return self.states.currentTileWork.borrowedIdentity(kernel)

    @contextmanager
    def allocPapTileIdentity(self, kernel, subtile=False):
        names = self.papTileIdentityNames(kernel)
        if subtile:
            base = self.vgprPool.checkOutAligned(len(names), 1, "Subtile PAP tile identity")
            try:
                yield {name: base + i for i, name in enumerate(names)}
            finally:
                self.vgprPool.checkIn(base)
        else:
            with self.allocTmpSgpr(len(names), alignment=1, tag="PAP tile identity") as tmp:
                yield {name: tmp.idx + i for i, name in enumerate(names)}

    def papCheckpointCurrentTileIdentity(self, kernel, prevTile, subtile=False):
        module = Module("papCheckpointCurrentTileIdentity")
        move, register = (VMovB32, vgpr) if subtile else (SMovB32, sgpr)
        for name in self.papTileIdentityNames(kernel):
            module.add(move(dst=register(prevTile[name]), src=sgpr(name), comment="checkpoint %s for PAP restore" % name))
        return module

    def papRestoreCurrentTileIdentity(self, kernel, prevTile, subtile=False):
        module = Module("papRestoreCurrentTileIdentity")
        move, register = (VReadfirstlaneB32, vgpr) if subtile else (SMovB32, sgpr)
        for name in self.papTileIdentityNames(kernel):
            module.add(move(dst=sgpr(name), src=register(prevTile[name]), comment="restore current %s after PAP" % name))
        return module

    def halfPlrPrefetchAcrossPersistentLabel(self):
      if not hasattr(self.states, "halfPlrPapLabel"):
        self.states.halfPlrPapLabel = Label(
            self.labels.getNameInc("HalfPlrPrefetchAcrossPersistent"), "")
      return self.states.halfPlrPapLabel

    def halfPlrPrefetchAcrossPersistentReturnLabel(self, loopCopy):
      if not hasattr(self.states, "halfPlrPapReturnLabels"):
        self.states.halfPlrPapReturnLabels = {}
      if loopCopy not in self.states.halfPlrPapReturnLabels:
        self.states.halfPlrPapReturnLabels[loopCopy] = Label(
            self.labels.getNameInc("ReturnFromHalfPlrPAP_%u" % loopCopy), "")
      return self.states.halfPlrPapReturnLabels[loopCopy]

    def halfPlrPrefetchAcrossPersistentEntryLabel(self, loopCopy):
      if not hasattr(self.states, "halfPlrPapEntryLabels"):
        self.states.halfPlrPapEntryLabels = {}
      if loopCopy not in self.states.halfPlrPapEntryLabels:
        self.states.halfPlrPapEntryLabels[loopCopy] = Label(
            self.labels.getNameInc("HalfPlrPAPEntry_%u" % loopCopy), "")
      return self.states.halfPlrPapEntryLabels[loopCopy]

    def callHalfPlrPrefetchAcrossPersistent(self, kernel, loopCopy):
      """Jump to the shared PAP block before the final HalfPLR loop trip."""
      module = Module("callHalfPlrPrefetchAcrossPersistent")
      if not (kernel["HalfPLR"] and kernel["PrefetchAcrossPersistent"]):
        return module

      module.add(SCmpEQU32(
          src0=self.loopCounter(kernel, self.states.unrollIdx),
          src1=1,
          comment="HalfPLR PAP before final unrolled-loop trip"))
      module.add(SCBranchSCC1(
          labelName=self.halfPlrPrefetchAcrossPersistentEntryLabel(loopCopy).getLabelName(),
          comment="short branch to out-of-line PAP entry when LoopCounter == 1"))
      module.add(self.halfPlrPrefetchAcrossPersistentReturnLabel(loopCopy))
      return module

    def emitHalfPlrPrefetchAcrossPersistentBlock(
        self, kernel, tensorParametersA, tensorParametersB):
      """Emit one PAP body shared by all three rotating HalfPLR loop copies."""
      module = Module("halfPlrPrefetchAcrossPersistentBlock")
      if not (kernel["HalfPLR"] and kernel["PrefetchAcrossPersistent"]):
        return module

      afterLabel = Label(self.labels.getNameInc("AfterHalfPlrPAPBlock"), "")
      module.add(SBranch(
          labelName=afterLabel.getLabelName(),
          comment="normal loop-exit path skips out-of-line HalfPLR PAP block"))
      entryLabels = getattr(self.states, "halfPlrPapEntryLabels", {})
      returnLabels = getattr(self.states, "halfPlrPapReturnLabels", {})
      assert entryLabels, "no HalfPLR loop copy registered a PAP entry"
      assert set(entryLabels) == set(returnLabels), \
          "every HalfPLR PAP entry needs its own return label"
      # The selector is live from an entry trampoline, through the shared body, to the
      # return dispatch, so hold it across all of them; nested PAP code then cannot
      # reuse it.
      with self.allocTmpSgpr(1, tag="HalfPlrPAPReturnSelector") as selector:
        returnSelector = selector.idx
        for loopCopy in sorted(entryLabels):
          module.add(entryLabels[loopCopy])
          module.add(SMovB32(
              dst=sgpr(returnSelector),
              src=loopCopy,
              comment="select branch-back label for HalfPLR loop copy %u" % loopCopy))
          module.add(SBranch(
              labelName=self.halfPlrPrefetchAcrossPersistentLabel().getLabelName(),
              comment="join shared HalfPLR PAP body"))
        module.add(self.halfPlrPrefetchAcrossPersistentLabel())
        module.add(self.prefetchAcrossPersistent(
            kernel, tensorParametersA, tensorParametersB, skipBarrier=False))

        returnIds = sorted(returnLabels)
        for loopCopy in returnIds[:-1]:
          module.add(SCmpEQU32(
              src0=sgpr(returnSelector),
              src1=loopCopy,
              comment="return to HalfPLR loop copy %u" % loopCopy))
          module.add(SCBranchSCC1(
              labelName=returnLabels[loopCopy].getLabelName(),
              comment="return to matching HalfPLR loop copy"))
        module.add(SBranch(
            labelName=returnLabels[returnIds[-1]].getLabelName(),
            comment="return to one-trip HalfPLR loop entry"))
      module.add(afterLabel)
      return module

    def isPrefetchAcrossPersistentEnabled(self, kernel):
      """Consume PAP capability resolved during solution derivation."""
      return bool(kernel.get("_PrefetchAcrossPersistentEnabled",
                             kernel.get("PrefetchAcrossPersistent", 0)))

    # ReuseAcrossPersistent has no predicate of its own: emitters read
    # kernel["ReuseAcrossPersistent"] the way they read kernel["HalfPLR"]. Every
    # precondition RAP has is a reject in Solution.assignDerivedParameters -- or,
    # for nonpersistent work, a clear of the flag itself -- so a solution that
    # reaches codegen with the flag set has already been checked.
    #
    # RAP is deliberately independent of PrefetchAcrossPersistent. They share a
    # persistent loop and nothing else: PAP overlaps the next tile's loads with
    # this tile's compute, RAP holds A across tiles, and RAP 1 with PAP 0 is a
    # supported combination. RAP requires full-K tiles; PAP also supports partitioned work.

    def rapResidentKTiles(self, kernel):
      """How many k-tiles of A/MXSA are held resident; 1 (i.e. no residency) when RAP is off."""
      if not kernel["ReuseAcrossPersistent"]:
        return 1
      return kernel["_RAPNumResidentKTiles"]

    # Suffix worn by the labels of the reuse copy of the compute section. Empty
    # everywhere else, so every other kernel -- and RAP's own fill copy -- keeps the
    # names it had before this feature existed.
    RAP_ITERN_SUFFIX = "_RAPIterN"
    rapLabelSuffix = ""

    @contextmanager
    def rapIterNLabels(self):
      """Suffix every label built in this block, for the reuse copy of the section."""
      self.rapLabelSuffix = self.RAP_ITERN_SUFFIX
      try:
        yield
      finally:
        self.rapLabelSuffix = ""

    def rapLabel(self, name):
      """Disambiguate a label built from a literal across the two emissions.

      Labels from labels.getNameInc are already unique: the counter keeps running
      across both copies. The ones that collide are built from literals, which is
      deliberate -- it is what lets a branch emitted at one site agree with a target
      emitted at another. Emitting the section twice then defines the same label
      twice, which the CFG builder rejects outright.
      """
      return name + self.rapLabelSuffix

    def unrollLoopEndLabelName(self, kernel, loopIdx, nta=0, ntb=0):
      """Name of the unroll loop's end label.

      Built in one place because closeLoop defines it and anything leaving the loop
      early has to name the same thing. Two independent constructions of one label
      name is how the reuse copy's "do not enter LoopL" escape came to point at the
      fill copy's end.
      """
      loopChar = self.states.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      strNta = "" if kernel["AdaptiveGemmNTAB"] == 0 else "_NTA%s"%nta
      strNtb = "" if kernel["AdaptiveGemmNTAB"] == 0 else "_NTB%s"%ntb
      return self.rapLabel("LoopEnd%s%s%s"%(loopChar, strNta, strNtb))

    def rapGetName(self, name):
      """labels.getName with the reuse copy's suffix applied.

      getName deliberately returns the same string every time so a branch and its
      target agree, which is exactly what collides when the section is emitted
      twice, so the suffix goes on top.
      """
      return self.rapLabel(self.labels.getName(name))

    def rapPersistentLoopEntryLabel(self, kernel):
      """Label the persistent loop branches back to.

      With the compute section peeled, only the very first tile runs the copy that
      fills the resident A registers; every later tile re-enters at the second copy.
      """
      return "RAP_IterN" if kernel["ReuseAcrossPersistent"] else "PersistentLoopStart"

    # Per-tensor emitter state that advances as the compute section is emitted.
    # Cannot go through saveLocalPointers: these keys are created during emission,
    # so at snapshot time (before the first copy) they do not exist yet.
    _RAP_TENSOR_KEYS = ("localReadOffset", "localReadSwapByteOffset", "localWriteSwapByteOffset")

    def _rapTensorParams(self, kernel, tPA, tPB):
      params = [tPA, tPB]
      if kernel["ProblemType"]["MXBlockA"]:
        params.append(tPA["MX"])
      if kernel["ProblemType"]["MXBlockB"]:
        params.append(tPB["MX"])
      return params

    def rapSnapshotEmitterState(self, kernel, tPA, tPB):
      """Capture the emitter state that emitting the compute section once mutates.

      Containers are deep-copied: the section mutates several of them in place
      (freeSgprVarPool, lraTileProperties, the per-iteration local-write skip list),
      so keeping a reference would "restore" an object that had already been
      changed, and the second copy would silently allocate different temporaries.
      """
      states = {}
      for name, value in vars(self.states).items():
        states[name] = deepcopy(value) if isinstance(value, (dict, list, set)) else value
      tensors = [{key: tp[key] for key in self._RAP_TENSOR_KEYS if key in tp}
                 for tp in self._rapTensorParams(kernel, tPA, tPB)]
      # The register pools and the SGPR definition table go along: the section
      # checks registers out and in and undefines SGPRs, so without this the second
      # copy would release what the first already released. This is the same
      # deepcopy-and-swap-back the OptNLL alternative path uses.
      pools = (deepcopy(self.vgprPool), deepcopy(self.sgprPool), deepcopy(self.sgprs))
      return states, tensors, pools

    def rapRestoreEmitterState(self, kernel, tPA, tPB, snapshot):
      states, tensors, pools = snapshot
      for name, value in states.items():
        setattr(self.states, name, value)
      for tp, saved in zip(self._rapTensorParams(kernel, tPA, tPB), tensors):
        for key in self._RAP_TENSOR_KEYS:
          if key in saved:
            tp[key] = saved[key]
          elif key in tp:
            del tp[key]
      savedVgprPool, savedSgprPool, savedSgprs = pools
      # Keep whatever peak the first copy reached; allocation is driven by pool size.
      savedVgprPool.appendPool(self.vgprPool.size())
      savedSgprPool.appendPool(self.sgprPool.size())
      self.vgprPool = savedVgprPool
      self.sgprPool = savedSgprPool
      self.sgprs = savedSgprs

    def rapUnrolledLoopCopies(self, kernel):
      """How many copies of the unroll loop body RAP emits inside the loop shell.

      One per resident k-tile: each must live in a section that is emitted once,
      because the register set it addresses is a codegen-time constant, and under RAP
      the loop shell owns all of them.

      The NGLL and NLL sections used to own the last PrefetchGlobalRead of them, but
      their k-tile indices are absolute (numKTiles-1-remainPgr and numKTiles-1), so
      they only ever fit a K that uses every resident k-tile. Owning the whole range
      here is what lets one kernel serve a range of K. What the drain sections did is
      now done inside the body: not issuing global reads near the end is the
      branchless TDM disable, and not issuing the lookahead local reads is replaced by
      draining them at the exit.
      """
      if not kernel["ReuseAcrossPersistent"]:
        return 1
      return self.rapResidentKTiles(kernel)

    def rapStoreWithheldVgprs(self, kernel):
      """Registers RAP keeps out of the store's hands: the resident A plus its scales.

      Derived from the same ranges the reclaim sites use, so the guard and the
      reclaim cannot drift apart.
      """
      if not kernel["ReuseAcrossPersistent"]:
        return 0
      abStart, _ = self.rapReclaimableValuABRange(kernel)
      mxsStart, _ = self.rapReclaimableValuMXSABRange(kernel)
      return (abStart - self.states.a.startVgprValu) + mxsStart

    def rapResidentBufferIdx(self, kernel, tc, unwrappedIdx, defaultIdx):
      """Buffer-set index for an A/MXSA access, or None when it leaves the resident block.

      Without RAP the buffer index wraps every LoopIters (`u % numVgprBuffer`),
      because only the PLR window is held. With RAP the whole K extent is held, so
      the index is absolute: the section's own k-tile times LoopIters, plus the
      iteration within it.

      Returning None means "do not emit this access". That happens for local reads,
      which run one iteration ahead of the MFMAs: on the last resident k-tile the
      lookahead addresses the k-tile after the block. That access is the prefetch
      of the *next* tile's A -- exactly the transfer RAP exists to remove -- and
      without the guard the index would wrap onto resident k-tile 0 and overwrite
      it.
      """
      if tc not in ("A", "MXSA") or not kernel["ReuseAcrossPersistent"]:
        return defaultIdx
      if self.rapLookaheadLeavesBlock(kernel, unwrappedIdx):
        return None
      return self.states.rapKTileIdx * kernel["LoopIters"] + unwrappedIdx

    def rapIsLastResidentSection(self, kernel):
      """Is the section being emitted the last one of the resident block?

      Work whose only consumer is the next section is dead here. Two things qualify:
      the local-read address swaps, which select the buffer the next section would
      read from, and the loop counter decrement, whose readers were the next
      section's silencing gate and this section's own early exit -- and the last
      section has no early exit, while after the loop the counter is written before
      it is read again.

      Leaving the addresses unswapped does not leak into the next persistent
      iteration: every tile entry resets them with v_and 0xffff.
      """
      if not kernel["ReuseAcrossPersistent"]:
        return False
      return self.states.rapKTileIdx == self.rapResidentKTiles(kernel) - 1

    def rapTdmPrefetchIsDead(self, kernel):
      """Is this section's TDM prefetch silenced on every path that reaches it?

      The runtime gate in globalReadDo zeroes the descriptor when the loop counter
      has PrefetchGlobalRead or fewer k-tiles left. Section i (1-based) only runs
      when K covers it, and it sees the counter at (K / DepthU) - (i - 1), so it is
      silenced exactly when i >= K / DepthU - PrefetchGlobalRead + 1. K / DepthU
      ranges up to the count the kernel holds, so the sections silenced for *every*
      K it serves are the last PrefetchGlobalRead of the block, and only those: with
      8 resident k-tiles and PGR 2, section 6 is live at K = 8 tiles and cannot go.

      For those last sections the descriptor writes and the transfer are dead weight
      rather than a runtime decision, so the load need not be issued at all. This is
      what the NGLL/NLL drain used to achieve by not containing a prefetch.

      Callers must also be in the unroll loop (mode 1). The pre-loop prologue issues
      the first PrefetchGlobalRead transfers and has to keep them, and rapKTileIdx
      does not describe a section there -- it still holds whatever the previous
      section left.
      """
      if not kernel["ReuseAcrossPersistent"] or not kernel["PrefetchGlobalRead"]:
        return False
      return self.states.rapKTileIdx >= \
          self.rapResidentKTiles(kernel) - kernel["PrefetchGlobalRead"]

    def rapLookaheadLeavesBlock(self, kernel, unwrappedIdx):
      """Does this access run past the last resident k-tile?

      Local reads run an iteration ahead of the MFMAs, so on the last section the
      lookahead addresses a k-tile that is not there. For A that would wrap onto
      resident k-tile 0 and overwrite it, which is why rapResidentBufferIdx refuses
      the access. B and its scales cannot wrap -- they are re-read every tile from
      a PLR window, so their buffer index is unaffected -- but the read is still
      pointless: nothing consumes it, because the section it was fetched for does
      not exist. Skipping it saves the LDS traffic and, more importantly, stops
      handing the exit path loads that are still in flight with no consumer.

      Only the section's own position decides this, so B asks the same question
      without going through the A-only buffer-index path.
      """
      if not kernel["ReuseAcrossPersistent"]:
        return False
      idx = self.states.rapKTileIdx * kernel["LoopIters"] + unwrappedIdx
      return idx >= self.rapResidentKTiles(kernel) * kernel["LoopIters"]

    def rapReclaimableValuABRange(self, kernel):
      """(start, size) of the ValuA/B block that may be lent out as scratch.

      Under RAP the ValuA half stays live across the whole persistent loop -- that
      is the feature -- so only the ValuB half is lendable. ValuA and ValuB are
      allocated contiguously, so the B half is [b.startVgprValu, lastValuAB).
      """
      start = self.states.b.startVgprValu if kernel["ReuseAcrossPersistent"] \
              else self.states.a.startVgprValu
      return start, self.states.lastValuAB - start

    def rapReclaimableValuMXSABRange(self, kernel):
      """(start, size) of the ValuMXSA/B block that may be lent out as scratch.

      MXSA is pinned at the bottom of the register file (s_set_vgpr_msb has no
      field for the WMMA scale operands, so scales must live in v0-v255), so the
      resident MXSA block is a prefix and the lendable part starts after it.
      """
      start = (self.states.mxsa.startVgprValu + self.states.mxsa.numVgprValu) \
              if kernel["ReuseAcrossPersistent"] else 0
      return start, self.states.lastValuMXSAB - start

class PersistentLoop(Component):
    """
    Persistent loop code.
    """
    def __call__(self):
        assert(0)

    def initialize(self, writer, kernel):
        processing = Component.TileProcessingStrategy.find(writer)
        writer.states.currentTileWork = processing.tileWork(kernel)
        return Component.WorkAssignment.find(writer).initialize(writer, kernel, processing)

    def activateReservedOrAcquire(self, writer, kernel, tPA, tPB):
        processing = Component.TileProcessingStrategy.find(writer)
        return Component.WorkAssignment.find(writer).activateReservedOrAcquire(writer, kernel, processing, tPA, tPB)

    def compute(self, writer, kernel, tensorParametersA, tensorParametersB, module, expand, tPM):
        if kernel["ReuseAcrossPersistent"]:
          # Peel the compute section in two: the first tile runs a copy that fills the
          # resident A registers, every later tile re-enters at the second copy and
          # reuses them. Only the compute half is duplicated -- the store is the bulk
          # of the kernel and stays shared, which keeps the instruction cache footprint
          # roughly unchanged.
          # Record which batch the fill below loads A for. Read here, at the loop head,
          # because tile activation advances the cursor past this tile and PAP goes on to
          # overwrite WorkGroup2 with the next tile's, so neither survives the section.
          module.add(Component.WorkAssignment.find(writer).peekTileBatch(writer, kernel, "RAPResidentBatch"))
          snapshot = writer.rapSnapshotEmitterState(kernel, tensorParametersA, tensorParametersB)
          writer.states.rapDeferSgprUndef = True
          pack = writer._persistentComputeSection(kernel, tensorParametersA, tensorParametersB, module, expand, tPM)

          storeJoin = Label("RAP_StoreJoin", "")
          # Conditional, not s_branch: the CFG builder gives an unconditional branch no
          # fall-through edge, so iterN would have no predecessor, the backend would
          # skip it, and it would come out with no waitcnts and no barriers at all --
          # which a functional simulator still reports as PASSED.
          cursor = Component.TileProcessingStrategy.find(writer).staticPartition().cursor
          module.add(SCmpEQU32(src0=sgpr(cursor), src1=sgpr(cursor),
                               comment="RAP: always true; keeps a CFG edge into iterN"))
          module.add(SCBranchSCC1(labelName=storeJoin.getLabelName(),
                                  comment="RAP: first tile skips the reuse copy"))
          module.add(Label("RAP_IterN", ""))
          # A is indexed by the batch, so the resident copy only serves tiles in the
          # batch it was filled from. This copy has no A loads to supply any other, so
          # send a tile that crossed a batch boundary through the fill copy instead --
          # it pays one tile's worth of reloads and leaves A resident for the tiles
          # after it. Ahead of everything else in the section: the cursor still names
          # this tile and nothing has claimed WorkGroup* for it, so the loop head can
          # take it from the top.
          with writer.allocTmpSgpr(1, tag="RAPBatchGuard") as sBatch:
            module.add(Component.WorkAssignment.find(writer).peekTileBatch(writer, kernel, sBatch.idx))
            module.add(SCmpEQU32(src0=sgpr(sBatch.idx), src1=sgpr("RAPResidentBatch"),
                                 comment="RAP: is the resident A this tile's batch?"))
            module.add(writer.longBranchScc0(Label("PersistentLoopStart", ""), posNeg=-1,
                                           comment="RAP: batch changed, refill A"))
          module.add(self.reinitWaveIdx(writer, kernel))
          writer.rapRestoreEmitterState(kernel, tensorParametersA, tensorParametersB, snapshot)
          # From here on A and its scales come from the resident registers, so the
          # reuse copy issues neither their LDS reads nor their global transfers.
          # numReadsPerIter* feeds the analytically computed s_wait_dscnt immediate
          # and the scheduler's latency budget, so it has to shrink with them.
          writer.states.rapDropAResidentLoads = True
          writer.states.numReadsPerIterA = 0
          writer.states.numReadsPerIterMXSA = 0
          with writer.rapIterNLabels():
            pack = writer._persistentComputeSection(kernel, tensorParametersA, tensorParametersB, module, expand, tPM)
          module.add(storeJoin)
          # Drain the local reads a small-K exit left in flight, before the store
          # reuses their registers.
          #
          # A section reads one k-tile ahead. When K does not fill the block, the
          # section that takes the early exit has already issued that lookahead for a
          # successor this K never runs, so those ds_reads are still outstanding with
          # nothing to consume them. The store then borrows the value registers as
          # scratch -- the MX scale blocks sit at the bottom of the register file, so
          # computeStoreVgprs writes v36/v37 while a pending read of ValuMXSB targets
          # v34-v37 -- and a read landing late puts LDS data over the wave offset the
          # store is about to compute with. It is a write-after-write, so a functional
          # simulator retires the read at issue, always sees the VALU win, and cannot
          # fail on it.
          #
          # Here rather than at the loop exit for two reasons. This is where every path
          # into the store converges, so one wait covers both copies and every K. And
          # it is late: the tile-mapping and store setup run in between, so the reads
          # have long landed and the wait is normally already satisfied. Only the
          # exit that skips the PAP block -- a workgroup's last tile -- actually needs
          # it; the PAP block drains before its own LDS writes.
          module.add(SWaitCnt(dscnt=0,
                              comment="RAP: drain a small-K exit's unconsumed local reads"))
        else:
          pack = writer._persistentComputeSection(kernel, tensorParametersA, tensorParametersB, module, expand, tPM)

        return pack

    def prefetch(self, writer, kernel, tPA, tPB, *, subtile=False, preloopGrModule=None, skipBarrier=False):
        module = Module("prefetchAcrossPersistentSubtile" if subtile else "prefetchAcrossPersistent")
        if not writer.isPrefetchAcrossPersistentEnabled(kernel):
            return module
        if subtile and (not kernel.get("UseSubtileImpl") or tPA is None or tPB is None):
            return module
        processing = Component.TileProcessingStrategy.find(writer)
        assignment = Component.WorkAssignment.find(writer)
        skip = Label(writer.labels.getNameInc("PersistentSkipSubtilePrefetch" if subtile else "PersistentSkipPrefetch"), "")
        module.add(processing.prefetchEligibility(writer, kernel, skip))
        module.add(assignment.reserveNext(writer, kernel, skip))
        module.add(SBitcmp1B32(src0=sgpr("PersistentPrefetchState"), src1=0, comment="Next assignment data already issued?"))
        module.add(SCBranchSCC1(labelName=skip.getLabelName(), comment="Do not prefetch a reservation twice"))
        if not skipBarrier:
            module.add(SBarrier(comment="Subtile PAP: sync before next-tile prefetch" if subtile else "PAP: sync before next-tile prefetch"))
        with writer.allocPapTileIdentity(kernel, subtile=subtile) as previous:
            module.add(writer.papCheckpointCurrentTileIdentity(kernel, previous, subtile=subtile))
            module.add(processing.prefetchAcrossPersistentSetupNextTile(writer, kernel, tPA, tPB, skipLroReset=True))
            rapOuter = writer.states.rapInPapNextTilePrefetch
            writer.states.rapInPapNextTilePrefetch = kernel["ReuseAcrossPersistent"]
            try:
                if subtile:
                    module.add(writer.setupPrefetchAcrossPersistentSubtileLoads(kernel, tPA, tPB, preloopGrModule))
                else:
                    module.add(self.prefetchClassicLoads(writer, kernel, tPA, tPB))
            finally:
                writer.states.rapInPapNextTilePrefetch = rapOuter
            module.add(writer.papRestoreCurrentTileIdentity(kernel, previous, subtile=subtile))
        if (not subtile and kernel["enableTDMA"] and kernel["enableTDMB"] and not kernel["NoTailLoop"] and not kernel["HalfPLR"]):
            module.add(writer.papTdmUpdateDescriptor(kernel, tPA, tPB, preservePapBank=False))
            if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
                module.add(writer.papTdmUpdateDescriptor(kernel, tPA["MX"], tPB["MX"], preservePapBank=False))
        module.add(skip)
        return module

    def prefetchClassicLoads(self, writer, kernel, tPA, tPB):
        module = Module("Persistent classic PAP load handoff")
        if kernel["enableTDMA"] and kernel["enableTDMB"]:
            module.add(writer.papTdmUpdateDescriptor(kernel, tPA, tPB))
            if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
                module.add(writer.papTdmUpdateDescriptor(kernel, tPA["MX"], tPB["MX"]))
        loopCounterName = writer.loopCounterName(kernel, writer.states.unrollIdx)
        snapshotLoopCounter = kernel["HalfPLR"] or writer.states.currentTileWork.k_start is not None
        if snapshotLoopCounter:
            previous = writer.vgprPool.checkOutAligned(2, 1, "PAP loop counters")
            module.add(VMovB32(dst=vgpr(previous), src=sgpr(loopCounterName), comment="checkpoint LoopCounter for PAP restore"))
            module.add(VMovB32(dst=vgpr(previous + 1), src=sgpr("OrigLoopCounter"), comment="checkpoint OrigLoopCounter for PAP restore"))
        module.add(writer.calculateLoopNumIter(kernel, tPA, tPB, writer.states.unrollIdx))
        module.add(writer.setupPrefetchAcrossPersistentLoads(kernel, tPA, tPB, isOptNLL=True))
        if snapshotLoopCounter:
            module.add(VReadfirstlaneB32(dst=sgpr(loopCounterName), src=vgpr(previous), comment="restore LoopCounter after PAP"))
            module.add(VReadfirstlaneB32(dst=sgpr("OrigLoopCounter"), src=vgpr(previous + 1), comment="restore OrigLoopCounter after PAP"))
            writer.vgprPool.checkIn(previous)
        if kernel["enableTDMA"] and kernel["enableTDMB"]:
            module.add(writer.papTdmSaveLdsBank(kernel))
        return module

    @abc.abstractmethod
    def openPersistentLoop(self, writer, kernel):
        pass

    @abc.abstractmethod
    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        pass

    @abc.abstractmethod
    def recalcLocalReadAddressesAB(self, writer, kernel):
        pass

    @abc.abstractmethod
    def closePersistentLoop(self, writer, kernel):
        pass

class PersistentLoopOff(PersistentLoop):
    kernel = {"TileProcessingStrategy": "None"}

    def initialize(self, writer, kernel):
        return Module("Ordinary kernel initialization")

    def activateReservedOrAcquire(self, writer, kernel, tPA, tPB):
        return Module("Ordinary workgroup activation")

    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop Off openPersistentLoop")
        return module

    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        module = Module("PersistentLoop Off recalcLocalWriteAddresses")
        return module

    def recalcLocalReadAddressesAB(self, writer, kernel):
        module = Module("PersistentLoop Off recalcLocalReadAddressesAB")
        return module

    def closePersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop Off closePersistentLoop")
        return module

class PersistentLoopOn(PersistentLoop):
    # Shared persistent-loop boundaries

    @classmethod
    def matches(cls, writer, debug=False):
        return isPersistent(writer.states.kernel)

    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop On openPersistentLoop")

        # Label start of persistent loop
        module.addComment2("Persistent Loop Start")
        persistentLabel = Label(label="PersistentLoopStart", comment="")
        module.add(persistentLabel)

        module.add(self.reinitWaveIdx(writer, kernel))

        # TODO remove?
        # kStr += inst("s_add_u32", sgpr("PersistentLoopIter"), sgpr("PersistentLoopIter"), hex(1), "Inc PersistentLoop Iter")     # Back-up: not needed now
        #kStr += str(Code.WaitCnt(self.version, 0,0,"wait for outstanding stores"))
        return module

    def reinitWaveIdx(self, writer, kernel):
        """Re-init sgprWaveIdx, which every persistent iteration needs afresh.

        TDM init reads s[sgprWaveIdx], but the same sgpr is later UNDEFed and
        reused as a temp, so on the second iteration the value would be stale.
        Emitted at whichever label the persistent loop actually branches back to,
        which under ReuseAcrossPersistent is the peeled iterN entry rather than
        the loop head.
        """
        module = Module("PersistentLoop On reinitWaveIdx")
        if kernel["enableTDMA"] or kernel["enableTDMB"]:
            wavelen = kernel["WavefrontSize"]
            with writer.allocTmpSgpr(1, tag="PersistentLoopOn_openPersistentLoop_tmpSgprRes") as tmpSgprRes:
                module.add(VReadfirstlaneB32(sgpr(tmpSgprRes.idx), vgpr("Serial"), "first tId"))
                module.add(SLShiftRightB32(sgpr("WaveIdx"), ceil(log2(wavelen)), sgpr(tmpSgprRes.idx),
                                           "re-init WaveIdx for persistent loop iteration"))
        return module

    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        module = Module("PersistentLoop On recalcLocalWriteAddresses")

        if getattr(writer, "oriLwa%s" % tc) is None:
            setattr(writer, "oriLwa%s" % tc, writer.vgprPool.checkOut(1, "OriLocalWriteddr%s" % tc))
            module.add(VMovB32(dst=vgpr(getattr(writer, "oriLwa%s" % tc)), src=vgpr("LocalWriteAddr%s" % tc), comment="back up LWA for persistent kernel + wider local read"))

        return module

    def recalcLocalReadAddressesAB(self, writer, kernel):
        # Tile activation rebuilds local-read addresses. Backend tail handling
        # owns any temporary pointer save/restore inside a tile.
        return Module("PersistentLoop On recalcLocalReadAddressesAB")

    def closePersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop closePersistentLoop")
        module.add(Label("PersistentLoopClose", ""))
        if kernel.get("DebugPersistentKernelLoopForever", False):
            with writer.allocTmpSgpr(3, tag="PersistentLoop_close") as tmp:
                module.add(SLongBranchNegative(Label("PersistentLoopStart", ""), tmp))
        else:
            module.add(Component.WorkAssignment.find(writer).closeLoop(writer, kernel))
        return module
