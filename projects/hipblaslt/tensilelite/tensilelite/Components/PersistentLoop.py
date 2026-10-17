# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent lifecycle, prefetch/reuse orchestration, and emitter state."""

from contextlib import contextmanager
from copy import deepcopy
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister, EXEC, VCC, MemTokenData
from rocisa.instruction import SMovB32, VMovB32, VReadfirstlaneB32, SCmpEQU32, SCBranchSCC1, SBranch, SWaitCnt, SBitcmp1B32, SBarrier, SLShiftRightB32, SLongBranchNegative
from ..ExecutionPolicy import isPersistent, isStreamK
from math import ceil, log2
from ..Component import Component
from ..Common import persistentSpatialCluster
import abc
from rocisa.instruction import SAddU32, SAndB32, SCBranchSCC0, SCSelectB32, SLShiftLeftB32, SMinU32, SMulI32, SOrB32, SSubU32
from rocisa.instruction import VReadlaneB32, VWritelaneB32
from rocisa.functions import scalarUInt32DivideAndRemainder

class PersistentKernelState:
    """Emitter state and borrowed registers for the persistent loop lifecycle."""

    @staticmethod
    def papClearPreparedAddress(comment):
        # Bit 0 (issued), bit 1 (bank), and bit 31 (reserved) remain untouched.
        return SAndB32(dst=sgpr("PersistentPrefetchState"),
                       src0=sgpr("PersistentPrefetchState"), src1=hex(0xfffffffb),
                       comment=comment)

    def papAddressCarrierPlan(self, kernel, tPA, tPB, *, enabled=True):
        """Check every simultaneous scratch shape on copies of the real pools."""
        if (not enabled or self.states.tailloopInNll or kernel.get("AdaptiveGemmNTAB", 0)
                or not self.papTdmSupportsShadowAddressPair(kernel, tPA, tPB)):
            return None
        scales = bool(kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"])
        if scales and not self.papTdmSupportsShadowAddressPair(kernel, tPA["MX"], tPB["MX"]):
            return None
        streamK = isStreamK(kernel)
        if streamK and getattr(self.states, "papNextTileVgpr", -1) < 0:
            return None
        if streamK:
            # The packed path borrows only temporary scalars. Keep its complete
            # preparation/consumer scratch within the existing scalar budget;
            # fragmentation must not turn this optimization into pool growth.
            scalarLimit = min(self.sgprPool.size(), self.states.regCaps["MaxSgpr"])
            shapes = (((2, 2), (4, 2)),                   # address preparation
                      ((2, 2), (1, 1), (3, 2), (1, 1)), # descriptor consumer
                      ((4, 2), (1, 1)))                 # rebuilt increments
            if streamK:
                shapes = (((2, 2), (3, 2)), *shapes[1:],
                          ((4, 2), (1, 1), (1, 1)))  # SK mapper and magic constants
            try:
                for shape in shapes:
                    scratch = deepcopy(self.sgprPool)
                    for count, alignment in shape:
                        scratch.checkOutAligned(count, alignment, "PAP packed preflight", preventOverflow=False)
                    if scratch.size() > scalarLimit:
                        return None
                # The unchanged ordinary fallback has its own scratch lifetime
                # after preparation has returned every temporary to the pool.
                fallback = deepcopy(self.sgprPool)
                fallback.checkOutAligned(3, 2, "PAP mapping fallback", preventOverflow=False)
                if persistentSpatialCluster(kernel):
                    fallback.checkOutAligned(4, 2, "PAP cluster fallback", preventOverflow=False)
                if fallback.size() > self.states.regCaps["MaxSgpr"]:
                    return None
                vectors = deepcopy(self.vgprPool)
                vectors.checkOutAligned(4, 2, "PAP packed mapper preflight", preventOverflow=False)
                if vectors.size() > self.states.regCaps["MaxVgpr"]:
                    return None
            except RuntimeError:
                return None
            return {"packed": True}
        limit = self.states.regCaps["MaxSgpr"]
        # A kernel already at the architectural budget retains its old path.
        if self.sgprPool.size() >= limit:
            return None
        # Producer, swap, ordinary mapper/WGM, descriptor refresh/full tail,
        # counter and elected-handoff scratch lifetimes. These are nested
        # checkouts, so alignment gaps and fragmentation matter as well as sum.
        # Rebuilt NGLL increments nest a parity temporary under four wrap words.
        shapes = (((4, 2),), ((1, 1),),
                  ((4, 2), (1, 1)),
                  ((1, 1), (2, 2), (1, 1), (1, 1)),
                  ((1, 1), (1, 1), (1, 1), (1, 1)),
                  ((1, 1), (3, 2), (1, 1)), ((1, 1), (1, 1), (2, 2)),
                  ((4, 2),))
        needsNestedDivide = (persistentSpatialCluster(kernel)
                             or (not kernel["NoTailLoop"] and kernel["DepthU"] & (kernel["DepthU"] - 1)))
        shapes += (((3, 2), (4, 2)),) if needsNestedDivide else (((3, 2),),)
        try:
            vectors = deepcopy(self.vgprPool)
            # DefaultWGM's ordinary fallback borrows four aligned VGPRs;
            # producer division and descriptor arithmetic need only two.
            vectors.checkOutAligned(4, 2, "PAP mapper preflight", preventOverflow=False)
            if vectors.size() > self.states.regCaps["MaxVgpr"]:
                return None
        except RuntimeError:
            return None
        for size in ((7, 5) if scales else (5,)):
            pool = deepcopy(self.sgprPool)
            try:
                base = pool.checkOutAligned(size, 2, "PAP address carrier preflight", preventOverflow=False)
                fits = True
                for shape in shapes:
                    scratch = deepcopy(pool)
                    for count, alignment in shape:
                        scratch.checkOutAligned(count, alignment, "PAP scratch preflight", preventOverflow=False)
                    if scratch.size() > limit:
                        fits = False
                        break
                if fits:
                    return dict(base=base, size=size, data=base, mx=base + 2 if size == 7 else None,
                                coordinates={i: base + size - 3 + i for i in range(3)})
            except RuntimeError:
                continue
        return None

    @contextmanager
    def allocPapAddressCarrier(self, kernel, tPA, tPB, *, enabled=True):
        plan = self.papAddressCarrierPlan(kernel, tPA, tPB, enabled=enabled)
        self.states.papAddressCarrier = plan
        if plan is None:
            yield None
            return
        base = None
        if not plan.get("packed"):
            base = self.sgprPool.checkOutAligned(plan["size"], 2, "PAP address carrier", preventOverflow=False)
            assert base == plan["base"]
        # These placeholders were emitted before any tile activation or NGLL
        # bypass can run. Fill them only after the bounded allocation succeeds.
        self.states.papAddressActivationReset.add(self.papClearPreparedAddress(
            "PAP: invalidate prepared address at tile activation"))
        self.states.papAddressRoutingReset.add(self.papClearPreparedAddress(
            "PAP: invalidate prepared address before NGLL entry or bypass"))
        try:
            yield plan
        finally:
            if base is not None:
                self.sgprPool.checkIn(base)
            self.states.papAddressCarrier = None

    def papPrepareNextTileAddresses(self, kernel, tPA, tPB):
        """Return mapping and pure addresses together for standalone callers."""
        mapper, address = self.papNextTileAddressModules(kernel, tPA, tPB)
        module = Module("PAP NGLL address preparation")
        module.add(mapper)
        module.add(address)
        return module

    def papNextTileAddressModules(self, kernel, tPA, tPB):
        """Prepare StreamK addresses in packed lanes; keep the scalar DP path."""
        plan = getattr(self.states, "papAddressCarrier", None)
        if plan is not None and plan.get("packed") and isStreamK(kernel):
            return self.papStreamKNextTileAddressModules(kernel, tPA, tPB)
        return self._papNextTileAddressModules(kernel, tPA, tPB)

    def _papNextTileAddressModules(self, kernel, tPA, tPB):
        """Separate EXEC-changing mapping from schedulable scalar addresses."""
        mapper = Module("PAP next coordinate mapping")
        address = Module("PAP pure next addresses")
        module = mapper
        plan = getattr(self.states, "papAddressCarrier", None)
        if plan is None:
            return mapper, address
        c = plan["coordinates"]
        # The data-address output does not become live until mapping finishes.
        # Borrow it to save EXEC, and restore the exact incoming mask before the
        # pure leaf overwrites that word. Full EXEC makes the divider's readlane
        # independent of whichever lanes the current computation has active.
        module.add(SMovB32(dst=sgpr(plan["data"]), src=EXEC(), comment="PAP mapper save current EXEC"))
        module.add(SMovB32(dst=EXEC(), src=-1, comment="PAP mapper scratch uses full EXEC"))
        with self.allocTmpSgpr(4, alignment=2, tag="PAP address preparation scratch") as scratch:
            a, b, d, rem = (scratch.idx + i for i in range(4))
            vectors = self.vgprPool.checkOut(2, "PAP next tile divide")
            try:
                vr = ContinuousRegister(vectors, 2)
                module.add(SMovB32(dst=sgpr(a), src=sgpr("NextTile"), comment="PAP speculative next tile"))
                if persistentSpatialCluster(kernel):
                    cs, ck = kernel["ClusterDim"]
                    module.add(SLShiftRightB32(dst=sgpr(a), shiftHex=int(log2(cs * ck)), src=sgpr(a)))
                    module.add(SAddU32(dst=sgpr(b), src0=sgpr("NumWorkGroups0"), src1=cs - 1))
                    module.add(SLShiftRightB32(dst=sgpr(b), shiftHex=int(log2(cs)), src=sgpr(b)))
                    module.add(SAddU32(dst=sgpr(d), src0=sgpr("NumWorkGroups1"), src1=ck - 1))
                    module.add(SLShiftRightB32(dst=sgpr(d), shiftHex=int(log2(ck)), src=sgpr(d)))
                    module.add(SMulI32(dst=sgpr(d), src0=sgpr(d), src1=sgpr(b)))
                    divisor = b
                else:
                    module.add(SMulI32(dst=sgpr(d), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1")))
                    divisor = "NumWorkGroups0"
                module.add(scalarUInt32DivideAndRemainder(qReg=c[2], dReg=a, divReg=d, rReg=rem,
                    tmpVgprRes=vr, wavewidth=32, doRemainder=True, comment="PAP next tile batch"))
                module.add(scalarUInt32DivideAndRemainder(qReg=c[1], dReg=rem, divReg=divisor, rReg=c[0],
                    tmpVgprRes=vr, wavewidth=32, doRemainder=True, comment="PAP next tile coordinates"))
                if persistentSpatialCluster(kernel):
                    module.add(SAndB32(dst=sgpr(a), src0=sgpr("NextTile"), src1=cs * ck - 1))
                    module.add(SLShiftLeftB32(dst=sgpr(c[0]), shiftHex=int(log2(cs)), src=sgpr(c[0])))
                    module.add(SAndB32(dst=sgpr(rem), src0=sgpr(a), src1=cs - 1))
                    module.add(SAddU32(dst=sgpr(c[0]), src0=sgpr(c[0]), src1=sgpr(rem)))
                    module.add(SLShiftLeftB32(dst=sgpr(c[1]), shiftHex=int(log2(ck)), src=sgpr(c[1])))
                    module.add(SLShiftRightB32(dst=sgpr(a), shiftHex=int(log2(cs)), src=sgpr(a)))
                    module.add(SAddU32(dst=sgpr(c[1]), src0=sgpr(c[1]), src1=sgpr(a)))
                    for i in (0, 1):
                        module.add(SSubU32(dst=sgpr(rem), src0=sgpr("NumWorkGroups%u" % i), src1=1))
                        module.add(SMinU32(dst=sgpr(c[i]), src0=sgpr(c[i]), src1=sgpr(rem), comment="PAP clamp phantom coordinate"))
            finally:
                self.vgprPool.checkIn(vectors)
            module.add(SMovB32(dst=EXEC(), src=sgpr(plan["data"]), comment="PAP mapper restore exact current EXEC"))
            module = address
            module.add(self.papTdmPrepareAddressPair(kernel, tPA, tPB, c, plan["data"], scratch))
            if plan["mx"] is not None:
                module.add(self.papTdmPrepareAddressPair(kernel, tPA["MX"], tPB["MX"], c, plan["mx"], scratch))
            module.add(SMovB32(dst=sgpr(b), src=hex(0x80000004)))
            if not persistentSpatialCluster(kernel):
                # Equality to +1 needs only the encoded low field; sign extension
                # cannot turn any other low value into +1. No branch or join label
                # separates the useful address leaf from NGLL WMMA scheduling.
                mask = 0x3ff if kernel["WorkGroupMappingXCC"] == -1 else 0xffff
                module.add(SAndB32(dst=sgpr(a), src0=sgpr("WGM"), src1=hex(mask)))
                module.add(SCmpEQU32(src0=sgpr(a), src1=1, comment="PAP prepared mapping requires runtime WGM +1"))
                module.add(SCSelectB32(dst=sgpr(b), src0=sgpr(b), src1=0))
            if kernel["ProblemType"].get("SupportUserArgs"):
                module.add(SAndB32(dst=sgpr(a), src0=sgpr("ArgType"), src1=0xff))
                module.add(SCmpEQU32(src0=sgpr(a), src1=0, comment="PAP prepared address requires direct ArgType 0"))
                module.add(SCSelectB32(dst=sgpr(b), src0=sgpr(b), src1=0))
            module.add(SOrB32(dst=sgpr("PersistentPrefetchState"), src0=sgpr("PersistentPrefetchState"), src1=sgpr(b),
                              comment="PAP record branchless address preparation validity"))
        return mapper, address

    @staticmethod
    def papPackedIdentityNames(kernel):
        names = ["WorkGroup0", "WorkGroup1", "WorkGroup2"]
        if isStreamK(kernel):
            names += ["StreamKLocalStart", "StreamKLocalEnd"]
        elif persistentSpatialCluster(kernel):
            names += ["PersistentPhantomTile"]
        return names

    def papStreamKNextTileAddressModules(self, kernel, tPA, tPB):
        """Prepare a static K range without scalar carriers across compute.

        Lanes 0..4 hold next coordinates and bounds; 5..8 hold addresses until
        issue. Lanes 9..13 save current identity and 14..15 save EXEC/VCC while
        mapping. All borrowed lanes are free before the descriptor snapshot.
        """
        mapper = Module("PAP StreamK next range mapping")
        address = Module("PAP pure next addresses")
        payload = vgpr(self.states.papNextTileVgpr)
        mapper.add(self.papPackedSaveIdentity(kernel))
        with self.allocTmpSgpr(1, tag="PAP StreamK save masks") as mask:
            for lane, register in ((14, EXEC()), (15, VCC())):
                mapper.add(SMovB32(dst=sgpr(mask.idx), src=register))
                mapper.add(VWritelaneB32(dst=payload, src0=sgpr(mask.idx), src1=lane))
        mapper.add(SMovB32(dst=EXEC(), src=-1, comment="PAP mapper scratch uses full EXEC"))
        processing = Component.TileProcessingStrategy.find(self)
        mapper.add(processing.prefetchAcrossPersistentSetupNextTile(self, kernel, tPA, tPB, skipLroReset=True))
        for lane, name in enumerate(self.papPackedIdentityNames(kernel)):
            mapper.add(VWritelaneB32(dst=payload, src0=sgpr(name), src1=lane,
                                     comment="PAP retain next " + name))
        with self.allocTmpSgpr(1, tag="PAP StreamK restore masks") as mask:
            for lane, register in ((14, EXEC()), (15, VCC())):
                mapper.add(VReadlaneB32(dst=sgpr(mask.idx), src0=payload, src1=lane))
                mapper.add(SMovB32(dst=register, src=sgpr(mask.idx), comment="PAP mapper restore exact mask"))
        with self.allocTmpSgpr(2, alignment=2, tag="PAP StreamK address scratch") as output:
            with self.allocTmpSgpr(3, alignment=2, tag="PAP StreamK pure address scratch") as scratch:
                coordinates = {i: self.sgprs["WorkGroup%u" % i] for i in range(3)}
                pairs = [(tPA, tPB, 5)]
                if kernel["ProblemType"]["MXBlockA"]:
                    pairs += [(tPA["MX"], tPB["MX"], 7)]
                for a, b, lane in pairs:
                    address.add(self.papTdmPrepareAddressPair(kernel, a, b, coordinates, output.idx, scratch))
                    for i in range(2):
                        address.add(VWritelaneB32(dst=payload, src0=sgpr(output.idx + i), src1=lane + i,
                                                  comment="PAP retain prepared StreamK address"))
                address.add(self.papPackedRestoreIdentity(kernel))
                a, b = scratch.idx, scratch.idx + 1
                mask = 0x3ff if kernel["WorkGroupMappingXCC"] == -1 else 0xffff
                address.add(SAndB32(dst=sgpr(a), src0=sgpr("WGM"), src1=hex(mask)))
                address.add(SCmpEQU32(src0=sgpr(a), src1=1, comment="PAP prepared mapping requires runtime WGM +1"))
                address.add(SCSelectB32(dst=sgpr(b), src0=hex(0x80000004), src1=0))
                if kernel["ProblemType"].get("SupportUserArgs"):
                    address.add(SAndB32(dst=sgpr(a), src0=sgpr("ArgType"), src1=0xff))
                    address.add(SCmpEQU32(src0=sgpr(a), src1=0, comment="PAP prepared address requires direct ArgType 0"))
                    address.add(SCSelectB32(dst=sgpr(b), src0=sgpr(b), src1=0))
                address.add(SOrB32(dst=sgpr("PersistentPrefetchState"), src0=sgpr("PersistentPrefetchState"),
                                  src1=sgpr(b), comment="PAP record StreamK address preparation validity"))
        return mapper, address

    def papPackedSaveIdentity(self, kernel):
        """Borrow lanes that become free before the descriptor snapshot."""
        module = Module("PAP packed save current identity")
        first = 9 if isStreamK(kernel) else 8
        for lane, name in enumerate(self.papPackedIdentityNames(kernel), first):
            module.add(VWritelaneB32(dst=vgpr(self.states.papNextTileVgpr),
                src0=sgpr(name), src1=lane, comment="PAP pack current " + name))
        return module

    def papPackedRestoreIdentity(self, kernel):
        module = Module("PAP packed restore current identity")
        first = 9 if isStreamK(kernel) else 8
        for lane, name in enumerate(self.papPackedIdentityNames(kernel), first):
            module.add(VReadlaneB32(dst=sgpr(name), src0=vgpr(self.states.papNextTileVgpr),
                src1=lane, comment="PAP unpack current " + name))
        return module

    def papPackedUpdateDescriptors(self, kernel, tPA, tPB):
        """Consume both encoded addresses through one temporary scalar pair."""
        module = Module("PAP consume packed addresses")
        payload = vgpr(self.states.papNextTileVgpr)
        names = self.papPackedIdentityNames(kernel)
        # Multicast handoff still belongs to the current phantom peer. Only
        # StreamK also activates the next range bounds for descriptor setup.
        for i, name in enumerate(names if isStreamK(kernel) else names[:3]):
            module.add(VReadlaneB32(dst=sgpr(name), src0=payload,
                src1=i, comment="PAP unpack next " + name))
        with self.allocTmpSgpr(2, alignment=2, tag="PAP unpack address") as tmp:
            first = 5 if isStreamK(kernel) else 4
            pairs = [(tPA, tPB, first)]
            if kernel["ProblemType"]["MXBlockA"]:
                pairs += [(tPA["MX"], tPB["MX"], first + 2)]
            for a, b, lane in pairs:
                for i in range(2):
                    module.add(VReadlaneB32(dst=sgpr(tmp.idx + i), src0=payload,
                        src1=lane + i, comment="PAP unpack prepared address"))
                module.add(self.papTdmUpdateDescriptor(kernel, a, b, preparedGlobalAddr=tmp.idx))
        return module

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
        module = Module("Persistent activation")
        writer.states.papAddressActivationReset = Module("PAP address activation reset")
        module.add(writer.states.papAddressActivationReset)
        module.add(Component.WorkAssignment.find(writer).activateReservedOrAcquire(writer, kernel, processing, tPA, tPB))
        return module

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
        def handoff():
            if persistentSpatialCluster(kernel):
                return assignment.persistentClusterPrefetchHandoff(writer, kernel)
            handoffModule = Module("Persistent PAP local handoff")
            if not skipBarrier:
                handoffModule.add(SBarrier(comment="Subtile PAP: sync before next-tile prefetch" if subtile else "PAP: sync before next-tile prefetch"))
            return handoffModule

        # This path only prepares scalar descriptors and loop state. Let it run
        # while the previous bank's reads finish, then release the bank just
        # before issuing the same first-PGR group. Keep other paths unchanged.
        lateHandoff = (not subtile and kernel.get("ScheduleIterAlg") == 4
                       and writer.papTdmSupportsIncrementalUpdate(kernel))
        if not lateHandoff:
            module.add(handoff())
        carrier = getattr(writer.states, "papAddressCarrier", None)
        if carrier is not None and not subtile:
            module.add(self.prefetchPreparedClassic(writer, kernel, tPA, tPB, processing, carrier,
                                                   beforeIssue=handoff if lateHandoff else None))
        else:
            module.add(self.prefetchBorrowedIdentity(writer, kernel, tPA, tPB, processing,
                subtile=subtile, preloopGrModule=preloopGrModule, beforeIssue=handoff if lateHandoff else None))
        if (not subtile and kernel["enableTDMA"] and kernel["enableTDMB"] and not kernel["NoTailLoop"] and not kernel["HalfPLR"]):
            module.add(writer.papTdmUpdateDescriptor(kernel, tPA, tPB, preservePapBank=False))
            if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
                module.add(writer.papTdmUpdateDescriptor(kernel, tPA["MX"], tPB["MX"], preservePapBank=False))
        module.add(skip)
        return module

    def prefetchBorrowedIdentity(self, writer, kernel, tPA, tPB, processing, *,
                                 subtile=False, preloopGrModule=None, beforeIssue=None):
        module = Module("Persistent PAP borrowed identity")
        with writer.allocPapTileIdentity(kernel, subtile=subtile) as previous:
            module.add(writer.papCheckpointCurrentTileIdentity(kernel, previous, subtile=subtile))
            module.add(processing.prefetchAcrossPersistentSetupNextTile(writer, kernel, tPA, tPB, skipLroReset=True))
            rapOuter = writer.states.rapInPapNextTilePrefetch
            writer.states.rapInPapNextTilePrefetch = kernel["ReuseAcrossPersistent"]
            try:
                if subtile:
                    module.add(writer.setupPrefetchAcrossPersistentSubtileLoads(kernel, tPA, tPB, preloopGrModule))
                else:
                    module.add(self.prefetchClassicLoads(
                        writer, kernel, tPA, tPB,
                        beforeIssue=beforeIssue))
            finally:
                writer.states.rapInPapNextTilePrefetch = rapOuter
            module.add(writer.papRestoreCurrentTileIdentity(kernel, previous, subtile=subtile))
        return module

    def prefetchPreparedClassic(self, writer, kernel, tPA, tPB, processing, carrier, *, beforeIssue=None):
        if carrier.get("packed"):
            return self.prefetchPackedClassic(writer, kernel, tPA, tPB, processing, beforeIssue=beforeIssue)
        module = Module("Persistent PAP prepared address consumer")
        fallback = Label(writer.labels.getNameInc("PapAddressFallback"), "")
        ready = Label(writer.labels.getNameInc("PapAddressReady"), "")
        coordinates = carrier["coordinates"]
        module.add(SBitcmp1B32(src0=sgpr("PersistentPrefetchState"), src1=2, comment="PAP next addresses prepared by NGLL?"))
        module.add(SCBranchSCC0(labelName=fallback.getLabelName(), comment="PAP use ordinary mapping on preparation bypass"))
        with writer.allocTmpSgpr(1, tag="PAP swap coordinates") as tmp:
            for i in range(3):
                module.add(SMovB32(dst=sgpr(tmp.idx), src=sgpr("WorkGroup%u" % i)))
                module.add(SMovB32(dst=sgpr("WorkGroup%u" % i), src=sgpr(coordinates[i]), comment="PAP borrow prepared next coordinate"))
                module.add(SMovB32(dst=sgpr(coordinates[i]), src=sgpr(tmp.idx), comment="PAP retain current coordinate for common restore"))
        module.add(writer.papTdmUpdateDescriptor(kernel, tPA, tPB, preparedGlobalAddr=carrier["data"]))
        if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
            module.add(writer.papTdmUpdateDescriptor(kernel, tPA["MX"], tPB["MX"], preparedGlobalAddr=carrier["mx"]))
        module.add(SBranch(labelName=ready.getLabelName()))
        module.add(fallback)
        for i in range(3):
            module.add(SMovB32(dst=sgpr(coordinates[i]), src=sgpr("WorkGroup%u" % i), comment="checkpoint current coordinate on PAP fallback"))
        if persistentSpatialCluster(kernel):
            module.add(SMovB32(dst=sgpr(carrier["data"]), src=sgpr("PersistentPhantomTile"), comment="PAP fallback checkpoint phantom locally"))
        module.add(processing.prefetchAcrossPersistentSetupNextTile(writer, kernel, tPA, tPB, skipLroReset=True))
        if persistentSpatialCluster(kernel):
            module.add(SMovB32(dst=sgpr("PersistentPhantomTile"), src=sgpr(carrier["data"]), comment="PAP fallback restore current phantom before issue"))
        module.add(writer.papTdmUpdateDescriptor(kernel, tPA, tPB))
        if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
            module.add(writer.papTdmUpdateDescriptor(kernel, tPA["MX"], tPB["MX"]))
        module.add(ready)
        module.add(self.prefetchClassicLoads(writer, kernel, tPA, tPB, beforeIssue=beforeIssue, updateDescriptors=False))
        for i in range(3):
            module.add(SMovB32(dst=sgpr("WorkGroup%u" % i), src=sgpr(coordinates[i]), comment="restore current WorkGroup%u after PAP" % i))
        return module

    def prefetchPackedClassic(self, writer, kernel, tPA, tPB, processing, *, beforeIssue=None):
        module = Module("PAP packed ordinary consumer")
        fallback = Label(writer.labels.getNameInc("PapPackedFallback"), "")
        ready = Label(writer.labels.getNameInc("PapPackedReady"), "")
        module.add(writer.papPackedSaveIdentity(kernel))
        module.add(SBitcmp1B32(src0=sgpr("PersistentPrefetchState"), src1=2))
        module.add(SCBranchSCC0(labelName=fallback.getLabelName()))
        module.add(writer.papPackedUpdateDescriptors(kernel, tPA, tPB))
        module.add(SBranch(labelName=ready.getLabelName()))
        module.add(fallback)
        module.add(processing.prefetchAcrossPersistentSetupNextTile(writer, kernel, tPA, tPB, skipLroReset=True))
        if persistentSpatialCluster(kernel):
            module.add(VReadlaneB32(dst=sgpr("PersistentPhantomTile"),
                src0=vgpr(writer.states.papNextTileVgpr), src1=11))
        module.add(writer.papTdmUpdateDescriptor(kernel, tPA, tPB))
        if kernel["ProblemType"]["MXBlockA"]:
            module.add(writer.papTdmUpdateDescriptor(kernel, tPA["MX"], tPB["MX"]))
        module.add(ready)
        module.add(self.prefetchClassicLoads(writer, kernel, tPA, tPB,
            beforeIssue=beforeIssue, updateDescriptors=False))
        module.add(writer.papPackedRestoreIdentity(kernel))
        return module

    def prefetchClassicLoads(self, writer, kernel, tPA, tPB, *, beforeIssue=None, updateDescriptors=True):
        module = Module("Persistent classic PAP load handoff")
        if updateDescriptors and kernel["enableTDMA"] and kernel["enableTDMB"]:
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
        if beforeIssue is not None:
            # Build this here, while the identity checkpoint remains allocated,
            # so the elected-wave temporary cannot alias the saved identity.
            # Descriptor parity branches have joined. Keep any predicate
            # created or consumed by the issue helper after this handoff;
            # the current OptNLL PAP path does not need an issue predicate.
            module.add(beforeIssue())
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
        if kernel.get("_PersistentVectorEpilogueLds", False) and not kernel["PrefetchAcrossPersistent"]:
            # PAP0 has no live successor tile in LDS. Keep the original shared
            # layout and hand it back to compute only after every wave finishes
            # its vector reads. Emit after GW (and the main-loop barrier pass),
            # so this protects the persistent back edge without an early drain.
            module.add(SWaitCnt(dscnt=0, comment="finish vector epilogue before compute LDS reuse"))
            barrier = SBarrier(comment="hand epilogue LDS back to persistent compute")
            barrier.setMemToken(MemTokenData([writer.states.memTokenEpilogue]))
            module.add(barrier)
        module.add(Component.WorkAssignment.find(writer).persistentClusterNextTileArrive(writer, kernel))
        if kernel.get("DebugPersistentKernelLoopForever", False):
            with writer.allocTmpSgpr(3, tag="PersistentLoop_close") as tmp:
                module.add(SLongBranchNegative(Label("PersistentLoopStart", ""), tmp))
        else:
            module.add(Component.WorkAssignment.find(writer).closeLoop(writer, kernel))
        return module
