# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cluster (multicast) TDM load component.

Owns the multicast ("cluster load") mask machinery: mask value compute, the
``MulticastMask*`` SGPR declare/undeclare, the combined-vs-split topology
decision, and the per-load-site descriptor attach. Each method receives the
SGPR operands the caller already holds rather than re-allocating them.
Capability-selected (``HasTDM`` + ``TDMInst == 3``), like ``TensorDataMoverLoad``.
"""

from ..Component import ClusterLoad
from ..Common import clusterEnabled, streamKDual2DMulticast, streamKMulticast
from typing import Mapping
from rocisa.code import Module, Label
from rocisa.container import sgpr
from rocisa.instruction import SLShiftLeftB32, SMulI32, SBitcmp1B32, SCBranchSCC1, SBranch


class ClusterLoadTDM(ClusterLoad):
    asmCaps = {"HasTDM": True}
    kernel  = {"TDMInst": 3}

    def __call__(self, writer: "KernelWriterAssembly", kernel: Mapping):
        # Abstract-satisfying no-op, mirrors TensorDataMoverLoad.__call__.
        pass

    # -- topology decision ---------------------------------------------------

    def usesCombinedMask(self, kernel: Mapping) -> bool:
        """True when the single-parity combined ``MulticastMask`` applies.

        Single source of truth for the combined-vs-split decision. Subtile and
        StreamK DP multicast both need the split A/B masks: subtile issues A and
        B on every wave (no wave-parity split), and StreamK broadcasts only B
        across the [C,1] cluster while A stays per-workgroup -- the combined
        parity mask would be wrong for both.
        """
        if streamKMulticast(kernel):
            return False
        tdmA: bool = kernel["enableTDMA"]
        tdmB: bool = kernel["enableTDMB"]
        return tdmA and tdmB and kernel["NumWaves"] > 1 and not kernel.get("UseSubtileImpl")

    def maskSgprName(self, kernel: Mapping, tc: str, *, subtile: bool = False,
                     waveSeparated: bool = False) -> str:
        """Resolve the multicast-mask SGPR name.

        Wave-separated (non-subtile) uses the combined ``"MulticastMask"``;
        dense/subtile and StreamK multicast use the split ``f"MulticastMask{tc}"``
        (any ``MXS`` prefix stripped). StreamK forces the split name so B never
        resolves to the never-declared combined SGPR.
        """
        if streamKMulticast(kernel):
            string = tc.removeprefix("MXS") if tc.startswith("MXS") else tc
            return f"MulticastMask{string}"
        if waveSeparated and not subtile:
            return "MulticastMask"
        string = tc.removeprefix("MXS") if tc.startswith("MXS") else tc
        return f"MulticastMask{string}"

    def cooperativeThreadPartition(self, kernel: Mapping, tc: str) -> int:
        """Cooperating-workgroup count for ``tc``: ClusterDim[1] (A) / [0] (B)."""
        subTc: str = tc[-1]
        return kernel["ClusterDim"][1] if subTc == "A" else kernel["ClusterDim"][0]

    # -- SGPR declare / undeclare -------------------------------------------

    def declareSgprs(self, writer: "KernelWriter", kernel: Mapping) -> None:
        """Allocate the ``MulticastMask*`` SGPRs."""
        if not kernel["Multicast"]:
            return
        tdmM: bool = kernel["enableTDMMetadata"]
        if self.usesCombinedMask(kernel):
            writer.defineSgpr("MulticastMask", 1)
        else:
            writer.defineSgpr("MulticastMaskA", 1)
            writer.defineSgpr("MulticastMaskB", 1)
        if tdmM:
            writer.defineSgpr("MulticastMaskMetadata", 1)

    def papRefreshesMask(self, kernel: Mapping) -> bool:
        """True when PrefetchAcrossPersistent re-applies the mask after prologue.

        PAP re-emits the TDM descriptor setup (``applyToDescriptor``) on every
        persistent-loop iteration, so the StreamK multicast mask SGPR must stay
        live past the prologue -- freeing it makes those reuses reference an
        undeclared SGPR (``expected absolute expression`` at assembly time).
        """
        return bool(kernel.get("PrefetchAcrossPersistent") and streamKMulticast(kernel))

    def papDropsSelfOnlyMaskA(self, kernel: Mapping) -> bool:
        """True when the PAP-live A mask can be freed because it is self-only.

        ``maskA`` is self-only (``1 << wg_x``) exactly when ``computeMasks`` uses
        ``aPeers == 1`` -- i.e. on the StreamK B-multicast path whenever A is NOT
        a genuine 2-D dual/factored A-multicast.  This mirrors the ``aPeers``
        derivation directly (``streamKMulticast(kernel) and not
        streamKDual2DMulticast(kernel)``, which under PAP is equivalent to
        ``papRefreshesMask(kernel) and not streamKDual2DMulticast(kernel)``):

          * ``[C,1]`` pure multicast  -> aPeers==1 -> self-only -> DROP.
          * FACTORED ``[Cs,Ck]`` (FDPO=0, no StreamKDualMulticast) -> Ck is a
            K-split *reduction* axis, so A is still per-workgroup (aPeers==1) ->
            self-only -> DROP.  This is what lets factored 2-D + PAP + FDPO=0 fit
            the 106-SGPR budget (was sgprs=107 -> ``s_endpgm`` stub overflow).
          * dual-2D (ForceDPOnly 2-D or StreamKDualMulticast) -> Ck peers reuse A
            on N-adjacent tiles, so A is a real multicast (aPeers>1) -> KEEP live.

        Freeing the self-only mask is a no-op re-apply, so codegen is byte-
        identical to keeping it (minus the freed SGPR); the only observable
        effect is the reduced SGPR count.
        """
        return self.papRefreshesMask(kernel) and not streamKDual2DMulticast(kernel)

    def undeclareSgprs(self, writer: "KernelWriter", kernel: Mapping) -> Module:
        """Free the ``MulticastMask*`` SGPRs."""
        mod = Module()
        if not (kernel["Multicast"] and kernel["TDMInst"] != 0):
            return mod
        tdmM: bool = kernel["enableTDMMetadata"]
        refresh: bool = self.papRefreshesMask(kernel)
        dropMaskA: bool = self.papDropsSelfOnlyMaskA(kernel)
        if self.usesCombinedMask(kernel):
            mod.add(writer.undefineSgpr("MulticastMask"))
        else:
            # Under PAP the A mask stays live unless it is self-only (freed then).
            if not refresh or dropMaskA:
                mod.add(writer.undefineSgpr("MulticastMaskA"))
            # Under PAP the B broadcast mask is re-applied every iteration: keep live.
            if not refresh:
                mod.add(writer.undefineSgpr("MulticastMaskB"))
        if tdmM:
            mod.add(writer.undefineSgpr("MulticastMaskMetadata"))
        return mod

    # -- mask value computation ---------------------------------------------

    def computeMasks(self, writer: "KernelWriterAssembly", kernel: Mapping, *,
                     sgprWgX: int, sgprWgY: int, sgprNWgX: int, sTmp: int) -> Module:
        """Compute the multicast mask value(s) into the ``MulticastMask*`` SGPRs.

        The caller passes the operands it already holds (``sgprWgX``/``sgprWgY``/
        ``sgprNWgX`` and ``sTmp`` whose ``+4`` slot is scratch).
        """
        mod = Module()
        if not kernel["Multicast"]:
            return mod
        mod.addComment0("Calculate multicast mask")

        # A-multicast peer count. The dense/subtile path multicasts A across
        # ClusterDim[1] peers. StreamK keeps A per-workgroup (self-only) -- for the
        # 1-D [C,1] cluster ClusterDim[1]==1 already yields maskA=1; for the 2-D
        # factored StreamK cluster ClusterDim[1]=Ck is the K-split (reduction) axis,
        # NOT an A-multicast axis, so force self-only there too. (StreamK's B mask
        # is recomputed in preLoop regardless; only maskA must stay self-only.)
        #
        # EXCEPTION -- 2-D DUAL-multicast (ForceDPOnly 2-D dual multicast AND the
        # standard StreamKDualMulticast path): here the Ck (Y) axis maps to
        # N-ADJACENT output tiles, so A IS multicast across the Ck peers exactly
        # like a dense cluster. Use the dense peer count so the kernel-init
        # maskA/maskB below is byte-correct for BOTH operands on the DP round, and
        # no preLoop overwrite is needed (see StreamK.preLoop /
        # streamKMulticastMaskPredicate). On the standard path the masks are later
        # dropped to self-only at the DP->SK boundary (streamKMulticastBoundaryClear).
        if streamKMulticast(kernel) and not streamKDual2DMulticast(kernel):
            aPeers = 1
        else:
            aPeers = kernel["ClusterDim"][1]
        maskA = 1
        for idx in range(aPeers):
            maskA |= (1 << (idx * kernel["ClusterDim"][0]))

        maskB = (1 << kernel["ClusterDim"][0]) - 1

        if kernel["enableTDMMetadata"]:
            if kernel["ProblemType"]["Sparse"] == 1:
                mod.add(SLShiftLeftB32(dst=sgpr("MulticastMaskMetadata"), shiftHex=sgpr(sgprWgX), src=hex(maskA),\
                                        comment="Setting metadata mask (follows sparse A)"))
            elif kernel["ProblemType"]["Sparse"] == 2:
                mod.add(SMulI32(dst=sgpr(sTmp+4), src0=sgpr(sgprWgY), src1=sgpr(sgprNWgX),\
                                comment="Shift factor: wg_y * nwg_x (metadata)"))
                mod.add(SLShiftLeftB32(dst=sgpr("MulticastMaskMetadata"), shiftHex=sgpr(sTmp+4), src=hex(maskB),\
                                        comment="Setting metadata mask (follows sparse B)"))

        if self.usesCombinedMask(kernel):
            setMulticastMaskLblOdd = Label(f"setMulticastMask_OddWave", "")
            setMulticastMaskLblEven = Label(f"setMulticastMask_EvenWave", "")
            setMulticastMaskLblEnd = Label(f"setMulticastMaskEnd", "")

            mod.add(SBitcmp1B32(sgpr("WaveIdx"), 0, "Check parity of wId"))
            mod.add(SCBranchSCC1(setMulticastMaskLblOdd.getLabelName(), "Jump if wId is odd"))

            mod.add(setMulticastMaskLblEven)
            mod.add(SLShiftLeftB32(dst=sgpr("MulticastMask"), shiftHex=sgpr(sgprWgX), src=hex(maskA),\
                                    comment="Setting maskA for even wave"))
            mod.add(SBranch(setMulticastMaskLblEnd.getLabelName()))
            mod.add(setMulticastMaskLblOdd)
            mod.add(SMulI32(dst=sgpr(sgprWgY), src0=sgpr(sgprWgY), src1=sgpr(sgprNWgX),\
                            comment="Shift factor: wg_y * nwg_x"))
            mod.add(SLShiftLeftB32(dst=sgpr("MulticastMask"), shiftHex=sgpr(sgprWgY), src=hex(maskB),\
                                    comment="Setting maskB for odd wave"))
            mod.add(setMulticastMaskLblEnd)

        else:
            mod.add(SLShiftLeftB32(dst=sgpr("MulticastMaskA"), shiftHex=sgpr(sgprWgX), src=hex(maskA),\
                                    comment="Setting maskA"))

            mod.add(SMulI32(dst=sgpr(sgprWgY), src0=sgpr(sgprWgY), src1=sgpr(sgprNWgX),\
                            comment="Shift factor: wg_y * nwg_x"))
            mod.add(SLShiftLeftB32(dst=sgpr("MulticastMaskB"), shiftHex=sgpr(sgprWgY), src=hex(maskB),\
                                    comment="Setting maskB"))
        return mod

    # -- descriptor attach ---------------------------------------------------

    def applyToDescriptor(self, writer: "KernelWriterAssembly", kernel: Mapping,
                          group1: int | str, tc: str, *, subtile: bool = False,
                          waveSeparated: bool = False) -> Module:
        """OR the multicast mask into descriptor ``Group1[word0]``.

        Folds the ``Multicast and enableCluster`` gate, mask-name choice, and the
        ``SOrB32`` attach; returns an empty ``Module`` when the gate is not met.
        """
        from .TensorDataMover import TensorDataMoverLoad
        mod = Module()
        if kernel["Multicast"] and clusterEnabled(kernel["ClusterDim"]):
            mask = self.maskSgprName(kernel, tc, subtile=subtile, waveSeparated=waveSeparated)
            # Under PAP the self-only A-side mask SGPR is freed (see
            # papDropsSelfOnlyMaskA): the StreamK [C,1] A mask carries no multicast
            # peers, so re-applying it is a no-op. Skip it so the freed SGPR is not
            # referenced across the persistent-loop refresh.
            if self.papDropsSelfOnlyMaskA(kernel) and mask == "MulticastMaskA":
                return mod
            tdm = TensorDataMoverLoad.find(writer)
            mod.add(tdm.setMulticastMask(group1, mask, writer))
        return mod
