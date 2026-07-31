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
from ..Common import clusterEnabled, streamK2DMulticast, streamKMulticast
from typing import Mapping
from rocisa.code import Module, Label
from rocisa.container import sgpr
from rocisa.instruction import SLShiftLeftB32, SMulI32, SBitcmp1B32, SCBranchSCC1, SBranch, \
    SMovB32, SAndB32


class ClusterLoadTDM(ClusterLoad):
    asmCaps = {"HasTDM": True}
    kernel  = {"TDMInst": 3}

    def __call__(self, writer: "KernelWriterAssembly", kernel: Mapping):
        # Abstract-satisfying no-op, mirrors TensorDataMoverLoad.__call__.
        pass

    # -- topology decision ---------------------------------------------------

    def usesCombinedMask(self, kernel: Mapping) -> bool:
        """True when the single-parity combined ``MulticastMask`` applies.

        Subtile and StreamK cluster multicast both need the split A/B masks
        (subtile issues A and B on every wave; the StreamK cluster broadcasts A
        and B along different cluster axes), so the combined parity mask applies
        only to the wave-separated dense case.
        """
        if streamKMulticast(kernel):
            return False
        tdmA: bool = kernel["enableTDMA"]
        tdmB: bool = kernel["enableTDMB"]
        return tdmA and tdmB and kernel["NumWaves"] > 1 and not kernel.get("UseSubtileImpl")

    def maskSgprName(self, kernel: Mapping, tc: str, *, subtile: bool = False,
                     waveSeparated: bool = False) -> str:
        """Resolve the multicast-mask SGPR name.

        Wave-separated (non-subtile, non-StreamK-multicast) uses the combined
        ``"MulticastMask"``; dense/subtile and StreamK multicast use the split
        ``f"MulticastMask{tc}"`` (any ``MXS`` prefix stripped) so B never resolves
        to the never-declared combined SGPR.
        """
        if waveSeparated and not subtile and not streamKMulticast(kernel):
            return "MulticastMask"
        return f"MulticastMask{tc.removeprefix('MXS')}"

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

        With ``Ck == 1`` A has no peers, so its mask collapses to the self bit
        (``maskA == 1`` -> ``1 << wg_x``) and re-applying it is a no-op. Free the
        SGPR to stay within the 106-SGPR budget (at 107 SGPRs the kernel is
        replaced by an ``s_endpgm`` stub and the output tensor is left unwritten).
        With ``Ck > 1`` A is a real multicast and must stay live.
        """
        return self.papRefreshesMask(kernel) and not streamK2DMulticast(kernel)

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

        # sTmp+0 / sTmp+4 double as scratch for the reduced-bit masks of a padded
        # boundary cluster: maskCol = (1<<(validY*cx))-1 (AND with maskA), maskRow =
        # (1<<validX)-1 (AND with maskB). computeMulticastMaskReduction writes them.
        maskColSgpr = sTmp + 0
        maskRowSgpr = sTmp + 4

        maskA = 1
        for idx in range(kernel["ClusterDim"][1]):
            maskA |= (1 << (idx * kernel["ClusterDim"][0]))

        maskB = (1 << kernel["ClusterDim"][0]) - 1

        # Reduce the broadcast mask to the WGs actually present in a padded boundary
        # cluster (grid rounded up to ClusterDim): writes maskColSgpr/maskRowSgpr and
        # returns True, else returns False and we fall back to the full mask.
        reduced = writer.computeMulticastMaskReduction(
            kernel, mod, sgprWgX, sgprWgY, maskColSgpr, maskRowSgpr)
        if not reduced:
            maskColSgpr = None
            maskRowSgpr = None

        def setMask(dst, maskConst, shiftReg, reducedBits, comment):
            # dst = (maskConst [& reducedBits]) << shiftReg. When reducedBits is present
            # the constant is first ANDed to the WGs that really exist in this cluster;
            # otherwise it degenerates to the original single shift of the immediate.
            if reducedBits is not None:
                mod.add(SMovB32(dst=sgpr(dst), src=hex(maskConst), comment=comment))
                mod.add(SAndB32(dst=sgpr(dst), src0=sgpr(dst), src1=sgpr(reducedBits),
                                comment="reduce to real WGs"))
                mod.add(SLShiftLeftB32(dst=sgpr(dst), shiftHex=sgpr(shiftReg), src=sgpr(dst),
                                       comment=comment))
            else:
                mod.add(SLShiftLeftB32(dst=sgpr(dst), shiftHex=sgpr(shiftReg), src=hex(maskConst),
                                       comment=comment))

        if kernel["enableTDMMetadata"]:
            if kernel["ProblemType"]["Sparse"] == 1:
                setMask("MulticastMaskMetadata", maskA, sgprWgX, maskColSgpr,
                        "Setting metadata mask (follows sparse A)")
            elif kernel["ProblemType"]["Sparse"] == 2:
                with writer.allocTmpSgpr(1, tag="multicastMetadataShift") as shiftTmp:
                    mod.add(SMulI32(dst=sgpr(shiftTmp.idx), src0=sgpr(sgprWgY), src1=sgpr(sgprNWgX),\
                                    comment="Shift factor: wg_y * nwg_x (metadata)"))
                    setMask("MulticastMaskMetadata", maskB, shiftTmp.idx, maskRowSgpr,
                            "Setting metadata mask (follows sparse B)")

        if self.usesCombinedMask(kernel):
            setMulticastMaskLblOdd = Label(f"setMulticastMask_OddWave", "")
            setMulticastMaskLblEven = Label(f"setMulticastMask_EvenWave", "")
            setMulticastMaskLblEnd = Label(f"setMulticastMaskEnd", "")

            mod.add(SBitcmp1B32(sgpr("WaveIdx"), 0, "Check parity of wId"))
            mod.add(SCBranchSCC1(setMulticastMaskLblOdd.getLabelName(), "Jump if wId is odd"))

            mod.add(setMulticastMaskLblEven)
            setMask("MulticastMask", maskA, sgprWgX, maskColSgpr, "Setting maskA for even wave")
            mod.add(SBranch(setMulticastMaskLblEnd.getLabelName()))
            mod.add(setMulticastMaskLblOdd)
            mod.add(SMulI32(dst=sgpr(sgprWgY), src0=sgpr(sgprWgY), src1=sgpr(sgprNWgX),\
                            comment="Shift factor: wg_y * nwg_x"))
            setMask("MulticastMask", maskB, sgprWgY, maskRowSgpr, "Setting maskB for odd wave")
            mod.add(setMulticastMaskLblEnd)

        else:
            setMask("MulticastMaskA", maskA, sgprWgX, maskColSgpr, "Setting maskA")

            mod.add(SMulI32(dst=sgpr(sgprWgY), src0=sgpr(sgprWgY), src1=sgpr(sgprNWgX),\
                            comment="Shift factor: wg_y * nwg_x"))
            setMask("MulticastMaskB", maskB, sgprWgY, maskRowSgpr, "Setting maskB")
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
            # papDropsSelfOnlyMaskA): with Ck == 1 the A mask carries no multicast
            # peers, so re-applying it is a no-op. Skip it so the freed SGPR is not
            # referenced across the persistent-loop refresh.
            if self.papDropsSelfOnlyMaskA(kernel) and mask == "MulticastMaskA":
                return mod
            tdm = TensorDataMoverLoad.find(writer)
            mod.add(tdm.setMulticastMask(group1, mask, writer))
        return mod
