# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cluster (multicast) TDM load component.

Centralizes the multicast ("cluster load") mask machinery (value compute,
``MulticastMask*`` SGPR declare/undeclare, combined-vs-split topology decision,
and per-load-site descriptor attach) that was previously duplicated across
``KernelWriter``/``KernelWriterAssembly``/``SubtileGREmit``. Behavior-preserving:
every method emits byte-identical assembly, receiving the SGPR operands the
caller already holds rather than re-allocating. Capability-selected
(``HasTDM`` + ``TDMInst == 3``), like ``TensorDataMoverLoad``.
"""

from ..Component import ClusterLoad
from ..Common import clusterEnabled
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

        Single source of truth for the combined-vs-split decision. Subtile needs
        the split A/B masks (it issues A and B on every wave, no wave-parity
        split, so the combined parity mask would OR one tensor's mask into both
        descriptors).
        """
        tdmA: bool = kernel["enableTDMA"]
        tdmB: bool = kernel["enableTDMB"]
        return tdmA and tdmB and kernel["NumWaves"] > 1 and not kernel.get("UseSubtileImpl")

    def maskSgprName(self, kernel: Mapping, tc: str, *, subtile: bool = False,
                     waveSeparated: bool = False) -> str:
        """Resolve the multicast-mask SGPR name.

        Wave-separated (non-subtile) uses the combined ``"MulticastMask"``;
        dense/subtile use the split ``f"MulticastMask{tc}"`` (any ``MXS`` prefix
        stripped).
        """
        if waveSeparated and not subtile:
            return "MulticastMask"
        string = tc.removeprefix("MXS") if tc.startswith("MXS") else tc
        return f"MulticastMask{string}"

    # -- SGPR declare / undeclare -------------------------------------------

    def declareSgprs(self, writer: "KernelWriter", kernel: Mapping) -> None:
        """Allocate the ``MulticastMask*`` SGPRs (lift of KernelWriter)."""
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

    def undeclareSgprs(self, writer: "KernelWriter", kernel: Mapping) -> Module:
        """Free the ``MulticastMask*`` SGPRs (lift of KernelWriter)."""
        mod = Module()
        if not (kernel["Multicast"] and kernel["TDMInst"] != 0):
            return mod
        tdmM: bool = kernel["enableTDMMetadata"]
        if self.usesCombinedMask(kernel):
            mod.add(writer.undefineSgpr("MulticastMask"))
        else:
            mod.add(writer.undefineSgpr("MulticastMaskA"))
            mod.add(writer.undefineSgpr("MulticastMaskB"))
        if tdmM:
            mod.add(writer.undefineSgpr("MulticastMaskMetadata"))
        return mod

    # -- mask value computation ---------------------------------------------

    def computeMasks(self, writer: "KernelWriterAssembly", kernel: Mapping, *,
                     sgprWgX: int, sgprWgY: int, sgprNWgX: int, sTmp: int) -> Module:
        """Compute the multicast mask value(s) into the ``MulticastMask*`` SGPRs.

        Verbatim lift of the ``defineAndResources`` mask compute; the caller
        passes the operands it already holds (``sgprWgX``/``sgprWgY``/``sgprNWgX``
        and ``sTmp`` whose ``+4`` slot is scratch) so the output is byte-identical.
        """
        mod = Module()
        if not kernel["Multicast"]:
            return mod
        mod.addComment0("Calculate multicast mask")

        maskA = 1
        for idx in range(kernel["ClusterDim"][1]):
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
            tdm = TensorDataMoverLoad.find(writer)
            mod.add(tdm.setMulticastMask(group1, mask, writer))
        return mod
