# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cluster (multicast) TDM load component.

Centralizes the multicast ("cluster load") mask machinery that was previously
duplicated across ``KernelWriter``/``KernelWriterAssembly``/``SubtileGREmit``:

  * the mask *value* computation (``computeMasks``),
  * the ``MulticastMask*`` SGPR declare/undeclare (``declareSgprs`` /
    ``undeclareSgprs``),
  * the topology decision (``usesCombinedMask`` / ``maskSgprName``), and
  * the descriptor attach at each load site (``applyToDescriptor``).

This is a behavior-preserving extraction: every method emits byte-identical
assembly to the original inline code. ``computeMasks`` therefore receives the
exact SGPR operands the caller already holds (it does not re-allocate) so the
instruction stream and register indices are unchanged.

Selection is capability-based (``HasTDM`` + ``TDMInst == 3``), identical to how
``TensorDataMoverLoad`` is found: ``ClusterLoad.find(writer)`` returns the TDM
impl on gfx1250 and ``None`` (fallback -> no multicast) elsewhere.
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
        """Single-parity combined ``MulticastMask`` predicate.

        Subtile issues both A and B loads on every wave (no wave-parity load
        split), so the single-parity ``MulticastMask`` is wrong there -- it
        would OR one tensor's mask into both descriptors. Use the split A/B
        masks in that case. This is the single source of truth for the
        combined-vs-split decision used by declare/undeclare/computeMasks.
        """
        # StreamK DP cooperative multicast needs the SPLIT A/B masks: A is
        # loaded per-workgroup (MulticastMaskA = self bit) while B is broadcast
        # across the [C,1] cluster (MulticastMaskB = all-C bits). The combined
        # single-parity mask instead selects maskA on even waves / maskB on odd
        # waves (for the wave-separated A-even/B-odd load split), which is wrong
        # here, so force split whenever StreamKMulticast is on. Inert otherwise.
        if kernel.get("StreamKMulticast", 0):
            return False
        tdmA: bool = kernel["enableTDMA"]
        tdmB: bool = kernel["enableTDMB"]
        return tdmA and tdmB and kernel["NumWaves"] > 1 and not kernel.get("UseSubtileImpl")

    def maskSgprName(self, kernel: Mapping, tc: str, *, subtile: bool = False,
                     waveSeparated: bool = False) -> str:
        """Central multicast-mask SGPR name resolver.

        Reproduces the three prior naming rules exactly:
          * wave-separated (non-subtile): the combined ``"MulticastMask"``;
          * dense and subtile: the split ``f"MulticastMask{tc}"`` with any
            ``MXS`` prefix stripped (dense passed ``MXSA``/``MXSB`` tensor
            chars; subtile only ever passes ``A``/``B`` so the strip is a
            no-op there).
        """
        # StreamK DP cooperative multicast always uses the split A/B masks
        # (usesCombinedMask() returns False for it). The wave-separated dense
        # apply site would otherwise resolve to the combined "MulticastMask"
        # name, which is never declared on this path -> the B descriptor would
        # OR an undefined SGPR. Force the split name so A binds MulticastMaskA
        # (self) and B binds MulticastMaskB (broadcast).
        if kernel.get("StreamKMulticast", 0):
            string = tc.removeprefix("MXS") if tc.startswith("MXS") else tc
            return f"MulticastMask{string}"
        if waveSeparated and not subtile:
            return "MulticastMask"
        string = tc.removeprefix("MXS") if tc.startswith("MXS") else tc
        return f"MulticastMask{string}"

    def cooperativeThreadPartition(self, kernel: Mapping, tc: str) -> int:
        """Number of cooperating workgroups for tensor ``tc``.

        ``ClusterDim[1]`` for A, ``ClusterDim[0]`` for B. Shared math with the
        GL2 prefetch cooperative loads.
        """
        subTc: str = tc[-1]
        return kernel["ClusterDim"][1] if subTc == "A" else kernel["ClusterDim"][0]

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

        Verbatim lift of the ``defineAndResources`` mask compute. The caller
        passes the exact SGPR operands it already holds (``sgprWgX`` = wg_x,
        ``sgprWgY`` = wg_y, ``sgprNWgX`` = nwg_x, and ``sTmp`` whose ``+4`` slot
        is scratch) so the emitted instructions and register indices are
        byte-identical to the original inline code.
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

        Folds the ``kernel["Multicast"] and enableCluster`` gate, the mask-name
        choice, and the ``SOrB32`` attach. Returns an empty ``Module`` when the
        gate is not satisfied -- identical to today's skipped ``if``.
        """
        from .TensorDataMover import TensorDataMoverLoad
        mod = Module()
        if kernel["Multicast"] and clusterEnabled(kernel["ClusterDim"]):
            mask = self.maskSgprName(kernel, tc, subtile=subtile, waveSeparated=waveSeparated)
            tdm = TensorDataMoverLoad.find(writer)
            mod.add(tdm.setMulticastMask(group1, mask, writer))
        return mod
