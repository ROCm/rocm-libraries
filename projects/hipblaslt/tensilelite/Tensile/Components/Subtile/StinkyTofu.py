# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""StinkyTofu module options for the subtile kernel path.

Every supported gfx1250 subtile kernel is emitted through StinkyTofu. The opt
level mirrors the classic protocol: ScheduleIterAlg=4 sets ``_StinkyTofuOptLevel``
(=3), which selects that opt level and turns wait-count insertion on; otherwise
the body runs at a basic level (OptLevel=0) with wait-count insertion off. The
basic level still runs the required kernel-scope passes (notably VGPR MSB
handling), so default subtile kernels keep that coverage.

Barriers stay Python-owned (``ClusterBarrier`` is forced off); subtile owns
barrier emission via ``ClusterBarrier.py``.

Wait-count insertion is honored only for kernels whose LDS producers are all TDM
``tensor_load_to_lds``. Kernels that feed LDS with ``buffer_load...lds`` (DTL)
producers -- non-TDM A/B reads or MX-scale reads -- are not safe: StinkyTofu
classifies such a load as a plain MUBUF load (not an LDS writer), so it would
strip a Python producer wait it cannot re-derive. For those kernels the guard
forces ``EnableWaitCntInsertion=False`` and stays at the basic level.
"""

from ...Common import print2, printWarning
from ...Common.GlobalParameters import globalParameters


# Basic opt level for subtile kernels that do not select ScheduleIterAlg=4.
# O0 still runs the required passes (including InsertVgprMsbPass).
SUBTILE_STINKYTOFU_BASIC_OPTLEVEL = 0


def subtileKernelIsWaitInsertionSafe(kernel):
    """True when StinkyTofu may own this subtile kernel's wait counts.

    A kernel is safe only when every LDS producer is a TDM
    ``tensor_load_to_lds``. Two emission paths instead write LDS via
    ``buffer_load...lds`` (DTL):

      * Non-TDM A/B global reads (``emitSingleBufferLoad``) -- taken whenever
        ``enableTDMA``/``enableTDMB`` is not set for that tensor.
      * MX-scale global reads (``globalReadDoScaleSubtile``) -- always DTL and
        emitted whenever ``MXBlockA`` or ``MXBlockB`` is set (scales have no
        TDM path).

    StinkyTofu (gfx1250) treats a DTL load as a plain MUBUF load, so it cannot
    re-derive a stripped producer wait for it. Safe iff both A and B use TDM
    AND there are no MX-scale DTL producers.
    """
    tdmA = bool(kernel.get("enableTDMA", False))
    tdmB = bool(kernel.get("enableTDMB", False))
    problemType = kernel.get("ProblemType", {}) or {}
    hasMXScale = bool(problemType.get("MXBlockA", 0)) or bool(problemType.get("MXBlockB", 0))
    return tdmA and tdmB and not hasMXScale


def buildSubtileStinkyTofuOptions(kernel, writer):
    """Build the StinkyTofu options dict for a subtile kernel body.

    ScheduleIterAlg=4 sets ``_StinkyTofuOptLevel`` (=3): that opt level is used
    and ``EnableWaitCntInsertion`` is on. Otherwise the body runs at the basic
    level (OptLevel=0) with wait-count insertion off.

    ClusterBarrier is forced off so subtile keeps owning barrier emission. The
    wait-insertion guard forces ``EnableWaitCntInsertion=False`` (and the basic
    level) for kernels with ``buffer_load...lds`` (DTL) producers, which keep
    their Python-emitted waits.
    """
    siaOptLevel = kernel.get("_StinkyTofuOptLevel", SUBTILE_STINKYTOFU_BASIC_OPTLEVEL)
    waitCntSelected = siaOptLevel not in (None, 0)
    optLevel = int(siaOptLevel) if waitCntSelected else SUBTILE_STINKYTOFU_BASIC_OPTLEVEL
    enableWaitCnt = waitCntSelected
    if enableWaitCnt and not subtileKernelIsWaitInsertionSafe(kernel):
        # DTL (buffer_load...lds) producers are classified as plain MUBUF loads
        # by StinkyTofu on gfx1250, so it would strip a producer wait it cannot
        # reconstruct. Keep the Python waits and stay at the basic level.
        enableWaitCnt = False
        optLevel = SUBTILE_STINKYTOFU_BASIC_OPTLEVEL
        kernelName = getattr(getattr(writer, "states", None), "kernelName", "")
        printWarning("StinkyTofu wait-count insertion disabled for subtile kernel "
                     "%s: it has buffer_load-to-LDS (DTL) producers; keeping "
                     "Python-emitted waits." % (kernelName or "<unnamed>"))
        print2("[subtile StinkyTofu] %s runs at the basic level (no wait-count "
               "insertion) for DTL-producer kernels." % (kernelName or "<unnamed>"))

    return {"OptLevel": optLevel,
            "EnableRemarks": bool(globalParameters.get("StinkyTofuEnableRemarks") or False),
            "DebugLevel": int(globalParameters.get("StinkyTofuDebugLevel") or 0),
            "PrintBeforePass": str(globalParameters.get("StinkyTofuPrintBeforePass") or ""),
            "PrintAfterPass": str(globalParameters.get("StinkyTofuPrintAfterPass") or ""),
            "DebugPass": str(globalParameters.get("StinkyTofuDebugPass") or ""),
            "PassOrderSnapshotJson": str(globalParameters.get("StinkyTofuPassOrderSnapshotJson") or ""),
            # On only for ScheduleIterAlg=4 wait-insertion-safe kernels; basic
            # kernels keep their own split waits.
            "EnableWaitCntInsertion": enableWaitCnt,
            # gfx1250: skip an unnecessary s_wait_tensorcnt(0) across unroll copies
            # in double-buffered TDM loops; the wait pass won't re-propagate
            # tensorcnt status for tensor_load_to_lds. Only affects the
            # wait-insertion path, which the DTL guard restricts to pure-TDM
            # kernels matching this rule's precondition.
            "EnableLoopCarriedTokenDeps": True,
            # True: expert scheduling mode2; False: mode 0. Independent of OptLevel.
            "EnableESM2": kernel["EnableStinkyTofuESM2"],
            "TileA0": kernel["ThreadTile0"],
            "TileB0": kernel["ThreadTile1"],
            "TileM0": kernel["MacroTile0"],
            "wavefrontSize": kernel["WavefrontSize"],
            "SubGroup0": kernel["SubGroup0"],
            "SubGroup1": kernel["SubGroup1"],
            "WaveGroup0": kernel["MIWaveGroup"][0],
            "WaveGroup1": kernel["MIWaveGroup"][1],
            # Subtile forces unit vector widths.
            "VectorWidthA": 1,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": kernel["GlobalReadVectorWidthA"],
            "GlobalReadVectorWidthB": kernel["GlobalReadVectorWidthB"],
            "DirectToLdsA": bool(kernel["DirectToLdsA"]),
            "DirectToLdsB": bool(kernel["DirectToLdsB"]),
            "UseSgprForGRO": kernel["_UseSgprForGRO"],
            # -1 disables SwInstructionPrefetch in Gfx1250Backend; else scratch pool index
            "SwPrefetchScratchSgpr": int(writer.sgprs.get("SwPrefetchScratch", -1)),
            # Neutral pass-through: subtile owns barrier emission (ClusterBarrier.py).
            "ClusterBarrier": False,
            "PrefetchGlobalRead": int(kernel.get("PrefetchGlobalRead", 1)),
            "PrefetchLocalRead": int(kernel.get("PrefetchLocalRead", 1)),
           }
