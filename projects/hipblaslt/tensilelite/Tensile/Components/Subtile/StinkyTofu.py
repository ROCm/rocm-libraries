# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""StinkyTofu module options for the subtile kernel path.

The subtile body emits its own waits and barriers. By default this option set
keeps StinkyTofu as a neutral pass-through: wait-count insertion and the
cluster-barrier handshake are disabled. The remaining keys mirror the classic
option set but use subtile-correct values (subtile forces VectorWidthA/B=1).

The ``SUBTILE_STINKYTOFU_WAITCNT`` environment variable opts the subtile path
into StinkyTofu-owned wait counts. When enabled, StinkyTofu strips the
subtile-emitted split waits (``StinkyRemoveWaitCntPass``) and re-inserts its own
(``StinkyWaitCntInsertionPass``). The subtile wait emission itself is left in
place untouched; this flag only chooses whether StinkyTofu replaces them.
Barriers stay Python-owned (``ClusterBarrier`` remains off) in either mode.

The wait toggle is honored only for kernels whose LDS producers are all TDM
``tensor_load_to_lds``. Kernels that feed LDS with ``buffer_load...lds`` (DTL)
producers -- non-TDM A/B reads or MX-scale reads -- are not safe: StinkyTofu
classifies such a load as a plain MUBUF load (not an LDS writer), so it would
strip a Python producer wait it cannot re-derive. For those kernels the guard
forces ``EnableWaitCntInsertion=False`` regardless of the toggle.
"""

import os

from ...Common import print2, printWarning
from ...Common.GlobalParameters import globalParameters


# Environment toggle (default off) handing the subtile wait COUNTS to StinkyTofu.
# Kept local to the subtile path so no shared-code plumbing is required.
SUBTILE_WAITCNT_ENV = "SUBTILE_STINKYTOFU_WAITCNT"


def subtileStinkyTofuWaitCntEnabled():
    """True when the subtile StinkyTofu wait-count toggle is opted in.

    Off unless ``SUBTILE_STINKYTOFU_WAITCNT`` is set to 1/true/yes/on.
    """
    val = os.environ.get(SUBTILE_WAITCNT_ENV)
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


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


def buildSubtileStinkyTofuOptions(kernel, stinky_opt_level, writer):
    """Build the StinkyTofu options dict for a subtile kernel body.

    ClusterBarrier is forced off so subtile keeps owning barrier emission.
    EnableWaitCntInsertion defaults off (pass-through); it flips on only when
    the SUBTILE_STINKYTOFU_WAITCNT toggle opts the subtile path into
    StinkyTofu-owned wait counts AND the kernel is wait-insertion-safe (all LDS
    producers are TDM tensor_load_to_lds). Kernels with buffer_load...lds (DTL)
    producers keep their Python-emitted waits.
    """
    enableWaitCnt = subtileStinkyTofuWaitCntEnabled()
    if enableWaitCnt and not subtileKernelIsWaitInsertionSafe(kernel):
        # DTL (buffer_load...lds) producers are classified as plain MUBUF loads
        # by StinkyTofu on gfx1250, so it would strip a producer wait it cannot
        # reconstruct. Keep the Python waits for this kernel.
        enableWaitCnt = False
        kernelName = getattr(getattr(writer, "states", None), "kernelName", "")
        printWarning("StinkyTofu wait-count insertion disabled for subtile kernel "
                     "%s: it has buffer_load-to-LDS (DTL) producers; keeping "
                     "Python-emitted waits." % (kernelName or "<unnamed>"))
        print2("[subtile StinkyTofu] %s honors SUBTILE_STINKYTOFU_WAITCNT only "
               "for pure-TDM kernels." % SUBTILE_WAITCNT_ENV)
    return {"OptLevel": stinky_opt_level,
            "EnableRemarks": bool(globalParameters.get("StinkyTofuEnableRemarks") or False),
            "DebugLevel": int(globalParameters.get("StinkyTofuDebugLevel") or 0),
            "PrintBeforePass": str(globalParameters.get("StinkyTofuPrintBeforePass") or ""),
            "PrintAfterPass": str(globalParameters.get("StinkyTofuPrintAfterPass") or ""),
            "DebugPass": str(globalParameters.get("StinkyTofuDebugPass") or ""),
            "PassOrderSnapshotJson": str(globalParameters.get("StinkyTofuPassOrderSnapshotJson") or ""),
            # Off (default): subtile keeps its own split waits. When the
            # SUBTILE_STINKYTOFU_WAITCNT toggle is on, StinkyTofu strips them
            # (StinkyRemoveWaitCntPass) and re-inserts its own.
            "EnableWaitCntInsertion": enableWaitCnt,
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
