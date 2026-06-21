################################################################################
#
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################

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
"""

import os

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


def buildSubtileStinkyTofuOptions(kernel, stinky_opt_level, writer):
    """Build the StinkyTofu options dict for a subtile kernel body.

    ClusterBarrier is forced off so subtile keeps owning barrier emission.
    EnableWaitCntInsertion defaults off (pass-through); it flips on only when
    the SUBTILE_STINKYTOFU_WAITCNT toggle opts the subtile path into
    StinkyTofu-owned wait counts.
    """
    enableWaitCnt = subtileStinkyTofuWaitCntEnabled()
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
