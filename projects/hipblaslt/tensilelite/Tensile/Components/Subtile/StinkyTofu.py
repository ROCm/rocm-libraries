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

The subtile body already emits its own waits and barriers, so this option set
keeps StinkyTofu as a neutral pass-through: wait-count insertion and the
cluster-barrier handshake are disabled. The remaining keys mirror the classic
option set but use subtile-correct values (subtile forces VectorWidthA/B=1).
"""

from ...Common.GlobalParameters import globalParameters


def buildSubtileStinkyTofuOptions(kernel, stinky_opt_level, writer):
    """Build the StinkyTofu options dict for a subtile kernel body.

    EnableWaitCntInsertion and ClusterBarrier are forced off so the conversion
    does not insert or strip waits/barriers; subtile owns those for now.
    """
    return {"OptLevel": stinky_opt_level,
            "EnableRemarks": bool(globalParameters.get("StinkyTofuEnableRemarks") or False),
            "DebugLevel": int(globalParameters.get("StinkyTofuDebugLevel") or 0),
            "PrintBeforePass": str(globalParameters.get("StinkyTofuPrintBeforePass") or ""),
            "PrintAfterPass": str(globalParameters.get("StinkyTofuPrintAfterPass") or ""),
            "DebugPass": str(globalParameters.get("StinkyTofuDebugPass") or ""),
            "PassOrderSnapshotJson": str(globalParameters.get("StinkyTofuPassOrderSnapshotJson") or ""),
            # Neutral pass-through: subtile emits its own split waits.
            "EnableWaitCntInsertion": False,
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
