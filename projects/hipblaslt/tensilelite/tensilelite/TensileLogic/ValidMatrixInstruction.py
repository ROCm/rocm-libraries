################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
ValidMatrixInstruction
---
Format: (M x N x K x B)
    XDLOPS tile definition, only valid for gfx908, gfx90a
    MxNxKxB specifies matrix instruction variants
    MxNxB determines the shape of the C tile each instruction worked on
        K determines the unroll depth

Alternative format: (M x N x K x B x MIBlockM x WaveTileM x WaveTileN x WaveM x WaveN)
    (Note: MxN means M-by-N in the following comments)
    MIBlockM determines how many blocks along M dimension for multi-block MI variants. Concrete examples:
    - MI 16x16x1x4 (4-block variant) with MIBlockM=4 -> (16x16)*(4x1)=64x16 tile per instruction executed
    - MI 32x32x1x2 (2-block variant) with MIBlockM=1 -> (32x32)*(1x2)=32x64 tile per instruction executed
    WaveTileM/N are dimensions of the C tile each wave works on, and is close to the concept of ThreadTile in classic VALU kernels
    - WT 4x1 -> each wave executes 4x1 matrix instructions on the C tile of total area (4*MITileM)x(1*MITileN)
    WaveM/N are dimensions of waves spawned for one workgroup where each wave consists of 64 threads
    - Wave2x2 -> a total of 4 waves in one workgroup of shape 2x2
    Putting it all together:
    - [32, 32, 1, 2, 1,  4, 1,  2, 2]
       ^^^^^^^^^^^^  ^   ^^^^   ^^^^
        MatrixInst  BlkM  WT    Wave
    - means (32x64) per MI * (4x1) per wave * (2x2) per workgroup = (32*4*2)x(64*1*2) = 256x128 macro tile
    Tensile will ignore the parameters ThreadTile and WorkGroup when the alternative format is used

Notes:
    - If empty, do not use these instructions
"""

from typing import Dict
from pathlib import Path

from Tensile.Common import IsaVersion, IsaInfo, elineno
from Tensile.SolutionStructs.Validators.MatrixInstruction import validateMIParameters


def _validateMatrixInstruction(
    solution: dict, isaInfoMap: Dict[IsaVersion, IsaInfo], filepath: Path
) -> bool:
    try:
        validateMIParameters(solution, isaInfoMap)
        assert solution["Valid"], f"Solution was rejected: {elineno()}"
        return True
    except AssertionError as e:
        print(
            f"Error: Validation failed: {e} (file: {filepath}, index: {solution['SolutionIndex']})"
        )
        return False
