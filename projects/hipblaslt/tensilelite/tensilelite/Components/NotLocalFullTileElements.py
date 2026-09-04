################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import NotLocalFullTileElements
from math import ceil

class NotLocalFullTileElementsVALU(NotLocalFullTileElements):
    kernel = {"EnableMatrixInstruction": False}

    """
    Partition thread-tile into writeElements for store code
    This function creates the writeElement mapping for full tiles
    (ie non-edge cases)
    """
    def __call__(self, writer, kernel):
        elements     = []
        vectorwidths = []
        vectorwidth  = kernel["VectorWidthA"] if kernel["_VectorStore"] else 1
        vectorwidth  = min(vectorwidth, writer.maxGwvw(kernel))

        while vectorwidth > 0:
            elements_temp = []
            # Full tile loop:
            for tt1 in range(0, kernel["ThreadTile1"]//kernel["VectorWidthA"]):
                for vc1 in range(0, kernel["VectorWidthA"]):
                    for tt0 in range(0, kernel["ThreadTile0"]//kernel["VectorWidthA"]):
                        for vc0 in range(0, kernel["VectorWidthA"], vectorwidth): # note step by fullVw
                            element = (tt1, tt0, vc1, vc0)
                            elements_temp.append(element)
            vectorwidths.append(vectorwidth)
            elements.append(elements_temp)
            vectorwidth = int(vectorwidth / 2) # decrease vectorwidth by half

        return (vectorwidths, elements, vectorwidths, elements)

class NotLocalFullTileElementsMFMA(NotLocalFullTileElements):
    kernel = {"EnableMatrixInstruction": True}

    def getElements(self, writer, kernel):
        # When singleBuffer/atomic is enabled. We will have 2 different type of store.
        # One is GSU=1 normal store and another is GSU>1 atomic store.
        # storeVectorWidth indicates the atomic store vectorWidth.
        # storeVectorWidth_1 indicates the normal store vectorWidth.
        # For non-atomic cases, these two are the same.
        elements          = []
        storeVectorWidths = []
        storeVectorWidth  = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
        storeVectorWidth  = min(storeVectorWidth, writer.maxGwvw(kernel))

        # handle mfma 4x4 instruction
        matrixInstM  = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN  = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
        matrixInstBM = 1                                              if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
        matrixInstBN = 1                                              if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

        outputsPerThread = matrixInstM * matrixInstN // kernel["WavefrontSize"]

        # handle SourceSwap
        totalTT0     = matrixInstBM * kernel["MIWaveTile"][0]
        totalTT1     = matrixInstBN * kernel["MIWaveTile"][1]

        totalTT0     = totalTT0                      if kernel["SourceSwap"] else (totalTT0 * outputsPerThread)
        totalTT1     = (totalTT1 * outputsPerThread) if kernel["SourceSwap"] else totalTT1
        vectorWidth0 = kernel["VectorWidthA"]        if kernel["SourceSwap"] else kernel["VectorWidthA"] * kernel["MIOutputVectorWidth"]
        vectorWidth1 = kernel["VectorWidthB"] * kernel["MIOutputVectorWidth"] if kernel["SourceSwap"] else kernel["VectorWidthB"]

        while storeVectorWidth > 0:
            elements_temp = []
            for tt1 in range(0, ceil(totalTT1//vectorWidth1)):
                for vc1 in range(0, vectorWidth1):
                    for tt0 in range(0, ceil(totalTT0//vectorWidth0)):
                        for vc0 in range(0, vectorWidth0, storeVectorWidth): # note step by storeVectorWidth
                            element = (tt1, tt0, vc1, vc0)
                            elements_temp.append(element)
            storeVectorWidths.append(storeVectorWidth)
            elements.append(elements_temp)
            storeVectorWidth = int(storeVectorWidth / 2) # decrease storeVectorWidth by half

        return (storeVectorWidths, elements)

    """
    Partition thread-tile into writeElements for store code
    This function creates the writeElement mapping for full tiles
    (ie non-edge cases)
    """
    def __call__(self, writer, kernel):
        if kernel["GlobalSplitU"] == 0:
            (storeVectorWidths, elements) = self.getElements(writer, kernel)
            return (storeVectorWidths, elements, storeVectorWidths, elements)
        else:
            gsuBackup = kernel["GlobalSplitU"]
            kernel["GlobalSplitU"] = 2
            (storeVectorWidths, elements) = self.getElements(writer, kernel)
            kernel["GlobalSplitU"] = 1
            (storeVectorWidths_1, elements_1) = self.getElements(writer, kernel)
            kernel["GlobalSplitU"] = gsuBackup
            return (storeVectorWidths, elements, storeVectorWidths_1, elements_1)
