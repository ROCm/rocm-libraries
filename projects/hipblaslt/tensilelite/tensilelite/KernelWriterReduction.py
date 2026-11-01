################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy

from .Common import INDEX_CHARS
from .Common.DataType import DataType
from .KernelWriterBase import KernelWriterBase

class KernelWriterReduction(KernelWriterBase):

    def __init__(self, state):
        super().__init__()

        self.state["ProblemType"] = deepcopy(state["ProblemType"])

        # C dimensions
        self.indicesStr = ""
        for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
            c = INDEX_CHARS[i].lower()
            self.indicesStr += INDEX_CHARS[i].lower()

        # derive parameter
        self.language = "HIP"
        self.kernelName = self.getKernelName()
        self.datatype = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
        if self.state["ProblemType"]["DataType"].isInt8() and self.state["ProblemType"]["ComputeDataType"].isSingle() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
            self.datatype = DataType('int32').toDevice(self.language)

    @staticmethod
    def kernelName(solution, btype):
        state = solution._state if hasattr(solution, "_state") else solution.state
        # C dimensions
        indexChars = INDEX_CHARS
        indicesStr = ""
        for i in range(0, state["ProblemType"]["NumIndicesC"]):
            c = indexChars[i].lower()
            indicesStr += indexChars[i].lower()
        name = "D"
        name += indicesStr
        name += "_%s%s"%(btype.toChar(), state["ProblemType"]["ComputeDataType"].toChar())
        name += "_Reduction"
        return name


    def getKernelName(self):
        return KernelWriterReduction.kernelName(self, self.state["ProblemType"]["BiasDataType"])


    def _kernels(self):
        """(signature, body) of each reduction kernel, one per macro tile and vector width."""
        computeStr  = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
        computeChar = self.state["ProblemType"]["ComputeDataType"].toChar()
        MTVW = [[256, 1, 1], [128, 2, 1], [64, 4, 1], [32, 8, 1], [16, 16, 1], [8, 32, 1], [32, 32, 4]]
        dstStr  = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
        dstChar = self.state["ProblemType"]["BiasDataType"].toChar()
        kernels = []
        for mtvw in MTVW:
            signature = "extern \"C\" __global__ void D%s_%s%s_MT%dx%d_VW%d_Reduction(%s *in, %s* out, int m, int n, int strideJ)"%(self.indicesStr, dstChar, computeChar, mtvw[0], mtvw[1], mtvw[2], computeStr, dstStr)
            body = "    reductionKernel_%s<%s, %s, %d, %d, %d>(in, out, m, n, strideJ);\n"%(self.indicesStr, computeStr, dstStr, mtvw[0], mtvw[1], mtvw[2])
            kernels.append((signature, body))
        return kernels


    def getHeaderFileString(self):
        fileString = "" # CHeader

        for signature, _ in self._kernels():
            fileString += signature + ";\n"

        fileString += "\n"

        return fileString


    # The kernels are defined here (Kernels.cpp) and only declared in Kernels.h, like the other
    # helper kernels: a definition in the header would be compiled into every file including it.
    # Each definition starts with a line break, as in the other helper sources: writeHelpers
    # concatenates the sources, and the activation helpers' source is a comment with no newline.
    def getSourceFileString(self):
        fileString = ""

        for signature, body in self._kernels():
            fileString += "\n"
            fileString += signature + "\n"
            fileString += "{\n"
            fileString += body
            fileString += "}\n"

        fileString += "\n"

        return (0, fileString)
