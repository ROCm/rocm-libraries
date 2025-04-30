################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
from .KernelWriterBase import KernelWriterBase
from .TensileInstructions import DataType

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

    # Currently dummy
    def kernelBody(self):
        kStr = ""
        return kStr


    def getKernelName(self):
        name = "D"
        name += self.indicesStr
        name += "_%s%s"%(self.state["ProblemType"]["BiasDataType"].toChar(), self.state["ProblemType"]["ComputeDataType"].toChar())
        name += "_Reduction"
        return name


    def getHeaderFileString(self):
        fileString = "" # CHeader

        # C dimensions
        indicesStr = ""
        for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
            c = INDEX_CHARS[i].lower()
            if c == 'k':
                continue
            indicesStr += INDEX_CHARS[i].lower()

        computeStr  = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
        computeChar = self.state["ProblemType"]["ComputeDataType"].toChar()
        MTVW = [[256, 1, 1], [128, 2, 1], [64, 4, 1], [32, 8, 1], [16, 16, 1], [8, 32, 1], [32, 32, 4]]
        dstStr  = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
        dstChar = self.state["ProblemType"]["BiasDataType"].toChar()
        for mtvw in MTVW:
            fileString += "extern \"C\" __global__ void D%s_%s%s_MT%dx%d_VW%d_Reduction(%s *in, %s* out, int m, int n, int strideJ)\n"%(self.indicesStr, dstChar, computeChar, mtvw[0], mtvw[1], mtvw[2], computeStr, dstStr)
            fileString += "{\n"
            fileString += "    reductionKernel_%s<%s, %s, %d, %d, %d>(in, out, m, n, strideJ);\n"%(self.indicesStr, computeStr, dstStr, mtvw[0], mtvw[1], mtvw[2])
            fileString += "}\n"

        fileString += "\n"

        return fileString


    def getSourceFileString(self):
        fileString = ""
        fileString += self.kernelBody()

        return (0, fileString)
