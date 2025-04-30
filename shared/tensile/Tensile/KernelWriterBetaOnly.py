################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import globalParameters, CHeader
from .KernelWriterBase import KernelWriterBase

class KernelWriterBetaOnly(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[self.state["ProblemType"]["Index0"]] = "0" + self.indexChars[self.state["ProblemType"]["Index0"]]
    self.indexChars[self.state["ProblemType"]["Index1"]] = "1" + self.indexChars[self.state["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[self.state["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[self.state["ProblemType"]["Index1"]]


  def functionSignature(self):
    kStr = ""

    # self.state name
    kStr += self.endLine
    kStr += "extern \"C\"" + self.endLine
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # pointers
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    if self.state["_GlobalAccumulation"]:
      ptrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)

    isStridedBuffer = self.state["ProblemType"]["StridedBatched"] or self.state["_GlobalAccumulation"]
    ptrStr += "" if isStridedBuffer else "*"
    batch   = "" if isStridedBuffer else "Batch"
    kStr += "  " + ptrStr + " * " + batch + "D," + self.endLine

    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    isStridedBuffer = self.state["ProblemType"]["StridedBatched"]
    ptrStr += "" if isStridedBuffer else "*"
    batch   = "" if isStridedBuffer else "Batch"
    kStr += "  " + ptrStr + " const * " + batch + "C," + self.endLine

    # offset
    if not isStridedBuffer:
      kStr += "  uint64_t offsetD,%s" % self.endLine
      kStr += "  uint64_t offsetC,%s" % self.endLine

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideC%s,%s" % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s,%s" % (self.indexChars[i], self.endLine)

    # beta
    kStr += "  %s const beta)%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine )

    return kStr


  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBodyBetaOnly(self):
    problemType = self.state["ProblemType"]
    globalAccum = self.state["_GlobalAccumulation"]

    kStr = ""
    kStr += "{%s" % self.endLine

    ########################################
    # defined initial strides
    firstStride = 0
    if problemType["UseInitialStridesCD"]:
      # no strides #defined
      lastStrideC = 0
      assert 0  # need to fix beta-clear routine to pass initial stride parms
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" % self.endLine
      lastStrideC = 1
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_C()
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)
    kStr += "  if (id >= (size%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += "*size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      kStr += "  id%d = id %% size%s;%s" % (i, self.indexChars[i], self.endLine)
      kStr += "  id  = id / size%s;%s" % (self.indexChars[i], self.endLine)

    nonTileFreeIndices = []

    # apply batch
    if not self.state["ProblemType"]["StridedBatched"]:
      nonTileFreeIndices = list(range(0, self.state["ProblemType"]["NumIndicesC"]))
      nonTileFreeIndices.remove(self.state["ProblemType"]["Index0"])
      nonTileFreeIndices.remove(self.state["ProblemType"]["Index1"])

      kStr += self.endLine
      kStr += "  uint64_t wg = 0"
      batchStride = "1"
      for i in nonTileFreeIndices:
        kStr += " + id%d * %s " % (i, batchStride)
        batchStride += " * size%s" % self.indexChars[i]
      kStr += ";" + self.endLine

      if not self.state["_GlobalAccumulation"]:
        ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
        kStr += "  " + ptrStr + " * D = BatchD[wg];" + self.endLine
      ptrStr  = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      zeroStr = self.state["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
      kStr += "  " + ptrStr + f" const* C = (beta == {zeroStr}) ? nullptr : BatchC[wg];" + self.endLine

      # apply offset only for general batch
      kStr += self.endLine
      if not self.state["_GlobalAccumulation"]:
        kStr += "  D = D + offsetD;" + self.endLine
      kStr += "  C = C + offsetC;" + self.endLine


    kStr += self.endLine
    ########################################
    # D index
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      tmpStr = ''
      if self.state["_GlobalAccumulation"]:
        tmpStr = 'id%d' % i
      elif i in nonTileFreeIndices:
        tmpStr = '0'
      else:
        tmpStr = 'id%d' % i
      kStr += ', ' if i else ''
      kStr += tmpStr
    kStr += ");%s" % (self.endLine)

    # C index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
    kStr += ");%s" % (self.endLine)

    ########################################
    # zero
    if globalAccum:
      ptrStr = problemType["ComputeDataType"].toDevice(self.language)
    else:
      ptrStr = problemType["DestDataType"].toDevice(self.language)
    kStr += "#define SCALAR_ZERO ((%s)(0))%s" % (ptrStr, self.endLine )

    ########################################
    # zero
    computeType = problemType["ComputeDataType"].toDevice(self.language)
    kStr += "  if( beta == (%s)0) {%s" % (computeType, self.endLine)
    kStr += "    D[idxD] = SCALAR_ZERO;%s" % self.endLine
    kStr += "  } else {%s" % self.endLine
    kStr += "    D[idxD] = ((%s)(C[idxC])) * beta;%s" % (computeType, self.endLine)
    kStr += "  }%s" % self.endLine

    ########################################
    # end
    kStr += "}%s" % self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideC" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)
    kStr += "#undef SCALAR_ZERO%s" % ( self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    name += "" if self.state["ProblemType"]["StridedBatched"] else "_GB" # legacy
    name += "_GA" if self.state["_GlobalAccumulation"] else ""

    return name


  def getSourceFileString(self):
    fileString = ""

    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % self.kernelName
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += self.kernelBodyBetaOnly()

    return (0, fileString)

  def getHeaderFileString(self):
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += CHeader
      fileString += "#pragma once\n\n"
      fileString += "\n"
      fileString += "#include <KernelHeader.h>\n\n"
      fileString += "#include <hip/hip_runtime.h>\n"
      fileString += "#include <hip/hip_fp16.h>\n"
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += ";\n"

    return fileString
