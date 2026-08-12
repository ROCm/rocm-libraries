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

from Tensile.Common import INDEX_CHARS
from Tensile.Common.DataType import DataType
from .KernelWriterBase import KernelWriterBase

class KernelWriterBetaOnly(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()
    self.datatype = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(INDEX_CHARS)):
      self.indexChars.append(INDEX_CHARS[i])
    self.indexChars[self.state["ProblemType"]["Index0"]] = "0" + self.indexChars[self.state["ProblemType"]["Index0"]]
    self.indexChars[self.state["ProblemType"]["Index1"]] = "1" + self.indexChars[self.state["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[self.state["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[self.state["ProblemType"]["Index1"]]

    # Macro guards for f8 types
    # For now, it is enough to check dest type to determine if we are using f8 types
    # May need to include checks for input data type in the future.
    self.f8MacroGuardStart = "";
    self.f8MacroGuardEnd   = "";
    if (self.state["ProblemType"]["DestDataType"].isFloat8() or self.state["ProblemType"]["DestDataType"].isBFloat8()):
      self.f8MacroGuardStart = "\n#if HIP_FP8_TYPE_OCP\n"
      self.f8MacroGuardEnd   = "\n#endif // F8 macro guard\n"
    if (self.state["ProblemType"]["DestDataType"].isFloat8_fnuz() or self.state["ProblemType"]["DestDataType"].isBFloat8_fnuz()):
      self.f8MacroGuardStart = "\n#if HIP_FP8_TYPE_FNUZ\n"
      self.f8MacroGuardEnd   = "\n#endif // F8 macro guard\n"


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

    # bias
    if self.state["ProblemType"]["BetaOnlyUseBias"]:
      biasPtrStr = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
      kStr += "  " + biasPtrStr + " const * " + "Bias," + self.endLine


    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideC%s,%s" % (self.indexChars[i], self.endLine)

    if self.state["ProblemType"]["BetaOnlyUseBias"]:
      kStr += "  unsigned int strideBias,%s" % (self.endLine)
      if self.state["ProblemType"]["UseBias"] == 3:
        kStr += "  unsigned int factorDim,%s" % (self.endLine)

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

    # GLOBAL_BIAS()
    if self.state["ProblemType"]["BetaOnlyUseBias"] and self.state["ProblemType"]["NumIndicesC"] > 2:
      kStr += "#define GLOBAL_BIAS(IDX%s" % self.indexChars[0]
      kStr += ", IDX%s" % self.indexChars[2]
      indexChar = self.indexChars[0]
      kStr += ") (( (IDX%s)" % (indexChar)
      indexChar = self.indexChars[2]
      kStr += " + (IDX%s)*strideBias" % (indexChar)
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
      if problemType["DataType"].isHalf() and problemType["HighPrecisionAccumulate"]:
        ptrStr = DataType('s').toDevice(self.language)
    else:
      ptrStr = problemType["DataType"].toDevice(self.language)
    kStr += "#define SCALAR_ZERO ((%s)(0))%s" % (ptrStr, self.endLine )

    biasStr = ""
    if self.state["ProblemType"]["BetaOnlyUseBias"]:
      id_str = "id0"
      if self.state["ProblemType"]["UseBias"] == 3:
        id_str = "idb"
        kStr += "  %s idb = ( factorDim == 0 ? (%s)id0 : id1);%s" % (self.uint64Str, self.uint64Str, self.endLine)
      elif self.state["ProblemType"]["UseBias"] == 2:
        id_str = "id1"
      if problemType["NumIndicesC"] > 2:
        biasStr = " + ((" + self.datatype + ")(Bias == 0 ? 0 : GLOBAL_BIAS((%s)%s, id2)))"% (self.uint64Str, id_str)
      else:
        biasStr = " + ((" + self.datatype + ")(Bias == 0 ? 0 : Bias[%s]))"%id_str

    ########################################
    # zero
    kStr += "  if( beta == (%s)0) {%s" % (self.datatype, self.endLine)
    kStr += "    D[idxD] = SCALAR_ZERO%s;%s" % (biasStr, self.endLine)
    kStr += "  } else {%s" % self.endLine
    kStr += "    D[idxD] = ((%s)(C[idxC])) * beta%s;%s" % (self.datatype, biasStr, self.endLine)
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
    if self.state["ProblemType"]["BetaOnlyUseBias"] and self.state["ProblemType"]["NumIndicesC"] > 2:
      kStr += "#undef  GLOBAL_BIAS%s" % ( self.endLine)
    kStr += "#undef SCALAR_ZERO%s" % ( self.endLine)

    return kStr


  @staticmethod
  def kernelName(solution, btype=None):
    state = solution._state if hasattr(solution, "_state") else solution.state
    indexChars = INDEX_CHARS
    # C dimensions
    name = "C"
    for i in range(0, state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += state["ProblemType"]["DestDataType"].toChar()
    if state["ProblemType"]["GroupedGemm"]:
      name += "_GG"
    else:
      name += "" if state["ProblemType"]["StridedBatched"] else "_GB"
    if state["ProblemType"]["BetaOnlyUseBias"]:
      name += "_Bias%s"%btype.toChar()
    name += "_GA" if state["_GlobalAccumulation"] else ""

    return name


  def getKernelName(self):
    btype = self.state["ProblemType"]["BiasDataType"] if self.state["ProblemType"]["BetaOnlyUseBias"] else None
    return KernelWriterBetaOnly.kernelName(self, btype)


  def getSourceFileString(self):
    fileString = ""

    for toggle in [True, False]:
      self.state["ProblemType"]["GroupedGemm"] = toggle
      self.kernelName = self.getKernelName()
      fileString += self.f8MacroGuardStart
      fileString += self.functionSignature()
      fileString += self.kernelBodyBetaOnly()
      fileString += self.f8MacroGuardEnd

    return (0, fileString)

  def getHeaderFileString(self):
    fileString = "" # CHeader

    for toggle in [True, False]:
      self.state["ProblemType"]["GroupedGemm"] = toggle
      self.kernelName = self.getKernelName()
      fileString += self.f8MacroGuardStart
      fileString += self.functionSignature()
      fileString += ";\n"
      fileString += self.f8MacroGuardEnd

    return fileString
