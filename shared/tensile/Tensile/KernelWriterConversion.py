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

class KernelWriterConversion(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]
    self.state["GlobalSplitU"] = state["GlobalSplitU"] if state["_GlobalAccumulation"] == 'MultipleBuffer' else 1
    self.state["GSUUnrollUnit"] = state["GSUUnrollUnit"] # number of unroll for large GSU
    # mod part of GSU unroll. This will be fully unrolled.
    mod = self.state["GlobalSplitU"] % self.state["GSUUnrollUnit"]
    self.state["GSUmod"] = self.state["GSUUnrollUnit"] if mod == 0 else mod
    self.state["VectorWidth"] = state["VectorWidth"]
    self.state["Reduction"] = state["Reduction"]

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()
    self.datatype = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)

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

    # kernel name
    kStr += self.endLine
    kStr += "extern \"C\"\n"
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # pointers
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    ptrStr += '' if self.state["ProblemType"]["StridedBatched"] else '*'
    bStr = '' if self.state["ProblemType"]["StridedBatched"] else 'Batch'

    kStr += "  " + ptrStr + " * " + bStr + "D," + self.endLine
    kStr += "  " + self.datatype + " * W," + self.endLine
    kStr += "  " + ptrStr + " const * " + bStr + "C," + self.endLine

    # offset
    if not self.state["ProblemType"]["StridedBatched"]:
      kStr += "  uint64_t offsetD,%s" % self.endLine
      kStr += "  uint64_t offsetC,%s" % self.endLine

    # alpha & beta
    kStr += "  %s const alpha,%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "  %s const beta" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language))

    midEnd = ",%s"%self.endLine

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "%s  unsigned int const strideD%s" % (midEnd, self.indexChars[i])
    for i in range(firstStrideCD, lastStrideC):
      kStr += "%s  unsigned int const strideW%s" % (midEnd, self.indexChars[i])
    for i in range(firstStrideCD, lastStrideC):
      kStr += "%s  unsigned int const strideC%s" % (midEnd, self.indexChars[i])

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "%s  unsigned int const size%s" % (midEnd, self.indexChars[i])

    # gsu
    kStr += "%s  unsigned int const gsu" % midEnd
    # SR
    if self.state["ProblemType"]["DestDataType"].is8bitFloat() \
            and self.state["ProblemType"]["StochasticRounding"]:
      kStr += "%s  const uint32_t RNDSeeds" % midEnd

    # put final end
    kStr += ")%s" % self.endLine

    return kStr


  def kernelBody(self):
    kStr = ""
    kStr += "{%s" % self.endLine
    problemType = self.state["ProblemType"]

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
      kStr += "#define strideW" + self.indexChars[i] + " 1" + self.endLine
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

    # GLOBAL_W()
    kStr += "#define GLOBAL_W(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideW%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideW%s" % (indexChar, indexChar)
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

    # define NUM_ELEMENT_LOAD and NUM_GSU for GlobalSplitUSeparatePost
    mul_NEL = ""
    div_NEL = ""
    kStr += "#define NUM_ELEMENT_LOAD %d%s" % (self.state["VectorWidth"], self.endLine)
    mul_NEL = "*NUM_ELEMENT_LOAD"
    div_NEL = "/NUM_ELEMENT_LOAD"
    # parallel reduction
    kStr += "#define NUM_REDUCTION %d%s" % (self.state["Reduction"], self.endLine)
    div_R = "/NUM_REDUCTION"

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)
    kStr += "  if (id%s >= (size%s" % (mul_NEL+div_R, self.indexChars[0])
    for i in range(1, problemType["NumIndicesC"]):
      kStr += "*size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    # parallel reduction
    if self.state["Reduction"] > 1:
      kStr += "  int idR = (int)(id %% NUM_REDUCTION);%s" % (self.endLine)
      kStr += "  id = id / NUM_REDUCTION;%s" % (self.endLine)
    for i in range(0, problemType["NumIndicesC"]):
      kStr += "  id%d = (id %% (size%s%s))%s;%s" % (i, self.indexChars[i], div_NEL, mul_NEL,self.endLine)
      kStr += "  id  = id / (size%s%s);%s" % (self.indexChars[i], div_NEL, self.endLine)
      div_NEL = "" # for first iter only
      mul_NEL = "" # for first iter only

    nonTileFreeIndices = []

    ########################################
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

      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      kStr += "  " + ptrStr + " * D = BatchD[wg];" + self.endLine
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      zeroStr = self.state["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
      kStr += "  " + ptrStr + f" const* C = (beta == {zeroStr}) ? nullptr : BatchC[wg];" + self.endLine

      # apply offset only for general batch
      kStr += self.endLine
      kStr += "  D = D + offsetD;" + self.endLine
      kStr += "  C = C + offsetC;" + self.endLine

    ########################################
    # D index
    kStr += self.endLine
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
    kStr += ");%s" % (self.endLine)

    # W index
    kStr += "  %s idxW = GLOBAL_W( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += 'id%d' % i
    kStr += ");%s" % (self.endLine)

    # D index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
    kStr += ");%s" % (self.endLine)

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  %s strideW = 1 + (size%s - 1) * strideW%s" % (self.uint64Str, indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (size%s - 1) * strideW%s" % (indexChar, indexChar)
    kStr += ";" + self.endLine
    destTypeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    if self.state["VectorWidth"] == 1:
      loadTypeStr = self.datatype
      storeTypeStr = destTypeStr
    else:
      # wider load optimization
      loadTypeStr = "%s%s" % (self.datatype, self.state["VectorWidth"] )
      storeByte = self.state["ProblemType"]["DestDataType"].numBytes() * self.state["VectorWidth"]
      # storeByte should be >=2 (because VectorWidth > 1 here)
      if (storeByte == 2):
        storeTypeStr = "tensile_half"
      else:
        storeTypeStr = "float%u"% (storeByte // 4)

    # parallel reduction
    if self.state["Reduction"] > 1:
      kStr += "  idxW += strideW * idR;%s" % (self.endLine)
      kStr += "  strideW *= NUM_REDUCTION;%s" % (self.endLine)

    # define accum variable(s)
    for vi in range(self.state["VectorWidth"]):
      kStr += "  %s accum%u = 0;%s" % (self.datatype, vi, self.endLine)
    # define result buffer
    kStr += "  %s result[NUM_ELEMENT_LOAD];%s"%(destTypeStr, self.endLine)

    idxStr = [""] if self.state["VectorWidth"] == 1 else [".x",".y",".z",".w"] # element access for wider load
    if self.state["GlobalSplitU"] > self.state["GSUUnrollUnit"]:
      # generate loop for large GSU
      iterUnit = self.state["GSUUnrollUnit"] // self.state["Reduction"]
      minusGSUmod = " - %d"%self.state["GSUmod"] if self.state["GSUmod"] == self.state["GSUUnrollUnit"] else ""
      kStr += "  uint32_t gsu_div = (gsu%s) / %u;%s" % (minusGSUmod, self.state["GSUUnrollUnit"], self.endLine)
      kStr += "  for (int i=0; i<gsu_div; i++) {%s" % self.endLine
      for gsuIdx in range(iterUnit):
        kStr += "    %s temp%d = *((%s*)(W+idxW));%s" % (loadTypeStr, gsuIdx, loadTypeStr, self.endLine)
        kStr += "    idxW  += strideW;%s" % self.endLine
      for gsuIdx in range(iterUnit):
        for vi in range(self.state["VectorWidth"]):
          kStr += "    accum%u += temp%u%s;%s" % (vi, gsuIdx, idxStr[vi], self.endLine)
      kStr += "  }%s" % self.endLine
    # unroll mod part
    iterMod = self.state["GSUmod"]//self.state["Reduction"]
    for gsuIdx in range(iterMod):
      kStr += "  %s temp%d = *((%s*)(W+idxW));%s" % (loadTypeStr, gsuIdx, loadTypeStr, self.endLine)
      kStr += "  idxW  += strideW;%s" % self.endLine
      for vi in range(self.state["VectorWidth"]):
        kStr += "  accum%u += temp%u%s;%s" % (vi, gsuIdx, idxStr[vi], self.endLine)

    # parallel reduction
    # do parallel reduction before alpha
    if self.state["Reduction"] > 1:
      r = 1
      while r < self.state["Reduction"]:
        for vi in range(self.state["VectorWidth"]):
          kStr += "  accum%d += __shfl_down(accum%d, %d, %d);%s" % (vi, vi, r, self.state["Reduction"], self.endLine)
        r *= 2

      # do alpha-beta only for idR==0 (representative index)
      kStr += "  if( idR != 0)%s" % (self.endLine)
      kStr += "    return;%s" % self.endLine

    #alpha
    for vi in range(self.state["VectorWidth"]):
      kStr += "  accum%d *= (%s)alpha;%s" % (vi, self.datatype, self.endLine)
    #Beta
    kStr += "  if( beta != (%s)0){%s" % (self.datatype, self.endLine)
    for vi in range(self.state["VectorWidth"]):
      # load C here
      kStr += "    accum%d += beta * (%s)C[idxC+%d];%s" % (vi, self.datatype, vi, self.endLine)
    kStr += "  }%s" % self.endLine

    # Stochastic Rounding? need to use explicit_downcast
    if self.state["ProblemType"]["DestDataType"].is8bitFloat() \
            and self.state["ProblemType"]["StochasticRounding"]:
      for vi in range(self.state["VectorWidth"]):
        # generate RND... For F8, computeDataType is always f32
        kStr += "  uint32_t x = reinterpret_cast<uint32_t &>(accum%d);%s" % (vi, self.endLine)
        kStr += "  uint32_t drop_bits = x & 0xFFFFu;%s" % (self.endLine)
        kStr += "  drop_bits ^= x >> 16;%s" % (self.endLine)
        kStr += "  drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);%s" % (self.endLine)
        kStr += "  drop_bits *= 0x7000149;%s" % (self.endLine)
        kStr += "  uint32_t rng = (drop_bits ^ 0x13371337 ^ (idxD * 229791) ^ RNDSeed);%s" % (self.endLine)

        # call explicit_downcast
        cmpTypeStr =  self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
        kStr += "  result[%d] = explicit_downcast<%s, %s, true>(accum%d, rng);%s" % (vi, destTypeStr, cmpTypeStr, vi, self.endLine)
    else:
      #covert to output
      for vi in range(self.state["VectorWidth"]):
        kStr += "  result[%d] = (%s)accum%d;%s" % (vi, destTypeStr, vi, self.endLine)

    kStr += "  *((%s*)(D+idxD)) = *((%s*)(result));%s" % (storeTypeStr, storeTypeStr, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideW" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideC" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_W%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)
    kStr += "#undef NUM_ELEMENT_LOAD%s" % (self.endLine)
    # parallel reduction
    kStr += "#undef NUM_REDUCTION%s" % (self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    name += "" if self.state["ProblemType"]["StridedBatched"] else "_GB"
    name += "_PostGSU"
    # add extra string for gsu (only for GSUUnrollUnit > 1)
    if self.state["GSUUnrollUnit"] > 1:
      # This part must match client code (in ContractionSolution.cpp)
      gsuMod = self.state["GSUmod"]
      modStr = ""
      if self.state["GlobalSplitU"] > self.state["GSUUnrollUnit"]:
        modStr += "_mod%u"%self.state["GSUUnrollUnit"]
      name += "%u%s"%(gsuMod, modStr)
    if self.state["VectorWidth"] > 1:
      name += "_VW" + str(self.state["VectorWidth"])
    if self.state["Reduction"] > 1:
      name += "_R" + str(self.state["Reduction"])
    return name


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


  def getSourceFileString(self):
    fileString = ""
    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % self.kernelName
      fileString += "\n"
    fileString += self.functionSignature()
    fileString += self.kernelBody()

    return (0, fileString)
