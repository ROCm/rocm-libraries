################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from copy import deepcopy

from .Common import INDEX_CHARS
from .KernelWriterBase import KernelWriterBase

class KernelWriterActivationOnly(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()

    # determine chars for fast access
    self.states.indexChars = []
    for i in range(0, len(INDEX_CHARS)):
      self.states.indexChars.append(INDEX_CHARS[i])
    self.states.indexChars[self.state["ProblemType"]["Index0"]] = "0" + self.states.indexChars[self.state["ProblemType"]["Index0"]]
    self.states.indexChars[self.state["ProblemType"]["Index1"]] = "1" + self.states.indexChars[self.state["ProblemType"]["Index1"]]
    self.tileChar0 = self.states.indexChars[self.state["ProblemType"]["Index0"]]
    self.tileChar1 = self.states.indexChars[self.state["ProblemType"]["Index1"]]


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
    isStridedBuffer = self.state["ProblemType"]["StridedBatched"]
    ptrStr += "" if isStridedBuffer else "*"
    batch   = "" if isStridedBuffer else "Batch"
    kStr += "  " + ptrStr + " * " + batch + "D," + self.endLine

    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    isStridedBuffer = self.state["ProblemType"]["StridedBatched"]
    ptrStr += "" if isStridedBuffer else "*"
    batch   = "" if isStridedBuffer else "Batch"

    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    enumName = "Tensile::ActivationType_%s"%activationCDataType.toChar()
    if self.state["ProblemType"]["ActivationType"] != 'none':
      if self.state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
        kStr += "  %s const activationType,%s" % (enumName, self.endLine)
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        kStr += "  %s const %s,%s" % (activationCDataType.toDevice(self.language), name, self.endLine)

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.states.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s,%s" % (self.states.indexChars[i], self.endLine)

    return kStr


  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBody(self):
    problemType = self.state["ProblemType"]

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
      kStr += "#define strideD" + self.states.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.states.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.states.indexChars[i]
    indexChar = self.states.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.states.indexChars[i]
      kStr += " + (IDX%s)*strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.states.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)
    kStr += "  if (id >= (size%s" % self.states.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += "*size%s" % self.states.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      kStr += "  id%d = id %% size%s;%s" % (i, self.states.indexChars[i], self.endLine)
      kStr += "  id  = id / size%s;%s" % (self.states.indexChars[i], self.endLine)

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
        batchStride += " * size%s" % self.states.indexChars[i]
      kStr += ";" + self.endLine

      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      kStr += "  " + ptrStr + " * D = BatchD[wg];" + self.endLine


    kStr += self.endLine
    ########################################
    # D index
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      tmpStr = ''
      if i in nonTileFreeIndices:
        tmpStr = '0'
      else:
        tmpStr = 'id%d' % i
      kStr += ', ' if i else ''
      kStr += tmpStr
    kStr += ");%s" % (self.endLine)

    ########################################
    # Activation
    typeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    typeActivationStr = self.state["ProblemType"]["ActivationComputeDataType"].toDevice(self.language)
    if self.state["ProblemType"]["ActivationType"] != 'none':
      names = ""
      if self.state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
        names += ", activationType"
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        names += (", " + name)
      rvalueStr = "activation%s((%s)D[idxD]%s)" % (self.gaurdStr, typeActivationStr, names)

      if self.state["ProblemType"]["DestDataType"].isInt8() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
        rvalueStr = "min(127, max(-128, (int32_t)std::nearbyint(%s)))" % rvalueStr

      kStr += "  D[idxD] = (%s)%s;%s" % (typeStr, rvalueStr, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.states.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = INDEX_CHARS
    # C dimensions
    name = "D"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    if self.state["ProblemType"]["ActivationType"] != 'none':
      if self.state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
        name += "_%s"%"A"
      else:
        name += "_%s"%str(self.state["ProblemType"]["ActivationType"]).upper()
      name += self.state["ProblemType"]["ActivationComputeDataType"].toChar()
    name += ("ng" if self.state["ProblemType"]["ActivationNoGuard"] else "")

    return name


  def getSourceFileString(self):
    fileString = ""
    fileString += self.functionSignature()
    fileString += self.kernelBody()

    return (0, fileString)

  def getHeaderFileString(self):
    fileString = "" # CHeader
    fileString += self.functionSignature()
    fileString += ";\n"

    return fileString
