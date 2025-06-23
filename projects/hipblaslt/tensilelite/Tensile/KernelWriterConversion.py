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

from typing import Dict
from copy import deepcopy

from .KernelWriterBase import KernelWriterBase

from Tensile.Common.Architectures import isaToGfx
from Tensile.Common import INDEX_CHARS, IsaInfo
from Tensile.Common.DataType import DataType

class KernelWriterConversion(KernelWriterBase):

  def __init__(self, state, load_vw, isaInfoMap: Dict[str, IsaInfo]):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["GenPGRPostKernels"] = state["GenPGRPostKernels"]
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]
    self.state["ActivationFused"] = state["ActivationFused"]
    self.state["GlobalSplitU"] = state["GlobalSplitU"]

    self.state["UnrollOnly"] = state["UnrollOnly"]

    self.actGradientPrefix = ""
    if self.state["ProblemType"]["Gradient"]:
      self.actGradientPrefix = "Gradient"
    self.gaurdStr = "NG" if self.state["ProblemType"]["ActivationNoGuard"] else ""

    # setup load vector width
    self.num_elements_load = load_vw

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

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()
    self.isaInfoMap = isaInfoMap
    self.datatype = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
    self.int32Str = DataType('int32').toDevice(self.language)
    if self.state["ProblemType"]["DataType"].isInt8() and self.state["ProblemType"]["ComputeDataType"].isSingle() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      self.datatype = self.int32Str

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(INDEX_CHARS)):
      self.indexChars.append(INDEX_CHARS[i])
    self.indexChars[self.state["ProblemType"]["Index0"]] = "0" + self.indexChars[self.state["ProblemType"]["Index0"]]
    self.indexChars[self.state["ProblemType"]["Index1"]] = "1" + self.indexChars[self.state["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[self.state["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[self.state["ProblemType"]["Index1"]]

    self.gsuKernels = [self.state["GlobalSplitU"]]
    if self.state["GenPGRPostKernels"]:
      pgrgsu = int(self.state["GlobalSplitU"] / 2)
      while pgrgsu > 1:
        self.gsuKernels.append(pgrgsu)
        pgrgsu = int(pgrgsu / 2)

  def functionArgument(self):
    kStr = ""

    # argument structure start
    kStr += self.endLine
    kStr += "struct __attribute__((__packed__)) argument_%s" % ( self.kernelName )
    kStr += "{" + self.endLine

    # pointers
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    ptrStr += '' if self.state["ProblemType"]["StridedBatched"] else '*'
    bStr = '' if self.state["ProblemType"]["StridedBatched"] else 'Batch'

    if self.state["ProblemType"]["UseE"]:
      ptrCStr = self.state["ProblemType"]["DataTypeE"].toDevice(self.language)
      ptrCStr += '' if self.state["ProblemType"]["StridedBatched"] else '*'
      kStr += "  " + ptrCStr + " * " + bStr + "E;" + self.endLine
    kStr += "  " + ptrStr + " * " + bStr + "D;" + self.endLine
    kStr += "  " + self.datatype + " * W;" + self.endLine
    kStr += "  " + ptrStr + " * " + bStr + "C;" + self.endLine

    # bias
    if self.state["ProblemType"]["UseBias"]:
      if (not self.state["ProblemType"]["Gradient"]):
        biasPtrStr = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
        kStr += "  " + biasPtrStr + " * " + "Bias;" + self.endLine
      elif self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"):
        biasPtrStr = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
        kStr += "  " + biasPtrStr + "* " + "Bias;" + self.endLine

    # ScaleAB
    if self.state["ProblemType"]["UseScaleAB"]:
      scalePtrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
      kStr += "  " + scalePtrStr + " * " + "ScaleA;" + self.endLine
      kStr += "  " + scalePtrStr + " * " + "ScaleB;" + self.endLine

    # ScaleCD
    if self.state["ProblemType"]["UseScaleCD"]:
      scalePtrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
      kStr += "  " + scalePtrStr + " * " + "ScaleC;" + self.endLine
      kStr += "  " + scalePtrStr + " * " + "ScaleD;" + self.endLine

    enableFactorDim = False
    # interface: ScaleAlphaVec GSU>1 GSUA "MUL"
    if self.state["ProblemType"]["UseScaleAlphaVec"]:
      scaleAlphaVecPtrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
      kStr += "  " + scaleAlphaVecPtrStr + " * " + "ScaleAlphaVec;" + self.endLine
      if self.state["ProblemType"]["UseScaleAlphaVec"] == 3:
        enableFactorDim = True

    # alpha & beta
    kStr += "  %s alpha;%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "  %s beta;%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)

    # activation
    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    enumName = "Tensile::%sActivationType_%s"%(self.actGradientPrefix, activationCDataType.toChar())
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        kStr += "  %s %s;%s" % (self.state["ProblemType"]["ActivationComputeDataType"].toDevice(self.language), name, self.endLine)
      if self.state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
        kStr += "  %s activationType;%s" % (enumName, self.endLine)

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    if self.state["ProblemType"]["UseE"]:
      for i in range(firstStrideCD, lastStrideC):
        kStr += "  unsigned int strideE%s;%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int strideD%s;%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int strideW%s;%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int strideC%s;%s" % (self.indexChars[i], self.endLine)

    if self.state["ProblemType"]["UseBias"] and \
        (not self.state["ProblemType"]["Gradient"] or \
          (self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"))):
      kStr += "  unsigned int strideBias;%s" % (self.endLine)
      if self.state["ProblemType"]["UseBias"] == 3:
        enableFactorDim = True

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      if i < (self.state["ProblemType"]["NumIndicesC"] - 1):
        kStr += "  unsigned int size%s;%s" % (self.indexChars[i], self.endLine)
      else:
        kStr += "  unsigned int size%s;%s" % (self.indexChars[i], self.endLine)
    kStr += "  unsigned int gsu;%s" % (self.endLine)

    if enableFactorDim:
      kStr += "  unsigned int factorDim;%s" % (self.endLine)

    # argument structure end
    kStr += "};" + self.endLine

    return kStr

  def functionSignature(self):
    kStr = ""

    # kernel name
    kStr += self.endLine
    kStr += "extern \"C\"\n"
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # kernel argument
    if self.state["ProblemType"]["GroupedGemm"]:
      kStr += "  uint32_t* wiTablePtr, void* deviceUserArgsPtr, argument_%s* argsPtr, uint32_t gemm_count)" % ( self.kernelName ) + self.endLine
    else:
      kStr += "  argument_%s arg)" % ( self.kernelName ) + self.endLine

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
    if self.state["ProblemType"]["UseE"]:
      for i in range(firstStride, lastStrideC):
        kStr += "#define strideE" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideW" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_E()
    if self.state["ProblemType"]["UseE"]:
      kStr += "#define GLOBAL_E(IDX%s" % self.indexChars[0]
      for i in range(1, problemType["NumIndicesC"]):
        kStr += ", IDX%s" % self.indexChars[i]
      indexChar = self.indexChars[0]
      kStr += ") (( (IDX%s)*strideE%s" % (indexChar, indexChar)
      for i in range(1, problemType["NumIndicesC"]):
        indexChar = self.indexChars[i]
        kStr += " + (IDX%s)*arg.strideE%s" % (indexChar, indexChar)
      kStr += " ))" + self.endLine

    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*arg.strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_W()
    kStr += "#define GLOBAL_W(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideW%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*arg.strideW%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_C()
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*arg.strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_BIAS()
    if self.state["ProblemType"]["UseBias"] and \
       (not self.state["ProblemType"]["Gradient"] or \
            (self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"))) \
       and self.state["ProblemType"]["NumIndicesC"] > 2:
      kStr += "#define GLOBAL_BIAS(IDX%s" % self.indexChars[0]
      kStr += ", IDX%s" % self.indexChars[2]
      indexChar = self.indexChars[0]
      kStr += ") (( (IDX%s)" % (indexChar)
      indexChar = self.indexChars[2]
      kStr += " + (IDX%s)*arg.strideBias" % (indexChar)
      kStr += " ))" + self.endLine

    self.num_dword_load = int(self.num_elements_load * self.state["ProblemType"]["ComputeDataType"].numBytes() / 4)
    self.num_dword_store = int(self.num_elements_load * self.state["ProblemType"]["DestDataType"].numBytes() / 4)
    if self.num_dword_store == 0:
      self.num_dword_store = self.num_elements_load * self.state["ProblemType"]["DestDataType"].numBytes() / 4
    if self.state["ProblemType"]["DataType"].isDouble():
      self.num_dword_load  = self.num_dword_load // 2
    if self.state["ProblemType"]["DestDataType"].isDouble():
      self.num_dword_store = self.num_dword_store // 2
    kStr += "#define NUM_ELEMENT_LOAD %d%s" % ( self.num_elements_load, self.endLine)
    kStr += "#define NUM_GSU %d%s" % (self.state["GlobalSplitU"], self.endLine)

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)

    ########################################
    # Grouped gemm: find index of gemm
    if self.state["ProblemType"]["GroupedGemm"]:
      kStr += self.endLine
      kStr += "  uint32_t left = 0;" + self.endLine
      kStr += "  uint32_t middle;" + self.endLine
      kStr += "  uint32_t right = gemm_count;" + self.endLine
      kStr += "  uint32_t wiMiddle, wiLeft;" + self.endLine
      kStr += "  uint32_t targetP1 = id + 1;" + self.endLine
      kStr += "  while(left < right)" + self.endLine
      kStr += "  {" + self.endLine
      kStr += "    middle = (left + right) / 2;" + self.endLine
      kStr += "    wiMiddle = wiTablePtr[middle];" + self.endLine
      kStr += "    if(wiMiddle < targetP1)" + self.endLine
      kStr += "    {" + self.endLine
      kStr += "      left = middle + 1;" + self.endLine
      kStr += "      wiLeft = wiMiddle;" + self.endLine
      kStr += "    }" + self.endLine
      kStr += "    else" + self.endLine
      kStr += "      right = middle;" + self.endLine
      kStr += "  }" + self.endLine
      kStr += "  id = id - wiLeft;"  + self.endLine

      # kStr += "  argument_%s arg = argsPtr[left-1];" % ( self.kernelName ) + self.endLine
      kStr += "  argument_%s arg;" % ( self.kernelName ) + self.endLine
      kStr += "  int loadsInBytes = 0;" + self.endLine
      kStr += "  for(; loadsInBytes + 16 <= sizeof(argument_%s); loadsInBytes += 16)" % ( self.kernelName ) + self.endLine
      kStr += "    s_buffer_load<float4, sizeof(float4)>(*((float4*) &arg + loadsInBytes/16), argsPtr+left-1, loadsInBytes);" + self.endLine
      kStr += "  for(; loadsInBytes + 8 <= sizeof(argument_%s); loadsInBytes += 8)" % ( self.kernelName ) + self.endLine
      kStr += "    s_buffer_load<float2, sizeof(float2)>(*((float2*) &arg + loadsInBytes/8), argsPtr+left-1, loadsInBytes);" + self.endLine
      kStr += "  for(; loadsInBytes + 4 <= sizeof(argument_%s); loadsInBytes += 4)" % ( self.kernelName ) + self.endLine
      kStr += "    s_buffer_load<float1, sizeof(float1)>(*((float1*) &arg + loadsInBytes/4), argsPtr+left-1, loadsInBytes);" + self.endLine

    ########################################
    # kernel start
    kStr += self.endLine
    kStr += "  if (id*NUM_ELEMENT_LOAD >= (arg.size%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += " * arg.size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      if i == 0:
        kStr += "  id%d = (id %% (arg.size%s/NUM_ELEMENT_LOAD)) * NUM_ELEMENT_LOAD;%s" % (i, self.indexChars[i], self.endLine)
        kStr += "  id  = id / (arg.size%s/NUM_ELEMENT_LOAD);%s" % (self.indexChars[i], self.endLine)
      else:
        kStr += "  id%d = id %% arg.size%s;%s" % (i, self.indexChars[i], self.endLine)
        kStr += "  id  = id / arg.size%s;%s" % (self.indexChars[i], self.endLine)

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
        batchStride += " * arg.size%s" % self.indexChars[i]
      kStr += ";" + self.endLine

      if self.state["ProblemType"]["UseE"]:
        ptrStr = self.state["ProblemType"]["DataTypeE"].toDevice(self.language)
        kStr += "  " + ptrStr + " * arg.E = arg.BatchE[wg];" + self.endLine
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      kStr += "  " + ptrStr + " * arg.D = arg.BatchD[wg];" + self.endLine
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      zeroStr = self.state["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
      kStr += "  " + ptrStr + f" const* arg.C = (arg.beta == {zeroStr}) ? nullptr : arg.BatchC[wg];" + self.endLine

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

    # C index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
    kStr += ");%s" % (self.endLine)

    if self.state["ProblemType"]["UseBias"] and \
       (not self.state["ProblemType"]["Gradient"] or \
         (self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"))):

      id_str = "id0"
      if self.state["ProblemType"]["UseBias"] == 3:
        id_str = "idb"
        kStr += "  %s idb = ( arg.factorDim == 0 ? (%s)id0 : id1);%s" % (self.uint64Str, self.uint64Str, self.endLine)
      elif self.state["ProblemType"]["UseBias"] == 2:
        id_str = "id1"
      if problemType["NumIndicesC"] > 2:
        kStr += "  %s idxBias = GLOBAL_BIAS((%s)%s, id2);%s" % (self.uint64Str, self.uint64Str, id_str, self.endLine)
      else:
        kStr += "  %s idxBias = %s;%s" % (self.uint64Str, id_str, self.endLine)


    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    intermediateDataType = self.datatype
    if self.state["ProblemType"]["DataType"].isInt8() and self.state["ProblemType"]["ComputeDataType"].isSingle() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      intermediateDataType = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)

    destTypeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)

    indexChar = self.indexChars[0]
    kStr += "  %s strideW = 1 + (arg.size%s - 1) * strideW%s" % (self.uint64Str, indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (arg.size%s - 1) * arg.strideW%s" % (indexChar, indexChar)
    kStr += ";" + self.endLine
    kStr += "  %s strideWLimit = strideW * arg.gsu * sizeof(%s);"%(self.uint64Str, self.datatype) + self.endLine

    kStr += "  " + intermediateDataType + " accum[NUM_ELEMENT_LOAD] = {0};" + self.endLine
    kStr += "  " + destTypeStr + " result[NUM_ELEMENT_LOAD];" + self.endLine

    #Load scaleAB
    if self.state["ProblemType"]["UseScaleAB"] == "Scalar":
      kStr += "  " + intermediateDataType + " scaleA_data, scaleB_data;" + self.endLine
      kStr += "  " + "scaleA_data = arg.ScaleA == nullptr ? 1 : *(arg.ScaleA);" + self.endLine
      kStr += "  " + "scaleB_data = arg.ScaleB == nullptr ? 1 : *(arg.ScaleB);" + self.endLine

    #Load scaleCD
    if self.state["ProblemType"]["UseScaleCD"]:
      kStr += "  " + intermediateDataType + " scaleC_data, scaleD_data;" + self.endLine
      kStr += "  " + "scaleC_data = arg.ScaleC == nullptr ? 1 : *(arg.ScaleC);" + self.endLine
      kStr += "  " + "scaleD_data = arg.ScaleD == nullptr ? 1 : *(arg.ScaleD);" + self.endLine

    #TODO: workspace type is half precision
    if self.state["ProblemType"]["UseBias"] and self.state["ProblemType"]["Gradient"] and self.state["ProblemType"]["BiasSrc"] == "D":
      kStr += "  auto idxW_ori = idxW;%s"%self.endLine

    typeStr = "int" if self.state["ProblemType"]["DataType"].isInt8() or self.state["ProblemType"]["DataType"].isInt32() else ("double" if self.state["ProblemType"]["DataType"].isDouble() else "float")
    typeStr2 = "int16_t" if self.state["ProblemType"]["DestDataType"].isInt8() else ("tensile_half" if self.state["ProblemType"]["DestDataType"].isAnyFloat8() else "tensile_bfloat16")
    loadTypeStr = "%s%s" % (typeStr, "" if self.num_dword_load == 1 else self.num_dword_load)
    storeTypeStr = "%s%s" % (typeStr, self.num_dword_store) if self.num_dword_store >= 1 else typeStr2 if self.num_dword_store == 0.5 else destTypeStr

    #Bias A/B
    if self.state["ProblemType"]["UseBias"] and self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"):
      size          = "arg.size0I" if self.state["ProblemType"]["BiasSrc"] == "A" else "arg.size1J"
      barrier       = "id1" if self.state["ProblemType"]["BiasSrc"] == "A" else "id0"
      biasIdxStr    = "id0" if self.state["ProblemType"]["BiasSrc"] == "A" else "id1"
      biasIdxGsuStr = biasIdxStr + "Gsu"
      biasPtrStr    = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
      kStr += "  if(%s == 0 && id2 == 0)%s"%(barrier, self.endLine)
      kStr += "  {%s" % self.endLine
      kStr += "    auto offset = strideW * arg.gsu;%s"% self.endLine
      kStr += "    auto strideBias = %s;%s"%(size, self.endLine)
      kStr += "    auto %s = %s + offset;%s"%(biasIdxGsuStr, biasIdxStr, self.endLine)
      biasLoadCount = 1
      if self.num_dword_load != 1 and self.state["ProblemType"]["BiasSrc"] == "A":
        biasLoadCount = self.num_dword_load
      kStr += "    " + intermediateDataType + " biasAccum[%d] = {0};%s" % (biasLoadCount ,self.endLine)
      kStr += "    for (int i = 0; i < arg.gsu; i++) {%s" % self.endLine
      for vIdx in range(biasLoadCount):
        kStr += "      biasAccum[%d] += arg.W[%s+%d];%s" % (vIdx, biasIdxGsuStr, vIdx, self.endLine)
      kStr += "      %s  += strideBias;%s" % (biasIdxGsuStr, self.endLine)
      kStr += "    }%s" % self.endLine
      for vIdx in range(biasLoadCount):
        kStr += "    arg.Bias[%s+%d] = (%s)biasAccum[%d];%s"%(biasIdxStr, vIdx, biasPtrStr, vIdx, self.endLine)
      kStr += "  }%s" % self.endLine
    kStr += self.endLine

    #Load GSU D buffer
    if self.state["UnrollOnly"]:
      kStr += "  %s temp[NUM_GSU];" % loadTypeStr + self.endLine
      for gsuIdx in range(self.state["GlobalSplitU"]):
        # kStr += "  temp[%d] = *((%s*)(arg.W+idxW));%s" % (gsuIdx, loadTypeStr, self.endLine)
        kStr += "  buffer_load<%s, sizeof(%s), CacheOperation::Kind::Always>(temp[%d], arg.W, idxW * sizeof(%s), 0, strideWLimit);%s" % (loadTypeStr, loadTypeStr, gsuIdx, self.datatype, self.endLine)
        kStr += "  idxW  += strideW;" + self.endLine
      kStr += self.endLine
      castToIntermidate = ("(%s)" % intermediateDataType) if intermediateDataType != self.datatype else ""
      #Accumlate all D buffer
      for gsuIdx in range(self.state["GlobalSplitU"]):
        if self.num_dword_load == 1:
          kStr += "  accum[0] += %stemp[%d];" % (castToIntermidate, gsuIdx) + self.endLine
        elif self.num_dword_load >= 2:
          kStr += "  accum[0] += %stemp[%d].x;" % (castToIntermidate, gsuIdx) + self.endLine
          kStr += "  accum[1] += %stemp[%d].y;" % (castToIntermidate, gsuIdx) + self.endLine
        if self.num_dword_load == 4:
          kStr += "  accum[2] += %stemp[%d].z;" % (castToIntermidate, gsuIdx) + self.endLine
          kStr += "  accum[3] += %stemp[%d].w;" % (castToIntermidate, gsuIdx) + self.endLine
      kStr += self.endLine
    else:
      if self.state["ProblemType"]["ComputeDataType"].isSingle():
        if self.num_dword_load > 1:
          kStr += "  float2 accumVec(accum[0], accum[1]);" + self.endLine
          if self.num_dword_load > 2:
            kStr += "  float2 accumVec2(accum[2], accum[3]);" + self.endLine
      canPKF32Arch = []
      for isa in self.isaInfoMap.keys():
        if self.isaInfoMap[isa].asmCaps['v_pk_add_f32']:
          canPKF32Arch.append(isa)
      defineStr = []
      if len(canPKF32Arch) > 0:
        defineStr = "#if defined(__%s__)"%isaToGfx(canPKF32Arch[0])
        for arch in canPKF32Arch[1:]:
          defineStr += "|| defined(__%s__)"%isaToGfx(arch)
      else:
        defineStr = "#if 0"
      # PGR=2
      kStr += "  %s temp[NUM_GSU];" % loadTypeStr + self.endLine
      for gsuIdx in range(self.state["GlobalSplitU"]):
        kStr += "  buffer_load<%s, sizeof(%s), CacheOperation::Kind::Always>(temp[%d], arg.W, idxW * sizeof(%s), 0, strideWLimit);%s" % (loadTypeStr, loadTypeStr, gsuIdx, self.datatype, self.endLine)
        kStr += "  idxW  += strideW;" + self.endLine
      kStr += self.endLine
      kStr += "  int gsuRemain = (int)arg.gsu - NUM_GSU;" + self.endLine
      kStr += "  while(gsuRemain >= NUM_GSU)" + self.endLine
      kStr += "  {" + self.endLine
      kStr += "    gsuRemain -= NUM_GSU;" + self.endLine
      for gsuIdx in range(self.state["GlobalSplitU"]):
        castToIntermidate = ("(%s)" % intermediateDataType) if intermediateDataType != self.datatype else ""
        if self.state["ProblemType"]["ComputeDataType"].isSingle():
          kStr += self.getAsm(defineStr, castToIntermidate, gsuIdx, space="    ")
        else:
          if self.num_dword_load == 1:
            kStr += "  accum[0] += %stemp[%d];" % (castToIntermidate, gsuIdx) + self.endLine
          elif self.num_dword_load >= 2:
            kStr += "  accum[0] += %stemp[%d].x;" % (castToIntermidate, gsuIdx) + self.endLine
            kStr += "  accum[1] += %stemp[%d].y;" % (castToIntermidate, gsuIdx) + self.endLine
          if self.num_dword_load == 4:
            kStr += "  accum[2] += %stemp[%d].z;" % (castToIntermidate, gsuIdx) + self.endLine
            kStr += "  accum[3] += %stemp[%d].w;" % (castToIntermidate, gsuIdx) + self.endLine
        kStr += "    __builtin_amdgcn_sched_barrier(0);" + self.endLine
        kStr += "    buffer_load<%s, sizeof(%s), CacheOperation::Kind::Always>(temp[%d], arg.W, idxW * sizeof(%s), 0, strideWLimit);%s" % (loadTypeStr, loadTypeStr, gsuIdx, self.datatype, self.endLine)
        kStr += "    __builtin_amdgcn_sched_barrier(0);" + self.endLine
        kStr += "    idxW  += strideW;" + self.endLine
      kStr += "  }" + self.endLine
      # Switch method
      kStr += "  switch(gsuRemain)" + self.endLine
      kStr += "  {" + self.endLine
      for gsuIdx in reversed(range(self.state["GlobalSplitU"])):
        kStr += ("    case %d:"%gsuIdx if gsuIdx > 0 else "    default:") + self.endLine
        kStr += "    {" + self.endLine
        castToIntermidate = ("(%s)" % intermediateDataType) if intermediateDataType != self.datatype else ""
        caseRemain = min(gsuIdx, self.state["GlobalSplitU"])
        for gsuIdx2 in range(self.state["GlobalSplitU"]):
          castToIntermidate = ("(%s)" % intermediateDataType) if intermediateDataType != self.datatype else ""
          if self.state["ProblemType"]["ComputeDataType"].isSingle():
            kStr += self.getAsm(defineStr, castToIntermidate, gsuIdx2, space="      ")
          else:
            if self.num_dword_load == 1:
              kStr += "  accum[0] += %stemp[%d];" % (castToIntermidate, gsuIdx2) + self.endLine
            elif self.num_dword_load >= 2:
              kStr += "  accum[0] += %stemp[%d].x;" % (castToIntermidate, gsuIdx2) + self.endLine
              kStr += "  accum[1] += %stemp[%d].y;" % (castToIntermidate, gsuIdx2) + self.endLine
            if self.num_dword_load == 4:
              kStr += "  accum[2] += %stemp[%d].z;" % (castToIntermidate, gsuIdx2) + self.endLine
              kStr += "  accum[3] += %stemp[%d].w;" % (castToIntermidate, gsuIdx2) + self.endLine
          if caseRemain > gsuIdx2:
            kStr += "      __builtin_amdgcn_sched_barrier(0);" + self.endLine
            kStr += "      buffer_load<%s, sizeof(%s), CacheOperation::Kind::Always>(temp[%d], arg.W, idxW * sizeof(%s), 0, strideWLimit);%s" % (loadTypeStr, loadTypeStr, gsuIdx2, self.datatype, self.endLine)
            kStr += "      __builtin_amdgcn_sched_barrier(0);" + self.endLine
            kStr += "      idxW  += strideW;" + self.endLine
        for gsuIdx2 in range(caseRemain):
          castToIntermidate = ("(%s)" % intermediateDataType) if intermediateDataType != self.datatype else ""
          if self.state["ProblemType"]["ComputeDataType"].isSingle():
            kStr += self.getAsm(defineStr, castToIntermidate, gsuIdx2, space="      ")
          else:
            if self.num_dword_load == 1:
              kStr += "  accum[0] += %stemp[%d];" % (castToIntermidate, gsuIdx2) + self.endLine
            elif self.num_dword_load >= 2:
              kStr += "  accum[0] += %stemp[%d].x;" % (castToIntermidate, gsuIdx2) + self.endLine
              kStr += "  accum[1] += %stemp[%d].y;" % (castToIntermidate, gsuIdx2) + self.endLine
            if self.num_dword_load == 4:
              kStr += "  accum[2] += %stemp[%d].z;" % (castToIntermidate, gsuIdx2) + self.endLine
              kStr += "  accum[3] += %stemp[%d].w;" % (castToIntermidate, gsuIdx2) + self.endLine
        kStr += "    } break;" + self.endLine
      kStr += "  }" + self.endLine

      kStr += defineStr + self.endLine
      if self.state["ProblemType"]["ComputeDataType"].isSingle():
        if self.num_dword_load > 1:
          kStr += "  accum[0] = accumVec.x;" + self.endLine
          kStr += "  accum[1] = accumVec.y;" + self.endLine
          if self.num_dword_load > 2:
            kStr += "  accum[2] = accumVec2.x;" + self.endLine
            kStr += "  accum[3] = accumVec2.y;" + self.endLine
      kStr += "#endif" + self.endLine

    accumStr = "accum"
    resultStr = "result"

    #scaleAB
    if self.state["ProblemType"]["UseScaleAB"] == "Scalar":
      kStr += "  arg.alpha = arg.alpha*scaleA_data*scaleB_data;%s" % (self.endLine)
      kStr += self.endLine
    elif self.state["ProblemType"]["UseScaleAB"] == "Vector":
      kStr += "  if(arg.ScaleA != nullptr) {" + self.endLine
      for vIdx in range(self.num_dword_load):
        kStr += "    %s[%d] *= (%s)arg.ScaleA[id0+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
      kStr += "  }" + self.endLine
      kStr += "  if(arg.ScaleB != nullptr) {" + self.endLine
      for vIdx in range(self.num_dword_load):
        kStr += "      %s[%d] *= (%s)arg.ScaleB[id1];%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
      kStr += "  }" + self.endLine
      kStr += self.endLine

    #alpha
    for vIdx in range(self.num_dword_load):
      kStr += "  %s[%d] *= (%s)arg.alpha;%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
    kStr += self.endLine

    if self.state["ProblemType"]["UseScaleAlphaVec"]:
      kStr += "  if(arg.ScaleAlphaVec != nullptr){" + self.endLine

      if self.state["ProblemType"]["UseScaleAlphaVec"] == 3:
        kStr += "    if(arg.factorDim == 0){" + self.endLine
        for vIdx in range(self.num_dword_load):
          kStr += "      %s[%d] *= (%s)arg.ScaleAlphaVec[id0+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
        kStr += "    }else{" + self.endLine
        for vIdx in range(self.num_dword_load):
          kStr += "      %s[%d] *= (%s)arg.ScaleAlphaVec[id1];%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
        kStr += "    }" + self.endLine
        kStr += "  }" + self.endLine
      elif self.state["ProblemType"]["UseBias"] == 2:
        for vIdx in range(self.num_dword_load):
          kStr += "    %s[%d] *= (%s)arg.ScaleAlphaVec[id1];%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
        kStr += "  }" + self.endLine
      else:
        for vIdx in range(self.num_dword_load):
          kStr += "    %s[%d] *= (%s)arg.ScaleAlphaVec[id0+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
        kStr += "  }" + self.endLine

      kStr += self.endLine

    #scaleC
    if self.state["ProblemType"]["UseScaleCD"]:
      kStr += "  arg.beta = arg.beta*scaleC_data;%s" % (self.endLine)
    kStr += self.endLine

    #Beta
    kStr += "  if(arg.beta != (%s)0){%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    for vIdx in range(self.num_dword_load):
      kStr += "    %s[%d] += arg.beta * (%s)arg.C[idxC+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
    kStr += "  }" + self.endLine
    kStr += self.endLine

    #Bias
    if self.state["ProblemType"]["UseBias"] and (not self.state["ProblemType"]["Gradient"]):
      kStr += "  if(arg.Bias != 0){" + self.endLine
      if self.state["ProblemType"]["UseBias"] == 3:
        kStr += "    if(arg.factorDim == 0){" + self.endLine
        for vIdx in range(self.num_dword_load):
          kStr += "      %s[%d] += (%s)arg.Bias[idxBias+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
        kStr += "    }else{" + self.endLine
        for vIdx in range(self.num_dword_load):
          kStr += "      %s[%d] += (%s)arg.Bias[idxBias];%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
        kStr += "    }" + self.endLine
        kStr += "  }" + self.endLine
      elif self.state["ProblemType"]["UseBias"] == 2:
        for vIdx in range(self.num_dword_load):
          kStr += "    %s[%d] += (%s)arg.Bias[idxBias];%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
        kStr += "  }" + self.endLine
      else:
        for vIdx in range(self.num_dword_load):
          kStr += "    %s[%d] += (%s)arg.Bias[idxBias+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
        kStr += "  }" + self.endLine

      kStr += self.endLine

    #Handle E
    if self.state["ProblemType"]["UseE"]:
      dataTypeE = self.state["ProblemType"]["DataTypeE"].toDevice(self.language)
      if self.state["ProblemType"]["Gradient"]:
        kStr += "  %s idxE = GLOBAL_E( (%s)" % (self.uint64Str, self.uint64Str)
        for i in range(problemType["NumIndicesC"]):
          kStr += ', ' if i else ''
          kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
        kStr += ");%s" % (self.endLine)
        kStr += "  %s dataE[%d];%s" % (intermediateDataType, self.num_dword_load, self.endLine)
        for vIdx in range(self.num_dword_load):
          kStr += "  dataE[%d] = (%s)arg.E[idxE+%d];%s" % ( vIdx, intermediateDataType, vIdx, self.endLine)
      else:
        # E index
        kStr += "  if( arg.E != nullptr)%s" % (self.endLine)
        kStr += "  {%s" % (self.endLine)
        kStr += "    %s idxE = GLOBAL_E( (%s)" % (self.uint64Str, self.uint64Str)
        for i in range(problemType["NumIndicesC"]):
          kStr += ', ' if i else ''
          kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
        kStr += ");%s" % (self.endLine)
        for vIdx in range(self.num_dword_load):
          kStr += "    arg.E[idxE+%d] = (%s)(accum[%d]);%s" % (vIdx, dataTypeE, vIdx, self.endLine)
        kStr += "  }%s" % (self.endLine)

    #Activation
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      typeActivationStr = self.state["ProblemType"]["ActivationComputeDataType"].toDevice(self.language)
      actArgs = ""
      if self.state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
        actArgs += ", arg.activationType"
      for args in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        actArgs += (", " + "arg." + args)
      for vIdx in range(self.num_dword_load):
        if self.state["ProblemType"]["Gradient"]:
          kStr += "  %s[%d] *= activation%s((%s)dataE[%d]%s);%s" % (accumStr, vIdx, self.gaurdStr, typeActivationStr, vIdx, actArgs, self.endLine)
        else:
          kStr += "  %s[%d] = activation%s((%s)%s[%d]%s);%s" % (accumStr, vIdx, self.gaurdStr, typeActivationStr, accumStr, vIdx, actArgs, self.endLine)
      kStr += self.endLine

    #scaleD
    if self.state["ProblemType"]["UseScaleCD"]:
      for vIdx in range(self.num_dword_load):
        kStr += "  %s[%d] *= scaleD_data;%s" % (accumStr, vIdx, self.endLine)
    kStr += self.endLine

    #Output high precision D to WS
    if self.state["ProblemType"]["UseBias"] and self.state["ProblemType"]["Gradient"] and self.state["ProblemType"]["BiasSrc"] == "D":
      for vIdx in range(self.num_dword_load):
        kStr += "  arg.W[idxW_ori+%d] = accum[%d];%s" % (vIdx, vIdx, self.endLine)

    #Saturation
    if self.state["ProblemType"]["DestDataType"].isInt8() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      for vIdx in range(self.num_dword_load):
        kStr += "  %s[%d] = min(127, max(-128, (int32_t)std::nearbyint(%s[%d])));%s" % (accumStr, vIdx, accumStr, vIdx, self.endLine)
      kStr += self.endLine

    #covert to output
    for vIdx in range(self.num_dword_load):
      kStr += "  %s[%d] = (%s)%s[%d];%s" % (resultStr, vIdx, destTypeStr, accumStr, vIdx, self.endLine)

    # kStr += "  *(%s *)(arg.D+idxD) = *(%s *)%s;%s" % (storeTypeStr, storeTypeStr, resultStr, self.endLine)
    kStr += "  buffer_store<%s, sizeof(%s), CacheOperation::Kind::Always>(*(%s *)%s, arg.D, idxD * sizeof(%s), 0);%s" % (storeTypeStr, storeTypeStr, storeTypeStr, resultStr, destTypeStr, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    kStr += "#undef NUM_GSU" + self.endLine
    kStr += "#undef NUM_ELEMENT_LOAD" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideW" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideC" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_W%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)
    if self.state["ProblemType"]["UseBias"]:
      kStr += "#undef GLOBAL_BIAS%s" % (self.endLine)
    if self.state["ProblemType"]["UseE"]:
      kStr += "#undef GLOBAL_E%s" % (self.endLine)

    return kStr


  @staticmethod
  def kernelName(solution, num_elements_load, btype=None):
    state = solution._state if hasattr(solution, "_state") else solution.state
    indexChars = INDEX_CHARS
    # C dimensions
    name = "C"
    for i in range(0, state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"

    # add input datatype into kernel name (the datatype of workspace)
    inputTypeStr = DataType("I").toChar() if state["ProblemType"]["DataType"].isInt8() or state["ProblemType"]["DataType"].isInt32() else \
                                  (DataType("D").toChar() if state["ProblemType"]["DataType"].isDouble() else DataType("S").toChar())

    name += (inputTypeStr + state["ProblemType"]["DestDataType"].toChar())

    if state["ProblemType"]["GroupedGemm"]:
      name += "_GG"
    else:
      name += "" if state["ProblemType"]["StridedBatched"] else "_GB"
    if btype:
      if state["ProblemType"]["Gradient"]:
        name += "_DBias%s"%(btype.toChar())
        name += "_BiasSrc%s"%(state["ProblemType"]["BiasSrc"])
      else:
        name += "_Bias%s"%btype.toChar()

    factorDim =  0 if state["ProblemType"]["Gradient"] else state["ProblemType"]["UseBias"]
    factorDim =  max(factorDim, state["ProblemType"]["UseScaleAlphaVec"])
    if factorDim > 1:
        name += "_FD%s"%("N" if factorDim == 2 else "MN")

    if state["ProblemType"]["UseE"]:
      if state["ProblemType"]["Gradient"]:
        name += "_Grad%s"%state["ProblemType"]["DataTypeE"].toChar()
      else:
        name += "_Aux%s"%state["ProblemType"]["DataTypeE"].toChar()

    if ((state["ProblemType"]["ActivationType"] != 'none') and state["ActivationFused"]):
      if state["ProblemType"]["ActivationType"] == 'all':
        name += "_A"
      elif state["ProblemType"]["ActivationType"] == 'hipblaslt_all':
        name += "_HA"
      else:
        name += "_%s"%str(state["ProblemType"]["ActivationType"]).upper()
      name += state["ProblemType"]["ActivationComputeDataType"].toChar()
      name += ("ng" if state["ProblemType"]["ActivationNoGuard"] else "")
    if state["ProblemType"]["UseScaleAB"] == "Scalar":
      name += "_ScaleAB"
    elif state["ProblemType"]["UseScaleAB"] == "Vector":
      name += "_ScaleABVec"
    name += "_ScaleCD" if state["ProblemType"]["UseScaleCD"] else ""
    name += "_ScaleAlphaVec" if state["ProblemType"]["UseScaleAlphaVec"] else ""
    name += "_PostGSU" + str(state["GlobalSplitU"])
    if num_elements_load != None:
      name += "_VW" + str(num_elements_load)
    return name


  def getKernelName(self):
    btype = self.state["ProblemType"]["BiasDataType"] if self.state["ProblemType"]["UseBias"] else None
    return KernelWriterConversion.kernelName(self, self.num_elements_load, btype)


  def getHeaderFileString(self):
    fileString = "" # CHeader
    backupGSU    = self.state["GlobalSplitU"]
    backupUnroll = self.state["UnrollOnly"]
    for gsu in self.gsuKernels:
      for toggle in [True, False]:
        self.state["GlobalSplitU"] = gsu
        self.state["ProblemType"]["GroupedGemm"] = toggle
        self.kernelName = self.getKernelName()
        fileString += self.f8MacroGuardStart
        fileString += self.functionArgument()
        fileString += self.functionSignature()
        fileString += ";\n"
        fileString += self.f8MacroGuardEnd
      if not self.state["UnrollOnly"]:
        self.state["UnrollOnly"] = True
    self.state["GlobalSplitU"] = backupGSU
    self.state["UnrollOnly"] = backupUnroll


    return fileString


  def getSourceFileString(self):
    fileString = ""
    backupGSU    = self.state["GlobalSplitU"]
    backupUnroll = self.state["UnrollOnly"]
    for gsu in self.gsuKernels:
      for toggle in [True, False]:
        self.state["GlobalSplitU"] = gsu
        self.state["ProblemType"]["GroupedGemm"] = toggle
        self.kernelName = self.getKernelName()
        fileString += self.f8MacroGuardStart
        fileString += self.functionSignature()
        fileString += self.kernelBody()
        fileString += self.f8MacroGuardEnd
      if not self.state["UnrollOnly"]:
        self.state["UnrollOnly"] = True
    self.state["GlobalSplitU"] = backupGSU
    self.state["UnrollOnly"] = backupUnroll

    return (0, fileString)

  def getAsm(self, defineStr, castToIntermidate, gsuIdx, space=""):
    kStr = ""
    kStr += space + "asm __volatile__(" + self.endLine
    if self.num_dword_load == 1:
      if self.datatype == self.int32Str:
        kStr += space + "    \"v_cvt_f32_i32 v0, %1 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %0, v0, %0 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accum[0]): \"v\"(temp[%d])"% (gsuIdx) + self.endLine
      else:
        kStr += space + "    \"v_add_f32 %0, %1, %0 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accum[0]): \"v\"(%stemp[%d])"% (castToIntermidate, gsuIdx) + self.endLine
    elif self.num_dword_load == 2:
      if self.datatype == self.int32Str:
        kStr += defineStr + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v0, %1 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v1, %2 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_pk_add_f32 %0, v[0:1], %0 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accumVec): \"v\"(temp[%d].x), \"v\"(temp[%d].y)"% (gsuIdx, gsuIdx) + self.endLine
        kStr += "#else" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v0, %2 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v1, %3 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %0, v0, %0 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %1, v1, %1 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accum[0]), \"+v\"(accum[1]): \"v\"(temp[%d].x), \"v\"(temp[%d].y)"% (gsuIdx, gsuIdx) + self.endLine
        kStr += "#endif" + self.endLine
      else:
        kStr += defineStr + self.endLine
        kStr += space + "    \"v_pk_add_f32 %0, %1, %0 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accumVec): \"v\"(%stemp[%d])"% (castToIntermidate, gsuIdx) + self.endLine
        kStr += "#else" + self.endLine
        kStr += space + "    \"v_add_f32 %0, %2, %0 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %1, %3, %1 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accum[0]), \"+v\"(accum[1]): \"v\"(%stemp[%d].x), \"v\"(%stemp[%d].y)"% (castToIntermidate, gsuIdx, castToIntermidate, gsuIdx) + self.endLine
        kStr += "#endif" + self.endLine
    elif self.num_dword_load == 4:
      if self.datatype == self.int32Str:
        kStr += defineStr + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v0, %2 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v1, %3 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v2, %4 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v3, %5 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_pk_add_f32 %0, v[0:1], %0 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_pk_add_f32 %1, v[2:3], %1 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accumVec), \"+v\"(accumVec2): \"v\"(temp[%d].x), \"v\"(temp[%d].y), \"v\"(temp[%d].z), \"v\"(temp[%d].w)"% (gsuIdx, gsuIdx, gsuIdx, gsuIdx) + self.endLine
        kStr += "#else" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v0, %4 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v1, %5 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v2, %6 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_cvt_f32_i32 v3, %7 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %0, v0, %0 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %1, v1, %1 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %2, v2, %2 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %3, v3, %3 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accum[0]), \"+v\"(accum[1]), \"+v\"(accum[2]), \"+v\"(accum[3]): \"v\"(temp[%d].x), \"v\"(temp[%d].y), \"v\"(temp[%d].z), \"v\"(temp[%d].w)"% (gsuIdx, gsuIdx, gsuIdx, gsuIdx) + self.endLine
        kStr += "#endif" + self.endLine
      else:
        kStr += defineStr + self.endLine
        kStr += space + "    \"v_pk_add_f32 %0, %2, %0 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_pk_add_f32 %1, %3, %1 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accumVec), \"+v\"(accumVec2): \"v\"(%smake_float2(temp[%d].x,temp[%d].y)), \"v\"(%smake_float2(temp[%d].z,temp[%d].w))"% (castToIntermidate, gsuIdx, gsuIdx, castToIntermidate, gsuIdx, gsuIdx) + self.endLine
        kStr += "#else" + self.endLine
        kStr += space + "    \"v_add_f32 %0, %4, %0 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %1, %5, %1 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %2, %6, %2 \\n\\t\"" + self.endLine
        kStr += space + "    \"v_add_f32 %3, %7, %3 \\n\\t\"" + self.endLine
        kStr += space + "    : \"+v\"(accum[0]), \"+v\"(accum[1]), \"+v\"(accum[2]), \"+v\"(accum[3]): \"v\"(%stemp[%d].x), \"v\"(%stemp[%d].y), \"v\"(%stemp[%d].z), \"v\"(%stemp[%d].w)"% (castToIntermidate, gsuIdx, castToIntermidate, gsuIdx, castToIntermidate, gsuIdx, castToIntermidate, gsuIdx) + self.endLine
        kStr += "#endif" + self.endLine
    else:
      assert 0 and "Does not support this dword load"
    vgprStr = ""
    if castToIntermidate:
      for i in range(self.num_dword_load):
        if i != 0:
          vgprStr += ","
        vgprStr += "\"v%d\""%i
    kStr += space + "    :%s);"%vgprStr + self.endLine
    return kStr
