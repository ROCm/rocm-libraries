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

from rocisa.enum import DataTypeEnum
from collections import OrderedDict
from collections.abc import Mapping

from typing import List

from Tensile.Activation import ActivationType
from Tensile.Common import fastdeepcopy as deepcopy
from Tensile.Common.Constants import INDEX_CHARS
from Tensile.Common.DataType import DataType
from Tensile.Common.Utilities import assignParameterWithDefault, printWarning, print2, printExit




class ProblemSizeRange:

  def __init__(self, problemType, config):
    self.totalIndices = 1+max(problemType["IndexAssignmentsA"]) + problemType["NumIndicesLD"]
    if len(config) < self.totalIndices:
      for i in range(len(config), self.totalIndices):
        if i < self.totalIndices - problemType["NumIndicesLD"]:
          config.append(0)
        else:
          config.append([0])

    self.indexMax = []
    self.indexIsSized = []
    self.indicesSized = []
    self.indicesMapped = []
    for i in range(0, self.totalIndices):
      dim = deepcopy(config[i])
      if isinstance(dim, list):
        if len(dim) == 1:
          self.indicesSized.append([dim[0], 1, 0, dim[0]])
        elif len(dim) == 2:
          self.indicesSized.append([dim[0], dim[0], 0, dim[1]])
        elif len(dim) == 3:
          self.indicesSized.append([dim[0], dim[1], 0, dim[2]])
        elif len(dim) == 4:
          self.indicesSized.append([dim[0], dim[1], dim[2], dim[3]])
        else:
          printExit("dimension[%u] config (%s) has %u descriptors rather than 1-4."
              % ( i, dim, len(dim) ))
        self.indexIsSized.append(True)
        self.indexMax.append(self.indicesSized[len(self.indicesSized)-1][3])

      elif isinstance(dim, int):
        self.indicesMapped.append(dim)
        self.indexIsSized.append(False)
        self.indexMax.append(self.indicesSized[self.indicesMapped[ \
            len(self.indicesMapped)-1]][3])

    # max num elements in each tensor
    self.maxNumElements = [ 1, 1, 1 ]
    for i in range(0, problemType["NumIndicesC"]):
      self.maxNumElements[0] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsA"]:
      self.maxNumElements[1] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsB"]:
      self.maxNumElements[2] *= self.indexMax[i]

    self.totalProblemSizes = 1
    self.numProblemSizes = [] # per index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append({})
      if self.indexIsSized[i]:
        self.numProblemSizes.append(0)
        index = self.indicesSized[sizedIdx]
        sizedIdx += 1
        currentSize = index[0]
        currentIncrement = index[1]
        while currentSize <= index[3]:
          currentSize += currentIncrement
          currentIncrement += index[2]
          self.numProblemSizes[i] += 1
      else:
        self.numProblemSizes.append(1)
      self.totalProblemSizes *= self.numProblemSizes[i]

    ########################################
    # enumerate problem sizes
    currentSizedIndexSizes = []
    currentSizedIndexIncrements = []
    for i in range(0, len(self.indicesSized)):
      currentSizedIndexSizes.append(self.indicesSized[i][0])
      currentSizedIndexIncrements.append(self.indicesSized[i][1])

    # iterate over all problem sizes
    self.problemSizes = []
    moreProblemSizes = True
    problemIdx = 0
    problemSize = [0]*self.totalIndices
    while moreProblemSizes:
      #/ convert current sized and mapped indices to full sizes
      currentSizedIdx = 0
      currentMappedIdx = 0
      for i in range(0, self.totalIndices):
        if self.indexIsSized[i]:
          problemSize[i] = currentSizedIndexSizes[currentSizedIdx]
          currentSizedIdx+=1
        else:
          problemSize[i] = problemSize[self.indicesMapped[currentMappedIdx]]
          currentMappedIdx+=1
      self.problemSizes.append(tuple(problemSize))

      #/ increment sizes for next benchmark
      currentSizedIndexSizes[0] += currentSizedIndexIncrements[0]
      currentSizedIndexIncrements[0] += self.indicesSized[0][2]
      for i in range(1, len(self.indicesSized)+1):
        # if prior index past max, reset to min and increment next index
        if currentSizedIndexSizes[i-1] > self.indicesSized[i-1][3]:
          #/ reset prior index
          currentSizedIndexSizes[i-1] = self.indicesSized[i-1][0]
          currentSizedIndexIncrements[i-1] = self.indicesSized[i-1][1]
          # increment next index
          if i >= len(self.indicesSized):
            moreProblemSizes = False
          else:
            currentSizedIndexSizes[i] += currentSizedIndexIncrements[i]
            currentSizedIndexIncrements[i] += self.indicesSized[i][2]

      problemIdx+=1

  ########################################
  # YAML format
  def __str__(self):
    state = "[ "
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      if self.indexIsSized[i]:
        indices = self.indicesSized[sizedIdx]
        state += "[ %u, %u, %u, %u ]" \
            % (indices[0], indices[1], indices[2], indices[3])
        sizedIdx += 1
      else:
        indices = self.indicesSized[self.indicesMapped[mappedIdx]]
        state += str(self.indicesMapped[mappedIdx])
        mappedIdx += 1
      if i < len(self.indexIsSized)-1:
        state += ", "
    state += " ]"
    return state

class Problem:
  """ Problem sizes, strides, padding and other info"""
  def __init__(self, sizes=None, stridesA=None, stridesB=None, stridesC=None, stridesD=None, count=None):
    self.sizes = tuple(sizes) if sizes else None
    self.stridesA = tuple(stridesA) if stridesA else None
    self.stridesB = tuple(stridesB) if stridesB else None
    self.stridesC = tuple(stridesC) if stridesC else None
    self.stridesD = tuple(stridesD) if stridesD else None

    self.count = count

  def __str__(self):
    rv= "{ sizes:" + str(list(self.sizes))
    if self.stridesA:
      rv += ", stridesA:" + str(list(self.stridesA))
    if self.stridesB:
      rv += ", stridesB:" + str(list(self.stridesB))
    if self.stridesC:
      rv += ", stridesC:" + str(list(self.stridesC))
    if self.stridesD:
      rv += ", stridesD:" + str(list(self.stridesD))
    rv += " }"
    return rv

class ExactList(Problem):
  def __init__(self, e, problemType):
    if len(e) == problemType["TotalIndices"]:
      if -1 in e:
        printExit("ExactSize %s contains -1" % (e))
      if problemType["OperationType"] == "GEMM":
        e += [-1, -1, -1, -1]
        e = ExactList.convertLeadingDims(problemType, tuple(e))
      sizes=e

    elif len(e) == (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
      sizes = ExactList.convertLeadingDims(problemType, tuple(e))
    else:
      printExit("ExactSize %s doesn't match indices of ProblemType %s, totalIndices=%d, len e=%d, NumIndicesLD = %d" \
          % (e, problemType, problemType["TotalIndices"], len(e), problemType["NumIndicesLD"]) )

    # TODO- pass strides here, remove calls to convertLeadingDims
    Problem.__init__(self, sizes=sizes)

  def __str__(self):
    return str(list(self.sizes))

  @staticmethod
  def convertLeadingDims(problemType, problemSize, stridesA = None, stridesB = None, stridesC = None, stridesD = None):
    # FIXME-problem: refactor to eliminate max, pass strides in strideB parm rather than hacked
    # onto the end of the sizes list
    predStridesD = stridesD is not None and stridesD[1] != -1
    predStridesC = stridesC is not None and stridesC[1] != -1
    predStridesA = stridesA is not None and stridesA[1] != -1
    predStridesB = stridesB is not None and stridesB[1] != -1
    return problemSize[:problemType["NumIndicesC"]+1] + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][0]]) if not predStridesD else stridesD[1], ) + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][1]]) if not predStridesC else stridesC[1], ) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][2]],
                problemSize[problemType["IndexAssignmentsA"][0]]) if not predStridesA else stridesA[1], ) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][3]],
                problemSize[problemType["IndexAssignmentsB"][0]]) if not predStridesB else stridesB[1], )


class ExactDict(Problem):
  AllowedFields = [ 'count', 'sizes', 'stridesA', 'stridesB', 'stridesC', 'stridesD' ]

  def __init__(self, e, problemType):
    Problem.__init__(self)

    for f in e:
      if f in ExactDict.AllowedFields:
        setattr(self, f, e[f])
      else:
        raise RuntimeError ("specified field '%s' is not a valid Exact dict field"%f)

    if problemType:
      if "OperationType" in problemType and problemType["OperationType"] == "GEMM":
        sizesTuple = tuple(self.sizes + [-1, -1, -1, -1])
        self.sizes = ExactList.convertLeadingDims(problemType, sizesTuple, self.stridesA, self.stridesB, self.stridesC, self.stridesD)

    if problemType:
      if "OperationType" in problemType and problemType["OperationType"] == "GEMM":
        if len(self.sizes) != (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
        # FIXME-ExactDict size descriptor still (but preferrably not so) uses 8-tuple for GEMM problems
          raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"]+problemType["NumIndicesLD"], len(self.sizes)))
      elif len(self.sizes) != problemType["TotalIndices"]:
        raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"], len(self.sizes)))


################################################################################
# ProblemSizes
################################################################################
"""
Adapter class for class `ProblemSizes`. It satisfies the implicit usage requirement
of ClientWriter.writeClientConfig() by converting ExactLogic to list of `Problem` objects
"""
class ProblemSizesMock:
  def __init__(self, exactLogic):
    self.problems = [Problem(problem) for problem, solution in exactLogic]

class ProblemSizesMockDummy:
  def __init__(self):
    self.problems = [Problem(sizes=[128, 128, 1, 512])]

class ProblemSizes:

  ########################################
  def __init__(self, problemType, config):
    self.problemType = problemType
    self.ranges = []
    self.exacts = []
    self.minStrides = None
    if config:
      for dictionary in config:
        for sizeTypeKey in dictionary:
          #print ("PROBLEM parsed:", sizeTypeKey, dictionary[sizeTypeKey])
          if sizeTypeKey == "Range":
            psr = ProblemSizeRange(problemType, dictionary[sizeTypeKey])
            self.ranges.append( psr )
          elif sizeTypeKey == "Exact":
            e= dictionary[sizeTypeKey]
            if isinstance(e,list):
              self.exacts.append(ExactList(e, problemType))
            elif isinstance(e,dict):
              self.exacts.append(ExactDict(e, problemType))
            else:
              printExit("Unsupported Exact type==%s"%type(e))
          elif sizeTypeKey == "MinStride":
            e = dictionary[sizeTypeKey]
            if len(e) != problemType["TotalIndices"]:
              printExit("MinStride %s doesn't match indices of ProblemType %s" \
                  % (e, problemType) )
            if self.minStrides:
              printExit("Only one MinStride command is allowed in a ProblemsSizes definition.  Previous minStrides:%s, New minstride:%s" \
                  % (self.minStrides, e) )

            self.minStrides=(tuple(e))
          else:
            printExit("ProblemSize Type %s not supported"%sizeTypeKey)

    if not self.minStrides:
      # set harmless default mins of 0
      self.minStrides = ([0]* problemType["TotalIndices"])

    # not the ideal spot, but convert leading dims that are below the minimum size
    if problemType["OperationType"] == "GEMM":
      for i in range(0, len(self.ranges)):
        self.ranges[i].problemSizes[:] = \
          [ExactList.convertLeadingDims(self.problemType, problemSize) for problemSize in self.ranges[i].problemSizes]

    self.problems = OrderedDict()
    for sizeRange in self.ranges:
      for rangeSize in sizeRange.problemSizes:
        self.problems.update({Problem(rangeSize) : 1})
    for e in self.exacts:
        self.problems.update({e : 1})
    self.problems =  list(self.problems.keys())
    self.totalProblemSizes = len(self.problems)

    # max sizes
    self.maxD = 0
    self.maxC = 0
    self.maxA = 0
    self.maxB = 0
    for problem in self.problems:
      problemSize = problem.sizes # FIXME-problem.   This should use problem.strides*

      sizeLdd = problemSize[self.problemType["IndexAssignmentsLD"][0]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeD = max(self.minStrides[0], sizeLdd)
      for i in range(1, problemType["NumIndicesC"]):
        sizeD *= max(self.minStrides[i], problemSize[i])

      sizeLdc = problemSize[self.problemType["IndexAssignmentsLD"][1]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeC = max(self.minStrides[0], sizeLdc)
      for i in range(1, problemType["NumIndicesC"]):
        sizeC *= max(self.minStrides[i], problemSize[i])

      sizeLda = problemSize[self.problemType["IndexAssignmentsLD"][2]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsA"][0]]
      sizeA = max(self.minStrides[self.problemType["IndexAssignmentsA"][0]], sizeLda)
      for i in self.problemType["IndexAssignmentsA"][1:]:
        sizeA *= max(self.minStrides[i], problemSize[i])

      sizeLdb = problemSize[self.problemType["IndexAssignmentsLD"][3]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsB"][0]]
      sizeB = max(self.minStrides[self.problemType["IndexAssignmentsB"][0]], sizeLdb)
      for i in self.problemType["IndexAssignmentsB"][1:]:
        sizeB *= max(self.minStrides[i], problemSize[i])

      self.maxD = max(self.maxD, sizeD)
      self.maxC = max(self.maxC, sizeC)
      self.maxA = max(self.maxA, sizeA)
      self.maxB = max(self.maxB, sizeB)

  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s

################################################################################
# ProblemType
# name of solution should begin with name of problemType, and arguments can be listed out explicitly

################################################################################
# Default Problem Type
################################################################################
_defaultProblemType = {
    # =GEMM uses TransposeA,B parameters and makes the problem type more readable for users
    # =TensorContraction  requires specifying
    "OperationType": "GEMM",  # GEMM, TensorContraction, ConvolutionForward, ConvolutionBackwardData, ConvolutionBackwardWeights
    "DataType": 0,  # data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeA": 0,  # A data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeB": 0,  # B data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeE": 0,  # E data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DataTypeAmaxD": 0,  # AmaxD data type can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DestDataType": 0,  # destination data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "ComputeDataType": 0,  # compute data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "F32XdlMathOp": 0,  # reducing intermediate precision from f32 to a specific type, such as "x", as listed in SolutionStructs.py::DataType.
    # in:f32, intermediate:xf32, out:f32. f32 = xf32(f32) * xf32(f32)
    "UseBeta": True,  # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "UseE": False,  # =True use output E to output gemm results before activation
    "Gradient": False,  # =True set globalWriteElements to gradient mode
    "UseBias": 0,  # =1 support bias vector on M direction, =2 support bias vector on N direction, =3 support bias vector on both M,N direction
    "BiasSrc": "D",  # This parameter is used in gradient + bias. Support A, B, D.
    "UseScaleAB": "",  # Support "", "Scalar", and "Vector"
    "UseScaleCD": False,  # =True use scaleC, scaleD
    "UseScaleAlphaVec": 0,  # =1 support alpha vector on M direction, =2 support bias vector on N direction, =3 support alpha vector on both M,N direction
    "HighPrecisionAccumulate": False,  # f32 += f16*f16
    "SilentHighPrecisionAccumulate": False,  # Keep kernel names the same for HPA mode.  Useful for testing.
    "Sparse": 0,  # 4:2 Structured Sparse A Matrix, 0=Non Sparse, 1=Sparse Matrix A, 2=Sparse Matrix B
    "ComplexConjugateA": False,  # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB": False,
    "StochasticRounding": False,  # By default, IEEE RNE rounding
    # for OperationType == GEMM
    "TransposeA": False,  # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB": True,
    "Batched": False,  # add batching dimension
    "StridedBatched": True,  # use to select general batch or strided batch
    "GroupedGemm": False,  # use to select general batch or strided batch
    # for OperationType == TensorContraction
    # - Indices < NumIndicesC are Free or Batch indices and appear in C and D
    # - Indices which appear in both A and B, and are < NumIndicesC are batch.  A and B must have same number of batch indices.
    # - Indices which appear in both A and B, and are >= NumIndicesC are summation. A and B must have same number of summation indices.
    # - Indices which appear in A or B (but not both), are Free.  A and B may have different numbers of free indices.
    # - Summation loops are nested from smallest index number to largest, with the largest summation index as the 'unroll' loop.
    # - Memory order of C and D matrices is always 0..NumIndicesC-1, with 0 as the fastest-moving.
    #   - By choosing index assignments the output can be 'transposed'.  For example if IA=[1,2] IB=[0,2] then 0 is the coalesced dim for C/D.
    #   - Likewise batch index may be assigned between two free indices to control the output order, ie to write in CNHW format.
    #   - For example : IA=[0,1,3] IB=[2,1,3].  0,2 are free indices;  1 is batch.
    "IndexAssignmentsA": [0, 2],
    "IndexAssignmentsB": [1, 2],
    "NumIndicesC": 2,
    # use initial strides for AB.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - GlobalReadWidth must be 1 (since elements are not guaranteed to be adjacent in memory)
    "UseInitialStridesAB": False,
    # use initial strides for CD.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - Additional multiply on the store address path
    #   -VectorStore must be 0.  If VectorStore is -1, it will be silently set to 0 internally.
    "UseInitialStridesCD": False,
    "AllowNoFreeDims": False,  # allow A or B to specify no free dims
    # (if false, A and B must have at least one free dim)
    # (if true, A and B must have at least one free or batch dim)
    # SetConstStride* sets the specified stride in the problem.
    # These no longer generate predicates - see AssertStrideEqualA/B below
    # List of pairs of [index, constValue].
    # Index is a member of the global index assignments (not an offset into IndexAssignmentsA/B)
    # EX: SetConstStrideA: [ [3, 1], [2, 4] ] sets
    #     strideA for index3 to constant '1' and stride for index2 to constant '4'.
    "SetConstStrideA": [],
    "SetConstStrideB": [],
    "SetConstStrideBias": [],
    # Summation dimension indices
    "MirrorDimsA": [],
    "MirrorDimsB": [],
    "MirrorDimsMetadata": [],
    # for LD description
    "NumIndicesLD": 4,
    "IndexAssignmentsLD": [3, 4, 5, 6],  # order is LDD, LDC, LDA, LDB
    # Tile aware solution selection
    "TileAwareSelection": False,
    # Activation
    "Activation": False,
    "ActivationNoGuard": False,
    # AmaxD
    "OutputAmaxD": False,
    # For kernels putting arguments in workspaces instead of kernel arguments, they can choose to support user arguments input instead.
    "SupportUserArgs": True,
    "SwizzleTensorA": False,
    "SwizzleTensorB": False,
}

# The supported typed GEMM, each entry is (Ti, To, Tc).
# DataType (Ti)        = The data-type of the input matrices: A/B
# DestDataType (To)    = The data-type of the output matrices: C/D
# ComputeDataType (Tc) = The data-type of computation: alpha/beta:
# Cinternal: basically should == ComputeDataType
# This is used in _checkIfSupportedGEMMType()
_validGEMMTypes = [
    ("H", "H", "H"),
    ("S", "S", "S"),
    ("D", "D", "D"),
    ("C", "C", "C"),
    ("Z", "Z", "Z"),
    ("H", "H", "S"),
    ("H", "S", "S"),
    ("B", "B", "S"),
    ("B", "S", "S"),
    ("B", "H", "S"),
    ("I8", "I", "I"),
    ("4xi8", "I", "I"),
    ("I8", "I8", "I"),
    ("I8", "I", "S"),
    ("I8", "I8", "S"),
    ("I8", "H", "S"),
    ("I8", "B", "S"),
    ("F8", "S", "S"),
    ("B8", "S", "S"),
    ("F8B8", "S", "S"),
    ("B8F8", "S", "S"),
    ("F8", "H", "S"),
    ("B8", "H", "S"),
    ("F8B8", "H", "S"),
    ("B8F8", "H", "S"),
    ("B8", "B", "S"),
    ("H", "F8", "S"),
    ("F8", "B", "S"),
    ("F8B8", "B", "S"),
    ("B8F8", "B", "S"),  # in/out are both R8
    ("F8", "F8", "S"),
    ("B8", "B8", "S"),
    ("F8B8", "B8", "S"),
    ("B8F8", "B8", "S"),
    ("F8", "B8", "S"),
    ("B8", "F8", "S"),
    ("F8B8", "F8", "S"),
    ("B8F8", "F8", "S"),  # F8 NANOO
    ("F8N", "S", "S"),
    ("B8N", "S", "S"),
    ("F8B8N", "S", "S"),
    ("B8F8N", "S", "S"),
    ("F8N", "H", "S"),
    ("B8N", "H", "S"),
    ("F8B8N", "H", "S"),
    ("B8F8N", "H", "S"),
    ("B8N", "B", "S"),
    ("H", "F8N", "S"),
    ("F8N", "B", "S"),
    ("F8B8N", "B", "S"),
    ("B8F8N", "B", "S"),  # in/out are both R8
    ("F8N", "F8N", "S"),
    ("B8N", "B8N", "S"),
    ("F8B8N", "B8N", "S"),
    ("B8F8N", "B8N", "S"),
    ("F8N", "B8N", "S"),
    ("B8N", "F8N", "S"),
    ("F8B8N", "F8N", "S"),
    ("B8F8N", "F8N", "S"),
]


# All HPA types are listed here (HPA=T). The name of the library logic files for these types is:
# *_TiToTc_BH*.yaml where Ti, To, and Tc are the data types of A/B, C/D, and computation, respectively.
# The name of the library logic files for non-HPA (HPA=F) types is: *_TiB*.yaml.
_HPATypes = [
    ("H", "S", "S"),
    ("H", "H", "S"),
    ("B", "B", "S"),
    ("B", "S", "S"),
    ("B", "H", "S"),
    ("I8", "I", "I"),
    ("4xi8", "I", "I"),
    ("I8", "I", "S"),
    ("I8", "I8", "S"),
    ("I8", "H", "S"),
    ("I8", "B", "S"),
    ("F8", "S", "S"),
    ("B8", "S", "S"),
    ("F8B8", "S", "S"),
    ("B8F8", "S", "S"),
    ("F8", "H", "S"),
    ("B8", "H", "S"),
    ("F8B8", "H", "S"),
    ("B8F8", "H", "S"),
    ("H", "F8", "S"),
    ("F8", "B", "S"),
    ("F8B8", "B", "S"),  # in/out are both R8
    ("F8", "F8", "S"),
    ("B8", "B8", "S"),
    ("F8B8", "B8", "S"),
    ("B8F8", "B8", "S"),
    ("F8", "B8", "S"),
    ("B8", "F8", "S"),
    ("F8B8", "F8", "S"),
    ("B8F8", "F8", "S"),
    ("F8N", "S", "S"),
    ("B8N", "S", "S"),
    ("F8B8N", "S", "S"),
    ("B8F8N", "S", "S"),
    ("F8N", "H", "S"),
    ("B8N", "H", "S"),
    ("F8B8N", "H", "S"),
    ("B8F8N", "H", "S"),
    ("H", "F8N", "S"),
    ("F8N", "B", "S"),
    ("F8B8N", "B", "S"),  # in/out are both R8
    ("F8N", "F8N", "S"),
    ("B8N", "B8N", "S"),
    ("F8B8N", "B8N", "S"),
    ("B8F8N", "B8N", "S"),
    ("F8N", "B8N", "S"),
    ("B8N", "F8N", "S"),
    ("F8B8N", "F8N", "S"),
    ("B8F8N", "F8N", "S"),
]

def problemTypeToEnum(problemType):
  problemType["DataType"] = \
          problemType["DataType"].value
  problemType["DataTypeA"] = \
          problemType["DataTypeA"].value
  problemType["DataTypeB"] = \
          problemType["DataTypeB"].value
  problemType["DataTypeE"] = \
          problemType["DataTypeE"].value
  problemType["DataTypeAmaxD"] = \
          problemType["DataTypeAmaxD"].value
  problemType["DestDataType"] = \
          problemType["DestDataType"].value
  problemType["ComputeDataType"] = \
          problemType["ComputeDataType"].value
  problemType["BiasDataTypeList"] = \
          [btype.value for btype in problemType["BiasDataTypeList"]]
  problemType["ActivationComputeDataType"] = \
          problemType["ActivationComputeDataType"].value
  problemType["ActivationType"] = \
          problemType["ActivationType"].value
  problemType["F32XdlMathOp"] = \
      problemType["F32XdlMathOp"].value
  if "DataTypeMetadata" in problemType:
      problemType["DataTypeMetadata"] = \
          problemType["DataTypeMetadata"].value

class ProblemType(Mapping):
  ########################################

  @classmethod
  def FromDefaultConfig(printIndexAssignmentInfo: bool):
    return ProblemType(_defaultProblemType, printIndexAssignmentInfo)

  def __init__(self, config, printIndexAssignmentInfo: bool):
    self.state = {}

    for key in _defaultProblemType:
      assignParameterWithDefault(self.state, key, config, _defaultProblemType)

    # adjusting all data types
    if "DataType" in config:
      self["DataType"]  = DataType(config["DataType"])
      self["DataTypeA"] = self["DataType"]
      self["DataTypeB"] = self["DataType"]
    else:
      raise Exception("NO data type specified")
      self["DataType"]  = DataType(0)
      self["DataTypeA"] = DataType(0)
      self["DataTypeB"] = DataType(0)

    if "DataTypeA" in config:
      self["DataTypeA"] = DataType(config["DataTypeA"])

    if "DataTypeB" in config:
      self["DataTypeB"] = DataType(config["DataTypeB"])

    if "DestDataType" in config:
      self["DestDataType"] = DataType(config["DestDataType"])
    else:
      if "DataType" in config:
        self["DestDataType"] = DataType(config["DataType"])
      else:
        raise Exception("NO dest data type or data type specified")
        self["DataType"] = DataType(0)

    self["DataTypeE"] = self["DestDataType"]
    if "DataTypeE" in config:
      self["DataTypeE"] = DataType(config["DataTypeE"])

    if "ComputeDataType" in config:
      self["ComputeDataType"] = DataType(config["ComputeDataType"])
    else:
      if "DestDataType" in config:
        self["ComputeDataType"] = DataType(config["DestDataType"])
      else:
        if "DataType" in config:
          self["ComputeDataType"] = DataType(config["DataType"])
        else:
          raise Exception("NO compute data type, or dest data type, or data type specified")
          self["DataType"] = DataType(0)

    # Just like DataTypeE is DestDataType by default; DataTypeAmaxD if ComputeDataType by default.
    # So far we don't have to set it in config yamls
    self["DataTypeAmaxD"] = self["ComputeDataType"]
    if "DataTypeAmaxD" in config:
      self["DataTypeAmaxD"] = DataType(config["DataTypeAmaxD"])

    if self["Sparse"]:
      self["DataTypeMetadata"] = DataType("I8")

    if "F32XdlMathOp" in config:
        self["F32XdlMathOp"] = DataType(config["F32XdlMathOp"])
    else:
        self["F32XdlMathOp"] = DataType(DataTypeEnum.Float)

    # Modifying ComputeDataType for HHH+HPA: if (HHH+HPA), convert it to HHS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isHalf() and self["DataType"].isHalf() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == f16, DestType == f16, ComputeDataType == f16, but HPA == True (HHH+HPA, no such a type); Converting HHH+HPA to HHS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for BBB+HPA: if (BBB+HPA), convert it to BBS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isBFloat16() and self["DataType"].isBFloat16() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == bf16, DestType == bf16, ComputeDataType == bf16, but HPA == True (BBB+HPA, no such a type); Converting BBB+HPA to BBS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for I8I8I_BH: if (I8I8I8+HPA), convert it to I8I8I_BH by setting ComputeDataType to i.
    if self["ComputeDataType"].isInt8() and DataType(config["DataType"]).isInt8() and self["HighPrecisionAccumulate"]:
      print2("DataType == i8 and HPA == True; setting compute data type to int32")
      self["ComputeDataType"] = DataType('i')

    if self["OperationType"] == "GEMM":
      self._checkIfSupportedGEMMType()
      self.initGEMM()
    else:
      raise Exception("Unsupported OperationType = %s" % self["OperationType"])

    self.state["AssignedDerivedParameters"] = False
    ProblemType.assignDerivedParameters(self.state, printIndexAssignmentInfo)

    for tc in ('A', 'B'):
      for sc in self["SetConstStride%s"%tc] :
          (anchorDim, stride) = sc[:2]
          if anchorDim not in self.state["IndexAssignments%s"%tc]:
              raise Exception("SetConstStride%s=%s anchorDim=%u is not in IndexAssignments%s"%(tc, sc, anchorDim, tc))

    # Bias
    # If compute data type is not equal to dest data type, tensile will run conversion kernel.
    # In this case we don't need to apply bias in beta only kernel.
    if "UseBias" in config:
      if self["ComputeDataType"] != self["DestDataType"]:
        self["BetaOnlyUseBias"] = False
      else:
        self["BetaOnlyUseBias"] = True if self["UseBias"] > 0 else False
      if "BiasDataTypeList" in config:
        self["BiasDataTypeList"] = [DataType(btype) for btype in config["BiasDataTypeList"]]
        self["BiasDataTypeList"].sort() # Make name unique
      else:
        self["BiasDataTypeList"] = getBiasDataTypeListDefault(self)
    else:
      self["BetaOnlyUseBias"] = False
      self["BiasDataTypeList"] = []

    # Activation
    # Currently, ActivationType supports only 'all' and 'hipblaslt_all', and is active only when the Activation configuration is set to True.
    # Otherwise, ActivationType will be set to 'none'.
    if "Activation" in config:
      typeStr = config.get("ActivationType", 'none')
      if typeStr not in ['all', 'hipblaslt_all']:
        typeStr = 'none'
    else:
      typeStr = 'none'
    self["ActivationType"] = ActivationType(typeStr)
    if "ActivationComputeDataType" in config:
      self["ActivationComputeDataType"] = DataType(config["ActivationComputeDataType"])
    else:
      self["ActivationComputeDataType"] = self["ComputeDataType"]

    if self["ActivationType"] != 'none':
      # This is a dummy guard in case we currently don't have a converter to convert data from compute type to activation compute type
      if self["ActivationComputeDataType"] not in [self["ComputeDataType"], self["DestDataType"]]:
        printWarning("TensileLite currently only supports ActivationComputeDataType (%s) = ComputeDataType (%s) or DestDataType (%s). \
                      ActivationComputeDataType will be set to ComputeDataType automatically."%(self["ActivationComputeDataType"].toChar(), \
                                                                                                self["ComputeDataType"], \
                                                                                                self["DestDataType"]))
        self["ActivationComputeDataType"] = self["ComputeDataType"]
      if (self["ActivationComputeDataType"].numRegisters() != self["ComputeDataType"].numRegisters()) and \
        (self["DataType"].numRegisters() < self["DestDataType"].numRegisters()):
        printWarning("TensileLite only supports ActivationComputeDataType = ComputeDataType if DestDataType > DataType. \
                      ActivationComputeDataType will be set to ComputeDataType automatically.")
        self["ActivationComputeDataType"] = self["ComputeDataType"]

    if "UseE" in config:
      if config["UseE"]:
        if self["ActivationType"] == 'none':
          printWarning("Use E is disabled cause Activation is set to False.")
          self["UseE"] = False
        else:
          self["UseE"] = config["UseE"]
      else:
        self["UseE"] = config["UseE"]

    if "Gradient" in config:
      if config["Gradient"]:
        if (not self["UseBias"]) and self["ActivationType"] == 'none':
          printWarning("Gradient is disabled cause bias and activation are both disabled.")
          self["Gradient"] = False
        if self["ActivationType"] != 'none' and self["UseE"] == False:
          printWarning("Use E is enabled cause Activation is enabled.")
          self["UseE"] = True
        elif self["ActivationType"] != 'none' and self["UseE"] == False:
          printWarning("Use E is disabled cause Activation is disabled.")
          self["UseE"] = False
        # if self["UseScaleAlphaVec"]:
        #   printWarning("Use scaleAlphaVec is disabled cause Gradient is enabled.")
        #   self["UseScaleAlphaVec"] = False
      self["Gradient"] = config["Gradient"]

    # Need gradient info
    biasSrcList = ["A", "B", "D"]
    if "BiasSrc" in config:
      if not self["Gradient"] and config["BiasSrc"] != "D":
        printWarning("BiasSrc is set to D cause Gradient is disabled.")
        self["BiasSrc"] = "D"
      elif self["Gradient"]:
        # # Currently only supports D :)
        # if config["BiasSrc"] != "D":
        #   raise Exception("BiasSrc currently only supports D.")
        if config["BiasSrc"] not in biasSrcList:
          raise Exception("BiasSrc only supports A, B, D.")

    if "ActivationNoGuard" in config:
      self["ActivationNoGuard"] = config["ActivationNoGuard"]
      if self["ActivationNoGuard"]:
        if self["ActivationType"] == 'none':
          printWarning("ActivationNoGuard is set to False cause Acivation is off.")
          self["ActivationNoGuard"] = False
        if (not self["Gradient"]):
          printWarning("ActivationNoGuard is set to False cause Gradient is off.")
          self["ActivationNoGuard"] = False

  ################################################################################
   # Function checkIfSupportedGEMMType:
  #   Assures 3 data-types are valid, supported and well-assigned
  #   See the discussion in ValidParameters.py for validGEMMTypes
  ################################################################################
  def _checkIfSupportedGEMMType(self):
    inType = self["DataType"]
    outType = self["DestDataType"]
    computeType = self["ComputeDataType"]

    gemmType = ( inType.toChar(), outType.toChar(), computeType.toChar() )
    if gemmType not in _validGEMMTypes:
      raise Exception("This typed-GEMM (Ti, To, Tc) = (%s, %s, %s) is not supported yet."%(gemmType[0], gemmType[1], gemmType[2]))

  ########################################
  def initGEMM(self):
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self.state["Sparse"] == 2:
      self["IndexAssignmentsMetadata"] = [sumIdx, 1] # N (ref B)
    else:
      self["IndexAssignmentsMetadata"] = [sumIdx, 0] # T (ref A)
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["TransposeB"]:
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["IndexAssignmentsMetadata"].append(2)
      self["NumIndicesC"] = 3
    else:
      self["NumIndicesC"] = 2

    self["NumIndicesLD"] = 4
    self["IndexAssignmentsLD"][0] = self["NumIndicesC"] + 1
    for i in range(1, len(self["IndexAssignmentsLD"])):
      self["IndexAssignmentsLD"][i] = self["IndexAssignmentsLD"][i-1] + 1

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  # determine d0, d1, dU
  @staticmethod
  def assignDerivedParameters(state, printIndexAssignmentInfo: bool=False):
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    state["TotalIndices"] = max(max(state["IndexAssignmentsA"])+1, \
        max(state["IndexAssignmentsB"])+1)

    # determine num free, batch
    state["IndicesFree"] = []
    state["IndicesBatch"] = []
    state["IndicesSummation"] = []

    for i in range(0, state["NumIndicesC"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesBatch"].append(i)

      elif inA or inB:
        state["IndicesFree"].append(i)
      else:
        raise Exception("invalid index %u (inC but not (inA or inB))" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesSummation"].append(i)
      else:
        raise Exception("invalid index %u (expected summation but not (inA and inB))" % i)
    # print index assignments
    if printIndexAssignmentInfo:
      print("IndicesFree:  %s" % state["IndicesFree"])
      print("IndicesBatch: %s" % state["IndicesBatch"])
      print("IndicesSum:   %s" % state["IndicesSummation"])
      print("IndexAssignmentsA:   %s" % state["IndexAssignmentsA"])
      print("IndexAssignmentsB:   %s" % state["IndexAssignmentsB"])
      print("NumIndicesC:  %s" % state["NumIndicesC"])

    for k in ('IndexAssignmentsA','IndexAssignmentsB'):
      if len(state[k]) != len(set(state[k])):
        raise Exception("duplicate index in %s=%s"% (k,state[k]))

    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if not state["AllowNoFreeDims"] and state["NumIndicesFree"] < 2 :
      raise Exception("Tensile requires >= 2 free indices or set AllowNoFreeDims; FreeIndices=%s."% state["IndicesFree"])

    # by default, unroll index will be the last/inner summation index
    state["IndexUnroll"] = state["IndicesSummation"][len(state["IndicesSummation"])-1]
    for i in range(0, len(state["IndexAssignmentsA"])):
      if state["IndexAssignmentsA"][i] == state["IndexUnroll"]:
        state["IndexUnrollA"] = i
        break
    for i in range(0, len(state["IndexAssignmentsB"])):
      if state["IndexAssignmentsB"][i] == state["IndexUnroll"]:
        state["IndexUnrollB"] = i
        break
    for i in range(0, len(state["IndexAssignmentsMetadata"])):
      if state["IndexAssignmentsMetadata"][i] == state["IndexUnroll"]:
        state["IndexUnrollM"] = i
        break
    #print2("IndexUnrollA: %u" % state["IndexUnrollA"])
    #print2("IndexUnrollB: %u" % state["IndexUnrollB"])

    # assign d0, d1
    if state["AllowNoFreeDims"]:
      dimList = state["IndicesFree"] + state["IndicesBatch"]
    else:
      dimList = state["IndicesFree"]
    state["Index01A"] = [i for i in state["IndexAssignmentsA"] if i in dimList][0]
    state["Index01B"] = [i for i in state["IndexAssignmentsB"] if i in dimList][0]
    #print2("Index01A: %u" % state["Index01A"])
    #print2("Index01B: %u" % state["Index01B"])
    # Store code is optimized for 0 as the fastest-moving in memory
    # whichever has lower stride in C (lower value), is 0, other is 1
    if state["Index01A"] < state["Index01B"]:
      state["Index0"]  = state["Index01A"]
      state["Index1"]  = state["Index01B"]
      state["Tensor0"] = 0
      state["Tensor1"] = 1
      state["TileA"] = 0
      state["TileB"] = 1
    else:
      state["Index0"]  = state["Index01B"]
      state["Index1"]  = state["Index01A"]
      state["Tensor0"] = 1
      state["Tensor1"] = 0
      state["TileA"] = 1
      state["TileB"] = 0

    # generalize transpose
    strideIdxA = state["IndexAssignmentsA"].index(state["Index01A"])
    strideIdxB = state["IndexAssignmentsB"].index(state["Index01B"])
    unrollIdxA = state["IndexAssignmentsA"].index(state["IndexUnroll"])
    unrollIdxB = state["IndexAssignmentsB"].index(state["IndexUnroll"])
    state["TLUA"] = strideIdxA < unrollIdxA
    state["TLUB"] = strideIdxB < unrollIdxB
    #state["TLUB"] = True # hack

    if printIndexAssignmentInfo:
      print("TLUA:  %s (stridePosA(%d) <? unrollIdxA(%d)" % \
			(state["TLUA"], strideIdxA, unrollIdxA))
      print("TLUB:  %s (stridePosB(%d) <? unrollIdxB(%d)" % \
	  		(state["TLUB"], strideIdxB, unrollIdxB))
      print("Index01A:  %s" % state["Index01A"])
      print("Index01B:  %s" % state["Index01B"])
    #unrollDimStrideGreaterThanTileDimStrideA = TLUA = !transA = fast
    #!unrollDimStrideLessThanTileDimStrideB   = TLUB =  transB = fast
    state["AssignedDerivedParameters"] = True

    if state["Sparse"]:
      state["Index01Metadata"] = [i for i in state["IndexAssignmentsMetadata"] if i in dimList][0]
      strideIdxM = state["IndexAssignmentsMetadata"].index(state["Index01Metadata"])
      unrollIdxM = state["IndexAssignmentsMetadata"].index(state["IndexUnroll"])
      state["TLUMetadata"] = strideIdxM < unrollIdxM
      if printIndexAssignmentInfo:
        print("TLUMetadata:  %s (stridePosM(%d) <? unrollIdxM(%d)" % \
          (state["TLUMetadata"], strideIdxM, unrollIdxM))
        print("Index01Metadata:  %s" % state["Index01Metadata"])

  ########################################
  def __str__(self):
    indexChars = INDEX_CHARS
    # C dimensions
    name = ["C" + "".join(indexChars[i].lower() for i in range(0, self["NumIndicesC"]))]

    # A dimensions
    name.append("A" + "".join(indexChars[i] if i in self["MirrorDimsA"] else indexChars[i].lower() for i in self["IndexAssignmentsA"]))
    if self["ComplexConjugateA"]:
      name.append("C")

    # B dimensions
    name.append("B" + "".join(indexChars[i] if i in self["MirrorDimsB"] else indexChars[i].lower() for i in self["IndexAssignmentsB"]))
    if self["ComplexConjugateB"]:
      name.append("C")

    # DataTypes
    if self["DataType"] != self["DataTypeA"] or self["DataType"] != self["DataTypeB"]:
      name.append(self["DataTypeA"].toChar() + self["DataTypeB"].toChar())
    name.append(self["DataType"].toChar()) # Type of A/B

    # Special condition for some newly supported kernels:
    #   HHS, HSS, BSS and I8II kernels, use a clearer naming _TiToTc_
    # TODO: Distinguish all kernels by _TiToTc_ to be more consistent with rocblas
    gemmType = (self["DataType"].toChar(),self["DestDataType"].toChar(),self["ComputeDataType"].toChar() )
    if gemmType in _HPATypes:
      name[-1] += "".join([self["DestDataType"].toChar(), self["ComputeDataType"].toChar()])

    if not self["F32XdlMathOp"].isSingle() and self["DataType"].isSingle():
      name.append("".join(["M", self["F32XdlMathOp"].toChar()]))

    if self["SwizzleTensorA"]:
      name.append("STA")

    if self["SwizzleTensorB"]:
      name.append("STB")

    # Other
    other = ""
    if self["UseBeta"]: other += "B"
    if self["HighPrecisionAccumulate"] and not self["SilentHighPrecisionAccumulate"]: other += "H"
    if self["UseInitialStridesAB"]: other += "I"
    if self["UseInitialStridesCD"]: other += "Ic"
    if other: name.append(other)

    if self["UseBias"]:
      name.append("Bias")
      if self["BiasDataTypeList"] != getBiasDataTypeListDefault(self):
        name.append("".join(i.toChar() for i in self["BiasDataTypeList"]))
      if self["BiasSrc"] and self["Gradient"]: # Show bias src if gradient = True
        name.append(f"{self['BiasSrc']}")

    factorDim = max(self["UseScaleAlphaVec"], self["UseBias"])
    if factorDim > 1 :
        s = ("N" if factorDim == 2 else "MN")
        name.append(f"FD{s}")

    if self["UseE"]:
      if self["Gradient"]:
        name.append(f"Grad{self['DataTypeE'].toChar()}")
      else:
        name.append(f"Aux{self['DataTypeE'].toChar()}")
    if self["OutputAmaxD"]:
      name.append("AmaxD")
    if self["Sparse"]:
      if self["Sparse"] == 2:
        name.append("SPB")
      else:
        name.append("SPA")

    # precision and other
    # name += "_SB" if self["StridedBatched"] else "_GB"
    if self["GroupedGemm"]:
      name.append("GG")
    else:
      if not self["StridedBatched"]:
        name.append("GB") # legacy

    # Activation Naming
    if self["ActivationType"] != 'none':
      if self["ActivationType"] == 'all':
        name.append("A")
      elif self["ActivationType"] == 'hipblaslt_all':
        name.append("HA")
      else:
        name.append(f"{str(self['ActivationType']).upper()}")
      name.append(self["ActivationComputeDataType"].toChar())
    if self["ActivationNoGuard"]: name[-1] += "NG"

    if self["UseScaleAB"] == "Scalar":
      name.append("SAB")
    elif self["UseScaleAB"] == "Vector":
      name.append("SABV")
    if self["UseScaleCD"]: name.append("SCD")
    if self["UseScaleAlphaVec"]: name.append("SAV")

    if self["SupportUserArgs"]: name.append("UserArgs")

    return "_".join(name)

  def keys(self):
    return list(self.state.keys())
  def __len__(self):
    return len(self.state)
  def __iter__(self):
    return iter(self.state)
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(str(self))
  def __eq__(self, other):
    return isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

  def get(self, key, default=None):
    try:
      return self.state[key]
    except:
      return default

################################################################################
# Bias Type
################################################################################

def getBiasDataTypeListDefault(problem: ProblemType) -> List[DataType]:
  bList = []
  for d in ["DataType", "ComputeDataType", "DestDataType"]:
    dtype = DataType(problem[d])
    # filter out int8/f8/b8, because it is not supported by bias datatype
    # TODO
    if not dtype.isInt8() and not dtype.is8bitFloat():
      bList.append(dtype)

  biasDataTypeList = list(set(bList))
  biasDataTypeList.sort() # Make name unique
  return biasDataTypeList
