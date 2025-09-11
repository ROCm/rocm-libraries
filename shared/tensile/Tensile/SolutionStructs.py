################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import assignParameterRequired, assignParameterWithDefault, \
                    defaultProblemType, defaultSolution, \
                    globalParameters, \
                    tPrint, printExit, printWarning, \
                    validActivationFormats, validConvolutionConfig, \
                    validMFMA, validWMMA, validParameters, validWeightFormats, \
                    validGEMMTypes, HPATypes, parameterNameAbbreviations
from .DataType import DataType
from .Utils import roundUpToNearestMultiple

from .KernelWriterStreamKInit import KernelWriterStreamKInit
from .KernelWriterBetaOnly import KernelWriterBetaOnly
from .KernelWriterConversion import KernelWriterConversion

from .CustomKernels import isCustomKernelConfig

from collections import namedtuple,OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from functools import reduce

import collections
import math
import operator
import sys

########################################
# Print a reject message :
def reject(state, *args):
  if state and "NoReject" in state and state["NoReject"]:
    return

  if globalParameters["PrintSolutionRejectionReason"]:
    sys.stdout.write("\nreject: ")
    for a in args:
      print(a)
    #traceback.print_stack(None, 2)
    solutionIndex = state["SolutionIndex"] if (state != None and "SolutionIndex" in state) else -1
    if solutionIndex != -1:
      # If we have valid solutionIndex, this means we are during TensileCreateLibrary stage
      # In this stage, all solutions in the logic should be valid
      # So if any rejection happens, print the warning for further check
      # This will be done only when --global-parameters=PrintSolutionRejectionReason=True
      solutionNameMin = state["SolutionNameMin"] if ("SolutionNameMin" in state) else None
      # if we don't have SolutionNameMin, we simply use the problemTypeName
      solutionNameMin = str(state["ProblemType"]) if (solutionNameMin == None) else solutionNameMin
      print("!! Warning: Any rejection of a LibraryLogic is not expected, please check. \
        SolutionIndex: %d (or SolutionName/ProblemType: %s)"%(solutionIndex, solutionNameMin))
  if state != None:
    state["Valid"] = False

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(math.ceil(float(dividend) / float(divisor)))

class Fbs(Enum):
  Free=0     # Expect to be free dimension
  Batch=1    # Expect to be batch dimension
  Sum=2      # Expect to be summation dimension

################################################################################
RegDim=namedtuple("RegDim", ["idx", "fbs", "dim"])

class ConvolutionConfig:
  def __init__(self, fil=None, stride=None, dilation=None, \
                  spatial=None, groupCount=1, \
                  padStart=None, padEnd=None):
    self.fil = fil
    self.stride = stride
    self.dilation = dilation
    self.spatial = spatial
    self.groupCount = groupCount
    self.padStart = padStart
    self.padEnd = padEnd

  @staticmethod
  def copyField(tag, selfValues, refValues):
    """ Use selfValues if specified (after validating they match cc), or conv config values if not"""
    if selfValues:
      if refValues:
          assert(len(selfValues) == len(refValues))
          for (i,(selfVal, refVal)) in enumerate(zip(selfValues, refValues)):
            if selfVal == -1:
              selfValues[i] = refVal
            if selfVal != -1 and refVal != -1 and refVal != selfVal:
              raise RuntimeError("Mismatch between ConvolutionConfig value (%d) and ConvProblem value (%d) for %s[%d]." %
                        (refVal, selfVal, tag, i))
      return selfValues
    else:
      return refValues

  def copyFromRef(self, ref):
    """
    For all fields which are -1 in self, copy from reference implementation.
    For any fields that are specified in self (not -1), ensure they match reference
    """
    self.fil = self.copyField("filter", self.fil, ref.fil)
    self.stride = self.copyField("stride", self.stride, ref.stride)
    self.dilation = self.copyField("dilation", self.dilation, ref.dilation)
    self.spatial = self.copyField("spatial", self.spatial, ref.spatial)
    self.padStart = self.copyField("padStart", self.padStart, ref.padStart)
    self.padEnd = self.copyField("padEnd", self.padEnd, ref.padEnd)

    if self.groupCount == -1:
      self.groupCount = ref.groupCount

  def checkFullySpecified(self, ref):
    """
    Throw exception if the config is not fully specified.
    """
    for field in ('fil', 'stride', 'dilation', 'spatial', 'groupCount', 'padStart', 'padEnd'):
        val = getattr(self,field)
        if val==None:
          raise RuntimeError("ConvolutionConfig field '%s' == None and must be specified'" % field)
        elif isinstance(val,int) and val==-1 or type(val) in (tuple,list) and -1 in val:
          raise RuntimeError("ConvolutionConfig field '%s' == %s must be fully specified.'" % (field,val))


  def __str__ (self):
      return("filter:%s stride:%s dilation:%s spatial:%s group:%d padStart:%s padEnd:%s" \
            % (str(self.fil) if self.fil else "tbd",
               self.stride, self.dilation, \
               str(self.spatial) if self.spatial else "tbd", \
               self.groupCount, self.padStart, self.padEnd))

class Convolution:
  class Dimension:
    """
    A description of the dimension - short char, usage, and const strides
    Dimensions are later assigned tensile indices and assigned to A/B
    based on the desired formats.
    """
    # stride=-1 indicates TBD stride; >=0 indicates a compile-time constant
    def __init__(self, shortChar, description, size=-1, strideA=-1, strideB=-1):
      self.shortChar = shortChar
      self.description = description
      self.size=size
      self.strideA=strideA
      self.strideB=strideB

    def __str__(self):
      s = "%5s : %s" % ("'%s'"%self.shortChar, self.description)
      if self.size != -1:
        s+=" [size:%d]"%self.size
      if self.strideA != -1:
        s+=" [strideA:%d]"%self.strideA
      if self.strideB != -1:
        s+=" [strideB:%d]"%self.strideB
      return s
    def __repr__(self):
      return self.shortChar

  SummaryProblemProperties=[\
        'OperationType','DestDataType','DataType','HighPrecisionAccumulate',\
        'TensorAFormat','TensorBFormat','TensorDFormat',\
        'Filter', 'Stride','Dilation','PadStart','PadEnd','GroupCount',\
        'NumIndicesC', 'IndexAssignmentsA','IndexAssignmentsB',\
        'IndicesFree', 'IndicesBatch', 'IndicesSummation',\
        'SetConstStrideA', 'SetConstStrideB', 'ZeroPadA', \
        'UseBeta', 'UseInitialStridesAB', "AllowNoFreeDims", \
        ]
  SummarySolutionProperties=[\
        'AssertSizeEqual', 'AssertStrideAEqual', 'AssertStrideBEqual',\
         'AssertSizeGreaterThan', 'AssertSizeLessThan', "AssertSizeMultiple"
        ]

  # valid lowest filter dimensions, these we can attach compile-time constant strides:
  ValidLowestFilterDim= ('X','XY', 'XYZ', 'W', 'HW', 'DHW')

  def initForwardConvolution(self, problemTypeOut, config, \
                             formatA, formatB, formatD,
                             ndim, cdim, kdim, sdims, fdims, \
                             fil, stride, dilation):
    """
    Output : registerA and registerB
    """
    # Make index assignments following standard Tensile Index assignment rules (see Common.py)
    # - Indices < NumCindices are batch or free indices and are present in TensorD
    # - Indices >= NumCindices are summation indices.  cidx is cin / summation so must be after nidx
    # - Memory order for TensorD is NumCindices...0, with 0 the fastest-moving dim.

    # The index assignment is captured in the RegDim.idx parm. This idx is in global space.

    # Specific assignments to A and B (and associated impact on memory order of those tensors) is
    # specified by order of parms to registerA and registerB below.

    # Control output space dimension order:S
    # Note:
    #   - 'C' in output format refers to output channels, ie 'K'
    #   - backward-weight formats swap (C,K)
    #   - fastest moving in format is rightmost and should have idx=0, then
    #     increase index moving left through the format.
    #print ("formatA=", formatA, "formatB=", formatB, "formatD=", formatD)

    self.normalizedCC = ConvolutionConfig(fil=fil, stride=stride, dilation=dilation, \
                                          spatial=None)
    if formatD in ('NCHW','NCDHW'):
      sidx = 0
      i = len(sdims)
      kidx = i ; i+=1
      nidx = i ; i+=1
      sumIdx = i
    elif formatD in ('NHWC','NDHWC'):
      i = 0
      kidx = i ; i+=1
      sidx = i ; i+=len(sdims)
      nidx = i ; i+=1
      sumIdx = i
    elif formatD in ("CNHW", "CNDHW"):
      sidx = 0
      # re-order batch dim to control memory order in output space
      i = len(sdims)
      nidx = i ; i+=1
      kidx = i ; i+=1
      sumIdx = i
    elif formatD in ("CHWN", "CDHWN"):
      i = 0
      nidx = i ; i+=1
      sidx = i ; i+=len(sdims)
      kidx = i ; i+=1
      sumIdx = i
    else:
      raise RuntimeError ("unknown formatD '%s'"%formatD)

    if not self.unrollOnChannel:
      # place cidx at lowest summation then filters -> filters are unroll
      cidx = sumIdx
      sumIdx = sumIdx+1

    self.filterRegDims = []
    for filterDim in fdims:
      self.filterRegDims.append( RegDim(sumIdx, Fbs.Sum, filterDim) )
      sumIdx = sumIdx+1


    if self.unrollOnChannel:
      # place cidx at highest summation, after filters
      cidx = sumIdx

    self.spatialRegDims = []
    # reverse dims  so can pass spatialRegDims to register functions in 'convolution' order
    for si,sdim in enumerate(sdims):
      self.spatialRegDims.insert(0, RegDim(sidx+si, Fbs.Free, sdim))

    chinRegDim = [RegDim(cidx,Fbs.Sum,cdim)]
    choutRegDim= [RegDim(kidx,Fbs.Free,kdim)]

    if formatA in ("NCHW", "NCDHW"):
      self.registerA( [RegDim(nidx,Fbs.Batch,ndim)] + chinRegDim + self.spatialRegDims + self.filterRegDims )
    elif formatA in ("NHWC", "NDHWC"):
      self.registerA( [RegDim(nidx,Fbs.Batch,ndim)] + self.spatialRegDims + self.filterRegDims + chinRegDim )
    elif formatA in ("CNHW", "CNDHW"):
      self.registerA( chinRegDim + [RegDim(nidx,Fbs.Batch,ndim)] + self.spatialRegDims + self.filterRegDims )
    elif formatA in ("CHWN", "CDHWN"):
      self.registerA( chinRegDim + self.spatialRegDims + self.filterRegDims + [RegDim(nidx,Fbs.Batch,ndim)] )
    else:
      raise RuntimeError ("unknown formatA '%s'"%formatA)

    ndim.strideB = 0
    if formatB in ("KCYX",'KCZYX') :
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + choutRegDim + chinRegDim + self.filterRegDims )
    elif formatB in ("CKYX",'CKZYX'):
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + chinRegDim + choutRegDim + self.filterRegDims )
    elif formatB in ("CYXK",'CZYXK'):
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + chinRegDim + self.filterRegDims + choutRegDim )
    elif formatB in ("KYXC",'KZYXC'):
      self.registerB( [RegDim(nidx,Fbs.Batch,ndim)] + choutRegDim + self.filterRegDims + chinRegDim )
    else:
      raise RuntimeError ("unknown formatB '%s'"%formatB)

    problemTypeOut["NumIndicesC"] = 2+len(self.spatialRegDims)

    problemTypeOut["ZeroPadA"] = self.makeZeroPadConvProblemType(self.cc.padStart, self.cc.padEnd)

    # Attach constant strides to A, if possible:
    nonFilterDims = [dim for dim in self.regDimsA if dim not in self.filterRegDims]
    setStride=False
    if sdims and nonFilterDims[-1].dim == sdims[0]:
      setStride = True
      sdims[0].strideA = self.cc.stride[0]
    elif nonFilterDims[-1].dim==cdim:
      setStride = True
      cdim.strideA = 1
    elif nonFilterDims[-1].dim==ndim:
      setStride = True
      ndim.strideA = 1

    if self.filterRegDims and self.regDimsA[-1].dim == self.filterRegDims[-1].dim:
      if self.filterRegDims[-1].dim.shortChar in self.ValidLowestFilterDim:
        self.filterRegDims[-1].dim.strideA = self.cc.dilation[0]
    elif not setStride:
      raise RuntimeError ("unexpected lowest dimension in tensorAFormat(%s)"%self.tensorAFormat)

    # Attach constant strides to B, if possible:
    if self.regDimsB[-1].dim == cdim:
      cdim.strideB=1
    elif self.regDimsB[-1].dim == kdim:
      kdim.strideB=1
    elif self.filterRegDims and self.regDimsB[-1].dim == self.filterRegDims[-1].dim:
      if self.filterRegDims[-1].dim.shortChar in self.ValidLowestFilterDim:
        self.filterRegDims[-1].dim.strideB = 1
    else:
      raise RuntimeError ("unexpected lowest dimension in tensorBFormat(%s)"%self.tensorAFormat)

  @staticmethod
  def swap(targetStr, replStr1, replStr2):
    tmp = '$'
    assert(tmp not in targetStr)
    for (char1,char2) in zip(replStr1,replStr2):
      if char1==',':
        continue
      targetStr = targetStr.replace(char1, tmp).replace(char2, char1).replace(tmp, char2)
    return targetStr

  def makeZeroPadConvProblemType(self, padStart, padEnd):
    """
    Convert padStart/padEnd into the format expected by ProblemType ZeroPad*
    Tensile drops any compile-time padding info here; this must be provided with each problem.
    """
    rv = []
    spatialChars='WHD'
    filterChars='XYZ'
    for i in range(self.numSpatialDims):
      if padStart[i] or padEnd[i]:
          anchorIdx = self.convolutionDims[spatialChars[i]].idx
          sumIdx    = self.convolutionDims[filterChars[i]].idx
          rv.append([anchorIdx, sumIdx, -1, -1])
    return rv

  def makeZeroPadProblemType(self, zps, padStart, padEnd, c, cc):
    """ Convert padStart/padEnd into the format expected by ProblemType ZeroPad* """
    rv = []
    ss = c if self.regDimsA[0].dim.shortChar == 'C' else 1

    for (i,zp) in enumerate(zps):
      (anchorIdx, sumIdx) = zp[:2]
      rv.append([anchorIdx, sumIdx, padStart[i]*ss, padEnd[i]*ss])
      assert(cc.spatial[i] != -1)
      ss *= cc.spatial[i]
    return rv

  def __init__(self, problemTypeOut, convolutionType, config):
    """
    problemTypeOut contains problem type parms created by this constructor.
    """

    self.convolutionDims={};
    self.convolutionType = convolutionType
    self.config = config # input configuration
    self.problemTypeOut = problemTypeOut
    self.cc = ConvolutionConfig() # parsed configuration

    for k in config:
      if k not in validConvolutionConfig:
        raise RuntimeError ("unknown convolution config field '%s'"%k)

    self.tensorAFormat = config.get("TensorAFormat", "NCHW")
    assert self.tensorAFormat in validActivationFormats
    self.formatNumSpatialDims = len(self.tensorAFormat)-2
    assert (self.formatNumSpatialDims>=2 and self.formatNumSpatialDims<=3)

    if convolutionType in ('ConvolutionForward', 'ConvolutionBackwardData'):
      defaultFormatB = "KCYX" if self.formatNumSpatialDims==2 else 'KCZYX'
      defaultFormatD = self.tensorAFormat
    elif convolutionType == 'ConvolutionBackwardWeights':
      defaultFormatB = "NCHW" if self.formatNumSpatialDims==2 else 'NCDHW'
      defaultFormatD = "KCYX" if self.formatNumSpatialDims==2 else 'KCZYX'

    self.tensorBFormat = config.get("TensorBFormat", defaultFormatB)
    self.tensorDFormat = config.get("TensorDFormat", defaultFormatD)

    if convolutionType in ('ConvolutionForward', 'ConvolutionBackwardData'):
      assert self.tensorBFormat in validWeightFormats
      assert self.tensorDFormat in validActivationFormats
    elif convolutionType in ('ConvolutionBackwardWeights'):
      assert self.tensorBFormat in validActivationFormats
      assert self.tensorDFormat in validWeightFormats

    if self.tensorDFormat == 0:
      self.tensorDFormat = self.tensorAFormat
    assert len(self.tensorAFormat) == len(self.tensorBFormat) == len(self.tensorDFormat)

    # index 0,1,2 = W,H,D = X,Y,Z
    if config.get("Spatial",None):
      self.cc.spatial  = self.dimxParm(config, "Spatial",-1)
    else:
      self.cc.spatial = None
    self.cc.fil   = self.dimxParm(config, "Filter",1)
    self.cc.stride   = self.dimxParm(config, "Stride",1)
    self.cc.dilation = self.dimxParm(config, "Dilation",1)
    self.cc.padStart = self.dimxParm(config, "PadStart",0)
    self.cc.padEnd   = self.dimxParm(config, "PadEnd",0)
    self.packedSpatialDims = config.get("PackedSpatialDims", 1)
    self.packedFilterDims  = config.get("PackedFilterDims", 1)
    self.unrollOnChannel = config.get("UnrollOnChannel", 1)

    assert(type(self.packedFilterDims) == int)
    assert(type(self.packedSpatialDims) == int)
    assert(type(self.unrollOnChannel) == int)
    if not all(i==1 for i in self.cc.dilation[1:]) and not all (i==1 for i in self.cc.fil) :
      self.packedFilterDims = 0
    if not (\
       all(i==1 for i in self.cc.stride[1:]) and \
       all(i==0 for i in self.cc.padStart) and \
       all(i==0 for i in self.cc.padEnd) \
       ):
      self.packedSpatialDims = 0

    assert (len(self.cc.fil)==len(self.cc.stride)==len(self.cc.dilation) \
            ==len(self.cc.padStart)==len(self.cc.padEnd))

    self.groupCount = config.get("GroupCount", 1)
    self.indexAssignments = []

    # Index assignment have fastest-moving first
    ndim = Convolution.Dimension('N',   'Minibatch dimension. size#T=N.')
    kdim = Convolution.Dimension('K',   'Cout. size#T=Cout.')
    cdim = Convolution.Dimension('C', 'Cin.  size#T=Cin.')

    if self.packedSpatialDims:
      if self.formatNumSpatialDims==2:
        sdims = [Convolution.Dimension('HW', \
            'Spatially packed HW. size#T=H_o*W_o. strideA#T=strideW(#S0).')]
      elif self.formatNumSpatialDims==3:
        sdims = [Convolution.Dimension('DHW', \
            'Spatially packed DHW. size#T=D_o*H_o*W_o. strideA#T=strideW(#S0).')]
      else:
        raise RuntimeError ("unsupported formatNumSpatialDims")
    else:
      sdims = []
      schars = [1,'W','H','D']
      # sdims[0] is W
      for si in range(self.formatNumSpatialDims):
        sc=schars[si+1]
        if si==0:
            strideMsg = "stride%s(#S0)"%sc
        else:
            strideMsg = "%s_in*stride%s(#S%d)"%(schars[si],sc,si)
        sdims.append(Convolution.Dimension(sc,  \
            'Spatial %s. size#T=%s_o strideA#T=%s.'%(sc,sc,strideMsg)))

    # dims actually used in the tensor.
    self.numSpatialDims = len(sdims)
    if self.packedSpatialDims:
      assert (self.numSpatialDims <= self.formatNumSpatialDims)
    else:
      assert (self.numSpatialDims == self.formatNumSpatialDims)

    fdims = []
    for (rfi,filterValue) in enumerate(self.cc.fil[::-1]):
      if not self.packedFilterDims or filterValue != 1:
        fi = self.formatNumSpatialDims - rfi - 1 # forward filter index, 0...
        filterChar = chr(ord('X')+fi)
        filterValueStr = "TBD" if filterValue==-1 else str(filterValue)
        prevChar = ['1', 'W', 'W*H']
        # TODO - stride setconst maybe applies only for NCHW/CNHW format not NHWC
        # can modify message here based on format or position of indices?
        filterMsg = "Filter%s. size#T=Filter%s(%s). strideA#T=Dilation%s(#D%d)*%s." \
            % (filterChar, filterChar, filterValueStr, filterChar, fi, \
               prevChar[fi])
        fdims.append(Convolution.Dimension(filterChar, filterMsg, size=filterValue))

    # Create summation dimensions for non-unit filters and assign summation indices
    assert(len(self.cc.fil)) == self.formatNumSpatialDims

    # Output format: C->K -> doesn't matter
    # what about if both C and K present in output (ie weights)  CK
    if convolutionType in ("ConvolutionForward"):
      self.initForwardConvolution(problemTypeOut, config, \
                                self.tensorAFormat, self.tensorBFormat, self.tensorDFormat,
                                ndim=ndim, cdim=cdim, kdim=kdim, sdims=sdims, fdims=fdims, \
                                fil=self.cc.fil, stride=self.cc.stride, dilation=self.cc.dilation)
    elif convolutionType in ("ConvolutionBackwardData"):
      # swaps cdim and kdim
      formatB=self.swap(self.tensorBFormat, 'C', 'K')
      self.initForwardConvolution(problemTypeOut, config, \
                                self.tensorAFormat, formatB, self.tensorDFormat,
                                ndim=ndim, cdim=kdim, kdim=cdim, sdims=sdims, fdims=fdims, \
                                fil=self.cc.fil, stride=self.cc.dilation, dilation=self.cc.stride)
    elif convolutionType in ("ConvolutionBackwardWeights"):
      # swaps ndim and cdim; filter and spatial
      formatA=self.swap(self.tensorBFormat, 'C', 'N')
      # convert activation->weight format, ie NCHW->KCYX
      formatB=self.swap(self.tensorBFormat, 'WHD', 'XYZ').replace('C','K').replace('N','C')
      formatD=self.swap(self.tensorDFormat, 'C,K,XYZ', 'N,C,WHD')  # ie CKYX -> NCHW
      self.initForwardConvolution(problemTypeOut, config, \
                                  formatA, formatB, formatD,
                                  ndim=cdim, cdim=ndim, kdim=kdim, sdims=list(reversed(fdims)), \
                                  fdims=list(reversed(sdims)), \
                                  fil=self.cc.spatial, stride=self.cc.dilation, \
                                  dilation=self.cc.stride)
      # fdims (filter dims) become the free dims for backward-weights.
      # if these dims have size==1 and stride==default they may be collapsed to empty list.
      # set AllowNoFreeDims to tell Tensile to use the batch dim as a virtual free dims
      # this forces PackBatchDims and sets Index* appropriately.
      if not fdims:
        problemTypeOut["AllowNoFreeDims"] = True

    # convert from convolution order to tensor order:
    self.regDimsA.reverse()
    self.regDimsB.reverse()

    problemTypeOut["IndexAssignmentsA"] = [x[0] for x in self.regDimsA]
    problemTypeOut["IndexAssignmentsB"] = [x[0] for x in self.regDimsB]
    problemTypeOut["UseBeta"] = False # MI kernels don't use beta


    self.solutionParms = {}

    stridea=[]
    for (idx,fbs,dim) in self.regDimsA:
      if dim.strideA != -1:
        stridea.append([idx, dim.strideA])
    stridea.sort()
    self.solutionParms["AssertStrideAEqual"] = \
            {problemTypeOut["IndexAssignmentsA"].index(s[0]) : s[1] for s in stridea}
    problemTypeOut["SetConstStrideA"] = stridea

    strideb=[]
    for (idx,fbs,dim) in self.regDimsB:
      if dim.strideB != -1:
        strideb.append([idx,dim.strideB])
    strideb.sort()

    self.solutionParms["AssertStrideBEqual"] = \
            {problemTypeOut["IndexAssignmentsB"].index(s[0]) : s[1] for s in strideb}
    problemTypeOut["SetConstStrideB"] = strideb

    self.solutionParms["AssertSizeEqual"] = {regDim.idx:regDim.dim.size for regDim in self.indexAssignments if regDim.dim.size != -1}

    if self.solutionParms["AssertStrideAEqual"].get(0,-1) == 1 and \
       self.solutionParms["AssertStrideBEqual"].get(0,-1) == 1:
      # optimize if no initial stride needed in A or B
      # allow yaml to override this for testing UseInitialStridesAB
      if "UseInitialStridesAB" not in problemTypeOut:
        problemTypeOut["UseInitialStridesAB"] = False
    else:
      problemTypeOut["UseInitialStridesAB"] = True

    iaa = problemTypeOut["IndexAssignmentsA"]
    iab = problemTypeOut["IndexAssignmentsB"]
    allIndices = list(range(len(self.indexAssignments)))
    self.sumIndices  = list(range(problemTypeOut["NumIndicesC"], len(self.indexAssignments)))
    self.batchIndices = [idx for idx in allIndices \
                   if idx in iaa and idx in iab and idx not in self.sumIndices]
    self.freeIndices = [idx for idx in allIndices \
                   if idx not in self.sumIndices and idx not in self.batchIndices]
    self.checkDims(self.freeIndices, self.batchIndices, self.sumIndices)


  def dimIdx(self, convolutionChar):
    return self.convolutionDims[convolutionChar].idx

  def convolutionChar(self, dimIdx):
    return self.indexAssignments[dimIdx].dim.shortChar

  def markedConvolutionChar(self, dimIdx, tc):
    regDim = self.indexAssignments[dimIdx]
    assert tc in ('A','B')
    if tc=='A' and regDim.fbs==Fbs.Sum and regDim.dim.shortChar not in ['C','K','N']:
        return "_" + regDim.dim.shortChar
    else:
        return regDim.dim.shortChar

  def makeProblem(self, n, c, k, pcc):
    """
    Generate valid problem dims for specified convolution
    pcc is a ConvolutionConfig class with specified values for this problem.
    The function will attempt to initialize TBD values in pcc from the convolution base class,
    and will then check to ensure the problem is fully specified.

    Return [ [sizes], [stridesA] ]
    """
    numDims = 1 + max(max([x[0] for x in self.regDimsA]), max([x[0] for x in self.regDimsB]))
    sizes = [-1]*numDims
    astrides = [-1]*numDims
    bstrides = [-1]*numDims

    pcc.copyFromRef(self.cc)
    pcc.checkFullySpecified(self.cc)

    sizes[self.convolutionDims['N'].idx]=n
    sizes[self.convolutionDims['C'].idx]=c
    sizes[self.convolutionDims['K'].idx]=k

    xIndexOfA = [index for index in range(len(self.regDimsA)) if self.regDimsA[index].dim.shortChar == 'W']
    cIndexOfA = [index for index in range(len(self.regDimsA)) if self.regDimsA[index].dim.shortChar == 'C']
    if xIndexOfA < cIndexOfA:
      astrides[self.convolutionDims['C'].idx] = reduce((lambda x, y: x * y), pcc.spatial)

    astrides[self.convolutionDims['N'].idx] = reduce((lambda x, y: x * y), pcc.spatial) * c
    bstrides[self.convolutionDims['N'].idx] = 0 # broadcast b matrix

    if len(pcc.spatial) != self.formatNumSpatialDims:
      raise RuntimeError ("len(pcc.spatial=", pcc.spatial, ") must match formatNumSpatialDims(%d)"%self.formatNumSpatialDims)

    # convert to Output dimensions:
    spatialOut=[0]*len(pcc.spatial)
    for i in range(self.formatNumSpatialDims):
      spatialOut[i] = int((pcc.spatial[i] + pcc.padStart[i] + pcc.padEnd[i] - ((pcc.fil[i]-1) * pcc.dilation[i] + 1)) / pcc.stride[i]) + 1

    #print ("spatialOut=", spatialOut, "padStart=", pcc.padStart, "padEnd=", pcc.padEnd)

    cScalar = c if self.regDimsA[0].dim.shortChar == 'C' else 1

    for fi,filterValue in enumerate(pcc.fil):
      try:
        pos = self.convolutionDims[chr(ord('X')+fi)].idx
        sizes[pos] = filterValue
        astrides[pos] = pcc.dilation[0]*cScalar if fi==0 else pcc.spatial[fi-1]*pcc.dilation[fi]*cScalar
      except KeyError:
        None

    if self.numSpatialDims==1:
      spatialName="DHW"[3-self.formatNumSpatialDims:]
      pos=self.convolutionDims[spatialName].idx
      sizes[pos] = reduce((lambda x, y: x * y), spatialOut) # product of all spatial dimes
      astrides[pos] = pcc.stride[0]*cScalar
    else:
      for si,sout in enumerate(spatialOut):
        spatialChars=['W','H','D']
        pos = self.convolutionDims[spatialChars[si]].idx
        sizes[pos] = sout
        astrides[pos]=pcc.stride[0]*cScalar if si==0 else pcc.spatial[si-1]*pcc.stride[si]*cScalar

    assert all(i!=-1 for i in sizes)

    # translate to strides for A tensor in IndexAssignmentsA order:
    orderedStridesA = []
    orderedStridesB = []
    for (idx,fbs,dim) in self.regDimsA:
      orderedStridesA.append(astrides[idx])

    for (idx,fbs,dim) in self.regDimsB:
      orderedStridesB.append(bstrides[idx])

    #print("ordered=A", orderedStridesA, "b=", orderedStridesB)

    return (sizes, orderedStridesA, orderedStridesB)

  def registerA(self, regDimList):
    """
    Provide a list of indices in convolution order - these will be reversed when assigned to IndexAssignmentsAB
    The order of items in the list determines the IndexAssignment order.
    Each tuple in the list is a RegDim class.
     - idx is the tensor index
     - fbs indicates if the tensor is expected to be Free, Sum, or Batch.  This is used for later check.
     - dim is Convolution.Dimension class that describes the dimension (for Usage info)
    """
    for regDim in regDimList:
      try:
        self.indexAssignments[regDim.idx]
      except IndexError:
        self.indexAssignments.extend([None]*(1+regDim.idx-len(self.indexAssignments)))
      assert(self.indexAssignments[regDim.idx] == None or \
             self.indexAssignments[regDim.idx] == regDim)
      self.indexAssignments[regDim.idx] = regDim
      self.convolutionDims[regDim.dim.shortChar] = regDim
    self.regDimsA = regDimList

  def registerB(self, regDimList):
    """
    See registerA
    """
    for regDim in regDimList:
      try:
        self.indexAssignments[regDim.idx]
      except IndexError:
        self.indexAssignments.extend([None]*(1+regDim.idx-len(self.indexAssignments)))
      assert(self.indexAssignments[regDim.idx] == None or \
             self.indexAssignments[regDim.idx] == regDim)
      self.indexAssignments[regDim.idx] = regDim
      self.convolutionDims[regDim.dim.shortChar] = regDim

    self.regDimsB = regDimList

  def dimxParm(self, config, parmName, default):
    parm =config.get(parmName)
    if not parm:
      rv = [default ] * self.formatNumSpatialDims
    else:
      rv=[]
      for x in parm.split('x'):
        if x.upper()=='N':
          rv.append(-1)
        else:
          rv.append(int(x))
      rv.reverse() # rightmost number is 0
    if len(rv) != self.formatNumSpatialDims:
        raise RuntimeError ("%s parm '%s' must have %d spatial dims'"%(parmName, parm, self.formatNumSpatialDims))
    return rv

  def printUsage(self, problemType, details=False):
    print()
    print("Tensor Formats: A:%s B:%s D:%s\n" % (self.tensorAFormat, self.tensorBFormat, self.tensorDFormat))
    print("Input Conv: %s packedFilter:%d packedSpatiol:%d unrollOnChannel:%d\n" % \
            (str(self.cc), self.packedFilterDims, self.packedSpatialDims, \
             self.unrollOnChannel))
    print("Normalized Conv: %s unrollOnChannel:%d\n" % (str(self.normalizedCC), self.unrollOnChannel))
    print("Tensile Index Assignments and Usage:")
    print("   Tensile    : ConvChar: Explanation/Usage")
    for (idx,regDim) in enumerate(self.indexAssignments):
        tensileChar = globalParameters['IndexChars'][idx]
        usage = str(regDim.dim)
        usage = usage.replace('#T', tensileChar)
        for i in range(len(self.cc.stride)):
            usage = usage.replace('#S%d'%i, str(self.cc.stride[i]) if self.cc.stride[i]>=0 else 'TBD')
        for i in range(len(self.cc.dilation)):
            usage = usage.replace('#D%d'%i, str(self.cc.dilation[i]) if self.cc.dilation[i]>=0 else 'TBD')
        print("  %d('%c') %-5s:   %s" % (idx, tensileChar, str(regDim.fbs).split('.')[1], usage))

    if 0:
      print ()
      print ("  FreeIndices:", ','.join([str(x) for x in self.freeIndices]))
      print ("  BatchIndices:", ','.join([str(x) for x in self.batchIndices]))
      print ("  SumIndices:", ','.join([str(x) for x in self.sumIndices]))

    if details:
      print ()
      print ("- Spatial sizes D_i, H_i, W_i refer to size of INPUT dimension.")
      print ("- Spatial sizes D_o, H_o, W_o refer to size of OUTPUT dimension.")
      print ("     For example W_o =  (W_i - X - padStart - padEnd + 1)/stride")
      print ("- (TBD)' indicates the parm is flexible and must be specified at runtime.")
      print ("- (i)' where i is an integer constant, indicates the parm is hard-coded at compile time.")
      print ("  The runtime value must match the compile-time value.")
      print ("- Unspecified strides use default stride value:")
      print ("    stride[i] = (stride[i-1]*size[i]) for i>0 ; 1 for i==0.")
      print ("- [stride*,size*] in brackets list required values to run the generated solutions.")
      print ("- Tensile IndexAssignments list the fastest-moving (in memory) index first.")
      print ("- Dimension collapsing:")
      print ("    - spatial dims with default strides and no zero-pad are collapsed with adjacent dims.")
      print ("    - Nx1 filter with dilationY=1 collapse into a single filter.")
      print ("    - 1xN filter with dilationX=1 collapse into a single filter.")
      print ("    - PackSpatialDims=0 / PackFilterDims=0 forcibly disables collapsing.")
      print ("- Overlapping / Hidden summation dimensions shown below with leading '_'.")

    print ()
    if problemType:
      print ("ProblemType Definition:")
      for k in Convolution.SummaryProblemProperties:
        try:
          if k in ['IndexAssignmentsA', 'IndexAssignmentsB']:
              comment = "# [" + ",".join([self.markedConvolutionChar(idx,k[-1]) for idx in problemType[k]]) + "]"
          elif k == 'NumIndicesC':
              comment = "# [" + ",".join([self.convolutionChar(idx) for idx in range(0,problemType[k])] ) + "]"
          else:
              comment = ""
          print ("  ", k, ":", problemType[k], comment)
        except KeyError:
          pass

    print ()
    print ("Solution Assertions:")
    for k in Convolution.SummarySolutionProperties:
      try:
        print ("  ", k, ":", self.solutionParms[k])
      except KeyError:
        pass


  def checkDims(self, freeIndices, batchIndices, sumIndices):
    for dimList in (self.regDimsA, self.regDimsB):
      for (idx,fbs,dim) in dimList:
        if fbs==Fbs.Free and idx not in freeIndices:
          raise RuntimeError ("dimension %d('%s') expected to be free dimension" % (idx, dim.shortChar))
        elif fbs==Fbs.Batch and idx not in batchIndices:
          raise RuntimeError ("dimension %d('%s') expected to be batch dimension" % (idx, dim.shortChar))
        elif fbs==Fbs.Sum and idx not in sumIndices:
          raise RuntimeError ("dimension %d('%s') expected to be summation dimension" % (idx, dim.shortChar))


  def identifier(self, problem = None):

    if problem == None:
      id = self.convolutionType
      id += "_" + self.tensorAFormat
      id += "_" + self.tensorBFormat
      id += "_" + self.tensorDFormat
      id += "_spatialDims:" + str(self.numSpatialDims)
      id += "_indices:" + '.'.join([x.dim.shortChar for x in self.indexAssignments])
    else:
      id = ''
      problemCC = problem.convConfig
      id += ",".join([str(x) for x in problemCC.spatial])
      id += "," + ",".join([str(x) for x in problemCC.fil])
      id += "," + ",".join([str(x) for x in problemCC.stride])
      id += "," + ",".join([str(x) for x in problemCC.dilation])
      id += "," + ",".join([str(x) for x in problemCC.padStart])
      id += "," + ",".join([str(x) for x in problemCC.padEnd])

    return id


################################################################################
# ProblemType
# name of solution should begin with name of problemType, and arguments can be listed out explicitly
class ProblemType(Mapping):
  ########################################
  def __init__(self, config):
    self.state = {}

    for key in defaultProblemType:
      assignParameterWithDefault(self.state, key, config, defaultProblemType)

    # adjusting all data types
    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      printExit("NO data type specified")
      self["DataType"] = DataType(0)

    if "DestDataType" in config:
      self["DestDataType"] = DataType(config["DestDataType"])
    else:
      if "DataType" in config:
        self["DestDataType"] = DataType(config["DataType"])
      else:
        printExit("NO dest data type or data type specified")
        self["DataType"] = DataType(0)

    if "ComputeDataType" in config:
      self["ComputeDataType"] = DataType(config["ComputeDataType"])
    else:
      if "DestDataType" in config:
        self["ComputeDataType"] = DataType(config["DestDataType"])
      else:
        if "DataType" in config:
          self["ComputeDataType"] = DataType(config["DataType"])
        else:
          printExit("NO compute data type, or dest data type, or data type specified")
          self["DataType"] = DataType(0)

    if "F32XdlMathOp" in config:
        self["F32XdlMathOp"] = DataType(config["F32XdlMathOp"])
    else:
        self["F32XdlMathOp"] = DataType(0)

    # Modifying ComputeDataType for HHH+HPA: if (HHH+HPA), convert it to HHS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isHalf() and self["DataType"].isHalf() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == f16, DestType == f16, ComputeDataType == f16, but HPA == True (HHH+HPA, no such a type); Converting HHH+HPA to HHS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for BBB+HPA: if (BBB+HPA), convert it to BBS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isBFloat16() and self["DataType"].isBFloat16() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == bf16, DestType == bf16, ComputeDataType == bf16, but HPA == True (BBB+HPA, no such a type); Converting BBB+HPA to BBS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    self.convolution = None
    if self["OperationType"] == "GEMM":
      self.checkIfSupportedGEMMType()
      self.initGEMM()
    elif self["OperationType"] == "TensorContraction":
      self.initTensorContraction(self.state)
    elif self["OperationType"] in ("ConvolutionForward", "ConvolutionBackwardData", "ConvolutionBackwardWeights"):
      self.initConvolution(config, self["OperationType"])
    else:
      printExit("Unsupported OperationType = %s" % self["OperationType"])

    self.state["AssignedDerivedParameters"] = False
    ProblemType.assignDerivedParameters(self.state)

    if self.convolution:
      if globalParameters["PrintConvolutionUsage"] & 0x3 :
        print()
        self.convolution.printUsage(self, globalParameters["PrintConvolutionUsage"]&0x2)
        print()
      self.convolution.checkDims(self.state["IndicesFree"], self.state["IndicesBatch"], self.state["IndicesSummation"])


    for tc in ('A', 'B'):
      freeDims={}
      sumDims={}
      for zp in self["ZeroPad%s"%tc] :
        (freeDim, sumDim, leading, trailing) = zp
        if freeDim not in self.state["IndicesFree"]:
          printExit("ZeroPad%s=%s dim=%u is not a free index"%(tc, zp, freeDim))
        if freeDim not in self.state["IndexAssignments%s"%tc]:
          printExit("ZeroPad%s=%s dim=%u is not in IndexAssignments%s"%(tc, zp, freeDim, tc))
        if sumDim not in self.state["IndicesSummation"]:
          printExit("ZeroPad%s=%s dim=%u is not a summation index"%(tc, zp, sumDim))
        if freeDim in freeDims:
          printExit("ZeroPad%s=%s freeDim=%u occurs in more than one tuple (prev:%s)"%(tc, zp, freeDim,freeDims[freeDim]))
        freeDims[freeDim] = zp
        if sumDim in sumDims:
          printExit("ZeroPad%s=%s sumDim=%u occurs in more than one tuple"%(tc, zp, sumDim))
        sumDims[sumDim] = zp

    for tc in ('A', 'B'):
      for sc in self["SetConstStride%s"%tc] :
          (anchorDim, stride) = sc[:2]
          if anchorDim not in self.state["IndexAssignments%s"%tc]:
              printExit("SetConstStride%s=%s anchorDim=%u is not in IndexAssignments%s"%(tc, sc, anchorDim, tc))

  ################################################################################
   # Function checkIfSupportedGEMMType:
  #   Assures 3 data-types are valid, supported and well-assigned
  #   See the discussion on Common.py for validGEMMTypes
  ################################################################################
  def checkIfSupportedGEMMType(self):
    inType = self["DataType"]
    outType = self["DestDataType"]
    computeType = self["ComputeDataType"]

    gemmType = ( inType.toChar(), outType.toChar(), computeType.toChar() )

    if gemmType not in validGEMMTypes:
      printExit("This typed-GEMM (Ti, To, Tc) = (%s, %s, %s) is not supported yet."%(gemmType[0],gemmType[1],gemmType[2]))

  ########################################
  def initGEMM(self):
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["TransposeB"]:
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["NumIndicesC"] = 3
    else:
      self["NumIndicesC"] = 2

    self["NumIndicesLD"] = 4
    self["IndexAssignmentsLD"][0] = self["NumIndicesC"] + 1
    for i in range(1, len(self["IndexAssignmentsLD"])):
      self["IndexAssignmentsLD"][i] = self["IndexAssignmentsLD"][i-1] + 1

  ########################################
  def initTensorContraction(self, config):
    assignParameterRequired(self.state, "NumIndicesC", config)
    assignParameterRequired(self.state, "IndexAssignmentsA", config)
    assignParameterRequired(self.state, "IndexAssignmentsB", config)
    self["NumIndicesLD"] = 0

  ########################################
  def initConvolution(self, config, convolutionType):
    convolutionConfig = {}
    try:
      if config['ConvolutionConfig'] != None:
        for dict in config['ConvolutionConfig']:
          for k,v in dict.items():
            convolutionConfig[k] = v
    except KeyError:
      raise RuntimeError ("OperationType %s must include ConvolutionConfig section in ProblemType"%convolutionType)

    self.convolution = Convolution(self, convolutionType, convolutionConfig)
    self["NumIndicesLD"] = 0
    # For Conv with filter 1x1, unit-stride, no padding case, we can let UseBeta = True
    self["UseBeta"] = ("UseBeta" in config and config["UseBeta"] == True) and (self.canExpressedAsGEMM() == True)

  ########################################
  def canExpressedAsGEMM(self):
    rv = self.convolution != None and \
         self.convolution.cc != None and \
         self.convolution.cc.fil == [1,1] and \
         self.convolution.cc.stride == [1,1] and \
         self.convolution.cc.dilation == [1,1] and \
         self.convolution.cc.padStart == [0,0] and \
         self.convolution.cc.padEnd == [0,0]
    return rv

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  def isTensorContraction(self):
    return self.operationType == 1

  ########################################
  # determine d0, d1, dU
  @staticmethod
  def assignDerivedParameters(state):
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
        printExit("invalid index %u (inC but not (inA or inB))" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u (expected summation but not (inA and inB))" % i)
    # print index assignments
    if globalParameters["PrintIndexAssignments"]:
      print("IndicesFree:  %s" % state["IndicesFree"])
      print("IndicesBatch: %s" % state["IndicesBatch"])
      print("IndicesSum:   %s" % state["IndicesSummation"])
      print("IndexAssignmentsA:   %s" % state["IndexAssignmentsA"])
      print("IndexAssignmentsB:   %s" % state["IndexAssignmentsB"])
      print("NumIndicesC:  %s" % state["NumIndicesC"])

    for k in ('IndexAssignmentsA','IndexAssignmentsB'):
      if len(state[k]) != len(set(state[k])):
        printExit("duplicate index in %s=%s"% (k,state[k]))

    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if not state["AllowNoFreeDims"] and state["NumIndicesFree"] < 2 :
      printExit("Tensile requires >= 2 free indices or set AllowNoFreeDims; FreeIndices=%s."% state["IndicesFree"])

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
    #tPrint(3, "IndexUnrollA: %u" % state["IndexUnrollA"])
    #tPrint(3, "IndexUnrollB: %u" % state["IndexUnrollB"])

    # assign d0, d1
    if state["AllowNoFreeDims"]:
      dimList = state["IndicesFree"] + state["IndicesBatch"]
    else:
      dimList = state["IndicesFree"]
    state["Index01A"] = [i for i in state["IndexAssignmentsA"] if i in dimList][0]
    state["Index01B"] = [i for i in state["IndexAssignmentsB"] if i in dimList][0]
    #tPrint(3, "Index01A: %u" % state["Index01A"])
    #tPrint(3, "Index01B: %u" % state["Index01B"])
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

    if globalParameters["PrintIndexAssignments"]:
      print("TLUA:  %s (stridePosA(%d) <? unrollIdxA(%d)" % \
            (state["TLUA"], strideIdxA, unrollIdxA))
      print("TLUB:  %s (stridePosB(%d) <? unrollIdxB(%d)" % \
              (state["TLUB"], strideIdxB, unrollIdxB))
      print("Index01A:  %s" % state["Index01A"])
      print("Index01B:  %s" % state["Index01B"])
    #unrollDimStrideGreaterThanTileDimStrideA = TLUA = !transA = fast
    #!unrollDimStrideLessThanTileDimStrideB   = TLUB =  transB = fast
    state["AssignedDerivedParameters"] = True


  ########################################
  def __str__(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self["NumIndicesC"]):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i] if i in self["MirrorDimsA"] else indexChars[i].lower()
    if self["ComplexConjugateA"]:
      name += "C"
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i] if i in self["MirrorDimsB"] else indexChars[i].lower()
    if self["ComplexConjugateB"]:
      name += "C"

    # DataTypes
    name += "_"
    name += self["DataType"].toChar() # Type of A/B

    # Special condition for some newly supported kernels:
    #   HHS, HSS, BSS and I8II kernels, use a clearer naming _TiToTc_
    # TODO: Distinguish all kernels by _TiToTc_ to be more consistent with rocblas
    gemmType = (self["DataType"].toChar(),self["DestDataType"].toChar(),self["ComputeDataType"].toChar() )
    if gemmType in HPATypes:
      name += self["DestDataType"].toChar()    # Type of C/D
      name += self["ComputeDataType"].toChar() # Type of Alpha/Beta
      name += "_"

    # Other
    if self["UseBeta"]: name += "B"
    if self["HighPrecisionAccumulate"] and not self["SilentHighPrecisionAccumulate"]: name += "H"
    if self["Fp16AltImpl"]:
      if self["Fp16AltImplRound"]: name += "RZ"
      else: name += "R"
    if self["UseInitialStridesAB"]: name += "I"
    if self["UseInitialStridesCD"]: name += "Ic"

    # precision and other
    # name += "_SB" if self["StridedBatched"] else "_GB"
    name += "" if self["StridedBatched"] else "_GB" # legacy

    if not self["F32XdlMathOp"].isSingle() and self["DataType"].isSingle():
      name += "_M"
      name += self["F32XdlMathOp"].toChar()

    # Rounding mode: IEEE vs SR
    if self["StochasticRounding"]:  name += "_SR"

    return name

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
# ProblemSizeRange
################################################################################
class ProblemSizeRange:

  ########################################
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
  def __init__(self, sizes=None, stridesA=None, stridesB=None, stridesC=None, stridesD=None, zeroPadA=None, zeroPadB=None, count=None):
    self.sizes = tuple(sizes) if sizes else None
    self.stridesA = tuple(stridesA) if stridesA else None
    self.stridesB = tuple(stridesB) if stridesB else None
    self.stridesC = tuple(stridesC) if stridesC else None
    self.stridesD = tuple(stridesD) if stridesD else None

    self.zeroPadA = zeroPadA
    self.zeroPadB = zeroPadB
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


class ConvProblem(Problem):
  ConvField = namedtuple ("ConvField", ('shortChar', 'descrip', 'default'))
  AllowedConvFields = [ ConvField('n', 'Batch Count', None),
                        ConvField('c', 'Channel In', None),
                        ConvField('k', 'Channel Out',  None),

                        ConvField('d', 'Spatial Depth', -1),
                        ConvField('h', 'Spatial Height',-1),
                        ConvField('w', 'Spatial Width', -1),

                        ConvField('z', 'Filter Z',  -1),
                        ConvField('y', 'Filter Y',  -1),
                        ConvField('x', 'Filter X',  -1),

                        ConvField('#', 'Stride for Depth', -1),
                        ConvField('u', 'Stride for Height', -1),
                        ConvField('v', 'Stride for Width', -1),

                        ConvField('^', 'Dilation for filter Depth Z', -1),
                        ConvField('l', 'Dilation for filter Height Y', -1),
                        ConvField('j', 'Dilation for filter Width X', -1),

                        ConvField('$', 'Pad for Depth', -1),
                        ConvField('p', 'Pad for Height', -1),
                        ConvField('q', 'Pad for WidthX', -1),

                        ConvField('$_', 'Pad End for Depth (overrides $ for end)', -1),
                        ConvField('p_', 'Pad End for Height (overrides p for end)', -1),
                        ConvField('q_', 'Pad End for Width (overrides q for end)', -1),

                        ConvField('g', 'Group Count',  1),

                        ConvField('count', 'Layer execution Count',  -1),
                        ]
  AllowedConfFieldsDict = {field.shortChar : field for field in AllowedConvFields}

  @staticmethod
  def initParm(e, chars, skipFields):
    fields = []
    for s in (chars):
      if s not in skipFields:
        fields.append(e[s])
    return fields

  def __init__(self, e, convolution):

    self.inputConfig = deepcopy(e)

    if convolution.formatNumSpatialDims==2:
      skipFields = ('d', 'z', '#', '^', '$')
    else:
      skipFields = ()

    if not isinstance(e,dict):
        raise RuntimeError ("ConvProblem must be a dictionary, for example '{n: 64, ...}' not '[n: 64, ...]'")

    for k in e:
      if k not in ConvProblem.AllowedConfFieldsDict:
        # TODO  - detect and print message for common error n:32 w/o space
        raise RuntimeError ("unknown ConvProblem field '%s'"%k)

    for (k,field) in ConvProblem.AllowedConfFieldsDict.items():
      if k not in e and k not in skipFields:
        if field.default == None:
          raise RuntimeError ("required ConvProblem field '%s' not present in ConvProblem:%s"%(k,e))
        elif isinstance(field.default, int):
          e[k] = field.default

    padStart = self.initParm(e, ('q','p','$'), skipFields)
    padEnd = self.initParm(e, ('q_','p_','$_'), skipFields)
    padEnd = [ps if pe==-1 else pe for (ps,pe) in zip(padStart,padEnd) ] # use padStart as default
    self.convConfig = ConvolutionConfig(
                fil = self.initParm(e, ('x','y','z'), skipFields),
                stride = self.initParm(e, ('v','u','#'), skipFields),
                dilation   = self.initParm(e, ('j','l','^'), skipFields),
                spatial =    self.initParm(e, ('w','h','d'), skipFields),
                padStart = padStart,
                padEnd = padEnd,
                groupCount = e['g']
              )

    (sizes, stridesA, stridesB) = convolution.makeProblem(e['n'], e['c'], e['k'], self.convConfig)
    zeroPadA = convolution.makeZeroPadProblemType(convolution.problemTypeOut["ZeroPadA"],
        self.convConfig.padStart, self.convConfig.padEnd, e['c'], self.convConfig)

    Problem.__init__(self, sizes, stridesA, stridesB=stridesB, zeroPadA=zeroPadA, count=e['count'])

    #print ("sizes=", self.sizes, "stridesA=", self.stridesA, "stridesB=", self.stridesB, "zeroPadA=", self.zeroPadA)


  def toExactDict(self):
    """ Return a dict with ExactDict fields, after converting the ConvProblem to tensor sizes and strides"""
    padStartA = [zp[2] for zp in self.zeroPadA]
    padEndA = [zp[3] for zp in self.zeroPadA]
    exactFields = OrderedDict()

    exactFields['count'] = self.count
    exactFields['sizes'] = list(self.sizes)
    exactFields['stridesA'] = list(self.stridesA)

    if padStartA:
      exactFields['padStartA'] = padStartA
    if padEndA:
      exactFields['padEndA'] = padEndA

    return exactFields


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
      printExit("ExactSize %s doesn't match indices of ProblemType %s, totalIndices=%d" \
          % (e, problemType, problemType["TotalIndices"]) )

    # TODO- pass strides here, remove calls to convertLeadingDims
    Problem.__init__(self, sizes=sizes, zeroPadA=problemType["ZeroPadA"], zeroPadB=problemType["ZeroPadB"])

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
  # padStartA is list of pad starts for A dimension in order of ZeroPadA list.
  # padEndA is list of pad ends for A dimension in order of ZeroPadA list.
  AllowedFields = [ 'count', 'sizes', 'stridesA', 'stridesB', 'stridesC', 'stridesD', 'padStartA', 'padEndA', 'padStartB', 'padEndB']

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
      zp={}
      zp['A'] = deepcopy(problemType["ZeroPadA"])
      zp['B'] = deepcopy(problemType["ZeroPadB"])

      for (tc, padName, zpField) in (
          ("A", "padStartA",2), ("A", "padEndA", 3),
          ("B", "padStartB",2), ("B", "padEndB", 3) ):
          try:
            problemPad = getattr(self, padName)
            if len(problemPad) != len (zp[tc]):
                raise RuntimeError ("problem-specified %s==%s does not match length of problem-type pad==%s." % (padName, problemPad, zp[tc]))
            for (i,p) in enumerate(problemPad):
              if not (zp[tc][i][zpField] == -1 or zp[tc][i][zpField] == p):
                raise RuntimeError ("problem-specified %s==%d does not match problem-type==%d." % (padName, p, zp[tc][i][zpField]))
              zp[tc][i][zpField] = p
          except AttributeError:
            None

      for (tc) in ("A", "B"):
        for p in zp[tc]:
          if p[2] == -1 or p[3]==-1:
            raise RuntimeError ("padStart/padEnd for %s must be specified in problem-type or problem - can't be left -1/TBD" % zp[tc])

      self.zeroPadA = zp['A']
      self.zeroPadB = zp['B']
    else:
      self.zeroPadA = self.zeroPadB = []

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

          elif sizeTypeKey == "Conv":
            if problemType.convolution == None:
              printExit("ConvProblem requires OperationType==Convolution*")
            else:
              self.exacts.append(ConvProblem(dictionary[sizeTypeKey], problemType.convolution))

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
            self.problems.update({Problem(rangeSize, zeroPadA=problemType["ZeroPadA"]) : 1 })
    for e in self.exacts:
        self.problems.update({e : 1})
    if globalParameters["SortProblems"]:
      self.problems =  sorted(list( self.problems.keys()), key=operator.attrgetter("sizes"))
    else:
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

    if globalParameters["PrintConvolutionUsage"] & 0x4:
      for problem in self.problems:
        if isinstance(problem, ConvProblem):
          print (problem.inputConfig, '->\n  ', ", ".join(["%s: %s"%(k,v) for (k,v) in problem.toExactDict().items()]))


  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s

# kds is class Solution or class Kernel
# If PackFreeDims=1 then all free dims are packed ; else only 1 free dim/matrix is supported
# PackBatchDims can pack batches into A or B (has stride==0 requirements for non-packed tensor);
# batchMask controls which bit in PackBatchDims detects batch index
def isPackedIndex(ks, index, batchMask=0x3):
  problemType = ks["ProblemType"]
  return index in problemType["IndicesFree"] and ks["PackFreeDims"] or \
         index in problemType["IndicesBatch"] and (ks["PackBatchDims"] & batchMask)

def isExtractableIndex(ks, index, tc='x'):
  xA = index in ks['PackedC0IndicesX'][:-1]
  xB = index in ks['PackedC1IndicesX'][:-1]
  if tc=='A':
    return xA
  elif tc=='B':
    return xB
  else:
    return xA or xB

################################################################################
# Solution
################################################################################
class Solution(collections.abc.Mapping):

  ########################################
  def __init__(self, config):
    self._name = None

    state = {}
    # problem type
    if "ProblemType" in config:
      state["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      state["ProblemType"] = ProblemType(defaultProblemType)

    # assign parameters with defaults
    for key in defaultSolution:
      assignParameterWithDefault(state, key, config, defaultSolution)

    if 'ISA' not in state:
      if 'ISA' in config:
        state['ISA'] = config['ISA']
      elif config['KernelLanguage'] == 'Assembly':
        state['ISA'] = list(globalParameters["CurrentISA"])
      else:
        state['ISA'] = [0,0,0]

    if "CodeObjectVersion" not in state:
      if "CodeObjectVersion" in config:
        state["CodeObjectVersion"] = config["CodeObjectVersion"]
      else:
        state["CodeObjectVersion"] = globalParameters["CodeObjectVersion"]

    # assign parameters without defaults
    for key in config:
      if key != "ProblemType" and key not in state:
        state[key] = config[key]
    state["Valid"] = True
    # this could prevent OriginalSolution from re-assigning the parameters, save lots of time
    if "AssignedProblemIndependentDerivedParameters" not in state:
      state["AssignedProblemIndependentDerivedParameters"] = False
    if "AssignedDerivedParameters" not in state:
      state["AssignedDerivedParameters"] = False

    if state["ProblemType"].convolution:
        for (key,value) in state["ProblemType"].convolution.solutionParms.items():
            state[key]=value
    Solution.assignDerivedParameters(state)
    self._name = config["CustomKernelName"] if isCustomKernelConfig(config) else None
    self._state = state
    self.initHelperKernelObjects()

  # these keys are copied from ProblemType to internal that may be overridden
  InternalKeys = ["UseSgprForGRO","VectorStore"]


  ########################################
  # get a list of kernel parameters for this solution
  def getKernels(self):
    self["Kernel"] = True
    return self


  ########################################
  # create Helper Kernels
  def initHelperKernelObjects(self):
    self.initStreamKInitKernelObjects()
    self.initBetaOnlyKernelObjects()
    self.initConversionKernelObjects()


  ########################################
  # create StreamKInit Kernels
  def initStreamKInitKernelObjects(self):
    self.streamKInitKernelObjects = []
    if self["StreamK"] > 0 and self["StreamKAtomic"] == 0:
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["KernelLanguage"] = "Source"
      state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
      self.streamKInitKernelObjects.append(KernelWriterStreamKInit(state))


  ########################################
  # create BetaOnly Kernels
  def initBetaOnlyKernelObjects(self):
    self.betaOnlyKernelObjects = []
    if self["GlobalSplitU"] > 1 or self["StreamK"] > 0 and self["StreamKAtomic"] == 1:
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["KernelLanguage"] = "Source"
      state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
      self.betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))


  ########################################
  # create Conversion Kernels
  def initConversionKernelObjects(self):
    self.conversionKernelObjects = []
    gsu = self["GlobalSplitU"]
    if (gsu > 1) and self["_GlobalAccumulation"]:
      # wider load for GSU is single compute type only
      supportedTypeForVWopt = self["ProblemType"]["ComputeDataType"].isSingle() or self["ProblemType"]["ComputeDataType"].isDouble()
      vwMax = 1
      if (supportedTypeForVWopt):
        vwMax = 2

      # reduction for GSU is single compute type + gus = power of 2 only
      supportedTypeForReductionOpt = self["ProblemType"]["ComputeDataType"].isSingle() or self["ProblemType"]["ComputeDataType"].isDouble()
      maxReduction = 1
      maxReductionConst = 4 # this must match the value in client code (ContractionSolution.cpp)
      minGSUperReduction = 32; # Minimum GSU=128 for Reduction=4, GSU=64 for Reduction2
      applicableReduction = max(1, gsu // minGSUperReduction)
      if (supportedTypeForReductionOpt and ((gsu & (gsu - 1)) == 0) and self["_GlobalAccumulation"] == "MultipleBuffer"):
        maxReduction = min(applicableReduction, maxReductionConst) # not exceeding reductionThreshold

      # loop unroll opt for postGSU
      supportedTypeForUnrollOpt = self["ProblemType"]["ComputeDataType"].isSingle() or self["ProblemType"]["ComputeDataType"].isDouble()

      vw = 1
      while vw <= vwMax:
        reduction = 1
        while reduction <= maxReduction:
          # so far, reduction=2 does not perform well. Skip 2
          if reduction == 2 and self["ProblemType"]["ComputeDataType"].isSingle():
            reduction *= 2
            continue
          state = {}
          state["ProblemType"] = deepcopy(self["ProblemType"])
          state["KernelLanguage"] = "Source"
          state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
          state["GlobalSplitU"] = self["GlobalSplitU"]
          state["VectorWidth"] = vw
          state["Reduction"] = reduction
          # number of unroll for large GSU (must match client code)
          state["GSUUnrollUnit"] = 16 * state["Reduction"] if supportedTypeForUnrollOpt and self["_GlobalAccumulation"] == "MultipleBuffer" else 1
          self.conversionKernelObjects.append(KernelWriterConversion(state))
          reduction *= 2
        vw *= 2


  ########################################
  # get Helper Kernels
  def getHelperKernelObjects(self):
    return self.streamKInitKernelObjects + self.betaOnlyKernelObjects + self.conversionKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelStreamKInitObjects(self):
    return self.streamKInitKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelBetaOnlyObjects(self):
    return self.betaOnlyKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelConversionObjects(self):
    return self.conversionKernelObjects


  @staticmethod
  def getMIOutputInfo(state):
    outputVectorWidth = 4
    RegsPerOut = 1

    isa = tuple(state["ISA"])
    if globalParameters["AsmCaps"][isa]['HasMFMA']:
      if state["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64':
        outputVectorWidth, RegsPerOut = 1, 2
      else:
        outputVectorWidth, RegsPerOut = 4, 1
    elif globalParameters["AsmCaps"][isa]['HasWMMA']:
      outputVectorWidth, RegsPerOut = 1, 1
    else:
      print("WARNING: unexpected code flow")

    return outputVectorWidth, RegsPerOut


  ########################################
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state):

    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False
    if "Valid" not in state:
      state["Valid"] = True

    if (not state["ProblemType"]["StridedBatched"]) and (not state["ProblemType"]['Batched']):
      reject(state, "General Batched GEMM only support Batched Problem")

    if (not state["ProblemType"]["StridedBatched"]) and (state["ProblemType"]["OperationType"] != 'GEMM'):
      reject(state, "General Batched GEMM only support GEMM OperationType")

    EnableMatrixInstruction = state["EnableMatrixInstruction"] if "EnableMatrixInstruction" in state else None
    if EnableMatrixInstruction == None:
      if  ("MIBlock" in state and len(state["MIBlock"]) == 6) \
          and ("MIWaveGroup" in state and len(state["MIWaveGroup"]) == 2) \
          and ("MIWaveTile" in state and len(state["MIWaveTile"]) == 2):
        EnableMatrixInstruction = True
      elif ("WorkGroup" in state and len(state["WorkGroup"]) == 3) \
          and ("ThreadTile" in state and len(state["ThreadTile"]) == 2) :
        EnableMatrixInstruction = False
      else:
        reject(state, "EnableMatrixInstruction undetermined")

    if EnableMatrixInstruction == True:
      state["MatrixInstM"]         = state["MIBlock"][0]
      state["MatrixInstN"]         = state["MIBlock"][1]
      state["MatrixInstK"]         = state["MIBlock"][2]
      state["MatrixInstB"]         = state["MIBlock"][3]
      state["MatrixInstBM"]        = state["MIBlock"][4]
      state["MatrixInstBN"]        = state["MIBlock"][5]

      state["MIOutputVectorWidth"], state["MIRegPerOut"] = Solution.getMIOutputInfo(state)

      if state["MatrixInstM"] == 4:
        state["ThreadTile0"] = state["MIWaveTile"][0] * state["MIOutputVectorWidth"]
        state["ThreadTile1"] = state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * state["MatrixInstM"] * state["MatrixInstBM"] // state["MIOutputVectorWidth"]
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"] * state["MatrixInstBN"]
      else:
        state["ThreadTile0"] = state["MatrixInstBM"] * state["MIWaveTile"][0] * (state["MatrixInstM"] * state["MatrixInstN"] // state["WavefrontSize"])
        state["ThreadTile1"] = state["MatrixInstBN"] * state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * (state["WavefrontSize"] // state["MatrixInstN"])
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"]

    elif EnableMatrixInstruction == False:
      state["ThreadTile0"] = state["ThreadTile"][0]
      state["ThreadTile1"] = state["ThreadTile"][1]

      state["SubGroup0"]   = state["WorkGroup"][0]
      state["SubGroup1"]   = state["WorkGroup"][1]

    state["LocalSplitU"] = state["WorkGroup"][2]
    # enable MatrixInstruction store only for LSU=1
    # LSU>1 case, use non-MI store instead
    state["EnableMatrixInstructionStore"] = EnableMatrixInstruction and state["LocalSplitU"]==1

    if "SubGroup0" in state and "SubGroup1" in state and "LocalSplitU" in state:
      state["NumThreads"]  = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"]
      if (state["NumThreads"] % state['WavefrontSize']) != 0:
        reject(state, f"size of WorkGroup {state['NumThreads']} should be multiple of WavefrontSize {state['WavefrontSize']}")
      if EnableMatrixInstruction == True:
        if ((state["SubGroup0"] * state["SubGroup1"]) % state['WavefrontSize']) != 0:
          reject(state, f"SubGroup0 {state['SubGroup0']} * SubGroup1 {state['SubGroup1']}should be multiple of WavefrontSize {state['WavefrontSize']}")
        if (state["LocalSplitU"] > 4):
          reject(state, f"LocalSplitU {state['LocalSplitU']} should not be larger than 4 with MatrixInstruction")

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "MacroTile" in state:
      if state["MacroTile0"] != state["MacroTile"][0] \
          or state["MacroTile1"] != state["MacroTile"][1]:
        reject(state, "MacroTile mismatch")

    if state["Valid"] and "MacroTileShapeMax" in state \
        and "MacroTileShapeMin" in state:
      macroTileShape = max(state["MacroTile0"]//state["MacroTile1"], \
          state["MacroTile1"]//state["MacroTile0"])
      if macroTileShape > state["MacroTileShapeMax"] \
          or macroTileShape < state["MacroTileShapeMin"]:
        reject(state, "rejecting MacroTile Shape %u:%u for Min:Max %u:%u" \
            % (state["MacroTile0"], state["MacroTile1"], \
            state["MacroTileShapeMin"], state["MacroTileShapeMax"]))

    if "WorkGroupMappingType" in state:
      if state["WorkGroupMappingType"] == "Z":
        if abs(state["WorkGroupMapping"]) > 2:
          reject(state, "WorkGroupMappingType=Z only supports WorkGroupMapping=1, 2")

    # done
    state["AssignedProblemIndependentDerivedParameters"] = True

  ########################################
  # This is the "classic" algorithm which requires that each threads load the same number of bytes
  # Called with tc=A and then with tc=B
  # totalVectors is totalElements/GRVW, this is #vectors loaded by the LoadTile
  # Reduces the GlobalLoadVectorWidth if necessary if each thread has a small amount of work to do.
  # Output from this function:
  #  state[GlobalLoadVectorWidth*]
  #  state[NumLoads*] # only used in SolutionStructs, with classic alg
  @staticmethod
  def setGlobalLoadVectorWidth(state, tc, totalElements, grvw):
    validDepthU = True
    numThreadGrvw = int(state["NumThreads"] * grvw)
    if totalElements < numThreadGrvw:
      # Try to reduce size of vector so every thread has a load to do
      pv = numThreadGrvw //totalElements
      if not state["FractionalLoad"]:
        if numThreadGrvw % totalElements != 0:
          reject(None, "(NumThreads * grvw) %u %% totalElements %u != 0" \
              % (numThreadGrvw, totalElements))
          validDepthU = False
        if pv * totalElements != numThreadGrvw:
          reject(None, "pv %u * totalElements %u != (NumThreads * grvw) %u " \
              % (pv, totalElements, numThreadGrvw))
          validDepthU = False
        if grvw < 1 or grvw % pv != 0:
          reject(None, "GlobalLoadVectorWidth%s %u %% pv %u != 0" \
              % (tc, grvw, pv))
          validDepthU = False
        grvw = grvw//pv
        numThreadGrvw = numThreadGrvw//pv
    else:
      pv = 1 # no partial vector required
      if totalElements % numThreadGrvw != 0:
        if not state["FractionalLoad"]:
          reject(None, "totalElements %u %% (NumThreads * grvw) %u != 0" \
              % (totalElements, numThreadGrvw))
          validDepthU = False

    state["GlobalLoadVectorWidth%s"%tc] = grvw

    # NumLoads is NOT used on the fractional path
    # NumLoads is number of vector loads per-thread
    state["NumLoads%s"%tc] = totalElements // numThreadGrvw
    #print "result: ", pvar(state, "GlobalLoadVectorWidth%s"%tc), \
    #        pvar(state, "NumLoads%s"%tc)

    return validDepthU

  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalescedA]
  #   state[NumLoadsPerpendicularA]
  #   state[LSCA]
  #   state[LSPA]
  @staticmethod
  def setGlobalLoadTileDimClassic(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp):

    if state["WaveSeparateGlobalRead%s"%tc]:
      totalElementsPerp = roundupRatio(totalElementsPerp, state["NumThreads"] // state["WavefrontSize"])

    # nlc = 1
    if state["NumLoadsCoalesced%s"%tc] == 1 :
      foundValid = False
      nlcStart = 1
      if state["DirectToVgpr%s"%tc]:
        # adjust nlc for DirectToVgpr
        if state["ProblemType"]["TLU%s"%tc]:
          nlcStart = roundupRatio(state["MIWaveTile%s"%tc], state["GlobalLoadVectorWidth%s"%tc])
        else:
          nlcStart = roundupRatio(state["DepthU"], state["MatrixInstK"] * state["GlobalLoadVectorWidth%s"%tc] * state["LocalSplitU"] // state["MIInputPerThread"])
      for nlc in range(nlcStart, int(state["NumLoads%s"%tc]+1)):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          #print("NumLoadsCoalesced",state["NumLoadsCoalesced%s"%tc])
          #print("NumLoadsPerpendicular",state["NumLoadsPerpendicular%s"%tc])
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=1 found"%tc)
        return False

    # nlc = -1
    elif state["NumLoadsCoalesced%s"%tc] == -1:
      foundValid = False
      for nlc in range(state["NumLoads%s"%tc], 0, -1):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=-1 found"%tc)
        return False

    # nlc = other
    else:
      if state["NumLoadsCoalesced%s"%tc] > state["NumLoads%s"%tc]:
        reject(state, "%s nlc > numLoads"%tc)
        return False

      state["NumLoadsPerpendicular%s"%tc] = state["NumLoads%s"%tc] \
          // state["NumLoadsCoalesced%s"%tc]

      if state["NumLoads%s"%tc] % state["NumLoadsCoalesced%s"%tc] != 0:
        reject(state, "%s: numLoads %u %% numLoadsCoalesced %u != 0" \
            % (tc, state["NumLoads%s"%tc], state["NumLoadsCoalesced%s"%tc]))
        return False

      if totalVectorsCoalesced % state["NumLoadsCoalesced%s"%tc] != 0 :
        reject(state, "%s: totalVectorsCoalesced %u %% numLoadsPara %u != 0" \
              % (tc, totalVectorsCoalesced, state["NumLoadsCoalesced%s"%tc]))
        return False
      if totalElementsPerp % state["NumLoadsPerpendicular%s"%tc] != 0:
        reject(state, "%s: totalElementsPerp %u %% numLoadsPerp %u != 0" \
              % (tc, totalElementsPerp, state["NumLoadsPerpendicular%s"%tc]))
        return False

    if state["ProblemType"]["TLU%s"%tc]:
      state["LSC%s"%tc] = state["MacroTile%s"%tc] // state["NumLoadsCoalesced%s"%tc]
      state["LSP%s"%tc] = int(math.ceil(float(state["DepthU"]) / state["NumLoadsPerpendicular%s"%tc]))
    else:
      state["LSC%s"%tc] = int(math.ceil(float(state["DepthU"]) / state["NumLoadsCoalesced%s"%tc]))
      state["LSP%s"%tc] = state["MacroTile%s"%tc] // state["NumLoadsPerpendicular%s"%tc]

    if state["WaveSeparateGlobalRead%s"%tc]:
      state["LSP%s"%tc] = roundupRatio(state["LSP%s"%tc], state["NumThreads"] // state["WavefrontSize"])

    return True


  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalesced*]
  #   state[NumLoadsPerpendicular*]
  #   state[LSC*]
  #   state[LSP*]
  #   state[GlobalLoadVectorWidthA/B]
  #
  # LSC and LSP define the shape of the PerLoadTile, measured in elements.
  #   LSC*LSP is the elements loaded by a single instruction across all
  #   threads in the group.
  #   LSC is the number of elements loaded in the para(coalesced) dimension
  #   LSP is the number of elements loaded in the perp(noncoalesced) dimension
  #   PerLoadTile is always rectangular.
  #   When BufferLoad=1, the area (LSC*LSP) can be larger than NumThreads.
  #   In this case, some threads will generate a dummy OOB GRO.
  #   Related fields:
  #     LVC = LSC/GRVW  (LVCA = LSCA/GLVWA)
  #     LVP = LSP/GRVW  (LVPA = LSPA/GLVWA)
  #
  # NumLoadsCoalesced and NumLoadsPerpendicular define the number of times the
  #   PerLoadTile is loaded in each dimension to fetch the LoadTile
  # LoadTile = (LSC * NumLoadsCoalesced) * (LSP * NumLoadsPerpendicular).
  #   For Fractional, the LoadTile can be larger than the MacroTile. Buffer
  #   loads will clip any OOB references to 0 and will also avoid writing these
  #   into LDS.

  # Fractional load algorithm:
  #  - Each load instruction loads one or more (complete) rows of the load tile.
  #     - Each row is LSC elements wide
  #     - Rows are complete and do not wrap. This allows a single base GRO VGPR
  #       to be used for all loads in the tile.
  #     - Some work-items in the load may not perform useful work. These WI will
  #       set their GRO to a large OOB number so as to do no harm
  #     - Some G2L registers space may be unused as well.
  #     - The 'used' message at the bottom of this routine computes and prints the
  #       wasted register space.
  #     - The wasted space is removed when the data is written to LDS- the LWO
  #       for work-items beyond the valid ones are set to safely write to OOB locations.

  #     - In cases where each load is loading multiple rows (multiple lines of lsc
  #       elements), the last load is allowed to load fewer lines than the others.
  #       The KernelWriterAssembly will modify the LWO for the last load.  This allows
  #       flexibility in the unroll factors for example.
  @staticmethod
  def setGlobalLoadTileDimFractional(state, tc, depthU, glvwOrig):
    isa = tuple(state["ISA"])

    assert(depthU > 0)
    dbFract = 0

    # parDim, perpDim define the LoadTile and are measured in elements
    if state["ProblemType"]["TLU%s"%tc]:
      parDim  = state["MacroTile%s"%tc]
      perpDim = depthU
    else:
      parDim  = depthU
      perpDim = state["MacroTile%s"%tc]

    if dbFract:
        print("\ninfo: %s Fractional MT%u_%u_%u Par=%u Perp=%u WG%02u_%02u_%02u NumThreads=%u GRWV=%u" \
          % (tc, state["MacroTile0"], state["MacroTile1"], depthU, \
            parDim, perpDim, \
            state["WorkGroup"][0], state["WorkGroup"][1], state["LocalSplitU"], \
            state["NumThreads"], glvwOrig))

    # Try to find a GRVW which is smaller than the LSC and also does not force
    # the LSC to wrap - both of these conditions can be tested with lsc % grvw ==0.
    # Each iteration divides GRWV by 2 which provides finer granularity
    # and a possible opportunity to handle the lsc
    grvw = glvwOrig
    minGrvw = 2 if state["ProblemType"]["DataType"].isHalf() and \
                globalParameters["ArchCaps"][isa]["HasEccHalf"] else 1
    # TODO- check this for int8 and fractional load
    # minGrvw = 4 if state["ProblemType"]["DataType"].isInt8() and \
    #             globalParameters["ArchCaps"][isa]["HasEccHalf"] else 1
    bestVw = -1
    while grvw >= minGrvw:
      # Per instruction across the entire group:
      elementsLoadedPerInst = state["NumThreads"]*grvw
      mik = 1
      if (state["DirectToVgpr%s"%tc] and state["ProblemType"]["TLU%s"%tc]):
        mik = state["MatrixInstK"] * state["LocalSplitU"] // state["MIInputPerThread"]
        elementsLoadedPerInst //= mik
      # LSC, LSP - #elements loaded along specified dim with each load
      if parDim >= elementsLoadedPerInst:
        # entire work-group can work on (part) of the same row
        # DirectToVgpr case, LSC is limited to elementsLoadedPerInst // (state["MatrixInstK"] * state["LocalSplitU"])
        state["LSC%s"%tc] = elementsLoadedPerInst
        state["LSP%s"%tc] = mik
        state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim , state["LSC%s"%tc])
        state["NumLoadsPerpendicular%s"%tc] = 1
      else:
        # work-group exceeds read dimension so wraps to multiple rows
        state["LSC%s"%tc] = parDim
        state["LSP%s"%tc] = min(perpDim, elementsLoadedPerInst // parDim)
        state["NumLoadsCoalesced%s"%tc] = 1
        state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

      # Vector loads can't wrap to next P dim, so LSC must be divisible by vector elements;
      if dbFract:
        print("  lsc search : lsc(%u) %% grvw(%u) = %u (?0)" % (state["LSC%s"%tc], grvw, state["LSC%s"%tc] % grvw))
      if state["LSC%s"%tc] % grvw == 0:
        bestVw = grvw
        # Try to shrink GRVW if possible while keeping same LSC and LSP:
        # For example, avoid cases where we use a GRVW=4 with many empty addresses
        # when a GRVW=1 will do instead.
        validElementsLoadedPerInst = state["LSC%s"%tc] * state["LSP%s"%tc]
        grvw //= 2
        while grvw >= minGrvw:
          elementsLoadedPerInst = state["NumThreads"]*grvw//mik
          if elementsLoadedPerInst < validElementsLoadedPerInst:
            break # Went too far, not enough load elements at this VW
          if state["LSC%s"%tc] % grvw == 0:
            if dbFract:
              print("  stepdown success (valid)elementsLoadedPerInst=", validElementsLoadedPerInst, "/", elementsLoadedPerInst, "grvw=", grvw, "lsc=", state["LSC%s"%tc])
            bestVw = grvw
          grvw //= 2
        break

      # TODO - could have this generate dwordx3 loads in addition, step down by 1 instead of div2
      # Would need to change asm code gen to generate x3
      grvw //= 2
      # end-- while loop

    if bestVw == -1:
      if dbFract:
        print ("reject fractional - no acceptable tile dim? GlobalLoadVectorWidthA/B", \
         glvwOrig)
      return False  # could not find a solution, perhaps only possible for half ?

    state["GlobalLoadVectorWidth%s"%tc] = bestVw
    if bestVw != glvwOrig:
      if dbFract:
        print("  reducing GlobalLoadVectorWidth%s from %u to %u" \
            % (tc, glvwOrig, bestVw))

    # How many loads per threads in each dimension.
    # threads which are outside the global read tile bounds will be clipped
    # in the assembly code generator.
    # Multiply the LSC*GRVW
    state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim, state["LSC%s"%tc])
    state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

    nlc = state["NumLoadsCoalesced%s"%tc]
    nlp = state["NumLoadsPerpendicular%s"%tc]

    # LoadTile must at least cover the MacroTile:
    assert(nlc*state["LSC%s"%tc] >= parDim)
    assert(nlp*state["LSP%s"%tc] >= perpDim)

    perpOverhang = perpDim % state["LSP%s"%tc]
    state["fractionalPerpOverhang%s"%tc] = perpOverhang
    if dbFract:
      # how many threads compute Global Read Offsets (GRO) that are not used
      print("  PerLoadTile=%ux%u elements Loads/WI=%ux%u LoadTile/WI=%ux%u (MT=%ux%u), %u/%u = %.1f%% WI GRO used %s" \
          % (state["LSC%s"%tc], state["LSP%s"%tc], \
             nlc, nlp, \
             nlc*state["LSC%s"%tc], nlp*state["LSP%s"%tc], \
             parDim, perpDim, \
             parDim*perpDim, \
             nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc], \
             float(parDim*perpDim), \
             float(nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc]) * 100.0) \
             )

      for p in range(0,nlp):
        elementWidth = 4
        if p != nlp-1:
          perp = state["LSP%s"%tc]
        else:
          perp = perpOverhang if perpOverhang else state["LSP%s"%tc]

        validElements = state["LSC%s"%tc] * perp
        print("  buffer_load_element_x%u %ux%ux%u bytes,  %u/%u valid GRO" %\
              (state["GlobalLoadVectorWidth%s"%tc], \
              state["LSC%s"%tc], perp, \
              elementWidth, \
              validElements//state["GlobalLoadVectorWidth%s"%tc],
              state["NumThreads"]))

    return True


  @staticmethod
  def parameterWrapper(state):
    isa = tuple(state["ISA"])
    if len(state["MatrixInstruction"]) == 9:
      waves = state["MatrixInstruction"][7]* state["MatrixInstruction"][8]
      state["ThreadTile"][0] = state["MatrixInstruction"][5]
      state["ThreadTile"][1] = state["MatrixInstruction"][6] * state["MatrixInstruction"][1]
      state["WorkGroup"][0] = state["MatrixInstruction"][4] * state["MatrixInstruction"][0] * state["MatrixInstruction"][7]
      state["WorkGroup"][1] = waves*state["WavefrontSize"] // state["WorkGroup"][0]
      #print("9-tuple: ", state["MatrixInstruction"], " TT=", state["ThreadTile"], " WG=", state["WorkGroup"])
    if state["MatrixInstruction"]:
      state["MatrixInstruction"] = [state["MatrixInstruction"][0],state["MatrixInstruction"][1],state["MatrixInstruction"][2],state["MatrixInstruction"][3]]

    if state["MatrixInstruction"] != [] and len(state["MatrixInstruction"]) == 4:
      state["MFMA_BF16_1K"] = False
      if globalParameters["AsmCaps"][isa]["HasMFMA"]:
        miDataType = state["ProblemType"]["DataType"] if (not state["EnableF32XdlMathOp"]) else state["ProblemType"]["F32XdlMathOp"]
        # check if requested MFMA instruction is not in the list of valid instructions
        if not (state["ProblemType"]["DataType"].toChar() in validMFMA and \
          state["MatrixInstruction"] in validMFMA[miDataType.toChar()]):
          # check separate list for B1k instructions
          if miDataType.isBFloat16() and \
            state["MatrixInstruction"] in validMFMA["B1k"]:
            state["MFMA_BF16_1K"] = True
          else:
            reject(state, "MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], miDataType))

        # check if requested instruction is available on current architecture
        if state["ProblemType"]["DataType"].toChar() == 'B':
          if state["MatrixInstruction"] in validMFMA["B"] and not globalParameters["AsmCaps"][isa]["HasMFMA_bf16_original"]:
            reject(state, "MatrixInstruction %s not available on %s" % (state["MatrixInstruction"], isa))
          if state["MatrixInstruction"] in validMFMA["B1k"] and not globalParameters["AsmCaps"][isa]["HasMFMA_bf16_1k"]:
            reject(state, "MatrixInstruction %s not available on %s" % (state["MatrixInstruction"], isa))
        elif state["ProblemType"]["DataType"].toChar() == 'I8':
          if globalParameters["AsmCaps"][isa]["HasMFMA_i8_908"] and state["MatrixInstruction"] not in validMFMA["I8_908"]:
            reject(state, "MatrixInstruction %s not available on %s" % (state["MatrixInstruction"], isa))
          if globalParameters["AsmCaps"][isa]["HasMFMA_i8_940"] and state["MatrixInstruction"] not in validMFMA["I8_940"]:
            reject(state, "MatrixInstruction %s not available on %s" % (state["MatrixInstruction"], isa))

      elif globalParameters["AsmCaps"][isa]["HasWMMA"]:
        if state["MatrixInstruction"] not in validWMMA:
          reject(state, "MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], state["ProblemType"]["DataType"]))

      if (state["ThreadTile"][1] % state["MatrixInstruction"][0]) != 0:
        reject(state, "invalid ThreadTile1 %u for MatrixInstM %u" % (state["ThreadTile"][1], state["MatrixInstruction"][0]))

      # set EnableMatrixInstruction
      state["EnableMatrixInstruction"] = True

      # set MIBlock
      miwg0      = state["MatrixInstruction"][0] if (state["WorkGroup"][0] < state["MatrixInstruction"][0]) else state["WorkGroup"][0]
      MIBlock_BM = miwg0 // state["MatrixInstruction"][0]
      MIBlock_BM = min(MIBlock_BM, state["MatrixInstruction"][3])
      MIBlock_BN = state["MatrixInstruction"][3] // MIBlock_BM

      state["MIBlock"]    = [32, 32, 2, 1, 1, 1]
      state["MIBlock"][0] = state["MatrixInstruction"][0]
      state["MIBlock"][1] = state["MatrixInstruction"][1]
      state["MIBlock"][2] = state["MatrixInstruction"][2]
      state["MIBlock"][3] = state["MatrixInstruction"][3]
      state["MIBlock"][4] = MIBlock_BM
      state["MIBlock"][5] = MIBlock_BN

      # set MIWaveGroup
      numOfWave                = max((state["WorkGroup"][0] * state["WorkGroup"][1]) // state["WavefrontSize"], 1) # should be >0
      state['MIWaveGroup']     = [1, 1]
      state['MIWaveGroup'][0]  = min((miwg0 // state["MatrixInstruction"][0]) // MIBlock_BM, numOfWave)
      state['MIWaveGroup'][1]  = numOfWave // state['MIWaveGroup'][0]

      # set MIWaveTIle
      state['MIWaveTile']      = [1, 1]
      state['MIWaveTile'][0]   = state["ThreadTile"][0]
      state['MIWaveTile'][1]   = state["ThreadTile"][1] // state["MatrixInstruction"][1]

      # set MIInputPerThread
      isa = tuple(state["ISA"])
      state['MIInputPerThread'] = state["MatrixInstruction"][0] * state["MatrixInstruction"][2] * state["MatrixInstruction"][3] // state["WavefrontSize"]
      if (not globalParameters["AsmCaps"][isa]['HasMFMA']) and globalParameters["AsmCaps"][isa]['HasWMMA']:
        state['MIInputPerThread'] = state["MatrixInstruction"][2]

    else:
      state["EnableMatrixInstruction"] = False


  ##############################################
  # check and calculate Wave Separate Global Read
  @staticmethod
  def checkAndAssignWaveSeparateGlobalRead(state, tc):
    # check can we use WaveSeparateGlobalRead
    numOfWaves = state["NumThreads"] // state["WavefrontSize"]
    if state["WaveSeparateGlobalRead%s"%tc]:
      if state["FractionalLoad"] != 0:
        reject(state, "didn't support WaveSeparateGlobalRead with FractionalLoad(%u) != 0" % state["FractionalLoad"])
      if state["LocalDotLayout"]>1:
        reject(state, "didn't support WaveSeparateGlobalRead when LocalDotLayout(%u) > 1" % state["LocalDotLayout"])
      if state["ProblemType"]["TLU%s"%tc] and (state["DepthU"] > 0) and (state["DepthU"] % numOfWaves != 0):
        reject(state, "didn't support WaveSeparateGlobalRead when DepthU is not multiple of wave %u in TLU%s" % (state["DepthU"], tc))
      if not state["ProblemType"]["TLU%s"%tc] and (state["MacroTile%s" % tc] % numOfWaves != 0):
        reject(state, "didn't support WaveSeparateGlobalRead when MacroTile is not multiple of wave %u in TLU%s" % (state["MacroTile%s"%tc], tc))


  ########################################
  # determine can we use VgprForLocalReadPacking
  @staticmethod
  def isVgprForLocalReadPackingDoable(state):
    isa = tuple(state["ISA"])
    rejectComment = ""
    doable = True
    # MatrixInstruction only
    if not state["EnableMatrixInstruction"]:
      rejectComment = "VgprForLocalReadPacking is for MatrixInstruction only"
      doable = False
    # only for HasEccHalf
    if not globalParameters["ArchCaps"][isa]["HasEccHalf"]:
      rejectComment = "VgprForLocalReadPacking is for EccHalf only"
      doable = False
    # only for SIA=3 + PLR>=1
    if not (state["ScheduleIterAlg"] == 3 and state["PrefetchLocalRead"] >= 1):
      rejectComment = "VgprForLocalReadPacking is effective only fof SIA=3 and PLR>=1"
      doable = False
    # only for 1 or 2 byte input (numRegister < 1) + UnrollMajorLDSA or B is False
    if not (state["ProblemType"]["DataType"].numRegisters() < 1 and (state["UnrollMajorLDSA"] == False or state["UnrollMajorLDSB"] == False)):
      rejectComment = "VgprForLocalReadPacking is effective only fof 1 or 2 byte input + UnrollMajorLDSA or B =false"
      doable = False
    return doable, rejectComment

  ########################################
  # determine if current datatype can support DirectToVgpr
  @staticmethod
  def isDirectToVgprSupportDataType(state):
    return (state["ProblemType"]["DataType"].isSingle() or state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isComplex() or \
            state["ProblemType"]["DataType"].isHalf() or state["ProblemType"]["DataType"].isBFloat16() or state["ProblemType"]["DataType"].isInt8()) or \
            state["ProblemType"]["DataType"].is8bitFloat()

  ########################################
  # determine can we use DirectToVgpr
  @staticmethod
  def isDirectToVgprDoable(state, tc):
    MIindex = 0 if tc == 'A' else 1
    numBytes = state["ProblemType"]["DataType"].numBytes()
    # Does not support DirectToVgprA+DirectToVgprB+PrefetchGlobalRead=2
    # Need more than double-vgpr buffers to avoid overwritting loaded data on vgpr
    if state["DirectToVgprA"] and state["DirectToVgprB"] and state["PrefetchGlobalRead"]==2:
      reject(state, "DirectToVgprA + DirectToVgprB + PrefetchGlobalRead=2 is not supported")
      return False

    # With MatrixInstruction only (tentative)
    if not state["EnableMatrixInstruction"] :
      reject(state, "DirectToVgpr is for MatrixInstruction only")
      return False

    # check if the DataType can support DirectToVgpr
    if not Solution.isDirectToVgprSupportDataType(state):
      reject(state, "so far, DirectToVgpr is for single, double or double complex only")
      return False

    # Does not work with TLU = False and PrefetchLocalRead = 0
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchLocalRead"] == 0:
      reject(state, "DirectToVgpr%c does not supports TLU%c = False and PrefetchLocalRead = 0"%(tc, tc))
      return False

    # Does not work with TLU = False and SGEMM/CGEMM (not supported)
    if (not state["ProblemType"]["TLU%c"%tc]) and (state["ProblemType"]["DataType"].isSingle() or state["ProblemType"]["DataType"].isSingleComplex()):
      reject(state, "DirectToVgpr%c does not supports TLU%c = False + SGEMM/CGEMM"%(tc, tc))
      return False

    # numBytes < 4 case
    if numBytes < 4:
      # Does not work with TLU = True and numBytes < 4 (not supported)
      if state["ProblemType"]["TLU%c"%tc]:
        doable, _ = Solution.isVgprForLocalReadPackingDoable(state)
        if numBytes * state["VectorWidth%s"%tc] >= 4 and doable:
          # use pack logic (with v_perm) same as local read (only if VgprForLocalReadPacking is doable)
          # numBytes * VW should be 4 or larger
          # force VgprForLocalReadPacking + ClusterLocalRead
          state["VgprForLocalReadPacking"] = True
          state["ClusterLocalRead"] = True
        else:
          reject(state, "DirectToVgpr%c does not supports TLU%c = True + numByte < 4"%(tc, tc))
          return False

    # MIWaveGroup, MatrixInstBM,BN check
    #  for A, MIWaveGroup[1] and MatrixInstBN should be 1
    #  for B, MIWaveGroup[0] and MatrixInstBM should be 1
    # This is to limit the number of Vgpr
    if tc == 'A' and not (state['MIWaveGroup'][1] == 1 and state['MatrixInstBN'] == 1):
      reject(state, "MIWaveGroup[1] and MatrixInstBN should be 1 for DirectToVgprA. Current value is [%d, %d]"%(state['MIWaveGroup'][1], state['MatrixInstBN']))
      return False
    if tc == 'B' and not (state['MIWaveGroup'][0] == 1 and state['MatrixInstBM'] == 1):
      reject(state, "MIWaveGroup[0] and MatrixInstBM should be 1 for DirectToVgprB. Current value is [%d, %d]"%(state['MIWaveGroup'][0], state['MatrixInstBM']))
      return False

    # Does not work with WaveSeparateGlobalRead
    if state["WaveSeparateGlobalRead%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports WaveSeparateGlobalRead%c"%(tc, tc))
      return False

    # Does not work with TLU and NumLoadsCoalesced != MIWaveTile / GlobalLoadVectorWidth
    # (only for FractionalLoad = False)
    if state["FractionalLoad"] == False:
      if state["ProblemType"]["TLU%s"%tc] and state["NumLoadsCoalesced%c"%tc] != state['MIWaveTile'][MIindex] / state["GlobalLoadVectorWidth%c"%tc]:
        reject(state, "DirectToVgpr%c does not supports NumLoadsCoalesced%c(=%u) != MIWaveTile[%u](=%u) / GlobalLoadVectorWidth%c(=%u)"\
                       %(tc, tc, state["NumLoadsCoalesced%c"%tc], MIindex, state['MIWaveTile'][MIindex], tc, state["GlobalLoadVectorWidth%c"%tc]))
        return False
    # Does not work with MIWaveTile < VectorWidth
    if state['MIWaveTile'][MIindex] < state["VectorWidth%s"%tc]:
      reject(state, "DirectToVgpr%c does not supports MIWaveTile[%u](=%u) < VectorWidth%s(=%u)"\
                     %(tc, MIindex, state['MIWaveTile'][MIindex], tc, state["VectorWidth%s"%tc]))
      return False

    # Does not work with ExpandPointerSwap = False
    if not state["ExpandPointerSwap"]:
      reject(state, "DirectToVgpr%c does not supports ExpandPointerSwap = False"%(tc))
      return False

    # Does not work with TLU + VectorWidth != GlobalLoadVectorWidth (VW = 2 + GRVW = 1 or VW = 1 + GRVW = 2 does not work)
    if state["ProblemType"]["TLU%c"%tc] and state["VectorWidth%s"%tc] != state["GlobalLoadVectorWidth%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports TLU + VectorWidth%s(=%u) != GlobalLoadVectorWidth%c(%u)"%(tc, tc, state["VectorWidth%s"%tc], tc, state["GlobalLoadVectorWidth%c"%tc]))
      return False

    # Does not work with FractionalLoad and (not TLU)
    if state["FractionalLoad"] and (not state["ProblemType"]["TLU%c"%tc]):
      reject(state, "DirectToVgpr%c does not supports FractionalLoad + TLU=False"%(tc))
      return False

    # Does not work with TLU=False and NumLoadsCoalesced != DepthU//(MatrixInstK*GRVW*LSU//MIInputPerThread)
    if (not state["ProblemType"]["TLU%c"%tc]) and \
        state["NumLoadsCoalesced%c"%tc] != state["DepthU"] // (state["MatrixInstK"] * state["GlobalLoadVectorWidth%c"%tc] * state["LocalSplitU"] // state["MIInputPerThread"]):
      reject(state, "DirectToVgpr%c does not supports TLU=False and NumLoadsCoalesced%c != DepthU//(MatrixInstK*GlobalLoadVectorWidth*LocalSplitU//MIInputPerThread(=%u))"%(tc, tc, state["MIInputPerThread"]))
      return False

    # TLU=False case, need GlobalLoadVectorWidth == LocalReadVectorWidth
    if (not state["ProblemType"]["TLU%c"%tc]) and \
       state["GlobalLoadVectorWidth%c"%tc] != state["LocalReadVectorWidth"]:
      reject(state, "DirectToVgpr%c does not supports TLU=False GlobalLoadVectorWidth%c(%u) != LocalReadVectorWidth(%u)"%(tc, tc, state["GlobalLoadVectorWidth%c"%tc], state["LocalReadVectorWidth"]))
      return False

    # Does not work with TLU=False and PrefetchLocalRead=1 and VectorWidth>1
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchLocalRead"] == 1 and state["VectorWidth%s"%tc] > 1:
      reject(state, "DirectToVgpr%c does not supports TLU=False and PrefetchLocalRead=1 and VectorWidth%s>1)"%(tc, tc))
      return False

    # Does not work with SIA<3 and PGR=2
    if state["ScheduleIterAlg"] < 3 and state["PrefetchGlobalRead"] == 2:
      reject(state, "DirectToVgpr%c does not supports ScheduleIterAlg < 3 and PrefetchGlobalRead==2"%(tc))
      return False

    # Does not work with InnerUnroll>1
    if state["InnerUnroll"]>1:
      reject(state, "DirectToVgpr%c does not supports InnerUnroll>1"%(tc))
      return False

    # Does not work with ThreadSeparateGlobalRead
    if state["ThreadSeparateGlobalRead%s"%tc]:
      reject(state, "DirectToVgpr%c does not supports ThreadSeparateGlobalRead%c"%(tc,tc))
      return False

    # Reject TLU = UnrollMajorLDS (B only)
    if tc == 'B' and (state["ProblemType"]["TLUA"] == state["UnrollMajorLDSA"] or state["ProblemType"]["TLUB"] == state["UnrollMajorLDSB"]):
      reject(state, "DirectToVgpr%c does not supports TLU = UnrollMajorLDS"%(tc))
      return False

    # force GlobalReadCoalesceGroup == TLU in DTV case
    state["GlobalReadCoalesceGroup%s"%tc] = state["ProblemType"]["TLU%c"%tc]

    # Does not work with DirectToLDS
    # -> this will be checked after DirectToLDS doable check is done

    return True

  ########################################
  # determine can we use DirectToLds
  @staticmethod
  def isDirectToLdsDoable(state, tc):
    isa = tuple(state["ISA"])
    numBytes = state["ProblemType"]["DataType"].numBytes()
    asem = state["AssertSummationElementMultiple"]
    gsu = state["GlobalSplitU"]

    # x2/x4 support for directToLds (no longer supported)

    # numelements_perlane = 4/numBytes
    # TN with transposeLDS feature should work as long as state["AssertSummationElementMultiple"] % (numelements_perlane*2) = 0
    #                                                     state["AssertSummationElementMultiple"] % (numelements_perlane*4) = 0

    #NT
    # use only for all precisions (except for bpe > 4)
    #TN
    # use for all precisions (except for bpe > 4) with TransposeLDS=1

    numBytes = state["ProblemType"]["DataType"].numBytes()
    numBytesPerLoad = int(state["GlobalLoadVectorWidth%c"%tc] * numBytes)
    # DTLx2 and DTLx4 for gfx950
    if not globalParameters["ArchCaps"][isa]["HasDTLx4"]:
      if numBytesPerLoad != 4:
        reject(state, "DirectToLds can only be used with buffer loads requiring 1 register")
        return False

    if numBytes < 4:
      # numBytes < 4 and TLU=false case
      # need AssertSummationElementMultiple
      if state["ProblemType"]["TLU%c"%tc] == False and \
         ((asem % gsu != 0) or ((asem//gsu) % (state["GlobalLoadVectorWidth%c"%tc])  != 0)):
        reject(state, "can't use DirectToLds for numBytes < 4 with TLU%c=False and AssertSummationElementMultiple=%u and GlobalSplitU=%u" % (tc, asem, gsu))
        return False
      # numBytes < 4 and TLU=true case
      # need AssertSummationElementMultiple
      afem = "AssertFree0ElementMultiple" if tc == "A" else "AssertFree1ElementMultiple"
      if state["ProblemType"]["TLU%c"%tc] and \
         (state[afem] % state["GlobalLoadVectorWidth%c"%tc]  != 0):
        reject(state, "can't use DirectToLds for numBytes < 4 with TLU%c=True and %s%%GlobalLoadVectorWidth!=0" % (tc, afem))
        return False

    if state["NumThreads"] % state["WavefrontSize"] != 0:
      reject(state, "can't use DirectToLds for NumThreads % WavefrontSize != 0")
      return False

    if state["ProblemType"]["TLU%c"%tc] == state["UnrollMajorLDS%c" % tc]:
      reject(state, "can't use DirectToLds for TLU%c == UnrollMajorLDS%c"%(tc, tc))
      return False

    # avoid picking x2&x4 for precisions < f32/f64 in [ProblemType][TLU] == TRUE
    if not state["EnableMatrixInstruction"]:
      if numBytesPerLoad * state["WavefrontSize"] > 256:
        reject(state, "can't use DirectToLds for not EnableMatrixInstruction and GlobalLoadVectorWidth%c * bpe * WavefrontSize > 256"%tc)
        return False

    if state["WaveSeparateGlobalRead%c" % tc]:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytes != state["WavefrontSize"] * numBytesPerLoad:
        reject(state, "can't use DirectToLds for LSC%c and LSP%c * bpe!= WavefrontSize * GlobalLoadVectorWidth%c * bpe > 4"%(tc, tc, tc))
        return False
    else:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytes != state["NumThreads"] * numBytesPerLoad:
        reject(state, "can't use DirectToLds for LSC%c and LSP%c * bpe != NumThreads * GlobalLoadVectorWidth%c * bpe > 4"%(tc, tc, tc))
        return False

    if (state["LdsBlockSizePerPad%c"%tc] == 0) \
        and (state["LdsPad%c"%tc] != 0):
#        and ((state["LSC%c"%tc] * numBytes) != (state["NumThreads"] * 4)): // TODO:
#        and ((state["LSC%c"%tc] * numBytes) % (state["WavefrontSize"] * 4) != 0):
      reject(state, "can't use DirectToLds for LdsBlockSizePerPad%c == 0 and LdsPad%c != 0"%(tc, tc))
      return False

    if (state["LdsBlockSizePerPad%c"%tc] != 0) \
        and (state["LdsPad%c"%tc] != 0) \
        and (state["LdsBlockSizePerPad%c"%tc] != state["WavefrontSize"] * numBytesPerLoad):
#        and (state["LdsBlockSizePerPad%tc"] % (state["WavefrontSize"] * 4) != 0): // TODO:
      reject(state, "can't use DirectToLds for LdsBlockSizePerPad%c != 0 and LdsPad%c != 0 and \
              LdsBlockSizePerPad%c != WavefrontSize * GlobalLoadVectorWidth%c * bpe"%(tc, tc, tc, tc))
      return False

    # so far, DirectToLds does not work well with PGR=2
    # performance is not good and a lot of ds_read for DTL can cause scheduling issue(need fix)
    # limit this reject condition for numBytes >= 8
    if numBytes >= 8 and state["PrefetchGlobalRead"] == 2 and not (state["DirectToVgprA"] or state["DirectToVgprB"]):
      reject(state, "can't use DirectToLds for PrefetchGlobalRead == 2 without DirectToVgpr")
      return False

    if state["NumLoadsCoalesced%c"%tc] > 1:
      # NumLoadsCoalesced > 1 not working with TLU=False
      if (not state["ProblemType"]["TLU%c"%tc]):
        reject(state, "Can't use NumLoadsCoalesced > 1 with DirectToLds + TLU=False")
        return False
      # Does not work with (NumLoadsCoalesced>1 and UseInstOffsetForGRO) + DGEMM
      if state["ProblemType"]["DataType"].isDouble() and state["UseInstOffsetForGRO"]:
        reject(state, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 and UseInstOffsetForGRO for dgemm"%(tc, tc))
        return False

    # Does not work with PAPMode 1 and (AssertSummationElementMultiple/GlobalSplitU) % (DepthU * 2) != 0
    # This is because DirectToLds use second LDS buffer after Odd exit. It does not match local read at the beginning of PK loop.
    if state["PrefetchAcrossPersistentMode"] == 1 and ((asem%gsu != 0) or ((asem//gsu) % (state["DepthU"] * 2) != 0)):
      reject(state, "DirectToLds%c does not work with PAPMode 1 and (AssertSummationElementMultiple//GlobalSplitU) is not multiple of (DepthU * 2)"%(tc))
      return False

    # Does not work with PrefetchGlobalRead=2 and PrefetchLocalRead=1 (cannot schedule DTL global read after local read)
    if state["PrefetchGlobalRead"] == 2 and state["PrefetchLocalRead"] == 1:
      reject(state, "DirectToLds%c does not work with PrefetchGlobalRead=2 and PrefetchLocalRead=1"%(tc))
      return False

    # Does not work with PrefetchGlobalRead=2 and numBytes < 4 and TLU=True
    # local reads for high portion can be scheduled after DirectToLds load
    if state["PrefetchGlobalRead"] == 2 and numBytes < 4 and state["ProblemType"]["TLU%c"%tc]:
      reject(state, "DirectToLds%c does not work with PrefetchGlobalRead=2 and numBytes < 4 and TLU"%(tc))
      return False

    # Does not work with PrefetchGlobalRead=2 and MatrixInstB > 1
    if state["PrefetchGlobalRead"] == 2 and state["MatrixInstB"] > 1:
      reject(state, "DirectToLds%c does not work with PrefetchGlobalRead=2 and MatrixInstB > 1"%(tc))
      return False

    # DirectToLds does not work if MacroTile is not power of 2
    # LDS offset swap/rotate logic works only when MacroTile is power of 2
    mt = state["MacroTile%c"%tc]
    if mt & (mt - 1) != 0 and state["NumLoadsCoalesced%c"%tc] > 1:
      reject(state, "can't use DirectToLds if MacroTile%s is not power of 2 and NumLoadsCoalesced%s > 1"%(tc,tc))
      return False

    # check for DirectToLds + ThreadSeparateGlobalRead
    if state["ThreadSeparateGlobalRead%c"%tc]:
      if numBytes > 4:
        reject(state, "ThreadSeparateGlobalRead%c + DTL does not work if numBytes(%d) > 4"%(tc, numBytes))
      if state["ProblemType"]["TLU%c"%tc]:
        reject(state, "ThreadSeparateGlobalRead%c does not work with DTL%c + TLU%c"%(tc, tc, tc))
      if state["NumLoadsCoalesced%c"%tc] > 1:
        reject(state, "ThreadSeparateGlobalRead%c does not work with DirectToLds + NumLoadsCoalesced > 1."%(tc))
      if int(state["WavefrontSize"] * state["GlobalLoadVectorWidth%c"%tc]) < state["_DepthULds"] * state["VectorWidth%s"%tc]:
        reject(state, "ThreadSeparateGlobalRead%c does not work with WavefrontSize * GlobalLoadVectorWidth%c < _DepthULds * VectorWidth%s."%(tc, tc, tc))

    # Does not work with LocalSplitU
    if state["LocalSplitU"] > 1:
      reject(state, "DirectToLds%c does not work with LocalSplitU>1"%(tc))
      return False

    return True

  @staticmethod
  def getDivisorName(state, tC):
    if state["GlobalReadCoalesceGroup{}".format(tC)]:
      if state["GlobalReadCoalesceVector{}".format(tC)]:
        divisorName = "LVC{}".format(tC)
      else:
        # Fractional load use the more accurate lsc, multiply by VW later
        divisorName = "LSC{}".format(tC)
    else:
      if state["GlobalReadCoalesceVector{}".format(tC)]:
        divisorName = "LSP{}".format(tC)
      else:
        divisorName = "LVP{}".format(tC)
    return divisorName

  ########################################
  # get number of elements in LDS for each tc (A or B)
  @staticmethod
  def getLdsNumElements(state, tc):
    bpeAB = int(4*state["ProblemType"]["DataType"].numRegisters())
    if state["LdsBlockSizePerPad%s"%tc]:
      padInterval = state["LdsBlockSizePerPad%s"%tc] // bpeAB
    else:
      if state["UnrollMajorLDS%s"%tc]:
        padInterval = state["_DepthULds"]
      else:
        padInterval = state["MacroTile%s"%tc]
    numElements = state["MacroTile%s"%tc] * state["_DepthULds"]
    ldsNumElements = numElements + (numElements // padInterval) * (state["LdsPad%s"%tc])
    if state["DirectToVgpr%s"%tc]:
      # DirectToVgpr does not use LDS. Set to 0.
      ldsNumElements = 0
    return ldsNumElements

  ########################################
  # get number of aligned elements in LDS for each tc (A or B)
  @staticmethod
  def getLdsNumElementsAligned(state, tc):
    ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())
    ldsNumElements = Solution.getLdsNumElements(state, tc)
    ldsNumElementsAligned = roundUpToNearestMultiple(ldsNumElements, ldsAlign)
    return ldsNumElementsAligned


  ########################################
  # determine auto LdsPad and LdsBlockSizePerPad
  @staticmethod
  def ldsPaddingAuto(state, isa):
    # LDS padding
    # Resolve -1 before isDirectToLdsDoable check
    numBytes = state["ProblemType"]["DataType"].numBytes()
    optPad = state["LocalReadVectorWidth"]
    readRegs = state["LocalReadVectorWidth"]*numBytes//4
    if (not globalParameters["AsmCaps"][isa]['HasWMMA']) and readRegs > 4:
      reject(state, "LocalReadVectorWidth=%u results in attemping to read LDS larger than b128, reject")

    autoAdjusted = {"LdsPadA": False, "LdsPadB": False, "LdsBlockSizePerPadA": False, "LdsBlockSizePerPadB": False}
    for tc in ('A','B'):
      # set pad as readRegs to avoid unaligned read
      idx01 = 0 if tc == 'A' else 1
      charMN = 'M' if tc == 'A' else 'N'
      numBank = 32
      vw = state["VectorWidth%s"%tc]

      # LdsBlockSizePerPad and LdsPad setting
      autoCalcLBSPP = False
      if state["LdsBlockSizePerPad%s"%tc] == -1:
        state["LdsBlockSizePerPad%s"%tc] = 0
        autoCalcLBSPP = True
      autoAdjusted["LdsBlockSizePerPad%s"%tc] = autoCalcLBSPP
      autoCalcLP = False
      if state["LdsPad%s"%tc] == -1:
        autoCalcLP = True
        if state["ProblemType"]["TLU%s"%tc] and (not state["UnrollMajorLDS%s"%tc]):
          state["LdsPad%s"%tc] = 0
        else:
          state["LdsPad%s"%tc] = vw
      autoAdjusted["LdsPad%s"%tc] = autoCalcLP

      if state["EnableMatrixInstruction"]:
        # MI case
        LRstrideLine = 0 # for LBSPP value check
        LRstride = 0
        comment = ""
        depthU = state["_DepthULds"]
        if state["UnrollMajorLDS%s"%tc]:
          LRstrideLine = depthU
          comment = "DepthULds"
          LRstride = LRstrideLine * vw
        else:
          LRstrideLine = state["MacroTile%d"%idx01]
          comment = "MT0"
          if state["MIInputPerThread"] > 1:
            # MIInputPerThread > 1 case, we still need padding to mitigate bank conflict even for non-UnrollMajorLDS case
            LRstride = LRstrideLine * state["LocalReadVectorWidth"]
        # if LRstrideLine is not power of 2, adjust ldsPad at each line (keep LRstride = 0)
        if LRstrideLine <= 0 or (LRstrideLine & (LRstrideLine - 1)) != 0:
          LRstride = 0
          state["LdsBlockSizePerPad%s"%tc] = 0 # force LdsBlockSizePerPad to 0
        # auto calc for LBSPP
        if autoCalcLBSPP and LRstride > 0:
          state["LdsBlockSizePerPad%s"%tc] = max(int(2**(math.ceil(math.log(LRstride * numBytes, 2)))), 128)
        # value check
        if state["LdsBlockSizePerPad%s"%tc]:
          if state["LdsBlockSizePerPad%s"%tc] < LRstrideLine:
            reject(state, "reject: %s %u x bpe > LdsBlockSizePerPad%s %u" % (comment, LRstrideLine, tc, state["LdsBlockSizePerPad%s"%tc]))
        # auto calc for LdsPad
        if autoCalcLP:
          miWidth = state["MatrixInst%s"%charMN] * state["MatrixInstB%s"%charMN]
          if tc == 'B' and state["MatrixInstBM"] > 1:
            # B and state["MatrixInstBM"] > 1 case, BN is not continuous. Use MatrixInstN only
            miWidth = state["MatrixInstN"]
          if (not state["UnrollMajorLDS%s"%tc]):
            extra = 0
            if miWidth < numBank and miWidth * vw < 128 // numBytes:
              extra = (miWidth * vw) % (128 // numBytes)
            if extra:
              divisor = 128//numBytes
              mod = LRstride % divisor
              state["LdsPad%s"%tc] = (divisor + extra - mod) % divisor
          else:
            optPadMN = optPad
            # for readRegs = 1 or 4, we need to double pad for MI16x16xNx1 to avoid bank conflict.
            if miWidth < numBank and (readRegs == 4 or readRegs == 1):
              # UnrollMajorLds and miWidth < 32 case, same M or N location (with K+1(,2,...) is accessed
              # We need to offset (32//miWidth) times for the next M or N
              #  ex.) MI16x16, miWidth=16. To avoid Bank conflict, we need to double padding for M1.
              #        [M0,K0]. [M0,K1]
              #        offset K0 and K1 -> [M1,K0]. [M1,K1]
              #                         offset K0 and K1 -> [M2,K0]. [M2,K1]
              #                                           offset K0 and K1 -> [M3,K0]. [M3,K1]
              optPadMN = optPad * (numBank//miWidth)
            optPadMN = max(state["GlobalLoadVectorWidth%s"%tc], optPadMN)
            # if depthU is not power of 2, adjust ldsPad to make depthU part multiple of 128/numBytes
            if depthU > 0 and (depthU & (depthU - 1)) != 0:
              divisor = 128//numBytes
              ratio = roundupRatio(128//numBytes, depthU)
              extraPadDU = (ratio * depthU) % (128//numBytes)
              optPadMN_Plus = (extraPadDU + optPadMN) % (128//numBytes)
              optPadMN_Minus = (extraPadDU - optPadMN) % (128//numBytes)
              optPadMN = min(optPadMN_Plus, optPadMN_Minus)
            state["LdsPad%s"%tc] = optPadMN
        ## turn-off padding for directToLds
        if state["DirectToLds%s"%tc]:
          state["LdsPad%s"%tc] = 0

      else:
        # non MI case
        if state["UnrollMajorLDS%s"%tc]:
          reject(state, "didn't support UnrollMajorLDS in VALU mode yet")
        if state["LdsBlockSizePerPad%s"%tc] != 0:
          reject(state, "didn't support LdsBlockSizePerPad in VALU mode yet")

      assert(state["LdsPad%s"%tc] >= 0)

      # DirectToVgpr case, set LdsPad to 0
      if state["DirectToVgpr%s"%tc]:
        state["LdsPad%s"%tc] = 0

      # set LdsBlockSizePerPad = 0 if LdsPad is 0
      if state["LdsPad%s"%tc] == 0:
        state["LdsBlockSizePerPad%s"%tc] = 0

    # LDS size check for auto adjustment
    # if LDS size is over the limit, change LdsPad to 0
    # check UnrollMajorLDS=False for A, B first, then true
    for umlds in [False, True]:
      for tc in ('A','B'):
        # A side (aligned) + B side (not aligned)
        ldsNumElementsAB = Solution.getLdsNumElementsAligned(state, 'A') + \
                           Solution.getLdsNumElements(state, 'B')
        if state["1LDSBuffer"] != 1:
          # not 1LDSBuffer case, double num element
          ldsNumElementsAB *= 2
        if autoAdjusted["LdsPad%s"%tc] and (ldsNumElementsAB * numBytes) > globalParameters["MaxLDS"]:
          # auto adjusted LdsPad and LDS overflow
          if state["UnrollMajorLDS%s"%tc] == umlds:
            # change LdsPad and LdsBlockSizePerPad to 0
            state["LdsPad%s"%tc] = 0
            state["LdsBlockSizePerPad%s"%tc] = 0

  ########################################
  # assign all derived parameters
  @staticmethod
  def assignDerivedParameters(state):
    isa = tuple(state["ISA"])

    state["EnableF32XdlMathOp"] = False #ignore the F32 xDL MathOp by default.
    #enable F32 xDL MathOp only when the input type is f32.
    if "F32XdlMathOp" in state["ProblemType"] \
       and (not state["ProblemType"]["F32XdlMathOp"].isSingle()) \
       and (state["ProblemType"]["DataType"].isSingle()):
      state["EnableF32XdlMathOp"] = True

    Solution.parameterWrapper(state)

    Solution.assignProblemIndependentDerivedParameters(state)

    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    for s in Solution.InternalKeys:
        state['_'+s] = state[s]
        #del state[s]

    if ("_GlobalAccumulation" not in state) or ("_WorkspaceSizePerElemC" not in state):
      state["_GlobalAccumulation"] = None
      state["_WorkspaceSizePerElemC"] = 0

      if state["StreamK"] > 0 and state["StreamKAtomic"] == 0:
        # StreamK Workspace size
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        state["_GlobalAccumulation"] = 'PartialsBuffer'
        state["_WorkspaceSizePerElemC"] = computeBytes

      if state["GlobalSplitU"] > 1:
        computeName  = state["ProblemType"]["ComputeDataType"].toName()
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()

        if state["GlobalSplitUAlgorithm"] == 'SingleBuffer':
          # For SingleBuffer algorithm, _GA and _WorkspaceSizePerElemC is updated only if the gemm function is HPA (excluding int8).
          # The workspace is used to convert the final output from ComputeDataType to DestDataType.
          # If ComputeDataType and DestDataType are the same ,the _GA and _Workspace remain unchanged.
          # for HPA cases with ComputeDataType!=DestDataType : HHS/BBS (_GA should be singlebuffer for these types)
          if (computeName != state["ProblemType"]["DestDataType"].toName()):
            state["_GlobalAccumulation"] = 'SingleBuffer'
            state["_WorkspaceSizePerElemC"] = computeBytes
        elif state["GlobalSplitUAlgorithm"] == 'MultipleBuffer':
          state["_GlobalAccumulation"] = 'MultipleBuffer'
          state["_WorkspaceSizePerElemC"] = computeBytes * state["GlobalSplitU"]

    if state["StreamK"] != 0:
      if state["MIWaveGroup"][0] * state["MIWaveGroup"][1] != 4:
        reject(state, "Stream-K requries MIWaveGroup0*MIWaveGroup1=4")
      if state["EnableMatrixInstruction"] and globalParameters["AsmCaps"][isa]["HasWMMA"]:
        reject(state, "Stream-K untested with WMMA")
      if state["GlobalSplitU"] > 1:
        reject(state, "Cannot enable both Stream-K and GSU")
      if state["PersistentKernel"]:
        reject(state, "Cannot enable both Stream-K and PersistentKernel")
      if not state["ProblemType"]["StridedBatched"]:
        reject(state, "General batch not supported with Stream-K")
      if state["StreamKXCCMapping"] > 0:
        if isa != (9,4,2):
          reject(state, "XCC mapping currently only on gfx942")
      if state["StreamKAtomic"] == 1:
        if not state["ProblemType"]["DataType"].isSingle():
          reject(state, "Atomic Stream-K currently only tested for SGEMM")
        if not state["BufferStore"]:
          reject(state, "Atomic Stream-K requires BufferStore")
        if state["LocalSplitU"] > 1:
          reject(state, "Atomic Stream-K not working with LocalSplitU")
    else:
      # If not using StreamK, set StreamKAtomic to 0 to avoid possibility of duplicate kernels
      state["StreamKAtomic"] = 0
      state["StreamKXCCMapping"] = 0
      # If not using StreamK, clear debug modes to avoid duplicate kernels
      state["DebugStreamK"] = 0

    if state["KernelLanguage"] == "Assembly" and not state["BufferLoad"]:
      # StaggerU only works with source kernels, or with BufferLoad.
      # Since StaggerU defaults to 32, override it to 0 if not supported.
      state["StaggerU"] = 0

    def supportsPreloadKernelArguments():
      if not globalParameters["AsmCaps"][isa]["KernargPreloading"]:
        # force to disable preloading (instead of rejecting kernel)
        state["PreloadKernelArguments"] = 0
        return False, f"{isa} doesn't support preloading."

      if not state["ProblemType"]["StridedBatched"]:
        return False, "Preloading only supported in StridedBatched mode."

      dt = state["ProblemType"]["DataType"]
      if not (dt.isSingle() or dt.isHalf() or dt.isBFloat16()):
        return False, f"Preloading not supported for data type {dt}."

      if not state["ProblemType"]["UseBeta"]:
        return False, "Preloading only supported with UseBeta."

      if state["ProblemType"]["UseInitialStridesAB"]:
        return False, "Preloading not supported with initial strides AB."

      if abs(state["WorkGroupMapping"]) > 1:
        return False, "Preloading not supported with WorkGroupMapping."

      if not state["PrefetchGlobalRead"]:
        return False, "Preloading must use PrefetchGlobalRead."

      if state["EdgeType"] == "ShiftPtr":
        return False, "Preloading not compatible with ShiftPtr."

      if state["StaggerU"] != 0:
        return False, "Preloading not compatible with StaggerU."

      return True, ""

    pkaSupported, pkaMsg = supportsPreloadKernelArguments()
    if state["PreloadKernelArguments"] == -1:
      if pkaSupported:
        state["PreloadKernelArguments"] = 1
      else:
        state["PreloadKernelArguments"] = 0
    elif state["PreloadKernelArguments"] == 1:
      if not pkaSupported:
        reject(state, pkaMsg)

    if "DelayRemainingArguments" in state:
      if state["DelayRemainingArguments"] and state["PreloadKernelArguments"] != 1:
          reject(state, "Delayed kernel arguments only supported when preloading.")
    else:
      state["DelayRemainingArguments"] = False

    if state["VectorStore"] == -1:
        state["_VectorStore"] = 1 # default, may be changed if needed to generate a valid kernel

    ProblemType.assignDerivedParameters(state["ProblemType"])
    if not state["Valid"]:
      tPrint(3, "in assignDerivedParameters, state['Valid'] = False")
      return

    atomic = ((state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer')) or state["AtomicAddC"] or (state["StreamK"] > 0 and state["StreamKAtomic"] == 1)
    if atomic and globalParameters["DebugSkipAtomic"]:
      reject(state, "DEBUG: DebugSkipAtomic enabled, rejecting atomic kernel")
    if not atomic and globalParameters["DebugSkipNonAtomic"]:
      reject(state, "DEBUG: DebugSkipNonAtomic enabled, rejecting non-atomic kernel")

    # Init LoopIters parameter in case of early exit
    # For backwards compatibility with older yaml files
    state["LoopIters"] = 0
    if "LoopUnroll" in state:
      state["LoopIters"] = state["LoopUnroll"]

    if state["ScheduleIterAlg"] == 2:
      state["InnerUnroll"] = state["DepthU"] // state["MatrixInstK"]
      state["PrefetchLocalRead"] = 1
      state["ExpandPointerSwap"] = 1
      state["1LDSBuffer"] = 1
      tPrint(3, "\nSet SIA=2, force PrefetchLocalRead=1, ExpandPointerSwap=1, 1LDSBuffer=1")

    if "MemoryModifierFormat" not in state or state["MemoryModifierFormat"] not in validParameters["MemoryModifierFormat"]:
      if globalParameters["AsmCaps"][isa]["HasGLCModifier"]:
        state["MemoryModifierFormat"] = "GLC"
      else:
        state["MemoryModifierFormat"] = "SC0"

    if ("ForceStoreSC1" not in state) or (state["ForceStoreSC1"] == "Auto") or \
       (state["ForceStoreSC1"] not in validParameters["ForceStoreSC1"]):
      state["ForceStoreSC1"] = globalParameters["ArchCaps"][isa]["ForceStoreSC1"]

    if not globalParameters["AsmCaps"][isa]["HasNTModifier"]:
      # force to disable nt flag if it is not supported by arch
      for ch in ["A", "B", "C", "D"]:
        if state["NonTemporal%s"%ch] >= 4:
          state["NonTemporal%s"%ch] -= 4

    if state["WavefrontSize"] == 32 and not globalParameters["ArchCaps"][isa]["HasWave32"]:
      reject(state, "WavefrontSize=32 not supported for ISA {}".format(isa))

    if state["WavefrontSize"] == 32 and state["KernelLanguage"] == "Source":
      reject(state, "WavefrontSize=32 not yet supported for source kernels.")

    if state["EnableMatrixInstruction"]:
      if not (globalParameters["AsmCaps"][isa]["HasMFMA"] or globalParameters["AsmCaps"][isa]["HasWMMA"]):
        reject(state, f"isa {isa} doesn't support matrix instruction")
        return
      if not (state["ProblemType"]["DataType"].isSingle() \
              or state["ProblemType"]["DataType"].isDouble() \
              or state["ProblemType"]["DataType"].isBFloat16() \
              or state["ProblemType"]["DataType"].isHalf() \
              or state["ProblemType"]["DataType"].isComplex() \
              or state["ProblemType"]["DataType"].is8bitFloat() \
              or state["ProblemType"]["DataType"].isInt8()):
        reject(state, "didn't support Matrix Instruction with type %s" % str(state["ProblemType"]["DataType"]))
        return
      if (not globalParameters["AsmCaps"][isa]["HasMFMA"] and globalParameters["AsmCaps"][isa]["HasWMMA"] and (state["WavefrontSize"] == 64)):
        reject(state, "WMMA only suppport on WGP mode, wave size = 32")
        return
      if not state["MIBlock"] or len(state["MIBlock"]) != 6:
        reject(state, "invalid MIBlock")
        return
      if not state["MIWaveGroup"] or len(state["MIWaveGroup"]) != 2:
        reject(state, "invalid MIWaveGroup")
        return
      if not state["MIWaveTile"] or len(state["MIWaveTile"]) != 2:
        reject(state, "invalid MIWaveTile")
        return
      if globalParameters["AsmCaps"][isa]["HasMFMA"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] \
           and state["ProblemType"]["DataType"].numRegisters() < 1 :
          reject(state, "Matrix instructions for half, bf16, f8, b8 (or i8) types are natively accumulated" + \
           " in fp32 (or i32) precision. Please add the following config:" + \
           "\n - HighPrecisionAccumulate: True")
          return
      if globalParameters["AsmCaps"][isa]["HasWMMA"]:
        if state["ProblemType"]["DataType"].numRegisters() >=1:
          reject(state, "WMMA only supports half, bf16 and i8 types")
          return
      if state["InterleaveAlpha"]:
        reject(state, "Matrix instruction does not support InterleaveAlpha")
        return
    else:
      if not state["ProblemType"]["HighPrecisionAccumulate"] \
         and state["ProblemType"]["ComputeDataType"].numRegisters() > state["ProblemType"]["DataType"].numRegisters():
        reject(state, "For non-MI Kernel, if sizeof(ComputeDataType) > sizeof(DataType), " + \
         "Please add the following config:" + \
         "\n - HighPrecisionAccumulate: True")

      if state["ThreadTile0"] > 16 or state["ThreadTile1"] > 16:
        reject(state, "Invalid value for ThreadTile")

      if state["ScheduleIterAlg"] == 2 or state["ScheduleIterAlg"] == 3:
        reject(state, "SIA2 and SIA3 only support MatrixInstruction")

    if state["ProblemType"]["Tensor0"]==0:
      state["ThreadTileA"] = state["ThreadTile0"]
      state["ThreadTileB"] = state["ThreadTile1"]
      state["SubGroupA"] = state["SubGroup0"]
      state["SubGroupB"] = state["SubGroup1"]
      state["MacroTileA"] = state["MacroTile0"]
      state["MacroTileB"] = state["MacroTile1"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileA"] = state["MIWaveTile"][0]
        state["MIWaveTileB"] = state["MIWaveTile"][1]
    else:
      state["ThreadTileB"] = state["ThreadTile0"]
      state["ThreadTileA"] = state["ThreadTile1"]
      state["SubGroupB"] = state["SubGroup0"]
      state["SubGroupA"] = state["SubGroup1"]
      state["MacroTileB"] = state["MacroTile0"]
      state["MacroTileA"] = state["MacroTile1"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileA"] = state["MIWaveTile"][1]
        state["MIWaveTileB"] = state["MIWaveTile"][0]

    Solution.checkAndAssignWaveSeparateGlobalRead(state, 'A')
    Solution.checkAndAssignWaveSeparateGlobalRead(state, 'B')

    # Init vars early since there are early-exit return statements below
    state["LocalWriteUseSgprA"] = False
    state["LocalWriteUseSgprB"] = False

    if state["KernelLanguage"] != "Assembly" or state["WorkGroupMappingType"] == "Z":
      # WGM < 0 is only for Asm + WorkGroupMappingType==B
      # convert to > 0
      state["WorkGroupMapping" ] = abs(state["WorkGroupMapping"])
    if state["WorkGroupMapping" ] == -1:
      # -1 code is same as 1. Convert to 1.
      state["WorkGroupMapping" ] = 1

    # avoid bug somehow related to GlobalSplitU + Persistent
    # avoid bug related to WGM<0
    # General Batch doesn't support PersistentKernel
    if state["PersistentKernel"] and (\
            (state["KernelLanguage"] == "Assembly" and state["GlobalSplitU"] != 1) or \
            (state["KernelLanguage"] == "Assembly" and state["WorkGroupMapping"] < 0)):
      state["PersistentKernel"] = 0

    if state["PersistentKernelAlongBatch"] and (\
            (state["PersistentKernel"] == 0) or \
            (state["KernelLanguage"] == "Source" and state["GlobalSplitU"] != 1)):
      tPrint(3, "PersistentKernelAlongBatch requires PersistentKernel != 0, forcing PersistentKernelAlongBatch = False")
      tPrint(3, "PersistentKernelAlongBatch not support GSU on HIP, forcing PersistentKernelAlongBatch = False")
      state["PersistentKernelAlongBatch"] = False

    if state["PrefetchAcrossPersistent"]:
      if state["KernelLanguage"] == "Source" or \
         state["PersistentKernel"] == 0 or \
         state["PrefetchGlobalRead"] == 0 or \
         state["SuppressNoLoadLoop"]:
        tPrint(3, "PAP requires Assembly, PK!=0, PGR!=0, SuppressNoLoadLoop=True, forcing PAP=False")
        state["PrefetchAcrossPersistent"] = False
        state["PrefetchAcrossPersistentMode"] = False # PAPM should be 0 here to avoid getting rejected later with a logic file
      if state["PrefetchAcrossPersistentMode"] == 0 and state["PrefetchGlobalRead"] == 2:
        reject(state, "PAPMode 0 does not support PGR=2")
        return
    else:
      if state["PrefetchAcrossPersistentMode"] != 0:
        reject(state, "PAPMode requires PrefetchAcrossPersistent enabled")
        return

    problemType = state["ProblemType"]
    if not problemType["UseInitialStridesAB"]:
      for (tc) in ('A','B'):
        state["AssertStride%sEqual"%tc][0]=1

    if not problemType["UseInitialStridesCD"]:
      for (tc) in ('C','D'):
        state["AssertStride%sEqual"%tc][0]=1

    # Add AssertStride*Equal for PackBatchDims, if needed
    for (mask, tc) in ((0x1,'B'), (0x2,'A')):
      if state["PackBatchDims"] & mask:
        for bi in problemType["IndicesBatch"]:
          state["AssertStride%sEqual"%tc][problemType["IndexAssignments%s"%tc].index(bi)] = 0

    for (tc,batchMask) in (('A', 0x1), ('B', 0x2)):
      freeDims = [i for i in problemType["IndexAssignments%s"%tc] if i in problemType["IndicesFree"]]
      if not freeDims and (not problemType["AllowNoFreeDims"] or not (state["PackBatchDims"] & batchMask)):
        reject(state, "tensor%s contains no free indices.  Set AllowNoFreeDims and PackBatchDims&%s" % (tc, batchMask))
        return False

    # Determine which indices will be packed together as this impacts several different parms (sizes, magic numbers, etc)
    # The order in PackedC*Indices also determines the order that dimensions are packed - the first elements in
    # the list are the fastest-moving elements.
    # The store code optimizes for C0 being the coalesced dimension and C1 the perp dimension.
    # C0/C1 indices can come from IndexAssignmentsA or IndexAssignmentsB
    # grid size [0,1]
    state["PackedC0IdxChars"] = []
    state["PackedC0IndicesX"] = []
    indexChars = globalParameters["IndexChars"]
    # Pack all the dimensions (batch and free) of A into grid[0]

    if problemType["Index0"] in problemType["IndexAssignmentsA"]:
      tc0 = 'A'
      tc1 = 'B'
      batch0Mask = 0x1
      batch1Mask = 0x2
    else:
      tc0 = 'B'
      tc1 = 'A'
      batch0Mask = 0x2
      batch1Mask = 0x1
    assert(isPackedIndex(state, problemType["Index01A"], 0x1))
    assert(isPackedIndex(state, problemType["Index01B"], 0x2))

    # Pack all the dimensions (batch and free) of A into grid[0]
    for idx in problemType["IndexAssignments%s"%tc0]:
      if isPackedIndex(state, idx, batch0Mask):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC0IdxChars"].append("%s" % indexChars[idx])
        state["PackedC0IndicesX"].append(idx)

    state["PackedC1IdxChars"] = []
    state["PackedC1IndicesX"] = []
    for idx in problemType["IndexAssignments%s"%tc1]:
      if isPackedIndex(state, idx, batch1Mask):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC1IdxChars"].append("%s" % indexChars[idx])
        state["PackedC1IndicesX"].append(idx)

    # If dims are packed, then need to ensure a global vector load isn't split by a tensor dim
    # (since this could result in non-contiguous addresses)
    # Current implementation ensures that the vector load is not partial across the Free* boundary:
    # GlobalLoadVectorWidth=1 will always meet this requirement.
    # (TODO - could make this more sophisticated if dims use default strides and are thus contiguous)
    packedC0 = len(state["PackedC0IdxChars"])>1
    packedC1 = len(state["PackedC1IdxChars"])>1

    bufferLoad = state["BufferLoad"] and state["KernelLanguage"] == "Assembly"
    if not bufferLoad:
      state["DirectToLdsA"] = False
      state["DirectToLdsB"] = False
      state["_UseSgprForGRO"] = False
      state["FractionalLoad"] = False

    #These modes only work under certain conditions, apply them here:
    #  - The "NoLoad" loop is only generated if PrefetchGlobalRead>0
    #  - And Suppress does not work if GSU>1 for some reason
    if state["SuppressNoLoadLoop"] == 1:
      if not (bufferLoad and state["PrefetchGlobalRead"] == 1 and (state["GlobalSplitU"]==1)):
        state["SuppressNoLoadLoop"] = 0

    if state["ExpandPointerSwap"] == 1:
      # Pointer swap only used if PGR==1 or (PGR>1 and double/double complex) - so set ExpandPointerSwap=0 here
      # So far, EPS=1 and PGR>1 works only with double/double complex.
      # DirectToVgpr case, bufferLoad=False can work with ExpandPointerSwap=1
      #if not (bufferLoad and state["PrefetchGlobalRead"] == 1):
      if not ((bufferLoad or state["DirectToVgprA"] or state["DirectToVgprB"]) and ( state["PrefetchGlobalRead"] == 1 \
              or (state["PrefetchGlobalRead"] > 1 and Solution.isDirectToVgprSupportDataType(state)))):
        state["ExpandPointerSwap"] = 0
      # EPS not supported with SplitLDS yet
      if state["DepthULdsDivisor"] > 1:
        state["ExpandPointerSwap"] = 0

    # Can optimize preLoop LW Vmcnt only when PAP
    # TODO- less restriction? Haven't tested for not BufferLoad
    state["OptPreLoopVmcnt"] = state["OptPreLoopVmcnt"] and \
                               state["PrefetchAcrossPersistent"]

    #print("PackedC0IdxChars", state["PackedC0IdxChars"])
    #print("PackedC1IdxChars", state["PackedC1IdxChars"])

    if state["StaggerUStride"] == 0 and state["StaggerU"] != 0:
      reject(state, "StaggerUStride={} is only valid if StaggerU is 0 (StaggerU={})".format(state["StaggerUStride"], state["StaggerU"]))
      return

    # Set up stagger shift:
    bpeAB = int(4*state["ProblemType"]["DataType"].numRegisters())
    # (1<<staggerStrideShift) is number of loop iterations to traverse the stride
    if state["StaggerU"] == 0:
      state["StaggerUMapping"] = 0
      state["StaggerUStride"] = 0
    try:
        staggerStrideShift = (int)(math.ceil(math.log(state["StaggerUStride"] / \
                (state["DepthU"] * bpeAB), 2)))
    except ValueError: # i.e., StaggerUStride == 0
        staggerStrideShift = 0
    if staggerStrideShift < 0:
      reject(state, "StaggerUStride=%u is less than size of DepthU=%u * BytesPerElement=%u" \
        % (state["StaggerUStride"], state["DepthU"], bpeAB))
      return
    #print "staggerStrideShift=", staggerStrideShift, "depthu=", state["DepthU"]
    state["_staggerStrideShift"] = staggerStrideShift

    # Use GlobalReadVectorWidth if it is not -1
    if state["GlobalReadVectorWidth"] != -1:
      state["GlobalLoadVectorWidthA"] = state["GlobalReadVectorWidth"]
      state["GlobalLoadVectorWidthB"] = state["GlobalReadVectorWidth"]

    if  state["EnableMatrixInstruction"]:
      # MI case
      # VectorWidth default handling
      if state["VectorWidth"] < 1:
        regPerElem = state["ProblemType"]["DataType"].numRegisters()
        # half: regPE=0.5, vw=2 / int8: regPE=0.25, vw=4
        state["VectorWidth"] = int(1//regPerElem) if (regPerElem < 1) else 1

      if state["VectorWidthB"] < 1:
        regPerElem = state["ProblemType"]["DataType"].numRegisters()
        # half: regPE=0.5, vw=2 / int8: regPE=0.25, vw=4
        state["VectorWidthB"] = int(1//regPerElem) if (regPerElem < 1) else 1
        # state["VectorWidthB"] = 1 # use 1 to avoid existing kernels getting rejected (backward compatibility)
        # DirectToVgpr special case
        # allow LocalReadVectorWidthB > 1 for TLUB + MatrixInstruction (this is applicable for B only)
        # some more limitations necessary to make this logic work
        # - MatrixInstruction
        # - TLUB and not UnrollMajorLDSB
        # - MIInputPerThread == 1
        # - SourceSwap
        # - DirectToVgprB or DirectToVgprA
        # - MIWaveTile1 must be multiple of VectorWidthB
        # need state["LocalReadVectorWidth"] != -1 for DTVA
        # need state["GlobalLoadVectorWidthB"] != -1 for DTVB
        if state["EnableMatrixInstruction"] and \
           state["ProblemType"]["TLUB"] and (not state["UnrollMajorLDSB"]) and \
           state["MIInputPerThread"] == 1 and state["SourceSwap"]:
          if state["DirectToVgprB"] and state["GlobalLoadVectorWidthB"] != -1:
            state["VectorWidthB"] = state["GlobalLoadVectorWidthB"]
          elif state["DirectToVgprA"] and state["ProblemType"]["TLUA"] and state["LocalReadVectorWidth"] != -1:
            state["VectorWidthB"] = state["LocalReadVectorWidth"]

      # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
      if ((state["MIWaveTile"][0] % state["VectorWidth"]) != 0):
        if not state["SourceSwap"]:
          # non SourceSwap case, change VectorWidth to 1 for backward compatibility (before SourceSwap + VectorWidth support)
          state["VectorWidth"] = 1
        else:
          reject(state, "MIWaveTile0(%u) should be multiple of VectorWidth(%u)" % (state["MIWaveTile"][0], state["VectorWidth"]))
          return
      if (state["MIWaveTile"][1] % state["VectorWidthB"]) != 0:
        #reject(state, "MIWaveTile1(%u) should be multiple of VectorWidthB(%u)" % (state["MIWaveTile"][1], state["VectorWidthB"]))
        #return
        # change VectorWidthB to 1 for backward compatibility (before VectorWidthB support)
        state["VectorWidthB"] = 1
      if state["StoreRemapVectorWidth"]:
        # change VectorWidth and VectorWidthB to 1 for backward compatibility (before SourceSwap + VectorWidth support)
        state["VectorWidth"] = 1
        state["VectorWidthB"] = 1

    else:
      # non MI case
      # VectorWidth default handling
      if state["VectorWidth"] < 1:
        state["VectorWidth"] = int(4 / state["ProblemType"]["DataType"].numRegisters())
        while state["ThreadTile0"] % state["VectorWidth"] != 0 \
            or state["ThreadTile1"] % state["VectorWidth"] != 0:
          state["VectorWidth"] //= 2

      # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
      if state["ThreadTile0"] % state["VectorWidth"] != 0 \
          or state["ThreadTile1"] % state["VectorWidth"] != 0:
        reject(state, "ThreadTile0 %u or ThreadTile1 %u not a multiple of VectorWidth %u" \
            % (state["ThreadTile0"], state["ThreadTile1"], \
            state["VectorWidth"]))
        return
      # use VectorWidthB=VectorWidth for non MI case
      state["VectorWidthB"] = state["VectorWidth"]

    # use VectorWidthA internally
    state["VectorWidthA"] = state["VectorWidth"]

    if len(problemType["IndicesSummation"]) > 1:
      # not supported with multiple summations, bug is maybe something with
      # how stagger iteration is wrapped when unroll loop exits
      state["StaggerU"] = 0

    # Some restrictions for half:
    if state["KernelLanguage"] == "Assembly" \
      and state["ProblemType"]["DataType"].isHalf():

      # Vector-width must be at least 2 for Half (since unroll loop uses packed operations?)
      if (not state["EnableMatrixInstruction"]) and state["VectorWidth"] < 2:
        reject(state, "VectorWidth must be >= 2 for half")
      if globalParameters["ArchCaps"][isa]["HasEccHalf"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] and state["AssertFree0ElementMultiple"] % 2 != 0:
          # beta-on-edge has AF0EM requirement except for HPA kernels
          reject(state, "Archs with HasEccHalf require AF0EM%2==0 except for HPA kernels")

    # Some restrictions for int8 and fp8 or bf8:
    if state["KernelLanguage"] == "Assembly" \
        and (state["ProblemType"]["DataType"].isInt8() or state["ProblemType"]["DataType"].is8bitFloat()):
      if (not state["EnableMatrixInstruction"]) and state["VectorWidth"] < 4:
        reject(state, "VectorWidth must be >= 4 for Int8 or 8bitFloat")

    #if state["KernelLanguage"] == "Assembly" and state["PackSummationDims"]:
    #    reject(state, "PackSummationDims does not yet support assembly")

    # Default GlobalLoadVectorWidthA, B
    for tc in ('A','B'):
      if state["GlobalLoadVectorWidth%s"%tc] == -1:
        state["GlobalLoadVectorWidth%s"%tc] = state["VectorWidth"]
        if state["KernelLanguage"] == "Assembly":
          # asm kernel case, GlobalLoadVectorWidth needs to be at least numElemPerReg
          numElemPerReg = int(math.ceil(1 / state["ProblemType"]["DataType"].numRegisters()))
          state["GlobalLoadVectorWidth%s"%tc] = max(state["VectorWidth"], numElemPerReg)
      # reject - GLVWA/B too big
      if (state["GlobalLoadVectorWidth%s"%tc] * state["ProblemType"]["DataType"].numBytes()) > 16:
        reject(state, "GlobalLoadVectorWidth%s * DataType.numBytes() > 16"%tc)
        return

    # Default GlobalStoreVectorWidth
    if state["StoreVectorWidth"] == -1:
      #TODO : re-enable later after running testlists
      #state["StoreVectorWidth"] = state["VectorWidth"]
      # use wider store for best store optimization
      if state["SourceSwap"]:
        state["StoreVectorWidth"] = state["VectorWidth"]
      elif state["ProblemType"]["DataType"].numRegisters() <= 1:
        state["StoreVectorWidth"] = 4
      else:
        state["StoreVectorWidth"] = 4//state["ProblemType"]["DataType"].numRegisters()

    if state["EnableMatrixInstruction"]:
      if state["SourceSwap"]:
        if ((state["VectorWidthA"] % state["StoreVectorWidth"]) != 0):
          reject(state, "MFMA SourceSwap mode doesn't support vw(%u) with svw(%u)" % (state["VectorWidthA"], state["StoreVectorWidth"]))
          return
      else:
        if ((state["MIOutputVectorWidth"] % state["StoreVectorWidth"]) != 0):
          reject(state, "MFMA non-SourceSwap mode doesn't support miovw(%u) with svw(%u)" % (state["MIOutputVectorWidth"], state["StoreVectorWidth"]))
          return

    # reject - VW too big
    if (state["VectorWidthA"] * state["ProblemType"]["DataType"].numBytes()) > 16:
      reject(state, "VWA * DataType.numBytes() > 16")
      return
    if (state["VectorWidthB"] * state["ProblemType"]["DataType"].numBytes()) > 16:
      reject(state, "VWB * DataType.numBytes() > 16")
      return

    # LocalSplitU too large?
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]

    if numElementsPerWorkGroup < state["NumThreads"]:
      reject(state, "NumElementsPerWorkGroup %u < NumThreads %u; reduce LocalSplitU" \
          % (numElementsPerWorkGroup, state["NumThreads"]))
      return

    state["NumElementsPerThread"] = numElementsPerWorkGroup // state["NumThreads"]
    state["GlobalWriteVectorWidth"] = min(state["VectorWidthA"], state["NumElementsPerThread"] )
    if state["NumElementsPerThread"] % state["GlobalWriteVectorWidth"] != 0:
      reject(state, "LSU NumElementsPerThread %u not divisible into GWVW %u" \
          % (state["NumElementsPerThread"], state["GlobalWriteVectorWidth"]))
      return
    state["NumGlobalWriteVectorsPerThread"] = state["NumElementsPerThread"] \
        // state["GlobalWriteVectorWidth"]


    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways store
    if state["LocalSplitU"] > 1:
      if state["NumThreads"] % state["MacroTile0"] != 0:
        reject(state, "LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways store" \
            % (state["NumThreads"], state["MacroTile0"]))
        return
      if state["MacroTile0"]*state["MacroTile1"] % state["NumThreads"] != 0:
        reject(state, "LocalSplitU but MT0*MT1=%u elements doesn't divide into NumThreads=%u" \
            % (state["MacroTile0"]*state["MacroTile1"], state["NumThreads"]))
        return
      if (not state["EnableMatrixInstruction"]) and state["ProblemType"]["DataType"].isInt8():
        reject(state, "int8 doesn't support non-MFMA + LocalSplitU")
        return
      if (not state["EnableMatrixInstruction"]) and state["ProblemType"]["DataType"].is8bitFloat():
        reject(state, "Float8 or BFloat8 doesn't support non-MFMA + LocalSplitU")
        return

    # to eliminate identical/duplicate kernels when GSU=1
    if state["GlobalSplitU"] == 1:
      state["MinKForGSU"] = 256
      state["GlobalSplitUAlgorithm"] = 'SingleBuffer' # force to use SingleBuffer for GSU=1 (instead of rejecting it)
      # GlobalSplitUAtomicAdd is True
      if state["GlobalSplitUAtomicAdd"]:
        reject(state, " GlobalSplitU=1 and GlobalSplitUAtomicAdd=True. Rejecting to avoid duplicate kernels.")

    # GlobalSplitU doesn't work with some other things:
    if state["GlobalSplitU"] > 1:
      if not state["GlobalSplitUSummationAssignmentRoundRobin"] and state["LoopTail"]:
        reject(state, "GlobalSplitU and LoopTail require SummationAssignmentRoundRobin=True since strongly breaks Tensile kernel architecture")
        return

      supported = \
        (state["ProblemType"]["DataType"].isSingle()) or \
        (state["ProblemType"]["DataType"].is8bitFloat()) or \
        (state["ProblemType"]["DataType"].isDouble() and state["BufferStore"]) or \
        (state["ProblemType"]["DestDataType"].isInt32()) or \
        (state["KernelLanguage"] == "Assembly" and state["ProblemType"]["ComputeDataType"].isHalf()) or \
        (state["_GlobalAccumulation"]) or \
        (state["EnableMatrixInstruction"]) # MFMA case, support all data types with BufferStore=0 or 1

      if not supported:
        reject(state, "GlobalSplitU only compatible with single, or asm and (half or mixed) precision, or EnableMatrixInstruction")
        return

      if state["GlobalSplitUAtomicAdd"]:
        # use atomic_add for SingleBuffer algorithm
        # limit to f32 + BufferStore + VAW=1 only
        if not globalParameters["AsmCaps"][isa]["HasAtomicAdd"]:
          reject(state, "GlobalSplitUAtomicAdd is not supported by this arch")
        if state["GlobalSplitUAlgorithm"] != 'SingleBuffer':
          reject(state, "GlobalSplitUAtomicAdd only compatible with SingleBuffer aloghrithm")
        if not state["ProblemType"]["ComputeDataType"].isSingle():
          reject(state, "GlobalSplitUAtomicAdd only compatible with single precision ComputeDataType")
        if not state["BufferStore"]:
          reject(state, "GlobalSplitUAtomicAdd only compatible with BufferStore")
        if state["VectorAtomicWidth"] != 1:
          reject(state, "GlobalSplitUAtomicAdd only compatible with VectorAtomicWidth=1")

        # print warning message if GlobalSplitUAtomicAdd is enabled
        printWarning("Using GlobalSplitUAtomicAdd is not recommended")

    # set minimum and maximum of VectorAtomicWidth
    minVectorAtomicWidth = 2 if (state["ProblemType"]["ComputeDataType"].numBytes() == 2) else 1
    if state["GlobalSplitUAtomicAdd"]:
      maxVectorAtomicWidth = minVectorAtomicWidth
    else:
      # cmpswap_b64 is applicable only for bpe>4 data types due to alignment restriction
      # atomicAdd case, Wdth=1 only.
      # TODO: add VectorAtomicWidth=2 support for smaller data types by introducing alignment assertion

      # maximum is b64 (8 byte)
      #computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
      #maxVectorAtomicWidth = (8 // computeBytes) if computeBytes <= 8 else 1
      maxVectorAtomicWidth = minVectorAtomicWidth

    useAtomic = state["GlobalSplitU"] > 1 and state["GlobalSplitUAlgorithm"] == 'SingleBuffer'
    if state["VectorAtomicWidth"] == -1:
      if useAtomic:
        # atomic case, use max
        state["VectorAtomicWidth"] = maxVectorAtomicWidth
      else:
        # not atomic case, this is not used. Set min
        state["VectorAtomicWidth"] = minVectorAtomicWidth

    if state["VectorAtomicWidth"] < minVectorAtomicWidth:
      reject(state, "VectorAtomicWidth should not be smaller than min(=%u)"%minVectorAtomicWidth)
    if state["VectorAtomicWidth"] > maxVectorAtomicWidth:
      reject(state, "VectorAtomicWidth should not be larger than max(=%u)"%maxVectorAtomicWidth)
    if (not useAtomic) and state["VectorAtomicWidth"] > minVectorAtomicWidth:
      reject(state, "Rejecting (GlobalSplitU=1 or MultipleBuffer) and VectorAtomicWidth>min(=%u) to avoid duplicate kernels."%minVectorAtomicWidth)

    if useAtomic and state["VectorAtomicWidth"] > state["GlobalWriteVectorWidth"]:
      reject (state, "GSU + SingleBuffer + VectorAtomicWidth(%u) > GlobalWriteVectorWidth(%u) not supported"%(state["VectorAtomicWidth"], state["GlobalWriteVectorWidth"]))

    if state["ProblemType"]["DataType"].isHalf() and state["KernelLanguage"] == "Assembly":

      if (not state["EnableMatrixInstruction"]) and state["VectorWidth"] < 2:
        reject(state, "Assembly half requires VectorWidth >= 2 for non-MFMA mode")

      if state["GlobalSplitU"] > 1 and (not state["_GlobalAccumulation"]):
        if state["VectorAtomicWidth"] < 2:
          reject(state, "Assembly GSU half requires VectorWidth >= 2 (for 32-bit CAS)")

        if state["AssertFree0ElementMultiple"] < 2:
          reject(state, "Assembly GSU half requires AF0EM>=2 (for atomics on edge tiles)")

        if state["EnableMatrixInstruction"] and globalParameters["AsmCaps"][isa]['HasWMMA']:
          reject(state, "Half WMMA doesn't support single buffer GSU")
          return

    ########################################
    # Initial DepthU
    ########################################
    userDepthU = state["DepthU"]
    # DepthU == -1 means glvw=1
    if state["DepthU"] == -1:
      if state["MacroTile0"] != state["MacroTile1"]:
        reject(state, "DepthU=0 requires square MacroTile")
        return

    if userDepthU < 0:
      depthU     = 2
      depthULds  = 2
      maxDepthU  = globalParameters["MaxDepthU"]
      numOfWaves = state["NumThreads"] // state["WavefrontSize"]
      if state["ProblemType"]["TLUA"] and state["WaveSeparateGlobalReadA"]:
        depthU = max(depthU, numOfWaves)
      if state["ProblemType"]["TLUB"] and state["WaveSeparateGlobalReadB"]:
        depthU = max(depthU, numOfWaves)
    else:
      depthU = userDepthU
      depthULds = userDepthU//state["DepthULdsDivisor"]
      maxDepthU = userDepthU

    state["_DepthULds"] = state["DepthU"]//state["DepthULdsDivisor"] # internal

    # Default LocalReadVectorWidth
    if state["LocalReadVectorWidth"] == -1:
      if state["EnableMatrixInstruction"]:
        state["LocalReadVectorWidth"] = state["MIInputPerThread"]
      else:
        state["LocalReadVectorWidth"] = state["VectorWidth"]

    # reject if FractionalLoad and depthU is not power of 2 (does not work)
    if state["FractionalLoad"] and (depthU & (depthU - 1)) != 0:
      reject(state, "FractionalLoad requires DepthU = power of 2")

    ########################################
    # Search DepthU
    # Inputs:
    #  - depthU, userDepthU, state["LocalSplitU"], state["InnerUnroll"], state["MacroTile0/1"], state["GlobalLoadVectorWidthA/B"]
    #  - state["MatrixInstK"], ...
    # Outputs:
    #  - totalVectorsCoalescedA, totalVectorsCoalescedB, totalElementsPerpA, totalElementsPerpB, state["DepthU"]
    #######################################
    GlobalLoadVectorWidthAorig = state["GlobalLoadVectorWidthA"] # keep original value
    GlobalLoadVectorWidthBorig = state["GlobalLoadVectorWidthB"] # keep original value
    while True: # exit criteria at end
      validDepthU = True
      # peek LoopIters
      loopIters = (depthULds // state["LocalSplitU"]) // state["InnerUnroll"]
      if "MatrixInstK" in state:
        loopIters //= state["MatrixInstK"]
      if loopIters < 1:
        reject(state, "LoopIters need to greater than 0")
        return

      # Make sure the prefetch VGPR index plr[x] can be aligned for each loop
      # for example, if PLR3 result in 4 VGPR:
      #   PGR  - pre  : plr[0], plr[1], plr[2]
      #   loop - iter0: plr[3], iter1: plr[0], iter2: plr[1], iter3: plr[2] -> restart LOOP (from plr[3]...) -> OK
      #
      # but if PLR2 result in 3 VGPR:
      #   PGR  - pre  : plr[0], plr[1]
      #   loop - iter0: plr[2], iter1: plr[0], iter2: plr[1], iter3: plr[2] -> restart LOOP (from plr[2]...) -> !!
      if (depthULds % ((state["PrefetchLocalRead"]%loopIters)+1)) != 0:
        validDepthU = False

      # how many elements to load
      if state["ProblemType"]["TLUA"]:
        totalElementsCoalescedA = state["MacroTileA"]
        totalElementsPerpA = depthU
      else:
        totalElementsCoalescedA = depthU
        totalElementsPerpA = state["MacroTileA"]

      if state["ProblemType"]["TLUB"]:
        totalElementsCoalescedB = state["MacroTileB"]
        totalElementsPerpB = depthU
      else:
        totalElementsCoalescedB = depthU
        totalElementsPerpB = state["MacroTileB"]

      totalElementsA = totalElementsCoalescedA * totalElementsPerpA
      totalElementsB = totalElementsCoalescedB * totalElementsPerpB

      if state["FractionalLoad"]:
        if not Solution.setGlobalLoadTileDimFractional(state, "A", depthU, GlobalLoadVectorWidthAorig):
          validDepthU = False
        if not Solution.setGlobalLoadTileDimFractional(state, "B", depthU, GlobalLoadVectorWidthBorig):
          validDepthU = False
      else:
        GlobalLoadVectorWidthA = GlobalLoadVectorWidthAorig
        # TODO: Enable DTLx2 and DTLx4 for gfx950
        #if not globalParameters["ArchCaps"][isa]["HasDTLx4"]:
        if state["DirectToLdsA"] and (bpeAB * GlobalLoadVectorWidthA) > 4:
          # bpe * grvw must be <= 4 for DirectToLds (lds flag only for <= 32bit load)
          GlobalLoadVectorWidthA = 4 / bpeAB
          # use float only for <1. Otherwise, convert to int
          if GlobalLoadVectorWidthA >= 1:
            GlobalLoadVectorWidthA = int(GlobalLoadVectorWidthA)
        if not Solution.setGlobalLoadVectorWidth(state, "A", totalElementsA, GlobalLoadVectorWidthA):
          validDepthU = False
        GlobalLoadVectorWidthB = GlobalLoadVectorWidthBorig
        # TODO: Enable DTLx2 and DTLx4 for gfx950
        #if not globalParameters["ArchCaps"][isa]["HasDTLx4"]:
        if (not state["DirectToVgprB"]) and state["DirectToLdsB"] and (bpeAB * GlobalLoadVectorWidthB) > 4:
          # bpe * grvw must be <= 4 for DirectToLds
          GlobalLoadVectorWidthB = 4 / bpeAB
          # use float only for <1. Otherwise, convert to int
          if GlobalLoadVectorWidthB >= 1:
            GlobalLoadVectorWidthB = int(GlobalLoadVectorWidthB)
        if not Solution.setGlobalLoadVectorWidth(state, "B", totalElementsB, GlobalLoadVectorWidthB):
          validDepthU = False

      if validDepthU and state["KernelLanguage"] == "Assembly" \
        and (state["ProblemType"]["DataType"].isHalf() \
              or state["ProblemType"]["DataType"].isBFloat16()):
        if globalParameters["ArchCaps"][isa]["HasEccHalf"]:
          if state["GlobalLoadVectorWidthA"] <= 1 or state["GlobalLoadVectorWidthB"] <= 1:
            reject(state, "HalfEcc requires GLVWA > 1")

      # TODO- Need this restrict ?
      if validDepthU and state["KernelLanguage"] == "Assembly" \
        and (state["ProblemType"]["DataType"].isInt8() or state["ProblemType"]["DataType"].is8bitFloat()):
        if state["GlobalLoadVectorWidthA"] < 4:
          reject(state, "Int8 requires GLVWA >= 4, current is %u"%state["GlobalLoadVectorWidthA"])
        if state["GlobalLoadVectorWidthB"] < 4:
          reject(state, "Int8 requires GLVWB >= 4, current is %u"%state["GlobalLoadVectorWidthB"])


      # Now convert elements to vectors based on GlobalLoadVectorWidth
      GlobalLoadVectorWidthA = GlobalLoadVectorWidthAorig if state["GlobalLoadVectorWidthA"] == 0 else state["GlobalLoadVectorWidthA"]
      GlobalLoadVectorWidthB = GlobalLoadVectorWidthBorig if state["GlobalLoadVectorWidthB"] == 0 else state["GlobalLoadVectorWidthB"]
      totalVectorsCoalescedA = int(totalElementsCoalescedA / GlobalLoadVectorWidthA)
      totalVectorsCoalescedB = int(totalElementsCoalescedB / GlobalLoadVectorWidthB)
      totalVectorsA = int(totalElementsA / GlobalLoadVectorWidthA)
      totalVectorsB = int(totalElementsB / GlobalLoadVectorWidthB)

      if 0:
        print("info:", pvar(state, "NumThreads"), pvar(state, "DepthU"), pvar(state, "DepthULdsDivisor"),
                      "TT=%ux%u" % (state["ThreadTile0"], state["ThreadTile1"]),
                      "WG=%ux%u" % (state["WorkGroup"][0], state["WorkGroup"][1]),
                      "MT=%ux%u" % (state["MacroTile0"], state["MacroTile1"]))
        print("info: totalElementsCoalescedA=", totalElementsCoalescedA,
              " totalVectorsCoalescedA=", totalVectorsCoalescedA, " totalVectorsA=", totalVectorsA)
        print("info: totalElementsCoalescedB=", totalElementsCoalescedB,
              " totalVectorsCoalescedB=", totalVectorsCoalescedB, " totalVectorsB=", totalVectorsB)
        print("info", pvar(state, "VectorWidth")
                , pvar(state, "GlobalLoadVectorWidthA"), pvar(state, "GlobalLoadVectorWidthB"))

      #if state["ProblemType"]["DataType"].isHalf() \
      #    and (state["GlobalLoadVectorWidthA"] == 1 \
      #    or state["GlobalLoadVectorWidthB"] == 1):
      #  validDepthU = False

      if not state["FractionalLoad"]:
        if userDepthU == -1: # no vectors
          if state["GlobalLoadVectorWidthA"] != 1 \
              or state["GlobalLoadVectorWidthB"] != 1:
            validDepthU = False
        elif userDepthU == -2:
          if state["GlobalLoadVectorWidthA"] < GlobalLoadVectorWidthAorig and \
            state["GlobalLoadVectorWidthB"] < GlobalLoadVectorWidthBorig:
            validDepthU = False
        elif userDepthU <= -3:
          if state["GlobalLoadVectorWidthA"] < GlobalLoadVectorWidthAorig or \
            state["GlobalLoadVectorWidthB"] < GlobalLoadVectorWidthBorig:
            validDepthU = False

      if validDepthU:
        if not state["ProblemType"]["TLUA"]:
          if depthU < state["GlobalLoadVectorWidthA"]:
            validDepthU = False

        if not state["ProblemType"]["TLUB"]:
          if depthU < state["GlobalLoadVectorWidthB"]:
            validDepthU = False

      if validDepthU:
        # check depthU and ThreadSeparateGlobalReadA==1 depthU*bpe <= 64 bytes reject ThreadSeparateGlobalRead =1 (TLU=0 only)
        # reject depthU for cases requiring < minimum lanes per fragment. depthU * bpe  must be multiple of cache-line sizes(l2)
        # minimum lane is 4//bpe for bpe < 4, 1 for bpe >= 4
        minLanes = max( 4 // state["ProblemType"]["DataType"].numBytes(), 1)
        depthULds = depthU // state["DepthULdsDivisor"]
        if state["ThreadSeparateGlobalReadA"] and (not state["ProblemType"]["TLUA"]):
          #if state["ThreadSeparateGlobalReadA"] and (((depthU//state["GlobalLoadVectorWidthA"])// (2 * state["ThreadSeparateGlobalReadA"])) < 2):
          if (depthU < minLanes * state["GlobalLoadVectorWidthA"] * (2 * state["ThreadSeparateGlobalReadA"])):
            validDepthU= False
          # reject if KelementsPerMFrag (= (_DepthULds  // (ThreadSeparateGlobalRead * 2))) < inputPerThread = max(lrvwA,lrvwB)
          if (depthULds  // (2 * state["ThreadSeparateGlobalReadA"])) < state["LocalReadVectorWidth"]:
            validDepthU= False
        if state["ThreadSeparateGlobalReadB"] and (not state["ProblemType"]["TLUB"]):
          #if state["ThreadSeparateGlobalReadB"] and (((depthU//state["GlobalLoadVectorWidthB"])// (2 * state["ThreadSeparateGlobalReadB"])) < 2):
          if (depthU < minLanes * state["GlobalLoadVectorWidthB"] * (2 * state["ThreadSeparateGlobalReadB"])):
            validDepthU= False
          # reject if NblockSizePerLoad (= (waveWidth * GlobalLoadVectorWidthB // depthULds)) > MatrixInstN
          if (state["WavefrontSize"] * state["GlobalLoadVectorWidthB"] // depthULds  > state["MatrixInstN"]):
            validDepthU= False
          # reject if KelementsPerMFrag (= (_DepthULds  // (ThreadSeparateGlobalRead * 2))) < inputPerThread = max(lrvwA,lrvwB)
          if (depthULds  // (2 * state["ThreadSeparateGlobalReadB"])) < state["LocalReadVectorWidth"]:
            validDepthU= False

      # this depthU is valid, done unless user wants to double (for TN)
      if validDepthU:
        if userDepthU < -3: # for every int below -3, use next doubled value
          userDepthU += 1
          depthU *= 2
          depthULds = 2
          continue
        else: # use this found value
          state["DepthU"] = depthU
          state["_DepthULds"] = depthU//state["DepthULdsDivisor"]
          break

      # this depthU not valid
      else:
        # keep looking
        if depthU < maxDepthU:
          depthU += 2
          depthULds = depthU//state["DepthULdsDivisor"]
          continue
        # give up
        else:
          reject(state, "No valid DepthU found")
          return
    ########################################
    # end DepthU loop
    ########################################

    assert(state["DepthU"]> 0)

    if state["UnrollIncIsDepthU"] or state["PackSummationDims"] == 1 \
       or bool(problemType["ZeroPadA"]) or bool(problemType["ZeroPadB"]):
        # unrollIncIsDepthU does not support tail loop, so add asem requirement to reject
        # problems that require tail loop.
        if state["DepthU"] % state["AssertSummationElementMultiple"] != 0:
          reject(state, "PackSummationDims=1 requires DepthU is integer multiple of ASEM")
        else:
          state["AssertSummationElementMultiple"] = state["DepthU"]
        # not supported with PSD, has some interaction with iter
        state["StaggerU"] = 0

    if not state["FractionalLoad"]:
      if not Solution.setGlobalLoadTileDimClassic(state, "A", state["NumLoadsA"], \
          totalVectorsCoalescedA, totalElementsPerpA):
        return
      if not Solution.setGlobalLoadTileDimClassic(state, "B", state["NumLoadsB"], \
          totalVectorsCoalescedB, totalElementsPerpB):
        return

    # set UnrollMajorLDSA,B before isDirectToVgprDoable
    state["UnrollMajorLDSA"]     = (state["TransposeLDS"] and (not state["ProblemType"]["TLUA"])) or state["UnrollMajorLDSA"]
    state["UnrollMajorLDSB"]     = (state["TransposeLDS"] and (not state["ProblemType"]["TLUB"])) or state["UnrollMajorLDSB"]

    # TODO
    if (0 and state["LSCA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lsca % grvw != 0")
      return
    if (0 and state["LSPA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lspa % grvw != 0")
      return
    if (0 and state["LSCB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lscb % grvw != 0")
      return
    if (0 and state["LSPB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lspb % grvw != 0")
      return

    state["LVCA"] = roundupRatio(state["LSCA"] , state["GlobalLoadVectorWidthA"])
    state["LVPA"] = roundupRatio(state["LSPA"] , state["GlobalLoadVectorWidthA"])
    state["LVCB"] = roundupRatio(state["LSCB"] , state["GlobalLoadVectorWidthB"])
    state["LVPB"] = roundupRatio(state["LSPB"] , state["GlobalLoadVectorWidthB"])

    for tc in ('A','B'):
      if problemType["TLU%s"%tc]:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["Index01%s"%tc])
      else:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["IndexUnroll"])

      unitStride = False
      stride = -1
      stride = state["AssertStride%sEqual"%tc].get(pos,-1)
      if stride==1:
        unitStride = True
      if not unitStride and state["GlobalLoadVectorWidth%s"%tc] != 1:
        reject(state,
            "Non-unit stride(%s) for coalesced dimension (index=%d) requires GlobalLoadVectorWidth%s==1" \
                % ("TBD" if stride==-1 else str(stride), \
                   problemType["IndexAssignments%s"%tc][pos], tc))

      for p in state["AssertStride%sEqual"%tc].keys():
        if p>len(problemType["IndexAssignments%s"%tc]):
          raise RuntimeError ("AssertStride%sEqual index position %d is > len(IndexAssignments%s)" % \
                                tc, p, tc)

    maxIndex = max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])
    for p in state["AssertSizeEqual"]:
      if p>maxIndex:
        raise RuntimeError ("AssertSize index position=%d is > maxIndex=%d" % (p, maxIndex))


    # Some of these might become 0?
    if 0:
      print("info: ", pvar(state, "LVCA"), pvar(state, "LVPA"), \
            pvar(state, "LVCB"), pvar(state, "LVPB"))

    # lds buffer size for A, B
    if state["KernelLanguage"] == "Source":
      # source kernel
      # use 0 for auto adjust (-1)
      for tc in ('A','B'):
        if state["LdsBlockSizePerPad%s"%tc] == -1:
          state["LdsBlockSizePerPad%s"%tc] = 0
        if state["LdsPad%s"%tc] == -1:
          state["LdsPad%s"%tc] = 0
      if state["LdsPadA"] != state["LdsPadB"]:
        reject(state, "Source KernelLanguage only supports LdsPadA == LdsPadB")
        return

    ########################################
    # LDS
    ########################################
    # allow LocalReadVectorWidthB > 1 for TLUB + MatrixInstruction (this is applicable for B only)
    # some more limitations necessary to make this logic work
    # - MatrixInstruction
    # - TLUB and not UnrollMajorLDSB
    # - MIInputPerThread == 1
    # - SourceSwap
    # - DirectToVgprA
    # need to check after state["LocalReadVectorWidth"] = -1 is resolved
    wideLRVWBforTLUBallowed = False
    if state["EnableMatrixInstruction"] and \
       state["ProblemType"]["TLUB"] and (not state["UnrollMajorLDSB"]) and \
       state["MIInputPerThread"] == 1 and state["SourceSwap"]:
      if state["DirectToVgprA"] and state["ProblemType"]["TLUA"] and state["VectorWidthB"] == state["LocalReadVectorWidth"]:
        wideLRVWBforTLUBallowed = True


    # LocalReadVectorWidth check
    if state["EnableMatrixInstruction"]:
      if state["LocalReadVectorWidth"] < state["MIInputPerThread"]:
        reject(state, "LocalReadVectorWidth < %u" %(state["MIInputPerThread"]))
      if state["LocalReadVectorWidth"] > state["MIInputPerThread"] and not (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]) \
         and not (state["DirectToVgprA"] and state["LocalReadVectorWidth"] == state["VectorWidthB"] and wideLRVWBforTLUBallowed):
        reject(state, "LocalReadVectorWidth require Transpose LDS")
      if state["LocalReadVectorWidth"] > state["MIInputPerThread"] and \
         (state["UnrollMajorLDSA"] and (not state["UnrollMajorLDSB"])) and \
         state["DirectToVgprA"]:
        reject(state, "LocalReadVectorWidth + DirectToVgprA + does not work for TT")
    else:
      if state["LocalReadVectorWidth"] != state["VectorWidth"]:
        reject(state, "LocalReadVectorWidth must equal VectorWidth for non MI kernels")

    # Determine if we can load directly-to-Vgpr
    # need to check after state["LocalReadVectorWidth"] = -1 is resolved
    if state["DirectToVgprA"]:
      if not Solution.isDirectToVgprDoable(state, 'A'):
        return  # rejected
      # disable DTL
      state["DirectToLdsA"] = False
    if state["DirectToVgprB"]:
      if not  Solution.isDirectToVgprDoable(state, 'B'):
        return  # rejected
      # disable DTL
      state["DirectToLdsB"] = False

    # LDS padding
    # Resolve -1 before isDirectToLdsDoable check
    Solution.ldsPaddingAuto(state, isa)

    # Determine if we can load directly-to-LDS.
    # Transpose requires a trip through registers to perform the transpose so can't use DirectToLdsA
    # LDS loads always write 4 bytes apart so can use only 4-byte operations
    # The matrix must not require transposing since that is done by reading to VGPR and writing in different order
    # The LSC (load size coalesced) must load some multiple of 256 bytes since that is what each DirectToLds load provides
    # Note for these matrices LSC is same as MacroTile dim
    # MatrixInstruction rules:
    # DirectToLDS is supported for TLU=0  (make sure transposeLDS=1)
    # LDS (load size coalesced) * LSPA must load some multiple of 256 bytes.
    # No longer support loadX2/loadx4 .
    # need to check after DirectToVgpr check
    for tc in ('A','B'):
      if state["DirectToLds%s"%tc]:
        if Solution.isDirectToLdsDoable(state, tc):
          state["LocalWriteUseSgpr%s"%tc] = True
        else:
          return  # rejected

    if state["1LDSBuffer"] == -1 and (state["DirectToLdsA"] or state["DirectToLdsB"]):
      #1LDS buffer must be 0 for DirectToLdsA
      state["1LDSBuffer"] = 0

    # set NoLdsWriteCode if both A and B use DirectToLds or DirectToVgpr
    state["NoLdsWriteCode"] = False
    if (state["DirectToVgprA"] or state["DirectToLdsA"]) and (state["DirectToVgprB"] or state["DirectToLdsB"]):
      state["NoLdsWriteCode"] = True

    # noTailLoop condition check
    # need to check after DirectToVgpr and DirectToLds check
    # no tail loop optimization setting
    # noTailLoop=1: remove TailLoop
    # noTailLoop=2: remove TailLoop and generate TailLoop in NoLoadLoop with early exit
    # noTailLoop=3: remove TailLoop and generate TailLoop in NoLoadLoop without early exit
    # here, only check condition 1

    # Reject the following cases if noTailLoop is not enabled
    #  - PrefetchAcrossPersistent and PrefetchAcrossPersistentMode
    #    PrefetchAcrossPersistentMode does not support TailLoop (TLU is necessary for NoTailLoop)
    #  - DirectToLds + TLU + NumLoadsCoalesced > 1 (special local read offset conversion is not implemented in tail loop code)
    #  - DirectToLds + LRVW > 1

    # global load width for tail loop (based on AssertFree0, 1 or AssertSummationElementMultiple)
    asem = state["AssertSummationElementMultiple"]
    # need to adjust asem for GSU
    gsu = state["GlobalSplitU"]
    asemDivGSU = 1 if asem%gsu !=0 else asem//gsu

    noTailLoop = 0
    if (asemDivGSU % state["DepthU"] == 0):
      noTailLoop = 1

    # reject conditions for noTailLoop==0
    rejected = False
    rejectMessage = ""
    if state["PersistentKernel"] and state["PrefetchAcrossPersistent"] and state["PrefetchAcrossPersistentMode"] == 1:
      rejectMessage = "PK + PAP + PAPMode"
      rejected = True
    elif (state["DirectToLdsA"] or state["DirectToLdsB"]):
      if (not rejected) and state["EnableMatrixInstruction"] and state["LocalReadVectorWidth"] > state["MIInputPerThread"]:
        rejectMessage = "DirectToLds + LocalReadVectorWidth>MIInputPerThread"
        rejected = True
      for tc in ('A','B'):
        if (not rejected) and state["ProblemType"]["TLU%c"%tc] and state["NumLoadsCoalesced%c"%tc] > 1:
          rejectMessage = "DirectToLds + TLU%c + NumLoadsCoalesced%c>1"%(tc, tc)
          rejected = True

    if noTailLoop == 0 and rejected:
      # if reject condition for NoTailLoop is true and NoTailLoop is not enabled, reject this kernel
      rejectMessage += " requires NoTailLoop."
      rejectMessage += "\n" + "To enable NoTailLoop, "
      rejectMessage += "\n" + " - AssertSummationElementMultiple/GlobalSplitU) is multiple of DepthU or"
      rejectMessage += "\n" + " - BufferLoad and MatrixInstruction + MatrixInstK > 1 and"
      rejectMessage += "\n" + "   (global read width for TailLoop decided by assert is multiple of GlobalLoadVectorWidth) and"
      rejectMessage += "\n" + "   (StaggerU = 0 or NT(+BufferLoad))"
      reject(state, rejectMessage)
      return

    # reject condition for PAPM + PGR=2
    # (need to check after DepthU calculation (for negative value) is done)
    if state["PersistentKernel"] and state["PrefetchAcrossPersistent"] and state["PrefetchAcrossPersistentMode"] == 1 and \
      state["PrefetchGlobalRead"] == 2:
      # PAPM + PGR=2 requires at least 2 loop iterations to execute global read address calculation in NGLL
      # it means K should be >= DepthU * 2 (means larger than DepthU * 2 - 1)
      if not (3 in state["AssertSizeGreaterThan"].keys() and state["AssertSizeGreaterThan"][3] >= state["DepthU"] * 2 - 1):
        reject(state, "PAPM + PGR=2 does not work if AssertSizeGreaterThan for K is not greater than DepthU * 2 - 1")
        return

    #ldsNumElementsA        = Solution.getLdsNumElements(state, 'A') # not used
    ldsNumElementsB        = Solution.getLdsNumElements(state, 'B')
    ldsNumElementsAlignedA = Solution.getLdsNumElementsAligned(state, 'A')
    ldsNumElementsAlignedB = Solution.getLdsNumElementsAligned(state, 'B')

    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumElementsAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumElementsAlignedB
      state["LdsOffsetB"] = state["LdsOffsetA"] + state["LdsNumElementsAlignedA"]

      offsetBlk = state["LdsOffsetB"] + ldsNumElementsAlignedB
      if offsetBlk>0: # need 0 check to avoid an error
        offsetBlk = int(2**(math.ceil(math.log(offsetBlk, 2))))

      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetB_Blk"] = state["LdsOffsetA_Blk"] + state["LdsNumElementsAlignedA"]
      ldsNumElementsAB = state["LdsOffsetB_Blk"]+ ldsNumElementsB
    else:
      state["LdsOffsetB"] = ldsNumElementsAlignedA
      ldsNumElementsAB = ldsNumElementsAlignedA + ldsNumElementsB

    # lds buffer size for reduction
    bytesPerComElem = state["ProblemType"]["ComputeDataType"].numBytes()
    bytesPerLoadElem = state["ProblemType"]["DataType"].numBytes()
    multiplier = bytesPerComElem // bytesPerLoadElem

    ldsNumElementsReduction = multiplier*state["LocalSplitU"]*state["MacroTile0"]*state["MacroTile1"] if state["LocalSplitU"] > 1 else 0

    # lds max occupancy
    ldsSizeOccupancy = globalParameters["DeviceLDS"] // state["MaxOccupancy"]
    ldsNumElementsOccupancy = ldsSizeOccupancy // state["ProblemType"]["DestDataType"].numBytes()

    #print("ldsNumElementsA", ldsNumElementsA)
    #print("ldsNumElementsB", ldsNumElementsB)
    #print("ldsNumElementsAlignedA", ldsNumElementsAlignedA)
    #print("ldsNumElementsAlignedB", ldsNumElementsAlignedB)
    #print("ldsNumElementsAB", ldsNumElementsAB)

    if state["EnableMatrixInstruction"]:
      if (state["DirectToLdsA"] or state["DirectToLdsB"]) and state["1LDSBuffer"]:
        reject(state, "1LDSBuffer must be 0 for directToLds")

    # ThreadSeparateGlobalRead + no DirectToLds case (equivalent to previous SplitGlobalRead)
    # No local read offset conversion in this case. A and B must have equivalent ThreadSeparateGlobalRead setting
    for tc in ('A','B'):
     if (not (state["DirectToLds%s"%tc] and state["ProblemType"]["TLU%s"%tc] == False)) and state["ThreadSeparateGlobalRead%s"%tc]:
       divisorName = Solution.getDivisorName(state, tc)
       divisor = state[divisorName]
       tsgrNum = state["ThreadSeparateGlobalRead%s"%tc] * 2
       if tsgrNum >= divisor:
         reject(state, "ThreadSeparateGlobalRead * 2 (=%d) must be less than lvc/lsc/lvp/lsp"%(tsgrNum))
       if divisor > state["WavefrontSize"]:
         reject(state, "WavefrontSize >= divisor(lvc/lsc/lvp/lsp) is required for ThreadSeparateGlobalRead")
       if state["DirectToVgpr%s"%tc]:
         # set TSGR = 0 if DirectToVgpr is enabled
         state["ThreadSeparateGlobalRead%s"%tc] = 0

    if state["1LDSBuffer"] == -1:
      if ldsNumElementsAB * state["ProblemType"]["DataType"].numBytes() > globalParameters["MaxLDS"]:
        state["1LDSBuffer"] = 1
      else:
        state["1LDSBuffer"] = 0

    if state["1LDSBuffer"]:
      if not state["PrefetchGlobalRead"]:
        reject(state, "PGR=0 already use 1 LDS buffer only")
      # Should be able to support as long as NO scheduleLocalWrite
      if (not state["ScheduleIterAlg"] == 2) and (not state["ScheduleIterAlg"] == 3) and (state["ScheduleLocalWrite"]):
        reject(state, "1LDSBuffer only support SIA2 or SIA3, or SIA1 without SLW")
      state["LdsOffsetB"] = ldsNumElementsAlignedA
      ldsNumElementsAB = ldsNumElementsAlignedA + ldsNumElementsB

    # lds size is the greater of the two
    ldsNumElements = max(ldsNumElementsAB, ldsNumElementsReduction, ldsNumElementsOccupancy)

    if state["StoreRemapVectorWidth"] == -1:
      # use de_read_b64 as default in storeRemap to avoid bank conflict
      defaultRemap = 8 // state["ProblemType"]["DestDataType"].numBytes()
      defaultRemap = max(defaultRemap, state["MacroTile0"]//state["WavefrontSize"])
      ldsRemapPad = max(defaultRemap, state["MIOutputVectorWidth"])
      ldsNumElementsRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]
      if state["_GlobalAccumulation"]:
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        ldsNumElementsRemapC *= (computeBytes / state["ProblemType"]["DestDataType"].numBytes())
      ldsSize = ldsNumElementsRemapC * state["ProblemType"]["DestDataType"].numBytes()
      if not math.log(state["MacroTile0"],2).is_integer() or \
          ldsSize > globalParameters["MaxLDS"] or \
          state["SourceSwap"] or \
          (state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer') or \
          state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4:
        state["StoreRemapVectorWidth"] = 0
      else:
        state["StoreRemapVectorWidth"] = defaultRemap

    # GuaranteeNoPartial
    if state["ProblemType"]["TLUA"]:
      state["GuaranteeNoPartialA"] = state["AssertFree0ElementMultiple"]%state["GlobalLoadVectorWidthA"]==0
    else:
      state["GuaranteeNoPartialA"] = True

    if state["ProblemType"]["TLUB"]:
      state["GuaranteeNoPartialB"] = state["AssertFree1ElementMultiple"]%state["GlobalLoadVectorWidthB"]==0
    else:
      state["GuaranteeNoPartialB"] = True

    # SourceSwap
    if state["SourceSwap"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, "SourceSwap only applies to MatrixInstruction kernels")
        return
      if state["StoreRemapVectorWidth"]:
        reject(state, "SourceSwap not compatible with StoreRemap")
        return
    # non-SourceSwap+MFMA 4x4 check
    if (not state["SourceSwap"]) and state["EnableMatrixInstruction"]:
      if state["MatrixInstBM"] > 1 and state["MatrixInstN"] == 4 and (state["MatrixInstM"] > state["MIOutputVectorWidth"]) and \
        state["AssertFree0ElementMultiple"] % state["GlobalLoadVectorWidthA"] != 0:
        reject(state, "MI4x4 + non-SourceSwap + MatrixInstBM > 1 + MatrixInstN == 4 + MatrixInstM > MIOutputVectorWidth \
                       AssertFree0ElementMultiple %% bGlobalLoadVectorWidthA != 0 not supported")
        return

    # check if need to use lds init Acc vgprs
    state["LdsInitCVgprs"] = False
    if globalParameters["ArchCaps"][isa]["HasAccCD"] and \
         state["EnableMatrixInstruction"] and state["StorePriorityOpt"] and \
         state["ProblemType"]["DataType"].isDouble():
      state["LdsInitCVgprs"] = True

    # force MIArchVgpr when using WMMA
    if state["EnableMatrixInstruction"] and globalParameters["AsmCaps"][isa]["HasWMMA"]:
      state["MIArchVgpr"] = True

    if state["MIArchVgpr"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, "MIArchVgpr only support for MatrixInstruction")
        return
      if not (globalParameters["AsmCaps"][isa]["HasMFMA_vgpr"] or globalParameters["AsmCaps"][isa]["HasWMMA"]):
        reject(state, "MIArchVgpr is not supported by this arch")
        return
      if globalParameters["AsmCaps"][isa]["HasMFMA"]:
        if not (state["ProblemType"]["ComputeDataType"].isDouble() or \
                state["ProblemType"]["ComputeDataType"].isSingle() or \
                (state["ProblemType"]["ComputeDataType"].isHalf() and state["ProblemType"]["HighPrecisionAccumulate"]) or \
                state["ProblemType"]["ComputeDataType"].isInt32() or \
                state["ProblemType"]["ComputeDataType"].isComplex()):
          reject(state, "MIArchVgpr now only support fp64, fp64c, fp32, fp32c, fp16, int8 MatrixInstruction.")
          return
      if state["ProblemType"]["ComputeDataType"].isSingleComplex() and (not globalParameters["AsmCaps"][isa]["v_fma_f32"]):
        reject(state, "MIArchVgpr + fp32c requires v_fma_f32.")
        return

    if state["AtomicAddC"]:
      if not state["ProblemType"]["DataType"].isDouble():
        reject(state, "AtomicAddC currently only available for dgemm")
        return
      if state["AssertBetaValue"] != 1:
        reject(state, "AtomicAddC requires AssertBetaValue = 1")
        return
      if not state["AssertCEqualsD"]:
        reject(state, "AtomicAddC requires AssertCEqualsD")
        return

    if state["ProblemType"]["Fp16AltImpl"]:
      if not (state["ProblemType"]["DataType"].isHalf() and \
              state["ProblemType"]["HighPrecisionAccumulate"] and \
              state["EnableMatrixInstruction"]):
        reject(state, "Fp16AltImpl requires FP16 HPA MFMA")
        return

    if state["ProblemType"]["StochasticRounding"]:
      if not (state["ProblemType"]["DataType"].is8bitFloat()):
        reject(state, "StochasticRounding requires F8 types")
        return

    #check not support cases and calculate lds resources
    if state["StoreRemapVectorWidth"]:
      if not state["BufferStore"]:
        reject(state, "storeRemap only support BufferStore")
        return
      if not state["EnableMatrixInstruction"]:
        reject(state, "storeRemap only support MatrixInstruction kernel")
        return
      if (state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer'):
        reject(state, "storeRemap doesn't support GlobalSplitU yet, except GSU algorithm 2")
        return
      if packedC0 or packedC1:
        reject(state, "storeRemap doesn't support packedC0 and packedC1 yet")
        return
      if state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4:
        reject(state, "storeRemap doesn't support MI4x4 multi blocks in N direction yet")
        return
      if not math.log(state["MacroTile0"],2).is_integer():
        reject(state, "storeRemap only supports power-of-2 MT0")
        # TODO - this return should be here, but this is a hotfix,
        # Somehow we have a "Validation Failed" kernel in rocBLAS now (SRVW=4 and MT0=96) and this will stop the whole building process
        # Actions: 1. Hotfix, comment out this "return" temporarily for that invalidated kernel
        #          2. Remove / replace that invalidated kernel
        #          3. Put back this return
        #          4. How to design a better way to prevent from invalid kernel in rocBLAS?
        # return

      storeInstMinWidth = 1 # minimum dwordx1
      storeInstMaxWidth = 4 # maximum dwordx4
      srMinVw = max(storeInstMinWidth, int(storeInstMinWidth/state["ProblemType"]["DestDataType"].numRegisters()))
      numReg  = state["ProblemType"]["DestDataType"].numRegisters()
      if state["_GlobalAccumulation"]:
        numReg = state["ProblemType"]["ComputeDataType"].numRegisters()

      srMaxVw = int(storeInstMaxWidth/numReg)
      if srMinVw > state["StoreRemapVectorWidth"] or srMaxVw < state["StoreRemapVectorWidth"]:
        reject(state, "StoreRemapVectorWidth %u is not allowed for this data type" % state["StoreRemapVectorWidth"])
        return

      if state["StoreRemapVectorWidth"] * state["WavefrontSize"] < state["MacroTile0"]:
        reject(state, "storeRemap: Per wave single global write instruction doesn't enough to write one M column." + \
               " Please use larger StoreRemapVectorWidth.")
        return
      if (state["MacroTile0"]*state["MatrixInstN"])//state["MIWaveGroup"][0] < state["StoreRemapVectorWidth"]*state["WavefrontSize"]:
        reject(state, "storeRemap: number elements of lds less than per wave per local read elements." + \
               " Please use smaller StoreRemapVectorWidth.")
        return
      ldsRemapPad = max(state["StoreRemapVectorWidth"],state["MIOutputVectorWidth"])
      ldsNumElementsRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]

      if state["_GlobalAccumulation"]:
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        multiplier = computeBytes // state["ProblemType"]["DataType"].numBytes()
      elif state["ProblemType"]["DestDataType"].numBytes() > state["ProblemType"]["DataType"].numBytes():
        # Determine ratio of output to input element size.
        # SRVW remaps output so we need to scale up resources.
        multiplier = state["ProblemType"]["DestDataType"].numBytes() // state["ProblemType"]["DataType"].numBytes()
      else:
        multiplier = 1

      ldsNumElementsRemapC *= multiplier

      #print("ldsNumElementsRemapC=%u" % ldsNumElementsRemapC)

      # if LDS is bound by RemapC (SRVW), then 1LDSBuffer actually doesn't help in SIA3
      # since LDS usage couldn't be reduced
      if state["1LDSBuffer"] and (state["ScheduleIterAlg"] == 3) and (ldsNumElements < ldsNumElementsRemapC):
        # TODO- Remove this DataType test condition,
        # Currently we do this test is just because we don't want to affect existing logic in rocBLAS
        if state["ProblemType"]["DataType"].isInt8() or state["ProblemType"]["DataType"].is8bitFloat():
          reject(state, "LDS usage is bound be StoreRemap, thus 1LDSBuffer wouldn't have any help. Skip.")
          return

      ldsNumElements = max(ldsNumElements, ldsNumElementsRemapC)

    state["LdsNumElements"] = ldsNumElements
    ldsSize = ldsNumElements * state["ProblemType"]["DataType"].numBytes()
    if ldsSize > globalParameters["MaxLDS"]:
      reject(state, "Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      return

    # LoopUnroll  = DepthU / LocalSplitU
    if "LocalSplitU" in state and "_DepthULds" in state:
      state["LoopUnroll"] = state["_DepthULds"] // state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["_DepthULds"]:
      state["Valid"] = False
    if state["KernelLanguage"] != "Assembly" and state["InnerUnroll"] != 1:
      reject(state, "InnerUnroll only supported on assembly")
    state["LoopUnroll"] //= state["InnerUnroll"]

    #constraints for StoreCInUnroll feature
    if state["StoreCInUnroll"]:
      if not (state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isDoubleComplex()):
        reject(state, "StoreCInUnroll currently only available for dgemm/zgemm")
        return
      if state["LocalSplitU"]>1:
        reject(state, "LocalSplitU is not supported for StoreCinUnroll")
        return
      if state["MIArchVgpr"]:
        reject(state, "MIArchVgpr is not supported for StoreCinUnroll")
        return
      if not state["PersistentKernel"]:
        reject(state, "StoreCInUnroll requires PersistentKernel feature")
        return
      if not state["PrefetchAcrossPersistent"]:
        reject(state, "StoreCInUnroll requires PrefetchAcrossPersistent feature")
        return
      if state["PrefetchAcrossPersistentMode"] == 0:
        reject(state, "StoreCInUnroll requires PrefetchAcrossPersistentMode")
        return
      if state["ProblemType"]["DataType"].isDouble() and state["VectorWidthA"] != 2:
        reject(state, "StoreCInUnroll requires VectorWidth=2 for dgemm")
        return
      if state["AtomicAddC"] and state["StoreVectorWidth"] != 1:
        reject(state, "StoreCInUnroll requires AtomicAddC with StoreVectorWidth=1")
        return
      if state["ScheduleGlobalRead"] != 1:
        reject(state, "StoreCInUnroll requires ScheduleGlobalRead=1")
        return
      if state["PrefetchGlobalRead"] == 0:
        reject(state, "StoreCInUnroll requires PrefetchGlobalRead!=0")
        return
      if not state["ExpandPointerSwap"]:
        reject(state, "StoreCInUnroll requires ExpandPointerSwap")
        return
      if state["ScheduleIterAlg"] != 3:
        reject(state, "StoreCInUnroll requires ScheduleIterAlg=3")
        return
      if state['MIWaveGroup'][1] != 1 and state['MIWaveGroup'][1] != 4:
        reject(state, "StoreCInUnroll requires [MIWaveGroup][1]=1 or 4")
        return
      if state["StoreCInUnrollExact"] and state["StoreCInUnrollPostLoop"] :
        reject(state, "StoreCInUnrollPostLoop does not work with StoreCInUnrollExact")
        return
      if not state["SourceSwap"]:
        reject(state, "StoreCInUnroll requires SourceSwap feature")
        return
      if state["ProblemType"]["DataType"].isDouble() and state["NumElementsPerBatchStore"] == 1:
        reject(state, "StoreCInUnroll does not work with NumElementsPerBatchStore = 1 for dgemm")
        return
      if not state["BufferStore"]:
        reject(state, "StoreCInUnroll requires BufferStore feature")
        return
      if state["VectorWidthB"] > 1:
        reject(state, "StoreCInUnroll + VectorWidthB > 1 not supported yet")
        return

      # minimum K check
      # PGR=2 requires minimum K
      if state["PrefetchGlobalRead"] == 2:
        # PGR=2 case, K > DepthU * 2 is necessary
        minDUnum = 2
        if not (3 in state["AssertSizeGreaterThan"].keys() and state["AssertSizeGreaterThan"][3] >= state["DepthU"] * minDUnum):
          reject(state, "StoreCInUnroll does not work if AssertSizeGreaterThan for K is not greater than DepthU * %u"%minDUnum)
          return

      # exact K check
      # StoreCInUnrollExact requires exact K
      if state["StoreCInUnrollExact"]:
        # K == DepthU * ThreadTile0 * ThreadTile1 // VectorWidth is necessary
        exactK = state["DepthU"] * state["ThreadTile0"] * state["ThreadTile1"] // state["VectorWidthA"]
        if not (3 in state["AssertSizeEqual"].keys() and state["AssertSizeEqual"][3] == exactK):
          reject(state, "StoreCInUnrollExact does not work if AssertSizeEqual for K is not DepthU * ThreadTile0 * ThreadTile1 / VectorWidth")
          return

    else:
      # force to disable if StoreCInUnroll related parameter is enabled
      if state["StoreCInUnrollPostLoop"] :
        state["StoreCInUnrollPostLoop"] = False
      if state["StoreCInUnrollExact"] :
        state["StoreCInUnrollExact"] = False

    # check LocalDotLayout
    ldl = state["LocalDotLayout"]
    if ldl> 1:
      state["DirectToLdsA"] = False
      state["DirectToLdsB"] = False

      if state["KernelLanguage"] == "Assembly":
        if state["EnableMatrixInstruction"]:
          reject(state, "doesn't support LocalDotLayout > 1 in MFMA mode")
        else: # VALU mode
          if state["ProblemType"]["DataType"].isInt8():
            if (ldl != 4) or (state["ProblemType"]["HighPrecisionAccumulate"] != True):
              reject(state, "Only support Int8 HPA and LocalDotLayout 4")
              return
          elif state["ProblemType"]["DataType"].is8bitFloat():
            if (ldl != 4) or (state["ProblemType"]["HighPrecisionAccumulate"] != True):
              reject(state, "Only support 8bitFloat HPA and LocalDotLayout 4")
              return
          elif state["ProblemType"]["DataType"].isHalf():
            if ldl > 2:
              reject(state, "doesn't support FP16 with LocalDotLayout > 2")
              return
            elif (ldl == 2) and (state["ProblemType"]["HighPrecisionAccumulate"] != True):
              reject(state, "doesn't support non HPA FP16 with LocalDotLayout == 2")
              return
          else: # other type
              reject(state, "doesn't support LocalDotLayout with type {}".format(str(state["ProblemType"]["DataType"])))
              return

          if ldl != state["InnerUnroll"]:
            reject(state, "only support LocalDotLayout = InnerUnroll when LocalDotLayout > 1")
            return

          if ((state["LSPA"] % ldl) != 0) or ((state["LSPB"] % ldl) != 0):
            reject(state, "LSPA/B should be multiple of LocalDotLayout")
            return

    if 0:
      print("info: ", pvar(state, "LoopUnroll"), " LDS Stats:", pvar(state, "LdsOffsetA"), pvar(state, "LdsOffsetB"))
      print("info: ", pvar(state["ProblemType"], "TLUA"), \
          pvar(state, "NumLoadsCoalescedA"), pvar(state, "NumLoadsPerpendicularA"), \
          pvar(state, "LSCA"), pvar(state, "LSPA"))
      print("info:", pvar(state["ProblemType"], "TLUB"), \
          pvar(state, "NumLoadsCoalescedB"), pvar(state, "NumLoadsPerpendicularB"), \
          pvar(state, "LSCB"), pvar(state, "LSPB"))

    state["LoopIters"] = state["LoopUnroll"]
    if "MatrixInstK" in state:
      state["LoopIters"] //= state["MatrixInstK"]

    if state["LoopIters"] < 1:
      reject(state, "LoopIters need to greater than 0")
      return

    # PLR > 2xLoopIters is redundant setting
    # TODO- Why need to x2 ? Why not (if state["PrefetchLocalRead"] >= state["LoopIters"]:)
    if state["PrefetchLocalRead"] >= 2*state["LoopIters"]:
      reject(state, "Reject since PrefetchLocalRead %u >= 2x LoopIters %u" % (state["PrefetchLocalRead"],state["LoopIters"]))

    # reject low performance
    if state["PrefetchLocalRead"]%state["LoopIters"] > 1:
      reject(state, "PrefetchLocalRead: %u, LoopIters: %u performance is low" % (state["PrefetchLocalRead"],state["LoopIters"]))

    # prefetch wider read iteration > LoopIters, no enough iterations for prefetching
    if state["EnableMatrixInstruction"] and state["PrefetchLocalRead"] > 0:
      # Multiple = WLR-size / input-size = how many iters could be covered by one WLR ?
      wlrMultiple = state["LocalReadVectorWidth"]//state["MIInputPerThread"]
      if wlrMultiple == 0:
        reject(state, "LocalReadVectorWidth %u is less than MIInput" % (state["LocalReadVectorWidth"]))
        return
      # for example, if the original ds_read is b32...
      #   1. if LoopIters = 5 (b32 x 5 times), WLR-Multiple = 2 (b64), then we can fit the WLR
      #   2. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 4 (b128), this is not allowed
      #   3. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 2 (b64), this is allowed
      if state["LoopIters"] % wlrMultiple != 0:
        reject(state, "LocalReadVectorWidth %u cannot be distributed evenly, LoopIters %u should be divisible by WLR-Multiple %u" \
          % (state["LocalReadVectorWidth"], state["LoopIters"], wlrMultiple))

      PLR = (state["PrefetchLocalRead"] % state["LoopIters"])
      if PLR != 0 and state["LoopIters"] - (PLR * wlrMultiple) <= 0 :
        reject(state, "with PrefetchLocalRead %u LoopIters %u LocalReadVectorWidth %u, not enough LoopIters to prefetch %ux%u iterations, " \
          % (state["PrefetchLocalRead"],state["LoopIters"],state["LocalReadVectorWidth"], PLR , wlrMultiple) )
      if state["PrefetchLocalRead"] > 1 and PLR == 0:
        reject(state, "not good performance with PrefetchLocalRead %u LoopIters %u" \
          % (state["PrefetchLocalRead"],state["LoopIters"]) )

    # # reject conditions with lower performance
    # if state["ScheduleIterAlg"] == 2 and \
    # (state["ExpandPointerSwap"] != 1 or state["LoopIters"] != 1 or state["ScheduleGlobalRead"] != 1):
    #   reject(state, "ScheduleIterAlg 2 only work with EPS1_SGR1, LoopIter=1")

    if state["TransposeLDS"] == 1:
      if not state["EnableMatrixInstruction"]:
        reject(state, "TransposeLds Supports only in MatrixInstruction=1")
      if state["ProblemType"]["TLUA"] and state["ProblemType"]["TLUB"]:
          # TODO: Now in rocBLAS, lot of logic yamls are Type=NT and TLDS=1? Why aren't they rejected and how to get rid of them?
          reject(state, "TransposeLds requires TLUA=0 or TLUB=0")
    if state["EnableMatrixInstruction"]:
      # enable widerLocalRead
      if state["LocalReadVectorWidth"] > state["MIInputPerThread"]:
        # wider localRead support 2 types
        # 1. prefetch all lds to register
        # 2. using larger InnerUnroll
        if not (state["PrefetchLocalRead"] >= state["LoopIters"] and state["InnerUnroll"] == 1) and \
          not state["InnerUnroll"] >= state["LocalReadVectorWidth"] // state["MIInputPerThread"]:
          reject(state, "wider localRead only support (PrefetchLocalRead %u >= LoopIters %u) or (InnerUnroll %u > LocalReadxN)" % (state["PrefetchLocalRead"],state["LoopIters"],state["InnerUnroll"]))

    if state["DepthULdsDivisor"] > 1:
      if state["PrefetchGlobalRead"] == 2:
        reject(state, "DepthULdsDivisor > 1 does not support PrefetchGlobalRead=2")
      if state["ScheduleIterAlg"] != 3:
        reject(state, "DepthULdsDivisor > 1 does not support ScheduleIterAlg other than 3")
      if (state["DirectToLdsA"] == True or state["DirectToLdsB"] == True):
        reject(state, "DepthULdsDivisor > 1 does not support DirectToLds")
      if state["ProblemType"]["TLUA"] or state["ProblemType"]["TLUB"] or not (state["UnrollMajorLDSA"] and state["UnrollMajorLDSB"]):
        reject(state, "DepthULdsDivisor > 1: Only works with TN problem layout and UnrollMajorLDS")
      if state["PrefetchGlobalRead"]==1 and state["PrefetchLocalRead"]==0:
        reject(state, "PGR1 + PLR0 in SplitLDS requires double G2L buffer which is yet to be implemented")
      for tc in ('A', 'B'):
        if state["ProblemType"]["DataType"].numRegisters()*state["GlobalLoadVectorWidth%s"%tc] < state["DepthULdsDivisor"]:
          reject(state, "SplitLDS requires wider GlobalLoadVectorWidth%s; needs RegisterPerElem (%f) * GLVW%s (%u) >= DepthULdsDivisor (%u)"%
            (tc, state["ProblemType"]["DataType"].numRegisters(),tc,state["GlobalLoadVectorWidth%s"%tc],state["DepthULdsDivisor"]))

    if state["GlobalReadPerMfma"] > 1 and state["PrefetchGlobalRead"] == 2:
      reject(state, "GlobalReadPerMfma need to be 1 if PGR2")

    if state["UseInstOffsetForGRO"] == -1:
      state["UseInstOffsetForGRO"] = 1 if (state["DirectToLdsA"] or state["DirectToLdsB"]) else 0

    # Precise bounds check uses the "num_records" field in the buffer to
    # precisely detect when we are inbounds or not.  Only a one-dimensional
    # check is used since this is faster and also for computation we only
    # need to ensure that none of the loads fault.  threads which are
    # computing bogus sections of the C tile will later be ignored.
    # precise checking only works when all elements of the load are in-bounds
    # since if the vload crosses boundary we ignore all components not just the
    # ones that are OOB. See comments for groOffsetInMacroTile in KernelWriterAssembly.py
    #
    # So check for the cases where the unroll loop can
    # generate partial loads here and reject PBC solutions:
    # For non-TLU the free dim is in perp dim - should always be TRUE?  TODO

    #--
    # ShiftPtr can't use UseSgprForGRO since it needs to modify the VGPR pointers
    if bufferLoad and state["_UseSgprForGRO"] and state["EdgeType"]=="ShiftPtr":
      if not state["GuaranteeNoPartialA"] or not state["GuaranteeNoPartialB"]:
        state["_UseSgprForGRO"] = False
        #reject(state, "PBC with wide load has insufficient overlap guarantees- try GRVW=1 or adding appropriate Assert*ElementMultiple")

    if state["EnableMatrixInstruction"]:
      cont1 = not state["GuaranteeNoPartialB"]
      cont2 = ((state["MatrixInstN"] % state["GlobalLoadVectorWidthB"]) != 0)
      if cont1 and cont2:
        reject(state, "MatrixInstN %u %% GlobalLoadVectorWidthB %u must be 0" % \
          (state["MatrixInstN"], state["GlobalLoadVectorWidthB"]))
    else:
      if not bufferLoad or not state["GuaranteeNoPartialA"]:
        # Restrict GRVW/VW combos so shift-ptr logic will work
        if state["GlobalLoadVectorWidthA"] > 1 \
            and state["GlobalLoadVectorWidthA"] != state["VectorWidth"]:
            reject(state, "GlobalLoadVectorWidthA %u must be == VectorWidth %u or == 1" % \
                    (state["GlobalLoadVectorWidthA"], state["VectorWidth"]))

      if not bufferLoad or not state["GuaranteeNoPartialB"]:
        # Restrict GRVW/VW combos so shift-ptr logic will work
        if state["GlobalLoadVectorWidthB"] > 1 \
            and state["GlobalLoadVectorWidthB"] != state["VectorWidth"]:
            reject(state, "GlobalLoadVectorWidthB %u must be == VectorWidth %u or == 1" % \
                    (state["GlobalLoadVectorWidthB"], state["VectorWidth"]))

    # these work everywhere, no special restrictions
    state["AssertMinApproxSize"] = 0

    if state["KernelLanguage"] == "Assembly":
      if state["VectorWidth"] > 1:
        # VW>1 kernels require dims>1
        state["AssertMinApproxSize"] = 3
    elif state["VectorWidth"] > 1:
      # VW>1 kernels require dims>1
      state["AssertMinApproxSize"] = 2

    # Use SGPR to store an offset from GlobalReadOffsetA+0.
    # (as opposed to using dedicated VGPR for each GRO
    # Requires preciseBounds check since we rely on the buffer bounds check, not
    # individual vector registers doing bounds compares.
    if not bufferLoad and state["FractionalLoad"]:
        reject(state, "Fractional requires BufferLoad")

    if state["_UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough - better to leave PBC enabled even if we have to use VGPR
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      if numLoadsA + numLoadsB > 35 or state["DirectToVgprA"] or state["DirectToVgprB"]: # force _UseSgprForGRO = 0 if DirectToVgpr is enabled
        #print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["_UseSgprForGRO"] = 0
      else:
        state["_UseSgprForGRO"] = 1


    if packedC0 and not state["GuaranteeNoPartialA"]:
      reject(state, "packedC0 requires GuaranteeNoPartialA")
    if packedC1 and not state["GuaranteeNoPartialB"]:
      reject(state, "packedC1 requires GuaranteeNoPartialB")

    if packedC0 or packedC1:
      state["_UseSgprForGRO"] = 0

      if state["EdgeType"] != "ShiftPtr":
        reject(state, "Packed dims requires EdgeType==ShiftPtr")
      if state["KernelLanguage"] == "Assembly":
        if not bufferLoad:
          reject(state, "Packed dims for Assembly requires BufferLoad")

    if packedC0 and state["PackGranularity"]==2:
      if state["KernelLanguage"] == "Source":
        if state["AssertFree0ElementMultiple"]<state["VectorWidth"]:
          reject(state, "packedC0 Source requires AF0EM>=VectorWidth (for loads and stores)")
      else:
        if state["AssertFree0ElementMultiple"]<state["VectorWidth"]\
          or state["AssertFree0ElementMultiple"] == 1:
            if state["VectorStore"] <= 0:
              state["_VectorStore"] = 0
            else:
              reject(state, "packedC0 Assembly requires AF0EM>=VectorWidth or not VectorStore (for stores)")

    if state["AssertStrideCEqual"].get(0,-1) != 1 or state["AssertStrideDEqual"].get(0,-1) != 1:
      # Disable vector stores if not allowed:
      if state["VectorStore"] <= 0:
        state["_VectorStore"] = 0
      else:
        reject(state, "UseInitialStridesCD requires not VectorStore since store locations not adjacent")

    # Not currently suppored.  Support would require some changes in the
    # zeroPadRegs management:
    #   - don't allocate VGPRs for multiple perp/pad cases
    #   - guardZeroPad needs to add soffset to scalar calc
    if problemType["ZeroPadA"] or problemType["ZeroPadB"]:
      state["_UseSgprForGRO"] = 0

    # current requirement to avoid buffer loads that span multiple entries
    # if the summation dim participating in the ZeroPad is not fast-moving then
    # likely have more performant options.
    for tc in ('A', 'B'):
      if problemType["ZeroPad%s"%tc] and state["KernelLanguage"] == "Assembly":
        if state["GlobalLoadVectorWidth%s"%tc] != 1 \
            and problemType["IndexAssignments%s"%tc][0] in problemType["ZeroPad%s"%tc][0][0:1]:
          reject(state, "asm ZeroPad requires GlobalLoadVectorWidth==1")
        if not bufferLoad:
          reject(state, "asm ZeroPad requires BufferLoad")

    # Ensure AssertCEqualsD is always used with LdcEqualsLdd --DISABLED CURRENTLY
    #if state["AssertCEqualsD"]:
    #  if not ("LdcEqualsLdd" in state["ProblemType"] and state["ProblemType"]["LdcEqualsLdd"]):
    #    import pdb; pdb.set_trace()
    #    reject(state, "AssertCEqualsD requires LdcEqualsLdd=True")

    state["AssignedDerivedParameters"] = True

    # UnrollLoopEfficiencyEnable does not work with f16/bf16/int8x4
    if globalParameters["UnrollLoopEfficiencyEnable"] and (state["ProblemType"]["DataType"].isHalf() or \
       state["ProblemType"]["DataType"].isBFloat16() or state["ProblemType"]["DataType"].isInt8x4()):
      reject(state, "UnrollLoopEfficiencyEnable does not support f16/bf16/int8x4")

    # UnrollLoopEfficiencyEnable supports only ThreadTile0,1=[6,4] or [4,6] or [4,4] or [6.6] or [8,4] or [4,8]
    if globalParameters["UnrollLoopEfficiencyEnable"] and \
      not ((state["ThreadTile0"] == 6 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 6) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 6 and state["ThreadTile1"] == 6) or \
           (state["ThreadTile0"] == 8 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 8)):
      reject(state, "UnrollLoopEfficiencyEnable does not support ThreadTile0,1 = [%u,%u]"%(state["ThreadTile0"], state["ThreadTile1"]))

    # reject check for VgprForLocalReadPacking
    if state["VgprForLocalReadPacking"]:
      doable, rejectComment = Solution.isVgprForLocalReadPackingDoable(state)
      if not doable:
        reject(state, rejectComment)
        return

    # reject check for ClusterLocalRead
    if state["ClusterLocalRead"]:
        # Requires VgprForLocalReadPacking
        if not state["VgprForLocalReadPacking"]:
          reject(state, "ClusterLocalRead works with VgprForLocalReadPacking")
          return
        if (state["UnrollMajorLDSA"] == False and state["VectorWidthA"] > state["MIInputPerThread"]) or\
           (state["UnrollMajorLDSB"] == False and state["VectorWidthB"] > state["MIInputPerThread"]):
          reject(state, "ClusterLocalRead does not support VectorWidth or VectorWidthB > MIInputPerThread")
          return

    # change negative ExtraLatencyForLR to 0 for non DirectToVgpr
    if state["ExtraLatencyForLR"] < 0 and not (state["DirectToVgprA"] or state["DirectToVgprB"]):
      state["ExtraLatencyForLR"] = 0

    # SourceKernel with PrefetchGlobalRead / LoopDoWhile case
    # - SourceKernel + PGR2 does not work
    # - SourceKernel + LoopDoWhile requires at least 1 loop iteration (K > (DepthU * GSU * 2) - 1)
    if state["KernelLanguage"] == "Source":
      if state["PrefetchGlobalRead"] == 2:
        reject(state, "Source kernel does not support PGR2")
      else:
        factor = 0
        # DoWhile case, minimum iteration is 1
        if state["LoopDoWhile"]:
          factor = 1
        if factor > 0 and (not (3 in state["AssertSizeGreaterThan"].keys() and state["AssertSizeGreaterThan"][3] >= state["DepthU"] * state["GlobalSplitU"] * factor - 1)):
          reject(state, "Source kernel requires AssertSizeGreaterThan for K > (DepthU(%u) * GlobalSplitU(%u) * PGR/LoopDoWhile factor(%u) - 1)"%(state["DepthU"], state["GlobalSplitU"], factor))

  ########################################
  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
  def getMinNaming(objs):
    nonCKObjs = [obj for obj in objs if not isCustomKernelConfig(obj)]

    # early return
    if len(nonCKObjs) == 0:
      return {}

    # determine keys
    requiredParameters = {}
    if isinstance(nonCKObjs[0], Solution):
      keys = list(nonCKObjs[0]._state.keys())
    else:
      keys = list(nonCKObjs[0].keys())
    # only 1, rather than name being nothing, it'll be everything
    if len(nonCKObjs) == 1:
      for key in keys:
        if key in list(validParameters.keys()):
          requiredParameters[key] = False
    else:
      for key in keys:
        required = False
        if key in list(validParameters.keys()):
          for i in range(1, len(nonCKObjs)):
            if nonCKObjs[0][key] != nonCKObjs[i][key]:
              required = True
              break
        if required:
          requiredParameters[key] = True
        else:
          requiredParameters[key] = False

    requiredParameters["ProblemType"]       = False # always prepended
    requiredParameters["MacroTile0"]        = False # always prepended
    requiredParameters["MacroTile1"]        = False # always prepended
    requiredParameters["DepthU"]            = False # always prepended
    requiredParameters["LdcEqualsLdd"]      = False # always prepended
    requiredParameters["MatrixInstruction"] = False # always prepended
    requiredParameters["MatrixInstM"]       = False # always prepended
    requiredParameters["MatrixInstN"]       = False # always prepended
    requiredParameters["MatrixInstK"]       = False # always prepended
    requiredParameters["MatrixInstB"]       = False # always prepended
    requiredParameters["MatrixInstBM"]      = False # always prepended
    requiredParameters["MatrixInstBN"]      = False # always prepended
    requiredParameters["CustomKernelName"]  = False # Will not affect naming
    requiredParameters["Fp16AltImpl"]       = False # Will show up as a different type
    requiredParameters["Fp16AltImplRound"]  = False # Will show up as a different type
    requiredParameters["StochasticRounding"]= False # Will show up as a different type

    requiredParameters["Kernel"]            = True  # distinguish kernels from solutions
                                                    # for single-source compilation
    return requiredParameters

  ########################################
  @ staticmethod
  def getNameFull(state):
    requiredParameters = {}
    for key in state:
      if key in validParameters:
        requiredParameters[key] = True
    return Solution.getNameMin(state, requiredParameters)

  ########################################
  # Get Name Min
  @ staticmethod
  def getNameMin(state, requiredParameters):
    if isCustomKernelConfig(state):
      return state["CustomKernelName"]

    name = ""
    first = True
    # put problem first
    if "ProblemType" in state:
      name += str(state["ProblemType"]) + "_"
    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      name += "%s%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MacroTile"), \
          state["MacroTile0"], state["MacroTile1"], state["DepthU"] )
    if "MatrixInstM" in state:
      name += "%s%ux%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MatrixInstruction"), \
          state["MatrixInstM"], state["MatrixInstN"], state["MatrixInstK"], state["MatrixInstB"])
    if "LdcEqualsLdd" in state:
      if state["LdcEqualsLdd"]:
        name += "SE_"
      else:
        name += "SN_"
    for key in sorted(state.keys()):
      if key in requiredParameters and key[0] != '_':
        if requiredParameters[key] and key != "CustomKernelName":
          if not first:
            name += "_"
          else:
            first = False
          name += "%s%s" % ( Solution.getParameterNameAbbreviation(key), \
              Solution.getParameterValueAbbreviation(key, state[key]) )
    return name

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for objIdx in range(0, len(objs)):
      obj = objs[objIdx]
      for paramName in sorted(obj.keys()):
        if paramName in list(validParameters.keys()):
          paramValue = obj[paramName]
          if paramName in data:
            if paramValue not in data[paramName]:
              data[paramName].append(paramValue)
          else:
            data[paramName] = [ paramValue ]
    maxObjs = 1
    for paramName in data:
      if not isinstance(data[paramName][0],dict):
        data[paramName] = sorted(data[paramName])
      maxObjs *= len(data[paramName])
    numDigits = len(str(maxObjs))
    return [ data, numDigits ]

  ########################################
  # Get Name Serial
  @ staticmethod
  def getNameSerial(state, serialNaming):
    data = serialNaming[0]
    numDigits = serialNaming[1]

    serial = 0
    multiplier = 1
    for paramName in sorted(state.keys()):
      if paramName in list(validParameters.keys()):
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    return name


  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
    s += "%sProblemType: %s\n" % (indent, str(state["ProblemType"]))
    for key in sorted(state):
      s += "%s%s: %s\n" % (indent, str(key), str(state[key]))
    return s

  ########################################
  @ staticmethod
  def getParameterNameAbbreviation(name):
    return parameterNameAbbreviations[name]


  ########################################
  @ staticmethod
  def getParameterValueAbbreviation( key, value ):
    if key == 'ISA':
      return str(value[0]) + str(value[1]) + ('%x' % value[2])
    elif isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, tuple):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += str(value[i])
      return abbrev
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += Solution.getParameterValueAbbreviation(key, value[i])
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    elif isinstance(value, dict):
      s =  "_".join(["%d%d"%(pos,k) for pos,k in value.items()])
      return s
    elif isinstance(value, float):
      val1 = int(value)
      val2 = int(round(value*100)) - int(value)*100
      if val2 > 0:
        s =  "%dp%s" % (val1,str(val2).zfill(2))
      else:
        s = "%d" % (val1)
      return s
    else:
      printExit('Parameter {key}={value} is new object type ({t})'.format(key=key, value=value, t=type(value)))
      return str(value)

  ##########################
  # make class look like dict
  def keys(self):
    return self._state.keys()

  def __len__(self):
    return len(self._state)

  def __iter__(self):
    return iter(self._state)

  def __getitem__(self, key):
    return self._state[key]

  def __setitem__(self, key, value):
    self._name = None
    self._state[key] = value

  def __str__(self):
    if self._name is None:
      self._name = Solution.getNameFull(self._state)
    return self._name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return self._state

  def __hash__(self):
    return hash(str(self) + self.get("codeObjectFile", ""))

  def __eq__(self, other):
    return isinstance(other, Solution) and str(self) == str(other)

  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

  @property
  def enabledSplitLDS(self):
    return self["DepthULdsDivisor"] > 1

  @property
  def enabledSetPrioSplitLDS(self):
    # The interaction between SplitLDS's priority policy and StorePriorityOpt's is yet to be
    # investigated. For now, disable SplitLDS's priority policy when StorePriorityOpt is present
    # TODO: determine suitable priority policy when both are present
    return self.enabledSplitLDS and not self["StorePriorityOpt"]
