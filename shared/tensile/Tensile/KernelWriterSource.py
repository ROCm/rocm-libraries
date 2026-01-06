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

from . import Code
from .DataType import DataType
from .SolutionStructs import isPackedIndex
from .Common import globalParameters, printExit, roundUp
from .KernelWriter import KernelWriter

################################################################################
# Make OpenCL Kernel String
################################################################################
class KernelWriterSource(KernelWriter):

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming, removeTemporaries=True ):
    super(KernelWriterSource, self).__init__( \
        kernelMinNaming, kernelSerialNaming, removeTemporaries)
    self.language = globalParameters["RuntimeLanguage"]

    self.endLine = "\n"
    self.endLinePP =  "\\" + self.endLine
    self.quote = "\""
    self.endLineQuote = "\\n\""

    self.getGroupIdStr = "hc_get_group_id"
    self.getNumGroupsStr = "hc_get_num_groups"
    self.getLocalIdStr = "hc_get_workitem_id"
    self.getGlobalIdStr = "hc_get_workitem_absolute_id"
    self.sharedDeclStr = "__shared__ "
    self.sharedPtrStr = ""
    self.globalPtrStr = ""
    self.syncStr = "__syncthreads();"
    self.fenceStr = self.syncStr
    self.macFStr = "fmaf"
    self.macDStr = "fma"
    self.int64Str = "int64_t"
    self.uint64Str = "uint64_t"
    self.vectorComponents = ["x", "y", "z", "w"]
    self.atomicCasStr = "atomicCAS"
    self.volatileStr = ""
    self.deviceFunctionStr = "__device__ "

    self.commentPrefix = "/*"
    self.commentSuffix = "*/"
    self.commentHR = "*"*40
    self.indent = "  "

    self.psdUuseMagic = 1 # use magic number calc for pack summaton dims

    self.db={}
    self.db["PrintStagger"] = 0


  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterSource, self).initKernel( kernel, tPA, tPB )
    self.definedIter=set()
    pass

  ##############################################################################
  # Function Prefix
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""
    if kernel["ProblemType"]["DataType"].isHalf():
      self.vectorComponents = ["p[0]", "p[1]"]

    kStr += self.endLine

    ####################################
    # kernel preprocessor definitions
    kStr += self.endLine
    kStr += "/* tile parameters */" + self.endLine
    kStr += "#define NUM_THREADS %3d%s" \
        % (kernel["NumThreads"], self.endLine )
    kStr += "#define SG%s %d%s" \
        % (self.tileChar0, kernel["SubGroup0"], self.endLine )
    kStr += "#define SG%s %d%s" \
        % (self.tileChar1, kernel["SubGroup1"], self.endLine )
    kStr += "#define TT%s %d%s" \
        % (self.tileChar0, kernel["ThreadTile0"], self.endLine )
    kStr += "#define TT%s %d%s" \
        % (self.tileChar1, kernel["ThreadTile1"], self.endLine )
    kStr += "#define MT%s (SG%s*TT%s)%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine )
    kStr += "#define MT%s (SG%s*TT%s)%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine )
    kStr += "#define VECTOR_WIDTH %u%s" % (kernel["VectorWidth"], self.endLine)
    kStr += "#define GLOBAL_LOAD_VECTOR_WIDTH_A %u%s" \
        % (kernel["GlobalLoadVectorWidthA"], self.endLine)
    kStr += "#define GLOBAL_LOAD_VECTOR_WIDTH_B %u%s" \
        % (kernel["GlobalLoadVectorWidthB"], self.endLine)
    kStr += "#define GLOBAL_WRITE_VECTOR_WIDTH %u%s" \
        % (kernel["GlobalWriteVectorWidth"], self.endLine)
    kStr += self.endLine
    kStr += "/* DepthU parameters*/%s" % self.endLine
    kStr += "#define CPSV (NUM_THREADS / MT%s * VECTOR_WIDTH)%s" \
        % (self.tileChar0, self.endLine)
    kStr += "#define LOCAL_SPLITU %d%s" \
        % (kernel["LocalSplitU"], self.endLine )
    kStr += "#define UNROLL %d%s" \
        % (kernel["LoopUnroll"], self.endLine )
    kStr += "#define LOCAL_DEPTHU (LOCAL_SPLITU*UNROLL)%s" % (self.endLine )
    if kernel["GlobalSplitU"] > 1:
      kStr += "#define GLOBAL_SPLITU %u%s" \
          % (kernel["GlobalSplitU"], self.endLine )
    kStr += self.endLine
    kStr += "/* other */%s" % self.endLine
    kStr += "#define PAD %u%s" % (kernel["LdsPadA"], self.endLine)  # TODO - ignore LdsPadB
    kStr += "#define WORK_GROUP_MAPPING %u%s" \
        % (abs(kernel["WorkGroupMapping"]), self.endLine)
    kStr += self.endLine

    ####################################
    # num loads
    kStr += "/* num loads parallel and perpendicular to coalesced */%s" \
        % self.endLine
    kStr += "#define NLCA %d%s" % (kernel["NumLoadsCoalescedA"], self.endLine )
    kStr += "#define NLCB %d%s" % (kernel["NumLoadsCoalescedB"], \
        self.endLine )

    kStr += "#define NLPA %d%s" % (kernel["NumLoadsPerpendicularA"], \
        self.endLine )
    kStr += "#define NLPB %d%s" % (kernel["NumLoadsPerpendicularB"], \
        self.endLine )
    kStr += self.endLine

    ####################################
    # load sizes
    kStr += "/* load sizes parallel and perpendicular to coalesced */%s" % self.endLine
    if kernel["ProblemType"]["TLUA"]:
      kStr += "#define LSCA (MT%s/NLCA)%s" \
          % (self.tileCharA, self.endLine)
      kStr += "#define LSPA (LOCAL_DEPTHU/NLPA)" + self.endLine
    else:
      kStr += "#define LSCA (LOCAL_DEPTHU/NLCA)%s" \
          % (self.endLine)
      kStr += "#define LSPA (MT%s/NLPA)%s" \
          % ( self.tileCharA, self.endLine)
    if kernel["ProblemType"]["TLUB"]:
      kStr += "#define LSCB (MT%s/NLCB)%s" \
          % (self.tileCharB, self.endLine)
      kStr += "#define LSPB (LOCAL_DEPTHU/NLPB)" + self.endLine
    else:
      kStr += "#define LSCB (LOCAL_DEPTHU/NLCB)%s" \
          % (self.endLine)
      kStr += "#define LSPB (MT%s/NLPB)%s" % (self.tileCharB, self.endLine)
    kStr += "#define LVCA (LSCA/GLOBAL_LOAD_VECTOR_WIDTH_A)%s" % (self.endLine)
    kStr += "#define LVCB (LSCB/GLOBAL_LOAD_VECTOR_WIDTH_B)%s" % (self.endLine)
    kStr += "#define LVPA (LSPA/GLOBAL_LOAD_VECTOR_WIDTH_A)%s" % (self.endLine)
    kStr += "#define LVPB (LSPB/GLOBAL_LOAD_VECTOR_WIDTH_B)%s" % (self.endLine)


    # local buffer size
    kStr += "#define LDS_OFFSET_B %u%s" % (kernel["LdsOffsetB"], self.endLine)
    kStr += "#define LDS_NUM_ELEMENTS %u%s" % (kernel["LdsNumElements"], \
        self.endLine)

    # prefetch local buffer offsets
    # layout is redA, redB, blkA, blkB
    if kernel["PrefetchGlobalRead"]:
      kStr += "#define LDS_OFFSET_BLK %u%s" \
         % (kernel["LdsOffsetA_Blk"], self.endLine)

    ########################################
    # z-ordering
    if kernel["WorkGroupMappingType"] == "Z":
      kStr += self.endLine
      kStr += "#ifndef Z_ORDER_FUNCTIONS%s" % self.endLine
      kStr += "#define Z_ORDER_FUNCTIONS%s" % self.endLine
      kStr += "%svoid z_order(%s" % (self.deviceFunctionStr, self.endLine)
      kStr += "    unsigned int *z0, // 16-bits output%s" % self.endLine
      kStr += "    unsigned int *z1, // 16-bits output%s" % self.endLine
      kStr += "    unsigned int serial ) { // 32-bits input%s" % self.endLine
      kStr += "  *z0 = serial;%s" % (self.endLine)
      kStr += "  *z1 = (serial >> 1);%s" % (self.endLine)
      kStr += "  *z0 &= 0x55555555;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x55555555;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 1 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 1 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x33333333;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x33333333;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 2 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 2 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x0f0f0f0f; %s" % (self.endLine)
      kStr += "  *z1 &= 0x0f0f0f0f;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 4 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 4 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x00ff00ff;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x00ff00ff;%s"  % (self.endLine)
      kStr += "  *z0 |= ( (*z0) >> 8 );%s" % (self.endLine)
      kStr += "  *z1 |= ( (*z1) >> 8 );%s" % (self.endLine)
      kStr += "  *z0 &= 0x0000ffff;%s"  % (self.endLine)
      kStr += "  *z1 &= 0x0000ffff;%s"  % (self.endLine)
      kStr += "}%s" % self.endLine
      kStr += self.endLine
      kStr += "%sunsigned int round_down_power_of_2( unsigned int d0, unsigned int d1) {%s" % (self.deviceFunctionStr, self.endLine)
      kStr += "  unsigned int pow2 = min(d0, d1);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 1);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 2);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 4);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 8);%s" % self.endLine
      kStr += "  pow2 = pow2 | (pow2 >> 16);%s" % self.endLine
      kStr += "  pow2 = pow2 - (pow2 >> 1);%s" % self.endLine
      kStr += "  return pow2;%s" % self.endLine
      kStr += "}%s" % self.endLine
      kStr += self.endLine
      kStr += "%svoid generalized_z_order(%s" % (self.deviceFunctionStr, self.endLine)
      kStr += "    unsigned int *z0,%s" % self.endLine
      kStr += "    unsigned int *z1,%s" % self.endLine
      kStr += "    unsigned int d0,%s" % self.endLine
      kStr += "    unsigned int d1,%s" % self.endLine
      kStr += "    unsigned int maxPow2,%s" % self.endLine
      kStr += "    unsigned int max0,%s" % self.endLine
      kStr += "    unsigned int max1 ) {%s" % self.endLine
      kStr += "  if (! maxPow2) maxPow2 = round_down_power_of_2( max0, max1 );%s" % self.endLine
      kStr += "  // determine which tile wg is in and relative coord in tile%s" % self.endLine
      kStr += "  unsigned int offset0 = 0; // coord of tile%s" % self.endLine
      kStr += "  unsigned int offset1 = 0; // coord of tile%s" % self.endLine
      kStr += "  unsigned int start0 = 0;%s" % self.endLine
      kStr += "  unsigned int start1 = 0;%s" % self.endLine
      kStr += "  unsigned int tile = maxPow2;%s" % self.endLine
      kStr += "  unsigned int tilem1 = tile - 1;%s" % self.endLine
      kStr += "  for ( unsigned int i = 0; i < 16; i++ ) {%s" % self.endLine
      kStr += "    start0 = d0 & ~tilem1; // (d0 / tile) * tile;%s" % self.endLine
      kStr += "    start1 = d1 & ~tilem1; // (d1 / tile) * tile;%s" % self.endLine
      kStr += "    offset0 |= start0; // +=%s" % self.endLine
      kStr += "    offset1 |= start1;%s" % self.endLine
      kStr += "    d0 &= ~start0; // -=%s" % self.endLine
      kStr += "    d1 &= ~start1;%s" % self.endLine
      kStr += "    unsigned int end0 = start0 + tile; // cant be | b/c evals to 0+4->4 or 4+4->8%s" % self.endLine
      kStr += "    unsigned int end1 = start1 + tile;%s" % self.endLine
      kStr += "    if ( end0 <= max0 && end1 <= max1 ) break; // both end and max can be non-pow2%s" % self.endLine
      kStr += "    max0 -= start0; // cant be &~ b/c max0 doesnt necessarily have multiple of start0 to turn off%s" % self.endLine
      kStr += "    max1 -= start1;%s" % self.endLine
      kStr += "    tile >>= 1;%s" % self.endLine
      kStr += "    tilem1 >>= 1;%s" % self.endLine
      kStr += "  }%s" % self.endLine
      kStr += "  // d0, d1 is relative coord within tile%s" % self.endLine
      kStr += self.endLine
      kStr += "  // z-order relative coord%s" % self.endLine
      kStr += "  unsigned int serial = d0 + d1 * tile;%s" % self.endLine
      kStr += "  z_order( z0, z1, serial );%s" % self.endLine
      kStr += "  // add tile offset onto z-ordered index%s" % self.endLine
      kStr += "  *z0 |= offset0;%s" % self.endLine
      kStr += "  *z1 |= offset1;%s" % self.endLine
      #kStr += "  if (get_local_id(0)==0) printf(\\\"%%u, %%u -> %%u, %%u\\\\n\\\", d0, d1, (*z0), (*z1));%s" % self.endLine
      kStr += "}%s" % self.endLine
      kStr += "#endif%s" % self.endLine

    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # D
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( ((int64_t)IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + ((int64_t)IDX%s)*strideD%s" % (indexChar, indexChar)
    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      kStr += " + (gsuSumIdx)*strideW"
    kStr += " ))" + self.endLine
    # C
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( ((int64_t)IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + ((int64_t)IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    # A non-vector
    kStr += "#define GLOBAL_OFFSET_A(IDX%s" \
        % self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      kStr += ", IDX%s" \
          % self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    kStr += ") (( ((int64_t)IDX%s)*strideA%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
      kStr += " + ((int64_t)IDX%s)*strideA%s" % (indexChar, indexChar)
    kStr += " ))%s" % self.endLine
    # B non-vector
    kStr += "#define GLOBAL_OFFSET_B(IDX%s" \
        % self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      kStr += ", IDX%s" \
          % self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
    indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    kStr += ") (( ((int64_t)IDX%s)*strideB%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
      kStr += " + ((int64_t)IDX%s)*strideB%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    kStr += self.endLine

    ####################################
    # data types
    kStr += "/* data types */" + self.endLine
    kStr += "#define DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DataType"].toDevice(self.language), \
        self.endLine)
    kStr += "#define DEST_DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DestDataType"].toDevice(self.language), \
        self.endLine)
    kStr += "#define COMPUTE_DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["ComputeDataType"].toDevice(self.language), \
        self.endLine)
    #vecStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    #if kernel["VectorWidth"] > 1:
    #  vecStr += str(kernel["VectorWidth"])
    #kStr += "#define VECTOR_TYPE %s%s" % (vecStr, self.endLine)

    if self.language == "HIP" and kernel["ProblemType"]["DataType"].isComplex():
      kStr += "#define s0 x" + self.endLine
      kStr += "#define s1 y" + self.endLine

    ####################################
    # Atomic Global MAC
    if kernel["GlobalSplitU"] > 1 and kernel["_GlobalAccumulation"] != 'MultipleBuffer':
      kStr += self.comment("atomic add float")
      kStr += "#ifndef ATOMIC_FLOAT_FUNCTION%s" % (self.endLine)
      kStr += "#define ATOMIC_FLOAT_FUNCTION%s" % (self.endLine)
      """
      kStr += "%svoid atomicAddType(%s%sfloat *fPtr, float operand) {%s" \
          % ("__device__ " if self.language == "HIP" else "", \
          self.volatileStr, self.globalPtrStr, self.endLine)
      kStr += "  %s%sunsigned int *uPtr = (%s%sunsigned int *)fPtr;%s" \
          % (self.volatileStr, self.globalPtrStr, self.volatileStr, \
          self.globalPtrStr, self.endLine)
      #kStr += "  unsigned int old = *uPtr;%s" % (self.endLine)
      kStr += "  unsigned int old = atomicAdd(uPtr, 0); // atomic read%s" % (self.endLine)
      kStr += "  unsigned int assumed, newValue;%s" % (self.endLine)
      kStr += "  do {%s" % (self.endLine)
      kStr += "    assumed = old;%s" % (self.endLine)
      kStr += "    newValue = __float_as_uint(operand + __uint_as_float(assumed));%s" % (self.endLine)
      kStr += "    old = %s(uPtr, assumed, newValue);%s" \
          % (self.atomicCasStr, self.endLine)
      kStr += "  } while (assumed != old);%s" % (self.endLine)
      kStr += "}%s" % (self.endLine)
      """
      kStr += self.endLine
      kStr += "__device__ inline int atomicAddType(int *fPtr, int operand)%s" % (self.endLine)
      kStr += "{%s" % (self.endLine)
      kStr += "  return atomicAdd(fPtr,operand);%s" % (self.endLine)
      kStr += "}%s" % (self.endLine)
      kStr += self.endLine
      kStr += "__device__ inline unsigned int atomicAddType(unsigned int *fPtr, unsigned int operand)%s" % (self.endLine)
      kStr += "{%s" % (self.endLine)
      kStr += "  return atomicAdd(fPtr,operand);%s" % (self.endLine)
      kStr += "}%s" % (self.endLine)
      kStr += self.endLine
      kStr += "__device__ inline unsigned long long int atomicAddType(unsigned long long int *fPtr, unsigned long long int operand)%s" % (self.endLine)
      kStr += "{%s" % (self.endLine)
      kStr += "  return atomicAdd(fPtr,operand);%s" % (self.endLine)
      kStr += "}%s" % (self.endLine)
      kStr += self.endLine
      kStr += "__device__ inline float atomicAddType(float *fPtr, float operand)%s" % (self.endLine)
      kStr += "{%s" % (self.endLine)
      kStr += "  return atomicAdd(fPtr,operand);%s" % (self.endLine)
      kStr += "}%s" % (self.endLine)
      kStr += self.endLine
      kStr += "__device__ inline double atomicAddType(double *fPtr, double operand)%s" % (self.endLine)
      kStr += "{%s" % (self.endLine)
      kStr += "  return atomicAdd(fPtr,operand);%s" % (self.endLine)
      kStr += "}%s" % (self.endLine)
      kStr += self.endLine
      kStr += "#endif%s" % self.endLine

    kStr += "#define MAGIC_DIV1(dividend, magicNumber, magicShift) ((uint64_t)(dividend) * magicNumber >> magicShift)%s" % self.endLine


    ####################################
    # MACs
    kStr += self.endLine
    kStr += "/* MAC's */" + self.endLine

    if kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isHalf():
      kStr += "#define MAC(A,B,DST) DST += static_cast<float>(A) * static_cast<float>(B)"
    elif kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isInt8x4():
      kStr += "#define MAC(A,B,DST) DST = GenDot4(static_cast<int>(A), static_cast<int>(B), static_cast<int>(DST))"
    elif kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isBFloat16():
      kStr += "#define MAC(A,B,DST) DST += static_cast<float>(A) * static_cast<float>(B);"
    else:
      kStr += "#define MAC(A,B,DST) DST += A*B"
    kStr += self.endLine

    if kernel["ProblemType"]["DataType"].isReal():
      # real data
      if ((kernel["ThreadTileA"] % 2 == 0) and (kernel["ProblemType"]["DataType"].isHalf())):
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          kStr += "#define TYPE_MAC(MULA0,MULB0,DST0,MULA1,MULB1,DST1) " + self.endLinePP
          kStr += " DST0 = MAC(MULA0,MULB0,DST0);" + self.endLinePP
          kStr += " DST1 = MAC(MULA1,MULB1,DST1);" + self.endLinePP
          kStr += self.endLine
        else:
          kStr += "#define TYPE_MAC(MULA0,MULB0,DST0,MULA1,MULB1,DST1) " + self.endLinePP
          kStr += "  a_pk_fma[0] = MULA0; %s " % (self.endLinePP)
          kStr += " a_pk_fma[1] = MULA1; %s " % (self.endLinePP)
          kStr += " b_pk_fma[0] = MULB0; %s " % (self.endLinePP)
          kStr += " b_pk_fma[1] = MULB1; %s " % (self.endLinePP)
          kStr += " c_pk_fma[0] = DST0; %s " % (self.endLinePP)
          kStr += " c_pk_fma[1] = DST1; %s " % (self.endLinePP)
          kStr += " c_pk_fma = tensile_fmadd_half2(a_pk_fma, b_pk_fma, c_pk_fma); %s " % (self.endLinePP)
          kStr += " DST0 = c_pk_fma[0]; %s " % (self.endLinePP)
          kStr += " DST1 = c_pk_fma[1]; %s " % (self.endLinePP)
          kStr += self.endLine
      else:
        kStr += "#define TYPE_MAC(MULA,MULB,DST) " \
            + "DST = MAC(MULA,MULB,DST);" + self.endLine

      # GSU
      if kernel["GlobalSplitU"] > 1:
        if kernel["_GlobalAccumulation"] != 'MultipleBuffer': # 1st kernel would take care of Beta
          if kernel["ProblemType"]["UseBeta"]:
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) atomicAddType(&(DST), (ALPHA)*(REG));"
          else:
            kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) atomicAddType(&(DST), (ALPHA)*(REG));"
        elif kernel["_GlobalAccumulation"] == 'MultipleBuffer': # 2nd kernel would take care of Alpha and Beta
          if kernel["ProblemType"]["UseBeta"]:
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) DST = (REG);" + self.endLine
          else:
            kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) DST = (REG);" + self.endLine
      else:
        if kernel["ProblemType"]["UseBeta"]:
          # dst = alpha*reg + dst*beta
          if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
            kernel["ProblemType"]["DataType"].isBFloat16() and \
            kernel["ProblemType"]["DestDataType"].isBFloat16():
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) " \
              + "DST = 0 != (BETA) ? " \
              + "static_cast<tensile_bfloat16>((ALPHA)*(REG) + (BETA) * static_cast<float>(SRC)) : " \
              + "static_cast<tensile_bfloat16>((ALPHA)*(REG));" + self.endLine

          else:
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) " \
              + "DST = 0 != (BETA) ? (ALPHA)*(REG) + (BETA)*(SRC) : (ALPHA)*(REG);" + self.endLine
        else:
          # dst = alpha*reg
          kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) " \
              + "DST = (ALPHA)*(REG);" + self.endLine
    else:
      # complex data
      if not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # neither conjugate
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC( -MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]:
        # B conjugate (negate imaginary B.s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC( -MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      else:
        # A & B conjugate (negate imaginary .s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAC(  MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAC(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAC( -MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )

      if kernel["GlobalSplitU"] > 1: # 1st kernel will have taken care of B
        if kernel["_GlobalAccumulation"] != 'MultipleBuffer': # 1st kernel would take care of Beta
          if kernel["ProblemType"]["UseBeta"]:
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) atomicAddType(&(DST), (ALPHA)*(REG));" + self.endLine
          else:
            kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) atomicAddType(&(DST), (ALPHA)*(REG));" + self.endLine
        elif kernel["_GlobalAccumulation"] == 'MultipleBuffer': # 2nd kernel would take care of Alpha and Beta
          if kernel["ProblemType"]["UseBeta"]:
            kStr += "#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) DST = (REG);" + self.endLine
          else:
            kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) DST = (REG);" + self.endLine

      else:
        if kernel["ProblemType"]["UseBeta"]:
          # dst = alpha*reg + beta*dst
          kStr += (
            "#define TYPE_MAC_WRITE( DST, SRC, ALPHA, REG, BETA ) "+self.endLinePP +
            "  /* (1) */ " + self.endLinePP +
            "  type_mac_tmp = REG.s0; " + self.endLinePP +
            "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s0 = MAC( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
            "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s1 = MAC(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
            "  /* (2) */ " + self.endLinePP +
            "  if(BETA.s0 != 0) { " + self.endLinePP +
            "  REG.s0 = MAC(  BETA.s0, SRC.s0, REG.s0 ); " + self.endLinePP +
            "  REG.s1 = MAC(  BETA.s0, SRC.s1, REG.s1 ); " + self.endLinePP +
            "  } " + self.endLinePP +
            "  if (BETA.s1 != 0) { " + self.endLinePP +
            "  REG.s0 = MAC( -BETA.s1, SRC.s1, REG.s0 ); " + self.endLinePP +
            "  REG.s1 = MAC(  BETA.s1, SRC.s0, REG.s1 ); " + self.endLinePP +
            "  } " + self.endLinePP +
            "  /* (3) */ " + self.endLinePP +
            "  DST = REG;" + self.endLine )
        else:
          # dst = alpha*reg
          kStr += (
            "#define TYPE_MAC_WRITE( DST, ALPHA, REG ) "+self.endLinePP+
            "  /* (1) */ " + self.endLinePP +
            "  type_mac_tmp = REG.s0; " + self.endLinePP +
            "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s0 = MAC( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
            "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s1 = MAC(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
            "  /* (3) */ " + self.endLinePP +
            "  DST = REG;" + self.endLine )

    ####################################
    # sumation unroll
    kStr += self.endLine
    kStr += "/* %dx%d micro-tile */%s" \
      % (kernel["ThreadTile0"], kernel["ThreadTile1"], self.endLine)
    numMacs = 2 if kernel["PrefetchLocalRead"] else 1

    for m in range(0, numMacs):
      kStr += "#define MAC_%ux%u" \
          % (kernel["ThreadTile0"], kernel["ThreadTile1"])
      if kernel["PrefetchLocalRead"]:
        kStr += ("" if m==0 else "_BLK")
      kStr += self.endLinePP

      """
    if False:
      if kernel["VectorWidth"] == 1:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0], rA[1], rA[2], rA[3], rA[4], rA[5], rA[6], rA[7], rB[0], rB[1], rB[2], rB[3], rB[4], rB[5], rB[6], rB[7]); %s" % (self.endLinePP)
      if kernel["VectorWidth"] == 2:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0].%s, rA[0].%s, rA[1].%s, rA[1].%s, rA[2].%s, rA[2].%s, rA[3].%s, rA[3].%s, rB[0].%s, rB[0].%s, rB[1].%s, rB[1].%s, rB[2].%s, rB[2].%s, rB[3].%s, rB[3].%s); %s" % ( \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.endLinePP)
      if kernel["VectorWidth"] == 4:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0].%s, rA[0].%s, rA[0].%s, rA[0].%s, rA[1].%s, rA[1].%s, rA[1].%s, rA[1].%s, rB[0].%s, rB[0].%s, rB[0].%s, rB[0].%s, rB[1].%s, rB[1].%s, rB[1].%s, rB[1].%s); %s" % ( \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.endLinePP)
      """

      for idx1 in range(0, kernel["ThreadTile1"]):
        for idx0 in range(0, kernel["ThreadTile0"]):
          strC = "rC[%d+%d*TT%s]" % (idx0, idx1, self.tileChar0 )
          strA = "rA[%d%s]" % (idx0 if self.tPB["tile01Idx"] else idx1, ("+TT%s"%self.tileCharA) if m>0 else "")
          strB = "rB[%d%s]" % (idx1 if self.tPB["tile01Idx"] else idx0, ("+TT%s"%self.tileCharB) if m>0 else "")
          if ((kernel["ThreadTile0"] % 2 == 0) and (kernel["ProblemType"]["DataType"].isHalf())):
            if idx0 % 2 == 0:
              kStr += "  TYPE_MAC(%s,%s,%s , " % (strA, strB, strC)
            else:
              kStr += "%s,%s,%s); %s" % (strA, strB, strC, self.endLinePP)
          else:
            kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, \
                self.endLinePP)

      if kernel["UnrollMemFence"]:
        kStr += "  " + self.fenceStr
      kStr += self.endLine

      """
      for b in range(0, kernel["ThreadTileB"]):
        for a in range(0, kernel["ThreadTileA"]):
          # a
          vecA = a / kernel["VectorWidth"]
          elemA = a % kernel["VectorWidth"]
          strA = "rA[%d%s]" % (vecA, ("+TT%s/VECTOR_WIDTH"%self.tileCharA) \
              if m>0 else "")
          if kernel["VectorWidth"] > 1:
            strA += ".%s" % self.vectorComponents[elemA]
          # b
          vecB = b / kernel["VectorWidth"]
          elemB = b % kernel["VectorWidth"]
          strB = "rB[%d%s]" % (vecB, ("+TT%s/VECTOR_WIDTH"%self.tileCharB) \
              if m>0 else "")
          if kernel["VectorWidth"] > 1:
            strB += ".%s" % self.vectorComponents[elemB]
          # c
          strC = "rC[%d+%d*TT%s/VECTOR_WIDTH]" % (vecA, b, self.tileChar0 )
          elemC = elemA
          if kernel["VectorWidth"] > 1:
            strC += ".%s" % self.vectorComponents[elemC]
          #kStr += "  printf(\\\"T[%%u,%u,%u]: %s:%%.0f += %s:%%.0f * %s:%%.0f\\\\n\\\", serial, %s, %s, %s); %s" % (a, b, strC, strA, strB, strC, strA, strB, self.endLinePP)
          kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, \
              self.endLinePP)
      if kernel["UnrollMemFence"]:
        kStr += "  " + self.fenceStr
      kStr += self.endLine
      """

    ####################################
    # preprocessor definitions of kernel arguments
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStridesCD"]:
      # no strides #defined
      lastStrideD = 0
      lastStrideC = 0
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides CD*/%s" \
          % self.endLine
      lastStrideD = 1
      lastStrideC = 1

    if kernel["ProblemType"]["UseInitialStridesAB"]:
      lastStrideA = 0
      lastStrideB = 0
    else:
      kStr += "/* hard-coded initial strides AB */%s" \
          % self.endLine
      lastStrideA = 1
      lastStrideB = 1

    for i in range(firstStride, lastStrideD):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideA):
      kStr += "#define strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
          + " 1" + self.endLine
    for i in range(firstStride, lastStrideB):
      kStr += "#define strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
          + " 1" + self.endLine
    return kStr


  ##############################################################################
  # Function Signature Prefix
  ##############################################################################
  def functionSignaturePrefix(self, kernel):
    s = ""
    if self.language == "HIP":
      s += "#pragma clang diagnostic push" + self.endLine
      s += "#pragma clang diagnostic ignored \"-Wunused-parameter\"" + self.endLine
    return s


  ##############################################################################
  # Function Signature
  ##############################################################################
  def functionSignature(self, kernel ):
    kernelName = self.getKernelName(kernel)
    problemType = kernel["ProblemType"]

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]

    s = ""
    # kernel name
    s += "extern \"C\"\n"
    s += "__global__ "
    # the new default of 1024 degrades HGEMM performance too much
    s += "void\n__launch_bounds__(256)\n%s" % ( kernelName )
    s += "(" + self.endLine
    # pointers
    globalStr = "__global "
    if self.language == "HIP":
      #s += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    ptrStr = kernel["ProblemType"]["DestDataType"].toDevice(self.language)
    if kernel["_GlobalAccumulation"]:
      ptrStr = kernel["ProblemType"]["ComputeDataType"].toDevice(self.language)

    isStridedBuffer = kernel["ProblemType"]["StridedBatched"] or kernel["_GlobalAccumulation"]
    ptrStr  += ("" if isStridedBuffer else "*")
    batchStr = ("" if isStridedBuffer else "Batch")
    s += "  " + globalStr + ptrStr + " *"+ batchStr + "D,"
    s += self.endLine
    s += "  " + globalStr + ptrStr + " const * " + batchStr + "C,"
    s += self.endLine

    ptrStr   = kernel["ProblemType"]["DataType"].toDevice(self.language)
    ptrStr  += ("" if kernel["ProblemType"]["StridedBatched"] else "*")
    batchStr = ("" if kernel["ProblemType"]["StridedBatched"] else "Batch")
    s += "  " + globalStr + ptrStr + " const * " + batchStr + "A,"
    s += self.endLine
    s += "  " + globalStr + ptrStr + " const * " + batchStr + "B"

    # offset
    if not kernel["ProblemType"]["StridedBatched"]:
      s += "," + self.endLine + "  uint64_t offsetD"
      s += "," + self.endLine + "  uint64_t offsetC"
      s += "," + self.endLine + "  uint64_t offsetA"
      s += "," + self.endLine + "  uint64_t offsetB"

    # alpha & beta
    s += "," + self.endLine + "  " \
        + kernel["ProblemType"]["ComputeDataType"].toDevice(self.language) + " const alpha"
    if kernel["ProblemType"]["UseBeta"]:
      s += "," + self.endLine + "  " \
          + kernel["ProblemType"]["ComputeDataType"].toDevice(self.language) + " const beta"

    # strides
    firstStrideAB = firstStrideCD = 1
    if kernel["ProblemType"]["UseInitialStridesAB"]:
      firstStrideAB = 0
    if kernel["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideD = kernel["ProblemType"]["NumIndicesC"]
    lastStrideC = kernel["ProblemType"]["NumIndicesC"]
    lastStrideA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    for i in range(firstStrideCD, lastStrideD):
      s += "," + self.endLine + "  unsigned int const strideD" + self.indexChars[i]
    for i in range(firstStrideCD, lastStrideC):
      s += "," + self.endLine + "  unsigned int const strideC" + self.indexChars[i]
    for i in range(firstStrideAB, lastStrideA):
      s += "," + self.endLine + "  unsigned int const strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    for i in range(firstStrideAB, lastStrideB):
      s += "," + self.endLine + "  unsigned int const strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]

    # sizes
    for i in range(0, kernel["ProblemType"]["TotalIndices"]):
      s += "," + self.endLine + "  unsigned int size" + self.indexChars[i]

    for idxChar in self.magicSumChars:
      s += ",%s  unsigned magicNumberNumIter%s /*PSD*/" % (self.endLine, idxChar)
      s += ",%s  unsigned magicShiftNumIter%s /*PSD*/" % (self.endLine, idxChar)
      if kernel["GlobalSplitU"]>1 and idxChar==self.unrollChar:
          s += ",%s  unsigned magicNumberNumIter%s_GsuRemainder /*PSD */" % (self.endLine, idxChar)
          s += ",%s  unsigned magicShiftNumIter%s_GsuRemainder /*PSD */" % (self.endLine, idxChar)

    for idxChar in self.magicNonSumChars:
      s += ",%s  unsigned magicNumberSize%s" % (self.endLine, idxChar)
      s += ",%s  unsigned magicShiftSize%s" % (self.endLine, idxChar)

    for idx in problemType["IndicesSummation"]:
      for tc in ('A','B'):
        for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
          (freeDim, sumDim) = zp[:2]
          freeDimChar = globalParameters["IndexChars"][freeDim]
          sumChar = self.indexChars[sumDim]
          if sumDim == idx:
            s += ",%s  int padStart%s%s%s" % (self.endLine, tc, freeDimChar, sumChar)
            s += ",%s  int padEnd%s%s%s" % (self.endLine, tc, freeDimChar, sumChar)

    s += "," + self.endLine + "  unsigned int staggerUIterParm"

    # kernel["PersistentKernel"]:
    s += "," + self.endLine + "  unsigned int problemNumGroupTiles0"
    s += "," + self.endLine + "  unsigned int problemNumGroupTiles1"

    s += " )"
    return s

  ##############################################################################
  # Function Signature Suffix
  ##############################################################################
  def functionSignatureSuffix(self, kernel):
    s = ""
    if self.language == "HIP":
      s += self.endLine
      s += "#pragma clang diagnostic pop" + self.endLine
    return s

  ##############################################################################
  # Function Begin
  ##############################################################################
  def functionBegin(self, kernel):
    s = ""
    s += " {" + self.endLine
    return s


  ##############################################################################
  # Allocate Resources
  ##############################################################################
  def allocateResources(self, kernel, lraCode=None):
    kStr = ""

    kStr += "  unsigned int serial = %s(0);%s" \
        % (self.getLocalIdStr, self.endLine)
    kStr += "  unsigned int sgId = serial / (SG%s*SG%s);%s" \
        % (self.tileChar0, self.tileChar1, self.endLine)

    ####################################
    # zero
    if kernel["ProblemType"]["DataType"].isHalf() \
        and kernel["VectorWidth"] > 1 \
        and (kernel["LoopTail"] or kernel["EdgeType"] == "Branch"):
      kStr += "#define SCALAR_ZERO 0%s" % self.endLine
    elif kernel["ProblemType"]["DestDataType"].isBFloat16():
      kStr += "#define SCALAR_ZERO 0.0f%s" % self.endLine
    else:
      kStr += "#define SCALAR_ZERO %s%s" % ( kernel["ProblemType"][\
         "DataType"].zeroString(self.language, 1), \
         self.endLine )

    # TODO - use a different value for OOB data
    #        Currently use zero since Tensile already has handy functions to create zero in different types
    if kernel["ProblemType"]["HighPrecisionAccumulate"] and kernel["ProblemType"]["DataType"].isBFloat16():
      kStr += "#define SCALAR_OOB_DATA static_cast<tensile_bfloat16>(0.0f)%s" % self.endLine
    else:
      kStr += "#define SCALAR_OOB_DATA SCALAR_ZERO%s" % self.endLine

    kStr += "  /* registers for MAC's */" + self.endLine
    # TODO: change to kStr += "  COMPUTE_DATA_TYPE rC[TT%s*TT%s];%s" \ % (self.tileChar0, self.tileChar1, self.endLine )
    # with above there is no need for the if below
    if kernel["ProblemType"]["HighPrecisionAccumulate"] and (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
        kStr += "  float rC[TT%s*TT%s];%s" \
            % (self.tileChar0, self.tileChar1, self.endLine )
    else:
        kStr += "  DEST_DATA_TYPE rC[TT%s*TT%s];%s" \
            % (self.tileChar0, self.tileChar1, self.endLine )

    # registers for valuAB
    kStr += "  DATA_TYPE rA[TT%s%s];%s" \
        % (self.tPA["tileChar"], ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)
    kStr += "  DATA_TYPE rB[TT%s%s];%s" \
        % (self.tPB["tileChar"], ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)

    ####################################
    # registers for global -> local load
    kStr += self.endLine
    kStr += "  /* registers for global->local */%s" % self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for sPerp in range(0, self.numReadsPerpVecCompA):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          for sPara in range(0, self.numReadsCoalVecCompA):
            kStr += "  DATA_TYPE a_%u_%u_%u_%u;%s" \
                % (para, sPara, perp, sPerp, self.endLine)
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for sPerp in range(0, self.numReadsPerpVecCompB):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          for sPara in range(0, self.numReadsCoalVecCompB):
            kStr += "  DATA_TYPE b_%u_%u_%u_%u;%s" \
                % (para, sPara, perp, sPerp, self.endLine)
    """
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += "a_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedA"]-1 \
            and perp == kernel["NumLoadsPerpendicularA"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += "b_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedB"]-1 \
            and perp == kernel["NumLoadsPerpendicularB"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    """

    ####################################
    # allocate tensile_half2 memory
    if kernel["ProblemType"]["DataType"].isHalf():
      kStr += self.endLine
      kStr += "  /* allocate tensile_half2 memory */" + self.endLine
      kStr += "  tensile_half2 a_pk_fma;" + self.endLine
      kStr += "  tensile_half2 b_pk_fma;" + self.endLine
      kStr += "  tensile_half2 c_pk_fma;" + self.endLine

    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += "  /* allocate local memory */" + self.endLine
    kStr += "  %sDATA_TYPE localMemory[LDS_NUM_ELEMENTS];%s" \
        % (self.sharedDeclStr, self.endLine )


    ####################################
    # apply general batch
    if not kernel["ProblemType"]["StridedBatched"]:
      zeroStr = kernel["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)

      kStr += self.endLine
      kStr += "  unsigned int sizeUnroll = 1"
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]):
        loopDim  = kernel["ProblemType"]["IndicesSummation"][i]
        loopChar = self.indexChars[loopDim]
        kStr += f" * size{loopChar}"
      kStr += ";" + self.endLine

      kStr += "  unsigned int wg = " + self.getGroupIdStr + "(2);" + self.endLine
      if not kernel["_GlobalAccumulation"]:
        kStr +=  "  DEST_DATA_TYPE      * D = BatchD[wg];" + self.endLine
        kStr += f"  DEST_DATA_TYPE const* C = (beta == {zeroStr}) ? nullptr : BatchC[wg];" + self.endLine
      kStr += f"  DATA_TYPE      const* A = ((alpha == {zeroStr}) || (sizeUnroll == 0)) ? nullptr : BatchA[wg];" + self.endLine
      kStr += f"  DATA_TYPE      const* B = ((alpha == {zeroStr}) || (sizeUnroll == 0)) ? nullptr : BatchB[wg];" + self.endLine

      # apply offset only for general batch
      kStr += self.endLine
      if not kernel["_GlobalAccumulation"]:
        kStr += "  D = D + offsetD;" + self.endLine
        kStr += "  C = C + offsetC;" + self.endLine
      kStr += "  A = A + offsetA;" + self.endLine
      kStr += "  B = B + offsetB;" + self.endLine

    if 0:
      # in some cases we know the pad values at compile time and could hard-code here.  Not enabled.
      for tc in ('A', 'B'):
        for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
          (freeDim, sumDim, padStart, padEnd) = zp
          freeDimChar = globalParameters["IndexChars"][freeDim]
          sumChar = self.indexChars[sumDim]
          kStr += self.endLine
          kStr += "  unsigned int padStart%s%s%s = %u;" % (tc, freeDimChar, sumChar, padStart) + self.endLine
          kStr += "  unsigned int padEnd%s%s%s = %u;" % (tc, freeDimChar, sumChar, padEnd) + self.endLine

    self.magicSumChars = []
    if kernel["PackSummationDims"]:
      self.magicSumChars += [globalParameters["IndexChars"][c] for \
          c in kernel["ProblemType"]["IndicesSummation"][1:]]

    self.magicNonSumChars = kernel["PackedC0IdxChars"][:-1] + kernel["PackedC1IdxChars"][:-1]

    if kernel["MagicDivAlg"] == 2:
      kStr += self.endLine
      kStr += "  typedef struct MagicStruct {unsigned M; int a; int s;} MagicStruct;" + self.endLine
      kStr += "  const unsigned MAGIC_STRUCT_A = 0x80000000; // for extracting a-bit from shift kernarg" + self.endLine
      kStr += "#define MAGIC_DIV2(dividend, magic) (((((uint64_t)(dividend) * magic.M) >> 32) + dividend*magic.a) >> magic.s)%s" % self.endLine

      sumParms=[(idxChar, "magicStruct%s"%idxChar, "NumIter%s"%idxChar) for idxChar in self.magicSumChars]
      if kernel["PackSummationDims"] and kernel["GlobalSplitU"] > 1 and sumParms:
          sumParms.append([self.unrollChar, "magicStruct%s_GsuRemainder"%self.unrollChar, "NumIter%s_GsuRemainder" % self.unrollChar])
      for (idxChar, magicStruct, parmName) in sumParms + [(idxChar, "magicStruct%s"%idxChar, "Size%s"%idxChar) for idxChar in self.magicNonSumChars]:
        kStr += self.endLine
        kStr += "  MagicStruct %s;"%(magicStruct) + self.endLine
        kStr += "  %s.M = magicNumber%s;" % (magicStruct, parmName) + self.endLine
        kStr += "  %s.a = (magicShift%s & MAGIC_STRUCT_A) ? 1:0;" %(magicStruct, parmName) + self.endLine
        kStr += "  %s.s = magicShift%s & (~MAGIC_STRUCT_A);" %(magicStruct, parmName) + self.endLine


    return kStr

  ##############################################################################
  # Open Persistent Loop
  # init iteration counter, define loop target
  ##############################################################################
  def openPersistentLoop(self, kernel):
    kStr = ""
    if kernel["PersistentKernel"]:
      wg0 = "wg%s" % self.tileChar0
      wg1 = "wg%s" % self.tileChar1
      kStr += "  %s serialWgIter = %s(0);%s" \
        % (self.uint64Str, self.getGroupIdStr, self.endLine)
      kStr += "  unsigned int n%s = problemNumGroupTiles0;%s" \
          % ( wg0, self.endLine)
      kStr += "  unsigned int n%s = problemNumGroupTiles1;%s" \
          % ( wg1, self.endLine)
      kStr += "  unsigned int %s;%s" % ( wg0, self.endLine)
      kStr += "  unsigned int %s;%s" % ( wg1, self.endLine)

      # PersistentKernel along batch dimension
      if kernel["PersistentKernelAlongBatch"]:
        kStr += "  unsigned int wgKSerial;" + self.endLine
        kStr += "  unsigned int wgIJSerial;" + self.endLine

      #kStr += "if (serial==0) printf(\"WG%%u_%%u probWG:%%u_%%u  %s\", hc_get_group_id(0), hc_get_group_id(1), %s, %s);" % (self.endLinePP, wg0, wg1)+ self.endLine
      kStr += "%swhile (1) { // persistent loop %s" % (self.endLine, self.endLine)
    return kStr


  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  def graWorkGroup(self, kernel, isPap):
    kStr = ""

    wg0 = "wg%s" % self.tileChar0
    wg1 = "wg%s" % self.tileChar1
    nwgg = kernel["WorkGroupMapping"] >= 0
    n0 = 0 if nwgg else 1
    n1 = 1 if nwgg else 0

    if kernel["PersistentKernel"]:
      # TODO - PK not support GSU in Assembly, but HIP is OK
      kStr += "  unsigned int numWGIJ = problemNumGroupTiles0*problemNumGroupTiles1;" + self.endLine
      if kernel["PersistentKernelAlongBatch"]:
        wgKSerial = "wgKSerial"
        wgIJSerial = "wgIJSerial"
        # compare serialWgIter against problem groups
        # TODO - AlongBatch not support GSU in HIP now
        kStr += "  if (serialWgIter >= numWGIJ * sizeK) break; // persistent loop" + self.endLine
        kStr += "  %s  = serialWgIter / numWGIJ;%s" % ( wgKSerial, self.endLine)
        kStr += "  %s  = serialWgIter %% numWGIJ;%s" % ( wgIJSerial, self.endLine)
        kStr += "  %s  = %s %% problemNumGroupTiles0;%s" % ( wg0, wgIJSerial, self.endLine)
        kStr += "  %s  = %s / problemNumGroupTiles0;%s" % ( wg1, wgIJSerial, self.endLine)
        if not kernel["ProblemType"]["StridedBatched"]:
          zeroStr = kernel["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
          if not kernel["_GlobalAccumulation"]:
            kStr +=  "  D = BatchD[wgKSerial] + offsetD;%s" % self.endLine
            kStr += f"  C = ((beta == {zeroStr}) ? nullptr : BatchC[wgKSerial]) + offsetC;%s" % self.endLine
          kStr += f"  A = (((alpha == {zeroStr}) || (sizeUnroll == 0)) ? nullptr : BatchA[wgKSerial]) + offsetA;" + self.endLine
          kStr += f"  B = (((alpha == {zeroStr}) || (sizeUnroll == 0)) ? nullptr : BatchB[wgKSerial]) + offsetB;" + self.endLine
      else:
        # compare serialWgIter against problem groups
        if kernel["GlobalSplitU"] > 1:
          kStr += "  if (serialWgIter >= numWGIJ * GLOBAL_SPLITU) break; // persistent loop" + self.endLine
        else:
          kStr += "  if (serialWgIter >= numWGIJ) break; // persistent loop" + self.endLine
        kStr += "  %s  = serialWgIter %% problemNumGroupTiles0;%s" % ( wg0, self.endLine)
        kStr += "  %s  = serialWgIter / problemNumGroupTiles0;%s" % ( wg1, self.endLine)
    else:
      # optionally transpose work-group grid
      kStr += "  unsigned int %s = %s(%u);%s" \
          % ( wg0, self.getGroupIdStr, n0, self.endLine)
      kStr += "  unsigned int %s = %s(%u);%s" \
          % ( wg1, self.getGroupIdStr, n1, self.endLine)
      kStr += "  unsigned int n%s = %s(%u);%s" \
          % ( wg0, self.getNumGroupsStr, n0, self.endLine)
      kStr += "  unsigned int n%s = %s(%u);%s" \
          % ( wg1, self.getNumGroupsStr, n1, self.endLine)
      if kernel["GlobalSplitU"] > 1:
        kStr += "  n%s /= GLOBAL_SPLITU;%s" % (wg1, self.endLine)

    # split up work-group grid
    if kernel["GlobalSplitU"] > 1:
      kStr += "  unsigned int gsuSumIdx;%s" % self.endLine
      if kernel["GlobalSplitUWorkGroupMappingRoundRobin"]:
        kStr += "  gsuSumIdx = %s / n%s;%s" \
            % (wg1, wg1, self.endLine)
        kStr += "  %s = %s %% n%s;%s" \
            % (wg1, wg1, wg1, self.endLine)
      else:
        kStr += "  gsuSumIdx = %s %% GLOBAL_SPLITU;%s" \
            % (wg1, self.endLine)
        kStr += "  %s = %s / GLOBAL_SPLITU;%s" \
            % (wg1, wg1, self.endLine)

      ########################################
      # Blocked rows or columns
    if kernel["WorkGroupMappingType"] == "B" and abs(kernel["WorkGroupMapping"]) > 1:
      kStr += self.endLine
      kStr += "  %s wgSerial = %s + (%s %% WORK_GROUP_MAPPING) * n%s;// within block%s" \
        % (self.uint64Str, wg0, wg1, wg0, self.endLine)
      kStr += "  unsigned int block = %s / WORK_GROUP_MAPPING;%s" \
          % (wg1, self.endLine );
      kStr += "  unsigned int blockRemainder = (%s < n%s-(n%s %% WORK_GROUP_MAPPING) ) ? 0 : n%s %% WORK_GROUP_MAPPING;%s" % \
          ( wg1, wg1, wg1, wg1, self.endLine )
      for blockRemainder in range(0, abs(kernel["WorkGroupMapping"])):
        blockWidth = abs(kernel["WorkGroupMapping"]) if blockRemainder==0 else blockRemainder
        if blockRemainder > 0:
          kStr += " else "
        else:
          kStr += "  "
        if blockRemainder < abs(kernel["WorkGroupMapping"])-1:
          kStr += "if ( blockRemainder == %u) " % (blockRemainder)
        kStr += "{%s" % self.endLine
        kStr += "    %s = wgSerial / %u;%s" \
            % (wg0, blockWidth, self.endLine)
        kStr += "    %s = wgSerial %% %u + block*WORK_GROUP_MAPPING;%s" \
            % (wg1, blockWidth, self.endLine)
        kStr += "  }"
      kStr += "%s" % self.endLine


    ########################################
    # Generalized Z-Order
    elif kernel["WorkGroupMappingType"] == "Z":

      kStr += "  unsigned int nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += "  unsigned int nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)

      if abs(kernel["WorkGroupMapping"]) == 1: # Generalized Z-Order
        kStr += "  generalized_z_order(&%s, &%s, %s, %s, 0, nwg0, nwg1);%s" \
            % ( wg0, wg1, wg0, wg1, self.endLine)

      elif abs(kernel["WorkGroupMapping"]) == 2: # Z-Order round up and return early
        kStr += "  unsigned int wgSerial = %s + %s * n%s;%s" % (wg0, wg1, wg0 if nwgg else wg1, self.endLine)
        kStr += "  z_order(&%s, &%s, wgSerial);%s" % (wg0, wg1, self.endLine)
        kStr += "  if (%s >= nwg0 || %s >= nwg1) return; // wg mapped out of bounds after z-ordering%s" \
            % (wg0, wg1, self.endLine)
      else:
        printExit("WorkGroupMappingType=Z and WorkGroupMapping=%u not supported"%kernel["WorkGroupMapping"])

    #kStr += "if (serial==0) printf(\"WG:%%u_%%u progWG:%%u_%%u \\n\", hc_get_group_id(0), hc_get_group_id(1), %s, %s);" \
    #      % (wg0, wg1)+ self.endLine
    return kStr


  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int globalReadOffset%s%s = (serial%s" \
        % (tP["tensorChar"], tP["tileChar"], ("%" if tP["grcg"] == tP["tlu"] else "/") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")"
    if tP["grcv"] == tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s" % tP["tensorChar"]
    kStr += " + ("
    kStr += "wg%s" % (tP["tileChar"])
    kStr += ")*MT%s;%s" % (tP["tileChar"], self.endLine)
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    kStr = "  unsigned int globalReadOffset%s%s = " % (tP["tensorChar"], self.unrollChar)
    if kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"% tP["tensorChar"]]:
      kStr += "size%s - 1 - " % self.unrollChar
    kStr += "(serial" + ("/" if tP["grcg"] == tP["tlu"] else "%")
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")"
    if tP["grcv"] != tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s"% tP["tensorChar"]
    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += " + LOCAL_DEPTHU*"
      else:
        kStr += " + (size%s/GLOBAL_SPLITU)*" % self.unrollChar
      kStr += "gsuSumIdx"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    kStr = ""
    # packed free dims don't use 'wg' level vars for dims
    nonTileFreeIndices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    if kernel["PersistentKernel"] and kernel["PersistentKernelAlongBatch"]:
      kStr += "  unsigned int wgK = wgKSerial % sizeK;" + self.endLine
    else:
      for i in range(0, len(nonTileFreeIndices)):
        index = nonTileFreeIndices[i]
        if isPackedIndex(kernel, index):
          continue
        kStr += "  unsigned int wg" + self.indexChars[index] \
            + " = ( " + self.getGroupIdStr + "(2)"
        for j in reversed(list(range( i+1, len(nonTileFreeIndices)))):
          index2 = nonTileFreeIndices[j]
          kStr += " / size" + self.indexChars[index2]
        kStr += " ) % size" + self.indexChars[index] + ";" + self.endLine

    return kStr

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    kStr = ""
    for i in range(self.otherSummations):
      index = i + kernel["ProblemType"]["NumIndicesC"]
      if index in kernel["ProblemType"]["MirrorDimsA"]:
        kStr += "unsigned int globalReadOffsetA%s = size%s - 1;%s" \
            % (self.indexChars[index], self.indexChars[index], self.endLine)
      else:
        kStr += "#define globalReadOffsetA%s 0%s" \
            % (self.indexChars[index], self.endLine)
      if index in kernel["ProblemType"]["MirrorDimsB"]:
        kStr += "unsigned int globalReadOffsetB%s = size%s - 1;%s" \
            % (self.indexChars[index], self.indexChars[index], self.endLine)
      else:
        kStr += "#define globalReadOffsetB%s 0%s" \
            % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    for l in range(0, tP["nrt"]):
      for s in range(0, 1 if tP["rc"] else tP["nrtv"]):
        flattenedOffset = "flattenedOffset%s_%u_%u"%(tc,l,s)
        gro = "globalReadOffset%s%s_%u_%u" % (tc, tP["tileChar"], l, s)
        kStr += "  unsigned int %s = globalReadOffset%s%s + %u + %d*%s;%s" \
            % (flattenedOffset, tc, tP["tileChar"], s, l, \
            (tP["lsc"] if tP["tlu"] else tP["lsp"]), \
            self.endLine)

        # clip to edge if the flattened offset is OOB:
        tP["packedSizeList"] = ["size%s"%self.indexChars[idx] for idx in kernel["PackedC%dIndicesX"%(tP["tile01Idx"])]]
        sizeStr = " * ".join(tP["packedSizeList"])

        kStr += "  %s = (%s > (%s-1)) ? (%s-1):%s;%s" \
            % (flattenedOffset, flattenedOffset, sizeStr, sizeStr, flattenedOffset, self.endLine)

        # Create additional vector address components for any packed dimensions
        lastGro = flattenedOffset
        firstPrintedIdx=1
        lastIdx = -1
        for idx in kernel["ProblemType"]["IndexAssignments%s"%tc]:
          if idx < kernel["ProblemType"]["NumIndicesC"] and isPackedIndex(kernel, idx, tP["PackBatchDims"]):
            gro = "globalReadOffset%s%s_%u_%u" % (tc, self.indexChars[idx], l, s)
            # unpacked batch dims do not to declare a GRO ; they use WG
            # packed batch dims and free dims do need a GRO defined here, and may need to 'unpack'
            # process in order of index assignments for A/B.
            if firstPrintedIdx:
              # no unpacking from prev needed:
              firstPrintedIdx = 0
              kStr += "  unsigned int %s = %s;%s" % (gro, flattenedOffset, self.endLine)
              #kStr += "printf(\"gro: serial:%%u wg0:%%u wg1:%%u %s:%%u\\n\", serial, wg0I, wg1J, %s);%s" % (gro, gro, self.endLine)
            else:
              # if another free dim or a packed batch dim
              if kernel["MagicDivAlg"]:
                c = globalParameters["IndexChars"][lastIdx]
                if kernel["MagicDivAlg"]==1:
                  kStr += "  unsigned int %s = MAGIC_DIV1(%s, magicNumberSize%s, magicShiftSize%s);%s" \
                          % (gro, lastGro, c, c, self.endLine)
                elif kernel["MagicDivAlg"]==2:
                  kStr += "  unsigned int %s = MAGIC_DIV2(%s, magicStruct%s);%s" \
                          % (gro, lastGro, c, self.endLine)
                kStr += "  %s -= (%s*size%s);%s" \
                    % (lastGro, gro, self.indexChars[lastIdx], self.endLine)
              else:
                kStr += "  unsigned int %s = %s / size%s; // extract packed index%s" \
                        % (gro, lastGro, self.indexChars[lastIdx], self.endLine)
                kStr += "  %s %%= size%s;%s" % (lastGro, self.indexChars[lastIdx], self.endLine)
            lastGro = gro
            lastIdx = idx

          if 0 and tP["isA"]:
            kStr += "printf(\"gro-0: serial:%%u wg0:%%u wg1:%%u globalReadOffsetA0I_0_0:%%u\\n\", serial, wg0I, wg1J, globalReadOffsetA0I_0_0);%s" \
                    % (self.endLine)
          if 0 and tP["isB"]:
            kStr += "printf(\"gro-0: serial:%%u wg0:%%u wg1:%%u globalReadOffsetA0J_0_0:%%u\\n\", serial, wg0I, wg1J, globalReadOffsetA0J_0_0);%s" \
                    % (self.endLine)

    if 0 and tP["isB"]:
      kStr += "printf(\"gro-1: serial:%%u wg0:%%u wg1:%%u globalReadOffsetA0I_0_0:%%u globalReadOffsetB1J_0_0:%%u\\n\", serial, wg0I, wg1J, globalReadOffsetA0I_0_0, globalReadOffsetB1J_0_0);%s" \
        % (self.endLine)

    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    kStr = ""
    isMirrorUnroll = kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"% tP["tensorChar"]]
    for l in range(0, tP["nru"]):
      for s in range(0, 1 if tP["rc"] else kernel["VectorWidth"]):
        kStr += "  unsigned int globalReadOffset%s%s_%u_%u = globalReadOffset%s%s + %u %s %d*%s;%s" \
            % (tP["tensorChar"], self.unrollChar, l, s, \
            tP["tensorChar"], self.unrollChar, s,    \
            "-" if isMirrorUnroll else "+", l, \
            (tP["lsp"] if tP["tlu"] else tP["lsc"]), \
            self.endLine)
      #else:
      #  kStr += "  unsigned int globalReadOffset%s%s_%u = globalReadOffset%s%s + %d*%s;%s" \
      #      % (tP["tensorChar"], self.unrollChar, l, tP["tensorChar"], self.unrollChar, l, \
      #      (tP["lsp"] if tP["tlu"] else tP["lsc"]), \
      #      self.endLine)
    return kStr


  ##############################################################################
  # Global Read Addresses: Branch A/B - TODO
  ##############################################################################
  def graBranch(self, kernel, tP):
    kStr = ""
    for l in range(0, tP["nrt"]):
      gro = "(globalReadOffset%s%s_%u_0%s)" \
          % (tP["tensorChar"], tP["tileChar"], l, \
          (" + (VECTOR_WIDTH-1)" if tP["rtc"] else "") )
      limit = "size%s" % (tP["tileChar"])
      kStr += "  bool inBounds%s_%u = %s < %s;%s" \
          % (tP["tensorChar"], l, gro, limit, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  def graShift(self, kernel, tP):
    kStr = ""
    for l in range(0, tP["nrt"]):
      for s in range(0, 1 if tP["rc"] else tP["nrtv"]):
        #gro = "globalReadOffset%s%s_%u_%u" \
        #    % (tP["tensorChar"], tP["tileChar"], l, s )

        #limit = "(size%s-GLOBAL_LOAD_VECTOR_WIDTH_%s)" % (tP["tileChar"], tP["tensorChar"] )

        #kStr += "  %s = (%s > %s) ? %s+%u : %s;%s" \
        #    % (gro, gro, limit, limit, s, gro, self.endLine)

        kStr += "  globalReadOffset%s%s_%u_%u" \
            % (tP["tensorChar"], tP["tileChar"], l, s )
        kStr += " = ("
        kStr += "  globalReadOffset%s%s_%u_%u" \
            % (tP["tensorChar"], tP["tileChar"], l, s )
        kStr += " > "
        kStr += "size%s-%s" % (tP["tileChar"], "GLOBAL_LOAD_VECTOR_WIDTH_%s+%u"%(tP["tensorChar"], s) if tP["rtv"] else "1")
        kStr += ") ? "
        kStr += "size%s-%s" % (tP["tileChar"], "GLOBAL_LOAD_VECTOR_WIDTH_%s+%u"%(tP["tensorChar"], s) if tP["rtv"] else "1")
        kStr += " : "
        kStr += "globalReadOffset%s%s_%u_%u" \
            % (tP["tensorChar"], tP["tileChar"], l, s )
        kStr += ";%s" % self.endLine

    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    problemType = kernel["ProblemType"]
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            # Pass parms to GLOBAL_OFFSET_ macro:
            gro = "globalReadOffset%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp)

            kStr += "  %s %s = GLOBAL_OFFSET_%s( " \
                % (self.int64Str, gro, tP["tensorChar"])
            for i in range(0, len(tP["ia"])):
              index = tP["ia"][i]
              if index < kernel["ProblemType"]["NumIndicesC"]:
                if index == tP["tileIdx"]:
                  kStr += "(globalReadOffset%s%s_%u_%u)" \
                      % (tP["tensorChar"], tP["tileChar"], \
                      (para if tP["tlu"] else perp), \
                      (sPara if tP["tlu"] else sPerp) )
                else:
                  if isPackedIndex(kernel, index):
                    # pass vector per-tensor-dim offset (rather than scalar wg)
                    if index in problemType["IndicesBatch"] and not tP["PackBatchDims"]:
                      # pass 0, this is is the non-packed batch dim and must be 0
                      kStr += "0"
                    else:
                      kStr += "(globalReadOffset%s%s_%u_%u)" \
                          % (tc, \
                          self.indexChars[index],
                          (para if tP["tlu"] else perp), \
                          (sPara if tP["tlu"] else sPerp) )
                  else:
                    # just a non-vector group index
                    if kernel["ProblemType"]["StridedBatched"]:
                      kStr += "wg" + self.indexChars[index]
                    else:
                      kStr += "0"
              else: # summation index
                if index == kernel["ProblemType"]["IndexUnroll"]:
                  kStr += "(globalReadOffset%s%s_%u_%u)" \
                      % (tP["tensorChar"], self.unrollChar, \
                      (perp if tP["tlu"] else para), \
                      (sPerp if tP["tlu"] else sPara) )
                else:
                  kStr += "(globalReadOffset%s%s)" \
                      % (tP["tensorChar"], self.indexChars[index])
              if i < len(tP["ia"])-1:
                kStr += ", "
            kStr += " );%s" % self.endLine
            for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
              # subtract pad - this both helps efficiently detect OOB on the summation start and also
              # corrects the valid offsets for the start pad.
              (freeDim,sumDim) = zp[:2]
              freeDimChar = globalParameters["IndexChars"][freeDim]
              freeDimChar2 = self.indexChars[freeDim]
              sumChar = self.indexChars[sumDim]
              kStr += self.indent + gro + " -= padStart%s%s%s;"%(tc,freeDimChar, sumChar) + self.endLine
              freeOffset = "globalReadOffset%s%s_%u_%u" \
                      % (tc, freeDimChar2, \
                        (para if tP["tlu"] else perp), \
                        (sPara if tP["tlu"] else sPerp) )
              if sumDim == kernel["ProblemType"]["IndexUnroll"]:
                sumOffset = "globalReadOffset%s%s_%u_%u" \
                        % (tc, sumChar,
                        (perp if tP["tlu"] else para), \
                        (sPerp if tP["tlu"] else sPara) )
              else:
                sumOffset = "globalReadOffset%s%s" % (tc, sumChar)
              kStr += self.indent + \
                      "unsigned" + " " +\
                      gro + "_ZP%s%s =  %s*stride%s%s + %s*stride%s%s - padStart%s%s%s;" \
                            % (freeDimChar, sumChar,
                               freeOffset, tc,freeDimChar2,  sumOffset, tc, sumChar,   \
                               tc, freeDimChar, sumChar) + \
                      self.endLine
            if 0 and tP["isA"]:
              kStr += "printf(%sgid0=%%u %s=%%lu%s, %s(0), %s);" \
                       % (self.quote, gro, self.endLineQuote, \
                          self.getGlobalIdStr, gro) + self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP, isPap=False):
    kStr = ""

    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            kStr += "  %sDATA_TYPE const *globalRead%s_%u_%u_%u_%u = %s + globalReadOffset%s_%u_%u_%u_%u;%s" \
                % (self.globalPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, \
                tP["tensorChar"], tP["tensorChar"], \
                para, sPara, perp, sPerp, \
                self.endLine)
        #else:
        #    kStr += "  %sVECTOR_TYPE const *globalRead%s_%u_%u = (%sVECTOR_TYPE const *)(%s + globalReadOffset%s_%u_%u);%s" \
        #        % (self.globalPtrStr, tP["tensorChar"], para, perp, self.globalPtrStr, \
        #        tP["tensorChar"], tP["tensorChar"], para, perp, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A/B
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    tc = tP["tensorChar"]
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    isMirrorIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] in kernel["ProblemType"]["MirrorDims%s"%(tc)]
    declStr = "%s%s globalReadInc%s%s = %s(%s)stride%s%s" \
        % (self.indent, self.int64Str, tc, loopChar, \
        "-" if isMirrorIdx else "", self.int64Str, tc, loopChar)
    if loopIdx==self.unrollIdx:
      kStr += declStr
      if not kernel["PackSummationDims"]:
        # PSD recomputes load address using globalReadIncrementFromBase and includes LOCAL_DEPTHU multiple
        #- don't include it here
        kStr += "*LOCAL_DEPTHU"
      if kernel["GlobalSplitU"] > 1 \
          and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        kStr += "*GLOBAL_SPLITU"
    else:
      if kernel["PackSummationDims"]:
        # Skip the subtract of previous iteration since PSD compute load address using globalReadIncrementFromBase
        kStr += declStr
      else:
        # For Source kernel the address moves during the unroll loop
        # but not during the tail loop - so higher-order summations
        # need to only subtract the increments performed in the unroll
        # loop (truncate the iterations that are handled in tail loop).
        tmpChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx+1]]
        isPervMirrorIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx+1] in kernel["ProblemType"]["MirrorDims%s"%(tc)]
        if loopIdx+1 == self.unrollIdx:
          # special case needs to adjust (subtract) address incs made during unroll loop
          if kernel["GlobalSplitU"] > 1:
            numIter = "incNumIter%s_%s" % (self.unrollChar, tc)
            kStr += self.indent + "unsigned int %s = size%s/LOCAL_DEPTHU;" \
                    % (numIter, tmpChar) + self.endLine
            kStr += self.calculateLoopNumIterGsu(kernel, numIter, numIter, hidden=True)
            numIter += "*GLOBAL_SPLITU"
          else:
            numIter = "size%s/LOCAL_DEPTHU" % tmpChar
          kStr += declStr
          kStr += " %s stride%s%s*(" % ("+" if isPervMirrorIdx else "-", tc, tmpChar) + numIter + ")*LOCAL_DEPTHU"
        else:
          # other summation that does not immediately wrap the unroll inc:
          kStr += declStr
          kStr += " %s stride%s%s*(size%s)" % ("+" if isPervMirrorIdx else "-", tc, tmpChar, tmpChar)
    kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    kStr = self.comment("local write addresses: tile assignment %s"%tP["tensorChar"])
    kStr += "  unsigned int lw%s%s = (serial%s" \
        % (tP["tensorChar"], tP["tileChar"], ("%" if tP["grcg"] \
        == tP["tlu"] else "/") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")";
    if tP["grcv"] == tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s" % tP["tensorChar"]
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    kStr = self.comment("local write addresses: unroll assignment %s"%tP["tensorChar"])
    kStr += "  unsigned int lw%s%s = (serial%s" \
        % (tP["tensorChar"], self.unrollChar, ("/" if tP["grcg"] \
        == tP["tlu"] else "%") )
    if tP["grcg"]:
      kStr += (tP["lvc"] if tP["grcv"] else tP["lsc"])
    else:
      kStr += (tP["lsp"] if tP["grcv"] else tP["lvp"])
    kStr += ")";
    if tP["grcv"] != tP["tlu"]:
      kStr += "*GLOBAL_LOAD_VECTOR_WIDTH_%s" % tP["tensorChar"]
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Release tile related vgpr
  ##############################################################################
  def lwaReleaseTileVgpr(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP, uDu=0):
    kStr = ""
    kStr += "  unsigned int localWriteFirstOffset%s = lw%s%s + lw%s%s*(MT%s+PAD)%s;%s" \
        % (tP["tensorChar"], tP["tensorChar"], tP["tileChar"], \
        tP["tensorChar"], self.unrollChar, tP["tileChar"], \
        " + LDS_OFFSET_B" if tP["isB"] else "", self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    kStr = self.comment("local write addresses: final offsets %s" % tP["tensorChar"])
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "  unsigned int localWriteOffset%s_%u_%u_%u_%u = localWriteFirstOffset%s + (%u + %d*%s)" \
                % (tP["tensorChar"], para, sPara, perp, sPerp, \
                tP["tensorChar"], sPara if tP["tlu"] else sPerp, para, \
                (tP["lsc"] if not tP["tlu"] else tP["lsc"]) )
            if not tP["tlu"]:
              kStr += "*(MT%s+PAD)" % (tP["tileChar"])
            kStr += " + (%u + %d*%s)" % (
                sPerp if tP["tlu"] else sPara, perp, \
                (tP["lsp"] if tP["tlu"] else tP["lsp"]) )
            if tP["tlu"]:
              kStr += "*(MT%s+PAD)" % (tP["tileChar"])
            kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    kStr = self.comment("local write addresses: declare addresses %s" % tP["tensorChar"])
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "  %sDATA_TYPE *localWrite%s_%u_%u_%u_%u;%s"\
                % (self.sharedPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine )
    return kStr

  ##############################################################################
  # Local Write Addresses: Allocate tmpSgpr for initOpt
  ##############################################################################
  def lwaInitOptAllocate(self):
    return ""

  ##############################################################################
  # Local Write Addresses: Release tmpSgpr for initOpt
  ##############################################################################
  def lwaInitOptRelease(self):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment A/B
  ##############################################################################
  def lraTileAssignment(self, kernel, tPA, tPB):
    kStr = ""

    tP0 = tPA if tPB["tile01Idx"] else tPB
    tP1 = tPB if tPB["tile01Idx"] else tPA

    kStr += "  unsigned int lr%s = (serial %% SG%s);%s" \
        % (tP0["tileChar"], self.tileChar0, self.endLine)
    kStr += "  unsigned int lr%s = (serial / SG%s) %% SG%s;%s" \
        % (tP1["tileChar"], self.tileChar0, self.tileChar1, self.endLine)

    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int localReadOffset%s = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD)%s;%s" \
        % ( tP["tensorChar"], tP["tileChar"], tP["tileChar"], \
        " + LDS_OFFSET_B" if tP["isB"] else "", self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    kStr = ""
    kStr += "  %sDATA_TYPE *localRead%s;%s" % (self.sharedPtrStr, \
        tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Recalculate local write addresses A/B
  ##############################################################################
  def recalcLocalWriteAddresses(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Recalculate local read addresses A/B
  ##############################################################################
  def recalcLocalReadAddressesAB(self, kernel):
    return ""

  ##############################################################################
  # openShadowInit
  ##############################################################################
  def openShadowInit(self, kernel):
    return ""

  ##############################################################################
  # closeShadowInit
  ##############################################################################
  def closeShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    kStr = ""

    # init rC, in pf this is called twice
    kStr += self.endLine
    for i in range(0, kernel["ThreadTile0"]*kernel["ThreadTile1"]):
        kStr += "  rC[%u] = SCALAR_ZERO;%s" % (i, self.endLine)

    return kStr

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    kStr = ""
    for loopIdx in kernel["ProblemType"]["IndicesSummation"]:
      loopChar = self.indexChars[loopIdx]
      kStr += "%sint numIter%s;%s" \
          % (self.indent, loopChar, self.endLine)

    return kStr


  ##############################################################################
  # Declare stagger parms used for both A and B
  # Input is the number of loop iterations
  # Defines staggerUIter
  # staggerUIter must be power-of-2 to simplify masking implementation
  ##############################################################################
  def declareStaggerParms(self, kernel):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]

    # Number of elements in U accessed by the unroll loop:
    # Does not include elements accessed in tail loop
    kStr += "  const unsigned origNumIter = numIter%s;%s" % (loopChar, self.endLine)

    if kernel["StaggerUMapping"] == 0:
      staggerInput = ("wg%s" % self.tileChar0)
    elif kernel["StaggerUMapping"] == 1:
      staggerInput = "wg%s" % self.tileChar1
    elif kernel["StaggerUMapping"] == 2:
      staggerInput = "wg2"
    elif kernel["StaggerUMapping"] == 3:
      staggerInput = "wgSerial"
    elif kernel["StaggerUMapping"] == 4:
      staggerInput = "0xffffffff" # all WG compute same stagger, this is test mode
    else:
      assert(0) # unsupported
    #kStr += "if (serial==0) printf(\"xWG:%u_%u progWG:%u_%u staggerUIterParm=%u\\n\", hc_get_group_id(0), hc_get_group_id(1), wg0I, wg1J, staggerUIterParm);"  + self.endLine
    kStr += "  unsigned staggerUIter = (%s & staggerUIterParm);%s" % (staggerInput, self.endLine)

    bpeAB = int(4*kernel["ProblemType"]["DataType"].numRegisters())
    kStr += "  staggerUIter = (staggerUIter << %u); // shift so each stagger has %u-byte stride%s" \
            % (kernel["_staggerStrideShift"], \
              (1<<kernel["_staggerStrideShift"])*kernel["DepthU"]*bpeAB, self.endLine)
    #kStr += "if (serial==0) printf(\"WG:%u_%u progWG:%u_%u staggerUIter=%u\\n\", hc_get_group_id(0), hc_get_group_id(1), wg0I, wg1J, staggerUIter);"  + self.endLine
    #kStr += "  staggerUIter = 0;\n"

    if self.db["PrintStagger"]:
      kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) == 0)%s" % \
              (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
      kStr += "  printf(%sStaggerOffset loop init: numIter=%%u, staggerUIter=%%u, globalReadIncAL=%%lu globalReadIncBL=%%lu %s,\
                        numIter%s, staggerUIter, globalReadIncAL, globalReadIncBL);%s" \
                      % (self.quote, self.endLineQuote, loopChar, self.endLine)
    return kStr

  ##############################################################################
  # Calculate and apply stagger offsets
  #
  # To help with cache and memory parking, offset the start position in the
  # summation dimension so each group starts at a different K
  ##############################################################################
  def calculateStagger(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    loopIdx = self.unrollIdx
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            gr = "globalRead%s_%u_%u_%u_%u" \
                  % (tc,   para, sPara, perp, sPerp)

            ti= "globalReadOffset%s%s_%u_%u" \
                % (tc, tP["tileChar"], \
                (para if tP["tlu"] else perp), \
                (sPara if tP["tlu"] else sPerp) )

            kStr += "  %s += (staggerUIter * globalReadInc%s%s); // apply stagger offset%s" \
                    % (gr, tc, loopChar, self.endLine)

            if self.db["PrintStagger"]:
              kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 16)%s" % \
                      (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
              # typecasting to work around hcc printf bugs:
              kStr += "printf(%sStaggerOffset init: gid=%%u.%%u.%%u, ti=0x%%x  %s-%s=0x%%x%s, \
                              %s(2),%s(1),%s(0), %s,  (unsigned)(size_t)(%s-%s));%s" \
                             % (self.quote,\
                                gr, tc,
                                self.endLineQuote,\
                                self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr,\
                                ti,  gr, tc, \
                                self.endLine)

    # if prefetching then wrap iteration needs to change since already prefetched
    # some tiles before entering the unroll loop
    kStr += self.endLine
    if tP["isB"]:
      kStr += "  staggerUIter += %u; // add PrefetchGlobalRead%s" \
          % (kernel["PrefetchGlobalRead"], self.endLine)
      # StaggerUIter is now the loop iteration where we should wrap the offset back to 0

    return kStr

  ##############################################################################
  # Remove the stagger offset for the kernel
  # (used in tail loop)
  ##############################################################################
  def removeStagger(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    loopIdx = self.unrollIdx
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            gr = "globalRead%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp)

            if self.staggerU:
              kStr += "  %s += ((origNumIter - (staggerUIter - %u)) * globalReadInc%s%s); // remove stagger offset%s" \
                      % (gr, kernel["PrefetchGlobalRead"], tc, loopChar, self.endLine)

              if self.db["PrintStagger"]:
                kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 8)%s" % \
                        (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
                kStr += "printf(%sStaggerOffset remove: gid=%%u.%%u.%%u, origNumIter=%%u staggerUIter=%%u %s=%%p %s=%%p %s, \
                                %s(2),%s(1),%s(0), origNumIter, staggerUIter, %s, %s);%s" \
                               % (self.quote, \
                                  tc, gr, \
                                  self.endLineQuote, \
                                  self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr,
                                  tc, gr, \
                                  self.endLine)

    return kStr


  ##############################################################################
  # Emit code to compute loop iterations for GSU.
  # If the unroll summation size is not evenly divisible by GSU, then
  # some of the CUs working on same output space may perform different
  # numbers of unroll loop iterations.  Account for that here.
  # This is a separate function since the graInc for multiple summations
  # needs to know the #loop iters as well, so this code allows the
  # code to be replicated in multiple places.
  ##############################################################################
  def calculateLoopNumIterGsu(self, kernel, srcIterVar, dstIterVar, hidden):
    kStr = ""
    if hidden:
      kStr += self.indent + "{" + self.endLine
      indent = self.indent + "  "
    else:
      indent = self.indent
    kStr += "%sunsigned int numIterMyWg = %s / GLOBAL_SPLITU;%s" \
        % (indent, srcIterVar, self.endLine)
    kStr += "%sunsigned int numIterPerWgRemainder = %s %% GLOBAL_SPLITU;%s" \
        % (indent, srcIterVar, self.endLine)
    kStr += "%sif (gsuSumIdx < numIterPerWgRemainder) {%s" \
        % (indent, self.endLine)
    kStr += indent + "  numIterMyWg ++;" + self.endLine
    kStr += "%s}%s" % (indent, self.endLine)
    kStr += "%s%s = numIterMyWg;%s" \
        % (indent, dstIterVar, self.endLine)
    if hidden:
      kStr += self.indent + "}" + self.endLine
    return kStr


  ##############################################################################
  # Calculate Loop Num Iterations
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx, isPap):
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx

    kStr = ""
    problemType = kernel["ProblemType"]
    loopDim  = problemType["IndicesSummation"][loopIdx]
    loopChar = self.indexChars[loopDim]
    if tailLoop:
      kStr += self.endLine + "  /* Compute tail loop num iter */" + self.endLine
      if kernel["PackSummationDims"]:
          totalIters = "(size%s" % self.unrollChar
          for os in range(self.otherSummations):
            otherSumChar = self.indexChars[problemType["IndicesSummation"][os]]
            totalIters += "*size%s" % otherSumChar
          totalIters += ")"
      else:
          totalIters = "size%s" % self.unrollChar
      kStr += "%snumIter%s = (((%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU);%s" \
          % (self.indent, self.unrollChar, totalIters, self.endLine)
      if kernel["GlobalSplitU"] > 1:
        kStr += "%sif (gsuSumIdx != numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIter%s = 0;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
        #kStr += "if (serial==0) printf(\\\"WG%u_%u TK:%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    else:
      kStr += self.endLine + "  /* Compute summation loop num iter */" + self.endLine

      # Check alpha == 0
      zeroStr = kernel["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
      kStr += self.indent + "if(alpha == %s) size%s = 0;"%(zeroStr, loopChar) + "  // Short circuit check alpha=0, skip A*B " + self.endLine

      if loopIdx == self.unrollIdx and kernel["GlobalSplitU"] > 1:
        kStr += self.calculateLoopNumIterGsu(kernel, "(size%s / LOCAL_DEPTHU)"%loopChar, \
                                             "numIter%s"%loopChar, hidden=False)
        #kStr += "if (serial==0) printf(\\\"WG%u_%u UK:%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine

        if self.unrollIncIsDepthU:
            kStr += self.indent + "numIter%s *= LOCAL_DEPTHU;"%loopChar + self.endLine
      else:
        kStr += self.indent + "numIter%s = size%s" \
            % (loopChar, loopChar)
        if not self.unrollIncIsDepthU and loopIdx == self.unrollIdx:
            kStr += " / LOCAL_DEPTHU"
        kStr += ";" + self.endLine

      if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
        kStr += self.indent + "unsigned int psdIter=0; // packed summation dim iterator" + self.endLine

      zpA = self.zpForSumIdx(loopDim, problemType["ZeroPadA"])
      zpB = self.zpForSumIdx(loopDim, problemType["ZeroPadB"])
      for (zp,tc) in ((zpA,'A'), (zpB,'B')):
        if zp:
          (freeDim,sumDim) = zp[:2]
          freeDimChar = globalParameters["IndexChars"][freeDim]
          freeDimChar2 = self.indexChars[freeDim]
          sumChar = self.indexChars[sumDim]
          kStr += "%sunsigned int elementEdge%s%s = stride%s%s * size%s + stride%s%s*(size%s - 1) - padStart%s%s%s - padEnd%s%s%s;" \
              % (self.indent, tc, loopChar, tc, freeDimChar2, freeDimChar2, tc, loopChar, loopChar, tc, freeDimChar, sumChar, tc, freeDimChar, sumChar) \
              + self.endLine

          if sumChar not in self.definedIter:
            kStr += self.indent + "unsigned int iter%s = 0;" % sumChar + self.endLine
            self.definedIter.add(sumChar)

    return kStr



  ##############################################################################
  # Open Loop
  ##############################################################################
  def openLoop(self, kernel, loopIdx, uDu=0, noLabelGen=False, beginLabelOnly=False):
    problemType = kernel["ProblemType"]
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx

    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if kernel["LoopDoWhile"] and not tailLoop:
      kStr += "%sdo {%s" % (self.indent, self.endLine)
      self.indent += "  "
      kStr += "%snumIter%s--;%s" % (self.indent, loopChar, self.endLine)
      assert(not self.unrollIncIsDepthU)
    else:
      if self.unrollIncIsDepthU and loopIdx==self.unrollIdx and not tailLoop:
        if kernel["PackSummationDims"]:
          totalIters = "("
          totalIters += "*".join(["numIter%s"%(self.indexChars[os]) for os in problemType["IndicesSummation"]])
          totalIters += ")"
        else:
          totalIters = "numIter%s" % loopChar
        kStr += self.indent \
                + "while (psdIter < %s) {" % (totalIters) \
                + self.endLine
      else:
        kStr += "%swhile (numIter%s-- > %u) {%s" \
            % (self.indent, loopChar, \
            (1 if (kernel["PrefetchGlobalRead"] and loopIdx == self.unrollIdx \
            and not tailLoop) else 0), self.endLine)
      self.indent += "  "
    #if tailLoop:
    #  kStr += "if (serial==0) printf(\\\"WG%u_%u: ti=%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    #else:
    #  kStr += "if (serial==0) printf(\\\"WG%u_%u: ui=%u\\\\n\\\", get_group_id(0), get_group_id(1), numIterK);" + self.endLine
    return kStr

  ##############################################################################
  # Close Loop
  ##############################################################################
  def closeLoop(self, kernel, loopIdx, finalLoop, loopCopies, uDu=0, emitEndLabelOnly=False, oddLabel=False, \
                skipCondJumpCounter=-1, isOptNLL=False, skipJump=False):
    tailLoop = loopIdx < 0
    kStr = ""
    if emitEndLabelOnly:
      return kStr
    problemType = kernel["ProblemType"]
    loopDim = problemType["IndicesSummation"][loopIdx]
    loopChar = self.indexChars[loopDim]

    for tc in ('A', 'B'):
      # assume A and B don't specify same summation idx
      zp = next((zpi for zpi in problemType["ZeroPad%s"%tc] if zpi[1] == loopDim), None)
      if zp:
        if loopIdx == self.unrollIdx:
          incAmount = "LOCAL_DEPTHU"
          if kernel["GlobalSplitU"] > 1 \
              and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
            incAmount += "*GLOBAL_SPLITU"
        else:
          incAmount = "1"

    self.indent = self.indent[2:]
    if kernel["LoopDoWhile"] and not tailLoop:
      kStr += "%s} while (numIter%s > %u);%s" \
          % (self.indent, loopChar, \
          (1 if kernel["PrefetchGlobalRead"] else 0), self.endLine )
    else:
      kStr += "%s}%s" % (self.indent, self.endLine)
    #kStr += "if (serial==0) printf(\\\"WG%u_%u: rc0=%.0f\\\\n\\\", get_group_id(0), get_group_id(1), rC[0]);" + self.endLine
    return kStr

  ##############################################################################
  # Close Loop
  ##############################################################################
  def openLoopCopy(self, kernel, lc):
    return ""

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self,kernel, label = None, isOptNLL = False):
    return ""

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, black, iuiCount, useMacro, isTail=False):
    kStr = ""
    for iui in range(0,iuiCount):
        kStr += "%sMAC_%ux%u" % (self.indent, \
            kernel["ThreadTile0"],kernel["ThreadTile1"])
        if black:
          kStr += "_BLK"
        kStr += self.endLine
    return kStr

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isPap):
    kStr = ""
    if kernel["GlobalSplitU"] > 1:
      kStr += "%sif (numIterMyWg >= 1) {%s" \
          % (self.indent, self.endLine)
    else:
      kStr += "%sif (size%s >= LOCAL_DEPTHU) {%s" \
          % (self.indent, self.unrollChar, self.endLine)
    self.indent += "  "
    return kStr

  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isPap, isNGLL):
    kStr = ""
    self.indent = self.indent[2:]
    kStr += "%s} // end %s%s" % \
        (self.indent, "PrefetchGlobalRead" if prefetch else "unroll", self.endLine)
    if prefetch:
      kStr += "%selse { // still need to initC even if skipped prefetch%s" % (self.indent, self.endLine)
      kStr += self.initC(kernel)
      kStr += "%s}%s" % (self.indent, self.endLine)

    return kStr


  def globalReadIncCheckStagger(self, iterVar, loopChar, tP, para, sPara, perp, sPerp):
    kStr = ""
    tc = tP["tensorChar"]

    # Check to see if GRA wraps around edge:
    gr = "globalRead%s_%u_%u_%u_%u" \
            % (tP["tensorChar"], para, sPara, perp, sPerp)

    kStr += "%sif ((%s) == staggerUIter) {%s" \
            % (self.indent, iterVar, self.endLine)

    if self.db["PrintStagger"]:
      # note loop counter numIterK/numIterL hard-coded, manually hack if needed
      kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 16)%s" % \
              (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
      kStr += "printf(%sStaggerOffset wrap-gro: gid=%%u.%%u.%%u, old GR-%s=0x%%x numIter=%%u staggerUIter=%%u%s,\
                        %s(2),%s(1),%s(0), (unsigned)(size_t)(%s-%s), numIterL, staggerUIter);%s" \
                       % (self.quote, \
                          tc, \
                          self.endLineQuote, \
                          self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, \
                          gr,tc, \
                          self.endLine)

    kStr += "  %s%s -= (origNumIter * globalReadInc%s%s); // wrap staggered offset back to row start%s" \
            % (self.indent, \
               gr,  tc, loopChar,
               self.endLine)
    kStr += "%s}%s" % (self.indent, self.endLine)
    if self.db["PrintStagger"]:
      kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 8)%s" % \
              (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
      kStr += "printf(%sStaggerOffset check-gro: gid=%%u.%%u.%%u, GR-%s=0x%%x %s, \
                      %s(2),%s(1),%s(0), (unsigned)(size_t)(%s-%s));%s" \
                     % (self.quote, \
                        tc, \
                        self.endLineQuote, \
                        self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, \
                        gr,tc, \
                        self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Increment either A or B
  # Called from globalReadIncrementAB below
  ##############################################################################
  def globalReadIncrement(self, kernel, loopIdx, tP, prefetchIndex, incs=1):
    kStr = ""
    tc = tP["tensorChar"]
    loopChar = self.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += self.comment("global read inc %s for sum%c"%(tc,loopChar))
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            globalRead = "globalRead%s_%u_%u_%u_%u" % (tc, para, sPara, perp, sPerp)
            kStr += "%s%s = (%sDATA_TYPE const *)( ((%sDATA_TYPE const *)%s) + %s*globalReadInc%s%s);%s" \
                % (self.indent, globalRead,
                self.globalPtrStr, self.globalPtrStr,
                globalRead,
                incs, tc, loopChar, \
                self.endLine)

            if self.staggerU and loopIdx==self.unrollIdx:
              kStr += self.globalReadIncCheckStagger("numIter%s"%loopChar, loopChar, tP, para, sPara, perp, sPerp)

          #else:
          #  kStr += "%sglobalRead%s_%u_%u%s += globalReadInc%s%s%s;%s" \
          #      % (self.indent, tP["tensorChar"], para, perp, \
          #      (("_s%u"%s) if (tP["rtc"] \
          #      or tP["ruc"]) else ""), \
          #      tP["tensorChar"], loopChar, "" if (tP["rtc"] \
          #      or tP["ruc"]) else "/VECTOR_WIDTH", \
          #      self.endLine)
    return kStr

  def globalReadIncrementFromBase(self, kernel, tP, sumOffset, loopChar):
    """ Recompute the address, starting from base address pointer + initial offset + summation offset """
    kStr = ""
    tc = tP["tensorChar"]
    kStr += self.comment("global read inc %s from base"%(tc))
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1 if tP["rc"] else tP["nrcv"]):
            globalRead = "globalRead%s_%u_%u_%u_%u" % (tc, para, sPara, perp, sPerp)
            kStr += self.indent + \
                "%s = %s + globalReadOffset%s_%u_%u_%u_%u + %s;" % (
                  globalRead,
                  tc,
                  tc, para, sPara, perp, sPerp,
                  sumOffset
                ) + self.endLine
            if self.staggerU:
              kStr += self.globalReadIncCheckStagger("iter%s"%loopChar, loopChar, tP, para, sPara, perp, sPerp)

    return kStr


  ##############################################################################
  # Global Read: Increment A and B
  # Called from KernelWriter
  # If PackSummationDims=1, this increments all counters for A and B
  ##############################################################################
  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    imod = Code.Module("globalReadIncrementAB%s")

    problemType = kernel["ProblemType"]
    unrollChar = self.indexChars[problemType["IndicesSummation"][self.unrollIdx]]

    headerCode = ""
    if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
      headerCode += self.endLine
      headerCode += self.indent + "psdIter += LOCAL_DEPTHU;" + self.endLine

    if loopIdx==self.unrollIdx and kernel["PackSummationDims"] and self.actualSummationLoops==1:
      kStr = headerCode
      if prefetchIndex>0:
        psdPackedBits = "(LOCAL_DEPTHU)"
      else:
        psdPackedBits = "(psdIter)"
      for os in reversed(range(problemType["NumIndicesSummation"])):
        # Only get here if we are packing summation dims
        sumDim  = problemType["IndicesSummation"][os]
        sumChar = self.indexChars[sumDim]
        firstIter = (os==problemType["NumIndicesSummation"]-1)
        lastIter  = (os==0)

        kStr += self.endLine
        iterType = "" if sumChar in self.definedIter else "unsigned int "
        if not lastIter:
          c = "//" if self.psdUuseMagic else "" # show non-magic code commented out
          kStr += self.indent + c + iterType + "iter%s = %s %% numIter%s;" % \
              (sumChar, psdPackedBits, sumChar) + self.endLine

          kStr += self.indent + c
          if firstIter:
            kStr += "unsigned int "
          kStr += "psdPackedBits = %s / numIter%s;" % (psdPackedBits, sumChar) + self.endLine

          if self.psdUuseMagic:
            assert kernel["MagicDivAlg"] == 2  # older alg not supported
            kStr += self.indent
            if firstIter:
              kStr += "unsigned int "
            if os == self.unrollIdx and kernel["GlobalSplitU"]>1:
              magicStruct = "((gsuSumIdx < numIterPerWgRemainder) ? magicStruct%s_GsuRemainder : magicStruct%s)"\
                  % (sumChar, sumChar)
            else:
              magicStruct = "magicStruct%s" % sumChar
            kStr += "tmpBits = MAGIC_DIV2(%s, %s);" % (psdPackedBits, magicStruct) + self.endLine
            kStr += self.indent + iterType + "iter%s = %s - tmpBits*numIter%s;" % \
                (sumChar, psdPackedBits, sumChar) + self.endLine

            kStr += self.indent
            if firstIter:
              kStr += "unsigned int "
            kStr += "psdPackedBits = tmpBits;" + self.endLine

          # set up bits for next iteration:
          psdPackedBits = "psdPackedBits"
        else:
          # last iteration:
          kStr += self.indent + iterType + "iter%s = %s;" % (sumChar, psdPackedBits) + self.endLine

        # update psdOffset:
        for (tc) in ('A','B'):
          kStr += self.indent
          if firstIter:
            kStr += self.int64Str + " psdOffset%s = " % tc
          else:
            kStr += "psdOffset%s += " % tc
          kStr += "iter%s*globalReadInc%s%s;" % (sumChar, tc, sumChar)
          kStr += self.endLine

      # Add the psdOffsets for A and B:
      for (tc,tP) in (('A',self.tPA),('B',self.tPB)):
        # makeSchedule is linked to the modules names - update both together
        incCode = Code.Module("globalReadIncrement%s"%tc)
        kStr += self.indent + self.globalReadIncrementFromBase(kernel, tP, "psdOffset%s"%tc, unrollChar)
        incCode.addText(kStr)
        kStr = ""
        imod.addCode(incCode)
    else:
      # Non pack-summation-dims code path:
      incA = Code.Module("globalReadIncrementA")
      incA.addText(headerCode)
      incA.addText(self.globalReadIncrement(kernel, loopIdx, self.tPA, prefetchIndex, incs))
      imod.addCode(incA)

      incB = Code.Module("globalReadIncrementB")
      incB.addText(self.globalReadIncrement(kernel, loopIdx, self.tPB, prefetchIndex, incs))
      imod.addCode(incB)

    return imod

  ##############################################################################
  # DirectToLds M0 update: Do It A/B
  ##############################################################################
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    tc = tP["tensorChar"]
    imod = Code.Module("directToLdsM0Update%s_%u"%(tc,mode))
    return imod


  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    kStr = ""
    tc = tP["tensorChar"]

    guardK = (mode==2)
    kStr += self.comment("global read %s")%tc

    #for perp in range(0, tP["nrp"]):
    #  for para in range(0, tP["nrc"]):
    #    for s in range(0, numUnrollVectorComponents):
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]):
            dest ="%s_%u_%u_%u_%u" \
                % (tP["tensorChar"].lower(), \
                para, sPara, perp, sPerp )
            kStr += "%s%s = " % (self.indent, dest)
            isMirrorIdx = kernel["ProblemType"]["IndicesSummation"][self.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"% tc]
            # guard around K
            guarded = 0
            if guardK:
              guarded = 1
              guardMirror = ""
              if isMirrorIdx:
                guardMirror += "- (size%s / LOCAL_DEPTHU)*LOCAL_DEPTHU" % (self.unrollChar)
              kStr += "( globalReadOffset%s%s_%u_%u %s %s %u >= (size%s %% LOCAL_DEPTHU%s)%s )" \
                  % (tP["tensorChar"], self.unrollChar, \
                  (perp if tP["tlu"] else para), (sPerp if tP["tlu"] else 0), \
                  guardMirror, "-" if isMirrorIdx and tc == 'B' else "+", \
                  (0 if tP["tlu"] else sPara), self.unrollChar, \
                  (" + LOCAL_DEPTHU*gsuSumIdx" if kernel["GlobalSplitU"]>1 \
                  else ""), (" || !numIter%s"%self.unrollChar) \
                  if kernel["GlobalSplitU"] > 1 else "")

            # guard around pad
            for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
              if guarded:
                kStr += " || "
              guarded = 1
              (freeDim, sumDim) = zp[:2]
              freeDimChar = globalParameters["IndexChars"][freeDim]
              sumChar = self.indexChars[sumDim]
              assert self.unrollIncIsDepthU
              if kernel["PackSummationDims"]:
                iterVar = "iter"+sumChar
              elif sumDim == kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]:
                iterVar = "psdIter"
              else:
                raise RuntimeError("ZP not supported with multiple summations and PSD==0")
              if sumDim in kernel["ProblemType"]["MirrorDims%s"%(tc)]:
                iterVar = "-" + iterVar

              globalReadOffsetZp = "globalReadOffset%s_%u_%u_%u_%u_ZP%s%s" \
                  % (tc, para, 0 if tP["rc"] else sPara, perp, sPerp, \
                     freeDimChar, sumChar);
              kStr += " ( (%s * stride%s%s + %s) >= elementEdge%s%s )" \
                      % (iterVar, tc, sumChar, globalReadOffsetZp, tc, sumChar)

            # guard around edge
            if kernel["EdgeType"] == "Branch":
              if guarded:
                kStr += " || "
              guarded = 1
              kStr += "( !inBounds%s_%u )" % ( \
                  (tP["tensorChar"], para if tP["tlu"] else perp) )
            if guarded:
              kStr += " ? SCALAR_OOB_DATA : "
            kStr += "*(globalRead%s_%u_%u_%u_%u %s %u);%s" \
                % (tP["tensorChar"], para, 0 if tP["rc"] else sPara, perp, sPerp, "-" if isMirrorIdx and tc == 'B' else "+", sPara if tP["rc"] else 0, \
                self.endLine)

            #if self.db["PrintStagger"] and tP["isA"]:
            if 0 and self.db["PrintStagger"]:
              kStr += "if (%s(2)==0 && %s(1)==0 && %s(0) <= 16)%s" % \
                      (self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr, self.endLine)
              kStr += "  printf(%sglobalRead: gid=%%u.%%u.%%u, %s loaded:%%.0f%s, \
                                %s(2),%s(1),%s(0), %s );%s" \
                             % (self.quote,\
                                tc,
                                self.endLineQuote, \
                                self.getGlobalIdStr, self.getGlobalIdStr, self.getGlobalIdStr,\
                                dest, \
                                self.endLine)
    return kStr

  ##############################################################################
  # Global Read A/B completed
  ##############################################################################
  def doneGlobalABReads(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "%slocalWriteOffset%s_%u_%u_%u_%u = (localWriteOffset%s_%u_%u_%u_%u + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
                % (self.indent, tP["tensorChar"], \
                para, sPara, perp, sPerp, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine )
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "%slocalWriteOffset%s_%u_%u_%u_%u %%= LDS_OFFSET_BLK;%s" \
                % (self.indent, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine )
    return kStr

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    kStr = self.comment("local write init pointers %s" % tP["tensorChar"])
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, 1): # tP["nwcv"]):
            kStr += "%slocalWrite%s_%u_%u_%u_%u = (%sDATA_TYPE *)(localMemory + localWriteOffset%s_%u_%u_%u_%u);%s"\
                % (self.indent, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.sharedPtrStr, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine)
    return kStr

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    kStr = ""
    LWCodeA = self.localWriteDo(kernel, tPA)
    LWCodeB = self.localWriteDo(kernel, tPB)
    kStr += self.comment("local write a")
    kStr += LWCodeA
    kStr += self.comment("local write b")
    kStr += LWCodeB
    return kStr

  ##############################################################################
  # Replace the determined vmcnt in PreLoop LocalWrite
  ##############################################################################
  def replacePreLoopLWVmcnt(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  def localWriteDo(self, kernel, tP, uDu=0):
    kStr = ""
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nwpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, roundUp(tP["nwcv"])):
            kStr += "%s*(localWrite%s_%u_%u_%u_%u + %u) = %s_%u_%u_%u_%u;%s" \
                % (self.indent, tP["tensorChar"], \
                para, 0, perp, sPerp, sPara, \
                tP["tensorChar"].lower(), \
                para, \
                sPara if tP["tlu"] else sPerp, \
                perp, \
                sPerp if tP["tlu"] else sPara, \
                self.endLine)
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine
    if False and tP["isB"]:
      kStr += "%s%s" % (self.syncStr, self.endLine)
      kStr += "    /* print Local state */" + self.endLine
      kStr += "    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine
      kStr += "      printf(\\\"lds[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" % self.endLine
      kStr += "    }" + self.endLine
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    kStr = ""
    kStr += "%slocalReadOffset%s = (localReadOffset%s + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
        % (self.indent, tP["tensorChar"], tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    kStr = ""
    kStr += "%slocalReadOffset%s %%= LDS_OFFSET_BLK;%s" \
        % (self.indent, tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    kStr = ""
    kStr += "%slocalRead%s = (%sDATA_TYPE *)(localMemory + localReadOffset%s);%s" \
        % (self.indent, tP["tensorChar"], self.sharedPtrStr, \
        tP["tensorChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    kStr = ""
    kStr += "%slocalRead%s += LOCAL_SPLITU*(MT%s+PAD);%s" \
        % (self.indent, tP["tensorChar"], tP["tileChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  def localReadDo(self, kernel, black, iui, epsi, tP):
    tc   = tP["tensorChar"]
    imod = Code.Module("LocalReadDo%s_I%s"%(tc,iui))
    pack = Code.Module("pack%s_I%s"%(tc,iui))

    for r in range(0, kernel[tP["tt"]]//kernel["VectorWidth"]):
      for s in range(0, kernel["VectorWidth"]):
        imod.addCode("%sr%s[%u*VECTOR_WIDTH+%u%s] = localRead%s[%u*SG%s*VECTOR_WIDTH + %u]; %s" \
            % (self.indent, tP["tensorChar"], r, s, \
            (("+TT%s"%tP["tileChar"]) if black else ""), \
            tP["tensorChar"], r, tP["tileChar"], s, self.endLine))

    return imod, pack

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - wg%s*MT%s;%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(tP["tileChar"], tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% GLOBAL_LOAD_VECTOR_WIDTH_%s;%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tensorChar"], self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH) %% SG%s) == (serial %s SG%s)%s ) {%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tileChar"], "%" if tP["isA"] else "/", \
        self.tileChar0, (" %% SG%s"%self.tileChar1) if tP["isB"] else "", self.endLine)

    # old
    #kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
    #    % (tP["tileChar"], tP["tileChar"], tP["tileChar"], self.endLine)
    # new
    # (wgMT/(SG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
    kStr += "    unsigned int s%s = (wgMT%s%%VECTOR_WIDTH)/GLOBAL_LOAD_VECTOR_WIDTH_%s + (wgMT%s/(SG%s*VECTOR_WIDTH))*(VECTOR_WIDTH/GLOBAL_LOAD_VECTOR_WIDTH_%s);%s" \
        % (tP["tileChar"], tP["tileChar"], tP["tensorChar"], \
        tP["tileChar"], tP["tileChar"], tP["tensorChar"], self.endLine)

    for r in range(1, tP["glvw"]):
      kStr += "    if (r%s == %u) {%s" % (tP["tileChar"], r, self.endLine)
      numVectors = kernel["ThreadTile%s"%(tP["tile01Idx"])]//tP["glvw"]
      for vIdx in range(0, numVectors):
        if vIdx == 0:
          kStr += "      "
        else:
          kStr += " else "
        if vIdx < numVectors-1:
          kStr += "if (s%s == %u) " % (tP["tileChar"], vIdx)
        kStr += "{%s" % self.endLine
        for tt in range(0, kernel["ThreadTile%u"%(((tP["tile01Idx"])+1)%2)]):
          for s in range(0, r):
            if tP["isA"]:
              kStr += "        rC[%u + %u*GLOBAL_LOAD_VECTOR_WIDTH_A + %u*TT%s] = rC[%u + %u*GLOBAL_LOAD_VECTOR_WIDTH_A + %u*TT%s];%s" \
                % (s, vIdx, tt, self.tileChar0, \
                s+tP["glvw"]-r, vIdx, tt, self.tileChar0, \
                self.endLine)
            else:
              kStr += "        rC[%u + %u*TT%s*GLOBAL_LOAD_VECTOR_WIDTH_B + %u*TT%s] = rC[%u + %u*TT%s*GLOBAL_LOAD_VECTOR_WIDTH_B + %u*TT%s];%s" \
                % (tt, vIdx, self.tileChar0, s, self.tileChar0, \
                tt, vIdx, self.tileChar0, \
                s+tP["glvw"]-r, self.tileChar0, self.endLine)
        #kStr += "printf(\\\"sv %u %u\\\");%s" % (r, vIdx, self.endLine)
        kStr += "      }"
        if vIdx == numVectors-1:
          kStr += self.endLine
      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr


  ##############################################################################
  # Shift Vectors Components d1
  ##############################################################################
  def shiftVectorComponents1(self, kernel, tP):
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - %s*MT%s;%s" \
        % (self.tileChar1, self.tileChar1, "wg%s"%self.tileChar1, \
        self.tileChar1, self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar1, self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% VECTOR_WIDTH;%s" \
        % (self.tileChar1, self.tileChar1, self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH) %% SG%s) == ((serial / SG%s) %% SG%s) ) {%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar0, self.tileChar1, \
        self.endLine)
    kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
    for r1 in range(1, tP["glvw"]):
      kStr += "    if (r%s == %u) {%s" % (self.tileChar1, r1, self.endLine)
      numVectors = kernel["ThreadTile1"]/tP["glvw"]
      for vIdx in range(0, numVectors):
        if vIdx == 0:
          kStr += "      "
        else:
          kStr += " else "
        if vIdx < numVectors - 1:
          kStr += "if (s%s == %u) " % (self.tileChar1, vIdx)
        kStr += "{%s" % self.endLine

        for s in range(0, r1):
          for tt0 in range(0, kernel["ThreadTile0"]):
            kStr += "        rC[%u+%u*TT%s*VECTOR_WIDTH + %u*TT%s] = rC[%u+%u*TT%s*VECTOR_WIDTH + %u*TT%s];%s" \
                % (tt0, vIdx, self.tileChar0, s, self.tileChar0, \
                tt0, vIdx, self.tileChar0, \
                s+tP["glvw"]-r1, self.tileChar0, self.endLine)

        kStr += "      }"
        if vIdx == numVectors - 1:
          kStr += self.endLine

      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Complex Declare Tmp Registers
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    kStr = ""
    if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
      kStr += "  float type_mac_tmp;" + self.endLine
    if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
      kStr += "  double type_mac_tmp;" + self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    kStr += "  %sCOMPUTE_DATA_TYPE *localLocalSplitU = (%sCOMPUTE_DATA_TYPE *)(localMemory);%s" \
      % (self.sharedPtrStr, self.sharedPtrStr, self.endLine)
    for j in range(0, kernel["ThreadTile1"] // kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"] // kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"]):
            kStr += "%slocalLocalSplitU[%u + (lr%s + %u*SG%s + (MT%s/VECTOR_WIDTH)*(lr%s*VECTOR_WIDTH + %u + SG%s*VECTOR_WIDTH*%u) + (MT%s*MT%s/VECTOR_WIDTH)*sgId)*VECTOR_WIDTH] = rC[%u + (%u+%u*(TT%s/VECTOR_WIDTH)+%u*TT%s)*VECTOR_WIDTH];%s" \
              % (self.indent, vc, self.tileChar0, i, self.tileChar0, \
                self.tileChar0, self.tileChar1, \
                s, self.tileChar1, j, self.tileChar0, self.tileChar1, vc, i, s, \
                self.tileChar0, j, self.tileChar0, self.endLine)
    kStr += self.syncThreads(kernel);
    """

    kStr += "    /* print Local state */" + self.endLine
    kStr += "    for (unsigned int i = serial; i < MT0I*MT1J*LOCAL_SPLITU; i+=NUM_THREADS) {%s" % self.endLine
    kStr += "      printf(\\\"localLocalSplitU[%%06u] = %%10.0f, %%10.0f\\\\n\\\", i, localLocalSplitU[i], localLocalSplitU[i]);%s" \
        % self.endLine
    kStr += "    }" + self.endLine
    """
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for s in range(0, kernel["VectorWidth"]):
        kStr += "  rC[%u + %3u*GLOBAL_WRITE_VECTOR_WIDTH] = localLocalSplitU[%u + (serial+%u*NUM_THREADS)*GLOBAL_WRITE_VECTOR_WIDTH];%s" \
            % (s, i, s, i, self.endLine)
    kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"]):
          kStr += "  rC[%u + %3u*GLOBAL_WRITE_VECTOR_WIDTH] += localLocalSplitU[(%u + serial*GLOBAL_WRITE_VECTOR_WIDTH+%u*NUM_THREADS*GLOBAL_WRITE_VECTOR_WIDTH + %u*MT%s*MT%s)];%s" \
              % (s, i, s, i, r, self.tileChar0, self.tileChar1, self.endLine)
      kStr += self.endLine
    return kStr



  ##############################################################################
  # extractGlobalCDims: Extract the packed dims from mask(s)
  #
  # tensorIdx:
  ##############################################################################
  def extractGlobalCDims(self, kernel, base, tensorIdx):
    kStr = ""
    lastIndex = None
    if tensorIdx == 0:
      flattenedGlobalC = "flattenedGlobalC0"
    elif tensorIdx == 1:
      flattenedGlobalC = "flattenedGlobalC1"

    first = 1
    for idx in kernel["PackedC%sIndicesX"%tensorIdx]:
      kStr += "  globalC%s = " % (self.indexChars[idx])

      if first:
        # first print just copies flattenedGlobalC1 - no div / mod
        first = 0
        kStr += "  %s" % (flattenedGlobalC)
        if base:
          kStr += " + %s" % base
        kStr += ";" + self.endLine
      else:
        # later iterations extract dimension from previous using mod,
        # then div to remove the extracted bits for next iteration
        #kStr += "printf(\"pre: serial:%%u wg0:%%u wg1:%%u globalC0I:%%u globalC1J:%%u\\n\", serial, wg0I, wg1J, globalC0I, globalC1J);%s" % (self.endLine)
        if kernel["MagicDivAlg"]:
          c = globalParameters["IndexChars"][lastIndex]
          if kernel["MagicDivAlg"]==1:
            kStr += "MAGIC_DIV1(globalC%s, magicNumberSize%s, magicShiftSize%s);%s" \
                    % (self.indexChars[lastIndex], c, c, self.endLine)
          elif kernel["MagicDivAlg"]==2:
            kStr += "MAGIC_DIV2(globalC%s, magicStruct%s);%s" \
                    % (self.indexChars[lastIndex], c, self.endLine)
          kStr += "  globalC%s -= (globalC%s*size%s);%s" \
                  % (self.indexChars[lastIndex], self.indexChars[idx], \
                     self.indexChars[lastIndex], self.endLine)
        else:
          kStr += "(globalC%s) / size%s;%s" % (self.indexChars[lastIndex], self.indexChars[lastIndex], self.endLine)
          kStr += "  globalC%s %%= size%s;%s" % (self.indexChars[lastIndex], self.indexChars[lastIndex], self.endLine)

      lastIndex = idx
      #kStr += "printf(\"post: serial:%%u wg0:%%u wg1:%%u globalC0I:%%u globalCK=%%u\\n\", serial, wg0I, wg1J, globalC0I, globalCK);%s" % (self.endLine)

    return kStr

  ##############################################################################
  # globalWriteWorkGroupInitBeforePersistentLoop:
  ##############################################################################
  def globalWriteWorkGroupInitBeforePersistentLoop(self, kernel):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################
  def globalWriteWorkGroupInit(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    # Add Index0
    index0 = kernel["ProblemType"]["Index0"]
    kStr += "  unsigned int localC%s = (serial %% (MT%s/GLOBAL_WRITE_VECTOR_WIDTH))*GLOBAL_WRITE_VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  unsigned int globalC%s = (wg%s)" % (self.indexChars[index0], self.indexChars[index0])
    kStr += "*MT%s + localC%s;%s" % (self.tileChar0, self.tileChar0, self.endLine)
    # Save original flattened C0 before extracting batch components:
    kStr += "  unsigned int flattenedGlobalC0 = globalC%s;%s" \
        % (self.indexChars[index0], self.endLine)

    # Add Index1
    index1 = kernel["ProblemType"]["Index1"]
    kStr += "  unsigned int localC%s = serial / (MT%s/GLOBAL_WRITE_VECTOR_WIDTH);%s" \
        % (self.tileChar1, self.tileChar0, self.endLine)
    kStr += "  unsigned int globalC%s = (wg%s)" % (self.indexChars[index1], self.indexChars[index1])
    kStr += "*MT%s + localC%s;%s" % (self.tileChar1, self.tileChar1, self.endLine)
    kStr += "  unsigned int flattenedGlobalC1 = globalC%s;%s" \
        % (self.indexChars[index1], self.endLine)

    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      if i != index0 and i != index1:
        kStr += "  unsigned int globalC%s = " \
            % (self.indexChars[i])
        if isPackedIndex(kernel,i):
          kStr += "0; // define, will be set below%s" % (self.endLine)
        elif kernel["ProblemType"]["StridedBatched"] or kernel["_GlobalAccumulation"]:
          kStr += "(wg%s);%s" % (self.indexChars[i], self.endLine)
        else:
          kStr += "0;%s" % (self.endLine)

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      indexChar = self.indexChars[0]
      kStr += "  %s strideW = 1 + (size%s - 1) " % (self.uint64Str, indexChar)
      for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
        strideStr = "1"
        for j in range(0, i):
          strideStr += " * size%s" % (self.indexChars[j])
        indexChar = self.indexChars[i]
        kStr += " + (size%s - 1) * %s" % (indexChar, strideStr)
      kStr += ";" + self.endLine

    return kStr

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    kStr = ""
    packGranularity = kernel["PackGranularity"]
    addTensorDimCheck0 = addTensorDimCheck1 = 0
    for b in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      loadOffset1 = " %u*CPSV" %b
      if packGranularity==2:
        addTensorDimCheck1 = 1
        base1 = loadOffset1
        loadOffset1 = "0"
      for s in range(0, kernel["GlobalWriteVectorWidth"]):
        loadOffset0 = "%u" % (s)
        if packGranularity==2:
          addTensorDimCheck0 = 1
          base0 = loadOffset0
          loadOffset0 = "0"
        if kernel["EdgeType"] != "None":
          if addTensorDimCheck0 or addTensorDimCheck1:
            kStr += self.endLine
          if addTensorDimCheck0:
            kStr += "  /* new 0 offset - inc and extract tensor dims */%s" % (self.endLine)
            kStr += self.extractGlobalCDims(kernel, base0, 0)
            addTensorDimCheck0 = 0
          if addTensorDimCheck1:
            kStr += "  /* new 1 offset - inc and extract tensor dims */%s" % (self.endLine)
            kStr += self.extractGlobalCDims(kernel, base1, 1)
            addTensorDimCheck1 = 0

          ### Bounds checks:
          # if packed, check flattened against product of all packed sizes
          # The flattened base never changes so add all address offsets before comparison
          globalC0ForCheck = "flattenedGlobalC0"
          size0ForCheck = " * ".join(self.tPA["packedSizeList"])
          kStr += "  if (%s%s < %s) {" \
              % (globalC0ForCheck, \
              ((" + %u" %s) if kernel["GlobalWriteVectorWidth"]>1 else ""), \
              size0ForCheck)

          globalC1ForCheck = "flattenedGlobalC1"
          size1ForCheck = " * ".join(self.tPB["packedSizeList"])
          kStr += "  if (%s + %u*CPSV < %s) {" \
              % (globalC1ForCheck, b, size1ForCheck)

        kStr += "  TYPE_MAC_WRITE( D[ GLOBAL_D( (%s)" % self.uint64Str
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC%s" % self.indexChars[i]
          if i == kernel["ProblemType"]["Index0"] and kernel["GlobalWriteVectorWidth"]>1:
            kStr += " + %s" % (loadOffset0)
          if i == kernel["ProblemType"]["Index1"]:
            kStr += " + %s" % (loadOffset1)
          if i < kernel["ProblemType"]["NumIndicesC"]-1:
            kStr += ", (%s)" % self.uint64Str
        kStr += ") ]"

        if kernel["ProblemType"]["UseBeta"]:
          kStr += ", C[ GLOBAL_C( (%s)" % self.uint64Str
          for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
            kStr += " globalC%s" % self.indexChars[i]
            if i == kernel["ProblemType"]["Index0"] and kernel["GlobalWriteVectorWidth"]>1:
              kStr += " + %s" % (loadOffset0)
            if i == kernel["ProblemType"]["Index1"]:
              kStr += " + %s" % (loadOffset1)
            if i < kernel["ProblemType"]["NumIndicesC"]-1:
              kStr += ", (%s)" % self.uint64Str
          kStr += ") ]"

        kStr += ", alpha"
        kStr += ", rC[%u + %u*GLOBAL_WRITE_VECTOR_WIDTH]" % (s, b )

        if kernel["ProblemType"]["UseBeta"]:
          kStr += ", beta"
        kStr += ")"

        if kernel["EdgeType"] != "None":
          kStr += "} }"
        kStr += self.endLine
    return kStr


  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    # Add Index0 and Index1:
    index0 = kernel["ProblemType"]["Index0"]
    kStr += "  unsigned int flattenedGlobalC0 = "
    kStr += "(wg%s)*MT%s + (serial %% SG%s)*VECTOR_WIDTH;%s" \
            % (self.indexChars[index0], self.tileChar0, self.tileChar0, self.endLine)

    index1 = kernel["ProblemType"]["Index1"]
    kStr += "  unsigned int flattenedGlobalC1 = "
    kStr += "(wg%s)*MT%s + (serial / SG%s)*VECTOR_WIDTH;%s" \
            % (self.indexChars[index1], self.tileChar1, self.tileChar0, self.endLine)

    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC%s = " % self.indexChars[i]
      if i == index0 and len(kernel["PackedC0IndicesX"]) == 1:
        kStr += "flattenedGlobalC0;"
      elif i == index1 and len(kernel["PackedC1IndicesX"]) == 1:
        kStr += "flattenedGlobalC1;"
      elif isPackedIndex(kernel,i):
        kStr += "0; // will be set below"
      elif kernel["ProblemType"]["StridedBatched"] or kernel["_GlobalAccumulation"]:
        kStr += "(wg%s);" % (self.indexChars[i])
      else:
        kStr += "0;"
      kStr += "%s" % self.endLine

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      indexChar = self.indexChars[0]
      kStr += "  %s strideW = 1 + (size%s - 1) " % (self.uint64Str, indexChar)
      for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
        strideStr = "1"
        for j in range(0, i):
          strideStr += " * size%s" % (self.indexChars[j])
        indexChar = self.indexChars[i]
        kStr += " + (size%s - 1) * %s" % (indexChar, strideStr)
      kStr += ";" + self.endLine

    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    kStr = ""
    packGranularity = kernel["PackGranularity"]
    addTensorDimCheck0 = addTensorDimCheck1 = 0

    for b in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
      for a in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
        if packGranularity==2:
          addTensorDimCheck0 = 1
          base0 = " %u*SG%s*VECTOR_WIDTH" % (a,self.tileChar0)
        for s1 in range(0, kernel["VectorWidth"]):
          if packGranularity==2:
            addTensorDimCheck1 = 1
            base1 = "%u + %u*SG%s*VECTOR_WIDTH" % (s1, b,self.tileChar1)
            offsetS1 = ""
          else:
            offsetS1 = ((" + %u"%s1) if kernel["VectorWidth"]>1 else "")
          for s0 in range(0, kernel["VectorWidth"]):
            # set default offsets, may be overridden in packed mode:
            offsetS0 = ((" + %u"%s0) if kernel["VectorWidth"]>1 else "")
            offset0 = "%s + %u*SG%s*VECTOR_WIDTH" \
                      % (offsetS0, a, self.tileChar0)
            offset1 = "%s + %u*SG%s*VECTOR_WIDTH" % (\
                ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                b, self.tileChar1)

            if kernel["EdgeType"] == "Branch":
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar0, a, self.tileChar0, self.tileChar0)
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar1, b, self.tileChar1, self.tileChar1)
            elif kernel["EdgeType"] == "ShiftPtr":
              if addTensorDimCheck0 or addTensorDimCheck1:
                kStr += self.endLine
              if addTensorDimCheck0:
                kStr += "  /* new vw0 offset - inc and extract tensor dims */%s" % (self.endLine)
                kStr += self.extractGlobalCDims(kernel, base0, 0)
                addTensorDimCheck0 = 0
              if addTensorDimCheck1:
                kStr += "  /* new vw1 offset - inc and extract tensor dims */%s" % (self.endLine)
                kStr += self.extractGlobalCDims(kernel, base1, 1)
                addTensorDimCheck1 = 0

              tP0 = self.tPA if self.tPB["tile01Idx"] else self.tPB
              tP1 = self.tPB if self.tPB["tile01Idx"] else self.tPA

              ### Bounds checks:
              # if packed, check flattened against product of all packed sizes
              # The flattened base never changes so add all address offsets before comparison
              if packGranularity == 2:
                # base contains some addressing components, so just offset here:
                offset0 = offsetS0
              globalC0ForCheck = "flattenedGlobalC0"
              size0ForCheck = " * ".join(tP0["packedSizeList"])

              # Check 0 dimension against appropriate size limit
              kStr += "  if (%s%s + %u*SG%s*VECTOR_WIDTH < %s) {" \
                  % (globalC0ForCheck,
                  ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                  a, self.tileChar0, size0ForCheck)

              if packGranularity == 2:
                offset1 = offsetS1
              globalC1ForCheck = "flattenedGlobalC1"
              size1ForCheck = " * ".join(tP1["packedSizeList"])

              kStr += "  if (%s%s + %u*SG%s*VECTOR_WIDTH < %s) {" \
                  % (globalC1ForCheck,
                  ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                  b, self.tileChar1, size1ForCheck)

            # Write the result
            kStr += "  TYPE_MAC_WRITE( D[ GLOBAL_D( (%s)" % self.uint64Str
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
              kStr += " globalC%s" % self.indexChars[i]
              if i == kernel["ProblemType"]["Index0"]:
                kStr += offset0
              if i == kernel["ProblemType"]["Index1"]:
                kStr += offset1
              if i < kernel["ProblemType"]["NumIndicesC"]-1:
                kStr += ", (%s)" % self.uint64Str
            kStr += ") ]"

            if kernel["ProblemType"]["UseBeta"]:
              kStr += ", C[ GLOBAL_C( (%s)" % self.uint64Str
              for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
                kStr += " globalC%s" % self.indexChars[i]
                if i == kernel["ProblemType"]["Index0"]:
                  kStr += offset0
                if i == kernel["ProblemType"]["Index1"]:
                  kStr += offset1
                if i < kernel["ProblemType"]["NumIndicesC"]-1:
                  kStr += ", (%s)" % self.uint64Str
              kStr += ") ]"

            kStr += ", alpha"
            #kStr += ", rC[%d+%d*(TT%s/VECTOR_WIDTH)+%d*TT%s]%s" \
            #    % (a, s1, self.tileChar0, b, self.tileChar0, \
            #    ((".%s"%self.vectorComponents[s0]) if kernel["VectorWidth"]>1\
            #    else "") )
            kStr += ", rC[%u*VECTOR_WIDTH+%u + (%u*VECTOR_WIDTH+%u)*TT%s]" \
                % (a, s0, b, s1, self.tileChar0 )
            if kernel["ProblemType"]["UseBeta"]:
              kStr += ", beta"
            kStr += ")"

            if kernel["EdgeType"] != "None":
              kStr += " } }"
            kStr += self.endLine
    return kStr

  def openPrefetchAcrossPersistent(self, kernel, isOptNLL, useBufferOOB=False):
    return ""

  def closePrefetchAcrossPersistent(self, kernel, isOptNLL, useBufferOOB=False):
    return ""

  ##############################################################################
  # PrefetchGlobalRead2
  ##############################################################################
  def openPrefetchGlobalRead2(self, kernel):
    return ""

  def closePrefetchGlobalRead2(self, kernel):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel, addLabel):
    kStr = ""

    if kernel["PersistentKernel"]:
      kStr += "  serialWgIter += %s(0);%s" \
        % (self.getNumGroupsStr, self.endLine)
      kStr += "} // End Persistent Loop" + self.endLine



    kStr += self.endLine
    kStr += "}" + self.endLine

    return kStr

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    kStr = ""
    if globalParameters["MergeFiles"] and self.language == "HIP":
      kStr += "#undef UNROLL%s" % self.endLine
      kStr += "#undef LOCAL_SPLITU%s" % self.endLine
      kStr += "#undef LOCAL_DEPTHU%s" % self.endLine
      kStr += "#undef SG%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef SG%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef TT%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef TT%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef MT%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef MT%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef NLCA%s" % (self.endLine )
      kStr += "#undef NLCB%s" % (self.endLine )
      kStr += "#undef NLPA%s" % (self.endLine )
      kStr += "#undef NLPB%s" % (self.endLine )
      kStr += "#undef LSCA%s" % (self.endLine)
      kStr += "#undef LSPA%s" % (self.endLine)
      kStr += "#undef LSCB%s" % (self.endLine)
      kStr += "#undef LSPB%s" % (self.endLine)
      kStr += "#undef GLOBAL_D%s" % (self.endLine)
      kStr += "#undef GLOBAL_C%s" % (self.endLine)
      kStr += "#undef GLOBAL_OFFSET_A%s" % (self.endLine)
      kStr += "#undef GLOBAL_OFFSET_B%s" % (self.endLine)
      kStr += "#undef DATA_TYPE%s" % (self.endLine)
      kStr += "#undef DEST_DATA_TYPE%s" % (self.endLine)
      kStr += "#undef COMPUTE_DATA_TYPE%s" % (self.endLine)
      #kStr += "#undef VECTOR_TYPE%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_B%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_BLK%s" % (self.endLine)
      kStr += "#undef LDS_NUM_ELEMENTS%s" % (self.endLine)
      kStr += "#undef NUM_THREADS%s" % (self.endLine)
      kStr += "#undef PAD%s" % (self.endLine)
      kStr += "#undef WORK_GROUP_MAPPING%s" % (self.endLine)
      kStr += "#undef VECTOR_WIDTH%s" % (self.endLine)
      kStr += "#undef GLOBAL_LOAD_VECTOR_WIDTH_A%s" % (self.endLine)
      kStr += "#undef GLOBAL_LOAD_VECTOR_WIDTH_B%s" % (self.endLine)
      kStr += "#undef GLOBAL_WRITE_VECTOR_WIDTH%s" % (self.endLine)
      kStr += "#undef MAC%s" % (self.endLine)
      kStr += "#undef TYPE_MAC%s" % (self.endLine)
      kStr += "#undef TYPE_MAC_WRITE%s" % (self.endLine)
      kStr += "#undef GLOBAL_SPLITU%s" % (self.endLine)
      # zero
      kStr += "#undef SCALAR_ZERO%s" % (self.endLine )
      kStr += "#undef SCALAR_OOB_DATA%s" % (self.endLine )

      numMacs = 2 if kernel["PrefetchLocalRead"] else 1
      for m in range(0, numMacs):
        kStr += "#undef MAC_%ux%u" \
            % (kernel["ThreadTile0"], kernel["ThreadTile1"])
        if kernel["PrefetchLocalRead"]:
          kStr += ("" if m==0 else "_BLK")
        kStr += self.endLine
      # initial strides
      firstStride = 0
      if kernel["ProblemType"]["UseInitialStridesCD"]:
        lastStrideD = 0
        lastStrideC = 0
      else:
        lastStrideD = 1
        lastStrideC = 1
      if kernel["ProblemType"]["UseInitialStridesAB"]:
        lastStrideA = 0
        lastStrideB = 0
      else:
        lastStrideA = 1
        lastStrideB = 1
      for i in range(firstStride, lastStrideD):
        kStr += "#undef strideD" + self.indexChars[i] + self.endLine
      for i in range(firstStride, lastStrideC):
        kStr += "#undef strideC" + self.indexChars[i] + self.endLine
      for i in range(firstStride, lastStrideA):
        kStr += "#undef strideA" \
            + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
            + self.endLine
      for i in range(firstStride, lastStrideB):
        kStr += "#undef strideB" \
            + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
            + self.endLine
      # other summation indices
      for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
        index = i + kernel["ProblemType"]["NumIndicesC"]
        kStr += "#undef globalReadOffsetA%s%s" \
            % (self.indexChars[index], self.endLine)
        kStr += "#undef globalReadOffsetB%s%s" \
            % (self.indexChars[index], self.endLine)
      kStr += self.endLine + self.endLine
    return kStr

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    kStr = ""
    kernelName = self.getKernelFileBase(kernel)
    if not globalParameters["MergeFiles"]:
      kStr += "\n"
      kStr += "#include \"%s.h\"\n" % kernelName
      kStr += "\n"

    return kStr

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  def kernelBodySuffix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # WaitCnt
  ##############################################################################
  def wait(self, kernel, tPA, tPB, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  def getWaitcntCodeForDirectToVgpr(self, kernel, localWriteEndIter, u, firstIter, isPap=True, beforeBarrier=False, NLLlast=False, oddLast=False):
    return ""

  ##############################################################################
  # waitcnt code for PrefetchGlobalRead
  ##############################################################################
  def getWaitcntCodeForPGR(self, kernel):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel, comment=""):
    return self.indent + self.syncStr + " //" + comment + self.endLine
# TODO: can be uncommented once source kernels support WavefrontSize=32
#    if kernel["NumThreads"] > kernel["WavefrontSize"]:
#        return self.indent + self.syncStr + " //" + comment + self.endLine
#    else:
#        return self.indent + "// Skip barrier: NumThreads=%s"%(kernel["NumThreads"]) + \
#               self.endLine

  ##############################################################################
  # MapAcctoArch
  ##############################################################################
  def MapAcctoArchRegs(self, kernel, option):
    return ""

  ##############################################################################
  # openmovaccVgpr
  ##############################################################################
  def openmovaccVgpr(self, kernel, backupSgpr):
    return ""

  ##############################################################################
  # getAccVgprCode
  ##############################################################################
  def getAccVgprCode(self,kernel,odd):
    return ""

  ##############################################################################
  # closemovaccVgpr
  ##############################################################################
  def closemovaccVgpr(self, kernel, backupSgpr):
    return ""

  ##############################################################################
  # init for StoreCInUnroll
  ##############################################################################
  def initStoreCInUnroll(self, kernel):
    return ""

  ##############################################################################
  # init for StoreCInUnroll per Persistent Loop
  ##############################################################################
  def initStoreCInUnrollPerPersistentLoop(self, kernel):
    return ""

  ##############################################################################
  # init for StoreCInUnroll per Unroll Loop
  ##############################################################################
  def initStoreCInUnrollPerUnrollLoop(self, kernel, needInit):
    return ""

  ##############################################################################
  # swap SrdC and SrdCbackup, SrdD and SrdDbackup
  ##############################################################################
  def swapSrdCDandBackup(self, kernel):
    return ""

  ##############################################################################
  # C/D address increment value for StoreCInUnroll
  ##############################################################################
  def generateCorDaddrIncrementForStoreCInUnroll(self, kernel, CorD, odd, tmpSgprWork):
    return ""

  ##############################################################################
  # get address/gpr index increment frequency for StoreCInUnroll
  ##############################################################################
  def getAddrGprIdxIncrementFrequencyForStoreCInUnroll(self, kernel):
    return ""

  ##############################################################################
  # generate post process for StoreCInUnroll loop
  ##############################################################################
  def generatePostProcessForStoreCInUnrollLoop(self, kernel, needPost):
    return ""

  ##############################################################################
  # restore SrdCbackup and SrdDbackup
  ##############################################################################
  def restoreSrdCandDBackup(self, kernel):
    return ""

  ##############################################################################
  # set storeC sync objects
  ##############################################################################
  def setStoreCsyncObject(self, kernel):
    return ""

  ##############################################################################
  # reset storeC sync objects
  ##############################################################################
  def resetStoreCsyncObject(self, kernel):
    return ""

  ##############################################################################
  # end process for StoreCInUnroll per PersistentLoop (OptNLL)
  ##############################################################################
  def endProcessPersistentLoopforStoreCInUnrollOptNLL(self, kernel):
    return ""

  ##############################################################################
  # end process for StoreCInUnroll per PersistentLoop (NoOptNLL)
  ##############################################################################
  def endProcessPersistentLoopforStoreCInUnrollNoOptNLL(self, kernel):
    return ""

  ##############################################################################
  # number of storeC code in template for StoreCInUnroll
  ##############################################################################
  def getNumberOfStoreCInTemplate(self, kernel):
    return ""

  ##############################################################################
  # number of LoadC code in template for StoreCInUnroll
  ##############################################################################
  def getNumberOfLoadCInForLoadC(self, kernel):
    return ""

  ##############################################################################
  # generate storeCInUnroll post loop code
  ##############################################################################
  def generateStoreInUnrollPostLoop(self, kernel, isOptNLL, isDTVodd):
    return ""

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def openOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def closeOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  def generateEvenEndLabeNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  def generateOddEndVgprCopyForDTV(self, kernel):
    return ""

  ##############################################################################
  # isSwapGlobalReadOrderForDtvOrDtl
  ##############################################################################
  def isSwapGlobalReadOrderForDtvOrDtl(self, kernel):
    return False
