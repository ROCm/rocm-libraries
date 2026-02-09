################################################################################
#
# Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import Signature
from ..Common import globalParameters, getCOVFromParam, gfxName

from math import ceil

def getSrcValueType(kernel, isTypeA):
    # special cases for F8 datatypes
    if kernel["ProblemType"]["DataType"].isFloat8():
        srcValueType = "FP8"
    elif kernel["ProblemType"]["DataType"].isBFloat8():
        srcValueType = "BF8"
    elif kernel["ProblemType"]["DataType"].isFloat8BFloat8():
        srcValueType = "FP8" if isTypeA else "BF8"
    elif kernel["ProblemType"]["DataType"].isBFloat8Float8():
        srcValueType = "BF8" if isTypeA else "FP8"
    else:
        srcValueType = kernel["ProblemType"]["DataType"].toNameAbbrev().upper()

    srcValueType = srcValueType.lower()
    return srcValueType

def getDstValueType(kernel):
    # special cases for F8 datatypes
    if kernel["ProblemType"]["DataType"].isFloat8():
        dstValueType = "FP8"
    elif kernel["ProblemType"]["DataType"].isBFloat8():
        dstValueType = "BF8"
    else:
        dstValueType = kernel["ProblemType"]["DataType"].toNameAbbrev().upper()

    dstValueType = dstValueType.lower()
    return dstValueType


# Creates kernel header, compatible with code object version 4 and up. V2 and V3 no longer supported.
class SignatureDefault(Signature):

    # Formats an argument to add to the header
    def addArgument(self, name, size, offset, valueKind, valueType = None, AddrSpaceQual = None):
        self.offset += size
        if valueType is None:
            valueType = f"u{size*8}"
        kStr = ""
        kStr += "      - .name:            %s\n" % name
        kStr += "        .size:            %s\n" % size
        kStr += "        .offset:          %s\n" % offset
        kStr += "        .value_kind:      %s\n" % valueKind
        kStr += "        .value_type:      %s\n" % valueType
        if AddrSpaceQual != None:
            kStr += "        .address_space:   %s\n" % AddrSpaceQual
        return kStr

    def __call__(self, writer):
        kernel = writer.kernel

        kStr = self.commentHeader()

        # begin kernel descriptor
        kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--%s\"%s" \
            % (gfxName(writer.version), writer.endLine)

        kStr += ".text%s" % writer.endLine
        kStr += ".protected %s%s" % (writer.kernelName, writer.endLine)
        kStr += ".globl %s%s" % (writer.kernelName, writer.endLine)
        kStr += ".p2align 8%s" % writer.endLine
        kStr += ".type %s,@function%s" % (writer.kernelName, writer.endLine)

        kStr += ".section .rodata,#alloc%s" % writer.endLine
        kStr += ".p2align 6%s" % writer.endLine
        tWord = "amdhsa_kernel"
        kStr += ".%s %s%s" % (tWord, writer.kernelName, writer.endLine)
        tWord = ".amdhsa_user_sgpr_kernarg_segment_ptr"
        kStr += "  %s 1%s" % (tWord, writer.endLine)

        supportsPreload = globalParameters["AsmCaps"][writer.version]["KernargPreloading"]

        numPreloaded = 0
        numUserSgprs = 2

        if kernel["PreloadKernelArguments"] == 1:
            numPreloaded = writer.sgprPool.numPreloadSGPRs
            numUserSgprs += numPreloaded

        if supportsPreload:
            kStr += f"  .amdhsa_user_sgpr_kernarg_preload_offset 0{writer.endLine}"
            kStr += f"  .amdhsa_user_sgpr_kernarg_preload_length {numPreloaded}{writer.endLine}"

        kStr += f"  .amdhsa_user_sgpr_count {numUserSgprs}{writer.endLine}"

        # kern arg size
        kernArgReg = 0
        kernArgReg += 3*writer.rpga
        kernArgReg += max(1,int(writer.bpeAB/4)) # alpha
        if kernel["ProblemType"]["UseBeta"]:
            kernArgReg += max(1,int(writer.bpeCexternal/4)) # beta
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
        kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
        if not kernel["ProblemType"]["UseInitialStridesAB"]:
            kernArgReg -= 2 # strides
        if not kernel["ProblemType"]["UseInitialStridesCD"]:
            kernArgReg -= 2 # strides
        kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
        kernArgReg += kernel["ProblemType"]["NumIndicesC"]
        if globalParameters["DebugKernel"]:
            kernArgReg += writer.rpga # debug buffer
        # kernArgBytes = kernArgReg * 4 # bytes/reg

        # register allocation
        totalVgprs = writer.vgprPool.size()
        totalSgprs = writer.sgprPool.size()

        # accumulator offset for Unified Register Files
        vgprCount = totalVgprs
        if writer.archCaps["ArchAccUnifiedRegs"]:
            agprStart = ceil(totalVgprs/8)*8
            vgprCount = agprStart + writer.agprPool.size()

            tWord = ".amdhsa_accum_offset"
            kStr += "  %s %u // accvgpr offset%s" % (tWord, agprStart, writer.endLine)

        tWord = ".amdhsa_next_free_vgpr"
        kStr += "  %s %u // vgprs%s" % (tWord, vgprCount, writer.endLine)
        tWord = ".amdhsa_next_free_sgpr"
        kStr += "  %s %u // sgprs%s" % (tWord, totalSgprs, writer.endLine)

        tWord = ".amdhsa_group_segment_fixed_size"
        if kernel["AggressivePerfMode"]>=2 and kernel["ProblemType"]["DataType"].isDouble() and \
            kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 4 and kernel["WorkGroup"] == [16,16,1]:
            group_segment_size = 32768 # Pad LDS to ensure we run exactly two waves
        else:
            group_segment_size = kernel["LdsNumElements"] * writer.bpeAB
        kStr += "  %s %u // lds bytes%s" % ( tWord, group_segment_size, writer.endLine )

        if writer.archCaps["HasWave32"]:
            if kernel["WavefrontSize"] == 32:
                kStr += "  .amdhsa_wavefront_size32 1 // 32-thread wavefronts%s" % writer.endLine
            else:
                kStr += "  .amdhsa_wavefront_size32 0 // 64-thread wavefronts%s" % writer.endLine

        # other
        kStr += "  .amdhsa_private_segment_fixed_size 0%s" % writer.endLine
        kStr += "  .amdhsa_system_sgpr_workgroup_id_x 1%s" % writer.endLine
        kStr += "  .amdhsa_system_sgpr_workgroup_id_y 1%s" % writer.endLine
        kStr += "  .amdhsa_system_sgpr_workgroup_id_z %u%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, writer.endLine)
        kStr += "  .amdhsa_system_vgpr_workitem_id 0%s" % writer.endLine
        kStr += "  .amdhsa_float_denorm_mode_32 3%s" % writer.endLine
        kStr += "  .amdhsa_float_denorm_mode_16_64 3%s" % writer.endLine
        kStr += ".end_amdhsa_kernel%s" % writer.endLine
        kStr += ".text%s" % writer.endLine

        kStr += writer.comment3("Optimizations and Config:")
        kStr += writer.comment1("ThreadTile= %u x %u" % (kernel["ThreadTile0"], kernel["ThreadTile1"]))
        kStr += writer.comment1("SubGroup= %u x %u" % (kernel["SubGroup0"], kernel["SubGroup1"]))
        if kernel["EnableMatrixInstruction"]:
          kStr += writer.comment1("VectorWidthA=%u" % (kernel["VectorWidthA"]))
          kStr += writer.comment1("VectorWidthB=%u" % (kernel["VectorWidthB"]))
        else:
          kStr += writer.comment1("VectorWidth=%u" % (kernel["VectorWidth"]))
        glvw = kernel["GlobalLoadVectorWidthA"]
        glvwAStr = "GlobalLoadVectorWidthA=%f"%glvw if glvw < 1 else "GlobalLoadVectorWidthA=%u"%glvw
        glvw = kernel["GlobalLoadVectorWidthB"]
        glvwBStr = "GlobalLoadVectorWidthB=%f"%glvw if glvw < 1 else "GlobalLoadVectorWidthB=%u"%glvw
        kStr += writer.comment1("%s, %s" % (glvwAStr, glvwBStr))
        kStr += writer.comment1("DirectToLdsA=%s" % kernel["DirectToLdsA"])
        kStr += writer.comment1("DirectToLdsB=%s" % kernel["DirectToLdsB"])
        kStr += writer.comment1("UseSgprForGRO=%s" % kernel["_UseSgprForGRO"])

        kStr += ".amdgpu_metadata\n"
        kStr += "---\n"
        kStr += "amdhsa.version:\n"
        kStr += "  - 1\n"
        cov = getCOVFromParam(kernel["CodeObjectVersion"])
        if cov == 4:
            kStr += "  - 1\n"
        elif cov == 5:
            kStr += "  - 2\n"
        kStr += "amdhsa.target: amdgcn-amd-amdhsa--%s%s" \
            % (gfxName(writer.version), writer.endLine)
        kStr += "amdhsa.kernels:\n"
        kStr += "  - .name: %s%s" % (writer.kernelName, writer.endLine)
        kStr += "    .symbol: '%s.kd'%s" % (writer.kernelName, writer.endLine)
        kStr += "    .language:                   %s%s" % ("OpenCL C", writer.endLine)
        kStr += "    .language_version:%s" % writer.endLine
        kStr += "      - 2%s" % writer.endLine
        kStr += "      - 0%s" % writer.endLine
        kStr += "    .args:%s" % writer.endLine
        self.offset = 0

        for arg in writer.sgprPool.kernargs:
            kStr += self.addArgument(name=arg['name'], size=arg['size']*4, offset=self.offset, valueKind="by_value")

        kStr += "    .group_segment_fixed_size:   %u%s" % ( group_segment_size, writer.endLine ) #XXXXXX
        kStr += "    .kernarg_segment_align:      %u%s" % ( 8, writer.endLine )
        kStr += "    .kernarg_segment_size:       %u%s" % (((self.offset+7)//8)*8, writer.endLine) # round up to .kernarg_segment_align
        kStr += "    .max_flat_workgroup_size:    %u%s" % ( kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"], writer.endLine )
        kStr += "    .private_segment_fixed_size: %u%s" % ( 0, writer.endLine )
        kStr += "    .sgpr_count:                 %u%s" % ( totalSgprs, writer.endLine )
        kStr += "    .sgpr_spill_count:           %u%s" % ( 0, writer.endLine )
        kStr += "    .vgpr_count:                 %u%s" % ( totalVgprs, writer.endLine )
        kStr += "    .vgpr_spill_count:           %u%s" % ( 0, writer.endLine )
        kStr += "    .wavefront_size:             %u%s" % ( kernel["WavefrontSize"], writer.endLine )

        kStr += "...\n"

        kStr += ".end_amdgpu_metadata\n"

        kStr += "%s:%s" % (writer.kernelName, writer.endLine)

        return kStr
