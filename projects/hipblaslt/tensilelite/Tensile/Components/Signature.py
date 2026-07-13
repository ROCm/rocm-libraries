################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import SignatureBase
from rocisa.enum import SignatureValueKind as SVK
from ..Component import Signature
from ..Common import DataDirection
from ..Activation import ActivationType

from dataclasses import dataclass, field

# Fused GEMM.A2A kernarg segment layout (Task 5).
#
# When kernel["FusedGemmA2A"] is set, the Signature appends a fixed-size segment
# of kernarg metadata at the very end of the kernarg buffer (after every GEMM /
# store arg). These args are registered as kernarg metadata ONLY -- they are
# deliberately NOT loaded in the prologue (no defineSgpr, not counted in
# numSgprToLoad). The A2A fusion logic runs solely in the D-store epilogue
# (Task 6-9), so the epilogue reads each arg on demand via
# loadKernArg(..., sgprOffset=hex(fused_base + intra-offset), dword=...) into a
# scratch SGPR that is freed immediately after use. A WG maps to a single
# dst_rank, so it reads exactly one recv_ptr + one flag_ptr (via a switch on
# dst_rank), never the whole array.
#
# The array slot count is a COMPILE-TIME constant (8), independent of the runtime
# world size W; unused slots cost nothing here because nothing enters SGPR.
FUSED_A2A_MAX_RANKS = 8

# Intra-segment byte layout, in emission order. Pointers are 8 bytes
# (SIG_GLOBALBUFFER), scalars are 4 bytes (u32). Task 6-9 add fused_base (the
# byte offset of recv_ptr_0 in kernarg memory, exposed as
# writer.states.fusedA2AKernArgBase) to these to get an absolute sgprOffset.
def fusedA2AKernArgLayout():
    """Return {argName: intra-segment byte offset} for the fused-A2A segment.

    The offsets are relative to the segment base (recv_ptr_0 == 0). Order and
    sizes MUST match the addArg() sequence in SignatureDefault.__call__:
      recv_ptr_0..7  : 8B each  (remote recv base pointers, incl. self)
      flag_ptr_0..7  : 8B each  (remote flag base pointers)
      counter_ptr    : 8B       (this device's counter base)
      FusedMyRank    : 4B (u32)
      FusedTarget    : 4B (u32)  == M_tiles * tiles_per_rank
      FusedW         : 4B (u32)  world size
      FusedNShard    : 4B (u32)
      FusedDrain     : 4B (u32)  runtime drain flag (NOT a compile-time gate)
      FusedAN        : 4B (u32)  A2A column count (first AN cols PUSH, rest local)

    FusedAN is appended at the END of the scalar list (after FusedDrain) so the
    offsets of every preceding arg are unchanged from earlier tasks.
    """
    layout = {}
    off = 0
    for j in range(FUSED_A2A_MAX_RANKS):
        layout["recv_ptr_%u" % j] = off
        off += 8
    for j in range(FUSED_A2A_MAX_RANKS):
        layout["flag_ptr_%u" % j] = off
        off += 8
    layout["counter_ptr"] = off
    off += 8
    for name in ("FusedMyRank", "FusedTarget", "FusedW", "FusedNShard", "FusedDrain", "FusedAN"):
        layout[name] = off
        off += 4
    return layout

# Total bytes of the fused-A2A kernarg segment.
FUSED_A2A_SEGMENT_BYTES = (2 * FUSED_A2A_MAX_RANKS + 1) * 8 + 6 * 4

def _currentKernArgOffset(signature) -> int:
    """Byte offset the NEXT addArg() would receive (== accumulated kernarg size).

    SignatureCodeMeta accumulates a running byte offset per arg but does not
    expose it to Python. We recover it from the already-emitted metadata: each
    arg prints ``.size:`` and ``.offset:`` lines, so the next offset is the last
    arg's offset + size. This keeps the fused-segment base byte-identical to
    what Signature actually emits, with no rocisa C++ change.
    """
    text = str(signature)
    lastSize = None
    lastOffset = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(".size:"):
            lastSize = int(s.split(":", 1)[1].strip())
        elif s.startswith(".offset:"):
            lastOffset = int(s.split(":", 1)[1].strip())
    if lastOffset is None or lastSize is None:
        return 0
    return lastOffset + lastSize

@dataclass
class UserArgumentsInfo:
    # Common args
    commonArgsNum: int  = 0
    commonArgsSize: int = 0
    # variable related fixed parameters
    alphaMaxSize: int = 16
    alphaMaxRegisterSize: int = field(init=False)
    betaMaxSize: int = 16
    betaMaxRegisterSize: int = field(init=False)
    scaleASize: int = 0
    scaleBSize: int = 0
    scaleCSize: int = 0
    scaleDSize: int = 0
    actMaxSize: int = 4
    actMaxRegisterSize: int = field(init=False)
    # gemm related
    gemmArgumentSize: int = 0
    # Epilogue related
    scaleAlphaVecSize: int = 0
    biasSize: int = 0
    eSize: int = 0
    gateSize: int = 0
    activationSize: int = 0
    factorDimSize: int = 0
    # Total argument size
    totalSize: int = 0

    def __post_init__(self):
        self.alphaMaxRegisterSize = self.alphaMaxSize // 4
        self.betaMaxRegisterSize  = self.betaMaxSize // 4
        self.actMaxRegisterSize   = self.actMaxSize // 4

def getSrcValueType(kernel, isTypeA):
    # special cases for F8 datatypes
    tc='A' if isTypeA else 'B'
    if kernel["ProblemType"]["MacDataType%s"%tc].isAnyFloat8():
        srcValueType = "FP8"
    elif kernel["ProblemType"]["MacDataType%s"%tc].isAnyBFloat8():
        srcValueType = "BF8"
    else:
        srcValueType = kernel["ProblemType"]["DataType%s"%tc].toNameAbbrev().upper()

    srcValueType = srcValueType.lower()
    return srcValueType


# Creates kernel header, compatible with code object version 4 and up. V2 and V3 no longer supported.
class SignatureDefault(Signature):

    def __call__(self, writer) -> SignatureBase:
        kernel = writer.states.kernel

        userArgumentsInfo = UserArgumentsInfo()

        # kern arg size
        kernArgReg = 0
        kernArgReg += 3*writer.states.rpga
        # TODO: Check correctness of the following
        kernArgReg += max(1,int(writer.states.bpeA/4)) # alpha
        # TODO: alpha and beta should be computeType
        if kernel["ProblemType"]["UseBeta"]:
            kernArgReg += max(1,int(writer.states.bpeCexternal/4)) # beta
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
        if writer.debugConfig.debugKernel:
            kernArgReg += writer.states.rpga # debug buffer
        # kernArgBytes = kernArgReg * 4 # bytes/reg

        group_segment_size = kernel["LdsNumBytes"]

        # When modify the size, please also update TENSILE_COMMON_KERNEL_ARGS_SIZE in ContractionSolution.hpp
        userArgumentsInfo.commonArgsNum += 4
        userArgumentsInfo.commonArgsSize = userArgumentsInfo.commonArgsNum * writer.states.bpr

        sgprWgZ = 1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0
        numSgprToLoad = writer.states.numSgprToLoad + userArgumentsInfo.commonArgsNum
        writer.states.numSgprPreload = min(numSgprToLoad, writer.states.numSgprPreload)
        signature = SignatureBase(kernelName=writer.states.kernelName,
                                    kernArgsVersion=kernel["InternalSupportParams"]["KernArgsVersion"],
                                    codeObjectVersion=kernel["CodeObjectVersion"],
                                    groupSegmentSize=group_segment_size,
                                    sgprWorkGroup=(1, 1, sgprWgZ),
                                    vgprWorkItem=0,
                                    flatWorkGroupSize=(kernel["NumThreads"]),
                                    numSgprPreload=writer.states.numSgprPreload)

       # General Argument info
        signature.addArg(   "Gemm info", SVK.SIG_VALUE, "u32")
        signature.addArg("kernel info0", SVK.SIG_VALUE, "u32")
        signature.addArg("kernel info1", SVK.SIG_VALUE, "u32")
        signature.addArg("numWG",        SVK.SIG_VALUE, "u32")

        srcValueTypeA = getSrcValueType(kernel, True)
        srcValueTypeB = getSrcValueType(kernel, False)
        dstValueType  = kernel["ProblemType"]["DestDataType"].toNameAbbrev()
        cptValueType  = kernel["ProblemType"]["ComputeDataType"].toNameAbbrev()
        biasValueType = "void"
        actValueType  = kernel["ProblemType"]["ActivationComputeDataType"].toNameAbbrev()

        for i in range(0, writer.states.numSgprSizesFree):
            signature.addArg(            "SizesFree%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.numSgprSizesSum):
            signature.addArg(             "SizesSum%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        if writer.debugConfig.debugKernel:
            signature.addArg("AddressDbg", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("D", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
        signature.addArg("C", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
        signature.addArg("A", SVK.SIG_GLOBALBUFFER, srcValueTypeA, "generic")
        if kernel["ProblemType"]["MXBlockA"]:
            signature.addArg("MXSA", SVK.SIG_GLOBALBUFFER, "void", "generic")
        signature.addArg("B", SVK.SIG_GLOBALBUFFER, srcValueTypeB, "generic")
        if kernel["ProblemType"]["MXBlockB"]:
            signature.addArg("MXSB", SVK.SIG_GLOBALBUFFER, "void", "generic")
        userArgumentsInfo.gemmArgumentSize += (8 + 8 + 8 + 8)  # A, B, C, D buffer
        if kernel["ProblemType"]["MXBlockA"]:
            userArgumentsInfo.gemmArgumentSize += 8
        if kernel["ProblemType"]["MXBlockB"]:
            userArgumentsInfo.gemmArgumentSize += 8
        if kernel["ProblemType"]["Sparse"]:
            signature.addArg("MetaData", SVK.SIG_GLOBALBUFFER, "void" , "generic")

        # StreamKForceDPOnly (SK3 DP-first, gfx1250) never touches the workspace
        # partials/fixup path, so AddressWS/AddressFlags are dead: they are dropped
        # from the SGPR define (KernelWriter.py) and here from the .kd metadata. The
        # host (ContractionSolution.cpp singleCallArgs) matches by not appending
        # ws/Flags under streamKForceDPOnly, so the positional kernarg layout stays
        # consistent host<->device.
        if kernel["StreamK"] > 0 and kernel["StreamKAtomic"] == 0 and not kernel["StreamKForceDPOnly"]:
            signature.addArg("AddressWS", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg("AddressFlags", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")

        for i in range(0, writer.states.d.numSgprStrides):
            signature.addArg(              "strideD%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.c.numSgprStrides):
            signature.addArg(              "strideC%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.a.numSgprStrides):
            signature.addArg(              "strideA%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        if kernel["ProblemType"]["MXBlockA"]:
            for i in range(0, writer.states.mxsa.numSgprStrides):
                signature.addArg(          "strideMXSA%u"%i, SVK.SIG_VALUE,            "u32")
                userArgumentsInfo.gemmArgumentSize += 4

        for i in range(0, writer.states.b.numSgprStrides):
            signature.addArg(              "strideB%u"%i, SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        if kernel["ProblemType"]["MXBlockB"]:
            for i in range(0, writer.states.mxsb.numSgprStrides):
                signature.addArg(          "strideMXSB%u"%i, SVK.SIG_VALUE,            "u32")
                userArgumentsInfo.gemmArgumentSize += 4

        if kernel["ProblemType"]["Sparse"]:
            for i in range(0, writer.states.m.numSgprStrides):
                signature.addArg(   "strideMetadata%u"%i, SVK.SIG_VALUE,               "u32")

        for idxChar in kernel["PackedC0IdxChars"][:-1]:
            signature.addArg("MagicNumberSize%s"%idxChar, SVK.SIG_VALUE,               "u32")
            signature.addArg( "MagicShiftSize%s"%idxChar, SVK.SIG_VALUE,               "u32")

        # Note: We use packed f16 if alpha and beta are f16
        pack_cptValueType = 'pkf16' if kernel["ProblemType"]["ComputeDataType"].isHalf() else cptValueType
        signature.addArg(   "alpha",        SVK.SIG_VALUE, pack_cptValueType)
        if kernel["ProblemType"]["UseBeta"]:
            signature.addArg("beta",        SVK.SIG_VALUE, pack_cptValueType)
        # These are fixed sizes
        userArgumentsInfo.gemmArgumentSize += userArgumentsInfo.alphaMaxSize
        userArgumentsInfo.gemmArgumentSize += userArgumentsInfo.betaMaxSize

        if kernel["ExpertSchedulingMode"] > 0 and kernel["ESMRuntimeGate"]:
            signature.addArg( "ESMRuntimeSupported", SVK.SIG_VALUE,               "u32")
            userArgumentsInfo.gemmArgumentSize += 4

        if kernel["StreamK"] == 4:
            signature.addArg("ItersPerTile",                       SVK.SIG_VALUE, "u32")
            signature.addArg("TotalItems",                         SVK.SIG_VALUE, "u32")
            signature.addArg("SKTiles",                            SVK.SIG_VALUE, "u32")
            signature.addArg("SKSplit",                            SVK.SIG_VALUE, "u32")
            signature.addArg("SKItersPerWI",                       SVK.SIG_VALUE, "u32")
            signature.addArg("SKGrid",                             SVK.SIG_VALUE, "u32")
            userArgumentsInfo.gemmArgumentSize += 24
        elif kernel["StreamK"] == 5:
            # Hybrid SK3+SK4. The host pushes only the 6 args matching the
            # mode it selected for this launch; the SK4 reader names
            # (TotalItems, SKTiles, SKSplit, SKItersPerWI, SKGrid) are emitted
            # as RegSet aliases (see the SK5 block in KernelWriterAssembly.py)
            # onto the same physical SGPRs as the SK3 primary names
            # (MagicNumberItersPerTile, MagicShiftItersPerTile, SKItersPerWG,
            # skGrid, skTiles) respectively.
            #
            # The mode bit (bit 30 of slot 2) selects the active path. The
            # signature metadata uses SK3 names as the primary kernarg labels
            # because they are what defineSgpr() declares; SK4 names exist
            # only as register aliases.
            signature.addArg("ItersPerTile",                       SVK.SIG_VALUE, "u32")
            signature.addArg("MagicNumberItersPerTile",            SVK.SIG_VALUE, "u32")
            signature.addArg("MagicShiftItersPerTile",             SVK.SIG_VALUE, "u32")
            signature.addArg("SKItersPerWG",                       SVK.SIG_VALUE, "u32")
            signature.addArg("skGrid",                             SVK.SIG_VALUE, "u32")
            signature.addArg("skTiles",                            SVK.SIG_VALUE, "u32")
            userArgumentsInfo.gemmArgumentSize += 24
        elif kernel["StreamK"] == 3:  # SK3 two-tile ABI
            # StreamK args
            signature.addArg("ItersPerTile",                       SVK.SIG_VALUE, "u32")
            signature.addArg("MagicNumberItersPerTile",            SVK.SIG_VALUE, "u32")
            signature.addArg("MagicShiftItersPerTile",             SVK.SIG_VALUE, "u32")
            signature.addArg("SKItersPerWG",                       SVK.SIG_VALUE, "u32")
            userArgumentsInfo.gemmArgumentSize += 16
            signature.addArg("skGrid",                             SVK.SIG_VALUE, "u32")
            signature.addArg("skTiles",                            SVK.SIG_VALUE, "u32")
            userArgumentsInfo.gemmArgumentSize += 8

        if kernel["ProblemType"]["UseScaleAB"]:
            signature.addArg("AddressScaleA", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg("AddressScaleB", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
        userArgumentsInfo.scaleASize += 8
        userArgumentsInfo.scaleBSize += 8
        if kernel["ProblemType"]["UseScaleCD"]:
            signature.addArg("AddressScaleC", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg("AddressScaleD", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
        userArgumentsInfo.scaleCSize += 8
        userArgumentsInfo.scaleDSize += 8

        if kernel["ProblemType"]["UseScaleAlphaVec"]:
            signature.addArg("AddressScaleAlphaVec", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            if kernel["ProblemType"]["UseScaleAlphaVec"] == 3:
                userArgumentsInfo.factorDimSize =4

        userArgumentsInfo.scaleAlphaVecSize += 8

        if writer.states.useBias != DataDirection.NONE:
            signature.addArg("bias", SVK.SIG_GLOBALBUFFER, biasValueType, "generic")  # Note: We append the data in ws_d
            if writer.states.needBiasType:
                signature.addArg("biasType",        SVK.SIG_VALUE,        "u32")
                signature.addArg("StrideBias",      SVK.SIG_VALUE,        "u32")
                if kernel["ProblemType"]["UseBias"] == 3:
                    userArgumentsInfo.factorDimSize = 4
        userArgumentsInfo.biasSize += (8 + 4 + 4)

        if writer.states.useGateResidual:
            signature.addArg("gate",     SVK.SIG_GLOBALBUFFER, srcValueTypeB, "generic")
            signature.addArg("gateType", SVK.SIG_VALUE,        "u32")
            for i in range(0, writer.states.gate.numSgprStrides):
                signature.addArg("strideG%u"%i, SVK.SIG_VALUE, "u32")
        # Gate is not part of the grouped-gemm UserArgs struct (totalSize); it is
        # delivered via the normal kernarg above, so gate adds nothing to totalSize.

        if userArgumentsInfo.factorDimSize == 4:
            signature.addArg("factorDim", SVK.SIG_VALUE, "u32")

        if kernel["ProblemType"]["UseE"]:
            signature.addArg(      "E", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            for i in range(0, writer.states.e.numSgprStrides):
                signature.addArg("StrideE%u"%i,        SVK.SIG_VALUE,        "u32")
        userArgumentsInfo.eSize += 8
        for i in range(0, writer.states.e.numSgprStrides):
            userArgumentsInfo.eSize += 4

        if ((kernel["ProblemType"]["ActivationType"] != 'none') and kernel["ActivationFused"]):
            if kernel["ProblemType"]["ActivationComputeDataType"].isHalf():
                actValueType = 'pkf16'
            for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
                signature.addArg(                   name, SVK.SIG_VALUE,        actValueType)
            if kernel["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all'] :
                signature.addArg(       "activationType", SVK.SIG_VALUE,               "u32")

        # TODO- combine one workspace
        if (kernel["ProblemType"]["OutputAmaxD"]):
            signature.addArg(    "AddrAmaxOut", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg(    "AmaxWS",      SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg(    "AmaxSync",    SVK.SIG_GLOBALBUFFER, "u32",        "generic")

        if (kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel" or kernel["AdaptiveGemmGSUA"] == 1):
            signature.addArg(    "dstD", SVK.SIG_GLOBALBUFFER, dstValueType, "generic")
            signature.addArg(               "Synchronizer", SVK.SIG_GLOBALBUFFER, cptValueType, "generic")
            signature.addArg(               "GSUSync", SVK.SIG_VALUE,              "u32")

        # Batch offset support for general batched mode (pointer array).
        # Placed at the tail of the kernarg buffer (after the dstD/Synchronizer block)
        # so no later arg is shifted; the host appends them in the same position,
        # after the dstD/Synchronizer/seed block. Record each arg's kernarg byte
        # offset so the assembly loads them from the accurate position rather
        # than re-deriving it.
        #
        # signature.offset counts from the very first arg including the common header.
        # The assembly loads these args with KernArgAddress already advanced past
        # that header by commonArgsSize, so subtract it.
        if not kernel["ProblemType"]["GroupedGemm"]:
            commonArgsSize = userArgumentsInfo.commonArgsSize
            writer.states.batchOffsetDKernArgOffset = signature.offset - commonArgsSize
            signature.addArg("batchOffsetD", SVK.SIG_VALUE, "u64")
            writer.states.batchOffsetCKernArgOffset = signature.offset - commonArgsSize
            signature.addArg("batchOffsetC", SVK.SIG_VALUE, "u64")
            writer.states.batchOffsetAKernArgOffset = signature.offset - commonArgsSize
            signature.addArg("batchOffsetA", SVK.SIG_VALUE, "u64")
            writer.states.batchOffsetBKernArgOffset = signature.offset - commonArgsSize
            signature.addArg("batchOffsetB", SVK.SIG_VALUE, "u64")
            userArgumentsInfo.gemmArgumentSize += 32  # 4 offsets * 8 bytes each

        # Fused GEMM.A2A kernarg metadata (Task 5). Registered LAST so the fused
        # args occupy the tail of the kernarg buffer. These are metadata-only:
        # no defineSgpr / no numSgprToLoad change -- the epilogue (Task 6-9)
        # reads each on demand by absolute byte offset. See the module-level
        # fusedA2AKernArgLayout() docstring for the offset contract.
        if kernel["FusedGemmA2A"]:
            # Byte offset of the first fused arg (recv_ptr_0) in kernarg memory,
            # i.e. the accumulated size of every preceding kernarg. Recover it
            # from the metadata already emitted so it matches Signature exactly.
            fusedBase = _currentKernArgOffset(signature)
            for j in range(FUSED_A2A_MAX_RANKS):
                signature.addArg("recv_ptr_%u" % j, SVK.SIG_GLOBALBUFFER, "void", "generic")
            for j in range(FUSED_A2A_MAX_RANKS):
                signature.addArg("flag_ptr_%u" % j, SVK.SIG_GLOBALBUFFER, "void", "generic")
            signature.addArg("counter_ptr", SVK.SIG_GLOBALBUFFER, "void", "generic")
            signature.addArg("FusedMyRank", SVK.SIG_VALUE, "u32")
            signature.addArg("FusedTarget", SVK.SIG_VALUE, "u32")
            signature.addArg("FusedW",      SVK.SIG_VALUE, "u32")
            signature.addArg("FusedNShard", SVK.SIG_VALUE, "u32")
            signature.addArg("FusedDrain",  SVK.SIG_VALUE, "u32")
            signature.addArg("FusedAN",     SVK.SIG_VALUE, "u32")
            # Publish the segment base for the epilogue (Task 6-9). The epilogue
            # dereferences fused args against sgprKernArgAddress, which the
            # prologue has already advanced past the common-args header by
            # commonArgsSize (Bypass_ArgType3_to_ArgType0 "Shift common args" in
            # KernelWriterAssembly.py; argType 0/3 single-GEMM path, the only
            # path fused stage-1 takes). Normal GEMM arg loads reset to that
            # shifted base; the fused loads use metadata offsets that INCLUDE the
            # header, so subtract commonArgsSize once here to rebase them onto the
            # same shifted address. Absolute offset of arg X (relative to the
            # shifted base) = fusedA2AKernArgBase + fusedA2AKernArgLayout()[X].
            writer.states.fusedA2AKernArgBase = fusedBase - userArgumentsInfo.commonArgsSize

        activationType = ActivationType("all")
        for name in activationType.getAdditionalArgStringList():
            userArgumentsInfo.activationSize += userArgumentsInfo.actMaxSize
        userArgumentsInfo.activationSize += 4  # Type size

        # Calculate total size
        userArgumentsInfo.totalSize = userArgumentsInfo.gemmArgumentSize + \
                                      userArgumentsInfo.scaleASize + \
                                      userArgumentsInfo.scaleBSize + \
                                      userArgumentsInfo.scaleCSize + \
                                      userArgumentsInfo.scaleDSize + \
                                      userArgumentsInfo.scaleAlphaVecSize + \
                                      userArgumentsInfo.biasSize + \
                                      userArgumentsInfo.factorDimSize + \
                                      userArgumentsInfo.eSize + \
                                      userArgumentsInfo.activationSize + \
                                      userArgumentsInfo.gateSize

        writer.states.userArgsInfo = userArgumentsInfo

        self.addOptConfigComment(signature,
                                tt=[kernel["ThreadTile0"], kernel["ThreadTile1"]],
                                sg=[kernel["SubGroup0"], kernel["SubGroup1"]],
                                vwA=kernel["VectorWidthA"],
                                vwB=kernel["VectorWidthB"],
                                glvwA=kernel["GlobalReadVectorWidthA"],
                                glvwB=kernel["GlobalReadVectorWidthB"],
                                d2lA=kernel["DirectToLdsA"],
                                d2lB=kernel["DirectToLdsB"],
                                useSgprForGRO=kernel["_UseSgprForGRO"])

        return signature

    def addOptConfigComment(self, signature: SignatureBase, tt, sg, vwA, vwB, glvwA, glvwB, d2lA, d2lB, useSgprForGRO):
        signature.addDescriptionTopic("Optimizations and Config:")
        signature.addDescriptionBlock("ThreadTile= %u x %u" % (tt[0], tt[1]) )
        signature.addDescriptionBlock("SubGroup= %u x %u" % (sg[0], sg[1]) )
        signature.addDescriptionBlock("VectorWidthA=%u" % vwA )
        signature.addDescriptionBlock("VectorWidthB=%u" % vwB )
        signature.addDescriptionBlock("GlobalReadVectorWidthA=%u, GlobalReadVectorWidthB=%u" % (glvwA, glvwB) )
        signature.addDescriptionBlock("DirectToLdsA=%s" % d2lA )
        signature.addDescriptionBlock("DirectToLdsB=%s" % d2lB )
        signature.addDescriptionBlock("UseSgprForGRO=%s" % ("True" if useSgprForGRO else "False") )
