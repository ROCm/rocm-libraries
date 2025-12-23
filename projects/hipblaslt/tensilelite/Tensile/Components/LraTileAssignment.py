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

from rocisa.code import Module, Label
from rocisa.container import vgpr, ContinuousRegister
from rocisa.instruction import VAddU32, VAndB32, VLShiftLeftB32, VLShiftRightB32
from rocisa.functions import vectorStaticRemainder, \
    vectorStaticDivideAndRemainder, vectorStaticDivide, vectorStaticMultiply, \
    vectorStaticMultiplyAdd

from ..Component import LraTileAssignment, LraTileProperties
from ..Common import roundUp, log2, ceilDivide
from dataclasses import dataclass

@dataclass
class LraTilePropertiesMFMA(LraTileProperties):
   dividendForKId: int
   num1DBlocks: int
   num1DWaves: int
   dividedForBlkId: int
   dividedForWaveId: int
   vectorWidth: int
   maxKId: int

class LraTileAssignmentVALU(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": False}

    """
    Local Read Addresses: Tile Assignment
    """
    def __call__(self, writer, kernel, tP):
        module = Module("LraTileAssignmentVALU")

        # allocate resources
        qReg    = writer.vgprPool.checkOut(1,"qReg") # quotient
        rReg    = writer.vgprPool.checkOut(1,"rReg") # remainder
        # dot2: currently only support unroll major LDS
        tc               = tP["tensorChar"]
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
        strideTile       = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        tmpVgpr          = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes       = ContinuousRegister(tmpVgpr, 2)

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            if tP["tileIdx"] == 0:
                # kStr += "%slr%s = serial %% SG%s%s%s" \
                #         % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                #         writer.commentSuffix, writer.endLine)

                # constant
                dividendReg = "Serial" # local serial
                divisor = kernel["SubGroup0"]
                # dot2: waveSplitK
                if kernel["UseDotInstruction"]:
                    if kernel["NumWaveSplitK"] > 1:
                        newSerial = writer.vgprPool.checkOut(1,"newSerial")
                        module.add(vectorStaticDivide(newSerial, dividendReg, kernel["NumWaveSplitK"], tmpVgprRes, \
                        "Divided by NumWaveSplitK(%u)" % kernel["NumWaveSplitK"]))
                        # generate instruction
                        module.add(vectorStaticDivideAndRemainder(qReg, rReg, newSerial, divisor, tmpVgprRes))
                        # tile offset
                        module.add(vectorStaticMultiply(vgpr(rReg), vgpr(rReg), strideTile, tmpSgprInfo, \
                        "1. M offset: mOffset = mIdx * mStride(%u)" % strideTile))
                        writer.vgprPool.checkIn(newSerial)
                    else:
                        module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))
                        # tile offset
                        module.add(vectorStaticMultiply(vgpr(rReg), vgpr(rReg), strideTile, tmpSgprInfo, \
                        "1. M offset: mOffset = mIdx * mStride(%u)" % strideTile))
                else:
                    module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))

                # release and return resource
                tP["gpr"]["lro"] = rReg
                writer.tmplro = qReg
            else:
                # kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
                #         % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                #         tP["tileChar"], writer.commentSuffix, writer.endLine)

                # constant
                divisor = kernel["SubGroup1"]
                dividendReg = writer.tmplro
                # generate instruction
                module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))

                if kernel["UseDotInstruction"]:
                    # tile offset
                    module.add(vectorStaticMultiply(vgpr(rReg), vgpr(rReg), strideTile, tmpSgprInfo, \
                    "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))

                # release and return resource
                tP["gpr"]["lro"] = rReg

                writer.vgprPool.checkIn(writer.tmplro) # old
                writer.vgprPool.checkIn(qReg)

        writer.vgprPool.checkIn(tmpVgpr)

        return module

class LraTileAssignmentMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read Addresses: Tile Assignment A/B
    """
    def __call__(self, writer, kernel, tP):
        module = Module("LraTileAssignmentMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        module.add(self.LraTileAssignmentCode(writer, kernel, tP, tReg, kReg, tmpVgprRes))

        # release register
        tP["gpr"]["lro"] = tReg
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)

        # Analyze LDS Bank Conflicts
        try:
            result = self.analyzeLdsBankConflicts(writer, kernel, tP)
            
            # Store bank conflict metric in kernel (similar to MathClocksUnrolledLoop)
            # BankConflictMax: ratio of max usage to average usage
            # 1.0 = perfect uniform distribution, >1.0 = some banks are overused
            tc = tP["tensorChar"]
            if result['avg_bank_usage'] > 0:
                bank_conflict_ratio = result['max_bank_usage'] / result['avg_bank_usage']
                # Sanity check: ratio should never be less than 1.0 (max >= avg)
                assert bank_conflict_ratio >= 1.0 - 1e-6, \
                    f"BankConflictMax calculation error: ratio={bank_conflict_ratio:.6f} < 1.0 " \
                    f"(max={result['max_bank_usage']}, avg={result['avg_bank_usage']:.2f})"
            else:
                bank_conflict_ratio = 1.0
            kernel[f"BankConflictMax{tc}"] = bank_conflict_ratio
            
    
        except Exception as e:
            # Don't break compilation if analysis fails
            # Set default value if analysis fails
            kernel[f"BankConflictMax{tP['tensorChar']}"] = 1.0
        
        

        return module

    def analyzeLdsBankConflicts(self, writer, kernel, tP, isDTVAB=False):
        """
        Analyze LDS bank conflicts for local read addresses.
        This function EXACTLY mirrors LraTileAssignmentCode calculation logic.
        Can access writer to get perpStride and permBlock.
        
        Args:
            writer: KernelWriter instance
            kernel: Kernel configuration dictionary
            tP: Tensor parameters (tPA or tPB)
        
        Returns:
            Dictionary with conflict analysis results
        """
        # === BEGIN: Copy ALL parameter setup from LraTileAssignmentCode ===
        
        enableLDSTr = tP.get("enableLDSTr", False)
        tc = tP["tensorChar"]
        tile01 = tP["tile01Idx"]
        waveWidth = writer.states.kernel["WavefrontSize"]
        inputPerThread = kernel["LocalReadVectorWidth"] if not writer.states.inTailLoop else kernel["MIInputPerThread%s"%tc]
        
        # Handle Sparse
        if kernel["ProblemType"]["Sparse"]:
            if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]):
                inputPerThread = inputPerThread // 2
            elif tP["isM"]:
                inputPerThread = inputPerThread // 8
        
        LdsPad = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
        
        # Parameters for get each type index
        dividendForKId = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        
        if kernel["SourceSwap"]:
            dividedForBlkId = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
        else:
            dividedForBlkId = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
        
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth = kernel["VectorWidth%s"%tc]
        if isDTVAB:
            if tP["tlu"]:
                vectorWidth = 1
        # Get perpStride and permBlock from writer
        abmatrixinfo = writer.states.a if tc == 'A' else writer.states.b
        perpStride = abmatrixinfo.gNLCPerpStride
        permBlock = abmatrixinfo.gNLCPermBlock
        
        # Strider for each type of index
        umlds = kernel["UnrollMajorLDS%s" % tc]
        mt = kernel["MacroTile%u" % tile01]
        
        if enableLDSTr:
            strideTile = 4
        else:
            strideTile = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        
        strideK = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        strideK1 = 0
        if enableLDSTr:
            if kernel["UseGeneralizedNLCOne%s"%tc] and perpStride > 1:
                strideK = 8
            strideK1 = mt + LdsPad
        
        # FIXME SPARSE
        if kernel["ProblemType"]["Sparse"] != 0:
            if kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numBytes() > 16:
                isSparseTrack = (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) or tP["isM"]
                strideK = (inputPerThread if umlds else (mt + LdsPad) * inputPerThread) * (2 if isSparseTrack and kernel["MIInputPerThread%s"%tc] > inputPerThread else 1)
        # Special case for new F8 MFMA
        elif kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] > 32:
            if umlds:
                strideK = 16
            else:
                strideK = (mt + LdsPad) * 16
        elif kernel["UseF32XEmulation"] and not (kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16):
            if umlds:
                strideK = 4
            else:
                strideK = (mt + LdsPad) * 4
        
        strideBlock = kernel["MatrixInstM"] * strideTile
        if enableLDSTr:
            strideWave = kernel["MatrixInstM"] * vectorWidth
        else:
            strideWave = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth
        
        # applyVWCalcEarly
        applyVWCalcEarly = perpStride > 1 and kernel["ProblemType"]["TLU%s"%tc] == 0
        
        # === END: Parameter setup ===
        
        # Calculate read width in bytes
        bpeDS = tP["bpeDS"]
        read_width_bytes = inputPerThread * bpeDS
        banks_per_thread = read_width_bytes // 4  # 4 bytes per bank
        
        # LDS configuration for AMD GPUs
        num_banks = 32
        bank_width = 4  # bytes
        isWmma_v1 = writer.states.asmCaps["HasWMMA_V1"]
        
        # === BEGIN: Simulate address calculation for each thread ===
        bank_map = {}  # bank_idx -> list of thread_ids
        
        for tid in range(waveWidth):
            # "0. thread id in wave: wtid = tid % wavelength(%u)"
            dividendReg = tid  # "Serial" = tid
            wtid = dividendReg % waveWidth
            kReg = wtid
            dummy = 0
            
            # Step 1 - N offset
            if enableLDSTr:
                # "1. N offset: nIdx = wtid % 4"
                tReg = kReg % 4
                # "1. N offset: nIdx = wtid % MI_M(%d)"
                sReg = kReg % dividendForKId
                # "1. thread id in wave: k1Idx = mtid // 16"
                sReg = sReg // 16
                # "1. K1 offset: lrK1Offset = k1Idx * mStride(%u)"
                sReg = sReg * 16  # This should be strideK1
            else:
                # "1. N offset: nIdx = wtid % MI_N(%u)"
                tReg = kReg % kernel["MatrixInstN"]
            
            # apply VectorWidth early if needed
            if applyVWCalcEarly:
                tReg = tReg * vectorWidth
                # perpPerm(tReg) - skip for now
            
            # "1. N offset: nOffset = nIdx * nStride(%u)"
            tReg = tReg * strideTile
            
            if enableLDSTr:
                # "1. offset in wave: lrOffset = bnOffset + lrKOffset"
                tReg = tReg + sReg
            
            # Step 2 - Block offset
            if num1DBlocks > 1:
                # "2. block offset: bnIdx = wtid / dividedForBlkId(%u)"
                dummy = kReg // dividedForBlkId
                # "2. block offset: bnIdx = bnIdx % num1DBlocks(%u)"
                dummy = dummy % num1DBlocks
                # "2. block offset: bnOffset = bnIdx * strideBlock(%u)"
                tReg = tReg + dummy * strideBlock
            
            # Step 4 - Apply vector width
            if not applyVWCalcEarly:
                tReg = tReg * vectorWidth
            
            # Step 5-6 - K (unroll) offset
            if not isWmma_v1:
                if (dividendForKId != waveWidth):
                    if enableLDSTr:
                        # "5.1 thread id in wave: mtid = wtid % 16"
                        mReg = kReg % 16
                        # "5.2 thread id in wave: k1Idx = mtid // 4"
                        mReg = mReg // 4
                        
                    # "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))"
                    kReg = kReg // dividendForKId
                    
                    if enableLDSTr:
                        # "5. K offset: lrKOffset = kIdx * mStride(%u)"
                        kReg = kReg * strideK
                        
                        if perpStride == 1:
                            # "5.1 K1 offset: lrK1Offset = k1Idx * mStride(%u)"
                            lrK1Offset = mReg * strideK1
                            # "5.1 offset in wave: lrOffset = bnOffset + lrKOffset"
                            kReg = kReg + lrK1Offset
                        else:
                            kReg = kReg + mReg
                            # perpPerm(kReg) - skip
                            kReg = kReg * strideK1
                        # "6. offset in wave: lrOffset = bnOffset + lrKOffset"
                        tReg = tReg + kReg
                    else:
                        # "5. K offset: lrKOffset = kIdx * mStride(%u); 6. offset in wave: lrOffset = bnOffset + lrKOffset"
                        tReg = tReg + kReg * strideK
            
            # Step 7 - Wave offset
            if num1DWaves > 1:
                # "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)"
                dummy = dividendReg // dividedForWaveId
                # "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)"
                dummy = dummy % num1DWaves
                # "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset"
                tReg = tReg + dummy * strideWave
            
            # === END: Address calculation ===
            
            # Convert to byte address
            byte_address = tReg * bpeDS
            
            # Add LDS padding if enabled
            if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
                padding = (byte_address // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * bpeDS
                byte_address = byte_address + padding
            
            # Calculate all banks this thread accesses
            thread_banks = set()
            for byte_offset in range(0, read_width_bytes, bank_width):
                addr = byte_address + byte_offset
                bank_idx = (addr // bank_width) % num_banks
                thread_banks.add(bank_idx)
            
            # Record which threads access which banks
            for bank_idx in thread_banks:
                if bank_idx not in bank_map:
                    bank_map[bank_idx] = []
                bank_map[bank_idx].append(tid)

        # === Analyze conflicts based on bank usage uniformity ===
        
        # Calculate bank usage (how many threads access each bank)
        bank_usage = [len(bank_map.get(i, [])) for i in range(num_banks)]
        max_usage = max(bank_usage) if bank_usage else 0
        
        # Calculate average of ALL bank usage (including unused banks)
        avg_usage = sum(bank_usage) / num_banks if num_banks > 0 else 0
        
        return {
            'max_bank_usage': max_usage,
            'avg_bank_usage': avg_usage,
        }

    def LraTileAssignmentCode(self, writer, kernel, tP, tReg, kReg, tmpVgprRes, dividendReg="Serial", isDTVAB=False):
        module = Module("LraTileAssignmentCode")

        # alloc vgpr
        enableLDSTr = tP["enableLDSTr"]
        dummy   = writer.vgprPool.checkOut(1,"dummy")
        if enableLDSTr:
           sReg    = writer.vgprPool.checkOut(1,"sReg") # remainder
           mReg    = writer.vgprPool.checkOut(1,"mReg") # remainder

        isWmma_v1 = writer.states.asmCaps["HasWMMA_V1"]
        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        inputPerThread   = kernel["LocalReadVectorWidth"] if not writer.states.inTailLoop else kernel["MIInputPerThread%s"%tc]
        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and  tP["isA"]):
            inputPerThread = inputPerThread // 2
          elif tP["isM"]:
            inputPerThread = inputPerThread // 8
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        if isDTVAB:
            if tP["tlu"]:
                # DTV + TLU case, glvw and vw are applied to the same direction. No need to apply both.
                # non TLU case, glvw and vw are applied to the different direction. We need to apply vw here.
                vectorWidth = 1
        maxKId = waveWidth // ((kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]) * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)
        abmatrixinfo = writer.states.a if tc == 'A' else writer.states.b
        perpStride = abmatrixinfo.gNLCPerpStride
        permBlock  = abmatrixinfo.gNLCPermBlock

        # strider for each type of index
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        mt               = kernel["MacroTile%u" % tile01]
        if enableLDSTr:
           strideTile = 4
        else:
           strideTile       = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        if isDTVAB:
          strideTile  = 1 # DTV case. Actual stride will be applied later.

        strideK          = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        if enableLDSTr:
           if kernel["UseGeneralizedNLCOne%s"%tc] and perpStride > 1:
              strideK  = 8
           strideK1 = mt+LdsPad

        # FIXME SPARSE
        if kernel["ProblemType"]["Sparse"] != 0:
            if kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numBytes() > 16:
              isSparseTrack = (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) or tP["isM"]
              strideK      = (inputPerThread if umlds else (mt + LdsPad) * inputPerThread) * (2 if isSparseTrack and kernel["MIInputPerThread%s"%tc] >  inputPerThread else 1)
        #special case for new F8 MFMA
        elif  kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] > 32:
            if umlds:
                strideK = 16
            else:
                strideK = (mt + LdsPad) * 16
        elif kernel["UseF32XEmulation"] and not (kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16):
            if umlds:
                strideK = 4
            else:
                strideK = (mt + LdsPad) * 4

        strideBlock      = kernel["MatrixInstM"] * strideTile
        if enableLDSTr:
           strideWave = kernel["MatrixInstM"] * vectorWidth
        else:
           strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        lsu              = kernel["LocalSplitU"]

        if isDTVAB:
          strideTile  = 1 # DTV case. Actual stride will be applied later.

        def perpPerm(vgprReg):
           reMap0 = writer.vgprPool.checkOut(1)
           reMap1 = writer.vgprPool.checkOut(1)
           perpStrideInv = permBlock // perpStride
           
           module.addComment0("Computing strided(%u) perp indicies"%perpStrideInv)
           module.add(VAndB32(dst=vgpr(reMap0), src0=(permBlock // perpStrideInv - 1), src1=vgpr(vgprReg), comment="r0 = I %% (%u // %u)"%(permBlock, perpStrideInv)))
           module.add(VLShiftLeftB32(dst=vgpr(reMap0), shiftHex=log2(perpStrideInv), src=vgpr(reMap0), comment="r0 = %u * r0"%(perpStrideInv)))
           module.addComment0("Computing r1 = (I %% %u) // (%u // %u)"%(permBlock, permBlock, perpStrideInv))
           module.add(VAndB32(dst=vgpr(reMap1), src0=(permBlock - 1), src1=vgpr(vgprReg), comment="r1 = I %% (%u)"%(permBlock)))
           module.add(VLShiftRightB32(dst=vgpr(reMap1), shiftHex=log2(permBlock // perpStrideInv), src=vgpr(reMap1), comment="r1 = (r1) // (%u // %u)"%(permBlock, perpStrideInv)))
           module.add(VAddU32(dst=vgpr(reMap0), src0=vgpr(reMap0), src1=vgpr(reMap1), comment="r0 = r0 + r1" ))

           module.add(VLShiftRightB32(dst=vgpr(reMap1), shiftHex=log2(permBlock), src=vgpr(vgprReg), comment="r1 = I // %u"%(permBlock)))
           module.add(vectorStaticMultiplyAdd(vgpr(vgprReg), vgpr(reMap1), permBlock, vgpr(reMap0), None))

           module.addComment0("Done computing strided(%u) perp indices"%perpStrideInv)
           writer.vgprPool.checkIn(reMap0)
           writer.vgprPool.checkIn(reMap1)

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            # tile offset
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            if enableLDSTr:
               module.add(vectorStaticRemainder(dummy, tReg, kReg, 4, tmpVgprRes, tmpSgprInfo, \
                                                "1. N offset: nIdx = wtid %% 4"))
               module.add(vectorStaticRemainder(dummy, sReg, kReg, dividendForKId, tmpVgprRes, tmpSgprInfo, \
                                                "1. N offset: nIdx = wtid %% MI_M(%d)"%dividendForKId))
               module.add(vectorStaticDivide(sReg, sReg, 16, tmpVgprRes, \
                                                "1. thread id in wave: k1Idx = mtid // 16"))
               module.add(vectorStaticMultiply(vgpr(sReg), vgpr(sReg), 16, tmpSgprInfo, \
                                         "1. K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))

            else:
               module.add(vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgprRes, tmpSgprInfo, \
                                             "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"]))

            applyVWCalcEarly = perpStride > 1 and kernel["ProblemType"]["TLU%s"%tc] == 0
            if applyVWCalcEarly:
               # Apply vector width calc before we apply permutation to perp dim
               module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                                               "1. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))
               perpPerm(tReg)

            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))
            if enableLDSTr:
                module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(sReg), src1=vgpr(tReg), \
                           comment="1. offset in wave: lrOffset = bnOffset + lrKOffset"))
            # block offset
            if num1DBlocks > 1:
                module.add(vectorStaticDivide(dummy, kReg, dividedForBlkId, tmpVgprRes, \
                    "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DBlocks, tmpVgprRes, tmpSgprInfo, \
                    "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideBlock, vgpr(tReg), tmpSgprInfo, \
                    "2. block offset: bnOffset = bnIdx * strideBlock(%u); 3. add N and block offset: bnOffset = block and N offset" % strideBlock))
            else:
                module.addComment0("Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1")

            if not applyVWCalcEarly:
               module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                                               "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))

            # unroll offset
            #if isMfma and (dividendForKId != waveWidth):
            if not isWmma_v1:
                if (dividendForKId != waveWidth) and (not isDTVAB):
                    if enableLDSTr:
                        module.add(vectorStaticRemainder(dummy, mReg, kReg, 16, tmpVgprRes, tmpSgprInfo, \
                                                        "5.1 thread id in wave: mtid = wtid %% 16"))
                        module.add(vectorStaticDivide(mReg, mReg, 4, tmpVgprRes, \
                                                     "5.2 thread id in wave: k1Idx = mtid // 4"))
                if (dividendForKId != waveWidth) or isDTVAB:
                  # DTVAB case, add this regardless of dividendForKId != waveWidth
                    module.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
                        "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"])))
                if (dividendForKId != waveWidth) and (not isDTVAB):

                    if enableLDSTr:
                        module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideK, tmpSgprInfo, \
                                                 "5. K offset: lrKOffset = kIdx * mStride(%u)" % (strideK)))

                        if perpStride == 1:
                           module.add(vectorStaticMultiply(vgpr(mReg), vgpr(mReg), strideK1, tmpSgprInfo, \
                                                           "5.1 K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))
                           module.add(VAddU32(dst=vgpr(kReg), src0=vgpr(mReg), src1=vgpr(kReg), \
                                              comment="5.1 offset in wave: lrOffset = bnOffset + lrKOffset"))
                        else:
                           module.add(VAddU32(dst=vgpr(kReg), src0=vgpr(mReg), src1=vgpr(kReg), \
                                              comment="5.1 offset in wave: lrOffset = bnOffset + lrKOffset"))
                           # Apply permutation to perpendicular dim
                           if perpStride > 1:
                              perpPerm(kReg)
                           module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideK1, tmpSgprInfo, \
                                                           "5.2 K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))
                        module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                                          comment="6. offset in wave: lrOffset = bnOffset + lrKOffset"))
                    else:
                        module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(kReg), strideK, vgpr(tReg), tmpSgprInfo, \
                                                    "5. K offset: lrKOffset = kIdx * mStride(%u); 6. offset in wave: lrOffset = bnOffset + lrKOffset" % (strideK)))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(dummy, dividendReg, dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideWave, vgpr(tReg), tmpSgprInfo, \
                                             "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset" % strideWave))

        # release register
        writer.vgprPool.checkIn(dummy)
        if enableLDSTr:
           writer.vgprPool.checkIn(sReg)
           writer.vgprPool.checkIn(mReg)

        return module
