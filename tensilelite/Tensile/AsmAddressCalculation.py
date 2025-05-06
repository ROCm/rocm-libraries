################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import Module
from rocisa.container import EXEC, VCC, vgpr, sgpr
from rocisa.instruction import MacroInstruction, SAddCU32, SAddU32, SAndB32, \
    SAndB64, SLShiftLeftB32, SMovB32, SMovB64, SMulI32, SSubBU32, SSubU32, \
    VAddCCOU32, VAddCOU32, VAddI32, VAddLShiftLeftU32, VAddU32, VAndB32, VBfiB32, \
    VCmpEQU32, VCmpGtU32, VCmpLtU32, VCmpXNeU32, VCndMaskB32, VLShiftLeftB32, \
    VMadI32I24, VMovB32, VMulLOU32, VSubI32, VSubU32

from .Common import INDEX_CHARS, DataDirection, log2

##############################################################################
# Fields associated with computing address
##############################################################################
class AddrCalculation:
    # rowInc is number of rows to add to the base address
    # coord0Vgpr : This is VGPR that holds coord0.  Coord0 is element-space
    #    packed index for the 0 coordinate of the C/D matrix.
    # coord1Vgpr : VGPR which tracks the last coord1 calculation.
    #          If this is new coord1, just overwrite it with latest calc.
    def __init__(self, kernelWriter, ss, addrCVgpr, addrDVgpr, addrGSUSyncVgprs, addrEVgpr, addrBiasVgpr, addrScaleAVecVgpr, addrScaleBVecVgpr, addrScaleAlphaVecVgpr, element, \
        coordOffset0, coord1Vgpr, coordOffset1, rowInc, newCoord1, vectorDataTypes):
        self.kernelWriter = kernelWriter

        # vgprs for address, could be more than one (for flat)
        self.addrEVgpr    = addrEVgpr
        self.addrDVgpr    = addrDVgpr
        self.addrGSUSyncVgprs    = addrGSUSyncVgprs
        self.addrCVgpr    = addrCVgpr
        self.addrBiasVgpr = addrBiasVgpr
        self.addrScaleAVecVgpr = addrScaleAVecVgpr
        self.addrScaleBVecVgpr = addrScaleBVecVgpr
        self.addrScaleAlphaVecVgpr = addrScaleAlphaVecVgpr
        self.coord1Vgpr = coord1Vgpr # vgpr that stores coord1Vgpr

        self.element = element
        self.coordOffset0 = coordOffset0
        self.coordOffset1 = coordOffset1
        self.rowInc = rowInc
        self.rowIncDirtyRowPtr = 0 # rowInc was used to modify rowPtr, need to recompute addr
        self.newCoord1 = newCoord1 # vgpr that stores newCoord1
        self.vectorDataTypes = vectorDataTypes

        self.biasOffset = [0, 0]
        self.scaleAVecOffset = 0
        self.scaleBVecOffset = 0
        self.scaleAlphaVecOffset = [0, 0]
        if ss.optSingleColVgpr:
            # optimized stores use the load offset for coordOffset0 calculations.
            self.biasOffset[0]   = coordOffset0 * kernelWriter.states.bpeCinternal + self.vectorDataTypes.bias(0).ldsOffset
            self.biasOffset[1]   = coordOffset1 * kernelWriter.states.bpeCinternal + self.vectorDataTypes.bias(1).ldsOffset
            self.scaleAVecOffset = coordOffset0 * kernelWriter.states.bpeCinternal + self.vectorDataTypes.scaleA.ldsOffset
            self.scaleBVecOffset = coordOffset1 * kernelWriter.states.bpeCinternal + self.vectorDataTypes.scaleB.ldsOffset
            self.scaleAlphaVecOffset[0]   = coordOffset0 * kernelWriter.states.bpeCinternal + self.vectorDataTypes.scaleAlpha(0).ldsOffset
            self.scaleAlphaVecOffset[1]   = coordOffset1 * kernelWriter.states.bpeCinternal + self.vectorDataTypes.scaleAlpha(1).ldsOffset
            self.globalOffset  = coordOffset0 * kernelWriter.states.bpeCexternal
            self.globalOffsetE = coordOffset0 * kernelWriter.states.bpeE
            self.globalOffsetInternal = coordOffset0 * kernelWriter.states.bpeCinternal
        else:
            # else non-opt stores include the coord0 offset into VGPR address calcs
            self.biasOffset[0]   = self.vectorDataTypes.bias(0).ldsOffset
            self.biasOffset[1]   = self.vectorDataTypes.bias(1).ldsOffset
            self.scaleAVecOffset = self.vectorDataTypes.scaleA.ldsOffset
            self.scaleBVecOffset = self.vectorDataTypes.scaleB.ldsOffset
            self.scaleAlphaVecOffset[0] = self.vectorDataTypes.scaleAlpha(0).ldsOffset
            self.scaleAlphaVecOffset[1] = self.vectorDataTypes.scaleAlpha(1).ldsOffset
            self.globalOffset = 0
            self.globalOffsetE = 0
            self.globalOffsetInternal = 0

    def addScaled(self, destV, src0, src1, scale1, tmpS01, comment=""):
        """
        Use minimally efficient instructions to add stride*scale
        """

        module = Module("addScaled")
        if scale1 == 1:
            module.add(VAddU32(dst=destV, src0=src0, src1=src1, comment=comment))
        else:
            module.add(SMulI32(dst=sgpr(tmpS01), src0=src1, src1=scale1, comment="scale stride"))
            module.add(VAddI32(dst=destV, src0=src0, src1=sgpr(tmpS01), comment=comment))
        return module


    def emitAddressCoordIncrement(self, kernel, ss, tmpVgpr, tmpS01, updateCoord1):
        """
        Emit code that computes the coord0 and coord1 for this element
        sets self.coord0Vgpr with the address that holds the coord0 value for this element.
        Input:
          - tmpVgpr is a 1 temporary VGPR used for coord0 calculation on edges
        """

        module = Module("emitAddressCoordIncrement")
        kw = self.kernelWriter
        (d1,d0,vc1,vc0) = self.element
        self.coord0Vgpr = None # will set below

        # module.addComment0("store addr=v%u coordOffset0=%u"% \
        #    (self.addr, self.coordOffset0))
        module.addComment0("(d1,vc1,d0,vc0)=(%u,%u,%u,%u)"\
            % (d1,vc1,d0,vc0))
        if ss.optSingleColVgpr:
            self.coord0Vgpr = kw.vgprs.coord0
        elif not ss.optSharedColVgpr or (d1 == vc1 == 0):
            # not share mode or first row always does the address calc math:

            if self.coordOffset0 == 0:
                self.coord0Vgpr = kw.vgprs.coord0
            elif self.coordOffset0 <= 64:
                self.coord0Vgpr = tmpVgpr
                module.add(VAddCOU32(dst=vgpr(self.coord0Vgpr), dst1=VCC(), src0=vgpr(kw.vgprs.coord0), src1=self.coordOffset0, \
                          comment="coord0.1: coord0 += d0*sg0*VW + vc0"))
            else:
                self.coord0Vgpr = tmpVgpr
                module.add(SMovB32(dst=sgpr(tmpS01), src=self.coordOffset0, comment="coordOffset0 d0=%u vc0=%u"%(d0, vc0)))
                module.add(VAddCOU32(dst=vgpr(self.coord0Vgpr), dst1=VCC(), src0=vgpr(kw.vgprs.coord0), src1=sgpr(tmpS01), \
                          comment="coord0.2: coord0 += d0*sg0*VW + vc0"))

            if self.newCoord1:
                if not kernel["BufferStore"] or updateCoord1:
                    if self.rowInc== 0:
                        None
                    elif self.rowInc <= 64 and self.rowInc > 0:
                        # rowInc fits in instruction:
                        module.add(VAddCOU32(dst=vgpr(self.coord1Vgpr), dst1=VCC(), \
                                  src0=vgpr(self.kernelWriter.vgprs.coord1), src1=self.rowInc, \
                                  comment="coord1.1: coord1Vgpr += d1*sg1*VW + vc1"))
                    elif self.rowInc > 0:
                        module.add(SMovB32(dst=sgpr(tmpS01), src=self.rowInc, comment="rowInc d1=%u vc1=%u"%(d0, vc0)))
                        module.add(VAddCOU32(dst=vgpr(self.coord1Vgpr), dst1=VCC(), \
                                  src0=vgpr(self.kernelWriter.vgprs.coord1), src1=sgpr(tmpS01), \
                                  comment="coord1.2: coord1 += d1*sg1*VW + vc1"))
                    else:
                        # rowInc < 0
                        module.add(SMovB32(dst=sgpr(tmpS01), src=self.rowInc, comment="rowInc d1=%u vc1=%u"%(d0, vc0)))
                        module.add(VAddI32(dst=vgpr(self.coord1Vgpr), \
                                  src0=vgpr(self.kernelWriter.vgprs.coord1), src1=sgpr(tmpS01), \
                                  comment="coord1.3: coord1 += d1*sg1*VW + vc1"))
        return module

    def getRowPtr(self, kw, tc):
        if tc == 'C':
            return kw.vgprs.cinRowPtr
        elif tc == 'E':
            return kw.vgprs.coutRowPtrE
        elif tc == 'Bias':
            return kw.vgprs.coutRowPtrBias
        else:
            return kw.vgprs.coutRowPtrD

    def getAddrVgpr(self, kw, tc):
        if tc == 'C':
            return self.addrCVgpr
        elif tc == 'E':
            return self.addrEVgpr
        elif tc == 'Bias':
            return self.addrBiasVgpr
        elif tc == 'TD':
            return self.addrGSUSyncVgprs
        else:
            return self.addrDVgpr

    # storeChar is 'C' or 'D'
    # elementVgpr is coord0Vgpr*strideCD0, or optimized to just coord0Vgpr if strideCD0 is unit const
    def emitExtractAndScalePackedDims(self, kernel, ss, tmpVgpr, storeChar):
        module = Module("emitExtractAndScalePackedDims")
        kw = self.kernelWriter
        packedIndices = kernel["PackedC0IndicesX"]
        packedBits = self.coord0Vgpr # start with coord0, will move to temp below
        rowPtr = self.getRowPtr(kw, storeChar)
        addrVgpr = self.getAddrVgpr(kw, storeChar)
        bpe = kw.states.bpeCinternal if (tc == 'Bias') else (kw.states.bpeE if (tc == 'E') else kw.states.bpeCexternal)

        for i,idx in enumerate(packedIndices[:-1]):
            # vgprTmp assignments:
            #   - tmp+0 may be the incoming packed coordinate 0, used on replay too
            #   - tmp+1 is DIV output
            #   - tmp+2 is scratch
            idxChar= INDEX_CHARS[idx]
            module.addComment0("extract %s"%kw.sizeRef(idx))
            assert(tmpVgpr+1 != packedBits) # bad since we still need packedBits below for remainder (can't overwrite here)
            module.add(MacroInstruction("V_MAGIC_DIV", \
                           args=[tmpVgpr+1, vgpr(packedBits), sgpr("MagicNumberSize%s"%idxChar), \
                           sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0"]))
            # tmpVgpr+1 returns the quotient, tmpVgpr+2 is overwritten

            # compute remainder, packedBits % sizeIdx - this is the 'extracted' index that must be scaled
            # remainder is mul and sub
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+2), src0=vgpr(tmpVgpr+1), src1=kw.sizeRef(idx), \
                           comment="remainder part 1"))
            module.add(VSubU32(dst=vgpr(tmpVgpr+2), src0=vgpr(packedBits), src1=vgpr(tmpVgpr+2),
                           comment="remainder part 2"))

            if i==0:
                module.add(VMulLOU32(dst=vgpr(addrVgpr), src0=vgpr(tmpVgpr+2), \
                          src1=kw.strideRef(storeChar, idx), comment="addrCalc <- scaled extracted dim"))
            else:
                module.add(VMulLOU32(dst=vgpr(tmpVgpr+2), src0=vgpr(tmpVgpr+2), \
                          src1=kw.strideRef(storeChar, idx), comment="scale extracted dim"))
                module.add(VAddU32(dst=vgpr(addrVgpr), src0=vgpr(addrVgpr), \
                          src1=vgpr(tmpVgpr+2), comment="addrCalc += scaled extracted dim "))

            if i < len(packedIndices)-2:
                # TODO - might be able to eliminate this
                module.add(VMovB32(dst=vgpr(tmpVgpr+0), src=vgpr(tmpVgpr+1), \
                          comment="Copy remaining bits for next divide"))
                packedBits = tmpVgpr+0

        if len(packedIndices)>1:
            # if we unpacked something, then scale it to BPE
            module.addComment0("extract final %s"%kw.sizeRef(packedIndices[-1]))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+2), src0=vgpr(tmpVgpr+1), \
                      src1=kw.strideRef(storeChar, packedIndices[-1]), comment="scale final extracted dim"))
            module.add(VAddU32(dst=vgpr(addrVgpr), src0=vgpr(addrVgpr), \
                      src1=vgpr(tmpVgpr+2), comment="addrCalc += scaled extracted dim "))

            module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                      src0=vgpr(rowPtr), \
                      src1=vgpr(addrVgpr), \
                      shiftHex=hex(log2(bpe)), \
                      comment="packed: add rowPtr and scaleToBpe"))

        return module

    def emitScaleToBpe(self, kernel, ss, tmpVgpr, tmpSgpr, singleUpdate, tc, dim):
        """
        Needs 3 temporary VGPRs
        """

        module = Module("emitScaleToBpe")
        kw = self.kernelWriter
        (d1,d0,vc1,vc0) = self.element
        rowPtr = self.getRowPtr(kw, tc)
        addrVgpr = self.getAddrVgpr(kw, tc)
        bpe = kw.states.bpeCinternal if (tc == 'Bias') else (kw.states.bpeE if (tc == 'E') else kw.states.bpeCexternal)
        if (tc == 'C' or tc == 'TD'):
            bpe = bpe if (kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel") else kw.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()
        # set when we generate code that updates the address
        # optSingleColVgpr and optSharedColVgpr attempt to minimize these updates
        updatedAddr = False

        # scale and set final address:
        stride0 = kw.strideRef('D', 0) if ((tc == 'Bias') or (tc == 'ScaleAlphaVec') or (tc == 'ScaleAVec') or (tc == 'ScaleBVec')) else kw.strideRef(tc, 0)
        if kw.isConstUnitStride(stride0):
            elementVgpr = self.coord0Vgpr
        else:
            module.add(VMulLOU32(dst=vgpr(addrVgpr), \
                src0=vgpr(self.coord0Vgpr), \
                src1=stride0, \
                comment="scale element by non-unit stride"))
            elementVgpr = addrVgpr

        isSingleKernel = ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel") or kernel["StreamK"] > 0
        if ss.optSingleColVgpr:
            # This is first element in the first batch, create a byte address that will
            # be re-used by subsequent elements:
            # if this element is firstInBatch - may need to set up a bpe-scaled row pointer for the batch:
            #  - need row-ptr start of each batch
            assert (kw.vgprs.coord0 == self.coord0Vgpr) # elementAddr assignment above assumes these are the same
            if singleUpdate:
                updatedAddr = True
                if tc == 'C':
                    singleColAddrUpdated = ss.singleColCAddrUpdated
                elif tc == 'E':
                    singleColAddrUpdated = ss.singleColEAddrUpdated
                elif tc == 'Bias':
                    singleColAddrUpdated = ss.singleColBiasAddrUpdated
                elif tc == 'TD':
                    singleColAddrUpdated = ss.singleColTDAddrUpdated
                else:
                    singleColAddrUpdated = ss.singleColDAddrUpdated
                if not singleColAddrUpdated or not ss.optSrdIncForRow:
                    if tc == 'Bias' and kw.states.useBias == DataDirection.READ:
                        coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                        module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                        module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                                shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                src=vgpr(self.addrBiasVgpr), \
                                                comment="Bias address scaled by BPE"))
                        if kernel["LdsOffsetBias"] != 0:
                          module.add(VAddU32(dst=vgpr(self.addrBiasVgpr), \
                                             src0=(kernel["LdsOffsetBias"]), \
                                             src1=vgpr(self.addrBiasVgpr), \
                                             comment="add lds offset"))
                        ss.singleColBiasAddrUpdated = True
                        return module
                    if tc == 'ScaleAlphaVec' and kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
                        if self.addrScaleAlphaVecVgpr:
                            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                            coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                            module.add(VSubU32(dst=vgpr(self.addrScaleAlphaVecVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                            module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                                    shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                                    src=vgpr(self.addrScaleAlphaVecVgpr), \
                                                    comment="ScaleAlpha address scaled by BPE"))
                            if kernel["LdsOffsetBias"] != 0:
                                module.add(VAddU32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                                   src0=(kernel["LdsOffsetBias"]), \
                                                   src1=vgpr(self.addrScaleAlphaVecVgpr), \
                                                   comment="add lds offset"))
                        return module
                    if tc == 'ScaleAVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                        if self.addrScaleAVecVgpr:
                            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                            module.add(VSubU32(dst=vgpr(self.addrScaleAVecVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                            module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAVecVgpr), \
                                                    shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                    src=vgpr(self.addrScaleAVecVgpr), \
                                                    comment="ScaleAVec address scaled by BPE"))
                            if kernel["LdsOffsetBias"] != 0:
                                module.add(VAddU32(dst=vgpr(self.addrScaleAVecVgpr), \
                                                   src0=(kernel["LdsOffsetBias"]), \
                                                   src1=vgpr(self.addrScaleAVecVgpr), \
                                                   comment="add lds offset"))
                        return module
                    if tc == 'ScaleBVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                        if self.addrScaleBVecVgpr:
                            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr("WorkGroup1"), comment="wgp1 * MT1"))
                            module.add(VSubU32(dst=vgpr(self.addrScaleBVecVgpr), src0=vgpr(self.coord1Vgpr), src1=sgpr(tmpSgpr)))
                            module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleBVecVgpr), \
                                                    shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                    src=vgpr(self.addrScaleBVecVgpr), \
                                                    comment="ScaleBVec address scaled by BPE"))
                            if kernel["LdsOffsetBias"] != 0:
                                module.add(VAddU32(dst=vgpr(self.addrScaleBVecVgpr), \
                                                   src0=(kernel["LdsOffsetBias"]), \
                                                   src1=vgpr(self.addrScaleBVecVgpr), \
                                                   comment="add lds offset"))
                        return module
                    if tc == 'C':
                        ss.singleColCAddrUpdated    = True
                    elif tc == 'E':
                        ss.singleColEAddrUpdated    = True
                    elif tc == 'Bias':
                        ss.singleColBiasAddrUpdated = True
                    elif tc == 'TD':
                        ss.singleColTDAddrUpdated    = True
                    else:
                        ss.singleColDAddrUpdated    = True
                    module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                      src0=vgpr(rowPtr), \
                      src1=vgpr(elementVgpr), \
                      shiftHex=hex(log2(bpe)), \
                      comment="optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=%d, coord0Vgpr=%d"%(kw.vgprs.coord0, self.coord0Vgpr)))
        elif ss.optSharedColVgpr:
            # Need an address calculation for the first address in each row:
            if d1==0 and vc1==0:
                if tc == 'Bias' and kw.states.useBias == DataDirection.READ:
                    coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                    module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                            shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                            src=vgpr(self.addrBiasVgpr), \
                                            comment="Bias address scaled by BPE"))
                    if kernel["LdsOffsetBias"] != 0:
                        module.add(VAddU32(dst=vgpr(self.addrBiasVgpr), \
                                           src0=(kernel["LdsOffsetBias"]), \
                                           src1=vgpr(self.addrBiasVgpr), \
                                           comment="add lds offset"))
                    return module
                if tc == 'ScaleAlphaVec' and kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
                    if self.addrScaleAlphaVecVgpr:
                        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                        coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                        module.add(VSubU32(dst=vgpr(self.addrScaleAlphaVecVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                        module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                                shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                                src=vgpr(self.addrScaleAlphaVecVgpr), \
                                                comment="ScaleAlpha address scaled by BPE"))
                        if kernel["LdsOffsetBias"] != 0:
                            module.add(VAddU32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                               src0=(kernel["LdsOffsetBias"]), \
                                               src1=vgpr(self.addrScaleAlphaVecVgpr), \
                                               comment="add lds offset"))
                    return module
                if tc == 'ScaleAVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                    if self.addrScaleAVecVgpr:
                        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                        module.add(VSubU32(dst=vgpr(self.addrScaleAVecVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                        module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAVecVgpr), \
                                                shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                src=vgpr(self.addrScaleAVecVgpr), \
                                                comment="ScaleAVec address scaled by BPE"))
                        if kernel["LdsOffsetBias"] != 0:
                            module.add(VAddU32(dst=vgpr(self.addrScaleAVecVgpr), \
                                               src0=(kernel["LdsOffsetBias"]), \
                                               src1=vgpr(self.addrScaleAVecVgpr), \
                                               comment="add lds offset"))
                    return module
                if tc == 'ScaleBVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                    if self.addrScaleBVecVgpr:
                        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr("WorkGroup1"), comment="wgp1 * MT1"))
                        module.add(VSubU32(dst=vgpr(self.addrScaleBVecVgpr), src0=vgpr(self.coord1Vgpr), src1=sgpr(tmpSgpr)))
                        module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleBVecVgpr), \
                                                shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                src=vgpr(self.addrScaleBVecVgpr), \
                                                comment="ScaleBVec address scaled by BPE"))
                        if kernel["LdsOffsetBias"] != 0:
                            module.add(VAddU32(dst=vgpr(self.addrScaleBVecVgpr), \
                                               src0=(kernel["LdsOffsetBias"]), \
                                               src1=vgpr(self.addrScaleBVecVgpr), \
                                               comment="add lds offset"))
                    return module
                packedIndices = kernel["PackedC0IndicesX"]
                if len(packedIndices) > 1:
                    updatedAddr = True
                    module.add(self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc))
                else:
                    updatedAddr = True
                    module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                      src0=vgpr(rowPtr), \
                      src1=vgpr(elementVgpr), \
                      shiftHex=hex(log2(bpe)), \
                      comment="optSharedColVgpr scaleToBpe for first row: col addr <- cinRowPtr + coord0, scaled by BPE"))
        else:
            # Generate final address calculation (to bytes) for each element
            # The unpacking takes 8-10 instructions so could be worth optimizing someday :
            # each col has same offset so could create a class to hold column-specific state including
            # the byte address offset for that col and the mask in/out.
            if tc == 'Bias' and kw.states.useBias == DataDirection.READ:
                coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                        shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                        src=vgpr(self.addrBiasVgpr), \
                                        comment="Bias address scaled by BPE"))
                if kernel["LdsOffsetBias"] != 0:
                    module.add(VAddU32(dst=vgpr(self.addrBiasVgpr), \
                                       src0=(kernel["LdsOffsetBias"]), \
                                       src1=vgpr(self.addrBiasVgpr), \
                                       comment="add lds offset"))
                return module
            if tc == 'ScaleAlphaVec' and kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
                if self.addrScaleAlphaVecVgpr:
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                    coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                    module.add(VSubU32(dst=vgpr(self.addrScaleAlphaVecVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                            shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                            src=vgpr(self.addrScaleAlphaVecVgpr), \
                                            comment="ScaleAlpha address scaled by BPE"))
                    if kernel["LdsOffsetBias"] != 0:
                        module.add(VAddU32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                           src0=(kernel["LdsOffsetBias"]), \
                                           src1=vgpr(self.addrScaleAlphaVecVgpr), \
                                           comment="add lds offset"))
                return module
            if tc == 'ScaleAVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                if self.addrScaleAVecVgpr:
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                    module.add(VSubU32(dst=vgpr(self.addrScaleAVecVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAVecVgpr), \
                                            shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                            src=vgpr(self.addrScaleAVecVgpr), \
                                            comment="ScaleAVec address scaled by BPE"))
                    if kernel["LdsOffsetBias"] != 0:
                        module.add(VAddU32(dst=vgpr(self.addrScaleAVecVgpr), \
                                           src0=(kernel["LdsOffsetBias"]), \
                                           src1=vgpr(self.addrScaleAVecVgpr), \
                                           comment="add lds offset"))
                return module
            if tc == 'ScaleBVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                if self.addrScaleBVecVgpr:
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr("WorkGroup1"), comment="wgp1 * MT1"))
                    module.add(VSubU32(dst=vgpr(self.addrScaleBVecVgpr), src0=vgpr(self.coord1Vgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleBVecVgpr), \
                                            shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                            src=vgpr(self.addrScaleBVecVgpr), \
                                            comment="ScaleBVec address scaled by BPE"))
                if kernel["LdsOffsetBias"] != 0:
                    module.add(VAddU32(dst=vgpr(self.addrScaleBVecVgpr), \
                                       src0=(kernel["LdsOffsetBias"]), \
                                       src1=vgpr(self.addrScaleBVecVgpr), \
                                       comment="add lds offset"))
                return module
            packedIndices = kernel["PackedC0IndicesX"]
            if len(packedIndices) > 1:
                updatedAddr = True
                module.add(self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc))
            else:
                updatedAddr = True
                module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                    src0=vgpr(rowPtr), \
                    src1=vgpr(elementVgpr), \
                    shiftHex=hex(log2(bpe)), \
                    comment="scaleToBpe: accumulate d0 lower and *= bpe into Cin addr"))

        # if not optSrdIncForRow then we may have moved the row pointer
        # and depending on paths above may not have refreshed addrVgpr already.
        # if so - do it here:
        if self.rowIncDirtyRowPtr and not updatedAddr:
            module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
              src0=vgpr(rowPtr), \
              src1=vgpr(kw.vgprs.coord0), \
              shiftHex=hex(log2(bpe)), \
              comment="scaleToBpe: Update address with new rowPtr"))

        return module

    def edgeProtectCode(self, kernel, edge, beta, atomic, mask, tmpSgpr):
        """
        Generate code to protect address offset in edge case
        """

        module = Module("edgeProtectCode")
        kw = self.kernelWriter
        tmpS01 = tmpSgpr
        tmpS23 = tmpSgpr+self.kernelWriter.states.laneSGPRCount

        laneSGPRCount = self.kernelWriter.states.laneSGPRCount
        wavefrontSize = kernel["WavefrontSize"]

        # Now do the edge check and compute the address in bytes:
        if kernel["BufferStore"]:
            if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and (beta or kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel"))):
                # Set address to -1 if OOB on either dimension
                # and only check the x/coord0 index here, save a couple inst
                sizeBoundary = [0,0]
                sizeBoundary[0] = \
                    sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
                    else kw.sizeRef(kernel["ProblemType"]["Index0"])
                sizeBoundary[1] = \
                    sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
                    else kw.sizeRef(kernel["ProblemType"]["Index1"])

                module.add(VCmpLtU32(dst=sgpr(tmpS01,laneSGPRCount), src0=vgpr(self.coord0Vgpr), src1=sizeBoundary[0], comment="coord0 < size0" ))
                module.add(VCmpLtU32(dst=sgpr(mask,laneSGPRCount), src0=vgpr(self.coord1Vgpr), src1=sizeBoundary[1], comment="coord1 < size1" ))
                SAndX = SAndB64 if wavefrontSize == 64 else SAndB32
                module.add(SAndX(dst=sgpr(mask,laneSGPRCount), src0=sgpr(tmpS01,laneSGPRCount), src1=sgpr(mask,laneSGPRCount), comment="in0 && in1" ))
        else:
            module.add(VCmpLtU32(dst=sgpr(tmpS01,laneSGPRCount), src0=vgpr(self.coord0Vgpr), src1=sgpr("SizesFree+0"), comment="coord0 < size0" ))
            module.add(VCmpLtU32(dst=sgpr(tmpS23,laneSGPRCount), src0=vgpr(self.coord1Vgpr), src1=sgpr("SizesFree+1"), comment="coord1 < size1" ))
            SAndX = SAndB64 if wavefrontSize == 64 else SAndB32
            module.add(SAndX(dst=sgpr(mask,laneSGPRCount), src0=sgpr(tmpS01,laneSGPRCount), src1=sgpr(tmpS23,laneSGPRCount), comment="in0 && in1" ))

            if (beta or atomic):
                SMovX = SMovB64 if wavefrontSize == 64 else SMovB32
                module.add(SMovX(dst=EXEC(), src=sgpr(mask,laneSGPRCount), comment="sgprs -> exec" ))

        return module

    # TODO - mask should be part of AddrCalc state not passed as parm
    def emitAddressSetupCode(self, kernel, tPB, ss, tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrVgpr):
        """
        Generate code to set up the address vgpr
        Input:
          tmpVgpr : two temp vgprs
        Output:
          Returns kStr with appropriate setup code
          Sets self.coord0Vgpr with vgpr that contains the coord0 for this element.  This enables
            optimization - if no setup code is required the coord0 can be the input.
        """

        module = Module("emitAddressSetupCode")
        kw = self.kernelWriter

        updateCoord1 = (edge or len(kernel["PackedC1IndicesX"]) > 1)
        module.add(self.emitAddressCoordIncrement(kernel, ss, tmpVgpr, tmpS01, updateCoord1))

        # calculate flat load offset
        if not kernel["BufferStore"]:
            # flat: in-bounds exec mask
            # global offset macro (requires 3 tmpVgpr)
            # final address = C + index*bytes
            params = ["%u" % addrVgpr]
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
                if i == kernel["ProblemType"]["Index0"]:
                    params.append("%s" % (self.coord0Vgpr))
                elif i == kernel["ProblemType"]["Index1"]:
                    params.append("%s" % (self.coord1Vgpr))
                else: # just a group index
                    params.append("sgprWorkGroup%u"%i)
            params.append("%s" % (tmpVgpr+2))
            module.add(MacroInstruction(name="GLOBAL_OFFSET_C", args=params))
            module.add(VMovB32(dst=vgpr(tmpVgpr+2), src=vgpr(addrVgpr+0), comment="temp store offset 0"))
            module.add(VMovB32(dst=vgpr(tmpVgpr+3), src=vgpr(addrVgpr+1), comment="temp store offset 1"))

        # Move the row ptr VGPR
        # optSrdIncForRow moves the SRD so don't move here
        if not ss.optSrdIncForRow and kernel["BufferStore"]:
            if self.rowInc != 0:
                self.rowIncDirtyRowPtr = 1
                #assert (not kernel["ProblemType"]["UseInitialStridesCD"])
                module.addComment1("Fix for UseInitialStridesCD, emitAddressSetupCode")

                if len(kernel["PackedC1IndicesX"]) == 1:
                    strideChar = self.kernelWriter.states.indexChars[kernel["PackedC1IndicesX"][0]]
                    module.add(self.addScaled(vgpr(kw.vgprs.cinRowPtr),  vgpr(kw.vgprs.cinRowPtr),  \
                              sgpr("StrideC%s"%strideChar), self.rowInc, tmpS01, "ROWINC- Move cinRowPtr to next row"))
                    module.add(self.addScaled(vgpr(kw.vgprs.coutRowPtrD), vgpr(kw.vgprs.coutRowPtrD), \
                              sgpr("StrideD%s"%strideChar), self.rowInc, tmpS01, "Move coutRowPtrD to next row"))
                    if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1):
                        module.add(self.addScaled(vgpr(kw.vgprs.coutRowPtrE), vgpr(kw.vgprs.coutRowPtrE), \
                                  sgpr("StrideE%s"%strideChar), self.rowInc, tmpS01, "Move coutRowPtrE to next row"))
                    if kw.vgprs.coutRowPtrBias != -1:
                        index = kernel["PackedC1IndicesX"][0] - 1
                        strideW1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (kw.states.indexChars[index]))
                        module.add(self.addScaled(vgpr(kw.vgprs.coutRowPtrBias), vgpr(kw.vgprs.coutRowPtrBias), \
                                  sgpr(strideW1), self.rowInc, tmpS01, "Move coutRowPtrBias to next row"))
                elif len(kernel["PackedC1IndicesX"]) > 1:
                    module.add(kw.extractPackedCoord1ToRowStart(kernel, kernel["PackedC1IndicesX"] , self.coord1Vgpr, 'D'))

        # Shift Pointer for MFMA:
        #   For MFMA shift pointer, correct data is stored in another thread.
        #   Therefore, MFMA cannot use v_mov to amend store data
        #   It needs to modify the coord1 of thread directly.
        # (Use ShiftVectorComponentsMFMA now)
        if 0 and (not kernel["SourceSwap"]) and (not kernel["GuaranteeNoPartialB"]) and tPB["rtv"] and kernel["EnableMatrixInstruction"] and edge:
            (d1,d0,vc1,vc0) = self.element
            if (d1 == vc1 == d0 == vc0 == 0) or self.newCoord1:
                sgprCnt = self.kernelWriter.states.laneSGPRCount
                waveSize = kernel["WavefrontSize"]
                packedC1 = kernel["PackedC1IndicesX"]
                strideC1 = "StrideC%s" % (kw.states.indexChars[packedC1[0]])
                strideD1 = "StrideD%s" % (kw.states.indexChars[packedC1[0]])
                strideE1 = "StrideE%s" % (kw.states.indexChars[packedC1[0]])
                index = packedC1[0] - 1
                strideW1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (kw.states.indexChars[index]))

                module.addComment1("shift vector components d1")
                vw = kernel["GlobalReadVectorWidthB"]
                vTmp1 = tmpVgpr
                vTmp2 = tmpVgpr+1
                sTmp1 = tmpS01
                sTmp2 = tmpS01+sgprCnt
                # check conditions
                module.add(VBfiB32(dst=vgpr(vTmp1), src0=vw-1, src1=0, src2=vgpr(self.coord1Vgpr), comment="coord1 & ~(vw-1)"))
                module.add(VBfiB32(dst=vgpr(vTmp2), src0=vw-1, src1=0, src2=sgpr("SizesFree+%u"%tPB["idx"]), comment="sizeFree & ~(vw-1)"))
                module.add(VCmpEQU32(dst=sgpr(sTmp1,sgprCnt), src0=vgpr(vTmp1), src1=vgpr(vTmp2), comment="if coord1 is in edge glvw"))
                module.add(VAndB32(dst=vgpr(vTmp2), src0=sgpr("SizesFree+%u"%tPB["idx"]), src1=vw-1, comment="sizeFree mod VW"))
                module.add(VCmpGtU32(dst=sgpr(sTmp2,sgprCnt), src0=vgpr(vTmp2), src1=0, comment="this problem is not multiple size of glvw"))
                SAndBX = SAndB64 if waveSize == 64 else SAndB32
                module.add(SAndBX(dst=sgpr(sTmp1,sgprCnt), src0=sgpr(sTmp1,sgprCnt), src1=sgpr(sTmp2,sgprCnt), comment="AND both conditions"))
                # calculate new coord
                module.add(VAddU32(dst=vgpr(vTmp1), src0=vgpr(self.coord1Vgpr), src1=vgpr(vTmp2), comment="shift coord1"))
                module.add(VBfiB32(dst=vgpr(vTmp1), src0=vw-1, src1=vgpr(vTmp1), src2=sgpr("SizesFree+%u"%tPB["idx"]), comment="new coord1 = (shift coord1 & (vw-1)) |  (sizeFree & ~(vw-1))"))
                module.add(VSubI32(dst=vgpr(vTmp2), src0=vgpr(vTmp1), src1=vgpr(self.coord1Vgpr), comment="shift how many column"))
                module.add(VCndMaskB32(dst=vgpr(self.coord1Vgpr), src0=vgpr(self.coord1Vgpr), src1=vgpr(vTmp1), \
                              src2=sgpr(sTmp1,sgprCnt), comment="set new coord1 if meet conditions" ))

                module.add(VMadI32I24(dst=vgpr(vTmp1), src0=sgpr(strideC1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.cinRowPtr), \
                             comment="new rowStart address += shift column * StridesC"))
                module.add(VCndMaskB32(dst=vgpr(kw.vgprs.cinRowPtr), src0=vgpr(kw.vgprs.cinRowPtr), src1=vgpr(vTmp1), src2=sgpr(sTmp1,sgprCnt), \
                             comment="set new rowStart if meet conditions" ))
                module.add(VMadI32I24(dst=vgpr(vTmp1), src0=sgpr(strideD1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.coutRowPtrD), \
                             comment="new rowStart address += shift column * StridesD"))
                module.add(VCndMaskB32(dst=vgpr(kw.vgprs.coutRowPtrD), src0=vgpr(kw.vgprs.coutRowPtrD), src1=vgpr(vTmp1), src2=sgpr(sTmp1,sgprCnt), \
                             comment="set new rowStart if meet conditions" ))
                if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1):
                    module.add(VMadI32I24(dst=vgpr(vTmp1), src0=sgpr(strideE1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.coutRowPtrE), \
                             comment="new rowStart address += shift column * StridesE"))
                    module.add(VCndMaskB32(dst=vgpr(kw.vgprs.coutRowPtrE), src0=vgpr(kw.vgprs.coutRowPtrE), src1=vgpr(vTmp1), src2=sgpr(sTmp1,sgprCnt), \
                             comment="set new rowStart if meet conditions" ))
                if kw.vgprs.coutRowPtrBias != -1:
                    module.add(VMadI32I24(dst=vgpr(vTmp1), src0=sgpr(strideW1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.coutRowPtrBias), \
                             comment="new rowStart address += shift column * StridesW"))
                    module.add(VCndMaskB32(dst=vgpr(kw.vgprs.coutRowPtrBias), src0=vgpr(kw.vgprs.coutRowPtrBias), src1=vgpr(vTmp1), src2=sgpr(sTmp1,sgprCnt), \
                             comment="set new rowStart if meet conditions" ))

                if kernel["StoreRemapVectorWidth"]:
                    ldsPad = max(kernel["StoreRemapVectorWidth"],kernel["MIOutputVectorWidth"])
                    module.add(VMovB32(dst=vgpr(vTmp1), src=hex((kernel["MacroTile0"]+ldsPad)*kw.states.bpeCexternal), \
                                comment="lds byte stride = (MT0 + PAD) * bpe"))
                    module.add(VMadI32I24(dst=vgpr(vTmp1), src0=vgpr(vTmp1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.storeRemapLW), \
                                comment="new lds write address += shift column * Lds byte Stride"))
                    module.add(VCndMaskB32(dst=vgpr(kw.vgprs.storeRemapLW), src0=vgpr(kw.vgprs.storeRemapLW), src1=vgpr(vTmp1), \
                                  src2=sgpr(sTmp1,sgprCnt), comment="set new rowStart if meet conditions" ))
                    if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1):
                        printExit("Output E does not support StoreRemapVectorWidth")
                    if kw.vgprs.coutRowPtrBias != -1:
                        printExit("Bias reduction does not support StoreRemapVectorWidth")
                module.addSpaceLine()

        return module

    def emitLdChange(self, kernel, ss, tc, edge, beta, mask, bufferOOB, singleUpdate, tmpVgpr, tmpSgpr, addrVgpr, BufAddr, dim):
        """
        Generate code for final C read/D write address
        """

        laneSGPRCount = self.kernelWriter.states.laneSGPRCount
        module = Module("emitLdChange")
        isSingleKernel = ((kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1) or kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel") or kernel["StreamK"] > 0
        if kernel["BufferStore"]:
            module.add(self.emitScaleToBpe(kernel, ss, tmpVgpr, tmpSgpr, singleUpdate, tc, dim))
            if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and (beta or kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel"))) and \
                (tc != 'ScaleAlphaVec' and (not (tc == 'Bias' and self.kernelWriter.states.useBias == DataDirection.READ)) and tc != 'ScaleAVec' and tc != 'ScaleBVec'):
                module.add(VCndMaskB32(dst=vgpr(addrVgpr), src0=vgpr(bufferOOB), src1=vgpr(addrVgpr), \
                               src2=sgpr(mask,laneSGPRCount), comment="LD%s clip if OOB. offset" % tc ))
            # dot2: for every NumWaveSplitK contiguous threads, only the rightmost (with the largest threadIdx) one need to write value to D matrix.
            if kernel["NumWaveSplitK"] > 1 and (tc == 'D' or tc == 'TD'):
                kidx = bufferOOB-1
                module.add(VAndB32(dst=vgpr(kidx), src0=(kernel["NumWaveSplitK"]-1), src1=vgpr("Serial"), comment="kidx = Serial % NumWaveSplitK"))
                module.add(VCmpXNeU32(dst=EXEC(), src0=(kernel["NumWaveSplitK"]-1), src1=vgpr(kidx), comment="kidx == NumWaveSplitK-1" ))
                module.add(VMovB32(dst=vgpr(addrVgpr), src=vgpr(bufferOOB), comment="LD%s clip if kidx != NumWaveSplitK-1." % tc))
                module.add(SMovB64(dst=EXEC(), src=-1, comment="Reset exec mask"))

        else:
            if tc == 'Bias' and kernel["ProblemType"]["UseBias"] and isSingleKernel:
                module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                        shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                        src=vgpr(self.addrBiasVgpr), \
                                        comment="Bias address scaled by BPE"))
                if kernel["LdsOffsetBias"] != 0:
                    module.add(VAddU32(dst=vgpr(self.addrBiasVgpr), \
                                       src0=(kernel["LdsOffsetBias"]), \
                                       src1=vgpr(self.addrBiasVgpr), \
                                       comment="add lds offset"))
            elif tc == 'ScaleAlphaVec' and kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
                if self.addrScaleAlphaVecVgpr:
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile%u"%dim], src1=sgpr("WorkGroup%u"%dim), comment="wgp%u * MT%u"%(dim, dim)))
                    coordVgpr = self.coord0Vgpr if dim == 0 else self.coord1Vgpr
                    module.add(VSubU32(dst=vgpr(self.addrScaleAlphaVecVgpr), src0=vgpr(coordVgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                            shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                            src=vgpr(self.addrScaleAlphaVecVgpr), \
                                            comment="ScaleAlpha address scaled by BPE"))
                    if kernel["LdsOffsetBias"] != 0:
                        module.add(VAddU32(dst=vgpr(self.addrScaleAlphaVecVgpr), \
                                           src0=(kernel["LdsOffsetBias"]), \
                                           src1=vgpr(self.addrScaleAlphaVecVgpr), \
                                           comment="add lds offset"))
            elif tc == 'ScaleA' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                if self.addrScaleAVecVgpr:
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                    module.add(VSubU32(dst=vgpr(self.addrScaleAVecVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleAVecVgpr), \
                                            shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                            src=vgpr(self.addrScaleAVecVgpr), \
                                            comment="ScaleAVec address scaled by BPE"))
                if kernel["LdsOffsetBias"] != 0:
                    module.add(VAddU32(dst=vgpr(self.addrScaleAVecVgpr), \
                                       src0=(kernel["LdsOffsetBias"]), \
                                       src1=vgpr(self.addrScaleAVecVgpr), \
                                       comment="add lds offset"))
            elif tc == 'ScaleBVec' and (kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
                if self.addrScaleBVecVgpr:
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr("WorkGroup1"), comment="wgp1 * MT1"))
                    module.add(VSubU32(dst=vgpr(self.addrScaleBVecVgpr), src0=vgpr(self.coord1Vgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleBVecVgpr), \
                                            shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                            src=vgpr(self.addrScaleBVecVgpr), \
                                            comment="ScaleBVec address scaled by BPE"))
                if kernel["LdsOffsetBias"] != 0:
                    module.add(VAddU32(dst=vgpr(self.addrScaleBVecVgpr), \
                                       src0=(kernel["LdsOffsetBias"]), \
                                       src1=vgpr(self.addrScaleBVecVgpr), \
                                       comment="add lds offset"))
            else:
                # store a copy of the offset in 2 of the tmpVgpr for D
                module.add(VAddCOU32(dst=vgpr(addrVgpr+0), dst1=VCC(), src0=vgpr(BufAddr+0), src1=vgpr(tmpVgpr+2), \
                            comment="addrVgpr = C(D) + index*bytes (lo)" ))
                module.add(VAddCCOU32(dst=vgpr(addrVgpr+1), dst1=VCC(), src0=vgpr(BufAddr+1), src1=vgpr(tmpVgpr+3), \
                            src2=VCC(), comment="addrVgpr = C(D) + index*bytes (hi)"))
        return module

    @staticmethod
    def incrementSrdMultipleRows(srcDstBaseSgpr: str, strideSgpr: str, tmpSgpr: str, numRows: int, bpe: int) -> Module:
        module = Module("incrementToNextRows")
        if numRows > 1:
            module.add(SMulI32(dst=sgpr(tmpSgpr), \
                               src0=sgpr(strideSgpr), \
                               src1=numRows*bpe, \
                               comment="scale %s *= numRows(%u) * bpe"%(strideSgpr, numRows)))
        elif numRows < 0:
            module.add(SMulI32(dst=sgpr(tmpSgpr), \
                               src0=sgpr(strideSgpr), \
                                src1=(-numRows)*bpe, \
                                comment="scale %s *= numRows(%u) * bpe"%(strideSgpr, numRows)))
        else:
            module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), \
                                      src=sgpr(strideSgpr), \
                                      shiftHex=log2(bpe), \
                                      comment="incToNextRow: Scale by BPE"))
        dstLow = f"{srcDstBaseSgpr}+0"
        dstHigh = f"{srcDstBaseSgpr}+1"

        if numRows >= 0:
            module.add(SAddU32(dst=sgpr(dstLow), \
                                        src0=sgpr(dstLow), \
                                        src1=sgpr(tmpSgpr), \
                                        comment="incToNextRow: gra SRD += inc(lower)" ))
            module.add(SAddCU32(dst=sgpr(dstHigh), \
                                        src0=sgpr(dstHigh), \
                                        src1=0, \
                                        comment="incToNextRow: gra SRD += inc(upper)" ))
        else:
            module.add(SSubU32(dst=sgpr(dstLow), \
                                        src0=sgpr(dstLow), \
                                        src1=sgpr(tmpSgpr), \
                                        comment="incToNextRow: gra SRD -= inc(lower)" ))
            module.add(SSubBU32(dst=sgpr(dstHigh), \
                                        src0=sgpr(dstHigh), \
                                        src1=0, \
                                        comment="incToNextRow: gra SRD -= inc(upper)" ))
        return module

    def incrementToNextRow(self, kernel, tc, ss, stmp, bpeType=None, dst=-1):
        """
        Generate code to move to the next row(s)
        If optSrdIncForRow, this will move the SRD forward
        If not, this could generate some other instructions
        """

        module = Module("incrementToNextRow")
        numRows = self.rowInc
        tmpBpe = bpeType if bpeType else self.kernelWriter.states.bpeCexternal
        if (tc == 'C' or tc == 'TD') and (kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel"):
            tmpBpe = int(self.kernelWriter.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())
        if ss.optSrdIncForRow:
            if numRows:
                packedC1 = kernel["PackedC1IndicesX"]
                assert(len(packedC1) == 1)  # would need to extract each dim and scale
                if tc == 'Bias' and (not kernel["WorkGroupReduction"]):
                    index = packedC1[0] - 1
                    strideCD1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (self.kernelWriter.states.indexChars[index]))
                else:
                    td = "D" if tc == 'TD' else tc
                    strideCD1 = "Stride%s%s"%(td ,self.kernelWriter.states.indexChars[packedC1[0]])
                if numRows > 1:
                    module.add(SMulI32(dst=sgpr(stmp), \
                                src0=sgpr(strideCD1), \
                                src1=numRows*tmpBpe, \
                                comment="scale Stride%s *= numRows(%u) * bpe"%(tc,numRows)))
                elif numRows < 0:
                    module.add(SMulI32(dst=sgpr(stmp), \
                                src0=sgpr(strideCD1), \
                                src1=(-numRows)*tmpBpe, \
                                comment="scale Stride%s *= numRows(%u) * bpe"%(tc,numRows)))
                else:
                    module.add(SLShiftLeftB32(dst=sgpr(stmp), \
                                src=sgpr(strideCD1), \
                                shiftHex=log2(tmpBpe), \
                                comment="incToNextRow: Scale by BPE"))

                if dst == -1:
                    dstLow = "Srd%s+0"%(tc)
                    dstHigh = "Srd%s+1"%(tc)
                else:
                    dstLow = dst+0
                    dstHigh = dst+1

                if numRows >= 0:
                    module.add(SAddU32(dst=sgpr(dstLow), \
                                        src0=sgpr(dstLow), \
                                        src1=sgpr(stmp), \
                                        comment="incToNextRow: gra SRD += inc(lower)" ))
                    module.add(SAddCU32(dst=sgpr(dstHigh), \
                                        src0=sgpr(dstHigh), \
                                        src1=0, \
                                        comment="incToNextRow: gra SRD += inc(upper)" ))
                else: # numRows < 0
                    module.add(SSubU32(dst=sgpr(dstLow), \
                                        src0=sgpr(dstLow), \
                                        src1=sgpr(stmp), \
                                        comment="incToNextRow: gra SRD -= inc(lower)" ))
                    module.add(SSubBU32(dst=sgpr(dstHigh), \
                                        src0=sgpr(dstHigh), \
                                        src1=0, \
                                        comment="incToNextRow: gra SRD -= inc(upper)" ))

        return module
