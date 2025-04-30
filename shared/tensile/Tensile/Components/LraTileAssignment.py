################################################################################
#
# Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import LraTileAssignment
from ..AsmUtils import inst, vgpr, sgpr, vectorStaticDivideAndRemainder, vectorStaticDivide, staticMultiply, vectorStaticRemainder, instCommentOnly

class LraTileAssignmentVALU(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": False}

    """
    Local Read Addresses: Tile Assignment
    """
    def __call__(self, writer, kernel, tP):
        kStr = ""

        # allocate resources
        qReg    = writer.vgprPool.checkOut(1,"qReg") # quotient
        rReg    = writer.vgprPool.checkOut(1,"rReg") # remainder
        tmpSgpr = writer.getTmpSgpr(1).idx()

        if tP["tileIdx"] == 0:
            kStr += "%slr%s = serial %% SG%s%s%s" \
                    % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                    writer.commentSuffix, writer.endLine)

            # constant
            dividendReg = "Serial" # local serial
            divisor = kernel["SubGroup0"]

            # generate instruction
            kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpSgpr)

            # release and return resource
            tP["gpr"]["lro"] = rReg
            writer.tmplro = qReg
        else:
            kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
                    % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                    tP["tileChar"], writer.commentSuffix, writer.endLine)

            # constant
            divisor = kernel["SubGroup1"]
            dividendReg = writer.tmplro

            # generate instruction
            kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpSgpr)

            # release and return resource
            tP["gpr"]["lro"] = rReg

            writer.vgprPool.checkIn(writer.tmplro) # old
            writer.vgprPool.checkIn(qReg)

        return kStr


class LraTileAssignmentMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read Addresses: Tile Assignment A/B
    """
    def __call__(self, writer, kernel, tP):
        kStr = ""

        kStr += "%slr%s%s%s" \
                % (writer.commentPrefix, tP["tileChar"], writer.commentSuffix, writer.endLine)

        isMfma = writer.asmCaps["HasMFMA"]

        # get constant parameter
        tc               = tP["tensorChar"]
        umlds            = kernel["UnrollMajorLDS%s" % tc]

        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        dtlTsgr = kernel["DirectToLds%s" % tc] and kernel["ThreadSeparateGlobalRead%c"%tc] and umlds
        if dtlTsgr:
          mReg    = writer.vgprPool.checkOut(1,"mReg")

         # alloc sgpr
        tmpSgpr = writer.getTmpSgpr(1).idx()

        # get constant parameter
        tile01           = tP["tile01Idx"]
        waveWidth        = kernel["WavefrontSize"]
        inputPerThreadA  = writer.lrvwA if kernel["UnrollMajorLDSA"] else kernel["MIInputPerThread"]
        inputPerThreadB  = writer.lrvwB if kernel["UnrollMajorLDSB"] else kernel["MIInputPerThread"]
        inputPerThread   = max(inputPerThreadA, inputPerThreadB)
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

        # strider for each type of index
        mt               = kernel["MacroTile%u" % tile01]
        strideTile       = kernel["_DepthULds"] + LdsPad if umlds else 1
        strideK          = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        strideBlock      = kernel["MatrixInstM"] * strideTile
        strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        # tile offset
        kStr += vectorStaticRemainder(kReg, "Serial", waveWidth, tmpSgpr, \
            "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth)
        kStr += vectorStaticRemainder(tReg, kReg, kernel["MatrixInstN"], tmpSgpr, \
            "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"])
        if dtlTsgr:
          tmpVgpr = writer.vgprPool.checkOut(1,"tmpVgpr")
          # WSGR splits global fetch 2D tile MblockxdepthU into (WSPR *2)xMblockxdepthU/(WSPR*2)  (Mblock = waveWidth * glvw  / depthU)
          # LDS layout stored as 3D tile K1xMblockxK0
          # Padding is not allowed in directToLds
          NblockSizePerLoad = int(waveWidth * kernel["GlobalLoadVectorWidth%c"%tc]) // kernel["_DepthULds"] // vectorWidth
          # Nidx offset calculation
          # each load fetches tuple<K1,Nidx,K0> mapped to wavefront load tuple<TSGR<<1,wavefront/depthU/TSGR<<1, depth//TSGR<<1)
          kStr += vectorStaticDivide(mReg, tReg, NblockSizePerLoad, tmpSgpr, \
              "1. N offset: nIdx_upper = nIdx / NblockSizePerLoad(%u)" % NblockSizePerLoad)
          kStr += staticMultiply(vgpr(mReg), vgpr(mReg), NblockSizePerLoad*kernel["_DepthULds"], sgpr(tmpSgpr), \
              "1. N offset: nIdx_upper_offset = nIdx_upper * nStride(%u)" % (NblockSizePerLoad*kernel["_DepthULds"]))
          KelementsPerMFrag = kernel["_DepthULds"]//(kernel["ThreadSeparateGlobalRead%c"%tc]*2)
          kStr += vectorStaticRemainder(tmpVgpr, tReg, NblockSizePerLoad, tmpSgpr, \
              "1. N offset: nIdx_lower = nIdx %% NblockSizePerLoad(%u)" % NblockSizePerLoad)
          kStr += staticMultiply(vgpr(tReg), vgpr(tmpVgpr), KelementsPerMFrag, sgpr(tmpSgpr), \
              "1. N offset: nIdx_lower = nIdx_load * nStride(%u)" % KelementsPerMFrag)

          kStr += inst("_v_add_u32", vgpr(tReg), vgpr(mReg), vgpr(tReg), \
              "1. N offset: nOffset =  nIdx_lower + nIdx_upper")
          writer.vgprPool.checkIn(tmpVgpr)
        else:
          kStr += staticMultiply(vgpr(tReg), vgpr(tReg), strideTile, sgpr(tmpSgpr), \
            "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile)
        # block offset
        if num1DBlocks > 1:
            tmpVgpr = writer.vgprPool.checkOut(1,"tmpVgpr")
            # generate the code only when num1DBlocks > 1.
            # if num1DBlocks is 1, % num1DBlocks is always 0 and no difference in tReg value
            kStr += vectorStaticDivide(tmpVgpr, kReg, dividedForBlkId, tmpSgpr, \
                "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId)
            kStr += vectorStaticRemainder(tmpVgpr, tmpVgpr, num1DBlocks, tmpSgpr, \
                "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks)  # assuming num1DBlocks is power of 2 to use same vreg for src and dst
            kStr += staticMultiply(vgpr(tmpVgpr), vgpr(tmpVgpr), strideBlock, sgpr(tmpSgpr), \
                "2. block offset: bnOffset = bnIdx * strideBlock(%u)" % strideBlock)
            kStr += inst("_v_add_u32", vgpr(tReg), vgpr(tmpVgpr), vgpr(tReg), \
                "3. add N and block offset: bnOffset = block and N offset")
            writer.vgprPool.checkIn(tmpVgpr)
        else:
            # comment only because bnIdx = bnIdx % num1DBlocks(1) = 0
            kStr += instCommentOnly("2. block offset: bnIdx = bnIdx %% num1DBlocks(%u) is 0. do nothing" % num1DBlocks)
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, sgpr(tmpSgpr), \
            "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth)

        # unroll offset
        if isMfma and (dividendForKId != waveWidth):
            kStr += vectorStaticDivide(kReg, kReg, dividendForKId, tmpSgpr, \
                "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"]))
            if dtlTsgr:
              # ThreadSeparateGlobalRead + DirectToLds case
              # kIdx_lower = (kIdx % KelementsPerMFrag) + (kIdx // KelementsPerMFrag) * (KelementsPerMFrag * NblockSizePerLoad)
              # Here, KelementsPerMFrag needs to be divided by inputPerThread.
              # inputPerThread will be multiplied later
              KelementsPerMFrag //= inputPerThread
              kStr += vectorStaticRemainder(mReg, kReg, KelementsPerMFrag, tmpSgpr, \
                  "5. K offset: kIdx_lower = kIdx %% KelementsPerMFrag(%u)" % (KelementsPerMFrag))
              kStr += vectorStaticDivide(kReg, kReg, KelementsPerMFrag, tmpSgpr, \
                  "5. K offset: kIdx_higher = kIdx / KelementsPerMFrag(%u)" % (KelementsPerMFrag))
              kStr += staticMultiply(vgpr(kReg), vgpr(kReg), (KelementsPerMFrag * NblockSizePerLoad * vectorWidth), sgpr(tmpSgpr), \
                  "5. K offset: kIdx_higher = kIdx_higher * mStride(%u)" % (KelementsPerMFrag * NblockSizePerLoad * vectorWidth))
              kStr += inst("_v_add_u32", vgpr(kReg), vgpr(kReg), vgpr(mReg), \
                  "5. K offset: kIdx = kIdx_lower + kIdx_higher")
            kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideK, sgpr(tmpSgpr), \
                "5. K offset: lrKOffset = kIdx * mStride(%u)" % strideK)

            kStr += inst("_v_add_u32", vgpr(tReg), vgpr(kReg), vgpr(tReg), \
                "6. offset in wave: lrOffset = bnOffset + lrKOffset")

        # wave offset
        if num1DWaves > 1:
            tmpVgpr = writer.vgprPool.checkOut(1,"tmpVgpr")
            kStr += vectorStaticDivide(tmpVgpr, "Serial", dividedForWaveId, tmpSgpr, \
                "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId)
            kStr += vectorStaticRemainder(kReg, tmpVgpr, num1DWaves, tmpSgpr, \
                "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves)
            kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideWave, sgpr(tmpSgpr), \
                "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u)" % strideWave)
            kStr += inst("_v_add_u32", vgpr(tReg), vgpr(kReg), vgpr(tReg), \
                "8. final local read offset: flrOffset = lrOffset + WOffset")
            writer.vgprPool.checkIn(tmpVgpr)

        # localSplitU case. Calculate LSU offset here
        if kernel["LocalSplitU"] > 1:
            subGroup  = kernel["SubGroup0"] * kernel["SubGroup1"]
            kInterval = waveWidth // dividendForKId
            # generate instruction
            # (use kreg to calculate sgid)
            kStr += vectorStaticDivide(kReg, "Serial", subGroup, tmpSgpr, \
              "LSU offset: sgid = Serial / subGroup(%u)" % subGroup)
            kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideK * kInterval, sgpr(tmpSgpr), \
              "LSU offset: lsuOffset = sgid * kInterval(%u) *  strideK(%u)" % (kInterval, strideK))
            kStr += inst("_v_add_u32", vgpr(tReg), vgpr(kReg), vgpr(tReg), \
              "LSU offset: lrOffset += lsuOffset")

        # release register
        tP["gpr"]["lro"] = tReg
        writer.vgprPool.checkIn(kReg)
        if dtlTsgr:
          writer.vgprPool.checkIn(mReg)

        return kStr
