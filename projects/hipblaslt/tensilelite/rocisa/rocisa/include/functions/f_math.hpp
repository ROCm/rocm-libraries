/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/extension.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <string>

namespace rocisa
{
    ///////////////////////////////////////
    // Divide & Remainder
    // quotient register, remainder register, dividend register, divisor, tmpVgprx2
    ///////////////////////////////////////

    template <typename QREG, typename DREG>
    std::shared_ptr<Module>
        vectorStaticDivideAndRemainder(QREG                              qReg,
                                       int                               rReg,
                                       DREG                              dReg,
                                       int                               divisor,
                                       std::optional<ContinuousRegister> tmpVgprRes,
                                       bool                              doRemainder = true,
                                       const std::string&                comment     = "")
    {
        auto qRegStr = [&qReg]() {
            if constexpr(std::is_same_v<QREG, int>)
            {
                return std::to_string(qReg);
            }
            else
            {
                return qReg;
            }
        }();
        auto dRegStr = [&dReg]() {
            if constexpr(std::is_same_v<DREG, int>)
            {
                return std::to_string(dReg);
            }
            else
            {
                return dReg;
            }
        }();
        std::string dComment = comment.empty()
                                   ? qRegStr + " = " + dRegStr + " / " + std::to_string(divisor)
                                   : comment;
        std::string rComment = comment.empty() ? std::to_string(rReg) + " = " + dRegStr + " % "
                                                     + std::to_string(divisor)
                                               : comment;

        auto module = std::make_shared<Module>("vectorStaticDivideAndRemainder");

        auto qRegVgpr = vgpr(qReg);
        auto rRegVgpr = vgpr(rReg);
        auto dRegVgpr = vgpr(dReg);

        if((divisor & (divisor - 1)) == 0)
        { // power of 2
            int divisor_log2 = static_cast<int>(std::log2(divisor));
            module->addT<VLShiftRightB32>(qRegVgpr, divisor_log2, dRegVgpr, dComment);
            if(doRemainder)
            {
                module->addT<VAndB32>(rRegVgpr, divisor - 1, dRegVgpr, rComment);
            }
        }
        else
        {
            if(!tmpVgprRes || tmpVgprRes->size < 2)
            {
                throw std::runtime_error("Invalid tmpVgprRes, must be at least 2");
            }
            int  tmpVgprIdx = tmpVgprRes->idx;
            auto tmpVgpr    = vgpr(tmpVgprIdx);
            auto tmpVgpr1   = vgpr(tmpVgprIdx + 1);

            int shift = 32 + 1;
            int magic = ((1ULL << shift) / divisor) + 1;
            /*
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        */
            if(magic <= 64 && magic >= -16)
            {
                module->addT<VMulHIU32>(tmpVgpr1, dRegVgpr, magic, dComment);
                module->addT<VMulLOU32>(tmpVgpr, dRegVgpr, magic, dComment);
            }
            else
            {
                module->addT<VMovB32>(tmpVgpr, magic);
                module->addT<VMulHIU32>(tmpVgpr1, dRegVgpr, tmpVgpr, dComment);
                module->addT<VMulLOU32>(tmpVgpr, dRegVgpr, tmpVgpr, dComment);
            }

            module->addT<VLShiftRightB64>(
                vgpr(tmpVgprIdx, 2), shift, vgpr(tmpVgprIdx, 2), dComment);
            module->addT<VMovB32>(qRegVgpr, tmpVgpr, std::nullopt, dComment);

            if(doRemainder)
            {
                if(divisor <= 64 && divisor >= -16)
                {
                    module->addT<VMulLOU32>(tmpVgpr, qRegVgpr, divisor, rComment);
                }
                else
                {
                    module->addT<VMovB32>(tmpVgpr1, divisor);
                    module->addT<VMulLOU32>(tmpVgpr, qRegVgpr, tmpVgpr1, rComment);
                }
                module->addT<VSubU32>(rRegVgpr, dRegVgpr, tmpVgpr, rComment);
            }
        }

        return module;
    }

    template <typename QREG, typename DREG>
    std::shared_ptr<Module> vectorStaticDivide(QREG                              qReg,
                                               DREG                              dReg,
                                               int                               divisor,
                                               std::optional<ContinuousRegister> tmpVgprRes,
                                               const std::string&                comment = "")
    {
        int  rReg = -1; // unused
        auto module
            = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgprRes, false, comment);
        module->name = "vectorStaticDivide (reg=-1)";
        return module;
    }

    inline std::shared_ptr<Module> vectorUInt32DivideAndRemainder(int  qReg,
                                                                  int  dReg,
                                                                  int  divReg,
                                                                  int  rReg,
                                                                  bool doRemainder           = true,
                                                                  const std::string& comment = "")
    {
        auto qRegVgpr   = vgpr(qReg);
        auto rRegVgpr   = vgpr(rReg);
        auto dRegVgpr   = vgpr(dReg);
        auto divRegVgpr = vgpr(divReg);

        std::string dComment = comment.empty() ? qRegVgpr->toString() + " = " + dRegVgpr->toString()
                                                     + " / " + divRegVgpr->toString()
                                               : comment;
        std::string rComment = comment.empty() ? rRegVgpr->toString() + " = " + dRegVgpr->toString()
                                                     + " % " + divRegVgpr->toString()
                                               : comment;

        auto pEXEC = MAKE(EXEC);

        auto module = std::make_shared<Module>("vectorUInt32DivideAndRemainder");
        module->addT<VCvtU32toF32>(qRegVgpr, divRegVgpr, std::nullopt, dComment);
        module->addT<VRcpIFlagF32>(qRegVgpr, qRegVgpr, dComment);
        module->addT<VCvtU32toF32>(rRegVgpr, dRegVgpr, std::nullopt, dComment);
        module->addT<VMulF32>(qRegVgpr, qRegVgpr, rRegVgpr, std::nullopt, dComment);
        module->addT<VCvtF32toU32>(qRegVgpr, qRegVgpr, std::nullopt, dComment);
        module->addT<VMulU32U24>(rRegVgpr, qRegVgpr, divRegVgpr, dComment);
        module->addT<VSubU32>(rRegVgpr, dRegVgpr, rRegVgpr, dComment);
        module->addT<VCmpXEqU32>(pEXEC, rRegVgpr, divRegVgpr, std::nullopt, dComment);
        module->addT<VAddU32>(qRegVgpr, 1, qRegVgpr, dComment);
        if(doRemainder)
        {
            module->addT<VMovB32>(rRegVgpr, 0, std::nullopt, rComment);
        }
        module->addT<SMovB64>(pEXEC, -1, dComment);
        return module;
    }

    inline std::shared_ptr<Module> vectorUInt32CeilDivideAndRemainder(int  qReg,
                                                                      int  dReg,
                                                                      int  divReg,
                                                                      int  rReg,
                                                                      bool doRemainder = true,
                                                                      const std::string& comment
                                                                      = "")
    {
        auto qRegVgpr   = vgpr(qReg);
        auto rRegVgpr   = vgpr(rReg);
        auto dRegVgpr   = vgpr(dReg);
        auto divRegVgpr = vgpr(divReg);

        std::string dComment = comment.empty()
                                   ? qRegVgpr->toString() + " = ceil(" + dRegVgpr->toString()
                                         + " / " + divRegVgpr->toString() + ")"
                                   : comment;
        std::string rComment = comment.empty() ? rRegVgpr->toString() + " = " + dRegVgpr->toString()
                                                     + " % " + divRegVgpr->toString()
                                               : comment;

        auto pVCC  = MAKE(VCC);
        auto pEXEC = MAKE(EXEC);

        auto module = std::make_shared<Module>("vectorUInt32CeilDivideAndRemainder");
        module->addT<VCvtU32toF32>(qRegVgpr, divRegVgpr, std::nullopt, dComment);
        module->addT<VRcpIFlagF32>(qRegVgpr, qRegVgpr, dComment);
        module->addT<VCvtU32toF32>(rRegVgpr, dRegVgpr, std::nullopt, dComment);
        module->addT<VMulF32>(qRegVgpr, qRegVgpr, rRegVgpr, std::nullopt, dComment);
        module->addT<VCvtF32toU32>(qRegVgpr, qRegVgpr, std::nullopt, dComment);
        module->addT<VMulU32U24>(rRegVgpr, qRegVgpr, divRegVgpr, dComment);
        module->addT<VSubU32>(rRegVgpr, dRegVgpr, rRegVgpr, dComment);
        module->addT<VCmpNeU32>(pVCC, rRegVgpr, 0, std::nullopt, dComment);
        module->addT<VAddCCOU32>(qRegVgpr, pVCC, qRegVgpr, 0, pVCC, "ceil");
        if(doRemainder)
        {
            module->addT<VCmpXEqU32>(pEXEC, rRegVgpr, divRegVgpr, std::nullopt, rComment);
            module->addT<VMovB32>(rRegVgpr, 0, std::nullopt, rComment);
            module->addT<SMovB64>(pEXEC, -1, dComment);
        }
        return module;
    }

    template <typename RREG, typename DREG>
    std::shared_ptr<Module> vectorStaticRemainder(int                               qReg,
                                                  RREG                              rReg,
                                                  DREG                              dReg,
                                                  int                               divisor,
                                                  std::optional<ContinuousRegister> tmpVgprRes,
                                                  std::optional<ContinuousRegister> tmpSgprRes,
                                                  const std::string&                comment = "")
    {
        auto qRegVgpr = vgpr(qReg);
        auto rRegVgpr = vgpr(rReg);
        auto dRegVgpr = vgpr(dReg);

        std::string dComment = comment.empty() ? rRegVgpr->toString() + " = " + dRegVgpr->toString()
                                                     + " % " + std::to_string(divisor)
                                               : comment;

        auto module = std::make_shared<Module>("vectorStaticRemainder");

        if((divisor & (divisor - 1)) == 0)
        { // power of 2
            module->addT<VAndB32>(rRegVgpr, divisor - 1, dRegVgpr, dComment);
        }
        else
        {
            if(!tmpVgprRes || tmpVgprRes->size < 2)
            {
                throw std::runtime_error("Invalid tmpVgprRes, must be at least 2");
            }
            int  tmpVgprIdx = tmpVgprRes->idx;
            auto tmpVgpr    = vgpr(tmpVgprIdx);
            auto tmpVgpr1   = vgpr(tmpVgprIdx + 1);

            if(!tmpSgprRes || tmpSgprRes->size < 1)
            {
                throw std::runtime_error("Invalid tmpSgprRes, must be at least 1");
            }
            int  tmpSgprIdx = tmpSgprRes->idx;
            auto tmoSgpr    = sgpr(tmpSgprIdx);

            int shift = 32 + 1;
            int magic = ((1ULL << shift) / divisor) + 1;

            /*
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        */

            if(magic <= 64 && magic >= -16)
            {
                module->addT<VMulHIU32>(tmpVgpr1, dRegVgpr, magic, dComment);
                module->addT<VMulLOU32>(tmpVgpr, dRegVgpr, magic, dComment);
            }
            else
            {
                module->addT<SMovB32>(tmoSgpr, magic, dComment);
                module->addT<VMulHIU32>(tmpVgpr1, dRegVgpr, tmoSgpr, dComment);
                module->addT<VMulLOU32>(tmpVgpr, dRegVgpr, tmoSgpr, dComment);
            }

            module->addT<VLShiftRightB64>(
                vgpr(tmpVgprIdx, 2), shift, vgpr(tmpVgprIdx, 2), dComment);
            module->addT<VMovB32>(qRegVgpr, tmpVgpr, std::nullopt, dComment);

            if(divisor <= 64 && divisor >= -16)
            {
                module->addT<VMulLOU32>(tmpVgpr, qRegVgpr, divisor, dComment);
            }
            else
            {
                module->addT<SMovB32>(tmoSgpr, divisor, dComment);
                module->addT<VMulLOU32>(tmpVgpr, qRegVgpr, tmoSgpr, dComment);
            }

            module->addT<VSubU32>(rRegVgpr, dRegVgpr, tmpVgpr, dComment);
        }

        return module;
    }

    // only used for loop unroll and GlobalSplitU
    // doRemainder==0 : compute quotient only
    // doRemainder==1 : compute quotient and remainder
    // doRemainder==2 : only compute remainder (not quotient unless required for remainder)
    // dreg == dividend
    // tmpSgpr must be 2 SPGRs
    // qReg and dReg can be "sgpr[..]" or names of sgpr (will call sgpr)
    template <typename QREG, typename RREG, typename DREG>
    std::shared_ptr<Module>
        scalarStaticDivideAndRemainder(QREG                              qReg,
                                       RREG                              rReg,
                                       DREG                              dReg,
                                       int                               divisor,
                                       std::optional<ContinuousRegister> tmpSgprRes,
                                       int                               doRemainder = 1)
    {
        auto qRegSgpr = [&qReg]() {
            if constexpr(std::is_same_v<QREG, int> || std::is_same_v<QREG, std::string>)
            {
                return sgpr(qReg);
            }
            else
            {
                return qReg;
            }
        }();

        auto rRegSgpr = [&rReg]() {
            if constexpr(std::is_same_v<RREG, int> || std::is_same_v<RREG, std::string>)
            {
                return sgpr(rReg);
            }
            else
            {
                return rReg;
            }
        }();

        auto dRegSgpr = [&dReg]() {
            if constexpr(std::is_same_v<DREG, int> || std::is_same_v<DREG, std::string>)
            {
                return sgpr(dReg);
            }
            else
            {
                return dReg;
            }
        }();

        auto module = std::make_shared<Module>("scalarStaticDivideAndRemainder");

        if((divisor & (divisor - 1)) == 0)
        { // power of 2
            int divisor_log2 = static_cast<int>(std::log2(divisor));
            if(doRemainder != 2)
            {
                module->addT<SLShiftRightB32>(qRegSgpr,
                                              divisor_log2,
                                              dRegSgpr,
                                              qRegSgpr->toString() + " = " + dRegSgpr->toString()
                                                  + " / " + std::to_string(divisor));
            }
            if(doRemainder)
            {
                module->addT<SAndB32>(rRegSgpr,
                                      divisor - 1,
                                      dRegSgpr,
                                      rRegSgpr->toString() + " = " + dRegSgpr->toString() + " % "
                                          + std::to_string(divisor));
            }
        }
        else
        {
            if(!tmpSgprRes || tmpSgprRes->size < 2)
            {
                throw std::runtime_error("Invalid tmpSgprRes, must be at least 2");
            }
            int  tmpSgprIdx = tmpSgprRes->idx;
            auto tmpSgpr    = sgpr(tmpSgprIdx);
            auto tmpSgpr1   = sgpr(tmpSgprIdx + 1);
            auto tmp2Sgpr   = sgpr(tmpSgprIdx, 2);
            /*
            if divisor == 30:
                shift = 32+2
            elif divisor >= 14:
                shift = 32+4
            elif divisor >= 6:
                shift = 32+3
            elif divisor >= 5:
                shift = 32+2
            elif divisor >= 3:
                shift = 32+1
            */
            int shift   = 32 + 1;
            int magic   = ((1ULL << shift) / divisor) + 1;
            int magicHi = magic >> 16;
            int magicLo = magic & 0xFFFF;

            module->addT<SMovB32>(tmpSgpr1, 0, "STATIC_DIV: divisor=" + std::to_string(divisor));
            module->addT<SMulI32>(tmpSgpr, magicHi, dRegSgpr, "tmp1 = dividend * magic hi");
            module->addT<SLShiftLeftB64>(tmp2Sgpr, 16, tmp2Sgpr, "left shift 16 bits");
            module->addT<SMulI32>(qRegSgpr, dRegSgpr, magicLo, "tmp0 = dividend * magic lo");
            module->addT<SAddU32>(tmpSgpr, qRegSgpr, tmpSgpr, "add lo");
            module->addT<SAddCU32>(tmpSgpr1, tmpSgpr1, 0, "add hi");
            module->addT<SLShiftRightB64>(
                tmp2Sgpr, shift, tmp2Sgpr, "tmp1 = (dividend * magic) << shift");
            module->addT<SMovB32>(qRegSgpr, tmpSgpr, "quotient");

            if(doRemainder)
            {
                module->addT<SMulI32>(tmpSgpr, qRegSgpr, divisor, "quotient*divisor");
                module->addT<SSubU32>(
                    rRegSgpr, dRegSgpr, tmpSgpr, "rReg = dividend - quotient*divisor");
            }
        }

        return module;
    }

    template <typename QREG, typename DREG>
    std::shared_ptr<Module> scalarStaticCeilDivide(QREG                              qReg,
                                                   DREG                              dReg,
                                                   int                               divisor,
                                                   std::optional<ContinuousRegister> tmpSgprRes)
    {
        auto qRegSgpr = [&qReg]() {
            if constexpr(std::is_same_v<QREG, int> || std::is_same_v<QREG, std::string>)
            {
                return sgpr(qReg);
            }
            else
            {
                return qReg;
            }
        }();

        auto dRegSgpr = [&dReg]() {
            if constexpr(std::is_same_v<DREG, int> || std::is_same_v<DREG, std::string>)
            {
                return sgpr(dReg);
            }
            else
            {
                return dReg;
            }
        }();

        auto module = std::make_shared<Module>("scalarStaticCeilDivide");

        if((divisor & (divisor - 1)) == 0)
        { // power of 2
            int divisor_log2 = static_cast<int>(std::log2(divisor));
            module->addT<SLShiftRightB32>(qRegSgpr,
                                          divisor_log2,
                                          dRegSgpr,
                                          qRegSgpr->toString() + " = " + dRegSgpr->toString()
                                              + " / " + std::to_string(divisor));
            auto tmpSgpr = sgpr(tmpSgprRes->idx);
            module->addT<SAndB32>(tmpSgpr,
                                  divisor - 1,
                                  dRegSgpr,
                                  tmpSgpr->toString() + " = " + dRegSgpr->toString() + " % "
                                      + std::to_string(divisor));
            module->addT<SAddCU32>(qRegSgpr, qRegSgpr, 0);
        }
        else
        {
            if(!tmpSgprRes || tmpSgprRes->size < 2)
            {
                throw std::runtime_error("Invalid tmpSgprRes, must be at least 2");
            }
            int  tmpSgprIdx = tmpSgprRes->idx;
            auto tmpSgpr    = sgpr(tmpSgprIdx);
            auto tmpSgpr1   = sgpr(tmpSgprIdx + 1);
            auto tmp2Sgpr   = sgpr(tmpSgprIdx, 2);
            /*
            if divisor == 30:
                shift = 32+2
            elif divisor >= 14:
                shift = 32+4
            elif divisor >= 6:
                shift = 32+3
            elif divisor >= 5:
                shift = 32+2
            elif divisor >= 3:
                shift = 32+1
            */
            int shift   = 32 + 1;
            int magic   = ((1ULL << shift) / divisor) + 1;
            int magicHi = magic >> 16;
            int magicLo = magic & 0xFFFF;

            module->addT<SMovB32>(tmpSgpr1, 0, "STATIC_DIV: divisor=" + std::to_string(divisor));
            module->addT<SMulI32>(tmpSgpr, magicHi, dRegSgpr, "tmp1 = dividend * magic hi");
            module->addT<SLShiftLeftB64>(tmp2Sgpr, 16, tmp2Sgpr, "left shift 16 bits");
            module->addT<SMulI32>(qRegSgpr, dRegSgpr, magicLo, "tmp0 = dividend * magic lo");
            module->addT<SAddU32>(tmpSgpr, qRegSgpr, tmpSgpr, "add lo");
            module->addT<SAddCU32>(tmpSgpr1, tmpSgpr1, 0, "add hi");
            module->addT<SLShiftRightB64>(tmp2Sgpr, shift, tmp2Sgpr, "tmp0 = quotient");
            module->addT<SMulI32>(tmpSgpr1, tmpSgpr, divisor, "tmp1 = quotient * divisor");
            module->addT<SCmpLgU32>(
                tmpSgpr1, dRegSgpr, "if (quotient * divisor != dividend), result+=1");
            module->addT<SAddCU32>(
                qRegSgpr, tmpSgpr, 0, "if (quotient * divisor != dividend), result+=1");
        }

        return module;
    }

    template <typename QREG, typename RREG, typename DREG>
    std::shared_ptr<Module> scalarStaticRemainder(QREG                              qReg,
                                                  RREG                              rReg,
                                                  DREG                              dReg,
                                                  int                               divisor,
                                                  std::optional<ContinuousRegister> tmpSgprRes,
                                                  const std::string&                comment = "")
    {
        auto module = std::make_shared<Module>("scalarStaticRemainder");

        auto rRegSgpr = sgpr(rReg);
        auto dRegSgpr = sgpr(dReg);

        std::string dComment = comment.empty() ? rRegSgpr->toString() + " = " + dRegSgpr->toString()
                                                     + " % " + std::to_string(divisor)
                                               : comment;

        if((divisor & (divisor - 1)) == 0)
        { // power of 2
            module->addT<SAndB32>(rRegSgpr, divisor - 1, dRegSgpr, dComment);
        }
        else
        {
            auto qRegSgpr = sgpr(qReg);

            if(!tmpSgprRes || tmpSgprRes->size < 3)
            {
                throw std::runtime_error("Invalid tmpSgprRes, must be at least 3");
            }
            int  tmpSgprIdx = tmpSgprRes->idx;
            auto tmpSgpr    = sgpr(tmpSgprIdx);
            auto tmpSgpr1   = sgpr(tmpSgprIdx + 1);
            auto tmpSgpr2   = sgpr(tmpSgprIdx + 2);
            auto tmp2Sgpr   = sgpr(tmpSgprIdx, 2);
            /*
            if divisor == 30:
                shift = 32+2
            elif divisor >= 14:
                shift = 32+4
            elif divisor >= 7:
                shift = 32+3
            elif divisor >= 6:
                shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
            elif divisor >= 5:
                shift = 32+2
            elif divisor >= 3:
                shift = 32+1
            */
            int shift = 32 + 1;
            int magic = ((1ULL << shift) / divisor) + 1;

            if(magic <= 64 && magic >= -16)
            {
                module->addT<SMulHIU32>(tmpSgpr1, dRegSgpr, magic, dComment);
                module->addT<SMulI32>(tmpSgpr, dRegSgpr, magic, dComment);
            }
            else
            {
                module->addT<SMovB32>(tmpSgpr2, magic, dComment);
                module->addT<SMulHIU32>(tmpSgpr1, dRegSgpr, tmpSgpr2, dComment);
                module->addT<SMulI32>(tmpSgpr, dRegSgpr, tmpSgpr2, dComment);
            }

            module->addT<SLShiftRightB64>(tmp2Sgpr, shift, tmp2Sgpr, dComment);
            module->addT<SMovB32>(qRegSgpr, tmpSgpr, dComment);

            if(divisor <= 64 && divisor >= -16)
            {
                module->addT<SMulI32>(tmpSgpr, qRegSgpr, divisor, dComment);
            }
            else
            {
                module->addT<SMovB32>(tmpSgpr2, divisor, dComment);
                module->addT<SMulI32>(tmpSgpr, qRegSgpr, tmpSgpr2, dComment);
            }

            module->addT<SSubU32>(rRegSgpr, dRegSgpr, tmpSgpr, dComment);
        }

        return module;
    }

    // Use fp64 reciprocal instruction to compute integer division.
    // This gives accurate results for dividends up to 2^24 and divisors up to 2^16.
    template <typename QREG, typename DREG, typename DIVREG, typename RREG>
    std::shared_ptr<Module> scalarUInt24DivideAndRemainder(QREG                qReg,
                                                           DREG                dReg,
                                                           DIVREG              divReg,
                                                           RREG                rReg,
                                                           ContinuousRegister& tmpVgprRes,
                                                           int                 wavewidth,
                                                           bool                doRemainder = true,
                                                           bool                doQuotient = true,
                                                           const std::string&  comment     = "")
    {
        auto module = std::make_shared<Module>("scalarUInt24DivideAndRemainder");

        auto qRegSgpr   = sgpr(qReg);
        auto rRegSgpr   = sgpr(rReg);
        auto dRegSgpr   = sgpr(dReg);
        auto divRegSgpr = sgpr(divReg);

        std::string dComment = comment.empty() ? qRegSgpr->toString() + " = " + dRegSgpr->toString()
                                                     + " / " + divRegSgpr->toString()
                                               : comment;

        std::string rComment = "";
        if(doRemainder)
        {
            rComment = comment.empty() ? sgpr(rReg)->toString() + " = " + dRegSgpr->toString()
                                             + " % " + divRegSgpr->toString()
                                       : comment;
        }

        if(tmpVgprRes.size < 4)
        {
            throw std::runtime_error("Invalid tmpVgprRes, must be at least 4");
        }
        auto tmpVgpr  = tmpVgprRes.idx;
        auto tmpVgpr1 = tmpVgprRes.idx + 2;

        auto pEXEC = MAKE(EXEC);

        module->addT<VCvtU32toF64>(vgpr(tmpVgpr, 2), divRegSgpr, std::nullopt, dComment);
        module->addT<VRcpF64>(vgpr(tmpVgpr, 2), vgpr(tmpVgpr, 2), dComment);
        module->addT<VCvtU32toF64>(vgpr(tmpVgpr1, 2), dRegSgpr, std::nullopt, dComment);
        module->addT<VMulF64>(vgpr(tmpVgpr, 2), vgpr(tmpVgpr, 2), vgpr(tmpVgpr1, 2), std::nullopt, dComment);
        module->addT<VCvtF64toU32>(vgpr(tmpVgpr), vgpr(tmpVgpr, 2), std::nullopt, dComment);

        module->addT<VMulLOU32>(vgpr(tmpVgpr + 1), vgpr(tmpVgpr), divRegSgpr, dComment);
        module->addT<VSubU32>(vgpr(tmpVgpr1), dRegSgpr, vgpr(tmpVgpr + 1), dComment);
        module->addT<VCmpXGeU32>(pEXEC, vgpr(tmpVgpr1), divRegSgpr, std::nullopt, dComment);
        module->addT<VAddU32>(vgpr(tmpVgpr), vgpr(tmpVgpr), 1, dComment);

        auto resetExec = [module, pEXEC](int wavewidth) {
            if(wavewidth == 64)
            {
                module->addT<SMovB64>(pEXEC, -1, "Reset exec");
            }
            else
            {
                module->addT<SMovB32>(pEXEC, -1, "Reset exec");
            }
        };
        resetExec(wavewidth);

        if(doRemainder)
        {
            module->addT<VMulLOU32>(vgpr(tmpVgpr + 1), vgpr(tmpVgpr), divRegSgpr, dComment);
            module->addT<VSubU32>(vgpr(tmpVgpr1), dRegSgpr, vgpr(tmpVgpr + 1), dComment);
        }

        if(doQuotient)
        {
            module->addT<VReadfirstlaneB32>(qRegSgpr, vgpr(tmpVgpr), "quotient");
        }
        else
        {
            module->addT<SNop>(0);
        }

        if(doRemainder)
        {
            module->addT<VReadfirstlaneB32>(sgpr(rReg), vgpr(tmpVgpr1), "remainder");
        }

        return module;
    }

    template <typename QREG, typename DREG, typename DIVREG, typename RREG>
    std::shared_ptr<Module> scalarUInt32DivideAndRemainder(QREG                qReg,
                                                           DREG                dReg,
                                                           DIVREG              divReg,
                                                           RREG                rReg,
                                                           ContinuousRegister& tmpVgprRes,
                                                           int                 wavewidth,
                                                           bool                doRemainder = true,
                                                           const std::string&  comment     = "")
    {
        auto module = std::make_shared<Module>("scalarUInt32DivideAndRemainder");

        auto qRegSgpr   = sgpr(qReg);
        auto rRegSgpr   = sgpr(rReg);
        auto dRegSgpr   = sgpr(dReg);
        auto divRegSgpr = sgpr(divReg);

        std::string dComment = comment.empty() ? qRegSgpr->toString() + " = " + dRegSgpr->toString()
                                                     + " / " + divRegSgpr->toString()
                                               : comment;

        std::string rComment = "";
        if(doRemainder)
        {
            rComment = comment.empty() ? sgpr(rReg)->toString() + " = " + dRegSgpr->toString()
                                             + " % " + divRegSgpr->toString()
                                       : comment;
        }

        if(tmpVgprRes.size < 2)
        {
            throw std::runtime_error("Invalid tmpVgprRes, must be at least 2");
        }
        auto tmpVgpr  = vgpr(tmpVgprRes.idx);
        auto tmpVgpr1 = vgpr(tmpVgprRes.idx + 1);

        auto pEXEC = MAKE(EXEC);

        module->addT<VCvtU32toF32>(tmpVgpr, divRegSgpr, std::nullopt, dComment);
        module->addT<VRcpIFlagF32>(tmpVgpr, tmpVgpr, dComment);
        module->addT<VCvtU32toF32>(tmpVgpr1, dRegSgpr, std::nullopt, dComment);
        module->addT<VMulF32>(tmpVgpr, tmpVgpr, tmpVgpr1, std::nullopt, dComment);
        module->addT<VCvtF32toU32>(tmpVgpr, tmpVgpr, std::nullopt, dComment);
        module->addT<VMulU32U24>(tmpVgpr1, tmpVgpr, divRegSgpr, dComment);
        module->addT<VSubU32>(tmpVgpr1, dRegSgpr, tmpVgpr1, dComment);
        module->addT<VCmpXEqU32>(pEXEC, tmpVgpr1, divRegSgpr, std::nullopt, dComment);
        module->addT<VAddU32>(tmpVgpr, 1, tmpVgpr, dComment);

        if(doRemainder)
        {
            module->addT<VMovB32>(tmpVgpr1, 0, std::nullopt, rComment);
        }

        auto resetExec = [module, pEXEC](int wavewidth) {
            if(wavewidth == 64)
            {
                module->addT<SMovB64>(pEXEC, -1, "Reset exec");
            }
            else
            {
                module->addT<SMovB32>(pEXEC, -1, "Reset exec");
            }
        };
        resetExec(wavewidth);
        module->addT<VCmpXGtU32>(
            pEXEC, tmpVgpr1, divRegSgpr, std::nullopt, "overflow happened in remainder");
        module->addT<VSubU32>(tmpVgpr, tmpVgpr, 1, "quotient - 1");

        if(doRemainder)
        {
            module->addT<VMulU32U24>(tmpVgpr1, tmpVgpr, divRegSgpr, "re-calculate remainder");
            module->addT<VSubU32>(tmpVgpr1, dRegSgpr, tmpVgpr1, "re-calculate remainder");
        }

        resetExec(wavewidth);
        module->addT<VReadfirstlaneB32>(qRegSgpr, tmpVgpr, "quotient");

        if(doRemainder)
        {
            module->addT<VReadfirstlaneB32>(sgpr(rReg), tmpVgpr1, "remainder");
        }

        return module;
    }

    // Perform a magic division (mul by magic number and shift)
    // dest is two consec SGPR, used for intermediate temp as well as final result
    // result quotient returned in sgpr(dest,1)
    // tmpVgpr: Size 2
    template <typename DEST>
    std::shared_ptr<Module> sMagicDiv(DEST                dest,
                                      bool                hasSMulHi,
                                      int                 dividend,
                                      int                 magicNumber,
                                      int                 magicShift,
                                      ContinuousRegister& tmpVgpr)
    {
        auto module = std::make_shared<Module>("sMagicDiv");

        auto destSgpr  = sgpr(dest, 2);
        auto destSgpr0 = sgpr(dest);
        auto destSgpr1 = [&dest]() {
            if constexpr(std::is_same_v<DEST, int>)
            {
                return sgpr(dest + 1);
            }
            else if constexpr(std::is_same_v<DEST, std::string>)
            {
                std::string destStr = dest + "+1";
                return sgpr(destStr);
            }
            return sgpr(-1);
        }();
        auto continuousReg = ContinuousRegister(tmpVgpr.idx, 2);

        module->addModuleAsFlatItems(SMulInt64to32(destSgpr0,
                                                   destSgpr1,
                                                   dividend,
                                                   magicNumber,
                                                   continuousReg,
                                                   hasSMulHi,
                                                   false,
                                                   "s_magic mul"));
        module->addT<SLShiftRightB64>(destSgpr, magicShift, destSgpr, "sMagicDiv");
        return module;
    }

    // Perform a sgpr version of magic division algo 2 (mul by magic number, Abit and shift)
    // dest is three consec SGPR, used for intermediate temp as well as final result
    // result quotient returned in sgpr(dest,1)
    inline std::shared_ptr<Module>
        sMagicDiv2(const std::shared_ptr<RegisterContainer>& dst,
                   const std::shared_ptr<RegisterContainer>& dst2,
                   const std::shared_ptr<RegisterContainer>& dividend,
                   const std::shared_ptr<RegisterContainer>& magicNumber,
                   const std::shared_ptr<RegisterContainer>& magicShiftAbit,
                   const std::shared_ptr<RegisterContainer>& tmpSgpr)
    {
        auto module = std::make_shared<Module>("sMagicDiv2");

        module->addT<SMulHIU32>(dst2, dividend, magicNumber, "s_magic mul, div alg 2");
        module->addT<SLShiftRightB32>(tmpSgpr, 31, magicShiftAbit, "tmpS = extract abit");
        module->addT<SMulI32>(dst, dividend, tmpSgpr, "s_magic mul, div alg 2");
        module->addT<SAddU32>(dst, dst, dst2);

        module->addT<SAndB32>(
            tmpSgpr, magicShiftAbit, 0x7fffffff, "tmpS = remove abit to final shift");
        module->addT<SLShiftRightB32>(dst, tmpSgpr, dst, "sMagicDiv Alg 2");

        return module;
    }

    // Multiply
    // product register, operand register, multiplier
    std::shared_ptr<Module> vectorStaticMultiply(const std::shared_ptr<RegisterContainer>& product,
                                                 const std::shared_ptr<RegisterContainer>& operand,
                                                 int multiplier,
                                                 const std::optional<ContinuousRegister>& tmpSgprRes
                                                 = std::nullopt,
                                                 const std::string& comment = "");

    // MultiplyAdd
    // product register, operand register, multiplier, accumulator
    std::shared_ptr<Module>
        vectorStaticMultiplyAdd(const std::shared_ptr<RegisterContainer>& product,
                                const std::shared_ptr<RegisterContainer>& operand,
                                int                                       multiplier,
                                const std::shared_ptr<RegisterContainer>& accumulator,
                                const std::optional<ContinuousRegister>&  tmpSgprRes = std::nullopt,
                                const std::string&                        comment    = "");

    // Multiply scalar for 64bit
    // product register, operand register, multiplier
    std::shared_ptr<Module>
        scalarStaticMultiply64(const std::shared_ptr<RegisterContainer>& product,
                               const std::shared_ptr<RegisterContainer>& operand,
                               int                                       multiplier,
                               const std::optional<ContinuousRegister>&  tmpSgprRes = std::nullopt,
                               const std::string&                        comment    = "");
} // namespace rocisa
