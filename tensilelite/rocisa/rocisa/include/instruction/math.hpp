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

#include <cassert>
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
            assert(tmpVgprRes && tmpVgprRes->size >= 2);
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
            assert(tmpVgprRes && tmpVgprRes->size >= 2);
            int  tmpVgprIdx = tmpVgprRes->idx;
            auto tmpVgpr    = vgpr(tmpVgprIdx);
            auto tmpVgpr1   = vgpr(tmpVgprIdx + 1);

            assert(tmpSgprRes && tmpSgprRes->size >= 1);
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
} // namespace rocisa
