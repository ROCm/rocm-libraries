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
#include "functions/f_math.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace rocisa
{
    // template of vectorStaticDivide
    template std::shared_ptr<Module> vectorStaticDivideAndRemainder<int, int>(
        int, int, int, int, std::optional<ContinuousRegister>, bool, const std::string&);
    template std::shared_ptr<Module> vectorStaticDivideAndRemainder<int, std::string>(
        int, int, std::string, int, std::optional<ContinuousRegister>, bool, const std::string&);
    template std::shared_ptr<Module> vectorStaticDivide<int, int>(
        int, int, int, std::optional<ContinuousRegister>, const std::string&);
    template std::shared_ptr<Module> vectorStaticDivide<int, std::string>(
        int, std::string, int, std::optional<ContinuousRegister>, const std::string&);
    template std::shared_ptr<Module> vectorStaticDivide<std::string, std::string>(
        std::string, std::string, int, std::optional<ContinuousRegister>, const std::string&);
    // template of vectorStaticRemainder
    template std::shared_ptr<Module>
        vectorStaticRemainder<int, int>(int,
                                        int,
                                        int,
                                        int,
                                        std::optional<ContinuousRegister>,
                                        std::optional<ContinuousRegister>,
                                        const std::string&);
    template std::shared_ptr<Module>
        vectorStaticRemainder<int, std::string>(int,
                                                int,
                                                std::string,
                                                int,
                                                std::optional<ContinuousRegister>,
                                                std::optional<ContinuousRegister>,
                                                const std::string&);
    template std::shared_ptr<Module>
        vectorStaticRemainder<std::string, int>(int,
                                                std::string,
                                                int,
                                                int,
                                                std::optional<ContinuousRegister>,
                                                std::optional<ContinuousRegister>,
                                                const std::string&);
    // template of scalarStaticDivideAndRemainder
    template std::shared_ptr<Module> scalarStaticDivideAndRemainder<int, int, int>(
        int, int, int, int, std::optional<ContinuousRegister>, int);
    template std::shared_ptr<Module> scalarStaticDivideAndRemainder<int, int, std::string>(
        int, int, std::string, int, std::optional<ContinuousRegister>, int);
    template std::shared_ptr<Module> scalarStaticDivideAndRemainder<int, std::string, int>(
        int, std::string, int, int, std::optional<ContinuousRegister>, int);
    template std::shared_ptr<Module> scalarStaticDivideAndRemainder<std::string, int, std::string>(
        std::string, int, std::string, int, std::optional<ContinuousRegister>, int);
    template std::shared_ptr<Module> scalarStaticDivideAndRemainder<int, std::string, std::string>(
        int, std::string, std::string, int, std::optional<ContinuousRegister>, int);
    template std::shared_ptr<Module>
        scalarStaticDivideAndRemainder<int, int, std::shared_ptr<RegisterContainer>>(
            int,
            int,
            std::shared_ptr<RegisterContainer>,
            int,
            std::optional<ContinuousRegister>,
            int);
    // template of scalarStaticCeilDivide
    template std::shared_ptr<Module> scalarStaticCeilDivide<std::shared_ptr<RegisterContainer>,
                                                            std::shared_ptr<RegisterContainer>>(
        std::shared_ptr<RegisterContainer>,
        std::shared_ptr<RegisterContainer>,
        int,
        std::optional<ContinuousRegister>);
    // template of scalarUInt24DivideAndRemainder
    #define ExplicitInstantiation(QREG, DREG, DIVREG, RREG) \
        template std::shared_ptr<Module> scalarUInt24DivideAndRemainder<QREG, DREG, DIVREG, RREG>( \
        QREG, DREG, DIVREG, RREG, ContinuousRegister&, int, bool, bool, const std::string&);
    ExplicitInstantiation(std::string, std::string, std::string, std::string)
    ExplicitInstantiation(std::string, std::string, std::string, int)
    ExplicitInstantiation(std::string, std::string, int,         std::string)
    ExplicitInstantiation(std::string, std::string, int,         int)
    ExplicitInstantiation(std::string, int,         std::string, std::string)
    ExplicitInstantiation(std::string, int,         std::string, int)
    ExplicitInstantiation(std::string, int,         int,         std::string)
    ExplicitInstantiation(std::string, int,         int,         int)
    ExplicitInstantiation(int,         std::string, std::string, std::string)
    ExplicitInstantiation(int,         std::string, std::string, int)
    ExplicitInstantiation(int,         std::string, int,         std::string)
    ExplicitInstantiation(int,         std::string, int,         int)
    ExplicitInstantiation(int,         int,         std::string, std::string)
    ExplicitInstantiation(int,         int,         std::string, int)
    ExplicitInstantiation(int,         int,         int,         std::string)
    ExplicitInstantiation(int,         int,         int,         int)
    #undef ExplicitInstantiation
    // template of scalarStaticRemainder
    template std::shared_ptr<Module> scalarStaticRemainder<int, int>(
        int, int, int, int, std::optional<ContinuousRegister>, const std::string&);
    // template of scalarUInt32DivideAndRemainder
    #define ExplicitInstantiation(QREG, DREG, DIVREG, RREG) \
        template std::shared_ptr<Module> scalarUInt32DivideAndRemainder<QREG, DREG, DIVREG, RREG>( \
        QREG, DREG, DIVREG, RREG, ContinuousRegister&, int, bool, const std::string&);
    ExplicitInstantiation(std::string, std::string, std::string, std::string)
    ExplicitInstantiation(std::string, std::string, std::string, int)
    ExplicitInstantiation(std::string, std::string, int,         std::string)
    ExplicitInstantiation(std::string, std::string, int,         int)
    ExplicitInstantiation(std::string, int,         std::string, std::string)
    ExplicitInstantiation(std::string, int,         std::string, int)
    ExplicitInstantiation(std::string, int,         int,         std::string)
    ExplicitInstantiation(std::string, int,         int,         int)
    ExplicitInstantiation(int,         std::string, std::string, std::string)
    ExplicitInstantiation(int,         std::string, std::string, int)
    ExplicitInstantiation(int,         std::string, int,         std::string)
    ExplicitInstantiation(int,         std::string, int,         int)
    ExplicitInstantiation(int,         int,         std::string, std::string)
    ExplicitInstantiation(int,         int,         std::string, int)
    ExplicitInstantiation(int,         int,         int,         std::string)
    ExplicitInstantiation(int,         int,         int,         int)
    #undef ExplicitInstantiation
    // template of sMagicDiv
    template std::shared_ptr<Module> sMagicDiv<int>(int                 dest,
                                                    bool                hasSMulHi,
                                                    int                 dividend,
                                                    int                 magicNumber,
                                                    int                 magicShift,
                                                    ContinuousRegister& tmpVgpr);

    std::shared_ptr<Module>
        vectorStaticMultiply(const std::shared_ptr<RegisterContainer>& product,
                             const std::shared_ptr<RegisterContainer>& operand,
                             int                                       multiplier,
                             const std::optional<ContinuousRegister>&  tmpSgprRes,
                             const std::string&                        comment)
    {
        std::string dComment = comment.empty() ? product->toString() + " = " + operand->toString()
                                                     + " * " + std::to_string(multiplier)
                                               : comment;
        auto        module   = std::make_shared<Module>("vectorStaticMultiply");
        if(multiplier == 0)
        {
            module->addT<VMovB32>(product, multiplier, std::nullopt, dComment);
        }
        else if((multiplier & (multiplier - 1)) == 0) // pow of 2
        {
            int multiplier_log2 = static_cast<int>(std::log2(multiplier));
            if(multiplier_log2 == 0 && (*product == *operand))
            {
                module->addCommentAlign(dComment + " (multiplier is 1, do nothing)");
            }
            else
            {
                module->addT<VLShiftLeftB32>(product, multiplier_log2, operand, dComment);
            }
        }
        else
        {
            if(multiplier <= 64 && multiplier >= -16)
            {
                module->addT<VMulLOU32>(product, multiplier, operand, dComment);
            }
            else
            {
                if(!tmpSgprRes || tmpSgprRes->size < 1)
                {
                    throw std::runtime_error("Invalid tmpSgprRes, must be at least 1");
                }
                auto tmpSgpr = sgpr(tmpSgprRes->idx);
                module->addT<SMovB32>(tmpSgpr, multiplier, dComment);
                module->addT<VMulLOU32>(product, tmpSgpr, operand, dComment);
            }
        }
        return module;
    }

    std::shared_ptr<Module>
        vectorStaticMultiplyAdd(const std::shared_ptr<RegisterContainer>& product,
                                const std::shared_ptr<RegisterContainer>& operand,
                                int                                       multiplier,
                                const std::shared_ptr<RegisterContainer>& accumulator,
                                const std::optional<ContinuousRegister>&  tmpSgprRes,
                                const std::string&                        comment)
    {
        std::string dComment = comment.empty() ? product->toString() + " = " + operand->toString()
                                                     + " * " + std::to_string(multiplier)
                                               : comment;
        auto        module   = std::make_shared<Module>("vectorStaticMultiplyAdd");
        if(multiplier == 0)
        {
            module->addT<VMovB32>(product, accumulator, std::nullopt, dComment);
        }
        else if((multiplier & (multiplier - 1)) == 0) // pow of 2
        {
            int multiplier_log2 = static_cast<int>(std::log2(multiplier));
            if(multiplier_log2 == 0)
            {
                module->addT<VAddU32>(product, operand, accumulator, dComment);
            }
            else
            {
                module->addT<VLShiftLeftAddU32>(
                    product, multiplier_log2, operand, accumulator, dComment);
            }
        }
        else // not pow of 2
        {
            if(multiplier <= 64 && multiplier >= -16)
            {
                module->addT<VMadU32U24>(
                    product, multiplier, operand, accumulator, std::nullopt, dComment);
            }
            else
            {
                if(!tmpSgprRes || tmpSgprRes->size < 1)
                {
                    throw std::runtime_error("Invalid tmpSgprRes, must be at least 1");
                }
                auto tmpSgpr = sgpr(tmpSgprRes->idx);
                module->addT<SMovB32>(tmpSgpr, multiplier, dComment);
                module->addT<VMadU32U24>(
                    product, tmpSgpr, operand, accumulator, std::nullopt, dComment);
            }
        }
        return module;
    }

    std::shared_ptr<Module>
        scalarStaticMultiply64(const std::shared_ptr<RegisterContainer>& product,
                               const std::shared_ptr<RegisterContainer>& operand,
                               int                                       multiplier,
                               const std::optional<ContinuousRegister>&  tmpSgprRes,
                               const std::string&                        comment)
    {
        std::string commentStr = comment.empty() ? product->toString() + " = " + operand->toString()
                                                       + " * " + std::to_string(multiplier)
                                                 : comment;
        auto        module     = std::make_shared<Module>("scalarStaticMultiply64");
        if(multiplier == 0)
        {
            module->addT<SMovB64>(product, 0, commentStr);
        }

        // TODO- to support non-pow2, need to use mul_32 and mul_hi_32 ?
        if((multiplier & (multiplier - 1)) != 0)
        {
            throw std::runtime_error("Multiplier must be a power of 2");
        }

        int multiplier_log2 = static_cast<int>(std::log2(multiplier));
        if(multiplier_log2 == 0 && (*product == *operand))
        {
            module->addCommentAlign(comment + " (multiplier is 1, do nothing)");
        }
        else
        {
            // notice that the src-order of s_lshl_b64 is different from v_lshlrev_b32.
            module->addT<SLShiftLeftB64>(product, multiplier_log2, operand, commentStr);
        }
        return module;
    }
} // namespace rocisa

void math_func(nb::module_ m)
{
    m.def("vectorStaticDivideAndRemainder",
          nb::overload_cast<int,
                            int,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            bool,
                            const std::string&>(&rocisa::vectorStaticDivideAndRemainder<int, int>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes")  = std::nullopt,
          nb::arg("doRemainder") = true,
          nb::arg("comment")     = "");
    m.def("vectorStaticDivideAndRemainder",
          nb::overload_cast<int,
                            int,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            bool,
                            const std::string&>(
              &rocisa::vectorStaticDivideAndRemainder<int, std::string>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes")  = std::nullopt,
          nb::arg("doRemainder") = true,
          nb::arg("comment")     = "");
    m.def("vectorStaticDivide",
          nb::overload_cast<int,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::vectorStaticDivide<int, int>),
          nb::arg("qReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("vectorStaticDivide",
          nb::overload_cast<int,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::vectorStaticDivide<int, std::string>),
          nb::arg("qReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("vectorStaticDivide",
          nb::overload_cast<std::string,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(
              &rocisa::vectorStaticDivide<std::string, std::string>),
          nb::arg("qReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("vectorUInt32DivideAndRemainder",
          &rocisa::vectorUInt32DivideAndRemainder,
          nb::arg("qReg"),
          nb::arg("dReg"),
          nb::arg("divReg"),
          nb::arg("rReg"),
          nb::arg("doRemainder") = true,
          nb::arg("comment")     = "");
    m.def("vectorUInt32CeilDivideAndRemainder",
          &rocisa::vectorUInt32CeilDivideAndRemainder,
          nb::arg("qReg"),
          nb::arg("dReg"),
          nb::arg("divReg"),
          nb::arg("rReg"),
          nb::arg("doRemainder") = true,
          nb::arg("comment")     = "");
    m.def("vectorStaticRemainder",
          nb::overload_cast<int,
                            int,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::vectorStaticRemainder<int, int>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes") = std::nullopt,
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("vectorStaticRemainder",
          nb::overload_cast<int,
                            int,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::vectorStaticRemainder<int, std::string>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes") = std::nullopt,
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("vectorStaticRemainder",
          nb::overload_cast<int,
                            std::string,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::vectorStaticRemainder<std::string, int>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpVgprRes") = std::nullopt,
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");

    m.def("scalarStaticDivideAndRemainder",
          nb::overload_cast<int, int, int, int, std::optional<rocisa::ContinuousRegister>, int>(
              &rocisa::scalarStaticDivideAndRemainder<int, int, int>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes")  = std::nullopt,
          nb::arg("doRemainder") = 1);
    m.def("scalarStaticDivideAndRemainder",
          nb::overload_cast<int,
                            int,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            int>(&rocisa::scalarStaticDivideAndRemainder<int, int, std::string>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes")  = std::nullopt,
          nb::arg("doRemainder") = 1);
    m.def("scalarStaticDivideAndRemainder",
          nb::overload_cast<int,
                            std::string,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            int>(&rocisa::scalarStaticDivideAndRemainder<int, std::string, int>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes")  = std::nullopt,
          nb::arg("doRemainder") = 1);
    m.def("scalarStaticDivideAndRemainder",
          nb::overload_cast<std::string,
                            int,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            int>(
              &rocisa::scalarStaticDivideAndRemainder<std::string, int, std::string>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes")  = std::nullopt,
          nb::arg("doRemainder") = 1);
    m.def("scalarStaticDivideAndRemainder",
          nb::overload_cast<int,
                            std::string,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            int>(
              &rocisa::scalarStaticDivideAndRemainder<int, std::string, std::string>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes")  = std::nullopt,
          nb::arg("doRemainder") = 1);
    m.def("scalarStaticDivideAndRemainder",
          nb::overload_cast<int,
                            int,
                            std::shared_ptr<rocisa::RegisterContainer>,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            int>(
              &rocisa::scalarStaticDivideAndRemainder<int,
                                                      int,
                                                      std::shared_ptr<rocisa::RegisterContainer>>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes")  = std::nullopt,
          nb::arg("doRemainder") = 1);
    m.def("scalarStaticCeilDivide",
          nb::overload_cast<std::shared_ptr<rocisa::RegisterContainer>,
                            std::shared_ptr<rocisa::RegisterContainer>,
                            int,
                            std::optional<rocisa::ContinuousRegister>>(
              &rocisa::scalarStaticCeilDivide<std::shared_ptr<rocisa::RegisterContainer>,
                                              std::shared_ptr<rocisa::RegisterContainer>>),
          nb::arg("qReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes") = std::nullopt);
    m.def("scalarStaticRemainder",
          nb::overload_cast<int,
                            int,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::scalarStaticRemainder<int, int, int>),
          nb::arg("qReg"),
          nb::arg("rReg"),
          nb::arg("dReg"),
          nb::arg("divisor"),
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    #define ExplicitInstantiation(QREG, DREG, DIVREG, RREG) \
    m.def("scalarUInt24DivideAndRemainder", \
          nb::overload_cast<QREG, \
                            DREG, \
                            DIVREG, \
                            RREG, \
                            rocisa::ContinuousRegister&, \
                            int, \
                            bool, \
                            bool, \
                            const std::string&>( \
              &rocisa::scalarUInt24DivideAndRemainder<QREG, DREG, DIVREG, RREG>), \
          nb::arg("qReg"), \
          nb::arg("dReg"), \
          nb::arg("divReg"), \
          nb::arg("rReg"), \
          nb::arg("tmpVgprRes"), \
          nb::arg("wavewidth"), \
          nb::arg("doRemainder") = true, \
          nb::arg("doQuotient") = true, \
          nb::arg("comment")     = "");
    ExplicitInstantiation(std::string, std::string, std::string, std::string)
    ExplicitInstantiation(std::string, std::string, std::string, int)
    ExplicitInstantiation(std::string, std::string, int,         std::string)
    ExplicitInstantiation(std::string, std::string, int,         int)
    ExplicitInstantiation(std::string, int,         std::string, std::string)
    ExplicitInstantiation(std::string, int,         std::string, int)
    ExplicitInstantiation(std::string, int,         int,         std::string)
    ExplicitInstantiation(std::string, int,         int,         int)
    ExplicitInstantiation(int,         std::string, std::string, std::string)
    ExplicitInstantiation(int,         std::string, std::string, int)
    ExplicitInstantiation(int,         std::string, int,         std::string)
    ExplicitInstantiation(int,         std::string, int,         int)
    ExplicitInstantiation(int,         int,         std::string, std::string)
    ExplicitInstantiation(int,         int,         std::string, int)
    ExplicitInstantiation(int,         int,         int,         std::string)
    ExplicitInstantiation(int,         int,         int,         int)
    #undef ExplicitInstantiation
    #define ExplicitInstantiation(QREG, DREG, DIVREG, RREG) \
    m.def("scalarUInt32DivideAndRemainder", \
          nb::overload_cast<QREG, \
                            DREG, \
                            DIVREG, \
                            RREG, \
                            rocisa::ContinuousRegister&, \
                            int, \
                            bool, \
                            const std::string&>( \
              &rocisa::scalarUInt32DivideAndRemainder<QREG, DREG, DIVREG, RREG>), \
          nb::arg("qReg"), \
          nb::arg("dReg"), \
          nb::arg("divReg"), \
          nb::arg("rReg"), \
          nb::arg("tmpVgprRes"), \
          nb::arg("wavewidth"), \
          nb::arg("doRemainder") = true, \
          nb::arg("comment")     = "");
    ExplicitInstantiation(std::string, std::string, std::string, std::string)
    ExplicitInstantiation(std::string, std::string, std::string, int)
    ExplicitInstantiation(std::string, std::string, int,         std::string)
    ExplicitInstantiation(std::string, std::string, int,         int)
    ExplicitInstantiation(std::string, int,         std::string, std::string)
    ExplicitInstantiation(std::string, int,         std::string, int)
    ExplicitInstantiation(std::string, int,         int,         std::string)
    ExplicitInstantiation(std::string, int,         int,         int)
    ExplicitInstantiation(int,         std::string, std::string, std::string)
    ExplicitInstantiation(int,         std::string, std::string, int)
    ExplicitInstantiation(int,         std::string, int,         std::string)
    ExplicitInstantiation(int,         std::string, int,         int)
    ExplicitInstantiation(int,         int,         std::string, std::string)
    ExplicitInstantiation(int,         int,         std::string, int)
    ExplicitInstantiation(int,         int,         int,         std::string)
    ExplicitInstantiation(int,         int,         int,         int)
    #undef ExplicitInstantiation
    m.def("sMagicDiv",
          nb::overload_cast<int, bool, int, int, int, rocisa::ContinuousRegister&>(
              &rocisa::sMagicDiv<int>),
          nb::arg("dest"),
          nb::arg("hasSMulHi"),
          nb::arg("dividend"),
          nb::arg("magicNumber"),
          nb::arg("magicShift"),
          nb::arg("tmpVgprRes"));
    m.def("sMagicDiv2",
          &rocisa::sMagicDiv2,
          nb::arg("dst"),
          nb::arg("dst2"),
          nb::arg("dividend"),
          nb::arg("magicNumber"),
          nb::arg("magicShiftAbit"),
          nb::arg("tmpSgpr"));
    m.def("vectorStaticMultiply",
          &rocisa::vectorStaticMultiply,
          nb::arg("product"),
          nb::arg("operand"),
          nb::arg("multiplier"),
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("vectorStaticMultiplyAdd",
          &rocisa::vectorStaticMultiplyAdd,
          nb::arg("product"),
          nb::arg("operand"),
          nb::arg("multiplier"),
          nb::arg("accumulator"),
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");
    m.def("scalarStaticMultiply64",
          &rocisa::scalarStaticMultiply64,
          nb::arg("product"),
          nb::arg("operand"),
          nb::arg("multiplier"),
          nb::arg("tmpSgprRes") = std::nullopt,
          nb::arg("comment")    = "");
}
