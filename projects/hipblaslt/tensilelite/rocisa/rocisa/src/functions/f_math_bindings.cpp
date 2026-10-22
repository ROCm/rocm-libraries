// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "functions/f_math.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

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
    m.def("vectorStaticDivideAndRemainder",
          nb::overload_cast<std::string,
                            int,
                            int,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            bool,
                            const std::string&>(
              &rocisa::vectorStaticDivideAndRemainder<std::string, int>),
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
    m.def("scalarStaticRemainder",
          nb::overload_cast<int,
                            int,
                            std::string,
                            int,
                            std::optional<rocisa::ContinuousRegister>,
                            const std::string&>(&rocisa::scalarStaticRemainder<int, int, std::string>),
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
    m.def("vectorAddMultiplyBpe",
          &rocisa::vectorAddMultiplyBpe,
          nb::arg("dst"),
          nb::arg("src0"),
          nb::arg("src1"),
          nb::arg("bpe"),
          nb::arg("comment")    = "");
    m.def("vectorMultiplyBpe",
        nb::overload_cast<int, int, float, const std::string&>(
            &rocisa::vectorMultiplyBpe<int, int>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("comment")    = "");
    m.def("vectorMultiplyBpe",
        nb::overload_cast<std::string, std::string, float, const std::string&>(
            &rocisa::vectorMultiplyBpe<std::string, std::string>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("comment")    = "");
    m.def("vectorMultiply64Bpe",
          nb::overload_cast<int, int, float, int, const std::string&>(
              &rocisa::vectorMultiply64Bpe<int, int, int>),
          nb::arg("dst"),
          nb::arg("src"),
          nb::arg("bpe"),
          nb::arg("tmp"),
          nb::arg("comment")    = "");
    m.def("vectorMultiply64Bpe",
          nb::overload_cast<std::string, std::string, float, int, const std::string&>(
              &rocisa::vectorMultiply64Bpe<std::string, std::string, int>),
          nb::arg("dst"),
          nb::arg("src"),
          nb::arg("bpe"),
          nb::arg("tmp"),
          nb::arg("comment")    = "");
    m.def("scalarMultiplyBpe",
        nb::overload_cast<int, int, float, const std::string&>(
            &rocisa::scalarMultiplyBpe<int, int>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("comment")    = "");
    m.def("scalarMultiplyBpe",
        nb::overload_cast<std::string, std::string, float, const std::string&>(
            &rocisa::scalarMultiplyBpe<std::string, std::string>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("comment")    = "");
    m.def("scalarMultiplyBpe",
        nb::overload_cast<int, std::string, float, const std::string&>(
            &rocisa::scalarMultiplyBpe<int, std::string>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("comment")    = "");
    m.def("scalarMultiplyBpe",
        nb::overload_cast<std::string, int, float, const std::string&>(
            &rocisa::scalarMultiplyBpe<std::string, int>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("comment")    = "");
    m.def("scalarMultiply64Bpe",
          nb::overload_cast<int, int, float, int, const std::string&>(
              &rocisa::scalarMultiply64Bpe<int, int, int>),
          nb::arg("dst"),
          nb::arg("src"),
          nb::arg("bpe"),
          nb::arg("tmp"),
          nb::arg("comment")    = "");
    m.def("scalarMultiply64Bpe",
        nb::overload_cast<std::string, std::string, float, int, const std::string&>(
            &rocisa::scalarMultiply64Bpe<std::string, std::string, int>),
        nb::arg("dst"),
        nb::arg("src"),
        nb::arg("bpe"),
        nb::arg("tmp"),
        nb::arg("comment")    = "");
}
