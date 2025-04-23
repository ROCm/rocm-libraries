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
#include "instruction/math.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace rocisa
{
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
}

void math_inst(nb::module_ m)
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
}
