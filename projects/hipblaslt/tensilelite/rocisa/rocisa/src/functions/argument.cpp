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
#include "functions/argument.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;

void argument_func(nb::module_ m)
{
    nb::class_<rocisa::ArgumentLoader>(m, "ArgumentLoader")
        .def(nb::init<>())
        .def("resetOffset", &rocisa::ArgumentLoader::resetOffset)
        .def("setOffset", &rocisa::ArgumentLoader::setOffset)
        .def("getOffset", &rocisa::ArgumentLoader::getOffset)
        .def("loadKernArg",
             nb::overload_cast<int, int, std::optional<InstructionInput>, int, bool>(
                 &rocisa::ArgumentLoader::loadKernArg<int, int>),
             nb::arg("dst"),
             nb::arg("srcAddr"),
             nb::arg("sgprOffset") = std::nullopt,
             nb::arg("dword")      = 1,
             nb::arg("writeSgpr")  = true)
        .def(
            "loadKernArg",
            nb::overload_cast<std::string, std::string, std::optional<InstructionInput>, int, bool>(
                &rocisa::ArgumentLoader::loadKernArg<std::string, std::string>),
            nb::arg("dst"),
            nb::arg("srcAddr"),
            nb::arg("sgprOffset") = std::nullopt,
            nb::arg("dword")      = 1,
            nb::arg("writeSgpr")  = true)
        .def("loadKernArg",
             nb::overload_cast<int, std::string, std::optional<InstructionInput>, int, bool>(
                 &rocisa::ArgumentLoader::loadKernArg<int, std::string>),
             nb::arg("dst"),
             nb::arg("srcAddr"),
             nb::arg("sgprOffset") = std::nullopt,
             nb::arg("dword")      = 1,
             nb::arg("writeSgpr")  = true)
        .def("loadKernArg",
             nb::overload_cast<std::string, int, std::optional<InstructionInput>, int, bool>(
                 &rocisa::ArgumentLoader::loadKernArg<std::string, int>),
             nb::arg("dst"),
             nb::arg("srcAddr"),
             nb::arg("sgprOffset") = std::nullopt,
             nb::arg("dword")      = 1,
             nb::arg("writeSgpr")  = true)
        .def("loadAllKernArg",
             nb::overload_cast<int, int, int, int>(&rocisa::ArgumentLoader::loadAllKernArg<int>),
             nb::arg("sgprStartIndex"),
             nb::arg("srcAddr"),
             nb::arg("numSgprToLoad"),
             nb::arg("numSgprPreload") = 0)
        .def("loadAllKernArg",
             nb::overload_cast<int, std::string, int, int>(
                 &rocisa::ArgumentLoader::loadAllKernArg<std::string>),
             nb::arg("sgprStartIndex"),
             nb::arg("srcAddr"),
             nb::arg("numSgprToLoad"),
             nb::arg("numSgprPreload") = 0);
}
