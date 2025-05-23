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
#include "instruction/mfma.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void mfma_inst(nb::module_ m_mfma)
{
    nb::class_<rocisa::MFMAInstruction, rocisa::Instruction>(m_mfma, "MFMAInstruction")
        .def(nb::init<rocisa::InstType,
                      rocisa::InstType,
                      const std::vector<int>&,
                      bool,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      bool,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("mfma1k"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("acc2")    = nullptr,
             nb::arg("neg")     = false,
             nb::arg("comment") = "")
        .def_rw("a", &rocisa::MFMAInstruction::a)
        .def_rw("b", &rocisa::MFMAInstruction::b)
        .def_rw("acc", &rocisa::MFMAInstruction::acc)
        .def_rw("acc2", &rocisa::MFMAInstruction::acc2)
        .def("getParams", &rocisa::MFMAInstruction::getParams)
        .def("__str__", &rocisa::MFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::MFMAInstruction& self, const nb::dict&) {
            return new rocisa::MFMAInstruction(self);
        });

    nb::class_<rocisa::SMFMAInstruction, rocisa::Instruction>(m_mfma, "SMFMAInstruction")
        .def(nb::init<rocisa::InstType,
                      rocisa::InstType,
                      const std::vector<int>&,
                      bool,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("mfma1k"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("metadata"),
             nb::arg("comment") = "")
        .def_rw("a", &rocisa::SMFMAInstruction::a)
        .def_rw("b", &rocisa::SMFMAInstruction::b)
        .def_rw("acc", &rocisa::SMFMAInstruction::acc)
        .def_rw("metadata", &rocisa::SMFMAInstruction::metadata)
        .def("getParams", &rocisa::SMFMAInstruction::getParams)
        .def("__str__", &rocisa::SMFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::SMFMAInstruction& self, const nb::dict&) {
            return new rocisa::SMFMAInstruction(self);
        });
}
