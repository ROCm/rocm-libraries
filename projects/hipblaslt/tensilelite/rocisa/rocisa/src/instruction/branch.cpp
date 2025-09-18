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
#include "instruction/branch.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;

void branch_inst(nb::module_ m_branch)
{
    nb::class_<rocisa::BranchInstruction, rocisa::Instruction>(m_branch, "BranchInstruction")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def_rw("labelName", &rocisa::BranchInstruction::labelName)
        .def("getParams", &rocisa::BranchInstruction::getParams)
        .def("toString", &rocisa::BranchInstruction::toString);

    nb::class_<rocisa::SBranch, rocisa::BranchInstruction>(m_branch, "SBranch")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SBranch& self, nb::dict&) { return new rocisa::SBranch(self); });

    nb::class_<rocisa::SCBranchSCC0, rocisa::BranchInstruction>(m_branch, "SCBranchSCC0")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCBranchSCC0& self, nb::dict&) {
            return new rocisa::SCBranchSCC0(self);
        });

    nb::class_<rocisa::SCBranchSCC1, rocisa::BranchInstruction>(m_branch, "SCBranchSCC1")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCBranchSCC1& self, nb::dict&) {
            return new rocisa::SCBranchSCC1(self);
        });

    nb::class_<rocisa::SCBranchVCCNZ, rocisa::BranchInstruction>(m_branch, "SCBranchVCCNZ")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCBranchVCCNZ& self, nb::dict&) {
            return new rocisa::SCBranchVCCNZ(self);
        });

    nb::class_<rocisa::SCBranchVCCZ, rocisa::BranchInstruction>(m_branch, "SCBranchVCCZ")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCBranchVCCZ& self, nb::dict&) {
            return new rocisa::SCBranchVCCZ(self);
        });

    nb::class_<rocisa::SSetPCB64, rocisa::BranchInstruction>(m_branch, "SSetPCB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&, const std::string&>(),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SSetPCB64& self, nb::dict&) { return new rocisa::SSetPCB64(self); });

    nb::class_<rocisa::SSwapPCB64, rocisa::BranchInstruction>(m_branch, "SSwapPCB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSwapPCB64& self, nb::dict&) {
            return new rocisa::SSwapPCB64(self);
        });

    nb::class_<rocisa::SCBranchExecZ, rocisa::BranchInstruction>(m_branch, "SCBranchExecZ")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCBranchExecZ& self, nb::dict&) {
            return new rocisa::SCBranchExecZ(self);
        });

    nb::class_<rocisa::SCBranchExecNZ, rocisa::BranchInstruction>(m_branch, "SCBranchExecNZ")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("labelName"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCBranchExecNZ& self, nb::dict&) {
            return new rocisa::SCBranchExecNZ(self);
        });
}
