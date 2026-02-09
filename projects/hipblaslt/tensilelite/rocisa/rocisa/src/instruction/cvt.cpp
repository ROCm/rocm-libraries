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
#include "instruction/cvt.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void cvt_inst(nb::module_ m_inst)
{
    nb::class_<rocisa::VCvtInstruction, rocisa::CommonInstruction>(m_inst, "VCvtInstruction")
        .def(nb::init<rocisa::CvtType,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::vector<InstructionInput>&,
                      const std::optional<rocisa::SDWAModifiers>,
                      const std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("cvtType"),
             nb::arg("dst"),
             nb::arg("srcs"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "");

    nb::class_<rocisa::VCvtF16toF32, rocisa::VCvtInstruction>(m_inst, "VCvtF16toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtF16toF32& self, const nb::dict&) {
            return new rocisa::VCvtF16toF32(self);
        });

    nb::class_<rocisa::VCvtF32toF16, rocisa::VCvtInstruction>(m_inst, "VCvtF32toF16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtF32toF16& self, const nb::dict&) {
            return new rocisa::VCvtF32toF16(self);
        });

    nb::class_<rocisa::VCvtF32toU32, rocisa::VCvtInstruction>(m_inst, "VCvtF32toU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtF32toU32& self, const nb::dict&) {
            return new rocisa::VCvtF32toU32(self);
        });

    nb::class_<rocisa::VCvtU32toF32, rocisa::VCvtInstruction>(m_inst, "VCvtU32toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtU32toF32& self, const nb::dict&) {
            return new rocisa::VCvtU32toF32(self);
        });

    nb::class_<rocisa::VCvtF64toU32, rocisa::VCvtInstruction>(m_inst, "VCvtF64toU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtF64toU32& self, const nb::dict&) {
            return new rocisa::VCvtF64toU32(self);
        });

    nb::class_<rocisa::VCvtU32toF64, rocisa::VCvtInstruction>(m_inst, "VCvtU32toF64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtU32toF64& self, const nb::dict&) {
            return new rocisa::VCvtU32toF64(self);
        });

    nb::class_<rocisa::VCvtI32toF32, rocisa::VCvtInstruction>(m_inst, "VCvtI32toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtI32toF32& self, const nb::dict&) {
            return new rocisa::VCvtI32toF32(self);
        });

    nb::class_<rocisa::VCvtF32toI32, rocisa::VCvtInstruction>(m_inst, "VCvtF32toI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtF32toI32& self, const nb::dict&) {
            return new rocisa::VCvtF32toI32(self);
        });

    nb::class_<rocisa::VCvtFP8toF32, rocisa::VCvtInstruction>(m_inst, "VCvtFP8toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtFP8toF32& self, const nb::dict&) {
            return new rocisa::VCvtFP8toF32(self);
        });

    nb::class_<rocisa::VCvtBF8toF32, rocisa::VCvtInstruction>(m_inst, "VCvtBF8toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtBF8toF32& self, const nb::dict&) {
            return new rocisa::VCvtBF8toF32(self);
        });

    nb::class_<rocisa::VCvtPkFP8toF32, rocisa::VCvtInstruction>(m_inst, "VCvtPkFP8toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtPkFP8toF32& self, const nb::dict&) {
            return new rocisa::VCvtPkFP8toF32(self);
        });

    nb::class_<rocisa::VCvtPkBF8toF32, rocisa::VCvtInstruction>(m_inst, "VCvtPkBF8toF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtPkBF8toF32& self, const nb::dict&) {
            return new rocisa::VCvtPkBF8toF32(self);
        });

    nb::class_<rocisa::VCvtPkF32toBF8, rocisa::VCvtInstruction>(m_inst, "VCvtPkF32toBF8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtPkF32toBF8& self, const nb::dict&) {
            return new rocisa::VCvtPkF32toBF8(self);
        });

    nb::class_<rocisa::VCvtSRF32toFP8, rocisa::VCvtInstruction>(m_inst, "VCvtSRF32toFP8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtSRF32toFP8& self, const nb::dict&) {
            return new rocisa::VCvtSRF32toFP8(self);
        });

    nb::class_<rocisa::VCvtSRF32toBF8, rocisa::VCvtInstruction>(m_inst, "VCvtSRF32toBF8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtSRF32toBF8& self, const nb::dict&) {
            return new rocisa::VCvtSRF32toBF8(self);
        });

    nb::class_<rocisa::VCvtScalePkFP8toF16, rocisa::VCvtInstruction>(m_inst, "VCvtScalePkFP8toF16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScalePkFP8toF16& self, nb::dict&) {
            return new rocisa::VCvtScalePkFP8toF16(self);
        });

    nb::class_<rocisa::VCvtScalePkBF8toF16, rocisa::VCvtInstruction>(m_inst, "VCvtScalePkBF8toF16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScalePkBF8toF16& self, nb::dict&) {
            return new rocisa::VCvtScalePkBF8toF16(self);
        });

    nb::class_<rocisa::VCvtScaleFP8toF16, rocisa::VCvtInstruction>(m_inst, "VCvtScaleFP8toF16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScaleFP8toF16& self, nb::dict&) {
            return new rocisa::VCvtScaleFP8toF16(self);
        });

    nb::class_<rocisa::VCvtScalePkF16toFP8, rocisa::VCvtInstruction>(m_inst, "VCvtScalePkF16toFP8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScalePkF16toFP8& self, nb::dict&) {
            return new rocisa::VCvtScalePkF16toFP8(self);
        });

    nb::class_<rocisa::VCvtScalePkF16toBF8, rocisa::VCvtInstruction>(m_inst, "VCvtScalePkF16toBF8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScalePkF16toBF8& self, nb::dict&) {
            return new rocisa::VCvtScalePkF16toBF8(self);
        });

    nb::class_<rocisa::VCvtScaleSRF16toFP8, rocisa::VCvtInstruction>(m_inst, "VCvtScaleSRF16toFP8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScaleSRF16toFP8& self, nb::dict&) {
            return new rocisa::VCvtScaleSRF16toFP8(self);
        });

    nb::class_<rocisa::VCvtScaleSRF16toBF8, rocisa::VCvtInstruction>(m_inst, "VCvtScaleSRF16toBF8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("scale"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtScaleSRF16toBF8& self, nb::dict&) {
            return new rocisa::VCvtScaleSRF16toBF8(self);
        });

    nb::class_<rocisa::VCvtPkF32toFP8, rocisa::VCvtInstruction>(m_inst, "VCvtPkF32toFP8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtPkF32toFP8& self, nb::dict&) {
            return new rocisa::VCvtPkF32toFP8(self);
        });

    nb::class_<rocisa::PVCvtBF16toFP32, rocisa::VCvtInstruction>(m_inst, "PVCvtBF16toFP32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::PVCvtBF16toFP32& self, nb::dict&) {
            return new rocisa::PVCvtBF16toFP32(self);
        });

    nb::class_<rocisa::VCvtPkF32toBF16, rocisa::VCvtInstruction>(m_inst, "VCvtPkF32toBF16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCvtPkF32toBF16& self, nb::dict&) {
            return new rocisa::VCvtPkF32toBF16(self);
        });
}
