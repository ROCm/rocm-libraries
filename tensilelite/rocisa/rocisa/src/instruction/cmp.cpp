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
#include "instruction/cmp.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;

void cmp_inst(nb::module_ m_cmp)
{
    nb::class_<rocisa::VCmpInstruction, rocisa::CommonInstruction>(m_cmp, "VCmpInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::string>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__str__", &rocisa::VCmpInstruction::toString);

    nb::class_<rocisa::VCmpXInstruction, rocisa::CommonInstruction>(m_cmp, "VCmpXInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::string>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__str__", &rocisa::VCmpXInstruction::toString);

    nb::class_<rocisa::SCmpEQI32, rocisa::CommonInstruction>(m_cmp, "SCmpEQI32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpEQI32& self, const nb::dict&) {
            return new rocisa::SCmpEQI32(self);
        });

    nb::class_<rocisa::SCmpEQU32, rocisa::CommonInstruction>(m_cmp, "SCmpEQU32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpEQU32& self, const nb::dict&) {
            return new rocisa::SCmpEQU32(self);
        });

    nb::class_<rocisa::SCmpEQU64, rocisa::CommonInstruction>(m_cmp, "SCmpEQU64")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpEQU64& self, const nb::dict&) {
            return new rocisa::SCmpEQU64(self);
        });

    nb::class_<rocisa::SCmpGeI32, rocisa::CommonInstruction>(m_cmp, "SCmpGeI32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpGeI32& self, const nb::dict&) {
            return new rocisa::SCmpGeI32(self);
        });

    nb::class_<rocisa::SCmpGeU32, rocisa::CommonInstruction>(m_cmp, "SCmpGeU32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpGeU32& self, const nb::dict&) {
            return new rocisa::SCmpGeU32(self);
        });

    nb::class_<rocisa::SCmpGtI32, rocisa::CommonInstruction>(m_cmp, "SCmpGtI32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpGtI32& self, const nb::dict&) {
            return new rocisa::SCmpGtI32(self);
        });

    nb::class_<rocisa::SCmpGtU32, rocisa::CommonInstruction>(m_cmp, "SCmpGtU32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpGtU32& self, const nb::dict&) {
            return new rocisa::SCmpGtU32(self);
        });

    nb::class_<rocisa::SCmpLeI32, rocisa::CommonInstruction>(m_cmp, "SCmpLeI32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLeI32& self, const nb::dict&) {
            return new rocisa::SCmpLeI32(self);
        });

    nb::class_<rocisa::SCmpLeU32, rocisa::CommonInstruction>(m_cmp, "SCmpLeU32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLeU32& self, const nb::dict&) {
            return new rocisa::SCmpLeU32(self);
        });

    nb::class_<rocisa::SCmpLgU32, rocisa::CommonInstruction>(m_cmp, "SCmpLgU32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLgU32& self, const nb::dict&) {
            return new rocisa::SCmpLgU32(self);
        });

    nb::class_<rocisa::SCmpLgI32, rocisa::CommonInstruction>(m_cmp, "SCmpLgI32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLgI32& self, const nb::dict&) {
            return new rocisa::SCmpLgI32(self);
        });

    nb::class_<rocisa::SCmpLgU64, rocisa::CommonInstruction>(m_cmp, "SCmpLgU64")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLgU64& self, const nb::dict&) {
            return new rocisa::SCmpLgU64(self);
        });

    nb::class_<rocisa::SCmpLtI32, rocisa::CommonInstruction>(m_cmp, "SCmpLtI32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLtI32& self, const nb::dict&) {
            return new rocisa::SCmpLtI32(self);
        });

    nb::class_<rocisa::SCmpLtU32, rocisa::CommonInstruction>(m_cmp, "SCmpLtU32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpLtU32& self, const nb::dict&) {
            return new rocisa::SCmpLtU32(self);
        });

    nb::class_<rocisa::SBitcmp1B32, rocisa::CommonInstruction>(m_cmp, "SBitcmp1B32")
        .def(nb::init<const InstructionInput&, const InstructionInput&, const std::string&>(),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SBitcmp1B32& self, const nb::dict&) {
            return new rocisa::SBitcmp1B32(self);
        });

    nb::class_<rocisa::SCmpKEQU32, rocisa::CommonInstruction>(m_cmp, "SCmpKEQU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&, const int, const std::string&>(),
             nb::arg("src"),
             nb::arg("simm16"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpKEQU32& self, const nb::dict&) {
            return new rocisa::SCmpKEQU32(self);
        });

    nb::class_<rocisa::SCmpKGeU32, rocisa::CommonInstruction>(m_cmp, "SCmpKGeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&, const int, const std::string&>(),
             nb::arg("src"),
             nb::arg("simm16"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpKGeU32& self, const nb::dict&) {
            return new rocisa::SCmpKGeU32(self);
        });

    nb::class_<rocisa::SCmpKGtU32, rocisa::CommonInstruction>(m_cmp, "SCmpKGtU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&, const int, const std::string&>(),
             nb::arg("src"),
             nb::arg("simm16"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpKGtU32& self, const nb::dict&) {
            return new rocisa::SCmpKGtU32(self);
        });

    nb::class_<rocisa::SCmpKLGU32, rocisa::CommonInstruction>(m_cmp, "SCmpKLGU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&, const int, const std::string&>(),
             nb::arg("src"),
             nb::arg("simm16"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCmpKLGU32& self, const nb::dict&) {
            return new rocisa::SCmpKLGU32(self);
        });

    nb::class_<rocisa::VCmpEQF32, rocisa::VCmpInstruction>(m_cmp, "VCmpEQF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpEQF32& self, nb::dict&) { return new rocisa::VCmpEQF32(self); });

    nb::class_<rocisa::VCmpEQF64, rocisa::VCmpInstruction>(m_cmp, "VCmpEQF64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpEQF64& self, nb::dict&) { return new rocisa::VCmpEQF64(self); });

    nb::class_<rocisa::VCmpEQU32, rocisa::VCmpInstruction>(m_cmp, "VCmpEQU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpEQU32& self, nb::dict&) { return new rocisa::VCmpEQU32(self); });

    nb::class_<rocisa::VCmpEQI32, rocisa::VCmpInstruction>(m_cmp, "VCmpEQI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpEQI32& self, nb::dict&) { return new rocisa::VCmpEQI32(self); });

    nb::class_<rocisa::VCmpGEF16, rocisa::VCmpInstruction>(m_cmp, "VCmpGEF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGEF16& self, nb::dict&) { return new rocisa::VCmpGEF16(self); });

    nb::class_<rocisa::VCmpGTF16, rocisa::VCmpInstruction>(m_cmp, "VCmpGTF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGTF16& self, nb::dict&) { return new rocisa::VCmpGTF16(self); });

    nb::class_<rocisa::VCmpGEF32, rocisa::VCmpInstruction>(m_cmp, "VCmpGEF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGEF32& self, nb::dict&) { return new rocisa::VCmpGEF32(self); });

    nb::class_<rocisa::VCmpGTF32, rocisa::VCmpInstruction>(m_cmp, "VCmpGTF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGTF32& self, nb::dict&) { return new rocisa::VCmpGTF32(self); });

    nb::class_<rocisa::VCmpGEF64, rocisa::VCmpInstruction>(m_cmp, "VCmpGEF64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGEF64& self, nb::dict&) { return new rocisa::VCmpGEF64(self); });

    nb::class_<rocisa::VCmpGTF64, rocisa::VCmpInstruction>(m_cmp, "VCmpGTF64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGTF64& self, nb::dict&) { return new rocisa::VCmpGTF64(self); });

    nb::class_<rocisa::VCmpGEI32, rocisa::VCmpInstruction>(m_cmp, "VCmpGEI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGEI32& self, nb::dict&) { return new rocisa::VCmpGEI32(self); });

    nb::class_<rocisa::VCmpGTI32, rocisa::VCmpInstruction>(m_cmp, "VCmpGTI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGTI32& self, nb::dict&) { return new rocisa::VCmpGTI32(self); });

    nb::class_<rocisa::VCmpGEU32, rocisa::VCmpInstruction>(m_cmp, "VCmpGEU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGEU32& self, nb::dict&) { return new rocisa::VCmpGEU32(self); });

    nb::class_<rocisa::VCmpGtU32, rocisa::VCmpInstruction>(m_cmp, "VCmpGtU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpGtU32& self, nb::dict&) { return new rocisa::VCmpGtU32(self); });

    nb::class_<rocisa::VCmpLeU32, rocisa::VCmpInstruction>(m_cmp, "VCmpLeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpLeU32& self, nb::dict&) { return new rocisa::VCmpLeU32(self); });

    nb::class_<rocisa::VCmpLeI32, rocisa::VCmpInstruction>(m_cmp, "VCmpLeI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpLeI32& self, nb::dict&) { return new rocisa::VCmpLeI32(self); });

    nb::class_<rocisa::VCmpLtI32, rocisa::VCmpInstruction>(m_cmp, "VCmpLtI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpLtI32& self, nb::dict&) { return new rocisa::VCmpLtI32(self); });

    nb::class_<rocisa::VCmpLtU32, rocisa::VCmpInstruction>(m_cmp, "VCmpLtU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpLtU32& self, nb::dict&) { return new rocisa::VCmpLtU32(self); });

    nb::class_<rocisa::VCmpUF32, rocisa::VCmpInstruction>(m_cmp, "VCmpUF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpUF32& self, nb::dict&) { return new rocisa::VCmpUF32(self); });

    nb::class_<rocisa::VCmpNeI32, rocisa::VCmpInstruction>(m_cmp, "VCmpNeI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpNeI32& self, nb::dict&) { return new rocisa::VCmpNeI32(self); });

    nb::class_<rocisa::VCmpNeU32, rocisa::VCmpInstruction>(m_cmp, "VCmpNeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpNeU32& self, nb::dict&) { return new rocisa::VCmpNeU32(self); });

    nb::class_<rocisa::VCmpNeU64, rocisa::VCmpInstruction>(m_cmp, "VCmpNeU64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VCmpNeU64& self, nb::dict&) { return new rocisa::VCmpNeU64(self); });

    nb::class_<rocisa::VCmpClassF32, rocisa::VCmpInstruction>(m_cmp, "VCmpClassF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpClassF32& self, nb::dict&) {
            return new rocisa::VCmpClassF32(self);
        });

    nb::class_<rocisa::VCmpXClassF32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXClassF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXClassF32& self, nb::dict&) {
            return new rocisa::VCmpXClassF32(self);
        });

    nb::class_<rocisa::VCmpXEqU32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXEqU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXEqU32& self, nb::dict&) {
            return new rocisa::VCmpXEqU32(self);
        });

    nb::class_<rocisa::VCmpXGeU32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXGeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXGeU32& self, nb::dict&) {
            return new rocisa::VCmpXGeU32(self);
        });

    nb::class_<rocisa::VCmpXGtU32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXGtU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXGtU32& self, nb::dict&) {
            return new rocisa::VCmpXGtU32(self);
        });

    nb::class_<rocisa::VCmpXLeU32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXLeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXLeU32& self, nb::dict&) {
            return new rocisa::VCmpXLeU32(self);
        });

    nb::class_<rocisa::VCmpXLeI32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXLeI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXLeI32& self, nb::dict&) {
            return new rocisa::VCmpXLeI32(self);
        });

    nb::class_<rocisa::VCmpXLtF32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXLtF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXLtF32& self, nb::dict&) {
            return new rocisa::VCmpXLtF32(self);
        });

    nb::class_<rocisa::VCmpXLtI32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXLtI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXLtI32& self, nb::dict&) {
            return new rocisa::VCmpXLtI32(self);
        });

    nb::class_<rocisa::VCmpXLtU32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXLtU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXLtU32& self, nb::dict&) {
            return new rocisa::VCmpXLtU32(self);
        });

    nb::class_<rocisa::VCmpXLtU64, rocisa::VCmpXInstruction>(m_cmp, "VCmpXLtU64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXLtU64& self, nb::dict&) {
            return new rocisa::VCmpXLtU64(self);
        });

    nb::class_<rocisa::VCmpXNeU16, rocisa::VCmpXInstruction>(m_cmp, "VCmpXNeU16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXNeU16& self, nb::dict&) {
            return new rocisa::VCmpXNeU16(self);
        });

    nb::class_<rocisa::VCmpXNeU32, rocisa::VCmpXInstruction>(m_cmp, "VCmpXNeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCmpXNeU32& self, nb::dict&) {
            return new rocisa::VCmpXNeU32(self);
        });
}
