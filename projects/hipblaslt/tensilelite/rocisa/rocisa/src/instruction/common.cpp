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
#include "instruction/common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;

void common_inst(nb::module_ m_common)
{
    nb::class_<rocisa::SAbsI32, rocisa::CommonInstruction>(m_common, "SAbsI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAbsI32& self, const nb::dict&) {
            return new rocisa::SAbsI32(self);
        });

    nb::class_<rocisa::SMaxI32, rocisa::CommonInstruction>(m_common, "SMaxI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMaxI32& self, const nb::dict&) {
            return new rocisa::SMaxI32(self);
        });

    nb::class_<rocisa::SMaxU32, rocisa::CommonInstruction>(m_common, "SMaxU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMaxU32& self, const nb::dict&) {
            return new rocisa::SMaxU32(self);
        });

    nb::class_<rocisa::SMinI32, rocisa::CommonInstruction>(m_common, "SMinI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMinI32& self, const nb::dict&) {
            return new rocisa::SMinI32(self);
        });

    nb::class_<rocisa::SMinU32, rocisa::CommonInstruction>(m_common, "SMinU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMinU32& self, const nb::dict&) {
            return new rocisa::SMinU32(self);
        });

    nb::class_<rocisa::SAddI32, rocisa::CommonInstruction>(m_common, "SAddI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAddI32& self, const nb::dict&) {
            return new rocisa::SAddI32(self);
        });

    nb::class_<rocisa::SAddU32, rocisa::CommonInstruction>(m_common, "SAddU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAddU32& self, const nb::dict&) {
            return new rocisa::SAddU32(self);
        });

    nb::class_<rocisa::SAddCU32, rocisa::CommonInstruction>(m_common, "SAddCU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAddCU32& self, const nb::dict&) {
            return new rocisa::SAddCU32(self);
        });

    nb::class_<rocisa::SMulI32, rocisa::CommonInstruction>(m_common, "SMulI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMulI32& self, const nb::dict&) {
            return new rocisa::SMulI32(self);
        });

    nb::class_<rocisa::SMulHII32, rocisa::CommonInstruction>(m_common, "SMulHII32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMulHII32& self, const nb::dict&) {
            return new rocisa::SMulHII32(self);
        });

    nb::class_<rocisa::SMulHIU32, rocisa::CommonInstruction>(m_common, "SMulHIU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMulHIU32& self, const nb::dict&) {
            return new rocisa::SMulHIU32(self);
        });

    nb::class_<rocisa::SMulLOU32, rocisa::CommonInstruction>(m_common, "SMulLOU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SMulLOU32& self, const nb::dict&) {
            return new rocisa::SMulLOU32(self);
        });

    nb::class_<rocisa::SSubI32, rocisa::CommonInstruction>(m_common, "SSubI32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSubI32& self, const nb::dict&) {
            return new rocisa::SSubI32(self);
        });

    nb::class_<rocisa::SSubU32, rocisa::CommonInstruction>(m_common, "SSubU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSubU32& self, const nb::dict&) {
            return new rocisa::SSubU32(self);
        });

    nb::class_<rocisa::SSubBU32, rocisa::CommonInstruction>(m_common, "SSubBU32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSubBU32& self, const nb::dict&) {
            return new rocisa::SSubBU32(self);
        });

    nb::class_<rocisa::SCSelectB32, rocisa::CommonInstruction>(m_common, "SCSelectB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCSelectB32& self, const nb::dict&) {
            return new rocisa::SCSelectB32(self);
        });

    nb::class_<rocisa::SCSelectB64, rocisa::CommonInstruction>(m_common, "SCSelectB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SCSelectB64& self, const nb::dict&) {
            return new rocisa::SCSelectB64(self);
        });

    nb::class_<rocisa::SAndB32, rocisa::CommonInstruction>(m_common, "SAndB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAndB32& self, const nb::dict&) {
            return new rocisa::SAndB32(self);
        });

    nb::class_<rocisa::SAndB64, rocisa::CommonInstruction>(m_common, "SAndB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAndB64& self, const nb::dict&) {
            return new rocisa::SAndB64(self);
        });

    nb::class_<rocisa::SAndN2B32, rocisa::CommonInstruction>(m_common, "SAndN2B32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAndN2B32& self, const nb::dict&) {
            return new rocisa::SAndN2B32(self);
        });

    nb::class_<rocisa::SOrB32, rocisa::CommonInstruction>(m_common, "SOrB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SOrB32& self, const nb::dict&) { return new rocisa::SOrB32(self); });

    nb::class_<rocisa::SXorB32, rocisa::CommonInstruction>(m_common, "SXorB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SXorB32& self, const nb::dict&) {
            return new rocisa::SXorB32(self);
        });

    nb::class_<rocisa::SOrB64, rocisa::CommonInstruction>(m_common, "SOrB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SOrB64& self, const nb::dict&) { return new rocisa::SOrB64(self); });

    nb::class_<rocisa::SGetPCB64, rocisa::CommonInstruction>(m_common, "SGetPCB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&, const std::string&>(),
             nb::arg("dst"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SGetPCB64& self, nb::dict&) { return new rocisa::SGetPCB64(self); });

    nb::class_<rocisa::SLShiftLeftB32, rocisa::CommonInstruction>(m_common, "SLShiftLeftB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SLShiftLeftB32& self, nb::dict&) {
            return new rocisa::SLShiftLeftB32(self);
        });

    nb::class_<rocisa::SLShiftRightB32, rocisa::CommonInstruction>(m_common, "SLShiftRightB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")

        .def("__deepcopy__", [](const rocisa::SLShiftRightB32& self, nb::dict&) {
            return new rocisa::SLShiftRightB32(self);
        });

    nb::class_<rocisa::SLShiftLeftB64, rocisa::CommonInstruction>(m_common, "SLShiftLeftB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")

        .def("__deepcopy__", [](const rocisa::SLShiftLeftB64& self, nb::dict&) {
            return new rocisa::SLShiftLeftB64(self);
        });

    nb::class_<rocisa::SLShiftRightB64, rocisa::CommonInstruction>(m_common, "SLShiftRightB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SLShiftRightB64& self, nb::dict&) {
            return new rocisa::SLShiftRightB64(self);
        });

    nb::class_<rocisa::SAShiftRightI32, rocisa::CommonInstruction>(m_common, "SAShiftRightI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAShiftRightI32& self, nb::dict&) {
            return new rocisa::SAShiftRightI32(self);
        });

    nb::class_<rocisa::SLShiftLeft1AddU32, rocisa::CommonInstruction>(m_common,
                                                                      "SLShiftLeft1AddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SLShiftLeft1AddU32& self) {
            return new rocisa::SLShiftLeft1AddU32(self);
        });

    nb::class_<rocisa::SLShiftLeft2AddU32, rocisa::CommonInstruction>(m_common,
                                                                      "SLShiftLeft2AddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SLShiftLeft2AddU32& self) {
            return new rocisa::SLShiftLeft2AddU32(self);
        });

    nb::class_<rocisa::SLShiftLeft3AddU32, rocisa::CommonInstruction>(m_common,
                                                                      "SLShiftLeft3AddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SLShiftLeft3AddU32& self) {
            return new rocisa::SLShiftLeft3AddU32(self);
        });

    nb::class_<rocisa::SLShiftLeft4AddU32, rocisa::CommonInstruction>(m_common,
                                                                      "SLShiftLeft4AddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SLShiftLeft4AddU32& self) {
            return new rocisa::SLShiftLeft4AddU32(self);
        });

    nb::class_<rocisa::SSetMask, rocisa::CommonInstruction>(m_common, "SSetMask")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SSetMask& self, nb::dict&) { return new rocisa::SSetMask(self); });

    nb::class_<rocisa::SMovB32, rocisa::CommonInstruction>(m_common, "SMovB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SMovB32& self, nb::dict&) { return new rocisa::SMovB32(self); });

    nb::class_<rocisa::SMovB64, rocisa::CommonInstruction>(m_common, "SMovB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SMovB64& self, nb::dict&) { return new rocisa::SMovB64(self); });

    nb::class_<rocisa::SCMovB32, rocisa::CommonInstruction>(m_common, "SCMovB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SCMovB32& self, nb::dict&) { return new rocisa::SCMovB32(self); });

    nb::class_<rocisa::SCMovB64, rocisa::CommonInstruction>(m_common, "SCMovB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SCMovB64& self, nb::dict&) { return new rocisa::SCMovB64(self); });

    nb::class_<rocisa::SFf1B32, rocisa::CommonInstruction>(m_common, "SFf1B32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SFf1B32& self, nb::dict&) { return new rocisa::SFf1B32(self); });

    nb::class_<rocisa::SBfmB32, rocisa::CommonInstruction>(m_common, "SBfmB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SBfmB32& self, nb::dict&) { return new rocisa::SBfmB32(self); });

    nb::class_<rocisa::SFlbitI32B32, rocisa::CommonInstruction>(m_common, "SFlbitI32B32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SFlbitI32B32& self, nb::dict&) { return new rocisa::SFlbitI32B32(self); });

    nb::class_<rocisa::SMovkI32, rocisa::CommonInstruction>(m_common, "SMovkI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SMovkI32& self, nb::dict&) { return new rocisa::SMovkI32(self); });

    nb::class_<rocisa::SSExtI16toI32, rocisa::CommonInstruction>(m_common, "SSExtI16toI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSExtI16toI32& self, nb::dict&) {
            return new rocisa::SSExtI16toI32(self);
        });

    nb::class_<rocisa::SAndSaveExecB32, rocisa::CommonInstruction>(m_common, "SAndSaveExecB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAndSaveExecB32& self, nb::dict&) {
            return new rocisa::SAndSaveExecB32(self);
        });

    nb::class_<rocisa::SAndSaveExecB64, rocisa::CommonInstruction>(m_common, "SAndSaveExecB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAndSaveExecB64& self, nb::dict&) {
            return new rocisa::SAndSaveExecB64(self);
        });

    nb::class_<rocisa::SOrSaveExecB32, rocisa::CommonInstruction>(m_common, "SOrSaveExecB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SOrSaveExecB32& self, nb::dict&) {
            return new rocisa::SOrSaveExecB32(self);
        });

    nb::class_<rocisa::SOrSaveExecB64, rocisa::CommonInstruction>(m_common, "SOrSaveExecB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SOrSaveExecB64& self, nb::dict&) {
            return new rocisa::SOrSaveExecB64(self);
        });

    nb::class_<rocisa::SSetPrior, rocisa::Instruction>(m_common, "SSetPrior")
        .def(nb::init<int, const std::string&>(), nb::arg("prior"), nb::arg("comment") = "")
        .def("getParams", &rocisa::SSetPrior::getParams)
        .def("__str__", &rocisa::SSetPrior::toString)
        .def("__deepcopy__",
             [](const rocisa::SSetPrior& self, nb::dict&) { return new rocisa::SSetPrior(self); });

    nb::class_<rocisa::SBarrier, rocisa::Instruction>(m_common, "SBarrier")
        .def(nb::init<const std::string&>(), nb::arg("comment") = "")
        .def("getParams", &rocisa::SBarrier::getParams)
        .def("__str__", &rocisa::SBarrier::toString)
        .def("__deepcopy__",
             [](const rocisa::SBarrier& self, nb::dict&) { return new rocisa::SBarrier(self); });

    nb::class_<rocisa::SDcacheWb, rocisa::Instruction>(m_common, "SDcacheWb")
        .def(nb::init<const std::string&>(), nb::arg("comment") = "")
        .def("getParams", &rocisa::SDcacheWb::getParams)
        .def("__str__", &rocisa::SDcacheWb::toString)
        .def("__deepcopy__",
             [](const rocisa::SDcacheWb& self, nb::dict&) { return new rocisa::SDcacheWb(self); });

    nb::class_<rocisa::SNop, rocisa::Instruction>(m_common, "SNop")
        .def(nb::init<int, const std::string&>(), nb::arg("waitState"), nb::arg("comment") = "")
        .def("getParams", &rocisa::SNop::getParams)
        .def("__str__", &rocisa::SNop::toString)
        .def("__deepcopy__",
             [](const rocisa::SNop& self, nb::dict&) { return new rocisa::SNop(self); });

    nb::class_<rocisa::SEndpgm, rocisa::Instruction>(m_common, "SEndpgm")
        .def(nb::init<const std::string&>(), nb::arg("comment") = "")
        .def("getParams", &rocisa::SEndpgm::getParams)
        .def("__str__", &rocisa::SEndpgm::toString)
        .def("__deepcopy__",
             [](const rocisa::SEndpgm& self, nb::dict&) { return new rocisa::SEndpgm(self); });

    nb::class_<rocisa::SSleep, rocisa::Instruction>(m_common, "SSleep")
        .def(nb::init<const int, const std::string&>(), nb::arg("simm16"), nb::arg("comment") = "")
        .def("getParams", &rocisa::SSleep::getParams)
        .def("__str__", &rocisa::SSleep::toString)
        .def("__deepcopy__",
             [](const rocisa::SSleep& self, nb::dict&) { return new rocisa::SSleep(self); });

    nb::class_<rocisa::SGetRegB32, rocisa::CommonInstruction>(m_common, "SGetRegB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SGetRegB32& self, nb::dict&) {
            return new rocisa::SGetRegB32(self);
        });

    nb::class_<rocisa::SSetRegB32, rocisa::CommonInstruction>(m_common, "SSetRegB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSetRegB32& self, nb::dict&) {
            return new rocisa::SSetRegB32(self);
        });

    nb::class_<rocisa::SSetRegIMM32B32, rocisa::CommonInstruction>(m_common, "SSetRegIMM32B32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SSetRegIMM32B32& self, nb::dict&) {
            return new rocisa::SSetRegIMM32B32(self);
        });

    nb::class_<rocisa::_SWaitCnt, rocisa::Instruction>(m_common, "_SWaitCnt")
        .def(nb::init<int, int, const std::string&>(),
             nb::arg("lgkmcnt") = -1,
             nb::arg("vmcnt")   = -1,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::_SWaitCnt::getParams)
        .def("__str__", &rocisa::_SWaitCnt::toString)
        .def("__deepcopy__",
             [](const rocisa::_SWaitCnt& self, nb::dict&) { return new rocisa::_SWaitCnt(self); });

    nb::class_<rocisa::_SWaitCntVscnt, rocisa::Instruction>(m_common, "_SWaitCntVscnt")
        .def(nb::init<int, const std::string&>(), nb::arg("vscnt") = -1, nb::arg("comment") = "")
        .def("getParams", &rocisa::_SWaitCntVscnt::getParams)
        .def("__str__", &rocisa::_SWaitCntVscnt::toString)
        .def("__deepcopy__", [](const rocisa::_SWaitCntVscnt& self, nb::dict&) {
            return new rocisa::_SWaitCntVscnt(self);
        });

    nb::class_<rocisa::_SWaitStorecnt, rocisa::Instruction>(m_common, "_SWaitStorecnt")
        .def(nb::init<int, const std::string&>(), nb::arg("storecnt") = -1, nb::arg("comment") = "")
        .def("getParams", &rocisa::_SWaitStorecnt::getParams)
        .def("__str__", &rocisa::_SWaitStorecnt::toString)
        .def("__deepcopy__", [](const rocisa::_SWaitStorecnt& self, nb::dict&) {
            return new rocisa::_SWaitStorecnt(self);
        });

    nb::class_<rocisa::_SWaitLoadcnt, rocisa::Instruction>(m_common, "_SWaitLoadcnt")
        .def(nb::init<int, const std::string&>(), nb::arg("loadcnt") = -1, nb::arg("comment") = "")
        .def("getParams", &rocisa::_SWaitLoadcnt::getParams)
        .def("__str__", &rocisa::_SWaitLoadcnt::toString)
        .def("__deepcopy__", [](const rocisa::_SWaitLoadcnt& self, nb::dict&) {
            return new rocisa::_SWaitLoadcnt(self);
        });

    nb::class_<rocisa::_SWaitKMcnt, rocisa::Instruction>(m_common, "_SWaitKMcnt")
        .def(nb::init<int, const std::string&>(), nb::arg("kmcnt") = -1, nb::arg("comment") = "")
        .def("getParams", &rocisa::_SWaitKMcnt::getParams)
        .def("__str__", &rocisa::_SWaitKMcnt::toString)
        .def("__deepcopy__", [](const rocisa::_SWaitKMcnt& self, nb::dict&) {
            return new rocisa::_SWaitKMcnt(self);
        });

    nb::class_<rocisa::_SWaitDscnt, rocisa::Instruction>(m_common, "_SWaitDscnt")
        .def(nb::init<int, const std::string&>(), nb::arg("dscnt") = -1, nb::arg("comment") = "")
        .def("getParams", &rocisa::_SWaitDscnt::getParams)
        .def("__str__", &rocisa::_SWaitDscnt::toString)
        .def("__deepcopy__", [](const rocisa::_SWaitDscnt& self, nb::dict&) {
            return new rocisa::_SWaitDscnt(self);
        });

    nb::class_<rocisa::SWaitCnt, rocisa::CompositeInstruction>(m_common, "SWaitCnt")
        .def(nb::init<int, int, int, int, const std::string&, bool>(),
             nb::arg("vlcnt")    = -1,
             nb::arg("vscnt")    = -1,
             nb::arg("dscnt")    = -1,
             nb::arg("kmcnt")    = -1,
             nb::arg("comment")  = "",
             nb::arg("waitAll")  = false)
        .def_rw("vlcnt", &rocisa::SWaitCnt::vlcnt)
        .def_rw("vscnt", &rocisa::SWaitCnt::vscnt)
        .def_rw("dscnt", &rocisa::SWaitCnt::dscnt)
        .def_rw("kmcnt", &rocisa::SWaitCnt::kmcnt)
        .def_rw("comment", &rocisa::SWaitCnt::comment)
        .def("__deepcopy__",
             [](const rocisa::SWaitCnt& self, nb::dict&) { return new rocisa::SWaitCnt(self); });

    nb::class_<rocisa::SWaitAlu, rocisa::Instruction>(m_common, "SWaitAlu")
        .def(nb::init<int, int, int, int, int, int, int, const std::string&>(),
             nb::arg("va_vdst")  = -1,
             nb::arg("va_sdst")  = -1,
             nb::arg("va_ssrc")  = -1,
             nb::arg("hold_cnt") = -1,
             nb::arg("vm_vsrc")  = -1,
             nb::arg("va_vcc")   = -1,
             nb::arg("sa_sdst")  = -1,
             nb::arg("comment")  = "")
        .def("getParams", &rocisa::SWaitAlu::getParams)
        .def("__deepcopy__",
             [](const rocisa::SWaitAlu& self, nb::dict&) { return new rocisa::SWaitAlu(self); });

    nb::class_<rocisa::SDelayAlu, rocisa::Instruction>(m_common, "SDelayAlu")
        .def(nb::init<rocisa::DelayALUType,
                      int,
                      std::optional<int>,
                      std::optional<rocisa::DelayALUType>,
                      std::optional<int>,
                      const std::string&>(),
             nb::arg("instid0type"),
             nb::arg("instid0cnt"),
             nb::arg("instskipCnt") = std::nullopt,
             nb::arg("instid1type") = std::nullopt,
             nb::arg("instid1cnt")  = std::nullopt,
             nb::arg("comment")     = "")
        .def("getParams", &rocisa::SDelayAlu::getParams)
        .def("__deepcopy__",
             [](const rocisa::SDelayAlu& self, nb::dict&) { return new rocisa::SDelayAlu(self); });

    nb::class_<rocisa::VAddF16, rocisa::CommonInstruction>(m_common, "VAddF16")
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
             [](const rocisa::VAddF16& self, nb::dict&) { return new rocisa::VAddF16(self); });

    nb::class_<rocisa::VAddF32, rocisa::CommonInstruction>(m_common, "VAddF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::DPPModifiers>,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("dpp")     = std::nullopt,
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAddF32& self, nb::dict&) { return new rocisa::VAddF32(self); });

    nb::class_<rocisa::VAddF64, rocisa::CommonInstruction>(m_common, "VAddF64")
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
             [](const rocisa::VAddF64& self, nb::dict&) { return new rocisa::VAddF64(self); });

    nb::class_<rocisa::VAddI32, rocisa::CommonInstruction>(m_common, "VAddI32")
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
             [](const rocisa::VAddI32& self, nb::dict&) { return new rocisa::VAddI32(self); });

    nb::class_<rocisa::VAddU32, rocisa::CommonInstruction>(m_common, "VAddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAddU32& self, nb::dict&) { return new rocisa::VAddU32(self); });

    nb::class_<rocisa::VAddCOU32, rocisa::CommonInstruction>(m_common, "VAddCOU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("dst1"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAddCOU32& self, nb::dict&) { return new rocisa::VAddCOU32(self); });

    nb::class_<rocisa::VAddCCOU32, rocisa::CommonInstruction>(m_common, "VAddCCOU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("dst1"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VAddCCOU32& self, nb::dict&) {
            return new rocisa::VAddCCOU32(self);
        });

    nb::class_<rocisa::VAddPKF16, rocisa::CommonInstruction>(m_common, "VAddPKF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAddPKF16& self, nb::dict&) { return new rocisa::VAddPKF16(self); });

    nb::class_<rocisa::_VAddPKF32, rocisa::CommonInstruction>(m_common, "_VAddPKF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::_VAddPKF32& self, nb::dict&) {
            return new rocisa::_VAddPKF32(self);
        });

    nb::class_<rocisa::VAddPKF32, rocisa::CompositeInstruction>(m_common, "VAddPKF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAddPKF32& self, nb::dict&) { return new rocisa::VAddPKF32(self); });

    nb::class_<rocisa::VAdd3U32, rocisa::CommonInstruction>(m_common, "VAdd3U32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAdd3U32& self, nb::dict&) { return new rocisa::VAdd3U32(self); });

    nb::class_<rocisa::VMulF16, rocisa::CommonInstruction>(m_common, "VMulF16")
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
             [](const rocisa::VMulF16& self, nb::dict&) { return new rocisa::VMulF16(self); });

    nb::class_<rocisa::VMulF32, rocisa::CommonInstruction>(m_common, "VMulF32")
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
             [](const rocisa::VMulF32& self, nb::dict&) { return new rocisa::VMulF32(self); });

    nb::class_<rocisa::VMulF64, rocisa::CommonInstruction>(m_common, "VMulF64")
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
             [](const rocisa::VMulF64& self, nb::dict&) { return new rocisa::VMulF64(self); });

    nb::class_<rocisa::VMulPKF16, rocisa::CommonInstruction>(m_common, "VMulPKF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMulPKF16& self, nb::dict&) { return new rocisa::VMulPKF16(self); });

    nb::class_<rocisa::VMulPKF32S, rocisa::CommonInstruction>(m_common, "VMulPKF32S")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VMulPKF32S& self, nb::dict&) {
            return new rocisa::VMulPKF32S(self);
        });

    nb::class_<rocisa::_VMulPKF32, rocisa::CommonInstruction>(m_common, "_VMulPKF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::_VMulPKF32& self, nb::dict&) {
            return new rocisa::_VMulPKF32(self);
        });

    nb::class_<rocisa::VMulPKF32, rocisa::CompositeInstruction>(m_common, "VMulPKF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMulPKF32& self, nb::dict&) { return new rocisa::VMulPKF32(self); });

    nb::class_<rocisa::VMulLOU32, rocisa::CommonInstruction>(m_common, "VMulLOU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMulLOU32& self, nb::dict&) { return new rocisa::VMulLOU32(self); });

    nb::class_<rocisa::VMulHII32, rocisa::CommonInstruction>(m_common, "VMulHII32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMulHII32& self, nb::dict&) { return new rocisa::VMulHII32(self); });

    nb::class_<rocisa::VMulHIU32, rocisa::CommonInstruction>(m_common, "VMulHIU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMulHIU32& self, nb::dict&) { return new rocisa::VMulHIU32(self); });

    nb::class_<rocisa::VMulI32I24, rocisa::CommonInstruction>(m_common, "VMulI32I24")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VMulI32I24& self, nb::dict&) {
            return new rocisa::VMulI32I24(self);
        });

    nb::class_<rocisa::VMulU32U24, rocisa::CommonInstruction>(m_common, "VMulU32U24")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VMulU32U24& self, nb::dict&) {
            return new rocisa::VMulU32U24(self);
        });

    nb::class_<rocisa::VSubF32, rocisa::CommonInstruction>(m_common, "VSubF32")
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
             [](const rocisa::VSubF32& self, nb::dict&) { return new rocisa::VSubF32(self); });

    nb::class_<rocisa::VSubI32, rocisa::CommonInstruction>(m_common, "VSubI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VSubI32& self, nb::dict&) { return new rocisa::VSubI32(self); });

    nb::class_<rocisa::VSubU32, rocisa::CommonInstruction>(m_common, "VSubU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VSubU32& self, nb::dict&) { return new rocisa::VSubU32(self); });

    nb::class_<rocisa::VSubCoU32, rocisa::CommonInstruction>(m_common, "VSubCoU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("dst1"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VSubCoU32& self, nb::dict&) { return new rocisa::VSubCoU32(self); });

    nb::class_<rocisa::VMacF32, rocisa::CommonInstruction>(m_common, "VMacF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMacF32& self, nb::dict&) { return new rocisa::VMacF32(self); });

    nb::class_<rocisa::VDot2CF32F16, rocisa::CommonInstruction>(m_common, "VDot2CF32F16")
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
        .def("__deepcopy__", [](const rocisa::VDot2CF32F16& self, nb::dict&) {
            return new rocisa::VDot2CF32F16(self);
        });

    nb::class_<rocisa::VDot2CF32BF16, rocisa::CommonInstruction>(m_common, "VDot2CF32BF16")
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
        .def("__deepcopy__", [](const rocisa::VDot2CF32BF16& self, nb::dict&) {
            return new rocisa::VDot2CF32BF16(self);
        });

    nb::class_<rocisa::VDot2F32F16, rocisa::CommonInstruction>(m_common, "VDot2F32F16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VDot2F32F16& self, nb::dict&) {
            return new rocisa::VDot2F32F16(self);
        });

    nb::class_<rocisa::VDot2F32BF16, rocisa::CommonInstruction>(m_common, "VDot2F32BF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VDot2F32BF16& self, nb::dict&) {
            return new rocisa::VDot2F32BF16(self);
        });

    nb::class_<rocisa::VFmaF16, rocisa::CommonInstruction>(m_common, "VFmaF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VFmaF16& self, nb::dict&) { return new rocisa::VFmaF16(self); });

    nb::class_<rocisa::VFmaF32, rocisa::CommonInstruction>(m_common, "VFmaF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VFmaF32& self, nb::dict&) { return new rocisa::VFmaF32(self); });

    nb::class_<rocisa::VFmaF64, rocisa::CommonInstruction>(m_common, "VFmaF64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VFmaF64& self, nb::dict&) { return new rocisa::VFmaF64(self); });

    nb::class_<rocisa::VFmaPKF16, rocisa::CommonInstruction>(m_common, "VFmaPKF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VFmaPKF16& self, nb::dict&) { return new rocisa::VFmaPKF16(self); });

    nb::class_<rocisa::VFmaMixF32, rocisa::CommonInstruction>(m_common, "VFmaMixF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VFmaMixF32& self, nb::dict&) {
            return new rocisa::VFmaMixF32(self);
        });

    nb::class_<rocisa::VMadI32I24, rocisa::CommonInstruction>(m_common, "VMadI32I24")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VMadI32I24& self, nb::dict&) {
            return new rocisa::VMadI32I24(self);
        });

    nb::class_<rocisa::VMadU32U24, rocisa::CommonInstruction>(m_common, "VMadU32U24")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VMadU32U24& self, nb::dict&) {
            return new rocisa::VMadU32U24(self);
        });

    nb::class_<rocisa::VMadMixF32, rocisa::CommonInstruction>(m_common, "VMadMixF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VMadMixF32& self, nb::dict&) {
            return new rocisa::VMadMixF32(self);
        });

    nb::class_<rocisa::VExpF16, rocisa::CommonInstruction>(m_common, "VExpF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VExpF16& self, nb::dict&) { return new rocisa::VExpF16(self); });

    nb::class_<rocisa::VExpF32, rocisa::CommonInstruction>(m_common, "VExpF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VExpF32& self, nb::dict&) { return new rocisa::VExpF32(self); });

    nb::class_<rocisa::VRcpF16, rocisa::CommonInstruction>(m_common, "VRcpF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VRcpF16& self, nb::dict&) { return new rocisa::VRcpF16(self); });

    nb::class_<rocisa::VRcpF32, rocisa::CommonInstruction>(m_common, "VRcpF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VRcpF32& self, nb::dict&) { return new rocisa::VRcpF32(self); });

    nb::class_<rocisa::VRcpIFlagF32, rocisa::CommonInstruction>(m_common, "VRcpIFlagF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VRcpIFlagF32& self, nb::dict&) {
            return new rocisa::VRcpIFlagF32(self);
        });

    nb::class_<rocisa::VRcpF64, rocisa::CommonInstruction>(m_common, "VRcpF64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VRcpF64& self, nb::dict&) { return new rocisa::VRcpF64(self); });

    nb::class_<rocisa::VRsqF16, rocisa::CommonInstruction>(m_common, "VRsqF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VRsqF16& self, nb::dict&) { return new rocisa::VRsqF16(self); });

    nb::class_<rocisa::VRsqF32, rocisa::CommonInstruction>(m_common, "VRsqF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VRsqF32& self, nb::dict&) { return new rocisa::VRsqF32(self); });

    nb::class_<rocisa::VRsqIFlagF32, rocisa::CommonInstruction>(m_common, "VRsqIFlagF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VRsqIFlagF32& self, nb::dict&) {
            return new rocisa::VRsqIFlagF32(self);
        });

    nb::class_<rocisa::VMaxF16, rocisa::CommonInstruction>(m_common, "VMaxF16")
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
             [](const rocisa::VMaxF16& self, nb::dict&) { return new rocisa::VMaxF16(self); });

    nb::class_<rocisa::VMaxF32, rocisa::CommonInstruction>(m_common, "VMaxF32")
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
             [](const rocisa::VMaxF32& self, nb::dict&) { return new rocisa::VMaxF32(self); });

    nb::class_<rocisa::VMaxF64, rocisa::CommonInstruction>(m_common, "VMaxF64")
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
             [](const rocisa::VMaxF64& self, nb::dict&) { return new rocisa::VMaxF64(self); });

    nb::class_<rocisa::VMaxI32, rocisa::CommonInstruction>(m_common, "VMaxI32")
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
             [](const rocisa::VMaxI32& self, nb::dict&) { return new rocisa::VMaxI32(self); });

    nb::class_<rocisa::VMaxPKF16, rocisa::CommonInstruction>(m_common, "VMaxPKF16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMaxPKF16& self, nb::dict&) { return new rocisa::VMaxPKF16(self); });

    nb::class_<rocisa::VMed3I32, rocisa::CommonInstruction>(m_common, "VMed3I32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMed3I32& self, nb::dict&) { return new rocisa::VMed3I32(self); });

    nb::class_<rocisa::VMed3F32, rocisa::CommonInstruction>(m_common, "VMed3F32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMed3F32& self, nb::dict&) { return new rocisa::VMed3F32(self); });

    nb::class_<rocisa::VMinF16, rocisa::CommonInstruction>(m_common, "VMinF16")
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
             [](const rocisa::VMinF16& self, nb::dict&) { return new rocisa::VMinF16(self); });

    nb::class_<rocisa::VMinF32, rocisa::CommonInstruction>(m_common, "VMinF32")
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
             [](const rocisa::VMinF32& self, nb::dict&) { return new rocisa::VMinF32(self); });

    nb::class_<rocisa::VMinF64, rocisa::CommonInstruction>(m_common, "VMinF64")
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
             [](const rocisa::VMinF64& self, nb::dict&) { return new rocisa::VMinF64(self); });

    nb::class_<rocisa::VMinI32, rocisa::CommonInstruction>(m_common, "VMinI32")
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
             [](const rocisa::VMinI32& self, nb::dict&) { return new rocisa::VMinI32(self); });

    nb::class_<rocisa::VAndB32, rocisa::CommonInstruction>(m_common, "VAndB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAndB32& self, nb::dict&) { return new rocisa::VAndB32(self); });

    nb::class_<rocisa::VAndOrB32, rocisa::CommonInstruction>(m_common, "VAndOrB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VAndOrB32& self, nb::dict&) { return new rocisa::VAndOrB32(self); });

    nb::class_<rocisa::VNotB32, rocisa::CommonInstruction>(m_common, "VNotB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VNotB32& self, nb::dict&) { return new rocisa::VNotB32(self); });

    nb::class_<rocisa::VOrB32, rocisa::CommonInstruction>(m_common, "VOrB32")
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
             [](const rocisa::VOrB32& self, nb::dict&) { return new rocisa::VOrB32(self); });

    nb::class_<rocisa::VXorB32, rocisa::CommonInstruction>(m_common, "VXorB32")
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
             [](const rocisa::VXorB32& self, nb::dict&) { return new rocisa::VXorB32(self); });

    nb::class_<rocisa::VPrngB32, rocisa::CommonInstruction>(m_common, "VPrngB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VPrngB32& self, nb::dict&) { return new rocisa::VPrngB32(self); });

    nb::class_<rocisa::VCndMaskB32, rocisa::CommonInstruction>(m_common, "VCndMaskB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SDWAModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2")    = std::make_shared<rocisa::VCC>(),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VCndMaskB32& self, nb::dict&) {
            return new rocisa::VCndMaskB32(self);
        });

    nb::class_<rocisa::VLShiftLeftB16, rocisa::CommonInstruction>(m_common, "VLShiftLeftB16")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftLeftB16& self, nb::dict&) {
            return new rocisa::VLShiftLeftB16(self);
        });

    nb::class_<rocisa::VLShiftLeftB32, rocisa::CommonInstruction>(m_common, "VLShiftLeftB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftLeftB32& self, nb::dict&) {
            return new rocisa::VLShiftLeftB32(self);
        });

    nb::class_<rocisa::VLShiftRightB32, rocisa::CommonInstruction>(m_common, "VLShiftRightB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftRightB32& self, nb::dict&) {
            return new rocisa::VLShiftRightB32(self);
        });

    nb::class_<rocisa::VLShiftLeftB64, rocisa::CommonInstruction>(m_common, "VLShiftLeftB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftLeftB64& self, nb::dict&) {
            return new rocisa::VLShiftLeftB64(self);
        });

    nb::class_<rocisa::VLShiftRightB64, rocisa::CommonInstruction>(m_common, "VLShiftRightB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftRightB64& self, nb::dict&) {
            return new rocisa::VLShiftRightB64(self);
        });

    nb::class_<rocisa::_VLShiftLeftOrB32, rocisa::CommonInstruction>(m_common, "_VLShiftLeftOrB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::_VLShiftLeftOrB32& self, nb::dict&) {
            return new rocisa::_VLShiftLeftOrB32(self);
        });

    nb::class_<rocisa::VAShiftRightI32, rocisa::CommonInstruction>(m_common, "VAShiftRightI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VAShiftRightI32& self, nb::dict&) {
            return new rocisa::VAShiftRightI32(self);
        });

    nb::class_<rocisa::VLShiftLeftOrB32, rocisa::CompositeInstruction>(m_common, "VLShiftLeftOrB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftLeftOrB32& self, nb::dict&) {
            return new rocisa::VLShiftLeftOrB32(self);
        });

    nb::class_<rocisa::_VAddLShiftLeftU32, rocisa::CommonInstruction>(m_common,
                                                                      "_VAddLShiftLeftU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::_VAddLShiftLeftU32& self, nb::dict&) {
            return new rocisa::_VAddLShiftLeftU32(self);
        });

    nb::class_<rocisa::VAddLShiftLeftU32, rocisa::CompositeInstruction>(m_common,
                                                                        "VAddLShiftLeftU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VAddLShiftLeftU32& self, nb::dict&) {
            return new rocisa::VAddLShiftLeftU32(self);
        });

    nb::class_<rocisa::_VLShiftLeftAddU32, rocisa::CommonInstruction>(m_common,
                                                                      "_VLShiftLeftAddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::_VLShiftLeftAddU32& self, nb::dict&) {
            return new rocisa::_VLShiftLeftAddU32(self);
        });

    nb::class_<rocisa::VLShiftLeftAddU32, rocisa::CompositeInstruction>(m_common,
                                                                        "VLShiftLeftAddU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("shiftHex"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VLShiftLeftAddU32& self, nb::dict&) {
            return new rocisa::VLShiftLeftAddU32(self);
        });

    nb::class_<rocisa::VMovB32, rocisa::CommonInstruction>(m_common, "VMovB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMovB32& self, nb::dict&) { return new rocisa::VMovB32(self); });

    nb::class_<rocisa::_VMovB64, rocisa::CommonInstruction>(m_common, "_VMovB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::_VMovB64& self, nb::dict&) { return new rocisa::_VMovB64(self); });

    nb::class_<rocisa::VMovB64, rocisa::CompositeInstruction>(m_common, "VMovB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VMovB64& self, nb::dict&) { return new rocisa::VMovB64(self); });

    nb::class_<rocisa::VSwapB32, rocisa::CommonInstruction>(m_common, "VSwapB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::optional<rocisa::SDWAModifiers>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VSwapB32& self, nb::dict&) { return new rocisa::VSwapB32(self); });

    nb::class_<rocisa::VBfeI32, rocisa::CommonInstruction>(m_common, "VBfeI32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VBfeI32& self, nb::dict&) { return new rocisa::VBfeI32(self); });

    nb::class_<rocisa::VBfeU32, rocisa::CommonInstruction>(m_common, "VBfeU32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VBfeU32& self, nb::dict&) { return new rocisa::VBfeU32(self); });

    nb::class_<rocisa::VBfiB32, rocisa::CommonInstruction>(m_common, "VBfiB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const InstructionInput&&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VBfiB32& self, nb::dict&) { return new rocisa::VBfiB32(self); });

    nb::class_<rocisa::VPackF16toB32, rocisa::CommonInstruction>(m_common, "VPackF16toB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VPackF16toB32& self, nb::dict&) {
            return new rocisa::VPackF16toB32(self);
        });

    nb::class_<rocisa::VAccvgprReadB32, rocisa::CommonInstruction>(m_common, "VAccvgprReadB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VAccvgprReadB32& self, nb::dict&) {
            return new rocisa::VAccvgprReadB32(self);
        });

    nb::class_<rocisa::VAccvgprWrite, rocisa::CommonInstruction>(m_common, "VAccvgprWrite")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VAccvgprWrite& self, nb::dict&) {
            return new rocisa::VAccvgprWrite(self);
        });

    nb::class_<rocisa::VAccvgprWriteB32, rocisa::CommonInstruction>(m_common, "VAccvgprWriteB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VAccvgprWriteB32& self, nb::dict&) {
            return new rocisa::VAccvgprWriteB32(self);
        });

    nb::class_<rocisa::VReadfirstlaneB32, rocisa::CommonInstruction>(m_common, "VReadfirstlaneB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VReadfirstlaneB32& self, nb::dict&) {
            return new rocisa::VReadfirstlaneB32(self);
        });

    nb::class_<rocisa::VReadlaneB32, rocisa::CommonInstruction>(m_common, "VReadlaneB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VReadlaneB32& self, nb::dict&) {
            return new rocisa::VReadlaneB32(self);
        });

    nb::class_<rocisa::VWritelaneB32, rocisa::CommonInstruction>(m_common, "VWritelaneB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::VWritelaneB32& self, nb::dict&) {
            return new rocisa::VWritelaneB32(self);
        });

    nb::class_<rocisa::VRndneF32, rocisa::CommonInstruction>(m_common, "VRndneF32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VRndneF32& self, nb::dict&) { return new rocisa::VRndneF32(self); });

    nb::class_<rocisa::VPermB32, rocisa::CommonInstruction>(m_common, "VPermB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      const InstructionInput&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("src2"),
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::VPermB32& self, nb::dict&) { return new rocisa::VPermB32(self); });
}
