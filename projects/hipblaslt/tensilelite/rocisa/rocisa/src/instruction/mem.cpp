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
#include "instruction/mem.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;

namespace rocisa
{
    struct PyReadWriteInstruction : public ReadWriteInstruction
    {
        NB_TRAMPOLINE(ReadWriteInstruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };

    struct PyGlobalReadInstruction : public GlobalReadInstruction
    {
        NB_TRAMPOLINE(GlobalReadInstruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };

    struct PyGlobalWriteInstruction : public GlobalWriteInstruction
    {
        NB_TRAMPOLINE(GlobalWriteInstruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };

    struct PyAtomicReadWriteInstruction : public AtomicReadWriteInstruction
    {
        NB_TRAMPOLINE(AtomicReadWriteInstruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };

    struct PyLocalReadInstruction : public LocalReadInstruction
    {
        NB_TRAMPOLINE(LocalReadInstruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };

    struct PyLocalWriteInstruction : public LocalWriteInstruction
    {
        NB_TRAMPOLINE(LocalWriteInstruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };
} // namespace rocisa

void mem_inst(nb::module_ m_mem)
{
    nb::class_<rocisa::ReadWriteInstruction, rocisa::Instruction, rocisa::PyReadWriteInstruction>(
        m_mem, "ReadWriteInstruction")
        .def(nb::init<rocisa::InstType, rocisa::ReadWriteInstruction::RWType, const std::string&>(),
             nb::arg("instType"),
             nb::arg("rwType"),
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::ReadWriteInstruction::issueLatency);

    nb::class_<rocisa::GlobalReadInstruction,
               rocisa::ReadWriteInstruction,
               rocisa::PyGlobalReadInstruction>(m_mem, "GlobalReadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("comment") = "");

    nb::class_<rocisa::FLATReadInstruction, rocisa::GlobalReadInstruction>(m_mem,
                                                                           "FLATReadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::FLATReadInstruction::getParams)
        .def("__str__", &rocisa::FLATReadInstruction::toString);

    nb::class_<rocisa::GLOBALLoadInstruction, rocisa::GlobalReadInstruction>(
        m_mem, "GLOBALLoadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::GLOBALModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("modifier") = std::nullopt,
             nb::arg("comment")  = "")
        .def("getParams", &rocisa::GLOBALLoadInstruction::getParams)
        .def("__str__", &rocisa::GLOBALLoadInstruction::toString);

    nb::class_<rocisa::MUBUFReadInstruction, rocisa::GlobalReadInstruction>(m_mem,
                                                                            "MUBUFReadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::MUBUFReadInstruction::getParams)
        .def("__str__", &rocisa::MUBUFReadInstruction::toString);

    nb::class_<rocisa::AtomicReadWriteInstruction,
               rocisa::ReadWriteInstruction,
               rocisa::PyAtomicReadWriteInstruction>(m_mem, "AtomicReadWriteInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("srcs"),
             nb::arg("comment") = "");

    nb::class_<rocisa::SMemAtomicDecInstruction, rocisa::AtomicReadWriteInstruction>(
        m_mem, "SMemAtomicDecInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::SMemAtomicDecInstruction::getParams)
        .def("__str__", &rocisa::SMemAtomicDecInstruction::toString);

    nb::class_<rocisa::SMemLoadInstruction, rocisa::GlobalReadInstruction>(m_mem,
                                                                           "SMemLoadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::SMemLoadInstruction::getParams)
        .def("__str__", &rocisa::SMemLoadInstruction::toString);

    nb::class_<rocisa::GlobalWriteInstruction,
               rocisa::ReadWriteInstruction,
               rocisa::PyGlobalWriteInstruction>(m_mem, "GlobalWriteInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("srcData"),
             nb::arg("comment") = "");

    nb::class_<rocisa::SMemStoreInstruction, rocisa::GlobalWriteInstruction>(m_mem,
                                                                             "SMemStoreInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("srcData"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::SMemStoreInstruction::getParams)
        .def("__str__", &rocisa::SMemStoreInstruction::toString);

    nb::class_<rocisa::FLATStoreInstruction, rocisa::GlobalWriteInstruction>(m_mem,
                                                                             "FLATStoreInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("vaddr"),
             nb::arg("srcData"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::FLATStoreInstruction::getParams)
        .def("__str__", &rocisa::FLATStoreInstruction::toString);

    nb::class_<rocisa::MUBUFStoreInstruction, rocisa::GlobalWriteInstruction>(
        m_mem, "MUBUFStoreInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("srcData"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::MUBUFStoreInstruction::getParams)
        .def("__str__", &rocisa::MUBUFStoreInstruction::toString);

    nb::class_<rocisa::LocalReadInstruction,
               rocisa::ReadWriteInstruction,
               rocisa::PyLocalReadInstruction>(m_mem, "LocalReadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("srcs"),
             nb::arg("comment") = "")
        .def_rw("dst", &rocisa::LocalReadInstruction::dst);

    nb::class_<rocisa::DSLoadInstruction, rocisa::LocalReadInstruction>(m_mem, "DSLoadInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("srcs"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("getParams", &rocisa::DSLoadInstruction::getParams)
        .def("__str__", &rocisa::DSLoadInstruction::toString);

    nb::class_<rocisa::LocalWriteInstruction,
               rocisa::ReadWriteInstruction,
               rocisa::PyLocalWriteInstruction>(m_mem, "LocalWriteInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dstAddr"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("comment") = "");

    nb::class_<rocisa::DSStoreInstruction, rocisa::LocalWriteInstruction>(m_mem,
                                                                          "DSStoreInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dstAddr"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_rw("ds", &rocisa::DSStoreInstruction::ds)
        .def("getParams", &rocisa::DSStoreInstruction::getParams)
        .def("__str__", &rocisa::DSStoreInstruction::toString);

    nb::class_<rocisa::BufferLoadU8, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadU8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadU8& self, const nb::dict&) {
            return new rocisa::BufferLoadU8(self);
        });

    nb::class_<rocisa::BufferLoadD16HIU8, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadD16HIU8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadD16HIU8& self, const nb::dict&) {
            return new rocisa::BufferLoadD16HIU8(self);
        });

    nb::class_<rocisa::BufferLoadD16U8, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadD16U8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadD16U8& self, const nb::dict&) {
            return new rocisa::BufferLoadD16U8(self);
        });

    nb::class_<rocisa::BufferLoadD16HIB16, rocisa::MUBUFReadInstruction>(m_mem,
                                                                         "BufferLoadD16HIB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadD16HIB16& self, const nb::dict&) {
            return new rocisa::BufferLoadD16HIB16(self);
        });

    nb::class_<rocisa::BufferLoadD16B16, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadD16B16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadD16B16& self, const nb::dict&) {
            return new rocisa::BufferLoadD16B16(self);
        });

    nb::class_<rocisa::BufferLoadB32, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadB32& self, const nb::dict&) {
            return new rocisa::BufferLoadB32(self);
        });

    nb::class_<rocisa::BufferLoadB64, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadB64& self, const nb::dict&) {
            return new rocisa::BufferLoadB64(self);
        });

    nb::class_<rocisa::BufferLoadB128, rocisa::MUBUFReadInstruction>(m_mem, "BufferLoadB128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferLoadB128& self, const nb::dict&) {
            return new rocisa::BufferLoadB128(self);
        });

    nb::class_<rocisa::FlatLoadD16HIU8, rocisa::FLATReadInstruction>(m_mem, "FlatLoadD16HIU8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadD16HIU8& self, const nb::dict&) {
            return new rocisa::FlatLoadD16HIU8(self);
        });

    nb::class_<rocisa::FlatLoadD16U8, rocisa::FLATReadInstruction>(m_mem, "FlatLoadD16U8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadD16U8& self, const nb::dict&) {
            return new rocisa::FlatLoadD16U8(self);
        });

    nb::class_<rocisa::FlatLoadD16HIB16, rocisa::FLATReadInstruction>(m_mem, "FlatLoadD16HIB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadD16HIB16& self, const nb::dict&) {
            return new rocisa::FlatLoadD16HIB16(self);
        });

    nb::class_<rocisa::FlatLoadD16B16, rocisa::FLATReadInstruction>(m_mem, "FlatLoadD16B16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadD16B16& self, const nb::dict&) {
            return new rocisa::FlatLoadD16B16(self);
        });

    nb::class_<rocisa::FlatLoadB32, rocisa::FLATReadInstruction>(m_mem, "FlatLoadB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadB32& self, const nb::dict&) {
            return new rocisa::FlatLoadB32(self);
        });

    nb::class_<rocisa::FlatLoadB64, rocisa::FLATReadInstruction>(m_mem, "FlatLoadB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadB64& self, const nb::dict&) {
            return new rocisa::FlatLoadB64(self);
        });

    nb::class_<rocisa::FlatLoadB128, rocisa::FLATReadInstruction>(m_mem, "FlatLoadB128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatLoadB128& self, const nb::dict&) {
            return new rocisa::FlatLoadB128(self);
        });

    nb::class_<rocisa::GlobalLoadTR8B64, rocisa::GLOBALLoadInstruction>(m_mem, "GlobalLoadTR8B64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::GLOBALModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("modifier"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::GlobalLoadTR8B64& self, const nb::dict&) {
            return new rocisa::GlobalLoadTR8B64(self);
        });

    nb::class_<rocisa::GlobalLoadTR16B128, rocisa::GLOBALLoadInstruction>(m_mem, "GlobalLoadTR16B128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::GLOBALModifiers>,
                      const std::string&>(),
             nb::arg("dst").none(),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("modifier"),
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::GlobalLoadTR16B128& self, const nb::dict&) {
            return new rocisa::GlobalLoadTR16B128(self);
        });

    nb::class_<rocisa::BufferStoreB8, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreB8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreB8& self, const nb::dict&) {
            return new rocisa::BufferStoreB8(self);
        });

    nb::class_<rocisa::BufferStoreD16HIU8, rocisa::MUBUFStoreInstruction>(m_mem,
                                                                          "BufferStoreD16HIU8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreD16HIU8& self, const nb::dict&) {
            return new rocisa::BufferStoreD16HIU8(self);
        });

    nb::class_<rocisa::BufferStoreD16U8, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreD16U8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreD16U8& self, const nb::dict&) {
            return new rocisa::BufferStoreD16U8(self);
        });

    nb::class_<rocisa::BufferStoreD16HIB16, rocisa::MUBUFStoreInstruction>(m_mem,
                                                                           "BufferStoreD16HIB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreD16HIB16& self, const nb::dict&) {
            return new rocisa::BufferStoreD16HIB16(self);
        });

    nb::class_<rocisa::BufferStoreD16B16, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreD16B16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreD16B16& self, const nb::dict&) {
            return new rocisa::BufferStoreD16B16(self);
        });

    nb::class_<rocisa::BufferStoreB16, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreB16& self, const nb::dict&) {
            return new rocisa::BufferStoreB16(self);
        });

    nb::class_<rocisa::BufferStoreB32, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreB32& self, const nb::dict&) {
            return new rocisa::BufferStoreB32(self);
        });

    nb::class_<rocisa::BufferStoreB64, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreB64& self, const nb::dict&) {
            return new rocisa::BufferStoreB64(self);
        });

    nb::class_<rocisa::BufferStoreB128, rocisa::MUBUFStoreInstruction>(m_mem, "BufferStoreB128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferStoreB128& self, const nb::dict&) {
            return new rocisa::BufferStoreB128(self);
        });

    nb::class_<rocisa::BufferAtomicAddF32, rocisa::MUBUFStoreInstruction>(m_mem,
                                                                          "BufferAtomicAddF32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__str__", &rocisa::BufferAtomicAddF32::toString)
        .def("__deepcopy__", [](const rocisa::BufferAtomicAddF32& self, const nb::dict&) {
            return new rocisa::BufferAtomicAddF32(self);
        });

    nb::class_<rocisa::BufferAtomicCmpswapB32, rocisa::MUBUFStoreInstruction>(
        m_mem, "BufferAtomicCmpswapB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferAtomicCmpswapB32& self, const nb::dict&) {
            return new rocisa::BufferAtomicCmpswapB32(self);
        });

    nb::class_<rocisa::BufferAtomicCmpswapB64, rocisa::MUBUFStoreInstruction>(
        m_mem, "BufferAtomicCmpswapB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const InstructionInput&,
                      std::optional<rocisa::MUBUFModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("saddr"),
             nb::arg("soffset"),
             nb::arg("mubuf")   = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::BufferAtomicCmpswapB64& self, const nb::dict&) {
            return new rocisa::BufferAtomicCmpswapB64(self);
        });

    nb::class_<rocisa::FlatStoreD16HIB16, rocisa::FLATStoreInstruction>(m_mem, "FlatStoreD16HIB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatStoreD16HIB16& self, const nb::dict&) {
            return new rocisa::FlatStoreD16HIB16(self);
        });

    nb::class_<rocisa::FlatStoreD16B16, rocisa::FLATStoreInstruction>(m_mem, "FlatStoreD16B16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatStoreD16B16& self, const nb::dict&) {
            return new rocisa::FlatStoreD16B16(self);
        });

    nb::class_<rocisa::FlatStoreB32, rocisa::FLATStoreInstruction>(m_mem, "FlatStoreB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatStoreB32& self, const nb::dict&) {
            return new rocisa::FlatStoreB32(self);
        });

    nb::class_<rocisa::FlatStoreB64, rocisa::FLATStoreInstruction>(m_mem, "FlatStoreB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatStoreB64& self, const nb::dict&) {
            return new rocisa::FlatStoreB64(self);
        });

    nb::class_<rocisa::FlatStoreB128, rocisa::FLATStoreInstruction>(m_mem, "FlatStoreB128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("vaddr"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::FlatStoreB128& self, const nb::dict&) {
            return new rocisa::FlatStoreB128(self);
        });

    nb::class_<rocisa::FlatAtomicCmpswapB32, rocisa::FLATStoreInstruction>(m_mem,
                                                                           "FlatAtomicCmpswapB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::FLATModifiers>,
                      const std::string&>(),
             nb::arg("vaddr"),
             nb::arg("tmp"),
             nb::arg("src"),
             nb::arg("flat")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__str__", &rocisa::FlatAtomicCmpswapB32::toString)
        .def("__deepcopy__", [](const rocisa::FlatAtomicCmpswapB32& self, const nb::dict&) {
            return new rocisa::FlatAtomicCmpswapB32(self);
        });

    nb::class_<rocisa::DSLoadU8, rocisa::DSLoadInstruction>(m_mem, "DSLoadU8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadU8& self, const nb::dict&) {
            return new rocisa::DSLoadU8(self);
        });

    nb::class_<rocisa::DSLoadD16HIU8, rocisa::DSLoadInstruction>(m_mem, "DSLoadD16HIU8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadD16HIU8& self, const nb::dict&) {
            return new rocisa::DSLoadD16HIU8(self);
        });

    nb::class_<rocisa::DSLoadU16, rocisa::DSLoadInstruction>(m_mem, "DSLoadU16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadU16& self, const nb::dict&) {
            return new rocisa::DSLoadU16(self);
        });

    nb::class_<rocisa::DSLoadD16HIU16, rocisa::DSLoadInstruction>(m_mem, "DSLoadD16HIU16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadD16HIU16& self, const nb::dict&) {
            return new rocisa::DSLoadD16HIU16(self);
        });

    nb::class_<rocisa::DSLoadB16, rocisa::DSLoadInstruction>(m_mem, "DSLoadB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadB16& self, const nb::dict&) {
            return new rocisa::DSLoadB16(self);
        });

    nb::class_<rocisa::DSLoadB32, rocisa::DSLoadInstruction>(m_mem, "DSLoadB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadB32& self, const nb::dict&) {
            return new rocisa::DSLoadB32(self);
        });

    nb::class_<rocisa::DSLoadB64, rocisa::DSLoadInstruction>(m_mem, "DSLoadB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadB64& self, const nb::dict&) {
            return new rocisa::DSLoadB64(self);
        });

    nb::class_<rocisa::DSLoadB64TrB16, rocisa::DSLoadInstruction>(m_mem, "DSLoadB64TrB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoadB64TrB16& self, const nb::dict&) {
            return new rocisa::DSLoadB64TrB16(self);
        });

    nb::class_<rocisa::DSLoadB128, rocisa::DSLoadInstruction>(m_mem, "DSLoadB128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSLoadB128::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSLoadB128& self, const nb::dict&) {
            return new rocisa::DSLoadB128(self);
        });

    nb::class_<rocisa::DSLoad2B32, rocisa::DSLoadInstruction>(m_mem, "DSLoad2B32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoad2B32& self, const nb::dict&) {
            return new rocisa::DSLoad2B32(self);
        });

    nb::class_<rocisa::DSLoad2B64, rocisa::DSLoadInstruction>(m_mem, "DSLoad2B64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSLoad2B64& self, const nb::dict&) {
            return new rocisa::DSLoad2B64(self);
        });

    nb::class_<rocisa::DSStoreU16, rocisa::DSStoreInstruction>(m_mem, "DSStoreU16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStoreU16::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSStoreU16& self, const nb::dict&) {
            return new rocisa::DSStoreU16(self);
        });

    nb::class_<rocisa::DSStoreB8, rocisa::DSStoreInstruction>(m_mem, "DSStoreB8")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSStoreB8& self, const nb::dict&) {
            return new rocisa::DSStoreB8(self);
        });

    nb::class_<rocisa::DSStoreB16, rocisa::DSStoreInstruction>(m_mem, "DSStoreB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSStoreB16& self, const nb::dict&) {
            return new rocisa::DSStoreB16(self);
        });

    nb::class_<rocisa::DSStoreB8HID16, rocisa::DSStoreInstruction>(m_mem, "DSStoreB8HID16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSStoreB8HID16& self, const nb::dict&) {
            return new rocisa::DSStoreB8HID16(self);
        });

    nb::class_<rocisa::DSStoreD16HIB16, rocisa::DSStoreInstruction>(m_mem, "DSStoreD16HIB16")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSStoreD16HIB16& self, const nb::dict&) {
            return new rocisa::DSStoreD16HIB16(self);
        });

    nb::class_<rocisa::DSStoreB32, rocisa::DSStoreInstruction>(m_mem, "DSStoreB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStoreB32::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSStoreB32& self, const nb::dict&) {
            return new rocisa::DSStoreB32(self);
        });

    nb::class_<rocisa::DSStoreB64, rocisa::DSStoreInstruction>(m_mem, "DSStoreB64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStoreB64::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSStoreB64& self, const nb::dict&) {
            return new rocisa::DSStoreB64(self);
        });

    nb::class_<rocisa::DSStoreB128, rocisa::DSStoreInstruction>(m_mem, "DSStoreB128")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStoreB128::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSStoreB128& self, const nb::dict&) {
            return new rocisa::DSStoreB128(self);
        });

    nb::class_<rocisa::DSStoreB256, rocisa::DSStoreInstruction>(m_mem, "DSStoreB256")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStoreB256::issueLatency)
        .def("__str__", &rocisa::DSStoreB256::toString)
        .def("__deepcopy__", [](const rocisa::DSStoreB256& self, const nb::dict&) {
            return new rocisa::DSStoreB256(self);
        });

    nb::class_<rocisa::DSStore2B32, rocisa::DSStoreInstruction>(m_mem, "DSStore2B32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStore2B32::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSStore2B32& self, const nb::dict&) {
            return new rocisa::DSStore2B32(self);
        });

    nb::class_<rocisa::DSStore2B64, rocisa::DSStoreInstruction>(m_mem, "DSStore2B64")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dstAddr"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def_static("issueLatency", &rocisa::DSStore2B64::issueLatency)
        .def("__deepcopy__", [](const rocisa::DSStore2B64& self, const nb::dict&) {
            return new rocisa::DSStore2B64(self);
        });

    nb::class_<rocisa::DSBPermuteB32, rocisa::DSStoreInstruction>(m_mem, "DSBPermuteB32")
        .def(nb::init<const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      std::optional<rocisa::DSModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("src0"),
             nb::arg("src1"),
             nb::arg("ds")      = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::DSBPermuteB32& self, const nb::dict&) {
            return new rocisa::DSBPermuteB32(self);
        });

    nb::class_<rocisa::SAtomicDec, rocisa::SMemAtomicDecInstruction>(m_mem, "SAtomicDec")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SAtomicDec& self, nb::dict&) {
            return new rocisa::SAtomicDec(self);
        });

    nb::class_<rocisa::SLoadB32, rocisa::SMemLoadInstruction>(m_mem, "SLoadB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SLoadB32& self, nb::dict&) { return new rocisa::SLoadB32(self); });

    nb::class_<rocisa::SLoadB64, rocisa::SMemLoadInstruction>(m_mem, "SLoadB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SLoadB64& self, nb::dict&) { return new rocisa::SLoadB64(self); });

    nb::class_<rocisa::SLoadB128, rocisa::SMemLoadInstruction>(m_mem, "SLoadB128")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SLoadB128& self, nb::dict&) { return new rocisa::SLoadB128(self); });

    nb::class_<rocisa::SLoadB256, rocisa::SMemLoadInstruction>(m_mem, "SLoadB256")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SLoadB256& self, nb::dict&) { return new rocisa::SLoadB256(self); });

    nb::class_<rocisa::SLoadB512, rocisa::SMemLoadInstruction>(m_mem, "SLoadB512")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("dst"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SLoadB512& self, nb::dict&) { return new rocisa::SLoadB512(self); });

    nb::class_<rocisa::SStoreB32, rocisa::SMemStoreInstruction>(m_mem, "SStoreB32")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SStoreB32& self, nb::dict&) { return new rocisa::SStoreB32(self); });

    nb::class_<rocisa::SStoreB64, rocisa::SMemStoreInstruction>(m_mem, "SStoreB64")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__",
             [](const rocisa::SStoreB64& self, nb::dict&) { return new rocisa::SStoreB64(self); });

    nb::class_<rocisa::SStoreB128, rocisa::SMemStoreInstruction>(m_mem, "SStoreB128")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SStoreB128& self, nb::dict&) {
            return new rocisa::SStoreB128(self);
        });

    nb::class_<rocisa::SStoreB256, rocisa::SMemStoreInstruction>(m_mem, "SStoreB256")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SStoreB256& self, nb::dict&) {
            return new rocisa::SStoreB256(self);
        });

    nb::class_<rocisa::SStoreB512, rocisa::SMemStoreInstruction>(m_mem, "SStoreB512")
        .def(nb::init<const std::shared_ptr<rocisa::Container>&,
                      const std::shared_ptr<rocisa::Container>&,
                      const InstructionInput&,
                      std::optional<rocisa::SMEMModifiers>,
                      const std::string&>(),
             nb::arg("src"),
             nb::arg("base"),
             nb::arg("soffset"),
             nb::arg("smem")    = std::nullopt,
             nb::arg("comment") = "")
        .def("__deepcopy__", [](const rocisa::SStoreB512& self, nb::dict&) {
            return new rocisa::SStoreB512(self);
        });
}
