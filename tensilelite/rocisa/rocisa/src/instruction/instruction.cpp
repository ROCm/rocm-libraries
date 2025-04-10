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
#include "instruction/instruction.hpp"
#include "instruction/common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;

namespace rocisa
{
    struct PyInstruction : public Instruction
    {
        NB_TRAMPOLINE(Instruction, 0);
        std::vector<InstructionInput> getParams() const override
        {
            NB_OVERRIDE_PURE(getParams);
        }
    };

    struct PyCompositeInstruction : public CompositeInstruction
    {
        NB_TRAMPOLINE(CompositeInstruction, 0);
        std::vector<std::shared_ptr<Instruction>> setupInstructions() const override
        {
            NB_OVERRIDE_PURE(setupInstructions);
        }
    };
}

void common_inst(nb::module_ m);
void branch_inst(nb::module_ m);
void cmp_inst(nb::module_ m);
void cvt_inst(nb::module_ m);
void mem_inst(nb::module_ m);
void mfma_inst(nb::module_ m);
void ext_inst(nb::module_ m);

using InstructionInputVector = std::vector<InstructionInput>;

NB_MAKE_OPAQUE(InstructionInputVector);

void init_inst(nb::module_ m)
{
    auto m_inst = m.def_submodule("instruction", "rocIsa instructions submodule.");

    nb::bind_vector<InstructionInputVector>(
        m_inst, "InstructionInputVector", "A vector of InstructionInputs.");

    // Base class
    nb::class_<rocisa::Instruction, rocisa::Item, rocisa::PyInstruction>(m_inst, "Instruction")
        .def(nb::init<rocisa::InstType, const std::string&>())
        .def_rw("instType", &rocisa::Instruction::instType)
        .def_rw("comment", &rocisa::Instruction::comment)
        .def("setInlineAsm", &rocisa::Instruction::setInlineAsm)
        .def("getParams", &rocisa::Instruction::getParams)
        .def("__str__", &rocisa::Instruction::toString)
        .def("__deepcopy__",
             [](const rocisa::Instruction& self, const nb::dict&) {
                 throw std::runtime_error("Deepcopy not supported for Instruction");
                 return nullptr;
             })
        .def("__reduce__", [](const rocisa::Instruction& self) {
            throw std::runtime_error("Pickling not supported for Instruction");
            return nullptr;
        });

    nb::class_<rocisa::CompositeInstruction, rocisa::Instruction>(m_inst, "CompositeInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::Container>&,
                      const std::vector<InstructionInput>&,
                      const std::string&>())
        .def_rw("dst", &rocisa::CompositeInstruction::dst)
        .def_rw("srcs", &rocisa::CompositeInstruction::srcs)
        .def("getInstructions", &rocisa::CompositeInstruction::getInstructions)
        .def("getParams", &rocisa::CompositeInstruction::getParams)
        .def("__str__", &rocisa::CompositeInstruction::toString);

    nb::class_<rocisa::CommonInstruction, rocisa::Instruction>(m_inst, "CommonInstruction")
        .def(nb::init<rocisa::InstType,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::vector<InstructionInput>&,
                      std::optional<rocisa::DPPModifiers>,
                      std::optional<rocisa::SDWAModifiers>,
                      std::optional<rocisa::VOP3PModifiers>,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("dst"),
             nb::arg("srcs"),
             nb::arg("dpp")     = std::nullopt,
             nb::arg("sdwa")    = std::nullopt,
             nb::arg("vop3")    = std::nullopt,
             nb::arg("comment") = "")
        .def_rw("dst", &rocisa::CommonInstruction::dst)
        .def_rw("srcs", &rocisa::CommonInstruction::srcs)
        .def_rw("dpp", &rocisa::CommonInstruction::dpp)
        .def_rw("sdwa", &rocisa::CommonInstruction::sdwa)
        .def_rw("vop3", &rocisa::CommonInstruction::vop3)
        .def_rw("comment", &rocisa::CommonInstruction::comment)
        .def("setSrc",
             [](rocisa::CommonInstruction& self, size_t idx, const InstructionInput& src) {
                 self.srcs[idx] = src;
             })
        .def("getParams", &rocisa::CommonInstruction::getParams)
        .def("__str__", &rocisa::CommonInstruction::toString);

    nb::class_<rocisa::MacroInstruction, rocisa::Instruction>(m_inst, "MacroInstruction")
        .def(nb::init<const std::string&,
                      const std::vector<InstructionInput>&,
                      const std::string&>(),
             nb::arg("name"),
             nb::arg("args"),
             nb::arg("comment") = "")
        .def_rw("args", &rocisa::MacroInstruction::args)
        .def_rw("comment", &rocisa::MacroInstruction::comment)
        .def("setSrc",
             [](rocisa::MacroInstruction& self, size_t idx, const InstructionInput& arg) {
                 self.args[idx] = arg;
             })
        .def("getParams", &rocisa::MacroInstruction::getParams)
        .def("__str__", &rocisa::MacroInstruction::toString);

    // Instructions
    common_inst(m_inst);
    branch_inst(m_inst);
    cmp_inst(m_inst);
    cvt_inst(m_inst);
    mem_inst(m_inst);
    mfma_inst(m_inst);
    ext_inst(m_inst);
}
