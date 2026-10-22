// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "instruction/mfma.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void mfma_inst(nb::module_ m_mfma)
{
    m_mfma
        .def("getMFMAIssueLatency",
             &rocisa::getMFMAIssueLatency<false>,
             nb::arg("dataType"),
             nb::arg("miM"),
             nb::arg("miB"))
        .def("getSMFMAIssueLatency",
             &rocisa::getMFMAIssueLatency<true>,
             nb::arg("dataType"),
             nb::arg("miM"),
             nb::arg("miB"));
    nb::class_<rocisa::MFMAInstruction, rocisa::Instruction>(m_mfma, "MFMAInstruction")
        .def(nb::init<rocisa::InstType,
                      rocisa::InstType,
                      const std::vector<int>&,
                      bool,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::optional<InstructionInput>&,
                      bool,
                      const std::string&,
                      bool,
                      bool>(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("mfma1k"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("acc2")    = nb::none(),
             nb::arg("neg")     = false,
             nb::arg("comment") = "",
             nb::arg("reuseA")  = false,
             nb::arg("reuseB")  = false)
        .def_rw("a", &rocisa::MFMAInstruction::a)
        .def_rw("b", &rocisa::MFMAInstruction::b)
        .def_rw("acc", &rocisa::MFMAInstruction::acc)
        .def_rw("acc2", &rocisa::MFMAInstruction::acc2)
        .def_rw("reuseA", &rocisa::MFMAInstruction::reuseA)
        .def_rw("reuseB", &rocisa::MFMAInstruction::reuseB)
        .def("getParams", &rocisa::MFMAInstruction::getParams)
        .def("getIssueLatency", &rocisa::MFMAInstruction::getIssueLatency)
        .def("__str__", &rocisa::MFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::MFMAInstruction& self, const nb::dict&) {
            return new rocisa::MFMAInstruction(self);
        })
        .def(nb::init<rocisa::InstType,
                      rocisa::InstType,
                      const std::vector<int>&,
                      bool,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const int,
                      bool,
                      const std::string&,
                      bool,
                      bool>(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("mfma1k"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("acc2_imm") = 0,
             nb::arg("neg")      = false,
             nb::arg("comment")  = "",
             nb::arg("reuseA")   = false,
             nb::arg("reuseB")   = false)
        .def_rw("a", &rocisa::MFMAInstruction::a)
        .def_rw("b", &rocisa::MFMAInstruction::b)
        .def_rw("acc", &rocisa::MFMAInstruction::acc)
        .def_rw("acc2_imm", &rocisa::MFMAInstruction::acc2_imm)
        .def_rw("reuseA", &rocisa::MFMAInstruction::reuseA)
        .def_rw("reuseB", &rocisa::MFMAInstruction::reuseB)
        .def("getParams", &rocisa::MFMAInstruction::getParams)
        .def("getIssueLatency", &rocisa::MFMAInstruction::getIssueLatency)
        .def("__str__", &rocisa::MFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::MFMAInstruction& self, const nb::dict&) {
            return new rocisa::MFMAInstruction(self);
        });

    nb::class_<rocisa::MXMFMAInstruction, rocisa::Instruction>(m_mfma, "MXMFMAInstruction")
        // The C++ constructor parameter order was reshuffled (mxScaleA/BType moved
        // from positions 3-4 to after mxsb). nb::kw_only() forces every Python
        // caller to spell out argument names, which prevents a silent positional
        // mis-binding if the C++ signature changes again. The lone in-tree caller
        // (KernelWriterAssembly.MXMFMAInstruction(...)) already uses keyword args,
        // so this is non-breaking.
        .def(nb::init<rocisa::InstType,
                      rocisa::InstType,
                      const std::vector<int>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      rocisa::InstType,
                      rocisa::InstType,
                      int,
                      const std::string&,
                      bool,
                      bool,
                      int,
                      int>(),
             nb::kw_only(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("acc2")         = nullptr,
             nb::arg("mxsa")         = nullptr,
             nb::arg("mxsb")         = nullptr,
             nb::arg("vop3")         = std::nullopt,
             nb::arg("mxScaleAType") = rocisa::InstType::INST_F32,
             nb::arg("mxScaleBType") = rocisa::InstType::INST_F32,
             nb::arg("block")        = 0,
             nb::arg("comment")      = "",
             nb::arg("reuseA")       = false,
             nb::arg("reuseB")       = false,
             nb::arg("mxScaleASel")  = 0,
             nb::arg("mxScaleBSel")  = 0)
        .def_rw("a", &rocisa::MXMFMAInstruction::a)
        .def_rw("b", &rocisa::MXMFMAInstruction::b)
        .def_rw("mxsa", &rocisa::MXMFMAInstruction::mxsa)
        .def_rw("mxsb", &rocisa::MXMFMAInstruction::mxsb)
        .def_rw("acc", &rocisa::MXMFMAInstruction::acc)
        .def_rw("acc2", &rocisa::MXMFMAInstruction::acc2)
        .def_rw("vop3", &rocisa::MXMFMAInstruction::vop3)
        .def_rw("reuseA", &rocisa::MXMFMAInstruction::reuseA)
        .def_rw("reuseB", &rocisa::MXMFMAInstruction::reuseB)
        .def_rw("mxScaleASel", &rocisa::MXMFMAInstruction::mxScaleASel)
        .def_rw("mxScaleBSel", &rocisa::MXMFMAInstruction::mxScaleBSel)
        .def("getParams", &rocisa::MXMFMAInstruction::getParams)
        .def("getIssueLatency", &rocisa::MXMFMAInstruction::getIssueLatency)
        .def("__str__", &rocisa::MXMFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::MXMFMAInstruction& self, const nb::dict&) {
            return new rocisa::MXMFMAInstruction(self);
        })
        .def(nb::init<rocisa::InstType,
                      rocisa::InstType,
                      const std::vector<int>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      int,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::shared_ptr<rocisa::RegisterContainer>&,
                      const std::optional<rocisa::VOP3PModifiers>&,
                      rocisa::InstType,
                      rocisa::InstType,
                      int,
                      const std::string&,
                      bool,
                      bool,
                      int,
                      int>(),
             nb::kw_only(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("acc2_imm")     = 0,
             nb::arg("mxsa")         = nullptr,
             nb::arg("mxsb")         = nullptr,
             nb::arg("vop3")         = std::nullopt,
             nb::arg("mxScaleAType") = rocisa::InstType::INST_F32,
             nb::arg("mxScaleBType") = rocisa::InstType::INST_F32,
             nb::arg("block")        = 0,
             nb::arg("comment")      = "",
             nb::arg("reuseA")       = false,
             nb::arg("reuseB")       = false,
             nb::arg("mxScaleASel")  = 0,
             nb::arg("mxScaleBSel")  = 0)
        .def_rw("a", &rocisa::MXMFMAInstruction::a)
        .def_rw("b", &rocisa::MXMFMAInstruction::b)
        .def_rw("mxsa", &rocisa::MXMFMAInstruction::mxsa)
        .def_rw("mxsb", &rocisa::MXMFMAInstruction::mxsb)
        .def_rw("acc", &rocisa::MXMFMAInstruction::acc)
        .def_rw("acc2_imm", &rocisa::MXMFMAInstruction::acc2_imm)
        .def_rw("vop3", &rocisa::MXMFMAInstruction::vop3)
        .def_rw("reuseA", &rocisa::MXMFMAInstruction::reuseA)
        .def_rw("reuseB", &rocisa::MXMFMAInstruction::reuseB)
        .def_rw("mxScaleASel", &rocisa::MXMFMAInstruction::mxScaleASel)
        .def_rw("mxScaleBSel", &rocisa::MXMFMAInstruction::mxScaleBSel)
        .def("getParams", &rocisa::MXMFMAInstruction::getParams)
        .def("getIssueLatency", &rocisa::MXMFMAInstruction::getIssueLatency)
        .def("__str__", &rocisa::MXMFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::MXMFMAInstruction& self, const nb::dict&) {
            return new rocisa::MXMFMAInstruction(self);
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
                      bool,
                      const std::string&>(),
             nb::arg("instType"),
             nb::arg("accType"),
             nb::arg("variant"),
             nb::arg("mfma1k"),
             nb::arg("acc"),
             nb::arg("a"),
             nb::arg("b"),
             nb::arg("metadata"),
             nb::arg("neg")     = false,
             nb::arg("comment") = "")
        .def_rw("a", &rocisa::SMFMAInstruction::a)
        .def_rw("b", &rocisa::SMFMAInstruction::b)
        .def_rw("acc", &rocisa::SMFMAInstruction::acc)
        .def_rw("metadata", &rocisa::SMFMAInstruction::metadata)
        .def("getParams", &rocisa::SMFMAInstruction::getParams)
        .def("getIssueLatency", &rocisa::SMFMAInstruction::getIssueLatency)
        .def("__str__", &rocisa::SMFMAInstruction::toString)
        .def("__deepcopy__", [](const rocisa::SMFMAInstruction& self, const nb::dict&) {
            return new rocisa::SMFMAInstruction(self);
        });
}
