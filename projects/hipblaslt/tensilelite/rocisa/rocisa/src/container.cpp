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
#include "container.hpp"
#include "code.hpp"
#include "instruction/branch.hpp"
#include "instruction/common.hpp"
#include "instruction/instruction.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;

namespace rocisa
{
    std::shared_ptr<Item> replaceHolder(std::shared_ptr<Item> inst, int dst)
    {
        if(auto mod = std::dynamic_pointer_cast<Module>(inst))
        {
            for(auto& item : mod->items())
            {
                // We don't need the output
                static_cast<void>(replaceHolder(item, dst));
            }
        }
        else if(auto _inst = std::dynamic_pointer_cast<Instruction>(inst))
        {
            for(auto& param : _inst->getParams())
            {
                if(auto container = std::get_if<std::shared_ptr<Container>>(&param))
                {
                    if(auto holder = std::dynamic_pointer_cast<HolderContainer>(*container))
                    {
                        holder->setRegNum(dst);
                        param = std::make_shared<RegisterContainer>(holder->getCopiedRC());
                    }
                }
            }
        }
        else if(auto waitcnt = std::dynamic_pointer_cast<SWaitCnt>(inst))
        {
            throw std::runtime_error("SWaitCnt is not supported yet");
        }
        return inst;
    }

    struct PyContainer : public Container
    {
        NB_TRAMPOLINE(Container, 0);
        std::shared_ptr<Container> clone() const override
        {
            NB_OVERRIDE_PURE(clone);
        }

        std::string toString() const override
        {
            NB_OVERRIDE_PURE(toString);
        }
    };

    // Helper function to create GPR containers
    std::shared_ptr<RegisterContainer>
        createGPR(const std::string& gprType, const Holder& holder, float regNum)
    {
        if(holder.idx == -1)
        {
            return std::make_shared<HolderContainer>(gprType, *holder.name, regNum);
        }
        return std::make_shared<HolderContainer>(gprType, holder.idx, regNum);
    }

    std::shared_ptr<RegisterContainer> createGPR(const std::string& gprType, int idx, float regNum)
    {
        return std::make_shared<RegisterContainer>(gprType, std::nullopt, idx, regNum);
    }

    std::shared_ptr<RegisterContainer> createGPR(const std::string& gprType,
                                                 const std::string& name,
                                                 float              regNum  = 1.f,
                                                 bool               isMacro = false,
                                                 bool               isAbs   = false)
    {
        RegName regname = generateRegName(name);
        return std::make_shared<RegisterContainer>(gprType, regname, isAbs, isMacro, -1, regNum);
    }

    // Overloaded functions to create specific GPR containers with default regNum = 1.f
    std::shared_ptr<RegisterContainer> vgpr(const Holder& holder, float regNum)
    {
        return createGPR("v", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> vgpr(int idx, float regNum)
    {
        return createGPR("v", idx, regNum);
    }

    std::shared_ptr<RegisterContainer>
        vgpr(const std::string& name, float regNum, bool isMacro, bool isAbs)
    {
        return createGPR("v", name, regNum, isMacro, isAbs);
    }

    std::shared_ptr<RegisterContainer> sgpr(const Holder& holder, float regNum)
    {
        return createGPR("s", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> sgpr(int idx, float regNum)
    {
        return createGPR("s", idx, regNum);
    }

    std::shared_ptr<RegisterContainer> sgpr(const std::string& name, float regNum, bool isMacro)
    {
        return createGPR("s", name, regNum, isMacro);
    }

    std::shared_ptr<RegisterContainer> accvgpr(const Holder& holder, float regNum)
    {
        return createGPR("acc", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> accvgpr(int idx, float regNum)
    {
        return createGPR("acc", idx, regNum);
    }

    std::shared_ptr<RegisterContainer> accvgpr(const std::string& name, float regNum)
    {
        return createGPR("acc", name, regNum);
    }

    std::shared_ptr<RegisterContainer> mgpr(const Holder& holder, float regNum)
    {
        return createGPR("m", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> mgpr(int idx, float regNum)
    {
        return createGPR("m", idx, regNum);
    }

    std::shared_ptr<RegisterContainer> mgpr(const std::string& name, float regNum)
    {
        return createGPR("m", name, regNum);
    }
} // namespace rocisa

void init_containers(nb::module_ m)
{
    auto m_con = m.def_submodule("container", "rocIsa container submodule.");
    m_con.def("replaceHolder", &rocisa::replaceHolder, nb::arg("inst"), nb::arg("dst"));
    m_con.def("vgpr",
              nb::overload_cast<const rocisa::Holder&, float>(&rocisa::vgpr),
              nb::arg("holder"),
              nb::arg("regNum") = 1.f);
    m_con.def("vgpr",
              nb::overload_cast<int, float>(&rocisa::vgpr),
              nb::arg("idx"),
              nb::arg("regNum") = 1.f);
    m_con.def("vgpr",
              nb::overload_cast<const std::string&, float, bool, bool>(&rocisa::vgpr),
              nb::arg("name"),
              nb::arg("regNum")  = 1.f,
              nb::arg("isMacro") = false,
              nb::arg("isAbs")   = false);

    m_con.def("sgpr",
              nb::overload_cast<const rocisa::Holder&, float>(&rocisa::sgpr),
              nb::arg("holder"),
              nb::arg("regNum") = 1.f);
    m_con.def("sgpr",
              nb::overload_cast<int, float>(&rocisa::sgpr),
              nb::arg("idx"),
              nb::arg("regNum") = 1.f);
    m_con.def("sgpr",
              nb::overload_cast<const std::string&, float, bool>(&rocisa::sgpr),
              nb::arg("name"),
              nb::arg("regNum")  = 1.f,
              nb::arg("isMacro") = false);

    m_con.def("accvgpr",
              nb::overload_cast<const rocisa::Holder&, float>(&rocisa::accvgpr),
              nb::arg("holder"),
              nb::arg("regNum") = 1.f);
    m_con.def("accvgpr",
              nb::overload_cast<int, float>(&rocisa::accvgpr),
              nb::arg("idx"),
              nb::arg("regNum") = 1.f);
    m_con.def("accvgpr",
              nb::overload_cast<const std::string&, float>(&rocisa::accvgpr),
              nb::arg("name"),
              nb::arg("regNum") = 1.f);

    m_con.def("mgpr",
              nb::overload_cast<const rocisa::Holder&, float>(&rocisa::mgpr),
              nb::arg("holder"),
              nb::arg("regNum") = 1.f);
    m_con.def("mgpr",
              nb::overload_cast<int, float>(&rocisa::mgpr),
              nb::arg("idx"),
              nb::arg("regNum") = 1.f);
    m_con.def("mgpr",
              nb::overload_cast<const std::string&, float>(&rocisa::mgpr),
              nb::arg("name"),
              nb::arg("regNum") = 1.f);

    nb::class_<rocisa::Holder>(m_con, "Holder")
        .def(nb::init<int>(), nb::arg("idx"))
        .def(nb::init<const std::string&>(), nb::arg("name"))
        .def_rw("idx", &rocisa::Holder::idx)
        .def_rw("name", &rocisa::Holder::name)
        .def("__deepcopy__",
             [](const rocisa::Holder& self, nb::dict&) { return rocisa::Holder(self); })
        .def("__getstate__",
             [](const rocisa::Holder& self) { return std::make_tuple(self.idx, self.name); })
        .def("__setstate__",
             [](rocisa::Holder& self, std::tuple<int, std::optional<rocisa::RegName>> t) {
                 new(&self) rocisa::Holder(std::get<0>(t));
                 self.name = std::move(std::get<1>(t));
             });

    nb::class_<rocisa::Container, rocisa::PyContainer>(m_con, "Container")
        .def(nb::init<>())
        .def("__str__", &rocisa::Container::toString);

    nb::class_<rocisa::DSModifiers, rocisa::Container>(m_con, "DSModifiers")
        .def(nb::init<int, int, int, int, bool>(),
             nb::arg("na")      = 1,
             nb::arg("offset")  = 0,
             nb::arg("offset0") = 0,
             nb::arg("offset1") = 0,
             nb::arg("gds")     = false)
        // TODO: will remove this when instructions are all moved to C++
        .def_rw("na", &rocisa::DSModifiers::na)
        .def_rw("offset", &rocisa::DSModifiers::offset)
        .def("__str__", &rocisa::DSModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::DSModifiers& self, nb::dict mamo) {
                 return rocisa::DSModifiers(self);
             })
        .def("__getstate__",
             [](const rocisa::DSModifiers& self) {
                 return std::make_tuple(self.na, self.offset, self.offset0, self.offset1, self.gds);
             })
        .def("__setstate__", [](rocisa::DSModifiers& self, std::tuple<int, int, int, int, bool> t) {
            new(&self) rocisa::DSModifiers(
                std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
        });
    nb::class_<rocisa::FLATModifiers, rocisa::Container>(m_con, "FLATModifiers")
        .def(nb::init<int, bool, bool, bool, bool>(),
             nb::arg("offset12") = 0,
             nb::arg("glc")      = false,
             nb::arg("slc")      = false,
             nb::arg("lds")      = false,
             nb::arg("isStore")  = false)
        .def_rw("isStore", &rocisa::FLATModifiers::isStore)
        .def("__str__", &rocisa::FLATModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::FLATModifiers& self, nb::dict mamo) {
                 return rocisa::FLATModifiers(self);
             })
        .def("__getstate__",
             [](const rocisa::FLATModifiers& self) {
                 return std::make_tuple(self.offset12, self.glc, self.slc, self.lds, self.isStore);
             })
        .def(
            "__setstate__",
            [](rocisa::FLATModifiers& self, std::tuple<int, bool, bool, bool, bool> t) {
                new(&self) rocisa::FLATModifiers(
                    std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
            });

    nb::class_<rocisa::GLOBALModifiers, rocisa::Container>(m_con, "GLOBALModifiers")
        .def(nb::init<int>(),
             nb::arg("offset") = 0)
        .def("__str__", &rocisa::GLOBALModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::GLOBALModifiers& self, nb::dict mamo) {
                 return rocisa::GLOBALModifiers(self);
             })
        .def("__getstate__",
             [](const rocisa::GLOBALModifiers& self) {
                 return std::make_tuple(self.offset);
             })
        .def("__setstate__",
             [](rocisa::GLOBALModifiers& self, std::tuple<int> t) {
                 new(&self) rocisa::GLOBALModifiers(
                     std::get<0>(t));
             });

    nb::class_<rocisa::MUBUFModifiers, rocisa::Container>(m_con, "MUBUFModifiers")
        .def(nb::init<bool, int, bool, bool, bool, bool, bool>(),
             nb::arg("offen")    = false,
             nb::arg("offset12") = 0,
             nb::arg("glc")      = false,
             nb::arg("slc")      = false,
             nb::arg("nt")       = false,
             nb::arg("lds")      = false,
             nb::arg("isStore")  = false)
        .def_rw("isStore", &rocisa::MUBUFModifiers::isStore)
        .def("__str__", &rocisa::MUBUFModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::MUBUFModifiers& self, nb::dict&) {
                 return rocisa::MUBUFModifiers(self);
             })
        .def(
            "__getstate__",
            [](const rocisa::MUBUFModifiers& self) {
                return std::make_tuple(
                    self.offen, self.offset12, self.glc, self.slc, self.nt, self.lds, self.isStore);
            })
        .def("__setstate__",
             [](rocisa::MUBUFModifiers&                             self,
                std::tuple<bool, int, bool, bool, bool, bool, bool> t) {
                 new(&self) rocisa::MUBUFModifiers(std::get<0>(t),
                                                   std::get<1>(t),
                                                   std::get<2>(t),
                                                   std::get<3>(t),
                                                   std::get<4>(t),
                                                   std::get<5>(t),
                                                   std::get<6>(t));
             });

    nb::class_<rocisa::SMEMModifiers, rocisa::Container>(m_con, "SMEMModifiers")
        .def(nb::init<bool, bool, int>(),
             nb::arg("glc")    = false,
             nb::arg("nv")     = false,
             nb::arg("offset") = 0)
        .def("__str__", &rocisa::SMEMModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::SMEMModifiers& self, nb::dict&) {
                 return rocisa::SMEMModifiers(self);
             })
        .def("__getstate__",
             [](const rocisa::SMEMModifiers& self) {
                 return std::make_tuple(self.glc, self.nv, self.offset);
             })
        .def("__setstate__", [](rocisa::SMEMModifiers& self, std::tuple<bool, bool, int> t) {
            new(&self) rocisa::SMEMModifiers(std::get<0>(t), std::get<1>(t), std::get<2>(t));
        });

    nb::class_<rocisa::SDWAModifiers, rocisa::Container>(m_con, "SDWAModifiers")
        .def(nb::init<rocisa::SelectBit, rocisa::UnusedBit, rocisa::SelectBit, rocisa::SelectBit>(),
             nb::arg("dst_sel")    = 0,
             nb::arg("dst_unused") = 0,
             nb::arg("src0_sel")   = 0,
             nb::arg("src1_sel")   = 0)
        .def("__str__", &rocisa::SDWAModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::SDWAModifiers& self, nb::dict&) {
                 return rocisa::SDWAModifiers(self);
             })
        .def("__getstate__",
             [](const rocisa::SDWAModifiers& self) {
                 return std::make_tuple(
                     self.dst_sel, self.dst_unused, self.src0_sel, self.src1_sel);
             })
        .def(
            "__setstate__",
            [](rocisa::SDWAModifiers& self,
               std::
                   tuple<rocisa::SelectBit, rocisa::UnusedBit, rocisa::SelectBit, rocisa::SelectBit>
                       t) {
                new(&self) rocisa::SDWAModifiers(
                    std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
            });

    nb::class_<rocisa::DPPModifiers, rocisa::Container>(m_con, "DPPModifiers")
        .def(nb::init<int, int, int>(),
             nb::arg("row_shr")    = -1,
             nb::arg("row_bcast")  = -1,
             nb::arg("bound_ctrl") = -1)
        .def("__str__", &rocisa::DPPModifiers::toString);

    nb::class_<rocisa::VOP3PModifiers, rocisa::Container>(m_con, "VOP3PModifiers")
        .def(nb::init<const std::vector<int>&, const std::vector<int>&, const std::vector<int>&>(),
             nb::arg("op_sel")    = std::vector<int>{},
             nb::arg("op_sel_hi") = std::vector<int>{},
             nb::arg("byte_sel")  = std::vector<int>{})
        .def_rw("op_sel", &rocisa::VOP3PModifiers::op_sel)
        .def_rw("op_sel_hi", &rocisa::VOP3PModifiers::op_sel_hi)
        .def_rw("byte_sel", &rocisa::VOP3PModifiers::byte_sel)
        .def("__str__", &rocisa::VOP3PModifiers::toString)
        .def("__deepcopy__",
             [](const rocisa::VOP3PModifiers& self, nb::dict&) {
                 return rocisa::VOP3PModifiers(self);
             })
        .def("__getstate__",
             [](const rocisa::VOP3PModifiers& self) {
                 return std::make_tuple(self.op_sel, self.op_sel_hi, self.byte_sel);
             })
        .def("__setstate__",
             [](rocisa::VOP3PModifiers&                                          self,
                std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> t) {
                 new(&self) rocisa::VOP3PModifiers(std::get<0>(t), std::get<1>(t), std::get<2>(t));
             });

    nb::class_<rocisa::EXEC, rocisa::Container>(m_con, "EXEC")
        .def(nb::init<bool>(), nb::arg("setHi") = false)
        .def("__str__", &rocisa::EXEC::toString)
        .def("__deepcopy__", [](const rocisa::EXEC& self, nb::dict&) { return rocisa::EXEC(self); })
        .def("__getstate__", [](const rocisa::EXEC& self) { return std::make_tuple(self.setHi); })
        .def("__setstate__", [](rocisa::EXEC& self, std::tuple<bool> t) {
            new(&self) rocisa::EXEC(std::get<0>(t));
        });

    nb::class_<rocisa::VCC, rocisa::Container>(m_con, "VCC")
        .def(nb::init<bool>(), nb::arg("setHi") = false)
        .def("__str__", &rocisa::VCC::toString)
        .def("__deepcopy__", [](const rocisa::VCC& self, nb::dict&) { return rocisa::VCC(self); })
        .def("__getstate__", [](const rocisa::VCC& self) { return std::make_tuple(self.setHi); })
        .def("__setstate__",
             [](rocisa::VCC& self, std::tuple<bool> t) { new(&self) rocisa::VCC(std::get<0>(t)); });

    nb::class_<rocisa::HWRegContainer, rocisa::Container>(m_con, "HWRegContainer")
        .def(nb::init<const std::string&, const std::vector<int>&>(),
             nb::arg("reg"),
             nb::arg("value"))
        .def("__str__", &rocisa::HWRegContainer::toString)
        .def("__deepcopy__",
             [](const rocisa::HWRegContainer& self, nb::dict&) {
                 return rocisa::HWRegContainer(self);
             })
        .def("__getstate__",
             [](const rocisa::HWRegContainer& self) {
                 return std::make_tuple(self.reg, self.value);
             })
        .def("__setstate__",
             [](rocisa::HWRegContainer& self, std::tuple<std::string, std::vector<int>> t) {
                 new(&self) rocisa::HWRegContainer(std::get<0>(t), std::get<1>(t));
             });

    nb::class_<rocisa::RegName>(m_con, "RegName")
        .def(nb::init<const std::string&, const std::vector<int>&>(),
             nb::arg("name"),
             nb::arg("offsets"))
        .def_rw("name", &rocisa::RegName::name)
        // Exposing the vector of offsets can assign a new list to offsets, but cannot modify the offsets.
        .def("getOffsets", [](const rocisa::RegName& self) { return self.offsets; })
        .def("setOffset",
             [](rocisa::RegName& self, int i, int offset) {
                 if(i >= self.offsets.size())
                     throw std::out_of_range("Index out of range");
                 self.offsets[i] = offset;
             })
        .def("addOffset", [](rocisa::RegName& self, int offset) { self.offsets.push_back(offset); })
        .def("getTotalOffsets", &rocisa::RegName::getTotalOffsets)
        .def("__eq__", &rocisa::RegName::operator==)
        .def("__ne__", &rocisa::RegName::operator!=)
        .def("__str__", &rocisa::RegName::toString)
        .def("__hash__", &rocisa::RegName::hash)
        .def("__deepcopy__",
             [](const rocisa::RegName& self, nb::dict&) { return rocisa::RegName(self); })
        .def("__getstate__",
             [](const rocisa::RegName& self) { return std::make_tuple(self.name, self.offsets); })
        .def("__setstate__",
             [](rocisa::RegName& self, std::tuple<std::string, std::vector<int>> t) {
                 new(&self) rocisa::RegName(std::get<0>(t), std::get<1>(t));
             });

    nb::class_<rocisa::RegisterContainer, rocisa::Container>(m_con, "RegisterContainer")
        .def(nb::init<const std::string&, const std::optional<rocisa::RegName>&, int, float>(),
             nb::arg("regType"),
             nb::arg("regName"),
             nb::arg("regIdx"),
             nb::arg("regNum"))
        .def_rw("regIdx", &rocisa::RegisterContainer::regIdx)
        .def_rw("regName", &rocisa::RegisterContainer::regName)
        .def_rw("regNum", &rocisa::RegisterContainer::regNum)
        .def_rw("regType", &rocisa::RegisterContainer::regType)
        .def("setInlineAsm", &rocisa::RegisterContainer::setInlineAsm)
        .def("setMinus", &rocisa::RegisterContainer::setMinus)
        .def("setAbs", &rocisa::RegisterContainer::setAbs)
        .def("getMinus", &rocisa::RegisterContainer::getMinus)
        .def("replaceRegName",
             nb::overload_cast<const std::string&, int>(&rocisa::RegisterContainer::replaceRegName))
        .def("replaceRegName",
             nb::overload_cast<const std::string&, const std::string&>(
                 &rocisa::RegisterContainer::replaceRegName))
        .def("getRegNameWithType", &rocisa::RegisterContainer::getRegNameWithType)
        .def("getCompleteRegNameWithType", &rocisa::RegisterContainer::getCompleteRegNameWithType)
        .def("splitRegContainer", &rocisa::RegisterContainer::splitRegContainer)
        .def(
            "__eq__",
            [](const rocisa::RegisterContainer& self, nb::object other) {
                if(!nb::isinstance<rocisa::RegisterContainer>(other))
                    return false;
                return nb::cast<rocisa::RegisterContainer>(other) == self;
            },
            nb::arg("other").none())
        .def("__hash__", &rocisa::RegisterContainer::hash)
        .def("__str__", &rocisa::RegisterContainer::toString)
        .def("__and__", &rocisa::RegisterContainer::operator&&)
        .def("__deepcopy__",
             [](const rocisa::RegisterContainer& self, nb::dict&) {
                 return rocisa::RegisterContainer(self);
             })
        .def("__getstate__",
             [](const rocisa::RegisterContainer& self) {
                 return std::make_tuple(self.regType, self.regName, self.regIdx, self.regNum);
             })
        .def("__setstate__",
             [](rocisa::RegisterContainer&                                        self,
                std::tuple<std::string, std::optional<rocisa::RegName>, int, int> t) {
                 new(&self) rocisa::RegisterContainer(
                     std::get<0>(t), std::get<1>(t), std::get<2>(t), (float)std::get<3>(t));
             });

    nb::class_<rocisa::HolderContainer, rocisa::RegisterContainer>(m_con, "HolderContainer")
        .def(nb::init<const std::string&, const std::string&, float>(),
             nb::arg("regType"),
             nb::arg("holderName"),
             nb::arg("regNum"))
        .def(nb::init<const std::string&, int, float>(),
             nb::arg("regType"),
             nb::arg("holderIdx"),
             nb::arg("regNum"))
        .def_rw("regIdx", &rocisa::RegisterContainer::regIdx)
        .def_rw("regName", &rocisa::RegisterContainer::regName)
        .def_rw("regNum", &rocisa::RegisterContainer::regNum)
        .def_rw("regType", &rocisa::RegisterContainer::regType)
        .def_rw("holderName", &rocisa::HolderContainer::holderName)
        .def_rw("holderIdx", &rocisa::HolderContainer::holderIdx)
        .def_rw("holderType", &rocisa::HolderContainer::holderType)
        .def("setRegNum", &rocisa::HolderContainer::setRegNum)
        .def("getCopiedRC", &rocisa::HolderContainer::getCopiedRC)
        .def("splitRegContainer", &rocisa::HolderContainer::splitRegContainer)
        .def("__deepcopy__",
             [](const rocisa::HolderContainer& self, nb::dict&) {
                 return rocisa::HolderContainer(self);
             })
        .def("__getstate__",
             [](const rocisa::HolderContainer& self) {
                 return std::make_tuple(self.holderName,
                                        self.holderIdx,
                                        self.holderType,
                                        self.regType,
                                        self.regName,
                                        self.regIdx,
                                        self.regNum);
             })
        .def("__setstate__",
             [](rocisa::HolderContainer& self,
                std::tuple<std::string,
                           int,
                           int,
                           std::string,
                           std::optional<rocisa::RegName>,
                           int,
                           int>          t) {
                 new(&self)
                     rocisa::HolderContainer(std::get<3>(t), std::get<0>(t), (float)std::get<6>(t));
                 self.holderName = std::get<0>(t);
                 self.holderIdx  = std::get<1>(t);
                 self.holderType = std::get<2>(t);
                 self.regType    = std::get<3>(t);
                 self.regName    = std::get<4>(t);
                 self.regIdx     = std::get<5>(t);
                 self.regNum     = std::get<6>(t);
             });

    nb::class_<rocisa::ContinuousRegister>(m_con, "ContinuousRegister")
        .def(
            "__init__",
            [](rocisa::ContinuousRegister& self, uint32_t idx, uint32_t size) {
                new(&self) rocisa::ContinuousRegister(idx, size);
            },
            nb::arg("idx"),
            nb::arg("size"))
        .def_ro("idx", &rocisa::ContinuousRegister::idx)
        .def_ro("size", &rocisa::ContinuousRegister::size)
        .def("__deepcopy__",
             [](const rocisa::ContinuousRegister& self, nb::dict&) {
                 return rocisa::ContinuousRegister(self);
             })
        .def("__getstate__",
             [](const rocisa::ContinuousRegister& self) {
                 return std::make_tuple(self.idx, self.size);
             })
        .def("__setstate__",
             [](rocisa::ContinuousRegister& self, std::tuple<uint32_t, uint32_t> t) {
                 new(&self) rocisa::ContinuousRegister(std::get<0>(t), std::get<1>(t));
             });
}
