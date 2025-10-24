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
#include "base.hpp"

namespace nb = nanobind;

namespace rocisa
{
    std::string isaToGfx(const nb::tuple& arch)
    {
        return getGfxNameTuple(
            IsaVersion{nb::cast<int>(arch[0]), nb::cast<int>(arch[1]), nb::cast<int>(arch[2])});
    }

    std::string isaToGfx(const IsaVersion& arch)
    {
        return getGfxNameTuple(arch);
    }

    std::string getGlcBitName(bool hasGLCModifier)
    {
        if(hasGLCModifier)
            return "glc";
        return "sc0";
    }

    std::string getSlcBitName(bool hasGLCModifier)
    {
        if(hasGLCModifier)
            return "slc";
        return "sc1";
    }

    // Force init the instance
    auto isInit = rocIsa::getInstance().isInit();

}

void init_base(nb::module_ m)
{
    m.def("isaToGfx", nb::overload_cast<const nb::tuple&>(&rocisa::isaToGfx));
    m.def("isaToGfx", nb::overload_cast<const IsaVersion&>(&rocisa::isaToGfx));
    m.def("getGlcBitName", &rocisa::getGlcBitName);
    m.def("getSlcBitName", &rocisa::getSlcBitName);

    nb::class_<rocisa::IsaInfo>(m, "IsaInfo")
        .def(nb::init<>())
        .def_ro("asmCaps", &rocisa::IsaInfo::asm_caps)
        .def_ro("archCaps", &rocisa::IsaInfo::arch_caps)
        .def_ro("regCaps", &rocisa::IsaInfo::reg_caps)
        .def_ro("asmBugs", &rocisa::IsaInfo::asm_bugs)
        .def("__getstate__",
             [](const rocisa::IsaInfo& self) {
                 return std::make_tuple(
                     self.asm_caps, self.arch_caps, self.reg_caps, self.asm_bugs);
             })
        .def("__setstate__",
             [](rocisa::IsaInfo&                               self,
                const std::tuple<std::map<std::string, int>,
                                 std::map<std::string, int>,
                                 std::map<std::string, int>,
                                 std::map<std::string, bool>>& state) {
                 new(&self) rocisa::IsaInfo{std::get<0>(state),
                                            std::get<1>(state),
                                            std::get<2>(state),
                                            std::get<3>(state)};
             });

    nb::class_<rocisa::rocIsa>(m, "rocIsa")
        .def_static("getInstance",
                    &rocisa::rocIsa::getInstance,
                    nb::rv_policy::reference,
                    "Get instance of ISA.")
        .def("init",
             &rocisa::rocIsa::init,
             nb::arg("arch"),
             nb::arg("assemblerPath"),
             nb::arg("debug") = false,
             "Init ISA.")
        .def("isInit", &rocisa::rocIsa::isInit, "Check if any ISA is init.")
        .def("setKernel", &rocisa::rocIsa::setKernel, "Set kernel with given ISA and wavefront.")
        .def("getKernel", &rocisa::rocIsa::getKernel, "Get current set kernel.")
        .def("getIsaInfo",
             nb::overload_cast<const nb::tuple&>(&rocisa::rocIsa::getIsaInfo),
             "Get ISA info.")
        .def("getIsaInfo",
             nb::overload_cast<const IsaVersion&>(&rocisa::rocIsa::getIsaInfo),
             "Get ISA info.")
        .def("getAsmCaps", &rocisa::rocIsa::getAsmCaps, "Get asm capabilities.")
        .def("getRegCaps", &rocisa::rocIsa::getRegCaps, "Get reg capabilities.")
        .def("getArchCaps", &rocisa::rocIsa::getArchCaps, "Get arch capabilities.")
        .def("getAsmBugs", &rocisa::rocIsa::getAsmBugs, "Get asm bugs.")
        .def("getData", &rocisa::rocIsa::getData, "Get data for pickling.")
        .def("setData", &rocisa::rocIsa::setData, "Set data for pickling.")
        .def("getOutputOptions", &rocisa::rocIsa::getOutputOptions, "Get output options.")
        .def("setOutputOptions", &rocisa::rocIsa::setOutputOptions, "Set output options.");

    auto m_base = m.def_submodule("base", "rocIsa base submodule.");
    nb::class_<IsaVersion>(m_base, "IsaVersion")
        .def(nb::init<>())
        .def("__getitem__",
             [](const IsaVersion& a, size_t i) {
                 if(i >= a.size())
                     throw std::out_of_range("Index out of range");
                 return a[i];
             })
        .def("__setitem__",
             [](IsaVersion& a, size_t i, int v) {
                 if(i >= a.size())
                     throw std::out_of_range("Index out of range");
                 a[i] = v;
             })
        .def("__len__", [](const IsaVersion& a) { return a.size(); })
        .def("__eq__",
             [](const IsaVersion& a, const IsaVersion& b) {
                 return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]);
             })
        .def("__eq__",
             [](const IsaVersion& a, const nb::tuple& b) {
                 if(b.size() != a.size())
                     return false;
                 return (a[0] == nb::cast<int>(b[0])) && (a[1] == nb::cast<int>(b[1]))
                        && (a[2] == nb::cast<int>(b[2]));
             })
        .def("__eq__",
             [](const nb::tuple& a, const IsaVersion& b) {
                 if(b.size() != a.size())
                     return false;
                 return (nb::cast<int>(a[0]) == b[0]) && (nb::cast<int>(a[1]) == b[1])
                        && (nb::cast<int>(a[2]) == b[2]);
             })
        .def("__str__",
             [](const IsaVersion& a) {
                 std::string s;
                 for(auto a_u : a)
                     s += std::to_string(a_u);
                 return s;
             })
        .def("__getstate__",
             [](const IsaVersion& self) { return nb::make_tuple(self[0], self[1], self[2]); })
        .def("__setstate__", [](IsaVersion& self, nb::tuple state) {
            self[0] = nb::cast<int>(state[0]);
            self[1] = nb::cast<int>(state[1]);
            self[2] = nb::cast<int>(state[2]);
        });

    nb::class_<rocisa::KernelInfo>(m_base, "KernelInfo")
        .def(nb::init<>())
        .def_rw("isa", &rocisa::KernelInfo::isaVersion)
        .def_rw("wavefrontSize", &rocisa::KernelInfo::wavefront)
        .def("__getstate__",
             [](const rocisa::KernelInfo& self) {
                 return std::make_tuple(self.isaVersion, self.wavefront);
             })
        .def("__setstate__",
             [](rocisa::KernelInfo& self, const std::tuple<IsaVersion, int>& state) {
                 new(&self) rocisa::KernelInfo(std::get<0>(state), std::get<1>(state));
             });

    nb::class_<rocisa::OutputOptions>(m_base, "OutputOptions")
        .def(nb::init<>())
        .def_rw("outputNoComment", &rocisa::OutputOptions::outputNoComment)
        .def(
            "__getstate__",
            [](const rocisa::OutputOptions& self) { return std::make_tuple(self.outputNoComment); })
        .def("__setstate__", [](rocisa::OutputOptions& self, const std::tuple<bool>& state) {
            new(&self) rocisa::OutputOptions{std::get<0>(state)};
        });

    nb::class_<rocisa::Item>(m_base, "Item")
        .def(nb::init<const char*>(), nb::arg("name") = "")
        .def_rw("name", &rocisa::Item::name)
        .def_rw("parent", &rocisa::Item::parent, nb::arg("parent").none(), nb::rv_policy::reference)
        .def_prop_ro("asmCaps", &rocisa::Item::getAsmCaps)
        .def_prop_ro("regCaps", &rocisa::Item::getRegCaps)
        .def_prop_ro("archCaps", &rocisa::Item::getArchCaps)
        .def_prop_ro("asmBugs", &rocisa::Item::getAsmBugs)
        .def_prop_ro("kernel", &rocisa::Item::kernel)
        .def("prettyPrint", &rocisa::Item::prettyPrint, "Print the instance and the name of Item.")
        .def("__str__", &rocisa::Item::toString)
        .def("__deepcopy__",
             [](rocisa::Item& self, nb::dict mamo) {
                 rocisa::Item* copy = new rocisa::Item(self.name);
                 copy->parent       = self.parent;
                 return copy;
             })
        .def("__getstate__", [](const rocisa::Item& self) { return std::make_tuple(self.name); })
        .def("__setstate__", [](rocisa::Item& self, const std::tuple<std::string>& state) {
            new(&self) rocisa::Item(std::get<0>(state));
        });

    nb::class_<rocisa::DummyItem, rocisa::Item>(m_base, "DummyItem")
        .def(nb::init<>())
        .def("__deepcopy__",
             [](rocisa::DummyItem& self, nb::dict mamo) { return new rocisa::DummyItem(self); })
        .def("__getstate__", [](const rocisa::DummyItem& self) { return std::make_tuple(); })
        .def("__setstate__", [](rocisa::DummyItem& self, const std::tuple<>& state) {
            new(&self) rocisa::DummyItem();
        });
}
