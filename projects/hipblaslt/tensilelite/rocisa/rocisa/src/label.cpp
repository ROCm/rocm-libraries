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
#include "label.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void init_label(nb::module_ m)
{
    auto m_label = m.def_submodule("label", "rocIsa label submodule.");
    m_label.def("magicGenerator",
                &rocisa::magicGenerator,
                "Generates a random string with length equals to 17.");
    nb::class_<rocisa::LabelManager>(m_label, "LabelManager")
        .def(nb::init<>())
        .def("addName", &rocisa::LabelManager::addName, "Add name to the LabelManager.")
        .def("getName", &rocisa::LabelManager::getName, "Get name from the LabelManager.")
        .def("getNameInc",
             &rocisa::LabelManager::getNameInc,
             "Get name increment if the name exists in the LabelManager.")
        .def("getNameIndex",
             &rocisa::LabelManager::getNameIndex,
             "Get specific name index in the LabelManager.")
        .def("getUniqueName",
             &rocisa::LabelManager::getUniqueName,
             "Get unique name from the LabelManager.")
        .def("getUniqueNamePrefix",
             &rocisa::LabelManager::getUniqueNamePrefix,
             "Get unique name with a custom prefix from the LabelManager.")
        .def("__deepcopy__",
             [](rocisa::LabelManager& self, nb::dict mamo) {
                 rocisa::LabelManager* copy = new rocisa::LabelManager(self.getData());
                 return copy;
             })
        .def("__getstate__",
             [](const rocisa::LabelManager& self) { return std::make_tuple(self.getData()); })
        .def("__setstate__",
             [](rocisa::LabelManager& self, const std::tuple<std::map<std::string, int>>& state) {
                 new(&self) rocisa::LabelManager(std::get<0>(state));
             });
}
