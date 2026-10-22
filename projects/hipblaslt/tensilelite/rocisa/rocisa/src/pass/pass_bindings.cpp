// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pass.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void init_pass(nb::module_ m)
{
    auto m_pass = m.def_submodule("asmpass", "rocIsa pass submodule.");
    m_pass.def("getActFuncModuleName", &rocisa::getActFuncModuleName, "getActFuncModuleName.");
    m_pass.def("getActFuncBranchModuleName",
               &rocisa::getActFuncBranchModuleName,
               "getActFuncBranchModuleName.");
    m_pass.def("rocIsaPass", &rocisa::rocIsaPass, "rocIsaPass.");

    nb::class_<rocisa::rocIsaPassOption>(m_pass, "rocIsaPassOption")
        .def(nb::init<>())
        .def_rw("insertDelayAlu", &rocisa::rocIsaPassOption::insertDelayAlu)
        .def_rw("removeDupFunc", &rocisa::rocIsaPassOption::removeDupFunc)
        .def_rw("removeDupAssign", &rocisa::rocIsaPassOption::removeDupAssign)
        .def_rw("getCycles", &rocisa::rocIsaPassOption::getCycles)
        .def_rw("numWaves", &rocisa::rocIsaPassOption::numWaves);

    nb::class_<rocisa::rocIsaPassResult>(m_pass, "rocIsaPassResult")
        .def(nb::init<>())
        .def_ro("cycles",   &rocisa::rocIsaPassResult::cycles)
        .def_ro("maxVgpr",  &rocisa::rocIsaPassResult::maxVgpr);
}
