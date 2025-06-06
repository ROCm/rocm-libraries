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
#include "pass.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace rocisa
{
    rocIsaPassResult rocIsaPass(std::shared_ptr<KernelBody>& kernel, const rocIsaPassOption& option)
    {
        rocIsaPassResult result;
        if(option.removeDupFunc)
        {
            removeDuplicatedFunction(kernel->body);
        }

        auto assignDict = getAssignmentDict(kernel->body);
        compositeToInstruction(kernel->body);
        if(option.doOpt())
        {
            auto maxVgpr = kernel->totalVgprs;
            auto maxSgpr = kernel->totalSgprs;
            auto graph   = buildGraph(kernel->body, maxVgpr, maxSgpr, assignDict);
            if(option.removeDupAssign)
            {
                removeDuplicateAssignment(graph);
            }
        }

        if(option.insertDelayAlu)
        {
            insertDelayAlu(kernel->body);
        }

        if(option.getCycles)
            result.cycles = getCycles(kernel->body, option.numWaves);

        return std::move(result);
    }
} // namespace rocisa

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
        .def_ro("cycles", &rocisa::rocIsaPassResult::cycles);
}
