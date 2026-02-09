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
#include "functions/argument.hpp"
#include "functions/f_math.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace rocisa
{
    ////////////////////////////////////////
    // init lds state
    ////////////////////////////////////////
    std::shared_ptr<Module> DSInit(const ContinuousRegister& tmpVgprRes,
                                   int                       numThreads,
                                   int                       ldsNumElements,
                                   int                       initValue)
    {
        if(tmpVgprRes.size <= 1)
        {
            throw std::runtime_error("tmpVgprRes.size must be greater than 1");
        }
        int tmp     = tmpVgprRes.idx;
        int tmpAddr = tmp + 1;

        auto module = std::make_shared<Module>("initLds");
        module->addT<SWaitCnt>(0, 0, 0, 0, "init lds state");
        module->addT<SBarrier>("init LDS");
        module->addT<VMovB32>(vgpr(tmp), initValue, std::nullopt, "Init value");
        module->addT<VLShiftLeftB32>(
            vgpr(tmpAddr), 2, vgpr("Serial"), "set per-thread address to init LDS");

        int writesPerThread = ((ldsNumElements - 1) / numThreads / 4) + 1;
        for(int i = 0; i < writesPerThread; ++i)
        {
            module->addT<DSStoreB32>(
                vgpr(tmpAddr), vgpr(tmp), DSModifiers(i * numThreads * 4), "init lds");
        }

        module->addT<SWaitCnt>(0, 0, 0, 0, "wait for LDS init to complete");
        module->addT<SBarrier>("init LDS exit");

        return module;
    }
} // namespace rocisa

void math_func(nb::module_ m);
void branch_func(nb::module_ m);
void cast_func(nb::module_ m);
void argument_func(nb::module_ m);

void init_func(nb::module_ m)
{
    auto m_func = m.def_submodule("functions", "rocIsa functions submodule.");
    m_func.def("DSInit",
               &rocisa::DSInit,
               nb::arg("tmpVgprRes"),
               nb::arg("numThreads"),
               nb::arg("ldsNumElements"),
               nb::arg("initValue") = 0,
               "Initialize LDS state.");

    math_func(m_func);
    branch_func(m_func);
    cast_func(m_func);
    argument_func(m_func);
}
