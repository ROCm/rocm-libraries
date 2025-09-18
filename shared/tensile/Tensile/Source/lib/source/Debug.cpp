/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <Tensile/Debug.hpp>

#include <mutex>

#ifndef DEBUG_SM
#define DEBUG_SM 0
#endif

#ifndef DEBUG_SM2
#define DEBUG_SM2 0
#endif

namespace Tensile
{
    std::once_flag debug_init;

    bool Debug::printPropertyEvaluation() const
    {
        return m_value & 0x2;
    }

    bool Debug::printSelectedKernelName() const
    {
        return m_value & 0x4;
    }

    bool Debug::printDeviceSelection() const
    {
        return m_value & 0x8;
    }

    bool Debug::printPredicateEvaluation() const
    {
        return m_value & 0x10;
    }

    bool Debug::printCodeObjectInfo() const
    {
        return m_value & 0x20;
    }

    bool Debug::printKernelArguments() const
    {
        return m_value & 0x40;
    }

    bool Debug::printTensorInfo() const
    {
        return m_value & 0x80;
    }

    bool Debug::printConvolutionReference1() const
    {
        return m_value & 0x100;
    }
    bool Debug::printConvolutionReference2() const
    {
        return m_value & 0x200;
    }
    bool Debug::printConvolutionReference3() const
    {
        return m_value & 0x400;
    }

    bool Debug::printTensorModeHex() const
    {
        return m_value & 0x800;
    }

    bool Debug::printDataInit() const
    {
        return m_value & 0x1000;
    }

    bool Debug::printLibraryVersion() const
    {
        return m_value & 0x2000;
    }

    bool Debug::printLookupEfficiency() const
    {
        return m_value & 0x4000;
    }

    bool Debug::printWinningKernelName() const
    {
        return m_value & 0x8000;
    }

    bool Debug::printKernelCommonParams() const
    {
        return m_value & 0x80000;
    }

    bool Debug::printSolutionSelectionTime() const
    {
        return m_value & 0x10000;
    }

    bool Debug::printLibraryLogicIndex() const
    {
        return m_value & 0x20000;
    }

    bool Debug::printOverrideLogs() const
    {
        return m_value & 0x40000;
    }

    bool Debug::printStreamKGridInfo() const
    {
        return m_value & 0x80000;
    }

    bool Debug::naivePropertySearch() const
    {
        return m_naivePropertySearch;
    }

    bool Debug::skipKernelLaunch() const
    {
        return m_value2 & 0x1;
    }

    bool Debug::skipInitKernelLaunch() const
    {
        return m_value2 & 0x2;
    }

    bool Debug::enableDebugSelection() const
    {
        return m_debugSelection;
    }

    int Debug::useExperimentalSelection() const
    {
        return m_experimentSelection;
    }

    std::string Debug::getMetric() const
    {
        return m_metric;
    }

    int Debug::getSolutionIndex() const
    {
        return m_solution_index;
    }

    bool Debug::getSolutionSelectionTrace() const
    {
        return m_solselTrace;
    }

    Debug::Debug()
        : m_value(DEBUG_SM)
        , m_value2(DEBUG_SM2)
    {
        const char* db = std::getenv("TENSILE_DB");
        if(db)
            m_value = strtol(db, nullptr, 0);

        const char* db2 = std::getenv("TENSILE_DB2");
        if(db2)
            m_value2 = strtol(db2, nullptr, 0);

        const char* naive = std::getenv("TENSILE_NAIVE_SEARCH");
        if(naive)
            m_naivePropertySearch = strtol(naive, nullptr, 0) != 0;

        const char* db_select = std::getenv("TENSILE_TAM_SELECTION_ENABLE");
        if(db_select)
            m_debugSelection = strtol(db_select, nullptr, 0) != 0;

        const char* exp_select = std::getenv("TENSILE_EXPERIMENTAL_SELECTION");
        if(exp_select)
            m_experimentSelection = strtol(exp_select, nullptr, 0);

        const char* solsel_trace = std::getenv("TENSILE_SOLUTION_SELECTION_TRACE");
        if(solsel_trace)
            m_solselTrace = strtol(solsel_trace, nullptr, 0) != 0;

        const char* solution_index = std::getenv("TENSILE_SOLUTION_INDEX");
        if(solution_index)
            m_solution_index = strtol(solution_index, nullptr, 0);

        const char* tensile_metric = std::getenv("TENSILE_METRIC");
        if(tensile_metric)
            m_metric = tensile_metric;
    }

} // namespace Tensile
