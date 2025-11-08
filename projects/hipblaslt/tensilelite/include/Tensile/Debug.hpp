/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <cstdlib>
#include <string>
#ifdef Tensile_ENABLE_MARKER
#include <roctracer/roctx.h>
#endif

#include <Tensile/Singleton.hpp>

namespace TensileLite
{
    /**
     * @brief Common place for defining flags which enable debug behaviour.
     */
    class Debug : public LazySingleton<Debug>
    {
    public:
        bool printPropertyEvaluation() const;
        bool printPredicateEvaluation() const;
        bool printDeviceSelection() const;
        bool printCodeObjectInfo() const;
        bool printKernelArguments() const;
        bool printDataInit() const;
        bool printTensorInfo() const; // print tensor dims, strides, memory sizes
        bool printTensorModeHex() const; // if tensors are printed, use hexadecimal output format
        bool printLibraryVersion() const;
        bool printLookupEfficiency() const;
        bool printWinningKernelName() const;

        bool usePredictionLibrary() const;

        bool printLibraryLogicIndex() const;

        bool naivePropertySearch() const;

        bool skipKernelLaunch() const;

        bool useStreamKDataParrallel() const;

        int useExperimentalSelection() const;

        std::string getMetric() const;

        bool getBenchmark() const;

        int getSolutionIndex() const;

        bool printSolutionSelectionTime() const;

        int getGridbasedTopSols() const;

        bool printStreamKGridInfo() const;

        bool gridBasedKDTree() const;

        bool gridBasedBatchExp() const;

        bool disableStaggerU() const;

        __attribute__((always_inline)) inline void markerStart(const char* name) const
        {
#ifdef Tensile_ENABLE_MARKER
            if(m_printMarker)
            {
                roctxRangePush(name);
            }
#endif
        }

        __attribute__((always_inline)) inline void markerStart(const char*        name,
                                                               const std::string& objPath) const
        {
#ifdef Tensile_ENABLE_MARKER
            if(m_printMarker)
            {
                std::string s = name + std::string(": ") + objPath;
                roctxRangePush(s.c_str());
            }
#endif
        }

        __attribute__((always_inline)) inline void markerStop() const
        {
#ifdef Tensile_ENABLE_MARKER
            if(m_printMarker)
            {
                roctxRangePop();
            }
#endif
        }

    private:
        friend LazySingleton<Debug>;

        int         m_value;
        int         m_value2;
        bool        m_naivePropertySearch = false;
        bool        m_dataParallel        = false;
        int         m_experimentSelection = 0;
        int         m_solution_index      = -1;
        bool        m_predictionLib       = false;
        std::string m_metric              = "";
        int         m_gridbasedTopSols    = 1;
        bool        m_benchmark           = false;
        bool        m_gridbasedKdTree     = false;
        bool        m_gridbasedBatchExp   = false;
        bool        m_printMarker         = false;
        bool        m_disableStaggerU     = false;

        Debug();
    };
} // namespace TensileLite
