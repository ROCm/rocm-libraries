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

#pragma once

#include <cstdlib>
#include <string>

#include <Tensile/Singleton.hpp>

namespace Tensile
{
    /**
 * @brief Common place for defining flags which enable debug behaviour.
 */
    class Debug : public LazySingleton<Debug>
    {
    public:
        bool printPropertyEvaluation() const;
        bool printSelectedKernelName() const;
        bool printPredicateEvaluation() const;
        bool printDeviceSelection() const;
        bool printCodeObjectInfo() const;

        bool printKernelArguments() const;

        bool printDataInit() const;

        // print tensor dims, strides, memory sizes
        bool printTensorInfo() const;

        // 3 levels of debugging for the convolution reference debug
        bool printConvolutionReference1() const;
        bool printConvolutionReference2() const;
        bool printConvolutionReference3() const;

        // if tensors are printed, use hexadecimal output format
        bool printTensorModeHex() const;

        bool printLibraryVersion() const;

        bool printLookupEfficiency() const;

        bool printWinningKernelName() const;

        bool printKernelCommonParams() const;

        bool printSolutionSelectionTime() const;

        bool printLibraryLogicIndex() const;

        bool printOverrideLogs() const;

        bool printStreamKGridInfo() const;

        bool naivePropertySearch() const;

        bool skipKernelLaunch() const;
        bool skipInitKernelLaunch() const;

        bool enableDebugSelection() const;

        int  useExperimentalSelection() const;
        bool getSolutionSelectionTrace() const;

        std::string getMetric() const;

        int getSolutionIndex() const;

    private:
        friend LazySingleton<Debug>;

        int         m_value;
        int         m_value2;
        bool        m_naivePropertySearch = false;
        bool        m_debugSelection      = false;
        int         m_experimentSelection = 0;
        int         m_solution_index      = -1;
        bool        m_solselTrace         = false;
        std::string m_metric              = "";

        Debug();
    };
} // namespace Tensile
