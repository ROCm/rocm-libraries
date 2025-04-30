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

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/Tensile.hpp>

#include <boost/program_options.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {

        namespace po = boost::program_options;

        class ClientProblemFactory
        {
        public:
            ClientProblemFactory(po::variables_map const& args);
            ~ClientProblemFactory();

            ClientProblemFactory(ContractionProblem const& problem)
                : m_problems({problem})
            {
            }

            template <typename Iterator>
            ClientProblemFactory(Iterator begin, Iterator end)
                : m_problems(begin, end)
            {
            }

            std::vector<ContractionProblem> const& problems() const;

            std::vector<ContractionProblem> createProblems();

        private:
            std::vector<ContractionProblem> m_problems;

            ContractionProblem::FreeIndices  m_freeIndices;
            ContractionProblem::BatchIndices m_batchIndices;
            ContractionProblem::BoundIndices m_boundIndices;

            DataType          m_aType;
            DataType          m_bType;
            DataType          m_cType;
            DataType          m_dType;
            DataType          m_alphaType;
            DataType          m_betaType;
            bool              m_stridedBatched;
            bool              m_highPrecisionAccumulate;
            bool              m_deterministicMode;
            bool              m_cEqualsD;
            DataType          m_f32XdlMathOp;
            ArithmeticUnit    m_arithmeticUnit;
            KernelLanguage    m_kernelLanguage;
            PerformanceMetric m_performanceMetric;
            bool              m_fp16AltImpl;
            bool              m_fp16AltImplRound;
            bool              m_stochasticRounding;

            std::vector<std::vector<size_t>> m_problemSizes;
            std::vector<std::vector<size_t>> m_aStrides;
            std::vector<std::vector<size_t>> m_bStrides;
            std::vector<std::vector<size_t>> m_cStrides;
            std::vector<std::vector<size_t>> m_dStrides;
            std::vector<std::vector<size_t>> m_aZeroPads;
            std::vector<std::vector<size_t>> m_bZeroPads;
            std::vector<std::vector<size_t>> m_convProblemSizes;

            TensorOps m_aOps;
            TensorOps m_bOps;
            TensorOps m_cOps;
            TensorOps m_dOps;

            size_t m_aOffset;
            size_t m_bOffset;
            size_t m_cOffset;
            size_t m_dOffset;

            double m_beta;
            double m_alpha;
        };

    } // namespace Client
} // namespace Tensile
