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

#include <Tensile/Comparison.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

namespace Tensile
{
    template <>
    struct Comparison<ContractionProblem::FreeIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblem::FreeIndex const& lhs,
                           ContractionProblem::FreeIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d, lhs.c, rhs.c, lhs.i, rhs.i, lhs.isA, rhs.isA);
        }
    };

    template <>
    struct Comparison<ContractionProblem::BatchIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblem::BatchIndex const& lhs,
                           ContractionProblem::BatchIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d, lhs.c, rhs.c, lhs.a, rhs.a, lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblem::BoundIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblem::BoundIndex const& lhs,
                           ContractionProblem::BoundIndex const& rhs)
        {
            return LexicographicCompare(lhs.a, rhs.a, lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblem>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblem const& lhs, ContractionProblem const& rhs)
        {
            return LexicographicCompare(lhs.operationIdentifier(),
                                        rhs.operationIdentifier(),
                                        lhs.highPrecisionAccumulate(),
                                        rhs.highPrecisionAccumulate(),
                                        lhs.kernelLanguage(),
                                        rhs.kernelLanguage(),
                                        lhs.deterministicMode(),
                                        rhs.deterministicMode(),
                                        lhs.arithmeticUnit(),
                                        rhs.arithmeticUnit(),
                                        lhs.a(),
                                        rhs.a(),
                                        lhs.b(),
                                        rhs.b(),
                                        lhs.c(),
                                        rhs.c(),
                                        lhs.d(),
                                        rhs.d(),
                                        lhs.workspaceSize(),
                                        rhs.workspaceSize(),
                                        lhs.stridedBatched(),
                                        rhs.stridedBatched(),
                                        lhs.performanceMetric(),
                                        rhs.performanceMetric(),
                                        lhs.fp16AltImpl(),
                                        rhs.fp16AltImpl(),
                                        lhs.fp16AltImplRound(),
                                        rhs.fp16AltImplRound(),
                                        lhs.stochasticRounding(),
                                        rhs.stochasticRounding(),
                                        lhs.f32XdlMathOp(),
                                        rhs.f32XdlMathOp());
        }
    };
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::ContractionProblem>
    {
        inline size_t operator()(Tensile::ContractionProblem const& problem) const
        {
            return Tensile::hash_combine(problem.operationIdentifier(),
                                         problem.a(),
                                         problem.b(),
                                         problem.c(),
                                         problem.d(),
                                         problem.highPrecisionAccumulate(),
                                         problem.kernelLanguage(),
                                         problem.deterministicMode(),
                                         problem.arithmeticUnit(),
                                         problem.workspaceSize(),
                                         problem.stridedBatched(),
                                         problem.performanceMetric(),
                                         problem.fp16AltImpl(),
                                         problem.fp16AltImplRound(),
                                         problem.stochasticRounding(),
                                         problem.f32XdlMathOp());
        }
    };

} // namespace std
