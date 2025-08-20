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

#include <Tensile/Comparison.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

namespace TensileLite
{
    template <>
    struct Comparison<ContractionProblemGemm::FreeIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm::FreeIndex const& lhs,
                           ContractionProblemGemm::FreeIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d, lhs.c, rhs.c, lhs.i, rhs.i, lhs.isA, rhs.isA);
        }
    };

    template <>
    struct Comparison<ContractionProblemGemm::BatchIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm::BatchIndex const& lhs,
                           ContractionProblemGemm::BatchIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d, lhs.c, rhs.c, lhs.a, rhs.a, lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblemGemm::BoundIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm::BoundIndex const& lhs,
                           ContractionProblemGemm::BoundIndex const& rhs)
        {
            return LexicographicCompare(lhs.a, rhs.a, lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblemGemm>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm const& lhs, ContractionProblemGemm const& rhs)
        {
            return LexicographicCompare(lhs.operationIdentifier(),
                                        rhs.operationIdentifier(),
                                        lhs.highPrecisionAccumulate(),
                                        rhs.highPrecisionAccumulate(),
                                        lhs.kernelLanguage(),
                                        rhs.kernelLanguage(),
                                        lhs.deterministicMode(),
                                        rhs.deterministicMode(),
                                        lhs.a(),
                                        rhs.a(),
                                        lhs.b(),
                                        rhs.b(),
                                        lhs.c(),
                                        rhs.c(),
                                        lhs.d(),
                                        rhs.d(),
                                        lhs.e(),
                                        rhs.e(),
                                        lhs.computeInputType(),
                                        rhs.computeInputType(),
                                        lhs.workspaceSize(),
                                        rhs.workspaceSize(),
                                        lhs.stridedBatched(),
                                        rhs.stridedBatched(),
                                        lhs.groupedGemm(),
                                        rhs.groupedGemm(),
                                        lhs.performanceMetric(),
                                        rhs.performanceMetric(),
                                        lhs.activationType(),
                                        rhs.activationType(),
                                        lhs.activationComputeType(),
                                        rhs.activationComputeType(),
                                        lhs.activationNoGuard(),
                                        rhs.activationNoGuard(),
                                        lhs.useGradient(),
                                        rhs.useGradient(),
                                        lhs.useBias(),
                                        rhs.useBias(),
                                        lhs.biasSrc(),
                                        rhs.biasSrc(),
                                        lhs.useE(),
                                        rhs.useE(),
                                        lhs.useScaleAB(),
                                        rhs.useScaleAB(),
                                        lhs.useScaleCD(),
                                        rhs.useScaleCD(),
                                        lhs.useScaleAlphaVec(),
                                        rhs.useScaleAlphaVec(),
                                        lhs.outputAmaxD(),
                                        rhs.outputAmaxD(),
                                        lhs.f32XdlMathOp(),
                                        rhs.f32XdlMathOp(),
                                        lhs.swizzleTensorA(),
                                        rhs.swizzleTensorA(),
                                        lhs.swizzleTensorB(),
                                        rhs.swizzleTensorB());
        }
    };
} // namespace TensileLite

namespace std
{
    template <>
    struct hash<TensileLite::ContractionProblemGemm>
    {
        inline size_t operator()(TensileLite::ContractionProblemGemm const& problem) const
        {
            return TensileLite::hash_combine(problem.operationIdentifier(),
                                             problem.a(),
                                             problem.b(),
                                             problem.c(),
                                             problem.d(),
                                             problem.e(),
                                             problem.computeInputType(),
                                             problem.highPrecisionAccumulate(),
                                             problem.kernelLanguage(),
                                             problem.deterministicMode(),
                                             problem.workspaceSize(),
                                             problem.stridedBatched(),
                                             problem.groupedGemm(),
                                             problem.performanceMetric(),
                                             problem.activationType(),
                                             problem.activationComputeType(),
                                             problem.activationNoGuard(),
                                             problem.useGradient(),
                                             problem.useBias(),
                                             problem.biasSrc(),
                                             problem.useE(),
                                             problem.useScaleAB(),
                                             problem.useScaleCD(),
                                             problem.useScaleAlphaVec(),
                                             problem.outputAmaxD(),
                                             problem.f32XdlMathOp(),
                                             problem.swizzleTensorA(),
                                             problem.swizzleTensorB());
        }
    };

    template <>
    struct hash<std::vector<TensileLite::ContractionProblemGemm>>
    {
        inline size_t
            operator()(std::vector<TensileLite::ContractionProblemGemm> const& problems) const
        {
            size_t hash = 0;
            for(int idx = 0; idx < problems.size(); idx++)
            {
                auto problem = problems[idx];
                hash += TensileLite::hash_combine(problem.operationIdentifier(),
                                                  problem.a(),
                                                  problem.b(),
                                                  problem.c(),
                                                  problem.d(),
                                                  problem.e(),
                                                  problem.computeInputType(),
                                                  problem.highPrecisionAccumulate(),
                                                  problem.kernelLanguage(),
                                                  problem.deterministicMode(),
                                                  problem.workspaceSize(),
                                                  problem.stridedBatched(),
                                                  problem.groupedGemm(),
                                                  problem.performanceMetric(),
                                                  problem.activationType(),
                                                  problem.activationComputeType(),
                                                  problem.activationNoGuard(),
                                                  problem.useGradient(),
                                                  problem.useBias(),
                                                  problem.biasSrc(),
                                                  problem.useE(),
                                                  problem.useScaleAB(),
                                                  problem.useScaleCD(),
                                                  problem.useScaleAlphaVec(),
                                                  problem.outputAmaxD(),
                                                  problem.f32XdlMathOp(),
                                                  problem.swizzleTensorA(),
                                                  problem.swizzleTensorB());
            }
            return hash;
        }
    };

} // namespace std
