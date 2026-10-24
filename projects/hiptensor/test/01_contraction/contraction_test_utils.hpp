/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <hiptensor/hiptensor_types.h>

#include "utils.hpp"

namespace hiptensor
{
    // Compares an output tensor of `elements` elements of `dataType` on the device with a
    // reference on the device. Both parts of every complex element are compared.
    inline std::pair<bool, double>
        compareContractionOutput(hiptensorDataType_t          dataType,
                                 void*                        output,
                                 void*                        reference,
                                 std::size_t                  elements,
                                 hiptensorComputeDescriptor_t computeType,
                                 double                       tolerance)
    {
        switch(dataType)
        {
        case HIPTENSOR_R_16F:
            return compareEqualLaunchKernel<_Float16>(
                (_Float16*)output, (_Float16*)reference, elements, computeType, tolerance);
        case HIPTENSOR_R_16BF:
            return compareEqualLaunchKernel<hip_bfloat16>(
                (hip_bfloat16*)output, (hip_bfloat16*)reference, elements, computeType, tolerance);
        case HIPTENSOR_R_32F:
            return compareEqualLaunchKernel<float>(
                (float*)output, (float*)reference, elements, computeType, tolerance);
        case HIPTENSOR_C_32F:
            return compareEqualLaunchKernel<float>(
                (float*)output, (float*)reference, 2 * elements, computeType, tolerance);
        case HIPTENSOR_R_64F:
            return compareEqualLaunchKernel<double>(
                (double*)output, (double*)reference, elements, computeType, tolerance);
        case HIPTENSOR_C_64F:
            return compareEqualLaunchKernel<double>(
                (double*)output, (double*)reference, 2 * elements, computeType, tolerance);
        default:
            return {false, std::numeric_limits<double>::quiet_NaN()};
        }
    }

    // Number of points in the iteration space of a contraction of A and B: the product of the
    // extents of their distinct modes, so a mode carried by both (contracted or batch) counts once.
    inline double contractionIterationPoints(std::vector<int32_t> const&     modesA,
                                             std::vector<std::size_t> const& lengthsA,
                                             std::vector<int32_t> const&     modesB,
                                             std::vector<std::size_t> const& lengthsB)
    {
        std::map<int32_t, double> extents;
        for(std::size_t i = 0; i < modesA.size(); i++)
        {
            extents.emplace(modesA[i], static_cast<double>(lengthsA[i]));
        }
        for(std::size_t i = 0; i < modesB.size(); i++)
        {
            extents.emplace(modesB[i], static_cast<double>(lengthsB[i]));
        }

        double points = 1.0;
        for(auto const& extent : extents)
        {
            points *= extent.second;
        }
        return points;
    }

    // Real floating-point operations per point of the iteration space: a multiply and an add,
    // or four multiplies and four adds for a complex compute type.
    inline double flopsPerPoint(hiptensorComputeDescriptor_t computeType)
    {
        bool isComplex = (computeType == HIPTENSOR_COMPUTE_DESC_C32F)
                         || (computeType == HIPTENSOR_COMPUTE_DESC_C64F);
        return isComplex ? 8.0 : 2.0;
    }

    // GFLOP of D = alpha * A * B + beta * C, whatever the order of the modes in each tensor.
    inline double contractionGFlops(std::vector<int32_t> const&     modesA,
                                    std::vector<std::size_t> const& lengthsA,
                                    std::vector<int32_t> const&     modesB,
                                    std::vector<std::size_t> const& lengthsB,
                                    hiptensorComputeDescriptor_t    computeType)
    {
        return flopsPerPoint(computeType)
               * contractionIterationPoints(modesA, lengthsA, modesB, lengthsB) * 1e-9;
    }

    // GFLOP of E = alpha * A * B * C + beta * D, run as T = A * B and E = alpha * T * C + beta * D.
    // T carries every mode of A and B except those contracted in the first step: modes carried
    // by both A and B but by neither C nor E.
    inline double trinaryContractionGFlops(std::vector<int32_t> const&     modesA,
                                           std::vector<std::size_t> const& lengthsA,
                                           std::vector<int32_t> const&     modesB,
                                           std::vector<std::size_t> const& lengthsB,
                                           std::vector<int32_t> const&     modesC,
                                           std::vector<std::size_t> const& lengthsC,
                                           std::vector<int32_t> const&     modesE,
                                           hiptensorComputeDescriptor_t    computeType)
    {
        auto carries = [](std::vector<int32_t> const& modes, int32_t mode) {
            return std::find(modes.cbegin(), modes.cend(), mode) != modes.cend();
        };

        // Modes of A, then of B.
        std::vector<int32_t> modesAB = modesA;
        modesAB.insert(modesAB.end(), modesB.cbegin(), modesB.cend());
        std::vector<std::size_t> lengthsAB = lengthsA;
        lengthsAB.insert(lengthsAB.end(), lengthsB.cbegin(), lengthsB.cend());

        std::vector<int32_t>     modesT;
        std::vector<std::size_t> lengthsT;
        for(std::size_t i = 0; i < modesAB.size(); i++)
        {
            auto mode       = modesAB[i];
            bool contracted = carries(modesA, mode) && carries(modesB, mode)
                              && !carries(modesC, mode) && !carries(modesE, mode);
            if(!contracted && !carries(modesT, mode))
            {
                modesT.push_back(mode);
                lengthsT.push_back(lengthsAB[i]);
            }
        }

        return flopsPerPoint(computeType)
               * (contractionIterationPoints(modesA, lengthsA, modesB, lengthsB)
                  + contractionIterationPoints(modesT, lengthsT, modesC, lengthsC))
               * 1e-9;
    }
} // namespace hiptensor
