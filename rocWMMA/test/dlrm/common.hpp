/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DLRM_TEST_COMMON_HPP
#define DLRM_TEST_COMMON_HPP

#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <rocwmma/internal/constants.hpp>
#include <rocwmma/internal/types.hpp>
#include <rocwmma/internal/utils.hpp>
#include <rocwmma/rocwmma.hpp>

#include "device/common.hpp"

namespace rocwmma
{

    struct validate_data_t
    {
        bool  pass;
        float maxRelativeDiff;
        float tolerance = 1e-5;
    };

    template <typename DataT>
    __host__ static inline void fill(
        DataT* mat, uint32_t numRows, uint32_t numCols, uint32_t batchSize, uint32_t normalization)
    {
        auto batchOffset = numRows * numCols;
        for(int k = 0; k < batchSize; ++k)
        {
            for(int i = 0; i < numRows; ++i)
            {
                for(int j = 0; j < numCols; ++j)
                {
                    // Random values normalized such that output is between 0 and 1
                    auto value = __float2half(static_cast<float>(rand() / normalization)
                                              / static_cast<float>(RAND_MAX));
                    mat[k * batchOffset + i * numCols + j] = static_cast<DataT>(value);
                }
            }
        }
    }

    template <typename DataT>
    __host__ static inline void
        fillInit(DataT* mat, uint32_t numRows, uint32_t numCols, uint32_t batchSize, DataT value)
    {
        auto batchOffset = numRows * numCols;
        for(int k = 0; k < batchSize; ++k)
        {
            for(int i = 0; i < numRows; ++i)
            {
                for(int j = 0; j < numCols; ++j)
                {
                    mat[k * batchOffset + i * numCols + j] = static_cast<DataT>(value);
                }
            }
        }
    }

    template <typename DataT>
    validate_data_t compareEqual(DataT const* matrixA,
                                 DataT const* matrixB,
                                 uint32_t     batchSize,
                                 uint32_t     b,
                                 double       tolerance = 10.0)
    {
        validate_data_t return_data;
        bool            retval             = true;
        double          max_relative_error = 0.0;

        // Some types don't have direct conversion to double.
        // Convert to float first then to double.
        auto toDoubleA
            = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };
        auto toDoubleB
            = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

        bool       isInf = false;
        bool       isNaN = false;
        std::mutex writeMutex;

#pragma omp parallel for
        for(int i = 0; i < batchSize * b; ++i) // Row
        {
            auto valA = matrixA[i];
            auto valB = matrixB[i];

            auto numerator = fabs(toDoubleA(valA) - toDoubleB(valB));
            auto divisor   = fabs(toDoubleA(valA)) + fabs(toDoubleB(valB)) + 1.0;

            if(std::isinf(numerator) || std::isinf(divisor))
            {
#pragma omp atomic
                isInf |= true;
            }
            else
            {
                auto relative_error = numerator / divisor;
                if(std::isnan(relative_error))
                {
#pragma omp atomic
                    isNaN |= true;
                }
                else if(relative_error > max_relative_error)
                {
                    const std::lock_guard<std::mutex> guard(writeMutex);
                    // Double check in case of stall
                    if(relative_error > max_relative_error)
                    {
                        max_relative_error = relative_error;
                    }
                }
            }

            if(isInf || isNaN)
            {
                i = batchSize;
            }
        }

        auto eps = toDoubleA(std::numeric_limits<DataT>::epsilon());
        if(isInf)
        {
            retval             = false;
            max_relative_error = std::numeric_limits<DataT>::infinity();
        }
        else if(isNaN)
        {
            retval             = false;
            max_relative_error = std::numeric_limits<DataT>::signaling_NaN();
        }
        else if(max_relative_error > (eps * tolerance))
        {
            retval = false;
        }

        return_data.pass            = retval;
        return_data.maxRelativeDiff = max_relative_error;
        return_data.tolerance       = eps * tolerance;

        return return_data;
    }

} // namespace rocwmma

#endif // DLRM_TEST_COMMON_HPP
