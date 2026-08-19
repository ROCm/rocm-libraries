/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_KERNELS_DENSE_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_DENSE_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    // Replace column vector
    template <typename ValueType, typename IndexType>
    __global__ void kernel_dense_replace_column_vector(const ValueType* __restrict__ vec,
                                                       IndexType idx,
                                                       IndexType nrow,
                                                       IndexType ncol,
                                                       ValueType* __restrict__ mat)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        mat[DENSE_IND(ai, idx, nrow, ncol)] = vec[ai];
    }

    // Replace row vector
    template <typename ValueType, typename IndexType>
    __global__ void kernel_dense_replace_row_vector(const ValueType* __restrict__ vec,
                                                    IndexType idx,
                                                    IndexType nrow,
                                                    IndexType ncol,
                                                    ValueType* __restrict__ mat)
    {
        IndexType aj = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(aj >= ncol)
        {
            return;
        }

        mat[DENSE_IND(idx, aj, nrow, ncol)] = vec[aj];
    }

    // Extract column vector
    template <typename ValueType, typename IndexType>
    __global__ void kernel_dense_extract_column_vector(ValueType* __restrict__ vec,
                                                       IndexType idx,
                                                       IndexType nrow,
                                                       IndexType ncol,
                                                       const ValueType* __restrict__ mat)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        vec[ai] = mat[DENSE_IND(ai, idx, nrow, ncol)];
    }

    // Extract row vector
    template <typename ValueType, typename IndexType>
    __global__ void kernel_dense_extract_row_vector(ValueType* __restrict__ vec,
                                                    IndexType idx,
                                                    IndexType nrow,
                                                    IndexType ncol,
                                                    const ValueType* __restrict__ mat)
    {
        IndexType aj = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(aj >= ncol)
        {
            return;
        }

        vec[aj] = mat[DENSE_IND(idx, aj, nrow, ncol)];
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_DENSE_HPP_
