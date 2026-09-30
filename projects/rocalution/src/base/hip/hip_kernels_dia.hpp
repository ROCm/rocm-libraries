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

#ifndef ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    // Nathan Bell and Michael Garland
    // Efficient Sparse Matrix-Vector Multiplication
    // NVR-2008-004 / NVIDIA Technical Report
    template <typename ValueType, typename IndexType>
    __global__ void kernel_dia_spmv(IndexType num_rows,
                                    IndexType num_cols,
                                    IndexType num_diags,
                                    const IndexType* __restrict__ Aoffsets,
                                    const ValueType* __restrict__ Aval,
                                    const ValueType* __restrict__ x,
                                    ValueType* __restrict__ y)
    {
        int row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

        if(row >= num_rows)
        {
            return;
        }

        ValueType sum = static_cast<ValueType>(0);

        for(IndexType n = 0; n < num_diags; ++n)
        {
            IndexType ind = DIA_IND(row, n, num_rows, num_diags);
            IndexType col = row + Aoffsets[n];

            if((col >= 0) && (col < num_cols))
            {
                sum = sum + Aval[ind] * x[col];
            }
        }

        y[row] = sum;
    }

    // Nathan Bell and Michael Garland
    // Efficient Sparse Matrix-Vector Multiplication
    // NVR-2008-004 / NVIDIA Technical Report
    template <typename ValueType, typename IndexType>
    __global__ void kernel_dia_add_spmv(IndexType num_rows,
                                        IndexType num_cols,
                                        IndexType num_diags,
                                        const IndexType* __restrict__ Aoffsets,
                                        const ValueType* __restrict__ Aval,
                                        ValueType scalar,
                                        const ValueType* __restrict__ x,
                                        ValueType* __restrict__ y)
    {
        int row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

        if(row >= num_rows)
        {
            return;
        }

        ValueType sum = static_cast<ValueType>(0);

        for(IndexType n = 0; n < num_diags; ++n)
        {
            IndexType ind = DIA_IND(row, n, num_rows, num_diags);
            IndexType col = row + Aoffsets[n];

            if((col >= 0) && (col < num_cols))
            {
                sum = sum + Aval[ind] * x[col];
            }
        }

        y[row] = y[row] + scalar * sum;
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_
