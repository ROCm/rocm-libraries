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

#ifndef ROCALUTION_HIP_HIP_KERNELS_COO_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_COO_HPP_

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType, typename IndexType>
    __global__ void kernel_coo_permute(IndexType nnz,
                                       const IndexType* __restrict__ in_row,
                                       const IndexType* __restrict__ in_col,
                                       const IndexType* __restrict__ perm,
                                       IndexType* __restrict__ out_row,
                                       IndexType* __restrict__ out_col)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        for(int i = ind; i < nnz; i += hipGridDim_x)
        {
            out_row[i] = perm[in_row[i]];
            out_col[i] = perm[in_col[i]];
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_COO_HPP_
