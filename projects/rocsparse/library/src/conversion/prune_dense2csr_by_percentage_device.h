/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once

#include "common.h"
#include "handle.h"

namespace rocsparse
{
    template <rocsparse_int BLOCK_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void abs_kernel(rocsparse_int m, rocsparse_int n, const T* A, int64_t lda, T* output)
    {
        rocsparse_int thread_id = hipThreadIdx_x + hipBlockIdx_x * BLOCK_SIZE;

        if(thread_id >= m * n)
        {
            return;
        }

        rocsparse_int row = thread_id % m;
        rocsparse_int col = thread_id / m;

        output[m * col + row] = rocsparse::abs(A[lda * col + row]);
    }
}
