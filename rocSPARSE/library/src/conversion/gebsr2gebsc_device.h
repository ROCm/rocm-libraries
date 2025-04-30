/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsr2gebsc_permute_kernel(rocsparse_int        nnzb,
                                    rocsparse_int        linsize_block,
                                    const rocsparse_int* in1,
                                    const T*             in2,
                                    const rocsparse_int* map,
                                    rocsparse_int*       out1,
                                    T*                   out2)
    {
        rocsparse_int gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= nnzb)
        {
            return;
        }

        rocsparse_int idx = map[gid];

        out1[gid] = in1[idx];

        for(rocsparse_int i = 0; i < linsize_block; ++i)
        {
            out2[gid * linsize_block + i] = in2[idx * linsize_block + i];
        }
    }
}
