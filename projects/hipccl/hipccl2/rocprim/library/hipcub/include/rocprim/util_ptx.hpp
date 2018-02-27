// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_ROCPRIM_UTIL_PTX_HPP_
#define HIPCUB_ROCPRIM_UTIL_PTX_HPP_

#include "../config.hpp"

#define HIPCUB_WARP_THREADS ::rocprim::warp_size()
#define HIPCUB_ARCH 1 // ignored with rocPRIM backend

BEGIN_HIPCUB_NAMESPACE

HIPCUB_DEVICE inline
int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (hipThreadIdx_z * block_dim_x * block_dim_y))
        + ((block_dim_y == 1) ? 0 : (hipThreadIdx_y * block_dim_x))
        + hipThreadIdx_x;
}

HIPCUB_DEVICE inline
unsigned int LaneId()
{
    return ::rocprim::lane_id();
}

HIPCUB_DEVICE inline
unsigned int WarpId()
{
    return ::rocprim::warp_id();
}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_PTX_HPP_
