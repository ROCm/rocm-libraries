/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_scalar.hpp"

namespace rocsparse
{
    template <uint32_t WF_SIZE,
              uint32_t LOOPS,
              uint32_t COLS,
              bool     NT,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(WF_SIZE) __global__
        void coommnn_segmented_atomic(rocsparse_operation trans_B,
                                      int64_t             nnz,
                                      I                   n,
                                      int64_t             batch_stride_A,
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                      const I* __restrict__ coo_row_ind,
                                      const I* __restrict__ coo_col_ind,
                                      const A* __restrict__ coo_val,
                                      const B* __restrict__ dense_B,
                                      int64_t ldb,
                                      int64_t batch_stride_B,
                                      C* __restrict__ dense_C,
                                      int64_t              ldc,
                                      int64_t              batch_stride_C,
                                      rocsparse_order      order_C,
                                      rocsparse_index_base idx_base,
                                      bool                 is_host_mode);
}
