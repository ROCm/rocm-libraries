/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <typename T>
    struct coordinate_t;

    template <uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(WF_SIZE) __global__
        void csrmmnt_merge_path_main_kernel(bool conj_A,
                                            bool conj_B,
                                            J    ncol_offset,
                                            J    ncol,
                                            J    m,
                                            J    n,
                                            J    k,
                                            I    nnz,
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                            const I* __restrict__ csr_row_ptr,
                                            const J* __restrict__ csr_col_ind,
                                            const A* __restrict__ csr_val,
                                            const coordinate_t<uint32_t>* __restrict__ coord0,
                                            const coordinate_t<uint32_t>* __restrict__ coord1,
                                            const B* __restrict__ dense_B,
                                            int64_t ldb,
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                            C* __restrict__ dense_C,
                                            int64_t              ldc,
                                            rocsparse_order      order_C,
                                            rocsparse_index_base idx_base,
                                            bool                 is_host_mode);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnt_merge_path_remainder_kernel(bool conj_A,
                                                 bool conj_B,
                                                 J    ncol_offset,
                                                 J    m,
                                                 J    n,
                                                 J    k,
                                                 I    nnz,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                 const I* __restrict__ csr_row_ptr,
                                                 const J* __restrict__ csr_col_ind,
                                                 const A* __restrict__ csr_val,
                                                 const coordinate_t<uint32_t>* __restrict__ coord0,
                                                 const coordinate_t<uint32_t>* __restrict__ coord1,
                                                 const B* __restrict__ dense_B,
                                                 int64_t ldb,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                 C* __restrict__ dense_C,
                                                 int64_t              ldc,
                                                 rocsparse_order      order_C,
                                                 rocsparse_index_base idx_base,
                                                 bool                 is_host_mode);
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnn_merge_path_kernel(bool conj_A,
                                       bool conj_B,
                                       J    m,
                                       J    n,
                                       J    k,
                                       I    nnz,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                       const I* __restrict__ csr_row_ptr,
                                       const J* __restrict__ csr_col_ind,
                                       const A* __restrict__ csr_val,
                                       const coordinate_t<uint32_t>* __restrict__ coord0,
                                       const coordinate_t<uint32_t>* __restrict__ coord1,
                                       const B* __restrict__ dense_B,
                                       int64_t ldb,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                       C* __restrict__ dense_C,
                                       int64_t              ldc,
                                       rocsparse_order      order_C,
                                       rocsparse_index_base idx_base,
                                       bool                 is_host_mode);
}
