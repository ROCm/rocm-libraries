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
#include "../../csrmm_device_merge.h"
#include "../csrmm_common.h"
#include "rocsparse_scalar.hpp"

namespace rocsparse
{

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
        void csrmmnt_merge_path_main_kernel(bool    conj_A,
                                            bool    conj_B,
                                            J       ncol_offset,
                                            J       ncol,
                                            J       m,
                                            J       n,
                                            J       k,
                                            I       nnz,
                                            int64_t batch_count,
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                            int64_t offsets_batch_stride_A,
                                            int64_t columns_values_batch_stride_A,
                                            const I* __restrict__ csr_row_ptr,
                                            const J* __restrict__ csr_col_ind,
                                            const A* __restrict__ csr_val,
                                            const coordinate_t<uint32_t>* __restrict__ coord0,
                                            const coordinate_t<uint32_t>* __restrict__ coord1,
                                            const B* __restrict__ dense_B,
                                            int64_t ldb,
                                            int64_t batch_stride_B,
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                            C* __restrict__ dense_C,
                                            int64_t              ldc,
                                            int64_t              batch_stride_C,
                                            rocsparse_order      order_C,
                                            rocsparse_index_base idx_base,
                                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            // Grid-stride loop over the batch dimension (grid y). Per-batch pointers
            // are computed with load_pointer so the device kernels stay batch-agnostic.
            // The merge-path coordinates are per batch when the batches use distinct
            // row pointers (offsets_batch_stride_A != 0) and shared (stride 0)
            // otherwise; coords_per_batch mirrors csrmm_analysis_template_merge.
            const uint64_t coord_total_work  = static_cast<uint64_t>(m) + nnz;
            const uint64_t coord_block_count = (coord_total_work - 1) / 256 + 1;
            const int64_t  coord_batch_stride
                = (offsets_batch_stride_A != 0)
                      ? static_cast<int64_t>(((coord_block_count - 1) / 256 + 1) * 256)
                      : 0;
            for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
            {
                rocsparse::csrmmnt_merge_path_main_device<WF_SIZE, ITEMS_PER_THREAD, LOOPS>(
                    conj_A,
                    conj_B,
                    ncol_offset,
                    ncol,
                    m,
                    n,
                    k,
                    nnz,
                    alpha,
                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                    load_pointer(coord0, batch, coord_batch_stride),
                    load_pointer(coord1, batch, coord_batch_stride),
                    load_pointer(dense_B, batch, batch_stride_B),
                    ldb,
                    beta,
                    load_pointer(dense_C, batch, batch_stride_C),
                    ldc,
                    order_C,
                    idx_base);
            }
        }
    }

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
        void csrmmnt_merge_path_remainder_kernel(bool    conj_A,
                                                 bool    conj_B,
                                                 J       ncol_offset,
                                                 J       m,
                                                 J       n,
                                                 J       k,
                                                 I       nnz,
                                                 int64_t batch_count,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                 int64_t offsets_batch_stride_A,
                                                 int64_t columns_values_batch_stride_A,
                                                 const I* __restrict__ csr_row_ptr,
                                                 const J* __restrict__ csr_col_ind,
                                                 const A* __restrict__ csr_val,
                                                 const coordinate_t<uint32_t>* __restrict__ coord0,
                                                 const coordinate_t<uint32_t>* __restrict__ coord1,
                                                 const B* __restrict__ dense_B,
                                                 int64_t ldb,
                                                 int64_t batch_stride_B,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                 C* __restrict__ dense_C,
                                                 int64_t              ldc,
                                                 int64_t              batch_stride_C,
                                                 rocsparse_order      order_C,
                                                 rocsparse_index_base idx_base,
                                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            // Grid-stride loop over the batch dimension (grid y). See main kernel.
            const uint64_t coord_total_work  = static_cast<uint64_t>(m) + nnz;
            const uint64_t coord_block_count = (coord_total_work - 1) / 256 + 1;
            const int64_t  coord_batch_stride
                = (offsets_batch_stride_A != 0)
                      ? static_cast<int64_t>(((coord_block_count - 1) / 256 + 1) * 256)
                      : 0;
            for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
            {
                rocsparse::
                    csrmmnt_merge_path_remainder_device<BLOCKSIZE, WF_SIZE, ITEMS_PER_THREAD>(
                        conj_A,
                        conj_B,
                        ncol_offset,
                        m,
                        n,
                        k,
                        nnz,
                        alpha,
                        load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                        load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                        load_pointer(csr_val, batch, columns_values_batch_stride_A),
                        load_pointer(coord0, batch, coord_batch_stride),
                        load_pointer(coord1, batch, coord_batch_stride),
                        load_pointer(dense_B, batch, batch_stride_B),
                        ldb,
                        beta,
                        load_pointer(dense_C, batch, batch_stride_C),
                        ldc,
                        order_C,
                        idx_base);
            }
        }
    }

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
        void csrmmnn_merge_path_kernel(bool    conj_A,
                                       bool    conj_B,
                                       J       m,
                                       J       n,
                                       J       k,
                                       I       nnz,
                                       int64_t batch_count,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                       int64_t offsets_batch_stride_A,
                                       int64_t columns_values_batch_stride_A,
                                       const I* __restrict__ csr_row_ptr,
                                       const J* __restrict__ csr_col_ind,
                                       const A* __restrict__ csr_val,
                                       const coordinate_t<uint32_t>* __restrict__ coord0,
                                       const coordinate_t<uint32_t>* __restrict__ coord1,
                                       const B* __restrict__ dense_B,
                                       int64_t ldb,
                                       int64_t batch_stride_B,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                       C* __restrict__ dense_C,
                                       int64_t              ldc,
                                       int64_t              batch_stride_C,
                                       rocsparse_order      order_C,
                                       rocsparse_index_base idx_base,
                                       bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            // Grid-stride loop over the batch dimension (grid y). See main kernel.
            const uint64_t coord_total_work  = static_cast<uint64_t>(m) + nnz;
            const uint64_t coord_block_count = (coord_total_work - 1) / 256 + 1;
            const int64_t  coord_batch_stride
                = (offsets_batch_stride_A != 0)
                      ? static_cast<int64_t>(((coord_block_count - 1) / 256 + 1) * 256)
                      : 0;

            for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
            {
                rocsparse::csrmmnn_merge_path_device<BLOCKSIZE, WF_SIZE, ITEMS_PER_THREAD>(
                    conj_A,
                    conj_B,
                    m,
                    n,
                    k,
                    nnz,
                    alpha,
                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                    load_pointer(coord0, batch, coord_batch_stride),
                    load_pointer(coord1, batch, coord_batch_stride),
                    load_pointer(dense_B, batch, batch_stride_B),
                    ldb,
                    beta,
                    load_pointer(dense_C, batch, batch_stride_C),
                    ldc,
                    order_C,
                    idx_base);
            }
        }
    }
}

#define CSRMMNN_MERGE_PATH_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, ITEM_PER_THREAD) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                               \
        rocsparse::csrmmnn_merge_path_kernel<BLOCKSIZE, WFSIZE, ITEM_PER_THREAD>(       \
            bool    conj_A,                                                             \
            bool    conj_B,                                                             \
            J       m,                                                                  \
            J       n,                                                                  \
            J       k,                                                                  \
            I       nnz,                                                                \
            int64_t batch_count,                                                        \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                              \
            int64_t offsets_batch_stride_A,                                             \
            int64_t columns_values_batch_stride_A,                                      \
            const I* __restrict__ csr_row_ptr,                                          \
            const J* __restrict__ csr_col_ind,                                          \
            const A* __restrict__ csr_val,                                              \
            const coordinate_t<uint32_t>* __restrict__ coord0,                          \
            const coordinate_t<uint32_t>* __restrict__ coord1,                          \
            const B* __restrict__ dense_B,                                              \
            int64_t ldb,                                                                \
            int64_t batch_stride_B,                                                     \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                               \
            C* __restrict__ dense_C,                                                    \
            int64_t              ldc,                                                   \
            int64_t              batch_stride_C,                                        \
            rocsparse_order      order_C,                                               \
            rocsparse_index_base idx_base,                                              \
            bool                 is_host_mode);

#define CSRMMNT_MERGE_PATH_MAIN_KERNEL(T, I, J, A, B, C, WFSIZE, ITEM_PER_THREAD, LOOPS) \
    template __launch_bounds__(WFSIZE) __global__ void                                   \
        rocsparse::csrmmnt_merge_path_main_kernel<WFSIZE, ITEM_PER_THREAD, LOOPS>(       \
            bool    conj_A,                                                              \
            bool    conj_B,                                                              \
            J       ncol_offset,                                                         \
            J       ncol,                                                                \
            J       m,                                                                   \
            J       n,                                                                   \
            J       k,                                                                   \
            I       nnz,                                                                 \
            int64_t batch_count,                                                         \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                               \
            int64_t offsets_batch_stride_A,                                              \
            int64_t columns_values_batch_stride_A,                                       \
            const I* __restrict__ csr_row_ptr,                                           \
            const J* __restrict__ csr_col_ind,                                           \
            const A* __restrict__ csr_val,                                               \
            const coordinate_t<uint32_t>* __restrict__ coord0,                           \
            const coordinate_t<uint32_t>* __restrict__ coord1,                           \
            const B* __restrict__ dense_B,                                               \
            int64_t ldb,                                                                 \
            int64_t batch_stride_B,                                                      \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                \
            C* __restrict__ dense_C,                                                     \
            int64_t              ldc,                                                    \
            int64_t              batch_stride_C,                                         \
            rocsparse_order      order_C,                                                \
            rocsparse_index_base idx_base,                                               \
            bool                 is_host_mode);

#define CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, ITEM_PER_THREAD) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                                         \
        rocsparse::csrmmnt_merge_path_remainder_kernel<BLOCKSIZE, WFSIZE, ITEM_PER_THREAD>(       \
            bool    conj_A,                                                                       \
            bool    conj_B,                                                                       \
            J       ncol_offset,                                                                  \
            J       m,                                                                            \
            J       n,                                                                            \
            J       k,                                                                            \
            I       nnz,                                                                          \
            int64_t batch_count,                                                                  \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                        \
            int64_t offsets_batch_stride_A,                                                       \
            int64_t columns_values_batch_stride_A,                                                \
            const I* __restrict__ csr_row_ptr,                                                    \
            const J* __restrict__ csr_col_ind,                                                    \
            const A* __restrict__ csr_val,                                                        \
            const coordinate_t<uint32_t>* __restrict__ coord0,                                    \
            const coordinate_t<uint32_t>* __restrict__ coord1,                                    \
            const B* __restrict__ dense_B,                                                        \
            int64_t ldb,                                                                          \
            int64_t batch_stride_B,                                                               \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                         \
            C* __restrict__ dense_C,                                                              \
            int64_t              ldc,                                                             \
            int64_t              batch_stride_C,                                                  \
            rocsparse_order      order_C,                                                         \
            rocsparse_index_base idx_base,                                                        \
            bool                 is_host_mode);

#define CSRMMNN_MERGE_PATH_256_16_256(T, I, J, A, B, C) \
    CSRMMNN_MERGE_PATH_KERNEL(T, I, J, A, B, C, 256, 16, 256)
#define CSRMMNN_MERGE_PATH_256_32_256(T, I, J, A, B, C) \
    CSRMMNN_MERGE_PATH_KERNEL(T, I, J, A, B, C, 256, 32, 256)
#define CSRMMNN_MERGE_PATH_256_64_256(T, I, J, A, B, C) \
    CSRMMNN_MERGE_PATH_KERNEL(T, I, J, A, B, C, 256, 64, 256)
#define CSRMMNN_MERGE_PATH_256_128_256(T, I, J, A, B, C) \
    CSRMMNN_MERGE_PATH_KERNEL(T, I, J, A, B, C, 256, 128, 256)
#define CSRMMNN_MERGE_PATH_256_256_256(T, I, J, A, B, C) \
    CSRMMNN_MERGE_PATH_KERNEL(T, I, J, A, B, C, 256, 256, 256)

#define CSRMMNT_MERGE_PATH_MAIN_64_256_1(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_MAIN_KERNEL(T, I, J, A, B, C, 64, 256, 1)
#define CSRMMNT_MERGE_PATH_MAIN_128_256_1(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_MAIN_KERNEL(T, I, J, A, B, C, 128, 256, 1)
#define CSRMMNT_MERGE_PATH_MAIN_256_256_1(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_MAIN_KERNEL(T, I, J, A, B, C, 256, 256, 1)

#define CSRMMNT_MERGE_PATH_REMAINDER_256_8_256(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 8, 256)
#define CSRMMNT_MERGE_PATH_REMAINDER_256_16_256(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 16, 256)
#define CSRMMNT_MERGE_PATH_REMAINDER_256_32_256(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 32, 256)
#define CSRMMNT_MERGE_PATH_REMAINDER_256_64_256(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 64, 256)
#define CSRMMNT_MERGE_PATH_REMAINDER_256_128_256(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 128, 256)
#define CSRMMNT_MERGE_PATH_REMAINDER_256_256_256(T, I, J, A, B, C) \
    CSRMMNT_MERGE_PATH_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 256, 256)
