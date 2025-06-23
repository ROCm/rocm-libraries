/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    namespace primitives
    {
        template <class T>
        class double_buffer
        {
        private:
            T* buffers[2];

            unsigned int selector;

        public:
            __device__ __host__ inline double_buffer()
            {
                selector   = 0;
                buffers[0] = nullptr;
                buffers[1] = nullptr;
            }
            __device__ __host__ inline double_buffer(T* current, T* alternate)
            {
                selector   = 0;
                buffers[0] = current;
                buffers[1] = alternate;
            }

            __device__ __host__ inline T* current() const
            {
                return buffers[selector];
            }
            __device__ __host__ inline T* alternate() const
            {
                return buffers[selector ^ 1];
            }
            __device__ __host__ inline void swap()
            {
                selector ^= 1;
            }
        };

        template <typename K, typename V>
        rocsparse_status radix_sort_pairs_buffer_size(rocsparse_handle handle,
                                                      size_t           length,
                                                      uint32_t         startbit,
                                                      uint32_t         endbit,
                                                      size_t*          buffer_size,
                                                      bool             using_double_buffers = true);

        template <typename K, typename V>
        rocsparse_status radix_sort_pairs(rocsparse_handle  handle,
                                          double_buffer<K>& keys,
                                          double_buffer<V>& values,
                                          size_t            length,
                                          uint32_t          startbit,
                                          uint32_t          endbit,
                                          size_t            buffer_size,
                                          void*             buffer);

        template <typename K, typename V>
        rocsparse_status radix_sort_pairs(rocsparse_handle handle,
                                          K*               keys_input,
                                          K*               keys_output,
                                          V*               values_input,
                                          V*               values_output,
                                          size_t           length,
                                          uint32_t         startbit,
                                          uint32_t         endbit,
                                          size_t           buffer_size,
                                          void*            buffer);

        template <typename K>
        rocsparse_status radix_sort_keys_buffer_size(rocsparse_handle handle,
                                                     size_t           length,
                                                     uint32_t         startbit,
                                                     uint32_t         endbit,
                                                     size_t*          buffer_size);

        template <typename K>
        rocsparse_status radix_sort_keys(rocsparse_handle  handle,
                                         double_buffer<K>& keys,
                                         size_t            length,
                                         uint32_t          startbit,
                                         uint32_t          endbit,
                                         size_t            buffer_size,
                                         void*             buffer);

        template <typename J>
        rocsparse_status run_length_encode_buffer_size(rocsparse_handle handle,
                                                       size_t           length,
                                                       size_t*          buffer_size);

        template <typename J>
        rocsparse_status run_length_encode(rocsparse_handle handle,
                                           J*               input,
                                           J*               unique_output,
                                           J*               counts_output,
                                           J*               runs_count_output,
                                           size_t           length,
                                           size_t           buffer_size,
                                           void*            buffer);

        template <typename I, typename J>
        rocsparse_status exclusive_scan_buffer_size(rocsparse_handle handle,
                                                    J                initial_value,
                                                    size_t           length,
                                                    size_t*          buffer_size);

        template <typename I, typename J>
        rocsparse_status exclusive_scan(rocsparse_handle handle,
                                        I*               input,
                                        J*               output,
                                        J                initial_value,
                                        size_t           length,
                                        size_t           buffer_size,
                                        void*            buffer);

        template <typename I, typename J>
        rocsparse_status
            inclusive_scan_buffer_size(rocsparse_handle handle, size_t length, size_t* buffer_size);

        template <typename I, typename J>
        rocsparse_status inclusive_scan(rocsparse_handle handle,
                                        I*               input,
                                        J*               output,
                                        size_t           length,
                                        size_t           buffer_size,
                                        void*            buffer);

        template <typename I, typename J>
        rocsparse_status
            find_max_buffer_size(rocsparse_handle handle, size_t length, size_t* buffer_size);

        template <typename I, typename J>
        rocsparse_status find_max(rocsparse_handle handle,
                                  I*               input,
                                  J*               max,
                                  size_t           length,
                                  size_t           buffer_size,
                                  void*            buffer);

        template <typename I, typename J>
        rocsparse_status
            find_sum_buffer_size(rocsparse_handle handle, size_t length, size_t* buffer_size);

        template <typename I, typename J>
        rocsparse_status find_sum(rocsparse_handle handle,
                                  I*               input,
                                  J*               sum,
                                  size_t           length,
                                  size_t           buffer_size,
                                  void*            buffer);

        template <typename K, typename V, typename I>
        rocsparse_status segmented_radix_sort_pairs_buffer_size(rocsparse_handle handle,
                                                                size_t           length,
                                                                size_t           segments,
                                                                uint32_t         startbit,
                                                                uint32_t         endbit,
                                                                size_t*          buffer_size);

        template <typename K, typename V, typename I>
        rocsparse_status segmented_radix_sort_pairs(rocsparse_handle  handle,
                                                    double_buffer<K>& keys,
                                                    double_buffer<V>& values,
                                                    size_t            length,
                                                    size_t            segments,
                                                    I*                begin_offsets,
                                                    I*                end_offsets,
                                                    uint32_t          startbit,
                                                    uint32_t          endbit,
                                                    size_t            buffer_size,
                                                    void*             buffer);

        template <typename K, typename I>
        rocsparse_status segmented_radix_sort_keys_buffer_size(rocsparse_handle handle,
                                                               size_t           length,
                                                               size_t           segments,
                                                               uint32_t         startbit,
                                                               uint32_t         endbit,
                                                               size_t*          buffer_size);

        template <typename K, typename I>
        rocsparse_status segmented_radix_sort_keys(rocsparse_handle  handle,
                                                   double_buffer<K>& keys,
                                                   size_t            length,
                                                   size_t            segments,
                                                   I*                begin_offsets,
                                                   I*                end_offsets,
                                                   uint32_t          startbit,
                                                   uint32_t          endbit,
                                                   size_t            buffer_size,
                                                   void*             buffer);

        template <typename I, typename J>
        rocsparse_status sort_csr_column_indices_buffer_size(
            rocsparse_handle handle, J m, J n, I nnz, const I* csr_row_ptr, size_t* buffer_size);

        template <typename I, typename J>
        rocsparse_status sort_csr_column_indices(rocsparse_handle handle,
                                                 J                m,
                                                 J                n,
                                                 I                nnz,
                                                 const I*         csr_row_ptr,
                                                 const J*         csr_col_ind,
                                                 J*               csr_col_ind_buffer1,
                                                 J*               csr_col_ind_buffer2,
                                                 void*            buffer);

        template <typename I, typename J>
        rocsparse_status sort_csr_column_indices(rocsparse_handle handle,
                                                 J                m,
                                                 J                n,
                                                 I                nnz,
                                                 const I*         csr_row_ptr,
                                                 J*               csr_col_ind,
                                                 J*               csr_col_ind_buffer1,
                                                 void*            buffer);
    }
}
