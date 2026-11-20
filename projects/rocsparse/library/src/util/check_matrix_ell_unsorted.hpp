/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "check_matrix_ell_device.h"
#include "check_matrix_workspace.hpp"

namespace rocsparse
{
    // Duplicate pass of the unsorted ELL check. It expects the status slot at the start of
    // temp_buffer to hold success, and temp_buffer to have the size that
    // check_matrix_ell_workspace_query returns for the same batch_limit.
    template <typename I>
    rocsparse_status check_matrix_ell_unsorted_duplicates(rocsparse_handle       handle,
                                                          I                      m,
                                                          I                      n,
                                                          I                      ell_width,
                                                          const I*               ell_col_ind,
                                                          rocsparse_index_base   idx_base,
                                                          rocsparse_data_status* data_status,
                                                          void*                  temp_buffer,
                                                          int64_t                batch_limit
                                                          = check_matrix_ell_sort_batch_limit)
    {
        check_matrix_ell_workspace ws;
        RETURN_IF_ROCSPARSE_ERROR(check_matrix_ell_workspace_query<I>(
            handle, m, n, ell_width, rocsparse_storage_mode_unsorted, ws, batch_limit));

        const int64_t batch_rows = check_matrix_ell_rows_per_batch(m, ell_width, batch_limit);

        char* ptr            = reinterpret_cast<char*>(temp_buffer) + ws.status_bytes;
        void* rocprim_buffer = ptr;
        ptr += ws.rocprim_slot;
        int64_t* offsets = reinterpret_cast<int64_t*>(ptr);
        ptr += ws.offset_bytes;
        I* gathered = reinterpret_cast<I*>(ptr);
        ptr += ws.array_bytes;
        I* gathered_alt = reinterpret_cast<I*>(ptr);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_ell_gather_device<256, I>),
                                           dim3((m - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           m,
                                           n,
                                           ell_width,
                                           ell_col_ind,
                                           idx_base,
                                           gathered);

        // The segment offsets are relative to the first row of a batch, so all batches
        // share them.
        const int64_t  offset_threads = batch_rows + 1;
        const uint32_t offset_blocks  = static_cast<uint32_t>((offset_threads - 1) / 256 + 1);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_ell_offsets_device<256>),
                                           dim3(offset_blocks),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           batch_rows,
                                           static_cast<int64_t>(ell_width),
                                           offsets);

        // A batch can end up sorted in either buffer. sorted is the buffer that holds the
        // first batch, and later batches are copied there if they end up in the other one.
        I* sorted = nullptr;
        for(int64_t first_row = 0; first_row < m; first_row += batch_rows)
        {
            const int64_t rows  = std::min(batch_rows, static_cast<int64_t>(m) - first_row);
            const int64_t begin = first_row * ell_width;
            const int64_t count = rows * ell_width;

            rocsparse::primitives::double_buffer<I> keys(gathered + begin, gathered_alt + begin);
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys<I, int64_t>(
                handle,
                keys,
                static_cast<size_t>(count),
                static_cast<size_t>(rows),
                offsets,
                offsets + 1,
                0,
                ws.end_bit,
                ws.rocprim_bytes,
                rocprim_buffer)));

            if(sorted == nullptr)
            {
                sorted = keys.current() - begin;
            }
            else if(keys.current() != sorted + begin)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(sorted + begin,
                                                             keys.current(),
                                                             sizeof(I) * static_cast<size_t>(count),
                                                             hipMemcpyDeviceToDevice,
                                                             handle->stream));
            }
        }

        rocsparse_data_status* d_data_status
            = reinterpret_cast<rocsparse_data_status*>(temp_buffer);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_ell_duplicate_device<256, I>),
                                           dim3((m - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           m,
                                           n,
                                           ell_width,
                                           sorted,
                                           d_data_status);

        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(data_status,
                                                     d_data_status,
                                                     sizeof(rocsparse_data_status),
                                                     hipMemcpyDeviceToHost,
                                                     handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
        return rocsparse_status_success;
    }
}
