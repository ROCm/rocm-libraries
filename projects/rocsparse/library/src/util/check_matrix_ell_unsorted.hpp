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
    // check_matrix_ell_workspace_query returns for the same limits.
    template <typename I>
    rocsparse_status check_matrix_ell_unsorted_duplicates(rocsparse_handle       handle,
                                                          I                      m,
                                                          I                      n,
                                                          I                      ell_width,
                                                          const I*               ell_col_ind,
                                                          rocsparse_index_base   idx_base,
                                                          rocsparse_data_status* data_status,
                                                          void*                  temp_buffer,
                                                          const check_matrix_ell_sort_limits& limits
                                                          = check_matrix_ell_sort_limits{})
    {
        check_matrix_ell_workspace ws;
        RETURN_IF_ROCSPARSE_ERROR(check_matrix_ell_workspace_query<I>(
            handle, m, n, ell_width, rocsparse_storage_mode_unsorted, ws, limits));

        const int64_t batch_rows = ws.batch_rows;
        void*    rocprim_buffer  = check_matrix_workspace_at<char>(temp_buffer, ws.rocprim_offset);
        int64_t* offsets  = check_matrix_workspace_at<int64_t>(temp_buffer, ws.offsets_offset);
        I*       keys_buf = check_matrix_workspace_at<I>(temp_buffer, ws.keys_offset);
        I*       keys_alt = check_matrix_workspace_at<I>(temp_buffer, ws.keys_alt_offset);
        rocsparse_data_status* d_data_status
            = check_matrix_workspace_at<rocsparse_data_status>(temp_buffer, 0);

        // The segment offsets are relative to the first row of a batch, so all batches
        // share them.
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::check_matrix_ell_offsets_device<256>),
            dim3(static_cast<uint32_t>(rocsparse::ceil_div(batch_rows + 1, 256))),
            dim3(256),
            0,
            handle->stream,
            batch_rows,
            static_cast<int64_t>(ell_width),
            offsets);

        for(int64_t first_row = 0; first_row < m; first_row += batch_rows)
        {
            const int64_t rows = std::min(batch_rows, static_cast<int64_t>(m) - first_row);
            const dim3    blocks(static_cast<uint32_t>(rocsparse::ceil_div(rows, 256)));

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_ell_gather_device<256, I>),
                                               blocks,
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               m,
                                               n,
                                               ell_width,
                                               first_row,
                                               rows,
                                               ell_col_ind,
                                               idx_base,
                                               keys_buf);

            rocsparse::primitives::double_buffer<I> keys(keys_buf, keys_alt);
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys<I, int64_t>(
                handle,
                keys,
                static_cast<size_t>(rows * ell_width),
                static_cast<size_t>(rows),
                offsets,
                offsets + 1,
                0,
                ws.end_bit,
                ws.rocprim_bytes,
                rocprim_buffer)));

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::check_matrix_ell_duplicate_device<256, I>),
                blocks,
                dim3(256),
                0,
                handle->stream,
                rows,
                n,
                ell_width,
                keys.current(),
                d_data_status);
        }

        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(data_status,
                                                     d_data_status,
                                                     sizeof(rocsparse_data_status),
                                                     hipMemcpyDeviceToHost,
                                                     handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
        return rocsparse_status_success;
    }
}
