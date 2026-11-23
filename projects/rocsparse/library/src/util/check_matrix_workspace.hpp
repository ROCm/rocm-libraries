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

#include "rocsparse_control.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

#include <algorithm>
#include <limits>

namespace rocsparse
{
    // Temporary buffer layout of the COO and ELL checks. The status slot is at offset 0 and
    // is present whenever there is data to check. The other sections are used by the
    // duplicate pass, which sorts a copy of the indices, and only exist for unsorted storage.
    // Every *_offset is a byte offset into the temporary buffer, and total is its size.
    struct check_matrix_coo_workspace
    {
        size_t   rocprim_bytes{};
        size_t   rocprim_offset{};
        size_t   cols_offset{};
        size_t   cols_alt_offset{};
        size_t   rows_offset{};
        size_t   rows_alt_offset{};
        uint32_t end_bit{};
        size_t   total{};
    };

    // The keys sections hold one batch of rows.
    struct check_matrix_ell_workspace
    {
        int64_t  batch_rows{};
        size_t   rocprim_bytes{};
        size_t   rocprim_offset{};
        size_t   offsets_offset{};
        size_t   keys_offset{};
        size_t   keys_alt_offset{};
        uint32_t end_bit{};
        size_t   total{};
    };

    // Appends a section of bytes to the layout and returns its offset.
    inline size_t check_matrix_workspace_append(size_t& total, size_t bytes)
    {
        const size_t offset = total;
        total += bytes;
        return offset;
    }

    template <typename T>
    inline T* check_matrix_workspace_at(void* temp_buffer, size_t offset)
    {
        return reinterpret_cast<T*>(reinterpret_cast<char*>(temp_buffer) + offset);
    }

    // Radix sort bits needed for keys in [0, max_key].
    inline uint32_t check_matrix_key_bits(int64_t max_key)
    {
        if(max_key <= 0)
        {
            return 1;
        }

        return static_cast<uint32_t>(64
                                     - __builtin_clzll(static_cast<unsigned long long>(max_key)));
    }

    // The duplicate pass sorts the stored indices, which the first pass has checked to lie
    // in [idx_base, max(m, n) - 1 + idx_base].
    template <typename I>
    inline rocsparse_status check_matrix_coo_workspace_query(rocsparse_handle            handle,
                                                             I                           m,
                                                             I                           n,
                                                             int64_t                     nnz,
                                                             rocsparse_index_base        idx_base,
                                                             rocsparse_storage_mode      storage,
                                                             check_matrix_coo_workspace& ws)
    {
        ws = {};

        if(nnz == 0)
        {
            return rocsparse_status_success;
        }

        check_matrix_workspace_append(ws.total, rocsparse::align_size<rocsparse_data_status>(1));

        if(storage == rocsparse_storage_mode_unsorted && nnz >= 2)
        {
            const int64_t dim = std::max(static_cast<int64_t>(m), static_cast<int64_t>(n));
            ws.end_bit        = check_matrix_key_bits(dim - 1 + idx_base);

            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<I, I>(
                handle, static_cast<size_t>(nnz), 0, ws.end_bit, &ws.rocprim_bytes, true)));

            const size_t array_bytes = rocsparse::align_size<I>(static_cast<size_t>(nnz));
            ws.rocprim_offset        = check_matrix_workspace_append(
                ws.total, rocsparse::align_size<char>(ws.rocprim_bytes));
            ws.cols_offset     = check_matrix_workspace_append(ws.total, array_bytes);
            ws.cols_alt_offset = check_matrix_workspace_append(ws.total, array_bytes);
            ws.rows_offset     = check_matrix_workspace_append(ws.total, array_bytes);
            ws.rows_alt_offset = check_matrix_workspace_append(ws.total, array_bytes);
        }

        return rocsparse_status_success;
    }

    // Limits of one batch of the unsorted ELL duplicate pass, which sorts whole rows as the
    // segments of rocprim::segmented_radix_sort_keys.
    struct check_matrix_ell_sort_limits
    {
        // The sort takes the number of keys and segments as unsigned int and reads each
        // segment offset as unsigned int.
        int64_t entries = std::numeric_limits<uint32_t>::max();

        // The sort configured in rocsparse_segmented_radix_sort_keys.cpp gives a segment at
        // most one block of 256 threads, and the work-items of one launch must fit in 32 bits.
        // The gather and duplicate kernels of a batch run one 256-thread block per 256 rows.
        int64_t rows = rocsparse::dispatch_limit_x(256);
    };

    // The unsorted ELL duplicate pass needs one row to fit in a batch.
    inline bool check_matrix_ell_sort_supported(rocsparse_storage_mode              storage,
                                                int64_t                             ell_width,
                                                const check_matrix_ell_sort_limits& limits)
    {
        return storage != rocsparse_storage_mode_unsorted || ell_width <= limits.entries;
    }

    // Rows in each batch of the unsorted ELL duplicate pass, for a supported ell_width > 0.
    inline int64_t check_matrix_ell_rows_per_batch(int64_t                             m,
                                                   int64_t                             ell_width,
                                                   const check_matrix_ell_sort_limits& limits)
    {
        return std::min({m, limits.entries / ell_width, limits.rows});
    }

    // The duplicate pass sorts zero-based column indices with padding stored as n.
    template <typename I>
    inline rocsparse_status
        check_matrix_ell_workspace_query(rocsparse_handle                    handle,
                                         I                                   m,
                                         I                                   n,
                                         I                                   ell_width,
                                         rocsparse_storage_mode              storage,
                                         check_matrix_ell_workspace&         ws,
                                         const check_matrix_ell_sort_limits& limits
                                         = check_matrix_ell_sort_limits{})
    {
        ws = {};

        const int64_t entries = static_cast<int64_t>(m) * ell_width;
        if(entries == 0)
        {
            return rocsparse_status_success;
        }

        if(!check_matrix_ell_sort_supported(storage, ell_width, limits))
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        check_matrix_workspace_append(ws.total, rocsparse::align_size<rocsparse_data_status>(1));

        if(storage == rocsparse_storage_mode_unsorted && ell_width > 1)
        {
            ws.batch_rows = check_matrix_ell_rows_per_batch(m, ell_width, limits);
            ws.end_bit    = check_matrix_key_bits(n);

            const int64_t batch_entries = ws.batch_rows * ell_width;
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<I, int64_t>(
                    handle,
                    static_cast<size_t>(batch_entries),
                    static_cast<size_t>(ws.batch_rows),
                    0,
                    ws.end_bit,
                    &ws.rocprim_bytes)));

            const size_t keys_bytes = rocsparse::align_size<I>(static_cast<size_t>(batch_entries));
            ws.rocprim_offset       = check_matrix_workspace_append(
                ws.total, rocsparse::align_size<char>(ws.rocprim_bytes));
            ws.offsets_offset = check_matrix_workspace_append(
                ws.total, rocsparse::align_size<int64_t>(static_cast<size_t>(ws.batch_rows) + 1));
            ws.keys_offset     = check_matrix_workspace_append(ws.total, keys_bytes);
            ws.keys_alt_offset = check_matrix_workspace_append(ws.total, keys_bytes);
        }

        return rocsparse_status_success;
    }
}
