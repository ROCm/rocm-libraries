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
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

#include <algorithm>
#include <limits>

namespace rocsparse
{
    // Temporary buffer layout of the COO and ELL checks. The status slot comes first and
    // is always present when there is data to check. The remaining parts are used by the
    // duplicate pass, which sorts a copy of the indices, and only exist for unsorted storage.
    struct check_matrix_coo_workspace
    {
        size_t   status_bytes{};
        size_t   rocprim_bytes{};
        size_t   rocprim_slot{};
        size_t   array_bytes{};
        uint32_t end_bit{};
        size_t   total{};
    };

    struct check_matrix_ell_workspace
    {
        size_t   status_bytes{};
        size_t   rocprim_bytes{};
        size_t   rocprim_slot{};
        size_t   offset_bytes{};
        size_t   array_bytes{};
        uint32_t end_bit{};
        size_t   total{};
    };

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

        ws.status_bytes = rocsparse::align_size<rocsparse_data_status>(1);
        ws.total        = ws.status_bytes;

        if(storage == rocsparse_storage_mode_unsorted && nnz >= 2)
        {
            const int64_t dim = std::max(static_cast<int64_t>(m), static_cast<int64_t>(n));
            ws.end_bit        = check_matrix_key_bits(dim - 1 + idx_base);

            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<I, I>(
                handle, static_cast<size_t>(nnz), 0, ws.end_bit, &ws.rocprim_bytes, true)));
            ws.rocprim_slot = rocsparse::align_size<char>(ws.rocprim_bytes);
            ws.array_bytes  = rocsparse::align_size<I>(static_cast<size_t>(nnz));
            ws.total += ws.rocprim_slot + 4 * ws.array_bytes;
        }

        return rocsparse_status_success;
    }

    // rocprim::segmented_radix_sort_keys takes the number of keys and segments as unsigned
    // int and reads each segment offset as unsigned int. The unsorted ELL duplicate pass
    // therefore sorts whole rows in batches of at most this many entries.
    constexpr int64_t check_matrix_ell_sort_batch_limit = std::numeric_limits<uint32_t>::max();

    // Rows in each batch of the unsorted ELL duplicate pass, for 0 < ell_width <= batch_limit.
    inline int64_t
        check_matrix_ell_rows_per_batch(int64_t m, int64_t ell_width, int64_t batch_limit)
    {
        return std::min(m, batch_limit / ell_width);
    }

    // The duplicate pass sorts zero-based column indices with padding stored as n.
    template <typename I>
    inline rocsparse_status check_matrix_ell_workspace_query(rocsparse_handle            handle,
                                                             I                           m,
                                                             I                           n,
                                                             I                           ell_width,
                                                             rocsparse_storage_mode      storage,
                                                             check_matrix_ell_workspace& ws,
                                                             int64_t                     batch_limit
                                                             = check_matrix_ell_sort_batch_limit)
    {
        ws = {};

        const int64_t entries = static_cast<int64_t>(m) * ell_width;
        if(entries == 0)
        {
            return rocsparse_status_success;
        }

        ws.status_bytes = rocsparse::align_size<rocsparse_data_status>(1);
        ws.total        = ws.status_bytes;

        if(storage == rocsparse_storage_mode_unsorted && ell_width > 1)
        {
            if(ell_width > batch_limit)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

            const int64_t batch_rows = check_matrix_ell_rows_per_batch(m, ell_width, batch_limit);
            ws.end_bit               = check_matrix_key_bits(n);

            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<I, int64_t>(
                    handle,
                    static_cast<size_t>(batch_rows * ell_width),
                    static_cast<size_t>(batch_rows),
                    0,
                    ws.end_bit,
                    &ws.rocprim_bytes)));
            ws.rocprim_slot = rocsparse::align_size<char>(ws.rocprim_bytes);
            ws.offset_bytes = rocsparse::align_size<int64_t>(static_cast<size_t>(batch_rows) + 1);
            ws.array_bytes  = rocsparse::align_size<I>(static_cast<size_t>(entries));
            ws.total += ws.rocprim_slot + ws.offset_bytes + 2 * ws.array_bytes;
        }

        return rocsparse_status_success;
    }
}
