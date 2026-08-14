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

#include "rocsparse-types.h"

namespace rocsparse
{
    // Persistent analysis result of the native ELL triangular solve.
    //
    // The analysis (preprocess) stage computes a row execution order (row map)
    // by a level-scheduling of the ELL matrix. This ordering is reused by every
    // (capture-safe) compute stage, so the device buffer holding it must outlive
    // the analysis stage. This structure owns that buffer for the lifetime of the
    // matrix descriptor's info object and remembers the configuration it was
    // computed for so that it can be recomputed if the configuration changes.
    struct ellsv_info_t
    {
    private:
        int64_t             m_num_rows{};
        rocsparse_indextype m_index_type{};
        rocsparse_datatype  m_value_type{};

        bool                m_computed{false};
        rocsparse_operation m_trans{};
        rocsparse_fill_mode m_fill_mode{};
        rocsparse_diag_type m_diag_type{};

        // Device array of size num_rows holding the row execution order.
        void* m_row_map{};

        // Transposed ELL matrix (only allocated for the transposed operations).
        // It is stored in the same column-major layout with leading dimension
        // num_rows and a (possibly different) width.
        int64_t m_transposed_width{};
        void*   m_transposed_col_ind{};
        void*   m_transposed_val{};

    public:
        ellsv_info_t() = delete;

        ellsv_info_t(int64_t             num_rows,
                     rocsparse_indextype index_type,
                     rocsparse_datatype  value_type,
                     hipStream_t         stream);

        hipError_t free_memory(hipStream_t stream);
        hipError_t free_transposed(hipStream_t stream);
        ~ellsv_info_t();

        // Whether a previously computed row map matches the requested config.
        bool matches(rocsparse_operation trans,
                     rocsparse_fill_mode fill_mode,
                     rocsparse_diag_type diag_type) const;

        // Record the configuration the row map has been computed for.
        void set_config(rocsparse_operation trans,
                        rocsparse_fill_mode fill_mode,
                        rocsparse_diag_type diag_type);

        // (Re)allocate the transposed ELL storage for the given width.
        rocsparse_status allocate_transposed(int64_t width, hipStream_t stream);

        int64_t             get_num_rows() const;
        rocsparse_indextype get_index_type() const;
        rocsparse_datatype  get_value_type() const;
        void*               get_row_map();
        const void*         get_row_map() const;

        int64_t     get_transposed_width() const;
        void*       get_transposed_col_ind();
        const void* get_transposed_col_ind() const;
        void*       get_transposed_val();
        const void* get_transposed_val() const;
    };
}
