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

#include "rocsparse_mat_descr.hpp"
#include "rocsparse_spmat_descr.hpp"

namespace rocsparse
{
    //
    // \brief Build a CSR matrix descriptor from a CSC one for triangular solve.
    //
    // CSR is the transpose of the CSC matrix that shares the same arrays, so the
    // descriptor (type, diagonal type, index base, storage mode, ...) is copied
    // unchanged and only the triangular fill mode is flipped (lower <-> upper).
    //
    // The output is a caller-provided stack object, so no heap allocation is
    // performed and there is nothing to free.
    //
    // \param[in]  csc_descr the source CSC matrix descriptor.
    // \param[out] csr_descr the CSR matrix descriptor (csc_descr with fill flipped).
    //
    static inline void build_csr_descr_from_csc(const _rocsparse_mat_descr& csc_descr,
                                                _rocsparse_mat_descr&       csr_descr)
    {
        csr_descr = csc_descr;
        switch(csc_descr.fill_mode)
        {
        case rocsparse_fill_mode_lower:
            csr_descr.fill_mode = rocsparse_fill_mode_upper;
            break;
        case rocsparse_fill_mode_upper:
            csr_descr.fill_mode = rocsparse_fill_mode_lower;
            break;
        }
    }

    //
    // \brief Build a CSR view of a CSC sparse matrix descriptor for triangular solve.
    //
    // A CSC matrix is the transpose of the CSR matrix that shares the same arrays:
    // the CSC column-pointer / row-index arrays act as the CSR row-pointer /
    // column-index arrays, the row and column dimensions are swapped, and the
    // triangular fill mode is flipped (lower <-> upper).
    //
    // Both outputs are caller-provided stack objects, so no heap allocation is
    // performed and there is nothing to free. The returned \p csr view only
    // references the data owned by \p csc and must not outlive it.
    //
    // \param[in]  csc       the source CSC sparse matrix descriptor.
    // \param[out] csr       the CSR sparse matrix descriptor viewing csc^T.
    // \param[out] csr_descr the CSR matrix descriptor (referenced by \p csr).
    //
    static inline void build_csr_from_csc(const _rocsparse_spmat_descr& csc,
                                          _rocsparse_spmat_descr&       csr,
                                          _rocsparse_mat_descr&         csr_descr)
    {
        // Copy the matrix descriptor and flip only the fill mode, since CSR is
        // the transpose of CSC.
        rocsparse::build_csr_descr_from_csc(*csc.descr, csr_descr);

        // Reinterpret the CSC arrays as the transposed CSR matrix.
        csr                = csc;
        csr.rows           = csc.cols;
        csr.cols           = csc.rows;
        csr.row_data       = csc.col_data;
        csr.col_data       = csc.row_data;
        csr.const_row_data = csc.const_col_data;
        csr.const_col_data = csc.const_row_data;
        csr.row_type       = csc.col_type;
        csr.col_type       = csc.row_type;
        csr.format         = rocsparse_format_csr;
        csr.descr          = &csr_descr;

        // rocsparse_csc_set_strided_batch stores the value batch stride in
        // columns_values_batch_stride and leaves batch_stride unset (unlike the CSR
        // setter, which also fills batch_stride). The triangular solve reads the
        // value stride from batch_stride, so map it explicitly here to keep batched
        // CSC solves correct.
        csr.batch_stride = csc.columns_values_batch_stride;
    }
}
