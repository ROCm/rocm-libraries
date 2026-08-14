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
    // Persistent CSR image of an ELL matrix.
    //
    // Some routines (e.g. rocsparse_spsv) only implement an ELL matrix by
    // converting it once to CSR and then delegating to the CSR backend. The
    // conversion needs a device allocation and a device->host copy (to obtain
    // the number of non-zeros), neither of which are legal under HIP graph
    // capture, so it must be performed during the blocking preprocess/analysis
    // stage and the resulting CSR arrays kept alive for the (capture-safe)
    // compute stage. This structure owns those arrays for the lifetime of the
    // matrix descriptor's info object.
    struct ell2csr_info_t
    {
    private:
        int64_t             m_num_rows{};
        int64_t             m_num_cols{};
        int64_t             m_csr_nnz{};
        rocsparse_indextype m_csr_row_ptr_indextype{};
        rocsparse_indextype m_csr_col_ind_indextype{};
        rocsparse_datatype  m_csr_val_datatype{};

        void* m_csr_row_ptr{};
        void* m_csr_col_ind{};
        void* m_csr_val{};

    public:
        ell2csr_info_t() = delete;

        ell2csr_info_t(int64_t             num_rows,
                       int64_t             num_cols,
                       rocsparse_indextype csr_row_ptr_indextype,
                       rocsparse_indextype csr_col_ind_indextype,
                       rocsparse_datatype  csr_val_datatype,
                       hipStream_t         stream);

        hipError_t free_memory(hipStream_t stream);
        ~ell2csr_info_t();

        // Convert the ELL matrix held by descriptor \p ell into the CSR arrays
        // owned by this structure. Must be called exactly once, outside of any
        // HIP graph capture.
        rocsparse_status calculate(rocsparse_handle handle, rocsparse_const_spmat_descr ell);

        int64_t             get_csr_nnz() const;
        rocsparse_indextype get_csr_row_ptr_indextype() const;
        rocsparse_indextype get_csr_col_ind_indextype() const;
        rocsparse_datatype  get_csr_val_datatype() const;

        const void* get_csr_row_ptr() const;
        const void* get_csr_col_ind() const;
        const void* get_csr_val() const;
    };
}
