/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csr2bsr.hpp"
#include "rocsparse_gcsr2bsr.hpp"

namespace rocsparse
{
    template <typename I>
    static rocsparse_status gcsr2bsr_nnz_a(rocsparse_handle          handle,
                                           rocsparse_direction       direction,
                                           int64_t                   m,
                                           int64_t                   n,
                                           const rocsparse_mat_descr csr_descr,
                                           const I*                  csr_row_ptr,
                                           rocsparse_indextype       csr_col_ind_indextype,
                                           const void*               csr_col_ind,
                                           int64_t                   block_dim,
                                           const rocsparse_mat_descr bsr_descr,
                                           I*                        bsr_row_ptr,
                                           I*                        nnzb)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(csr_col_ind_indextype)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#define CASE(VAL, TYPE)                                                                     \
    case VAL:                                                                               \
    {                                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2bsr_nnz_template(handle,                   \
                                                                  direction,                \
                                                                  (TYPE)m,                  \
                                                                  (TYPE)n,                  \
                                                                  csr_descr,                \
                                                                  csr_row_ptr,              \
                                                                  (const TYPE*)csr_col_ind, \
                                                                  (TYPE)block_dim,          \
                                                                  bsr_descr,                \
                                                                  bsr_row_ptr,              \
                                                                  nnzb));                   \
        return rocsparse_status_success;                                                    \
    }

            CASE(rocsparse_indextype_i32, int32_t);
            CASE(rocsparse_indextype_i64, int64_t);
#undef CASE
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

rocsparse_status rocsparse::gcsr2bsr_nnz(rocsparse_handle          handle,
                                         rocsparse_direction       direction,
                                         int64_t                   m,
                                         int64_t                   n,
                                         const rocsparse_mat_descr csr_descr,
                                         rocsparse_indextype       csr_row_ptr_indextype,
                                         const void*               csr_row_ptr,
                                         rocsparse_indextype       csr_col_ind_indextype,
                                         const void*               csr_col_ind,
                                         int64_t                   block_dim,
                                         const rocsparse_mat_descr bsr_descr,
                                         rocsparse_indextype       bsr_row_ptr_indextype,
                                         void*                     bsr_row_ptr,
                                         int64_t*                  bsr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              bsr_row_ptr_indextype != csr_row_ptr_indextype);

    switch(csr_row_ptr_indextype)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

#define CASE(VAL, I)                                                               \
    case VAL:                                                                      \
    {                                                                              \
        I local_bsr_nnz;                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsr2bsr_nnz_a(handle,                \
                                                            direction,             \
                                                            m,                     \
                                                            n,                     \
                                                            csr_descr,             \
                                                            (const I*)csr_row_ptr, \
                                                            csr_col_ind_indextype, \
                                                            csr_col_ind,           \
                                                            block_dim,             \
                                                            bsr_descr,             \
                                                            (I*)bsr_row_ptr,       \
                                                            &local_bsr_nnz));      \
        bsr_nnz[0] = local_bsr_nnz;                                                \
        return rocsparse_status_success;                                           \
    }

        CASE(rocsparse_indextype_i32, int32_t);
        CASE(rocsparse_indextype_i64, int64_t);
#undef CASE
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}
