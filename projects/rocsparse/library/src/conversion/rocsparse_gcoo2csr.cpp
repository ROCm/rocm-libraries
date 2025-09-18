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
#include "rocsparse_gcoo2csr.hpp"
#include "rocsparse_convert_array.hpp"
#include "rocsparse_coo2csr.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::gcoo2csr(rocsparse_handle     handle_,
                                     rocsparse_indextype  source_row_type_,
                                     const void*          source_row_,
                                     int64_t              nnz_,
                                     int64_t              m_,
                                     rocsparse_indextype  target_row_type_,
                                     void*                target_row_,
                                     rocsparse_index_base idx_base_)
{
    ROCSPARSE_ROUTINE_TRACE;

#define DO(SROW, TROW)                                                          \
    do                                                                          \
    {                                                                           \
        RETURN_IF_ROCSPARSE_ERROR(                                              \
            (rocsparse::coo2csr_template<TROW, SROW>)(handle_,                  \
                                                      (const SROW*)source_row_, \
                                                      nnz_,                     \
                                                      m_,                       \
                                                      (TROW*)target_row_,       \
                                                      idx_base_));              \
        return rocsparse_status_success;                                        \
    } while(false)

    switch(source_row_type_)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        switch(target_row_type_)
        {
        case rocsparse_indextype_u16:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        case rocsparse_indextype_i32:
            DO(int32_t, int32_t);
        case rocsparse_indextype_i64:
            DO(int32_t, int64_t);
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    case rocsparse_indextype_i64:
    {
        switch(target_row_type_)
        {
        case rocsparse_indextype_u16:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        case rocsparse_indextype_i32:
            DO(int64_t, int32_t);
        case rocsparse_indextype_i64:
            DO(int64_t, int64_t);
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    }
#undef DO
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

rocsparse_status rocsparse::spmat_coo2csr_buffer_size(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr source_,
                                                      rocsparse_spmat_descr       target_,
                                                      size_t*                     buffer_size_)
{
    ROCSPARSE_ROUTINE_TRACE;

    buffer_size_[0] = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_coo2csr(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr source_,
                                          rocsparse_spmat_descr       target_,
                                          size_t                      buffer_size_,
                                          void*                       buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcoo2csr(handle,
                                                  source_->row_type,
                                                  source_->const_row_data,
                                                  source_->nnz,
                                                  source_->rows,
                                                  target_->row_type,
                                                  target_->row_data,
                                                  source_->idx_base));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                       source_->nnz,
                                                       target_->col_type,
                                                       target_->col_data,
                                                       source_->col_type,
                                                       source_->const_col_data));

    if(source_->const_val_data != nullptr && target_->val_data != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                           source_->nnz,
                                                           target_->data_type,
                                                           target_->val_data,
                                                           source_->data_type,
                                                           source_->const_val_data));
    }

    return rocsparse_status_success;
}
