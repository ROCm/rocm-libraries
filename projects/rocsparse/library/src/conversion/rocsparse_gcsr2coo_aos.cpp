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
#include "rocsparse_gcsr2coo_aos.hpp"
#include "rocsparse_convert_array.hpp"
#include "rocsparse_gcsr2coo.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::gcsr2coo_aos_buffer_size(rocsparse_handle     handle,
                                                     int64_t              m,
                                                     int64_t              nnz,
                                                     rocsparse_indextype  source_row_type,
                                                     const void*          source_row_data,
                                                     rocsparse_indextype  source_col_type,
                                                     const void*          source_col_data,
                                                     rocsparse_index_base source_idx_base,
                                                     rocsparse_indextype  target_ind_type,
                                                     size_t*              buffer_size_)
{
    ROCSPARSE_ROUTINE_TRACE;

    buffer_size_[0] = rocsparse::indextype_sizeof(target_ind_type) * nnz;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::gcsr2coo_aos(rocsparse_handle     handle,
                                         int64_t              m,
                                         int64_t              nnz,
                                         rocsparse_indextype  source_row_type,
                                         const void*          source_row_data,
                                         rocsparse_indextype  source_col_type,
                                         const void*          source_col_data,
                                         rocsparse_datatype   source_data_type,
                                         const void*          source_val_data,
                                         rocsparse_index_base source_idx_base,
                                         rocsparse_indextype  target_ind_type,
                                         void*                target_ind_data,
                                         rocsparse_datatype   target_data_type,
                                         void*                target_val_data,
                                         rocsparse_index_base target_idx_base,
                                         size_t               buffer_size_,
                                         void*                buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsr2coo(handle,
                                                  source_row_type,
                                                  source_row_data,
                                                  nnz,
                                                  m,
                                                  target_ind_type,
                                                  buffer_, // this contains the row.
                                                  source_idx_base));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(
        handle, nnz, target_ind_type, target_ind_data, 2, target_ind_type, buffer_, 1));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(
        handle,
        nnz,
        target_ind_type,
        reinterpret_cast<char*>(target_ind_data) + rocsparse::indextype_sizeof(target_ind_type),
        2,
        source_col_type,
        source_col_data,
        1));

    if(source_val_data != nullptr && target_val_data != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(
            handle, nnz, target_data_type, target_val_data, source_data_type, source_val_data));
    }
    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_csr2coo_aos_buffer_size(rocsparse_handle            handle,
                                                          rocsparse_const_spmat_descr source_,
                                                          rocsparse_spmat_descr       target_,
                                                          size_t*                     buffer_size_)

{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsr2coo_aos_buffer_size(handle,
                                                                  source_->rows,
                                                                  source_->nnz,
                                                                  source_->row_type,
                                                                  source_->const_row_data,
                                                                  source_->col_type,
                                                                  source_->const_col_data,
                                                                  source_->idx_base,
                                                                  target_->row_type,
                                                                  buffer_size_));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_csr2coo_aos(rocsparse_handle            handle,
                                              rocsparse_const_spmat_descr source_,
                                              rocsparse_spmat_descr       target_,
                                              size_t                      buffer_size_,
                                              void*                       buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsr2coo_aos(handle,
                                                      source_->rows,
                                                      source_->nnz,
                                                      source_->row_type,
                                                      source_->const_row_data,
                                                      source_->col_type,
                                                      source_->const_col_data,
                                                      source_->data_type,
                                                      source_->const_val_data,
                                                      source_->idx_base,
                                                      target_->row_type,
                                                      target_->ind_data,
                                                      target_->data_type,
                                                      target_->val_data,
                                                      target_->idx_base,
                                                      buffer_size_,
                                                      buffer_));

    return rocsparse_status_success;
}
