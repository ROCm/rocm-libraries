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

#include "rocsparse_spmat_transfer_from.hpp"
#include "rocsparse_convert_array.hpp"

#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::internal_spmat_transfer_from(rocsparse_handle            handle,
                                                         rocsparse_spmat_descr       target,
                                                         rocsparse_const_spmat_descr source)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, target);
    ROCSPARSE_CHECKARG_POINTER(2, source);
    ROCSPARSE_CHECKARG(
        1, target, (target->get_format() != source->get_format()), rocsparse_status_invalid_value);

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_invalid_size,
                              target->get_rows() != source->get_rows());
    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_invalid_size,
                              target->get_cols() != source->get_cols());
    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_invalid_size,
                              target->get_nnz() != source->get_nnz());

    int64_t size_row_data = 0;
    int64_t size_col_data = 0;
    int64_t size_ind_data = 0;
    int64_t size_val_data = 0;
    switch(source->get_format())
    {
    case rocsparse_format_csr:
    {
        size_row_data = source->get_rows() + 1;
        size_col_data = source->get_nnz();
        size_val_data = source->get_nnz();
        break;
    }
    case rocsparse_format_csc:
    {
        size_col_data = source->get_cols() + 1;
        size_row_data = source->get_nnz();
        size_val_data = source->get_nnz();
        break;
    }
    case rocsparse_format_coo:
    {
        size_row_data = source->get_nnz();
        size_col_data = source->get_nnz();
        size_val_data = source->get_nnz();
        break;
    }
    case rocsparse_format_coo_aos:
    {
        size_ind_data = source->get_nnz() * 2;
        size_val_data = source->get_nnz();
        break;
    }
    case rocsparse_format_ell:
    {
        size_col_data = source->get_nnz();
        size_val_data = source->get_nnz();
        break;
    }
    case rocsparse_format_bell:
    {
        RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                  source->get_format() == rocsparse_format_bell);
        break;
    }
    case rocsparse_format_sell:
    {
        RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                  source->format == rocsparse_format_sell);
        break;
    }
    case rocsparse_format_bsr:
    {
        size_row_data = source->get_rows() + 1;
        size_col_data = source->get_nnz();
        size_val_data = source->get_block_dim() * source->get_block_dim() * source->get_nnz();
        break;
    }
    }

    //
    // ROW_DATA.
    //
    if(source->get_const_row_data() != nullptr)
    {
        ROCSPARSE_CHECKARG(
            1,
            target,
            ((source->get_const_row_data() != nullptr) && (target->get_row_data() == nullptr)),
            rocsparse_status_invalid_pointer);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                           size_row_data,
                                                           target->get_row_type(),
                                                           target->get_row_data(),
                                                           source->get_row_type(),
                                                           source->get_const_row_data()));
    }

    //
    // COL_DATA.
    //
    if(source->get_const_col_data() != nullptr)
    {
        ROCSPARSE_CHECKARG(
            1,
            target,
            ((source->get_const_col_data() != nullptr) && (target->get_col_data() == nullptr)),
            rocsparse_status_invalid_pointer);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                           size_col_data,
                                                           target->get_col_type(),
                                                           target->get_col_data(),
                                                           source->get_col_type(),
                                                           source->get_const_col_data()));
    }

    //
    // IND_DATA.
    //
    if(source->get_const_ind_data() != nullptr)
    {
        ROCSPARSE_CHECKARG(
            1,
            target,
            ((source->get_const_ind_data() != nullptr) && (target->get_ind_data() == nullptr)),
            rocsparse_status_invalid_pointer);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                           size_ind_data,
                                                           target->get_row_type(),
                                                           target->get_ind_data(),
                                                           source->get_row_type(),
                                                           source->get_const_ind_data()));
    }

    //
    // VAL_DATA.
    //
    if(source->get_const_val_data() != nullptr)
    {
        ROCSPARSE_CHECKARG(
            1,
            target,
            ((source->get_const_val_data() != nullptr) && (target->get_val_data() == nullptr)),
            rocsparse_status_invalid_pointer);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                           size_val_data,
                                                           target->get_data_type(),
                                                           target->get_val_data(),
                                                           source->get_data_type(),
                                                           source->get_const_val_data()));
    }

    return rocsparse_status_success;
}
