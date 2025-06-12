/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

/********************************************************************************
 * \brief Copy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse::copy_csrmv_info(rocsparse_csrmv_info       dest,
                                            const rocsparse_csrmv_info src)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(dest == nullptr || src == nullptr || dest == src)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;

    previously_created |= (dest->adaptive.size != 0);
    previously_created |= (dest->adaptive.first_row != 0);
    previously_created |= (dest->adaptive.last_row != 0);
    previously_created |= (dest->adaptive.row_blocks != nullptr);
    previously_created |= (dest->adaptive.wg_flags != nullptr);
    previously_created |= (dest->adaptive.wg_ids != nullptr);

    previously_created |= (dest->lrb.size != 0);
    previously_created |= (dest->lrb.wg_flags != nullptr);
    previously_created |= (dest->lrb.rows_offsets_scratch != nullptr);
    previously_created |= (dest->lrb.rows_bins != nullptr);
    previously_created |= (dest->lrb.n_rows_bins != nullptr);

    previously_created |= (dest->trans != rocsparse_operation_none);
    previously_created |= (dest->m != 0);
    previously_created |= (dest->n != 0);
    previously_created |= (dest->nnz != 0);
    previously_created |= (dest->max_rows != 0);
    previously_created |= (dest->descr != nullptr);
    previously_created |= (dest->csr_row_ptr != nullptr);
    previously_created |= (dest->csr_col_ind != nullptr);
    previously_created |= (dest->index_type_I != rocsparse_indextype_u16);
    previously_created |= (dest->index_type_J != rocsparse_indextype_u16);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->adaptive.size != src->adaptive.size);
        invalid |= (dest->adaptive.first_row != src->adaptive.first_row);
        invalid |= (dest->adaptive.last_row != src->adaptive.last_row);
        invalid |= (dest->lrb.size != src->lrb.size);
        invalid |= (dest->trans != src->trans);
        invalid |= (dest->m != src->m);
        invalid |= (dest->n != src->n);
        invalid |= (dest->nnz != src->nnz);
        invalid |= (dest->max_rows != src->max_rows);
        invalid |= (dest->index_type_I != src->index_type_I);
        invalid |= (dest->index_type_J != src->index_type_J);

        if(invalid)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }
    }

    size_t I_size = sizeof(uint16_t);
    switch(src->index_type_I)
    {
    case rocsparse_indextype_u16:
    {
        I_size = sizeof(uint16_t);
        break;
    }
    case rocsparse_indextype_i32:
    {
        I_size = sizeof(int32_t);
        break;
    }
    case rocsparse_indextype_i64:
    {
        I_size = sizeof(int64_t);
        break;
    }
    }

    size_t J_size = sizeof(uint16_t);
    switch(src->index_type_J)
    {
    case rocsparse_indextype_u16:
    {
        J_size = sizeof(uint16_t);
        break;
    }
    case rocsparse_indextype_i32:
    {
        J_size = sizeof(int32_t);
        break;
    }
    case rocsparse_indextype_i64:
    {
        J_size = sizeof(int64_t);
        break;
    }
    }

    if(src->adaptive.row_blocks != nullptr)
    {
        if(dest->adaptive.row_blocks == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->adaptive.row_blocks, I_size * src->adaptive.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->adaptive.row_blocks,
                                      src->adaptive.row_blocks,
                                      I_size * src->adaptive.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->adaptive.wg_flags != nullptr)
    {
        if(dest->adaptive.wg_flags == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->adaptive.wg_flags,
                                                    sizeof(uint32_t) * src->adaptive.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->adaptive.wg_flags,
                                      src->adaptive.wg_flags,
                                      sizeof(uint32_t) * src->adaptive.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->adaptive.wg_ids != nullptr)
    {
        if(dest->adaptive.wg_ids == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->adaptive.wg_ids, J_size * src->adaptive.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->adaptive.wg_ids,
                                      src->adaptive.wg_ids,
                                      J_size * src->adaptive.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->lrb.wg_flags != nullptr)
    {
        if(dest->lrb.wg_flags == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->lrb.wg_flags, sizeof(uint32_t) * src->lrb.size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->lrb.wg_flags,
                                      src->lrb.wg_flags,
                                      sizeof(uint32_t) * src->lrb.size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->lrb.rows_offsets_scratch != nullptr)
    {
        if(dest->lrb.rows_offsets_scratch == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->lrb.rows_offsets_scratch, J_size * src->m));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->lrb.rows_offsets_scratch,
                                      src->lrb.rows_offsets_scratch,
                                      J_size * src->m,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->lrb.rows_bins != nullptr)
    {
        if(dest->lrb.rows_bins == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->lrb.rows_bins, J_size * src->m));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->lrb.rows_bins, src->lrb.rows_bins, J_size * src->m, hipMemcpyDeviceToDevice));
    }

    if(src->lrb.n_rows_bins != nullptr)
    {
        if(dest->lrb.n_rows_bins == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->lrb.n_rows_bins, J_size * 32));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->lrb.n_rows_bins, src->lrb.n_rows_bins, J_size * 32, hipMemcpyDeviceToDevice));
    }

    dest->adaptive.size      = src->adaptive.size;
    dest->adaptive.first_row = src->adaptive.first_row;
    dest->adaptive.last_row  = src->adaptive.last_row;
    dest->lrb.size           = src->lrb.size;
    dest->trans              = src->trans;
    dest->m                  = src->m;
    dest->n                  = src->n;
    dest->nnz                = src->nnz;
    dest->max_rows           = src->max_rows;
    dest->index_type_I       = src->index_type_I;
    dest->index_type_J       = src->index_type_J;

    // Not owned by the info struct. Just pointers to externally allocated memory
    dest->descr       = src->descr;
    dest->csr_row_ptr = src->csr_row_ptr;
    dest->csr_col_ind = src->csr_col_ind;

    return rocsparse_status_success;
}
