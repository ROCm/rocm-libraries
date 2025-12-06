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

#include "rocsparse_bsric0_strided_batched_kernel_17_32.hpp"
#include "rocsparse_bsric0_strided_batched_kernel_2_8.hpp"
#include "rocsparse_bsric0_strided_batched_kernel_2_8_unrolled.hpp"
#include "rocsparse_bsric0_strided_batched_kernel_9_16.hpp"
#include "rocsparse_bsric0_strided_batched_kernel_general.hpp"
#include "rocsparse_bsric0_strided_batched_kernel_launch.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status
    rocsparse::bsric0_strided_batched_kernel_launch(rocsparse_handle     handle,
                                                    rocsparse_direction  dir,
                                                    int64_t              batch_count,
                                                    int64_t              mb,
                                                    rocsparse_datatype   bsr_val_datatype,
                                                    void*                bsr_val,
                                                    int64_t              bsr_val_stride,
                                                    rocsparse_indextype  bsr_row_ptr_indextype,
                                                    const void*          bsr_row_ptr,
                                                    rocsparse_indextype  bsr_col_ind_indextype,
                                                    const void*          bsr_col_ind,
                                                    const void*          bsr_diag_ind,
                                                    int64_t              bsr_dim,
                                                    int32_t*             done_array,
                                                    int64_t              done_array_stride,
                                                    const void*          map,
                                                    void*                zero_pivot,
                                                    int64_t              zero_pivot_stride,
                                                    rocsparse_index_base idx_base,
                                                    int64_t              max_nnzb)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::bsric0_kernel_general_launch_t bsric0_kernel_launch;
    if(handle->wavefront_size == 32)
    {
        bsric0_kernel_launch = find_bsric0_strided_batched_kernel_general_launch(
            32, 32, false, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
    }
    else
    {
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        const bool sleep = (gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2);
        if(sleep)
        {
            bsric0_kernel_launch = find_bsric0_strided_batched_kernel_general_launch(
                64, 64, true, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
        }
        else
        {
            if(max_nnzb <= 32)
            {
                if(bsr_dim <= 8)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_2_8_unrolled_launch(
                        bsr_dim * bsr_dim,
                        32,
                        bsr_dim,
                        bsr_val_datatype,
                        bsr_row_ptr_indextype,
                        bsr_col_ind_indextype);
                }
                else if(bsr_dim <= 16)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_9_16_launch(
                        64, 32, 16, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
                }
                else if(bsr_dim <= 32)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_17_32_launch(
                        64, 32, 32, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
                }
                else
                {
                    bsric0_kernel_launch
                        = find_bsric0_strided_batched_kernel_general_launch(64,
                                                                            64,
                                                                            false,
                                                                            bsr_val_datatype,
                                                                            bsr_row_ptr_indextype,
                                                                            bsr_col_ind_indextype);
                }
            }
            else if(max_nnzb <= 64)
            {
                if(bsr_dim <= 8)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_2_8_launch(
                        64, 64, 8, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
                }
                else if(bsr_dim <= 16)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_9_16_launch(
                        64, 64, 16, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
                }
                else if(bsr_dim <= 32)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_17_32_launch(
                        64, 64, 32, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
                }
                else
                {
                    bsric0_kernel_launch
                        = find_bsric0_strided_batched_kernel_general_launch(64,
                                                                            64,
                                                                            false,
                                                                            bsr_val_datatype,
                                                                            bsr_row_ptr_indextype,
                                                                            bsr_col_ind_indextype);
                }
            }
            else if(max_nnzb <= 128)
            {
                if(bsr_dim <= 8)
                {
                    bsric0_kernel_launch = find_bsric0_strided_batched_kernel_2_8_launch(
                        64, 128, 8, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
                }
                else if(bsr_dim <= 16)
                {
                    bsric0_kernel_launch
                        = find_bsric0_strided_batched_kernel_9_16_launch(64,
                                                                         128,
                                                                         16,
                                                                         bsr_val_datatype,
                                                                         bsr_row_ptr_indextype,
                                                                         bsr_col_ind_indextype);
                }
                else if(bsr_dim <= 32)
                {
                    bsric0_kernel_launch
                        = find_bsric0_strided_batched_kernel_17_32_launch(64,
                                                                          128,
                                                                          32,
                                                                          bsr_val_datatype,
                                                                          bsr_row_ptr_indextype,
                                                                          bsr_col_ind_indextype);
                }
                else
                {
                    bsric0_kernel_launch
                        = find_bsric0_strided_batched_kernel_general_launch(64,
                                                                            64,
                                                                            false,
                                                                            bsr_val_datatype,
                                                                            bsr_row_ptr_indextype,
                                                                            bsr_col_ind_indextype);
                }
            }
            else
            {
                bsric0_kernel_launch = find_bsric0_strided_batched_kernel_general_launch(
                    64, 64, false, bsr_val_datatype, bsr_row_ptr_indextype, bsr_col_ind_indextype);
            }
        }
    }

    if(bsric0_kernel_launch == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch,
                                               "No suitable kernel detected from bsric0");
    }

    RETURN_IF_ROCSPARSE_ERROR(bsric0_kernel_launch(handle,
                                                   dir,
                                                   batch_count,
                                                   mb,
                                                   bsr_row_ptr,
                                                   bsr_col_ind,
                                                   bsr_val,
                                                   bsr_val_stride,
                                                   bsr_diag_ind,
                                                   bsr_dim,
                                                   done_array,
                                                   done_array_stride,
                                                   map,
                                                   zero_pivot,
                                                   zero_pivot_stride,
                                                   idx_base));

    return rocsparse_status_success;
}

namespace rocsparse
{
    rocsparse_status bsric0_strided_batched_zero_pivot(rocsparse_handle      handle,
                                                       rocsparse_bsric0_info info,
                                                       rocsparse_indextype   indextype,
                                                       rocsparse_int         batch_count,
                                                       void*                 position)

    {
        ROCSPARSE_ROUTINE_TRACE;

        if(batch_count == 0)
        {
            return rocsparse_status_success;
        }
        // Stream
        hipStream_t stream = handle->stream;

        if(info == nullptr)
        {
            rocsparse::set_minus_one_async(
                handle->pointer_mode, indextype, batch_count, position, stream);
            return rocsparse_status_success;
        }
        auto status = info->copy_zero_pivot_async(
            batch_count, handle->pointer_mode, indextype, position, stream);
        if(status == rocsparse_status_zero_pivot)
        {
            return status;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }
}
