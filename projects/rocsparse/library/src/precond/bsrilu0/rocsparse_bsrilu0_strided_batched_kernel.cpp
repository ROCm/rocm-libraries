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

#include "rocsparse_bsrilu0_strided_batched_kernel_2_8.hpp"
#include "rocsparse_bsrilu0_strided_batched_kernel_33_64.hpp"
#include "rocsparse_bsrilu0_strided_batched_kernel_9_32.hpp"
#include "rocsparse_bsrilu0_strided_batched_kernel_general.hpp"
#include "rocsparse_bsrilu0_strided_batched_kernel_launch.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status
    rocsparse::bsrilu0_strided_batched_kernel_launch(rocsparse_handle     handle,
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
                                                     int                  boost,
                                                     size_t               boost_tol_size,
                                                     const void*          boost_tol,
                                                     const void*          boost_val)
{
    bsrilu0_kernel_2_8_launch_t bsrilu0_kernel_launch = nullptr;
    const std::string           gcn_arch_name         = rocsparse::handle_get_arch_name(handle);

    const bool sleep = (gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2);
    if(sleep || (handle->wavefront_size == 32)
       || (bsr_dim > 64 && bsr_val_datatype != rocsparse_datatype_f64_c)
       || (bsr_dim > 32 && bsr_val_datatype == rocsparse_datatype_f64_c))
    {
        bsrilu0_kernel_launch = find_bsrilu0_strided_batched_kernel_general_launch(
            128,
            (sleep) ? 64 : handle->wavefront_size,
            sleep,
            bsr_val_datatype,
            bsr_row_ptr_indextype,
            bsr_col_ind_indextype);
    }
    else if(bsr_dim <= 8)
    {
        bsrilu0_kernel_launch
            = find_bsrilu0_strided_batched_kernel_2_8_launch(64,
                                                             handle->wavefront_size,
                                                             8,
                                                             bsr_val_datatype,
                                                             bsr_row_ptr_indextype,
                                                             bsr_col_ind_indextype);
    }
    else if(bsr_dim <= 16)
    {
        bsrilu0_kernel_launch
            = find_bsrilu0_strided_batched_kernel_9_32_launch(64,
                                                              handle->wavefront_size,
                                                              16,
                                                              bsr_val_datatype,
                                                              bsr_row_ptr_indextype,
                                                              bsr_col_ind_indextype);
    }
    else if(bsr_dim <= 32)
    {
        bsrilu0_kernel_launch
            = find_bsrilu0_strided_batched_kernel_9_32_launch(64,
                                                              handle->wavefront_size,
                                                              32,
                                                              bsr_val_datatype,
                                                              bsr_row_ptr_indextype,
                                                              bsr_col_ind_indextype);
    }
    else
    {
        bsrilu0_kernel_launch
            = find_bsrilu0_strided_batched_kernel_33_64_launch(64,
                                                               handle->wavefront_size,
                                                               64,
                                                               bsr_val_datatype,
                                                               bsr_row_ptr_indextype,
                                                               bsr_col_ind_indextype);
    }

    if(bsrilu0_kernel_launch == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
    }

    RETURN_IF_ROCSPARSE_ERROR(bsrilu0_kernel_launch(handle,
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
                                                    idx_base,
                                                    boost,
                                                    boost_tol_size,
                                                    boost_tol,
                                                    boost_val));

    return rocsparse_status_success;
}

namespace rocsparse
{
    rocsparse_status bsrilu0_strided_batched_zero_pivot(rocsparse_handle       handle,
                                                        rocsparse_bsrilu0_info info,
                                                        rocsparse_indextype    indextype,
                                                        rocsparse_int          batch_count,
                                                        void*                  position)

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
