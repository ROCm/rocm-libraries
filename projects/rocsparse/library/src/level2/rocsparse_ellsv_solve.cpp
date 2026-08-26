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

#include "ellsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_ellsv.hpp"
#include "rocsparse_ellsv_info.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_solve_kernel(I       m,
                            I       n,
                            int64_t ell_width,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const I* __restrict__ ell_col_ind,
                            const T* __restrict__ ell_val,
                            const T* __restrict__ x,
                            int64_t x_inc,
                            T*      y,
                            int64_t y_inc,
                            int* __restrict__ done_array,
                            const I* __restrict__ map,
                            I* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_fill_mode  fill_mode,
                            rocsparse_diag_type  diag_type,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::ellsv_device<BLOCKSIZE, WF_SIZE, SLEEP, I, T>(m,
                                                                 n,
                                                                 ell_width,
                                                                 alpha,
                                                                 ell_col_ind,
                                                                 ell_val,
                                                                 x,
                                                                 x_inc,
                                                                 y,
                                                                 y_inc,
                                                                 done_array,
                                                                 map,
                                                                 zero_pivot,
                                                                 idx_base,
                                                                 fill_mode,
                                                                 diag_type);
    }

    // Solves against whichever matrix it is handed, which is either the matrix
    // the caller supplied or the materialized transpose cached by the analysis
    // phase. Everything the kernel needs beyond the vectors - dimensions, ELL
    // width, index base, fill mode and diagonal type - is read off that matrix.
    template <uint32_t WF_SIZE, bool SLEEP, typename I, typename T>
    static rocsparse_status launch_ellsv_solve_kernel(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr A,
                                                      const void*                 alpha,
                                                      rocsparse_const_dnvec_descr x,
                                                      rocsparse_dnvec_descr       y,
                                                      const void*                 row_map,
                                                      void*                       zero_pivot,
                                                      void*                       temp_buffer,
                                                      size_t                      buffer_size,
                                                      bool                        is_host_mode)
    {
        constexpr uint32_t BLOCKSIZE = 1024;

        hipStream_t stream = handle->stream;

        const I m = static_cast<I>(A->rows);

        const size_t done_bytes = ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));

        // Must stay in sync with rocsparse::ellsv_solve_buffer_size.
        if(buffer_size < 256 + done_bytes)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
        }

        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;

        int32_t* done_array = reinterpret_cast<int32_t*>(ptr);

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(done_array, 0, done_bytes, stream));

        dim3 blocks((static_cast<size_t>(m) * WF_SIZE - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        auto alpha_ = reinterpret_cast<const T*>(alpha);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::ellsv_solve_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I, T>),
            blocks,
            threads,
            0,
            stream,
            m,
            static_cast<I>(A->cols),
            A->ell_width,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_),
            reinterpret_cast<const I*>(A->const_col_data),
            reinterpret_cast<const T*>(A->const_val_data),
            reinterpret_cast<const T*>(x->const_values),
            x->inc,
            reinterpret_cast<T*>(y->values),
            y->inc,
            done_array,
            reinterpret_cast<const I*>(row_map),
            reinterpret_cast<I*>(zero_pivot),
            A->idx_base,
            A->descr->fill_mode,
            A->descr->diag_type,
            is_host_mode);

        return rocsparse_status_success;
    }

    typedef decltype(&launch_ellsv_solve_kernel<32, false, int32_t, float>) ellsv_solve_kernel_t;

    template <uint32_t WF_SIZE, bool SLEEP, typename I>
    static ellsv_solve_kernel_t find_t(rocsparse_datatype t)
    {
        switch(t)
        {
        case rocsparse_datatype_f32_r:
            return launch_ellsv_solve_kernel<WF_SIZE, SLEEP, I, float>;
        case rocsparse_datatype_f64_r:
            return launch_ellsv_solve_kernel<WF_SIZE, SLEEP, I, double>;
        case rocsparse_datatype_f32_c:
            return launch_ellsv_solve_kernel<WF_SIZE, SLEEP, I, rocsparse_float_complex>;
        case rocsparse_datatype_f64_c:
            return launch_ellsv_solve_kernel<WF_SIZE, SLEEP, I, rocsparse_double_complex>;
        default:
            return nullptr;
        }
    }

    template <uint32_t WF_SIZE, bool SLEEP, typename... P>
    static ellsv_solve_kernel_t find_i(rocsparse_indextype i, P... p)
    {
        return (i == rocsparse_indextype_i32)   ? find_t<WF_SIZE, SLEEP, int32_t>(p...)
               : (i == rocsparse_indextype_i64) ? find_t<WF_SIZE, SLEEP, int64_t>(p...)
                                                : nullptr;
    }

    static ellsv_solve_kernel_t find_ellsv_solve_kernel(uint32_t            wfsize_,
                                                        bool                sleep_,
                                                        rocsparse_indextype i_type,
                                                        rocsparse_datatype  t_type)
    {
        if((wfsize_ == 32) && (sleep_ == false))
        {
            return find_i<32, false>(i_type, t_type);
        }
        else if((wfsize_ == 64) && (sleep_ == false))
        {
            return find_i<64, false>(i_type, t_type);
        }
        else if((wfsize_ == 64) && (sleep_ == true))
        {
            return find_i<64, true>(i_type, t_type);
        }
        else
        {
            return nullptr;
        }
    }

    static rocsparse_status ellsv_init_zero_pivot(rocsparse_handle            handle,
                                                  rocsparse_ellsv_info        ei,
                                                  rocsparse_const_spmat_descr A)
    {
        hipStream_t stream = handle->stream;

        ei->create_singularity_numeric_exact(1, A->col_type, stream);

        switch(A->descr->diag_type)
        {
        case rocsparse_diag_type_unit:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::assign_max_async(1, A->col_type, ei->get_position(), stream));
            if(A->col_type == rocsparse_indextype_i32)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                    1,
                    (int32_t*)ei->get_singularity_numeric_exact()->get_position(),
                    (const int32_t*)ei->get_position(),
                    stream));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                    1,
                    (int64_t*)ei->get_singularity_numeric_exact()->get_position(),
                    (const int64_t*)ei->get_position(),
                    stream));
            }
            break;
        }
        case rocsparse_diag_type_non_unit:
        {
            if(A->col_type == rocsparse_indextype_i32)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                    1,
                    (int32_t*)ei->get_singularity_numeric_exact()->get_position(),
                    (const int32_t*)ei->get_position(),
                    stream));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                    1,
                    (int64_t*)ei->get_singularity_numeric_exact()->get_position(),
                    (const int64_t*)ei->get_position(),
                    stream));
            }
            break;
        }
        }

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::ellsv_solve_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_operation         trans,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnvec_descr x,
                                                    rocsparse_const_dnvec_descr y,
                                                    size_t* buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, buffer_size_in_bytes);

    const int64_t batch_count = (y) ? y->batch_count : A->batch_count;

    if(A->rows == 0 || batch_count == 0)
    {
        *buffer_size_in_bytes = 0;
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    const int64_t m = A->rows;

    size_t size = 256;
    size += rocsparse::ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m)
                                      * static_cast<size_t>(batch_count));

    *buffer_size_in_bytes = size;

    return rocsparse_status_success;
}

rocsparse_status rocsparse::ellsv_solve(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_datatype          alpha_datatype,
                                        const void*                 alpha,
                                        int64_t                     alpha_stride,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y,
                                        rocsparse_ellsv_info        ellsv_info,
                                        void*                       temp_buffer,
                                        size_t                      buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(2, alpha_datatype);
    ROCSPARSE_CHECKARG_POINTER(5, A);
    ROCSPARSE_CHECKARG_POINTER(8, ellsv_info);
    ROCSPARSE_CHECKARG_ARRAY(3, A->batch_count, alpha);
    ROCSPARSE_CHECKARG_POINTER(6, x);
    ROCSPARSE_CHECKARG_POINTER(7, y);

    (void)alpha_stride;

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    ROCSPARSE_CHECKARG(
        7, y, (y->batch_count > 1 || A->batch_count > 1), rocsparse_status_not_implemented);

    rocsparse_mat_descr descr = A->descr;

    if(ellsv_info == nullptr || ellsv_info->get(trans, descr->fill_mode) == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_internal_error,
            "ellsv row map is not available, it looks like the analysis phase of this "
            "algorithm was not previously executed.");
    }

    rocsparse_ellsv_info ei = ellsv_info;

    rocsparse::trm_info_t* trm_info = ei->get(trans, descr->fill_mode);

    // For a transposed solve the analysis phase materialized A^T and described it
    // with the opposite fill mode, so the solve simply runs against that matrix.
    rocsparse_const_spmat_descr solve_matrix
        = (trans == rocsparse_operation_none) ? A : trm_info->get_transposed_matrix();

    if(solve_matrix == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_internal_error,
            "ellsv transpose is not available, it looks like the analysis phase of this "
            "algorithm was not previously executed.");
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_init_zero_pivot(handle, ei, A));

    const bool is_host_mode = (handle->pointer_mode == rocsparse_pointer_mode_host);

    bool     sleep  = false;
    uint32_t wfsize = 0;
    rocsparse::ellsv_select_launch(handle, &sleep, &wfsize);

    rocsparse::ellsv_solve_kernel_t launch = rocsparse::find_ellsv_solve_kernel(
        wfsize, sleep, solve_matrix->col_type, solve_matrix->data_type);

    if(launch == nullptr)
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }

    RETURN_IF_ROCSPARSE_ERROR(launch(handle,
                                     solve_matrix,
                                     alpha,
                                     x,
                                     y,
                                     trm_info->get_row_map(),
                                     ei->get_singularity_numeric_exact()->get_position(),
                                     temp_buffer,
                                     buffer_size,
                                     is_host_mode));

    return rocsparse_status_success;
}
