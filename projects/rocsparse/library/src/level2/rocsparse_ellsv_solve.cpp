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

    template <uint32_t WF_SIZE, bool SLEEP, typename I, typename T>
    static rocsparse_status ellsv_launch_solve(rocsparse_handle     handle,
                                               I                    m,
                                               I                    n,
                                               int64_t              ell_width,
                                               const void*          alpha,
                                               const void*          ell_col_ind,
                                               const void*          ell_val,
                                               const void*          x,
                                               int64_t              x_inc,
                                               void*                y,
                                               int64_t              y_inc,
                                               const void*          row_map,
                                               void*                zero_pivot,
                                               rocsparse_index_base base,
                                               rocsparse_fill_mode  fill_mode,
                                               rocsparse_diag_type  diag_type,
                                               void*                temp_buffer,
                                               bool                 is_host_mode)
    {
        constexpr uint32_t BLOCKSIZE = 1024;

        hipStream_t stream = handle->stream;

        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;

        const size_t done_bytes = ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));
        int32_t*     done_array = reinterpret_cast<int32_t*>(ptr);

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
            n,
            ell_width,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_),
            reinterpret_cast<const I*>(ell_col_ind),
            reinterpret_cast<const T*>(ell_val),
            reinterpret_cast<const T*>(x),
            x_inc,
            reinterpret_cast<T*>(y),
            y_inc,
            done_array,
            reinterpret_cast<const I*>(row_map),
            reinterpret_cast<I*>(zero_pivot),
            base,
            fill_mode,
            diag_type,
            is_host_mode);

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    static rocsparse_status ellsv_solve_dispatch(rocsparse_handle     handle,
                                                 bool                 sleep,
                                                 uint32_t             wfsize,
                                                 I                    m,
                                                 I                    n,
                                                 int64_t              ell_width,
                                                 const void*          alpha,
                                                 const void*          ell_col_ind,
                                                 const void*          ell_val,
                                                 const void*          x,
                                                 int64_t              x_inc,
                                                 void*                y,
                                                 int64_t              y_inc,
                                                 const void*          row_map,
                                                 void*                zero_pivot,
                                                 rocsparse_index_base base,
                                                 rocsparse_fill_mode  fill_mode,
                                                 rocsparse_diag_type  diag_type,
                                                 void*                temp_buffer,
                                                 bool                 is_host_mode)
    {
        if(sleep)
        {
            return rocsparse::ellsv_launch_solve<64, true, I, T>(handle,
                                                                 m,
                                                                 n,
                                                                 ell_width,
                                                                 alpha,
                                                                 ell_col_ind,
                                                                 ell_val,
                                                                 x,
                                                                 x_inc,
                                                                 y,
                                                                 y_inc,
                                                                 row_map,
                                                                 zero_pivot,
                                                                 base,
                                                                 fill_mode,
                                                                 diag_type,
                                                                 temp_buffer,
                                                                 is_host_mode);
        }
        else if(wfsize == 64)
        {
            return rocsparse::ellsv_launch_solve<64, false, I, T>(handle,
                                                                  m,
                                                                  n,
                                                                  ell_width,
                                                                  alpha,
                                                                  ell_col_ind,
                                                                  ell_val,
                                                                  x,
                                                                  x_inc,
                                                                  y,
                                                                  y_inc,
                                                                  row_map,
                                                                  zero_pivot,
                                                                  base,
                                                                  fill_mode,
                                                                  diag_type,
                                                                  temp_buffer,
                                                                  is_host_mode);
        }

        return rocsparse::ellsv_launch_solve<32, false, I, T>(handle,
                                                              m,
                                                              n,
                                                              ell_width,
                                                              alpha,
                                                              ell_col_ind,
                                                              ell_val,
                                                              x,
                                                              x_inc,
                                                              y,
                                                              y_inc,
                                                              row_map,
                                                              zero_pivot,
                                                              base,
                                                              fill_mode,
                                                              diag_type,
                                                              temp_buffer,
                                                              is_host_mode);
    }

    template <typename I, typename T>
    static rocsparse_status ellsv_compute(rocsparse_handle            handle,
                                          rocsparse_ellsv_info        ei,
                                          bool                        sleep,
                                          uint32_t                    wfsize,
                                          rocsparse_operation         trans,
                                          rocsparse_const_spmat_descr A,
                                          const void*                 alpha,
                                          rocsparse_const_dnvec_descr x,
                                          rocsparse_dnvec_descr       y,
                                          void*                       temp_buffer,
                                          bool                        is_host_mode)
    {
        const I                   m    = static_cast<I>(A->rows);
        const I                   cols = static_cast<I>(A->cols);
        const rocsparse_fill_mode uplo = A->descr->fill_mode;
        rocsparse_fill_mode       fill = uplo;

        rocsparse::trm_info_t* trm_info = ei->get(trans, uplo);

        const void* col_ind  = A->const_col_data;
        const void* val      = A->const_val_data;
        I           n_solver = cols;
        int64_t     width    = A->ell_width;

        if(trans != rocsparse_operation_none)
        {
            fill     = rocsparse::ellsv_flip_fill(fill);
            col_ind  = trm_info->get_transposed_col_ind();
            val      = trm_info->get_transposed_val();
            n_solver = m;
            width    = trm_info->get_ell_width();
        }

        void* zero_pivot = ei->get_singularity_numeric_exact()->get_position();

        return rocsparse::ellsv_solve_dispatch<I, T>(handle,
                                                     sleep,
                                                     wfsize,
                                                     m,
                                                     n_solver,
                                                     width,
                                                     alpha,
                                                     col_ind,
                                                     val,
                                                     x->const_values,
                                                     x->inc,
                                                     y->values,
                                                     y->inc,
                                                     trm_info->get_row_map(),
                                                     zero_pivot,
                                                     A->idx_base,
                                                     fill,
                                                     A->descr->diag_type,
                                                     temp_buffer,
                                                     is_host_mode);
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
                                        void*                       temp_buffer)
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_init_zero_pivot(handle, ei, A));

    const bool is_host_mode = (handle->pointer_mode == rocsparse_pointer_mode_host);

    bool     sleep  = false;
    uint32_t wfsize = 0;
    rocsparse::ellsv_select_launch(handle, &sleep, &wfsize);

#define ELLSV_SOLVE_DISPATCH(ITYPE, TTYPE)  \
    rocsparse::ellsv_compute<ITYPE, TTYPE>( \
        handle, ei, sleep, wfsize, trans, A, alpha, x, y, temp_buffer, is_host_mode)

    switch(A->col_type)
    {
    case rocsparse_indextype_i32:
    {
        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, float));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, double));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, rocsparse_float_complex));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int32_t, rocsparse_double_complex));
            break;
        default:
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        break;
    }
    case rocsparse_indextype_i64:
    {
        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, float));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, double));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, rocsparse_float_complex));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(ELLSV_SOLVE_DISPATCH(int64_t, rocsparse_double_complex));
            break;
        default:
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        break;
    }
    case deprecated_rocsparse_indextype_u16:
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }
    }

#undef ELLSV_SOLVE_DISPATCH

    return rocsparse_status_success;
}
