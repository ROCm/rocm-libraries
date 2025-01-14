/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_spsv.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_coosv.hpp"
#include "rocsparse_csrsv.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status spsv_template(rocsparse_handle            handle,
                                   rocsparse_operation         trans,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr mat,
                                   rocsparse_const_dnvec_descr x,
                                   rocsparse_dnvec_descr       y,
                                   rocsparse_spsv_alg          alg,
                                   rocsparse_spsv_stage        stage,
                                   size_t*                     buffer_size,
                                   void*                       temp_buffer)
    {
        // STAGE 1 - compute required buffer size of temp_buffer
        if(stage == rocsparse_spsv_stage_buffer_size)
        {
            if(mat->format == rocsparse_format_csr)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrsv_buffer_size_template(handle,
                                                          trans,
                                                          (J)mat->rows,
                                                          (I)mat->nnz,
                                                          mat->descr,
                                                          (const T*)mat->const_val_data,
                                                          (const I*)mat->const_row_data,
                                                          (const J*)mat->const_col_data,
                                                          mat->info,
                                                          buffer_size));

                *buffer_size = rocsparse::max(static_cast<size_t>(4), *buffer_size);
                return rocsparse_status_success;
            }
            else if(mat->format == rocsparse_format_coo)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coosv_buffer_size_template(handle,
                                                          trans,
                                                          (I)mat->rows,
                                                          mat->nnz,
                                                          mat->descr,
                                                          (const T*)mat->const_val_data,
                                                          (const I*)mat->const_row_data,
                                                          (const I*)mat->const_col_data,
                                                          mat->info,
                                                          buffer_size));

                *buffer_size = rocsparse::max(static_cast<size_t>(4), *buffer_size);
                return rocsparse_status_success;
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        // STAGE 2 - preprocess stage
        if(stage == rocsparse_spsv_stage_preprocess)
        {
            if(mat->analysed == false)
            {
                if(mat->format == rocsparse_format_csr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::csrsv_analysis_template(handle,
                                                            trans,
                                                            (J)mat->rows,
                                                            (I)mat->nnz,
                                                            mat->descr,
                                                            (const T*)mat->const_val_data,
                                                            (const I*)mat->const_row_data,
                                                            (const J*)mat->const_col_data,
                                                            mat->info,
                                                            rocsparse_analysis_policy_force,
                                                            rocsparse_solve_policy_auto,
                                                            temp_buffer)));
                }
                else if(mat->format == rocsparse_format_coo)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coosv_analysis_template(handle,
                                                            trans,
                                                            (I)mat->rows,
                                                            mat->nnz,
                                                            mat->descr,
                                                            (const T*)mat->const_val_data,
                                                            (const I*)mat->const_row_data,
                                                            (const I*)mat->const_col_data,
                                                            mat->info,
                                                            rocsparse_analysis_policy_force,
                                                            rocsparse_solve_policy_auto,
                                                            temp_buffer)));
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                }

                mat->analysed = true;
            }

            return rocsparse_status_success;
        }

        // STAGE 3 - perform SpSV computation
        if(stage == rocsparse_spsv_stage_compute)
        {
            if(mat->format == rocsparse_format_csr)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrsv_solve_template(handle,
                                                    trans,
                                                    (J)mat->rows,
                                                    (I)mat->nnz,
                                                    (const T*)alpha,
                                                    mat->descr,
                                                    (const T*)mat->const_val_data,
                                                    (const I*)mat->const_row_data,
                                                    (const J*)mat->const_col_data,
                                                    mat->info,
                                                    (const T*)x->const_values,
                                                    (int64_t)1,
                                                    (T*)y->values,
                                                    rocsparse_solve_policy_auto,
                                                    temp_buffer));
                return rocsparse_status_success;
            }
            else if(mat->format == rocsparse_format_coo)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coosv_solve_template(handle,
                                                    trans,
                                                    (I)mat->rows,
                                                    mat->nnz,
                                                    (const T*)alpha,
                                                    mat->descr,
                                                    (const T*)mat->const_val_data,
                                                    (const I*)mat->const_row_data,
                                                    (const I*)mat->const_col_data,
                                                    mat->info,
                                                    (const T*)x->const_values,
                                                    (T*)y->values,
                                                    rocsparse_solve_policy_auto,
                                                    temp_buffer));
                return rocsparse_status_success;
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    template <typename... Ts>
    rocsparse_status spsv_dynamic_dispatch(rocsparse_indextype itype,
                                           rocsparse_indextype jtype,
                                           rocsparse_datatype  ctype,
                                           Ts&&... ts)
    {
        switch(ctype)
        {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                            \
    case ENUMVAL:                                                               \
    {                                                                           \
        switch(itype)                                                           \
        {                                                                       \
        case rocsparse_indextype_u16:                                           \
        {                                                                       \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);        \
        }                                                                       \
        case rocsparse_indextype_i32:                                           \
        {                                                                       \
            switch(jtype)                                                       \
            {                                                                   \
            case rocsparse_indextype_u16:                                       \
            case rocsparse_indextype_i64:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);    \
            }                                                                   \
            case rocsparse_indextype_i32:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(                                      \
                    (rocsparse::spsv_template<int32_t, int32_t, TYPE>(ts...))); \
                return rocsparse_status_success;                                \
            }                                                                   \
            }                                                                   \
        }                                                                       \
        case rocsparse_indextype_i64:                                           \
        {                                                                       \
            switch(jtype)                                                       \
            {                                                                   \
            case rocsparse_indextype_u16:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);    \
            }                                                                   \
            case rocsparse_indextype_i32:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(                                      \
                    (rocsparse::spsv_template<int64_t, int32_t, TYPE>(ts...))); \
                return rocsparse_status_success;                                \
            }                                                                   \
            case rocsparse_indextype_i64:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(                                      \
                    (rocsparse::spsv_template<int64_t, int64_t, TYPE>(ts...))); \
                return rocsparse_status_success;                                \
            }                                                                   \
            }                                                                   \
        }                                                                       \
        }                                                                       \
    }

            DATATYPE_CASE(rocsparse_datatype_f32_r, float);
            DATATYPE_CASE(rocsparse_datatype_f64_r, double);
            DATATYPE_CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
            DATATYPE_CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);
            //DATATYPE_CASE(rocsparse_datatype_i8_r, int8_t);
            //DATATYPE_CASE(rocsparse_datatype_u8_r, uint8_t);
            //DATATYPE_CASE(rocsparse_datatype_i32_r, int32_t);
            //DATATYPE_CASE(rocsparse_datatype_u32_r, uint32_t);

        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#undef DATATYPE_CASE
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spsv(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans, //1
                                           const void*                 alpha, //2
                                           rocsparse_const_spmat_descr mat, //3
                                           rocsparse_const_dnvec_descr x, //4
                                           const rocsparse_dnvec_descr y, //5
                                           rocsparse_datatype          compute_type, //6
                                           rocsparse_spsv_alg          alg, //7
                                           rocsparse_spsv_stage        stage, //8
                                           size_t*                     buffer_size, //9
                                           void*                       temp_buffer) // 10
try
{
    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spsv",
                         trans,
                         (const void*&)alpha,
                         (const void*&)mat,
                         (const void*&)x,
                         (const void*&)y,
                         compute_type,
                         alg,
                         stage,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, alpha);
    ROCSPARSE_CHECKARG_POINTER(3, mat);
    ROCSPARSE_CHECKARG_POINTER(4, x);
    ROCSPARSE_CHECKARG_POINTER(5, y);
    ROCSPARSE_CHECKARG_ENUM(6, compute_type);
    ROCSPARSE_CHECKARG_ENUM(7, alg);
    ROCSPARSE_CHECKARG_ENUM(8, stage);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        ROCSPARSE_CHECKARG_POINTER(9, buffer_size);
    }

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    ROCSPARSE_CHECKARG(3, mat, (mat->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, x, (x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(5, y, (y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(3, mat, (mat->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4, x, (x->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(5, y, (y->data_type != compute_type), rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsv_dynamic_dispatch(mat->row_type,
                                                               mat->col_type,
                                                               compute_type,
                                                               handle,
                                                               trans,
                                                               alpha,
                                                               mat,
                                                               x,
                                                               y,
                                                               alg,
                                                               stage,
                                                               buffer_size,
                                                               temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
