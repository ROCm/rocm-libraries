/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_spitsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csritsv.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spitsv_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spitsv_alg_default);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spitsv_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spitsv_stage_buffer_size);
        CASE(rocsparse_spitsv_stage_preprocess);
        CASE(rocsparse_spitsv_stage_compute);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spitsv_alg value_)
{
    switch(value_)
    {
    case rocsparse_spitsv_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spitsv_stage value_)
{
    switch(value_)
    {
    case rocsparse_spitsv_stage_buffer_size:
    case rocsparse_spitsv_stage_preprocess:
    case rocsparse_spitsv_stage_compute:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status spitsv_template(rocsparse_handle            handle,
                                     rocsparse_int*              host_nmaxiter,
                                     const void*                 host_tol,
                                     void*                       host_history,
                                     rocsparse_operation         trans,
                                     const void*                 alpha,
                                     const rocsparse_spmat_descr mat,
                                     const rocsparse_dnvec_descr x,
                                     rocsparse_dnvec_descr       y,
                                     rocsparse_spitsv_alg        alg,
                                     rocsparse_spitsv_stage      stage,
                                     size_t*                     buffer_size,
                                     void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(mat->format != rocsparse_format_csr)
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }

        switch(stage)
        {
        case rocsparse_spitsv_stage_buffer_size:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csritsv_buffer_size_template(handle,
                                                        trans,
                                                        (J)mat->rows,
                                                        (I)mat->nnz,
                                                        mat->descr,
                                                        (const T*)mat->val_data,
                                                        (const I*)mat->row_data,
                                                        (const J*)mat->col_data,
                                                        mat->info,
                                                        buffer_size));
            return rocsparse_status_success;
        }

        case rocsparse_spitsv_stage_preprocess:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csritsv_analysis_template(handle,
                                                     trans,
                                                     (J)mat->rows,
                                                     (I)mat->nnz,
                                                     mat->descr,
                                                     (const T*)mat->val_data,
                                                     (const I*)mat->row_data,
                                                     (const J*)mat->col_data,
                                                     mat->info,
                                                     rocsparse_analysis_policy_force,
                                                     rocsparse_solve_policy_auto,
                                                     temp_buffer));
            return rocsparse_status_success;
        }

        case rocsparse_spitsv_stage_compute:
        {
            static constexpr rocsparse_int host_nfreeiter = 0;
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritsv_solve_ex_template<I, J, T>)(handle,
                                                                host_nmaxiter,
                                                                host_nfreeiter,
                                                                (const floating_data_t<T>*)host_tol,
                                                                (floating_data_t<T>*)host_history,
                                                                trans,
                                                                (J)mat->rows,
                                                                (I)mat->nnz,
                                                                (const T*)alpha,
                                                                mat->descr,
                                                                (const T*)mat->val_data,
                                                                (const I*)mat->row_data,
                                                                (const J*)mat->col_data,
                                                                mat->info,
                                                                (const T*)x->values,
                                                                (T*)y->values,
                                                                rocsparse_solve_policy_auto,
                                                                temp_buffer));
            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... Ts>
    rocsparse_status spitsv_dynamic_dispatch(rocsparse_indextype itype,
                                             rocsparse_indextype jtype,
                                             rocsparse_datatype  ctype,
                                             Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(ctype)
        {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                              \
    case ENUMVAL:                                                                 \
    {                                                                             \
        switch(itype)                                                             \
        {                                                                         \
        case rocsparse_indextype_u16:                                             \
        {                                                                         \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);          \
        }                                                                         \
        case rocsparse_indextype_i32:                                             \
        {                                                                         \
            switch(jtype)                                                         \
            {                                                                     \
            case rocsparse_indextype_u16:                                         \
            case rocsparse_indextype_i64:                                         \
            {                                                                     \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);      \
            }                                                                     \
            case rocsparse_indextype_i32:                                         \
            {                                                                     \
                RETURN_IF_ROCSPARSE_ERROR(                                        \
                    (rocsparse::spitsv_template<int32_t, int32_t, TYPE>)(ts...)); \
                return rocsparse_status_success;                                  \
            }                                                                     \
            }                                                                     \
        }                                                                         \
        case rocsparse_indextype_i64:                                             \
        {                                                                         \
            switch(jtype)                                                         \
            {                                                                     \
            case rocsparse_indextype_u16:                                         \
            {                                                                     \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);      \
            }                                                                     \
            case rocsparse_indextype_i32:                                         \
            {                                                                     \
                RETURN_IF_ROCSPARSE_ERROR(                                        \
                    (rocsparse::spitsv_template<int64_t, int32_t, TYPE>)(ts...)); \
                return rocsparse_status_success;                                  \
            }                                                                     \
            case rocsparse_indextype_i64:                                         \
            {                                                                     \
                RETURN_IF_ROCSPARSE_ERROR(                                        \
                    (rocsparse::spitsv_template<int64_t, int64_t, TYPE>)(ts...)); \
                return rocsparse_status_success;                                  \
            }                                                                     \
            }                                                                     \
        }                                                                         \
        }                                                                         \
    }

            DATATYPE_CASE(rocsparse_datatype_f32_r, float);
            DATATYPE_CASE(rocsparse_datatype_f64_r, double);
            DATATYPE_CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
            DATATYPE_CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);

#undef DATATYPE_CASE
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
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

extern "C" rocsparse_status rocsparse_spitsv(rocsparse_handle            handle, //0
                                             rocsparse_int*              host_nmaxiter, //1
                                             const void*                 host_tol, //2
                                             void*                       host_history, //3
                                             rocsparse_operation         trans, //4
                                             const void*                 alpha, //5
                                             const rocsparse_spmat_descr mat, //6
                                             const rocsparse_dnvec_descr x, //7
                                             const rocsparse_dnvec_descr y, //8
                                             rocsparse_datatype          compute_type, //9
                                             rocsparse_spitsv_alg        alg, //10
                                             rocsparse_spitsv_stage      stage, //11
                                             size_t*                     buffer_size, //12
                                             void*                       temp_buffer) // 13
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spitsv",
                         (const void*&)host_nmaxiter,
                         (const void*&)host_tol,
                         (const void*&)host_history,
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

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(6, mat);
    ROCSPARSE_CHECKARG_POINTER(7, x);
    ROCSPARSE_CHECKARG_POINTER(8, y);

    // Check for valid pointers
    ROCSPARSE_CHECKARG_POINTER(5, alpha);
    // Check for valid pointers
    ROCSPARSE_CHECKARG_POINTER(1, host_nmaxiter);

    ROCSPARSE_CHECKARG_ENUM(4, trans);
    ROCSPARSE_CHECKARG_ENUM(9, compute_type);
    ROCSPARSE_CHECKARG_ENUM(10, alg);
    ROCSPARSE_CHECKARG_ENUM(11, stage);

    if(stage == rocsparse_spitsv_stage_buffer_size)
    {
        ROCSPARSE_CHECKARG_POINTER(12, buffer_size);
    }

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    if(mat->init == false || x->init == false || y->init == false)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_initialized);
    }
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != mat->data_type || //
       compute_type != x->data_type || //
       compute_type != y->data_type)
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spitsv_dynamic_dispatch(mat->row_type,
                                                                 mat->col_type,
                                                                 compute_type,
                                                                 handle,
                                                                 host_nmaxiter,
                                                                 host_tol,
                                                                 host_history,
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
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
