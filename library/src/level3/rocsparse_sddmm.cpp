/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/generic/rocsparse_sddmm.h"
#include "common.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_sddmm.hpp"

namespace rocsparse
{
    template <rocsparse_format FORMAT, typename I, typename J, typename T, typename... Ts>
    static rocsparse_status sddmm_buffer_size_dispatch_alg(rocsparse_sddmm_alg alg, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(alg)
        {
        case rocsparse_sddmm_alg_default:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_default, I, J, T>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }
        case rocsparse_sddmm_alg_dense:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_dense, I, J, T>::
                     buffer_size_template(ts...)));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename I, typename J, typename T, typename... Ts>
    static rocsparse_status sddmm_buffer_size_dispatch_format(rocsparse_format    format,
                                                              rocsparse_sddmm_alg alg,
                                                              Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(format)
        {
        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_buffer_size_dispatch_alg<rocsparse_format_coo, I, I, T>(alg,
                                                                                          ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_buffer_size_dispatch_alg<rocsparse_format_csr, I, J, T>(alg,
                                                                                          ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_coo_aos:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_buffer_size_dispatch_alg<rocsparse_format_coo_aos, I, I, T>(
                    alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_buffer_size_dispatch_alg<rocsparse_format_csc, I, J, T>(alg,
                                                                                          ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_buffer_size_dispatch_alg<rocsparse_format_ell, I, I, T>(alg,
                                                                                          ts...)));
            return rocsparse_status_success;
        }
        case rocsparse_format_bell:
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... Ts>
    static rocsparse_status sddmm_buffer_size_dispatch(rocsparse_format    format,
                                                       rocsparse_indextype itype,
                                                       rocsparse_indextype jtype,
                                                       rocsparse_datatype  ctype,
                                                       rocsparse_sddmm_alg alg,
                                                       Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(ctype)
        {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                                       \
    case ENUMVAL:                                                                          \
    {                                                                                      \
        switch(itype)                                                                      \
        {                                                                                  \
        case rocsparse_indextype_u16:                                                      \
        {                                                                                  \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                   \
        }                                                                                  \
        case rocsparse_indextype_i32:                                                      \
        {                                                                                  \
            switch(jtype)                                                                  \
            {                                                                              \
            case rocsparse_indextype_u16:                                                  \
            case rocsparse_indextype_i64:                                                  \
            {                                                                              \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);               \
            }                                                                              \
            case rocsparse_indextype_i32:                                                  \
            {                                                                              \
                RETURN_IF_ROCSPARSE_ERROR(                                                 \
                    (rocsparse::sddmm_buffer_size_dispatch_format<int32_t, int32_t, TYPE>( \
                        format, alg, ts...)));                                             \
                return rocsparse_status_success;                                           \
            }                                                                              \
            }                                                                              \
        }                                                                                  \
        case rocsparse_indextype_i64:                                                      \
        {                                                                                  \
            switch(jtype)                                                                  \
            {                                                                              \
            case rocsparse_indextype_u16:                                                  \
            {                                                                              \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);               \
            }                                                                              \
            case rocsparse_indextype_i32:                                                  \
            {                                                                              \
                RETURN_IF_ROCSPARSE_ERROR(                                                 \
                    (rocsparse::sddmm_buffer_size_dispatch_format<int64_t, int32_t, TYPE>( \
                        format, alg, ts...)));                                             \
                return rocsparse_status_success;                                           \
            }                                                                              \
            case rocsparse_indextype_i64:                                                  \
            {                                                                              \
                RETURN_IF_ROCSPARSE_ERROR(                                                 \
                    (rocsparse::sddmm_buffer_size_dispatch_format<int64_t, int64_t, TYPE>( \
                        format, alg, ts...)));                                             \
                return rocsparse_status_success;                                           \
            }                                                                              \
            }                                                                              \
        }                                                                                  \
        }                                                                                  \
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
        // LCOV_EXCL_START
#undef DATATYPE_CASE
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

extern "C" rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle            handle,
                                                        rocsparse_operation         trans_A,
                                                        rocsparse_operation         trans_B,
                                                        const void*                 alpha,
                                                        rocsparse_const_dnmat_descr A,
                                                        rocsparse_const_dnmat_descr B,
                                                        const void*                 beta,
                                                        const rocsparse_spmat_descr C,
                                                        rocsparse_datatype          compute_type,
                                                        rocsparse_sddmm_alg         alg,
                                                        size_t*                     buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_sddmm_buffer_size",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)A,
                         (const void*&)B,
                         (const void*&)beta,
                         (const void*&)C,
                         compute_type,
                         alg,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, A);
    ROCSPARSE_CHECKARG(4, A, A->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(5, B);
    ROCSPARSE_CHECKARG(5, B, B->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG_POINTER(7, C);
    ROCSPARSE_CHECKARG(7, C, C->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(8, compute_type);

    ROCSPARSE_CHECKARG(8,
                       compute_type,
                       (compute_type != A->data_type || compute_type != B->data_type
                        || compute_type != C->data_type),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG_ENUM(9, alg);
    ROCSPARSE_CHECKARG_POINTER(10, buffer_size);

    ROCSPARSE_CHECKARG(1,
                       trans_A,
                       (trans_A == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       trans_B,
                       (trans_B == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sddmm_buffer_size_dispatch(
        C->format,
        (C->format == rocsparse_format_csc) ? C->col_type : C->row_type,
        (C->format == rocsparse_format_csc) ? C->row_type : C->col_type,
        compute_type,
        alg,
        //
        handle,
        trans_A,
        trans_B,
        alpha,
        A,
        B,
        beta,
        C,
        compute_type,
        alg,
        buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    template <rocsparse_format FORMAT, typename I, typename J, typename T, typename... Ts>
    static rocsparse_status sddmm_preprocess_dispatch_alg(rocsparse_sddmm_alg alg, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(alg)
        {
        case rocsparse_sddmm_alg_default:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_default, I, J, T>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }
        case rocsparse_sddmm_alg_dense:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_dense, I, J, T>::
                     preprocess_template(ts...)));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename I, typename J, typename T, typename... Ts>
    static rocsparse_status sddmm_preprocess_dispatch_format(rocsparse_format    format,
                                                             rocsparse_sddmm_alg alg,
                                                             Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(format)
        {
        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_preprocess_dispatch_alg<rocsparse_format_coo, I, I, T>(alg,
                                                                                         ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_preprocess_dispatch_alg<rocsparse_format_csr, I, J, T>(alg,
                                                                                         ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_coo_aos:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_preprocess_dispatch_alg<rocsparse_format_coo_aos, I, I, T>(
                    alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_preprocess_dispatch_alg<rocsparse_format_csc, I, J, T>(alg,
                                                                                         ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_preprocess_dispatch_alg<rocsparse_format_ell, I, I, T>(alg,
                                                                                         ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... Ts>
    static rocsparse_status sddmm_preprocess_dispatch(rocsparse_format    format,
                                                      rocsparse_indextype itype,
                                                      rocsparse_indextype jtype,
                                                      rocsparse_datatype  ctype,
                                                      rocsparse_sddmm_alg alg,
                                                      Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(ctype)
        {
#define DATATYPE_CASE(ENUMVAL, TYPE)                                                      \
    case ENUMVAL:                                                                         \
    {                                                                                     \
        switch(itype)                                                                     \
        {                                                                                 \
        case rocsparse_indextype_u16:                                                     \
        {                                                                                 \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                  \
        }                                                                                 \
        case rocsparse_indextype_i32:                                                     \
        {                                                                                 \
            switch(jtype)                                                                 \
            {                                                                             \
            case rocsparse_indextype_u16:                                                 \
            case rocsparse_indextype_i64:                                                 \
            {                                                                             \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);              \
            }                                                                             \
            case rocsparse_indextype_i32:                                                 \
            {                                                                             \
                RETURN_IF_ROCSPARSE_ERROR(                                                \
                    (rocsparse::sddmm_preprocess_dispatch_format<int32_t, int32_t, TYPE>( \
                        format, alg, ts...)));                                            \
                return rocsparse_status_success;                                          \
            }                                                                             \
            }                                                                             \
        }                                                                                 \
        case rocsparse_indextype_i64:                                                     \
        {                                                                                 \
            switch(jtype)                                                                 \
            {                                                                             \
            case rocsparse_indextype_u16:                                                 \
            {                                                                             \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);              \
            }                                                                             \
            case rocsparse_indextype_i32:                                                 \
            {                                                                             \
                RETURN_IF_ROCSPARSE_ERROR(                                                \
                    (rocsparse::sddmm_preprocess_dispatch_format<int64_t, int32_t, TYPE>( \
                        format, alg, ts...)));                                            \
                return rocsparse_status_success;                                          \
            }                                                                             \
            case rocsparse_indextype_i64:                                                 \
            {                                                                             \
                RETURN_IF_ROCSPARSE_ERROR(                                                \
                    (rocsparse::sddmm_preprocess_dispatch_format<int64_t, int64_t, TYPE>( \
                        format, alg, ts...)));                                            \
                return rocsparse_status_success;                                          \
            }                                                                             \
            }                                                                             \
        }                                                                                 \
        }                                                                                 \
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
        // LCOV_EXCL_START
#undef DATATYPE_CASE
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

extern "C" rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle            handle, //0
                                                       rocsparse_operation         trans_A, //1
                                                       rocsparse_operation         trans_B, //2
                                                       const void*                 alpha, //3
                                                       rocsparse_const_dnmat_descr A, //4
                                                       rocsparse_const_dnmat_descr B, //5
                                                       const void*                 beta, //6
                                                       const rocsparse_spmat_descr C, //7
                                                       rocsparse_datatype          compute_type, //8
                                                       rocsparse_sddmm_alg         alg, //9
                                                       void*                       temp_buffer) //10
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         "rocsparse_sddmm_preprocess",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)A,
                         (const void*&)B,
                         (const void*&)beta,
                         (const void*&)C,
                         compute_type,
                         alg,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, A);
    ROCSPARSE_CHECKARG(4, A, A->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(5, B);
    ROCSPARSE_CHECKARG(5, B, B->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG_POINTER(7, C);
    ROCSPARSE_CHECKARG(7, C, C->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(8, compute_type);

    ROCSPARSE_CHECKARG(8,
                       compute_type,
                       (compute_type != A->data_type || compute_type != B->data_type
                        || compute_type != C->data_type),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG_ENUM(9, alg);

    ROCSPARSE_CHECKARG(1,
                       trans_A,
                       (trans_A == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       trans_B,
                       (trans_B == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);

    if(C->nnz == 0)
    {
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sddmm_preprocess_dispatch(
        C->format,
        (C->format == rocsparse_format_csc) ? C->col_type : C->row_type,
        (C->format == rocsparse_format_csc) ? C->row_type : C->col_type,
        compute_type,
        alg,
        //
        handle,
        trans_A,
        trans_B,
        alpha,
        A,
        B,
        beta,
        C,
        compute_type,
        alg,
        temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    template <rocsparse_format FORMAT, typename I, typename J, typename T, typename... Ts>
    static rocsparse_status sddmm_dispatch_alg(rocsparse_sddmm_alg alg, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(alg)
        {
        case rocsparse_sddmm_alg_default:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_default, I, J, T>::
                     compute_template(ts...)));
            return rocsparse_status_success;
        }
        case rocsparse_sddmm_alg_dense:
        {
            return rocsparse::rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_dense, I, J, T>::
                compute_template(ts...);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename I, typename J, typename T, typename... Ts>
    static rocsparse_status
        sddmm_dispatch_format(rocsparse_format format, rocsparse_sddmm_alg alg, Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(format)
        {
        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_dispatch_alg<rocsparse_format_coo, I, I, T>(alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_dispatch_alg<rocsparse_format_csr, I, J, T>(alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_coo_aos:
        {

            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_dispatch_alg<rocsparse_format_coo_aos, I, I, T>(alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_dispatch_alg<rocsparse_format_csc, I, J, T>(alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sddmm_dispatch_alg<rocsparse_format_ell, I, I, T>(alg, ts...)));
            return rocsparse_status_success;
        }

        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... Ts>
    static rocsparse_status sddmm_dispatch(rocsparse_format    format,
                                           rocsparse_indextype itype,
                                           rocsparse_indextype jtype,
                                           rocsparse_datatype  ctype,
                                           rocsparse_sddmm_alg alg,
                                           Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(ctype)
        {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                           \
    case ENUMVAL:                                                              \
    {                                                                          \
        switch(itype)                                                          \
        {                                                                      \
        case rocsparse_indextype_u16:                                          \
        {                                                                      \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);       \
        }                                                                      \
        case rocsparse_indextype_i32:                                          \
        {                                                                      \
            switch(jtype)                                                      \
            {                                                                  \
            case rocsparse_indextype_u16:                                      \
            case rocsparse_indextype_i64:                                      \
            {                                                                  \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);   \
            }                                                                  \
            case rocsparse_indextype_i32:                                      \
            {                                                                  \
                RETURN_IF_ROCSPARSE_ERROR(                                     \
                    (rocsparse::sddmm_dispatch_format<int32_t, int32_t, TYPE>( \
                        format, alg, ts...)));                                 \
                return rocsparse_status_success;                               \
            }                                                                  \
            }                                                                  \
        }                                                                      \
        case rocsparse_indextype_i64:                                          \
        {                                                                      \
            switch(jtype)                                                      \
            {                                                                  \
            case rocsparse_indextype_u16:                                      \
            {                                                                  \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);   \
            }                                                                  \
            case rocsparse_indextype_i32:                                      \
            {                                                                  \
                RETURN_IF_ROCSPARSE_ERROR(                                     \
                    (rocsparse::sddmm_dispatch_format<int64_t, int32_t, TYPE>( \
                        format, alg, ts...)));                                 \
                return rocsparse_status_success;                               \
            }                                                                  \
            case rocsparse_indextype_i64:                                      \
            {                                                                  \
                RETURN_IF_ROCSPARSE_ERROR(                                     \
                    (rocsparse::sddmm_dispatch_format<int64_t, int64_t, TYPE>( \
                        format, alg, ts...)));                                 \
                return rocsparse_status_success;                               \
            }                                                                  \
            }                                                                  \
        }                                                                      \
        }                                                                      \
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

        // LCOV_EXCL_START
#undef DATATYPE_CASE
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

extern "C" rocsparse_status rocsparse_sddmm(rocsparse_handle            handle, //0
                                            rocsparse_operation         trans_A, //1
                                            rocsparse_operation         trans_B, //2
                                            const void*                 alpha, //3
                                            rocsparse_const_dnmat_descr A, //4
                                            rocsparse_const_dnmat_descr B, //5
                                            const void*                 beta, //6
                                            const rocsparse_spmat_descr C, //7
                                            rocsparse_datatype          compute_type, //8
                                            rocsparse_sddmm_alg         alg, //9
                                            void*                       temp_buffer) //19
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_sddmm",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)A,
                         (const void*&)B,
                         (const void*&)beta,
                         (const void*&)C,
                         compute_type,
                         alg,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, A);
    ROCSPARSE_CHECKARG(4, A, A->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(5, B);
    ROCSPARSE_CHECKARG(5, B, B->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG_POINTER(7, C);
    ROCSPARSE_CHECKARG(7, C, C->init == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(8, compute_type);

    ROCSPARSE_CHECKARG(8,
                       compute_type,
                       (compute_type != A->data_type || compute_type != B->data_type
                        || compute_type != C->data_type),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG_ENUM(9, alg);

    ROCSPARSE_CHECKARG(1,
                       trans_A,
                       (trans_A == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       trans_B,
                       (trans_B == rocsparse_operation_conjugate_transpose),
                       rocsparse_status_not_implemented);

    if(C->nnz == 0)
    {
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::sddmm_dispatch(C->format,
                                  (C->format == rocsparse_format_csc) ? C->col_type : C->row_type,
                                  (C->format == rocsparse_format_csc) ? C->row_type : C->col_type,
                                  compute_type,
                                  alg,
                                  //
                                  handle,
                                  trans_A,
                                  trans_B,
                                  alpha,
                                  A,
                                  B,
                                  beta,
                                  C,
                                  compute_type,
                                  alg,
                                  temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
