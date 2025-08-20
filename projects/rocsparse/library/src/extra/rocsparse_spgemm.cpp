/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_spgemm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_csrgemm_numeric.hpp"
#include "rocsparse_csrgemm_symbolic.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spgemm_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spgemm_alg_default);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spgemm_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spgemm_stage_buffer_size);
        CASE(rocsparse_spgemm_stage_nnz);
        CASE(rocsparse_spgemm_stage_compute);
        CASE(rocsparse_spgemm_stage_symbolic);
        CASE(rocsparse_spgemm_stage_numeric);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spgemm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spgemm_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spgemm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spgemm_stage_buffer_size:
    case rocsparse_spgemm_stage_nnz:
    case rocsparse_spgemm_stage_compute:
    case rocsparse_spgemm_stage_symbolic:
    case rocsparse_spgemm_stage_numeric:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    static rocsparse_status spgemm_template(rocsparse_handle            handle,
                                            rocsparse_operation         trans_A,
                                            rocsparse_operation         trans_B,
                                            const void*                 alpha,
                                            rocsparse_const_spmat_descr A,
                                            rocsparse_const_spmat_descr B,
                                            const void*                 beta,
                                            rocsparse_const_spmat_descr D,
                                            rocsparse_spmat_descr       C,
                                            rocsparse_spgemm_alg        alg,
                                            rocsparse_spgemm_stage      stage,
                                            size_t*                     buffer_size,
                                            void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_format format_A = A->format;
        switch(stage)
        {
        case rocsparse_spgemm_stage_buffer_size:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgemm_buffer_size_template(handle,
                                                            trans_A,
                                                            trans_B,
                                                            (J)A->rows,
                                                            (J)B->cols,
                                                            (J)A->cols,
                                                            (const T*)alpha,
                                                            A->descr,
                                                            (I)A->nnz,
                                                            (const I*)A->const_row_data,
                                                            (const J*)A->const_col_data,
                                                            B->descr,
                                                            (I)B->nnz,
                                                            (const I*)B->const_row_data,
                                                            (const J*)B->const_col_data,
                                                            (const T*)beta,
                                                            D->descr,
                                                            (I)D->nnz,
                                                            (const I*)D->const_row_data,
                                                            (const J*)D->const_col_data,
                                                            C->info,
                                                            buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::bsrgemm_buffer_size_template(handle,
                                                            A->block_dir,
                                                            trans_A,
                                                            trans_B,
                                                            (J)A->rows,
                                                            (J)B->cols,
                                                            (J)A->cols,
                                                            (J)A->block_dim,
                                                            (const T*)alpha,
                                                            A->descr,
                                                            (I)A->nnz,
                                                            (const I*)A->const_row_data,
                                                            (const J*)A->const_col_data,
                                                            B->descr,
                                                            (I)B->nnz,
                                                            (const I*)B->const_row_data,
                                                            (const J*)B->const_col_data,
                                                            (const T*)beta,
                                                            D->descr,
                                                            (I)D->nnz,
                                                            (const I*)D->const_row_data,
                                                            (const J*)D->const_col_data,
                                                            C->info,
                                                            buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spgemm_stage_nnz:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                I nnz_C;
                // non-zeros of C need to be on host
                rocsparse_pointer_mode ptr_mode;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode(handle, &ptr_mode));
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                const rocsparse_status status
                    = rocsparse::csrgemm_nnz_template(handle,
                                                      trans_A,
                                                      trans_B,
                                                      (J)A->rows,
                                                      (J)B->cols,
                                                      (J)A->cols,
                                                      A->descr,
                                                      (I)A->nnz,
                                                      (const I*)A->const_row_data,
                                                      (const J*)A->const_col_data,
                                                      B->descr,
                                                      (I)B->nnz,
                                                      (const I*)B->const_row_data,
                                                      (const J*)B->const_col_data,
                                                      D->descr,
                                                      (I)D->nnz,
                                                      (const I*)D->const_row_data,
                                                      (const J*)D->const_col_data,
                                                      C->descr,
                                                      (I*)C->row_data,
                                                      &nnz_C,
                                                      C->info,
                                                      temp_buffer);

                RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, ptr_mode));
                RETURN_IF_ROCSPARSE_ERROR(status);
                C->nnz = nnz_C;

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            {
                I nnzb_C;
                // non-zeros blocks of C need to be on host
                rocsparse_pointer_mode ptr_mode;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode(handle, &ptr_mode));
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                const rocsparse_status status
                    = rocsparse::bsrgemm_nnzb_template(handle,
                                                       A->block_dir,
                                                       trans_A,
                                                       trans_B,
                                                       (J)A->rows,
                                                       (J)B->cols,
                                                       (J)A->cols,
                                                       (J)A->block_dim,
                                                       A->descr,
                                                       (I)A->nnz,
                                                       (const I*)A->const_row_data,
                                                       (const J*)A->const_col_data,
                                                       B->descr,
                                                       (I)B->nnz,
                                                       (const I*)B->const_row_data,
                                                       (const J*)B->const_col_data,
                                                       D->descr,
                                                       (I)D->nnz,
                                                       (const I*)D->const_row_data,
                                                       (const J*)D->const_col_data,
                                                       C->descr,
                                                       (I*)C->row_data,
                                                       &nnzb_C,
                                                       C->info,
                                                       temp_buffer);

                RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, ptr_mode));
                RETURN_IF_ROCSPARSE_ERROR(status);
                C->nnz = nnzb_C;

                return rocsparse_status_success;
            }
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spgemm_stage_compute:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                // CSR format
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_template(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      (J)A->rows,
                                                                      (J)B->cols,
                                                                      (J)A->cols,
                                                                      (const T*)alpha,
                                                                      A->descr,
                                                                      (I)A->nnz,
                                                                      (const T*)A->const_val_data,
                                                                      (const I*)A->const_row_data,
                                                                      (const J*)A->const_col_data,
                                                                      B->descr,
                                                                      (I)B->nnz,
                                                                      (const T*)B->const_val_data,
                                                                      (const I*)B->const_row_data,
                                                                      (const J*)B->const_col_data,
                                                                      (const T*)beta,
                                                                      D->descr,
                                                                      (I)D->nnz,
                                                                      (const T*)D->const_val_data,
                                                                      (const I*)D->const_row_data,
                                                                      (const J*)D->const_col_data,
                                                                      C->descr,
                                                                      (T*)C->val_data,
                                                                      (const I*)C->const_row_data,
                                                                      (J*)C->col_data,
                                                                      C->info,
                                                                      temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_template(handle,
                                                                      A->block_dir,
                                                                      trans_A,
                                                                      trans_B,
                                                                      (J)A->rows,
                                                                      (J)B->cols,
                                                                      (J)A->cols,
                                                                      (J)A->block_dim,
                                                                      (const T*)alpha,
                                                                      A->descr,
                                                                      (I)A->nnz,
                                                                      (const T*)A->const_val_data,
                                                                      (const I*)A->const_row_data,
                                                                      (const J*)A->const_col_data,
                                                                      B->descr,
                                                                      (I)B->nnz,
                                                                      (const T*)B->const_val_data,
                                                                      (const I*)B->const_row_data,
                                                                      (const J*)B->const_col_data,
                                                                      (const T*)beta,
                                                                      D->descr,
                                                                      (I)D->nnz,
                                                                      (const T*)D->const_val_data,
                                                                      (const I*)D->const_row_data,
                                                                      (const J*)D->const_col_data,
                                                                      C->descr,
                                                                      (T*)C->val_data,
                                                                      (const I*)C->const_row_data,
                                                                      (J*)C->col_data,
                                                                      C->info,
                                                                      temp_buffer));
                return rocsparse_status_success;
            }
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spgemm_stage_symbolic:
        {
            switch(format_A)
            {
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_bsr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgemm_symbolic_template(handle,
                                                         trans_A,
                                                         trans_B,
                                                         (J)A->rows,
                                                         (J)B->cols,
                                                         (J)A->cols,
                                                         A->descr,
                                                         (I)A->nnz,
                                                         (const I*)A->const_row_data,
                                                         (const J*)A->const_col_data,
                                                         B->descr,
                                                         (I)B->nnz,
                                                         (const I*)B->const_row_data,
                                                         (const J*)B->const_col_data,
                                                         D->descr,
                                                         (I)D->nnz,
                                                         (const I*)D->const_row_data,
                                                         (const J*)D->const_col_data,
                                                         C->descr,
                                                         (I)C->nnz,
                                                         (const I*)C->const_row_data,
                                                         (J*)C->col_data,
                                                         C->info,
                                                         temp_buffer));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_spgemm_stage_numeric:
        {
            switch(format_A)
            {
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_bsr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgemm_numeric_template(handle,
                                                        trans_A,
                                                        trans_B,
                                                        (J)A->rows,
                                                        (J)B->cols,
                                                        (J)A->cols,
                                                        (const T*)alpha,
                                                        A->descr,
                                                        (I)A->nnz,
                                                        (const T*)A->const_val_data,
                                                        (const I*)A->const_row_data,
                                                        (const J*)A->const_col_data,
                                                        B->descr,
                                                        (I)B->nnz,
                                                        (const T*)B->const_val_data,
                                                        (const I*)B->const_row_data,
                                                        (const J*)B->const_col_data,
                                                        (const T*)beta,
                                                        D->descr,
                                                        (I)D->nnz,
                                                        (const T*)D->const_val_data,
                                                        (const I*)D->const_row_data,
                                                        (const J*)D->const_col_data,
                                                        C->descr,
                                                        (I)C->nnz,
                                                        (T*)C->val_data,
                                                        (const I*)C->const_row_data,
                                                        (const J*)C->const_col_data,
                                                        C->info,
                                                        temp_buffer));
                return rocsparse_status_success;
            }
            }
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    template <typename... Ts>
    static rocsparse_status spgemm_template_dispatch(rocsparse_indextype itype,
                                                     rocsparse_indextype jtype,
                                                     rocsparse_datatype  ctype,
                                                     Ts&&... params)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(itype)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            switch(jtype)
            {
            case rocsparse_indextype_i64:
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_indextype_i32:
            {
                switch(ctype)
                {
                case rocsparse_datatype_f32_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int32_t, int32_t, float>(params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f64_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int32_t, int32_t, double>(params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f32_c:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int32_t, int32_t, rocsparse_float_complex>(
                            params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f64_c:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int32_t, int32_t, rocsparse_double_complex>(
                            params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_i8_r:
                case rocsparse_datatype_u8_r:
                case rocsparse_datatype_i32_r:
                case rocsparse_datatype_u32_r:
                case rocsparse_datatype_f16_r:
                case rocsparse_datatype_bf16_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                }
                }
            }
            }
        }
        case rocsparse_indextype_i64:
        {
            switch(jtype)
            {
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_indextype_i32:
            {
                switch(ctype)
                {
                case rocsparse_datatype_f32_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int32_t, float>(params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f64_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int32_t, double>(params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f32_c:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int32_t, rocsparse_float_complex>(
                            params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f64_c:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int32_t, rocsparse_double_complex>(
                            params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_i8_r:
                case rocsparse_datatype_u8_r:
                case rocsparse_datatype_i32_r:
                case rocsparse_datatype_u32_r:
                case rocsparse_datatype_f16_r:
                case rocsparse_datatype_bf16_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                }
                }
            }
            case rocsparse_indextype_i64:
            {
                switch(ctype)
                {
                case rocsparse_datatype_f32_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int64_t, float>(params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f64_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int64_t, double>(params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f32_c:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int64_t, rocsparse_float_complex>(
                            params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_f64_c:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spgemm_template<int64_t, int64_t, rocsparse_double_complex>(
                            params...)));
                    return rocsparse_status_success;
                }
                case rocsparse_datatype_i8_r:
                case rocsparse_datatype_u8_r:
                case rocsparse_datatype_i32_r:
                case rocsparse_datatype_u32_r:
                case rocsparse_datatype_f16_r:
                case rocsparse_datatype_bf16_r:
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                }
                }
            }
            }
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    static rocsparse_status spgemm_checkarg(rocsparse_handle            handle, //0
                                            rocsparse_operation         trans_A, //1
                                            rocsparse_operation         trans_B, //2
                                            const void*                 alpha, //3
                                            rocsparse_const_spmat_descr A, //4
                                            rocsparse_const_spmat_descr B, //5
                                            const void*                 beta, //6
                                            rocsparse_const_spmat_descr D, //7
                                            rocsparse_spmat_descr       C, //8
                                            rocsparse_datatype          compute_type, //9
                                            rocsparse_spgemm_alg        alg, //10
                                            rocsparse_spgemm_stage      stage, //11
                                            size_t*                     buffer_size, //12
                                            void*                       temp_buffer) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);

        ROCSPARSE_CHECKARG_POINTER(4, A);
        ROCSPARSE_CHECKARG_POINTER(5, B);
        ROCSPARSE_CHECKARG_POINTER(7, D);
        ROCSPARSE_CHECKARG_POINTER(8, C);
        ROCSPARSE_CHECKARG_ENUM(9, compute_type);
        ROCSPARSE_CHECKARG_ENUM(10, alg);
        ROCSPARSE_CHECKARG_ENUM(11, stage);

        switch(stage)
        {
        case rocsparse_spgemm_stage_buffer_size:
        {
            ROCSPARSE_CHECKARG_POINTER(12, buffer_size);
            break;
        }
        case rocsparse_spgemm_stage_nnz:
        case rocsparse_spgemm_stage_compute:
        case rocsparse_spgemm_stage_symbolic:
        case rocsparse_spgemm_stage_numeric:
        {
            break;
        }
        }

        //
        //    if(alpha == nullptr && beta == nullptr)
        //    {
        //        return rocsparse_status_invalid_pointer;
        //    }
        //

        ROCSPARSE_CHECKARG(4, A, (A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(5, B, (B->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(7, D, (D->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(8, C, (C->init == false), rocsparse_status_not_initialized);

        ROCSPARSE_CHECKARG(5, B, (B->format != A->format), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7, D, (D->format != A->format), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(8, C, (C->format != A->format), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(4, A, (A->data_type != compute_type), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(5, B, (B->data_type != compute_type), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7, D, (D->data_type != compute_type), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(8, C, (C->data_type != compute_type), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(5, B, (B->row_type != A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(7, D, (D->row_type != A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(8, C, (C->row_type != A->row_type), rocsparse_status_type_mismatch);

        ROCSPARSE_CHECKARG(5, B, (B->col_type != A->col_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(7, D, (D->col_type != A->col_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(8, C, (C->col_type != A->col_type), rocsparse_status_type_mismatch);

        return rocsparse_status_continue;
    }
}

extern "C" rocsparse_status rocsparse_spgemm(rocsparse_handle            handle,
                                             rocsparse_operation         trans_A,
                                             rocsparse_operation         trans_B,
                                             const void*                 alpha,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_spmat_descr B,
                                             const void*                 beta,
                                             rocsparse_const_spmat_descr D,
                                             rocsparse_spmat_descr       C,
                                             rocsparse_datatype          compute_type,
                                             rocsparse_spgemm_alg        alg,
                                             rocsparse_spgemm_stage      stage,
                                             size_t*                     buffer_size,
                                             void*                       temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace("rocsparse_spgemm",
                         handle,
                         trans_A,
                         trans_B,
                         alpha,
                         A,
                         B,
                         beta,
                         D,
                         C,
                         compute_type,
                         alg,
                         stage,
                         buffer_size,
                         temp_buffer);

    const rocsparse_status status = rocsparse::spgemm_checkarg(handle,
                                                               trans_A,
                                                               trans_B,
                                                               alpha,
                                                               A,
                                                               B,
                                                               beta,
                                                               D,
                                                               C,
                                                               compute_type,
                                                               alg,
                                                               stage,
                                                               buffer_size,
                                                               temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spgemm_template_dispatch(A->row_type,
                                                                  A->col_type,
                                                                  compute_type,

                                                                  handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  alpha,
                                                                  A,
                                                                  B,
                                                                  beta,
                                                                  D,
                                                                  C,
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
