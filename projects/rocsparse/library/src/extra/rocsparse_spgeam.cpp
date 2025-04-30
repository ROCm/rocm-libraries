/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <map>
#include <sstream>

#include "control.h"
#include "internal/generic/rocsparse_spgeam.h"
#include "to_string.hpp"
#include "utility.h"

#include "rocsparse_csrgeam.hpp"
#include "rocsparse_csrgeam_numeric.hpp"
#include "rocsparse_csrgeam_symbolic.hpp"

namespace rocsparse
{
    static rocsparse_status spgeam_buffer_size_template(rocsparse_handle            handle,
                                                        rocsparse_spgeam_descr      descr,
                                                        rocsparse_const_spmat_descr mat_A,
                                                        rocsparse_const_spmat_descr mat_B,
                                                        rocsparse_const_spmat_descr mat_C,
                                                        rocsparse_spgeam_stage      stage,
                                                        size_t*                     buffer_size)
    {
        const rocsparse_format format_A = mat_A->format;
        switch(stage)
        {
        case rocsparse_spgeam_stage_analysis:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_buffer_size_template(
                    handle,
                    descr->trans_A,
                    descr->trans_B,
                    mat_A->rows,
                    mat_B->cols,
                    mat_A->descr,
                    mat_A->nnz,
                    mat_A->const_row_data,
                    mat_A->const_col_data,
                    mat_B->descr,
                    mat_B->nnz,
                    mat_B->const_row_data,
                    mat_B->const_col_data,
                    mat_C != nullptr ? mat_C->const_row_data : nullptr,
                    buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
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
        case rocsparse_spgeam_stage_compute:
        case rocsparse_spgeam_stage_symbolic:
        case rocsparse_spgeam_stage_numeric:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        }
        return rocsparse_status_success;
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    static rocsparse_status spgeam_template(rocsparse_handle             handle,
                                            const rocsparse_spgeam_descr descr,
                                            const void*                  alpha,
                                            rocsparse_const_spmat_descr  mat_A,
                                            const void*                  beta,
                                            rocsparse_const_spmat_descr  mat_B,
                                            rocsparse_spmat_descr        mat_C,
                                            rocsparse_spgeam_stage       stage,
                                            size_t                       buffer_size,
                                            void*                        temp_buffer)
    {
        const rocsparse_format format_A = mat_A->format;
        switch(stage)
        {
        case rocsparse_spgeam_stage_analysis:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                if(mat_C == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::csrgeam_allocate_descr_memory_template(handle,
                                                                          mat_A->rows,
                                                                          mat_B->cols,
                                                                          alpha,
                                                                          mat_A->nnz,
                                                                          beta,
                                                                          mat_B->nnz,
                                                                          descr));
                }

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_record_descr_alpha_beta_template(
                    handle, mat_A->rows, mat_B->cols, alpha, mat_A->nnz, beta, mat_B->nnz, descr));

                I nnz_C;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_template(
                    handle,
                    descr->trans_A,
                    descr->trans_B,
                    mat_A->rows,
                    mat_B->cols,
                    mat_A->descr,
                    mat_A->nnz,
                    (const I*)mat_A->const_row_data,
                    (const J*)mat_A->const_col_data,
                    mat_B->descr,
                    mat_B->nnz,
                    (const I*)mat_B->const_row_data,
                    (const J*)mat_B->const_col_data,
                    mat_C != nullptr ? mat_C->descr : nullptr,
                    mat_C != nullptr ? (I*)mat_C->row_data : nullptr,
                    mat_C != nullptr ? &nnz_C : nullptr,
                    descr,
                    temp_buffer,
                    true));

                if(mat_C != nullptr)
                {
                    mat_C->nnz = nnz_C;
                }

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
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

        case rocsparse_spgeam_stage_compute:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgeam_copy_row_pointer_and_free_memory_template(
                        handle,
                        mat_A->rows,
                        mat_B->cols,
                        mat_C->descr,
                        (I*)mat_C->row_data,
                        &mat_C->nnz,
                        descr));

                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgeam_template(handle,
                                                descr->trans_A,
                                                descr->trans_B,
                                                mat_A->rows,
                                                mat_B->cols,
                                                (const T*)alpha,
                                                mat_A->descr,
                                                mat_A->nnz,
                                                (const A*)mat_A->const_val_data,
                                                (const I*)mat_A->const_row_data,
                                                (const J*)mat_A->const_col_data,
                                                (const T*)beta,
                                                mat_B->descr,
                                                mat_B->nnz,
                                                (const B*)mat_B->const_val_data,
                                                (const I*)mat_B->const_row_data,
                                                (const J*)mat_B->const_col_data,
                                                mat_C->descr,
                                                (C*)mat_C->val_data,
                                                (const I*)mat_C->const_row_data,
                                                (J*)mat_C->col_data,
                                                descr,
                                                temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
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

        case rocsparse_spgeam_stage_symbolic:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgeam_copy_row_pointer_and_free_memory_template(
                        handle,
                        mat_A->rows,
                        mat_B->cols,
                        mat_C->descr,
                        (I*)mat_C->row_data,
                        &mat_C->nnz,
                        descr));

                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgeam_symbolic_template(handle,
                                                         descr->trans_A,
                                                         descr->trans_B,
                                                         mat_A->rows,
                                                         mat_B->cols,
                                                         mat_A->descr,
                                                         mat_A->nnz,
                                                         (const I*)mat_A->const_row_data,
                                                         (const J*)mat_A->const_col_data,
                                                         mat_B->descr,
                                                         mat_B->nnz,
                                                         (const I*)mat_B->const_row_data,
                                                         (const J*)mat_B->const_col_data,
                                                         mat_C->descr,
                                                         (const I*)mat_C->const_row_data,
                                                         (J*)mat_C->col_data,
                                                         descr,
                                                         temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
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

        case rocsparse_spgeam_stage_numeric:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrgeam_numeric_template(handle,
                                                        descr->trans_A,
                                                        descr->trans_B,
                                                        mat_A->rows,
                                                        mat_B->cols,
                                                        (const T*)alpha,
                                                        mat_A->descr,
                                                        mat_A->nnz,
                                                        (const A*)mat_A->const_val_data,
                                                        (const I*)mat_A->const_row_data,
                                                        (const J*)mat_A->const_col_data,
                                                        (const T*)beta,
                                                        mat_B->descr,
                                                        mat_B->nnz,
                                                        (const B*)mat_B->const_val_data,
                                                        (const I*)mat_B->const_row_data,
                                                        (const J*)mat_B->const_col_data,
                                                        mat_C->descr,
                                                        (C*)mat_C->val_data,
                                                        (const I*)mat_C->const_row_data,
                                                        (const J*)mat_C->col_data,
                                                        descr,
                                                        temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
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
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    typedef rocsparse_status (*spgeam_template_t)(rocsparse_handle            handle,
                                                  rocsparse_spgeam_descr      descr,
                                                  const void*                 alpha,
                                                  rocsparse_const_spmat_descr mat_A,
                                                  const void*                 beta,
                                                  rocsparse_const_spmat_descr mat_B,
                                                  rocsparse_spmat_descr       mat_C,
                                                  rocsparse_spgeam_stage      stage,
                                                  size_t                      buffer_size,
                                                  void*                       temp_buffer);

    using spgeam_template_tuple
        = std::tuple<rocsparse_datatype, rocsparse_indextype, rocsparse_indextype>;
    // clang-format off
#define SPGEAM_TEMPLATE_CONFIG(T_, I_, J_)                                    \
    {                                                                         \
        spgeam_template_tuple(T_, I_, J_),                                    \
            spgeam_template<typename rocsparse::datatype_traits<T_>::type_t,  \
                            typename rocsparse::indextype_traits<I_>::type_t, \
                            typename rocsparse::indextype_traits<J_>::type_t, \
                            typename rocsparse::datatype_traits<T_>::type_t,  \
                            typename rocsparse::datatype_traits<T_>::type_t,  \
                            typename rocsparse::datatype_traits<T_>::type_t>  \
    }
    // clang-format on
    static const std::map<spgeam_template_tuple, spgeam_template_t> s_spgeam_template_dispatch{{

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f32_r, rocsparse_indextype_i32, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f32_r, rocsparse_indextype_i64, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f32_r, rocsparse_indextype_i64, rocsparse_indextype_i64),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f64_r, rocsparse_indextype_i32, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f64_r, rocsparse_indextype_i64, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f64_r, rocsparse_indextype_i64, rocsparse_indextype_i64),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f64_c, rocsparse_indextype_i32, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f64_c, rocsparse_indextype_i64, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f64_c, rocsparse_indextype_i64, rocsparse_indextype_i64),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f32_c, rocsparse_indextype_i32, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f32_c, rocsparse_indextype_i64, rocsparse_indextype_i32),

        SPGEAM_TEMPLATE_CONFIG(
            rocsparse_datatype_f32_c, rocsparse_indextype_i64, rocsparse_indextype_i64)}};

    static rocsparse_status spgeam_template_find(spgeam_template_t*  spgeam_function_,
                                                 rocsparse_datatype  compute_type_,
                                                 rocsparse_indextype i_type_,
                                                 rocsparse_indextype j_type_)
    {
        const auto& it = rocsparse::s_spgeam_template_dispatch.find(
            rocsparse::spgeam_template_tuple(compute_type_, i_type_, j_type_));

        if(it != rocsparse::s_spgeam_template_dispatch.end())
        {
            spgeam_function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "compute_type: " << rocsparse::to_string(compute_type_)
                 << ", i_type: " << rocsparse::to_string(i_type_)
                 << ", j_type: " << rocsparse::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

    static rocsparse_status spgeam_buffer_size_checkarg(rocsparse_handle            handle, //0
                                                        rocsparse_spgeam_descr      descr, //1
                                                        rocsparse_const_spmat_descr mat_A, //2
                                                        rocsparse_const_spmat_descr mat_B, //3
                                                        rocsparse_const_spmat_descr mat_C, //4
                                                        rocsparse_spgeam_stage      stage, //5
                                                        size_t*                     buffer_size) //6
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, mat_A);
        ROCSPARSE_CHECKARG_POINTER(3, mat_B);
        if(stage == rocsparse_spgeam_stage_compute)
        {
            ROCSPARSE_CHECKARG_POINTER(4, mat_C);
        }
        ROCSPARSE_CHECKARG_ENUM(5, stage);

        ROCSPARSE_CHECKARG(2, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(3, mat_B, (mat_B->init == false), rocsparse_status_not_initialized);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->format != mat_A->format), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            2, mat_A, (mat_A->data_type != descr->compute_type), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->data_type != descr->compute_type), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->col_type != mat_A->col_type), rocsparse_status_type_mismatch);

        if(stage == rocsparse_spgeam_stage_compute || mat_C != nullptr)
        {
            ROCSPARSE_CHECKARG(4, mat_C, (mat_C->init == false), rocsparse_status_not_initialized);

            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->format != mat_A->format), rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(4,
                               mat_C,
                               (mat_C->data_type != descr->compute_type),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->col_type != mat_A->col_type), rocsparse_status_type_mismatch);
        }

        return rocsparse_status_continue;
    }

    static rocsparse_status spgeam_checkarg(rocsparse_handle            handle, //0
                                            rocsparse_spgeam_descr      descr, //1
                                            const void*                 alpha, //2
                                            rocsparse_const_spmat_descr mat_A, //3
                                            const void*                 beta, //4
                                            rocsparse_const_spmat_descr mat_B, //5
                                            rocsparse_spmat_descr       mat_C, //6
                                            rocsparse_spgeam_stage      stage, //7
                                            size_t                      buffer_size, //8
                                            void*                       temp_buffer) //9
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(3, mat_A);
        ROCSPARSE_CHECKARG_POINTER(5, mat_B);
        if(stage == rocsparse_spgeam_stage_compute || stage == rocsparse_spgeam_stage_symbolic
           || stage == rocsparse_spgeam_stage_numeric)
        {
            ROCSPARSE_CHECKARG_POINTER(6, mat_C);
        }

        ROCSPARSE_CHECKARG_ENUM(7, stage);

        ROCSPARSE_CHECKARG(3, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(5, mat_B, (mat_B->init == false), rocsparse_status_not_initialized);

        ROCSPARSE_CHECKARG(
            5, mat_B, (mat_B->format != mat_A->format), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            3, mat_A, (mat_A->data_type != descr->compute_type), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            5, mat_B, (mat_B->data_type != descr->compute_type), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            5, mat_B, (mat_B->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            5, mat_B, (mat_B->col_type != mat_A->col_type), rocsparse_status_type_mismatch);

        if(stage == rocsparse_spgeam_stage_compute || stage == rocsparse_spgeam_stage_symbolic
           || stage == rocsparse_spgeam_stage_numeric || mat_C != nullptr)
        {
            ROCSPARSE_CHECKARG(6, mat_C, (mat_C->init == false), rocsparse_status_not_initialized);

            ROCSPARSE_CHECKARG(
                6, mat_C, (mat_C->format != mat_A->format), rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(6,
                               mat_C,
                               (mat_C->data_type != descr->compute_type),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(
                6, mat_C, (mat_C->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
            ROCSPARSE_CHECKARG(
                6, mat_C, (mat_C->col_type != mat_A->col_type), rocsparse_status_type_mismatch);
        }
        return rocsparse_status_continue;
    }
}

extern "C" rocsparse_status rocsparse_spgeam_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_spgeam_descr      descr,
                                                         rocsparse_const_spmat_descr mat_A,
                                                         rocsparse_const_spmat_descr mat_B,
                                                         rocsparse_const_spmat_descr mat_C,
                                                         rocsparse_spgeam_stage      stage,
                                                         size_t* buffer_size_in_bytes)
try
{
    rocsparse::log_trace("rocsparse_spgeam_buffer_size",
                         handle,
                         descr,
                         mat_A,
                         mat_B,
                         mat_C,
                         stage,
                         buffer_size_in_bytes);

    const rocsparse_status status = rocsparse::spgeam_buffer_size_checkarg(
        handle, descr, mat_A, mat_B, mat_C, stage, buffer_size_in_bytes);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::spgeam_buffer_size_template(
        handle, descr, mat_A, mat_B, mat_C, stage, buffer_size_in_bytes)));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_spgeam(rocsparse_handle            handle,
                                             rocsparse_spgeam_descr      descr,
                                             const void*                 alpha,
                                             rocsparse_const_spmat_descr mat_A,
                                             const void*                 beta,
                                             rocsparse_const_spmat_descr mat_B,
                                             rocsparse_spmat_descr       mat_C,
                                             rocsparse_spgeam_stage      stage,
                                             size_t                      buffer_size,
                                             void*                       temp_buffer)
try
{
    rocsparse::log_trace("rocsparse_spgeam",
                         handle,
                         descr,
                         alpha,
                         mat_A,
                         beta,
                         mat_B,
                         mat_C,
                         stage,
                         buffer_size,
                         temp_buffer);

    const rocsparse_status status = rocsparse::spgeam_checkarg(
        handle, descr, alpha, mat_A, beta, mat_B, mat_C, stage, buffer_size, temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    rocsparse::spgeam_template_t spgeam_function;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spgeam_template_find(
        &spgeam_function, descr->compute_type, mat_A->row_type, mat_A->col_type));

    RETURN_IF_ROCSPARSE_ERROR(spgeam_function(
        handle, descr, alpha, mat_A, beta, mat_B, mat_C, stage, buffer_size, temp_buffer));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
