/*! \file */
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

#include "rocsparse_control.hpp"
#include "rocsparse_coomm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename I, typename A>
    static rocsparse_status coomm_analysis_quickreturn(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_coomm_alg       alg,
                                                       I                         m,
                                                       I                         n,
                                                       I                         k,
                                                       int64_t                   nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const A*                  coo_val,
                                                       const I*                  coo_row_ind,
                                                       const I*                  coo_col_ind,
                                                       void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || k == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename I, typename A>
    static rocsparse_status coomm_analysis_checkarg(rocsparse_handle          handle, //0
                                                    rocsparse_operation       trans_A, //1
                                                    rocsparse_coomm_alg       alg, //2
                                                    I                         m, //3
                                                    I                         n, //4
                                                    I                         k, //5
                                                    int64_t                   nnz, //6
                                                    const rocsparse_mat_descr descr, //7
                                                    const A*                  coo_val, //8
                                                    const I*                  coo_row_ind, //9
                                                    const I*                  coo_col_ind, //10
                                                    void*                     temp_buffer) //11
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, alg);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(6, nnz);
        ROCSPARSE_CHECKARG_POINTER(7, descr);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_ARRAY(8, nnz, coo_val);
        ROCSPARSE_CHECKARG_ARRAY(9, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, coo_col_ind);

        const rocsparse_status status = rocsparse::coomm_analysis_quickreturn(handle,
                                                                              trans_A,
                                                                              alg,
                                                                              m,
                                                                              n,
                                                                              k,
                                                                              nnz,
                                                                              descr,
                                                                              coo_val,
                                                                              coo_row_ind,
                                                                              coo_col_ind,
                                                                              temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(11, temp_buffer);
        return rocsparse_status_continue;
    }

    template <typename I, typename A>
    static rocsparse_status coomm_analysis_core(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_coomm_alg       alg,
                                                I                         m,
                                                I                         n,
                                                I                         k,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                const A*                  coo_val,
                                                const I*                  coo_row_ind,
                                                const I*                  coo_col_ind,
                                                void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(alg)
        {
        case rocsparse_coomm_alg_default:
        case rocsparse_coomm_alg_atomic:
        {
            return rocsparse_status_success;
        }

        case rocsparse_coomm_alg_segmented:
        {
            return rocsparse_status_success;
        }

        case rocsparse_coomm_alg_segmented_atomic:
        {
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

template <typename I, typename A>
rocsparse_status rocsparse::coomm_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_coomm_alg       alg,
                                                    int64_t                   m,
                                                    int64_t                   n,
                                                    int64_t                   k,
                                                    int64_t                   nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               coo_val,
                                                    const void*               coo_row_ind,
                                                    const void*               coo_col_ind,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status
        = rocsparse::coomm_analysis_quickreturn<I, A>(handle,
                                                      trans_A,
                                                      alg,
                                                      m,
                                                      n,
                                                      k,
                                                      nnz,
                                                      descr,
                                                      static_cast<const A*>(coo_val),
                                                      static_cast<const I*>(coo_row_ind),
                                                      static_cast<const I*>(coo_col_ind),
                                                      temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::coomm_analysis_core<I, A>(handle,
                                              trans_A,
                                              alg,
                                              m,
                                              n,
                                              k,
                                              nnz,
                                              descr,
                                              static_cast<const A*>(coo_val),
                                              static_cast<const I*>(coo_row_ind),
                                              static_cast<const I*>(coo_col_ind),
                                              temp_buffer)));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename A>
    rocsparse_status coomm_analysis_impl(rocsparse_handle          handle,
                                         rocsparse_operation       trans_A,
                                         rocsparse_coomm_alg       alg,
                                         I                         m,
                                         I                         n,
                                         I                         k,
                                         int64_t                   nnz,
                                         const rocsparse_mat_descr descr,
                                         const A*                  coo_val,
                                         const I*                  coo_row_ind,
                                         const I*                  coo_col_ind,
                                         void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace(handle,
                             "rocsparse_coomm_analysis",
                             trans_A,
                             alg,
                             m,
                             n,
                             k,
                             nnz,
                             (const void*&)descr,
                             (const void*&)coo_val,
                             (const void*&)coo_row_ind,
                             (const void*&)coo_col_ind,
                             (const void*&)temp_buffer);

        const rocsparse_status status = rocsparse::coomm_analysis_checkarg(handle,
                                                                           trans_A,
                                                                           alg,
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           nnz,
                                                                           descr,
                                                                           coo_val,
                                                                           coo_row_ind,
                                                                           coo_col_ind,
                                                                           temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_analysis_core(handle,
                                                                 trans_A,
                                                                 alg,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 descr,
                                                                 coo_val,
                                                                 coo_row_ind,
                                                                 coo_col_ind,
                                                                 temp_buffer));

        return rocsparse_status_success;
    }
}

#define INSTANTIATE_ANALYSIS(I, A)                                      \
    template rocsparse_status rocsparse::coomm_analysis_template<I, A>( \
        rocsparse_handle          handle,                               \
        rocsparse_operation       trans_A,                              \
        rocsparse_coomm_alg       alg,                                  \
        int64_t                   m,                                    \
        int64_t                   n,                                    \
        int64_t                   k,                                    \
        int64_t                   nnz,                                  \
        const rocsparse_mat_descr descr,                                \
        const void*               coo_val,                              \
        const void*               coo_row_ind,                          \
        const void*               coo_col_ind,                          \
        void*                     temp_buffer);

// Uniform precisions
INSTANTIATE_ANALYSIS(int32_t, float);
INSTANTIATE_ANALYSIS(int64_t, float);
INSTANTIATE_ANALYSIS(int32_t, double);
INSTANTIATE_ANALYSIS(int64_t, double);
INSTANTIATE_ANALYSIS(int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int64_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_ANALYSIS(int32_t, _Float16);
INSTANTIATE_ANALYSIS(int64_t, _Float16);
INSTANTIATE_ANALYSIS(int32_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int64_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int32_t, int8_t);
INSTANTIATE_ANALYSIS(int64_t, int8_t);
#undef INSTANTIATE_ANALYSIS
