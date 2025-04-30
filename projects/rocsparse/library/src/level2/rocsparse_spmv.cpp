/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "internal/generic/rocsparse_spmv.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_bsrmv.hpp"
#include "rocsparse_coomv.hpp"
#include "rocsparse_coomv_aos.hpp"
#include "rocsparse_cscmv.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_ellmv.hpp"
#include "to_string.hpp"
#include <map>
#include <sstream>

namespace rocsparse
{

    static rocsparse_status check_spmv_alg(rocsparse_format format, rocsparse_spmv_alg alg)
    {
        switch(format)
        {
        case rocsparse_format_csr:
        case rocsparse_format_csc:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_csr_lrb:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_coo_atomic:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_coo_atomic:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_csr_lrb:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_format_ell:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_ell:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_coo_atomic:
            case rocsparse_spmv_alg_csr_lrb:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_format_bell:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_coo_atomic:
            case rocsparse_spmv_alg_csr_lrb:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        case rocsparse_format_bsr:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_bsr:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_coo_atomic:
            case rocsparse_spmv_alg_csr_lrb:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    static rocsparse_status spmv_alg2csrmv_alg(rocsparse_spmv_alg    spmv_alg,
                                               rocsparse::csrmv_alg& target)
    {
        switch(spmv_alg)
        {
        case rocsparse_spmv_alg_csr_rowsplit:
        {
            target = rocsparse::csrmv_alg_rowsplit;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_csr_adaptive:
        {
            target = rocsparse::csrmv_alg_adaptive;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_csr_lrb:
        {
            target = rocsparse::csrmv_alg_lrb;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_coo_atomic:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        {
            // LCOV_EXCL_START
            return rocsparse_status_invalid_value;
        }
        }
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    static rocsparse_status spmv_alg2coomv_alg(rocsparse_spmv_alg   spmv_alg,
                                               rocsparse_coomv_alg& coomv_alg)
    {
        switch(spmv_alg)
        {
        case rocsparse_spmv_alg_default:
        {
            coomv_alg = rocsparse_coomv_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo:
        {
            coomv_alg = rocsparse_coomv_alg_segmented;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo_atomic:
        {
            coomv_alg = rocsparse_coomv_alg_atomic;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_csr_rowsplit:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_csr_lrb:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    static rocsparse_status spmv_alg2coomv_aos_alg(rocsparse_spmv_alg       spmv_alg,
                                                   rocsparse_coomv_aos_alg& coomv_aos_alg)
    {
        switch(spmv_alg)
        {
        case rocsparse_spmv_alg_default:
        {
            coomv_aos_alg = rocsparse_coomv_aos_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo:
        {
            coomv_aos_alg = rocsparse_coomv_aos_alg_segmented;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo_atomic:
        {
            coomv_aos_alg = rocsparse_coomv_aos_alg_atomic;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_csr_rowsplit:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_csr_lrb:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status spmv_template(rocsparse_handle            handle,
                                   rocsparse_operation         trans,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr mat,
                                   rocsparse_const_dnvec_descr x,
                                   const void*                 beta,
                                   const rocsparse_dnvec_descr y,
                                   rocsparse_spmv_alg          alg,
                                   rocsparse_spmv_stage        stage,
                                   size_t*                     buffer_size,
                                   void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_spmv_alg(mat->format, alg)));

        switch(mat->format)
        {
        case rocsparse_format_coo:
        {
            rocsparse_coomv_alg coomv_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_alg(alg, coomv_alg)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_preprocess:
            {
                if(alg == rocsparse_spmv_alg_coo_atomic && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coomv_analysis_template(handle,
                                                            trans,
                                                            coomv_alg,
                                                            (I)mat->rows,
                                                            (I)mat->cols,
                                                            mat->nnz,
                                                            mat->descr,
                                                            (const A*)mat->const_val_data,
                                                            (const I*)mat->const_row_data,
                                                            (const I*)mat->const_col_data)));

                    mat->analysed = true;
                }
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_template(handle,
                                                                    trans,
                                                                    coomv_alg,
                                                                    (I)mat->rows,
                                                                    (I)mat->cols,
                                                                    mat->nnz,
                                                                    (const T*)alpha,
                                                                    mat->descr,
                                                                    (const A*)mat->const_val_data,
                                                                    (const I*)mat->const_row_data,
                                                                    (const I*)mat->const_col_data,
                                                                    (const X*)x->const_values,
                                                                    (const T*)beta,
                                                                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_coo_aos:
        {
            rocsparse_coomv_aos_alg coomv_aos_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_aos_alg(alg, coomv_aos_alg)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_preprocess:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coomv_aos_template(handle,
                                                  trans,
                                                  coomv_aos_alg,
                                                  (I)mat->rows,
                                                  (I)mat->cols,
                                                  mat->nnz,
                                                  (const T*)alpha,
                                                  mat->descr,
                                                  (const A*)mat->const_val_data,
                                                  (const I*)mat->const_ind_data,
                                                  (const X*)x->const_values,
                                                  (const T*)beta,
                                                  (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bsr:
        {
            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {
                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                if(alg == rocsparse_spmv_alg_default && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::bsrmv_analysis_template(handle,
                                                           mat->block_dir,
                                                           trans,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->const_val_data,
                                                           (const I*)mat->const_row_data,
                                                           (const J*)mat->const_col_data,
                                                           (J)mat->block_dim,
                                                           mat->info));
                    mat->analysed = true;
                }

                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template(handle,
                                                                    mat->block_dir,
                                                                    trans,
                                                                    (J)mat->rows,
                                                                    (J)mat->cols,
                                                                    (I)mat->nnz,
                                                                    (const T*)alpha,
                                                                    mat->descr,
                                                                    (const A*)mat->const_val_data,
                                                                    (const I*)mat->const_row_data,
                                                                    (const J*)mat->const_col_data,
                                                                    (J)mat->block_dim,
                                                                    mat->info,
                                                                    (const X*)x->const_values,
                                                                    (const T*)beta,
                                                                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csr:
        {
            rocsparse::csrmv_alg alg_csrmv;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {

                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive
                    || alg == rocsparse_spmv_alg_csr_lrb)
                   && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::csrmv_analysis_template(handle,
                                                           trans,
                                                           alg_csrmv,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->const_val_data,
                                                           (const I*)mat->const_row_data,
                                                           (const J*)mat->const_col_data,
                                                           mat->info));

                    mat->analysed = true;
                }

                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_template(
                    handle,
                    trans,
                    alg_csrmv,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (const T*)alpha,
                    mat->descr,
                    (const A*)mat->const_val_data,
                    (const I*)mat->const_row_data,
                    ((const I*)mat->const_row_data) + 1,
                    (const J*)mat->const_col_data,
                    (alg == rocsparse_spmv_alg_csr_rowsplit) ? nullptr : mat->info,
                    (const X*)x->const_values,
                    (const T*)beta,
                    (Y*)y->values,
                    false));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csc:
        {
            rocsparse::csrmv_alg alg_csrmv;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {
                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive
                    || alg == rocsparse_spmv_alg_csr_lrb)
                   && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::cscmv_analysis_template(handle,
                                                           trans,
                                                           alg_csrmv,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->const_val_data,
                                                           (const I*)mat->const_col_data,
                                                           (const J*)mat->const_row_data,
                                                           mat->info));

                    mat->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscmv_template(
                    handle,
                    trans,
                    alg_csrmv,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (const T*)alpha,
                    mat->descr,
                    (const A*)mat->const_val_data,
                    (const I*)mat->const_col_data,
                    (const J*)mat->const_row_data,
                    (alg == rocsparse_spmv_alg_csr_rowsplit) ? nullptr : mat->info,
                    (const X*)x->const_values,
                    (const T*)beta,
                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_ell:
        {
            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_template(handle,
                                                                    trans,
                                                                    (I)mat->rows,
                                                                    (I)mat->cols,
                                                                    (const T*)alpha,
                                                                    mat->descr,
                                                                    (const A*)mat->const_val_data,
                                                                    (const I*)mat->const_col_data,
                                                                    (I)mat->ell_width,
                                                                    (const X*)x->const_values,
                                                                    (const T*)beta,
                                                                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bell:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    typedef rocsparse_status (*spmv_template_t)(rocsparse_handle            handle,
                                                rocsparse_operation         trans,
                                                const void*                 alpha,
                                                rocsparse_const_spmat_descr mat,
                                                rocsparse_const_dnvec_descr x,
                                                const void*                 beta,
                                                const rocsparse_dnvec_descr y,
                                                rocsparse_spmv_alg          alg,
                                                rocsparse_spmv_stage        stage,
                                                size_t*                     buffer_size,
                                                void*                       temp_buffer);

    using spmv_template_tuple = std::tuple<rocsparse_datatype,
                                           rocsparse_indextype,
                                           rocsparse_indextype,
                                           rocsparse_datatype,
                                           rocsparse_datatype,
                                           rocsparse_datatype>;

#define SPMV_TEMPLATE_CONFIG(T_, I_, J_, A_, X_, Y_)                        \
    {                                                                       \
        spmv_template_tuple(T_, I_, J_, A_, X_, Y_),                        \
            spmv_template<typename rocsparse::datatype_traits<T_>::type_t,  \
                          typename rocsparse::indextype_traits<I_>::type_t, \
                          typename rocsparse::indextype_traits<J_>::type_t, \
                          typename rocsparse::datatype_traits<A_>::type_t,  \
                          typename rocsparse::datatype_traits<X_>::type_t,  \
                          typename rocsparse::datatype_traits<Y_>::type_t>  \
    }

    static const std::map<spmv_template_tuple, spmv_template_t> s_spmv_template_dispatch{{

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_i32_r,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_i32_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_i32_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_f32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_f32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_i8_r,
                             rocsparse_datatype_f32_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_r,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_r),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f64_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f64_r,
                             rocsparse_datatype_f64_c,
                             rocsparse_datatype_f64_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i32,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i32,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c),

        SPMV_TEMPLATE_CONFIG(rocsparse_datatype_f32_c,
                             rocsparse_indextype_i64,
                             rocsparse_indextype_i64,
                             rocsparse_datatype_f32_r,
                             rocsparse_datatype_f32_c,
                             rocsparse_datatype_f32_c)}};

    static rocsparse_status spmv_template_find(spmv_template_t*    spmv_function_,
                                               rocsparse_datatype  c_type_,
                                               rocsparse_indextype i_type_,
                                               rocsparse_indextype j_type_,
                                               rocsparse_datatype  a_type_,
                                               rocsparse_datatype  x_type_,
                                               rocsparse_datatype  y_type_)
    {
        const auto& it = rocsparse::s_spmv_template_dispatch.find(
            rocsparse::spmv_template_tuple(c_type_, i_type_, j_type_, a_type_, x_type_, y_type_));

        if(it != rocsparse::s_spmv_template_dispatch.end())
        {
            spmv_function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "c_type: " << rocsparse::to_string(c_type_) << std::endl
                      << ", i_type: " << rocsparse::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::to_string(j_type_) << std::endl
                      << ", a_type: " << rocsparse::to_string(a_type_) << std::endl
                      << ", x_type: " << rocsparse::to_string(x_type_) << std::endl
                      << ", y_type: " << rocsparse::to_string(y_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_spmv_template_dispatch)
            {
                const auto& t      = p.first;
                const auto  c_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  j_type = std::get<2>(t);
                const auto  a_type = std::get<3>(t);
                const auto  x_type = std::get<4>(t);
                const auto  y_type = std::get<5>(t);
                std::cout << std::endl
                          << std::endl
                          << "c_type: " << rocsparse::to_string(c_type) << std::endl
                          << ", i_type: " << rocsparse::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::to_string(j_type) << std::endl
                          << ", a_type: " << rocsparse::to_string(a_type) << std::endl
                          << ", x_type: " << rocsparse::to_string(x_type) << std::endl
                          << ", y_type: " << rocsparse::to_string(y_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "c_type: " << rocsparse::to_string(c_type_)
                 << ", i_type: " << rocsparse::to_string(i_type_)
                 << ", j_type: " << rocsparse::to_string(j_type_)
                 << ", a_type: " << rocsparse::to_string(a_type_)
                 << ", x_type: " << rocsparse::to_string(x_type_)
                 << ", y_type: " << rocsparse::to_string(y_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spmv(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans, //1
                                           const void*                 alpha, //2
                                           rocsparse_const_spmat_descr mat, //3
                                           rocsparse_const_dnvec_descr x, //4
                                           const void*                 beta, //5
                                           const rocsparse_dnvec_descr y, //6
                                           rocsparse_datatype          compute_type, //7
                                           rocsparse_spmv_alg          alg, //8
                                           rocsparse_spmv_stage        stage, //9
                                           size_t*                     buffer_size, //10
                                           void*                       temp_buffer) //11
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spmv",
                         trans,
                         (const void*&)alpha,
                         (const void*&)mat,
                         (const void*&)x,
                         (const void*&)beta,
                         (const void*&)y,
                         compute_type,
                         alg,
                         stage,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, alpha);
    ROCSPARSE_CHECKARG_POINTER(3, mat);
    ROCSPARSE_CHECKARG_POINTER(4, x);
    ROCSPARSE_CHECKARG_POINTER(5, beta);
    ROCSPARSE_CHECKARG_POINTER(6, y);
    ROCSPARSE_CHECKARG_ENUM(7, compute_type);
    ROCSPARSE_CHECKARG_ENUM(8, alg);
    ROCSPARSE_CHECKARG_ENUM(9, stage);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    ROCSPARSE_CHECKARG(10,
                       buffer_size,
                       (temp_buffer == nullptr && buffer_size == nullptr),
                       rocsparse_status_invalid_pointer);

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    ROCSPARSE_CHECKARG(3, mat, (mat->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, x, (x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(6, y, (y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    rocsparse::spmv_template_t spmv_function;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmv_template_find(&spmv_function,
                                                            compute_type,
                                                            rocsparse::determine_I_index_type(mat),
                                                            rocsparse::determine_J_index_type(mat),
                                                            mat->data_type,
                                                            x->data_type,
                                                            y->data_type));

    RETURN_IF_ROCSPARSE_ERROR(
        spmv_function(handle, trans, alpha, mat, x, beta, y, alg, stage, buffer_size, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
