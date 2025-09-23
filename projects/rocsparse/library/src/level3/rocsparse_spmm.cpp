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

#include <map>
#include <sstream>

#include "rocsparse.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_bellmm.hpp"
#include "rocsparse_bsrmm.hpp"
#include "rocsparse_coomm.hpp"
#include "rocsparse_cscmm.hpp"
#include "rocsparse_csrmm.hpp"
#include "rocsparse_determine_indextype.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spmm_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spmm_alg_default);
        CASE(rocsparse_spmm_alg_csr);
        CASE(rocsparse_spmm_alg_coo_segmented);
        CASE(rocsparse_spmm_alg_coo_atomic);
        CASE(rocsparse_spmm_alg_csr_row_split);
        CASE(rocsparse_spmm_alg_csr_nnz_split);
        CASE(rocsparse_spmm_alg_csr_merge_path);
        CASE(rocsparse_spmm_alg_coo_segmented_atomic);
        CASE(rocsparse_spmm_alg_bell);
        CASE(rocsparse_spmm_alg_bsr);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spmm_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spmm_stage_buffer_size);
        CASE(rocsparse_spmm_stage_preprocess);
        CASE(rocsparse_spmm_stage_compute);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spmm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spmm_alg_default:
    case rocsparse_spmm_alg_csr:
    case rocsparse_spmm_alg_coo_segmented:
    case rocsparse_spmm_alg_coo_atomic:
    case rocsparse_spmm_alg_csr_row_split:
    case rocsparse_spmm_alg_csr_nnz_split:
    case rocsparse_spmm_alg_csr_merge_path:
    case rocsparse_spmm_alg_coo_segmented_atomic:
    case rocsparse_spmm_alg_bell:
    case rocsparse_spmm_alg_bsr:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spmm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spmm_stage_buffer_size:
    case rocsparse_spmm_stage_preprocess:
    case rocsparse_spmm_stage_compute:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
    rocsparse_status spmm_alg2bellmm_alg(rocsparse_spmm_alg    spmm_alg,
                                         rocsparse_bellmm_alg& bellmm_alg)
    {
        switch(spmm_alg)
        {
        case rocsparse_spmm_alg_default:
        case rocsparse_spmm_alg_bell:
        {
            bellmm_alg = rocsparse_bellmm_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_bsr:
        case rocsparse_spmm_alg_csr:
        case rocsparse_spmm_alg_csr_row_split:
        case rocsparse_spmm_alg_csr_nnz_split:
        case rocsparse_spmm_alg_csr_merge_path:
        case rocsparse_spmm_alg_coo_segmented:
        case rocsparse_spmm_alg_coo_atomic:
        case rocsparse_spmm_alg_coo_segmented_atomic:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    rocsparse_status spmm_alg2csrmm_alg(rocsparse_spmm_alg spmm_alg, rocsparse_csrmm_alg& csrmm_alg)
    {
        switch(spmm_alg)
        {
        case rocsparse_spmm_alg_default:
        case rocsparse_spmm_alg_csr:
        {
            csrmm_alg = rocsparse_csrmm_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_csr_row_split:
        {
            csrmm_alg = rocsparse_csrmm_alg_row_split;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_csr_nnz_split:
        {
            csrmm_alg = rocsparse_csrmm_alg_nnz_split;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_csr_merge_path:
        {
            csrmm_alg = rocsparse_csrmm_alg_merge_path;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_bell:
        case rocsparse_spmm_alg_bsr:
        case rocsparse_spmm_alg_coo_segmented:
        case rocsparse_spmm_alg_coo_atomic:
        case rocsparse_spmm_alg_coo_segmented_atomic:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    rocsparse_status spmm_alg2coomm_alg(rocsparse_spmm_alg spmm_alg, rocsparse_coomm_alg& coomm_alg)
    {
        switch(spmm_alg)
        {
        case rocsparse_spmm_alg_default:
        {
            coomm_alg = rocsparse_coomm_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_coo_segmented:
        {
            coomm_alg = rocsparse_coomm_alg_segmented;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_coo_atomic:
        {
            coomm_alg = rocsparse_coomm_alg_atomic;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_coo_segmented_atomic:
        {
            coomm_alg = rocsparse_coomm_alg_segmented_atomic;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_bell:
        case rocsparse_spmm_alg_bsr:
        case rocsparse_spmm_alg_csr:
        case rocsparse_spmm_alg_csr_row_split:
        case rocsparse_spmm_alg_csr_nnz_split:
        case rocsparse_spmm_alg_csr_merge_path:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    rocsparse_status spmm_alg2bsrmm_alg(rocsparse_spmm_alg spmm_alg, rocsparse_bsrmm_alg& bsrmm_alg)
    {
        switch(spmm_alg)
        {
        case rocsparse_spmm_alg_default:
        case rocsparse_spmm_alg_bsr:
        {
            bsrmm_alg = rocsparse_bsrmm_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmm_alg_csr:
        case rocsparse_spmm_alg_csr_row_split:
        case rocsparse_spmm_alg_csr_nnz_split:
        case rocsparse_spmm_alg_csr_merge_path:
        case rocsparse_spmm_alg_bell:
        case rocsparse_spmm_alg_coo_segmented:
        case rocsparse_spmm_alg_coo_atomic:
        case rocsparse_spmm_alg_coo_segmented_atomic:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    rocsparse_status spmm(rocsparse_handle            handle,
                          rocsparse_operation         trans_A,
                          rocsparse_operation         trans_B,
                          rocsparse_datatype          alpha_type,
                          const void*                 alpha,
                          rocsparse_const_spmat_descr mat_A,
                          rocsparse_const_dnmat_descr mat_B,
                          rocsparse_datatype          beta_type,
                          const void*                 beta,
                          const rocsparse_dnmat_descr mat_C,
                          rocsparse_spmm_alg          alg,
                          rocsparse_spmm_stage        stage,
                          size_t*                     buffer_size,
                          void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(mat_A->format)
        {
        case rocsparse_format_csr:
        {
            rocsparse_csrmm_alg csrmm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2csrmm_alg(alg, csrmm_alg)));

            const int64_t m = mat_A->rows;
            const int64_t n = mat_C->cols;
            const int64_t k = mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size(handle,
                                                                       trans_A,
                                                                       csrmm_alg,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       mat_A->nnz,
                                                                       mat_A->descr,
                                                                       alpha_type,
                                                                       mat_A->data_type,
                                                                       mat_A->const_val_data,
                                                                       mat_A->row_type,
                                                                       mat_A->const_row_data,
                                                                       mat_A->col_type,
                                                                       mat_A->const_col_data,
                                                                       buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis(handle,
                                                                    trans_A,
                                                                    csrmm_alg,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    mat_A->nnz,
                                                                    mat_A->descr,
                                                                    mat_A->data_type,
                                                                    mat_A->const_val_data,
                                                                    mat_A->row_type,
                                                                    mat_A->const_row_data,
                                                                    mat_A->col_type,
                                                                    mat_A->const_col_data,
                                                                    temp_buffer));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm(handle,
                                                           trans_A,
                                                           trans_B,
                                                           csrmm_alg,
                                                           m,
                                                           n,
                                                           k,
                                                           mat_A->nnz,
                                                           mat_A->batch_count,
                                                           mat_A->offsets_batch_stride,
                                                           mat_A->columns_values_batch_stride,
                                                           alpha_type,
                                                           alpha,
                                                           mat_A->descr,
                                                           mat_A->data_type,
                                                           mat_A->const_val_data,
                                                           mat_A->row_type,
                                                           mat_A->const_row_data,
                                                           mat_A->col_type,
                                                           mat_A->const_col_data,
                                                           mat_B->data_type,
                                                           mat_B->const_values,
                                                           mat_B->ld,
                                                           mat_B->batch_count,
                                                           mat_B->batch_stride,
                                                           mat_B->order,
                                                           beta_type,
                                                           beta,
                                                           mat_C->data_type,
                                                           mat_C->values,
                                                           mat_C->ld,
                                                           mat_C->batch_count,
                                                           mat_C->batch_stride,
                                                           mat_C->order,
                                                           temp_buffer,
                                                           false));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csc:
        {
            rocsparse_csrmm_alg csrmm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2csrmm_alg(alg, csrmm_alg)));

            const int64_t m = mat_A->rows;
            const int64_t n = mat_C->cols;
            const int64_t k = mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscmm_buffer_size(handle,
                                                                       trans_A,
                                                                       csrmm_alg,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       mat_A->nnz,
                                                                       mat_A->descr,
                                                                       alpha_type,
                                                                       mat_A->data_type,
                                                                       mat_A->const_val_data,
                                                                       mat_A->col_type,
                                                                       mat_A->const_col_data,
                                                                       mat_A->row_type,
                                                                       mat_A->const_row_data,
                                                                       buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscmm_analysis(handle,
                                                                    trans_A,
                                                                    csrmm_alg,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    mat_A->nnz,
                                                                    mat_A->descr,
                                                                    mat_A->data_type,
                                                                    mat_A->const_val_data,
                                                                    mat_A->col_type,
                                                                    mat_A->const_col_data,
                                                                    mat_A->row_type,
                                                                    mat_A->const_row_data,
                                                                    temp_buffer));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscmm(handle,
                                                           trans_A,
                                                           trans_B,
                                                           csrmm_alg,
                                                           m,
                                                           n,
                                                           k,
                                                           mat_A->nnz,
                                                           mat_A->batch_count,
                                                           mat_A->offsets_batch_stride,
                                                           mat_A->columns_values_batch_stride,
                                                           alpha_type,
                                                           alpha,
                                                           mat_A->descr,
                                                           mat_A->data_type,
                                                           mat_A->const_val_data,
                                                           mat_A->col_type,
                                                           mat_A->const_col_data,
                                                           mat_A->row_type,
                                                           mat_A->const_row_data,
                                                           mat_B->data_type,
                                                           mat_B->const_values,
                                                           mat_B->ld,
                                                           mat_B->batch_count,
                                                           mat_B->batch_stride,
                                                           mat_B->order,
                                                           beta_type,
                                                           beta,
                                                           mat_C->data_type,
                                                           mat_C->values,
                                                           mat_C->ld,
                                                           mat_C->batch_count,
                                                           mat_C->batch_stride,
                                                           mat_C->order,
                                                           temp_buffer));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_coo:
        {
            rocsparse_coomm_alg coomm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2coomm_alg(alg, coomm_alg)));

            const int64_t m = mat_A->rows;
            const int64_t n = mat_C->cols;
            const int64_t k = mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_buffer_size(handle,
                                                                       trans_A,
                                                                       coomm_alg,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       mat_A->nnz,
                                                                       mat_C->batch_count,
                                                                       mat_A->descr,
                                                                       alpha_type,
                                                                       mat_A->data_type,
                                                                       mat_A->const_val_data,
                                                                       mat_A->row_type,
                                                                       mat_A->const_row_data,
                                                                       mat_A->col_type,
                                                                       mat_A->const_col_data,
                                                                       buffer_size));
                return rocsparse_status_success;
            }

            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_analysis(handle,
                                                                    trans_A,
                                                                    coomm_alg,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    mat_A->nnz,
                                                                    mat_A->descr,
                                                                    mat_A->data_type,
                                                                    mat_A->const_val_data,
                                                                    mat_A->row_type,
                                                                    mat_A->const_row_data,
                                                                    mat_A->col_type,
                                                                    mat_A->const_col_data,
                                                                    temp_buffer));
                return rocsparse_status_success;
            }

            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm(handle,
                                                           trans_A,
                                                           trans_B,
                                                           coomm_alg,
                                                           m,
                                                           n,
                                                           k,
                                                           mat_A->nnz,
                                                           mat_A->batch_count,
                                                           mat_A->batch_stride,
                                                           alpha_type,
                                                           alpha,
                                                           mat_A->descr,
                                                           mat_A->data_type,
                                                           mat_A->const_val_data,
                                                           mat_A->row_type,
                                                           mat_A->const_row_data,
                                                           mat_A->col_type,
                                                           mat_A->const_col_data,
                                                           mat_B->data_type,
                                                           mat_B->const_values,
                                                           mat_B->ld,
                                                           mat_B->batch_count,
                                                           mat_B->batch_stride,
                                                           mat_B->order,
                                                           beta_type,
                                                           beta,
                                                           mat_C->data_type,
                                                           mat_C->values,
                                                           mat_C->ld,
                                                           mat_C->batch_count,
                                                           mat_C->batch_stride,
                                                           mat_C->order,
                                                           temp_buffer));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bell:
        {
            rocsparse_bellmm_alg bellmm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2bellmm_alg(alg, bellmm_alg)));

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bellmm_buffer_size(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->block_dir,
                    (mat_C->rows / mat_A->block_dim),
                    mat_C->cols,

                    (trans_A == rocsparse_operation_none) ? (mat_A->cols / mat_A->block_dim)
                                                          : (mat_A->rows / mat_A->block_dim),

                    mat_A->ell_cols,
                    mat_A->block_dim,
                    alpha_type,
                    alpha,
                    mat_A->descr,
                    mat_A->col_type,
                    mat_A->const_col_data,
                    mat_A->data_type,
                    mat_A->const_val_data,
                    mat_B->data_type,
                    mat_B->const_values,
                    mat_B->ld,
                    mat_B->order,
                    beta_type,
                    beta,
                    mat_C->data_type,
                    mat_C->values,
                    mat_C->ld,
                    mat_C->order,
                    buffer_size)));
                return rocsparse_status_success;
            }

            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bellmm_preprocess(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->block_dir,
                    (mat_C->rows / mat_A->block_dim),
                    mat_C->cols,

                    (trans_A == rocsparse_operation_none) ? (mat_A->cols / mat_A->block_dim)
                                                          : (mat_A->rows / mat_A->block_dim),

                    mat_A->ell_cols,
                    mat_A->block_dim,
                    alpha_type,
                    alpha,
                    mat_A->descr,
                    mat_A->col_type,
                    mat_A->const_col_data,
                    mat_A->data_type,
                    mat_A->const_val_data,
                    mat_B->data_type,
                    mat_B->const_values,
                    mat_B->ld,
                    mat_B->order,
                    beta_type,
                    beta,
                    mat_C->data_type,
                    mat_C->values,
                    mat_C->ld,
                    mat_C->order,
                    temp_buffer)));
                return rocsparse_status_success;
            }

            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bellmm(handle,
                                                             trans_A,
                                                             trans_B,
                                                             mat_A->block_dir,
                                                             (mat_C->rows / mat_A->block_dim),
                                                             mat_C->cols,

                                                             (trans_A == rocsparse_operation_none)
                                                                 ? (mat_A->cols / mat_A->block_dim)
                                                                 : (mat_A->rows / mat_A->block_dim),

                                                             mat_A->ell_cols,
                                                             mat_A->block_dim,
                                                             mat_A->batch_count,
                                                             mat_A->batch_stride,
                                                             alpha_type,
                                                             alpha,
                                                             mat_A->descr,
                                                             mat_A->col_type,
                                                             mat_A->const_col_data,
                                                             mat_A->data_type,
                                                             mat_A->const_val_data,
                                                             mat_B->data_type,
                                                             mat_B->const_values,
                                                             mat_B->ld,
                                                             mat_B->batch_count,
                                                             mat_B->batch_stride,
                                                             mat_B->order,
                                                             beta_type,
                                                             beta,
                                                             mat_C->data_type,
                                                             mat_C->values,
                                                             mat_C->ld,
                                                             mat_C->batch_count,
                                                             mat_C->batch_stride,
                                                             mat_C->order,
                                                             temp_buffer)));
                return rocsparse_status_success;
            }
            }

            break;
        }

        case rocsparse_format_bsr:
        {
            rocsparse_bsrmm_alg bsrmm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2bsrmm_alg(alg, bsrmm_alg)));

            const int64_t mb = mat_A->rows;
            const int64_t n  = mat_C->cols;
            const int64_t kb = mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_buffer_size(handle,
                                                                       trans_A,
                                                                       bsrmm_alg,
                                                                       mb,
                                                                       n,
                                                                       kb,
                                                                       mat_A->nnz,
                                                                       mat_A->descr,
                                                                       mat_A->data_type,
                                                                       mat_A->const_val_data,
                                                                       mat_A->row_type,
                                                                       mat_A->const_row_data,
                                                                       mat_A->col_type,
                                                                       mat_A->const_col_data,
                                                                       mat_A->block_dim,
                                                                       buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_analysis(handle,
                                                                    trans_A,
                                                                    bsrmm_alg,
                                                                    mb,
                                                                    n,
                                                                    kb,
                                                                    mat_A->nnz,
                                                                    mat_A->descr,
                                                                    mat_A->data_type,
                                                                    mat_A->const_val_data,
                                                                    mat_A->row_type,
                                                                    mat_A->const_row_data,
                                                                    mat_A->col_type,
                                                                    mat_A->const_col_data,
                                                                    mat_A->block_dim,
                                                                    temp_buffer));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm(handle,
                                                           mat_A->block_dir,
                                                           trans_A,
                                                           trans_B,
                                                           bsrmm_alg,
                                                           mb,
                                                           n,
                                                           kb,
                                                           mat_A->nnz,
                                                           mat_A->batch_count,
                                                           mat_A->offsets_batch_stride,
                                                           mat_A->columns_values_batch_stride,
                                                           alpha_type,
                                                           alpha,
                                                           mat_A->descr,
                                                           mat_A->data_type,
                                                           mat_A->const_val_data,
                                                           mat_A->row_type,
                                                           mat_A->const_row_data,
                                                           mat_A->col_type,
                                                           mat_A->const_col_data,
                                                           mat_A->block_dim,
                                                           mat_B->data_type,
                                                           mat_B->const_values,
                                                           mat_B->ld,
                                                           mat_B->batch_count,
                                                           mat_B->batch_stride,
                                                           mat_B->order,
                                                           beta_type,
                                                           beta,
                                                           mat_C->data_type,
                                                           mat_C->values,
                                                           mat_C->ld,
                                                           mat_C->batch_count,
                                                           mat_C->batch_stride,
                                                           mat_C->order));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_ell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
            // LCOV_EXCL_START
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_spmm(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans_A, //1
                                           rocsparse_operation         trans_B, //2
                                           const void*                 alpha, //3
                                           rocsparse_const_spmat_descr mat_A, //4
                                           rocsparse_const_dnmat_descr mat_B, //5
                                           const void*                 beta, //6
                                           const rocsparse_dnmat_descr mat_C, //7
                                           rocsparse_datatype          compute_type, //8
                                           rocsparse_spmm_alg          alg, //9
                                           rocsparse_spmm_stage        stage, //10
                                           size_t*                     buffer_size, //11
                                           void*                       temp_buffer) //12
try
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         "rocsparse_spmm",
                         trans_A,
                         trans_B,
                         (const void*&)alpha,
                         (const void*&)mat_A,
                         (const void*&)mat_B,
                         (const void*&)mat_C,
                         compute_type,
                         alg,
                         stage,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, mat_A);
    ROCSPARSE_CHECKARG(4, mat_A, mat_A->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(5, mat_B);
    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG(5, mat_B, mat_B->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(7, mat_C);
    ROCSPARSE_CHECKARG(7, mat_C, mat_C->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_ENUM(8, compute_type);
    ROCSPARSE_CHECKARG(
        8, compute_type, (compute_type != mat_C->data_type), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ENUM(9, alg);
    ROCSPARSE_CHECKARG_ENUM(10, stage);
    ROCSPARSE_CHECKARG(11,
                       buffer_size,
                       (temp_buffer == nullptr && buffer_size == nullptr),
                       rocsparse_status_invalid_pointer);

    switch(stage)
    {

    case rocsparse_spmm_stage_buffer_size:
    {
        ROCSPARSE_CHECKARG_POINTER(11, buffer_size);
        break;
    }
    case rocsparse_spmm_stage_preprocess:
    {
        break;
    }
    case rocsparse_spmm_stage_compute:
    {
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmm(handle,
                                              trans_A,
                                              trans_B,
                                              compute_type,
                                              alpha,
                                              mat_A,
                                              mat_B,
                                              compute_type,
                                              beta,
                                              mat_C,
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
