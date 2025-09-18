/*! \file */
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

#include "rocsparse_enum_name.hpp"

#define CASE(VALUE) \
    case VALUE:     \
        return #VALUE
#define RETURN_INVALID return "invalid"

const char* rocsparse_indextype_name(rocsparse_indextype value)
{
    switch(value)
    {
        CASE(rocsparse_indextype_u16);
        CASE(rocsparse_indextype_i32);
        CASE(rocsparse_indextype_i64);
    }
    RETURN_INVALID;
}

const char* rocsparse_datatype_name(rocsparse_datatype value)
{
    switch(value)
    {
        CASE(rocsparse_datatype_f16_r);
        CASE(rocsparse_datatype_f32_r);
        CASE(rocsparse_datatype_f64_r);
        CASE(rocsparse_datatype_f32_c);
        CASE(rocsparse_datatype_f64_c);
        CASE(rocsparse_datatype_i8_r);
        CASE(rocsparse_datatype_u8_r);
        CASE(rocsparse_datatype_i32_r);
        CASE(rocsparse_datatype_u32_r);
        CASE(rocsparse_datatype_bf16_r);
    }
    RETURN_INVALID;
}

const char* rocsparse_index_base_name(rocsparse_index_base value)
{
    switch(value)
    {
        CASE(rocsparse_index_base_zero);
        CASE(rocsparse_index_base_one);
    }
    RETURN_INVALID;
}

const char* rocsparse_operation_name(rocsparse_operation value)
{
    switch(value)
    {
        CASE(rocsparse_operation_none);
        CASE(rocsparse_operation_transpose);
        CASE(rocsparse_operation_conjugate_transpose);
    }
    RETURN_INVALID;
}

const char* rocsparse_matrix_type_name(rocsparse_matrix_type value)
{
    switch(value)
    {
        CASE(rocsparse_matrix_type_general);
        CASE(rocsparse_matrix_type_symmetric);
        CASE(rocsparse_matrix_type_hermitian);
        CASE(rocsparse_matrix_type_triangular);
    }
    RETURN_INVALID;
}

const char* rocsparse_diag_type_name(rocsparse_diag_type value)
{
    switch(value)
    {
        CASE(rocsparse_diag_type_non_unit);
        CASE(rocsparse_diag_type_unit);
    }
    RETURN_INVALID;
}

const char* rocsparse_fill_mode_name(rocsparse_fill_mode value)
{
    switch(value)
    {
        CASE(rocsparse_fill_mode_lower);
        CASE(rocsparse_fill_mode_upper);
    }
    RETURN_INVALID;
}

const char* rocsparse_storage_mode_name(rocsparse_storage_mode value)
{
    switch(value)
    {
        CASE(rocsparse_storage_mode_sorted);
        CASE(rocsparse_storage_mode_unsorted);
    }
    RETURN_INVALID;
}

const char* rocsparse_action_name(rocsparse_action value)
{
    switch(value)
    {
        CASE(rocsparse_action_symbolic);
        CASE(rocsparse_action_numeric);
    }
    RETURN_INVALID;
}

const char* rocsparse_hyb_partition_name(rocsparse_hyb_partition value)
{
    switch(value)
    {
        CASE(rocsparse_hyb_partition_auto);
        CASE(rocsparse_hyb_partition_user);
        CASE(rocsparse_hyb_partition_max);
    }
    RETURN_INVALID;
}

const char* rocsparse_analysis_policy_name(rocsparse_analysis_policy value)
{
    switch(value)
    {
        CASE(rocsparse_analysis_policy_reuse);
        CASE(rocsparse_analysis_policy_force);
    }
    RETURN_INVALID;
}

const char* rocsparse_solve_policy_name(rocsparse_solve_policy value)
{
    switch(value)
    {
        CASE(rocsparse_solve_policy_auto);
    }
    RETURN_INVALID;
}

const char* rocsparse_direction_name(rocsparse_direction value)
{
    switch(value)
    {
        CASE(rocsparse_direction_row);
        CASE(rocsparse_direction_column);
    }
    RETURN_INVALID;
}

const char* rocsparse_order_name(rocsparse_order value)
{
    switch(value)
    {
        CASE(rocsparse_order_row);
        CASE(rocsparse_order_column);
    }
    RETURN_INVALID;
}

const char* rocsparse_format_name(rocsparse_format value)
{
    switch(value)
    {
        CASE(rocsparse_format_coo);
        CASE(rocsparse_format_coo_aos);
        CASE(rocsparse_format_csr);
        CASE(rocsparse_format_bsr);
        CASE(rocsparse_format_csc);
        CASE(rocsparse_format_ell);
        CASE(rocsparse_format_bell);
    }
    RETURN_INVALID;
}

const char* rocsparse_sddmm_alg_name(rocsparse_sddmm_alg value)
{
    switch(value)
    {
        CASE(rocsparse_sddmm_alg_default);
        CASE(rocsparse_sddmm_alg_dense);
    }
    RETURN_INVALID;
}

const char* rocsparse_itilu0_alg_name(rocsparse_itilu0_alg value)
{
    switch(value)
    {
        CASE(rocsparse_itilu0_alg_default);
        CASE(rocsparse_itilu0_alg_async_inplace);
        CASE(rocsparse_itilu0_alg_async_split);
        CASE(rocsparse_itilu0_alg_sync_split);
        CASE(rocsparse_itilu0_alg_sync_split_fusion);
    }
    RETURN_INVALID;
}

const char* rocsparse_spmv_alg_name(rocsparse_spmv_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spmv_alg_default);
        CASE(rocsparse_spmv_alg_bsr);
        CASE(rocsparse_spmv_alg_coo);
        CASE(rocsparse_spmv_alg_csr_adaptive);
        CASE(rocsparse_spmv_alg_csr_rowsplit);
        CASE(rocsparse_spmv_alg_ell);
        CASE(rocsparse_spmv_alg_coo_atomic);
        CASE(rocsparse_spmv_alg_csr_lrb);
        CASE(rocsparse_spmv_alg_csr_nnzsplit);
    }
    RETURN_INVALID;
}

const char* rocsparse_spsv_alg_name(rocsparse_spsv_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spsv_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_sptrsv_alg_name(rocsparse_sptrsv_alg value)
{
    switch(value)
    {
        CASE(rocsparse_sptrsv_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_spitsv_alg_name(rocsparse_spitsv_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spitsv_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_spsm_alg_name(rocsparse_spsm_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spsm_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_sptrsm_alg_name(rocsparse_sptrsm_alg value)
{
    switch(value)
    {
        CASE(rocsparse_sptrsm_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_spmm_alg_name(rocsparse_spmm_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spmm_alg_default);
        CASE(rocsparse_spmm_alg_bsr);
        CASE(rocsparse_spmm_alg_csr);
        CASE(rocsparse_spmm_alg_csr_merge_path);
        CASE(rocsparse_spmm_alg_coo_segmented);
        CASE(rocsparse_spmm_alg_coo_atomic);
        CASE(rocsparse_spmm_alg_bell);
        CASE(rocsparse_spmm_alg_coo_segmented_atomic);
        CASE(rocsparse_spmm_alg_csr_row_split);
        CASE(rocsparse_spmm_alg_csr_merge);
    }
    RETURN_INVALID;
}

const char* rocsparse_spgemm_alg_name(rocsparse_spgemm_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spgemm_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_spgeam_alg_name(rocsparse_spgeam_alg value)
{
    switch(value)
    {
        CASE(rocsparse_spgeam_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_sparse_to_dense_alg_name(rocsparse_sparse_to_dense_alg value)
{
    switch(value)
    {
        CASE(rocsparse_sparse_to_dense_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_dense_to_sparse_alg_name(rocsparse_dense_to_sparse_alg value)
{
    switch(value)
    {
        CASE(rocsparse_dense_to_sparse_alg_default);
    }
    RETURN_INVALID;
}

const char* rocsparse_gtsv_interleaved_alg_name(rocsparse_gtsv_interleaved_alg value)
{
    switch(value)
    {
        CASE(rocsparse_gtsv_interleaved_alg_default);
        CASE(rocsparse_gtsv_interleaved_alg_thomas);
        CASE(rocsparse_gtsv_interleaved_alg_lu);
        CASE(rocsparse_gtsv_interleaved_alg_qr);
    }
    RETURN_INVALID;
}

const char* rocsparse_gpsv_interleaved_alg_name(rocsparse_gpsv_interleaved_alg value)
{
    switch(value)
    {
        CASE(rocsparse_gpsv_interleaved_alg_default);
        CASE(rocsparse_gpsv_interleaved_alg_qr);
    }
    RETURN_INVALID;
}

#undef CASE
#undef RETURN_INVALID
