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

template <typename F, typename... P>
inline constexpr int token_count(F f)
{
    return 1;
}

template <typename F, typename... P>
inline constexpr int token_count(F f, P... p)
{
    return token_count(p...) + 1;
}

#define DEF(TOKEN, ...)                                           \
    struct TOKEN##_st                                             \
    {                                                             \
        using value_t                 = int;                      \
        static constexpr int   s      = token_count(__VA_ARGS__); \
        static constexpr TOKEN all[s] = {__VA_ARGS__};            \
    };                                                            \
    constexpr TOKEN TOKEN##_st::all[TOKEN##_st::s]

DEF(rocsparse_indextype, rocsparse_indextype_u16, rocsparse_indextype_i32, rocsparse_indextype_i64);

DEF(rocsparse_datatype,
    rocsparse_datatype_f16_r,
    rocsparse_datatype_f32_r,
    rocsparse_datatype_f64_r,
    rocsparse_datatype_f32_c,
    rocsparse_datatype_f64_c,
    rocsparse_datatype_i8_r,
    rocsparse_datatype_u8_r,
    rocsparse_datatype_i32_r,
    rocsparse_datatype_u32_r,
    rocsparse_datatype_bf16_r);

DEF(rocsparse_index_base, rocsparse_index_base_zero, rocsparse_index_base_one);

DEF(rocsparse_operation,
    rocsparse_operation_none,
    rocsparse_operation_transpose,
    rocsparse_operation_conjugate_transpose);

DEF(rocsparse_matrix_type,
    rocsparse_matrix_type_general,
    rocsparse_matrix_type_symmetric,
    rocsparse_matrix_type_hermitian,
    rocsparse_matrix_type_triangular);

DEF(rocsparse_diag_type, rocsparse_diag_type_non_unit, rocsparse_diag_type_unit);

DEF(rocsparse_fill_mode, rocsparse_fill_mode_lower, rocsparse_fill_mode_upper);

DEF(rocsparse_storage_mode, rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted);

DEF(rocsparse_action, rocsparse_action_symbolic, rocsparse_action_numeric);

DEF(rocsparse_hyb_partition,
    rocsparse_hyb_partition_auto,
    rocsparse_hyb_partition_user,
    rocsparse_hyb_partition_max);

DEF(rocsparse_analysis_policy, rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force);

DEF(rocsparse_solve_policy, rocsparse_solve_policy_auto);

DEF(rocsparse_direction, rocsparse_direction_row, rocsparse_direction_column);

DEF(rocsparse_order, rocsparse_order_row, rocsparse_order_column);

DEF(rocsparse_format,
    rocsparse_format_coo,
    rocsparse_format_coo_aos,
    rocsparse_format_csr,
    rocsparse_format_bsr,
    rocsparse_format_csc,
    rocsparse_format_ell,
    rocsparse_format_bell);

DEF(rocsparse_sddmm_alg, rocsparse_sddmm_alg_default, rocsparse_sddmm_alg_dense);

DEF(rocsparse_itilu0_alg,
    rocsparse_itilu0_alg_default,
    rocsparse_itilu0_alg_async_inplace,
    rocsparse_itilu0_alg_async_split,
    rocsparse_itilu0_alg_sync_split,
    rocsparse_itilu0_alg_sync_split_fusion);

DEF(rocsparse_spmv_alg,
    rocsparse_spmv_alg_default,
    rocsparse_spmv_alg_bsr,
    rocsparse_spmv_alg_coo,
    rocsparse_spmv_alg_csr_adaptive,
    rocsparse_spmv_alg_csr_rowsplit,
    rocsparse_spmv_alg_ell,
    rocsparse_spmv_alg_coo_atomic,
    rocsparse_spmv_alg_csr_lrb,
    rocsparse_spmv_alg_csr_nnzsplit);

DEF(rocsparse_spsv_alg, rocsparse_spsv_alg_default);
DEF(rocsparse_sptrsv_alg, rocsparse_sptrsv_alg_default);

DEF(rocsparse_spitsv_alg, rocsparse_spitsv_alg_default);

DEF(rocsparse_spsm_alg, rocsparse_spsm_alg_default);
DEF(rocsparse_sptrsm_alg, rocsparse_sptrsm_alg_default);

DEF(rocsparse_spmm_alg,
    rocsparse_spmm_alg_default,
    rocsparse_spmm_alg_bsr,
    rocsparse_spmm_alg_csr,
    rocsparse_spmm_alg_coo_segmented,
    rocsparse_spmm_alg_coo_atomic,
    rocsparse_spmm_alg_bell,
    rocsparse_spmm_alg_coo_segmented_atomic,
    rocsparse_spmm_alg_csr_row_split,
    rocsparse_spmm_alg_csr_merge);

DEF(rocsparse_spgemm_alg, rocsparse_spgemm_alg_default);

DEF(rocsparse_spgeam_alg, rocsparse_spgeam_alg_default);

DEF(rocsparse_sparse_to_dense_alg, rocsparse_sparse_to_dense_alg_default);

DEF(rocsparse_dense_to_sparse_alg, rocsparse_dense_to_sparse_alg_default);

DEF(rocsparse_gtsv_interleaved_alg,
    rocsparse_gtsv_interleaved_alg_default,
    rocsparse_gtsv_interleaved_alg_thomas,
    rocsparse_gtsv_interleaved_alg_lu,
    rocsparse_gtsv_interleaved_alg_qr);

DEF(rocsparse_gpsv_interleaved_alg,
    rocsparse_gpsv_interleaved_alg_default,
    rocsparse_gpsv_interleaved_alg_qr);

#define CASE(VALUE)               \
    case VALUE:                   \
    {                             \
        if(!strcmp(name, #VALUE)) \
        {                         \
            value = VALUE;        \
            return true;          \
        }                         \
        break;                    \
    }

#define RETURN_INVALID return true;

bool rocsparse_indextype_from_name(rocsparse_indextype& value, const char* name)
{
    for(auto v : rocsparse_indextype_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_indextype_u16);
            CASE(rocsparse_indextype_i32);
            CASE(rocsparse_indextype_i64);
        }
    }
    return false;
}

bool rocsparse_datatype_from_name(rocsparse_datatype value, const char* name)
{
    for(auto v : rocsparse_datatype_st::all)
    {
        switch(v)
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
    }
    return false;
}

bool rocsparse_index_base_from_name(rocsparse_index_base value, const char* name)
{
    for(auto v : rocsparse_index_base_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_index_base_zero);
            CASE(rocsparse_index_base_one);
        }
    }
    return false;
}

bool rocsparse_operation_from_name(rocsparse_operation value, const char* name)
{
    for(auto v : rocsparse_operation_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_operation_none);
            CASE(rocsparse_operation_transpose);
            CASE(rocsparse_operation_conjugate_transpose);
        }
    }
    return false;
}

bool rocsparse_matrix_type_from_name(rocsparse_matrix_type value, const char* name)
{
    for(auto v : rocsparse_matrix_type_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_matrix_type_general);
            CASE(rocsparse_matrix_type_symmetric);
            CASE(rocsparse_matrix_type_hermitian);
            CASE(rocsparse_matrix_type_triangular);
        }
    }
    return false;
}

bool rocsparse_diag_type_from_name(rocsparse_diag_type value, const char* name)
{
    for(auto v : rocsparse_diag_type_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_diag_type_non_unit);
            CASE(rocsparse_diag_type_unit);
        }
    }
    return false;
}

bool rocsparse_fill_mode_from_name(rocsparse_fill_mode value, const char* name)
{
    for(auto v : rocsparse_fill_mode_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_fill_mode_lower);
            CASE(rocsparse_fill_mode_upper);
        }
    }
    return false;
}

bool rocsparse_storage_mode_from_name(rocsparse_storage_mode value, const char* name)
{
    for(auto v : rocsparse_storage_mode_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_storage_mode_sorted);
            CASE(rocsparse_storage_mode_unsorted);
        }
    }
    return false;
}

bool rocsparse_action_from_name(rocsparse_action value, const char* name)
{
    for(auto v : rocsparse_action_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_action_symbolic);
            CASE(rocsparse_action_numeric);
        }
    }
    return false;
}

bool rocsparse_hyb_partition_from_name(rocsparse_hyb_partition value, const char* name)
{
    for(auto v : rocsparse_hyb_partition_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_hyb_partition_auto);
            CASE(rocsparse_hyb_partition_user);
            CASE(rocsparse_hyb_partition_max);
        }
    }
    return false;
}

bool rocsparse_analysis_policy_from_name(rocsparse_analysis_policy value, const char* name)
{
    for(auto v : rocsparse_analysis_policy_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_analysis_policy_reuse);
            CASE(rocsparse_analysis_policy_force);
        }
    }
    return false;
}

bool rocsparse_solve_policy_from_name(rocsparse_solve_policy value, const char* name)
{
    for(auto v : rocsparse_solve_policy_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_solve_policy_auto);
        }
    }
    return false;
}

bool rocsparse_direction_from_name(rocsparse_direction value, const char* name)
{
    for(auto v : rocsparse_direction_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_direction_row);
            CASE(rocsparse_direction_column);
        }
    }
    return false;
}

bool rocsparse_order_from_name(rocsparse_order value, const char* name)
{
    for(auto v : rocsparse_order_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_order_row);
            CASE(rocsparse_order_column);
        }
    }
    return false;
}

bool rocsparse_format_from_name(rocsparse_format value, const char* name)
{
    for(auto v : rocsparse_format_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_format_coo);
            CASE(rocsparse_format_coo_aos);
            CASE(rocsparse_format_csr);
            CASE(rocsparse_format_bsr);
            CASE(rocsparse_format_csc);
            CASE(rocsparse_format_ell);
            CASE(rocsparse_format_bell);
        }
    }
    return false;
}

bool rocsparse_sddmm_alg_from_name(rocsparse_sddmm_alg value, const char* name)
{
    for(auto v : rocsparse_sddmm_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_sddmm_alg_default);
            CASE(rocsparse_sddmm_alg_dense);
        }
    }
    return false;
}

bool rocsparse_itilu0_alg_from_name(rocsparse_itilu0_alg value, const char* name)
{
    for(auto v : rocsparse_itilu0_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_itilu0_alg_default);
            CASE(rocsparse_itilu0_alg_async_inplace);
            CASE(rocsparse_itilu0_alg_async_split);
            CASE(rocsparse_itilu0_alg_sync_split);
            CASE(rocsparse_itilu0_alg_sync_split_fusion);
        }
    }
    return false;
}

bool rocsparse_spmv_alg_from_name(rocsparse_spmv_alg value, const char* name)
{
    for(auto v : rocsparse_spmv_alg_st::all)
    {
        switch(v)
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
    }
    return false;
}

bool rocsparse_spsv_alg_from_name(rocsparse_spsv_alg value, const char* name)
{
    for(auto v : rocsparse_spsv_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_spsv_alg_default);
        }
    }
    return false;
}

bool rocsparse_sptrsv_alg_from_name(rocsparse_sptrsv_alg value, const char* name)
{
    for(auto v : rocsparse_sptrsv_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_sptrsv_alg_default);
        }
    }
    return false;
}

bool rocsparse_spitsv_alg_from_name(rocsparse_spitsv_alg value, const char* name)
{
    for(auto v : rocsparse_spitsv_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_spitsv_alg_default);
        }
    }
    return false;
}

bool rocsparse_spsm_alg_from_name(rocsparse_spsm_alg value, const char* name)
{
    for(auto v : rocsparse_spsm_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_spsm_alg_default);
        }
    }
    return false;
}

bool rocsparse_sptrsm_alg_from_name(rocsparse_sptrsm_alg value, const char* name)
{
    for(auto v : rocsparse_sptrsm_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_sptrsm_alg_default);
        }
    }
    return false;
}

bool rocsparse_spmm_alg_from_name(rocsparse_spmm_alg value, const char* name)
{
    for(auto v : rocsparse_spmm_alg_st::all)
    {
        switch(v)
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
    }
    return false;
}

bool rocsparse_spgemm_alg_from_name(rocsparse_spgemm_alg value, const char* name)
{
    for(auto v : rocsparse_spgemm_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_spgemm_alg_default);
        }
    }
    return false;
}

bool rocsparse_spgeam_alg_from_name(rocsparse_spgeam_alg value, const char* name)
{
    for(auto v : rocsparse_spgeam_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_spgeam_alg_default);
        }
    }
    return false;
}

bool rocsparse_sparse_to_dense_alg_from_name(rocsparse_sparse_to_dense_alg value, const char* name)
{
    for(auto v : rocsparse_sparse_to_dense_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_sparse_to_dense_alg_default);
        }
    }
    return false;
}

bool rocsparse_dense_to_sparse_alg_from_name(rocsparse_dense_to_sparse_alg value, const char* name)
{
    for(auto v : rocsparse_dense_to_sparse_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_dense_to_sparse_alg_default);
        }
    }
    return false;
}

bool rocsparse_gtsv_interleaved_alg_from_name(rocsparse_gtsv_interleaved_alg value,
                                              const char*                    name)
{
    for(auto v : rocsparse_gtsv_interleaved_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_gtsv_interleaved_alg_default);
            CASE(rocsparse_gtsv_interleaved_alg_thomas);
            CASE(rocsparse_gtsv_interleaved_alg_lu);
            CASE(rocsparse_gtsv_interleaved_alg_qr);
        }
    }
    return false;
}

bool rocsparse_gpsv_interleaved_alg_from_name(rocsparse_gpsv_interleaved_alg value,
                                              const char*                    name)
{
    for(auto v : rocsparse_gpsv_interleaved_alg_st::all)
    {
        switch(v)
        {
            CASE(rocsparse_gpsv_interleaved_alg_default);
            CASE(rocsparse_gpsv_interleaved_alg_qr);
        }
    }
    return false;
}

#undef CASE
#undef RETURN_INVALID
#undef DEF

#define DEF(TOKEN, ...)                                \
    do                                                 \
    {                                                  \
        if(!strcmp(enum_type, #TOKEN))                 \
        {                                              \
            TOKEN tok = (TOKEN)0;                      \
            bool  is  = TOKEN##_from_name(tok, value); \
            if(is == false)                            \
                return false;                          \
            v = (int)tok;                              \
            return true;                               \
        }                                              \
    } while(false)

bool convert(int& v, const char* enum_type, const char* value)
{
    DEF(rocsparse_indextype);
    DEF(rocsparse_datatype);
    DEF(rocsparse_index_base);
    DEF(rocsparse_operation);
    DEF(rocsparse_matrix_type);
    DEF(rocsparse_diag_type);
    DEF(rocsparse_fill_mode);
    DEF(rocsparse_storage_mode);
    DEF(rocsparse_action);
    DEF(rocsparse_hyb_partition);
    DEF(rocsparse_analysis_policy);
    DEF(rocsparse_solve_policy);
    DEF(rocsparse_direction);
    DEF(rocsparse_order);
    DEF(rocsparse_format);
    DEF(rocsparse_sddmm_alg);
    DEF(rocsparse_itilu0_alg);
    DEF(rocsparse_spmv_alg);
    DEF(rocsparse_spsv_alg);
    DEF(rocsparse_sptrsv_alg);
    DEF(rocsparse_spitsv_alg);
    DEF(rocsparse_spsm_alg);
    DEF(rocsparse_sptrsm_alg);
    DEF(rocsparse_spmm_alg);
    DEF(rocsparse_spgemm_alg);
    DEF(rocsparse_spgeam_alg);
    DEF(rocsparse_sparse_to_dense_alg);
    DEF(rocsparse_dense_to_sparse_alg);
    DEF(rocsparse_gtsv_interleaved_alg);
    DEF(rocsparse_gpsv_interleaved_alg);
    return false;
}

#undef DEF

#define DEF(TOKEN, ...)                \
    do                                 \
    {                                  \
        if(!strcmp(enum_type, #TOKEN)) \
        {                              \
            v = TOKEN##_st::s;         \
            return true;               \
        }                              \
    } while(false)

bool get_size(uint64_t& v, const char* enum_type)
{
    DEF(rocsparse_indextype);
    DEF(rocsparse_datatype);
    DEF(rocsparse_index_base);
    DEF(rocsparse_operation);
    DEF(rocsparse_matrix_type);
    DEF(rocsparse_diag_type);
    DEF(rocsparse_fill_mode);
    DEF(rocsparse_storage_mode);
    DEF(rocsparse_action);
    DEF(rocsparse_hyb_partition);
    DEF(rocsparse_analysis_policy);
    DEF(rocsparse_solve_policy);
    DEF(rocsparse_direction);
    DEF(rocsparse_order);
    DEF(rocsparse_format);
    DEF(rocsparse_sddmm_alg);
    DEF(rocsparse_itilu0_alg);
    DEF(rocsparse_spmv_alg);
    DEF(rocsparse_spsv_alg);
    DEF(rocsparse_sptrsv_alg);
    DEF(rocsparse_spitsv_alg);
    DEF(rocsparse_spsm_alg);
    DEF(rocsparse_sptrsm_alg);
    DEF(rocsparse_spmm_alg);
    DEF(rocsparse_spgemm_alg);
    DEF(rocsparse_spgeam_alg);
    DEF(rocsparse_sparse_to_dense_alg);
    DEF(rocsparse_dense_to_sparse_alg);
    DEF(rocsparse_gtsv_interleaved_alg);
    DEF(rocsparse_gpsv_interleaved_alg);
    return false;
}

#undef DEF

#define DEF(TOKEN, ...)                                         \
    do                                                          \
    {                                                           \
        if(!strcmp(enum_type, #TOKEN))                          \
        {                                                       \
            return rocsparse_enum_name(TOKEN##_st::all[index]); \
        }                                                       \
    } while(false)

const char* get_name(const char* enum_type, uint64_t index)
{
    DEF(rocsparse_indextype);
    DEF(rocsparse_datatype);
    DEF(rocsparse_index_base);
    DEF(rocsparse_operation);
    DEF(rocsparse_matrix_type);
    DEF(rocsparse_diag_type);
    DEF(rocsparse_fill_mode);
    DEF(rocsparse_storage_mode);
    DEF(rocsparse_action);
    DEF(rocsparse_hyb_partition);
    DEF(rocsparse_analysis_policy);
    DEF(rocsparse_solve_policy);
    DEF(rocsparse_direction);
    DEF(rocsparse_order);
    DEF(rocsparse_format);
    DEF(rocsparse_sddmm_alg);
    DEF(rocsparse_itilu0_alg);
    DEF(rocsparse_spmv_alg);
    DEF(rocsparse_spsv_alg);
    DEF(rocsparse_sptrsv_alg);
    DEF(rocsparse_spitsv_alg);
    DEF(rocsparse_spsm_alg);
    DEF(rocsparse_sptrsm_alg);
    DEF(rocsparse_spmm_alg);
    DEF(rocsparse_spgemm_alg);
    DEF(rocsparse_spgeam_alg);
    DEF(rocsparse_sparse_to_dense_alg);
    DEF(rocsparse_dense_to_sparse_alg);
    DEF(rocsparse_gtsv_interleaved_alg);
    DEF(rocsparse_gpsv_interleaved_alg);
    return nullptr;
}

#undef DEF
