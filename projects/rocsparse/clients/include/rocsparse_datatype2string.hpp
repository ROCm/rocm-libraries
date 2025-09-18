/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef ROCSPARSE_DATATYPE2STRING_HPP
#define ROCSPARSE_DATATYPE2STRING_HPP

#include "rocsparse.h"
#include <string>

#include <algorithm>

typedef enum rocsparse_matrix_init_kind_
{
    rocsparse_matrix_init_kind_default  = 0,
    rocsparse_matrix_init_kind_tunedavg = 1
} rocsparse_matrix_init_kind;

constexpr auto rocsparse_matrix_init_kind2string(rocsparse_matrix_init_kind matrix)
{
    switch(matrix)
    {
    case rocsparse_matrix_init_kind_default:
        return "default";
    case rocsparse_matrix_init_kind_tunedavg:
        return "tunedavg";
    }
    return "invalid";
}

typedef enum rocsparse_matrix_init_
{
    rocsparse_matrix_random           = 0, /**< Random initialization */
    rocsparse_matrix_laplace_2d       = 1, /**< Initialize 2D laplacian matrix */
    rocsparse_matrix_laplace_3d       = 2, /**< Initialize 3D laplacian matrix */
    rocsparse_matrix_file_mtx         = 3, /**< Read from matrix market file */
    rocsparse_matrix_file_smtx        = 4, /**< Read from machine learning csr file */
    rocsparse_matrix_file_bsmtx       = 5, /**< Read from machine learning bsr file */
    rocsparse_matrix_file_rocalution  = 6, /**< Read from rocalution file */
    rocsparse_matrix_zero             = 7, /**< Generates zero matrix */
    rocsparse_matrix_file_rocsparseio = 8, /**< Read from rocsparseio file */
    rocsparse_matrix_tridiagonal      = 9, /**< Initialize tridiagonal matrix */
    rocsparse_matrix_pentadiagonal    = 10 /**< Initialize pentadiagonal matrix */
} rocsparse_matrix_init;

constexpr auto rocsparse_matrix2string(rocsparse_matrix_init matrix)
{
    switch(matrix)
    {
    case rocsparse_matrix_random:
        return "rand";
    case rocsparse_matrix_laplace_2d:
        return "L2D";
    case rocsparse_matrix_laplace_3d:
        return "L3D";
    case rocsparse_matrix_file_mtx:
        return "mtx";
    case rocsparse_matrix_file_smtx:
        return "smtx";
    case rocsparse_matrix_file_bsmtx:
        return "bsmtx";
    case rocsparse_matrix_file_rocalution:
        return "csr";
    case rocsparse_matrix_zero:
        return "zero";
    case rocsparse_matrix_file_rocsparseio:
        return "bin";
    case rocsparse_matrix_tridiagonal:
        return "tri";
    case rocsparse_matrix_pentadiagonal:
        return "penta";
    }
    return "invalid";
}

constexpr auto rocsparse_indextype2string(rocsparse_indextype type)
{
    switch(type)
    {
    case rocsparse_indextype_u16:
        return "u16";
    case rocsparse_indextype_i32:
        return "i32";
    case rocsparse_indextype_i64:
        return "i64";
    }
    return "invalid";
}

constexpr auto rocsparse_datatype2string(rocsparse_datatype type)
{
    switch(type)
    {
    case rocsparse_datatype_f16_r:
        return "f16_r";
    case rocsparse_datatype_f32_r:
        return "f32_r";
    case rocsparse_datatype_f64_r:
        return "f64_r";
    case rocsparse_datatype_f32_c:
        return "f32_c";
    case rocsparse_datatype_f64_c:
        return "f64_c";
    case rocsparse_datatype_i8_r:
        return "i8_r";
    case rocsparse_datatype_u8_r:
        return "u8_r";
    case rocsparse_datatype_i32_r:
        return "i32_r";
    case rocsparse_datatype_u32_r:
        return "u32_r";
    case rocsparse_datatype_bf16_r:
        return "bf16_r";
    }
    return "invalid";
}

constexpr auto rocsparse_indexbase2string(rocsparse_index_base base)
{
    switch(base)
    {
    case rocsparse_index_base_zero:
        return "0b";
    case rocsparse_index_base_one:
        return "1b";
    }
    return "invalid";
}

constexpr auto rocsparse_operation2string(rocsparse_operation trans)
{
    switch(trans)
    {
    case rocsparse_operation_none:
        return "NT";
    case rocsparse_operation_transpose:
        return "T";
    case rocsparse_operation_conjugate_transpose:
        return "CT";
    }
    return "invalid";
}

constexpr auto rocsparse_matrixtype2string(rocsparse_matrix_type type)
{
    switch(type)
    {
    case rocsparse_matrix_type_general:
        return "general";
    case rocsparse_matrix_type_symmetric:
        return "symmetric";
    case rocsparse_matrix_type_hermitian:
        return "hermitian";
    case rocsparse_matrix_type_triangular:
        return "triangular";
    }
    return "invalid";
}

constexpr auto rocsparse_diagtype2string(rocsparse_diag_type diag)
{
    switch(diag)
    {
    case rocsparse_diag_type_non_unit:
        return "ND";
    case rocsparse_diag_type_unit:
        return "UD";
    }
    return "invalid";
}

constexpr auto rocsparse_fillmode2string(rocsparse_fill_mode uplo)
{
    switch(uplo)
    {
    case rocsparse_fill_mode_lower:
        return "L";
    case rocsparse_fill_mode_upper:
        return "U";
    }
    return "invalid";
}

constexpr auto rocsparse_storagemode2string(rocsparse_storage_mode storage)
{
    switch(storage)
    {
    case rocsparse_storage_mode_sorted:
        return "sorted";
    case rocsparse_storage_mode_unsorted:
        return "unsorted";
    }
    return "invalid";
}

constexpr auto rocsparse_action2string(rocsparse_action action)
{
    switch(action)
    {
    case rocsparse_action_symbolic:
        return "sym";
    case rocsparse_action_numeric:
        return "num";
    }
    return "invalid";
}

constexpr auto rocsparse_partition2string(rocsparse_hyb_partition part)
{
    switch(part)
    {
    case rocsparse_hyb_partition_auto:
        return "auto";
    case rocsparse_hyb_partition_user:
        return "user";
    case rocsparse_hyb_partition_max:
        return "max";
    }
    return "invalid";
}

constexpr auto rocsparse_analysis2string(rocsparse_analysis_policy policy)
{
    switch(policy)
    {
    case rocsparse_analysis_policy_reuse:
        return "reuse";
    case rocsparse_analysis_policy_force:
        return "force";
    }
    return "invalid";
}

constexpr auto rocsparse_solve2string(rocsparse_solve_policy policy)
{
    switch(policy)
    {
    case rocsparse_solve_policy_auto:
        return "auto";
    }
    return "invalid";
}

constexpr auto rocsparse_direction2string(rocsparse_direction direction)
{
    switch(direction)
    {
    case rocsparse_direction_row:
        return "row";
    case rocsparse_direction_column:
        return "column";
    }
    return "invalid";
}

constexpr auto rocsparse_order2string(rocsparse_order order)
{
    switch(order)
    {
    case rocsparse_order_row:
        return "row";
    case rocsparse_order_column:
        return "col";
    }
    return "invalid";
}

constexpr auto rocsparse_format2string(rocsparse_format format)
{
    switch(format)
    {
    case rocsparse_format_coo:
        return "coo";
    case rocsparse_format_coo_aos:
        return "coo_aos";
    case rocsparse_format_csr:
        return "csr";
    case rocsparse_format_bsr:
        return "bsr";
    case rocsparse_format_csc:
        return "csc";
    case rocsparse_format_ell:
        return "ell";
    case rocsparse_format_bell:
        return "bell";
    }
    return "invalid";
}

constexpr auto rocsparse_sddmmalg2string(rocsparse_sddmm_alg alg)
{
    switch(alg)
    {
    case rocsparse_sddmm_alg_default:
        return "default";
    case rocsparse_sddmm_alg_dense:
        return "dense";
    }
    return "invalid";
}

constexpr auto rocsparse_itilu0alg2string(rocsparse_itilu0_alg alg)
{
    switch(alg)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    {
        return "async_inplace";
    }
    case rocsparse_itilu0_alg_async_split:
    {
        return "async_split";
    }
    case rocsparse_itilu0_alg_sync_split:
    {
        return "sync_split";
    }
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return "sync_split_fusion";
    }
    }
    return "invalid";
}

constexpr auto rocsparse_spmvalg2string(rocsparse_spmv_alg alg)
{
    switch(alg)
    {
    case rocsparse_spmv_alg_default:
        return "default";
    case rocsparse_spmv_alg_bsr:
        return "bsr";
    case rocsparse_spmv_alg_coo:
        return "coo";
    case rocsparse_spmv_alg_csr_adaptive:
        return "csradaptive";
    case rocsparse_spmv_alg_csr_rowsplit:
        return "csrrowsplit";
    case rocsparse_spmv_alg_ell:
        return "ell";
    case rocsparse_spmv_alg_coo_atomic:
        return "cooatomic";
    case rocsparse_spmv_alg_csr_lrb:
        return "csrlrb";
    case rocsparse_spmv_alg_csr_nnzsplit:
        return "csrnnzsplit";
    }
    return "invalid";
}

constexpr auto rocsparse_spsvalg2string(rocsparse_spsv_alg alg)
{
    switch(alg)
    {
    case rocsparse_spsv_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_sptrsvalg2string(rocsparse_sptrsv_alg alg)
{
    switch(alg)
    {
    case rocsparse_sptrsv_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_spitsvalg2string(rocsparse_spitsv_alg alg)
{
    switch(alg)
    {
    case rocsparse_spitsv_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_spsmalg2string(rocsparse_spsm_alg alg)
{
    switch(alg)
    {
    case rocsparse_spsm_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_sptrsmalg2string(rocsparse_sptrsm_alg alg)
{
    switch(alg)
    {
    case rocsparse_sptrsm_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_spmmalg2string(rocsparse_spmm_alg alg)
{
    switch(alg)
    {
    case rocsparse_spmm_alg_default:
        return "default";
    case rocsparse_spmm_alg_bsr:
        return "bsr";
    case rocsparse_spmm_alg_csr:
        return "csr";
    case rocsparse_spmm_alg_coo_segmented:
        return "coosegmented";
    case rocsparse_spmm_alg_coo_atomic:
        return "cooatomic";
    case rocsparse_spmm_alg_bell:
        return "bell";
    case rocsparse_spmm_alg_coo_segmented_atomic:
        return "coosegmented_atomic";
    case rocsparse_spmm_alg_csr_row_split:
        return "csrrow_split";
    case rocsparse_spmm_alg_csr_nnz_split:
        return "csrnnz_split";
    case rocsparse_spmm_alg_csr_merge_path:
        return "csrmerge_path";
    }
    return "invalid";
}

constexpr auto rocsparse_spgemmalg2string(rocsparse_spgemm_alg alg)
{
    switch(alg)
    {
    case rocsparse_spgemm_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_spgeamalg2string(rocsparse_spgeam_alg alg)
{
    switch(alg)
    {
    case rocsparse_spgeam_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_sparsetodensealg2string(rocsparse_sparse_to_dense_alg alg)
{
    switch(alg)
    {
    case rocsparse_sparse_to_dense_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_densetosparsealg2string(rocsparse_dense_to_sparse_alg alg)
{
    switch(alg)
    {
    case rocsparse_dense_to_sparse_alg_default:
        return "default";
    }
    return "invalid";
}

constexpr auto rocsparse_gtsvinterleavedalg2string(rocsparse_gtsv_interleaved_alg alg)
{
    switch(alg)
    {
    case rocsparse_gtsv_interleaved_alg_default:
        return "default";
    case rocsparse_gtsv_interleaved_alg_thomas:
        return "thomas";
    case rocsparse_gtsv_interleaved_alg_lu:
        return "LU";
    case rocsparse_gtsv_interleaved_alg_qr:
        return "QR";
    }
    return "invalid";
}

constexpr auto rocsparse_gpsvalg2string(rocsparse_gpsv_interleaved_alg alg)
{
    switch(alg)
    {
    case rocsparse_gpsv_interleaved_alg_default:
        return "default";
    case rocsparse_gpsv_interleaved_alg_qr:
        return "qr";
    }
    return "invalid";
}

// Return a string without '/' or '\\'
inline std::string rocsparse_filename2string(const std::string& filename)
{
    std::string result(filename);
    std::replace(result.begin(), result.end(), '/', '_');
    std::replace(result.begin(), result.end(), '\\', '_');
    return result;
}

#endif // ROCSPARSE_DATATYPE2STRING_HPP
