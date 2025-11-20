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

#pragma once

#include "rocsparse-types.h"

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_indextype_name(rocsparse_indextype value);
///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_datatype_name(rocsparse_datatype value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_index_base_name(rocsparse_index_base value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_operation_name(rocsparse_operation value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_matrix_type_name(rocsparse_matrix_type value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_diag_type_name(rocsparse_diag_type value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_fill_mode_name(rocsparse_fill_mode value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_storage_mode_name(rocsparse_storage_mode value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_action_name(rocsparse_action value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_hyb_partition_name(rocsparse_hyb_partition value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_analysis_policy_name(rocsparse_analysis_policy value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_solve_policy_name(rocsparse_solve_policy value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_direction_name(rocsparse_direction value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_order_name(rocsparse_order value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_format_name(rocsparse_format value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_sddmm_alg_name(rocsparse_sddmm_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_itilu0_alg_name(rocsparse_itilu0_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spmv_alg_name(rocsparse_spmv_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spsv_alg_name(rocsparse_spsv_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_sptrsv_alg_name(rocsparse_sptrsv_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spitsv_alg_name(rocsparse_spitsv_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spsm_alg_name(rocsparse_spsm_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_sptrsm_alg_name(rocsparse_sptrsm_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spmm_alg_name(rocsparse_spmm_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spgemm_alg_name(rocsparse_spgemm_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_spgeam_alg_name(rocsparse_spgeam_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_sparse_to_dense_alg_name(rocsparse_sparse_to_dense_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_dense_to_sparse_alg_name(rocsparse_dense_to_sparse_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_gtsv_interleaved_alg_name(rocsparse_gtsv_interleaved_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
const char* rocsparse_gpsv_interleaved_alg_name(rocsparse_gpsv_interleaved_alg value);

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_indextype value)
{
    return rocsparse_indextype_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_datatype value)
{
    return rocsparse_datatype_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_index_base value)
{
    return rocsparse_index_base_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_operation value)
{
    return rocsparse_operation_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_matrix_type value)
{
    return rocsparse_matrix_type_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_diag_type value)
{
    return rocsparse_diag_type_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_fill_mode value)
{
    return rocsparse_fill_mode_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_storage_mode value)
{
    return rocsparse_storage_mode_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_action value)
{
    return rocsparse_action_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_hyb_partition value)
{
    return rocsparse_hyb_partition_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_analysis_policy value)
{
    return rocsparse_analysis_policy_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_solve_policy value)
{
    return rocsparse_solve_policy_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_direction value)
{
    return rocsparse_direction_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_order value)
{
    return rocsparse_order_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_format value)
{
    return rocsparse_format_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_sddmm_alg value)
{
    return rocsparse_sddmm_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_itilu0_alg value)
{
    return rocsparse_itilu0_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spmv_alg value)
{
    return rocsparse_spmv_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spsv_alg value)
{
    return rocsparse_spsv_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_sptrsv_alg value)
{
    return rocsparse_sptrsv_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spitsv_alg value)
{
    return rocsparse_spitsv_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spsm_alg value)
{
    return rocsparse_spsm_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_sptrsm_alg value)
{
    return rocsparse_sptrsm_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spmm_alg value)
{
    return rocsparse_spmm_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spgemm_alg value)
{
    return rocsparse_spgemm_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_spgeam_alg value)
{
    return rocsparse_spgeam_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_sparse_to_dense_alg value)
{
    return rocsparse_sparse_to_dense_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_dense_to_sparse_alg value)
{
    return rocsparse_dense_to_sparse_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_gtsv_interleaved_alg value)
{
    return rocsparse_gtsv_interleaved_alg_name(value);
}

///
/// @brief Get the litteral name of the enumeration.
/// @param[in] value value of the enumeration.
/// @return The litteral name of the enumeration.
///
inline const char* rocsparse_enum_name(rocsparse_gpsv_interleaved_alg value)
{
    return rocsparse_gpsv_interleaved_alg_name(value);
}
