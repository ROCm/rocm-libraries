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

/*! \file
 *  \brief auto_testing_bad_arg_get_status.hpp maps invalid status with types.
 */

#pragma once
#include "rocsparse-types.h"

template <typename T>
inline rocsparse_status auto_testing_bad_arg_get_status(T& p)
{
    return rocsparse_status_invalid_pointer;
}

template <typename T>
inline rocsparse_status auto_testing_bad_arg_get_status(const T& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_handle& p)
{
    return rocsparse_status_invalid_handle;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmat_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spvec_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_dnmat_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_dnvec_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(int32_t& p)
{
    return rocsparse_status_invalid_size;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(size_t& p)
{
    return rocsparse_status_invalid_size;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(int64_t& p)
{
    return rocsparse_status_invalid_size;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_operation& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_order& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_index_base& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_matrix_type& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_fill_mode& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_storage_mode& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_gtsv_interleaved_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sparse_to_dense_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_dense_to_sparse_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmv_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsv_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsv_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spitsv_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sddmm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_gpsv_interleaved_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spgemm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spgeam_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_itilu0_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_indextype& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_datatype& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_analysis_policy& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_solve_policy& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_direction& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_action& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_hyb_partition& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sparse_to_sparse_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sparse_to_sparse_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_extract_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_extract_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_check_spmat_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmv_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_v2_spmv_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsv_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsv_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spitsv_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spgemm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spgeam_stage& p)
{
    return rocsparse_status_invalid_value;
}
