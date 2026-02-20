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
 *  \brief auto_testing_bad_arg_set_invalid.hpp maps bad values with types.
 */

#pragma once
#include "rocsparse-types.h"

//
// PROVIDE TEMPLATE FOR AUTO TESTING BAD ARGUMENTS
//

template <typename T>
inline void auto_testing_bad_arg_set_invalid(T& p);

template <typename T>
inline void auto_testing_bad_arg_set_invalid(T*& p)
{
    p = nullptr;
}

template <>
inline void auto_testing_bad_arg_set_invalid(int32_t& p)
{
    p = -1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(size_t& p)
{
}

template <>
inline void auto_testing_bad_arg_set_invalid(int64_t& p)
{
    p = -1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(float& p)
{
    p = static_cast<float>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(double& p)
{
    p = static_cast<double>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_float_complex& p)
{
    p = static_cast<rocsparse_float_complex>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_double_complex& p)
{
    p = static_cast<rocsparse_double_complex>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_operation& p)
{
    p = (rocsparse_operation)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_order& p)
{
    p = (rocsparse_order)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_index_base& p)
{
    p = (rocsparse_index_base)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_matrix_type& p)
{
    p = (rocsparse_matrix_type)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_fill_mode& p)
{
    p = (rocsparse_fill_mode)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_storage_mode& p)
{
    p = (rocsparse_storage_mode)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_gtsv_interleaved_alg& p)
{
    p = (rocsparse_gtsv_interleaved_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sparse_to_dense_alg& p)
{
    p = (rocsparse_sparse_to_dense_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_dense_to_sparse_alg& p)
{
    p = (rocsparse_dense_to_sparse_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmv_alg& p)
{
    p = (rocsparse_spmv_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsv_alg& p)
{
    p = (rocsparse_spsv_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsv_alg& p)
{
    p = (rocsparse_sptrsv_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spitsv_alg& p)
{
    p = (rocsparse_spitsv_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsm_alg& p)
{
    p = (rocsparse_spsm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsm_alg& p)
{
    p = (rocsparse_sptrsm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sddmm_alg& p)
{
    p = (rocsparse_sddmm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmm_alg& p)
{
    p = (rocsparse_spmm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_gpsv_interleaved_alg& p)
{
    p = (rocsparse_gpsv_interleaved_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spgemm_alg& p)
{
    p = (rocsparse_spgemm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spgeam_alg& p)
{
    p = (rocsparse_spgeam_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_itilu0_alg& p)
{
    p = (rocsparse_itilu0_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_indextype& p)
{
    p = (rocsparse_indextype)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_datatype& p)
{
    p = (rocsparse_datatype)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_analysis_policy& p)
{
    p = (rocsparse_analysis_policy)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_direction& p)
{
    p = (rocsparse_direction)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_action& p)
{
    p = (rocsparse_action)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_hyb_partition& p)
{
    p = (rocsparse_hyb_partition)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_solve_policy& p)
{
    p = (rocsparse_solve_policy)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sparse_to_sparse_alg& p)
{
    p = (rocsparse_sparse_to_sparse_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sparse_to_sparse_stage& p)
{
    p = (rocsparse_sparse_to_sparse_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_extract_alg& p)
{
    p = (rocsparse_extract_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_extract_stage& p)
{
    p = (rocsparse_extract_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_check_spmat_stage& p)
{
    p = (rocsparse_check_spmat_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmv_stage& p)
{
    p = (rocsparse_spmv_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_v2_spmv_stage& p)
{
    p = (rocsparse_v2_spmv_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsv_stage& p)
{
    p = (rocsparse_spsv_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsv_stage& p)
{
    p = (rocsparse_sptrsv_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spitsv_stage& p)
{
    p = (rocsparse_spitsv_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsm_stage& p)
{
    p = (rocsparse_spsm_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsm_stage& p)
{
    p = (rocsparse_sptrsm_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmm_stage& p)
{
    p = (rocsparse_spmm_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spgemm_stage& p)
{
    p = (rocsparse_spgemm_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spgeam_stage& p)
{
    p = (rocsparse_spgeam_stage)-1;
}
