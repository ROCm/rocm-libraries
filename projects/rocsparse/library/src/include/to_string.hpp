/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    const char* to_string(rocsparse_status status);
    const char* to_string(rocsparse_matrix_type type);
    const char* to_string(rocsparse_data_status data_status);
    const char* to_string(rocsparse_sparse_to_sparse_stage value);
    const char* to_string(rocsparse_sparse_to_sparse_alg value);
    const char* to_string(rocsparse_pointer_mode value);
    const char* to_string(rocsparse_spmat_attribute value);
    const char* to_string(rocsparse_itilu0_alg value);
    const char* to_string(rocsparse_diag_type value);
    const char* to_string(rocsparse_fill_mode value_);
    const char* to_string(rocsparse_storage_mode value_);
    const char* to_string(rocsparse_index_base value_);
    const char* to_string(rocsparse_matrix_type value_);
    const char* to_string(rocsparse_direction value_);
    const char* to_string(rocsparse_operation value_);
    const char* to_string(rocsparse_indextype value_);
    const char* to_string(rocsparse_datatype value_);
    const char* to_string(rocsparse_order value_);
    const char* to_string(rocsparse_action value);
    const char* to_string(rocsparse_hyb_partition value);
    const char* to_string(rocsparse_gtsv_interleaved_alg value_);
    const char* to_string(rocsparse_sparse_to_dense_alg value_);
    const char* to_string(rocsparse_dense_to_sparse_alg value_);
    const char* to_string(rocsparse_spmv_alg value_);
    const char* to_string(rocsparse_spsv_alg value_);
    const char* to_string(rocsparse_spitsv_alg value_);
    const char* to_string(rocsparse_check_spmat_stage value_);
    const char* to_string(rocsparse_spmv_stage value_);
    const char* to_string(rocsparse_spsv_stage value_);
    const char* to_string(rocsparse_spitsv_stage value_);
    const char* to_string(rocsparse_spsm_alg value_);
    const char* to_string(rocsparse_spsm_stage value_);
    const char* to_string(rocsparse_spmm_alg value_);
    const char* to_string(rocsparse_spmm_stage value_);
    const char* to_string(rocsparse_sddmm_alg value_);
    const char* to_string(rocsparse_spgemm_alg value_);
    const char* to_string(rocsparse_spgemm_stage value_);
    const char* to_string(rocsparse_spgeam_alg value_);
    const char* to_string(rocsparse_spgeam_stage value_);
    const char* to_string(rocsparse_spgeam_input value_);
    const char* to_string(rocsparse_spgeam_output value_);
    const char* to_string(rocsparse_solve_policy value_);
    const char* to_string(rocsparse_analysis_policy value_);
    const char* to_string(rocsparse_format value_);
}
