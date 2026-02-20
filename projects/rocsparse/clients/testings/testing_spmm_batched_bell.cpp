/* ************************************************************************
* Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

#include <tuple>

template <typename I, typename T>
void testing_spmm_batched_bell_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    I                    m           = safe_size;
    I                    n           = safe_size;
    I                    k           = safe_size;
    I                    block_size  = safe_size;
    I                    ell_cols    = safe_size;
    void*                ell_val     = (void*)0x4;
    void*                ell_col_ind = (void*)0x4;
    void*                B           = (void*)0x4;
    void*                C           = (void*)0x4;
    size_t*              buffer_size = (size_t*)0x4;
    void*                temp_buffer = (void*)0x4;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_index_base base        = rocsparse_index_base_zero;
    rocsparse_order      order_B     = rocsparse_order_column;
    rocsparse_order      order_C     = rocsparse_order_column;
    rocsparse_direction  block_dir   = rocsparse_direction_row;
    rocsparse_spmm_alg   alg         = rocsparse_spmm_alg_default;
    rocsparse_spmm_stage stage       = rocsparse_spmm_stage_compute;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);

    // SpMM structures
    rocsparse_local_spmat local_mat_A(
        m, k, block_dir, block_size, ell_cols, ell_col_ind, ell_val, itype, base, ttype);
    rocsparse_local_dnmat local_mat_B(k, n, k, B, ttype, order_B);
    rocsparse_local_dnmat local_mat_C(m, n, m, C, ttype, order_C);

    rocsparse_spmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;
    rocsparse_dnmat_descr mat_C = local_mat_C;

#define PARAMS                                                                                    \
    handle, trans_A, trans_B, &alpha, mat_A, mat_B, &beta, mat_C, ttype, alg, stage, buffer_size, \
        temp_buffer

    rocsparse_int batch_count_B;
    rocsparse_int batch_count_C;
    int64_t       batch_stride_B;
    int64_t       batch_stride_C;

    // C_i = A * B_i
    batch_count_B  = 10;
    batch_count_C  = 5;
    batch_stride_B = k * n;
    batch_stride_C = m * n;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(PARAMS), rocsparse_status_invalid_value);

    // C_i = A_i * B
    batch_count_B  = 1;
    batch_count_C  = 5;
    batch_stride_B = 0;
    batch_stride_C = m * n;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(PARAMS), rocsparse_status_invalid_value);

    // C_i = A_i * B_i
    batch_count_B  = 10;
    batch_count_C  = 5;
    batch_stride_B = k * n;
    batch_stride_C = m * n;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(PARAMS), rocsparse_status_invalid_value);
#undef PARAMS
}

template <typename I, typename T>
void testing_spmm_batched_bell(const Arguments& arg)
{
}

#define INSTANTIATE(ITYPE, TTYPE)                                                        \
    template void testing_spmm_batched_bell_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmm_batched_bell<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_spmm_batched_bell_extra(const Arguments& arg) {}
