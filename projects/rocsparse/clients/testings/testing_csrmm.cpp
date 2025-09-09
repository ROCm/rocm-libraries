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

#include "testing.hpp"

template <typename T>
void testing_csrmm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Local decalrations.

    rocsparse_handle    handle  = local_handle;
    rocsparse_operation trans_A = rocsparse_operation_none;
    rocsparse_operation trans_B = rocsparse_operation_none;
    rocsparse_int       m       = safe_size;
    rocsparse_int       n       = safe_size;
    rocsparse_int       k       = safe_size;
    rocsparse_int       nnz     = safe_size;

    const T  local_alpha = static_cast<T>(2);
    const T* alpha       = &local_alpha;

    rocsparse_mat_descr  descr       = local_descr;
    const T*             csr_val     = (const T*)0x4;
    const rocsparse_int* csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int* csr_col_ind = (const rocsparse_int*)0x4;
    const T*             dense_B     = (const T*)0x4;
    rocsparse_int        ldb         = safe_size;
    const T              local_beta  = static_cast<T>(2);
    const T*             beta        = &local_beta;
    T*                   dense_C     = (T*)0x4;
    rocsparse_int        ldc         = safe_size;

#define PARAMS                                                                               \
    handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, \
        dense_B, ldb, beta, dense_C, ldc

    bad_arg_analysis(rocsparse_csrmm<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_not_implemented);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));
    //
    // Testing wrong leading dimensions.
    //
    {
        //
        // op(A) = A, op(B) = B
        //
        m       = 3;
        n       = 14;
        k       = 32;
        trans_A = rocsparse_operation_none;
        trans_B = rocsparse_operation_none;

        //  ldb < k
        ldb = k - 1;
        ldc = m;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = k;
        ldc = m - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);
    }

    {
        //
        // op(A) = A, op(B) = B^T
        //
        m       = 3;
        n       = 14;
        k       = 32;
        trans_A = rocsparse_operation_none;
        trans_B = rocsparse_operation_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = m;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = n;
        ldc = m - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_B = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = m;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = n;
        ldc = m - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);
    }

    {
        //
        // op(A) = A^T, op(B) = B
        //
        m       = 3;
        n       = 14;
        k       = 32;
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_none;

        //  ldb < m
        ldb = m - 1;
        ldc = k;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = m;
        ldc = k - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < m
        ldb = m - 1;
        ldc = k;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = m;
        ldc = k - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);
    }

    {
        //
        // op(A) = A^T, op(B) = B^T
        //
        m       = 3;
        n       = 14;
        k       = 32;
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = k;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = k - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = k;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = k - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_conjugate_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = k;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = k - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = k;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = k - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_invalid_size);
    }

#undef PARAMS
}

template <typename T>
void testing_csrmm(const Arguments& arg)
{
    rocsparse_int        M      = arg.M;
    rocsparse_int        N      = arg.N;
    rocsparse_int        K      = arg.K;
    rocsparse_operation  transA = arg.transA;
    rocsparse_operation  transB = arg.transB;
    rocsparse_index_base base   = arg.baseA;
    rocsparse_order      order  = rocsparse_order_column;
    //
    // order column
    //
    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    device_scalar<T> d_alpha(h_alpha);
    device_scalar<T> d_beta(h_beta);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Allocate host memory for matrix
    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, K);

    CHECK_ROCSPARSE_ERROR(hA.scale());

    M = hA.m;
    K = hA.n;
    auto Bm
        = (transB == rocsparse_operation_none) ? (transA == rocsparse_operation_none ? K : M) : N;
    auto Bn
        = (transB == rocsparse_operation_none) ? N : (transA == rocsparse_operation_none ? K : M);

    host_dense_matrix<T> hB(Bm, Bn, order);
    auto                 Cm = (transA == rocsparse_operation_none ? M : K);
    auto                 Cn = N;

    host_dense_matrix<T> hC(Cm, Cn, order);
    rocsparse_matrix_utils::init(hB);
    rocsparse_matrix_utils::init(hC);
    device_csr_matrix<T> dA(hA);

    //
    // Memory layout of matrix B and C on device.
    //
    device_dense_matrix_view<T> dB, dC;
    device_dense_vector<T>      layout(hB.m * hB.n + hC.m * hC.n);
    if((hB.n == hC.n) && (order == rocsparse_order_column))
    {
        //
        // column interleaved
        //
        dB(hB.m, hB.n, layout, hB.m + hC.m, order);
        dC(hC.m, hC.n, layout + hB.m, hB.m + hC.m, order);
    }
    else if((hB.m == hC.m) && (order == rocsparse_order_row))
    {
        //
        // row interleaved
        //
        dB(hB.m, hB.n, layout, hB.n + hC.n, order);
        dC(hC.m, hC.n, layout + hB.n, hB.n + hC.n, order);
    }
    else
    {
        //
        // BLOCK
        //
        dB(hB.m, hB.n, layout, (order == rocsparse_order_column) ? hB.m : hB.n);
        dC(hC.m, hC.n, layout + hB.m * hB.n, (order == rocsparse_order_column) ? hC.m : hC.n);
    }

    dB = hB;
    dC = hC;

    // Copy data from CPU to device
#define PARAMS(alpha_, A_, B_, beta_, C_)                                                      \
    handle, transA, transB, M, N, K, A_.nnz, alpha_, descr, A_.val, A_.ptr, A_.ind, B_, B_.ld, \
        beta_, C_, C_.ld

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode host", dC);
        }

        {
            host_dense_matrix<T> hC_copy(hC);
            // CPU csrmm
            host_csrmm<T, rocsparse_int, rocsparse_int, T, T, T>(M,
                                                                 N,
                                                                 K,
                                                                 transA,
                                                                 transB,
                                                                 *h_alpha,
                                                                 hA.ptr,
                                                                 hA.ind,
                                                                 hA.val,
                                                                 hB,
                                                                 hB.ld,
                                                                 order,
                                                                 *h_beta,
                                                                 hC,
                                                                 hC.ld,
                                                                 order,
                                                                 base,
                                                                 false);
            hC.near_check(dC);
            dC = hC_copy;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmm<T>(PARAMS(d_alpha, dA, dB, d_beta, dC)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode device", dC);
        }
        hC.near_check(dC);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_csrmm<T>, PARAMS(h_alpha, dA, dB, h_beta, dC));

        double gflop_count = csrmm_gflop_count<rocsparse_int, rocsparse_int>(
            N, dA.nnz, dC.m * dC.n, *h_beta != static_cast<T>(0));
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);
        double gbyte_count = csrmm_gbyte_count<T, T, T, rocsparse_int, rocsparse_int>(
            dA.m, dA.nnz, dB.m * dB.n, dC.m * dC.n, *h_beta != static_cast<T>(0));
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::trans_A,
                            rocsparse_operation2string(transA),
                            display_key_t::trans_B,
                            rocsparse_operation2string(transB),
                            display_key_t::nnz_A,
                            dA.nnz,
                            display_key_t::nnz_B,
                            dB.m * dB.n,
                            display_key_t::nnz_C,
                            dC.m * dC.n,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::beta,
                            *h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

#undef PARAMS
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_csrmm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrmm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrmm_extra(const Arguments& arg) {}
