/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "client_utility.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "unit.hpp"

// Check to see if rocblas_set_atomics_mode is working. This is done by:
// - Calling rocblas_set_atomics_mode to not allow atomics
// - Calling rocblas_sgemm for a size that would normally use a Tensile
//   kernel with GlobalSplitU > 1
// - Initializing matrices with random rational numbers, and checking
//   that the rocblas_sgemm is deterministic
// - If atomics are allowed in this case, the result is not deterministic.

template <typename T>
void testing_atomics_mode(const Arguments& arg)
{
    auto rocblas_gemm_fn = arg.api & c_API_FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    double               rocblas_gflops, cblas_gflops;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
    rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
    rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_fn(handle,
                                              transA,
                                              transB,
                                              M,
                                              N,
                                              K,
                                              nullptr,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              ldb,
                                              nullptr,
                                              nullptr,
                                              ldc),
                              rocblas_status_invalid_size);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(A_row, A_col, lda);
    host_matrix<T> hB(B_row, B_col, ldb);
    host_matrix<T> hC_1(M, N, ldc);
    host_matrix<T> hC_gold(M, N, ldc);
    host_matrix<T> hC_input(M, N, ldc);

    // Allocate device memory
    device_matrix<T> dA(A_row, A_col, lda);
    device_matrix<T> dB(B_row, B_col, ldb);
    device_matrix<T> dC(M, N, ldc);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());

    // Initialize data on host memory
    // For this test the arg.initialization has to be HPL and this has been set in the atomics_mode_gtest.yaml
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC_input, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // ROCBLAS rocblas_pointer_mode_host
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    //  From kernel selection file /library/src/blas3/Tensile/Logic/asm_full/vega20_Cijk_Ailk_Bljk_SB.yaml
    //  we know that for gchArch == 906 and [m,n,batch_count,k] = [1024, 16, 1, 500000] an
    //  algorithm with globalSplitU == 32  will be called. For this architecture and size result
    //  should not be deterministic.
    //  If this test fails, check to see if kernel selection file listed above has changed.

    std::string arch_name = rocblas_internal_get_arch_name();
    if(arch_name == "gfx906")
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_allowed));

        // calculate reference result
        CHECK_HIP_ERROR(dC.transfer_from(hC_input));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hC_gold.transfer_from(dC));

        // verify arbitary number of tests are not deterministic
        double err1                     = 0;
        int    arbitary_number_of_tests = 10;
        for(int i = 0; i < arbitary_number_of_tests; i++)
        {
            CHECK_HIP_ERROR(dC.transfer_from(hC_input));

            CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
                handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

            CHECK_HIP_ERROR(hC_1.transfer_from(dC));

            // compare hC_1 and hC_gold
            err1 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
            if(err1 != 0)
                break;
        }
        EXPECT_NE(err1, 0);
    }

    //  Verify that result is deterministic for size [m,n,batch_count,k] = [1024, 16, 1, 500000].
    //  We know that for this size gcnArch == 906 will use globalSplitU == 32. For other architectures
    //  we suspect globalSplitU > 32 will be used
    CHECK_ROCBLAS_ERROR(rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed));

    // calculate reference result
    CHECK_HIP_ERROR(dC.transfer_from(hC_input));

    CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
        handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

    CHECK_HIP_ERROR(hC_gold.transfer_from(dC));

    // verify arbitrary number of tests are deterministic
    double err2                      = 0;
    int    arbitrary_number_of_tests = 10;
    for(int i = 0; i < arbitrary_number_of_tests; i++)
    {
        CHECK_HIP_ERROR(dC.transfer_from(hC_input));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_fn(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hC_1.transfer_from(dC));

        // compare hC_1 and hC_gold
        err2 = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1));
        if(err2 != 0)
            break;
    }

    EXPECT_EQ(err2, 0);
}
