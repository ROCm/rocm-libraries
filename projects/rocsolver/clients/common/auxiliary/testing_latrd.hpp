/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/matrix_utils/matrix_utils.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

static bool latrd_use_hipgraph = std::getenv("LATRD_USE_HIPGRAPH") != nullptr ? true : false;
static bool print_debug_messages_latrd = std::getenv("PRINT_DEBUG") != nullptr ? true : false;

template <typename T, typename S>
void latrd_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        const rocblas_int k,
                        T dA,
                        const rocblas_int lda,
                        S dE,
                        T dTau,
                        T dW,
                        const rocblas_int ldw)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(nullptr, uplo, n, k, dA, lda, dE, dTau, dW, ldw),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, rocblas_fill_full, n, k, dA, lda, dE, dTau, dW, ldw),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, (T) nullptr, lda, dE, dTau, dW, ldw),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA, lda, (S) nullptr, dTau, dW, ldw),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA, lda, dE, (T) nullptr, dW, ldw),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA, lda, dE, dTau, (T) nullptr, ldw),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, 0, dA, lda, dE, dTau, (T) nullptr, ldw),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, 0, 0, (T) nullptr, lda, (S) nullptr,
                                          (T) nullptr, (T) nullptr, ldw),
                          rocblas_status_success);
}

template <typename T>
void testing_latrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int k = 1;
    rocblas_int lda = 1;
    rocblas_int ldw = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    device_strided_batch_vector<T> dW(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());

    // check bad arguments
    latrd_checkBadArgs(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(), dW.data(), ldw);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void latrd_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        /* for(rocblas_int i = 0; i < n; i++) */
        /* { */
        /*     for(rocblas_int j = 0; j < n; j++) */
        /*     { */
        /*         if(i == j || (i == j + 1) || (i == j - 1)) */
        /*             hA[0][i + j * lda] += 400; */
        /*         else */
        /*             hA[0][i + j * lda] -= 4; */
        /*     } */
        /* } */

        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j <= i; j++)
            {
                if(i == j)
                {
                    hA[0][i + j * lda] = hA[0][i + j * lda] + 400;
                }
                else if((i == j + 1) || (i == j - 1))
                {
                    auto tmp = hA[0][j + i * lda] + 400;
                    hA[0][i + j * lda] = tmp;
                    hA[0][j + i * lda] = tmp;
                }
                else
                {
                    auto tmp = hA[0][j + i * lda] - 4;
                    hA[0][i + j * lda] = tmp;
                    hA[0][j + i * lda] = tmp;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void latrd_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        /* for(rocblas_int i = 0; i < n; i++) */
        /* { */
        /*     for(rocblas_int j = 0; j < n; j++) */
        /*     { */
        /*         if(i == j) */
        /*             hA[0][i + j * lda] = hA[0][i + j * lda].real() + 400; */
        /*         else if((i == j + 1) || (i == j - 1)) */
        /*             hA[0][i + j * lda] += 400; */
        /*         else */
        /*             hA[0][i + j * lda] -= 4; */
        /*     } */
        /* } */

        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j <= i; j++)
            {
                if(i == j)
                {
                    hA[0][i + j * lda] = hA[0][i + j * lda].real() + 400;
                }
                else if((i == j + 1) || (i == j - 1))
                {
                    auto tmp = hA[0][j + i * lda] + 400;
                    hA[0][i + j * lda] = tmp;
                    hA[0][j + i * lda] = sconj(tmp);
                }
                else
                {
                    auto tmp = hA[0][j + i * lda] - 4;
                    hA[0][i + j * lda] = tmp;
                    hA[0][j + i * lda] = sconj(tmp);
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Sd, typename Td, typename Sh, typename Th>
void latrd_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dA,
                    const rocblas_int lda,
                    Sd& dE,
                    Td& dTau,
                    Td& dW,
                    const rocblas_int ldw,
                    Th& hA,
                    Th& hARes,
                    Sh& hE,
                    Th& hTau,
                    Th& hW,
                    Th& hWRes,
                    double* max_err)
{
    using S = decltype(std::real(T{}));
    using HMat = HostMatrix<T, rocblas_int>;
    using HMatS = HostMatrix<S, rocblas_int>;
    using BDesc = typename HMat::BlockDescriptor;

    std::size_t size_E = n;
    host_strided_batch_vector<S> hERes(size_E, 1, size_E, 1);

    // input data initialization
    latrd_initData<true, true, T>(handle, n, dA, lda, hA);

    // Create a copy of input matrix into iA
    rocblas_int b = 0;
    auto iAWrap = HMat::Wrap(hA[0] + b * lda * n, lda, n);
    auto iA = (*iAWrap).block(BDesc().nrows(n).ncols(n));
    // Prints for debugging (1. check if input A is self-adjoint; 2. print input A)
    /* std::cout << "max|A - A^*| = " << (iA - adjoint(iA)).max_coeff_norm() << std::endl; */
    /* iA.print(); */

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(),
                                        dW.data(), ldw));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));

    // CPU lapack
    cpu_latrd(uplo, n, k, hA[0], lda, hE[0], hTau[0], hW[0], ldw);

    // create copies of matrices A and W after calls to rocsolver and lapack
    auto AWrap = HMat::Wrap(hA[0] + b * lda * n, lda, n);
    auto WWrap = HMat::Wrap(hW[0] + b * ldw * n, ldw, n);
    auto AResWrap = HMat::Wrap(hARes[0] + b * lda * n, lda, n);
    auto WResWrap = HMat::Wrap(hWRes[0] + b * ldw * n, ldw, n);

    auto A = (*AWrap).block(BDesc().nrows(n).ncols(n));
    /* std::cout << "Matrix A.diag() (lapack):" << std::endl; */
    /* A.diag().print(); */
    /* std::cout << "Matrix A (lapack):" << std::endl; */
    /* A.print(); */
    auto E = *HMat::Convert(hE[0], 1, n - 1);
    /* std::cout << "Matrix E (lapack):" << std::endl; */
    /* E.print(); */
    // Create tridiagonal Tl (lapack) with diagonal and off-diagonal entries computed with lapack's LATRD
    // Tl is a (k + 1) x (k + 1) matrix
    auto Tl = HMat::Zeros(n, n);
    Tl.diag(A.diag());
    Tl.sub_diag(E);
    Tl.sup_diag(E);
    auto W = (*WWrap).block(BDesc().nrows(n).ncols(k));
    /* std::cout << "Matrix W (lapack):" << std::endl; */
    /* W.print(); */

    auto ARes = (*AResWrap).block(BDesc().nrows(n).ncols(n));
    /* std::cout << "Matrix Ares.diag() (rocsolver):" << std::endl; */
    /* ARes.diag().print(); */
    /* std::cout << "Matrix Ares (rocsolver):" << std::endl; */
    /* ARes.print(); */
    auto ERes = *HMat::Convert(hERes[0], 1, n - 1);
    /* std::cout << "Matrix E (rocsolver):" << std::endl; */
    /* E.print(); */
    // Create tridiagonal Tr (rocsolver) with diagonal and off-diagonal entries computed with rocsolver's LATRD
    // Tr is a (k + 1) x (k + 1) matrix
    auto Tr = HMat::Zeros(n, n);
    Tr.diag(ARes.diag());
    Tr.sub_diag(ERes);
    Tr.sup_diag(ERes);
    auto WRes = (*WResWrap).block(BDesc().nrows(n).ncols(k));
    /* std::cout << "Matrix Wres (rocsolver):" << std::endl; */
    /* WRes.print(); */

    //
    // Old error bounds, comparing rocsolver's outputs with lapack's outputs
    //
    // error is max(||hA - hARes|| / ||hA||, ||hW - hWRes|| / ||hW||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    /* rocblas_int offset = (uplo == rocblas_fill_lower) ? k : 0; */
    *max_err = 0;
    /* err = norm_error('F', n, n, lda, hA[0], hARes[0]); */
    /* *max_err = err > *max_err ? err : *max_err; */
    /* err = norm_error('F', n - k, k, ldw, hW[0] + offset, hWRes[0] + offset); */
    /* *max_err = err > *max_err ? err : *max_err; */

    //
    // New error bounds, compare (k + 1) eigenvalues of Tl and Tr
    //
    // A proper bound would be (following Weyl)
    //    max| l_lapack - l_rocsolver | < C * ulp * (k + 1) * max|l_lapack|
    //
    // where `l_lapack` (`l_rocsolver`) stand for eigenvalues computed from
    // lapack (rocsolver) results (diagonal and off-diagonal entries) of LATRD,
    // and `C` is a "small" constant.
    //
    // (Strictly speaking, the current input matrices should satisfy a bound
    // that grows with `sqrt(k + 1)` instead of `k + 1`).
    //
    if(uplo == rocblas_fill_lower)
    {
        // When `uplo == rocblas_fill_lower` LATRD will update the first `k` columns of A.
        //
        // Extract first `k + 1` columns of Tl and Tr.
        auto Tl_k = Tl.block(BDesc().nrows(k + 1).ncols(k + 1));
        auto Tr_k = Tr.block(BDesc().nrows(k + 1).ncols(k + 1));

        /* std::cout << "Matrix Tl (lapack):" << std::endl; */
        /* Tl_k.print(); */
        /* std::cout << "Matrix Tr (rocsolver):" << std::endl; */
        /* Tr_k.print(); */

        auto [Ul_k, eig_Tl_k] = eig_lower(real(Tl_k));
        if(print_debug_messages_latrd)
        {
            std::cout << "Eigenvalues of matrix Tl (lapack):" << std::endl;
            eig_Tl_k.print();
        }
        auto [Ur_k, eig_Tr_k] = eig_lower(real(Tr_k));
        if(print_debug_messages_latrd)
        {
            std::cout << "Eigenvalues of matrix Tr (rocsolver):" << std::endl;
            eig_Tr_k.print();
        }

        err = (eig_Tl_k - eig_Tr_k).max_coeff_norm() / (eig_Tl_k.max_coeff_norm());
        *max_err = err > *max_err ? err : *max_err;
    }
    else // if(uplo == rocblas_fill_upper)
    {
        // When `uplo == rocblas_fill_upper` LATRD will update the last `k` columns of A.
        //
        // Extract last `k + 1` columns of Tl and Tr.
        auto Tl_k
            = Tl.block(BDesc().from_row(n - k - 2).nrows(k + 1).from_col(n - k - 2).ncols(k + 1));
        auto Tr_k
            = Tr.block(BDesc().from_row(n - k - 2).nrows(k + 1).from_col(n - k - 2).ncols(k + 1));

        /* std::cout << "Matrix Tl (lapack):" << std::endl; */
        /* Tl_k.print(); */
        /* std::cout << "Matrix Tr (rocsolver):" << std::endl; */
        /* Tr_k.print(); */

        auto [Ul_k, eig_Tl_k] = eig_lower(real(Tl_k));
        if(print_debug_messages_latrd)
        {
            std::cout << "Eigenvalues of matrix Tl (lapack):" << std::endl;
            eig_Tl_k.print();
        }
        auto [Ur_k, eig_Tr_k] = eig_lower(real(Tr_k));
        if(print_debug_messages_latrd)
        {
            std::cout << "Eigenvalues of matrix Tr (rocsolver):" << std::endl;
            eig_Tr_k.print();
        }

        err = (eig_Tl_k - eig_Tr_k).max_coeff_norm() / (eig_Tl_k.max_coeff_norm());
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Sd, typename Td, typename Sh, typename Th>
void latrd_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int k,
                       Td& dA,
                       const rocblas_int lda,
                       Sd& dE,
                       Td& dTau,
                       Td& dW,
                       const rocblas_int ldw,
                       Th& hA,
                       Sh& hE,
                       Th& hTau,
                       Th& hW,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        latrd_initData<true, false, T>(handle, n, dA, lda, hA);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        memset(hW[0], 0, ldw * k * sizeof(T));
        cpu_latrd(uplo, n, k, hA[0], lda, hE[0], hTau[0], hW[0], ldw);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    if(latrd_use_hipgraph)
    {
        if(print_debug_messages_latrd)
        {
            std::cout << "Using hipGraph" << std::endl;
        }

        // cold calls
        rocblas_handle handle2;
        rocblas_create_handle(&handle2);

        hipStream_t stream;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle2, stream));
        for(int iter = 0; iter < 2; iter++)
        {
            latrd_initData<false, true, T>(handle2, n, dA, lda, hA);

            CHECK_ROCBLAS_ERROR(rocsolver_latrd(handle2, uplo, n, k, dA.data(), lda, dE.data(),
                                                dTau.data(), dW.data(), ldw));
        }

        // graph capture
        hipGraph_t graph;
        latrd_initData<false, true, T>(handle2, n, dA, lda, hA);
        CHECK_HIP_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        CHECK_ROCBLAS_ERROR(rocsolver_latrd(handle2, uplo, n, k, dA.data(), lda, dE.data(),
                                            dTau.data(), dW.data(), ldw));

        // graphExec
        CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));
        hipGraphExec_t graphExec;
        CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CHECK_HIP_ERROR(hipGraphDestroy(graph));

        // gpu-lapack performance
        /* hipStream_t stream; */
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        rocsolver_timer timer;

        if(profile > 0)
        {
            if(profile_kernels)
                rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                             | rocblas_layer_mode_ex_log_kernel);
            else
                rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
            rocsolver_log_set_max_levels(profile);
        }

        for(rocblas_int iter = 0; iter < hot_calls; iter++)
        {
            /* latrd_initData<false, true, T>(handle, n, dA, lda, hA); */
            latrd_initData<false, true, T>(handle2, n, dA, lda, hA);

            timer.start(stream);
            /* rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(), dW.data(), ldw); */
            CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));

            timer.end(stream);
        }
        *gpu_time_used = timer.get_combined();
    }
    else
    {
        latrd_initData<true, false, T>(handle, n, dA, lda, hA);

        // cold calls
        for(int iter = 0; iter < 2; iter++)
        {
            latrd_initData<false, true, T>(handle, n, dA, lda, hA);

            CHECK_ROCBLAS_ERROR(rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(),
                                                dTau.data(), dW.data(), ldw));
        }

        // gpu-lapack performance
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        rocsolver_timer timer;

        if(profile > 0)
        {
            if(profile_kernels)
                rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                             | rocblas_layer_mode_ex_log_kernel);
            else
                rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
            rocsolver_log_set_max_levels(profile);
        }

        for(rocblas_int iter = 0; iter < hot_calls; iter++)
        {
            latrd_initData<false, true, T>(handle, n, dA, lda, hA);

            timer.start(stream);
            rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(), dW.data(),
                            ldw);
            timer.end(stream);
        }
        *gpu_time_used = timer.get_combined();
    }
    /* *gpu_time_used = timer.get_combined(); */
}

template <typename T>
void testing_latrd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int k = argus.get<rocblas_int>("k", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldw = argus.get<rocblas_int>("ldw", n);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, (T*)nullptr, lda, (S*)nullptr,
                                              (T*)nullptr, (T*)nullptr, ldw),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = lda * n;
    size_t size_E = n;
    size_t size_tau = n;
    size_t size_W = ldw * k;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || k < 0 || k > n || lda < n || ldw < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, (T*)nullptr, lda, (S*)nullptr,
                                              (T*)nullptr, (T*)nullptr, ldw),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_latrd(handle, uplo, n, k, (T*)nullptr, lda, (S*)nullptr,
                                          (T*)nullptr, (T*)nullptr, ldw));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);
    host_strided_batch_vector<T> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<T> hWRes(size_WRes, 1, size_WRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    device_strided_batch_vector<T> dW(size_W, 1, size_W, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());

    // check quick return
    if(k == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(),
                                              dTau.data(), dW.data(), ldw),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        latrd_getError<T>(handle, uplo, n, k, dA, lda, dE, dTau, dW, ldw, hA, hARes, hE, hTau, hW,
                          hWRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        latrd_getPerfData<T>(handle, uplo, n, k, dA, lda, dE, dTau, dW, ldw, hA, hE, hTau, hW,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf);

    //
    // This error bound is very lax!
    //
    // validate results for rocsolver-test
    // using k*n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, k * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "k", "lda", "ldw");
            rocsolver_bench_output(uploC, n, k, lda, ldw);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_LATRD(...) extern template void testing_latrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LATRD, FOREACH_SCALAR_TYPE, APPLY_STAMP)
