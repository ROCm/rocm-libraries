/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <typename T, typename S>
void bdsqr_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        const rocblas_int nv,
                        const rocblas_int nu,
                        const rocblas_int nc,
                        S dD,
                        S dE,
                        T dV,
                        const rocblas_int ldv,
                        T dU,
                        const rocblas_int ldu,
                        T dC,
                        const rocblas_int ldc,
                        rocblas_int* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_bdsqr(nullptr, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc, dinfo),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, rocblas_fill_full, n, nv, nu, nc, dD, dE, dV, ldv,
                                          dU, ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S) nullptr, dE, dV, ldv, dU,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, (S) nullptr, dV, ldv, dU,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, (T) nullptr, ldv, dU,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, (T) nullptr,
                                          ldu, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu,
                                          (T) nullptr, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                          ldc, (rocblas_int*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, 0, nv, nu, nc, (S) nullptr, (S) nullptr,
                                          (T) nullptr, ldv, (T) nullptr, ldu, (T) nullptr, ldc, dinfo),
                          rocblas_status_success);
}

template <typename T>
void testing_bdsqr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 2;
    rocblas_int nv = 2;
    rocblas_int nu = 2;
    rocblas_int nc = 2;
    rocblas_int ldv = 2;
    rocblas_int ldu = 2;
    rocblas_int ldc = 2;

    // memory allocations
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dU(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    bdsqr_checkBadArgs(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv, dU.data(),
                       ldu, dC.data(), ldc, dinfo.data());
}

template <bool CPU, bool GPU, typename T, typename S, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nv,
                    const rocblas_int nu,
                    const rocblas_int nc,
                    Sd& dD,
                    Sd& dE,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dU,
                    const rocblas_int ldu,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hE,
                    Th& hV,
                    Th& hU,
                    Th& hC,
                    Uh& hInfo,
                    std::vector<S>& D,
                    std::vector<S>& E,
                    const bool test)
{
    if(CPU)
    {
        rocblas_init<S>(hD, true);
        rocblas_init<S>(hE, false);

        // Adding possible gaps to fully test the algorithm.
        for(rocblas_int i = 0; i < n - 1; ++i)
        {
            hE[0][i] -= 5;
            hD[0][i] -= 4;
        }
        hD[0][n - 1] -= 4;

        // (Forcing non-convergence expecting lapack and rocsolver to give
        // the same orthogonal equivalent matrix is not possible. Testing
        // implicitly the equivalent matrix is very complicated and it boils
        // down to essentially run the algorithm again and until convergence is achieved).

        // make copy of original data to test vectors if required
        if(test && (nv || nu || nc))
        {
            for(rocblas_int i = 0; i < n - 1; ++i)
            {
                E[i] = hE[0][i];
                D[i] = hD[0][i];
            }
            D[n - 1] = hD[0][n - 1];
        }

        // make V,U and C identities so that results are actually singular vectors
        // of B
        if(nv > 0)
        {
            memset(hV[0], 0, ldv * nv * sizeof(T));
            for(rocblas_int i = 0; i < std::min(n, nv); ++i)
                hV[0][i + i * ldv] = T(1.0);
        }
        if(nu > 0)
        {
            memset(hU[0], 0, ldu * n * sizeof(T));
            for(rocblas_int i = 0; i < std::min(n, nu); ++i)
                hU[0][i + i * ldu] = T(1.0);
        }
        if(nc > 0)
        {
            memset(hC[0], 0, ldc * nc * sizeof(T));
            for(rocblas_int i = 0; i < std::min(n, nc); ++i)
                hC[0][i + i * ldc] = T(1.0);
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
        if(nv > 0)
            CHECK_HIP_ERROR(dV.transfer_from(hV));
        if(nu > 0)
            CHECK_HIP_ERROR(dU.transfer_from(hU));
        if(nc > 0)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nv,
                    const rocblas_int nu,
                    const rocblas_int nc,
                    Sd& dD,
                    Sd& dE,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dU,
                    const rocblas_int ldu,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    const rocblas_int nvT,
                    const rocblas_int nuT,
                    const rocblas_int nvRes,
                    const rocblas_int nuRes,
                    Sh& hD,
                    Sh& hDRes,
                    Sh& hE,
                    Sh& hERes,
                    Th& hV,
                    Th& hU,
                    Th& hC,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err,
                    double* max_errv)
{
    using S = decltype(std::real(T{}));
    std::vector<S> hW(4 * n);
    std::vector<S> D(n);
    std::vector<S> E(n);

    // input data initialization
    bdsqr_initData<true, false, T>(handle, uplo, n, nvRes, nuRes, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                   ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E, true);

    // execute computations:
    // complementary execution to compute all singular vectors if needed
    if(nvT > 0 || nuT > 0)
    {
        // send data to GPU
        bdsqr_initData<false, true, T>(handle, uplo, n, nvT, nuT, 0, dD, dE, dV, ldv, dU, ldu, dC,
                                       ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E, false);

        CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nvT, nuT, 0, dD.data(), dE.data(),
                                            dV.data(), ldv, dU.data(), ldu, dC.data(), ldc,
                                            dInfo.data()));
    }

    // send data to GPU
    bdsqr_initData<false, true, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc,
                                   dInfo, hD, hE, hV, hU, hC, hInfo, D, E, false);

    // execute computations
    // CPU lapack
    cpu_bdsqr(uplo, n, 0, 0, 0, hD[0], hE[0], hV[0], ldv, hU[0], ldu, hC[0], ldc, hW.data(),
              hInfo[0]);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(),
                                        ldv, dU.data(), ldu, dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(nvRes > 0)
        CHECK_HIP_ERROR(hV.transfer_from(dV));
    if(nuRes > 0)
        CHECK_HIP_ERROR(hU.transfer_from(dU));
    if(nc > 0)
        CHECK_HIP_ERROR(hC.transfer_from(dC));

    // Check info for non-convergence
    *max_err = 0;
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    // error is ||hD - hDRes||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    double err;
    T tmp;
    *max_errv = 0;
    err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    // Check the singular vectors if required
    if(hInfo[0][0] == 0 && (nv || nu))
    {
        err = 0;
        rocblas_int n_comp = std::min(n, nvRes);

        if(uplo == rocblas_fill_upper)
        {
            // check singular vectors implicitly (A'*u_i = s_i*v_i)
            for(rocblas_int i = 0; i < n_comp; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    if(i > 0)
                        tmp = D[i] * hU[0][i + j * ldu] + E[i - 1] * hU[0][(i - 1) + j * ldu]
                            - hDRes[0][j] * hV[0][j + i * ldv];
                    else
                        tmp = D[i] * hU[0][i + j * ldu] - hDRes[0][j] * hV[0][j + i * ldv];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
        }
        else
        {
            // check singular vectors implicitly (A*v_i = s_i*u_i)
            for(rocblas_int i = 0; i < n_comp; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    if(i > 0)
                        tmp = D[i] * hV[0][j + i * ldv] + E[i - 1] * hV[0][j + (i - 1) * ldv]
                            - hDRes[0][j] * hU[0][i + j * ldu];
                    else
                        tmp = D[i] * hV[0][j + i * ldv] - hDRes[0][j] * hU[0][i + j * ldu];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
        }

        double normD = double(snorm('F', 1, n, D.data(), 1));
        double normE = double(snorm('F', 1, n - 1, E.data(), 1));
        err = std::sqrt(err) / std::sqrt(normD * normD + normE * normE);
        *max_errv = err > *max_errv ? err : *max_errv;
    }

    // C should be the transpose of U
    if(hInfo[0][0] == 0 && nc)
    {
        err = 0;
        for(rocblas_int i = 0; i < nc; ++i)
        {
            for(rocblas_int j = 0; j < n; ++j)
            {
                tmp = hC[0][j + i * ldc] - hU[0][i + j * ldu];
                err += std::abs(tmp) * std::abs(tmp);
            }
        }
        err = std::sqrt(err);
        *max_errv = err > *max_errv ? err : *max_errv;
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void bdsqr_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int nv,
                       const rocblas_int nu,
                       const rocblas_int nc,
                       Sd& dD,
                       Sd& dE,
                       Td& dV,
                       const rocblas_int ldv,
                       Td& dU,
                       const rocblas_int ldu,
                       Td& dC,
                       const rocblas_int ldc,
                       Ud& dInfo,
                       Sh& hD,
                       Sh& hE,
                       Th& hV,
                       Th& hU,
                       Th& hC,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    using S = decltype(std::real(T{}));
    std::vector<S> hW(4 * n);
    std::vector<S> D;
    std::vector<S> E;

    if(!perf)
    {
        bdsqr_initData<true, false, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                       ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E, false);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_bdsqr(uplo, n, nv, nu, nc, hD[0], hE[0], hV[0], ldv, hU[0], ldu, hC[0], ldc, hW.data(),
                  hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    bdsqr_initData<true, false, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc,
                                   dInfo, hD, hE, hV, hU, hC, hInfo, D, E, false);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        bdsqr_initData<false, true, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                       ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E, false);

        CHECK_ROCBLAS_ERROR(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(),
                                            dV.data(), ldv, dU.data(), ldu, dC.data(), ldc,
                                            dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

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
        bdsqr_initData<false, true, T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC,
                                       ldc, dInfo, hD, hE, hV, hU, hC, hInfo, D, E, false);

        start = get_time_us_sync(stream);
        rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(), dV.data(), ldv,
                        dU.data(), ldu, dC.data(), ldc, dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_bdsqr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nv = argus.get<rocblas_int>("nv", 0);
    rocblas_int nu = argus.get<rocblas_int>("nu", 0);
    rocblas_int nc = argus.get<rocblas_int>("nc", 0);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", nv > 0 ? n : 1);
    rocblas_int ldu = argus.get<rocblas_int>("ldu", nu > 0 ? nu : 1);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nc > 0 ? n : 1);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    if(argus.alg_mode == 1)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_set_alg_mode(handle, rocsolver_function_bdsqr, rocsolver_alg_mode_hybrid),
            rocblas_status_success);

        rocsolver_alg_mode alg_mode;
        EXPECT_ROCBLAS_STATUS(rocsolver_get_alg_mode(handle, rocsolver_function_bdsqr, &alg_mode),
                              rocblas_status_success);

        EXPECT_EQ(alg_mode, rocsolver_alg_mode_hybrid);
    }

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldv, (T*)nullptr, ldu, (T*)nullptr, ldc,
                                              (rocblas_int*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // sizes for testing singular vectors
    // TESTING OF SINGULAR VECTORS IS DONE IMPLICITLY (NOT EXPLICITLY COMPARING
    // WITH LAPACK), SO WE ALWAYS NEED TO COMPUTE THE SAME NUMBER OF ELEMENTS OF
    // THE RIGHT AND LEFT VECTORS
    rocblas_int nvA = nv, nuA = nu;
    rocblas_int nvT = 0, nuT = 0;
    rocblas_int nvRes = nv, nuRes = nu;
    rocblas_int ldvRes = ldv, lduRes = ldu;

    if(nv && nu)
    {
        nvRes = nvA = std::max(nv, nu);
        nuRes = nuA = std::max(nvRes, nc);
        lduRes = std::max(lduRes, nuRes);
    }
    else if(nu)
    {
        nvRes = nvT = nu;
        nuRes = nuA = std::max(nu, nc);
        ldvRes = n;
        lduRes = std::max(lduRes, nuRes);
    }
    else if(nv || nc)
    {
        nuRes = nuT = std::max(nv, nc);
        lduRes = nuRes;
    }

    // E, V, U, and C could have size zero in cases that are not quick-return or
    // invalid cases setting the size to one to avoid possible memory-access
    // errors in the rest of the unit test
    size_t size_D = size_t(n);
    size_t size_E = n > 1 ? size_t(n - 1) : 1;
    size_t size_V = std::max(size_t(ldv) * nv, size_t(1));
    size_t size_U = std::max(size_t(ldu) * n, size_t(1));
    size_t size_C = std::max(size_t(ldc) * nc, size_t(1));
    size_t size_VRes = std::max(size_t(ldvRes) * nvRes, size_t(1));
    size_t size_URes = std::max(size_t(lduRes) * n, size_t(1));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0, max_errorv = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1)
        || (nv > 0 && ldv < n) || (nc > 0 && ldc < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldv, (T*)nullptr, ldu, (T*)nullptr, ldc,
                                              (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, (S*)nullptr, (S*)nullptr,
                                          (T*)nullptr, ldv, (T*)nullptr, ldu, (T*)nullptr, ldc,
                                          (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        host_strided_batch_vector<S> hDRes(size_D, 1, size_D, 1);
        host_strided_batch_vector<S> hERes(size_E, 1, size_E, 1);
        host_strided_batch_vector<T> hV(size_VRes, 1, size_VRes, 1);
        host_strided_batch_vector<T> hU(size_URes, 1, size_URes, 1);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(size_VRes, 1, size_VRes, 1);
        device_strided_batch_vector<T> dU(size_URes, 1, size_URes, 1);
        if(size_VRes)
            CHECK_HIP_ERROR(dV.memcheck());
        if(size_URes)
            CHECK_HIP_ERROR(dU.memcheck());

        // check quick return
        if(n == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_bdsqr(handle, uplo, n, nv, nu, nc, dD.data(), dE.data(),
                                                  dV.data(), ldv, dU.data(), ldu, dC.data(), ldc,
                                                  dInfo.data()),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        bdsqr_getError<T>(handle, uplo, n, nvA, nuA, nc, dD, dE, dV, ldvRes, dU, lduRes, dC, ldc,
                          dInfo, nvT, nuT, nvRes, nuRes, hD, hDRes, hE, hERes, hV, hU, hC, hInfo,
                          hInfoRes, &max_error, &max_errorv);
    }

    // collect performance data
    if(argus.timing)
    {
        host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
        host_strided_batch_vector<T> hU(size_U, 1, size_U, 1);
        device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
        device_strided_batch_vector<T> dU(size_U, 1, size_U, 1);
        if(size_V)
            CHECK_HIP_ERROR(dV.memcheck());
        if(size_U)
            CHECK_HIP_ERROR(dU.memcheck());

        bdsqr_getPerfData<T>(handle, uplo, n, nv, nu, nc, dD, dE, dV, ldv, dU, ldu, dC, ldc, dInfo,
                             hD, hE, hV, hU, hC, hInfo, &gpu_time_used, &cpu_time_used, hot_calls,
                             argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);
        if(nv || nu || nc)
            ROCSOLVER_TEST_CHECK(T, max_errorv, 2 * n);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            if(nv || nu || nc)
                max_error = (max_error >= max_errorv) ? max_error : max_errorv;
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "nv", "nu", "nc", "ldv", "ldu", "ldc");
            rocsolver_bench_output(uploC, n, nv, nu, nc, ldv, ldu, ldc);
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

#define EXTERN_TESTING_BDSQR(...) extern template void testing_bdsqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_BDSQR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
