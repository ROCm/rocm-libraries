/* **************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename U>
void bdsvdx_checkBadArgs(const rocblas_handle handle,
                         const rocblas_fill uplo,
                         const rocblas_svect svect,
                         const rocblas_srange srange,
                         const rocblas_int n,
                         U dD,
                         U dE,
                         const T vl,
                         const T vu,
                         const rocblas_int il,
                         const rocblas_int iu,
                         rocblas_int* dNsv,
                         U dS,
                         U dZ,
                         const rocblas_int ldz,
                         rocblas_int* dIfail,
                         rocblas_int* dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(nullptr, uplo, svect, srange, n, dD, dE, vl, vu, il, iu,
                                           dNsv, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, rocblas_fill_full, svect, srange, n, dD, dE, vl,
                                           vu, il, iu, dNsv, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, rocblas_svect_all, srange, n, dD, dE, vl,
                                           vu, il, iu, dNsv, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, rocblas_srange(0), n, dD, dE, vl,
                                           vu, il, iu, dNsv, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, (U) nullptr, dE, vl, vu,
                                           il, iu, dNsv, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD, (U) nullptr, vl, vu,
                                           il, iu, dNsv, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu,
                                           (rocblas_int*)nullptr, dS, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu,
                                           dNsv, (U) nullptr, dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu,
                                           dNsv, dS, (U) nullptr, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu,
                                           dNsv, dS, dZ, ldz, (rocblas_int*)nullptr, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu,
                                           dNsv, dS, dZ, ldz, dIfail, (rocblas_int*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, 0, (U) nullptr, (U) nullptr,
                                           vl, vu, il, iu, dNsv, (U) nullptr, (U) nullptr, ldz,
                                           (rocblas_int*)nullptr, dInfo),
                          rocblas_status_success);
}

template <typename T>
void testing_bdsvdx_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_svect svect = rocblas_svect_singular;
    rocblas_srange srange = rocblas_srange_all;
    rocblas_int ldz = 4;
    T vl = 0;
    T vu = 0;
    rocblas_int il = 0;
    rocblas_int iu = 0;

    // memory allocations
    device_strided_batch_vector<T> dD(1, 1, 1, 1);
    device_strided_batch_vector<T> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dS(1, 1, 1, 1);
    device_strided_batch_vector<T> dZ(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dNsv(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIfail(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dZ.memcheck());
    CHECK_HIP_ERROR(dNsv.memcheck());
    CHECK_HIP_ERROR(dIfail.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    bdsvdx_checkBadArgs(handle, uplo, svect, srange, n, dD.data(), dE.data(), vl, vu, il, iu,
                        dNsv.data(), dS.data(), dZ.data(), ldz, dIfail.data(), dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void bdsvdx_initData(const rocblas_handle handle, const rocblas_int n, Td& dD, Td& dE, Th& hD, Th& hE)
{
    if(CPU)
    {
        rocblas_init<T>(hD, true);
        rocblas_init<T>(hE, true);

        // scale matrix and add fixed splits in the matrix to test split handling
        // (scaling ensures that all singular values are in [0, 20])
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 10;
            hE[0][i] = (hE[0][i] - 5) / 10;
            if(i == n / 4 || i == n / 2 || i == n - 1)
                hE[0][i] = 0;
            if(i == n / 7 || i == n / 5 || i == n / 3)
                hD[0][i] *= -1;
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void bdsvdx_getError(const rocblas_handle handle,
                     const rocblas_fill uplo,
                     const rocblas_svect svect,
                     const rocblas_srange srange,
                     const rocblas_int n,
                     Td& dD,
                     Td& dE,
                     const T vl,
                     const T vu,
                     const rocblas_int il,
                     const rocblas_int iu,
                     Ud& dNsv,
                     Td& dS,
                     Td& dZ,
                     const rocblas_int ldz,
                     Ud& dIfail,
                     Ud& dInfo,
                     Th& hD,
                     Th& hE,
                     Uh& hNsv,
                     Uh& hNsvRes,
                     Th& hS,
                     Th& hSRes,
                     Th& hZ,
                     Th& hZRes,
                     Uh& hIfailRes,
                     Uh& hInfo,
                     Uh& hInfoRes,
                     double* max_err)
{
    std::vector<T> work(14 * n);
    std::vector<int> iwork(12 * n);

    // input data initialization
    bdsvdx_initData<true, true, T>(handle, n, dD, dE, hD, hE);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD.data(), dE.data(), vl,
                                         vu, il, iu, dNsv.data(), dS.data(), dZ.data(), ldz,
                                         dIfail.data(), dInfo.data()));
    CHECK_HIP_ERROR(hNsvRes.transfer_from(dNsv));
    CHECK_HIP_ERROR(hSRes.transfer_from(dS));
    CHECK_HIP_ERROR(hZRes.transfer_from(dZ));
    CHECK_HIP_ERROR(hIfailRes.transfer_from(dIfail));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    // WORKAROUND: For some test cases, LAPACK's bdsvdx is returning incorrect singular values
    // when srange is rocblas_srange_index. In this case, we use rocblas_srange_all to get
    // all the singular values and offset and use il as an offset into the result array.
    rocblas_int ioffset = 0;
    if(srange == rocblas_srange_index)
    {
        cpu_bdsvdx(uplo, rocblas_svect_none, rocblas_srange_all, n, hD[0], hE[0], vl, vu, il, iu,
                   hNsv[0], hS[0], hZ[0], ldz, work.data(), iwork.data(), hInfo[0]);
        ioffset = il - 1;
        hNsv[0][0] = iu - il + 1;
    }
    else
    {
        cpu_bdsvdx(uplo, rocblas_svect_none, srange, n, hD[0], hE[0], vl, vu, il, iu, hNsv[0],
                   hS[0], hZ[0], ldz, work.data(), iwork.data(), hInfo[0]);
    }

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    // if finding singular values succeeded, check values
    double err;
    if(hInfoRes[0][0] == 0)
    {
        // check number of computed singular values
        rocblas_int nn = hNsvRes[0][0];
        *max_err += std::abs(nn - hNsv[0][0]);
        EXPECT_EQ(hNsv[0][0], hNsvRes[0][0]);

        // error is ||hS - hSRes|| / ||hS||
        // using frobenius norm
        err = norm_error('F', 1, nn, 1, hS[0] + ioffset, hSRes[0]);
        *max_err = err > *max_err ? err : *max_err;

        // Check the singular vectors if required
        // U is stored in hZRes, and V is stored in hZRes+n
        if(svect != rocblas_svect_none)
        {
            // U and V should be orthonormal, if they are then U^T*U and V^T*V should be the identity
            if(nn > 0)
            {
                std::vector<T> UUres(nn * nn, 0.0);
                std::vector<T> VVres(nn * nn, 0.0);
                std::vector<T> I(nn * nn, 0.0);

                for(rocblas_int i = 0; i < nn; i++)
                    I[i + i * nn] = T(1);

                cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, nn, nn, n,
                         T(1), hZRes[0], ldz, hZRes[0], ldz, T(0), UUres.data(), nn);
                err = norm_error('F', nn, nn, nn, I.data(), UUres.data());
                *max_err = err > *max_err ? err : *max_err;

                cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, nn, nn, n,
                         T(1), hZRes[0] + n, ldz, hZRes[0] + n, ldz, T(0), VVres.data(), nn);
                err = norm_error('F', nn, nn, nn, I.data(), VVres.data());
                *max_err = err > *max_err ? err : *max_err;
            }

            err = 0;
            // form bidiagonal matrix B
            std::vector<T> B(n * n);
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        B[i + j * n] = hD[0][i];
                    else if(i + 1 == j && uplo == rocblas_fill_upper)
                        B[i + j * n] = hE[0][i];
                    else if(i == j + 1 && uplo == rocblas_fill_lower)
                        B[i + j * n] = hE[0][j];
                    else
                        B[i + j * n] = 0;
                }
            }

            // check singular vectors implicitly (B*v_k = s_k*u_k)
            for(rocblas_int k = 0; k < nn; ++k)
            {
                cpu_gemv(rocblas_operation_none, n, n, T(1), B.data(), n, hZRes[0] + n + k * ldz, 1,
                         -hSRes[0][k], hZRes[0] + k * ldz, 1);
            }
            err = double(snorm('F', n, nn, hZRes[0], ldz)) / double(snorm('F', n, n, B.data(), n));
            *max_err = err > *max_err ? err : *max_err;

            // check ifail
            err = 0;
            for(int j = 0; j < nn; j++)
            {
                EXPECT_EQ(hIfailRes[0][j], 0) << "where j = " << j;
                if(hIfailRes[0][j] != 0)
                    err++;
            }
            *max_err = err > *max_err ? err : *max_err;
        }
    }
    else
    {
        if(svect != rocblas_svect_none)
        {
            // check ifail
            err = 0;
            for(int j = 0; j < hInfoRes[0][0]; j++)
            {
                EXPECT_NE(hIfailRes[0][j], 0) << "where j = " << j;
                if(hIfailRes[0][j] == 0)
                    err++;
            }
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void bdsvdx_getPerfData(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_svect svect,
                        const rocblas_srange srange,
                        const rocblas_int n,
                        Td& dD,
                        Td& dE,
                        const T vl,
                        const T vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        Ud& dNsv,
                        Td& dS,
                        Td& dZ,
                        const rocblas_int ldz,
                        Ud& dIfail,
                        Ud& dInfo,
                        Th& hD,
                        Th& hE,
                        Uh& hNsv,
                        Th& hS,
                        Th& hZ,
                        Uh& hInfo,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf)
{
    if(!perf)
    {
        std::vector<T> work(14 * n);
        std::vector<int> iwork(12 * n);

        bdsvdx_initData<true, false, T>(handle, n, dD, dE, hD, hE);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_bdsvdx(uplo, svect, srange, n, hD[0], hE[0], vl, vu, il, iu, hNsv[0], hS[0], hZ[0], ldz,
                   work.data(), iwork.data(), hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    bdsvdx_initData<true, false, T>(handle, n, dD, dE, hD, hE);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        bdsvdx_initData<false, true, T>(handle, n, dD, dE, hD, hE);

        CHECK_ROCBLAS_ERROR(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD.data(), dE.data(),
                                             vl, vu, il, iu, dNsv.data(), dS.data(), dZ.data(), ldz,
                                             dIfail.data(), dInfo.data()));
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
        bdsvdx_initData<false, true, T>(handle, n, dD, dE, hD, hE);

        start = get_time_us_sync(stream);
        rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD.data(), dE.data(), vl, vu, il, iu,
                         dNsv.data(), dS.data(), dZ.data(), ldz, dIfail.data(), dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_bdsvdx(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    char svectC = argus.get<char>("svect");
    char srangeC = argus.get<char>("srange");
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_svect svect = char2rocblas_svect(svectC);
    rocblas_srange srange = char2rocblas_srange(srangeC);

    T vl = T(argus.get<double>("vl", 0));
    T vu = T(argus.get<double>("vu", srangeC == 'V' ? 1 : 0));
    rocblas_int il = argus.get<rocblas_int>("il", srangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", srangeC == 'I' ? 1 : 0);

    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldz = argus.get<rocblas_int>("ldz", 2 * n);
    rocblas_int nsv_max = (srange == rocblas_srange_index ? iu - il + 1 : n);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if((uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
       || (svect != rocblas_svect_none && svect != rocblas_svect_singular))
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, (T*)nullptr,
                                               (T*)nullptr, vl, vu, il, iu, (rocblas_int*)nullptr,
                                               (T*)nullptr, (T*)nullptr, ldz, (rocblas_int*)nullptr,
                                               (rocblas_int*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    size_t size_S = nsv_max;
    size_t size_S_cpu = n;
    size_t size_Z = ldz * nsv_max;
    size_t size_Ifail = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_SRes = (argus.unit_check || argus.norm_check) ? size_S : 0;
    size_t size_ZRes = (argus.unit_check || argus.norm_check) ? size_Z : 0;
    size_t size_IfailRes = (argus.unit_check || argus.norm_check) ? size_Ifail : 0;

    // check invalid sizes
    bool invalid_size = (n < 0) || (svect == rocblas_svect_none && ldz < 1)
        || (svect != rocblas_svect_none && ldz < 2 * n)
        || (srange == rocblas_srange_value && (vl < 0 || vl >= vu))
        || (srange == rocblas_srange_index && ((iu > n) || (n > 0 && il > iu)))
        || (srange == rocblas_srange_index && (il < 1 || iu < 0));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, (T*)nullptr,
                                               (T*)nullptr, vl, vu, il, iu, (rocblas_int*)nullptr,
                                               (T*)nullptr, (T*)nullptr, ldz, (rocblas_int*)nullptr,
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
        CHECK_ALLOC_QUERY(rocsolver_bdsvdx(handle, uplo, svect, srange, n, (T*)nullptr, (T*)nullptr,
                                           vl, vu, il, iu, (rocblas_int*)nullptr, (T*)nullptr,
                                           (T*)nullptr, ldz, (rocblas_int*)nullptr,
                                           (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    // host
    host_strided_batch_vector<T> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<T> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hS(size_S_cpu, 1, size_S_cpu, 1); // extra space for cpu_bdsvdx
    host_strided_batch_vector<T> hSRes(size_SRes, 1, size_SRes, 1);
    host_strided_batch_vector<T> hZ(size_Z, 1, size_Z, 1);
    host_strided_batch_vector<T> hZRes(size_ZRes, 1, size_ZRes, 1);
    host_strided_batch_vector<rocblas_int> hNsv(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hNsvRes(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hIfailRes(size_IfailRes, 1, size_IfailRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
    // device
    device_strided_batch_vector<T> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<T> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dS(size_S, 1, size_S, 1);
    device_strided_batch_vector<T> dZ(size_Z, 1, size_Z, 1);
    device_strided_batch_vector<rocblas_int> dNsv(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIfail(size_Ifail, 1, size_Ifail, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);

    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    if(size_Z)
        CHECK_HIP_ERROR(dZ.memcheck());
    if(size_Ifail)
        CHECK_HIP_ERROR(dIfail.memcheck());
    CHECK_HIP_ERROR(dNsv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_bdsvdx(handle, uplo, svect, srange, n, dD.data(), dE.data(),
                                               vl, vu, il, iu, dNsv.data(), dS.data(), dZ.data(),
                                               ldz, dIfail.data(), dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        bdsvdx_getError<T>(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu, dNsv, dS, dZ,
                           ldz, dIfail, dInfo, hD, hE, hNsv, hNsvRes, hS, hSRes, hZ, hZRes,
                           hIfailRes, hInfo, hInfoRes, &max_error);

    // collect performance data
    if(argus.timing)
        bdsvdx_getPerfData<T>(handle, uplo, svect, srange, n, dD, dE, vl, vu, il, iu, dNsv, dS, dZ,
                              ldz, dIfail, dInfo, hD, hE, hNsv, hS, hZ, hInfo, &gpu_time_used,
                              &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                              argus.perf);

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "svect", "srange", "n", "vl", "vu", "il", "iu", "ldz");
            rocsolver_bench_output(uploC, svectC, srangeC, n, vl, vu, il, iu, ldz);

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

#define EXTERN_TESTING_BDSVDX(...) extern template void testing_bdsvdx<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_BDSVDX, FOREACH_REAL_TYPE, APPLY_STAMP)
