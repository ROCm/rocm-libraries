/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPMM_BELL_HPP
#define TESTING_SPMM_BELL_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_graph.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename I, typename T>
void testing_spmm_bell_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION))
    int32_t              n            = 100;
    int32_t              m            = 100;
    int32_t              k            = 100;
    int32_t              ellBlockSize = 2;
    int32_t              ell_cols     = 10;
    int32_t              safe_size    = 100;
    float                alpha        = 0.6;
    float                beta         = 0.2;
    hipsparseOperation_t transA       = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB       = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order        = HIPSPARSE_ORDER_COL;
    hipsparseIndexBase_t idxBase      = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType      = HIPSPARSE_INDEX_32I;
    hipDataType          dataType     = HIP_R_32F;
    hipsparseSpMMAlg_t   alg          = HIPSPARSE_SPMM_BLOCKED_ELL_ALG1;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int32_t* dind = (int32_t*)dind_managed.get();
    float*   dval = (float*)dval_managed.get();
    float*   dB   = (float*)dB_managed.get();
    float*   dC   = (float*)dC_managed.get();
    void*    dbuf = (void*)dbuf_managed.get();

    // SpMM structures
    hipsparseSpMatDescr_t matA;
    hipsparseDnMatDescr_t matB, matC;

    size_t bsize;

    // Create SpMM structures
    verify_hipsparse_status_success(
        hipsparseCreateBlockedEll(
            &matA, m, k, ellBlockSize, ell_cols, dind, dval, idxType, idxBase, dataType),
        "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&matB, k, n, k, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&matC, m, n, m, dC, dataType, order),
                                    "success");

    // SpMM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_bufferSize(
        nullptr, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, nullptr, matA, matB, &beta, matC, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, nullptr, matB, &beta, matC, dataType, alg, &bsize),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, nullptr, &beta, matC, dataType, alg, &bsize),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, nullptr, matC, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, &bsize),
        "Error: matC is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, nullptr),
        "Error: bsize is nullptr");

    // SpMM_preprocess
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_preprocess(
        nullptr, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, nullptr, matA, matB, &beta, matC, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, nullptr, matB, &beta, matC, dataType, alg, dbuf),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, nullptr, &beta, matC, dataType, alg, dbuf),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, matB, nullptr, matC, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, dbuf),
        "Error: matC is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // SpMM
    verify_hipsparse_status_invalid_handle(hipsparseSpMM(
        nullptr, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, nullptr, matA, matB, &beta, matC, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, nullptr, matB, &beta, matC, dataType, alg, dbuf),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, nullptr, &beta, matC, dataType, alg, dbuf),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, matB, nullptr, matC, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, dbuf),
        "Error: matC is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matB), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matC), "success");
#endif
}

template <typename I, typename T>
void testing_spmm_bell(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    I                    m        = argus.M;
    I                    n        = argus.N;
    I                    k        = argus.K;
    I                    blockDim = argus.block_dim;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseOrder_t     orderB   = argus.orderB;
    hipsparseOrder_t     orderC   = argus.orderC;
    hipsparseIndexBase_t idxBase  = argus.baseA;
    hipsparseSpMMAlg_t   alg      = argus.spmm_alg;
    std::string          filename = argus.filename;

    I mb = (m + blockDim - 1) / blockDim;
    I kb = (k + blockDim - 1) / blockDim;

    I ellBlockSize = blockDim;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    hipsparseLocalHandle_t handle(argus);

    // Generate the Blocked-ELL matrix.
    srand(12345ULL);
    I              ellCols = 0;
    std::vector<I> hbellColInd;
    std::vector<T> hbellVal;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_bell_matrix(filename,
                             (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? mb : kb,
                             (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? kb : mb,
                             ellCols,
                             blockDim,
                             hbellColInd,
                             hbellVal,
                             idxBase));

    m = mb * blockDim;
    k = kb * blockDim;

    // Some matrix properties
    I A_mb = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? mb : kb;
    I A_nb = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? kb : mb;
    I B_m  = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
    I B_n  = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;
    I C_m  = m;
    I C_n  = n;

    int64_t ldb = (orderB == HIPSPARSE_ORDER_COL)
                      ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n)
                      : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k);
    int64_t ldc = (orderC == HIPSPARSE_ORDER_COL) ? m : n;

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : B_m;
    int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? B_n : ldb;
    int64_t nrowC = (orderC == HIPSPARSE_ORDER_COL) ? ldc : C_m;
    int64_t ncolC = (orderC == HIPSPARSE_ORDER_COL) ? C_n : ldc;

    // ell_val has (A_mb * blockDim) * ellCols elements in row-major order.
    int64_t nnz_A = A_mb * blockDim * ellCols;
    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    // Allocate host memory for dense matrices.
    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, nnz_B, 1);
    hipsparseInit<T>(hC_1, nnz_C, 1);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto dbell_col_ind_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(I) * A_mb * (ellCols / ellBlockSize)), device_free};
    auto dbell_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dB_managed        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dbell_col_ind = (I*)dbell_col_ind_managed.get();
    T* dbell_val     = (T*)dbell_val_managed.get();
    T* dB            = (T*)dB_managed.get();
    T* dC_1          = (T*)dC_1_managed.get();
    T* dC_2          = (T*)dC_2_managed.get();
    T* d_alpha       = (T*)d_alpha_managed.get();
    T* d_beta        = (T*)d_beta_managed.get();

    // Copy data from CPU to device.
    CHECK_HIP_ERROR(hipMemcpy(dbell_col_ind,
                              hbellColInd.data(),
                              sizeof(I) * A_mb * (ellCols / ellBlockSize),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbell_val, hbellVal.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices.
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateBlockedEll(
        &matA, m, k, ellBlockSize, ellCols, dbell_col_ind, dbell_val, typeI, idxBase, typeT));

    // Create dense matrices.
    hipsparseDnMatDescr_t matB, matC1, matC2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matB, B_m, B_n, ldb, dB, typeT, orderB));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matC1, C_m, C_n, ldc, dC_1, typeT, orderC));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matC2, C_m, C_n, ldc, dC_2, typeT, orderC));

    // Query SpMM buffer size.
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_bufferSize(
        handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, &bufferSize));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, bufferSize));

    if(argus.call_preprocess)
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
        // Host pointer mode.
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
            handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, dbuffer));
#endif

#if(!defined(CUDART_VERSION))
        // Device pointer mode.
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
            handle, transA, transB, d_alpha, matA, matB, d_beta, matC2, typeT, alg, dbuffer));
#endif
    }

    if(argus.unit_check)
    {
        // Host pointer mode.
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
            handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, dbuffer));

#if(!defined(CUDART_VERSION))
        // Device pointer mode.
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
            handle, transA, transB, d_alpha, matA, matB, d_beta, matC2, typeT, alg, dbuffer));
#endif

        host_bellmm(A_mb,
                    n,
                    A_nb,
                    ellCols,
                    ellBlockSize,
                    transA,
                    transB,
                    h_alpha,
                    hbellColInd,
                    hbellVal,
                    hB,
                    ldb,
                    orderB,
                    h_beta,
                    hC_gold,
                    ldc,
                    orderC,
                    idxBase);

        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());

#if(!defined(CUDART_VERSION))
        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());
#endif
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
                handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
                handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = bellmm_gflop_count(static_cast<int>(A_mb),
                                                static_cast<int>(n),
                                                static_cast<int>(ellCols),
                                                static_cast<int>(blockDim),
                                                static_cast<int>(C_m * C_n),
                                                h_beta != make_DataType<T>(0));
        double gbyte_count = bellmm_gbyte_count<T>(static_cast<int>(A_mb),
                                                   static_cast<int>(ellCols),
                                                   static_cast<int>(blockDim),
                                                   static_cast<int>(B_m * B_n),
                                                   static_cast<int>(C_m * C_n),
                                                   h_beta != make_DataType<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::K,
                            k,
                            display_key_t::block_dim,
                            blockDim,
                            display_key_t::transA,
                            transA,
                            display_key_t::transB,
                            transB,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::algorithm,
                            hipsparse_spmmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(dbuffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matB));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matC1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matC2));
#endif
}
#endif // TESTING_SPMM_BELL_HPP
