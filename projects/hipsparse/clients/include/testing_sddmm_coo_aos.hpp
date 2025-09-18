/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SDDMM_COO_AOS_HPP
#define TESTING_SDDMM_COO_AOS_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_sddmm_coo_aos_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int32_t              n         = 100;
    int32_t              m         = 100;
    int32_t              k         = 100;
    int64_t              nnz       = 100;
    int32_t              safe_size = 100;
    float                alpha     = 0.6;
    float                beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     orderA    = HIPSPARSE_ORDER_COL;
    hipsparseOrder_t     orderB    = HIPSPARSE_ORDER_COL;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxTypeI  = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSDDMMAlg_t  alg       = HIPSPARSE_SDDMM_ALG_DEFAULT;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto drowcol_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dA_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int32_t* drowcol = (int32_t*)drowcol_managed.get();
    float*   dval    = (float*)dval_managed.get();
    float*   dB      = (float*)dB_managed.get();
    float*   dA      = (float*)dA_managed.get();
    void*    dbuf    = (void*)dbuf_managed.get();

    // SDDMM structures
    hipsparseDnMatDescr_t A, B;
    hipsparseSpMatDescr_t C;

    size_t bsize;

    // Create SDDMM structures
    verify_hipsparse_status_success(hipsparseCreateDnMat(&A, m, k, m, dA, dataType, orderA),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, k, n, k, dB, dataType, orderB),
                                    "success");
    verify_hipsparse_status_success(
        hipsparseCreateCooAoS(&C, m, n, nnz, drowcol, dval, idxTypeI, idxBase, dataType),
        "success");

    // SDDMM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSDDMM_bufferSize(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, &bsize),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, &bsize),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, nullptr),
        "Error: bsize is nullptr");

    // SDDMM
    verify_hipsparse_status_invalid_handle(hipsparseSDDMM_preprocess(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // SDDMM
    verify_hipsparse_status_invalid_handle(
        hipsparseSDDMM(nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroyDnMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(C), "success");

#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_sddmm_coo_aos(Arguments argus)
{
#if(!defined(CUDART_VERSION))
    I                    m        = argus.M;
    I                    n        = argus.N;
    I                    k        = argus.K;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    T                    h_beta   = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseOrder_t     orderA   = argus.orderA;
    hipsparseOrder_t     orderB   = argus.orderB;
    hipsparseIndexBase_t idx_base = argus.baseC;
    hipsparseSDDMMAlg_t  alg      = static_cast<hipsparseSDDMMAlg_t>(argus.sddmm_alg);
    std::string          filename = argus.filename;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr;
    std::vector<I> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    // Read or construct CSR matrix
    I nnz = 0;
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<I> hrowcol_ind(nnz * 2);
    // Convert to COO_AOS
    for(I i = 0; i < m; ++i)
    {
        for(I j = hcsr_row_ptr[i]; j < hcsr_row_ptr[i + 1]; ++j)
        {
            hrowcol_ind[2 * (j - idx_base) + 0] = i + idx_base;
            hrowcol_ind[2 * (j - idx_base) + 1] = hcsr_col_ind[j - idx_base];
        }
    }

    // Some matrix properties
    I A_m = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
    I A_n = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
    I B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
    I B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;
    I C_m = m;
    I C_n = n;

    int ld_multiplier_A = 1;
    int ld_multiplier_B = 1;

    int64_t lda
        = (orderA == HIPSPARSE_ORDER_COL)
              ? ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_A) * m)
                                                               : (int64_t(ld_multiplier_A) * k))
              : ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_A) * k)
                                                               : (int64_t(ld_multiplier_A) * m));
    int64_t ldb
        = (orderB == HIPSPARSE_ORDER_COL)
              ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * k)
                                                               : (int64_t(ld_multiplier_B) * n))
              : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * n)
                                                               : (int64_t(ld_multiplier_B) * k));

    lda = std::max(int64_t(1), lda);
    ldb = std::max(int64_t(1), ldb);

    int64_t nrowA = (orderA == HIPSPARSE_ORDER_COL) ? lda : A_m;
    int64_t ncolA = (orderA == HIPSPARSE_ORDER_COL) ? A_n : lda;
    int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : B_m;
    int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? B_n : ldb;

    int64_t nnz_A = nrowA * ncolA;
    int64_t nnz_B = nrowB * ncolB;

    std::vector<T> hA(A_m * A_n);
    std::vector<T> hB(B_m * B_n);

    hipsparseInit<T>(hA, nnz_A, 1);
    hipsparseInit<T>(hB, nnz_B, 1);

    // allocate memory on device
    auto drowcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz * 2), device_free};
    auto dval1_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dval2_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    auto dA_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dB_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};

    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* drowcol = (I*)drowcol_managed.get();
    T* dval1   = (T*)dval1_managed.get();
    T* dval2   = (T*)dval2_managed.get();

    T* dA      = (T*)dA_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(drowcol, hrowcol_ind.data(), sizeof(I) * nnz * 2, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval1, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval2, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t C1, C2;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCooAoS(&C1, C_m, C_n, nnz, drowcol, dval1, typeI, idx_base, typeT));
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCooAoS(&C2, C_m, C_n, nnz, drowcol, dval2, typeI, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t A, B;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&A, A_m, A_n, lda, dA, typeT, orderA));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));

    // Query SDDMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_bufferSize(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(
        handle, transA, transB, d_alpha, A, B, d_beta, C2, typeT, alg, buffer));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseSDDMM(
            handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSDDMM(handle, transA, transB, d_alpha, A, B, d_beta, C2, typeT, alg, buffer));

        // copy output from device to CPU.
        std::vector<T> hval1(nnz);
        std::vector<T> hval2(nnz);
        CHECK_HIP_ERROR(hipMemcpy(hval1.data(), dval1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hval2.data(), dval2, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        const int64_t incA = (orderA == HIPSPARSE_ORDER_COL)
                                 ? ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? lda : 1)
                                 : ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : lda);
        const int64_t incB = (orderB == HIPSPARSE_ORDER_COL)
                                 ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : ldb)
                                 : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? ldb : 1);

        for(I i = 0; i < nnz; ++i)
        {
            const I r = hrowcol_ind[2 * i] - idx_base;
            const I c = hrowcol_ind[2 * i + 1] - idx_base;

            const T* Aptr
                = (orderA == HIPSPARSE_ORDER_COL)
                      ? ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? &hA[r] : &hA[lda * r])
                      : ((transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? &hA[lda * r] : &hA[r]);

            const T* Bptr
                = (orderB == HIPSPARSE_ORDER_COL)
                      ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? &hB[ldb * c] : &hB[c])
                      : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? &hB[c] : &hB[ldb * c]);

            T sum = static_cast<T>(0);
            for(I j = 0; j < k; ++j)
            {
                sum = testing_fma(Aptr[incA * j], Bptr[incB * j], sum);
            }
            hcsr_val[i] = testing_mult(hcsr_val[i], h_beta) + testing_mult(h_alpha, sum);
        }

        unit_check_near(1, nnz, 1, hval1.data(), hcsr_val.data());
        unit_check_near(1, nnz, 1, hval2.data(), hcsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSDDMM(
                handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSDDMM(
                handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = sddmm_gflop_count(k, nnz, h_beta != make_DataType<T>(0));
        double gbyte_count
            = sddmm_coo_aos_gbyte_count<T>(m, n, k, nnz, h_beta != make_DataType<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::format,
                            hipsparse_format2string(HIPSPARSE_FORMAT_COO_AOS),
                            display_key_t::transA,
                            hipsparse_operation2string(transA),
                            display_key_t::transB,
                            hipsparse_operation2string(transB),
                            display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::K,
                            k,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::algorithm,
                            hipsparse_sddmmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    // free.
    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C2));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SDDMM_COO_AOS_HPP
