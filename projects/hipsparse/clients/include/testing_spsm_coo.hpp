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
#ifndef TESTING_SPSM_COO_HPP
#define TESTING_SPSM_COO_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

void testing_spsm_coo_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t              m         = 100;
    int64_t              n         = 100;
    int64_t              k         = 100;
    int64_t              nnz       = 100;
    int64_t              safe_size = 100;
    float                alpha     = 0.6;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseOrder_t     order     = HIPSPARSE_ORDER_COL;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpSMAlg_t   alg       = HIPSPARSE_SPSM_ALG_DEFAULT;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto drow_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   drow = (int*)drow_managed.get();
    int*   dcol = (int*)dcol_managed.get();
    float* dval = (float*)dval_managed.get();
    float* dB   = (float*)dB_managed.get();
    float* dC   = (float*)dC_managed.get();
    void*  dbuf = (void*)dbuf_managed.get();

    // SpSM structures
    hipsparseSpMatDescr_t A;
    hipsparseDnVecDescr_t B, C;

    hipsparseSpSMDescr_t descr;

    verify_hipsparse_status_success(hipsparseSpSM_createDescr(&descr), "success");

    size_t bsize;

    // Create SpSM structures
    verify_hipsparse_status_success(
        hipsparseCreateCoo(&A, m, n, nnz, drow, dcol, dval, idxType, idxBase, dataType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, m, k, m, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&C, m, k, m, dC, dataType, order),
                                    "success");

    // SpSM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpSM_bufferSize(
        nullptr, transA, transB, &alpha, A, B, C, dataType, alg, descr, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, nullptr, A, B, C, dataType, alg, descr, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, nullptr, B, C, dataType, alg, descr, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, C, dataType, alg, descr, &bsize),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, B, nullptr, dataType, alg, descr, &bsize),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, B, C, dataType, alg, nullptr, &bsize),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, B, C, dataType, alg, descr, nullptr),
        "Error: bsize is nullptr");

    // SpSM analysis
    verify_hipsparse_status_invalid_handle(hipsparseSpSM_analysis(
        nullptr, transA, transB, &alpha, A, B, C, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, nullptr, A, B, C, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, nullptr, B, C, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, nullptr, C, dataType, alg, descr, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, B, nullptr, dataType, alg, descr, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, B, C, dataType, alg, nullptr, dbuf),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, B, C, dataType, alg, descr, nullptr),
        "Error: dbuf is nullptr");

    // SpSM solve
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSM_solve(nullptr, transA, transB, &alpha, A, B, C, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(handle, transA, transB, nullptr, A, B, C, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(
            handle, transA, transB, &alpha, nullptr, B, C, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(
            handle, transA, transB, &alpha, A, nullptr, C, dataType, alg, descr, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(
            handle, transA, transB, &alpha, A, B, nullptr, dataType, alg, descr, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(handle, transA, transB, &alpha, A, B, C, dataType, alg, nullptr, dbuf),
        "Error: descr is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseSpSM_destroyDescr(descr), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(C), "success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_spsm_coo(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    I                    m        = argus.M;
    I                    n        = argus.N;
    I                    k        = argus.K;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseOrder_t     orderB   = argus.orderB;
    hipsparseOrder_t     orderC   = argus.orderC;
    hipsparseIndexBase_t idx_base = argus.baseA;
    hipsparseDiagType_t  diag     = argus.diag_type;
    hipsparseFillMode_t  uplo     = argus.fill_mode;
    hipsparseSpSMAlg_t   alg      = static_cast<hipsparseSpSMAlg_t>(argus.spsm_alg);
    std::string          filename = argus.filename;

#if(defined(CUDART_VERSION))
    if(orderB != orderC)
    {
        return HIPSPARSE_STATUS_SUCCESS;
    }
#endif

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hrow_ptr;
    std::vector<I> hcol_ind;
    std::vector<T> hval;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz;
    if(!generate_csr_matrix(filename, m, n, nnz, hrow_ptr, hcol_ind, hval, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    if(m != n)
    {
        // Skip non-square matrices
        return HIPSPARSE_STATUS_SUCCESS;
    }

    std::vector<I> hrow_ind(nnz);

    // Convert to COO
    for(I i = 0; i < m; ++i)
    {
        for(I j = hrow_ptr[i]; j < hrow_ptr[i + 1]; ++j)
        {
            hrow_ind[j - idx_base] = i + idx_base;
        }
    }

    // Some matrix properties
    I B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
    I B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
    I C_m = m;
    I C_n = k;

    int ld_multiplier_B = 1;
    int ld_multiplier_C = 1;

    int64_t ldb
        = (orderB == HIPSPARSE_ORDER_COL)
              ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * m)
                                                               : (int64_t(ld_multiplier_B) * k))
              : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * k)
                                                               : (int64_t(ld_multiplier_B) * m));
    int64_t ldc = (orderC == HIPSPARSE_ORDER_COL) ? (int64_t(ld_multiplier_C) * m)
                                                  : (int64_t(ld_multiplier_C) * k);

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : B_m;
    int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? B_n : ldb;
    int64_t nrowC = (orderC == HIPSPARSE_ORDER_COL) ? ldc : C_m;
    int64_t ncolC = (orderC == HIPSPARSE_ORDER_COL) ? C_n : ldc;

    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, 1, nnz_B);

    hC_1    = hB;
    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto drow_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* drow    = (I*)drow_managed.get();
    I* dcol    = (I*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(drow, hrow_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcol_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipsparseSpSMDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_createDescr(&descr));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCoo(&A, m, n, nnz, drow, dcol, dval, typeI, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, C_m, C_n, ldc, dC_1, typeT, orderC));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, C_m, C_n, ldc, dC_2, typeT, orderC));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    // Query SpSM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_bufferSize(
        handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_analysis(
        handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_analysis(
        handle, transA, transB, d_alpha, A, B, C2, typeT, alg, descr, buffer));

    if(argus.unit_check)
    {
        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseSpSM_solve(
            handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, buffer));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseSpSM_solve(
            handle, transA, transB, d_alpha, A, B, C2, typeT, alg, descr, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        I struct_pivot  = -1;
        I numeric_pivot = -1;
        host_coosm(m,
                   k,
                   nnz,
                   transA,
                   transB,
                   h_alpha,
                   hrow_ind,
                   hcol_ind,
                   hval,
                   hB,
                   (I)ldb,
                   orderB,
                   hC_gold,
                   (I)ldc,
                   orderC,
                   diag,
                   uplo,
                   idx_base,
                   &struct_pivot,
                   &numeric_pivot);

        if(struct_pivot == -1 && numeric_pivot == -1)
        {
            unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());
            unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSpSM_solve(
                handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSpSM_solve(
                handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spsv_gflop_count(m, nnz, diag) * k;
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = coosv_gbyte_count<T>(m, nnz) * k;
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::K,
                            k,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::algorithm,
                            hipsparse_spsmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_destroyDescr(descr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPSM_COO_HPP
