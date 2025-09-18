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
#ifndef TESTING_SPMM_BELL_HPP
#define TESTING_SPMM_BELL_HPP

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

void testing_spmm_bell_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int32_t              n             = 100;
    int32_t              m             = 100;
    int32_t              k             = 100;
    int32_t              ell_blocksize = 2;
    int32_t              ell_cols      = 10;
    int32_t              safe_size     = 100;
    float                alpha         = 0.6;
    float                beta          = 0.2;
    hipsparseOperation_t transA        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order         = HIPSPARSE_ORDER_COL;
    hipsparseIndexBase_t idxBase       = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType       = HIPSPARSE_INDEX_32I;
    hipDataType          dataType      = HIP_R_32F;
    hipsparseSpMMAlg_t   alg           = HIPSPARSE_SPMM_BLOCKED_ELL_ALG1;

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
    hipsparseSpMatDescr_t A;
    hipsparseDnMatDescr_t B, C;

    size_t bsize;

    // Create SpMM structures
    verify_hipsparse_status_success(
        hipsparseCreateBlockedEll(
            &A, m, k, ell_blocksize, ell_cols, dind, dval, idxType, idxBase, dataType),
        "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, k, n, k, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&C, m, n, m, dC, dataType, order),
                                    "success");

    // SpMM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_bufferSize(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, &bsize),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, &bsize),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, nullptr),
        "Error: bsize is nullptr");

    // SpMM_preprocess
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_preprocess(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // SpMM
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMM(nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(C), "success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_spmm_bell()
{

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)

    std::vector<T> hval = {make_DataType<T>(1.0),
                           make_DataType<T>(2.0),
                           make_DataType<T>(3.0),
                           make_DataType<T>(4.0),
                           make_DataType<T>(5.0),
                           make_DataType<T>(6.0),
                           make_DataType<T>(7.0),
                           make_DataType<T>(8.0),
                           make_DataType<T>(9.0),
                           make_DataType<T>(10.0),
                           make_DataType<T>(11.0),
                           make_DataType<T>(12.0),
                           make_DataType<T>(13.0),
                           make_DataType<T>(14.0),
                           make_DataType<T>(15.0),
                           make_DataType<T>(16.0)};

    std::vector<I> hcol_ind = {1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 5, 6};

    std::vector<I> hrow_ptr = {1, 3, 5, 9, 13, 15, 17};

    std::vector<T> hbell_val
        = {make_DataType<T>(1.0),  make_DataType<T>(2.0),  make_DataType<T>(3.0),
           make_DataType<T>(4.0),  make_DataType<T>(-7.0), make_DataType<T>(-7.0),
           make_DataType<T>(-7.0), make_DataType<T>(-7.0), make_DataType<T>(5.0),
           make_DataType<T>(6.0),  make_DataType<T>(7.0),  make_DataType<T>(8.0),
           make_DataType<T>(9.0),  make_DataType<T>(10.0), make_DataType<T>(11.0),
           make_DataType<T>(12.0), make_DataType<T>(13.0), make_DataType<T>(14.0),
           make_DataType<T>(15.0), make_DataType<T>(16.0), make_DataType<T>(-7.0),
           make_DataType<T>(-7.0), make_DataType<T>(-7.0), make_DataType<T>(-7.0)};

    std::vector<I> hbell_ind = {1, 0, 1, 2, 3, 0};

    I ell_cols      = 4;
    I ell_blocksize = 2;
    I m             = 6;
    I k             = 6;
    I nnz           = 16;
    I n             = 2;
    I ldb           = k;
    I ldc           = m;

    T                    h_alpha  = make_DataType<T>(2.0);
    T                    h_beta   = make_DataType<T>(1.0);
    hipsparseOperation_t transA   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order    = HIPSPARSE_ORDER_COL;
    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ONE;
    hipsparseSpMMAlg_t   alg      = HIPSPARSE_SPMM_BLOCKED_ELL_ALG1;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::vector<T> hB = {make_DataType<T>(1.0),
                         make_DataType<T>(1.0),
                         make_DataType<T>(-1.0),
                         make_DataType<T>(2.0),
                         make_DataType<T>(1.0),
                         make_DataType<T>(3.0),
                         make_DataType<T>(-1.0),
                         make_DataType<T>(4.0),
                         make_DataType<T>(1.0),
                         make_DataType<T>(5.0),
                         make_DataType<T>(-1.0),
                         make_DataType<T>(6.0)};

    std::vector<T> hC_1 = {make_DataType<T>(1.0),
                           make_DataType<T>(1.0),
                           make_DataType<T>(-1.0),
                           make_DataType<T>(2.0),
                           make_DataType<T>(1.0),
                           make_DataType<T>(3.0),
                           make_DataType<T>(-1.0),
                           make_DataType<T>(4.0),
                           make_DataType<T>(1.0),
                           make_DataType<T>(5.0),
                           make_DataType<T>(-1.0),
                           make_DataType<T>(6.0)};

    std::vector<T> hC_gold = {make_DataType<T>(35.0),
                              make_DataType<T>(41.0),
                              make_DataType<T>(115.0),
                              make_DataType<T>(126.0),
                              make_DataType<T>(25.0),
                              make_DataType<T>(31.0),
                              make_DataType<T>(149.0),
                              make_DataType<T>(172.0),
                              make_DataType<T>(155.0),
                              make_DataType<T>(169.0),
                              make_DataType<T>(45.0),
                              make_DataType<T>(58.0)};

    std::vector<T> hC_2(hC_1);

    // allocate memory on device
    auto drow_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    auto dbell_ind_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(I) * (ell_cols / ell_blocksize) * (m / ell_blocksize)), device_free};
    auto dbell_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * ell_cols * m), device_free};

    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * k * n), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * n), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * n), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* drow_ptr  = (I*)drow_ptr_managed.get();
    I* dcol_ind  = (I*)dcol_ind_managed.get();
    T* dval      = (T*)dval_managed.get();
    T* dB        = (T*)dB_managed.get();
    T* dC_1      = (T*)dC_1_managed.get();
    T* dC_2      = (T*)dC_2_managed.get();
    T* d_alpha   = (T*)d_alpha_managed.get();
    T* d_beta    = (T*)d_beta_managed.get();
    I* dbell_ind = (I*)dbell_ind_managed.get();
    T* dbell_val = (T*)dbell_val_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(drow_ptr, hrow_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol_ind, hcol_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * k * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * m * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * m * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbell_ind,
                              hbell_ind.data(),
                              sizeof(I) * (ell_cols / ell_blocksize) * (m / ell_blocksize),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbell_val, hbell_val.data(), sizeof(T) * ell_cols * m, hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t A_bell;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateBlockedEll(
        &A_bell, m, k, ell_blocksize, ell_cols, dbell_ind, dbell_val, typeI, idx_base, typeT));

    hipsparseSpMatDescr_t A_csr;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &A_csr, m, k, nnz, drow_ptr, dcol_ind, dval, typeI, typeI, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, k, n, ldb, dB, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, m, n, ldc, dC_1, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, m, n, ldc, dC_2, typeT, order));

    // Query SpMM buffer
    size_t bufferSize_bell;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_bufferSize(
        handle, transA, transB, &h_alpha, A_bell, B, &h_beta, C1, typeT, alg, &bufferSize_bell));

    void* buffer_bell;
    CHECK_HIP_ERROR(hipMalloc(&buffer_bell, bufferSize_bell));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
        handle, transA, transB, &h_alpha, A_bell, B, &h_beta, C1, typeT, alg, buffer_bell));
#endif
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
        handle, transA, transB, &h_alpha, A_bell, B, &h_beta, C1, typeT, alg, buffer_bell));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
        handle, transA, transB, d_alpha, A_bell, B, d_beta, C2, typeT, alg, buffer_bell));
#endif
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
        handle, transA, transB, d_alpha, A_bell, B, d_beta, C2, typeT, alg, buffer_bell));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * m * n, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * m * n, hipMemcpyDeviceToHost));
    unit_check_near(1, m * n, 1, hC_gold.data(), hC_1.data());
    unit_check_near(1, m * n, 1, hC_gold.data(), hC_2.data());

    CHECK_HIP_ERROR(hipFree(buffer_bell));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A_csr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A_bell));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif // TESTING_SPMM_BELL_HPP
