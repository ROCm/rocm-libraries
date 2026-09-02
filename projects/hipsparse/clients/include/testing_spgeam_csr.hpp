/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPGEAM_CSR_HPP
#define TESTING_SPGEAM_CSR_HPP

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

template <typename T>
void testing_spgeam_csr_bad_arg(const Arguments& argus)
{
    // clang-format 19 (math-ci) rewrites "#if(" to "#if (", which clang-format
    // 18 (the repo pre-commit hook) reverts; disable so both formatters agree.
    // clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
    // clang-format on
    int64_t              m         = 100;
    int64_t              n         = 100;
    int64_t              nnz_A     = 100;
    int64_t              nnz_B     = 100;
    int64_t              nnz_C     = 100;
    int64_t              safe_size = 100;
    float                alpha     = 0.6;
    float                beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpGEAMAlg_t alg       = HIPSPARSE_SPGEAM_ALG1;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<spgeam_struct> unique_ptr_descr(new spgeam_struct);
    hipsparseSpGEAMDescr_t         descr = unique_ptr_descr->descr;

    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_val_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcsr_row_ptr_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_col_ind_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_val_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_val_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   dcsr_row_ptr_A = (int*)dcsr_row_ptr_A_managed.get();
    int*   dcsr_col_ind_A = (int*)dcsr_col_ind_A_managed.get();
    float* dcsr_val_A     = (float*)dcsr_val_A_managed.get();
    int*   dcsr_row_ptr_B = (int*)dcsr_row_ptr_B_managed.get();
    int*   dcsr_col_ind_B = (int*)dcsr_col_ind_B_managed.get();
    float* dcsr_val_B     = (float*)dcsr_val_B_managed.get();
    int*   dcsr_row_ptr_C = (int*)dcsr_row_ptr_C_managed.get();
    int*   dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
    float* dcsr_val_C     = (float*)dcsr_val_C_managed.get();
    void*  dbuf           = (void*)dbuf_managed.get();

    hipsparseSpMatDescr_t A, B, C;

    size_t bufferSize;

    verify_hipsparse_status_success(hipsparseCreateCsr(&A,
                                                       m,
                                                       n,
                                                       nnz_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       dcsr_val_A,
                                                       idxType,
                                                       idxType,
                                                       idxBase,
                                                       dataType),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateCsr(&B,
                                                       m,
                                                       n,
                                                       nnz_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       dcsr_val_B,
                                                       idxType,
                                                       idxType,
                                                       idxBase,
                                                       dataType),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateCsr(&C,
                                                       m,
                                                       n,
                                                       nnz_C,
                                                       dcsr_row_ptr_C,
                                                       dcsr_col_ind_C,
                                                       dcsr_val_C,
                                                       idxType,
                                                       idxType,
                                                       idxBase,
                                                       dataType),
                                    "success");

    // SpGEAM buffer size
    verify_hipsparse_status_invalid_handle(hipsparseSpGEAM_bufferSize(
        nullptr, transA, transB, &alpha, A, &beta, B, C, dataType, alg, descr, &bufferSize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_bufferSize(
            handle, transA, transB, nullptr, A, &beta, B, C, dataType, alg, descr, &bufferSize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpGEAM_bufferSize(handle,
                                                                       transA,
                                                                       transB,
                                                                       &alpha,
                                                                       nullptr,
                                                                       &beta,
                                                                       B,
                                                                       C,
                                                                       dataType,
                                                                       alg,
                                                                       descr,
                                                                       &bufferSize),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, B, C, dataType, alg, descr, &bufferSize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpGEAM_bufferSize(handle,
                                                                       transA,
                                                                       transB,
                                                                       &alpha,
                                                                       A,
                                                                       &beta,
                                                                       nullptr,
                                                                       C,
                                                                       dataType,
                                                                       alg,
                                                                       descr,
                                                                       &bufferSize),
                                            "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpGEAM_bufferSize(handle,
                                                                       transA,
                                                                       transB,
                                                                       &alpha,
                                                                       A,
                                                                       &beta,
                                                                       B,
                                                                       nullptr,
                                                                       dataType,
                                                                       alg,
                                                                       descr,
                                                                       &bufferSize),
                                            "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_bufferSize(
            handle, transA, transB, &alpha, A, &beta, B, C, dataType, alg, descr, nullptr),
        "Error: bufferSize is nullptr");

    // SpGEAM nnz
    verify_hipsparse_status_invalid_handle(hipsparseSpGEAM_nnz(
        nullptr, transA, transB, &alpha, A, &beta, B, C, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_nnz(
            handle, transA, transB, nullptr, A, &beta, B, C, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_nnz(
            handle, transA, transB, &alpha, nullptr, &beta, B, C, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_nnz(
            handle, transA, transB, &alpha, A, &beta, nullptr, C, dataType, alg, descr, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM_nnz(
            handle, transA, transB, &alpha, A, &beta, B, nullptr, dataType, alg, descr, dbuf),
        "Error: C is nullptr");

    // SpGEAM compute
    verify_hipsparse_status_invalid_handle(hipsparseSpGEAM(
        nullptr, transA, transB, &alpha, A, &beta, B, C, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM(
            handle, transA, transB, nullptr, A, &beta, B, C, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM(
            handle, transA, transB, &alpha, nullptr, &beta, B, C, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM(
            handle, transA, transB, &alpha, A, &beta, nullptr, C, dataType, alg, descr, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEAM(
            handle, transA, transB, &alpha, A, &beta, B, nullptr, dataType, alg, descr, dbuf),
        "Error: C is nullptr");

    // Unsupported cases must return a well-defined error rather than an incorrect result.
    // Only non-transpose operations are supported.
    verify_hipsparse_status_not_supported(hipsparseSpGEAM_bufferSize(handle,
                                                                     HIPSPARSE_OPERATION_TRANSPOSE,
                                                                     transB,
                                                                     &alpha,
                                                                     A,
                                                                     &beta,
                                                                     B,
                                                                     C,
                                                                     dataType,
                                                                     alg,
                                                                     descr,
                                                                     &bufferSize),
                                          "Error: opA transpose is not supported");
    verify_hipsparse_status_not_supported(hipsparseSpGEAM_bufferSize(handle,
                                                                     transA,
                                                                     HIPSPARSE_OPERATION_TRANSPOSE,
                                                                     &alpha,
                                                                     A,
                                                                     &beta,
                                                                     B,
                                                                     C,
                                                                     dataType,
                                                                     alg,
                                                                     descr,
                                                                     &bufferSize),
                                          "Error: opB transpose is not supported");

    // Only the CSR format is supported.
    hipsparseSpMatDescr_t A_coo;
    verify_hipsparse_status_success(hipsparseCreateCoo(&A_coo,
                                                       m,
                                                       n,
                                                       nnz_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       dcsr_val_A,
                                                       idxType,
                                                       idxBase,
                                                       dataType),
                                    "success");
    verify_hipsparse_status_not_supported(
        hipsparseSpGEAM_bufferSize(
            handle, transA, transB, &alpha, A_coo, &beta, B, C, dataType, alg, descr, &bufferSize),
        "Error: COO format is not supported");
    verify_hipsparse_status_success(hipsparseDestroySpMat(A_coo), "success");

    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(C), "success");
#endif
}

template <typename T>
void testing_spgeam_csr(Arguments argus)
{
    // clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
    // clang-format on
    int                  m        = argus.M;
    int                  n        = argus.N;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseIndexBase_t idxBaseA = argus.baseA;
    hipsparseIndexBase_t idxBaseB = argus.baseB;
    hipsparseIndexBase_t idxBaseC = argus.baseC;
    hipsparseSpGEAMAlg_t alg      = HIPSPARSE_SPGEAM_ALG1;
    std::string          filename = argus.filename;

    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparseIndexType_t typeI = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t typeJ = HIPSPARSE_INDEX_32I;
    hipDataType          typeT = getDataType<T>();

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<spgeam_struct> unique_ptr_descr(new spgeam_struct);
    hipsparseSpGEAMDescr_t         descr = unique_ptr_descr->descr;

    // cusparseSpGEAM only supports host pointer mode scalars, so the device pointer mode
    // pass (matrix C2 below) is only exercised on the rocSPARSE backend.
    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    std::unique_ptr<spgeam_struct> unique_ptr_descr2(new spgeam_struct);
    hipsparseSpGEAMDescr_t         descr2 = unique_ptr_descr2->descr;
#endif

    // Host structures for A
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    srand(12345ULL);

    int nnz_A = 0;
    CHECK_GENERATE_MATRIX_ERROR(generate_csr_matrix(
        filename, m, n, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idxBaseA));

    // B is generated as an independent m x n matrix so that it has a different sparsity
    // pattern than A. This exercises the SpGEAM pattern-merge path (columns present in only
    // one of the operands) and the guarantee that the output C column indices are sorted.
    std::vector<int> hcsr_row_ptr_B;
    std::vector<int> hcsr_col_ind_B;
    std::vector<T>   hcsr_val_B;

    int nnz_B = 0;
    CHECK_GENERATE_MATRIX_ERROR(generate_csr_matrix(
        "*", m, n, nnz_B, hcsr_row_ptr_B, hcsr_col_ind_B, hcsr_val_B, idxBaseB));

    // Allocate device memory
    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto dcsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dcsr_row_ptr_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_B), device_free};
    auto dcsr_val_B_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dcsr_row_ptr_C_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    auto dcsr_row_ptr_C_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
#endif

    int* dcsr_row_ptr_A   = (int*)dcsr_row_ptr_A_managed.get();
    int* dcsr_col_ind_A   = (int*)dcsr_col_ind_A_managed.get();
    T*   dcsr_val_A       = (T*)dcsr_val_A_managed.get();
    int* dcsr_row_ptr_B   = (int*)dcsr_row_ptr_B_managed.get();
    int* dcsr_col_ind_B   = (int*)dcsr_col_ind_B_managed.get();
    T*   dcsr_val_B       = (T*)dcsr_val_B_managed.get();
    int* dcsr_row_ptr_C_1 = (int*)dcsr_row_ptr_C_1_managed.get();
    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    int* dcsr_row_ptr_C_2 = (int*)dcsr_row_ptr_C_2_managed.get();
    T*   d_alpha          = (T*)d_alpha_managed.get();
    T*   d_beta           = (T*)d_beta_managed.get();
#endif

    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
#endif

    // Create matrices
    hipsparseSpMatDescr_t A, B, C1;
    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    hipsparseSpMatDescr_t C2;
#endif
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&A,
                                             m,
                                             n,
                                             nnz_A,
                                             dcsr_row_ptr_A,
                                             dcsr_col_ind_A,
                                             dcsr_val_A,
                                             typeI,
                                             typeJ,
                                             idxBaseA,
                                             typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&B,
                                             m,
                                             n,
                                             nnz_B,
                                             dcsr_row_ptr_B,
                                             dcsr_col_ind_B,
                                             dcsr_val_B,
                                             typeI,
                                             typeJ,
                                             idxBaseB,
                                             typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &C1, m, n, 0, dcsr_row_ptr_C_1, nullptr, nullptr, typeI, typeJ, idxBaseC, typeT));
    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &C2, m, n, 0, dcsr_row_ptr_C_2, nullptr, nullptr, typeI, typeJ, idxBaseC, typeT));
#endif

    // Buffer size (host pointer mode)
    size_t bufferSize1;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM_bufferSize(
        handle, transA, transB, &h_alpha, A, &h_beta, B, C1, typeT, alg, descr, &bufferSize1));

    void* externalBuffer1;
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer1, bufferSize1));

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM_nnz(
        handle, transA, transB, &h_alpha, A, &h_beta, B, C1, typeT, alg, descr, externalBuffer1));

    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    // Buffer size (device pointer mode) - rocSPARSE backend only.
    size_t bufferSize2;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM_bufferSize(
        handle, transA, transB, d_alpha, A, d_beta, B, C2, typeT, alg, descr2, &bufferSize2));

    void* externalBuffer2;
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer2, bufferSize2));

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM_nnz(
        handle, transA, transB, d_alpha, A, d_beta, B, C2, typeT, alg, descr2, externalBuffer2));
#endif

    // Get nnz of C
    int64_t rows_C, cols_C, nnz_C_1;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(C1, &rows_C, &cols_C, &nnz_C_1));

    // Allocate C column indices and values
    auto dcsr_col_ind_C_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_C_1), device_free};
    auto dcsr_val_C_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C_1), device_free};

    int* dcsr_col_ind_C_1 = (int*)dcsr_col_ind_C_1_managed.get();
    T*   dcsr_val_C_1     = (T*)dcsr_val_C_1_managed.get();

    CHECK_HIPSPARSE_ERROR(
        hipsparseCsrSetPointers(C1, dcsr_row_ptr_C_1, dcsr_col_ind_C_1, dcsr_val_C_1));

    // Compute step (host pointer mode)
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM(
        handle, transA, transB, &h_alpha, A, &h_beta, B, C1, typeT, alg, descr, externalBuffer1));

    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    // Device pointer mode compute - rocSPARSE backend only.
    int64_t nnz_C_2;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(C2, &rows_C, &cols_C, &nnz_C_2));

    auto dcsr_col_ind_C_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_C_2), device_free};
    auto dcsr_val_C_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C_2), device_free};

    int* dcsr_col_ind_C_2 = (int*)dcsr_col_ind_C_2_managed.get();
    T*   dcsr_val_C_2     = (T*)dcsr_val_C_2_managed.get();

    CHECK_HIPSPARSE_ERROR(
        hipsparseCsrSetPointers(C2, dcsr_row_ptr_C_2, dcsr_col_ind_C_2, dcsr_val_C_2));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEAM(
        handle, transA, transB, d_alpha, A, d_beta, B, C2, typeT, alg, descr2, externalBuffer2));
#endif

    if(argus.unit_check)
    {
        std::vector<int> hcsr_row_ptr_C_1(m + 1);
        std::vector<int> hcsr_col_ind_C_1(nnz_C_1);
        std::vector<T>   hcsr_val_C_1(nnz_C_1);

        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_1.data(),
                                  dcsr_row_ptr_C_1,
                                  sizeof(int) * (m + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_1.data(),
                                  dcsr_col_ind_C_1,
                                  sizeof(int) * nnz_C_1,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_val_C_1.data(), dcsr_val_C_1, sizeof(T) * nnz_C_1, hipMemcpyDeviceToHost));

        // clang-format off
#if(!defined(CUDART_VERSION))
        // clang-format on
        std::vector<int> hcsr_row_ptr_C_2(m + 1);
        std::vector<int> hcsr_col_ind_C_2(nnz_C_2);
        std::vector<T>   hcsr_val_C_2(nnz_C_2);

        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_2.data(),
                                  dcsr_row_ptr_C_2,
                                  sizeof(int) * (m + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_2.data(),
                                  dcsr_col_ind_C_2,
                                  sizeof(int) * nnz_C_2,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_val_C_2.data(), dcsr_val_C_2, sizeof(T) * nnz_C_2, hipMemcpyDeviceToHost));
#endif

        // Compute host reference solution.
        //
        // Match the cuSPARSE SpGEAM behavior for zero scalars: when alpha is zero the output
        // sparsity pattern is that of B, when beta is zero it is that of A, and when both are
        // zero the output is empty. This is emulated by dropping the corresponding operand
        // (feeding an empty matrix) to the host csrgeam reference.
        const bool alpha_is_zero = (argus.alpha == 0.0 && argus.alphai == 0.0);
        const bool beta_is_zero  = (argus.beta == 0.0 && argus.betai == 0.0);

        std::vector<int> empty_row_ptr_A(m + 1, idxBaseA);
        std::vector<int> empty_row_ptr_B(m + 1, idxBaseB);
        std::vector<int> empty_col_ind;
        std::vector<T>   empty_val;

        const int* gold_row_ptr_A = alpha_is_zero ? empty_row_ptr_A.data() : hcsr_row_ptr_A.data();
        const int* gold_col_ind_A = alpha_is_zero ? empty_col_ind.data() : hcsr_col_ind_A.data();
        const T*   gold_val_A     = alpha_is_zero ? empty_val.data() : hcsr_val_A.data();
        const int* gold_row_ptr_B = beta_is_zero ? empty_row_ptr_B.data() : hcsr_row_ptr_B.data();
        const int* gold_col_ind_B = beta_is_zero ? empty_col_ind.data() : hcsr_col_ind_B.data();
        const T*   gold_val_B     = beta_is_zero ? empty_val.data() : hcsr_val_B.data();

        std::vector<int> hcsr_row_ptr_C_gold(m + 1);

        int nnz_C_gold = host_csrgeam_nnz(m,
                                          n,
                                          h_alpha,
                                          gold_row_ptr_A,
                                          gold_col_ind_A,
                                          h_beta,
                                          gold_row_ptr_B,
                                          gold_col_ind_B,
                                          hcsr_row_ptr_C_gold.data(),
                                          idxBaseA,
                                          idxBaseB,
                                          idxBaseC);

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        host_csrgeam(m,
                     n,
                     h_alpha,
                     gold_row_ptr_A,
                     gold_col_ind_A,
                     gold_val_A,
                     h_beta,
                     gold_row_ptr_B,
                     gold_col_ind_B,
                     gold_val_B,
                     hcsr_row_ptr_C_gold.data(),
                     hcsr_col_ind_C_gold.data(),
                     hcsr_val_C_gold.data(),
                     idxBaseA,
                     idxBaseB,
                     idxBaseC);

        // Verify nnz, row pointer, column indices and values
        int nnz_C_1_i = (int)nnz_C_1;
        unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C_1_i);
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C_1.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C_1.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_1.data());

        // clang-format off
#if(!defined(CUDART_VERSION))
        // clang-format on
        int nnz_C_2_i = (int)nnz_C_2;
        unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C_2_i);
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C_2.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C_2.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_2.data());
#endif
    }

    CHECK_HIP_ERROR(hipFree(externalBuffer1));

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C1));

    // clang-format off
#if(!defined(CUDART_VERSION))
    // clang-format on
    CHECK_HIP_ERROR(hipFree(externalBuffer2));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C2));
#endif
#endif
}

#endif // TESTING_SPGEAM_CSR_HPP
