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
#ifndef TESTING_SPMM_CSR_REUSE_DESCR_HPP
#define TESTING_SPMM_CSR_REUSE_DESCR_HPP

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

#include <algorithm>

using namespace hipsparse_test;

template <typename I, typename J, typename T>
void testing_spmm_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// Given a sparse matrix descriptor stored as (A_rows x A_cols) and the
// requested operations, returns the logical SpMM dimensions and the stored
// dimensions / leading dimensions of the dense B and C matrices. Column-major
// ordering is assumed for both dense matrices throughout this test.
template <typename J>
static void spmm_reuse_dims(J                    A_rows,
                            J                    A_cols,
                            J                    n,
                            hipsparseOperation_t transA,
                            hipsparseOperation_t transB,
                            J&                   B_m,
                            J&                   B_n,
                            J&                   C_m,
                            J&                   C_n,
                            int64_t&             ldb,
                            int64_t&             ldc)
{
    // C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C(m x n)
    const J spmm_k = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? A_cols : A_rows;
    const J spmm_m = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? A_rows : A_cols;

    // Stored dimensions of the dense B matrix (before applying op(B))
    B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? spmm_k : n;
    B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : spmm_k;

    C_m = spmm_m;
    C_n = n;

    // Column-major leading dimensions
    ldb = std::max(int64_t(1), int64_t(B_m));
    ldc = std::max(int64_t(1), int64_t(C_m));
}

// Exercises the per-call bufferSize / per-call buffer-allocation pattern for a
// single (transA, transB, algorithm) configuration: query
// hipsparseSpMM_bufferSize, hipMalloc a fresh externalBuffer of that size,
// optionally run hipsparseSpMM_preprocess, then run hipsparseSpMM. When called
// repeatedly with the same configuration on the same matA, the wrapper must
// continue to return correct results despite the bufferSize / buffer being
// re-queried and re-allocated on every call.
template <typename I, typename J, typename T>
static void call_spmm(hipsparseHandle_t&     handle,
                      hipsparseSpMatDescr_t& matA,
                      J                      M,
                      J                      N,
                      J                      K,
                      std::vector<I>&        hcsr_row_ptr,
                      std::vector<J>&        hcsr_col_ind,
                      std::vector<T>&        hcsr_val,
                      T                      alpha,
                      T                      beta,
                      hipsparseIndexBase_t   idx_base,
                      hipsparseOperation_t   transA,
                      hipsparseOperation_t   transB,
                      hipsparseSpMMAlg_t     alg,
                      bool                   use_preprocess)
{
    hipDataType typeT = getDataType<T>();

    const hipsparseOrder_t orderB = HIPSPARSE_ORDER_COL;
    const hipsparseOrder_t orderC = HIPSPARSE_ORDER_COL;

    J       B_m, B_n, C_m, C_n;
    int64_t ldb, ldc;
    spmm_reuse_dims<J>(M, K, N, transA, transB, B_m, B_n, C_m, C_n, ldb, ldc);

    const int64_t nnz_B = ldb * B_n;
    const int64_t nnz_C = ldc * C_n;

    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, nnz_B, 1);
    hipsparseInit<T>(hC_1, nnz_C, 1);

    hC_2    = hC_1;
    hC_gold = hC_1;

    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &beta, sizeof(T), hipMemcpyHostToDevice));

    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, C_m, C_n, ldc, dC_1, typeT, orderC));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, C_m, C_n, ldc, dC_2, typeT, orderC));

    // Query SpMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_bufferSize(
        handle, transA, transB, &alpha, matA, B, &beta, C1, typeT, alg, &bufferSize));

    // Ensure the externalBuffer pointer is never null even when no scratch space
    // is required, so the buffer remains valid for every backend.
    bufferSize = std::max(bufferSize, size_t(4));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    if(use_preprocess)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, B, &beta, C1, typeT, alg, buffer));
    }
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMM(handle, transA, transB, &alpha, matA, B, &beta, C1, typeT, alg, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    if(use_preprocess)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
            handle, transA, transB, d_alpha, matA, B, d_beta, C2, typeT, alg, buffer));
    }
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMM(handle, transA, transB, d_alpha, matA, B, d_beta, C2, typeT, alg, buffer));

    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    // Host SpMM
    host_csrmm<I, J, T>(M,
                        N,
                        K,
                        transA,
                        transB,
                        alpha,
                        hcsr_row_ptr.data(),
                        hcsr_col_ind.data(),
                        hcsr_val.data(),
                        hB.data(),
                        ldb,
                        orderB,
                        beta,
                        hC_gold.data(),
                        ldc,
                        orderC,
                        idx_base,
                        false);

    unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());
    unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));
}

// Exercises the multi-configuration shared-buffer path: query bufferSize once
// per (transA, transB, algorithm) configuration, allocate a single
// externalBuffer sized to the max, then repeatedly call hipsparseSpMM
// alternating among the configurations without ever calling
// hipsparseSpMM_bufferSize again. The dense B and C descriptors are recreated
// per configuration (their logical dimensions change with the operations)
// while the same sparse-matrix descriptor and the same user-provided buffer
// are reused throughout.
template <typename I, typename J, typename T>
static void call_spmm_shared_buffer(hipsparseHandle_t&                       handle,
                                    hipsparseSpMatDescr_t&                   matA,
                                    J                                        M,
                                    J                                        N,
                                    J                                        K,
                                    std::vector<I>&                          hcsr_row_ptr,
                                    std::vector<J>&                          hcsr_col_ind,
                                    std::vector<T>&                          hcsr_val,
                                    T                                        alpha,
                                    T                                        beta,
                                    hipsparseIndexBase_t                     idx_base,
                                    const std::vector<hipsparseOperation_t>& ops_A,
                                    const std::vector<hipsparseOperation_t>& ops_B,
                                    const std::vector<hipsparseSpMMAlg_t>&   algs,
                                    int                                      number_of_passes,
                                    bool                                     use_preprocess)
{
    hipDataType typeT = getDataType<T>();

    const hipsparseOrder_t orderB = HIPSPARSE_ORDER_COL;
    const hipsparseOrder_t orderC = HIPSPARSE_ORDER_COL;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Step 1: query the bufferSize for every (transA, transB, algorithm)
    // configuration we will use, and take the max. The externalBuffer we
    // allocate below is reused for all configurations.
    size_t buffer_size_max = 0;
    for(hipsparseOperation_t transA : ops_A)
    {
        for(hipsparseOperation_t transB : ops_B)
        {
            J       B_m, B_n, C_m, C_n;
            int64_t ldb, ldc;
            spmm_reuse_dims<J>(M, K, N, transA, transB, B_m, B_n, C_m, C_n, ldb, ldc);

            const int64_t nnz_B = ldb * B_n;
            const int64_t nnz_C = ldc * C_n;

            auto dB_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
            auto dC_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
            T*   dB         = (T*)dB_managed.get();
            T*   dC         = (T*)dC_managed.get();

            hipsparseDnMatDescr_t B, C;
            CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));
            CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C, C_m, C_n, ldc, dC, typeT, orderC));

            for(hipsparseSpMMAlg_t alg : algs)
            {
                size_t bufferSize;
                CHECK_HIPSPARSE_ERROR(hipsparseSpMM_bufferSize(
                    handle, transA, transB, &alpha, matA, B, &beta, C, typeT, alg, &bufferSize));
                buffer_size_max = std::max(buffer_size_max, bufferSize);
            }

            CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
            CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C));
        }
    }

    // Ensure a non-null buffer even if no scratch space is required.
    buffer_size_max = std::max(buffer_size_max, size_t(4));

    void* buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size_max));

    // Step 2: repeatedly loop over every configuration and call hipsparseSpMM
    // with the shared buffer, never calling bufferSize again. Verify each
    // call's result against a CPU reference.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t transA : ops_A)
        {
            for(hipsparseOperation_t transB : ops_B)
            {
                J       B_m, B_n, C_m, C_n;
                int64_t ldb, ldc;
                spmm_reuse_dims<J>(M, K, N, transA, transB, B_m, B_n, C_m, C_n, ldb, ldc);

                const int64_t nnz_B = ldb * B_n;
                const int64_t nnz_C = ldc * C_n;

                std::vector<T> hB(nnz_B);
                hipsparseInit<T>(hB, nnz_B, 1);

                auto dB_managed
                    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
                auto dC_managed
                    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
                T* dB = (T*)dB_managed.get();
                T* dC = (T*)dC_managed.get();

                CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));

                hipsparseDnMatDescr_t B, C;
                CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));
                CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C, C_m, C_n, ldc, dC, typeT, orderC));

                for(hipsparseSpMMAlg_t alg : algs)
                {
                    std::vector<T> hC(nnz_C);
                    hipsparseInit<T>(hC, nnz_C, 1);

                    CHECK_HIP_ERROR(
                        hipMemcpy(dC, hC.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));

                    if(use_preprocess)
                    {
                        CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
                            handle, transA, transB, &alpha, matA, B, &beta, C, typeT, alg, buffer));
                    }

                    CHECK_HIPSPARSE_ERROR(hipsparseSpMM(
                        handle, transA, transB, &alpha, matA, B, &beta, C, typeT, alg, buffer));

                    std::vector<T> hC_out(nnz_C);
                    CHECK_HIP_ERROR(
                        hipMemcpy(hC_out.data(), dC, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

                    std::vector<T> hC_gold(hC);
                    host_csrmm<I, J, T>(M,
                                        N,
                                        K,
                                        transA,
                                        transB,
                                        alpha,
                                        hcsr_row_ptr.data(),
                                        hcsr_col_ind.data(),
                                        hcsr_val.data(),
                                        hB.data(),
                                        ldb,
                                        orderB,
                                        beta,
                                        hC_gold.data(),
                                        ldc,
                                        orderC,
                                        idx_base,
                                        false);

                    unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_out.data());
                }

                CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
                CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C));
            }
        }
    }

    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename I, typename J, typename T>
void testing_spmm_csr_reuse_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    J                    m        = argus.M;
    J                    n        = argus.N;
    J                    k        = argus.K;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    // Index and data types
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures. The sparse matrix descriptor is created once as an
    // (m x k) CSR matrix and reused across every operation below.
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz_A;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename, m, k, nnz_A, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base));

    // Redefine sparse matrix values
    hipsparseInit<T>(hcsr_val, hcsr_val.size(), 1);

    // allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_A), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};

    I* dptr = (I*)dptr_managed.get();
    J* dcol = (J*)dcol_managed.get();
    T* dval = (T*)dval_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));

    // Create matrix
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&matA, m, k, nnz_A, dptr, dcol, dval, typeI, typeJ, idx_base, typeT));

    const std::vector<hipsparseOperation_t> ops_A
        = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
    const std::vector<hipsparseOperation_t> ops_B
        = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
    const std::vector<hipsparseSpMMAlg_t> algs
        = {HIPSPARSE_SPMM_ALG_DEFAULT, HIPSPARSE_SPMM_CSR_ALG1, HIPSPARSE_SPMM_CSR_ALG2};

    constexpr int number_of_passes = 3;

    // Scenario 1: per-call bufferSize / buffer allocation. Exercises that the
    // same sparse matrix descriptor produces correct results when bufferSize is
    // re-queried and the externalBuffer re-allocated on every call across all
    // configurations.
    for(bool use_preprocess : {false, true})
    {
        for(int pass = 0; pass < number_of_passes; ++pass)
        {
            for(hipsparseOperation_t transA : ops_A)
            {
                for(hipsparseOperation_t transB : ops_B)
                {
                    for(hipsparseSpMMAlg_t alg : algs)
                    {
                        call_spmm<I, J, T>(handle,
                                           matA,
                                           m,
                                           n,
                                           k,
                                           hcsr_row_ptr,
                                           hcsr_col_ind,
                                           hcsr_val,
                                           h_alpha,
                                           h_beta,
                                           idx_base,
                                           transA,
                                           transB,
                                           alg,
                                           use_preprocess);
                    }
                }
            }
        }
    }

    // Scenario 2: bufferSize is queried once per configuration up front, a
    // single externalBuffer is allocated to the max of those sizes, and
    // hipsparseSpMM is then called repeatedly across configurations with that
    // one shared buffer (no further bufferSize calls).
    for(bool use_preprocess : {false, true})
    {
        call_spmm_shared_buffer<I, J, T>(handle,
                                         matA,
                                         m,
                                         n,
                                         k,
                                         hcsr_row_ptr,
                                         hcsr_col_ind,
                                         hcsr_val,
                                         h_alpha,
                                         h_beta,
                                         idx_base,
                                         ops_A,
                                         ops_B,
                                         algs,
                                         number_of_passes,
                                         use_preprocess);
    }

    // Destroy matrix
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
#endif
}

#endif // TESTING_SPMM_CSR_REUSE_DESCR_HPP
