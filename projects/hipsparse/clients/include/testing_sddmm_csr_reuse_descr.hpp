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
#ifndef TESTING_SDDMM_CSR_REUSE_DESCR_HPP
#define TESTING_SDDMM_CSR_REUSE_DESCR_HPP

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
void testing_sddmm_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// Computes the stored dimensions, leading dimensions and element counts of the
// dense A and B matrices for a given (transA, transB, orderA, orderB)
// configuration. The sampled sparse output C is always (m x n).
//   C(m x n) = alpha * (op(A)(m x k) * op(B)(k x n)) .* spy(C) + beta * C
template <typename J>
static void sddmm_reuse_dense_dims(J                    m,
                                   J                    n,
                                   J                    k,
                                   hipsparseOperation_t transA,
                                   hipsparseOperation_t transB,
                                   hipsparseOrder_t     orderA,
                                   hipsparseOrder_t     orderB,
                                   J&                   A_m,
                                   J&                   A_n,
                                   J&                   B_m,
                                   J&                   B_n,
                                   int64_t&             lda,
                                   int64_t&             ldb,
                                   int64_t&             nnz_A,
                                   int64_t&             nnz_B)
{
    A_m = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
    A_n = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
    B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
    B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;

    lda = std::max(int64_t(1), (orderA == HIPSPARSE_ORDER_COL) ? int64_t(A_m) : int64_t(A_n));
    ldb = std::max(int64_t(1), (orderB == HIPSPARSE_ORDER_COL) ? int64_t(B_m) : int64_t(B_n));

    const int64_t nrowA = (orderA == HIPSPARSE_ORDER_COL) ? lda : int64_t(A_m);
    const int64_t ncolA = (orderA == HIPSPARSE_ORDER_COL) ? int64_t(A_n) : lda;
    const int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : int64_t(B_m);
    const int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? int64_t(B_n) : ldb;

    nnz_A = nrowA * ncolA;
    nnz_B = nrowB * ncolB;
}

// Exercises the per-call bufferSize / per-call buffer-allocation pattern for a
// single (transA, transB, orderA, orderB) configuration: query
// hipsparseSDDMM_bufferSize, hipMalloc a fresh tempBuffer of that size,
// optionally run hipsparseSDDMM_preprocess, then run hipsparseSDDMM in both
// host- and device-pointer modes. The sparse-matrix descriptor is reused and
// its values are reset before every call.
template <typename I, typename J, typename T>
static void call_sddmm(hipsparseHandle_t&     handle,
                       hipsparseSpMatDescr_t& matC,
                       T*                     dval,
                       J                      M,
                       J                      N,
                       J                      K,
                       I                      nnz,
                       std::vector<I>&        hcsr_row_ptr,
                       std::vector<J>&        hcsr_col_ind,
                       std::vector<T>&        hcsr_val,
                       T                      alpha,
                       T                      beta,
                       hipsparseIndexBase_t   idx_base,
                       hipsparseOperation_t   transA,
                       hipsparseOperation_t   transB,
                       hipsparseOrder_t       orderA,
                       hipsparseOrder_t       orderB,
                       hipsparseSDDMMAlg_t    alg,
                       bool                   use_preprocess)
{
    hipDataType typeT = getDataType<T>();

    J       A_m, A_n, B_m, B_n;
    int64_t lda, ldb, nnz_A, nnz_B;
    sddmm_reuse_dense_dims<J>(
        M, N, K, transA, transB, orderA, orderB, A_m, A_n, B_m, B_n, lda, ldb, nnz_A, nnz_B);

    std::vector<T> hA(nnz_A);
    std::vector<T> hB(nnz_B);
    hipsparseInit<T>(hA, nnz_A, 1);
    hipsparseInit<T>(hB, nnz_B, 1);

    auto dA_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dA      = (T*)dA_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &beta, sizeof(T), hipMemcpyHostToDevice));

    hipsparseDnMatDescr_t A, B;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&A, A_m, A_n, lda, dA, typeT, orderA));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));

    // Query SDDMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_bufferSize(
        handle, transA, transB, &alpha, A, B, &beta, matC, typeT, alg, &bufferSize));

    // Ensure the tempBuffer pointer is never null even when no scratch space is
    // required.
    bufferSize = std::max(bufferSize, size_t(4));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // Compute the host reference once: it is identical for both pointer modes.
    std::vector<T> hval_gold(hcsr_val);
    host_sddmm_csr(M,
                   N,
                   K,
                   nnz,
                   alpha,
                   hA.data(),
                   lda,
                   orderA,
                   transA,
                   hB.data(),
                   ldb,
                   orderB,
                   transB,
                   beta,
                   hval_gold.data(),
                   hcsr_row_ptr.data(),
                   hcsr_col_ind.data(),
                   idx_base);

    // HIPSPARSE pointer mode host
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    if(use_preprocess)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, matC, typeT, alg, buffer));
    }
    CHECK_HIPSPARSE_ERROR(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, B, &beta, matC, typeT, alg, buffer));

    std::vector<T> hval_1(nnz);
    CHECK_HIP_ERROR(hipMemcpy(hval_1.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));

    // HIPSPARSE pointer mode device
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    if(use_preprocess)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(
            handle, transA, transB, d_alpha, A, B, d_beta, matC, typeT, alg, buffer));
    }
    CHECK_HIPSPARSE_ERROR(
        hipsparseSDDMM(handle, transA, transB, d_alpha, A, B, d_beta, matC, typeT, alg, buffer));

    std::vector<T> hval_2(nnz);
    CHECK_HIP_ERROR(hipMemcpy(hval_2.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));

    unit_check_near(1, nnz, 1, hval_gold.data(), hval_1.data());
    unit_check_near(1, nnz, 1, hval_gold.data(), hval_2.data());

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
}

// Exercises the multi-configuration shared-buffer path: query bufferSize once
// per configuration, allocate a single tempBuffer sized to the max, then
// repeatedly call hipsparseSDDMM alternating among the configurations without
// ever calling hipsparseSDDMM_bufferSize again. The dense A and B descriptors
// are recreated per configuration (their logical dimensions change with the
// operations and orders) while the same sparse-matrix descriptor and the same
// user-provided buffer are reused throughout.
template <typename I, typename J, typename T>
static void call_sddmm_shared_buffer(hipsparseHandle_t&                       handle,
                                     hipsparseSpMatDescr_t&                   matC,
                                     T*                                       dval,
                                     J                                        M,
                                     J                                        N,
                                     J                                        K,
                                     I                                        nnz,
                                     std::vector<I>&                          hcsr_row_ptr,
                                     std::vector<J>&                          hcsr_col_ind,
                                     std::vector<T>&                          hcsr_val,
                                     T                                        alpha,
                                     T                                        beta,
                                     hipsparseIndexBase_t                     idx_base,
                                     const std::vector<hipsparseOperation_t>& ops_A,
                                     const std::vector<hipsparseOperation_t>& ops_B,
                                     const std::vector<hipsparseOrder_t>&     orders_A,
                                     const std::vector<hipsparseOrder_t>&     orders_B,
                                     hipsparseSDDMMAlg_t                      alg,
                                     int                                      number_of_passes,
                                     bool                                     use_preprocess)
{
    hipDataType typeT = getDataType<T>();

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Step 1: query the bufferSize for every configuration we will use, and
    // take the max. The tempBuffer we allocate below is reused for all
    // configurations.
    size_t buffer_size_max = 0;
    for(hipsparseOperation_t transA : ops_A)
    {
        for(hipsparseOperation_t transB : ops_B)
        {
            for(hipsparseOrder_t orderA : orders_A)
            {
                for(hipsparseOrder_t orderB : orders_B)
                {
                    J       A_m, A_n, B_m, B_n;
                    int64_t lda, ldb, nnz_A, nnz_B;
                    sddmm_reuse_dense_dims<J>(M,
                                              N,
                                              K,
                                              transA,
                                              transB,
                                              orderA,
                                              orderB,
                                              A_m,
                                              A_n,
                                              B_m,
                                              B_n,
                                              lda,
                                              ldb,
                                              nnz_A,
                                              nnz_B);

                    auto dA_managed
                        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
                    auto dB_managed
                        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
                    T* dA = (T*)dA_managed.get();
                    T* dB = (T*)dB_managed.get();

                    hipsparseDnMatDescr_t A, B;
                    CHECK_HIPSPARSE_ERROR(
                        hipsparseCreateDnMat(&A, A_m, A_n, lda, dA, typeT, orderA));
                    CHECK_HIPSPARSE_ERROR(
                        hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));

                    size_t bufferSize;
                    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_bufferSize(handle,
                                                                    transA,
                                                                    transB,
                                                                    &alpha,
                                                                    A,
                                                                    B,
                                                                    &beta,
                                                                    matC,
                                                                    typeT,
                                                                    alg,
                                                                    &bufferSize));
                    buffer_size_max = std::max(buffer_size_max, bufferSize);

                    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(A));
                    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
                }
            }
        }
    }

    // Ensure a non-null buffer even if no scratch space is required.
    buffer_size_max = std::max(buffer_size_max, size_t(4));

    void* buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size_max));

    // Step 2: repeatedly loop over every configuration and call hipsparseSDDMM
    // with the shared buffer, never calling bufferSize again. Verify each
    // call's result against a CPU reference.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t transA : ops_A)
        {
            for(hipsparseOperation_t transB : ops_B)
            {
                for(hipsparseOrder_t orderA : orders_A)
                {
                    for(hipsparseOrder_t orderB : orders_B)
                    {
                        J       A_m, A_n, B_m, B_n;
                        int64_t lda, ldb, nnz_A, nnz_B;
                        sddmm_reuse_dense_dims<J>(M,
                                                  N,
                                                  K,
                                                  transA,
                                                  transB,
                                                  orderA,
                                                  orderB,
                                                  A_m,
                                                  A_n,
                                                  B_m,
                                                  B_n,
                                                  lda,
                                                  ldb,
                                                  nnz_A,
                                                  nnz_B);

                        std::vector<T> hA(nnz_A);
                        std::vector<T> hB(nnz_B);
                        hipsparseInit<T>(hA, nnz_A, 1);
                        hipsparseInit<T>(hB, nnz_B, 1);

                        auto dA_managed
                            = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
                        auto dB_managed
                            = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
                        T* dA = (T*)dA_managed.get();
                        T* dB = (T*)dB_managed.get();

                        CHECK_HIP_ERROR(
                            hipMemcpy(dA, hA.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
                        CHECK_HIP_ERROR(
                            hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));

                        hipsparseDnMatDescr_t A, B;
                        CHECK_HIPSPARSE_ERROR(
                            hipsparseCreateDnMat(&A, A_m, A_n, lda, dA, typeT, orderA));
                        CHECK_HIPSPARSE_ERROR(
                            hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));

                        // Reset the sparse matrix values before each call.
                        CHECK_HIP_ERROR(hipMemcpy(
                            dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

                        if(use_preprocess)
                        {
                            CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(handle,
                                                                            transA,
                                                                            transB,
                                                                            &alpha,
                                                                            A,
                                                                            B,
                                                                            &beta,
                                                                            matC,
                                                                            typeT,
                                                                            alg,
                                                                            buffer));
                        }

                        CHECK_HIPSPARSE_ERROR(hipsparseSDDMM(
                            handle, transA, transB, &alpha, A, B, &beta, matC, typeT, alg, buffer));

                        std::vector<T> hval_out(nnz);
                        CHECK_HIP_ERROR(hipMemcpy(
                            hval_out.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));

                        std::vector<T> hval_gold(hcsr_val);
                        host_sddmm_csr(M,
                                       N,
                                       K,
                                       nnz,
                                       alpha,
                                       hA.data(),
                                       lda,
                                       orderA,
                                       transA,
                                       hB.data(),
                                       ldb,
                                       orderB,
                                       transB,
                                       beta,
                                       hval_gold.data(),
                                       hcsr_row_ptr.data(),
                                       hcsr_col_ind.data(),
                                       idx_base);

                        unit_check_near(1, nnz, 1, hval_gold.data(), hval_out.data());

                        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(A));
                        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
                    }
                }
            }
        }
    }

    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename I, typename J, typename T>
void testing_sddmm_csr_reuse_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION))
    J                    m        = argus.M;
    J                    n        = argus.N;
    J                    k        = argus.K;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseIndexBase_t idx_base = argus.baseC;
    std::string          filename = argus.filename;

    // Index and data types
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures. The sparse matrix descriptor is created once as an
    // (m x n) CSR matrix and reused across every operation below.
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz = 0;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base));

    // allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    I* dptr = (I*)dptr_managed.get();
    J* dcol = (J*)dcol_managed.get();
    T* dval = (T*)dval_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Create the sparse matrix C once and reuse it across all configurations.
    hipsparseSpMatDescr_t matC;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&matC, m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT));

    const std::vector<hipsparseOperation_t> ops_A
        = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
    const std::vector<hipsparseOperation_t> ops_B
        = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
    const std::vector<hipsparseOrder_t> orders_A = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
    const std::vector<hipsparseOrder_t> orders_B = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};

    const hipsparseSDDMMAlg_t alg = HIPSPARSE_SDDMM_ALG_DEFAULT;

    constexpr int number_of_passes = 3;

    // Scenario 1: per-call bufferSize / buffer allocation. Exercises that the
    // same sparse matrix descriptor produces correct results when bufferSize is
    // re-queried and the tempBuffer re-allocated on every call across all
    // configurations.
    for(bool use_preprocess : {false, true})
    {
        for(int pass = 0; pass < number_of_passes; ++pass)
        {
            for(hipsparseOperation_t transA : ops_A)
            {
                for(hipsparseOperation_t transB : ops_B)
                {
                    for(hipsparseOrder_t orderA : orders_A)
                    {
                        for(hipsparseOrder_t orderB : orders_B)
                        {
                            call_sddmm<I, J, T>(handle,
                                                matC,
                                                dval,
                                                m,
                                                n,
                                                k,
                                                nnz,
                                                hcsr_row_ptr,
                                                hcsr_col_ind,
                                                hcsr_val,
                                                h_alpha,
                                                h_beta,
                                                idx_base,
                                                transA,
                                                transB,
                                                orderA,
                                                orderB,
                                                alg,
                                                use_preprocess);
                        }
                    }
                }
            }
        }
    }

    // Scenario 2: bufferSize is queried once per configuration up front, a
    // single tempBuffer is allocated to the max of those sizes, and
    // hipsparseSDDMM is then called repeatedly across configurations with that
    // one shared buffer (no further bufferSize calls).
    for(bool use_preprocess : {false, true})
    {
        call_sddmm_shared_buffer<I, J, T>(handle,
                                          matC,
                                          dval,
                                          m,
                                          n,
                                          k,
                                          nnz,
                                          hcsr_row_ptr,
                                          hcsr_col_ind,
                                          hcsr_val,
                                          h_alpha,
                                          h_beta,
                                          idx_base,
                                          ops_A,
                                          ops_B,
                                          orders_A,
                                          orders_B,
                                          alg,
                                          number_of_passes,
                                          use_preprocess);
    }

    // Destroy matrix
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matC));
#endif
}

#endif // TESTING_SDDMM_CSR_REUSE_DESCR_HPP
