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
#ifndef TESTING_SPSM_CSR_REUSE_DESCR_HPP
#define TESTING_SPSM_CSR_REUSE_DESCR_HPP

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
void testing_spsm_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// Exercises the per-call bufferSize / per-call buffer-allocation pattern for a
// single (transA, fill mode, diagonal type, algorithm) configuration: set the
// fill-mode / diagonal-type attributes on the shared sparse-matrix descriptor,
// query hipsparseSpSM_bufferSize, hipMalloc a fresh externalBuffer of that
// size, create a fresh SpSM descriptor, run hipsparseSpSM_analysis and
// hipsparseSpSM_solve. When called repeatedly with the same configuration on
// the same matA, the wrapper must continue to return correct results despite
// the bufferSize / buffer / SpSM descriptor being re-created on every call.
// transB is fixed to NON_TRANSPOSE and B/C use column-major order to keep
// focus on sparse-matrix descriptor reuse.
template <typename I, typename J, typename T>
static void call_spsm(hipsparseHandle_t&     handle,
                      hipsparseSpMatDescr_t& matA,
                      J                      m,
                      J                      k,
                      I                      nnz,
                      std::vector<I>&        hcsr_row_ptr,
                      std::vector<J>&        hcsr_col_ind,
                      std::vector<T>&        hcsr_val,
                      T                      alpha,
                      hipsparseIndexBase_t   idx_base,
                      hipsparseOperation_t   transA,
                      hipsparseFillMode_t    uplo,
                      hipsparseDiagType_t    diag,
                      hipsparseSpSMAlg_t     alg)
{
    hipDataType typeT = getDataType<T>();

    // B: (m x k) col-major, ldb = m
    // C: (m x k) col-major, ldc = m
    const int64_t ldb   = std::max(int64_t(1), int64_t(m));
    const int64_t ldc   = std::max(int64_t(1), int64_t(m));
    const int64_t nnz_B = ldb * k;
    const int64_t nnz_C = ldc * k;

    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, 1, nnz_B);

    hC_1    = hB;
    hC_2    = hC_1;
    hC_gold = hC_1;

    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    // Set fill-mode / diagonal-type attributes on the shared descriptor.
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateDnMat(&B, m, k, ldb, dB, typeT, HIPSPARSE_ORDER_COL));
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateDnMat(&C1, m, k, ldc, dC_1, typeT, HIPSPARSE_ORDER_COL));
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateDnMat(&C2, m, k, ldc, dC_2, typeT, HIPSPARSE_ORDER_COL));

    hipsparseSpSMDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_createDescr(&descr));

    // Query SpSM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_bufferSize(handle,
                                 transA,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 matA,
                                 B,
                                 C1,
                                 typeT,
                                 alg,
                                 descr,
                                 &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_analysis(handle,
                               transA,
                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha,
                               matA,
                               B,
                               C1,
                               typeT,
                               alg,
                               descr,
                               buffer));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_solve(handle,
                            transA,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            matA,
                            B,
                            C1,
                            typeT,
                            alg,
                            descr,
                            buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_analysis(handle,
                               transA,
                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               d_alpha,
                               matA,
                               B,
                               C2,
                               typeT,
                               alg,
                               descr,
                               buffer));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_solve(handle,
                            transA,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            d_alpha,
                            matA,
                            B,
                            C2,
                            typeT,
                            alg,
                            descr,
                            buffer));

    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    // Host SpSM reference
    J struct_pivot  = -1;
    J numeric_pivot = -1;
    host_csrsm(m,
               k,
               nnz,
               transA,
               HIPSPARSE_OPERATION_NON_TRANSPOSE,
               alpha,
               hcsr_row_ptr,
               hcsr_col_ind,
               hcsr_val,
               hB,
               (J)ldb,
               HIPSPARSE_ORDER_COL,
               hC_gold,
               (J)ldc,
               HIPSPARSE_ORDER_COL,
               diag,
               uplo,
               idx_base,
               &struct_pivot,
               &numeric_pivot);

    // Only validate when the triangular system is non-singular (no structural
    // or numerical pivot was encountered by the host reference).
    if(struct_pivot == -1 && numeric_pivot == -1)
    {
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_destroyDescr(descr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));
}

// Exercises the multi-configuration shared-buffer path: query bufferSize once
// per (transA, fill mode, diagonal type, algorithm) configuration, allocate a
// single externalBuffer sized to the max, then repeatedly run
// hipsparseSpSM_analysis / hipsparseSpSM_solve alternating among the
// configurations without ever calling hipsparseSpSM_bufferSize again. A fresh
// SpSM descriptor is created per configuration while the same sparse-matrix
// descriptor and the same user-provided buffer are reused throughout.
// transB is fixed to NON_TRANSPOSE and B/C use column-major order.
template <typename I, typename J, typename T>
static void call_spsm_shared_buffer(hipsparseHandle_t&                       handle,
                                    hipsparseSpMatDescr_t&                   matA,
                                    J                                        m,
                                    J                                        k,
                                    I                                        nnz,
                                    std::vector<I>&                          hcsr_row_ptr,
                                    std::vector<J>&                          hcsr_col_ind,
                                    std::vector<T>&                          hcsr_val,
                                    T                                        alpha,
                                    hipsparseIndexBase_t                     idx_base,
                                    const std::vector<hipsparseOperation_t>& ops,
                                    const std::vector<hipsparseFillMode_t>&  uplos,
                                    const std::vector<hipsparseDiagType_t>&  diags,
                                    const std::vector<hipsparseSpSMAlg_t>&   algs,
                                    int                                      number_of_passes)
{
    hipDataType typeT = getDataType<T>();

    // B: (m x k) col-major, ldb = m
    // C: (m x k) col-major, ldc = m
    const int64_t ldb   = std::max(int64_t(1), int64_t(m));
    const int64_t ldc   = std::max(int64_t(1), int64_t(m));
    const int64_t nnz_B = ldb * k;
    const int64_t nnz_C = ldc * k;

    std::vector<T> hB(nnz_B);
    hipsparseInit<T>(hB, 1, nnz_B);

    auto dB_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    T*   dB         = (T*)dB_managed.get();
    T*   dC         = (T*)dC_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Step 1: query the bufferSize for every (transA, fill mode, diag type,
    // algorithm) configuration we will use, and take the max. The
    // externalBuffer allocated below is reused for all configurations.
    size_t buffer_size_max = 0;
    for(hipsparseOperation_t op : ops)
    {
        for(hipsparseFillMode_t uplo : uplos)
        {
            for(hipsparseDiagType_t diag : diags)
            {
                CHECK_HIPSPARSE_ERROR(hipsparseSpMatSetAttribute(
                    matA, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));
                CHECK_HIPSPARSE_ERROR(hipsparseSpMatSetAttribute(
                    matA, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

                hipsparseDnMatDescr_t B, C;
                CHECK_HIPSPARSE_ERROR(
                    hipsparseCreateDnMat(&B, m, k, ldb, dB, typeT, HIPSPARSE_ORDER_COL));
                CHECK_HIPSPARSE_ERROR(
                    hipsparseCreateDnMat(&C, m, k, ldc, dC, typeT, HIPSPARSE_ORDER_COL));

                for(hipsparseSpSMAlg_t alg : algs)
                {
                    hipsparseSpSMDescr_t descr;
                    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_createDescr(&descr));

                    size_t bufferSize;
                    CHECK_HIPSPARSE_ERROR(
                        hipsparseSpSM_bufferSize(handle,
                                                 op,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 matA,
                                                 B,
                                                 C,
                                                 typeT,
                                                 alg,
                                                 descr,
                                                 &bufferSize));
                    buffer_size_max = std::max(buffer_size_max, bufferSize);

                    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_destroyDescr(descr));
                }

                CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
                CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C));
            }
        }
    }

    void* buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size_max));

    // Step 2: repeatedly loop over every configuration and run SpSM analysis /
    // solve with the shared buffer, never calling bufferSize again. Verify each
    // call's result against a CPU reference.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t op : ops)
        {
            for(hipsparseFillMode_t uplo : uplos)
            {
                for(hipsparseDiagType_t diag : diags)
                {
                    CHECK_HIPSPARSE_ERROR(hipsparseSpMatSetAttribute(
                        matA, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));
                    CHECK_HIPSPARSE_ERROR(hipsparseSpMatSetAttribute(
                        matA, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

                    for(hipsparseSpSMAlg_t alg : algs)
                    {
                        std::vector<T> hC(nnz_C);
                        hipsparseInit<T>(hC, 1, nnz_C);

                        CHECK_HIP_ERROR(
                            hipMemcpy(dC, hC.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));

                        hipsparseDnMatDescr_t B, C;
                        CHECK_HIPSPARSE_ERROR(
                            hipsparseCreateDnMat(&B, m, k, ldb, dB, typeT, HIPSPARSE_ORDER_COL));
                        CHECK_HIPSPARSE_ERROR(
                            hipsparseCreateDnMat(&C, m, k, ldc, dC, typeT, HIPSPARSE_ORDER_COL));

                        hipsparseSpSMDescr_t descr;
                        CHECK_HIPSPARSE_ERROR(hipsparseSpSM_createDescr(&descr));

                        CHECK_HIPSPARSE_ERROR(
                            hipsparseSpSM_analysis(handle,
                                                   op,
                                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matA,
                                                   B,
                                                   C,
                                                   typeT,
                                                   alg,
                                                   descr,
                                                   buffer));
                        CHECK_HIPSPARSE_ERROR(
                            hipsparseSpSM_solve(handle,
                                               op,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               B,
                                               C,
                                               typeT,
                                               alg,
                                               descr,
                                               buffer));

                        std::vector<T> hC_out(nnz_C);
                        CHECK_HIP_ERROR(hipMemcpy(
                            hC_out.data(), dC, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

                        std::vector<T> hC_gold(hC);
                        J              struct_pivot  = -1;
                        J              numeric_pivot = -1;
                        host_csrsm(m,
                                   k,
                                   nnz,
                                   op,
                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                   alpha,
                                   hcsr_row_ptr,
                                   hcsr_col_ind,
                                   hcsr_val,
                                   hB,
                                   (J)ldb,
                                   HIPSPARSE_ORDER_COL,
                                   hC_gold,
                                   (J)ldc,
                                   HIPSPARSE_ORDER_COL,
                                   diag,
                                   uplo,
                                   idx_base,
                                   &struct_pivot,
                                   &numeric_pivot);

                        if(struct_pivot == -1 && numeric_pivot == -1)
                        {
                            unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_out.data());
                        }

                        CHECK_HIPSPARSE_ERROR(hipsparseSpSM_destroyDescr(descr));
                        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
                        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C));
                    }
                }
            }
        }
    }

    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename I, typename J, typename T>
void testing_spsm_csr_reuse_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    J                    m        = argus.M;
    J                    n        = argus.N;
    J                    k        = argus.K;
    T                    h_alpha  = argus.get_alpha<T>();
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures. The sparse matrix descriptor is created once as a
    // square (m x m) CSR matrix and reused across every configuration below.
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    // SpSM requires a square sparse matrix.
    n = m;

    I nnz;
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

    // Create matrix
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&matA, m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT));

    const std::vector<hipsparseOperation_t> ops
        = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
    const std::vector<hipsparseFillMode_t> uplos
        = {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER};
    const std::vector<hipsparseDiagType_t> diags
        = {HIPSPARSE_DIAG_TYPE_NON_UNIT, HIPSPARSE_DIAG_TYPE_UNIT};
    const std::vector<hipsparseSpSMAlg_t> algs = {HIPSPARSE_SPSM_ALG_DEFAULT};

    constexpr int number_of_passes = 3;

    // Scenario 1: per-call bufferSize / buffer allocation. Exercises that the
    // same sparse matrix descriptor produces correct results when bufferSize is
    // re-queried, the externalBuffer re-allocated, and the SpSM descriptor
    // re-created on every call across all configurations.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t op : ops)
        {
            for(hipsparseFillMode_t uplo : uplos)
            {
                for(hipsparseDiagType_t diag : diags)
                {
                    for(hipsparseSpSMAlg_t alg : algs)
                    {
                        call_spsm<I, J, T>(handle,
                                           matA,
                                           m,
                                           k,
                                           nnz,
                                           hcsr_row_ptr,
                                           hcsr_col_ind,
                                           hcsr_val,
                                           h_alpha,
                                           idx_base,
                                           op,
                                           uplo,
                                           diag,
                                           alg);
                    }
                }
            }
        }
    }

    // Scenario 2: bufferSize is queried once per configuration up front, a
    // single externalBuffer is allocated to the max of those sizes, and
    // hipsparseSpSM_analysis / hipsparseSpSM_solve are then called repeatedly
    // across configurations with that one shared buffer (no further bufferSize
    // calls).
    call_spsm_shared_buffer<I, J, T>(handle,
                                     matA,
                                     m,
                                     k,
                                     nnz,
                                     hcsr_row_ptr,
                                     hcsr_col_ind,
                                     hcsr_val,
                                     h_alpha,
                                     idx_base,
                                     ops,
                                     uplos,
                                     diags,
                                     algs,
                                     number_of_passes);

    // Destroy matrix
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
#endif
}

#endif // TESTING_SPSM_CSR_REUSE_DESCR_HPP
