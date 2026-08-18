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
#ifndef TESTING_SPSV_CSR_REUSE_DESCR_HPP
#define TESTING_SPSV_CSR_REUSE_DESCR_HPP

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
void testing_spsv_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// Exercises the per-call bufferSize / per-call buffer-allocation pattern for a
// single (operation, fill mode, diagonal type, algorithm) configuration: set
// the fill-mode / diagonal-type attributes on the shared sparse-matrix
// descriptor, query hipsparseSpSV_bufferSize, hipMalloc a fresh externalBuffer
// of that size, create a fresh SpSV descriptor, run hipsparseSpSV_analysis and
// hipsparseSpSV_solve. When called repeatedly with the same configuration on
// the same matA, the wrapper must continue to return correct results despite
// the bufferSize / buffer / SpSV descriptor being re-created on every call.
template <typename I, typename J, typename T>
static void call_spsv(hipsparseHandle_t&     handle,
                      hipsparseSpMatDescr_t& matA,
                      J                      m,
                      I                      nnz,
                      std::vector<I>&        hcsr_row_ptr,
                      std::vector<J>&        hcsr_col_ind,
                      std::vector<T>&        hcsr_val,
                      T                      alpha,
                      hipsparseIndexBase_t   idx_base,
                      hipsparseOperation_t   transA,
                      hipsparseFillMode_t    uplo,
                      hipsparseDiagType_t    diag,
                      hipsparseSpSVAlg_t     alg)
{
    hipDataType typeT = getDataType<T>();

    std::vector<T> hx(m);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, m);
    hipsparseInit<T>(hy_1, 1, m);

    hy_2    = hy_1;
    hy_gold = hy_1;

    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dx      = (T*)dx_managed.get();
    T* dy_1    = (T*)dy_1_managed.get();
    T* dy_2    = (T*)dy_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    // Set the fill-mode / diagonal-type attributes on the shared descriptor.
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, m, dx, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, m, dy_1, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, m, dy_2, typeT));

    hipsparseSpSVDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));

    // Query SpSV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_bufferSize(
        handle, transA, &alpha, matA, x, y1, typeT, alg, descr, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_analysis(handle, transA, &alpha, matA, x, y1, typeT, alg, descr, buffer));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_solve(handle, transA, &alpha, matA, x, y1, typeT, alg, descr));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_analysis(handle, transA, d_alpha, matA, x, y2, typeT, alg, descr, buffer));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_solve(handle, transA, d_alpha, matA, x, y2, typeT, alg, descr));

    CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

    // Host SpSV
    J struct_pivot  = -1;
    J numeric_pivot = -1;
    host_csrsv(transA,
               m,
               nnz,
               alpha,
               hcsr_row_ptr.data(),
               hcsr_col_ind.data(),
               hcsr_val.data(),
               hx.data(),
               hy_gold.data(),
               diag,
               uplo,
               idx_base,
               &struct_pivot,
               &numeric_pivot);

    // Only validate when the triangular system is non-singular (no structural
    // or numerical pivot was encountered by the host reference).
    if(struct_pivot == (m + 1) && numeric_pivot == (m + 1))
    {
        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
}

// Exercises the multi-configuration shared-buffer path: query bufferSize once
// per (operation, fill mode, diagonal type, algorithm) configuration, allocate
// a single externalBuffer sized to the max, then repeatedly run
// hipsparseSpSV_analysis / hipsparseSpSV_solve alternating among the
// configurations without ever calling hipsparseSpSV_bufferSize again. A fresh
// SpSV descriptor is created per configuration while the same sparse-matrix
// descriptor and the same user-provided buffer are reused throughout.
template <typename I, typename J, typename T>
static void call_spsv_shared_buffer(hipsparseHandle_t&                       handle,
                                    hipsparseSpMatDescr_t&                   matA,
                                    J                                        m,
                                    I                                        nnz,
                                    std::vector<I>&                          hcsr_row_ptr,
                                    std::vector<J>&                          hcsr_col_ind,
                                    std::vector<T>&                          hcsr_val,
                                    T                                        alpha,
                                    hipsparseIndexBase_t                     idx_base,
                                    const std::vector<hipsparseOperation_t>& ops,
                                    const std::vector<hipsparseFillMode_t>&  uplos,
                                    const std::vector<hipsparseDiagType_t>&  diags,
                                    const std::vector<hipsparseSpSVAlg_t>&   algs,
                                    int                                      number_of_passes)
{
    hipDataType typeT = getDataType<T>();

    std::vector<T> hx(m);
    hipsparseInit<T>(hx, 1, m);

    auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    T*   dx         = (T*)dx_managed.get();
    T*   dy         = (T*)dy_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Step 1: query the bufferSize for every configuration we will use, and
    // take the max. The externalBuffer we allocate below is reused for all
    // configurations.
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

                hipsparseDnVecDescr_t x, y;
                CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, m, dx, typeT));
                CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, m, dy, typeT));

                for(hipsparseSpSVAlg_t alg : algs)
                {
                    hipsparseSpSVDescr_t descr;
                    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));

                    size_t bufferSize;
                    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_bufferSize(
                        handle, op, &alpha, matA, x, y, typeT, alg, descr, &bufferSize));
                    buffer_size_max = std::max(buffer_size_max, bufferSize);

                    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));
                }

                CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
                CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));
            }
        }
    }

    void* buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size_max));

    // Step 2: repeatedly loop over every configuration and run SpSV analysis /
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

                    for(hipsparseSpSVAlg_t alg : algs)
                    {
                        std::vector<T> hy(m);
                        hipsparseInit<T>(hy, 1, m);

                        CHECK_HIP_ERROR(
                            hipMemcpy(dy, hy.data(), sizeof(T) * m, hipMemcpyHostToDevice));

                        hipsparseDnVecDescr_t x, y;
                        CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, m, dx, typeT));
                        CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, m, dy, typeT));

                        hipsparseSpSVDescr_t descr;
                        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));

                        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_analysis(
                            handle, op, &alpha, matA, x, y, typeT, alg, descr, buffer));
                        CHECK_HIPSPARSE_ERROR(
                            hipsparseSpSV_solve(handle, op, &alpha, matA, x, y, typeT, alg, descr));

                        std::vector<T> hy_out(m);
                        CHECK_HIP_ERROR(
                            hipMemcpy(hy_out.data(), dy, sizeof(T) * m, hipMemcpyDeviceToHost));

                        std::vector<T> hy_gold(hy);
                        J              struct_pivot  = -1;
                        J              numeric_pivot = -1;
                        host_csrsv(op,
                                   m,
                                   nnz,
                                   alpha,
                                   hcsr_row_ptr.data(),
                                   hcsr_col_ind.data(),
                                   hcsr_val.data(),
                                   hx.data(),
                                   hy_gold.data(),
                                   diag,
                                   uplo,
                                   idx_base,
                                   &struct_pivot,
                                   &numeric_pivot);

                        if(struct_pivot == (m + 1) && numeric_pivot == (m + 1))
                        {
                            unit_check_near(1, m, 1, hy_gold.data(), hy_out.data());
                        }

                        CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));
                        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
                        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));
                    }
                }
            }
        }
    }

    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename I, typename J, typename T>
void testing_spsv_csr_reuse_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
    J                    m        = argus.M;
    J                    n        = argus.N;
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

    // SpSV requires a square sparse matrix.
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
    const std::vector<hipsparseSpSVAlg_t> algs = {HIPSPARSE_SPSV_ALG_DEFAULT};

    constexpr int number_of_passes = 3;

    // Scenario 1: per-call bufferSize / buffer allocation. Exercises that the
    // same sparse matrix descriptor produces correct results when bufferSize is
    // re-queried, the externalBuffer re-allocated, and the SpSV descriptor
    // re-created on every call across all configurations.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t op : ops)
        {
            for(hipsparseFillMode_t uplo : uplos)
            {
                for(hipsparseDiagType_t diag : diags)
                {
                    for(hipsparseSpSVAlg_t alg : algs)
                    {
                        call_spsv<I, J, T>(handle,
                                           matA,
                                           m,
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
    // hipsparseSpSV_analysis / hipsparseSpSV_solve are then called repeatedly
    // across configurations with that one shared buffer (no further bufferSize
    // calls).
    call_spsv_shared_buffer<I, J, T>(handle,
                                     matA,
                                     m,
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

#endif // TESTING_SPSV_CSR_REUSE_DESCR_HPP
