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
#ifndef TESTING_SPMV_CSR_REUSE_DESCR_HPP
#define TESTING_SPMV_CSR_REUSE_DESCR_HPP

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

template <typename I, typename J, typename A, typename X, typename Y, typename T>
void testing_spmv_csr_reuse_descr_bad_arg(const Arguments& argus)
{
}

// Exercises the SpMV-only path: never call hipsparseSpMV_bufferSize and
// never call hipsparseSpMV_preprocess -- only hipsparseSpMV itself, and
// pass a null externalBuffer.
template <typename I, typename J, typename A, typename X, typename Y, typename T>
static void call_spmv_only(hipsparseHandle_t&                       handle,
                           hipsparseSpMatDescr_t&                   matA,
                           J                                        m,
                           J                                        n,
                           I                                        nnz,
                           std::vector<I>&                          hcsr_row_ptr,
                           std::vector<J>&                          hcsr_col_ind,
                           std::vector<A>&                          hcsr_val,
                           T                                        alpha,
                           T                                        beta,
                           hipsparseIndexBase_t                     idx_base,
                           const std::vector<hipsparseOperation_t>& ops,
                           const std::vector<hipsparseSpMVAlg_t>&   algs,
                           int                                      number_of_passes)
{
    hipDataType xType       = getDataType<X>();
    hipDataType yType       = getDataType<Y>();
    hipDataType computeType = getDataType<T>();

    const J max_dim = std::max(m, n);

    std::vector<X> hx(max_dim);
    hipsparseInit<X>(hx, 1, max_dim);

    auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(X) * max_dim), device_free};
    auto dy_managed = hipsparse_unique_ptr{device_malloc(sizeof(Y) * max_dim), device_free};
    X*   dx         = (X*)dx_managed.get();
    Y*   dy         = (Y*)dy_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(X) * max_dim, hipMemcpyHostToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t op : ops)
        {
            const J x_size = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
            const J y_size = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;

            std::vector<Y> hy(y_size);
            hipsparseInit<Y>(hy, 1, y_size);

            hipsparseDnVecDescr_t x, y;
            CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, x_size, dx, xType));
            CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, y_size, dy, yType));

            for(hipsparseSpMVAlg_t alg : algs)
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(dy, hy.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));

                // No bufferSize, no preprocess: only SpMV with a null
                // externalBuffer. The wrapper must handle all analysis
                // and compute-buffer bookkeeping internally.
                CHECK_HIPSPARSE_ERROR(hipsparseSpMV(
                    handle, op, &alpha, matA, x, &beta, y, computeType, alg, nullptr));

                std::vector<Y> hy_out(y_size);
                CHECK_HIP_ERROR(
                    hipMemcpy(hy_out.data(), dy, sizeof(Y) * y_size, hipMemcpyDeviceToHost));

                std::vector<Y> hy_ref(hy);
                host_csrmv<I, J, A, X, Y, T>(op,
                                             m,
                                             n,
                                             nnz,
                                             alpha,
                                             hcsr_row_ptr.data(),
                                             hcsr_col_ind.data(),
                                             hcsr_val.data(),
                                             hx.data(),
                                             beta,
                                             hy_ref.data(),
                                             idx_base);

                unit_check_near(1, y_size, 1, hy_ref.data(), hy_out.data());
            }

            CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
            CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));
        }
    }
}

// Exercises the per-call bufferSize / per-call buffer-allocation pattern
// for a single (operation, algorithm) configuration: query
// hipsparseSpMV_bufferSize, hipMalloc a fresh externalBuffer of that size,
// optionally run hipsparseSpMV_preprocess, then run hipsparseSpMV. When
// called repeatedly with the same (op, alg) on the same matA, the wrapper must
// continue to return correct results despite the bufferSize / buffer
// being re-queried and re-allocated on every call.
template <typename I, typename J, typename A, typename X, typename Y, typename T>
static void call_spmv(hipsparseHandle_t&     handle,
                      hipsparseSpMatDescr_t& matA,
                      J                      m,
                      J                      n,
                      I                      nnz,
                      std::vector<I>&        hcsr_row_ptr,
                      std::vector<J>&        hcsr_col_ind,
                      std::vector<A>&        hcsr_val,
                      T                      alpha,
                      T                      beta,
                      hipsparseIndexBase_t   idx_base,
                      hipsparseOperation_t   transA,
                      hipsparseSpMVAlg_t     alg,
                      bool                   use_preprocess)
{
    hipDataType xType       = getDataType<X>();
    hipDataType yType       = getDataType<Y>();
    hipDataType computeType = getDataType<T>();

    // For non-transpose: y(m) = alpha * A(m x n) * x(n) + beta * y(m)
    // For transpose/conj-transpose: y(n) = alpha * A^T/A^H(n x m) * x(m) + beta * y(n)
    J x_size = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
    J y_size = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;

    std::vector<X> hx(x_size);
    std::vector<Y> hy_1(y_size);
    std::vector<Y> hy_2(y_size);
    std::vector<Y> hy_gold(y_size);

    hipsparseInit<X>(hx, 1, x_size);
    hipsparseInit<Y>(hy_1, 1, y_size);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(X) * x_size), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(Y) * y_size), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(Y) * y_size), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    X* dx      = (X*)dx_managed.get();
    Y* dy_1    = (Y*)dy_1_managed.get();
    Y* dy_2    = (Y*)dy_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(X) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &beta, sizeof(T), hipMemcpyHostToDevice));

    // Create dense vectors
    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, x_size, dx, xType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, y_size, dy_1, yType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, y_size, dy_2, yType));

    // Query SpMV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_bufferSize(
        handle, transA, &alpha, matA, x, &beta, y1, computeType, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    if(use_preprocess)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSpMV_preprocess(
            handle, transA, &alpha, matA, x, &beta, y1, computeType, alg, buffer));
    }
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMV(handle, transA, &alpha, matA, x, &beta, y1, computeType, alg, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    if(use_preprocess)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSpMV_preprocess(
            handle, transA, d_alpha, matA, x, d_beta, y2, computeType, alg, buffer));
    }
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMV(handle, transA, d_alpha, matA, x, d_beta, y2, computeType, alg, buffer));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(Y) * y_size, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(Y) * y_size, hipMemcpyDeviceToHost));

    // Host SpMV
    host_csrmv<I, J, A, X, Y, T>(transA,
                                 m,
                                 n,
                                 nnz,
                                 alpha,
                                 hcsr_row_ptr.data(),
                                 hcsr_col_ind.data(),
                                 hcsr_val.data(),
                                 hx.data(),
                                 beta,
                                 hy_gold.data(),
                                 idx_base);

    unit_check_near(1, y_size, 1, hy_gold.data(), hy_1.data());
    unit_check_near(1, y_size, 1, hy_gold.data(), hy_2.data());

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
}

// Exercises the multi-configuration cache path: query bufferSize once per
// (operation, algorithm) configuration, allocate a single externalBuffer
// sized to the max, then repeatedly call hipsparseSpMV alternating among
// the configurations without ever calling hipsparseSpMV_bufferSize again.
template <typename I, typename J, typename A, typename X, typename Y, typename T>
static void call_spmv_shared_buffer(hipsparseHandle_t&                       handle,
                                    hipsparseSpMatDescr_t&                   matA,
                                    J                                        m,
                                    J                                        n,
                                    I                                        nnz,
                                    std::vector<I>&                          hcsr_row_ptr,
                                    std::vector<J>&                          hcsr_col_ind,
                                    std::vector<A>&                          hcsr_val,
                                    T                                        alpha,
                                    T                                        beta,
                                    hipsparseIndexBase_t                     idx_base,
                                    const std::vector<hipsparseOperation_t>& ops,
                                    const std::vector<hipsparseSpMVAlg_t>&   algs,
                                    int                                      number_of_passes,
                                    bool                                     use_preprocess)
{
    hipDataType xType       = getDataType<X>();
    hipDataType yType       = getDataType<Y>();
    hipDataType computeType = getDataType<T>();

    // For non-transpose: y(m) = alpha * A(m x n) * x(n) + beta * y(m)
    // For transpose/conj-transpose: y(n) = alpha * A^T/A^H(n x m) * x(m) + beta * y(n)
    // Use vectors sized to max(m, n) so the same allocation works for all
    // operations; the descriptors below restrict the actual logical lengths.
    const J max_dim = std::max(m, n);

    std::vector<X> hx(max_dim);
    hipsparseInit<X>(hx, 1, max_dim);

    auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(X) * max_dim), device_free};
    auto dy_managed = hipsparse_unique_ptr{device_malloc(sizeof(Y) * max_dim), device_free};
    X*   dx         = (X*)dx_managed.get();
    Y*   dy         = (Y*)dy_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(X) * max_dim, hipMemcpyHostToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    // Step 1: query the bufferSize for every (operation, algorithm)
    // configuration we will use, and take the max. The externalBuffer we
    // allocate below will be reused for all configurations.
    size_t buffer_size_max = 0;
    for(hipsparseOperation_t op : ops)
    {
        const J x_size = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
        const J y_size = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;

        hipsparseDnVecDescr_t x, y;
        CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, x_size, dx, xType));
        CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, y_size, dy, yType));

        for(hipsparseSpMVAlg_t alg : algs)
        {
            size_t bufferSize;
            CHECK_HIPSPARSE_ERROR(hipsparseSpMV_bufferSize(
                handle, op, &alpha, matA, x, &beta, y, computeType, alg, &bufferSize));
            buffer_size_max = std::max(buffer_size_max, bufferSize);
        }

        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
        CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));
    }

    void* buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size_max));

    // Step 2: repeatedly loop over every (operation, algorithm) configuration
    // and call hipsparseSpMV with the shared buffer, never calling
    // bufferSize again. Verify each call's result against a CPU reference.
    for(int pass = 0; pass < number_of_passes; ++pass)
    {
        for(hipsparseOperation_t op : ops)
        {
            const J x_size = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : m;
            const J y_size = (op == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : n;

            std::vector<Y> hy(y_size);
            std::vector<Y> hy_gold(y_size);
            hipsparseInit<Y>(hy, 1, y_size);
            hy_gold = hy;

            CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));

            hipsparseDnVecDescr_t x, y;
            CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, x_size, dx, xType));
            CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, y_size, dy, yType));

            for(hipsparseSpMVAlg_t alg : algs)
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(dy, hy.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));

                if(use_preprocess)
                {
                    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_preprocess(
                        handle, op, &alpha, matA, x, &beta, y, computeType, alg, buffer));
                }

                CHECK_HIPSPARSE_ERROR(
                    hipsparseSpMV(handle, op, &alpha, matA, x, &beta, y, computeType, alg, buffer));

                std::vector<Y> hy_out(y_size);
                CHECK_HIP_ERROR(
                    hipMemcpy(hy_out.data(), dy, sizeof(Y) * y_size, hipMemcpyDeviceToHost));

                std::vector<Y> hy_ref(hy);
                host_csrmv<I, J, A, X, Y, T>(op,
                                             m,
                                             n,
                                             nnz,
                                             alpha,
                                             hcsr_row_ptr.data(),
                                             hcsr_col_ind.data(),
                                             hcsr_val.data(),
                                             hx.data(),
                                             beta,
                                             hy_ref.data(),
                                             idx_base);

                unit_check_near(1, y_size, 1, hy_ref.data(), hy_out.data());
            }

            CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
            CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));
        }
    }

    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename I, typename J, typename A, typename X, typename Y, typename T>
void testing_spmv_csr_reuse_descr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    J                    m        = argus.M;
    J                    n        = argus.N;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    // Index and data types
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          aType = getDataType<A>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcol_ind;
    std::vector<A> hval;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcol_ind, hval, idx_base));

    // Redefine sparse matrix values
    hipsparseInit<A>(hval, hval.size(), 1);

    // allocate memory on device
    auto dptr_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(A) * nnz), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr    = (I*)dptr_managed.get();
    J* dcol    = (J*)dcol_managed.get();
    A* dval    = (A*)dval_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcol_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(A) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrix
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&matA, m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, aType));

    const std::vector<hipsparseOperation_t> ops = {HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                   HIPSPARSE_OPERATION_TRANSPOSE,
                                                   HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE};
    const std::vector<hipsparseSpMVAlg_t>   algs
        = {HIPSPARSE_SPMV_ALG_DEFAULT, HIPSPARSE_SPMV_CSR_ALG1, HIPSPARSE_SPMV_CSR_ALG2};

    constexpr int number_of_passes = 3;

    // Scenario 1: SpMV-only. Never call bufferSize or preprocess, only
    // hipsparseSpMV with a null externalBuffer.
    call_spmv_only<I, J, A, X, Y, T>(handle,
                                     matA,
                                     m,
                                     n,
                                     nnz,
                                     hcsr_row_ptr,
                                     hcol_ind,
                                     hval,
                                     h_alpha,
                                     h_beta,
                                     idx_base,
                                     ops,
                                     algs,
                                     number_of_passes);

    // Scenario 2: per-call bufferSize / buffer allocation. Exercises that
    // the per-(op,alg,datatype) cache entries survive being re-queried via
    // bufferSize on the same sparse matrix descriptor.
    for(bool use_preprocess : {false, true})
    {
        for(int pass = 0; pass < number_of_passes; ++pass)
        {
            for(hipsparseOperation_t op : ops)
            {
                for(hipsparseSpMVAlg_t alg : algs)
                {
                    call_spmv<I, J, A, X, Y, T>(handle,
                                                matA,
                                                m,
                                                n,
                                                nnz,
                                                hcsr_row_ptr,
                                                hcol_ind,
                                                hval,
                                                h_alpha,
                                                h_beta,
                                                idx_base,
                                                op,
                                                alg,
                                                use_preprocess);
                }
            }
        }
    }

    // Scenario 3: bufferSize is queried once per configuration up front, a
    // single externalBuffer is allocated to the max of those sizes, and
    // hipsparseSpMV is then called repeatedly across configurations with
    // that one shared buffer (no further bufferSize calls).
    for(bool use_preprocess : {false, true})
    {
        call_spmv_shared_buffer<I, J, A, X, Y, T>(handle,
                                                  matA,
                                                  m,
                                                  n,
                                                  nnz,
                                                  hcsr_row_ptr,
                                                  hcol_ind,
                                                  hval,
                                                  h_alpha,
                                                  h_beta,
                                                  idx_base,
                                                  ops,
                                                  algs,
                                                  number_of_passes,
                                                  use_preprocess);
    }

    // Destroy matrix
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
#endif
}

#endif // TESTING_SPMV_CSR_REUSE_DESCR_HPP
