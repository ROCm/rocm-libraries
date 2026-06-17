/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPMV_COO_HPP
#define TESTING_SPMV_COO_HPP

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

template <typename I, typename A, typename X, typename Y, typename T>
void testing_spmv_coo_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    int64_t              m           = 100;
    int64_t              n           = 100;
    int64_t              nnz         = 100;
    int64_t              safe_size   = 100;
    T                    alpha       = make_DataType<T>(0.6);
    T                    beta        = make_DataType<T>(0.2);
    hipsparseOperation_t transA      = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase     = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType     = HIPSPARSE_INDEX_32I;
    hipDataType          aType       = getDataType<A>();
    hipDataType          xType       = getDataType<X>();
    hipDataType          yType       = getDataType<Y>();
    hipDataType          computeType = getDataType<T>();

#if(!defined(CUDART_VERSION))
    hipsparseSpMVAlg_t alg = HIPSPARSE_MV_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
    hipsparseSpMVAlg_t alg = HIPSPARSE_SPMV_COO_ALG1;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
    hipsparseSpMVAlg_t alg = HIPSPARSE_MV_ALG_DEFAULT;
#endif
#endif

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto drow_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(A) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(X) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(Y) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    I*    drow = (I*)drow_managed.get();
    I*    dcol = (I*)dcol_managed.get();
    A*    dval = (A*)dval_managed.get();
    X*    dx   = (X*)dx_managed.get();
    Y*    dy   = (Y*)dy_managed.get();
    void* dbuf = (void*)dbuf_managed.get();

    // SpMV structures
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t x, y;

    size_t bsize;

    // Create SpMV structures
    verify_hipsparse_status_success(
        hipsparseCreateCoo(&matA, m, n, nnz, drow, dcol, dval, idxType, idxBase, aType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&x, n, dx, xType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, m, dy, yType), "success");

    // SpMV buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpMV_bufferSize(
        nullptr, transA, &alpha, matA, x, &beta, y, computeType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, nullptr, matA, x, &beta, y, computeType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, nullptr, x, &beta, y, computeType, alg, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, matA, nullptr, &beta, y, computeType, alg, &bsize),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, matA, x, nullptr, y, computeType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, matA, x, &beta, nullptr, computeType, alg, &bsize),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, matA, x, &beta, y, computeType, alg, nullptr),
        "Error: bsize is nullptr");

    // SpMV
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMV(nullptr, transA, &alpha, matA, x, &beta, y, computeType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, nullptr, matA, x, &beta, y, computeType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, nullptr, x, &beta, y, computeType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, nullptr, &beta, y, computeType, alg, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, x, nullptr, y, computeType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, x, &beta, nullptr, computeType, alg, dbuf),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, x, &beta, nullptr, computeType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "success");
#endif
}

template <typename I, typename A, typename X, typename Y, typename T>
void testing_spmv_coo(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    I                    m        = argus.M;
    I                    n        = argus.N;
    T                    h_alpha  = argus.get_alpha<T>();
    T                    h_beta   = argus.get_beta<T>();
    hipsparseOperation_t transA   = argus.transA;
    hipsparseIndexBase_t idx_base = argus.baseA;
    hipsparseSpMVAlg_t   alg      = argus.spmv_alg;
    std::string          filename = argus.filename;

    // Index and data types
    hipsparseIndexType_t typeI       = getIndexType<I>();
    hipDataType          aType       = getDataType<A>();
    hipDataType          xType       = getDataType<X>();
    hipDataType          yType       = getDataType<Y>();
    hipDataType          computeType = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hrow_ptr;
    std::vector<I> hcol_ind;
    std::vector<A> hval;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename, m, n, nnz, hrow_ptr, hcol_ind, hval, idx_base));

    // Redefine sparse matrix values
    hipsparseInit<A>(hval, hval.size(), 1);

    std::vector<I> hrow_ind(nnz);

    // Convert to COO
    for(I i = 0; i < m; ++i)
    {
        for(I j = hrow_ptr[i]; j < hrow_ptr[i + 1]; ++j)
        {
            hrow_ind[j - idx_base] = i + idx_base;
        }
    }

    std::vector<X> hx(n);
    std::vector<Y> hy_1(m);
    std::vector<Y> hy_2(m);
    std::vector<Y> hy_gold(m);

    hipsparseInit<X>(hx, 1, n);
    hipsparseInit<Y>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto drow_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(A) * nnz), device_free};
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(X) * n), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(Y) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(Y) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* drow    = (I*)drow_managed.get();
    I* dcol    = (I*)dcol_managed.get();
    A* dval    = (A*)dval_managed.get();
    X* dx      = (X*)dx_managed.get();
    Y* dy_1    = (Y*)dy_1_managed.get();
    Y* dy_2    = (Y*)dy_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(drow, hrow_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcol_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(A) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(X) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(Y) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(Y) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCoo(&matA, m, n, nnz, drow, dcol, dval, typeI, idx_base, aType));

    // Create dense vectors
    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, n, dx, xType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, m, dy_1, yType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, m, dy_2, yType));

    // Query SpMV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_bufferSize(
        handle, transA, &h_alpha, matA, x, &h_beta, y1, computeType, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    if(argus.unit_check)
    {
        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseSpMV(
            handle, transA, &h_alpha, matA, x, &h_beta, y1, computeType, alg, buffer));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpMV(handle, transA, d_alpha, matA, x, d_beta, y2, computeType, alg, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(Y) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(Y) * m, hipMemcpyDeviceToHost));

        // Host SpMV
        host_coomv<I, A, X, Y, T>(m,
                                  nnz,
                                  h_alpha,
                                  hrow_ind.data(),
                                  hcol_ind.data(),
                                  hval.data(),
                                  hx.data(),
                                  h_beta,
                                  hy_gold.data(),
                                  idx_base);

        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSpMV(
                handle, transA, &h_alpha, matA, x, &h_beta, y1, computeType, alg, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSpMV(
                handle, transA, &h_alpha, matA, x, &h_beta, y1, computeType, alg, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(m, nnz, h_beta != make_DataType<T>(0.0));
        double gbyte_count
            = coomv_gbyte_count<A, X, Y, I>(m, n, nnz, h_beta != make_DataType<T>(0.0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::transA,
                            transA,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::algorithm,
                            hipsparse_spmvalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
#endif
}

#endif // TESTING_SPMV_COO_HPP
