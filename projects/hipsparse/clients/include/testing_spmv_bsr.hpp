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
#ifndef TESTING_SPMV_BSR_HPP
#define TESTING_SPMV_BSR_HPP

#include <hipsparse/hipsparse-version.h>

#ifdef HIPSPARSE_WITH_SPMV_BSR

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
void testing_spmv_bsr_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION))
    int64_t              mb          = 10;
    int64_t              nb          = 10;
    int64_t              nnzb        = 10;
    int64_t              blockDim    = 2;
    int64_t              safe_size   = 100;
    T                    alpha       = make_DataType<T>(0.6);
    T                    beta        = make_DataType<T>(0.2);
    hipsparseOperation_t transA      = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase     = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseOrder_t     blockOrder  = HIPSPARSE_ORDER_ROW;
    hipsparseIndexType_t idxType     = HIPSPARSE_INDEX_32I;
    hipsparseSpMVAlg_t   alg         = HIPSPARSE_MV_ALG_DEFAULT;
    hipDataType          aType       = getDataType<A>();
    hipDataType          xType       = getDataType<X>();
    hipDataType          yType       = getDataType<Y>();
    hipDataType          computeType = getDataType<T>();

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(J) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(A) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(X) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(Y) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    I*    dptr = (I*)dptr_managed.get();
    J*    dcol = (J*)dcol_managed.get();
    A*    dval = (A*)dval_managed.get();
    X*    dx   = (X*)dx_managed.get();
    Y*    dy   = (Y*)dy_managed.get();
    void* dbuf = (void*)dbuf_managed.get();

    // SpMV structures
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t x, y;

    size_t bsize;

    // Create BSR sparse matrix descriptor
    verify_hipsparse_status_success(hipsparseCreateBsr(&matA,
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       blockDim,
                                                       blockDim,
                                                       dptr,
                                                       dcol,
                                                       dval,
                                                       idxType,
                                                       idxType,
                                                       idxBase,
                                                       aType,
                                                       blockOrder),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&x, nb * blockDim, dx, xType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, mb * blockDim, dy, yType), "success");

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
        "Error: matA is nullptr");
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

    // SpMV preprocess (optional)
    verify_hipsparse_status_invalid_handle(hipsparseSpMV_preprocess(
        nullptr, transA, &alpha, matA, x, &beta, y, computeType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(
            handle, transA, nullptr, matA, x, &beta, y, computeType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(
            handle, transA, &alpha, nullptr, x, &beta, y, computeType, alg, dbuf),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(
            handle, transA, &alpha, matA, nullptr, &beta, y, computeType, alg, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(
            handle, transA, &alpha, matA, x, nullptr, y, computeType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(
            handle, transA, &alpha, matA, x, &beta, nullptr, computeType, alg, dbuf),
        "Error: y is nullptr");

    // SpMV
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMV(nullptr, transA, &alpha, matA, x, &beta, y, computeType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, nullptr, matA, x, &beta, y, computeType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, nullptr, x, &beta, y, computeType, alg, dbuf),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, nullptr, &beta, y, computeType, alg, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, x, nullptr, y, computeType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, matA, x, &beta, nullptr, computeType, alg, dbuf),
        "Error: y is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "success");
#endif
}

template <typename I, typename J, typename A, typename X, typename Y, typename T>
void testing_spmv_bsr(Arguments argus)
{
#if(!defined(CUDART_VERSION))
    J                    m         = argus.M;
    J                    n         = argus.N;
    J                    blockDim  = argus.block_dim;
    T                    h_alpha   = make_DataType<T>(argus.alpha);
    T                    h_beta    = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA    = argus.transA;
    hipsparseIndexBase_t idx_base  = argus.baseA;
    hipsparseDirection_t block_dir = argus.dirA;
    hipsparseSpMVAlg_t   alg       = argus.spmv_alg;
    std::string          filename  = argus.filename;

    hipsparseOrder_t blockOrder
        = (block_dir == HIPSPARSE_DIRECTION_ROW) ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;

    // BSR SpMV on the rocSPARSE backend only currently supports non-transpose.
    if(transA != HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        return;
    }

    if(blockDim < 1)
    {
        return;
    }

    // Index and data type
    hipsparseIndexType_t typeI       = getIndexType<I>();
    hipsparseIndexType_t typeJ       = getIndexType<J>();
    hipDataType          aType       = getDataType<A>();
    hipDataType          xType       = getDataType<X>();
    hipDataType          yType       = getDataType<Y>();
    hipDataType          computeType = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host CSR structures (used as the source for the BSR conversion).
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

    // Convert the CSR matrix to BSR on the host. The helper expects int-based
    // row/column arrays, so temporarily project the CSR index arrays through
    // int vectors when I/J are not already int.
    J mb = (m + blockDim - 1) / blockDim;
    J nb = (n + blockDim - 1) / blockDim;

    I              nnzb = 0;
    std::vector<I> hbsr_row_ptr;
    std::vector<J> hbsr_col_ind;
    std::vector<A> hbsr_val;

    host_csr_to_bsr<I, J, A>(block_dir,
                             m,
                             n,
                             blockDim,
                             nnzb,
                             idx_base,
                             hcsr_row_ptr,
                             hcol_ind,
                             hval,
                             idx_base,
                             hbsr_row_ptr,
                             hbsr_col_ind,
                             hbsr_val);

    // BSR matrix dimensions in terms of scalar rows/cols.
    J x_size = nb * blockDim;
    J y_size = mb * blockDim;

    std::vector<X> hx(x_size);
    std::vector<Y> hy_1(y_size);
    std::vector<Y> hy_2(y_size);
    std::vector<Y> hy_gold(y_size);

    hipsparseInit<X>(hx, 1, x_size);
    hipsparseInit<Y>(hy_1, 1, y_size);

    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (mb + 1)), device_free};
    auto dbsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnzb), device_free};
    auto dbsr_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(A) * nnzb * blockDim * blockDim), device_free};
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(X) * x_size), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(Y) * y_size), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(Y) * y_size), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dbsr_row_ptr = (I*)dbsr_row_ptr_managed.get();
    J* dbsr_col_ind = (J*)dbsr_col_ind_managed.get();
    A* dbsr_val     = (A*)dbsr_val_managed.get();
    X* dx           = (X*)dx_managed.get();
    Y* dy_1         = (Y*)dy_1_managed.get();
    Y* dy_2         = (Y*)dy_2_managed.get();
    T* d_alpha      = (T*)d_alpha_managed.get();
    T* d_beta       = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(I) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(J) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_val, hbsr_val.data(), sizeof(A) * nnzb * blockDim * blockDim, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(X) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(Y) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create BSR matrix
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateBsr(&matA,
                                             mb,
                                             nb,
                                             nnzb,
                                             blockDim,
                                             blockDim,
                                             dbsr_row_ptr,
                                             dbsr_col_ind,
                                             dbsr_val,
                                             typeI,
                                             typeJ,
                                             idx_base,
                                             aType,
                                             blockOrder));

    // Create dense vectors
    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, x_size, dx, xType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, y_size, dy_1, yType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, y_size, dy_2, yType));

    // Query SpMV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_bufferSize(
        handle, transA, &h_alpha, matA, x, &h_beta, y1, computeType, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // Preprocess (optional)
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_preprocess(
        handle, transA, &h_alpha, matA, x, &h_beta, y1, computeType, alg, buffer));

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
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(Y) * y_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(Y) * y_size, hipMemcpyDeviceToHost));

        // Host SpMV reference using the int-projection of the BSR matrix.
        host_bsrmv<I, J, A, X, Y, T>(block_dir,
                                     transA,
                                     mb,
                                     nb,
                                     nnzb,
                                     h_alpha,
                                     hbsr_row_ptr.data(),
                                     hbsr_col_ind.data(),
                                     hbsr_val.data(),
                                     blockDim,
                                     hx.data(),
                                     h_beta,
                                     hy_gold.data(),
                                     idx_base);

        unit_check_near(1, y_size, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, y_size, 1, hy_gold.data(), hy_2.data());
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

        double gflop_count = spmv_gflop_count(static_cast<J>(mb * blockDim),
                                              static_cast<I>(nnzb * blockDim * blockDim),
                                              h_beta != make_DataType<T>(0.0));
        double gbyte_count = bsrmv_gbyte_count<A, X, Y, I, J>(
            mb, nb, nnzb, blockDim, h_beta != make_DataType<T>(0.0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::block_dim,
                            blockDim,
                            display_key_t::direction,
                            hipsparse_direction2string(block_dir),
                            display_key_t::nnzb,
                            nnzb,
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

#endif // HIPSPARSE_WITH_SPMV_BSR

#endif // TESTING_SPMV_BSR_HPP
