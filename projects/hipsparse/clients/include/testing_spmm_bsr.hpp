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
#ifndef TESTING_SPMM_BSR_HPP
#define TESTING_SPMM_BSR_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_graph.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

#include <algorithm>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename I, typename J, typename T>
void testing_spmm_bsr_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION))
    int64_t              mb         = 10;
    int64_t              kb         = 10;
    int64_t              n          = 10;
    int64_t              nnzb       = 10;
    int64_t              blockDim   = 2;
    int64_t              safe_size  = 100;
    T                    alpha      = make_DataType<T>(0.6);
    T                    beta       = make_DataType<T>(0.2);
    hipsparseOperation_t transA     = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB     = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order      = HIPSPARSE_ORDER_COL;
    hipsparseOrder_t     blockOrder = HIPSPARSE_ORDER_ROW;
    hipsparseIndexBase_t idxBase    = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType    = HIPSPARSE_INDEX_32I;
    hipDataType          dataType   = getDataType<T>();
    hipsparseSpMMAlg_t   alg        = HIPSPARSE_SPMM_BSR_ALG1;

    hipsparseLocalHandle_t handle;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(J) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    I*    dptr = (I*)dptr_managed.get();
    J*    dcol = (J*)dcol_managed.get();
    T*    dval = (T*)dval_managed.get();
    T*    dB   = (T*)dB_managed.get();
    T*    dC   = (T*)dC_managed.get();
    void* dbuf = (void*)dbuf_managed.get();

    // SpMM structures
    hipsparseSpMatDescr_t matA;
    hipsparseDnMatDescr_t matB, matC;

    size_t bsize;

    // Create BSR sparse matrix descriptor (mb x kb blocks for op(A) = A).
    verify_hipsparse_status_success(hipsparseCreateBsr(&matA,
                                                       mb,
                                                       kb,
                                                       nnzb,
                                                       blockDim,
                                                       blockDim,
                                                       dptr,
                                                       dcol,
                                                       dval,
                                                       idxType,
                                                       idxType,
                                                       idxBase,
                                                       dataType,
                                                       blockOrder),
                                    "success");
    verify_hipsparse_status_success(
        hipsparseCreateDnMat(&matB, kb * blockDim, n, kb * blockDim, dB, dataType, order),
        "success");
    verify_hipsparse_status_success(
        hipsparseCreateDnMat(&matC, mb * blockDim, n, mb * blockDim, dC, dataType, order),
        "success");

    // SpMM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_bufferSize(
        nullptr, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, nullptr, matA, matB, &beta, matC, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, nullptr, matB, &beta, matC, dataType, alg, &bsize),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, nullptr, &beta, matC, dataType, alg, &bsize),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, nullptr, matC, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, &bsize),
        "Error: matC is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, nullptr),
        "Error: bsize is nullptr");

    // SpMM_preprocess
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_preprocess(
        nullptr, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, nullptr, matA, matB, &beta, matC, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, nullptr, matB, &beta, matC, dataType, alg, dbuf),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, nullptr, &beta, matC, dataType, alg, dbuf),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, matB, nullptr, matC, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, dbuf),
        "Error: matC is nullptr");

    // SpMM
    verify_hipsparse_status_invalid_handle(hipsparseSpMM(
        nullptr, transA, transB, &alpha, matA, matB, &beta, matC, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, nullptr, matA, matB, &beta, matC, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, nullptr, matB, &beta, matC, dataType, alg, dbuf),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, nullptr, &beta, matC, dataType, alg, dbuf),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, matB, nullptr, matC, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(
            handle, transA, transB, &alpha, matA, matB, &beta, nullptr, dataType, alg, dbuf),
        "Error: matC is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matB), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matC), "success");
#endif
}

template <typename I, typename J, typename T>
void testing_spmm_bsr(Arguments argus)
{
#if(!defined(CUDART_VERSION))
    J                    m         = argus.M;
    J                    n         = argus.N;
    J                    k         = argus.K;
    J                    blockDim  = argus.block_dim;
    T                    h_alpha   = argus.get_alpha<T>();
    T                    h_beta    = argus.get_beta<T>();
    hipsparseOperation_t transA    = argus.transA;
    hipsparseOperation_t transB    = argus.transB;
    hipsparseOrder_t     orderB    = argus.orderB;
    hipsparseOrder_t     orderC    = argus.orderC;
    hipsparseIndexBase_t idx_base  = argus.baseA;
    hipsparseDirection_t block_dir = argus.dirA;
    hipsparseSpMMAlg_t   alg       = argus.spmm_alg;
    std::string          filename  = argus.filename;

    hipsparseOrder_t blockOrder
        = (block_dir == HIPSPARSE_DIRECTION_ROW) ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    hipsparseLocalHandle_t handle(argus);

    // Host CSR structures (used as the source for the BSR conversion).
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename,
                            (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k,
                            (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m,
                            nnz,
                            hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
                            idx_base));

    // Redefine sparse matrix values
    hipsparseInit<T>(hcsr_val, hcsr_val.size(), 1);

    J mb = (m + blockDim - 1) / blockDim;
    J kb = (k + blockDim - 1) / blockDim;

    I              nnzb_A = 0;
    std::vector<I> hbsr_row_ptr;
    std::vector<J> hbsr_col_ind;
    std::vector<T> hbsr_val;

    host_csr_to_bsr<I, J, T>(block_dir,
                             (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k,
                             (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m,
                             blockDim,
                             nnzb_A,
                             idx_base,
                             hcsr_row_ptr,
                             hcsr_col_ind,
                             hcsr_val,
                             idx_base,
                             hbsr_row_ptr,
                             hbsr_col_ind,
                             hbsr_val);
    m = mb * blockDim;
    k = kb * blockDim;

    // Some matrix properties
    J A_mb = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? mb : kb;
    J A_nb = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? kb : mb;
    J B_m  = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
    J B_n  = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;
    J C_m  = m;
    J C_n  = n;

    int64_t ldb = (orderB == HIPSPARSE_ORDER_COL)
                      ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n)
                      : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k);
    int64_t ldc = (orderC == HIPSPARSE_ORDER_COL) ? m : n;

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : B_m;
    int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? B_n : ldb;
    int64_t nrowC = (orderC == HIPSPARSE_ORDER_COL) ? ldc : C_m;
    int64_t ncolC = (orderC == HIPSPARSE_ORDER_COL) ? C_n : ldc;

    int64_t nnz_A = int64_t(nnzb_A) * blockDim * blockDim;
    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    // Allocate host memory for dense matrices
    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, nnz_B, 1);
    hipsparseInit<T>(hC_1, nnz_C, 1);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (A_mb + 1)), device_free};
    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnzb_A), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dB_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dbsr_row_ptr = (I*)dbsr_row_ptr_managed.get();
    J* dbsr_col_ind = (J*)dbsr_col_ind_managed.get();
    T* dbsr_val     = (T*)dbsr_val_managed.get();
    T* dB           = (T*)dB_managed.get();
    T* dC_1         = (T*)dC_1_managed.get();
    T* dC_2         = (T*)dC_2_managed.get();
    T* d_alpha      = (T*)d_alpha_managed.get();
    T* d_beta       = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(I) * (A_mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(J) * nnzb_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create BSR matrix
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateBsr(&matA,
                                             A_mb,
                                             A_nb,
                                             nnzb_A,
                                             blockDim,
                                             blockDim,
                                             dbsr_row_ptr,
                                             dbsr_col_ind,
                                             dbsr_val,
                                             typeI,
                                             typeJ,
                                             idx_base,
                                             typeT,
                                             blockOrder));

    // Create dense matrices
    hipsparseDnMatDescr_t matB, matC1, matC2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matB, B_m, B_n, ldb, dB, typeT, orderB));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matC1, C_m, C_n, ldc, dC_1, typeT, orderC));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matC2, C_m, C_n, ldc, dC_2, typeT, orderC));

    // Query SpMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM_bufferSize(
        handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, &bufferSize));

    if(bufferSize == 0)
    {
        bufferSize = 4;
    }

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // Preprocess (host pointer mode)
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM_preprocess(
        handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, buffer));

    // Preprocess (device pointer mode)
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM_preprocess(
        handle, transA, transB, d_alpha, matA, matB, d_beta, matC2, typeT, alg, buffer));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM(
            handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, buffer));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM(
            handle, transA, transB, d_alpha, matA, matB, d_beta, matC2, typeT, alg, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // Host SpMM reference using the int-projection of the BSR matrix.
        host_bsrmm<I, J, T>(A_mb,
                            n,
                            A_nb,
                            blockDim,
                            block_dir,
                            transA,
                            transB,
                            h_alpha,
                            hbsr_row_ptr,
                            hbsr_col_ind,
                            hbsr_val,
                            hB,
                            ldb,
                            h_beta,
                            hC_gold,
                            ldc,
                            idx_base);

        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM(
                handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(testing::hipsparseSpMM(
                handle, transA, transB, &h_alpha, matA, matB, &h_beta, matC1, typeT, alg, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = bsrmm_gflop_count(static_cast<int>(n),
                                               static_cast<int>(nnzb_A),
                                               static_cast<int>(blockDim),
                                               static_cast<int>(C_m * C_n),
                                               h_beta != make_DataType<T>(0));
        double gbyte_count = bsrmm_gbyte_count<T>(static_cast<int>(A_mb),
                                                  static_cast<int>(nnzb_A),
                                                  static_cast<int>(blockDim),
                                                  static_cast<int>(B_m * B_n),
                                                  static_cast<int>(C_m * C_n),
                                                  h_beta != make_DataType<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::K,
                            k,
                            display_key_t::block_dim,
                            blockDim,
                            display_key_t::direction,
                            hipsparse_direction2string(block_dir),
                            display_key_t::nnzb,
                            nnzb_A,
                            display_key_t::transA,
                            transA,
                            display_key_t::transB,
                            transB,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::algorithm,
                            hipsparse_spmmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matB));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matC1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matC2));
#endif
}

#endif // TESTING_SPMM_BSR_HPP
