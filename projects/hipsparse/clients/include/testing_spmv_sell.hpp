/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPMV_SELL_HPP
#define TESTING_SPMV_SELL_HPP

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

template <typename I, typename J, typename T>
void testing_spmv_sell_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 12011)
    int64_t              m                       = 100;
    int64_t              n                       = 100;
    int64_t              nnz                     = 100;
    int64_t              sellValuesSize          = 100;
    int64_t              sliceSize               = 100;
    int64_t              safe_size               = 100;
    float                alpha                   = 0.6;
    float                beta                    = 0.2;
    hipsparseOperation_t transA                  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase                 = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t sellSliceOffsetsIdxType = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t sellColInIdxType        = HIPSPARSE_INDEX_32I;
    hipDataType          dataType                = HIP_R_32F;

#if(!defined(CUDART_VERSION))
    hipsparseSpMVAlg_t alg = HIPSPARSE_MV_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
    hipsparseSpMVAlg_t alg = HIPSPARSE_SPMV_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
    hipsparseSpMVAlg_t alg = HIPSPARSE_MV_ALG_DEFAULT;
#endif
#endif

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   dptr = (int*)dptr_managed.get();
    int*   dcol = (int*)dcol_managed.get();
    float* dval = (float*)dval_managed.get();
    float* dx   = (float*)dx_managed.get();
    float* dy   = (float*)dy_managed.get();
    void*  dbuf = (void*)dbuf_managed.get();

    // SpMV structures
    hipsparseSpMatDescr_t A;
    hipsparseDnVecDescr_t x, y;

    size_t bsize;

    // Create SpMV structures
    verify_hipsparse_status_success(hipsparseCreateSlicedEll(&A,
                                                             m,
                                                             n,
                                                             nnz,
                                                             sellValuesSize,
                                                             sliceSize,
                                                             dptr,
                                                             dcol,
                                                             dval,
                                                             sellSliceOffsetsIdxType,
                                                             sellColInIdxType,
                                                             idxBase,
                                                             dataType),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&x, n, dx, dataType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, m, dy, dataType), "success");

    // SpMV buffer
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMV_bufferSize(nullptr, transA, &alpha, A, x, &beta, y, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(handle, transA, nullptr, A, x, &beta, y, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, nullptr, x, &beta, y, dataType, alg, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, A, nullptr, &beta, y, dataType, alg, &bsize),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(handle, transA, &alpha, A, x, nullptr, y, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(
            handle, transA, &alpha, A, x, &beta, nullptr, dataType, alg, &bsize),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_bufferSize(handle, transA, &alpha, A, x, &beta, y, dataType, alg, nullptr),
        "Error: bsize is nullptr");

    // SpMV preprocess (optional)
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMV_preprocess(nullptr, transA, &alpha, A, x, &beta, y, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(handle, transA, nullptr, A, x, &beta, y, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(handle, transA, &alpha, nullptr, x, &beta, y, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(handle, transA, &alpha, A, nullptr, &beta, y, dataType, alg, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(handle, transA, &alpha, A, x, nullptr, y, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(handle, transA, &alpha, A, x, &beta, nullptr, dataType, alg, dbuf),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV_preprocess(
            handle, transA, &alpha, A, x, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // SpMV
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMV(nullptr, transA, &alpha, A, x, &beta, y, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, nullptr, A, x, &beta, y, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, nullptr, x, &beta, y, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, A, nullptr, &beta, y, dataType, alg, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, A, x, nullptr, y, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, A, x, &beta, nullptr, dataType, alg, dbuf),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMV(handle, transA, &alpha, A, x, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "success");
#endif
}

template <typename I, typename J, typename T>
void testing_spmv_sell(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 12011)
    J                    m          = argus.M;
    J                    n          = argus.N;
    J                    slice_size = argus.slice_size;
    T                    h_alpha    = argus.get_alpha<T>();
    T                    h_beta     = argus.get_beta<T>();
    hipsparseOperation_t transA     = argus.transA;
    hipsparseIndexBase_t idx_base   = argus.baseA;
    hipsparseSpMVAlg_t   alg        = static_cast<hipsparseSpMVAlg_t>(argus.spmv_alg);
    std::string          filename   = argus.filename;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz;
    CHECK_GENERATE_MATRIX_ERROR(
        generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base));

    // Redefine sparse matrix values
    hipsparseInit<T>(hcsr_val, hcsr_val.size(), 1);

    I nslices = (m - 1) / slice_size + 1;

    std::vector<I> hsell_slice_offsets;
    std::vector<J> hsell_col_ind;
    std::vector<T> hsell_val;

    I sell_colval_size;
    host_csr_to_sell(m,
                     slice_size,
                     hcsr_row_ptr,
                     hcsr_col_ind,
                     hcsr_val,
                     hsell_slice_offsets,
                     hsell_col_ind,
                     hsell_val,
                     sell_colval_size,
                     idx_base,
                     idx_base);

    std::vector<T> hx(n);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, n);
    hipsparseInit<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * (nslices + 1)), device_free};
    auto dcol_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * sell_colval_size), device_free};
    auto dval_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * sell_colval_size), device_free};
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * n), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr    = (I*)dptr_managed.get();
    J* dcol    = (J*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dx      = (T*)dx_managed.get();
    T* dy_1    = (T*)dy_1_managed.get();
    T* dy_2    = (T*)dy_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dptr, hsell_slice_offsets.data(), sizeof(I) * (nslices + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol, hsell_col_ind.data(), sizeof(J) * sell_colval_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dval, hsell_val.data(), sizeof(T) * sell_colval_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateSlicedEll(&A,
                                                   m,
                                                   n,
                                                   nnz,
                                                   sell_colval_size,
                                                   slice_size,
                                                   dptr,
                                                   dcol,
                                                   dval,
                                                   typeI,
                                                   typeJ,
                                                   idx_base,
                                                   typeT));

    // Create dense vectors
    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, n, dx, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, m, dy_1, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, m, dy_2, typeT));

    // Query SpMV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_bufferSize(
        handle, transA, &h_alpha, A, x, &h_beta, y1, typeT, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // Preprocess (optional)
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMV_preprocess(handle, transA, &h_alpha, A, x, &h_beta, y1, typeT, alg, buffer));

    if(argus.unit_check)
    {
        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpMV(handle, transA, &h_alpha, A, x, &h_beta, y1, typeT, alg, buffer));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpMV(handle, transA, d_alpha, A, x, d_beta, y2, typeT, alg, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        host_sellmv(transA,
                    m,
                    n,
                    nnz,
                    slice_size,
                    sell_colval_size,
                    h_alpha,
                    hsell_slice_offsets.data(),
                    hsell_col_ind.data(),
                    hsell_val.data(),
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
            CHECK_HIPSPARSE_ERROR(
                hipsparseSpMV(handle, transA, &h_alpha, A, x, &h_beta, y1, typeT, alg, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseSpMV(handle, transA, &h_alpha, A, x, &h_beta, y1, typeT, alg, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(m, nnz, h_beta != make_DataType<T>(0.0));
        double gbyte_count = sellmv_gbyte_count<T>(
            m, n, nnz, slice_size, sell_colval_size, h_beta != make_DataType<T>(0.0));

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
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
#endif
}

#endif // TESTING_SPMV_SELL_HPP
