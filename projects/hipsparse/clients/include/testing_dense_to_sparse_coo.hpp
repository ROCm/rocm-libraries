/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_DENSE_TO_SPARSE_COO_HPP
#define TESTING_DENSE_TO_SPARSE_COO_HPP

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

void testing_dense_to_sparse_coo_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t safe_size = 100;
    int32_t m         = 10;
    int32_t n         = 10;
    int64_t nnz       = 10;
    int64_t ld        = m;

    hipsparseIndexBase_t        idxBase = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDenseToSparseAlg_t alg     = HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    hipsparseOrder_t            order   = HIPSPARSE_ORDER_COL;

    // Index and data type
    hipsparseIndexType_t iType    = HIPSPARSE_INDEX_64I;
    hipDataType          dataType = HIP_R_32F;

    // Create handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto ddense_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcoo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dcoo_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dcoo_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    float*   ddense_val   = (float*)ddense_val_managed.get();
    int32_t* dcoo_row_ind = (int32_t*)dcoo_row_ind_managed.get();
    int32_t* dcoo_col_ind = (int32_t*)dcoo_col_ind_managed.get();
    float*   dcoo_val     = (float*)dcoo_val_managed.get();
    void*    dbuf         = (void*)dbuf_managed.get();

    // Matrix structures
    hipsparseDnVecDescr_t matA;
    hipsparseSpMatDescr_t matB;

    size_t bsize;

    // Create matrix structures
    verify_hipsparse_status_success(
        hipsparseCreateDnMat(&matA, m, n, ld, ddense_val, dataType, order), "success");
    verify_hipsparse_status_success(
        hipsparseCreateCoo(
            &matB, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dcoo_val, iType, idxBase, dataType),
        "success");

    // denseToSparse buffer size
    verify_hipsparse_status_invalid_handle(
        hipsparseDenseToSparse_bufferSize(nullptr, matA, matB, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_bufferSize(handle, nullptr, matB, alg, &bsize),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_bufferSize(handle, matA, nullptr, alg, &bsize),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_bufferSize(handle, matA, matB, alg, nullptr),
        "Error: bsize is nullptr");

    // denseToSparse analysis
    verify_hipsparse_status_invalid_handle(
        hipsparseDenseToSparse_analysis(nullptr, matA, matB, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_analysis(handle, nullptr, matB, alg, &bsize),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_analysis(handle, matA, nullptr, alg, &bsize),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_analysis(handle, matA, matB, alg, nullptr),
        "Error: bsize is nullptr");

    // denseToSparse_convert
    verify_hipsparse_status_invalid_handle(
        hipsparseDenseToSparse_convert(nullptr, matA, matB, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_convert(handle, nullptr, matB, alg, dbuf), "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_convert(handle, matA, nullptr, alg, dbuf), "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDenseToSparse_convert(handle, matA, matB, alg, nullptr), "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(matB), "success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_dense_to_sparse_coo(Arguments argus)
{
#if(!defined(CUDART_VERSION))
    I                           m        = argus.M;
    I                           n        = argus.N;
    hipsparseIndexBase_t        idx_base = argus.baseA;
    hipsparseDenseToSparseAlg_t alg
        = static_cast<hipsparseDenseToSparseAlg_t>(argus.dense2sparse_alg);
    hipsparseOrder_t order = argus.orderA;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    I ld = (order == HIPSPARSE_ORDER_COL) ? m : n;

    // Host structures
    I              nrow = (order == HIPSPARSE_ORDER_COL) ? ld : m;
    I              ncol = (order == HIPSPARSE_ORDER_COL) ? n : ld;
    std::vector<T> hdense_val(nrow * ncol);

    if(order == HIPSPARSE_ORDER_COL)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < ld; ++j)
            {
                hdense_val[i * ld + j] = make_DataType<T>(-1);
            }
        }
    }
    else
    {
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < ld; ++j)
            {
                hdense_val[i * ld + j] = make_DataType<T>(-1);
            }
        }
    }

    srand(0);
    gen_dense_random_sparsity_pattern(m, n, hdense_val.data(), ld, order, 0.2);

    // allocate memory on device
    auto ddense_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nrow * ncol), device_free};

    T* ddense = (T*)ddense_managed.get();

    // Copy host dense matrix to device
    CHECK_HIP_ERROR(
        hipMemcpy(ddense, hdense_val.data(), sizeof(T) * nrow * ncol, hipMemcpyHostToDevice));

    // Create dense matrix
    hipsparseDnMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matA, m, n, ld, ddense, typeT, order));

    // Create matrices
    hipsparseSpMatDescr_t matB;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCoo(&matB, m, n, 0, nullptr, nullptr, nullptr, typeI, idx_base, typeT));

    // Query DenseToSparse buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_bufferSize(handle, matA, matB, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_analysis(handle, matA, matB, alg, buffer));

    int64_t rows, cols, nnz;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(matB, &rows, &cols, &nnz));

    auto drow_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    I* drow = (I*)drow_managed.get();
    I* dcol = (I*)dcol_managed.get();
    T* dval = (T*)dval_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseCooSetPointers(matB, drow, dcol, dval));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_convert(handle, matA, matB, alg, buffer));

        // copy output from device to CPU
        std::vector<I> hcoo_row_ind(nnz);
        std::vector<I> hcoo_col_ind(nnz);
        std::vector<T> hcoo_val(nnz);

        CHECK_HIP_ERROR(
            hipMemcpy(hcoo_row_ind.data(), drow, sizeof(I) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcoo_col_ind.data(), dcol, sizeof(I) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcoo_val.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        std::vector<I> hcoo_row_ind_cpu(nnz);
        std::vector<I> hcoo_col_ind_cpu(nnz);
        std::vector<T> hcoo_val_cpu(nnz);

        if(order == HIPSPARSE_ORDER_COL)
        {
            int index = 0;
            for(I i = 0; i < m; ++i)
            {
                for(I j = 0; j < n; ++j)
                {
                    if(hdense_val[j * ld + i] != make_DataType<T>(0.0))
                    {
                        hcoo_row_ind_cpu[index] = i + idx_base;
                        hcoo_col_ind_cpu[index] = j + idx_base;
                        hcoo_val_cpu[index]     = hdense_val[j * ld + i];

                        index++;
                    }
                }
            }
        }
        else
        {
            int index = 0;
            for(I i = 0; i < m; ++i)
            {
                for(I j = 0; j < n; ++j)
                {
                    if(hdense_val[i * ld + j] != make_DataType<T>(0.0))
                    {
                        hcoo_row_ind_cpu[index] = i + idx_base;
                        hcoo_col_ind_cpu[index] = j + idx_base;
                        hcoo_val_cpu[index]     = hdense_val[i * ld + j];

                        index++;
                    }
                }
            }
        }

        unit_check_general(1, nnz, 1, hcoo_row_ind_cpu.data(), hcoo_row_ind.data());
        unit_check_general(1, nnz, 1, hcoo_col_ind_cpu.data(), hcoo_col_ind.data());
        unit_check_general(1, nnz, 1, hcoo_val_cpu.data(), hcoo_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm-up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_convert(handle, matA, matB, alg, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_convert(handle, matA, matB, alg, buffer));
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = dense2coo_gbyte_count<T>(m, n, (I)nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::order,
                            order,
                            display_key_t::algorithm,
                            hipsparse_densetosparsealg2string(alg),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matB));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_DENSE_TO_SPARSE_COO_HPP
