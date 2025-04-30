/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPSV_CSR_HPP
#define TESTING_SPSV_CSR_HPP

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

void testing_spsv_csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t              m         = 100;
    int64_t              n         = 100;
    int64_t              nnz       = 100;
    int64_t              safe_size = 100;
    float                alpha     = 0.6;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpSVAlg_t   alg       = HIPSPARSE_SPSV_ALG_DEFAULT;

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

    // SpSV structures
    hipsparseSpMatDescr_t A;
    hipsparseDnVecDescr_t x, y;

    hipsparseSpSVDescr_t descr;

    verify_hipsparse_status_success(hipsparseSpSV_createDescr(&descr), "success");

    size_t bsize;

    // Create SpSV structures
    verify_hipsparse_status_success(
        hipsparseCreateCsr(&A, m, n, nnz, dptr, dcol, dval, idxType, idxType, idxBase, dataType),
        "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&x, m, dx, dataType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, m, dy, dataType), "success");

    // SpSV buffer
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSV_bufferSize(nullptr, transA, &alpha, A, x, y, dataType, alg, descr, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(handle, transA, nullptr, A, x, y, dataType, alg, descr, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(
            handle, transA, &alpha, nullptr, x, y, dataType, alg, descr, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(
            handle, transA, &alpha, A, nullptr, y, dataType, alg, descr, &bsize),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(
            handle, transA, &alpha, A, x, nullptr, dataType, alg, descr, &bsize),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(handle, transA, &alpha, A, x, y, dataType, alg, descr, nullptr),
        "Error: bsize is nullptr");

    // SpSV analysis
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSV_analysis(nullptr, transA, &alpha, A, x, y, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, nullptr, A, x, y, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, nullptr, x, y, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, A, nullptr, y, dataType, alg, descr, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, A, x, nullptr, dataType, alg, descr, dbuf),
        "Error: y is nullptr");
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, A, x, y, dataType, alg, descr, nullptr),
        "Error: dbuf is nullptr");
#endif

    // SpSV solve
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSV_solve(nullptr, transA, &alpha, A, x, y, dataType, alg, descr));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, nullptr, A, x, y, dataType, alg, descr),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, nullptr, x, y, dataType, alg, descr),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, A, nullptr, y, dataType, alg, descr),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, A, x, nullptr, dataType, alg, descr),
        "Error: y is nullptr");
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, A, x, y, dataType, alg, nullptr),
        "Error: descr is nullptr");
#endif

    // Destruct
    verify_hipsparse_status_success(hipsparseSpSV_destroyDescr(descr), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "success");
#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_spsv_csr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
    J                    m        = argus.M;
    J                    n        = argus.N;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseIndexBase_t idx_base = argus.baseA;
    hipsparseDiagType_t  diag     = argus.diag_type;
    hipsparseFillMode_t  uplo     = argus.fill_mode;
    hipsparseSpSVAlg_t   alg      = static_cast<hipsparseSpSVAlg_t>(argus.spsv_alg);
    std::string          filename = argus.filename;

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
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<T> hx(m);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, m);
    hipsparseInit<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dptr_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr    = (I*)dptr_managed.get();
    J* dcol    = (J*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dx      = (T*)dx_managed.get();
    T* dy_1    = (T*)dy_1_managed.get();
    T* dy_2    = (T*)dy_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipsparseSpSVDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&A, m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT));

    // Create dense vectors
    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, m, dx, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, m, dy_1, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, m, dy_2, typeT));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    // Query SpSV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_bufferSize(
        handle, transA, &h_alpha, A, x, y1, typeT, alg, descr, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_analysis(handle, transA, &h_alpha, A, x, y1, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_analysis(handle, transA, d_alpha, A, x, y2, typeT, alg, descr, buffer));

    if(argus.unit_check)
    {
        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpSV_solve(handle, transA, &h_alpha, A, x, y1, typeT, alg, descr));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseSpSV_solve(handle, transA, d_alpha, A, x, y2, typeT, alg, descr));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        J struct_pivot  = -1;
        J numeric_pivot = -1;
        host_csrsv(transA,
                   m,
                   nnz,
                   h_alpha,
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

        if(struct_pivot == -1 && numeric_pivot == -1)
        {
            unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
            unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
        }
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
                hipsparseSpSV_solve(handle, transA, &h_alpha, A, x, y1, typeT, alg, descr));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseSpSV_solve(handle, transA, &h_alpha, A, x, y1, typeT, alg, descr));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spsv_gflop_count(m, nnz, diag);
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = csrsv_gbyte_count<T>(m, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::algorithm,
                            hipsparse_spsvalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPSV_CSR_HPP
