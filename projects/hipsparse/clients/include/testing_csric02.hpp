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
#ifndef TESTING_CSRIC0_HPP
#define TESTING_CSRIC0_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <cmath>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csric02_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                    m         = 100;
    int                    nnz       = 100;
    int                    safe_size = 100;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csric02_struct> unique_ptr_csric02(new csric02_struct);
    csric02Info_t                   info = unique_ptr_csric02->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dptr    = (int*)dptr_managed.get();
    int*  dcol    = (int*)dcol_managed.get();
    T*    dval    = (T*)dval_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    int size;
    int position;

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_bufferSize(handle, m, nnz, descr, dval, (int*)nullptr, dcol, info, &size),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_bufferSize(handle, m, nnz, descr, dval, dptr, (int*)nullptr, info, &size),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_bufferSize(handle, m, nnz, descr, (T*)nullptr, dptr, dcol, info, &size),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_bufferSize(handle, m, nnz, descr, dval, dptr, dcol, info, (int*)nullptr),
        "Error: size is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_bufferSize(
            handle, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, &size),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_bufferSize(
            handle, m, nnz, descr, dval, dptr, dcol, (csric02Info_t) nullptr, &size),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsric02_bufferSize(
        (hipsparseHandle_t) nullptr, m, nnz, descr, dval, dptr, dcol, info, &size));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_analysis(
            handle, m, nnz, descr, dval, (int*)nullptr, dcol, info, policy, dbuffer),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_analysis(
            handle, m, nnz, descr, dval, dptr, (int*)nullptr, info, policy, dbuffer),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_analysis(
            handle, m, nnz, descr, (T*)nullptr, dptr, dcol, info, policy, dbuffer),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info, policy, (void*)nullptr),
        "Error: dbuffer is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_analysis(
            handle, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, policy, dbuffer),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, (csric02Info_t) nullptr, policy, dbuffer),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsric02_analysis(
        (hipsparseHandle_t) nullptr, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02(handle, m, nnz, descr, dval, (int*)nullptr, dcol, info, policy, dbuffer),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02(handle, m, nnz, descr, dval, dptr, (int*)nullptr, info, policy, dbuffer),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02(handle, m, nnz, descr, (T*)nullptr, dptr, dcol, info, policy, dbuffer),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02(handle, m, nnz, descr, dval, dptr, dcol, info, policy, (void*)nullptr),
        "Error: dbuffer is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02(
            handle, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, policy, dbuffer),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02(
            handle, m, nnz, descr, dval, dptr, dcol, (csric02Info_t) nullptr, policy, dbuffer),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsric02(
        (hipsparseHandle_t) nullptr, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_zeroPivot(handle, info, (int*)nullptr), "Error: position is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsric02_zeroPivot(handle, (csric02Info_t) nullptr, &position),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXcsric02_zeroPivot((hipsparseHandle_t) nullptr, info, &position));
#endif
}

template <typename T>
hipsparseStatus_t testing_csric02(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    int                    m        = argus.M;
    hipsparseIndexBase_t   idx_base = argus.baseA;
    hipsparseSolvePolicy_t policy   = argus.solve_policy;
    std::string            filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csric02_struct> unique_ptr_csric02(new csric02_struct);
    csric02Info_t                   info = unique_ptr_csric02->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    if(m == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse only accepts m > 1
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, m, m, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<T> hcsr_val_orig(hcsr_val);

    // Allocate memory on device
    auto dptr_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dval_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto d_analysis_pivot_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    auto d_solve_pivot_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dptr               = (int*)dptr_managed.get();
    int* dcol               = (int*)dcol_managed.get();
    T*   dval_1             = (T*)dval_1_managed.get();
    T*   dval_2             = (T*)dval_2_managed.get();
    int* d_analysis_pivot_2 = (int*)d_analysis_pivot_2_managed.get();
    int* d_solve_pivot_2    = (int*)d_solve_pivot_2_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval_1, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval_2, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain csric02 buffer size
    int bufferSize;
    CHECK_HIPSPARSE_ERROR(
        hipsparseXcsric02_bufferSize(handle, m, nnz, descr, dval_1, dptr, dcol, info, &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};
    void* dbuffer = (void*)dbuffer_managed.get();

    int h_analysis_pivot_gold;
    int h_analysis_pivot_1;
    int h_analysis_pivot_2;
    int h_solve_pivot_gold;
    int h_solve_pivot_1;
    int h_solve_pivot_2;

    hipsparseStatus_t status_analysis_1;
    hipsparseStatus_t status_analysis_2;
    hipsparseStatus_t status_solve_1;
    hipsparseStatus_t status_solve_2;

    if(argus.unit_check)
    {
        // csric02 analysis - host mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsric02_analysis(
            handle, m, nnz, descr, dval_1, dptr, dcol, info, policy, dbuffer));

        // Get pivot
        status_analysis_1 = hipsparseXcsric02_zeroPivot(handle, info, &h_analysis_pivot_1);
        if(h_analysis_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_analysis_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // csric02 analysis - device mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsric02_analysis(
            handle, m, nnz, descr, dval_2, dptr, dcol, info, policy, dbuffer));

        // Get pivot
        status_analysis_2 = hipsparseXcsric02_zeroPivot(handle, info, d_analysis_pivot_2);
        if(h_analysis_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_analysis_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // csric02 solve - host mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsric02(handle, m, nnz, descr, dval_1, dptr, dcol, info, policy, dbuffer));

        // Get pivot
        status_solve_1 = hipsparseXcsric02_zeroPivot(handle, info, &h_solve_pivot_1);
        if(h_solve_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_solve_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // csric02 solve - device mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsric02(handle, m, nnz, descr, dval_2, dptr, dcol, info, policy, dbuffer));

        // Get pivot
        status_solve_2 = hipsparseXcsric02_zeroPivot(handle, info, d_solve_pivot_2);
        if(h_solve_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_solve_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // Copy output from device to CPU
        std::vector<T> result_1(nnz);
        std::vector<T> result_2(nnz);

        CHECK_HIP_ERROR(hipMemcpy(result_1.data(), dval_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(result_2.data(), dval_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(&h_analysis_pivot_2, d_analysis_pivot_2, sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(&h_solve_pivot_2, d_solve_pivot_2, sizeof(int), hipMemcpyDeviceToHost));

        // Host csric02
        csric0(m,
               hcsr_row_ptr.data(),
               hcsr_col_ind.data(),
               hcsr_val.data(),
               idx_base,
               h_analysis_pivot_gold,
               h_solve_pivot_gold);

#ifndef __HIP_PLATFORM_NVIDIA__
        // Do not check pivots in cusparse
        unit_check_general(1, 1, 1, &h_analysis_pivot_gold, &h_analysis_pivot_1);
        unit_check_general(1, 1, 1, &h_analysis_pivot_gold, &h_analysis_pivot_2);
        unit_check_general(1, 1, 1, &h_solve_pivot_gold, &h_solve_pivot_1);
        unit_check_general(1, 1, 1, &h_solve_pivot_gold, &h_solve_pivot_2);
#endif

        if(h_analysis_pivot_gold == -1 && h_solve_pivot_gold == -1)
        {
            unit_check_near(1, nnz, 1, hcsr_val.data(), result_1.data());
            unit_check_near(1, nnz, 1, hcsr_val.data(), result_2.data());
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
            CHECK_HIP_ERROR(
                hipMemcpy(dval_1, hcsr_val_orig.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

            CHECK_HIPSPARSE_ERROR(hipsparseXcsric02(
                handle, m, nnz, descr, dval_1, dptr, dcol, info, policy, dbuffer));
        }

        double gpu_time_used = 0;

        // Solve run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(dval_1, hcsr_val_orig.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

            double temp = get_time_us();
            CHECK_HIPSPARSE_ERROR(hipsparseXcsric02(
                handle, m, nnz, descr, dval_1, dptr, dcol, info, policy, dbuffer));
            gpu_time_used += (get_time_us() - temp);
        }

        gpu_time_used = gpu_time_used / number_hot_calls;

        double gbyte_count = csric0_gbyte_count<T>(m, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::solve_policy,
                            hipsparse_solvepolicy2string(policy),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRIC0_HPP
