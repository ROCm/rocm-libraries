/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CSRILU0_HPP
#define TESTING_CSRILU0_HPP

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
void testing_csrilu02_bad_arg(void)
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

    std::unique_ptr<csrilu02_struct> unique_ptr_csrilu02(new csrilu02_struct);
    csrilu02Info_t                   info = unique_ptr_csrilu02->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};
    auto dboost_tol_managed = hipsparse_unique_ptr{device_malloc(sizeof(double)), device_free};
    auto dboost_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int*    dptr       = (int*)dptr_managed.get();
    int*    dcol       = (int*)dcol_managed.get();
    T*      dval       = (T*)dval_managed.get();
    void*   dbuffer    = (void*)dbuffer_managed.get();
    double* dboost_tol = (double*)dboost_tol_managed.get();
    T*      dboost_val = (T*)dboost_val_managed.get();

    int size;
    int position;

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_bufferSize(
            handle, m, nnz, descr, dval, (int*)nullptr, dcol, info, &size),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_bufferSize(
            handle, m, nnz, descr, dval, dptr, (int*)nullptr, info, &size),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_bufferSize(handle, m, nnz, descr, (T*)nullptr, dptr, dcol, info, &size),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_bufferSize(handle, m, nnz, descr, dval, dptr, dcol, info, (int*)nullptr),
        "Error: size is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_bufferSize(
            handle, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, &size),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_bufferSize(
            handle, m, nnz, descr, dval, dptr, dcol, (csrilu02Info_t) nullptr, &size),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrilu02_bufferSize(
        (hipsparseHandle_t) nullptr, m, nnz, descr, dval, dptr, dcol, info, &size));

    verify_hipsparse_status_invalid_handle(hipsparseXcsrilu02_numericBoost(
        (hipsparseHandle_t) nullptr, info, 1, dboost_tol, dboost_val));
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_numericBoost(
            handle, (csrilu02Info_t) nullptr, 1, dboost_tol, dboost_val),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_numericBoost(handle, info, 1, (double*)nullptr, dboost_val),
        "Error: boost_tol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_numericBoost(handle, info, 1, dboost_tol, (T*)nullptr),
        "Error: boost_val is nullptr");

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, (int*)nullptr, dcol, info, policy, dbuffer),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, (int*)nullptr, info, policy, dbuffer),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, (T*)nullptr, dptr, dcol, info, policy, dbuffer),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info, policy, (void*)nullptr),
        "Error: dbuffer is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_analysis(
            handle, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, policy, dbuffer),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, (csrilu02Info_t) nullptr, policy, dbuffer),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrilu02_analysis(
        (hipsparseHandle_t) nullptr, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02(handle, m, nnz, descr, dval, (int*)nullptr, dcol, info, policy, dbuffer),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02(handle, m, nnz, descr, dval, dptr, (int*)nullptr, info, policy, dbuffer),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02(handle, m, nnz, descr, (T*)nullptr, dptr, dcol, info, policy, dbuffer),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02(handle, m, nnz, descr, dval, dptr, dcol, info, policy, (void*)nullptr),
        "Error: dbuffer is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02(
            handle, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, policy, dbuffer),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02(
            handle, m, nnz, descr, dval, dptr, dcol, (csrilu02Info_t) nullptr, policy, dbuffer),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrilu02(
        (hipsparseHandle_t) nullptr, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_zeroPivot(handle, info, (int*)nullptr), "Error: position is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrilu02_zeroPivot(handle, (csrilu02Info_t) nullptr, &position),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXcsrilu02_zeroPivot((hipsparseHandle_t) nullptr, info, &position));
#endif
}

template <typename T>
hipsparseStatus_t testing_csrilu02(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    int                    m         = argus.M;
    int                    boost     = argus.numericboost;
    double                 boost_tol = argus.boosttol;
    T                      boost_val = make_DataType<T>(argus.boostval, argus.boostvali);
    hipsparseIndexBase_t   idx_base  = argus.baseA;
    hipsparseSolvePolicy_t policy    = argus.solve_policy;
    std::string            filename  = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csrilu02_struct> unique_ptr_csrilu02(new csrilu02_struct);
    csrilu02Info_t                   info = unique_ptr_csrilu02->info;

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
    auto dptr_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dval2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto d_position_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    auto boost_tol_managed  = hipsparse_unique_ptr{device_malloc(sizeof(double)), device_free};
    auto boost_val_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int*    dptr       = (int*)dptr_managed.get();
    int*    dcol       = (int*)dcol_managed.get();
    T*      dval1      = (T*)dval1_managed.get();
    T*      dval2      = (T*)dval2_managed.get();
    int*    d_position = (int*)d_position_managed.get();
    double* dboost_tol = (double*)boost_tol_managed.get();
    T*      dboost_val = (T*)boost_val_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval1, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval2, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain csrilu02 buffer size
    int bufferSize;
    CHECK_HIPSPARSE_ERROR(
        hipsparseXcsrilu02_bufferSize(handle, m, nnz, descr, dval1, dptr, dcol, info, &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dboost_tol, &boost_tol, sizeof(double), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dboost_val, &boost_val, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval1, dptr, dcol, info, policy, dbuffer));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsrilu02_numericBoost(handle, info, boost, &boost_tol, &boost_val));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsrilu02(handle, m, nnz, descr, dval1, dptr, dcol, info, policy, dbuffer));

        int               hposition_1;
        hipsparseStatus_t pivot_status_1;
        pivot_status_1 = hipsparseXcsrilu02_zeroPivot(handle, info, &hposition_1);

        // Pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval2, dptr, dcol, info, policy, dbuffer));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsrilu02_numericBoost(handle, info, boost, dboost_tol, dboost_val));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsrilu02(handle, m, nnz, descr, dval2, dptr, dcol, info, policy, dbuffer));

        int               hposition_2;
        hipsparseStatus_t pivot_status_2;
        pivot_status_2 = hipsparseXcsrilu02_zeroPivot(handle, info, d_position);
        CHECK_HIP_ERROR(hipMemcpy(&hposition_2, d_position, sizeof(int), hipMemcpyDeviceToHost));

        // Copy output from device to CPU
        std::vector<T> result1(nnz);
        std::vector<T> result2(nnz);
        CHECK_HIP_ERROR(hipMemcpy(result1.data(), dval1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(result2.data(), dval2, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Host csrilu02
        int position_gold = csrilu0(m,
                                    hcsr_row_ptr.data(),
                                    hcsr_col_ind.data(),
                                    hcsr_val.data(),
                                    idx_base,
                                    boost,
                                    boost_tol,
                                    boost_val);

        unit_check_general(1, 1, 1, &position_gold, &hposition_1);
        unit_check_general(1, 1, 1, &position_gold, &hposition_2);

        if(hposition_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

        if(hposition_2 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

#if defined(__HIP_PLATFORM_AMD__)
        unit_check_general(1, nnz, 1, hcsr_val.data(), result1.data());
        unit_check_general(1, nnz, 1, hcsr_val.data(), result2.data());
#elif defined(__HIP_PLATFORM_NVIDIA__)
        // do weaker check for cusparse
        unit_check_near(1, nnz, 1, hcsr_val.data(), result1.data());
        unit_check_near(1, nnz, 1, hcsr_val.data(), result2.data());
#endif
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
                hipMemcpy(dval1, hcsr_val_orig.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

            CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02(
                handle, m, nnz, descr, dval1, dptr, dcol, info, policy, dbuffer));
        }

        double gpu_time_used = 0;

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(dval1, hcsr_val_orig.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

            double temp = get_time_us();
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02(
                handle, m, nnz, descr, dval1, dptr, dcol, info, policy, dbuffer));
            gpu_time_used += (get_time_us() - temp);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csrilu0_gbyte_count<T>(m, nnz);
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

#endif // TESTING_CSRILU0_HPP
