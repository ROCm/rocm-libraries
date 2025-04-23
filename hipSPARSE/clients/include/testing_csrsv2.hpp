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
#ifndef TESTING_CSRSV2_HPP
#define TESTING_CSRSV2_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <cmath>
#include <hipsparse.h>
#include <limits>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrsv2_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                    m         = 100;
    int                    nnz       = 100;
    int                    safe_size = 100;
    T                      h_alpha   = 0.6;
    hipsparseOperation_t   transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csrsv2_struct> unique_ptr_csrsv2_info(new csrsv2_struct);
    csrsv2Info_t                   info = unique_ptr_csrsv2_info->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dptr    = (int*)dptr_managed.get();
    int*  dcol    = (int*)dcol_managed.get();
    T*    dval    = (T*)dval_managed.get();
    T*    dx      = (T*)dx_managed.get();
    T*    dy      = (T*)dy_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    int size;
    int position;

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, (int*)nullptr, dcol, info, &size),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr, (int*)nullptr, info, &size),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_bufferSize(
            handle, transA, m, nnz, descr, (T*)nullptr, dptr, dcol, info, &size),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr, dcol, info, (int*)nullptr),
        "Error: size is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_bufferSize(
            handle, transA, m, nnz, (hipsparseMatDescr_t) nullptr, dval, dptr, dcol, info, &size),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr, dcol, (csrsv2Info_t) nullptr, &size),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrsv2_bufferSize(
        (hipsparseHandle_t) nullptr, transA, m, nnz, descr, dval, dptr, dcol, info, &size));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, (int*)nullptr, dcol, info, policy, dbuffer),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, dptr, (int*)nullptr, info, policy, dbuffer),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_analysis(
            handle, transA, m, nnz, descr, (T*)nullptr, dptr, dcol, info, policy, dbuffer),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, dptr, dcol, info, policy, (void*)nullptr),
        "Error: dbuffer is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_analysis(handle,
                                                                      transA,
                                                                      m,
                                                                      nnz,
                                                                      (hipsparseMatDescr_t) nullptr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      info,
                                                                      policy,
                                                                      dbuffer),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_analysis(handle,
                                                                      transA,
                                                                      m,
                                                                      nnz,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      (csrsv2Info_t) nullptr,
                                                                      policy,
                                                                      dbuffer),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrsv2_analysis((hipsparseHandle_t) nullptr,
                                                                     transA,
                                                                     m,
                                                                     nnz,
                                                                     descr,
                                                                     dval,
                                                                     dptr,
                                                                     dcol,
                                                                     info,
                                                                     policy,
                                                                     dbuffer));

    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   dval,
                                                                   (int*)nullptr,
                                                                   dcol,
                                                                   info,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   (int*)nullptr,
                                                                   info,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   (T*)nullptr,
                                                                   dptr,
                                                                   dcol,
                                                                   info,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   info,
                                                                   (T*)nullptr,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: dx is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   info,
                                                                   dx,
                                                                   (T*)nullptr,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: dy is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   (T*)nullptr,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   info,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   info,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   (void*)nullptr),
                                            "Error: dbuffer is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   (hipsparseMatDescr_t) nullptr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   info,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_solve(handle,
                                                                   transA,
                                                                   m,
                                                                   nnz,
                                                                   &h_alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   (csrsv2Info_t) nullptr,
                                                                   dx,
                                                                   dy,
                                                                   policy,
                                                                   dbuffer),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrsv2_solve((hipsparseHandle_t) nullptr,
                                                                  transA,
                                                                  m,
                                                                  nnz,
                                                                  &h_alpha,
                                                                  descr,
                                                                  dval,
                                                                  dptr,
                                                                  dcol,
                                                                  info,
                                                                  dx,
                                                                  dy,
                                                                  policy,
                                                                  dbuffer));

    verify_hipsparse_status_invalid_pointer(hipsparseXcsrsv2_zeroPivot(handle, info, (int*)nullptr),
                                            "Error: position is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrsv2_zeroPivot(handle, (csrsv2Info_t) nullptr, &position),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXcsrsv2_zeroPivot((hipsparseHandle_t) nullptr, info, &position));
#endif
}

template <typename T>
hipsparseStatus_t testing_csrsv2(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                    m         = argus.M;
    hipsparseIndexBase_t   idx_base  = argus.baseA;
    hipsparseOperation_t   trans     = argus.transA;
    hipsparseDiagType_t    diag_type = argus.diag_type;
    hipsparseFillMode_t    fill_mode = argus.fill_mode;
    hipsparseSolvePolicy_t policy    = argus.solve_policy;
    T                      h_alpha   = make_DataType<T>(argus.alpha);
    std::string            filename  = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csrsv2_struct> unique_ptr_csrsv2_info(new csrsv2_struct);
    csrsv2Info_t                   info = unique_ptr_csrsv2_info->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    // Set matrix diag type
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatDiagType(descr, diag_type));

    // Set matrix fill mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatFillMode(descr, fill_mode));

    if(m == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
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

    std::vector<T> hx(m);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, m);

    // Allocate memory on device
    auto dptr_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_position_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dptr       = (int*)dptr_managed.get();
    int* dcol       = (int*)dcol_managed.get();
    T*   dval       = (T*)dval_managed.get();
    T*   dx         = (T*)dx_managed.get();
    T*   dy_1       = (T*)dy_1_managed.get();
    T*   dy_2       = (T*)dy_2_managed.get();
    T*   d_alpha    = (T*)d_alpha_managed.get();
    int* d_position = (int*)d_position_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrsv2 buffer size
    int bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_bufferSize(
        handle, trans, m, nnz, descr, dval, dptr, dcol, info, &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    // csrsv2 analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_analysis(
        handle, trans, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                     trans,
                                                     m,
                                                     nnz,
                                                     &h_alpha,
                                                     descr,
                                                     dval,
                                                     dptr,
                                                     dcol,
                                                     info,
                                                     dx,
                                                     dy_1,
                                                     policy,
                                                     dbuffer));

        int               hposition_1;
        hipsparseStatus_t pivot_status_1;
        pivot_status_1 = hipsparseXcsrsv2_zeroPivot(handle, info, &hposition_1);

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                     trans,
                                                     m,
                                                     nnz,
                                                     d_alpha,
                                                     descr,
                                                     dval,
                                                     dptr,
                                                     dcol,
                                                     info,
                                                     dx,
                                                     dy_2,
                                                     policy,
                                                     dbuffer));

        hipsparseStatus_t pivot_status_2;
        pivot_status_2 = hipsparseXcsrsv2_zeroPivot(handle, info, d_position);

        // Copy output from device to CPU
        int hposition_2;
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&hposition_2, d_position, sizeof(int), hipMemcpyDeviceToHost));

        // Host csrsv2
        hipDeviceProp_t prop;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));

        int position_gold;
        if((fill_mode == HIPSPARSE_FILL_MODE_LOWER && trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
           || (fill_mode == HIPSPARSE_FILL_MODE_UPPER && trans == HIPSPARSE_OPERATION_TRANSPOSE))
        {
            position_gold = csr_lsolve(trans,
                                       m,
                                       hcsr_row_ptr.data(),
                                       hcsr_col_ind.data(),
                                       hcsr_val.data(),
                                       h_alpha,
                                       hx.data(),
                                       hy_gold.data(),
                                       idx_base,
                                       diag_type,
                                       prop.warpSize);
        }
        else
        {
            position_gold = csr_usolve(trans,
                                       m,
                                       hcsr_row_ptr.data(),
                                       hcsr_col_ind.data(),
                                       hcsr_val.data(),
                                       h_alpha,
                                       hx.data(),
                                       hy_gold.data(),
                                       idx_base,
                                       diag_type,
                                       prop.warpSize);
        }

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
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                         trans,
                                                         m,
                                                         nnz,
                                                         &h_alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         info,
                                                         dx,
                                                         dy_1,
                                                         policy,
                                                         dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                         trans,
                                                         m,
                                                         nnz,
                                                         &h_alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         info,
                                                         dx,
                                                         dy_1,
                                                         policy,
                                                         dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = csrsv_gflop_count(m, nnz, diag_type);
        double gbyte_count = csrsv_gbyte_count<T>(m, nnz);

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::trans,
                            hipsparse_operation2string(trans),
                            display_key_t::diag_type,
                            hipsparse_diagtype2string(diag_type),
                            display_key_t::fill_mode,
                            hipsparse_fillmode2string(fill_mode),
                            display_key_t::solve_policy,
                            hipsparse_solvepolicy2string(policy),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRSV2_HPP
