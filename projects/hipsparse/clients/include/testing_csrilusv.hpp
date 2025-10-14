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
#ifndef TESTING_CSRILUSV_HPP
#define TESTING_CSRILUSV_HPP

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
hipsparseStatus_t testing_csrilusv(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    hipsparseIndexBase_t idx_base = argus.baseA;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_M(new descr_struct);
    hipsparseMatDescr_t           descr_M = unique_ptr_descr_M->descr;

    std::unique_ptr<csrilu02_struct> unique_ptr_csrilu02_info(new csrilu02_struct);
    csrilu02Info_t                   info_M = unique_ptr_csrilu02_info->info;

    // Initialize the matrix descriptor
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_M, idx_base));

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Initial Data on CPU
    int m;
    int n;
    int nnz;

    if(read_bin_matrix(argus.filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
       != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", argus.filename);
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto d_position_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dptr       = (int*)dptr_managed.get();
    int* dcol       = (int*)dcol_managed.get();
    T*   dval       = (T*)dval_managed.get();
    int* d_position = (int*)d_position_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain csrilu02 buffer size
    int size;
    CHECK_HIPSPARSE_ERROR(
        hipsparseXcsrilu02_bufferSize(handle, m, nnz, descr_M, dval, dptr, dcol, info_M, &size));

    // Allocate buffer on the device
    auto dbuffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    // csrilu02 analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02_analysis(handle,
                                                      m,
                                                      nnz,
                                                      descr_M,
                                                      dval,
                                                      dptr,
                                                      dcol,
                                                      info_M,
                                                      HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                      dbuffer));

    // Compute incomplete LU factorization
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02(handle,
                                             m,
                                             nnz,
                                             descr_M,
                                             dval,
                                             dptr,
                                             dcol,
                                             info_M,
                                             HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                             dbuffer));

    // Check for zero pivot
    int               hposition_1, hposition_2;
    hipsparseStatus_t pivot_status_1, pivot_status_2;

    // Host pointer mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    pivot_status_1 = hipsparseXcsrilu02_zeroPivot(handle, info_M, &hposition_1);

    // device pointer mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    pivot_status_2 = hipsparseXcsrilu02_zeroPivot(handle, info_M, d_position);

    // Copy output to CPU
    std::vector<T> iluresult(nnz);
    CHECK_HIP_ERROR(hipMemcpy(iluresult.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&hposition_2, d_position, sizeof(int), hipMemcpyDeviceToHost));

    // Compute host reference csrilu0
    int position_gold = csrilu0(m,
                                hcsr_row_ptr.data(),
                                hcsr_col_ind.data(),
                                hcsr_val.data(),
                                idx_base,
                                false,
                                0.0,
                                make_DataType<T>(0.0, 0.0));

    // Check zero pivot results
    unit_check_general(1, 1, 1, &position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &position_gold, &hposition_2);

    // If zero pivot was found, do not go further
    if(hposition_1 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_1, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    if(hposition_2 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_2, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

// Check csrilu0 factorization
#if defined(__HIP_PLATFORM_AMD__)
    unit_check_general(1, nnz, 1, hcsr_val.data(), iluresult.data());
#elif defined(__HIP_PLATFORM_NVIDIA__)
    unit_check_near(1, nnz, 1, hcsr_val.data(), iluresult.data());
#endif

    // Create info structs for lower and upper part
    std::unique_ptr<csrsv2_struct> unique_ptr_csrsv2_lower(new csrsv2_struct);
    std::unique_ptr<csrsv2_struct> unique_ptr_csrsv2_upper(new csrsv2_struct);

    csrsv2Info_t info_L = unique_ptr_csrsv2_lower->info;
    csrsv2Info_t info_U = unique_ptr_csrsv2_upper->info;

    // Create matrix descriptors for csrsv
    std::unique_ptr<descr_struct> unique_ptr_descr_L(new descr_struct);
    hipsparseMatDescr_t           descr_L = unique_ptr_descr_L->descr;

    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_L, idx_base));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatFillMode(descr_L, HIPSPARSE_FILL_MODE_LOWER));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatDiagType(descr_L, HIPSPARSE_DIAG_TYPE_UNIT));

    std::unique_ptr<descr_struct> unique_ptr_descr_U(new descr_struct);
    hipsparseMatDescr_t           descr_U = unique_ptr_descr_U->descr;

    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_U, idx_base));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatFillMode(descr_U, HIPSPARSE_FILL_MODE_UPPER));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatDiagType(descr_U, HIPSPARSE_DIAG_TYPE_NON_UNIT));

    // Obtain csrsv buffer sizes
    int size_lower, size_upper;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_bufferSize(handle,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      m,
                                                      nnz,
                                                      descr_L,
                                                      dval,
                                                      dptr,
                                                      dcol,
                                                      info_L,
                                                      &size_lower));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_bufferSize(handle,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      m,
                                                      nnz,
                                                      descr_U,
                                                      dval,
                                                      dptr,
                                                      dcol,
                                                      info_U,
                                                      &size_upper));

    // Pick maximum size so that we need only one buffer
    size = std::max(size_lower, size_upper);

    // Allocate buffer on the device
    auto dbuffer_sv_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer_sv = (void*)dbuffer_sv_managed.get();

    // csrsv analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_analysis(handle,
                                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                    m,
                                                    nnz,
                                                    descr_L,
                                                    dval,
                                                    dptr,
                                                    dcol,
                                                    info_L,
                                                    HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                    dbuffer_sv));

    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_analysis(handle,
                                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                    m,
                                                    nnz,
                                                    descr_U,
                                                    dval,
                                                    dptr,
                                                    dcol,
                                                    info_U,
                                                    HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                    dbuffer_sv));

    // Initialize some more structures required for Lz = x
    T h_alpha = static_cast<T>(1);

    std::vector<T> hx(m, static_cast<T>(1));
    std::vector<T> hy_gold(m);
    std::vector<T> hz_gold(m);

    // Allocate device memory
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dz_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dz_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dx      = (T*)dx_managed.get();
    T* dy_1    = (T*)dy_1_managed.get();
    T* dy_2    = (T*)dy_2_managed.get();
    T* dz_1    = (T*)dz_1_managed.get();
    T* dz_2    = (T*)dz_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Solve Lz = x

    // host pointer mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 m,
                                                 nnz,
                                                 &h_alpha,
                                                 descr_L,
                                                 dval,
                                                 dptr,
                                                 dcol,
                                                 info_L,
                                                 dx,
                                                 dz_1,
                                                 HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 dbuffer_sv));

    // Check for zero pivot
    pivot_status_1 = hipsparseXcsrsv2_zeroPivot(handle, info_L, &hposition_1);

    // device pointer mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 m,
                                                 nnz,
                                                 d_alpha,
                                                 descr_L,
                                                 dval,
                                                 dptr,
                                                 dcol,
                                                 info_L,
                                                 dx,
                                                 dz_2,
                                                 HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 dbuffer_sv));

    // Check for zero pivot
    pivot_status_2 = hipsparseXcsrsv2_zeroPivot(handle, info_L, d_position);

    // Host csrsv
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, 0));

    position_gold = csr_lsolve(HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               m,
                               hcsr_row_ptr.data(),
                               hcsr_col_ind.data(),
                               hcsr_val.data(),
                               h_alpha,
                               hx.data(),
                               hz_gold.data(),
                               idx_base,
                               HIPSPARSE_DIAG_TYPE_UNIT,
                               prop.warpSize);

    // Check zero pivot results
    unit_check_general(1, 1, 1, &position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &position_gold, &hposition_2);

    // If zero pivot was found, do not go further
    if(hposition_1 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_1, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    if(hposition_2 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_2, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Copy output from device to CPU
    std::vector<T> hz_1(m);
    std::vector<T> hz_2(m);

    CHECK_HIP_ERROR(hipMemcpy(hz_1.data(), dz_1, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hz_2.data(), dz_2, sizeof(T) * m, hipMemcpyDeviceToHost));

// Check z
#if defined(__HIP_PLATFORM_AMD__)
    unit_check_general(1, m, 1, hz_gold.data(), hz_1.data());
    unit_check_general(1, m, 1, hz_gold.data(), hz_2.data());
#elif defined(__HIP_PLATFORM_NVIDIA__)
    unit_check_near(1, m, 1, hz_gold.data(), hz_1.data());
    unit_check_near(1, m, 1, hz_gold.data(), hz_2.data());
#endif

    // Solve Uy = z

    // host pointer mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 m,
                                                 nnz,
                                                 &h_alpha,
                                                 descr_U,
                                                 dval,
                                                 dptr,
                                                 dcol,
                                                 info_U,
                                                 dz_1,
                                                 dy_1,
                                                 HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 dbuffer_sv));

    // Check for zero pivot
    pivot_status_1 = hipsparseXcsrsv2_zeroPivot(handle, info_U, &hposition_1);

    // device pointer mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsv2_solve(handle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 m,
                                                 nnz,
                                                 d_alpha,
                                                 descr_U,
                                                 dval,
                                                 dptr,
                                                 dcol,
                                                 info_U,
                                                 dz_2,
                                                 dy_2,
                                                 HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 dbuffer_sv));

    // Check for zero pivot
    pivot_status_2 = hipsparseXcsrsv2_zeroPivot(handle, info_U, d_position);

    // Host csrsv
    position_gold = csr_usolve(HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               m,
                               hcsr_row_ptr.data(),
                               hcsr_col_ind.data(),
                               hcsr_val.data(),
                               h_alpha,
                               hz_gold.data(),
                               hy_gold.data(),
                               idx_base,
                               HIPSPARSE_DIAG_TYPE_NON_UNIT,
                               prop.warpSize);

    // Check zero pivot results
    unit_check_general(1, 1, 1, &position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &position_gold, &hposition_2);

    // If zero pivot was found, do not go further
    if(hposition_1 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_1, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    if(hposition_2 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_2, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Copy output from device to CPU
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);

    CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

    // Check z
    unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
    unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRILUSOLVE_HPP
