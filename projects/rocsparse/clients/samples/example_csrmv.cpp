/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse/rocsparse.h"
#include "utils.hpp"
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            return -1;                                                         \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != rocsparse_status_success)                                         \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            return -1;                                                               \
        }                                                                            \
    }

int main(int argc, char* argv[])
{
    // Parse command line
    if(argc < 2)
    {
        std::cerr << argv[0] << " <ndim> [<trials> <batch_size>]" << std::endl;
        return -1;
    }

    rocsparse_int ndim       = atoi(argv[1]);
    int           trials     = 200;
    int           batch_size = 1;

    if(argc > 2)
    {
        trials = atoi(argv[2]);
    }
    if(argc > 3)
    {
        batch_size = atoi(argv[3]);
    }

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    hipDeviceProp_t devProp;
    int             device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem
    std::vector<rocsparse_int> hAptr;
    std::vector<rocsparse_int> hAcol;
    std::vector<double>        hAval;

    rocsparse_int m;
    rocsparse_int n;
    rocsparse_int nnz;

    utils_init_csr_laplace2d(hAptr, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    std::vector<double> hx(n);
    utils_init<double>(hx, 1, n, 1);

    // Offload data to device
    rocsparse_int* dAptr = NULL;
    rocsparse_int* dAcol = NULL;
    double*        dAval = NULL;
    double*        dx    = NULL;
    double*        dy    = NULL;

    HIP_CHECK(hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

    HIP_CHECK(
        hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice));

    double halpha = 1.0f;
    double hbeta  = 0.0;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse csrmv
        ROCSPARSE_CHECK(rocsparse_dcsrmv(handle,
                                         rocsparse_operation_none,
                                         m,
                                         n,
                                         nnz,
                                         &halpha,
                                         descrA,
                                         dAval,
                                         dAptr,
                                         dAcol,
                                         nullptr,
                                         dx,
                                         &hbeta,
                                         dy));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    double time = utils_time_us();

    // CSR matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            // Call rocsparse csrmv
            ROCSPARSE_CHECK(rocsparse_dcsrmv(handle,
                                             rocsparse_operation_none,
                                             m,
                                             n,
                                             nnz,
                                             &halpha,
                                             descrA,
                                             dAval,
                                             dAptr,
                                             dAcol,
                                             nullptr,
                                             dx,
                                             &hbeta,
                                             dy));
        }

        // Device synchronization
        HIP_CHECK(hipDeviceSynchronize());
    }

    time             = (utils_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth = static_cast<double>(sizeof(double) * (size_t(2) * m + nnz)
                                           + sizeof(rocsparse_int) * (size_t(m) + 1 + nnz))
                       / time / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dcsrmv WITHOUT meta data ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(12)
              << halpha << std::setw(12) << hbeta << std::setw(12) << gflops << std::setw(12)
              << bandwidth << std::setw(12) << time << std::endl;

    // Create meta data
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Analyse CSR matrix
    ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(
        handle, rocsparse_operation_none, m, n, nnz, descrA, dAval, dAptr, dAcol, info));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse csrmv
        ROCSPARSE_CHECK(rocsparse_dcsrmv(handle,
                                         rocsparse_operation_none,
                                         m,
                                         n,
                                         nnz,
                                         &halpha,
                                         descrA,
                                         dAval,
                                         dAptr,
                                         dAcol,
                                         info,
                                         dx,
                                         &hbeta,
                                         dy));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    time = utils_time_us();

    // CSR matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            // Call rocsparse csrmv
            ROCSPARSE_CHECK(rocsparse_dcsrmv(handle,
                                             rocsparse_operation_none,
                                             m,
                                             n,
                                             nnz,
                                             &halpha,
                                             descrA,
                                             dAval,
                                             dAptr,
                                             dAcol,
                                             info,
                                             dx,
                                             &hbeta,
                                             dy));
        }

        // Device synchronization
        HIP_CHECK(hipDeviceSynchronize());
    }

    time      = (utils_time_us() - time) / (trials * batch_size * 1e3);
    bandwidth = static_cast<double>(sizeof(double) * (size_t(2) * m + nnz)
                                    + sizeof(rocsparse_int) * (size_t(m) + 1 + nnz))
                / time / 1e6;
    gflops = static_cast<double>(size_t(2) * nnz) / time / 1e6;

    std::vector<double> hy(m);
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(int i = 0; i < std::min(20, m); i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dcsrmv WITH meta data ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(12)
              << halpha << std::setw(12) << hbeta << std::setw(12) << gflops << std::setw(12)
              << bandwidth << std::setw(12) << time << std::endl;

    // Clear up on device
    HIP_CHECK(hipFree(dAptr));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
