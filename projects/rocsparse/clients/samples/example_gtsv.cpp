/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utils.hpp"

#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <rocsparse/rocsparse.h>
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

    rocsparse_int m   = ndim;
    rocsparse_int n   = 32;
    rocsparse_int ldb = m;

    // Generate problem
    std::vector<double> hdl(m, 1.0);
    std::vector<double> hd(m, 1.0);
    std::vector<double> hdu(m, 1.0);
    std::vector<double> hB(ldb * n, 1.0);

    double* ddl = NULL;
    double* dd  = NULL;
    double* ddu = NULL;
    double* dB  = NULL;

    HIP_CHECK(hipMalloc((void**)&ddl, sizeof(double) * m));
    HIP_CHECK(hipMalloc((void**)&dd, sizeof(double) * m));
    HIP_CHECK(hipMalloc((void**)&ddu, sizeof(double) * m));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(double) * ldb * n));

    // Copy data to device
    HIP_CHECK(hipMemcpy(ddl, hdl.data(), sizeof(double) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, hd.data(), sizeof(double) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddu, hdu.data(), sizeof(double) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(double) * ldb * n, hipMemcpyHostToDevice));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dgtsv_buffer_size(handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size));

    // Allocate temporary buffer
    std::cout << "Allocating " << (buffer_size >> 10) << "kB temporary storage buffer" << std::endl;

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        ROCSPARSE_CHECK(rocsparse_dgtsv(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    double time = utils_time_us();

    // Call dgtsv to peform tri-diagonal solve
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            ROCSPARSE_CHECK(rocsparse_dgtsv(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer));

            // Device synchronization
            HIP_CHECK(hipDeviceSynchronize());
        }
    }

    double solve_time = (utils_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth  = (sizeof(double) * (size_t(3) * m + size_t(2) * m * n)) / solve_time / 1e6;

    // Print result
    HIP_CHECK(hipMemcpy(hB.data(), dB, sizeof(double) * ldb * n, hipMemcpyDeviceToHost));

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dgtsv ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "GB/s"
              << std::setw(12) << "solve msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << bandwidth
              << std::setw(12) << solve_time << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(ddl));
    HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(ddu));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
