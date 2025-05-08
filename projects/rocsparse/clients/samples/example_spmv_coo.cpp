/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
            exit(-1);                                                          \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != rocsparse_status_success)                                         \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            exit(-1);                                                                \
        }                                                                            \
    }

template <typename I, typename T>
void run_example(rocsparse_handle handle, int ndim, int trials, int batch_size)
{
    // Generate problem
    std::vector<I> hArow;
    std::vector<I> hAcol;
    std::vector<T> hAval;

    I m;
    I n;
    I nnz;

    utils_init_coo_laplace2d(hArow, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    // Sample some random data
    utils_seedrand();

    T halpha = utils_random<T>();
    T hbeta  = (T)0;

    std::vector<T> hx(n);
    utils_init<T>(hx, 1, n, 1);

    // Offload data to device
    I* dArow = NULL;
    I* dAcol = NULL;
    T* dAval = NULL;
    T* dx    = NULL;
    T* dy    = NULL;

    HIP_CHECK(hipMalloc((void**)&dArow, sizeof(I) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(I) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAval, sizeof(T) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(T) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(T) * m));

    HIP_CHECK(hipMemcpy(dArow, hArow.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));

    // Types
    rocsparse_indextype itype = utils_indextype<I>();
    rocsparse_datatype  ttype = utils_datatype<T>();

    // Create descriptors
    rocsparse_spmat_descr A;
    rocsparse_dnvec_descr x;
    rocsparse_dnvec_descr y;

    ROCSPARSE_CHECK(rocsparse_create_coo_descr(
        &A, m, n, nnz, dArow, dAcol, dAval, itype, rocsparse_index_base_zero, ttype));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&x, n, dx, ttype));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&y, m, dy, ttype));

    // Query for buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   rocsparse_operation_none,
                                   &halpha,
                                   A,
                                   x,
                                   &hbeta,
                                   y,
                                   ttype,
                                   rocsparse_spmv_alg_default,
                                   rocsparse_spmv_stage_buffer_size,
                                   &buffer_size,
                                   nullptr));

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse spmv
        ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                       rocsparse_operation_none,
                                       &halpha,
                                       A,
                                       x,
                                       &hbeta,
                                       y,
                                       ttype,
                                       rocsparse_spmv_alg_default,
                                       rocsparse_spmv_stage_compute,
                                       &buffer_size,
                                       temp_buffer));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    double time = utils_time_us();

    // COO matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            // Call rocsparse spmv
            ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                           rocsparse_operation_none,
                                           &halpha,
                                           A,
                                           x,
                                           &hbeta,
                                           y,
                                           ttype,
                                           rocsparse_spmv_alg_default,
                                           rocsparse_spmv_stage_compute,
                                           &buffer_size,
                                           temp_buffer));
        }

        // Device synchronization
        HIP_CHECK(hipDeviceSynchronize());
    }

    time             = (utils_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth = static_cast<double>(sizeof(T) * (size_t(2) * m + nnz) + sizeof(I) * (m + 1)
                                           + sizeof(I) * nnz)
                       / time / 1e6;
    double gflops = static_cast<double>(size_t(2) * nnz) / time / 1e6;

    std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
              << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
              << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(12)
              << halpha << std::setw(12) << hbeta << std::setw(12) << gflops << std::setw(12)
              << bandwidth << std::setw(12) << time << std::endl;

    // Clear up on device
    HIP_CHECK(hipFree(dArow));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(temp_buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(A));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(y));
}

int main(int argc, char* argv[])
{
    // Parse command line
    if(argc < 2)
    {
        std::cerr << argv[0] << " <ndim> [<trials> <batch_size>]" << std::endl;
        return -1;
    }

    int ndim       = atoi(argv[1]);
    int trials     = 200;
    int batch_size = 1;

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

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl;

    // single precision, real
    std::cout << "### rocsparse_spmv<int32_t, float> ###" << std::endl;
    run_example<int32_t, float>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_spmv<int64_t, float> ###" << std::endl;
    run_example<int64_t, float>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    // double precision, real
    std::cout << "### rocsparse_spmv<int32_t, double> ###" << std::endl;
    run_example<int32_t, double>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_spmv<int64_t, double> ###" << std::endl;
    run_example<int64_t, double>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    // single precision, complex
    std::cout << "### rocsparse_spmv<int32_t, rocsparse_float_complex> ###" << std::endl;
    run_example<int32_t, rocsparse_float_complex>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_spmv<int64_t, rocsparse_float_complex> ###" << std::endl;
    run_example<int64_t, rocsparse_float_complex>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    // double precision, complex
    std::cout << "### rocsparse_spmv<int32_t, rocsparse_double_complex> ###" << std::endl;
    run_example<int32_t, rocsparse_double_complex>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_spmv<int64_t, rocsparse_double_complex> ###" << std::endl;
    run_example<int64_t, rocsparse_double_complex>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
