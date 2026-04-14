/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename J, typename T>
void run_example(rocsparse_handle handle, int ndim, int trials, int batch_size)
{
    // Generate problem
    std::vector<I> hAptr;
    std::vector<J> hAcol;
    std::vector<T> hAval;

    J m;
    J n;
    I nnz;

    utils_init_csr_laplace2d(hAptr, hAcol, hAval, ndim, ndim, m, n, nnz, rocsparse_index_base_zero);

    // Sample some random data
    utils_seedrand();

    T halpha = utils_random<T>();
    T hbeta  = (T)0;

    std::vector<T> hx(n);
    utils_init<T>(hx, 1, n, 1);

    // Offload data to device
    I* dAptr = NULL;
    J* dAcol = NULL;
    T* dAval = NULL;
    T* dx    = NULL;
    T* dy    = NULL;

    HIP_CHECK(hipMalloc((void**)&dAptr, sizeof(I) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(J) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAval, sizeof(T) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(T) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(T) * m));

    HIP_CHECK(hipMemcpy(dAptr, hAptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));

    // Types
    const rocsparse_indextype itype = utils_indextype<I>();
    const rocsparse_indextype jtype = utils_indextype<J>();
    const rocsparse_datatype  ttype = utils_datatype<T>();

    // Create descriptors
    rocsparse_spmat_descr A;
    rocsparse_dnvec_descr x;
    rocsparse_dnvec_descr y;

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &A, m, n, nnz, dAptr, dAcol, dAval, itype, jtype, rocsparse_index_base_zero, ttype));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&x, n, dx, ttype));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&y, m, dy, ttype));

    rocsparse_spmv_descr spmv_descr;
    ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&spmv_descr));

    const rocsparse_spmv_alg spmv_alg = rocsparse_spmv_alg_csr_adaptive;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(
        handle, spmv_descr, rocsparse_spmv_input_alg, &spmv_alg, sizeof(spmv_alg), nullptr));

    const rocsparse_operation spmv_operation = rocsparse_operation_none;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_operation,
                                             &spmv_operation,
                                             sizeof(spmv_operation),
                                             nullptr));

    const rocsparse_datatype spmv_scalar_datatype = utils_datatype<T>();
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_scalar_datatype,
                                             &spmv_scalar_datatype,
                                             sizeof(spmv_scalar_datatype),
                                             nullptr));

    const rocsparse_datatype spmv_compute_datatype = utils_datatype<T>();
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_compute_datatype,
                                             &spmv_compute_datatype,
                                             sizeof(spmv_compute_datatype),
                                             nullptr));

    // Call spmv to get buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(
        handle, spmv_descr, A, x, y, rocsparse_v2_spmv_stage_analysis, &buffer_size, nullptr));

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Call rocsparse spmv
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &halpha,
                                      A,
                                      x,
                                      &hbeta,
                                      y,
                                      rocsparse_v2_spmv_stage_analysis,
                                      buffer_size,
                                      temp_buffer,
                                      nullptr));

    HIP_CHECK(hipFree(temp_buffer));
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(
        handle, spmv_descr, A, x, y, rocsparse_v2_spmv_stage_compute, &buffer_size, nullptr));

    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call rocsparse spmv
        ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                          spmv_descr,
                                          &halpha,
                                          A,
                                          x,
                                          &hbeta,
                                          y,
                                          rocsparse_v2_spmv_stage_compute,
                                          buffer_size,
                                          temp_buffer,
                                          nullptr));
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
            // Call rocsparse spmv
            ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                              spmv_descr,
                                              &halpha,
                                              A,
                                              x,
                                              &hbeta,
                                              y,
                                              rocsparse_v2_spmv_stage_compute,
                                              buffer_size,
                                              temp_buffer,
                                              nullptr));
        }

        // Device synchronization
        HIP_CHECK(hipDeviceSynchronize());
    }

    time             = (utils_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth = static_cast<double>(sizeof(T) * (size_t(2) * m + nnz) + sizeof(I) * (m + 1)
                                           + sizeof(J) * nnz)
                       / time / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;

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
    HIP_CHECK(hipFree(temp_buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(spmv_descr));
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
    std::cout << "### rocsparse_v2_spmv<int32_t, int32_t, float> ###" << std::endl;
    run_example<int32_t, int32_t, float>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int32_t, float> ###" << std::endl;
    run_example<int64_t, int32_t, float>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int64_t, float> ###" << std::endl;
    run_example<int64_t, int64_t, float>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    // double precision, real
    std::cout << "### rocsparse_v2_spmv<int32_t, int32_t, double> ###" << std::endl;
    run_example<int32_t, int32_t, double>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int32_t, double> ###" << std::endl;
    run_example<int64_t, int32_t, double>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int64_t, double> ###" << std::endl;
    run_example<int64_t, int64_t, double>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    // single precision, complex
    std::cout << "### rocsparse_v2_spmv<int32_t, int32_t, rocsparse_float_complex> ###"
              << std::endl;
    run_example<int32_t, int32_t, rocsparse_float_complex>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int32_t, rocsparse_float_complex> ###"
              << std::endl;
    run_example<int64_t, int32_t, rocsparse_float_complex>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int64_t, rocsparse_float_complex> ###"
              << std::endl;
    run_example<int64_t, int64_t, rocsparse_float_complex>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    // double precision, complex
    std::cout << "### rocsparse_v2_spmv<int32_t, int32_t, rocsparse_double_complex> ###"
              << std::endl;
    run_example<int32_t, int32_t, rocsparse_double_complex>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int32_t, rocsparse_double_complex> ###"
              << std::endl;
    run_example<int64_t, int32_t, rocsparse_double_complex>(handle, ndim, trials, batch_size);
    std::cout << "### rocsparse_v2_spmv<int64_t, int64_t, rocsparse_double_complex> ###"
              << std::endl;
    run_example<int64_t, int64_t, rocsparse_double_complex>(handle, ndim, trials, batch_size);
    std::cout << std::endl;

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
