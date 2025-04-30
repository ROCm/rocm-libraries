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
    bool set_triangular = false;
    // Parse command line
    if(argc < 2)
    {
        std::cerr << argv[0]
                  << " <ndim> [ <set_triangular> (=0), <trials> (=200)  <batch_size> (=1)]"
                  << std::endl;
        return -1;
    }

    rocsparse_int ndim       = atoi(argv[1]);
    int           trials     = 200;
    int           batch_size = 1;

    if(argc > 2)
    {
        int i = atoi(argv[2]);
        if(i == 1)
        {
            set_triangular = true;
        }
    }
    if(argc > 3)
    {
        trials = atoi(argv[3]);
    }
    if(argc > 4)
    {
        batch_size = atoi(argv[4]);
    }

    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // Transposition of the matrix
    rocsparse_operation trans = rocsparse_operation_none;

    // Analysis policy
    rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // Solve policy
    rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    hipDeviceProp_t devProp;
    int             device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcsr_col_ind;
    std::vector<double>        hcsr_val;

    rocsparse_int m;
    rocsparse_int nnz;
    double        alpha = 1.0f;

    utils_init_csr_laplace2d(
        hcsr_row_ptr, hcsr_col_ind, hcsr_val, ndim, ndim, m, m, nnz, rocsparse_index_base_zero);
    rocsparse_int nnz0 = nnz;
    utils_seedrand();

    if(set_triangular)
    {
        //
        // Extract Lower non unit diagonal with base 0, as configured here.
        //
        rocsparse_int* ptr = new rocsparse_int[m + 1];
        rocsparse_int* ind = new rocsparse_int[nnz];
        double*        val = new double[nnz];
        for(int i = 0; i <= m; ++i)
        {
            ptr[i] = 0;
        }
        for(int i = 0; i < m; ++i)
        {
            for(int k = hcsr_row_ptr[i]; k < hcsr_row_ptr[i + 1]; ++k)
            {
                const int j = hcsr_col_ind[k];
                if(j <= i)
                {
                    ptr[i + 1] += 1;
                }
            }
        }
        for(int i = 1; i <= m; ++i)
        {
            ptr[i] += ptr[i - 1];
        }
        nnz = ptr[m];
        for(int i = 0; i < m; ++i)
        {
            for(int k = hcsr_row_ptr[i]; k < hcsr_row_ptr[i + 1]; ++k)
            {
                const int j = hcsr_col_ind[k];
                if(j <= i)
                {
                    ind[ptr[i]] = j;
                    val[ptr[i]] = hcsr_val[k];
                    ptr[i] += 1;
                }
            }
        }
        for(int i = m; i > 0; --i)
        {
            ptr[i] = ptr[i - 1];
        }
        ptr[0] = 0;

        for(int i = 0; i <= m; ++i)
        {
            hcsr_row_ptr[i] = ptr[i];
        }
        for(int i = 0; i < nnz; ++i)
        {
            hcsr_col_ind[i] = ind[i];
        }
        for(int i = 0; i < nnz; ++i)
        {
            hcsr_val[i] = val[i];
        }
        delete[] val;
        delete[] ind;
        delete[] ptr;
    }

    std::vector<double> hx(m);
    std::vector<double> hy(m);
    utils_init<double>(hx, 1, m, 1);

    rocsparse_int* dcsr_row_ptr = NULL;
    rocsparse_int* dcsr_col_ind = NULL;
    double*        dcsr_val     = NULL;
    double*        dx           = NULL;
    double*        dy           = NULL;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * m));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

    // Copy data to device
    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(double) * m, hipMemcpyHostToDevice));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    if(set_triangular)
    {
        ROCSPARSE_CHECK(rocsparse_set_mat_type(descr, rocsparse_matrix_type_triangular));
    }

    // Matrix fill mode
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));

    // Matrix diagonal type
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit));

    // Matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dcsritsv_buffer_size(
        handle, trans, m, nnz, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, &buffer_size));

    // Allocate temporary buffer
    std::cout << "Allocating " << (buffer_size >> 10) << "kB temporary storage buffer" << std::endl;

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    static constexpr rocsparse_int s_nmaxiter = 400;
    rocsparse_int                  nmaxiter   = s_nmaxiter;
    double                         tol[1]     = {1.0e-5};
    double                         history[s_nmaxiter];
    std::cout << "iter " << s_nmaxiter << std::endl;
    std::cout << "tol  " << tol[0] << std::endl;

    // Perform analysis step
    ROCSPARSE_CHECK(rocsparse_dcsritsv_analysis(handle,
                                                trans,
                                                m,
                                                nnz,
                                                descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                info,
                                                analysis_policy,
                                                solve_policy,
                                                temp_buffer));
    HIP_CHECK(hipMemset(dy, 0, sizeof(double) * m));
    ROCSPARSE_CHECK(rocsparse_dcsritsv_solve_ex(handle,
                                                &nmaxiter,
                                                0,
                                                tol,
                                                history,
                                                trans,
                                                m,
                                                nnz,
                                                &alpha,
                                                descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                info,
                                                dx,
                                                dy,
                                                solve_policy,
                                                temp_buffer));
    std::cout << "history, niter  " << nmaxiter << std::endl;

    for(int i = 0; i < std::min(nmaxiter, 10); ++i)
        std::cout << "history[" << i << "] =  " << history[i] << std::endl;
    if(nmaxiter > 10)
        std::cout << "history[10] =  ... " << std::endl;
    for(int i = 0; i < 10; ++i)
    {
        if(nmaxiter - 10 + i > 10)
        {
            std::cout << "history[" << nmaxiter - 10 + i << "] =  " << history[nmaxiter - 10 + i]
                      << std::endl;
        }
    }
    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        nmaxiter = s_nmaxiter;
        HIP_CHECK(hipMemset(dy, 0, sizeof(double) * m));
        ROCSPARSE_CHECK(rocsparse_dcsritsv_solve_ex(handle,
                                                    &nmaxiter,
                                                    0,
                                                    tol,
                                                    history,
                                                    trans,
                                                    m,
                                                    nnz,
                                                    &alpha,
                                                    descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    info,
                                                    dx,
                                                    dy,
                                                    solve_policy,
                                                    temp_buffer));
    }

    // Device synchronization
    HIP_CHECK(hipDeviceSynchronize());

    // Start time measurement
    double ttime = 0;

    // Call dcsritsv to perform lower triangular solve Ly = x
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            nmaxiter = s_nmaxiter;
            HIP_CHECK(hipMemset(dy, 0, sizeof(double) * m));
            double time = utils_time_us();

            ROCSPARSE_CHECK(rocsparse_dcsritsv_solve_ex(handle,
                                                        &nmaxiter,
                                                        0,
                                                        tol,
                                                        history,
                                                        trans,
                                                        m,
                                                        nnz,
                                                        &alpha,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info,
                                                        dx,
                                                        dy,
                                                        solve_policy,
                                                        temp_buffer));

            // Device synchronization
            HIP_CHECK(hipDeviceSynchronize());
            // Device synchronization
            double solve_time = (utils_time_us() - time);
            ttime += solve_time;
        }
    }

    double solve_time = ttime / (trials * batch_size * 1e3);
    double bandwidth  = (sizeof(rocsparse_int) * (m + 1 + nnz0) + sizeof(double) * (m + m + nnz0))
                       / solve_time / 1e6;

    // Check for zero pivots
    rocsparse_int    pivot;
    rocsparse_status status = rocsparse_csritsv_zero_pivot(handle, descr, info, &pivot);

    if(status == rocsparse_status_zero_pivot)
    {
        std::cout << "WARNING: Found zero pivot in matrix row " << pivot << std::endl;
    }

    // Print result
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    std::cout << std::endl << "### rocsparse_dcsritsv ###" << std::endl;
    std::cout << std::setw(12) << "m" << std::setw(12) << "nnz" << std::setw(12) << "alpha"
              << std::setw(12) << "GB/s" << std::setw(12) << "solve msec" << std::endl;
    std::cout << std::setw(12) << m << std::setw(12) << nnz << std::setw(12) << alpha
              << std::setw(12) << bandwidth << std::setw(12) << solve_time << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
