/*! \file */
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

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>

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
    // Query device
    int ndev;
    HIP_CHECK(hipGetDeviceCount(&ndev));

    if(ndev < 1)
    {
        std::cerr << "No HIP device found" << std::endl;
        return -1;
    }

    // Query device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << std::endl;

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Print rocSPARSE version and revision
    int  ver;
    char rev[64];

    ROCSPARSE_CHECK(rocsparse_get_version(handle, &ver));
    ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev));

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    // Input data

    // Matrix A (m x k)
    // ( 1.0  2.0  0.0  3.0  0.0 )
    // ( 0.0  4.0  5.0  0.0  0.0 )
    // ( 6.0  0.0  0.0  7.0  8.0 )

    // Number of rows and columns
    rocsparse_int m = 3;
    rocsparse_int n = 2;
    rocsparse_int k = 5;

    // Number of non-zero entries
    rocsparse_int nnz = 8;

    // CSR row pointers
    rocsparse_int hcsr_row_ptr[4] = {0, 3, 5, 8};

    // CSR column indices
    rocsparse_int hcsr_col_ind[8] = {0, 1, 3, 1, 2, 0, 3, 4};

    // CSR values
    double hcsr_val[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // Transposition of the matrix
    rocsparse_operation transA = rocsparse_operation_none;
    rocsparse_operation transB = rocsparse_operation_none;

    // Matrix B (k x n)
    // (  9.0  10.0 )
    // ( 11.0  12.0 )
    // ( 13.0  14.0 )
    // ( 15.0  16.0 )
    // ( 17.0  18.0 )

    // Matrix B in column-major
    rocsparse_int ldb    = k;
    double        hB[10] = {9.0, 11.0, 13.0, 15.0, 17.0, 10.0, 12.0, 14.0, 16.0, 18.0};

    // Matrix C (m x n) in column-major
    rocsparse_int ldc   = m;
    double        hC[6] = {19.0, 21.0, 23.0, 20.0, 22.0, 24.0};

    // Scalar alpha and beta
    double alpha = 3.7;
    double beta  = 1.3;

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Offload data to device
    rocsparse_int* dcsr_row_ptr;
    rocsparse_int* dcsr_col_ind;
    double*        dcsr_val;
    double*        dB;
    double*        dC;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(double) * k * n));
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(double) * m * n));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(double) * k * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC, sizeof(double) * m * n, hipMemcpyHostToDevice));

    // Call dcsrmm
    ROCSPARSE_CHECK(rocsparse_dcsrmm(handle,
                                     transA,
                                     transB,
                                     m,
                                     n,
                                     k,
                                     nnz,
                                     &alpha,
                                     descr,
                                     dcsr_val,
                                     dcsr_row_ptr,
                                     dcsr_col_ind,
                                     dB,
                                     ldb,
                                     &beta,
                                     dC,
                                     ldc));

    // Print result
    HIP_CHECK(hipMemcpy(hC, dC, sizeof(double) * m * n, hipMemcpyDeviceToHost));

    std::cout << "C:" << std::endl;

    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            std::cout << " " << hC[i + j * ldc];
        }

        std::cout << std::endl;
    }

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}
