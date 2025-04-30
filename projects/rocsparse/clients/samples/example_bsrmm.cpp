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
    //     ( 1 2 0 3 0 0 )
    // A = ( 0 4 5 0 0 0 )
    //     ( 0 0 0 7 8 0 )
    //     ( 0 0 1 2 4 1 )

    // Number of rows and columns
    rocsparse_int block_dim = 2;
    rocsparse_int mb        = 2;
    rocsparse_int kb        = 3;
    rocsparse_int n         = 10;
    rocsparse_int m         = mb * block_dim;
    rocsparse_int k         = kb * block_dim;

    // Number of non-zero block entries
    rocsparse_int nnzb = 4;

    // BSR row pointers
    rocsparse_int hbsr_row_ptr[3] = {0, 2, 4};

    // BSR column indices
    rocsparse_int hbsr_col_ind[4] = {0, 1, 1, 2};

    // BSR values
    double hbsr_val[16]
        = {1.0, 2.0, 0.0, 4.0, 0.0, 3.0, 5.0, 0.0, 0.0, 7.0, 1.0, 2.0, 8.0, 0.0, 4.0, 1.0};

    // Transposition of the matrix
    rocsparse_direction dir    = rocsparse_direction_row;
    rocsparse_operation transA = rocsparse_operation_none;
    rocsparse_operation transB = rocsparse_operation_none;

    // Matrix B (k x n) column major order
    //     ( 9  11 13 15 17 10 12 14 16 18 )
    //     ( 8  10 1  10 6  11 7  3  12 17 )
    // B = ( 11 11 0  4  6  12 2  9  13 2  )
    //     ( 15 3  2  3  8  1  2  4  6  6  )
    //     ( 2  5  7  0  1  15 9  4  10 1  )
    //     ( 7  12 12 1  12 5  1  11 1  14 )

    // Matrix B in column-major
    rocsparse_int ldb = k;
    double        hB[6 * 10]
        = {9, 8, 11, 15, 2,  7, 11, 10, 11, 3,  5,  12, 13, 1, 0,  2,  7,  12, 15, 10,
           4, 3, 0,  1,  17, 6, 6,  8,  1,  12, 10, 11, 12, 1, 15, 5,  12, 7,  2,  2,
           9, 1, 14, 3,  9,  4, 4,  11, 16, 12, 13, 6,  10, 1, 18, 17, 2,  6,  1,  14};

    // Matrix C (m x n) column major order
    //     ( 0 0 0 0 0 0 0 0 0 0 )
    // C = ( 0 0 0 0 0 0 0 0 0 0 )
    //     ( 0 0 0 0 0 0 0 0 0 0 )
    //     ( 0 0 0 0 0 0 0 0 0 0 )

    // Matrix C (m x n) in column-major
    rocsparse_int ldc        = m;
    double        hC[4 * 10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Scalar alpha and beta
    double alpha = 1.0;
    double beta  = 0.0;

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Offload data to device
    rocsparse_int* dbsr_row_ptr;
    rocsparse_int* dbsr_col_ind;
    double*        dbsr_val;
    double*        dB;
    double*        dC;

    HIP_CHECK(hipMalloc((void**)&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsr_val, sizeof(double) * nnzb * block_dim * block_dim));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(double) * k * n));
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(double) * m * n));

    HIP_CHECK(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dbsr_val, hbsr_val, sizeof(double) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(double) * k * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC, sizeof(double) * m * n, hipMemcpyHostToDevice));

    // Call dbsrmm
    ROCSPARSE_CHECK(rocsparse_dbsrmm(handle,
                                     dir,
                                     transA,
                                     transB,
                                     mb,
                                     n,
                                     kb,
                                     nnzb,
                                     &alpha,
                                     descr,
                                     dbsr_val,
                                     dbsr_row_ptr,
                                     dbsr_col_ind,
                                     block_dim,
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
    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}
