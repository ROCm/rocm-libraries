/*! \file */
/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <iomanip>
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

    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Print rocSPARSE version and revision
    int  ver;
    char rev[64];

    ROCSPARSE_CHECK(rocsparse_get_version(handle, &ver));
    ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev));

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    // Input data

    // Number of rows and columns
    rocsparse_int m = 3;
    rocsparse_int n = 5;

    // Matrix A (m x n)
    // (  9 10 11 12 13)
    // ( 14 15 16 17 18)
    // ( 19 20 21 22 23)

    // Matrix A in column-major
    rocsparse_int lda    = m;
    double        hA[15] = {9, 14, 19, 10, 15, 20, 11, 16, 21, 12, 17, 22, 13, 18, 23};

    // Vector x
    // ( 1 )
    // ( 2 )
    // ( 0 )
    // ( 3 )
    // ( 0 )

    // Number of non-zero entries
    rocsparse_int nnz = 3;

    // Vector x indices
    rocsparse_int hx_ind[3] = {0, 1, 3};

    // Vector x values
    double hx_val[3] = {1, 2, 3};

    // Vector y
    // ( 4 )
    // ( 5 )
    // ( 6 )

    // Vector y values
    double hy[3] = {4, 5, 6};

    // Scalar alpha and beta
    double alpha = 3.7;
    double beta  = 1.3;

    // Matrix operation
    rocsparse_operation trans = rocsparse_operation_none;

    // Index base
    rocsparse_index_base base = rocsparse_index_base_zero;

    // Offload data to device
    double*        dA;
    rocsparse_int* dx_ind;
    double*        dx_val;
    double*        dy;

    HIP_CHECK(hipMalloc((void**)&dA, sizeof(double) * m * n));
    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(dA, hA, sizeof(double) * m * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx_val, hx_val, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(double) * m, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dgemvi_buffer_size(handle, trans, m, n, nnz, &buffer_size));

    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Call dgemvi
    ROCSPARSE_CHECK(rocsparse_dgemvi(
        handle, trans, m, n, &alpha, dA, lda, nnz, dx_val, dx_ind, &beta, dy, base, buffer));

    // Print result
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout.precision(2);
    std::cout << "y:";

    for(int i = 0; i < m; ++i)
    {
        std::cout << std::scientific << " " << hy[i];
    }
    std::cout << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dx_ind));
    HIP_CHECK(hipFree(dx_val));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(buffer));

    return 0;
}
