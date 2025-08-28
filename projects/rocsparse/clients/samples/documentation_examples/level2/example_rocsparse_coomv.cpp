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

#include <iostream>

#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                                        \
        }                                                                                     \
    }

#define ROCSPARSE_CHECK(stat)                                                         \
    {                                                                                 \
        if(stat != rocsparse_status_success)                                          \
        {                                                                             \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ \
                      << std::endl;                                                   \
            return -1;                                                                \
        }                                                                             \
    }

//! [doc example]
int main()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // A sparse matrix
    // 1 0 3 4
    // 0 0 5 1
    // 0 2 0 0
    // 4 0 0 8
    rocsparse_int hArow[8] = {0, 0, 0, 1, 1, 2, 3, 3};
    rocsparse_int hAcol[8] = {0, 2, 3, 2, 3, 1, 0, 3};
    double        hAval[8] = {1.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0};

    rocsparse_int m   = 4;
    rocsparse_int n   = 4;
    rocsparse_int nnz = 8;

    double halpha = 1.0;
    double hbeta  = 0.0;

    double hx[4] = {1.0, 2.0, 3.0, 4.0};

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA));

    // Offload data to device
    rocsparse_int* dArow;
    rocsparse_int* dAcol;
    double*        dAval;
    double*        dx;
    double*        dy;

    HIP_CHECK(hipMalloc(&dArow, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dAcol, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dAval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&dy, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(dArow, hArow, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice));

    // Call rocsparse coomv
    ROCSPARSE_CHECK(rocsparse_dcoomv(handle,
                                     rocsparse_operation_none,
                                     m,
                                     n,
                                     nnz,
                                     &halpha,
                                     descrA,
                                     dAval,
                                     dArow,
                                     dAcol,
                                     dx,
                                     &hbeta,
                                     dy));

    // Copy back to host
    double hy[4];
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(size_t i = 0; i < 4; i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear up on device
    HIP_CHECK(hipFree(dArow));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    return 0;
}
//! [doc example]
