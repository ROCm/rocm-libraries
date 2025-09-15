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
    // A sparse matrix
    // 1 0 3 4
    // 0 0 5 1
    // 0 2 0 0
    // 4 0 0 8
    rocsparse_int hAptr[5] = {0, 3, 5, 6, 8};
    rocsparse_int hAcol[8] = {0, 2, 3, 2, 3, 1, 0, 3};
    double        hAval[8] = {1.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0};

    rocsparse_int m   = 4;
    rocsparse_int n   = 4;
    rocsparse_int nnz = 8;

    double halpha = 1.0;
    double hbeta  = 0.0;

    double hx[4] = {1.0, 2.0, 3.0, 4.0};
    double hy[4] = {4.0, 5.0, 6.0, 7.0};

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Matrix descriptors
    rocsparse_mat_descr descrA;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA));

    rocsparse_mat_descr descrB;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrB));

    // Offload data to device
    rocsparse_int* dAptr;
    rocsparse_int* dAcol;
    double*        dAval;
    double*        dx;
    double*        dy;

    HIP_CHECK(hipMalloc(&dAptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dAcol, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dAval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&dy, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(dAptr, hAptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice));

    // Convert CSR matrix to ELL format
    rocsparse_int* dBcol;
    double*        dBval;

    // Determine ELL width
    rocsparse_int ell_width;
    ROCSPARSE_CHECK(rocsparse_csr2ell_width(handle, m, descrA, dAptr, descrB, &ell_width));

    // Allocate memory for ELL storage format
    HIP_CHECK(hipMalloc(&dBcol, sizeof(rocsparse_int) * ell_width * m));
    HIP_CHECK(hipMalloc(&dBval, sizeof(double) * ell_width * m));

    // Convert matrix from CSR to ELL
    ROCSPARSE_CHECK(rocsparse_dcsr2ell(
        handle, m, descrA, dAval, dAptr, dAcol, descrB, ell_width, dBval, dBcol));

    // Clean up CSR structures
    HIP_CHECK(hipFree(dAptr));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));

    // Call rocsparse ellmv
    ROCSPARSE_CHECK(rocsparse_dellmv(handle,
                                     rocsparse_operation_none,
                                     m,
                                     n,
                                     &halpha,
                                     descrB,
                                     dBval,
                                     dBcol,
                                     ell_width,
                                     dx,
                                     &hbeta,
                                     dy));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(size_t i = 0; i < m; i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear up on device
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrB));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dBcol));
    HIP_CHECK(hipFree(dBval));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    return 0;
}
//! [doc example]
