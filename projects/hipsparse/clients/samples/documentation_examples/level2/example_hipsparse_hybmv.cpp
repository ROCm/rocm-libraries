/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <hipsparse/hipsparse.h>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(stat)                                               \
    {                                                                 \
        if(stat != hipSuccess)                                        \
        {                                                             \
            fprintf(stderr, "Error: hip error in line %d", __LINE__); \
            return -1;                                                \
        }                                                             \
    }

#define HIPSPARSE_CHECK(stat)                                               \
    {                                                                       \
        if(stat != HIPSPARSE_STATUS_SUCCESS)                                \
        {                                                                   \
            fprintf(stderr, "Error: hipsparse error in line %d", __LINE__); \
            return -1;                                                      \
        }                                                                   \
    }

//! [doc example]
int main(int argc, char* argv[])
{
    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // A sparse matrix
    // 1 0 3 4
    // 0 0 5 1
    // 0 2 0 0
    // 4 0 0 8
    int    hAptr[5] = {0, 3, 5, 6, 8};
    int    hAcol[8] = {0, 2, 3, 2, 3, 1, 0, 3};
    double hAval[8] = {1.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0};

    int m   = 4;
    int n   = 4;
    int nnz = 8;

    double halpha = 1.0;
    double hbeta  = 0.0;

    double hx[4] = {1.0, 2.0, 3.0, 4.0};
    double hy[4] = {4.0, 5.0, 6.0, 7.0};

    // Matrix descriptor
    hipsparseMatDescr_t descrA;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrA));

    // Offload data to device
    int*    dAptr = NULL;
    int*    dAcol = NULL;
    double* dAval = NULL;
    double* dx    = NULL;
    double* dy    = NULL;

    HIP_CHECK(hipMalloc((void**)&dAptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(dAptr, hAptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice));

    // Convert CSR matrix to HYB format
    hipsparseHybMat_t hybA;
    HIPSPARSE_CHECK(hipsparseCreateHybMat(&hybA));

    HIPSPARSE_CHECK(hipsparseDcsr2hyb(
        handle, m, n, descrA, dAval, dAptr, dAcol, hybA, 0, HIPSPARSE_HYB_PARTITION_AUTO));

    // Clean up CSR structures
    HIP_CHECK(hipFree(dAptr));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));

    // Call hipsparse hybmv
    HIPSPARSE_CHECK(hipsparseDhybmv(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &halpha, descrA, hybA, dx, &hbeta, dy));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear up on device
    HIPSPARSE_CHECK(hipsparseDestroyHybMat(hybA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrA));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    return 0;
}
//! [doc example]
