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

#include "utility.hpp"

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

    // alpha * ( 1.0  0.0  2.0  0.0 ) * ( x_0 ) = ( 32.0 )
    //         ( 3.0  2.0  4.0  1.0 ) * ( x_1 ) = ( 14.7 )
    //         ( 5.0  6.0  1.0  3.0 ) * ( x_2 ) = ( 33.6 )
    //         ( 7.0  0.0  8.0  0.6 ) * ( x_3 ) = ( 10.0 )

    const int m   = 4;
    const int nnz = 13;

    // CSR row pointers
    int hcsrRowPtr[m + 1] = {0, 2, 6, 10, 13};

    // CSR column indices
    int hcsrColInd[nnz] = {0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 3};

    // CSR values
    double hcsrVal[nnz] = {1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 5.0, 6.0, 1.0, 3.0, 7.0, 8.0, 0.6};

    // Transposition of the matrix
    hipsparseOperation_t   trans  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

    // Scalar alpha
    double alpha = 1.0;

    // f and x
    double hf[m] = {32.0, 14.7, 33.6, 10.0};
    double hx[m];

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Set index base on descriptor
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    // Set fill mode on descriptor
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER));

    // Set diag type on descriptor
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT));

    // Csrsv info
    csrsv2Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&info));

    // Offload data to device
    int*    dcsrRowPtr;
    int*    dcsrColInd;
    double* dcsrVal;
    double* df;
    double* dx;

    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&df, sizeof(double) * m));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(df, hf, sizeof(double) * m, hipMemcpyHostToDevice));

    int bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(
        handle, trans, m, nnz, descr, dcsrVal, dcsrRowPtr, dcsrColInd, info, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(
        handle, trans, m, nnz, descr, dcsrVal, dcsrRowPtr, dcsrColInd, info, policy, dbuffer));

    // Call dcsrsv to perform alpha * A * x = f
    HIPSPARSE_CHECK(hipsparseDcsrsv2_solve(handle,
                                           trans,
                                           m,
                                           nnz,
                                           &alpha,
                                           descr,
                                           dcsrVal,
                                           dcsrRowPtr,
                                           dcsrColInd,
                                           info,
                                           df,
                                           dx,
                                           policy,
                                           dbuffer));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hx, dx, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "hx" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hx[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(info));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(df));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
