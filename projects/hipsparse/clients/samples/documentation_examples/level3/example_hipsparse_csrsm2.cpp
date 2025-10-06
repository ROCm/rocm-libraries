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

    // A = ( 1.0  0.0  0.0  0.0 )
    //     ( 2.0  3.0  0.0  0.0 )
    //     ( 4.0  5.0  6.0  0.0 )
    //     ( 7.0  0.0  8.0  9.0 )

    // Number of rows and columns
    int m = 4;
    int n = 4;

    // Number of right-hand-sides
    int nrhs = 4;

    // Number of non-zeros
    int nnz = 9;

    // CSR row pointers
    int hcsrRowPtr[5] = {0, 1, 3, 6, 9};

    // CSR column indices
    int hcsrColInd[9] = {0, 0, 1, 0, 1, 2, 0, 2, 3};

    // CSR values
    double hcsrVal[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Transposition of the matrix and rhs matrix
    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // Solve policy
    hipsparseSolvePolicy_t solve_policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    // Scalar alpha and beta
    double alpha = 1.0;

    // rhs and solution matrix
    int ldb = n;

    double hB[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // Offload data to device
    int*    dcsrRowPtr;
    int*    dcsrColInd;
    double* dcsrVal;
    double* dB;

    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(double) * n * nrhs));

    HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(double) * n * nrhs, hipMemcpyHostToDevice));

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Matrix fill mode
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER));

    // Matrix diagonal type
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_NON_UNIT));

    // Matrix info structure
    csrsm2Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateCsrsm2Info(&info));

    // Obtain required buffer size
    size_t buffer_size;
    HIPSPARSE_CHECK(hipsparseDcsrsm2_bufferSizeExt(handle,
                                                   0,
                                                   transA,
                                                   transB,
                                                   m,
                                                   nrhs,
                                                   nnz,
                                                   &alpha,
                                                   descr,
                                                   dcsrVal,
                                                   dcsrRowPtr,
                                                   dcsrColInd,
                                                   dB,
                                                   ldb,
                                                   info,
                                                   solve_policy,
                                                   &buffer_size));

    // Allocate temporary buffer
    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));

    // Perform analysis step
    HIPSPARSE_CHECK(hipsparseDcsrsm2_analysis(handle,
                                              0,
                                              transA,
                                              transB,
                                              m,
                                              nrhs,
                                              nnz,
                                              &alpha,
                                              descr,
                                              dcsrVal,
                                              dcsrRowPtr,
                                              dcsrColInd,
                                              dB,
                                              ldb,
                                              info,
                                              solve_policy,
                                              dbuffer));

    // Call dcsrsm to perform lower triangular solve LB = B
    HIPSPARSE_CHECK(hipsparseDcsrsm2_solve(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           dcsrVal,
                                           dcsrRowPtr,
                                           dcsrColInd,
                                           dB,
                                           ldb,
                                           info,
                                           solve_policy,
                                           dbuffer));

    // Check for zero pivots
    int               pivot;
    hipsparseStatus_t status = hipsparseXcsrsm2_zeroPivot(handle, info, &pivot);

    if(status == HIPSPARSE_STATUS_ZERO_PIVOT)
    {
        std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
    }

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hB, dB, sizeof(double) * m * nrhs, hipMemcpyDeviceToHost));

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroyCsrsm2Info(info));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]