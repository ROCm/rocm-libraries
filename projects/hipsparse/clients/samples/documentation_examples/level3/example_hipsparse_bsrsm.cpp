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
    //
    // with bsr_dim = 2
    //
    //      -------------------
    //   = | 1.0 0.0 | 0.0 0.0 |
    //     | 2.0 3.0 | 0.0 0.0 |
    //      -------------------
    //     | 4.0 5.0 | 6.0 0.0 |
    //     | 7.0 0.0 | 8.0 9.0 |
    //      -------------------

    // Number of rows and columns
    const int m = 4;

    // Number of block rows and block columns
    const int mb = 2;
    const int nb = 2;

    // BSR block dimension
    const int bsr_dim = 2;

    // Number of right-hand-sides
    const int nrhs = 4;

    // Number of non-zero blocks
    const int nnzb = 3;

    // BSR row pointers
    int hbsrRowPtr[mb + 1] = {0, 1, 3};

    // BSR column indices
    int hbsrColInd[nnzb] = {0, 0, 1};

    // BSR values
    double hbsrVal[nnzb * bsr_dim * bsr_dim]
        = {1.0, 2.0, 0.0, 3.0, 4.0, 7.0, 5.0, 0.0, 6.0, 8.0, 0.0, 9.0};

    // Storage scheme of the BSR blocks
    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_COLUMN;

    // Transposition of the matrix and rhs matrix
    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transX = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // Solve policy
    hipsparseSolvePolicy_t solve_policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    // Scalar alpha and beta
    double alpha = 1.0;

    // rhs and solution matrix
    const int ldb = nb * bsr_dim;
    const int ldx = mb * bsr_dim;

    double hB[ldb * nrhs] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    double hX[ldx * nrhs];

    // Offload data to device
    int*    dbsrRowPtr;
    int*    dbsrColInd;
    double* dbsrVal;
    double* dB;
    double* dX;

    HIP_CHECK(hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsrVal, sizeof(double) * nnzb * bsr_dim * bsr_dim));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(double) * nb * bsr_dim * nrhs));
    HIP_CHECK(hipMalloc((void**)&dX, sizeof(double) * mb * bsr_dim * nrhs));

    HIP_CHECK(hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dbsrVal, hbsrVal, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(double) * nb * bsr_dim * nrhs, hipMemcpyHostToDevice));

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Matrix fill mode
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER));

    // Matrix diagonal type
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_NON_UNIT));

    // Matrix info structure
    bsrsm2Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateBsrsm2Info(&info));

    // Obtain required buffer size
    int buffer_size;
    HIPSPARSE_CHECK(hipsparseDbsrsm2_bufferSize(handle,
                                                dir,
                                                transA,
                                                transX,
                                                mb,
                                                nrhs,
                                                nnzb,
                                                descr,
                                                dbsrVal,
                                                dbsrRowPtr,
                                                dbsrColInd,
                                                bsr_dim,
                                                info,
                                                &buffer_size));

    // Allocate temporary buffer
    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));

    // Perform analysis step
    HIPSPARSE_CHECK(hipsparseDbsrsm2_analysis(handle,
                                              dir,
                                              transA,
                                              transX,
                                              mb,
                                              nrhs,
                                              nnzb,
                                              descr,
                                              dbsrVal,
                                              dbsrRowPtr,
                                              dbsrColInd,
                                              bsr_dim,
                                              info,
                                              solve_policy,
                                              dbuffer));

    // Call dbsrsm to perform lower triangular solve LX = B
    HIPSPARSE_CHECK(hipsparseDbsrsm2_solve(handle,
                                           dir,
                                           transA,
                                           transX,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           &alpha,
                                           descr,
                                           dbsrVal,
                                           dbsrRowPtr,
                                           dbsrColInd,
                                           bsr_dim,
                                           info,
                                           dB,
                                           ldb,
                                           dX,
                                           ldx,
                                           solve_policy,
                                           dbuffer));

    // Check for zero pivots
    int               pivot;
    hipsparseStatus_t status = hipsparseXbsrsm2_zeroPivot(handle, info, &pivot);

    if(status == HIPSPARSE_STATUS_ZERO_PIVOT)
    {
        std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
    }

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hX, dX, sizeof(double) * mb * bsr_dim * nrhs, hipMemcpyDeviceToHost));

    std::cout << "hX" << std::endl;
    for(int i = 0; i < ldx * nrhs; i++)
    {
        std::cout << hX[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroyBsrsm2Info(info));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
