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

    // A sample square matrix A (4x4) in BSR format for ILU(0) factorization.
    // The 'S' in Sbsrilu02 indicates single precision float.
    // We'll use a block size of 1 for simplicity, making it behave like CSR ILU.
    // Matrix A:
    // ( 1  2  0  0 )
    // ( 3  4  5  0 )
    // ( 0  6  7  8 )
    // ( 0  0  9 10 )

    int m    = 4; // Number of rows
    int n    = 4; // Number of columns
    int bs   = 1; // Block size
    int mb   = m / bs; // Number of block rows
    int nb   = n / bs; // Number of block columns
    int nnzb = 10; // Number of non-zero blocks

    // BSR row pointers
    int hbsrRowPtr[5] = {0, 2, 5, 8, 10};

    // BSR column indices
    int hbsrColInd[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

    // BSR values (single precision float)
    float hbsrVal[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Set index base on descriptor
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    // For ILU(0), the L factor often has a unit diagonal.
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT));

    // BSRILU02 info
    bsrilu02Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateBsrilu02Info(&info));

    // Offload data to device
    int*   dbsrRowPtr;
    int*   dbsrColInd;
    float* dbsrVal; // This will store the factorized L and U values

    HIP_CHECK(hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsrVal, sizeof(float) * nnzb * bs * bs));

    HIP_CHECK(hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrVal, hbsrVal, sizeof(float) * nnzb * bs * bs, hipMemcpyHostToDevice));

    // 1. Get buffer size
    int bufferSize = 0;
    HIPSPARSE_CHECK(
        hipsparseSbsrilu02_bufferSize(handle,
                                      HIPSPARSE_DIRECTION_COLUMN, // Block storage direction
                                      mb,
                                      nnzb,
                                      descr,
                                      dbsrVal,
                                      dbsrRowPtr,
                                      dbsrColInd,
                                      bs,
                                      info,
                                      &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    // 2. Perform analysis (symbolic factorization)
    // This step analyzes the sparsity pattern of A to determine the structure of L and U.
    HIPSPARSE_CHECK(
        hipsparseSbsrilu02_analysis(handle,
                                    HIPSPARSE_DIRECTION_COLUMN,
                                    mb,
                                    nnzb,
                                    descr,
                                    dbsrVal,
                                    dbsrRowPtr,
                                    dbsrColInd,
                                    bs,
                                    info,
                                    HIPSPARSE_SOLVE_POLICY_USE_LEVEL, // Policy for analysis
                                    dbuffer));

    // 3. Perform factorization (numerical computation)
    // This step computes the actual numerical values of L and U, stored in dbsrVal.
    HIPSPARSE_CHECK(hipsparseSbsrilu02(handle,
                                       HIPSPARSE_DIRECTION_COLUMN,
                                       mb,
                                       nnzb,
                                       descr,
                                       dbsrVal,
                                       dbsrRowPtr,
                                       dbsrColInd,
                                       bs,
                                       info,
                                       HIPSPARSE_SOLVE_POLICY_USE_LEVEL, // Policy for factorization
                                       dbuffer));

    // 4. Check for zero pivots
    // A zero pivot can occur during factorization, indicating a numerical breakdown.
    int zeroPivot = 0; // -1 if no zero pivot, otherwise the block row index of the first zero pivot
    HIPSPARSE_CHECK(hipsparseXbsrilu02_zeroPivot(handle, info, &zeroPivot));
    if(zeroPivot != -1)
    {
        printf("Error: Zero pivot detected during ILU0 factorization at block row index %d\n",
               zeroPivot);
        // Handle the error (e.g., return, use a different preconditioner, etc.)
    }
    else
    {
        printf("BSRILU0 factorization completed successfully (no zero pivots detected).\n");
    }

    // Copy the factorized values (L and U combined) back to host
    float* hbsrVal_result = new float[nnzb * bs * bs];
    HIP_CHECK(
        hipMemcpy(hbsrVal_result, dbsrVal, sizeof(float) * nnzb * bs * bs, hipMemcpyDeviceToHost));

    // Print the result (the values of the factorized L and U combined)
    printf("\nFactorized BSR values (L and U combined):\n");
    for(int i = 0; i < nnzb * bs * bs; ++i)
    {
        printf("val[%d] = %f\n", i, hbsrVal_result[i]);
    }

    // Clean up
    delete[] hbsrVal_result;

    HIPSPARSE_CHECK(hipsparseDestroyBsrilu02Info(info));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
