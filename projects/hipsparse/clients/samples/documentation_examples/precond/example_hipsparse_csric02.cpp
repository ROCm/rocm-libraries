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

    // A sample symmetric positive definite matrix A (4x4) in CSR format.
    // The 'S' in Scsric02 indicates single precision float.
    // Matrix A:
    // ( 4  1  0  0 )
    // ( 1  5  2  0 )
    // ( 0  2  3  1 )
    // ( 0  0  1  2 )
    // This matrix is symmetric. For IC02, we typically provide the full matrix
    // or just the lower/upper part if using `HIPSPARSE_MATRIX_TYPE_SYMMETRIC`
    // with the descriptor. Here, we provide elements for both lower and upper parts
    // for simplicity, but the factorization will operate on the implicitly
    // symmetric matrix and produce the lower triangular factor L.

    int m   = 4; // Number of rows
    int n   = 4; // Number of columns (equal to m for Cholesky)
    int nnz = 10; // Number of non-zero elements (counting only one side for symmetric)

    // CSR row pointers
    int hcsrRowPtr[5] = {0, 2, 5, 8, 10};

    // CSR column indices
    // These indices correspond to the non-zero values used below.
    // For a symmetric matrix A, we implicitly work with A_lower.
    // The output will be L.
    int hcsrColInd[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

    // CSR values (single precision float for 'S'csric02)
    // The factorization computes the lower triangular L factor.
    // The input values represent the entries of A that correspond to the non-zero pattern.
    float hcsrVal[10] = {4.0f, 1.0f, 1.0f, 5.0f, 2.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f};

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Set index base on descriptor
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    // For incomplete Cholesky, the L factor is computed.
    // L is lower triangular with a unit diagonal.
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER));
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT));
    // Optionally set matrix type to symmetric if only storing one triangle of A
    // HIPSPARSE_CHECK(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_SYMMETRIC));

    // CSRIC02 info
    csric02Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateCsric02Info(&info));

    // Offload data to device
    int*   dcsrRowPtr;
    int*   dcsrColInd;
    float* dcsrVal; // This will store the factorized L values

    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    HIP_CHECK(
        hipMalloc((void**)&dcsrVal,
                  sizeof(float) * nnz)); // Note: Same size as input, values will be overwritten

    HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice));

    // 1. Get buffer size
    int bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseScsric02_bufferSize(
        handle, m, nnz, descr, dcsrVal, dcsrRowPtr, dcsrColInd, info, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    // 2. Perform analysis (symbolic factorization)
    // This step analyzes the sparsity pattern of A to determine the structure of L.
    HIPSPARSE_CHECK(
        hipsparseScsric02_analysis(handle,
                                   m,
                                   nnz,
                                   descr,
                                   dcsrVal,
                                   dcsrRowPtr,
                                   dcsrColInd,
                                   info,
                                   HIPSPARSE_SOLVE_POLICY_USE_LEVEL, // Policy for analysis
                                   dbuffer));

    // 3. Perform factorization (numerical computation)
    // This step computes the actual numerical values of L, stored in dcsrVal.
    HIPSPARSE_CHECK(hipsparseScsric02(handle,
                                      m,
                                      nnz,
                                      descr,
                                      dcsrVal,
                                      dcsrRowPtr,
                                      dcsrColInd,
                                      info,
                                      HIPSPARSE_SOLVE_POLICY_USE_LEVEL, // Policy for factorization
                                      dbuffer));

    // 4. Check for zero pivots
    // A zero pivot can occur during factorization, indicating a numerical breakdown.
    int zeroPivot = 0; // -1 if no zero pivot, otherwise the row index of the first zero pivot
    HIPSPARSE_CHECK(hipsparseXcsric02_zeroPivot(handle, info, &zeroPivot));
    if(zeroPivot != -1)
    {
        printf("Error: Zero pivot detected during IC02 factorization at row index %d\n", zeroPivot);
        // Depending on your application, you might want to handle this error
        // or switch to a different preconditioner.
    }
    else
    {
        printf("CSRIC02 factorization completed successfully (no zero pivots detected).\n");
    }

    // Copy the factorized values (L) back to host
    float* hcsrVal_result = new float[nnz];
    HIP_CHECK(hipMemcpy(hcsrVal_result, dcsrVal, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    // Print the result (the values of the factorized L)
    printf("\nFactorized CSR values (L factor):\n");
    for(int i = 0; i < nnz; ++i)
    {
        printf("val[%d] = %f\n", i, hcsrVal_result[i]);
    }

    // Clean up
    delete[] hcsrVal_result;

    HIPSPARSE_CHECK(hipsparseDestroyCsric02Info(info));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
