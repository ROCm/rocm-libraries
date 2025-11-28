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

#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include <stdio.h>

#define HIP_CHECK(stat)                                                 \
    {                                                                   \
        if(stat != hipSuccess)                                          \
        {                                                               \
            fprintf(stderr, "Error: hip error in line %d\n", __LINE__); \
            return -1;                                                  \
        }                                                               \
    }

#define HIPSPARSE_CHECK(stat)                                                 \
    {                                                                         \
        if(stat != HIPSPARSE_STATUS_SUCCESS)                                  \
        {                                                                     \
            fprintf(stderr, "Error: hipsparse error in line %d\n", __LINE__); \
            return -1;                                                        \
        }                                                                     \
    }

/*! [doc example start] */
int main(int argc, char* argv[])
{
    int m    = 2;
    int k    = 2;
    int n    = 3;
    int nnzA = 4;
    int nnzB = 4;

    float alpha = 1.0;
    float beta  = 0.0;

    hipsparseOperation_t opA         = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t opB         = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipDataType          computeType = HIP_R_32F;

    // A, B, and C are m×k, k×n, and m×n

    // A
    int   hcsrRowPtrA[] = {0, 2, 4};
    int   hcsrColIndA[] = {0, 1, 0, 1};
    float hcsrValA[]    = {1.0, 2.0, 3.0, 4.0};

    // B
    int   hcsrRowPtrB[] = {0, 2, 4};
    int   hcsrColIndB[] = {1, 2, 0, 2};
    float hcsrValB[]    = {5.0, 6.0, 7.0, 8.0};

    // Device memory management: Allocate and copy A, B
    int*   dcsrRowPtrA;
    int*   dcsrColIndA;
    float* dcsrValA;
    int*   dcsrRowPtrB;
    int*   dcsrColIndB;
    float* dcsrValB;
    int*   dcsrRowPtrC;
    int*   dcsrColIndC;
    float* dcsrValC;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValA, nnzA * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrB, (k + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValB, nnzB * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int)));

    HIP_CHECK(hipMemcpy(dcsrRowPtrA, hcsrRowPtrA, (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndA, hcsrColIndA, nnzA * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValA, hcsrValA, nnzA * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrRowPtrB, hcsrRowPtrB, (k + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndB, hcsrColIndB, nnzB * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValB, hcsrValB, nnzB * sizeof(float), hipMemcpyHostToDevice));

    hipsparseHandle_t     handle = NULL;
    hipsparseSpMatDescr_t matA, matB, matC;
    void*                 dBuffer1    = NULL;
    void*                 dBuffer2    = NULL;
    void*                 dBuffer3    = NULL;
    void*                 dBuffer4    = NULL;
    void*                 dBuffer5    = NULL;
    size_t                bufferSize1 = 0;
    size_t                bufferSize2 = 0;
    size_t                bufferSize3 = 0;
    size_t                bufferSize4 = 0;
    size_t                bufferSize5 = 0;

    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Create sparse matrix A in CSR format
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA,
                                       m,
                                       k,
                                       nnzA,
                                       dcsrRowPtrA,
                                       dcsrColIndA,
                                       dcsrValA,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO,
                                       HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matB,
                                       k,
                                       n,
                                       nnzB,
                                       dcsrRowPtrB,
                                       dcsrColIndB,
                                       dcsrValB,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO,
                                       HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matC,
                                       m,
                                       n,
                                       0,
                                       dcsrRowPtrC,
                                       NULL,
                                       NULL,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO,
                                       HIP_R_32F));

    hipsparseSpGEMMDescr_t spgemmDesc;
    HIPSPARSE_CHECK(hipsparseSpGEMM_createDescr(&spgemmDesc));

    // Determine size of first user allocated buffer
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_workEstimation(handle,
                                                        opA,
                                                        opB,
                                                        matA,
                                                        matB,
                                                        matC,
                                                        HIPSPARSE_SPGEMM_DEFAULT,
                                                        spgemmDesc,
                                                        &bufferSize1,
                                                        NULL));

    HIP_CHECK(hipMalloc((void**)&dBuffer1, bufferSize1));

    // Inspect the matrices A and B to determine the number of intermediate product in
    // C = alpha * A * B
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_workEstimation(handle,
                                                        opA,
                                                        opB,
                                                        matA,
                                                        matB,
                                                        matC,
                                                        HIPSPARSE_SPGEMM_DEFAULT,
                                                        spgemmDesc,
                                                        &bufferSize1,
                                                        dBuffer1));

    // Determine size of second, third, and fourth user allocated buffer
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_nnz(handle,
                                             opA,
                                             opB,
                                             matA,
                                             matB,
                                             matC,
                                             HIPSPARSE_SPGEMM_DEFAULT,
                                             spgemmDesc,
                                             &bufferSize2,
                                             NULL,
                                             &bufferSize3,
                                             NULL,
                                             &bufferSize4,
                                             NULL));

    HIP_CHECK(hipMalloc((void**)&dBuffer2, bufferSize2));
    HIP_CHECK(hipMalloc((void**)&dBuffer3, bufferSize3));
    HIP_CHECK(hipMalloc((void**)&dBuffer4, bufferSize4));

    // Compute sparsity pattern of C matrix and store in temporary buffers
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_nnz(handle,
                                             opA,
                                             opB,
                                             matA,
                                             matB,
                                             matC,
                                             HIPSPARSE_SPGEMM_DEFAULT,
                                             spgemmDesc,
                                             &bufferSize2,
                                             dBuffer2,
                                             &bufferSize3,
                                             dBuffer3,
                                             &bufferSize4,
                                             dBuffer4));

    // We can now free buffer 1 and 2
    HIP_CHECK(hipFree(dBuffer1));
    HIP_CHECK(hipFree(dBuffer2));

    // Get matrix C non-zero entries nnzC
    int64_t rowsC, colsC, nnzC;
    HIPSPARSE_CHECK(hipsparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC));

    // Allocate matrix C
    HIP_CHECK(hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC));
    HIP_CHECK(hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC));

    // Update matC with the new pointers. The C values array can be filled with data here
    // which is used if beta != 0.
    HIPSPARSE_CHECK(hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC));

    // Determine size of fifth user allocated buffer
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_copy(handle,
                                              opA,
                                              opB,
                                              matA,
                                              matB,
                                              matC,
                                              HIPSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc,
                                              &bufferSize5,
                                              NULL));

    HIP_CHECK(hipMalloc((void**)&dBuffer5, bufferSize5));

    // Copy data from temporary buffers to the newly allocated C matrix
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_copy(handle,
                                              opA,
                                              opB,
                                              matA,
                                              matB,
                                              matC,
                                              HIPSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc,
                                              &bufferSize5,
                                              dBuffer5));

    // We can now free buffer 3
    HIP_CHECK(hipFree(dBuffer3));

    // Compute C' = alpha * A * B + beta * C
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_compute(handle,
                                                 opA,
                                                 opB,
                                                 &alpha,
                                                 matA,
                                                 matB,
                                                 &beta,
                                                 matC,
                                                 computeType,
                                                 HIPSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc));

    // Copy results back to host if required
    int*  hcsrRowPtrC = (int*)malloc((m + 1) * sizeof(int));
    int*  hcsrColIndC = (int*)malloc((nnzC) * sizeof(int));
    float hcsrValC[nnzC];
    HIP_CHECK(hipMemcpy(hcsrRowPtrC, dcsrRowPtrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrColIndC, dcsrColIndC, sizeof(int) * nnzC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrValC, dcsrValC, sizeof(float) * nnzC, hipMemcpyDeviceToHost));

    // Update dcsrValA, dcsrValB with new values
    for(size_t i = 0; i < nnzA; i++)
    {
        hcsrValA[i] = 1.0;
    }
    for(size_t i = 0; i < nnzB; i++)
    {
        hcsrValB[i] = 2.0;
    }

    HIP_CHECK(hipMemcpy(dcsrValA, hcsrValA, sizeof(float) * nnzA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValB, hcsrValB, sizeof(float) * nnzB, hipMemcpyHostToDevice));

    // Compute C' = alpha * A * B + beta * C again with the new A and B values
    HIPSPARSE_CHECK(hipsparseSpGEMMreuse_compute(handle,
                                                 opA,
                                                 opB,
                                                 &alpha,
                                                 matA,
                                                 matB,
                                                 &beta,
                                                 matC,
                                                 computeType,
                                                 HIPSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc));

    // Copy results back to host if required
    HIP_CHECK(hipMemcpy(hcsrRowPtrC, dcsrRowPtrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrColIndC, dcsrColIndC, sizeof(int) * nnzC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrValC, dcsrValC, sizeof(float) * nnzC, hipMemcpyDeviceToHost));

    // Destroy matrix descriptors and handles
    HIPSPARSE_CHECK(hipsparseSpGEMM_destroyDescr(spgemmDesc));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matB));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Free device memory
    HIP_CHECK(hipFree(dBuffer4));
    HIP_CHECK(hipFree(dBuffer5));
    HIP_CHECK(hipFree(dcsrRowPtrA));
    HIP_CHECK(hipFree(dcsrColIndA));
    HIP_CHECK(hipFree(dcsrValA));
    HIP_CHECK(hipFree(dcsrRowPtrB));
    HIP_CHECK(hipFree(dcsrColIndB));
    HIP_CHECK(hipFree(dcsrValB));
    HIP_CHECK(hipFree(dcsrRowPtrC));
    HIP_CHECK(hipFree(dcsrColIndC));
    HIP_CHECK(hipFree(dcsrValC));

    return 0;
}
/*! [doc example end] */