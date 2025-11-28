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

    hipsparseSpMatDescr_t matA, matB, matC;

    hipsparseHandle_t handle;
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

    void*  dBuffer1    = NULL;
    void*  dBuffer2    = NULL;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;

    // Determine size of first user allocated buffer
    HIPSPARSE_CHECK(hipsparseSpGEMM_workEstimation(handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   computeType,
                                                   HIPSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc,
                                                   &bufferSize1,
                                                   NULL));
    HIP_CHECK(hipMalloc((void**)&dBuffer1, bufferSize1));

    // Inspect the matrices A and B to determine the number of intermediate product in
    // C = alpha * A * B
    HIPSPARSE_CHECK(hipsparseSpGEMM_workEstimation(handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   computeType,
                                                   HIPSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc,
                                                   &bufferSize1,
                                                   dBuffer1));

    // Determine size of second user allocated buffer
    HIPSPARSE_CHECK(hipsparseSpGEMM_compute(handle,
                                            opA,
                                            opB,
                                            &alpha,
                                            matA,
                                            matB,
                                            &beta,
                                            matC,
                                            computeType,
                                            HIPSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc,
                                            &bufferSize2,
                                            NULL));
    HIP_CHECK(hipMalloc((void**)&dBuffer2, bufferSize2));

    // Compute C = alpha * A * B and store result in temporary buffers
    HIPSPARSE_CHECK(hipsparseSpGEMM_compute(handle,
                                            opA,
                                            opB,
                                            &alpha,
                                            matA,
                                            matB,
                                            &beta,
                                            matC,
                                            computeType,
                                            HIPSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc,
                                            &bufferSize2,
                                            dBuffer2));

    // Get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    HIPSPARSE_CHECK(hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1));

    // Allocate the CSR structures for the matrix C
    HIP_CHECK(hipMalloc((void**)&dcsrColIndC, C_nnz1 * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValC, C_nnz1 * sizeof(float)));

    // Update matC with the new pointers
    HIPSPARSE_CHECK(hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC));

    // Copy the final products to the matrix C
    HIPSPARSE_CHECK(hipsparseSpGEMM_copy(handle,
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

    int*  hcsrRowPtrC = (int*)malloc((m + 1) * sizeof(int));
    int*  hcsrColIndC = (int*)malloc((C_nnz1) * sizeof(int));
    float hcsrValC[C_nnz1];

    // Copy back to the host
    HIP_CHECK(hipMemcpy(hcsrRowPtrC, dcsrRowPtrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrColIndC, dcsrColIndC, sizeof(int) * C_nnz1, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrValC, dcsrValC, sizeof(float) * C_nnz1, hipMemcpyDeviceToHost));

    printf("C\n");
    for(int i = 0; i < m; i++)
    {
        int start = hcsrRowPtrC[i];
        int end   = hcsrRowPtrC[i + 1];

        float* temp = (float*)malloc(n * sizeof(float));
        for(int j = start; j < end; j++)
        {
            temp[hcsrColIndC[j]] = hcsrValC[j];
        }

        for(int j = 0; j < n; j++)
        {
            printf("%f ", temp[j]);
        }
        printf("\n");
    }
    printf("\n");

    // Destroy matrix descriptors and handles
    HIPSPARSE_CHECK(hipsparseSpGEMM_destroyDescr(spgemmDesc));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matB));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Free device memory
    HIP_CHECK(hipFree(dBuffer1));
    HIP_CHECK(hipFree(dBuffer2));

    return 0;
}
/*! [doc example end] */