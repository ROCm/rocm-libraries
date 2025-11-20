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
    int m    = 4;
    int k    = 3;
    int n    = 2;
    int nnzA = 7;
    int nnzB = 3;
    int nnzD = 6;

    float alpha = 1.0;
    float beta  = 1.0;

    // A, B, and C are mxk, kxn, and m×n

    // A
    // 1 0 0
    // 3 4 0
    // 5 6 7
    // 0 0 9
    int   hcsrRowPtrA[] = {0, 1, 3, 6, 7};
    int   hcsrColIndA[] = {0, 0, 1, 0, 1, 2, 2};
    float hcsrValA[]    = {1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0};

    // B
    // 0 1
    // 1 0
    // 0 1
    int   hcsrRowPtrB[] = {0, 1, 2, 3};
    int   hcsrColIndB[] = {1, 0, 1};
    float hcsrValB[]    = {1.0, 1.0, 1.0};

    // D
    // 0 1
    // 2 3
    // 4 5
    // 0 6
    int   hcsrRowPtrD[] = {0, 1, 3, 5, 6};
    int   hcsrColIndD[] = {1, 0, 1, 0, 1, 1};
    float hcsrValD[]    = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Device memory management: Allocate and copy A, B
    int*   dcsrRowPtrA;
    int*   dcsrColIndA;
    float* dcsrValA;
    int*   dcsrRowPtrB;
    int*   dcsrColIndB;
    float* dcsrValB;
    int*   dcsrRowPtrD;
    int*   dcsrColIndD;
    float* dcsrValD;
    int*   dcsrRowPtrC;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValA, nnzA * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrB, (k + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValB, nnzB * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrD, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndD, nnzD * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValD, nnzD * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int)));

    HIP_CHECK(hipMemcpy(dcsrRowPtrA, hcsrRowPtrA, (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndA, hcsrColIndA, nnzA * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValA, hcsrValA, nnzA * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrRowPtrB, hcsrRowPtrB, (k + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndB, hcsrColIndB, nnzB * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValB, hcsrValB, nnzB * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrRowPtrD, hcsrRowPtrD, (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndD, hcsrColIndD, nnzD * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValD, hcsrValD, nnzD * sizeof(float), hipMemcpyHostToDevice));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    hipsparseMatDescr_t descrA;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrA));

    hipsparseMatDescr_t descrB;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrB));

    hipsparseMatDescr_t descrC;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrC));

    hipsparseMatDescr_t descrD;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrD));

    csrgemm2Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateCsrgemm2Info(&info));

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseScsrgemm2_bufferSizeExt(handle,
                                                     m,
                                                     n,
                                                     k,
                                                     &alpha,
                                                     descrA,
                                                     nnzA,
                                                     dcsrRowPtrA,
                                                     dcsrColIndA,
                                                     descrB,
                                                     nnzB,
                                                     dcsrRowPtrB,
                                                     dcsrColIndB,
                                                     &beta,
                                                     descrD,
                                                     nnzD,
                                                     dcsrRowPtrD,
                                                     dcsrColIndD,
                                                     info,
                                                     &bufferSize));

    void* dbuffer = NULL;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int nnzC;
    HIPSPARSE_CHECK(hipsparseXcsrgemm2Nnz(handle,
                                          m,
                                          n,
                                          k,
                                          descrA,
                                          nnzA,
                                          dcsrRowPtrA,
                                          dcsrColIndA,
                                          descrB,
                                          nnzB,
                                          dcsrRowPtrB,
                                          dcsrColIndB,
                                          descrD,
                                          nnzD,
                                          dcsrRowPtrD,
                                          dcsrColIndD,
                                          descrC,
                                          dcsrRowPtrC,
                                          &nnzC,
                                          info,
                                          dbuffer));

    int*   dcsrColIndC = NULL;
    float* dcsrValC    = NULL;
    HIP_CHECK(hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC));
    HIP_CHECK(hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC));

    HIPSPARSE_CHECK(hipsparseScsrgemm2(handle,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       descrA,
                                       nnzA,
                                       dcsrValA,
                                       dcsrRowPtrA,
                                       dcsrColIndA,
                                       descrB,
                                       nnzB,
                                       dcsrValB,
                                       dcsrRowPtrB,
                                       dcsrColIndB,
                                       &beta,
                                       descrD,
                                       nnzD,
                                       dcsrValD,
                                       dcsrRowPtrD,
                                       dcsrColIndD,
                                       descrC,
                                       dcsrValC,
                                       dcsrRowPtrC,
                                       dcsrColIndC,
                                       info,
                                       dbuffer));

    int*  hcsrRowPtrC = (int*)malloc((m + 1) * sizeof(int));
    int*  hcsrColIndC = (int*)malloc((nnzC) * sizeof(int));
    float hcsrValC[nnzC];

    // Copy back to the host
    HIP_CHECK(hipMemcpy(hcsrRowPtrC, dcsrRowPtrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrColIndC, dcsrColIndC, sizeof(int) * nnzC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrValC, dcsrValC, sizeof(float) * nnzC, hipMemcpyDeviceToHost));

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

    HIP_CHECK(hipFree(dcsrRowPtrA));
    HIP_CHECK(hipFree(dcsrColIndA));
    HIP_CHECK(hipFree(dcsrValA));
    HIP_CHECK(hipFree(dcsrRowPtrB));
    HIP_CHECK(hipFree(dcsrColIndB));
    HIP_CHECK(hipFree(dcsrValB));
    HIP_CHECK(hipFree(dcsrRowPtrC));
    HIP_CHECK(hipFree(dcsrColIndC));
    HIP_CHECK(hipFree(dcsrValC));
    HIP_CHECK(hipFree(dcsrRowPtrD));
    HIP_CHECK(hipFree(dcsrColIndD));
    HIP_CHECK(hipFree(dcsrValD));

    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrB));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrC));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrD));
    HIPSPARSE_CHECK(hipsparseDestroyCsrgemm2Info(info));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
/*! [doc example end] */
