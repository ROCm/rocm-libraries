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
#include <vector>

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
    const int m    = 4;
    const int n    = 4;
    const int nnzA = 9;
    const int nnzB = 6;

    float alpha{1.0f};
    float beta{1.0f};

    // A, B, and C are m×n

    // A
    // 1 0 0 2
    // 3 4 0 0
    // 5 6 7 8
    // 0 0 9 0
    std::vector<int>   hcsrRowPtrA = {0, 2, 4, 8, 9};
    std::vector<int>   hcsrColIndA = {0, 3, 0, 1, 0, 1, 2, 3, 2};
    std::vector<float> hcsrValA    = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // B
    // 0 1 0 0
    // 1 0 1 0
    // 0 1 0 1
    // 0 0 1 0
    std::vector<int>   hcsrRowPtrB = {0, 1, 3, 5, 6};
    std::vector<int>   hcsrColIndB = {1, 0, 2, 1, 3, 2};
    std::vector<float> hcsrValB    = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Device memory management: Allocate and copy A, B
    int*   dcsrRowPtrA;
    int*   dcsrColIndA;
    float* dcsrValA;
    int*   dcsrRowPtrB;
    int*   dcsrColIndB;
    float* dcsrValB;
    int*   dcsrRowPtrC;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValA, nnzA * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrB, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dcsrValB, nnzB * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int)));

    HIP_CHECK(
        hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsrColIndA, hcsrColIndA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValA, hcsrValA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsrRowPtrB, hcsrRowPtrB.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsrColIndB, hcsrColIndB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValB, hcsrValB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    hipsparseMatDescr_t descrA;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrA));

    hipsparseMatDescr_t descrB;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrB));

    hipsparseMatDescr_t descrC;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrC));

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseScsrgeam2_bufferSizeExt(handle,
                                                     m,
                                                     n,
                                                     &alpha,
                                                     descrA,
                                                     nnzA,
                                                     dcsrValA,
                                                     dcsrRowPtrA,
                                                     dcsrColIndA,
                                                     &beta,
                                                     descrB,
                                                     nnzB,
                                                     dcsrValB,
                                                     dcsrRowPtrB,
                                                     dcsrColIndB,
                                                     descrC,
                                                     nullptr,
                                                     dcsrRowPtrC,
                                                     nullptr,
                                                     &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int nnzC;
    HIPSPARSE_CHECK(hipsparseXcsrgeam2Nnz(handle,
                                          m,
                                          n,
                                          descrA,
                                          nnzA,
                                          dcsrRowPtrA,
                                          dcsrColIndA,
                                          descrB,
                                          nnzB,
                                          dcsrRowPtrB,
                                          dcsrColIndB,
                                          descrC,
                                          dcsrRowPtrC,
                                          &nnzC,
                                          dbuffer));

    int*   dcsrColIndC = nullptr;
    float* dcsrValC    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC));
    HIP_CHECK(hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC));

    HIPSPARSE_CHECK(hipsparseScsrgeam2(handle,
                                       m,
                                       n,
                                       &alpha,
                                       descrA,
                                       nnzA,
                                       dcsrValA,
                                       dcsrRowPtrA,
                                       dcsrColIndA,
                                       &beta,
                                       descrB,
                                       nnzB,
                                       dcsrValB,
                                       dcsrRowPtrB,
                                       dcsrColIndB,
                                       descrC,
                                       dcsrValC,
                                       dcsrRowPtrC,
                                       dcsrColIndC,
                                       dbuffer));

    std::vector<int>   hcsrRowPtrC(m + 1);
    std::vector<int>   hcsrColIndC(nnzC);
    std::vector<float> hcsrValC(nnzC);

    // Copy back to the host
    HIP_CHECK(
        hipMemcpy(hcsrRowPtrC.data(), dcsrRowPtrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsrColIndC.data(), dcsrColIndC, sizeof(int) * nnzC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrValC.data(), dcsrValC, sizeof(float) * nnzC, hipMemcpyDeviceToHost));

    std::cout << "C" << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = hcsrRowPtrC[i];
        int end   = hcsrRowPtrC[i + 1];

        std::vector<float> temp(n, 0.0f);
        for(int j = start; j < end; j++)
        {
            temp[hcsrColIndC[j]] = hcsrValC[j];
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;

    HIP_CHECK(hipFree(dcsrRowPtrA));
    HIP_CHECK(hipFree(dcsrColIndA));
    HIP_CHECK(hipFree(dcsrValA));
    HIP_CHECK(hipFree(dcsrRowPtrB));
    HIP_CHECK(hipFree(dcsrColIndB));
    HIP_CHECK(hipFree(dcsrValB));
    HIP_CHECK(hipFree(dcsrRowPtrC));
    HIP_CHECK(hipFree(dcsrColIndC));
    HIP_CHECK(hipFree(dcsrValC));

    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrB));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]
