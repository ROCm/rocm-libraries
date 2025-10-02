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
    //     1 0 0 0
    // A = 4 2 0 4
    //     0 3 7 0
    //     9 0 0 1
    int m   = 4;
    int n   = 4;
    int nnz = 8;

    std::vector<int>   hcsrRowPtrA = {0, 1, 4, 6, 8};
    std::vector<int>   hcsrColIndA = {0, 0, 1, 3, 1, 2, 0, 3};
    std::vector<float> hcsrValA    = {1.0f, 4.0f, 2.0f, 4.0f, 3.0f, 7.0f, 9.0f, 1.0f};

    int*   dcsrRowPtrA;
    int*   dcsrColIndA;
    float* dcsrValA;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrA, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndA, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrValA, sizeof(float) * nnz));

    HIP_CHECK(
        hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndA, hcsrColIndA.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValA, hcsrValA.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    float* ddenseB;
    HIP_CHECK(hipMalloc((void**)&ddenseB, sizeof(float) * m * n));

    hipsparseHandle_t     handle;
    hipsparseSpMatDescr_t matA;
    hipsparseDnMatDescr_t matB;

    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    hipsparseIndexType_t rowIdxTypeA = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t colIdxTypeA = HIPSPARSE_INDEX_32I;
    hipDataType          dataTypeA   = HIP_R_32F;
    hipsparseIndexBase_t idxBaseA    = HIPSPARSE_INDEX_BASE_ZERO;

    // Create sparse matrix A
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA,
                                       m,
                                       n,
                                       nnz,
                                       dcsrRowPtrA,
                                       dcsrColIndA,
                                       dcsrValA,
                                       rowIdxTypeA,
                                       colIdxTypeA,
                                       idxBaseA,
                                       dataTypeA));

    // Create dense matrix B
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matB, m, n, m, ddenseB, HIP_R_32F, HIPSPARSE_ORDER_COL));

    hipsparseSparseToDenseAlg_t alg = HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseSparseToDense_bufferSize(handle, matA, matB, alg, &bufferSize));

    void* tempBuffer;
    HIP_CHECK(hipMalloc((void**)&tempBuffer, bufferSize));

    // Complete the conversion
    HIPSPARSE_CHECK(hipsparseSparseToDense(handle, matA, matB, alg, tempBuffer));

    // Copy result back to host
    std::vector<float> hdenseB(m * n);
    HIP_CHECK(hipMemcpy(hdenseB.data(), ddenseB, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matB));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsrRowPtrA));
    HIP_CHECK(hipFree(dcsrColIndA));
    HIP_CHECK(hipFree(dcsrValA));
    HIP_CHECK(hipFree(ddenseB));
    HIP_CHECK(hipFree(tempBuffer));

    return 0;
}
//! [doc example]