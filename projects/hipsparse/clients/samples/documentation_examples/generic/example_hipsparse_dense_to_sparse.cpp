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
    int m = 4;
    int n = 4;

    std::vector<float> hdenseA = {1.0f,
                                  4.0f,
                                  0.0f,
                                  9.0f,
                                  0.0f,
                                  2.0f,
                                  3.0f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  7.0f,
                                  0.0f,
                                  0.0f,
                                  4.0f,
                                  0.0f,
                                  1.0f};

    float* ddenseA;
    HIP_CHECK(hipMalloc((void**)&ddenseA, sizeof(float) * m * n));
    HIP_CHECK(hipMemcpy(ddenseA, hdenseA.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    int* dcsrRowPtrB;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrB, sizeof(int) * (m + 1)));

    hipsparseHandle_t     handle;
    hipsparseDnMatDescr_t matA;
    hipsparseSpMatDescr_t matB;

    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Create dense matrix A
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matA, m, n, m, ddenseA, HIP_R_32F, HIPSPARSE_ORDER_COL));

    hipsparseIndexType_t rowIdxTypeB = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t colIdxTypeB = HIPSPARSE_INDEX_32I;
    hipDataType          dataTypeB   = HIP_R_32F;
    hipsparseIndexBase_t idxBaseB    = HIPSPARSE_INDEX_BASE_ZERO;

    // Create sparse matrix B
    HIPSPARSE_CHECK(hipsparseCreateCsr(
        &matB, m, n, 0, dcsrRowPtrB, NULL, NULL, rowIdxTypeB, colIdxTypeB, idxBaseB, dataTypeB));

    hipsparseDenseToSparseAlg_t alg = HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseDenseToSparse_bufferSize(handle, matA, matB, alg, &bufferSize));

    void* tempBuffer;
    HIP_CHECK(hipMalloc((void**)&tempBuffer, bufferSize));

    // Perform analysis which will determine the number of non-zeros in the CSR matrix
    HIPSPARSE_CHECK(hipsparseDenseToSparse_analysis(handle, matA, matB, alg, tempBuffer));

    // Grab the non-zero count from the B matrix decriptor
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    HIPSPARSE_CHECK(hipsparseSpMatGetSize(matB, &rows, &cols, &nnz));

    // Allocate the column indices and values arrays
    int*   dcsrColIndB;
    float* dcsrValB;
    HIP_CHECK(hipMalloc((void**)&dcsrColIndB, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrValB, sizeof(float) * nnz));

    // Set the newly allocated arrays on the sparse matrix descriptor
    HIPSPARSE_CHECK(hipsparseCsrSetPointers(matB, dcsrRowPtrB, dcsrColIndB, dcsrValB));

    // Complete the conversion
    HIPSPARSE_CHECK(hipsparseDenseToSparse_convert(handle, matA, matB, alg, tempBuffer));

    // Copy result back to host
    std::vector<int>   hcsrRowPtrB(m + 1);
    std::vector<int>   hcsrColIndB(nnz);
    std::vector<float> hcsrValB(nnz);
    HIP_CHECK(
        hipMemcpy(hcsrRowPtrB.data(), dcsrRowPtrB, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrColIndB.data(), dcsrColIndB, sizeof(int) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsrValB.data(), dcsrValB, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matB));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(ddenseA));
    HIP_CHECK(hipFree(dcsrRowPtrB));
    HIP_CHECK(hipFree(dcsrColIndB));
    HIP_CHECK(hipFree(dcsrValB));
    HIP_CHECK(hipFree(tempBuffer));

    return 0;
}
//! [doc example]