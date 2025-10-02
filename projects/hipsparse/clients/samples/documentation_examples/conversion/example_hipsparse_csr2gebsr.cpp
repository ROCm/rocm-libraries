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

    hipsparseMatDescr_t csr_descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&csr_descr));

    hipsparseMatDescr_t bsr_descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&bsr_descr));

    // Sparse matrix in CSR format
    //     1 2 0 3 0 0
    //     0 4 5 0 0 1
    // A = 6 0 0 7 8 0
    //     0 0 3 0 2 2
    //     1 0 0 0 4 3
    //     7 2 0 0 1 4
    int   hcsrRowPtr[7]  = {0, 3, 6, 9, 12, 15, 19};
    int   hcsrColInd[19] = {0, 1, 3, 1, 2, 5, 0, 3, 4, 2, 4, 5, 0, 4, 5, 0, 1, 4, 5};
    float hcsrVal[19]    = {1.0f,
                         2.0f,
                         3.0f,
                         4.0f,
                         5.0f,
                         1.0f,
                         6.0f,
                         7.0f,
                         8.0f,
                         3.0f,
                         2.0f,
                         2.0f,
                         1.0f,
                         4.0f,
                         3.0f,
                         7.0f,
                         2.0f,
                         1.0f,
                         4.0f};

    int                  m           = 6;
    int                  n           = 6;
    int                  nnz         = 19;
    int                  rowBlockDim = 3;
    int                  colBlockDim = 2;
    hipsparseDirection_t dir         = HIPSPARSE_DIRECTION_ROW;
    hipsparseIndexBase_t base        = HIPSPARSE_INDEX_BASE_ZERO;

    int mb = (m + rowBlockDim - 1) / rowBlockDim;
    int nb = (n + colBlockDim - 1) / colBlockDim;

    int*   dcsrRowPtr = nullptr;
    int*   dcsrColInd = nullptr;
    float* dcsrVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice));

    int* dbsrRowPtr = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1)));

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseScsr2gebsr_bufferSize(handle,
                                                   dir,
                                                   m,
                                                   n,
                                                   csr_descr,
                                                   dcsrVal,
                                                   dcsrRowPtr,
                                                   dcsrColInd,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int nnzb;
    HIPSPARSE_CHECK(hipsparseXcsr2gebsrNnz(handle,
                                           dir,
                                           m,
                                           n,
                                           csr_descr,
                                           dcsrRowPtr,
                                           dcsrColInd,
                                           bsr_descr,
                                           dbsrRowPtr,
                                           rowBlockDim,
                                           colBlockDim,
                                           &nnzb,
                                           dbuffer));

    int*   dbsrColInd = nullptr;
    float* dbsrVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb));

    HIPSPARSE_CHECK(hipsparseScsr2gebsr(handle,
                                        dir,
                                        m,
                                        n,
                                        csr_descr,
                                        dcsrVal,
                                        dcsrRowPtr,
                                        dcsrColInd,
                                        bsr_descr,
                                        dbsrVal,
                                        dbsrRowPtr,
                                        dbsrColInd,
                                        rowBlockDim,
                                        colBlockDim,
                                        dbuffer));

    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));

    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));

    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(csr_descr));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(bsr_descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]