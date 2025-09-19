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

    hipsparseMatDescr_t descrA;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrA));

    hipsparseMatDescr_t descrC;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrC));

    // Sparse matrix in BSR format
    //     1 2 | 0 3 | 0 0
    //     0 4 | 5 0 | 0 1
    // A = 6 0 | 0 7 | 8 0
    //     ---------------
    //     0 0 | 3 0 | 2 2
    //     1 0 | 0 0 | 4 3
    //     7 2 | 0 0 | 1 4
    int   hbsrRowPtrA[3] = {0, 3, 6};
    int   hbsrColIndA[6] = {0, 1, 2, 0, 1, 2};
    float hbsrValA[36]   = {1.0f, 2.0f, 0.0f, 4.0f, 6.0f, 0.0f, 0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 7.0f,
                          0.0f, 0.0f, 0.0f, 1.0f, 8.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 7.0f, 2.0f,
                          3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 4.0f, 3.0f, 1.0f, 4.0f};

    int                  m            = 6;
    int                  n            = 6;
    int                  rowBlockDimA = 3;
    int                  colBlockDimA = 2;
    int                  rowBlockDimC = 2;
    int                  colBlockDimC = 2;
    hipsparseDirection_t dirA         = HIPSPARSE_DIRECTION_ROW;

    int mbA   = (m + rowBlockDimA - 1) / rowBlockDimA;
    int nbA   = (n + colBlockDimA - 1) / colBlockDimA;
    int nnzbA = 6;

    int mbC = (m + rowBlockDimC - 1) / rowBlockDimC;
    int nbC = (n + colBlockDimC - 1) / colBlockDimC;

    int*   dbsrRowPtrA = nullptr;
    int*   dbsrColIndA = nullptr;
    float* dbsrValA    = nullptr;

    HIP_CHECK(hipMalloc((void**)&dbsrRowPtrA, sizeof(int) * (mbA + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsrColIndA, sizeof(int) * nnzbA));
    HIP_CHECK(hipMalloc((void**)&dbsrValA, sizeof(float) * rowBlockDimA * colBlockDimA * nnzbA));

    HIP_CHECK(hipMemcpy(dbsrRowPtrA, hbsrRowPtrA, sizeof(int) * (mbA + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrColIndA, hbsrColIndA, sizeof(int) * nnzbA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrValA,
                        hbsrValA,
                        sizeof(float) * rowBlockDimA * colBlockDimA * nnzbA,
                        hipMemcpyHostToDevice));

    int* dbsrRowPtrC = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsrRowPtrC, sizeof(int) * (mbC + 1)));

    int bufferSize;
    HIPSPARSE_CHECK(hipsparseSgebsr2gebsr_bufferSize(handle,
                                                     dirA,
                                                     mbA,
                                                     nbA,
                                                     nnzbA,
                                                     descrA,
                                                     dbsrValA,
                                                     dbsrRowPtrA,
                                                     dbsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int nnzbC;
    HIPSPARSE_CHECK(hipsparseXgebsr2gebsrNnz(handle,
                                             dirA,
                                             mbA,
                                             nbA,
                                             nnzbA,
                                             descrA,
                                             dbsrRowPtrA,
                                             dbsrColIndA,
                                             rowBlockDimA,
                                             colBlockDimA,
                                             descrC,
                                             dbsrRowPtrC,
                                             rowBlockDimC,
                                             colBlockDimC,
                                             &nnzbC,
                                             dbuffer));

    HIP_CHECK(hipDeviceSynchronize());

    int*   dbsrColIndC = nullptr;
    float* dbsrValC    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsrColIndC, sizeof(int) * nnzbC));
    HIP_CHECK(hipMalloc((void**)&dbsrValC, sizeof(float) * rowBlockDimC * colBlockDimC * nnzbC));

    HIPSPARSE_CHECK(hipsparseSgebsr2gebsr(handle,
                                          dirA,
                                          mbA,
                                          nbA,
                                          nnzbA,
                                          descrA,
                                          dbsrValA,
                                          dbsrRowPtrA,
                                          dbsrColIndA,
                                          rowBlockDimA,
                                          colBlockDimA,
                                          descrC,
                                          dbsrValC,
                                          dbsrRowPtrC,
                                          dbsrColIndC,
                                          rowBlockDimC,
                                          colBlockDimC,
                                          dbuffer));

    HIP_CHECK(hipFree(dbsrRowPtrA));
    HIP_CHECK(hipFree(dbsrColIndA));
    HIP_CHECK(hipFree(dbsrValA));

    HIP_CHECK(hipFree(dbsrRowPtrC));
    HIP_CHECK(hipFree(dbsrColIndC));
    HIP_CHECK(hipFree(dbsrValC));

    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]