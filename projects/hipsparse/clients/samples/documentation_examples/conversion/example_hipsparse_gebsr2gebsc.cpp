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

    // Sparse matrix in BSR format
    //     1 2 | 0 3 | 0 0
    //     0 4 | 5 0 | 0 1
    // A = 6 0 | 0 7 | 8 0
    //     ---------------
    //     0 0 | 3 0 | 2 2
    //     1 0 | 0 0 | 4 3
    //     7 2 | 0 0 | 1 4
    int   hbsrRowPtr[3] = {0, 3, 6};
    int   hbsrColInd[6] = {0, 1, 2, 0, 1, 2};
    float hbsrVal[36]   = {1.0f, 2.0f, 0.0f, 4.0f, 6.0f, 0.0f, 0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 7.0f,
                         0.0f, 0.0f, 0.0f, 1.0f, 8.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 7.0f, 2.0f,
                         3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 4.0f, 3.0f, 1.0f, 4.0f};

    int                  m           = 6;
    int                  n           = 6;
    int                  rowBlockDim = 3;
    int                  colBlockDim = 2;
    int                  nnzb        = 6;
    hipsparseDirection_t dir         = HIPSPARSE_DIRECTION_ROW;
    hipsparseAction_t    action      = HIPSPARSE_ACTION_NUMERIC;
    hipsparseIndexBase_t base        = HIPSPARSE_INDEX_BASE_ZERO;

    int mb = (m + rowBlockDim - 1) / rowBlockDim;
    int nb = (n + colBlockDim - 1) / colBlockDim;

    int*   dbsrRowPtr = nullptr;
    int*   dbsrColInd = nullptr;
    float* dbsrVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb));

    HIP_CHECK(hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dbsrVal, hbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb, hipMemcpyHostToDevice));

    int*   dbscRowInd = nullptr;
    int*   dbscColPtr = nullptr;
    float* dbscVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbscRowInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbscColPtr, sizeof(int) * (nb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbscVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb));

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseSgebsr2gebsc_bufferSize(handle,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     dbsrVal,
                                                     dbsrRowPtr,
                                                     dbsrColInd,
                                                     rowBlockDim,
                                                     colBlockDim,
                                                     &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    HIPSPARSE_CHECK(hipsparseSgebsr2gebsc(handle,
                                          mb,
                                          nb,
                                          nnzb,
                                          dbsrVal,
                                          dbsrRowPtr,
                                          dbsrColInd,
                                          rowBlockDim,
                                          colBlockDim,
                                          dbscVal,
                                          dbscRowInd,
                                          dbscColPtr,
                                          action,
                                          base,
                                          dbuffer));

    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));

    HIP_CHECK(hipFree(dbscRowInd));
    HIP_CHECK(hipFree(dbscColPtr));
    HIP_CHECK(hipFree(dbscVal));

    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]