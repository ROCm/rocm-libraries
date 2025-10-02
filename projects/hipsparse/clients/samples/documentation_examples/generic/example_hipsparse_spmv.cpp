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
    // A, x, and y are m×k, k×1, and m×1
    int                  m = 3, k = 4;
    int                  nnz_A  = 8;
    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // alpha and beta
    float alpha = 0.5f;
    float beta  = 0.25f;

    std::vector<int>   hcsrRowPtr = {0, 3, 5, 8};
    std::vector<int>   hcsrColInd = {0, 1, 3, 1, 2, 0, 2, 3};
    std::vector<float> hcsrVal    = {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<float> hx(k, 1.0f);
    std::vector<float> hy(m, 1.0f);

    int*   dcsrRowPtr;
    int*   dcsrColInd;
    float* dcsrVal;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A));

    HIP_CHECK(
        hipMemcpy(dcsrRowPtr, hcsrRowPtr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    hipsparseSpMatDescr_t matA;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA,
                                       m,
                                       k,
                                       nnz_A,
                                       dcsrRowPtr,
                                       dcsrColInd,
                                       dcsrVal,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO,
                                       HIP_R_32F));

    // Allocate memory for the vector x
    float* dx;
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(float) * k));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * k, hipMemcpyHostToDevice));

    hipsparseDnVecDescr_t vecX;
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecX, k, dx, HIP_R_32F));

    // Allocate memory for the resulting vector y
    float* dy;
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * m));
    HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(float) * m, hipMemcpyHostToDevice));

    hipsparseDnMatDescr_t vecY;
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecY, m, dy, HIP_R_32F));

    // Compute buffersize
    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(handle,
                                             transA,
                                             &alpha,
                                             matA,
                                             vecX,
                                             &beta,
                                             vecY,
                                             HIP_R_32F,
                                             HIPSPARSE_MV_ALG_DEFAULT,
                                             &bufferSize));

    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, bufferSize));

    // Preprocess operation (Optional)
    HIPSPARSE_CHECK(hipsparseSpMV_preprocess(handle,
                                             transA,
                                             &alpha,
                                             matA,
                                             vecX,
                                             &beta,
                                             vecY,
                                             HIP_R_32F,
                                             HIPSPARSE_MV_ALG_DEFAULT,
                                             buffer));

    // Perform operation
    HIPSPARSE_CHECK(hipsparseSpMV(handle,
                                  transA,
                                  &alpha,
                                  matA,
                                  vecX,
                                  &beta,
                                  vecY,
                                  HIP_R_32F,
                                  HIPSPARSE_MV_ALG_DEFAULT,
                                  buffer));

    // Copy device to host
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost));

    // Destroy matrix descriptors and handles
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecX));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecY));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(buffer));
    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    return 0;
}
//! [doc example]