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
    // Number of non-zeros of the sparse vector
    int nnz = 3;

    // Size of sparse and dense vector
    int size = 9;

    // Sparse index vector
    int hxInd[] = {0, 3, 5};

    // Sparse value vector
    float hxVal[] = {1.0, 2.0, 3.0};

    // Dense vector
    float hy[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Offload data to device
    int*   dxInd;
    float* dxVal;
    float* dy;
    HIP_CHECK(hipMalloc((void**)&dxInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dxVal, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(dxInd, hxInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dxVal, hxVal, sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(float) * size, hipMemcpyHostToDevice));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Create sparse vector X
    hipsparseSpVecDescr_t vecX;
    HIPSPARSE_CHECK(hipsparseCreateSpVec(
        &vecX, size, nnz, dxInd, dxVal, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));

    // Create dense vector Y
    hipsparseDnVecDescr_t vecY;
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F));

    // Obtain buffer size
    float  hresult = 0.0;
    size_t buffer_size;
    HIPSPARSE_CHECK(hipsparseSpVV_bufferSize(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, vecX, vecY, &hresult, HIP_R_32F, &buffer_size));

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // SpVV
    HIPSPARSE_CHECK(hipsparseSpVV(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, vecX, vecY, &hresult, HIP_R_32F, temp_buffer));

    HIP_CHECK(hipDeviceSynchronize());

    printf("hresult: %f\n", hresult);

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroySpVec(vecX));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecY));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dxInd));
    HIP_CHECK(hipFree(dxVal));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
/*! [doc example end] */