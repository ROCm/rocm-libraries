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
    // A = 4 2 0 0
    //     0 3 7 0
    //     0 0 0 1
    int m = 4;

    std::vector<int>   hcsr_row_ptr = {0, 1, 3, 5, 6};
    std::vector<int>   hcsr_col_ind = {0, 0, 1, 1, 2, 3};
    std::vector<float> hcsr_val     = {1, 4, 2, 3, 7, 1};
    std::vector<float> hx(m, 1.0f);
    std::vector<float> hy(m, 0.0f);

    // Scalar alpha
    float alpha = 1.0f;

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Offload data to device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dx;
    float* dy;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(float) * m));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * m));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * m, hipMemcpyHostToDevice));

    hipsparseHandle_t     handle;
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t vecX;
    hipsparseDnVecDescr_t vecY;

    hipsparseIndexType_t row_idx_type = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t col_idx_type = HIPSPARSE_INDEX_32I;
    hipDataType          data_type    = HIP_R_32F;
    hipDataType          computeType  = HIP_R_32F;
    hipsparseIndexBase_t idx_base     = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseOperation_t trans        = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Create sparse matrix A
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA,
                                       m,
                                       m,
                                       nnz,
                                       dcsr_row_ptr,
                                       dcsr_col_ind,
                                       dcsr_val,
                                       row_idx_type,
                                       col_idx_type,
                                       idx_base,
                                       data_type));

    // Create dense vector X
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecX, m, dx, data_type));

    // Create dense vector Y
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecY, m, dy, data_type));

    hipsparseSpSVDescr_t descr;
    HIPSPARSE_CHECK(hipsparseSpSV_createDescr(&descr));

    // Call spsv to get buffer size
    size_t buffer_size;
    HIPSPARSE_CHECK(hipsparseSpSV_bufferSize(handle,
                                             trans,
                                             &alpha,
                                             matA,
                                             vecX,
                                             vecY,
                                             computeType,
                                             HIPSPARSE_SPSV_ALG_DEFAULT,
                                             descr,
                                             &buffer_size));

    void* temp_buffer;
    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    // Call spsv to perform analysis
    HIPSPARSE_CHECK(hipsparseSpSV_analysis(handle,
                                           trans,
                                           &alpha,
                                           matA,
                                           vecX,
                                           vecY,
                                           computeType,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           descr,
                                           temp_buffer));

    // Call spsv to perform computation
    HIPSPARSE_CHECK(hipsparseSpSV_solve(
        handle, trans, &alpha, matA, vecX, vecY, computeType, HIPSPARSE_SPSV_ALG_DEFAULT, descr));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost));

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseSpSV_destroyDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matA));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecX));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecY));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
//! [doc example]