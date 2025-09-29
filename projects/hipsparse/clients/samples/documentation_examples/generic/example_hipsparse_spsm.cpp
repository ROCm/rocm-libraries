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
    int n = 2;

    std::vector<int>   hcsr_row_ptr = {0, 1, 3, 5, 6};
    std::vector<int>   hcsr_col_ind = {0, 0, 1, 1, 2, 3};
    std::vector<float> hcsr_val     = {1, 4, 2, 3, 7, 1};
    std::vector<float> hB(m * n);
    std::vector<float> hC(m * n);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            hB[m * i + j] = static_cast<float>(i + 1);
        }
    }

    // Scalar alpha
    float alpha = 1.0f;

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Offload data to device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dB;
    float* dC;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(float) * m * n));
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(float) * m * n));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    hipsparseHandle_t     handle;
    hipsparseSpMatDescr_t matA;
    hipsparseDnMatDescr_t matB;
    hipsparseDnMatDescr_t matC;

    hipsparseIndexType_t row_idx_type = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t col_idx_type = HIPSPARSE_INDEX_32I;
    hipDataType          dataType     = HIP_R_32F;
    hipDataType          computeType  = HIP_R_32F;
    hipsparseIndexBase_t idxBase      = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseOperation_t transA       = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB       = HIPSPARSE_OPERATION_NON_TRANSPOSE;

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
                                       idxBase,
                                       dataType));

    // Create dense matrix B
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matB, m, n, m, dB, dataType, HIPSPARSE_ORDER_COL));

    // Create dense matrix C
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matC, m, n, m, dC, dataType, HIPSPARSE_ORDER_COL));

    hipsparseSpSMDescr_t descr;
    HIPSPARSE_CHECK(hipsparseSpSM_createDescr(&descr));

    // Call SpSM to get buffer size
    size_t buffer_size;
    HIPSPARSE_CHECK(hipsparseSpSM_bufferSize(handle,
                                             transA,
                                             transB,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             computeType,
                                             HIPSPARSE_SPSM_ALG_DEFAULT,
                                             descr,
                                             &buffer_size));

    void* temp_buffer;
    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    // Call SpSM to perform analysis
    HIPSPARSE_CHECK(hipsparseSpSM_analysis(handle,
                                           transA,
                                           transB,
                                           &alpha,
                                           matA,
                                           matB,
                                           matC,
                                           computeType,
                                           HIPSPARSE_SPSM_ALG_DEFAULT,
                                           descr,
                                           temp_buffer));

    // Call SpSM to perform computation
    HIPSPARSE_CHECK(hipsparseSpSM_solve(handle,
                                        transA,
                                        transB,
                                        &alpha,
                                        matA,
                                        matB,
                                        matC,
                                        computeType,
                                        HIPSPARSE_SPSM_ALG_DEFAULT,
                                        descr,
                                        temp_buffer));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hC.data(), dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseSpSM_destroyDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matA));
    HIPSPARSE_CHECK(hipsparseDestroyDnMat(matB));
    HIPSPARSE_CHECK(hipsparseDestroyDnMat(matC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
//! [doc example]