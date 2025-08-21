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
    // hipsparseHandle_t     handle = NULL;
    // hipsparseSpMatDescr_t matA, matB, matC;
    // void*                 dBuffer1    = NULL;
    // void*                 dBuffer2    = NULL;
    // size_t                bufferSize1 = 0;
    // size_t                bufferSize2 = 0;

    // HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // // Create sparse matrix A in CSR format
    // HIPSPARSE_CHECK(hipsparseCreateCsr(&matA,
    //                                    m,
    //                                    k,
    //                                    nnzA,
    //                                    dcsrRowPtrA,
    //                                    dcsrColIndA,
    //                                    dcsrValA,
    //                                    HIPSPARSE_INDEX_32I,
    //                                    HIPSPARSE_INDEX_32I,
    //                                    HIPSPARSE_INDEX_BASE_ZERO,
    //                                    HIP_R_32F));
    // HIPSPARSE_CHECK(hipsparseCreateCsr(&matB,
    //                                    k,
    //                                    n,
    //                                    nnzB,
    //                                    dcsrRowPtrB,
    //                                    dcsrColIndB,
    //                                    dcsrValB,
    //                                    HIPSPARSE_INDEX_32I,
    //                                    HIPSPARSE_INDEX_32I,
    //                                    HIPSPARSE_INDEX_BASE_ZERO,
    //                                    HIP_R_32F));
    // HIPSPARSE_CHECK(hipsparseCreateCsr(&matC,
    //                                    m,
    //                                    n,
    //                                    0,
    //                                    dcsrRowPtrC,
    //                                    NULL,
    //                                    NULL,
    //                                    HIPSPARSE_INDEX_32I,
    //                                    HIPSPARSE_INDEX_32I,
    //                                    HIPSPARSE_INDEX_BASE_ZERO,
    //                                    HIP_R_32F));

    // hipsparseSpGEMMDescr_t spgemmDesc;
    // HIPSPARSE_CHECK(hipsparseSpGEMM_createDescr(&spgemmDesc));

    // // Determine size of first user allocated buffer
    // HIPSPARSE_CHECK(hipsparseSpGEMM_workEstimation(handle,
    //                                                opA,
    //                                                opB,
    //                                                &alpha,
    //                                                matA,
    //                                                matB,
    //                                                &beta,
    //                                                matC,
    //                                                computeType,
    //                                                HIPSPARSE_SPGEMM_DEFAULT,
    //                                                spgemmDesc,
    //                                                &bufferSize1,
    //                                                NULL));
    // hipMalloc((void**)&dBuffer1, bufferSize1);

    // // Inspect the matrices A and B to determine the number of intermediate product in
    // // C = alpha * A * B
    // HIPSPARSE_CHECK(hipsparseSpGEMM_workEstimation(handle,
    //                                                opA,
    //                                                opB,
    //                                                &alpha,
    //                                                matA,
    //                                                matB,
    //                                                &beta,
    //                                                matC,
    //                                                computeType,
    //                                                HIPSPARSE_SPGEMM_DEFAULT,
    //                                                spgemmDesc,
    //                                                &bufferSize1,
    //                                                dBuffer1));

    // // Determine size of second user allocated buffer
    // HIPSPARSE_CHECK(hipsparseSpGEMM_compute(handle,
    //                                         opA,
    //                                         opB,
    //                                         &alpha,
    //                                         matA,
    //                                         matB,
    //                                         &beta,
    //                                         matC,
    //                                         computeType,
    //                                         HIPSPARSE_SPGEMM_DEFAULT,
    //                                         spgemmDesc,
    //                                         &bufferSize2,
    //                                         NULL));
    // HIP_CHECK(hipMalloc((void**)&dBuffer2, bufferSize2));

    // // Compute C = alpha * A * B and store result in temporary buffers
    // HIPSPARSE_CHECK(hipsparseSpGEMM_compute(handle,
    //                                         opA,
    //                                         opB,
    //                                         &alpha,
    //                                         matA,
    //                                         matB,
    //                                         &beta,
    //                                         matC,
    //                                         computeType,
    //                                         HIPSPARSE_SPGEMM_DEFAULT,
    //                                         spgemmDesc,
    //                                         &bufferSize2,
    //                                         dBuffer2));

    // // Get matrix C non-zero entries C_nnz1
    // int64_t C_num_rows1, C_num_cols1, C_nnz1;
    // HIPSPARSE_CHECK(hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1));

    // // Allocate the CSR structures for the matrix C
    // HIP_CHECK(hipMalloc((void**)&dcsrColIndC, C_nnz1 * sizeof(int)));
    // HIP_CHECK(hipMalloc((void**)&dcsrValC, C_nnz1 * sizeof(float)));

    // // Update matC with the new pointers
    // HIPSPARSE_CHECK(hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC));

    // // Copy the final products to the matrix C
    // HIPSPARSE_CHECK(hipsparseSpGEMM_copy(handle,
    //                                      opA,
    //                                      opB,
    //                                      &alpha,
    //                                      matA,
    //                                      matB,
    //                                      &beta,
    //                                      matC,
    //                                      computeType,
    //                                      HIPSPARSE_SPGEMM_DEFAULT,
    //                                      spgemmDesc));

    // // Destroy matrix descriptors and handles
    // HIPSPARSE_CHECK(hipsparseSpGEMM_destroyDescr(spgemmDesc));
    // HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    // HIPSPARSE_CHECK(hipsparseDestroySpMat(matB));
    // HIPSPARSE_CHECK(hipsparseDestroySpMat(matC));
    // HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // // Free device memory
    // HIP_CHECK(hipFree(dBuffer1));
    // HIP_CHECK(hipFree(dBuffer2));

    return 0;
}
//! [doc example]