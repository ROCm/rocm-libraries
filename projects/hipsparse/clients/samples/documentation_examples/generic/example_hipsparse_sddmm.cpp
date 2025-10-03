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

#include <hip/hip_fp16.h>
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
    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    __half halpha = 1.0;
    __half hbeta  = 0.0;

    // A, B, and C are mxk, kxn, and mxn
    int m    = 4;
    int k    = 3;
    int n    = 2;
    int nnzC = 5;

    //     2  3  -1
    // A = 0  2   1
    //     0  0   5
    //     0 -2 0.5

    //      0  4
    // B =  1  0
    //     -2  0.5

    //      1 0            1 0
    // C =  2 3   spy(C) = 1 1
    //      0 0            0 0
    //      4 5            1 1

    std::vector<__half> hA = {2.0, 3.0, -1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 5.0, 0.0, -2.0, 0.5};
    std::vector<__half> hB = {0.0, 4.0, 1.0, 0.0, -2.0, 0.5};

    std::vector<int>    hcsr_row_ptrC = {0, 1, 3, 3, 5};
    std::vector<int>    hcsr_col_indC = {0, 0, 1, 0, 1};
    std::vector<__half> hcsr_valC     = {1.0, 2.0, 3.0, 4.0, 5.0};

    __half* dA = nullptr;
    __half* dB = nullptr;
    HIP_CHECK(hipMalloc((void**)&dA, sizeof(__half) * m * k));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(__half) * k * n));

    int*    dcsr_row_ptrC = nullptr;
    int*    dcsr_col_indC = nullptr;
    __half* dcsr_valC     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptrC, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_indC, sizeof(int) * nnzC));
    HIP_CHECK(hipMalloc((void**)&dcsr_valC, sizeof(__half) * nnzC));

    HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(__half) * m * k, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(__half) * k * n, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptrC, hcsr_row_ptrC.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_indC, hcsr_col_indC.data(), sizeof(int) * nnzC, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_valC, hcsr_valC.data(), sizeof(__half) * nnzC, hipMemcpyHostToDevice));

    hipsparseDnMatDescr_t matA;
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matA, m, k, k, dA, HIP_R_16F, HIPSPARSE_ORDER_ROW));

    hipsparseDnMatDescr_t matB;
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matB, k, n, n, dB, HIP_R_16F, HIPSPARSE_ORDER_ROW));

    hipsparseSpMatDescr_t matC;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matC,
                                       m,
                                       n,
                                       nnzC,
                                       dcsr_row_ptrC,
                                       dcsr_col_indC,
                                       dcsr_valC,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO,
                                       HIP_R_16F));

    size_t buffer_size = 0;
    HIPSPARSE_CHECK(hipsparseSDDMM_bufferSize(handle,
                                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                              &halpha,
                                              matA,
                                              matB,
                                              &hbeta,
                                              matC,
                                              HIP_R_16F,
                                              HIPSPARSE_SDDMM_ALG_DEFAULT,
                                              &buffer_size));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, buffer_size));

    HIPSPARSE_CHECK(hipsparseSDDMM_preprocess(handle,
                                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                              &halpha,
                                              matA,
                                              matB,
                                              &hbeta,
                                              matC,
                                              HIP_R_16F,
                                              HIPSPARSE_SDDMM_ALG_DEFAULT,
                                              dbuffer));

    HIPSPARSE_CHECK(hipsparseSDDMM(handle,
                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                   &halpha,
                                   matA,
                                   matB,
                                   &hbeta,
                                   matC,
                                   HIP_R_16F,
                                   HIPSPARSE_SDDMM_ALG_DEFAULT,
                                   dbuffer));

    HIP_CHECK(hipMemcpy(
        hcsr_row_ptrC.data(), dcsr_row_ptrC, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_col_indC.data(), dcsr_col_indC, sizeof(int) * nnzC, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_valC.data(), dcsr_valC, sizeof(__half) * nnzC, hipMemcpyDeviceToHost));

    std::cout << "C" << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = hcsr_row_ptrC[i];
        int end   = hcsr_row_ptrC[i + 1];

        std::vector<__half> temp(n, 0.0);
        for(int j = start; j < end; j++)
        {
            temp[hcsr_col_indC[j]] = hcsr_valC[j];
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matA));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matB));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(matC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dcsr_row_ptrC));
    HIP_CHECK(hipFree(dcsr_col_indC));
    HIP_CHECK(hipFree(dcsr_valC));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]