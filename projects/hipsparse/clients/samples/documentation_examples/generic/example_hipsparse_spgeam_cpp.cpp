/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <iostream>
#include <vector>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            return -1;                                                         \
        }                                                                      \
    }

#define HIPSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != HIPSPARSE_STATUS_SUCCESS)                                         \
        {                                                                            \
            std::cerr << "Error: hipsparse error in line " << __LINE__ << std::endl; \
            return -1;                                                               \
        }                                                                            \
    }

#ifdef HIPSPARSE_WITH_SPGEAM
//! [doc example start]
int main()
{
    // C = alpha * A + beta * B, where A, B and C are m x n CSR matrices.
    //     1 2 0 0
    // A = 0 0 3 0
    //     0 4 0 5
    int m = 3;
    int n = 4;

    std::vector<int>   hcsr_row_ptr_A = {0, 2, 3, 5};
    std::vector<int>   hcsr_col_ind_A = {0, 1, 2, 1, 3};
    std::vector<float> hcsr_val_A     = {1, 2, 3, 4, 5};

    //     0 1 0 6
    // B = 2 0 0 0
    //     0 0 7 8
    std::vector<int>   hcsr_row_ptr_B = {0, 2, 3, 5};
    std::vector<int>   hcsr_col_ind_B = {1, 3, 0, 2, 3};
    std::vector<float> hcsr_val_B     = {1, 6, 2, 7, 8};

    int nnz_A = hcsr_col_ind_A.size();
    int nnz_B = hcsr_col_ind_B.size();

    float alpha = 1.0f;
    float beta  = 1.0f;

    // Device memory for A and B
    int*   dcsr_row_ptr_A;
    int*   dcsr_col_ind_A;
    float* dcsr_val_A;
    int*   dcsr_row_ptr_B;
    int*   dcsr_col_ind_B;
    float* dcsr_val_B;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_A, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_A, sizeof(int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_A, sizeof(float) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_B, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_B, sizeof(int) * nnz_B));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_B, sizeof(float) * nnz_B));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice));

    // The C row offsets array can be allocated up front
    int* dcsr_row_ptr_C;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_C, sizeof(int) * (m + 1)));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    hipsparseConstSpMatDescr_t matA, matB;
    hipsparseSpMatDescr_t      matC;

    HIPSPARSE_CHECK(hipsparseCreateConstCsr(&matA,
                                            m,
                                            n,
                                            nnz_A,
                                            dcsr_row_ptr_A,
                                            dcsr_col_ind_A,
                                            dcsr_val_A,
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_BASE_ZERO,
                                            HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateConstCsr(&matB,
                                            m,
                                            n,
                                            nnz_B,
                                            dcsr_row_ptr_B,
                                            dcsr_col_ind_B,
                                            dcsr_val_B,
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_BASE_ZERO,
                                            HIP_R_32F));

    // Create C with a zero nnz for now, its column indices and values are allocated later
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matC,
                                       m,
                                       n,
                                       0,
                                       dcsr_row_ptr_C,
                                       nullptr,
                                       nullptr,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO,
                                       HIP_R_32F));

    hipsparseSpGEAMDescr_t spgeamDescr;
    HIPSPARSE_CHECK(hipsparseSpGEAM_createDescr(&spgeamDescr));

    // Query and allocate the shared workspace
    size_t bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseSpGEAM_bufferSize(handle,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               &beta,
                                               matB,
                                               matC,
                                               HIP_R_32F,
                                               HIPSPARSE_SPGEAM_ALG1,
                                               spgeamDescr,
                                               &bufferSize));

    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, bufferSize));

    // Compute the number of non-zeros and the row offsets of C
    HIPSPARSE_CHECK(hipsparseSpGEAM_nnz(handle,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        &beta,
                                        matB,
                                        matC,
                                        HIP_R_32F,
                                        HIPSPARSE_SPGEAM_ALG1,
                                        spgeamDescr,
                                        dbuffer));

    // Retrieve the number of non-zeros in C
    int64_t rows_C, cols_C, nnz_C;
    HIPSPARSE_CHECK(hipsparseSpMatGetSize(matC, &rows_C, &cols_C, &nnz_C));

    // Allocate the C column indices and values arrays and set them on the descriptor
    int*   dcsr_col_ind_C;
    float* dcsr_val_C;
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_C, sizeof(int) * nnz_C));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_C, sizeof(float) * nnz_C));
    HIPSPARSE_CHECK(hipsparseCsrSetPointers(matC, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C));

    // Compute C = alpha * A + beta * B
    HIPSPARSE_CHECK(hipsparseSpGEAM(handle,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    &beta,
                                    matB,
                                    matC,
                                    HIP_R_32F,
                                    HIPSPARSE_SPGEAM_ALG1,
                                    spgeamDescr,
                                    dbuffer));

    // Copy C back to the host and print it
    std::vector<int>   hcsr_row_ptr_C(m + 1);
    std::vector<int>   hcsr_col_ind_C(nnz_C);
    std::vector<float> hcsr_val_C(nnz_C);

    HIP_CHECK(hipMemcpy(
        hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * nnz_C, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(float) * nnz_C, hipMemcpyDeviceToHost));

    std::cout << "C = alpha * A + beta * B (nnz = " << nnz_C << ")" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::vector<float> row(n, 0.0f);
        for(int j = hcsr_row_ptr_C[i]; j < hcsr_row_ptr_C[i + 1]; j++)
        {
            row[hcsr_col_ind_C[j]] = hcsr_val_C[j];
        }
        for(int j = 0; j < n; j++)
        {
            std::cout << row[j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    HIPSPARSE_CHECK(hipsparseSpGEAM_destroyDescr(spgeamDescr));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matB));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matC));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dbuffer));
    HIP_CHECK(hipFree(dcsr_row_ptr_A));
    HIP_CHECK(hipFree(dcsr_col_ind_A));
    HIP_CHECK(hipFree(dcsr_val_A));
    HIP_CHECK(hipFree(dcsr_row_ptr_B));
    HIP_CHECK(hipFree(dcsr_col_ind_B));
    HIP_CHECK(hipFree(dcsr_val_B));
    HIP_CHECK(hipFree(dcsr_row_ptr_C));
    HIP_CHECK(hipFree(dcsr_col_ind_C));
    HIP_CHECK(hipFree(dcsr_val_C));

    return 0;
}
//! [doc example end]
#else
int main()
{
    // hipSPARSE was built with the generic SpGEAM API disabled (HIPSPARSE_WITH_SPGEAM off).
    return 0;
}
#endif
