/*! \file */
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

#include "rocsparse/rocsparse.h"
#include <cmath>
#include <hip/hip_runtime_api.h>
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

#define ROCSPARSE_CHECK(stat)                                                         \
    {                                                                                 \
        if(stat != rocsparse_status_success)                                          \
        {                                                                             \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ \
                      << std::endl;                                                   \
            return -1;                                                                \
        }                                                                             \
    }

int main(int argc, char* argv[])
{
    // Query device
    int ndev;
    HIP_CHECK(hipGetDeviceCount(&ndev));

    if(ndev < 1)
    {
        std::cerr << "No HIP device found" << std::endl;
        return -1;
    }

    // Query device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << std::endl;

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    // Print rocSPARSE version and revision
    int  ver;
    char rev[64];

    ROCSPARSE_CHECK(rocsparse_get_version(handle, &ver));
    ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev));

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    // Input data

    // Matrix A (m x n)
    // ( 1.0  2.0  0.0  3.0  0.0 )
    // ( 0.0  4.0  5.0  0.0  0.0 )
    // ( 6.0  0.0  0.0  7.0  8.0 )

    // Number of rows and columns
    int64_t m = 3;
    int64_t n = 5;

    // Number of non-zero entries
    int64_t nnz_A = 8;

    // CSR row pointers
    int64_t hcsr_row_ptr_A[4] = {0, 3, 5, 8};

    // CSR column indices
    int32_t hcsr_col_ind_A[8] = {0, 1, 3, 1, 2, 0, 3, 4};

    // CSR values
    double hcsr_val_A[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // Scalar alpha
    double alpha = 3.7;

    // Offload data to device
    int64_t* dcsr_row_ptr_A;
    int32_t* dcsr_col_ind_A;
    double*  dcsr_val_A;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_A, sizeof(int64_t) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_A, sizeof(int32_t) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_A, sizeof(double) * nnz_A));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(int64_t) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A, sizeof(int32_t) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(double) * nnz_A, hipMemcpyHostToDevice));

    // Matrix C has the same sparsity pattern as A (nnz(C) == nnz(A)).
    int64_t  nnz_C = nnz_A;
    int64_t* dcsr_row_ptr_C;
    int32_t* dcsr_col_ind_C;
    double*  dcsr_val_C;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_C, sizeof(int64_t) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_C, sizeof(int32_t) * nnz_C));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_C, sizeof(double) * nnz_C));

    // rocsparse_spmat_scale writes only C's values; the caller owns C's sparsity pattern, so copy
    // A's structure (row pointers and column indices) into C here.
    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_C, dcsr_row_ptr_A, sizeof(int64_t) * (m + 1), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_C, dcsr_col_ind_A, sizeof(int32_t) * nnz_C, hipMemcpyDeviceToDevice));

    // Create sparse matrices
    rocsparse_spmat_descr A;
    rocsparse_spmat_descr C;

    rocsparse_index_base base  = rocsparse_index_base_zero;
    rocsparse_indextype  itype = rocsparse_indextype_i64;
    rocsparse_indextype  jtype = rocsparse_indextype_i32;
    rocsparse_datatype   ttype = rocsparse_datatype_f64_r;

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &A, m, n, nnz_A, dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A, itype, jtype, base, ttype));
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &C, m, n, nnz_C, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C, itype, jtype, base, ttype));

    // Wrap the host scalar alpha in a self-describing size-one dense vector descriptor.
    rocsparse_dnvec_descr alpha_descr;
    ROCSPARSE_CHECK(rocsparse_dnvec_descr_create_scalar(
        handle, &alpha_descr, rocsparse_pointer_mode_host, ttype, &alpha, &alpha, nullptr));

    // Compute C = alpha * A. No temporary storage buffer is required.
    ROCSPARSE_CHECK(rocsparse_spmat_scale(handle, alpha_descr, A, C, nullptr));

    HIP_CHECK(hipStreamSynchronize(stream));

    // Copy result back to host
    std::vector<int64_t> hcsr_row_ptr_C(m + 1);
    std::vector<int32_t> hcsr_col_ind_C(nnz_C);
    std::vector<double>  hcsr_val_C(nnz_C);

    HIP_CHECK(hipMemcpy(
        hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int64_t) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int32_t) * nnz_C, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(double) * nnz_C, hipMemcpyDeviceToHost));

    // Print result
    std::cout << "Matrix C = " << alpha << " * A : " << nnz_C << " non-zero elements" << std::endl;

    std::cout << "C row pointer:";
    for(int i = 0; i < m + 1; ++i)
    {
        std::cout << " " << hcsr_row_ptr_C[i];
    }

    std::cout << std::endl << "C column indices:";
    for(int i = 0; i < nnz_C; ++i)
    {
        std::cout << " " << hcsr_col_ind_C[i];
    }

    std::cout << std::endl << "C values:";
    for(int i = 0; i < nnz_C; ++i)
    {
        std::cout << " " << hcsr_val_C[i];
    }
    std::cout << std::endl;

    // Validate the result against the host reference C = alpha * A.
    int    errors    = 0;
    double tolerance = 1e-10;
    for(int i = 0; i < nnz_C; ++i)
    {
        if(hcsr_col_ind_C[i] != hcsr_col_ind_A[i])
        {
            ++errors;
        }
        if(std::abs(hcsr_val_C[i] - alpha * hcsr_val_A[i]) > tolerance)
        {
            ++errors;
        }
    }

    if(errors == 0)
    {
        std::cout << "spmat_scale validation: PASSED" << std::endl;
    }
    else
    {
        std::cout << "spmat_scale validation: FAILED with " << errors << " errors" << std::endl;
    }

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(alpha_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(A));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(C));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr_A));
    HIP_CHECK(hipFree(dcsr_col_ind_A));
    HIP_CHECK(hipFree(dcsr_val_A));
    HIP_CHECK(hipFree(dcsr_row_ptr_C));
    HIP_CHECK(hipFree(dcsr_col_ind_C));
    HIP_CHECK(hipFree(dcsr_val_C));

    return 0;
}
