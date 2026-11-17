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

#include <iostream>
#include <vector>

#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                                        \
        }                                                                                     \
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

//! [doc example]
int main()
{
    // This example demonstrates computing the residual r = b - A * x
    // using rocsparse_spmv_set_extra. The key insight is:
    //
    //   y = alpha * A * x + beta * y + gamma * z
    //
    // Setting alpha = -1, beta = 0, gamma = 1, z = b gives:
    //
    //   r = -1 * A * x + 0 * r + 1 * b = b - A * x
    //
    // CSR sparse matrix A (4 x 6):
    //     1 4 0 0 0 0
    // A = 0 2 3 0 0 0
    //     5 0 0 7 8 0
    //     0 0 9 0 6 0
    int m = 4;
    int n = 6;

    std::vector<int>   hcsr_row_ptr = {0, 2, 4, 7, 9};
    std::vector<int>   hcsr_col_ind = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    std::vector<float> hcsr_val     = {1, 4, 2, 3, 5, 7, 8, 9, 6};

    // Approximate solution x = [1, 1, 1, 1, 1, 1]
    std::vector<double> hx(n, 1.0);

    // Right-hand side b. With x = [1,...,1], A*x = [5, 5, 20, 15].
    // Perturb b so the residual is non-zero: r = b - A*x = [1, 2, 2, 3].
    std::vector<double> hb = {6.0, 7.0, 22.0, 18.0};

    // Output: residual r = b - A * x (will be written by SpMV)
    std::vector<double> hr(m, 0.0);

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Allocate device memory
    int*    dcsr_row_ptr;
    int*    dcsr_col_ind;
    float*  dcsr_val;
    double* dx;
    double* db;
    double* dr;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc(&db, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&dr, sizeof(double) * m));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, hb.data(), sizeof(double) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dr, 0, sizeof(double) * m));

    rocsparse_handle      handle;
    rocsparse_error       p_error[1] = {};
    rocsparse_spmat_descr matA;
    rocsparse_dnvec_descr vecX;
    rocsparse_dnvec_descr vecB;
    rocsparse_dnvec_descr vecR;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Sparse matrix A stored in CSR format (float values, double compute)
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               n,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));

    // Dense vector x (approximate solution)
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecX, n, dx, rocsparse_datatype_f64_r));

    // Dense vector b (right-hand side), used as the extra z vector
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecB, m, db, rocsparse_datatype_f64_r));

    // Dense vector r (output residual)
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecR, m, dr, rocsparse_datatype_f64_r));

    // Create SpMV descriptor and configure inputs
    rocsparse_spmv_descr spmv_descr;
    ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&spmv_descr));

    const rocsparse_spmv_alg spmv_alg = rocsparse_spmv_alg_csr_adaptive;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(
        handle, spmv_descr, rocsparse_spmv_input_alg, &spmv_alg, sizeof(spmv_alg), p_error));

    const rocsparse_operation spmv_operation = rocsparse_operation_none;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_operation,
                                             &spmv_operation,
                                             sizeof(spmv_operation),
                                             p_error));

    const rocsparse_datatype spmv_scalar_datatype = rocsparse_datatype_f64_r;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_scalar_datatype,
                                             &spmv_scalar_datatype,
                                             sizeof(spmv_scalar_datatype),
                                             p_error));

    const rocsparse_datatype spmv_compute_datatype = rocsparse_datatype_f64_r;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_compute_datatype,
                                             &spmv_compute_datatype,
                                             sizeof(spmv_compute_datatype),
                                             p_error));

    // Set extra term: gamma = 1.0 and z = b, so the SpMV will add 1.0 * b to the result.
    // Combined with alpha = -1 and beta = 0, this computes r = -A*x + b = b - A*x.
    double gamma = 1.0;

    rocsparse_dnvec_descr gamma_vec;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&gamma_vec, 1, &gamma, rocsparse_datatype_f64_r));

    rocsparse_const_dnvec_descr z_vecs[1] = {vecB};
    ROCSPARSE_CHECK(rocsparse_spmv_set_extra(handle, spmv_descr, 1, gamma_vec, z_vecs, p_error));

    // alpha = -1: negates A*x contribution
    // beta  =  0: does not accumulate into r
    double alpha = -1.0;
    double beta  = 0.0;

    // Get buffer size for the analysis stage
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle,
                                                  spmv_descr,
                                                  matA,
                                                  vecX,
                                                  vecR,
                                                  rocsparse_v2_spmv_stage_analysis,
                                                  &buffer_size,
                                                  p_error));

    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Perform analysis
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &alpha,
                                      matA,
                                      vecX,
                                      &beta,
                                      vecR,
                                      rocsparse_v2_spmv_stage_analysis,
                                      buffer_size,
                                      buffer,
                                      p_error));

    HIP_CHECK(hipFree(buffer));

    // Get buffer size for the compute stage
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle,
                                                  spmv_descr,
                                                  matA,
                                                  vecX,
                                                  vecR,
                                                  rocsparse_v2_spmv_stage_compute,
                                                  &buffer_size,
                                                  p_error));

    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Compute r = alpha * A * x + beta * r + gamma * b
    //           = -1.0 * A * x + 0.0 * r + 1.0 * b
    //           = b - A * x
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &alpha,
                                      matA,
                                      vecX,
                                      &beta,
                                      vecR,
                                      rocsparse_v2_spmv_stage_compute,
                                      buffer_size,
                                      buffer,
                                      p_error));

    HIP_CHECK(hipFree(buffer));

    // Copy residual back to host
    HIP_CHECK(hipMemcpy(hr.data(), dr, sizeof(double) * m, hipMemcpyDeviceToHost));

    // With x=[1,...,1] and b=[6,7,22,18]: r = b - A*x = [1, 2, 2, 3]
    std::cout << "Residual r = b - A * x:" << std::endl;
    for(int i = 0; i < m; ++i)
    {
        std::cout << "  r[" << i << "] = " << hr[i] << std::endl;
    }

    // Clear extra parameters and release resources
    ROCSPARSE_CHECK(rocsparse_spmv_clear_extra(handle, spmv_descr, p_error));
    ROCSPARSE_CHECK(rocsparse_destroy_error(p_error[0]));
    ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(spmv_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(gamma_vec));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecX));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecB));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecR));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(db));
    HIP_CHECK(hipFree(dr));

    return 0;
}
//! [doc example]
