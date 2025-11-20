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

#include "utils.hpp"
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                 \
    {                                   \
        if(stat != hipSuccess)          \
        {                               \
            std::cerr << "HIP error\n"; \
            exit(-1);                   \
        }                               \
    }

#define ROCSPARSE_CHECK(stat)                 \
    {                                         \
        if(stat != rocsparse_status_success)  \
        {                                     \
            std::cerr << "rocSPARSE error\n"; \
            exit(-1);                         \
        }                                     \
    }

int main()
{
    // Matrix parameters
    const rocsparse_int m = 100, n = 100;
    const rocsparse_int nnz = 500;

    // Allocate and initialize CSR matrix
    std::vector<rocsparse_int> hA_ptr(m + 1);
    std::vector<rocsparse_int> hA_col(nnz);
    std::vector<float>         hA_val(nnz);

    // ... initialize matrix data ...

    // Allocate device memory
    rocsparse_int *dA_ptr, *dA_col;
    float *        dA_val, *dx, *dy, *dz;

    HIP_CHECK(hipMalloc(&dA_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dA_col, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dA_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc(&dx, sizeof(float) * n));
    HIP_CHECK(hipMalloc(&dy, sizeof(float) * m));
    HIP_CHECK(hipMalloc(&dz, sizeof(float) * m));

    // Copy to device
    HIP_CHECK(
        hipMemcpy(dA_ptr, hA_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    // ... copy other data ...

    // Create handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create descriptors
    rocsparse_spmat_descr A;
    rocsparse_dnvec_descr x_vec, y_vec, z_vec;
    rocsparse_spmv_descr  spmv_descr;

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&A,
                                               m,
                                               n,
                                               nnz,
                                               dA_ptr,
                                               dA_col,
                                               dA_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));

    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&x_vec, n, dx, rocsparse_datatype_f32_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&y_vec, m, dy, rocsparse_datatype_f32_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&z_vec, m, dz, rocsparse_datatype_f32_r));
    ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&spmv_descr));

    // Set spmv parameters
    rocsparse_spmv_alg alg = rocsparse_spmv_alg_csr_adaptive;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(
        handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));

    // Set extra residual parameters
    // Create gamma dnvec with host values
    float                 gamma = 0.5f;
    rocsparse_dnvec_descr gamma_vec;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&gamma_vec, 1, &gamma, rocsparse_datatype_f32_r));

    rocsparse_const_dnvec_descr z_vecs[1] = {extra};

    ROCSPARSE_CHECK(rocsparse_spmv_set_extra(handle, spmv_descr, 1, gamma_vec, z_vecs, nullptr));

    float alpha = 1.0f, beta = 0.0f;

    // Get buffer size and allocate
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle,
                                                  spmv_descr,
                                                  A,
                                                  x_vec,
                                                  y_vec,
                                                  rocsparse_v2_spmv_stage_analysis,
                                                  &buffer_size,
                                                  nullptr));

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Analysis stage
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &alpha,
                                      A,
                                      x_vec,
                                      &beta,
                                      y_vec,
                                      rocsparse_v2_spmv_stage_analysis,
                                      buffer_size,
                                      temp_buffer,
                                      nullptr));

    // Compute stage
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &alpha,
                                      A,
                                      x_vec,
                                      &beta,
                                      y_vec,
                                      rocsparse_v2_spmv_stage_compute,
                                      buffer_size,
                                      temp_buffer,
                                      nullptr));

    std::cout << "SpMV with extra vectors computed successfully!\n";

    // Demonstrate enable/disable extra functionality
    std::cout << "Demonstrating enable/disable extra functionality...\n";

    // Disable extra vectors temporarily
    int32_t disable_extra = 0;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_enable_extra,
                                             &disable_extra,
                                             sizeof(disable_extra),
                                             nullptr));
    std::cout << "Extra vectors disabled\n";

    // Run SpMV again without extra vectors
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &alpha,
                                      A,
                                      x_vec,
                                      &beta,
                                      y_vec,
                                      rocsparse_v2_spmv_stage_compute,
                                      buffer_size,
                                      temp_buffer,
                                      nullptr));

    std::cout << "SpMV without extra vectors computed\n";

    // Re-enable extra vectors
    int32_t enable_extra = 1;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle,
                                             spmv_descr,
                                             rocsparse_spmv_input_enable_extra,
                                             &enable_extra,
                                             sizeof(enable_extra),
                                             nullptr));
    std::cout << "Extra vectors re-enabled\n";

    // Run SpMV again with extra vectors
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle,
                                      spmv_descr,
                                      &alpha,
                                      A,
                                      x_vec,
                                      &beta,
                                      y_vec,
                                      rocsparse_v2_spmv_stage_compute,
                                      buffer_size,
                                      temp_buffer,
                                      nullptr));

    std::cout << "SpMV with re-enabled extra vectors computed\n";

    // Clear extra parameters
    ROCSPARSE_CHECK(rocsparse_spmv_clear_extra(handle, spmv_descr, nullptr));

    // Cleanup
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(dA_ptr));
    // ... free other allocations ...

    ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(spmv_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(A));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(x_vec));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(y_vec));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(z_vec));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(gamma_vec));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    std::cout << "Success!\n";
    return 0;
}
