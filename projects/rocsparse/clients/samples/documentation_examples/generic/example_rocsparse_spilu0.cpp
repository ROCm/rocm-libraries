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
    //
    // Define a matrix.
    //
    //     4 1 0 0
    // A = 2 8 1 0
    //     0 1 8 1
    //     0 0 2 4
    //
    static constexpr int32_t              m            = 4;
    static constexpr int32_t              nnz          = 10;
    static constexpr int64_t              batch_count  = 3;
    static constexpr rocsparse_index_base idx_base     = rocsparse_index_base_zero;
    static constexpr rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    static constexpr rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    static constexpr rocsparse_datatype   data_type    = rocsparse_datatype_f32_r;

    const int32_t hcsr_row_ptr[m + 1]
        = {idx_base + 0, idx_base + 2, idx_base + 5, idx_base + 8, idx_base + 10};

    const int32_t hcsr_col_ind[nnz] = {idx_base + 0,
                                       idx_base + 1,
                                       idx_base + 0,
                                       idx_base + 1,
                                       idx_base + 2,
                                       idx_base + 1,
                                       idx_base + 2,
                                       idx_base + 3,
                                       idx_base + 2,
                                       idx_base + 3};

    const float hcsr_val[nnz * batch_count] = {

        //
        //
        //
        4, 1, 2, 8,     1, 1, 8, 1, 2, 4, 4, 1, 0, 8, 1, 8, 1, 0, 2, 4,

        4, 1, 4, 1.001, 0, 1, 8, 1, 2, 4

    };

    const double singularity_tolerance = 0.01;

    const int32_t boost_enable    = 0;
    const double  boost_tolerance = 0.01;
    const double  boost_value     = 1;

    //
    // Offload data to device
    //
    int32_t* dcsr_row_ptr;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int32_t) * (m + 1)));
    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int32_t) * (m + 1), hipMemcpyHostToDevice));

    int32_t* dcsr_col_ind;
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int32_t) * nnz));
    HIP_CHECK(hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int32_t) * nnz, hipMemcpyHostToDevice));

    float* dcsr_val;
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz * batch_count));
    HIP_CHECK(
        hipMemcpy(dcsr_val, hcsr_val, sizeof(float) * nnz * batch_count, hipMemcpyHostToDevice));

    //
    // Create handle.
    //
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    //
    // Create sparse matrix A
    //
    rocsparse_spmat_descr matA;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
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

    ROCSPARSE_CHECK(rocsparse_csr_set_strided_batch(matA, batch_count, 0, nnz));

    //
    // Create the descriptior of the Incomplete Cholesky algorithm of level 0.
    //
    rocsparse_spilu0_descr spilu0_descr;
    ROCSPARSE_CHECK(rocsparse_spilu0_descr_create(handle, &spilu0_descr, nullptr));

    //
    // Configure the descriptor.
    //
    const rocsparse_spilu0_alg spilu0_alg = rocsparse_spilu0_alg_default;
    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_alg,
                                               &spilu0_alg,
                                               sizeof(spilu0_alg),
                                               nullptr));

    const rocsparse_datatype spilu0_compute_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_compute_datatype,
                                               &spilu0_compute_datatype,
                                               sizeof(spilu0_compute_datatype),
                                               nullptr));

    const rocsparse_analysis_policy spilu0_analysis_policy = rocsparse_analysis_policy_reuse;
    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_analysis_policy,
                                               &spilu0_analysis_policy,
                                               sizeof(spilu0_analysis_policy),
                                               nullptr));

    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_singularity_tolerance,
                                               &singularity_tolerance,
                                               sizeof(double),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_boost_enable,
                                               &boost_enable,
                                               sizeof(int32_t),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_boost_tolerance,
                                               &boost_tolerance,
                                               sizeof(double),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_spilu0_set_input(handle,
                                               spilu0_descr,
                                               rocsparse_spilu0_input_boost_value,
                                               &boost_value,
                                               sizeof(double),
                                               nullptr));

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    //
    // Spilu0 Analysis phase
    //
    size_t non_persistent_buffer_size_in_bytes;
    void*  non_persistent_buffer;

    ROCSPARSE_CHECK(rocsparse_spilu0_buffer_size(handle,
                                                 spilu0_descr,
                                                 matA,
                                                 matA,
                                                 rocsparse_spilu0_stage_analysis,
                                                 &non_persistent_buffer_size_in_bytes,
                                                 nullptr));
    HIP_CHECK(hipMalloc(&non_persistent_buffer, non_persistent_buffer_size_in_bytes));

    ROCSPARSE_CHECK(rocsparse_spilu0(handle,
                                     spilu0_descr,
                                     matA,
                                     matA,
                                     rocsparse_spilu0_stage_analysis,
                                     non_persistent_buffer_size_in_bytes,
                                     non_persistent_buffer,
                                     nullptr));

    //
    // Check for any singularities after analysis.
    //
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    rocsparse_singularity post_analysis_singularity[batch_count];

    ROCSPARSE_CHECK(rocsparse_spilu0_get_output(handle,
                                                spilu0_descr,
                                                rocsparse_spilu0_output_singularity,
                                                post_analysis_singularity,
                                                sizeof(rocsparse_singularity),
                                                nullptr));

    int64_t singularity_position[batch_count];
    ROCSPARSE_CHECK(rocsparse_spilu0_get_output(handle,
                                                spilu0_descr,
                                                rocsparse_spilu0_output_singularity_position,
                                                singularity_position,
                                                sizeof(int64_t),
                                                nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));
    for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        switch(post_analysis_singularity[batch_index])
        {
        case rocsparse_singularity_none:
        {
            break;
        }

        case rocsparse_singularity_symbolic:
        {

            std::cout << "symbolic singularity detected at batch_index: " << batch_index
                      << ", at position: " << singularity_position[batch_index] << std::endl;

            ROCSPARSE_CHECK(rocsparse_status_zero_pivot);
            break;
        }
        case rocsparse_singularity_numeric_exact:
        case rocsparse_singularity_numeric_near:
        {
            //
            // Not from analysis.
            //
            ROCSPARSE_CHECK(rocsparse_status_internal_error);
            break;
        }
        }
    }

    //
    // Compute phase.
    //
    ROCSPARSE_CHECK(rocsparse_spilu0_buffer_size(handle,
                                                 spilu0_descr,
                                                 matA,
                                                 matA,
                                                 rocsparse_spilu0_stage_compute,
                                                 &non_persistent_buffer_size_in_bytes,
                                                 nullptr));
    HIP_CHECK(hipFree(non_persistent_buffer));
    non_persistent_buffer = nullptr;
    HIP_CHECK(hipMalloc(&non_persistent_buffer, non_persistent_buffer_size_in_bytes));

    ROCSPARSE_CHECK(rocsparse_spilu0(handle,
                                     spilu0_descr,
                                     matA,
                                     matA,
                                     rocsparse_spilu0_stage_compute,
                                     non_persistent_buffer_size_in_bytes,
                                     non_persistent_buffer,
                                     nullptr));

    //
    // Check for any singularities after compute.
    //
    rocsparse_singularity post_compute_singularity[batch_count];
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    ROCSPARSE_CHECK(rocsparse_spilu0_get_output(handle,
                                                spilu0_descr,
                                                rocsparse_spilu0_output_singularity,
                                                post_compute_singularity,
                                                sizeof(rocsparse_singularity),
                                                nullptr));

    ROCSPARSE_CHECK(rocsparse_spilu0_get_output(handle,
                                                spilu0_descr,
                                                rocsparse_spilu0_output_singularity_position,
                                                singularity_position,
                                                sizeof(int64_t),
                                                nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));

    for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        switch(post_compute_singularity[batch_index])
        {
        case rocsparse_singularity_none:
        {
            break;
        }
        case rocsparse_singularity_symbolic:
        {
            std::cout << "numeric symbolic singularity detected at batch_index: " << batch_index
                      << ", at position: " << singularity_position[batch_index] << std::endl;
            ROCSPARSE_CHECK(rocsparse_status_internal_error);
            break;
        }

        case rocsparse_singularity_numeric_exact:
        {
            std::cout << "numeric exact singularity detected at batch_index: " << batch_index
                      << ", at position: " << singularity_position[batch_index] << std::endl;
            break;
        }
        case rocsparse_singularity_numeric_near:
        {
            std::cout << "numeric near singularity detected at batch_index: " << batch_index
                      << ", at position: " << singularity_position[batch_index] << std::endl;
            break;
        }
        }
    }

    HIP_CHECK(hipFree(non_persistent_buffer));

    ROCSPARSE_CHECK(rocsparse_spilu0_descr_destroy(handle, spilu0_descr, nullptr));

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
