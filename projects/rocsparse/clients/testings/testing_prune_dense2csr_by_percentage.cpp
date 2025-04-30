/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_prune_dense2csr_by_percentage_bad_arg(const Arguments& arg)
{
    static const size_t  safe_size                = 100;
    static rocsparse_int h_nnz_total_dev_host_ptr = 100;
    static size_t        h_buffer_size            = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptor
    rocsparse_local_mat_descr local_descr;

    // Create info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle                 = local_handle;
    rocsparse_int             m                      = safe_size;
    rocsparse_int             n                      = safe_size;
    const T*                  A                      = (const T*)0x4;
    rocsparse_int             lda                    = safe_size;
    const rocsparse_mat_descr descr                  = local_descr;
    T*                        csr_val                = (T*)0x4;
    rocsparse_int*            csr_row_ptr            = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind            = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_total_dev_host_ptr = &h_nnz_total_dev_host_ptr;
    rocsparse_mat_info        info                   = local_info;
    size_t*                   buffer_size            = &h_buffer_size;
    void*                     temp_buffer            = (void*)0x4;

    T percentage = static_cast<T>(1);

#define PARAMS_BUFFER_SIZE \
    handle, m, n, A, lda, percentage, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size
#define PARAMS_NNZ \
    handle, m, n, A, lda, percentage, descr, csr_row_ptr, nnz_total_dev_host_ptr, info, temp_buffer
#define PARAMS \
    handle, m, n, A, lda, percentage, descr, csr_val, csr_row_ptr, csr_col_ind, info, temp_buffer

    {
        int       nargs_to_exclude   = 3;
        const int args_to_exclude[3] = {5, 7, 9};

        select_bad_arg_analysis(rocsparse_prune_dense2csr_by_percentage_buffer_size<T>,
                                nargs_to_exclude,
                                args_to_exclude,
                                PARAMS_BUFFER_SIZE);
        select_bad_arg_analysis(rocsparse_prune_dense2csr_nnz_by_percentage<T>,
                                nargs_to_exclude,
                                args_to_exclude,
                                PARAMS_NNZ);
    }

    {
        int       nargs_to_exclude   = 3;
        const int args_to_exclude[3] = {5, 7, 9};
        select_bad_arg_analysis(
            rocsparse_prune_dense2csr_by_percentage<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(PARAMS_BUFFER_SIZE),
        rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    // Check percentage being less than 0
    percentage = -10;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(PARAMS_BUFFER_SIZE),
        rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(PARAMS_NNZ),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(PARAMS),
                            rocsparse_status_invalid_value);

    // Check percentage being greater than 100
    percentage = 110;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(PARAMS_BUFFER_SIZE),
        rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(PARAMS_NNZ),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(PARAMS),
                            rocsparse_status_invalid_value);
#undef PARAMS
#undef PARAMS_NNZ
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_prune_dense2csr_by_percentage(const Arguments& arg)
{
    rocsparse_int        M          = arg.M;
    rocsparse_int        N          = arg.N;
    rocsparse_int        LDA        = arg.denseld;
    rocsparse_index_base base       = arg.baseA;
    T                    percentage = static_cast<T>(arg.percentage);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    if(LDA < M)
    {
        EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           nullptr,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           nullptr,
                                                                           nullptr,
                                                                           nullptr,
                                                                           info,
                                                                           nullptr),
                                rocsparse_status_invalid_size);

        return;
    }

    // Allocate host memory
    host_vector<T>             h_A(LDA * N);
    host_vector<rocsparse_int> h_nnz_total_dev_host_ptr(1);

    // Allocate device memory
    device_vector<T>             d_A(LDA * N);
    device_vector<rocsparse_int> d_nnz_total_dev_host_ptr(1);
    device_vector<rocsparse_int> d_csr_row_ptr(M + 1);

    // Initialize a random matrix.
    rocsparse_seedrand();

    // Initialize the entire allocated memory.
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < LDA; ++i)
        {
            h_A[j * LDA + i] = -1;
        }
    }

    // Random initialization of the matrix.
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            h_A[j * LDA + i] = random_cached_generator_normal<T>();
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(T) * LDA * N, hipMemcpyHostToDevice));

    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(handle,
                                                                                 M,
                                                                                 N,
                                                                                 d_A,
                                                                                 LDA,
                                                                                 percentage,
                                                                                 descr,
                                                                                 nullptr,
                                                                                 d_csr_row_ptr,
                                                                                 nullptr,
                                                                                 info,
                                                                                 &buffer_size));

    T* d_temp_buffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&d_temp_buffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                         M,
                                                                         N,
                                                                         d_A,
                                                                         LDA,
                                                                         percentage,
                                                                         descr,
                                                                         d_csr_row_ptr,
                                                                         h_nnz_total_dev_host_ptr,
                                                                         info,
                                                                         d_temp_buffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                         M,
                                                                         N,
                                                                         d_A,
                                                                         LDA,
                                                                         percentage,
                                                                         descr,
                                                                         d_csr_row_ptr,
                                                                         d_nnz_total_dev_host_ptr,
                                                                         info,
                                                                         d_temp_buffer));

    host_vector<rocsparse_int> h_nnz_total_copied_from_device(1);
    CHECK_HIP_ERROR(hipMemcpy(h_nnz_total_copied_from_device,
                              d_nnz_total_dev_host_ptr,
                              sizeof(rocsparse_int),
                              hipMemcpyDeviceToHost));

    h_nnz_total_dev_host_ptr.unit_check(h_nnz_total_copied_from_device);

    device_vector<rocsparse_int> d_csr_col_ind(h_nnz_total_dev_host_ptr[0]);
    device_vector<T>             d_csr_val(h_nnz_total_dev_host_ptr[0]);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                                  M,
                                                                                  N,
                                                                                  d_A,
                                                                                  LDA,
                                                                                  percentage,
                                                                                  descr,
                                                                                  d_csr_val,
                                                                                  d_csr_row_ptr,
                                                                                  d_csr_col_ind,
                                                                                  info,
                                                                                  d_temp_buffer));

        host_vector<rocsparse_int> h_csr_row_ptr(M + 1);
        host_vector<rocsparse_int> h_csr_col_ind(h_nnz_total_dev_host_ptr[0]);
        host_vector<T>             h_csr_val(h_nnz_total_dev_host_ptr[0]);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            h_csr_row_ptr, d_csr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(h_csr_col_ind,
                                  d_csr_col_ind,
                                  sizeof(rocsparse_int) * h_nnz_total_dev_host_ptr[0],
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_csr_val, d_csr_val, sizeof(T) * h_nnz_total_dev_host_ptr[0], hipMemcpyDeviceToHost));

        // call host and check results
        host_vector<rocsparse_int> h_csr_row_ptr_cpu;
        host_vector<rocsparse_int> h_csr_col_ind_cpu;
        host_vector<T>             h_csr_val_cpu;
        host_vector<rocsparse_int> h_nnz_cpu(1);

        host_prune_dense2csr_by_percentage(M,
                                           N,
                                           h_A,
                                           LDA,
                                           base,
                                           percentage,
                                           h_nnz_cpu[0],
                                           h_csr_val_cpu,
                                           h_csr_row_ptr_cpu,
                                           h_csr_col_ind_cpu);

        h_nnz_cpu.unit_check(h_nnz_total_dev_host_ptr);
        h_csr_row_ptr_cpu.unit_check(h_csr_row_ptr);
        h_csr_col_ind_cpu.unit_check(h_csr_col_ind);
        h_csr_val_cpu.unit_check(h_csr_val);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_prune_dense2csr_by_percentage<T>,
                                               handle,
                                               M,
                                               N,
                                               d_A,
                                               LDA,
                                               percentage,
                                               descr,
                                               d_csr_val,
                                               d_csr_row_ptr,
                                               d_csr_col_ind,
                                               info,
                                               d_temp_buffer);

        double gbyte_count
            = prune_dense2csr_by_percentage_gbyte_count<T>(M, N, h_nnz_total_dev_host_ptr[0]);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::M,
                            N,
                            display_key_t::nnz,
                            h_nnz_total_dev_host_ptr[0],
                            display_key_t::percentage,
                            percentage,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(d_temp_buffer));
}

#define INSTANTIATE(TYPE)                                                                    \
    template void testing_prune_dense2csr_by_percentage_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_prune_dense2csr_by_percentage<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
// INSTANTIATE(rocsparse_float_complex);
// INSTANTIATE(rocsparse_double_complex);
void testing_prune_dense2csr_by_percentage_extra(const Arguments& arg) {}
