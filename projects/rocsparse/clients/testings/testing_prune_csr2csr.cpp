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
void testing_prune_csr2csr_bad_arg(const Arguments& arg)
{
    static const size_t  safe_size                = 1;
    static rocsparse_int h_nnz_total_dev_host_ptr = 1;
    static size_t        h_buffer_size            = 1;
    static T             h_threshold              = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptors
    rocsparse_local_mat_descr local_csr_descr_A;
    rocsparse_local_mat_descr local_csr_descr_C;

    host_dense_vector<rocsparse_int> hptr(safe_size + 1);
    hptr[0] = 0;
    hptr[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr_C(hptr);

    rocsparse_handle          handle                 = local_handle;
    rocsparse_int             m                      = safe_size;
    rocsparse_int             n                      = safe_size;
    rocsparse_int             nnz_A                  = safe_size;
    const rocsparse_mat_descr csr_descr_A            = local_csr_descr_A;
    const T*                  csr_val_A              = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr_A          = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind_A          = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr csr_descr_C            = local_csr_descr_C;
    T*                        csr_val_C              = (T*)0x4;
    rocsparse_int*            csr_row_ptr_C          = (rocsparse_int*)dcsr_row_ptr_C;
    rocsparse_int*            csr_col_ind_C          = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_total_dev_host_ptr = &h_nnz_total_dev_host_ptr;
    size_t*                   buffer_size            = &h_buffer_size;
    void*                     temp_buffer            = (void*)0x4;

    static constexpr int nargs_to_exclude_buffer_size                              = 3;
    const int            args_to_exclude_buffer_size[nargs_to_exclude_buffer_size] = {8, 10, 12};

    static constexpr int nargs_to_exclude_nnz                      = 2;
    const int            args_to_exclude_nnz[nargs_to_exclude_nnz] = {8, 12};

    static constexpr int nargs_to_exclude_solve                        = 4;
    const int            args_to_exclude_solve[nargs_to_exclude_solve] = {8, 10, 12, 13};

    const T* threshold = &h_threshold;

#define PARAMS_BUFFER_SIZE                                                                \
    handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, threshold, \
        csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, buffer_size
#define PARAMS_NNZ                                                                        \
    handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, threshold, \
        csr_descr_C, csr_row_ptr_C, nnz_total_dev_host_ptr, temp_buffer
#define PARAMS                                                                            \
    handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, threshold, \
        csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, temp_buffer
    select_bad_arg_analysis(rocsparse_prune_csr2csr_buffer_size<T>,
                            nargs_to_exclude_buffer_size,
                            args_to_exclude_buffer_size,
                            PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_prune_csr2csr_nnz<T>, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
    select_bad_arg_analysis(
        rocsparse_prune_csr2csr<T>, nargs_to_exclude_solve, args_to_exclude_solve, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr_A, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr_C, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_csr2csr_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_csr2csr_nnz<T>(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_csr2csr<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);
#undef PARAMS
#undef PARAMS_NNZ
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_prune_csr2csr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M          = arg.M;
    rocsparse_int               N          = arg.N;
    rocsparse_index_base        csr_base_A = arg.baseA;
    rocsparse_index_base        csr_base_C = arg.baseB;

    host_scalar<T> h_threshold(arg.threshold);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_local_mat_descr csr_descr_A;
    rocsparse_local_mat_descr csr_descr_C;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr_A, csr_base_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr_C, csr_base_C));

    // Allocate host memory for output CSR matrix
    host_vector<rocsparse_int> h_csr_row_ptr_A;
    host_vector<rocsparse_int> h_csr_col_ind_A;
    host_vector<T>             h_csr_val_A;

    host_vector<rocsparse_int> h_nnz_total_dev_host_ptr(1);

    // Generate uncompressed CSR matrix on host (or read from file)
    rocsparse_int nnz_A = 0;
    matrix_factory.init_csr(h_csr_row_ptr_A, h_csr_col_ind_A, h_csr_val_A, M, N, nnz_A, csr_base_A);

    // Allocate device memory for input CSR matrix
    device_vector<rocsparse_int> d_nnz_total_dev_host_ptr(1);
    device_vector<rocsparse_int> d_csr_row_ptr_C(M + 1);
    device_vector<rocsparse_int> d_csr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> d_csr_col_ind_A(nnz_A);
    device_vector<T>             d_csr_val_A(nnz_A);
    device_scalar<T>             d_threshold(h_threshold);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        d_csr_row_ptr_A, h_csr_row_ptr_A, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_csr_col_ind_A, h_csr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_csr_val_A, h_csr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));

    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_buffer_size<T>(handle,
                                                                 M,
                                                                 N,
                                                                 nnz_A,
                                                                 csr_descr_A,
                                                                 d_csr_val_A,
                                                                 d_csr_row_ptr_A,
                                                                 d_csr_col_ind_A,
                                                                 h_threshold,
                                                                 csr_descr_C,
                                                                 nullptr,
                                                                 d_csr_row_ptr_C,
                                                                 nullptr,
                                                                 &buffer_size));

    T* d_temp_buffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&d_temp_buffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_nnz<T>(handle,
                                                         M,
                                                         N,
                                                         nnz_A,
                                                         csr_descr_A,
                                                         d_csr_val_A,
                                                         d_csr_row_ptr_A,
                                                         d_csr_col_ind_A,
                                                         h_threshold,
                                                         csr_descr_C,
                                                         d_csr_row_ptr_C,
                                                         h_nnz_total_dev_host_ptr,
                                                         d_temp_buffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_csr2csr_nnz<T>(handle,
                                                         M,
                                                         N,
                                                         nnz_A,
                                                         csr_descr_A,
                                                         d_csr_val_A,
                                                         d_csr_row_ptr_A,
                                                         d_csr_col_ind_A,
                                                         d_threshold,
                                                         csr_descr_C,
                                                         d_csr_row_ptr_C,
                                                         d_nnz_total_dev_host_ptr,
                                                         d_temp_buffer));

    host_vector<rocsparse_int> h_nnz_total_copied_from_device(1);
    CHECK_HIP_ERROR(hipMemcpy(h_nnz_total_copied_from_device,
                              d_nnz_total_dev_host_ptr,
                              sizeof(rocsparse_int),
                              hipMemcpyDeviceToHost));

    h_nnz_total_dev_host_ptr.unit_check(h_nnz_total_copied_from_device);

    device_vector<rocsparse_int> d_csr_col_ind_C(h_nnz_total_dev_host_ptr[0]);
    device_vector<T>             d_csr_val_C(h_nnz_total_dev_host_ptr[0]);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(testing::rocsparse_prune_csr2csr<T>(handle,
                                                                  M,
                                                                  N,
                                                                  nnz_A,
                                                                  csr_descr_A,
                                                                  d_csr_val_A,
                                                                  d_csr_row_ptr_A,
                                                                  d_csr_col_ind_A,
                                                                  h_threshold,
                                                                  csr_descr_C,
                                                                  d_csr_val_C,
                                                                  d_csr_row_ptr_C,
                                                                  d_csr_col_ind_C,
                                                                  d_temp_buffer));

        host_vector<rocsparse_int> h_csr_row_ptr_C(M + 1);
        host_vector<rocsparse_int> h_csr_col_ind_C(h_nnz_total_dev_host_ptr[0]);
        host_vector<T>             h_csr_val_C(h_nnz_total_dev_host_ptr[0]);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(h_csr_row_ptr_C,
                                  d_csr_row_ptr_C,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(h_csr_col_ind_C,
                                  d_csr_col_ind_C,
                                  sizeof(rocsparse_int) * h_nnz_total_dev_host_ptr[0],
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(h_csr_val_C,
                                  d_csr_val_C,
                                  sizeof(T) * h_nnz_total_dev_host_ptr[0],
                                  hipMemcpyDeviceToHost));

        // call host and check results
        host_vector<rocsparse_int> h_csr_row_ptr_C_cpu;
        host_vector<rocsparse_int> h_csr_col_ind_C_cpu;
        host_vector<T>             h_csr_val_C_cpu;
        host_vector<rocsparse_int> h_nnz_C_cpu(1);

        host_prune_csr_to_csr(M,
                              N,
                              nnz_A,
                              h_csr_row_ptr_A,
                              h_csr_col_ind_A,
                              h_csr_val_A,
                              h_nnz_C_cpu[0],
                              h_csr_row_ptr_C_cpu,
                              h_csr_col_ind_C_cpu,
                              h_csr_val_C_cpu,
                              csr_base_A,
                              csr_base_C,
                              *h_threshold);

        h_nnz_C_cpu.unit_check(h_nnz_total_dev_host_ptr);
        h_csr_row_ptr_C_cpu.unit_check(h_csr_row_ptr_C);
        h_csr_col_ind_C_cpu.unit_check(h_csr_col_ind_C);
        h_csr_val_C_cpu.unit_check(h_csr_val_C);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_prune_csr2csr<T>,
                                                                      handle,
                                                                      M,
                                                                      N,
                                                                      nnz_A,
                                                                      csr_descr_A,
                                                                      d_csr_val_A,
                                                                      d_csr_row_ptr_A,
                                                                      d_csr_col_ind_A,
                                                                      h_threshold,
                                                                      csr_descr_C,
                                                                      d_csr_val_C,
                                                                      d_csr_row_ptr_C,
                                                                      d_csr_col_ind_C,
                                                                      d_temp_buffer);

        double gbyte_count = prune_csr2csr_gbyte_count<T>(M, nnz_A, h_nnz_total_dev_host_ptr[0]);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::nnz_C,
                            h_nnz_total_dev_host_ptr[0],
                            display_key_t::threshold,
                            *h_threshold,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(d_temp_buffer));
}

#define INSTANTIATE(TYPE)                                                    \
    template void testing_prune_csr2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_prune_csr2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
// INSTANTIATE(rocsparse_float_complex);
// INSTANTIATE(rocsparse_double_complex);
void testing_prune_csr2csr_extra(const Arguments& arg) {}
